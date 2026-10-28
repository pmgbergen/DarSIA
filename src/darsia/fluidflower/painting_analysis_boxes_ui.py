"""Inline rectangle editing on the painting photo, using the measured-box store.

Construction is passive. ``reload`` reads metadata in a worker; saving remains
an explicit action. A legacy source provider accepts ``(project, run_id)`` and
returns ``{"values": {"box1": ..., "box2": ...}, "source": <binding>}``, or
None. The provider must choose the actual source; no recent-job search occurs.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDoubleSpinBox, QFileDialog,
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem, QGroupBox, QHBoxLayout, QInputDialog,
    QLabel, QMenu, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
    QToolButton, QVBoxLayout,
)
from shiboken6 import isValid

from . import metric_analysis_domains as domains
from . import painting_analysis_boxes as boxes
from .painting_run_preview import _photo_geometry, read_preview_geometry


BOX_TOOLS_TIP = (
    "Boxes stay rectangles in measured millimetres. Shared rectangle sides "
    "can snap and move together. Polygon Follow edges and Join to neighbour "
    "do not apply to boxes. Right or middle drag still pans the photograph."
)
_SIDE_INDEX = {"x_min": 0, "x_max": 1, "y_min": 2, "y_max": 3}
_EDGE_SIDES = ("y_min", "x_max", "y_max", "x_min")
_CORNER_SIDES = (("x_min", "y_min"), ("x_max", "y_min"),
                 ("x_max", "y_max"), ("x_min", "y_max"))


def _bounds(outline):
    points = np.asarray(outline.points_mm, np.float64)
    return [float(points[:, 0].min()), float(points[:, 0].max()),
            float(points[:, 1].min()), float(points[:, 1].max())]


def _closest(point, first, second):
    p, a, b = (np.asarray(value, np.float64) for value in (point, first, second))
    direction = b - a
    length = float(direction @ direction)
    fraction = 0.0 if length == 0 else float(np.clip(((p-a) @ direction)/length, 0., 1.))
    result = a + fraction * direction
    return result, float(np.linalg.norm(result-p))


def passive_box_outlines(configuration, geometry, *, photo_path, shape, workspace_config=None):
    """Project an already verified configuration onto an exactly identified photo.

    The caller supplies its checked measured model. Display points are raw
    row/column pairs, never a bounding-box approximation of a projected quad.
    This helper reads and writes no files and is suitable for review overlays.
    """
    geometry = _photo_geometry(geometry, workspace_config or {}, photo_path, shape)
    result = []
    for row in configuration.get("domains", ()):
        if row["enabled"]:
            xy = domains.project_box(row, geometry)
            result.append(dict(domain_id=row["domain_id"], name=row["name"], role=row["role"],
                points_rc=tuple((float(y), float(x)) for x, y in xy)))
    return tuple(result)


class _Call(QObject):
    finished = Signal()

    def __init__(self, function):
        super().__init__()
        self.function = function
        self.result = None
        self.error = ""

    @Slot()
    def run(self):
        try:
            self.result = self.function()
        except Exception as error:  # reported through the owning controller
            self.error = f"{type(error).__name__}: {error}"
        self.finished.emit()


class PaintingAnalysisBoxesController(QObject):
    """Measured rectangles with a separate draft and one undo per box gesture.

    ``set_context`` accepts the result of an already verified metadata load;
    normal GUI ownership uses ``reload``. The parent must include ``busy`` and
    ``can_close`` in its existing operation/close guards. Layer tools should
    call ``set_mode(None)`` when activated; their session is never modified here.
    """
    changed = Signal()
    busy_changed = Signal(bool)
    saved = Signal(object)

    def __init__(self, view, group_layout=None, legacy_source_provider=None):
        super().__init__(view)
        self.view = view
        self.canvas = view.canvas
        self._viewport = self.canvas.viewport()
        self._disposed = False
        self.legacy_source_provider = legacy_source_provider
        self.draft = self.store = self.geometry = None
        self.models = {}
        self.geometry_binding = None
        self._context_key = None
        self._binding_error = "Load the measured frame models to edit analysis boxes."
        self._operation_error = ""
        self._thread = self._worker = None
        self._done = None
        self._mode = None
        self._changing_mode = False
        self._old_tool = None
        self._drag = None
        self._items = []
        self._undo, self._redo = [], []
        self._updating = False
        self._saved_configuration = None
        self.dirty = False
        self.legacy_conflicts = {}
        self.widget = self.group = QGroupBox("Analysis boxes", view)
        self.widget.setToolTip(BOX_TOOLS_TIP)
        layout = QVBoxLayout(self.widget)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Use", "Name", "x min [mm]", "x max [mm]", "y min [mm]", "y max [mm]", "Mass role"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.verticalHeader().hide()
        self.table.setMaximumHeight(220)
        layout.addWidget(self.table)
        row = QHBoxLayout()
        self.draw_button = QPushButton("Draw rectangle")
        self.edit_button = QPushButton("Move / resize")
        for button, mode in ((self.draw_button, "draw"), (self.edit_button, "edit")):
            button.setCheckable(True)
            button.setToolTip(BOX_TOOLS_TIP)
            button.toggled.connect(lambda checked, mode=mode: self._tool_toggled(mode, checked))
            if getattr(view, "tool_group", None) is not None:
                view.tool_group.addButton(button)
            row.addWidget(button)
        self.remove_button = QPushButton("Remove selected")
        self.remove_button.clicked.connect(self.remove_selected)
        row.addWidget(self.remove_button)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.shared_sides = QCheckBox("Move shared sides together")
        self.shared_sides.setChecked(True)
        self.snap_enabled = QCheckBox("Snap sides")
        self.snap_enabled.setChecked(True)
        self.snap_distance = QDoubleSpinBox()
        self.snap_distance.setRange(0., 100.)
        self.snap_distance.setDecimals(3)
        self.snap_distance.setValue(2.)
        self.snap_distance.setSuffix(" mm")
        for control in (self.shared_sides, self.snap_enabled, self.snap_distance):
            control.setToolTip(BOX_TOOLS_TIP)
            row.addWidget(control)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.undo_button, self.redo_button = QPushButton("Undo box edit"), QPushButton("Redo box edit")
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)
        self.reload_button = QPushButton("Reload boxes")
        self.reload_button.clicked.connect(self.reload)
        self.use_current_frame_button = QPushButton("Use saved boxes in current frame")
        self.use_current_frame_button.clicked.connect(self.use_saved_boxes_in_current_frame)
        self.use_current_frame_button.setToolTip(
            "Keep the box coordinates in a new draft bound to the currently selected measured frame. "
            "Save box revision keeps that choice.")
        self.save_button = QPushButton("Save box revision…")
        self.save_button.clicked.connect(self.save_revision)
        self.actions = QToolButton()
        self.actions.setText("Box actions")
        self.actions.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu = QMenu(self.actions)
        self.menu.addAction("Import original box1 / box2…", self.import_roi)
        self.menu.addAction("Review selected legacy-frame coordinates…", self.review_legacy)
        self.menu.addAction("Show retained input conflicts…", self.show_conflicts)
        self.menu.addAction("Choose measured frame models…", self.choose_models)
        self.actions.setMenu(self.menu)
        for button in (self.undo_button, self.redo_button, self.reload_button,
                       self.use_current_frame_button, self.save_button, self.actions):
            row.addWidget(button)
        layout.addLayout(row)
        self.note = QLabel(self._binding_error)
        self.note.setWordWrap(True)
        self.note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.note)
        from .analysis_box_reanalysis_ui import BoxMassReanalysis
        self.reanalysis = BoxMassReanalysis(self)
        layout.addWidget(self.reanalysis)
        if group_layout is not None:
            group_layout.addWidget(self.widget)
        self.table.itemChanged.connect(self._table_changed)
        self.table.itemSelectionChanged.connect(self._redraw)
        for watched in (view, self.canvas, self._viewport, self.widget):
            watched.destroyed.connect(self._invalidate)
        self._viewport.installEventFilter(self)
        self.refresh()

    def _alive(self):
        return (not self._disposed and isValid(self) and isValid(self.view)
                and isValid(self.canvas) and isValid(self._viewport) and isValid(self.widget))

    def _invalidate(self, *_args):
        # A viewport can send one more filtered event while its canvas is being
        # destroyed. This slot deliberately performs no Qt access.
        self._disposed = True
        self._mode, self._drag = None, None
        self._items = []

    def _key(self):
        if not self._alive():
            return None
        session, image = getattr(self.view, "session", None), getattr(self.view, "image", None)
        if session is None or image is None:
            return None
        preview = getattr(self.view, "run_preview", None)
        return (id(self.view.project), session.run_id, str(session.image_path),
                tuple(image.image_size), getattr(preview, "_geometry_selection", None),
                str(getattr(preview, "_index", "")))

    def busy(self):
        return self._thread is not None

    def can_close(self):
        return not self.busy()

    def wait_for_workers(self):
        if self._thread is not None:
            self._retire(self._thread)

    @property
    def mode(self):
        return self._mode

    def in_analysis_context(self):
        """Keep box graphics/tools out of the separate geometry/map routes.

        The original standalone editor (no scope, or its combined ``all``
        view) retains its existing box panel.
        """
        return getattr(self.view, "_workflow_scope", None) in {None, "all", "analysis_boxes"}

    def _editable(self):
        if not self._alive():
            return False
        preview = getattr(self.view, "run_preview", None)
        reference = preview is None or preview.is_reference()
        return (self.in_analysis_context() and self.draft is not None and self.geometry is not None and not self._binding_error
                and self._context_key == self._key() and reference and not self.busy()
                and not self._canvas_problem()
                and not getattr(self.view, "busy", lambda: False)())

    def _canvas_problem(self):
        if getattr(self.view, "_other_photo", None) is not None:
            return "Return to the reference photograph to edit analysis boxes."
        if getattr(self.canvas, "place_mode", False):
            return "Finish placing the rig-frame points before editing analysis boxes."
        expected = tuple(getattr(getattr(self.view, "image", None), "image_size", ()) or ())
        session_size = tuple(getattr(getattr(self.view, "session", None), "image_size", ()) or ())
        raw_size = tuple(getattr(self.canvas, "raw_size", ()) or ())
        if not expected or session_size != expected or raw_size != expected:
            return "The canvas and measured reference have different photograph grids."
        if hasattr(self.view, "_reference_token"):
            token = getattr(self.canvas, "preview", None)
            reference = self.view._reference_token
            background = getattr(self.view, "_background_token", None)
            is_background = (getattr(self.view, "_background_shown", False)
                             and background is not None and token is background)
            if not ((reference is not None and token is reference) or is_background):
                return "Return to the bound reference photograph to edit analysis boxes."
        return ""

    def _invoke(self, function, done):
        if not self._alive() or self.busy():
            return False
        thread, worker = QThread(self), _Call(function)
        self._thread, self._worker, self._done = thread, worker, done
        self._operation_error = ""
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._closed)
        self.refresh()
        thread.start()
        self.busy_changed.emit(True)
        return True

    @Slot()
    def _closed(self):
        thread = self.sender()
        if thread is self._thread:
            self._retire(thread)

    def _retire(self, thread):
        if thread is not self._thread:
            return
        thread.wait()
        worker, done = self._worker, self._done
        self._thread = self._worker = self._done = None
        if worker.error:
            self._operation_error = worker.error
        elif done is not None and self._alive():
            try:
                done(worker.result)
            except Exception as error:
                self._operation_error = f"{type(error).__name__}: {error}"
        # Release the worker on the GUI thread, after its thread has joined.
        del worker
        if isValid(thread):
            thread.deleteLater()
        self.refresh()
        if isValid(self):
            self.busy_changed.emit(self.busy())

    def reload(self, *_args, preserve_draft=False):
        if not self._alive() or self.busy():
            return False
        if self.dirty and not preserve_draft and QMessageBox.question(self.widget, "Reload boxes",
                "Reload the saved boxes and replace the unsaved box draft?") != QMessageBox.StandardButton.Yes:
            return False
        kept = None
        if preserve_draft and self.draft is not None:
            kept = (self.draft.configuration(), deepcopy(self._saved_configuration),
                    [value.clone() for value in self._undo], [value.clone() for value in self._redo],
                    deepcopy(self.legacy_conflicts), self._context_key)
        self.set_mode(None)
        preview = getattr(self.view, "run_preview", None)
        ensure_index = getattr(preview, "ensure_saved_geometry_index", None)
        if callable(ensure_index):
            ensure_index()
        key = self._key()
        project, session = self.view.project, getattr(self.view, "session", None)
        if key is None or session is None:
            self._binding_error = "Choose the reference photograph before loading analysis boxes."
            self.refresh()
            return False
        index, run = key[-1], session.run_id
        selection = getattr(preview, "_geometry_selection", None)
        index_error = str(getattr(preview, "_index_error", ""))
        provider = self.legacy_source_provider

        def load():
            store = domains.MetricAnalysisDomainStore(project)
            current = store.current().get("configuration", {})
            current = current if current.get("format") == domains.FORMAT else None
            models, geometry_binding, error = {}, None, ""
            try:
                from .painting_frame_models import verify_frame_model_selection
                if not index and index_error:
                    raise ValueError(index_error)
                if not index:
                    raise ValueError("No measured frame models are saved in user settings. "
                        "Use Box actions > Choose measured frame models to select the index once.")
                verify_frame_model_selection(selection)
                fresh = domains.new_configuration(project, index)
                models, geometry_binding = read_preview_geometry(index)
                if geometry_binding["geometry_index"] != fresh["geometry_index"]:
                    raise ValueError("The measured frame models changed while loading the box binding.")
                if current is None:
                    current = fresh
                report = domains.validate_configuration(project, current)
                if not domains.same_file_binding(current["geometry_index"], fresh["geometry_index"]) or not report["valid"]:
                    raise ValueError("Saved boxes bind another geometry revision or frame model; review that binding before editing.")
                verify_frame_model_selection(selection)
            except OSError as exc:
                error = ("Measured frame metadata cannot be read for " + index + ": " + str(exc) +
                         ". Use Box actions > Choose measured frame models to override the saved index.")
            except (ValueError, KeyError) as exc:
                error = str(exc)
            legacy = provider(project, run) if provider is not None else None
            return current, models, store, legacy, error, geometry_binding

        def loaded(result):
            if key != self._key():
                self._operation_error = "The reference changed while loading; reload its boxes."
                return
            config, models, store, legacy, error, geometry_binding = result
            if kept is not None:
                draft, baseline, undo, redo, conflicts, old_key = kept
                if (error or config is None or old_key is None or old_key[0] != key[0]
                        or not domains.same_configuration_binding(config, draft) or config != baseline):
                    self.geometry = None
                    self._context_key = key
                    self._binding_error = error or (
                        "The measured-box binding or saved revision changed. The box draft and undo history were retained; "
                        "use Reload boxes explicitly to review replacement.")
                    self._operation_error = self._binding_error
                    self.refresh()
                    return
                self.set_context(draft, models, store=store, legacy=legacy,
                                 geometry_binding=geometry_binding)
                self._saved_configuration = baseline
                self._undo, self._redo = undo, redo
                self.legacy_conflicts = dict(conflicts, **self.legacy_conflicts)
                self.dirty = self.draft.configuration() != baseline
                self._refresh_table()
                self.refresh()
                return
            self.set_context(config, models, store=store, legacy=legacy, binding_error=error,
                             geometry_binding=geometry_binding)
        return self._invoke(load, loaded)

    def set_context(self, configuration, models, *, store=None, legacy=None, binding_error="",
                    geometry_binding=None):
        """Install already checked metadata; never save or infer a missing frame."""
        if not self._alive():
            return False
        draft = boxes.AnalysisBoxesDraft(configuration) if configuration is not None else None
        added, conflicts = (), {}
        if draft is not None and legacy is not None:
            added, conflicts = self._import_legacy(draft, legacy)
        self.set_mode(None)
        self.draft = draft
        self.store, self.models = store, dict(models)
        self.geometry_binding = deepcopy(geometry_binding)
        self.geometry = None
        self._context_key = self._key()
        self._binding_error = str(binding_error)
        self._operation_error = ""
        self.legacy_conflicts = conflicts
        self._saved_configuration = deepcopy(configuration)
        self.dirty = bool(added)
        self._undo, self._redo = [], []
        if not self._binding_error and self._context_key is not None:
            try:
                run, path, size = self._context_key[1:4]
                model = self.models.get(run.casefold())
                if model is None:
                    raise ValueError("This reference run has no measured frame model.")
                self.geometry = _photo_geometry(model, getattr(self.view, "_workspace_frame_config", {}),
                                                path, (size[1], size[0]))
            except (OSError, ValueError, KeyError) as error:
                self._binding_error = str(error)
        self._refresh_table()
        self.refresh()

    @staticmethod
    def _import_legacy(draft, legacy):
        if not isinstance(legacy, dict) or set(legacy) != {"values", "source"} or not legacy["source"]:
            raise ValueError("Original boxes need their explicit values and source binding")
        values = legacy["values"]
        candidates = domains.import_legacy_boxes(values, source=legacy["source"])
        existing = {row["role"]: row for row in draft.configuration()["domains"] if row["role"]}
        conflicts = {row["role"]: dict(existing=deepcopy(existing[row["role"]]), original=deepcopy(row))
            for row in candidates if row["role"] in existing
            and (existing[row["role"]]["bounds_mm"] != row["bounds_mm"]
                 or (existing[row["role"]].get("legacy_import") or {}).get("original_value", values[row["role"]]) != values[row["role"]])}
        return draft.import_missing_legacy(values, source=legacy["source"]), conflicts

    def refresh(self):
        if not self._alive():
            return
        if self._context_key is not None and self._context_key != self._key():
            self.geometry = None
            self._binding_error = "The photograph or measured-model selection changed; reload the box binding."
            self.set_mode(None)
        editable = self._editable()
        if not editable and self._mode is not None:
            self.set_mode(None)
        for control in (self.draw_button, self.edit_button, self.remove_button, self.shared_sides,
                        self.snap_enabled, self.snap_distance):
            control.setEnabled(editable)
        self.undo_button.setEnabled(editable and bool(self._undo))
        self.redo_button.setEnabled(editable and bool(self._redo))
        self.reload_button.setEnabled(not self.busy())
        self.save_button.setEnabled(editable and self.store is not None and self.dirty)
        has_saved_boxes = bool(self.draft is not None and self.store is not None
            and self.draft.configuration()["domains"])
        self.use_current_frame_button.setVisible(bool(self._binding_error and has_saved_boxes))
        self.use_current_frame_button.setEnabled(has_saved_boxes and not self.busy())
        self.actions.setEnabled(not self.busy())
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed
                                  if editable else QAbstractItemView.EditTrigger.NoEditTriggers)
        analysis_scope = getattr(self.view, "_workflow_scope", None) == "analysis_boxes"
        self.table.setColumnHidden(6, analysis_scope)
        if self._operation_error and not self.busy():
            text = self._operation_error
        elif self._binding_error:
            text = self._binding_error
        elif self._canvas_problem():
            text = self._canvas_problem()
        elif not editable and not self.busy():
            text = "Analysis boxes are shown read only in this run; edit them on the reference photograph."
        elif self.busy():
            text = "Reading or saving analysis-box metadata…"
        else:
            text = "Measured millimetres, x right and y up. " + ("Unsaved box draft. " if self.dirty else "Saved box draft. ")
            if analysis_scope:
                text += "Optional analysis boxes; saved mass maps can be integrated for any enabled box."
            else:
                try:
                    self.draft.mass_boxes()
                    text += "Both mass roles are ready."
                except ValueError as error:
                    text += str(error)
        if self.legacy_conflicts:
            text += " Original input conflicts were retained; review them in Box actions."
        self.note.setText(text)
        self.reanalysis.refresh()
        self._redraw()

    def _refresh_table(self):
        selected = self.selected_ids()
        self._updating = True
        self.table.blockSignals(True)
        rows = self.draft.configuration()["domains"] if self.draft is not None else []
        self.table.setRowCount(len(rows))
        for number, row in enumerate(rows):
            for column, value in enumerate(("", row["name"], *row["bounds_mm"], row["role"])):
                item = QTableWidgetItem(f"{value:.12g}" if isinstance(value, float) else str(value))
                item.setData(Qt.ItemDataRole.UserRole, row["domain_id"])
                if column == 0:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Checked if row["enabled"] else Qt.CheckState.Unchecked)
                self.table.setItem(number, column, item)
            if row["domain_id"] in selected:
                self.table.selectRow(number)
        self.table.blockSignals(False)
        self._updating = False

    def selected_ids(self):
        if not self._alive():
            return ()
        return tuple(self.table.item(index.row(), 0).data(Qt.ItemDataRole.UserRole)
                     for index in self.table.selectionModel().selectedRows() if self.table.item(index.row(), 0))

    def _commit(self, candidate):
        if candidate.configuration() == self.draft.configuration():
            return False
        self._undo.append(self.draft.clone())
        self._undo = self._undo[-200:]
        self._redo.clear()
        self.draft = candidate
        self.dirty = candidate.configuration() != self._saved_configuration
        self._operation_error = ""
        self._refresh_table()
        self.refresh()
        self.changed.emit()
        return True

    def _change(self, function):
        if not self._editable():
            return False
        candidate = self.draft.clone()
        try:
            function(candidate)
            candidate.configuration()
        except (ValueError, KeyError, TypeError) as error:
            self._operation_error = str(error)
            self.refresh()
            return False
        return self._commit(candidate)

    def _table_changed(self, item):
        if self._updating or not self._editable():
            return
        key, column = item.data(Qt.ItemDataRole.UserRole), item.column()
        if column in (2, 3, 4, 5):
            try:
                number = float(item.text().replace(",", "."))
            except ValueError:
                self.note.setText("Enter a finite measured millimetre coordinate.")
                self._refresh_table()
                return
            side = tuple(_SIDE_INDEX)[column-2]
            self._change(lambda draft: draft.move_side(key, side, number,
                move_shared=self.shared_sides.isChecked()))
        else:
            values = {"name": item.text()} if column == 1 else {"role": item.text().strip()} if column == 6 else {
                "enabled": item.checkState() == Qt.CheckState.Checked}
            self._change(lambda draft: draft.update_metadata(key, **values))
        self._refresh_table()

    def undo(self):
        if not self._editable() or not self._undo:
            return False
        self._redo.append(self.draft.clone())
        self.draft = self._undo.pop()
        self.dirty = self.draft.configuration() != self._saved_configuration
        self._refresh_table(); self.refresh(); self.changed.emit()
        return True

    def redo(self):
        if not self._editable() or not self._redo:
            return False
        self._undo.append(self.draft.clone())
        self.draft = self._redo.pop()
        self.dirty = self.draft.configuration() != self._saved_configuration
        self._refresh_table(); self.refresh(); self.changed.emit()
        return True

    def remove_selected(self):
        selected = self.selected_ids()
        return self._change(lambda draft: [draft.remove(key) for key in selected]) if selected else False

    def _tool_toggled(self, mode, checked):
        if self._changing_mode or not self._alive():
            return
        if checked:
            if not self.set_mode(mode):
                pan = getattr(self.view, "tool_buttons", {}).get("pan")
                if pan is not None:
                    pan.setChecked(True)
                self.set_mode(None)
        elif self._mode == mode:
            # A different member of the host's existing exclusive group won.
            # Its own handler is responsible for the new canvas tool.
            self._mode, self._drag = None, None
            self._redraw()

    def set_mode(self, mode, *, restore_tool=True):
        if mode not in (None, "draw", "edit"):
            raise ValueError("Choose draw, edit, or no box tool")
        if not self._alive():
            self._mode, self._drag = None, None
            return False
        if mode is not None and not self._editable():
            return False
        canvas = self.view.canvas
        if mode is not None and self._mode is None:
            self._old_tool = getattr(canvas, "tool", "pan")
            if hasattr(canvas, "cancel_polygon"):
                canvas.cancel_polygon()
            if hasattr(canvas, "clear_selection"):
                canvas.clear_selection()
            canvas.set_tool("pan")
        restore = (mode is None and self._mode is not None and restore_tool
                   and getattr(canvas, "tool", None) == "pan")
        # The host deliberately returns to the reference photo when a drawing
        # tool is selected. Automatic box deactivation must preserve a newly
        # selected read-only photo or placement view instead of invoking that.
        restored_tool = (self._old_tool or "pan") if restore and self._editable() else "pan"
        # Cursor changes can synchronously re-enter viewport event filters.
        # Publish the inactive mode and guard before restoring any host tool.
        self._mode, self._drag = mode, None
        self._changing_mode = True
        try:
            if restore:
                canvas.set_tool(restored_tool)
                restore_host = getattr(self.view, "set_tool", None)
                if callable(restore_host):
                    restore_host(restored_tool)
                buttons = getattr(self.view, "tool_buttons", {})
                selected = buttons.get(restored_tool, buttons.get("pan"))
                if selected is not None:
                    selected.setChecked(True)
            for button, value in ((self.draw_button, "draw"), (self.edit_button, "edit")):
                button.setChecked(mode == value)
        finally:
            self._changing_mode = False
        self._redraw()
        return True

    def _layer_segments(self):
        matrix = self.geometry.metric_to_grid
        for polygon in getattr(self.view.session, "polygons", ()):
            points = [boxes.raw_point_to_metric(point, matrix) for point in polygon.points]
            if len(points) > 1:
                yield from zip(points, (*points[1:], points[0]))

    def _snap(self, draft, side, coordinate, anchor, *, exclude=None):
        if not self.snap_enabled.isChecked() or self.snap_distance.value() == 0:
            return coordinate
        axis = 0 if side.startswith("x") else 1
        candidates = []
        for outline in draft.outlines():
            if outline.domain_id == exclude:
                continue
            for number, (first, second) in enumerate(zip(outline.points_mm, (*outline.points_mm[1:], outline.points_mm[0]))):
                if _EDGE_SIDES[number].startswith(side[0]):
                    point, distance = _closest(anchor, first, second)
                    candidates.append((distance, float(point[axis])))
        for first, second in self._layer_segments():
            point, distance = _closest(anchor, first, second)
            candidates.append((distance, float(point[axis])))
        if candidates:
            distance, value = min(candidates, key=lambda row: row[0])
            if distance <= self.snap_distance.value():
                return value
        return coordinate

    def _rectangle_candidate(self, before, first_rc, second_rc, *, name=None):
        first, second = [boxes.raw_point_to_metric(point, self.geometry.metric_to_grid)
                         for point in (first_rc, second_rc)]
        values = [min(first[0], second[0]), max(first[0], second[0]),
                  min(first[1], second[1]), max(first[1], second[1])]
        candidate = before.clone()
        for side, index in _SIDE_INDEX.items():
            anchor = (values[index], second[1]) if side.startswith("x") else (second[0], values[index])
            values[index] = self._snap(candidate, side, values[index], anchor)
        if name is None:
            used = {outline.name.casefold() for outline in candidate.outlines()}
            number = len(used) + 1
            name = f"Analysis box {number}"
            while name.casefold() in used:
                number += 1
                name = f"Analysis box {number}"
        candidate.add_rectangle(name, values)
        return candidate

    def create_rectangle(self, first_rc, second_rc, *, name=None):
        """Two opposite corners define a rectangle in the measured frame."""
        if not self._editable():
            return False
        try:
            candidate = self._rectangle_candidate(self.draft, first_rc, second_rc, name=name)
        except ValueError as error:
            self._operation_error = str(error)
            self.refresh()
            return False
        return self._commit(candidate)

    def edit_side(self, domain_id, side, point_rc):
        if not self._editable() or side not in _SIDE_INDEX:
            return False
        point = boxes.raw_point_to_metric(point_rc, self.geometry.metric_to_grid)
        coordinate = point[0 if side.startswith("x") else 1]
        coordinate = self._snap(self.draft, side, coordinate, point, exclude=domain_id)
        return self._change(lambda draft: draft.move_side(domain_id, side, coordinate,
            move_shared=self.shared_sides.isChecked()))

    def _path(self, points):
        path = QPainterPath()
        for number, point in enumerate(points):
            value = self.view.canvas.raw_to_scene(*point)
            path.moveTo(value) if number == 0 else path.lineTo(value)
        path.closeSubpath()
        return path

    def _redraw(self):
        if not self._alive():
            self._items = []
            return
        scene = self.canvas.scene()
        if scene is None or not isValid(scene):
            self._items = []
            return
        for item in self._items:
            if isValid(item):
                scene.removeItem(item)
        self._items = []
        if (not self.in_analysis_context() or self.geometry is None or self.draft is None
                or self._context_key != self._key()):
            return
        preview = getattr(self.view, "run_preview", None)
        if preview is not None and not preview.is_reference():
            return
        if self._canvas_problem():
            return
        draft = self._drag.get("preview", self.draft) if self._drag else self.draft
        selected = self.selected_ids()
        for outline in draft.outlines():
            try:
                points = boxes.metric_to_raw(outline.points_mm, self.geometry.metric_to_grid)
            except ValueError as error:
                self._operation_error = str(error)
                self.note.setText(str(error))
                continue
            item = QGraphicsPathItem(self._path(points))
            color = QColor("#ffd200" if outline.role == "box1" else "#00ffff")
            pen = QPen(color, 2.)
            pen.setCosmetic(True)
            if not outline.enabled:
                pen.setStyle(Qt.PenStyle.DashLine)
            item.setPen(pen)
            item.setZValue(30.)
            item.setToolTip(outline.name + (" / " + outline.role if outline.role else ""))
            scene.addItem(item)
            self._items.append(item)
            if self._mode == "edit" and outline.domain_id in selected:
                for point in points:
                    handle = QGraphicsEllipseItem(-4., -4., 8., 8.)
                    handle.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
                    handle.setPos(self.view.canvas.raw_to_scene(*point))
                    handle.setPen(pen); handle.setBrush(color); handle.setZValue(31.)
                    scene.addItem(handle); self._items.append(handle)

    def _hit(self, point):
        canvas = self.view.canvas
        selected = set(self.selected_ids())
        outlines = sorted(self.draft.outlines(), key=lambda row: row.domain_id in selected, reverse=True)
        for outline in outlines:
            raw = boxes.metric_to_raw(outline.points_mm, self.geometry.metric_to_grid)
            pixels = [canvas.mapFromScene(canvas.raw_to_scene(*p)) for p in raw]
            for number, pixel in enumerate(pixels):
                if np.hypot(pixel.x()-point.x(), pixel.y()-point.y()) <= 8.:
                    return outline.domain_id, _CORNER_SIDES[number]
            for number, (first, second) in enumerate(zip(pixels, (*pixels[1:], pixels[0]))):
                _near, distance = _closest((point.x(), point.y()), (first.x(), first.y()), (second.x(), second.y()))
                if distance <= 8.:
                    return outline.domain_id, (_EDGE_SIDES[number],)
            if self._path(raw).contains(canvas.mapToScene(point)):
                return outline.domain_id, ()
        return None

    def _drag_candidate(self, raw):
        drag = self._drag
        candidate = drag["before"].clone()
        if tuple(raw) == tuple(drag["raw"]):
            return candidate
        if self._mode == "draw":
            return self._rectangle_candidate(candidate, drag["raw"], raw)
        key, sides = drag["hit"]
        matrix = self.geometry.metric_to_grid
        now = boxes.raw_point_to_metric(raw, matrix)
        start = boxes.raw_point_to_metric(drag["raw"], matrix)
        if sides:
            original = next(row for row in candidate.outlines() if row.domain_id == key)
            bounds = _bounds(original)
            changes = {}
            for side in sides:
                axis = 0 if side.startswith("x") else 1
                coordinate = bounds[_SIDE_INDEX[side]] + now[axis] - start[axis]
                anchor = (coordinate, now[1]) if axis == 0 else (now[0], coordinate)
                changes[(key, side)] = self._snap(candidate, side, coordinate, anchor, exclude=key)
            candidate.edit_sides(changes, move_shared=self.shared_sides.isChecked())
        else:
            candidate.move_box(key, now[0]-start[0], now[1]-start[1],
                               move_shared=self.shared_sides.isChecked())
        return candidate

    def eventFilter(self, watched, event):
        if self._changing_mode or not self._alive():
            return False
        if watched is not self._viewport or self._mode is None:
            return super().eventFilter(watched, event)
        if getattr(self.view.canvas, "tool", "pan") != "pan":
            self.set_mode(None)
            return False
        if not self._editable():
            self.set_mode(None)
            return False
        kind = event.type()
        if kind not in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove, QEvent.Type.MouseButtonRelease):
            return False
        point = event.position().toPoint()
        raw = self.view.canvas.scene_to_raw(self.view.canvas.mapToScene(point))
        if kind == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            hit = self._hit(point) if self._mode == "edit" else None
            if self._mode == "edit" and hit is None:
                self.table.clearSelection()
                return True
            self._drag = dict(raw=raw, before=self.draft.clone(), preview=self.draft.clone(), hit=hit)
            if hit is not None:
                for row in range(self.table.rowCount()):
                    if self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) == hit[0]:
                        self.table.selectRow(row)
            return True
        if self._drag is None:
            return False
        if kind == QEvent.Type.MouseMove:
            try:
                self._drag["preview"] = self._drag_candidate(raw)
                self._operation_error = ""
            except ValueError as error:
                self._drag["preview"] = self._drag["before"]
                self._operation_error = str(error)
            self.refresh()
            return True
        if kind == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            try:
                candidate = self._drag_candidate(raw)
                self._operation_error = ""
            except ValueError as error:
                candidate = None
                self._operation_error = str(error)
            self._drag = None
            if candidate is not None:
                self._commit(candidate)
            self.refresh()
            return True
        return False

    def choose_models(self):
        if not self._alive():
            return False
        preview = getattr(self.view, "run_preview", None)
        if preview is not None and hasattr(preview, "choose_geometry"):
            preview.choose_geometry()
            self.reload()
        else:
            self.note.setText("The painting view has no measured-frame chooser.")

    def import_roi(self):
        if not self._editable():
            return False
        path, _filter = QFileDialog.getOpenFileName(self.widget, "Import original box1 / box2", "", "ROI registry (*.toml)")
        if not path:
            return False
        key, before = self._key(), self.draft.clone()
        def imported(rows):
            if key != self._key():
                self._operation_error = "The reference changed; the imported boxes were not applied."
                return
            values = {row["role"]: row["legacy_import"]["original_value"] for row in rows}
            candidate = before.clone()
            added, conflicts = self._import_legacy(candidate,
                dict(values=values, source=rows[0]["legacy_import"]["source"]))
            self.legacy_conflicts.update(conflicts)
            if added:
                self._commit(candidate)
            else:
                self.refresh()
        return self._invoke(lambda: domains.load_legacy_registry(Path(path)), imported)

    def review_legacy(self):
        selected = self.selected_ids()
        if not self._editable() or not selected:
            return False
        config = self.draft.configuration()
        rows = [row for row in config["domains"] if row["domain_id"] in selected and row.get("legacy_import")]
        if not rows:
            self.note.setText("Select imported boxes whose frame interpretation needs review.")
            return False
        names = ", ".join(row["name"] for row in rows)
        answer = QMessageBox.question(self.widget, "Review imported box coordinates",
            f"Use the existing coordinates of {names} in the measured rig frame?\n"
            "Their original frame is nominally 900 × 490 mm. The measured models define the placement. "
            "No coordinate is scaled, clipped or moved by this decision.")
        if answer != QMessageBox.StandardButton.Yes:
            return False
        for row in rows:
            row["legacy_import"]["decision"] = "use_coordinates_in_measured_frame"
        return self._commit(boxes.AnalysisBoxesDraft(config))

    def show_conflicts(self):
        if not self._alive():
            return
        if not self.legacy_conflicts:
            self.note.setText("No original-box input conflicts are recorded.")
            return
        lines = []
        for role, value in self.legacy_conflicts.items():
            lines.append(f"{role}: existing {value['existing']['bounds_mm']} mm; "
                         f"original input {value['original']['bounds_mm']} mm. Existing draft retained.")
        QMessageBox.information(self.widget, "Retained original-box input conflicts", "\n".join(lines))

    def save_revision(self):
        if not self._editable() or self.store is None or not self.dirty:
            return False
        reason, accepted = QInputDialog.getText(self.widget, "Save analysis-box revision", "Reason:")
        if not accepted or not reason.strip():
            return False
        config, key, store = self.draft.configuration(), self._key(), self.store
        actor = getattr(self.view, "_reviewer", lambda: "")() or ""
        def saved(revision):
            if key != self._key():
                self._operation_error = "The box revision was saved for the previous reference; reload this view."
                return
            self._saved_configuration = deepcopy(config)
            self.dirty = self.draft.configuration() != self._saved_configuration
            self.saved.emit(revision)
        return self._invoke(lambda: store.save_revision(config, actor=actor, reason=reason.strip()), saved)

    def use_saved_boxes_in_current_frame(self):
        """Keep existing box values in an explicit unsaved draft for the selected frame."""
        if not self._alive() or self.busy() or self.draft is None or self.store is None:
            return False
        source, key = self.draft.configuration(), self._key()
        if key is None or not key[-1] or not source["domains"]:
            return False
        project, store = self.view.project, self.store
        baseline, conflicts = deepcopy(self._saved_configuration), deepcopy(self.legacy_conflicts)
        selection = getattr(getattr(self.view, "run_preview", None), "_geometry_selection", None)

        def prepare():
            from .painting_frame_models import verify_frame_model_selection
            verify_frame_model_selection(selection)
            config = domains.use_boxes_in_current_frame(project, source, key[-1])
            models, binding = read_preview_geometry(key[-1])
            if not domains.same_file_binding(config["geometry_index"], binding["geometry_index"]):
                raise ValueError("The selected measured frame changed while preparing the box draft.")
            verify_frame_model_selection(selection)
            return config, models, binding

        def prepared(result):
            if key != self._key():
                self._operation_error = "The selected frame changed; the saved boxes were not rebound."
                return
            config, models, binding = result
            self.set_context(config, models, store=store, geometry_binding=binding)
            self._saved_configuration = baseline
            self.legacy_conflicts = conflicts
            self.dirty = True
            self.refresh()
            self.changed.emit()

        return self._invoke(prepare, prepared)

    def passive_outlines(self, geometry, *, photo_path, shape, workspace_config=None):
        if self.draft is None:
            return ()
        return passive_box_outlines(self.draft.configuration(), geometry,
            photo_path=photo_path, shape=shape, workspace_config=workspace_config)

    def preview_geometry_problem(self, binding, *, source_run, target_run):
        """Match a worker's saved model snapshot, without reading on the GUI thread."""
        loaded = self.geometry_binding
        if not isinstance(binding, dict) or not isinstance(loaded, dict) or self.draft is None:
            return "Analysis-box overlay hidden: the preview has no checked measured-model identity."
        configuration = self.draft.configuration()
        if (not domains.same_file_binding(binding.get("geometry_index"), loaded.get("geometry_index"))
                or not domains.same_file_binding(binding.get("geometry_index"), configuration.get("geometry_index"))):
            return "Analysis-box overlay hidden: the measured geometry revision changed. Reload boxes to review its binding."
        loaded_models, observed_models = loaded.get("models"), binding.get("models")
        if not isinstance(loaded_models, dict) or not isinstance(observed_models, dict):
            return "Analysis-box overlay hidden: the preview has no checked measured-model identity."
        for run in (str(source_run).casefold(), str(target_run).casefold()):
            expected = loaded_models.get(run)
            observed = observed_models.get(run)
            model = self.models.get(run)
            if (not isinstance(expected, dict) or expected != observed or model is None
                    or expected.get("reference_image") != model.reference_image
                    or expected.get("shape_hw") != (None if model.shape_hw is None else list(model.shape_hw))
                    or not np.array_equal(np.asarray(expected.get("metric_to_grid")), model.metric_to_grid)):
                return "Analysis-box overlay hidden: the preview uses another measured source or target model. Reload boxes to review its binding."
        return ""


__all__ = ["PaintingAnalysisBoxesController", "passive_box_outlines", "BOX_TOOLS_TIP"]
