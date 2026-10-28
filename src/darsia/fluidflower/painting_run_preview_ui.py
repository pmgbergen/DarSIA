"""One painting run chooser; other runs are automatic, read-only previews."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, QSettings, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QShortcut
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                              QMessageBox, QProgressBar, QPushButton, QStackedWidget, QTableWidget,
                              QTableWidgetItem, QVBoxLayout, QWidget, QAbstractItemView)
from shiboken6 import isValid

from . import layer_painting as lp
from . import painting_run_preview as preview
from . import painting_frame_models as frame_models
from . import painting_preview_frame_domain as frame_domain
from . import painting_drawing_coverage as drawing_coverage
from . import painting_photo_continuation as photo_continuation

_SETTINGS = ("DarSIA", "FluidFlower layer painting")
_INDEX = "placement/geometry_index"
_FIRST = "placement/first_runs"


def _plain_error(error):
    return str(error) if isinstance(error, ValueError) else f"{type(error).__name__}: {error}"


class _PreviewWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)
    partial = Signal(object)

    def __init__(self, project, session, proposal, corrections, frame, options):
        super().__init__()
        self.project, self.session, self.proposal = project, session, proposal
        self.corrections, self.frame, self.options = corrections, frame, options
        self.result, self.error = None, ""

    @Slot()
    def run(self):
        try:
            frame_models.verify_frame_model_selection(self.options.get("geometry_selection"))
            self.progress.emit("Preparing the reference drawing")
            record = self.options.get("record")
            session = self.session
            if record and record.get("kind") == "drawing":
                from .layer_painting_version_archive import load_drawing_version
                session = load_drawing_version(self.project, session.run_id, record)
                record = None
                source = "drawing"
            else:
                source = self.options["source"]
            source_name = self.options["name"]
            if record is None and source == "manual":
                source_name = "Current manual map"
            elif record is None and source == "map" and self.proposal is not None:
                from . import layer_painting_polygons_only as polygons

                # Only the current-map convenience choice may use today's
                # drawing. Explicit saved versions retain their strict proof.
                if not polygons.proposal_current(self.proposal, session, self.frame):
                    source = "manual"
                    source_name = "Current manual map (the proposal is out of date)"
            session, layers = preview.preview_layers(session, self.proposal, self.corrections,
                                                     self.frame, source=source, record=record)
            # Only the live drawing/manual convenience route has extra photo
            # coverage. Explicit saved maps and proposals keep their exact arrays.
            live_drawing = self.options.get("record") is None and (
                source in {"manual", "drawing"} or self.proposal is None)
            added = 0
            if live_drawing:
                layers, added = drawing_coverage.add_explicit_outside(
                    session, self.frame, layers, drawing=source != "manual")
            contract = None
            if self.options.get("continue_photo_edges"):
                from .photo_edge_continuation import build_contracts
                run = str(self.options["run"]).casefold()
                contract = build_contracts(self.project, self.options["index"], session.run_id,
                    [run], [layer.layer_id for layer in session.layers], source_frame=self.frame)[run]

            full_photo = None

            def display_metadata(value):
                nonlocal full_photo
                if contract is not None:
                    from .photo_edge_continuation import verify_capture
                    verify_capture({contract["target_run"]: contract})
                value = frame_domain.attach_frame_domain(value, session, self.frame, layers)
                if live_drawing:
                    measurements = dict(value.measurements)
                    measurements[drawing_coverage.KEY] = dict(
                        display_only=True, explicit_outside_pixels=added,
                        stored_version_unchanged=True)
                    value = replace(value, measurements=measurements)
                value = value if contract is None else photo_continuation.attach(value, layers, contract)
                profile = self.options.get('full_photo_profile')
                if profile is not None:
                    from .photograph_io import is_camera_raw
                    if is_camera_raw(Path(value.photo_path)):
                        if full_photo is None:
                            self.progress.emit('Loading the complete lens-corrected RAW photograph')
                            from .raw_shape_basis_reader import reviewed_photo_path
                            physical_path = reviewed_photo_path(self.project, value.photo_path)
                            full_photo = self.options['full_photo_reader'](physical_path, profile)
                        if tuple(full_photo['camera_u_wh']) != tuple(value.native_shape_hw[::-1]):
                            raise ValueError('The full photograph and placed map have different camera grids')
                        value = replace(value, full_photo=full_photo)
                return value

            def partial(value):
                value = display_metadata(value)
                frame_models.verify_frame_model_selection(self.options.get("geometry_selection"))
                self.partial.emit((value, session))

            result = preview.prepare_preview(session, layers,
                geometry_index=self.options["index"], target_run=self.options["run"],
                config=self.options["config"], measure_offsets=True,
                photo_loader=self.options["photo_loader"], source_name=source_name,
                on_progress=self.progress.emit,
                on_preview=partial)
            result = display_metadata(result)
            frame_models.verify_frame_model_selection(self.options.get("geometry_selection"))
        except Exception as error:  # reported in the preview, no write fallback
            self.error = _plain_error(error)
            self.failed.emit(self.error)
            return
        self.result = result, session
        self.finished.emit((result, session))


class _SavedVersionWorker(QObject):
    finished = Signal()

    def __init__(self, project, current, record, frame, photo):
        super().__init__()
        self.project, self.current, self.record, self.frame, self.photo = project, current, record, frame, photo
        self.result, self.error = None, ""

    @Slot()
    def run(self):
        try:
            import cv2
            from .layer_painting_version_archive import load_drawing_version, _checked
            record = _checked(self.project, self.current.run_id, self.record, content=False)
            if record.get("kind") == "drawing":
                saved = load_drawing_version(self.project, self.current.run_id, record)
                layers = preview.drawing_layers(saved, self.frame)
            else:
                saved, layers = preview.preview_layers(self.current, None, [], self.frame, record=record)
            if (saved.run_id.casefold() != self.current.run_id.casefold()
                    or tuple(saved.image_size or ()) != tuple(self.current.image_size or ())
                    or not preview._same_path(saved.image_path, self.current.image_path)
                    or saved.image_stamp != self.current.image_stamp):
                raise ValueError("This version belongs to another reference photograph; use its original run.")
            height, width = self.photo.shape[:2]
            labels = cv2.resize(layers, (width, height), interpolation=cv2.INTER_NEAREST)
            self.result = (preview.RunPreview(saved.run_id.casefold(), self.photo, labels,
                labels == 0, saved.image_path, saved.run_id.casefold(),
                str(self.record.get("display_name") or self.record.get("name") or "Saved version")), saved)
        except Exception as error:
            self.error = _plain_error(error)
        finally:
            self.finished.emit()


class PaintingRunPreview(QObject):
    display_changed = Signal()
    reader_retired = Signal()

    def __init__(self, view, editable_column, canvas_class, *, run_inputs=None, set_reference=None):
        super().__init__(view)
        self.view, self.run_inputs, self.set_reference = view, run_inputs, set_reference
        self.thread = None
        self.worker = None
        self._pending = False
        self._active_key = self._waiting_key = self._result_key = None
        self._disposed = False
        self._started = None
        self._stage = ""
        self._loading_status = False
        self._retry_attempts = 0
        self._joining = False
        self.last_result = None
        self._last_session = None
        self._version_thread = self._version_worker = None
        self._version_requested = self._version_loading = None
        self._reference_version_key = None
        settings = QSettings(*_SETTINGS)
        settings.sync()
        self._index = str(settings.value(_INDEX, "") or "")
        self._geometry_selection = None
        self._index_error = ""
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Run"))
        self.runs = QComboBox()
        self.runs.setToolTip("The reference run is editable; other runs show a read-only placed preview")
        self.previous = QPushButton("Previous")
        self.next = QPushButton("Next")
        self.reference_button = QPushButton("Set reference…")
        self.geometry_button = QPushButton("Frame models…")
        for widget in (self.runs, self.previous, self.next, self.reference_button, self.geometry_button):
            row.addWidget(widget)
        layout.addLayout(row)
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Preview"))
        self.sources = QComboBox()
        self.sources.setToolTip("Preview an unfinished map, drawing or saved version; no approval is needed")
        source_row.addWidget(self.sources, 1)
        self.refresh_button = QPushButton("Refresh preview")
        source_row.addWidget(self.refresh_button)
        layout.addLayout(source_row)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setTextFormat(Qt.TextFormat.PlainText)
        self.note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.note)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        self.measurements = QTableWidget(0, 7)
        self.measurements.setHorizontalHeaderLabels(["Layers", "Median mm", "P10 mm", "P90 mm", "Δ reference mm", "Samples", "Status"])
        self.measurements.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.measurements.setMaximumHeight(180)
        self.measurements.hide()
        layout.addWidget(self.measurements)
        self.stack = QStackedWidget()
        self.stack.addWidget(editable_column)
        self.canvas = canvas_class()
        self.canvas.set_tool("pan")
        self.stack.addWidget(self.canvas)
        self._viewport = self.canvas.viewport()
        self.status_overlay = QWidget(self._viewport)
        self.status_overlay.setStyleSheet("background: #252525; color: white;")
        overlay_layout = QVBoxLayout(self.status_overlay)
        overlay_layout.addStretch(1)
        self.status_text = QLabel()
        self.status_text.setWordWrap(True)
        self.status_text.setTextFormat(Qt.TextFormat.PlainText)
        self.status_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        overlay_layout.addWidget(self.status_text)
        overlay_layout.addStretch(1)
        self.status_overlay.hide()
        self._viewport.installEventFilter(self)
        self._viewport.destroyed.connect(self._invalidate)
        self.canvas.destroyed.connect(self._invalidate)
        view.destroyed.connect(self._invalidate)
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._elapsed)
        self._retry_timer = QTimer(self)
        self._retry_timer.setSingleShot(True)
        self._retry_timer.setInterval(250)
        self._retry_timer.timeout.connect(self._retry_waiting)
        if hasattr(view, "busy_changed"):
            view.busy_changed.connect(self._parent_busy_changed)
        # These are the same controls as the working canvas uses. Keeping
        # them outside the stack makes display tools available for snapshots.
        for group in view._painting_photo_groups[:-1]:
            editable_column.layout().removeWidget(group)
            layout.addWidget(group)
        self.read_only_tools = QWidget()
        read_only_row = QHBoxLayout(self.read_only_tools)
        read_only_row.setContentsMargins(0, 0, 0, 0)
        self.read_only_note = QLabel("Read-only preview. Choose Current map to draw or edit; the saved version stays unchanged.")
        self.read_only_note.setWordWrap(True)
        self.edit_current_map = QPushButton("Edit Current map")
        self.edit_current_map.clicked.connect(self._edit_current_map)
        read_only_row.addWidget(self.read_only_note, 1)
        read_only_row.addWidget(self.edit_current_map)
        layout.addWidget(self.read_only_tools)
        self.read_only_tools.hide()
        layout.addWidget(self.stack, 1)
        display_controls = view._painting_photo_groups[-1]
        editable_column.layout().removeWidget(display_controls)
        layout.addWidget(display_controls)
        for control in (view.show_proposal, view.opacity, view.enhance, view.saturation):
            signal = control.toggled if hasattr(control, "toggled") else control.valueChanged
            signal.connect(self.redraw_display)
        self.runs.currentIndexChanged.connect(self._selected)
        self.sources.currentIndexChanged.connect(self._source_changed)
        self.previous.clicked.connect(lambda: self._step(-1))
        self.next.clicked.connect(lambda: self._step(1))
        self.geometry_button.clicked.connect(self.choose_geometry)
        self.reference_button.clicked.connect(self.change_reference)
        self.refresh_button.clicked.connect(self.request)
        self.navigation_keys = []
        for key, direction in (("Alt+Left", -1), ("Alt+Right", 1)):
            shortcut = QShortcut(key, self.widget)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda step=direction: self._step(step))
            self.navigation_keys.append(shortcut)
        self.previous.setToolTip("Previous run (Alt+Left)")
        self.next.setToolTip("Next run (Alt+Right)")

    def busy(self):
        return self.thread is not None or self._version_thread is not None

    def active_canvas(self):
        """Return the actual photo canvas targeted by shared display tools."""
        return self.canvas if self.stack.currentWidget() is self.canvas else self.view.canvas

    def _edit_current_map(self):
        reference = self.view.reference()
        index = self.runs.findData(reference.run_id) if reference is not None else -1
        blocked = self.runs.blockSignals(True)
        if index >= 0:
            self.runs.setCurrentIndex(index)
        self.runs.blockSignals(blocked)
        blocked = self.sources.blockSignals(True)
        self.sources.setCurrentIndex(0)
        self.sources.blockSignals(blocked)
        self._selected()
        self.view.show_reference_photo()

    def show_saved_version(self, record):
        self._refresh_sources()
        for index in range(self.sources.count()):
            source, value = self.sources.itemData(index) or (None, None)
            if source == "version" and value.get("folder") == record.get("folder"):
                blocked = self.sources.blockSignals(True)
                self.sources.setCurrentIndex(index)
                self.sources.blockSignals(blocked)
                break
        self._selected()

    def show_working_map(self):
        self.sources.setCurrentIndex(0)
        self._selected()

    def _reference_version(self):
        if getattr(self.view, '_workflow_scope', None) == 'analysis_boxes':
            return None
        source, record = self.sources.currentData() or ("map", None)
        return record if self.is_reference() and source == "version" else None

    def _show_reference_version(self, record):
        key = (str(self.view.project.root), self.view.session.run_id,
               record.get("folder"), record.get("sha256"))
        self._version_requested = key, deepcopy(record)
        self.stack.setCurrentWidget(self.canvas)
        self.view.set_reference_display_mode(False)
        if key == self._reference_version_key:
            if self._has_saved_result():
                # Navigation may have cleared the canvas without changing the
                # immutable snapshot key. Repaint those exact cached pixels.
                self._render_result(self.last_result, self._last_session)
                if self.canvas.preview is None and self.view.uses_full_reference_photo():
                    self.view._show_reference(reset_view=False)
                self._saved_photo_status()
                return
            # The same saved version may be reopened after its source locator
            # or live session identity changed. A stale cache is not a display.
            self._reference_version_key = None
        self.note.setText(f"Loading saved version {record.get('display_name') or record.get('name')}… Read only.")
        self.view.version_snapshot_note.setText(self.note.text() + " The saved map is loading; the working map is unchanged.")
        self.view._version_snapshot_session = None
        if self._version_thread is not None:
            return
        self.canvas.clear_photo()
        self._version_loading = key
        worker = _SavedVersionWorker(deepcopy(self.view.project), deepcopy(self.view.session),
            deepcopy(record), deepcopy(self.view.current_frame()), self.view._plain.copy())
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._version_retired)
        self._version_thread, self._version_worker = thread, worker
        thread.start()

    @Slot()
    def _version_retired(self):
        thread, worker = self._version_thread, self._version_worker
        if thread is None:
            return
        thread.wait()
        request = self._version_requested
        if (self._alive() and request is not None and request[0] == self._version_loading
                and self._reference_version() is not None and not self._joining):
            if worker.error:
                self.note.setText("Saved version unavailable: " + worker.error)
                self.view.version_snapshot_note.setText(self.note.text())
            elif worker.result is not None:
                result, session = worker.result
                self._render_result(result, session)
                self.last_result, self._last_session = result, session
                self._reference_version_key = request[0]
                self._result_key = self._key()
                self._saved_photo_status()
                self.view.show_version_metadata(session, request[1])
        self._version_thread = self._version_worker = None
        thread.deleteLater()
        if (not self._joining and self._alive() and self._version_requested is not None
                and self._version_requested[0] != self._version_loading and self._reference_version() is not None):
            self._show_reference_version(self._version_requested[1])
        if not self._joining and self._alive():
            self.reader_retired.emit()

    def _alive(self):
        return (not self._disposed and isValid(self) and isValid(self.view)
                and isValid(self.canvas) and isValid(self._viewport) and isValid(self.widget)
                and isValid(self.progress_bar))

    def _invalidate(self, *_args):
        self._disposed = True
        self._waiting_key = None
        self._last_session = None

    def _key(self):
        session = self.view.session
        source, record = self.sources.currentData() or ("map", None)
        version = None if record is None else (record.get("folder"), record.get("sha256"), record.get("kind"))
        return (id(self.view.project), id(session), getattr(session, "run_id", ""),
                getattr(session, "image_path", ""), tuple(getattr(session, "image_size", ()) or ()),
                getattr(session, "image_stamp", ""), getattr(session, "updated_utc", ""),
                id(getattr(self.view, "proposal", None)), str(self.runs.currentData() or "").casefold(),
                self._index, source, version, self._geometry_selection,
                self.view.reviewed_point_selection(self._geometry_selection))

    def _accepts_result(self):
        return (self._alive() and not self.is_reference() and self._active_key == self._key()
                and not self._pending)

    def _status(self, message, *, loading=False):
        if not self._alive():
            return
        self._stage = str(message)
        self._loading_status = loading
        self.progress_bar.setVisible(loading and not self.is_reference())
        self._elapsed(force=True)
        shown = self.last_result is not None and self._result_key == self._key()
        self.status_overlay.setGeometry(self._viewport.rect())
        self.status_overlay.setVisible(not shown and not self.is_reference())
        self.status_overlay.raise_()
        if loading:
            self._elapsed_timer.start()
        else:
            self._elapsed_timer.stop()

    def _elapsed(self, *, force=False):
        if not self._alive():
            if isValid(self._elapsed_timer):
                self._elapsed_timer.stop()
            return
        if not force and not self._loading_status:
            return
        elapsed = "" if self._started is None else f" · {int(time.monotonic()-self._started)} s"
        message = self._stage + elapsed
        self.note.setText(message)
        self.status_text.setText(message)

    def _cancel_waiting(self):
        self._waiting_key = None
        self._retry_timer.stop()

    def _parent_busy_changed(self, busy):
        if not busy and self._alive() and self._waiting_key is not None:
            self._retry_waiting()

    def _retry_waiting(self):
        if not self._alive():
            if isValid(self._retry_timer):
                self._retry_timer.stop()
            return
        if self.is_reference() or self._waiting_key != self._key():
            self._cancel_waiting()
            return
        if self.thread is not None:
            return
        if not self.view.busy():
            self._cancel_waiting()
            self.request()
        elif self._retry_attempts < 120:
            self._retry_attempts += 1
            self._retry_timer.start()
        else:
            self._status("The painting operation is still busy. The preview will resume when it finishes; "
                         "Refresh preview also retries.")

    def eventFilter(self, watched, event):
        if self._alive() and watched is self._viewport and event.type() == QEvent.Type.Resize:
            self.status_overlay.setGeometry(self._viewport.rect())
        return False

    def is_reference(self):
        session = self.view.session
        return session is None or str(self.runs.currentData() or "").casefold() == session.run_id.casefold()

    def ensure_saved_geometry_index(self):
        """Capture one series default or explicit USER override for all tools."""
        if not isValid(self) or self._disposed:
            return self._index
        if QThread.currentThread() != QObject.thread(self):
            raise RuntimeError("Frame-model settings must be read on the GUI thread.")
        current = self._geometry_selection
        override = self._index if self._index and (current is None
            or current.source == "user override" or current.index_path != self._index) else ""
        if not self._index or (current is not None and current.source == "series display binding"):
            settings = QSettings(*_SETTINGS)
            settings.sync()
            override = str(settings.value(_INDEX, "") or "")
        try:
            selected = frame_models.capture_frame_model_selection(self.view.project, override_path=override)
        except (AttributeError, TypeError, ValueError) as error:
            self._geometry_selection, self._index = None, ""
            self._index_error = str(error)
        else:
            self._geometry_selection, self._index = selected, selected.index_path
            self._index_error = ""
        return self._index

    def geometry_selection(self):
        self.ensure_saved_geometry_index()
        if self._geometry_selection is None:
            raise ValueError(self._index_error or "Frame models are unavailable. Retry after reopening the series.")
        return self.view.reviewed_point_selection(self._geometry_selection) or self._geometry_selection

    def refresh(self):
        self.ensure_saved_geometry_index()
        session = self.view.session
        reference = session.run_id if session else ""
        old = str(self.runs.currentData() or reference)
        try:
            runs = [row.run_id for row in self.run_inputs()] if self.run_inputs else self.view._series_runs()
        except (OSError, ValueError, KeyError):
            runs = []
        if reference and reference not in runs:
            runs.insert(0, reference)
        from .map_placement import review_order
        first = [part.strip() for part in str(QSettings(*_SETTINGS).value(_FIRST, "") or "").split(",") if part.strip()]
        runs = review_order(runs, reference, first)
        self.runs.blockSignals(True)
        self.runs.clear()
        for run in runs:
            self.runs.addItem(run + (" — reference (editable)" if run.casefold() == reference.casefold() else " — read only"), run)
        self.runs.setCurrentIndex(max(0, self.runs.findData(old)))
        self.runs.blockSignals(False)
        self._refresh_sources()
        self._selected()

    def _refresh_sources(self):
        old = self.sources.currentData()
        self.sources.blockSignals(True)
        self.sources.clear()
        self.sources.addItem("Current map (drawing when needed)", ("map", None))
        self.sources.addItem("Current drawing", ("drawing", None))
        for record in reversed(self.view.versions):
            self.sources.addItem(str(record.get("display_name") or record.get("name") or "Saved version"),
                                 ("version", dict(record)))
        index = next((i for i in range(self.sources.count()) if self.sources.itemData(i) == old), 0)
        self.sources.setCurrentIndex(index)
        self.sources.blockSignals(False)

    def _step(self, direction):
        self.runs.setCurrentIndex(max(0, min(self.runs.count()-1, self.runs.currentIndex()+direction)))

    def _selected(self, *_args):
        if not self._alive():
            return
        if getattr(self.view, '_workflow_scope', None) == 'analysis_boxes':
            self.view._show_analysis_boxes_editor()
            return
        reference = self.is_reference()
        record = self._reference_version()
        saved_reference = record is not None and self.view.image is not None
        self.read_only_tools.setVisible((saved_reference or not reference)
            and getattr(self.view, '_workflow_scope', None) != 'rig_frame'
            and not getattr(self.view, '_transfer_review_active', False))
        source, _source_record = self.sources.currentData() or ("map", None)
        for note in (self.view.working_layers_note, self.view.working_facies_note):
            note.setVisible(source == "version")
        for shortcut in self.view.findChildren(QShortcut):
            if shortcut not in self.navigation_keys:
                shortcut.setEnabled(reference and not saved_reference)
        self.measurements.setVisible(not reference and self.last_result is not None)
        self.reference_button.setEnabled(not reference and self.set_reference is not None and not self.view.busy())
        self.previous.setEnabled(self.runs.currentIndex() > 0)
        self.next.setEnabled(self.runs.currentIndex() < self.runs.count()-1)
        self.refresh_button.setEnabled(not reference)
        if saved_reference:
            self._cancel_waiting()
            self._pending = False
            self._elapsed_timer.stop()
            self._loading_status = False
            self.progress_bar.hide()
            self.measurements.hide()
            self.status_overlay.hide()
            self._show_reference_version(record)
            return
        self._version_requested = None
        self._reference_version_key = None
        self.stack.setCurrentIndex(0 if reference else 1)
        self.view.set_reference_display_mode(reference)
        if reference:
            self._cancel_waiting()
            self._pending = False
            self._elapsed_timer.stop()
            self._loading_status = False
            self.progress_bar.hide()
            self.status_overlay.hide()
            self.note.setText("Reference run: editable. Viewing other runs keeps this drawing on its own photograph.")
        else:
            self.request()
        self.display_changed.emit()

    def _source_changed(self, *_args):
        source, record = self.sources.currentData() or ("map", None)
        if source == "version":
            table = self.view.versions_table
            for row in range(table.rowCount()):
                if table.item(row, 0).data(Qt.ItemDataRole.UserRole) == record.get("folder"):
                    table.selectRow(row)
                    break
        self._selected()

    def choose_geometry(self):
        path, _ = QFileDialog.getOpenFileName(self.view, "Measured frame models (index.json)",
                                             self._index, "Geometry index (index.json);;JSON (*.json)")
        if path:
            self._index = path
            self._geometry_selection = frame_models.capture_frame_model_selection(self.view.project, override_path=path)
            self._index_error = ""
            settings = QSettings(*_SETTINGS)
            settings.setValue(_INDEX, path)
            settings.sync()
            self.request()
            self.display_changed.emit()

    def request(self):
        if not self._alive() or self.is_reference():
            return
        self.ensure_saved_geometry_index()
        key = self._key()
        if self.thread is not None:
            self._pending = key != self._active_key
            if self._pending:
                self._started = time.monotonic()
                self._status("Waiting for the previous preview before loading " + str(self.runs.currentData()), loading=True)
            else:
                self._status(self._stage or "This preview is already loading", loading=True)
            return
        self._started = time.monotonic()
        if self.view.busy():
            self._waiting_key = key
            self._retry_attempts = 0
            self._status("Waiting for the current painting operation before loading " + str(self.runs.currentData()),
                         loading=True)
            self._retry_timer.start()
            return
        self._cancel_waiting()
        self._pending = False
        self.canvas.clear_photo()
        self.last_result = None
        self._last_session = None
        self._result_key = None
        self.display_changed.emit()
        self.measurements.setRowCount(0)
        self.measurements.hide()
        if not self._index:
            self._status(self._index_error or "The series has no measured frame models. Use Frame models to set an override.")
            return
        session = self.view.session
        if session is None:
            self._status("Choose a reference run with a drawing first.")
            return
        try:
            source, record = self.sources.currentData() or ("map", None)
            if source == "map" and self.view._map_kind() == "manual map":
                source = "manual"
            options = {"run": str(self.runs.currentData()), "index": self._index,
                       "geometry_selection": self._geometry_selection,
                       "source": source, "record": deepcopy(record), "name": self.sources.currentText(),
                       "continue_photo_edges": True,
                       "config": deepcopy(self.view._workspace_frame_config),
                       "photo_loader": self.view._reference_photo_loader(),
                       "full_photo_profile": (lp.run_camera_u(self.view.project, str(self.runs.currentData()))
                            if self.view._workflow_scope == 'layers' else None),
                       "full_photo_reader": self.view._rig_photo_loader.reader}
            selected=self.geometry_selection()
            options.update(index=selected.index_path,geometry_selection=selected)
            worker = _PreviewWorker(deepcopy(self.view.project), deepcopy(session),
                                    self.view.proposal, deepcopy(self.view.corrections),
                                    deepcopy(self.view.current_frame()), options)
        except Exception as error:
            self._status(f"Preview unavailable: {_plain_error(error)}")
            return
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._progress)
        worker.partial.connect(self._partial)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.failed.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._retired)
        self.thread, self.worker = thread, worker
        self._active_key = key
        self.sources.setEnabled(False)
        self.reference_button.setEnabled(False)
        self._status("Loading " + options["run"] + " preview… Read only; no files are written.", loading=True)
        thread.start()

    @Slot(str)
    def _progress(self, message):
        if self.worker is not None and self.sender() is self.worker and self._accepts_result():
            self._status(message, loading=True)

    @Slot(object)
    def _partial(self, value):
        if self.worker is not None and self.sender() is self.worker and self._accepts_result():
            self._finished(value, partial=True)

    @Slot(object)
    def _finished(self, value, *, partial=False):
        if not self._accepts_result():
            return
        try:
            result, session = value
            if (result.run_id != str(self.runs.currentData() or "").casefold()
                    or result.source_run != self.view.session.run_id.casefold()):
                raise ValueError("The preview result belongs to another selected photograph")
            self._render_result(result, session)
            source, record = self.sources.currentData() or ("map", None)
            if source == "version":
                self.view.show_version_metadata(session, record)
        except Exception as error:
            self._failed(f"Could not display the preview: {_plain_error(error)}")
            return
        self.last_result = result
        self._last_session = session
        self._result_key = self._active_key
        self.status_overlay.hide()
        if partial:
            self._status("Photo and placed map shown; checking boundary measurements…", loading=True)
        else:
            self._elapsed_timer.stop()
            self._loading_status = False
            self.progress_bar.hide()
        self.display_changed.emit()

    def _render_result(self, result, session):
        photo_ready = True
        if result.full_photo is not None:
            self._show_full_photo(result)
        elif self.is_reference() and self.view.uses_full_reference_photo():
            photo_ready = self.view.show_full_reference_preview(self.canvas, result.run_id)
            if not photo_ready:
                self.canvas.clear_photo()
        else:
            self.canvas.set_photo(result.photo, (result.photo.shape[1], result.photo.shape[0]),
                                  token={"path": result.photo_path, "run_id": result.run_id}, reset_view=True)
        if photo_ready:
            self._render_overlay(result, session)
        offsets = result.measurements.get("layer_offsets") or {}
        rows = offsets.get("boundaries") or []
        self.measurements.setRowCount(len(rows))
        for index, row in enumerate(rows):
            values = (" / ".join(map(str, row["labels"])), row.get("median_mm"), row.get("p10_mm"),
                      row.get("p90_mm"), row.get("change_from_reference_mm"),
                      f"{row.get('measured_sample_count', 0)}/{row.get('sample_count', 0)}", row.get("status", ""))
            for column, value in enumerate(values):
                self.measurements.setItem(index, column, QTableWidgetItem("—" if value is None else str(value)))
        self.measurements.resizeColumnsToContents()
        self.measurements.setVisible(bool(rows))
        shift = result.measurements.get("offset_px") or {}
        checks = result.measurements.get("checks") or {}
        domain = result.measurements.get(frame_domain.KEY) or {}
        current_coverage = result.measurements.get(drawing_coverage.KEY) or {}
        continued = result.measurements.get(photo_continuation.KEY) or {}
        if domain.get("status") == "available":
            gap_note = (f"Inside-frame uncovered: {domain['inside_uncovered_pixels']:,} px "
                        f"({domain['inside_uncovered_cm2']:.4g} cm²). ")
            gap_note += ("Explicit drawing outside that frame is shown as context; the gap statistics keep the placed reference-frame domain. "
                         if current_coverage else "Outside the rig frame is transparent. ")
        else:
            gap_note = "Inside-frame gaps unavailable: " + str(domain.get("reason") or "The reference rig frame is not bound.") + " "
        if current_coverage:
            gap_note += "Areas outside the reference photo cannot be covered by extending polygons inside it. Saved manual versions keep their original map. "
        if continued:
            gap_note = (f"Target own frame ({continued['authority']}): {continued['inside_gap_pixels']:,} true gap pixels "
                f"({continued['inside_uncovered_cm2']:.4g} cm²); {continued['continued_pixels']:,} pixels continued beyond the reference photo (cyan hatch). "
                "The original filled/support provenance is retained. The separate placed-reference-frame diagnostic is unchanged. ")
        self.note.setText(f"{result.run_id}: {result.source_name} from {result.source_run}, placed through the saved frame models. "
                          "Read only · no approval, placement receipt or adoption is created. "
                          + ("Magenta: a true gap inside the target's own rig frame. " if continued
                             else "Magenta: uncovered inside the placed reference rig frame. ")
                          +
                          f"Displacement: median {shift.get('median_px', '—')} px, maximum {shift.get('max_px', '—')} px. "
                          + gap_note + f"Water pockets: {checks.get('water_pocket_pixels', '—')} px. "
                          "Boundary offsets use the placement review's descriptive method.")

    def _show_full_photo(self, result, *, reset_view=True):
        from .series_layer_view import photo_with_saturation
        packet = result.full_photo
        full = packet['image_rgb']
        display = packet['display_enhanced'] if self.view.enhance.isChecked() else packet['display_plain']
        display = photo_with_saturation(display, self.view.saturation.value() / 100.)
        self.canvas.set_photo(display, tuple(packet['camera_u_wh']),
            token=dict(packet, path=result.photo_path, run_id=result.run_id), reset_view=reset_view,
            detail=full, display_mapping=packet['mapping'], display_size=(full.shape[1], full.shape[0]),
            valid_support=packet['valid_support'])

    def refresh_full_reference_photo(self):
        """A completing full-photo read also updates an already shown saved version."""
        if self._alive() and self._has_saved_result():
            if self.view.show_full_reference_preview(self.canvas, self.last_result.run_id, reset_view=False):
                self._render_overlay(self.last_result, self._last_session)
            elif self.view.uses_full_reference_photo():
                self.canvas.clear_photo()
            self._saved_photo_status()

    def _has_saved_result(self):
        record = self._reference_version()
        if record is None or self.last_result is None or self._last_session is None:
            return False
        key = (str(self.view.project.root), self.view.session.run_id,
               record.get('folder'), record.get('sha256'))
        return (self._reference_version_key == key and self._version_requested is not None
                and self._version_requested[0] == key
                and self.last_result.run_id == self.view.session.run_id.casefold()
                and self._last_session.run_id.casefold() == self.view.session.run_id.casefold()
                and preview._same_path(self._last_session.image_path, self.view.session.image_path)
                and self._last_session.image_stamp == self.view.session.image_stamp
                and tuple(self._last_session.image_size or ()) == tuple(self.view.session.image_size or ()))

    def _saved_photo_status(self):
        """Photo completion replaces its loading/error note for this snapshot only."""
        if not self._has_saved_result():
            return
        if self.view.uses_full_reference_photo() and self.canvas.preview is None:
            self.note.setText(self.view.photo_note.text())
            return
        record = self._reference_version()
        self.note.setText(f"{self.last_result.source_name} · saved {record.get('kind')} · read only. "
            "Saved filled-map preview; choose Edit Current map to draw or edit the working map. "
            "The saved version stays unchanged. Restore version to edit this snapshot.")

    def _render_overlay(self, result, session):
        """Recolour the cached categorical display without reloading its photo."""
        rgba = np.zeros((*result.layers.shape, 4), dtype=np.uint8)
        style = getattr(self.view, "display_layer_styles", None)
        if callable(style):
            colors, hidden = style(session)
        else:
            colors, hidden = {layer.layer_id: QColor(layer.color) for layer in session.layers}, frozenset()
        for layer in session.layers:
            if layer.layer_id in hidden:
                continue
            color = QColor(colors.get(layer.layer_id, layer.color))
            rgba[result.layers == layer.layer_id] = (color.red(), color.green(), color.blue(),
                round(255 * self.view.opacity.value() / 100))
        # New positive packets distinguish authorized continuation from true
        # gaps; historical packets retain their original frame diagnostic.
        domain = result.measurements.get(frame_domain.KEY) or {}
        continued = result.measurements.get(photo_continuation.KEY) or {}
        if continued:
            inside, gaps, filled = continued["display_inside"], continued["display_gaps"], continued["display_continued"]
            if any(mask.shape != result.layers.shape for mask in (inside, gaps, filled)):
                raise ValueError("The continued preview masks belong to another display grid.")
            outside = ~inside
            if result.measurements.get(drawing_coverage.KEY):
                outside &= result.uncovered | (result.layers == lp.OUTSIDE)
            rgba[outside] = (0, 0, 0, 0)
            rows, columns = np.indices(result.layers.shape)
            hatch = ((rows + columns) // 3) % 2 == 0
            rgba[filled & hatch & (rgba[..., 3] > 0)] = (0, 220, 255, 150)
            rgba[gaps] = (255, 0, 255, 95)
        elif domain.get("status") == "available":
            inside, gaps = domain["display_inside"], domain["display_gaps"]
            if inside.shape != result.layers.shape or gaps.shape != result.layers.shape:
                raise ValueError("The displayed drawing-frame masks belong to another photograph grid.")
            if result.measurements.get(drawing_coverage.KEY):
                # Only actual live drawing coverage may remain as context.
                # Edge replication beyond the source photo is never coverage.
                rgba[~inside & (result.uncovered | (result.layers == lp.OUTSIDE))] = (0, 0, 0, 0)
            else:
                rgba[~inside] = (0, 0, 0, 0)
            # A hidden facies cannot hide a genuinely uncovered pixel.
            rgba[gaps] = (255, 0, 255, 95)
        else:
            # Unknown frame authority never becomes an all-photograph rig.
            rgba[result.uncovered | (result.layers == lp.OUTSIDE)] = (0, 0, 0, 0)
        self.canvas.set_proposal(rgba)
        self.canvas.set_overlays_visible(False, self.view.show_proposal.isChecked())

    @Slot()
    def redraw_display(self):
        """Update only the current cached overlay; retain progress, errors and zoom."""
        if not self._alive() or (self.is_reference() and self._reference_version() is None) or self._pending:
            return
        result, session = self.last_result, self._last_session
        if (result is None or session is None
                or (not self._has_saved_result() and self._result_key != self._key())):
            return
        if self.is_reference() and self.view.uses_full_reference_photo() and self.canvas.preview is None:
            return
        if (result.run_id != str(self.runs.currentData() or "").casefold()
                or self.view.session is None or result.source_run != self.view.session.run_id.casefold()):
            return
        try:
            if result.full_photo is not None:
                self._show_full_photo(result, reset_view=False)
            elif self.is_reference() and self.view.uses_full_reference_photo():
                self.view.show_full_reference_preview(self.canvas, result.run_id, reset_view=False)
            self._render_overlay(result, session)
        except Exception as error:
            self._status(self.note.text() + f" Display colours unavailable: {_plain_error(error)}",
                         loading=self._loading_status)

    @Slot(str)
    def _failed(self, message):
        if self._accepts_result():
            retained = self.last_result is not None and self._result_key == self._active_key
            self._status(("Photo and placed map retained; preview update unavailable: " if retained
                          else "Preview unavailable: ") + message)
            self.display_changed.emit()

    @Slot()
    def _retired(self):
        self._retire(self.sender())

    def _retire(self, thread):
        if thread is None or thread is not self.thread:
            return
        if thread is not None:
            thread.wait()
        worker = self.worker
        if self._accepts_result():
            if worker.error:
                self._failed(worker.error)
            elif worker.result is not None:
                self._finished(worker.result)
            else:
                self._failed("The preview worker stopped without a result.")
        # Keep the worker alive until its thread has finished, then drop its
        # last owned reference here in the GUI thread (the shared worker rule).
        self.thread = self.worker = None
        del worker
        if thread is not None and isValid(thread):
            thread.deleteLater()
        if not self._alive():
            return
        self._elapsed_timer.stop()
        self._loading_status = False
        self.progress_bar.hide()
        self.sources.setEnabled(True)
        self.reference_button.setEnabled(not self.is_reference() and self.set_reference is not None)
        if not self._joining and (self._pending or (not self.is_reference() and self._active_key != self._key())):
            self._pending = False
            self.request()
        if not self._joining:
            self.display_changed.emit()
            self.reader_retired.emit()

    def wait(self):
        self._cancel_waiting()
        self._pending = False
        self._active_key = None
        self._elapsed_timer.stop()
        self._loading_status = False
        if isValid(self.progress_bar):
            self.progress_bar.hide()
        self._joining = True
        try:
            self._version_requested = None
            if self._version_thread is not None:
                self._version_retired()
            if self.thread is not None:
                self._retire(self.thread)
        finally:
            self._joining = False

    def change_reference(self):
        if self.is_reference() or self.view.busy() or self.busy() or self.set_reference is None:
            return
        old = self.view.session.run_id if self.view.session else ""
        target = str(self.runs.currentData() or "")
        answer = QMessageBox.question(self.view, "Set reference run",
            f"Change the reference from {old} to {target}?\n\nThe drawing on {old} stays on that run's photograph. "
            f"The drawing belonging to {target} will open; no marks are copied between photographs.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.set_reference(target)
