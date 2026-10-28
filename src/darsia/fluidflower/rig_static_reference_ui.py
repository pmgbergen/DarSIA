"""Inline measured lines and point references, with shared run observations."""
from __future__ import annotations

from copy import deepcopy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import importlib
import math
import threading
import uuid

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QGraphicsPathItem,
    QGraphicsSimpleTextItem, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout,
)
from shiboken6 import isValid


def _backend():
    return importlib.import_module("darsia.fluidflower.rig_static_references")


def _guides():
    return importlib.import_module("darsia.fluidflower.rig_guide_lines")


def _measurements():
    return importlib.import_module("darsia.fluidflower.rig_measurement_lines")


def _spin():
    spin = QDoubleSpinBox()
    spin.setRange(-100000., 100000.)
    spin.setDecimals(3)
    spin.setSuffix(" mm")
    return spin


def _identity(packet):
    source = packet.get("source_raw") or {}
    profile = packet.get("profile") or {}
    return (str(packet.get("run_id", "")), str(source.get("path", "")),
            source.get("sha256"), source.get("size_bytes"),
            profile.get("sha256"), profile.get("profile_id"))


class _Recognize(QObject):
    finished = Signal()
    progress = Signal(int, int, str)

    def __init__(self, document, runs, reader):
        super().__init__()
        self.document, self.runs, self.reader = document, runs, reader
        self.cancelled = threading.Event()
        self.results, self.errors = {}, []

    @Slot()
    def run(self):
        def recognize(reference):
            run_id = str(reference.run_id)
            try:
                # Read and release each full RAW independently; never retain a cohort.
                packet = self.reader(reference)
                if str(packet.get("run_id")) != run_id:
                    raise ValueError("The loaded photograph belongs to another run")
                if not self.cancelled.is_set():
                    result = _backend().recognize_regions(self.document, packet)
                    if str(result.get("run_id")) != run_id:
                        raise ValueError("The recognition result belongs to another run")
                    return run_id, result, ""
                del packet
            except Exception as error:
                return run_id, None, str(error)
            return run_id, None, "Cancelled"

        # At most two RAW packets are live. Pending runs are not read on cancellation.
        iterator = iter(self.runs)
        number = 0
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="static-reference") as executor:
            pending = set()
            for _ in range(2):
                reference = next(iterator, None)
                if reference is not None:
                    pending.add(executor.submit(recognize, reference))
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    run_id, result, error = future.result()
                    if result is not None:
                        self.results[run_id] = result
                    if error:
                        self.errors.append(f"{run_id}: {error}")
                        self.results[run_id] = dict(run_id=run_id, observations={}, model=None,
                            status="failed", error=error)
                    number += 1
                    self.progress.emit(number, len(self.runs), run_id)
                    reference = None if self.cancelled.is_set() else next(iterator, None)
                    if reference is not None:
                        pending.add(executor.submit(recognize, reference))
        self.finished.emit()


class _LoadReferences(QObject):
    finished = Signal()

    def __init__(self, project_root):
        super().__init__()
        self.project_root, self.result, self.error = project_root, None, ""
        self.cancelled = threading.Event()

    @Slot()
    def run(self):
        try:
            self.result = _backend().load_references(self.project_root)
        except Exception as error:
            self.error = str(error)
        self.finished.emit()


class _SaveReferences(QObject):
    finished = Signal()

    def __init__(self, project_root, document, series_root=None):
        super().__init__()
        self.project_root, self.document, self.series_root = project_root, document, series_root
        self.result, self.error = None, ""
        self.cancelled = threading.Event()

    @Slot()
    def run(self):
        try:
            from .series_decisions import series_actor
            self.result = _backend().save_references(self.project_root, self.document,
                actor=series_actor(self.series_root),
                reason="Static references, measured lines and run observations updated")
        except Exception as error:
            self.error = str(error)
        self.finished.emit()


class StaticReferencePanel(QGroupBox):
    """Edit references explicitly; recognition and frame use never adopt science."""

    changed = Signal()
    busy_changed = Signal(bool)

    def __init__(self, rig_panel):
        super().__init__("Static reference areas", rig_panel)
        self.rig_panel, self.page, self.canvas = rig_panel, rig_panel.page, rig_panel.canvas
        self._thread = self._worker = None
        self._disposed = self._loading = False
        self._mode, self._clicks, self._items = "", [], []
        self._interaction_line_id = None
        self._interaction_key = self._work_key = self._last_context = None
        self._generation = 0
        self._loaded = False
        self._operation = ""
        self.dirty = False
        self.document = _backend().empty_document()
        self.document["measurement_uncertainty_mm"] = 2.

        layout = QVBoxLayout(self)
        run_row = QHBoxLayout()
        run_row.addWidget(QLabel("Source / run"))
        self.run_combo = QComboBox()
        self.run_combo.setToolTip("Inspect and correct this run's reference points and measurement lines on its own RAW photograph.")
        run_row.addWidget(self.run_combo, 1)
        layout.addLayout(run_row)
        self.measurement_box = QGroupBox("Measurement lines")
        measurement_layout = QVBoxLayout(self.measurement_box)
        measurement_help = QLabel("Choose two image endpoints and enter their measured physical length in mm. "
                                 "Select a saved line to give its first endpoint physical x/y and its direction. "
                                 "Positioned lines can define the rig frame; no redrawing is needed.")
        measurement_help.setWordWrap(True)
        measurement_layout.addWidget(measurement_help)
        measurement_row = QHBoxLayout()
        self.measurement_name = QLineEdit("Measurement 1")
        self.measurement_name.setPlaceholderText("Measurement name")
        self.measurement_length_mm = _spin()
        self.measurement_length_mm.setRange(0., 1000000.)
        self.measurement_length_mm.setToolTip("Physical distance between the two endpoints; 0 means needs measurement.")
        self.measurement_plane = QComboBox()
        self.measurement_plane.setEditable(True)
        self.measurement_plane.addItems(["rig", "bolt-heads"])
        for label, widget in (("Name", self.measurement_name), ("Length", self.measurement_length_mm),
                              ("Plane", self.measurement_plane)):
            measurement_row.addWidget(QLabel(label))
            measurement_row.addWidget(widget)
        measurement_layout.addLayout(measurement_row)
        measurement_row = QHBoxLayout()
        self.add_measurement_button = QPushButton("Add measurement line (2 clicks)")
        self.add_measurement_button.setCheckable(True)
        self.redraw_measurement_button = QPushButton("Redraw selected (2 clicks)")
        self.redraw_measurement_button.setCheckable(True)
        self.redraw_measurement_button.setToolTip("Correct both image endpoints in the selected run.")
        self.remove_measurement_button = QPushButton("Remove selected")
        for button in (self.add_measurement_button, self.redraw_measurement_button, self.remove_measurement_button):
            measurement_row.addWidget(button)
        measurement_layout.addLayout(measurement_row)
        self.measurement_table = QTableWidget(0, 6)
        self.measurement_table.setHorizontalHeaderLabels(["Name", "Length [mm]", "Plane", "Status", "Score", "ID"])
        self.measurement_table.setColumnHidden(5, True)
        self.measurement_table.horizontalHeader().setMinimumSectionSize(40)
        self.measurement_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 5):
            self.measurement_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.measurement_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.measurement_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.measurement_table.verticalHeader().hide()
        self.measurement_table.setMaximumHeight(132)
        self.measurement_table.itemChanged.connect(self._measurement_table_edited)
        self.measurement_table.itemSelectionChanged.connect(self._measurement_selection_changed)
        measurement_layout.addWidget(self.measurement_table)
        self.placement_box = QGroupBox("Physical placement of selected line")
        placement_layout = QVBoxLayout(self.placement_box)
        self.placement_note = QLabel()
        self.placement_note.setWordWrap(True)
        placement_layout.addWidget(self.placement_note)
        placement_row = QHBoxLayout()
        self.placement_x_mm, self.placement_y_mm = _spin(), _spin()
        for label, widget in (("Start x", self.placement_x_mm), ("Start y", self.placement_y_mm)):
            placement_row.addWidget(QLabel(label))
            placement_row.addWidget(widget)
        placement_layout.addLayout(placement_row)
        placement_row = QHBoxLayout()
        self.placement_direction = QComboBox()
        for label, direction in (("Choose physical direction", None), ("+x (right)", "x+"),
                                 ("−x (left)", "x-"), ("+y (up)", "y+"), ("−y (down)", "y-")):
            self.placement_direction.addItem(label, direction)
        self.placement_direction.setToolTip("Direction from the first clicked endpoint to the second in physical rig coordinates.")
        placement_row.addWidget(self.placement_direction, 1)
        self.apply_placement_button = QPushButton("Apply placement")
        self.clear_placement_button = QPushButton("Clear placement")
        placement_row.addWidget(self.apply_placement_button)
        placement_row.addWidget(self.clear_placement_button)
        placement_layout.addLayout(placement_row)
        measurement_layout.addWidget(self.placement_box)
        layout.addWidget(self.measurement_box)
        # Retain old annotations and readers; the direct four-point reference
        # editor replaces the separate measured-line workflow.
        self.measurement_box.hide()
        guide_box = QGroupBox("Support lines")
        guide_layout = QVBoxLayout(guide_box)
        guide_row = QHBoxLayout()
        self.guide_mode = QComboBox()
        for label, mode in (("Through points", "through"), ("Horizontal", "horizontal"),
                            ("Vertical", "vertical")):
            self.guide_mode.addItem(label, mode)
        self.guide_mode.setToolTip("Horizontal and vertical lines use the first point's height or column.")
        self.guide_extend = QCheckBox("Extend across photograph")
        self.guide_extend.setChecked(True)
        guide_row.addWidget(self.guide_mode)
        guide_row.addWidget(self.guide_extend)
        guide_layout.addLayout(guide_row)
        guide_row = QHBoxLayout()
        self.draw_guide_button = QPushButton("Draw line (2 clicks)")
        self.draw_guide_button.setCheckable(True)
        self.undo_guide_button = QPushButton("Undo line")
        self.clear_guides_button = QPushButton("Clear lines")
        self.guides_visible = QCheckBox("Show guides")
        self.guides_visible.setChecked(True)
        for widget in (self.draw_guide_button, self.undo_guide_button,
                       self.clear_guides_button, self.guides_visible):
            guide_row.addWidget(widget)
        guide_layout.addLayout(guide_row)
        guide_row = QHBoxLayout()
        guide_row.addWidget(QLabel("Select line"))
        self.guide_selection = QComboBox()
        self.guide_selection.setToolTip("Choose a support line in this photograph. The selected line has a thicker dashed stroke.")
        guide_row.addWidget(self.guide_selection, 1)
        self.remove_guide_button = QPushButton("Remove selected guide")
        guide_row.addWidget(self.remove_guide_button)
        guide_layout.addLayout(guide_row)
        layout.addWidget(guide_box)
        self.point_area_toggle = QToolButton()
        self.point_area_toggle.setText("Point / area references")
        self.point_area_toggle.setCheckable(True)
        self.point_area_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.point_area_toggle.setArrowType(Qt.ArrowType.RightArrow)
        layout.addWidget(self.point_area_toggle)
        self.point_area_box = QGroupBox("Point / area references")
        point_layout = QVBoxLayout(self.point_area_box)
        help_text = QLabel("Define an area with two opposite corners, then click its physical anchor. "
            "Enter measured x/y in one shared coordinate system for all runs. Only points in the "
            "same physical plane can define a model; frame corners and bolt heads may have different depths.")
        help_text.setWordWrap(True)
        point_layout.addWidget(help_text)
        self.point_area_box.setVisible(False)
        self.point_area_toggle.toggled.connect(self._toggle_point_areas)
        layout.addWidget(self.point_area_box)
        row = QHBoxLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("Reference name")
        self.x_mm, self.y_mm = _spin(), _spin()
        self.plane = QComboBox()
        self.plane.setEditable(True)
        self.plane.addItems(["rig", "bolt-heads"])
        for label, widget in (("Name", self.name),):
            row.addWidget(QLabel(label))
            row.addWidget(widget)
        point_layout.addLayout(row)
        row = QHBoxLayout()
        for label, widget in (("x", self.x_mm), ("y", self.y_mm)):
            row.addWidget(QLabel(label))
            row.addWidget(widget)
        point_layout.addLayout(row)
        uncertainty_row = QHBoxLayout()
        self.uncertainty_mm = _spin()
        self.uncertainty_mm.setRange(0., 20.)
        self.uncertainty_mm.setValue(2.)
        self.uncertainty_mm.setToolTip("Uncertainty of the physical x/y measurements. Image matching scores remain independent.")
        uncertainty_row.addWidget(QLabel("Measurement uncertainty ±"))
        uncertainty_row.addWidget(self.uncertainty_mm)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Name", "x [mm]", "y [mm]", "Plane", "Status", "Score", "ID"])
        self.table.setColumnHidden(6, True)
        self.table.horizontalHeader().setMinimumSectionSize(40)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 6):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().hide()
        self.table.setMaximumHeight(190)
        self.table.itemChanged.connect(self._table_edited)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        point_layout.addWidget(self.table)
        row = QHBoxLayout()
        self.define_button = QPushButton("Define area (3 clicks)")
        self.correct_button = QPushButton("Correct point")
        self.define_button.setCheckable(True)
        self.correct_button.setCheckable(True)
        self.remove_button = QPushButton("Remove area")
        self.update_button = QPushButton("Update selected measurements")
        for button in (self.define_button, self.correct_button):
            row.addWidget(button)
        point_layout.addLayout(row)
        row = QHBoxLayout()
        for button in (self.update_button, self.remove_button):
            row.addWidget(button)
        point_layout.addLayout(row)
        self.frame_model_box = QGroupBox("Rig frame from references")
        model_layout = QVBoxLayout(self.frame_model_box)
        layout.addWidget(self.frame_model_box)
        model_help = QLabel("Use one shared coordinate system: x to the right and y upwards. "
                            "Coordinates may start at the rig's inner bottom-left (0, 0). "
                            "Only references in the same physical plane are combined.")
        model_help.setWordWrap(True)
        model_layout.addWidget(model_help)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Physical plane"))
        model_row.addWidget(self.plane, 1)
        model_layout.addLayout(model_row)
        model_layout.addLayout(uncertainty_row)
        row = QHBoxLayout()
        self.find_button = QPushButton("Find in this run")
        self.find_all_button = QPushButton("Find in all runs")
        self.load_button = QPushButton("Load saved references")
        self.save_button = QPushButton("Save references")
        for button in (self.find_button, self.find_all_button):
            row.addWidget(button)
        layout.addLayout(row)
        row = QHBoxLayout()
        for button in (self.load_button, self.save_button):
            row.addWidget(button)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.origin_x_mm, self.origin_y_mm = _spin(), _spin()
        row.addWidget(QLabel("Rig bottom-left origin x/y"))
        row.addWidget(self.origin_x_mm)
        row.addWidget(self.origin_y_mm)
        model_layout.addLayout(row)
        self.use_button = QPushButton("Use for this run's rig frame")
        self.use_button.setToolTip("Use the fitted plane, the origin above and the existing inner width/height "
                                   "for the workspace rig frame. This does not adopt scientific geometry.")
        model_layout.addWidget(self.use_button)
        self.model_note = QLabel()
        self.model_note.setWordWrap(True)
        model_layout.addWidget(self.model_note)
        self.save_state = QLabel()
        self.save_state.setWordWrap(True)
        layout.addWidget(self.save_state)
        self.note = QLabel("Add a measurement line, enter its length in mm, then find it in each run.")
        self.note.setWordWrap(True)
        self.note.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.note)
        self.define_button.toggled.connect(lambda active: self._start("define", active))
        self.correct_button.toggled.connect(lambda active: self._start("correct", active))
        self.draw_guide_button.toggled.connect(lambda active: self._start("guide", active))
        self.add_measurement_button.toggled.connect(lambda active: self._start("measurement", active))
        self.redraw_measurement_button.toggled.connect(lambda active: self._start("measurement-redraw", active))
        self.remove_measurement_button.clicked.connect(self._remove_measurement)
        self.apply_placement_button.clicked.connect(self._apply_line_placement)
        self.clear_placement_button.clicked.connect(self._clear_line_placement)
        self.undo_guide_button.clicked.connect(self._undo_guide)
        self.clear_guides_button.clicked.connect(self._clear_guides)
        self.remove_guide_button.clicked.connect(self._remove_guide)
        self.guide_selection.currentIndexChanged.connect(self._draw_overlay)
        self.guides_visible.toggled.connect(self._guide_visibility_changed)
        self.remove_button.clicked.connect(self._remove)
        self.update_button.clicked.connect(self._update_selected)
        self.find_button.clicked.connect(lambda: self.find_regions(False))
        self.find_all_button.clicked.connect(lambda: self.find_regions(True))
        self.save_button.clicked.connect(self.save)
        self.load_button.clicked.connect(self.load)
        self.use_button.clicked.connect(self.use_for_rig_frame)
        self.plane.currentTextChanged.connect(self._frame_settings_changed)
        self.origin_x_mm.valueChanged.connect(self._frame_settings_changed)
        self.origin_y_mm.valueChanged.connect(self._frame_settings_changed)
        self.run_combo.currentIndexChanged.connect(self._run_selected)
        self.uncertainty_mm.valueChanged.connect(self._uncertainty_changed)
        if hasattr(self.canvas, "raw_point_selected"):
            self.canvas.raw_point_selected.connect(self.handle_raw_point)
        self.canvas.scene().sceneRectChanged.connect(self._draw_overlay)
        self.canvas.destroyed.connect(self._invalidate)
        self.page.destroyed.connect(self._invalidate)
        self._update_run_combo()
        self._refresh_table()
        self.refresh()
        self.frame_model_box.hide()

    def _toggle_point_areas(self, visible):
        self.point_area_box.setVisible(visible)
        self.frame_model_box.setVisible(visible)
        self.point_area_toggle.setArrowType(Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow)

    def _invalidate(self, *_args):
        self._disposed = True
        if self._worker is not None:
            self._worker.cancelled.set()
        self._items = []

    def _context(self):
        if self._disposed:
            return None
        getter = getattr(self.page, "static_reference_context", None)
        if not callable(getter):
            return None
        run_id, ready = getter()
        packet = getattr(self.canvas, "preview", None)
        if not ready or not isinstance(packet, dict) or str(packet.get("run_id")) != str(run_id):
            return None
        if "image_rgb" not in packet or "mapping" not in packet:
            return None
        return str(run_id), _identity(packet)

    def context_changed(self, *_args):
        context = self._context()
        if context != self._last_context:
            self.cancel_interaction()
            if self._worker is not None:
                self._worker.cancelled.set()
            self._last_context = context
        self._update_run_combo()
        self._refresh_table()
        self.refresh()
        if context is not None and not self._loaded and not self.dirty and not self.busy():
            self.load()

    def _current_run(self):
        getter = getattr(self.page, "static_reference_current_run", None)
        return getter() if callable(getter) else self.rig_panel.reference()

    def _update_run_combo(self):
        list_runs = getattr(self.page, "static_reference_runs", None)
        reference = self.rig_panel.reference()
        runs = list(list_runs()) if callable(list_runs) else ([] if reference is None else [reference])
        getter = getattr(self.page, "static_reference_context", None)
        current_id = str(getter()[0]) if callable(getter) else ""
        if not current_id:
            current = self._current_run()
            current_id = str(current.run_id) if current is not None else ""
        self.run_combo.blockSignals(True)
        try:
            self.run_combo.clear()
            for reference in runs:
                self.run_combo.addItem(str(reference.run_id), str(reference.run_id))
            index = self.run_combo.findData(current_id)
            self.run_combo.setCurrentIndex(index)
        finally:
            self.run_combo.blockSignals(False)

    def _run_selected(self, *_args):
        if self._disposed or self.busy():
            return
        run_id = self.run_combo.currentData()
        current = self._current_run()
        if run_id is None or (current is not None and str(current.run_id) == str(run_id)):
            return
        show = getattr(self.page, "show_static_reference_run", None)
        self.cancel_interaction()
        if not callable(show) or not show(str(run_id)):
            self.note.setText("This run's RAW reference photograph could not be shown.")
        self.context_changed()

    def busy(self):
        return self._thread is not None

    def can_close(self):
        return not self.busy()

    def wait_for_workers(self):
        if self._thread is not None:
            self._retire(self._thread)

    def cancel_interaction(self):
        active = bool(self._mode)
        self._mode, self._clicks, self._interaction_key = "", [], None
        self._interaction_line_id = None
        for button in (self.define_button, self.correct_button, self.draw_guide_button,
                       self.add_measurement_button, self.redraw_measurement_button):
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)
        if active and not self._disposed:
            self.canvas.place_mode = False
            self.rig_panel._host_hint("")
            self._draw_overlay()

    def _start(self, mode, active):
        if not active:
            if self._mode == mode:
                self.cancel_interaction()
            return
        context = self._context()
        if (context is None or self.busy() or (mode == "correct" and self._selected() is None)
                or (mode == "measurement-redraw" and self._selected_measurement() is None)):
            self.cancel_interaction()
            return
        self.rig_panel._set_mode("")
        self.cancel_interaction()
        self._mode, self._interaction_key = mode, context
        if mode == "measurement-redraw":
            self._interaction_line_id = self._selected_measurement()["id"]
        self._last_context = context
        button = {"define": self.define_button, "correct": self.correct_button,
                  "guide": self.draw_guide_button, "measurement": self.add_measurement_button,
                  "measurement-redraw": self.redraw_measurement_button}[mode]
        button.blockSignals(True)
        button.setChecked(True)
        button.blockSignals(False)
        self.canvas.place_mode = True
        self._hint({"define": "Click the first area corner.",
                    "correct": "Click the corrected physical anchor.",
                    "guide": "Click the first support-line point.",
                    "measurement": "Click the first measurement endpoint (camera U).",
                    "measurement-redraw": "Click the corrected first measurement endpoint in this run."}[mode])

    def _hint(self, text):
        self.note.setText(text)
        self.rig_panel._host_hint(text)

    def _valid_point(self, x, y):
        packet = self.canvas.preview
        point = packet["mapping"].camera_u_to_display_xy([[x, y]])[0]
        col, row = (int(round(float(value))) for value in point)
        support = packet.get("valid_support")
        height, width = packet["image_rgb"].shape[:2]
        return (0 <= row < height and 0 <= col < width and
                (support is None or bool(support[row, col])))

    def _result_for_edit(self, run_id):
        photo = _backend().photo_binding(self.canvas.preview)
        result = self.document.setdefault("runs", {}).get(run_id)
        if result is None or result.get("photo", {}).get("identity") != photo["identity"]:
            result = dict(run_id=run_id, photo=photo, observations={})
            self.document["runs"][run_id] = result
        return result

    def handle_raw_point(self, row, col):
        if not self._mode:
            return
        if self.busy() or not self.isEnabled() or self._context() != self._interaction_key:
            self.cancel_interaction()
            self.refresh()
            return
        if not all(math.isfinite(float(value)) for value in (row, col)) or not self._valid_point(col, row):
            self._hint("Choose a point supported by this RAW photograph.")
            return
        if self._mode in ("measurement", "measurement-redraw"):
            self._clicks.append((float(col), float(row)))
            if len(self._clicks) < 2:
                self._hint("Click the second measurement endpoint (camera U). The physical length is entered in mm.")
                self._draw_overlay()
                return
            try:
                redraw = self._mode == "measurement-redraw"
                if redraw:
                    line = next(line for line in self.document.get("measurement_lines", [])
                                if line["id"] == self._interaction_line_id)
                else:
                    line = _measurements().capture_measurement_line(self.canvas.preview,
                        line_id=uuid.uuid4().hex,
                        name=self.measurement_name.text().strip() or f"Measurement {len(self.document.get('measurement_lines', [])) + 1}",
                        points_u_xy=self._clicks, length_mm=self.measurement_length_mm.value(),
                        plane_id=self.measurement_plane.currentText().strip() or "rig")
                observation = _measurements().manual_measurement_observation(line, self.canvas.preview, self._clicks)
            except (ValueError, OSError, StopIteration) as error:
                self._clicks = self._clicks[:1]
                self._hint(str(error) or "Select an existing measurement line.")
                self._draw_overlay()
                return
            if not redraw:
                self.document.setdefault("measurement_lines", []).append(line)
            result = self._result_for_edit(self._interaction_key[0])
            result.setdefault("measurement_lines", {})[line["id"]] = observation
            self._generation += 1
            self.cancel_interaction()
            self._changed()
            if not redraw:
                self.measurement_name.setText(f"Measurement {len(self.document['measurement_lines']) + 1}")
                self.measurement_length_mm.setValue(0.)
            self.note.setText(("Measurement endpoints corrected in this run. " if redraw else "Measurement line added. ")
                             + ("Enter its physical length in mm. " if not line.get("length_mm") else "")
                             + "Save references to keep changes.")
            return
        if self._mode == "guide":
            self._clicks.append((float(col), float(row)))
            if len(self._clicks) < 2:
                self._hint("Click the second support-line point.")
                self._draw_overlay()
                return
            try:
                guide = _guides().make_guide(self.canvas.preview, guide_id=uuid.uuid4().hex,
                    points_u_xy=self._clicks, mode=self.guide_mode.currentData(),
                    extend=self.guide_extend.isChecked(),
                    photo_identity=_backend().photo_binding(self.canvas.preview)["identity"])
            except (ValueError, OSError) as error:
                self._clicks = self._clicks[:1]
                self._hint(str(error))
                self._draw_overlay()
                return
            self.document.setdefault("guides", []).append(guide)
            self.cancel_interaction()
            self._changed()
            self.note.setText("Support line added to this photograph. Save references to keep it.")
            return
        if self._mode == "correct":
            region = self._selected()
            run_id = self._interaction_key[0]
            result = self._result_for_edit(run_id)
            result.setdefault("observations", {})[region["id"]] = dict(
                anchor_u_xy=[float(col), float(row)], score=1., status="manual", method="manual",
                region_identity=region["identity"], photo_identity=result["photo"]["identity"])
            result["model"] = self._fit(result["observations"])
            self.cancel_interaction()
            self._changed()
            self.note.setText("Corrected point in this run. Save references to keep the observation.")
            return
        self._clicks.append((float(col), float(row)))
        if len(self._clicks) < 3:
            self._hint("Click the opposite area corner." if len(self._clicks) == 1 else "Click the physical anchor inside the area.")
            self._draw_overlay()
            return
        (x0, y0), (x1, y1), anchor = self._clicks
        rect = [min(x0, x1), min(y0, y1), abs(x1-x0), abs(y1-y0)]
        if rect[2] <= 0 or rect[3] <= 0 or not (rect[0] <= anchor[0] <= rect[0]+rect[2] and rect[1] <= anchor[1] <= rect[1]+rect[3]):
            self._clicks = self._clicks[:2]
            self._hint("The physical anchor must be inside a nonempty area; click the anchor again.")
            return
        try:
            region_id = uuid.uuid4().hex
            region = _backend().capture_region(self.canvas.preview, region_id=region_id,
                name=self.name.text().strip() or f"Reference {len(self.document['regions']) + 1}",
                rect_u_xywh=rect, anchor_u_xy=list(anchor), metric_xy_mm=[self.x_mm.value(), self.y_mm.value()],
                plane_id=self.plane.currentText().strip() or "rig")
        except (ValueError, OSError) as error:
            self.cancel_interaction()
            self.note.setText(str(error))
            return
        self.document.setdefault("regions", []).append(region)
        self._invalidate_models()
        run_id = self._interaction_key[0]
        result = self._result_for_edit(run_id)
        result.setdefault("observations", {})[region_id] = dict(anchor_u_xy=list(anchor), score=1., status="manual", method="manual",
            region_identity=region["identity"], photo_identity=result["photo"]["identity"])
        self.cancel_interaction()
        self._changed()
        self.note.setText("Area defined. Measurements are shared; image anchors belong to each run. Save references to keep changes.")

    def _selected(self):
        row = self.table.currentRow()
        if row < 0 or self.table.item(row, 6) is None:
            return None
        region_id = self.table.item(row, 6).text()
        return next((region for region in self.document.get("regions", []) if region["id"] == region_id), None)

    def _selected_measurement(self):
        row = self.measurement_table.currentRow()
        item = self.measurement_table.item(row, 5) if row >= 0 else None
        return next((line for line in self.document.get("measurement_lines", [])
                     if item is not None and line["id"] == item.text()), None)

    def _measurement_observations(self):
        context = self._context()
        if context is None or not self.document.get("measurement_lines"):
            return {}
        result = self.document.get("runs", {}).get(context[0])
        try:
            return {line["id"]: observation for line in self.document["measurement_lines"]
                    if (observation := _measurements().measurement_for_photo(line, self.canvas.preview, result)) is not None}
        except (ValueError, KeyError, AttributeError):
            return {}

    def _refresh_measurement_table(self):
        selected = self._selected_measurement()
        selected_id = selected["id"] if selected else None
        observations = self._measurement_observations()
        self.measurement_table.setRowCount(0)
        for line in self.document.get("measurement_lines", []):
            row = self.measurement_table.rowCount()
            self.measurement_table.insertRow(row)
            observation = observations.get(line["id"], {})
            status = observation.get("status", "not found").replace("_", " ")
            if not line.get("length_mm"):
                status = "needs measurement" + (f"; {status}" if observation else "")
            score = observation.get("score")
            values = [line.get("name", ""), f"{line.get('length_mm', 0.):.3f}", line.get("plane_id", "rig"),
                      status, "" if score is None else f"{score:.3f}", line["id"]]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column >= 3:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if observation.get("points_u_xy") is not None:
                    points = observation["points_u_xy"]
                    tooltip = f"Camera-U endpoints [px]: {points[0]} → {points[1]}"
                    if observation.get("length_u_px") is not None:
                        tooltip += f"\nObserved distance: {observation['length_u_px']:.3f} px"
                    item.setToolTip(tooltip)
                placement = line.get("placement")
                placement_tip = (f"\nPhysical placement: {placement['start_xy_mm']} mm, {placement['direction']}"
                                 if placement else "\nPhysical placement not set: this distance is not used to fit the frame.")
                item.setToolTip(item.toolTip() + placement_tip)
                self.measurement_table.setItem(row, column, item)
            if line["id"] == selected_id:
                self.measurement_table.selectRow(row)
        self._sync_line_placement()

    def _sync_line_placement(self):
        line = self._selected_measurement()
        placement = line.get("placement") if line is not None else None
        start = placement["start_xy_mm"] if placement else [0., 0.]
        self.placement_x_mm.setValue(start[0])
        self.placement_y_mm.setValue(start[1])
        self.placement_direction.setCurrentIndex(self.placement_direction.findData(
            placement["direction"] if placement else None))
        if line is None:
            self.placement_note.setText("Select a line in the list. Endpoints 1 and 2 will be labelled on the photograph.")
        elif placement:
            points = _measurements().measurement_metric_points(line)
            self.placement_note.setText(f"{line['name']}: endpoint 1 {start} mm; endpoint 2 "
                + (f"{points[1]} mm." if points is not None else "needs a positive physical length."))
        else:
            self.placement_note.setText(f"{line['name']}: enter endpoint 1's physical coordinates and the direction towards endpoint 2. "
                                        "The existing length determines endpoint 2.")

    def _set_line_placement(self, placement):
        line = self._selected_measurement()
        if line is None or self._context() is None or self.busy():
            return
        try:
            updated = _measurements().update_measurement_line(line, placement=placement)
        except ValueError as error:
            self.note.setText(str(error))
            return
        self.cancel_interaction()
        line.clear()
        line.update(updated)
        self._invalidate_models()
        self._changed()
        self.note.setText("Physical placement updated. Image endpoints are unchanged. Save references to keep the placement.")

    def _apply_line_placement(self):
        direction = self.placement_direction.currentData()
        if direction is None:
            self.note.setText("Choose the physical direction from endpoint 1 to endpoint 2 first.")
            return
        self._set_line_placement(dict(start_xy_mm=[self.placement_x_mm.value(), self.placement_y_mm.value()],
                                     direction=direction))

    def _clear_line_placement(self):
        self._set_line_placement(None)

    def _measurement_selection_changed(self):
        if self._loading:
            return
        if self._mode == "measurement-redraw":
            self.cancel_interaction()
        self._sync_line_placement()
        self.refresh()

    def _measurement_table_edited(self, item):
        if self._loading:
            return
        if self._context() is None or self.busy():
            self._refresh_table()
            return
        row = item.row()
        line_id = self.measurement_table.item(row, 5).text()
        line = next(line for line in self.document["measurement_lines"] if line["id"] == line_id)
        try:
            length = float(self.measurement_table.item(row, 1).text().replace(",", "."))
            if not math.isfinite(length) or length < 0:
                raise ValueError("Physical length must be a finite, nonnegative number in mm")
            name = self.measurement_table.item(row, 0).text().strip()
            plane = self.measurement_table.item(row, 2).text().strip()
            if not name or not plane:
                raise ValueError("Enter a measurement name and physical plane")
            updated = _measurements().update_measurement_line(line, name=name, length_mm=length, plane_id=plane)
        except ValueError as error:
            self.note.setText(str(error))
            self._refresh_table()
            return
        self.cancel_interaction()
        line.clear()
        line.update(updated)
        self._invalidate_models()
        self._changed()
        self.note.setText("Shared measurement updated. Image endpoints remain specific to each run. Save references to keep changes.")

    def _remove_measurement(self):
        line = self._selected_measurement()
        if line is None or self._context() is None or self.busy():
            return
        self.cancel_interaction()
        self.document["measurement_lines"].remove(line)
        for result in self.document.get("runs", {}).values():
            result.get("measurement_lines", {}).pop(line["id"], None)
        self._invalidate_models()
        self._changed()

    def _current_guides(self):
        if self._context() is None or not self.document.get("guides"):
            return []
        packet = self.canvas.preview
        try:
            identity = _backend().photo_binding(packet)["identity"]
            return [(guide, segment) for guide in self.document["guides"]
                    if (segment := _guides().guide_segment(guide, packet,
                        photo_identity=identity)) is not None]
        except (ValueError, KeyError, AttributeError):
            return []

    def _undo_guide(self):
        if self._context() is None or self.busy():
            return
        guides = self._current_guides()
        if guides:
            self.cancel_interaction()
            self.document["guides"].remove(guides[-1][0])
            self._changed()

    def _refresh_guide_selection(self):
        selected = self.guide_selection.currentData()
        self.guide_selection.blockSignals(True)
        try:
            self.guide_selection.clear()
            for number, (guide, _segment) in enumerate(self._current_guides(), 1):
                mode = {"through": "Through points", "horizontal": "Horizontal", "vertical": "Vertical"}[guide["mode"]]
                self.guide_selection.addItem(f"Line {number}: {mode}", guide["id"])
                points = guide["points_u_xy"]
                self.guide_selection.setItemData(number-1,
                    f"Camera-U endpoints [px]: {points[0]} → {points[1]}", Qt.ItemDataRole.ToolTipRole)
            index = self.guide_selection.findData(selected)
            if index >= 0:
                self.guide_selection.setCurrentIndex(index)
        finally:
            self.guide_selection.blockSignals(False)

    def _remove_guide(self):
        if self._context() is None or self.busy():
            return
        selected = self.guide_selection.currentData()
        guide = next((guide for guide, _segment in self._current_guides() if guide["id"] == selected), None)
        if guide is not None:
            self.cancel_interaction()
            self.document["guides"].remove(guide)
            self._changed()
            self.note.setText("Selected support line removed. Save references to keep changes.")

    def _clear_guides(self):
        if self._context() is None or self.busy():
            return
        guides = self._current_guides()
        if guides:
            self.cancel_interaction()
            current_ids = {guide["id"] for guide, _ in guides}
            self.document["guides"] = [guide for guide in self.document["guides"]
                                       if guide["id"] not in current_ids]
            self._changed()

    def _guide_visibility_changed(self, visible):
        if self._loading:
            return
        if self._context() is None or self.busy():
            self.refresh()
            return
        self.document["guides_visible"] = bool(visible)
        self._changed()

    def _observations(self):
        context = self._context()
        if context is None:
            return {}
        result = self.document.get("runs", {}).get(context[0], {})
        if not result.get("observations"):
            return {}
        try:
            identity = _backend().photo_binding(self.canvas.preview)["identity"]
        except (ValueError, KeyError, AttributeError):
            return {}
        if result.get("photo", {}).get("identity") != identity:
            return {}
        return result.get("observations", {})

    def _fit(self, observations):
        try:
            if self.document.get("measurement_lines"):
                return _measurements().fit_measurement_reference_model(self.document, observations,
                    self._measurement_observations(), plane_id=self.plane.currentText().strip() or "rig")
            return _backend().fit_reference_model(self.document.get("regions", []), observations,
                plane_id=self.plane.currentText().strip() or "rig",
                measurement_uncertainty_mm=self.document.get("measurement_uncertainty_mm", 0.))
        except ValueError as error:
            return dict(status="insufficient", reason=str(error))

    def _refresh_table(self):
        if self._disposed:
            return
        selected = self._selected()
        selected_id = selected["id"] if selected else None
        self._loading = True
        try:
            self.table.setRowCount(0)
            observations = self._observations()
            for region in self.document.get("regions", []):
                row = self.table.rowCount()
                self.table.insertRow(row)
                observation = observations.get(region["id"], {})
                metric = region.get("metric_xy_mm") or [0., 0.]
                values = [region.get("name", ""), f"{metric[0]:.3f}", f"{metric[1]:.3f}",
                    region.get("plane_id", "rig"), observation.get("status", "not found"),
                    "" if observation.get("score") is None else f"{observation['score']:.3f}", region["id"]]
                for column, value in enumerate(values):
                    item = QTableWidgetItem(str(value))
                    if column >= 4:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.table.setItem(row, column, item)
                if region["id"] == selected_id:
                    self.table.selectRow(row)
            self._refresh_measurement_table()
            self._refresh_guide_selection()
        finally:
            self._loading = False

    def _selection_changed(self):
        if self._loading:
            return
        region = self._selected()
        if region is not None:
            self.name.setText(region.get("name", ""))
            self.x_mm.setValue(region["metric_xy_mm"][0])
            self.y_mm.setValue(region["metric_xy_mm"][1])
            self.plane.setCurrentText(region.get("plane_id", "rig"))
        self.refresh()

    def _invalidate_models(self, region_id=None):
        self._generation += 1
        for result in self.document.get("runs", {}).values():
            result.pop("model", None)
            result.pop("plane_models", None)
            if region_id is not None:
                result.get("observations", {}).pop(region_id, None)

    def _uncertainty_changed(self, value):
        if self._loading:
            return
        if self._context() is None or self.busy():
            self.uncertainty_mm.blockSignals(True)
            self.uncertainty_mm.setValue(self.document.get("measurement_uncertainty_mm", 0.))
            self.uncertainty_mm.blockSignals(False)
            return
        self.document["measurement_uncertainty_mm"] = value
        self._invalidate_models()
        self._changed()

    def _frame_settings_changed(self, *_args):
        if self._loading:
            return
        if self._context() is None or self.busy():
            self._restore_frame_settings()
            return
        self.document['rig_origin_xy_mm'] = [self.origin_x_mm.value(), self.origin_y_mm.value()]
        self.document['model_plane_id'] = self.plane.currentText().strip() or 'rig'
        self._changed()

    def _restore_frame_settings(self):
        origin = self.document.get('rig_origin_xy_mm', [0., 0.])
        self.origin_x_mm.blockSignals(True)
        self.origin_y_mm.blockSignals(True)
        self.plane.blockSignals(True)
        self.origin_x_mm.setValue(origin[0])
        self.origin_y_mm.setValue(origin[1])
        self.plane.setCurrentText(self.document.get('model_plane_id', 'rig'))
        self.origin_x_mm.blockSignals(False)
        self.origin_y_mm.blockSignals(False)
        self.plane.blockSignals(False)

    def _update_selected(self):
        region = self._selected()
        if region is None or self._context() is None or self.busy():
            return
        self._update_region(region, self.name.text().strip(),
            [self.x_mm.value(), self.y_mm.value()], self.plane.currentText().strip())

    def _update_region(self, region, name, metric, plane):
        try:
            updated = _backend().update_region(region, name=name, metric_xy_mm=metric, plane_id=plane)
        except ValueError as error:
            self.note.setText(str(error))
            self._refresh_table()
            return
        region_id = region["id"]
        region.clear()
        region.update(updated)
        self._invalidate_models(region_id)
        self._changed()
        self.note.setText("Shared reference changed; its run observations and all fitted models were invalidated. Find it again or correct its point.")

    def _table_edited(self, item):
        if self._loading:
            return
        if self._context() is None or self.busy():
            self._refresh_table()
            return
        region_id = self.table.item(item.row(), 6).text()
        region = next(region for region in self.document["regions"] if region["id"] == region_id)
        try:
            metric = [float(self.table.item(item.row(), column).text().replace(",", ".")) for column in (1, 2)]
            if not all(math.isfinite(value) for value in metric):
                raise ValueError("Coordinates must be finite")
            plane = self.table.item(item.row(), 3).text().strip()
            if not plane:
                raise ValueError("Enter a physical plane")
        except ValueError as error:
            self.note.setText(str(error))
            self._refresh_table()
            return
        self._update_region(region, self.table.item(item.row(), 0).text().strip(), metric, plane)

    def _remove(self):
        region = self._selected()
        if region is None or self._context() is None or self.busy():
            return
        self.document["regions"].remove(region)
        self._invalidate_models(region["id"])
        self._changed()

    def _changed(self):
        self.dirty = True
        self._refresh_table()
        self.refresh()
        self.changed.emit()

    def find_regions(self, all_runs=False):
        context = self._context()
        if (context is None or self.busy()
                or not (self.document.get("regions") or self.document.get("measurement_lines"))):
            return False
        self.cancel_interaction()
        runs = list(self.page.static_reference_runs()) if all_runs else [self._current_run()]
        runs = [reference for reference in runs if reference is not None]
        if not runs:
            return False
        self._last_context = context
        self._work_key = context, self._generation
        self._operation = "recognize"
        worker = _Recognize(deepcopy(self.document), runs, self.page.read_static_reference_packet)
        self._launch(worker)
        self.note.setText(f"Finding references and measurement lines in {len(runs)} run(s)…")
        return True

    def load(self):
        if self._context() is None or self.busy() or self.dirty:
            return False
        self._loaded = True
        self._operation = "load"
        self._work_key = self._context(), self._generation
        self._launch(_LoadReferences(self.rig_panel.project.root))
        self.note.setText("Loading saved references and checking their source identities…")
        return True

    def _launch(self, worker):
        thread = QThread(self)
        self._worker, self._thread = worker, thread
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        if hasattr(worker, "progress"):
            worker.progress.connect(self._progress)
        thread.finished.connect(self._closed)
        self.refresh()
        thread.start()
        self.busy_changed.emit(True)

    @Slot(int, int, str)
    def _progress(self, number, total, run_id):
        if not self._disposed:
            self.note.setText(f"Finding references: {number}/{total} ({run_id})…")

    @Slot()
    def _closed(self):
        if self.sender() is self._thread:
            self._retire(self._thread)

    def _retire(self, thread):
        if thread is not self._thread:
            return
        thread.wait()
        worker = self._worker
        self._thread = self._worker = None
        if not self._disposed:
            if self._operation in ("load", "save"):
                if worker.error:
                    self.note.setText(f"Could not {self._operation} references: {worker.error}")
                elif self._work_key[1] == self._generation:
                    self.document = worker.result
                    if (not self.document.get("regions") and not self.document.get("measurement_lines")
                            and not self.document.get("runs") and not self.document.get("history")):
                        self.document["measurement_uncertainty_mm"] = 2.
                    self.uncertainty_mm.blockSignals(True)
                    self.uncertainty_mm.setValue(self.document.get("measurement_uncertainty_mm", 0.))
                    self.uncertainty_mm.blockSignals(False)
                    self._restore_frame_settings()
                    if self._operation == "save":
                        self.dirty = False
                    self.note.setText("Static references saved." if self._operation == "save" else
                        "Saved references loaded. Measurements are shared; observed endpoints and points belong to each run.")
                    self._refresh_table()
            elif worker.cancelled.is_set() or self._work_key != (self._context(), self._generation):
                self.note.setText("The photograph or references changed; recognition results were discarded.")
            else:
                self.document.setdefault("runs", {}).update(worker.results)
                self.dirty = self.dirty or bool(worker.results)
                self.note.setText("Recognition finished. Inspect scores and correct ambiguous or missing endpoints and points. "
                    "Save references to keep results." + (" " + "; ".join(worker.errors) if worker.errors else ""))
                self._refresh_table()
                self.changed.emit()
            self.refresh()
        del worker
        if isValid(thread):
            thread.deleteLater()
        if not self._disposed:
            self.busy_changed.emit(False)

    def save(self):
        if self._context() is None or self.busy() or not self.dirty:
            return False
        self.cancel_interaction()
        self._operation = "save"
        self._work_key = self._context(), self._generation
        from .series_decisions import project_series_root
        self._launch(_SaveReferences(self.rig_panel.project.root, deepcopy(self.document),
                                     project_series_root(self.rig_panel.project)))
        self.note.setText("Saving references and checking source identities…")
        return True

    def use_for_rig_frame(self):
        if self._context() is None or self.busy():
            return False
        if self.plane.currentText().strip() != "rig":
            self.note.setText("A rig frame needs references measured in the rig plane. Bolt heads or other depth planes require a measured plane conversion.")
            return False
        model = self._fit(self._observations())
        matrix = model.get("H_metric_mm_to_camera_u") if isinstance(model, dict) else None
        if matrix is None or self.rig_panel.width_m.value() <= 0 or self.rig_panel.height_m.value() <= 0:
            self.note.setText("Use sufficiently spread references in one plane, and enter the rig's inner width/height first.")
            return False
        width, height = self.rig_panel.width_m.value()*1000., self.rig_panel.height_m.value()*1000.
        x, y = self.origin_x_mm.value(), self.origin_y_mm.value()
        points = np.array([[x, y+height, 1.], [x+width, y+height, 1.],
                           [x+width, y, 1.], [x, y, 1.]])
        mapped = (np.asarray(matrix, dtype=float) @ points.T).T
        if not np.isfinite(mapped).all() or np.any(np.abs(mapped[:, 2]) < 1e-12):
            self.note.setText("The reference model cannot locate these rig corners.")
            return False
        if (np.min(mapped[:, 2])*np.max(mapped[:, 2]) <= 0
                or np.any(np.linalg.det(matrix)/mapped[:, 2]**3 >= 0)):
            self.note.setText("The measured model crosses a projective horizon or reverses orientation within the rig. Review the reference placements.")
            return False
        mapped = mapped[:, :2] / mapped[:, 2, None]
        apply_frame = getattr(self.page, "use_static_reference_frame", None)
        if callable(apply_frame):
            corners_rc = [(float(raw_row), float(col)) for col, raw_row in mapped]
            if not apply_frame(self._context()[0], corners_rc,
                    self.rig_panel.width_m.value(), self.rig_panel.height_m.value()):
                self.note.setText("This run's rig frame could not be updated.")
                return False
            self.note.setText("Workspace rig frame updated for this run. Scientific geometry has not been adopted.")
            return True
        self.rig_panel._loading = True
        try:
            from .series_rig_frame_panel import REFERENCE_PIXEL_COLUMN, _pixel_item

            for row, (col, raw_row) in enumerate(mapped):
                self.rig_panel.corners.setItem(row, REFERENCE_PIXEL_COLUMN, _pixel_item(raw_row, col))
        finally:
            self.rig_panel._loading = False
        self.rig_panel._store_frame()
        self.note.setText("Workspace rig frame updated from this plane and origin. Scientific geometry has not been adopted.")
        return True

    def refresh(self, *_args):
        if self._disposed:
            return
        available = self._context() is not None and not self.busy()
        self.guides_visible.blockSignals(True)
        self.guides_visible.setChecked(self.document.get("guides_visible", True))
        self.guides_visible.blockSignals(False)
        for widget in (self.guide_mode, self.guide_extend, self.guides_visible, self.draw_guide_button):
            widget.setEnabled(available)
        has_guides = available and bool(self._current_guides())
        self.undo_guide_button.setEnabled(has_guides)
        self.clear_guides_button.setEnabled(has_guides)
        self.guide_selection.setEnabled(has_guides)
        self.remove_guide_button.setEnabled(has_guides and self.guide_selection.currentData() is not None)
        self.run_combo.setEnabled(available and self.run_combo.count() > 0)
        for widget in (self.measurement_name, self.measurement_length_mm, self.measurement_plane,
                       self.add_measurement_button, self.measurement_table):
            widget.setEnabled(available)
        selected_measurement = self._selected_measurement() is not None
        self.placement_box.setEnabled(available and selected_measurement)
        self.clear_placement_button.setEnabled(available and selected_measurement
            and bool(self._selected_measurement().get('placement')))
        self.redraw_measurement_button.setEnabled(available and selected_measurement)
        self.remove_measurement_button.setEnabled(available and selected_measurement)
        selected = self._selected() is not None
        self.table.setEnabled(available)
        for widget in (self.name, self.x_mm, self.y_mm, self.plane, self.uncertainty_mm, self.origin_x_mm, self.origin_y_mm, self.define_button):
            widget.setEnabled(available)
        self.correct_button.setEnabled(available and selected)
        self.remove_button.setEnabled(available and selected)
        self.update_button.setEnabled(available and selected)
        has_references = bool(self.document.get("regions") or self.document.get("measurement_lines"))
        self.find_button.setEnabled(available and has_references)
        self.find_all_button.setEnabled(available and has_references)
        self.save_button.setEnabled(available and self.dirty)
        model = self._fit(self._observations()) if available and has_references else None
        self.load_button.setEnabled(available and not self.dirty)
        self.use_button.setEnabled(available and isinstance(model, dict) and model.get("H_metric_mm_to_camera_u") is not None
            and self.plane.currentText().strip() == "rig"
            and self.rig_panel.width_m.value() > 0 and self.rig_panel.height_m.value() > 0)
        if model:
            line_count = len(model.get("used_line_ids", []))
            count = len(model.get("used_region_ids", [])) + 2*line_count
            details = f"Plane {self.plane.currentText()}: {model.get('status', 'insufficient')} ({count} anchors, {line_count} lines). "
            if model.get("rms_u_px") is not None:
                details += f"RMS {model['rms_u_px']:.3f} px. "
            self.model_note.setText(details + model.get("reason", ""))
        else:
            self.model_note.setText("At least four spread, noncollinear anchors in one plane are needed for a model.")
        self.save_state.setText("Unsaved references — click Save references." if self.dirty
            else "All references saved." if self._loaded else "References have not loaded yet.")
        self._draw_overlay()

    def _draw_overlay(self, *_args):
        if self._disposed:
            return
        for item in self._items:
            if isValid(item) and item.scene() is not None:
                item.scene().removeItem(item)
        self._items = []
        context = self._context()
        if context is None:
            return
        observations = self._observations()
        measurement_observations = self._measurement_observations()
        selected_line = self._selected_measurement()
        for line in self.document.get("measurement_lines", []) if not self.measurement_box.isHidden() else []:
            observation = measurement_observations.get(line["id"], {})
            points = observation.get("points_u_xy")
            if points is None or observation.get("status") in ("ambiguous", "not_found", "not found", "missing"):
                continue
            first, second = [self.canvas.raw_to_scene(y, x) for x, y in points]
            color = QColor("#48d6ed")
            pen = QPen(color, 3. if selected_line is line else 2., Qt.PenStyle.SolidLine)
            pen.setCosmetic(True)
            path = QPainterPath(first)
            path.lineTo(second)
            segment = QGraphicsPathItem(path)
            segment.setData(0, "measurement-line")
            segment.setData(1, line["id"])
            markers = QPainterPath()
            for point in (first, second):
                markers.addEllipse(point, 4., 4.)
            endpoints = QGraphicsPathItem(markers)
            endpoints.setData(0, "measurement-endpoints")
            length = line.get("length_mm", 0.)
            label = QGraphicsSimpleTextItem(f"{line.get('name', line['id'])}: "
                                            + (f"{length:g} mm" if length else "needs measurement [mm]"))
            label.setBrush(color)
            label.setPos((first.x()+second.x())/2.+6., (first.y()+second.y())/2.+6.)
            label.setFlag(label.GraphicsItemFlag.ItemIgnoresTransformations, True)
            for item in (segment, endpoints, label):
                if isinstance(item, QGraphicsPathItem):
                    item.setPen(pen)
                item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                item.setZValue(35)
                self.canvas.scene().addItem(item)
                self._items.append(item)
            if selected_line is line:
                for number, point in enumerate((first, second), 1):
                    endpoint_label = QGraphicsSimpleTextItem(str(number))
                    endpoint_label.setData(0, "measurement-endpoint-number")
                    endpoint_label.setBrush(color)
                    endpoint_label.setPos(point.x()+6., point.y()-18.)
                    endpoint_label.setFlag(endpoint_label.GraphicsItemFlag.ItemIgnoresTransformations, True)
                    endpoint_label.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                    endpoint_label.setZValue(37)
                    self.canvas.scene().addItem(endpoint_label)
                    self._items.append(endpoint_label)
        if self.document.get("guides_visible", True):
            for guide, segment in self._current_guides():
                (x0, y0), (x1, y1) = segment
                path = QPainterPath(self.canvas.raw_to_scene(y0, x0))
                path.lineTo(self.canvas.raw_to_scene(y1, x1))
                item = QGraphicsPathItem(path)
                pen = QPen(QColor("#ffd166"), 3. if guide["id"] == self.guide_selection.currentData() else 2., Qt.PenStyle.DashLine)
                pen.setCosmetic(True)
                item.setPen(pen)
                item.setZValue(33)
                item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                self.canvas.scene().addItem(item)
                self._items.append(item)
        if self._clicks:
            path = QPainterPath()
            if self._mode in ("guide", "measurement", "measurement-redraw"):
                x, y = self._clicks[0]
                point = self.canvas.raw_to_scene(y, x)
                path.moveTo(point.x(), point.y()-5.)
                path.lineTo(point.x()+5., point.y())
                path.lineTo(point.x(), point.y()+5.)
                path.lineTo(point.x()-5., point.y())
                path.closeSubpath()
            elif len(self._clicks) >= 2:
                (x0, y0), (x1, y1) = self._clicks[:2]
                path.moveTo(self.canvas.raw_to_scene(y0, x0))
                for col, row in ((x1, y0), (x1, y1), (x0, y1), (x0, y0)):
                    path.lineTo(self.canvas.raw_to_scene(row, col))
            else:
                x, y = self._clicks[0]
                point = self.canvas.raw_to_scene(y, x)
                path.addEllipse(point, 4., 4.)
            item = QGraphicsPathItem(path)
            color = ("#ffd166" if self._mode == "guide" else "#48d6ed"
                     if self._mode in ("measurement", "measurement-redraw") else "#69b7ff")
            pen = QPen(QColor(color), 2.)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setZValue(36)
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.canvas.scene().addItem(item)
            self._items.append(item)
        for region in self.document.get("regions", []):
            observation = observations.get(region["id"], {})
            anchor = observation.get("anchor_u_xy")
            if anchor is None or observation.get("status") in ("ambiguous", "not_found", "not found", "missing"):
                continue
            color = QColor("#50e3a4" if observation.get("status") == "manual" else "#f4bc45")
            pen = QPen(color, 2.)
            pen.setCosmetic(True)
            rect = region.get("rect_u_xywh")
            original = region.get("anchor_u_xy")
            if rect and original:
                dx, dy = anchor[0]-original[0], anchor[1]-original[1]
                x, y, width, height = rect
                points = [(x+dx, y+dy), (x+width+dx, y+dy), (x+width+dx, y+height+dy), (x+dx, y+height+dy)]
                path = QPainterPath(self.canvas.raw_to_scene(points[0][1], points[0][0]))
                for col, row in points[1:]:
                    path.lineTo(self.canvas.raw_to_scene(row, col))
                path.closeSubpath()
                item = QGraphicsPathItem(path)
                item.setPen(pen)
                item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                item.setZValue(34)
                self.canvas.scene().addItem(item)
                self._items.append(item)
            point = self.canvas.raw_to_scene(anchor[1], anchor[0])
            path = QPainterPath()
            path.moveTo(point.x()-5., point.y())
            path.lineTo(point.x()+5., point.y())
            path.moveTo(point.x(), point.y()-5.)
            path.lineTo(point.x(), point.y()+5.)
            item = QGraphicsPathItem(path)
            item.setPen(pen)
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            item.setZValue(35)
            self.canvas.scene().addItem(item)
            label = QGraphicsSimpleTextItem(region.get("name", region["id"]))
            label.setBrush(color)
            label.setPos(point.x()+7., point.y()+4.)
            label.setFlag(label.GraphicsItemFlag.ItemIgnoresTransformations, True)
            label.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            label.setZValue(35)
            self.canvas.scene().addItem(label)
            self._items.extend([item, label])


__all__ = ["StaticReferencePanel"]
