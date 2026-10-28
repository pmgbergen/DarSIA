"""Rig frame and injection points for a registered series.

The panel replaces the project workflow's injection-point tab when the
geometry-and-facies page is hosted for a series folder. Four observed reference
points define a measured rectangle or quadrilateral in the photograph. Their
display coordinates use the bottom-left reference as origin, x right and y up.
Injection points can be entered in physical coordinates or placed in the photo.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QGuiApplication, QKeySequence, QPen, QPolygonF, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QComboBox,
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPolygonItem,
    QGraphicsSimpleTextItem,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.series_geometry import (
    FRAME_FROM_CONFIGURATION,
    FRAME_FROM_STATIC_REFERENCES,
    FRAME_MARKED,
    RigFrame,
    SeriesRunInputs,
    default_actor,
    photo_size,
    protocol_injection_points,
    remove_rig_frame,
    rig_frame_from_inputs,
    save_injection_points,
    save_rig_frame,
    saved_injection_points,
    saved_rig_frames,
)
from darsia.fluidflower.setup_crop_area import (
    CROP_REFERENCE_QUAD,
    CROP_SEPARATE_AREA,
    CropArea,
    crop_area_from_frame,
    crop_extent,
    save_crop_area,
    save_crop_choice,
    saved_crop_areas,
    saved_crop_choice,
    separate_crop_reason,
)
from darsia.fluidflower.storage_paths import same_storage_path
from darsia.fluidflower.templates import RIG_CLASSES

# Clicking order for the corners (clockwise from the top left).
CLICK_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")
CORNER_LABELS = {
    "top_left": "Top left",
    "top_right": "Top right",
    "bottom_right": "Bottom right",
    "bottom_left": "Bottom left",
}
POINT_COLUMNS = (
    "Location ID",
    "Name",
    "x [m]",
    "y [m]",
    "Row [px]",
    "Col [px]",
    "Source",
)
_DISPLAY_LIMIT_PX = 2400
REFERENCE_PIXEL_COLUMN = 1
REFERENCE_VISIBLE_COLUMN = 4
REFERENCE_DESCRIPTIONS = (("Rig frame", "rig_frame"), ("Top of free water", "top_of_free_water"))


def _number(text: str) -> float | None:
    try:
        value = float(str(text).replace(",", ".").strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _fmt(value: float | None, digits: int) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _pixel_item(row: float, col: float) -> QTableWidgetItem:
    item = QTableWidgetItem(f"{row:.1f}; {col:.1f}")
    item.setData(Qt.ItemDataRole.UserRole, (float(row), float(col)))
    item.setToolTip("Image position: row; column, in camera-U pixels. Enter decimals with a point, or separate the pair with a semicolon.")
    return item


def _pixel_value(item: QTableWidgetItem | None) -> tuple[float, float] | None:
    if item is None or not item.text().strip():
        return None
    original = item.data(Qt.ItemDataRole.UserRole)
    if original is not None and item.text() == f"{original[0]:.1f}; {original[1]:.1f}":
        return tuple(original)
    parts = item.text().split(';') if ';' in item.text() else item.text().split(',')
    values = [_number(part) for part in parts]
    if len(values) != 2 or any(value is None for value in values):
        raise ValueError("Enter the image position as row; column, for example 56.8; -31.4")
    return tuple(values)


class RigFramePanel(QWidget):
    """Rig frame of the reference run and the series' injection points."""

    changed = Signal()

    def __init__(
        self,
        page: Any,
        project: Any,
        reference: Callable[[], SeriesRunInputs | None],
        *,
        scrollable: bool = True,
        legacy_static_references: bool = False,
    ) -> None:
        """``page`` provides the canvas (``page.canvas``).

        A page that shows photographs itself provides ``show_frame_photo(path)``,
        which displays the photograph in its canvas and returns whether it did.
        ``scrollable`` puts the panel in its own scroll area.
        """

        super().__init__()
        self.page = page
        self.project = project
        self._reference = reference
        self.canvas = page.canvas
        self.frame: RigFrame | None = None
        self._mode = ""
        self._corner_index = 0
        self._corner_rows = list(range(len(CLICK_ORDER)))
        self._selected_reference_name = None
        self._photo_path: Path | None = None
        self._photo_size: tuple[int, int] | None = None
        self._shown_preview: Any = None
        self._loaded_run: str | None = None
        self._overlay: list[Any] = []
        self._show_reference_points = True
        self._show_reference_lines = True
        self._loading = False
        self._rig_class = ""
        # Setup crop: the series' choice, the reference run's separate area,
        # why a separate area cannot be chosen, the corners clicked so far and
        # why the last read or action failed.
        self._crop_choice = CROP_REFERENCE_QUAD
        self._crop_area: CropArea | None = None
        self._crop_reason = ""
        self._crop_points: dict[str, tuple[float, float]] = {}
        self._crop_error = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        content = QWidget()
        if scrollable:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            outer.addWidget(scroll)
            scroll.setWidget(content)
        else:
            outer.addWidget(content)
        layout = QVBoxLayout(content)
        self._content_layout = layout
        if not scrollable:
            layout.setContentsMargins(0, 0, 0, 0)

        text = (
            "Choose four visible reference points around a measured area. Inner rig corners are preferred; "
            "the water surface or other measured references can also be used. Enter their physical dimensions. "
            "The complete photograph stays visible. x points right and y upwards."
        )
        note = QLabel(text)
        note.setObjectName("secondaryText")
        note.setWordWrap(True)
        layout.addWidget(note)

        photo_row = QHBoxLayout()
        self.show_reference_button = QPushButton("Show reference photo")
        self.show_reference_button.clicked.connect(self.show_reference_photo)
        photo_row.addWidget(self.show_reference_button)
        self.show_other_button = QPushButton("Show other photo…")
        self.show_other_button.setToolTip(
            "Any raw photograph from the same camera position, for example one "
            "in which the injection port is visible"
        )
        self.show_other_button.clicked.connect(self._choose_photo)
        photo_row.addWidget(self.show_other_button)
        photo_row.addStretch(1)
        layout.addLayout(photo_row)
        self.photo_label = QLabel("No photograph shown.")
        self.photo_label.setObjectName("secondaryText")
        self.photo_label.setWordWrap(True)
        layout.addWidget(self.photo_label)
        if not scrollable:
            # The host shows the reference photograph itself.
            self.show_reference_button.hide()
            self.photo_label.hide()

        frame_box = QGroupBox("Reference quadrilateral")
        frame_layout = QVBoxLayout(frame_box)
        self.frame_status = QLabel()
        self.frame_status.setWordWrap(True)
        frame_layout.addWidget(self.frame_status)
        self.corners = QTableWidget(len(CLICK_ORDER), 5)
        self.corners.setHorizontalHeaderLabels(["Description", "Image [px]\n(row; col)", "x [mm]", "y [mm]", "Visible"])
        self.corners.setVerticalHeaderLabels(
            [f"{index+1}  {CORNER_LABELS[name]}" for index, name in enumerate(CLICK_ORDER)]
        )
        self.corners.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        for column in (0, 2, 3, 4):
            self.corners.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.corners.setMaximumHeight(170)
        self.corners.itemChanged.connect(self._corner_edited)
        self.corners.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.corners.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.corners.itemSelectionChanged.connect(self._reference_selection_changed)
        self.description_boxes = {}
        for row, name in enumerate(CLICK_ORDER):
            combo = QComboBox()
            for label, key in REFERENCE_DESCRIPTIONS:
                combo.addItem(label, key)
            combo.currentIndexChanged.connect(lambda _index, point=name: self._description_edited(point))
            self.description_boxes[name] = combo
            self.corners.setCellWidget(row, 0, combo)
        frame_layout.addWidget(self.corners)
        origin_note = QLabel("Point 4 (bottom left) is always (0, 0). x points right and y upwards; values are relative to this reference point.")
        origin_note.setWordWrap(True)
        frame_layout.addWidget(origin_note)
        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("Physical shape"))
        self.reference_shape = QComboBox()
        self.reference_shape.addItem("Rectangle", "rectangle")
        self.reference_shape.addItem("Four sides and a diagonal", "quadrilateral")
        shape_row.addWidget(self.reference_shape, 1)
        frame_layout.addLayout(shape_row)
        self.rectangle_size = QWidget()
        size_row = QHBoxLayout(self.rectangle_size)
        size_row.setContentsMargins(0, 0, 0, 0)
        self.width_m = QDoubleSpinBox()
        self.height_m = QDoubleSpinBox()
        for spin in (self.width_m, self.height_m):
            spin.setRange(0.0, 20.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.001)
            spin.setSuffix(" m")
            spin.editingFinished.connect(self._size_edited)
        size_row.addWidget(QLabel("1–2 / 4–3"))
        size_row.addWidget(self.width_m)
        size_row.addWidget(QLabel("1–4 / 2–3"))
        size_row.addWidget(self.height_m)
        size_row.addStretch(1)
        frame_layout.addWidget(self.rectangle_size)
        self.measured_sides = QGroupBox("Measured distances")
        side_layout = QVBoxLayout(self.measured_sides)
        self.side_lengths = {}
        for entries in ((('top', '1–2'), ('right', '2–3')), (('bottom', '3–4'), ('left', '4–1')),
                        (('diagonal_ac', '1–3 diagonal'),)):
            row = QHBoxLayout()
            for key, label in entries:
                spin = QDoubleSpinBox()
                spin.setRange(0., 20.)
                spin.setDecimals(4)
                spin.setSuffix(' m')
                spin.setSingleStep(.001)
                spin.editingFinished.connect(self._size_edited)
                self.side_lengths[key] = spin
                row.addWidget(QLabel(label))
                row.addWidget(spin)
            side_layout.addLayout(row)
        frame_layout.addWidget(self.measured_sides)
        self.measured_sides.hide()
        self.reference_error = QLabel()
        self.reference_error.setWordWrap(True)
        frame_layout.addWidget(self.reference_error)
        self.reference_shape.currentIndexChanged.connect(self._reference_shape_changed)
        self.class_row = QWidget()
        class_layout = QHBoxLayout(self.class_row)
        class_layout.setContentsMargins(0, 0, 0, 0)
        self.class_size = QLabel()
        self.class_size.setObjectName("secondaryText")
        self.class_size.setWordWrap(True)
        class_layout.addWidget(self.class_size, 1)
        self.use_class_size = QPushButton("Use class size")
        self.use_class_size.setToolTip(
            "Enter the nominal inner size of the rig class. Measured values of "
            "this rig are better when you have them."
        )
        self.use_class_size.clicked.connect(self.apply_class_size)
        class_layout.addWidget(self.use_class_size)
        self.class_row.hide()
        frame_layout.addWidget(self.class_row)
        frame_buttons = QHBoxLayout()
        single_point = QHBoxLayout()
        self.place_reference_point = QPushButton('Place selected reference point')
        self.place_reference = self.place_reference_point
        self.place_reference_point.setCheckable(True)
        self.place_reference_point.setEnabled(False)
        self.place_reference_point.toggled.connect(self._reference_place_toggled)
        single_point.addWidget(self.place_reference_point)
        self.reference_target = QLabel('Select a reference row')
        single_point.addWidget(self.reference_target, 1)
        frame_layout.addLayout(single_point)
        self.mark_corners = QPushButton("Mark all reference points")
        self.mark_corners.setCheckable(True)
        self.mark_corners.setToolTip(
            "Click points 1 to 4 around the reference area. Unseen points are not observations. "
            "Zoom with the mouse wheel for precision."
        )
        self.mark_corners.toggled.connect(self._mark_corners_toggled)
        frame_buttons.addWidget(self.mark_corners)
        self.copy_crop = QPushButton("Copy crop section")
        self.copy_crop.clicked.connect(self._copy_crop)
        frame_buttons.addWidget(self.copy_crop)
        frame_buttons.addStretch(1)
        frame_layout.addLayout(frame_buttons)
        configuration_row = QHBoxLayout()
        self.use_configuration = QPushButton("Use series configuration")
        self.use_configuration.setToolTip(
            "Discard the frame marked here and use the run's [corrections.curvature.crop]"
        )
        self.use_configuration.clicked.connect(self._use_configuration)
        configuration_row.addWidget(self.use_configuration)
        self.restore_reference = QPushButton('Restore approved reference')
        self.restore_reference.setToolTip('Restore this run\'s last approved reference points. Injection coordinates are unchanged.')
        self.restore_reference.clicked.connect(self._restore_approved_reference)
        configuration_row.addWidget(self.restore_reference)
        configuration_row.addStretch(1)
        frame_layout.addLayout(configuration_row)
        self.review_points = QPushButton('Review reference points in all runs…')
        self.review_points.clicked.connect(lambda: self.page.open_reference_point_review()
            if callable(getattr(self.page,'open_reference_point_review',None)) else None)
        self.review_points.setParent(frame_box)
        self.review_points.hide()
        layout.addWidget(frame_box)

        crop_box = QGroupBox("Crop area")
        crop_layout = QVBoxLayout(crop_box)
        self.crop_same = QRadioButton("Same as reference quad")
        self.crop_same.setToolTip("Setup crops to the reference quadrilateral, which also defines the alignment.")
        self.crop_separate = QRadioButton("Separate area")
        self.crop_separate.setToolTip(
            "Setup crops to the rig-aligned rectangle between four points marked on the reference photo. "
            "Alignment and physical size still come from the reference quadrilateral.")
        self.crop_same.setChecked(True)
        for button in (self.crop_same, self.crop_separate):
            button.clicked.connect(self._crop_choice_clicked)
            crop_layout.addWidget(button)
        crop_row = QHBoxLayout()
        self.mark_crop = QPushButton("Mark crop corners")
        self.mark_crop.setCheckable(True)
        self.mark_crop.setToolTip(
            "Click the four corners of the area to keep, in the reference points' order. "
            "Zoom with the mouse wheel for precision.")
        self.mark_crop.toggled.connect(self._mark_crop_toggled)
        crop_row.addWidget(self.mark_crop)
        crop_row.addStretch(1)
        crop_layout.addLayout(crop_row)
        self.crop_note = QLabel(
            "One crop area, marked on the reference run's photograph. Apply rig corners to Setup "
            "moves it to every run together with the reference quadrilateral.")
        self.crop_note.setObjectName("secondaryText")
        self.crop_note.setWordWrap(True)
        crop_layout.addWidget(self.crop_note)
        self.crop_status = QLabel()
        self.crop_status.setWordWrap(True)
        crop_layout.addWidget(self.crop_status)
        layout.addWidget(crop_box)

        self.static_references = None
        if legacy_static_references:
            from .rig_static_reference_ui import StaticReferencePanel
            self.static_references = StaticReferencePanel(self)
            layout.addWidget(self.static_references)

        points_box = QGroupBox("Injection points")
        self.injection_widget = points_box
        points_layout = QVBoxLayout(points_box)
        self.points = QTableWidget(0, len(POINT_COLUMNS))
        self.points.setHorizontalHeaderLabels(list(POINT_COLUMNS))
        self.points.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.points.verticalHeader().setVisible(False)
        header = self.points.horizontalHeader()
        for column in range(len(POINT_COLUMNS) - 1):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(
            len(POINT_COLUMNS) - 1, QHeaderView.ResizeMode.Stretch
        )
        self.points.itemChanged.connect(self._point_edited)
        self.points.itemSelectionChanged.connect(self._injection_selection_changed)
        if not scrollable:
            self.points.setMaximumHeight(120)
        points_layout.addWidget(self.points)
        point_buttons = QHBoxLayout()
        add = QPushButton("Add")
        add.clicked.connect(self._add_point)
        point_buttons.addWidget(add)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_points)
        point_buttons.addWidget(remove)
        self.place_point = QPushButton("Click to place selected")
        self.place_point.setCheckable(True)
        self.place_point.toggled.connect(self._place_toggled)
        point_buttons.addWidget(self.place_point)
        point_buttons.addStretch(1)
        points_layout.addLayout(point_buttons)
        point_buttons = QHBoxLayout()
        self.load_protocol = QPushButton("Load from injection protocol")
        self.load_protocol.clicked.connect(self._load_protocol)
        point_buttons.addWidget(self.load_protocol)
        self.copy_points = QPushButton("Copy coordinates")
        self.copy_points.clicked.connect(self._copy_points)
        point_buttons.addWidget(self.copy_points)
        point_buttons.addStretch(1)
        points_layout.addLayout(point_buttons)
        layout.addWidget(points_box)

        self.status = QLabel()
        self.status.setObjectName("secondaryText")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)

        self.canvas.image_point_selected.connect(self._canvas_clicked)
        self.canvas.scene().sceneRectChanged.connect(lambda *_: self._draw_overlay())
        self.cancel_placement_shortcut = QShortcut(QKeySequence('Esc'), page if isinstance(page, QWidget) else self)
        self.cancel_placement_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.cancel_placement_shortcut.setEnabled(False)
        self.cancel_placement_shortcut.activated.connect(self._cancel_placement)
        self.load()

    # ----------------------------------------------------------------- state

    def move_injection_controls(self, host: QWidget | None = None) -> None:
        """Move the same live controls into a host, or restore the combined panel."""
        destination = host.layout() if host is not None else self._content_layout
        if destination is None:
            raise ValueError("The injection-point host needs a layout")
        previous = self.injection_widget.parentWidget()
        if previous is not None and previous.layout() is not None:
            previous.layout().removeWidget(self.injection_widget)
        if host is None:
            destination.insertWidget(destination.indexOf(self.status), self.injection_widget)
        else:
            destination.addWidget(self.injection_widget)
        self.injection_widget.setTitle("" if host is not None else "Injection points")
        self.injection_widget.setEnabled(self.isEnabled())
        self.injection_widget.show()

    def setEnabled(self, enabled: bool) -> None:
        """Keep reparented injection controls under the controller's enable guard."""
        enabled = bool(enabled) and not getattr(self.page, '_point_review_active', False)
        super().setEnabled(enabled)
        if not enabled and self._mode:
            self._cancel_placement()
        widget = getattr(self, 'injection_widget', None)
        if widget is not None and not self.isAncestorOf(widget):
            widget.setEnabled(enabled)
        if hasattr(self, 'place_reference_point'):
            self._update_reference_placement_controls()

    def reference(self) -> SeriesRunInputs | None:
        return self._reference()

    def load(self) -> None:
        """Read the frame of the reference run and the saved injection points."""

        self._set_mode("")
        reference = self.reference()
        run_id = reference.run_id if reference is not None else ""
        if run_id != self._loaded_run:
            # Photographs shown for another run do not belong to this frame.
            self._loaded_run = run_id
            self._photo_path = None
            self._photo_size = None
            self._shown_preview = None
            self.photo_label.setText("No photograph shown.")
        frame = None
        if reference is not None:
            frame = saved_rig_frames(self.project).get(reference.run_id)
            if frame is None:
                frame = rig_frame_from_inputs(reference)
        self.frame = frame
        self._loading = True
        try:
            self._show_frame()
            self.points.setRowCount(0)
            for point in saved_injection_points(self.project):
                self._append_point(point)
            self._refresh_point_pixels()
        finally:
            self._loading = False
        if (
            self._photo_path is None
            and reference is not None
            and reference.baseline_image
        ):
            try:
                from darsia.fluidflower.layer_painting import reference_camera_u

                width, height = photo_size(reference.baseline_image,
                                           camera_u=reference_camera_u(self.project, reference))
                self._photo_size = (width, height)
            except (OSError, ValueError):
                self._photo_size = None
        self._load_crop()
        self._update_status()
        self._draw_overlay()

    def _show_frame(self) -> None:
        reference = self.reference()
        frame = self.frame
        geometry = getattr(frame, 'reference_geometry', None) or {}
        photo_corners = geometry.get('photo_corners_rc', frame.corners if frame is not None else {})
        from .rig_reference_points import reference_point_coordinates_mm, point_description
        physical = reference_point_coordinates_mm(frame) if frame is not None else {}
        disabled = set(frame.disabled_corners) if frame is not None else set()
        blocked = self.corners.blockSignals(True)
        for row, name in enumerate(CLICK_ORDER):
            combo = self.description_boxes[name]
            combo.blockSignals(True)
            description = point_description(frame, name) if frame is not None else 'rig_frame'
            if combo.findData(description) < 0:
                combo.addItem(description, description)
            combo.setCurrentIndex(combo.findData(description))
            combo.blockSignals(False)
            item = _pixel_item(*photo_corners[name]) if name in photo_corners else QTableWidgetItem()
            if name in disabled:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable & ~Qt.ItemFlag.ItemIsEnabled)
            self.corners.setItem(row, REFERENCE_PIXEL_COLUMN, item)
            for column, index in ((2, 0), (3, 1)):
                value = physical[name][index] if physical.get(name) is not None else None
                metric_item = QTableWidgetItem(_fmt(value, 1) if value is not None else '—')
                metric_item.setFlags(metric_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                metric_item.setToolTip("Physical millimetres relative to point 4 (bottom left). Defined by the measured reference shape.")
                self.corners.setItem(row, column, metric_item)
            visible = QTableWidgetItem()
            visible.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            visible.setCheckState(Qt.CheckState.Unchecked if name in disabled else Qt.CheckState.Checked)
            self.corners.setItem(row, REFERENCE_VISIBLE_COLUMN, visible)
        self.corners.blockSignals(blocked)
        self.width_m.setValue(float(frame.width_m or 0.0) if frame else 0.0)
        self.height_m.setValue(float(frame.height_m or 0.0) if frame else 0.0)
        shape = geometry.get('shape', 'rectangle')
        blocked = self.reference_shape.blockSignals(True)
        self.reference_shape.setCurrentIndex(self.reference_shape.findData(shape))
        self.reference_shape.blockSignals(blocked)
        measurements = geometry.get('measurements_m') or {}
        defaults = dict(top=self.width_m.value(), bottom=self.width_m.value(), left=self.height_m.value(),
                        right=self.height_m.value(), diagonal_ac=math.hypot(self.width_m.value(), self.height_m.value()))
        for key, spin in self.side_lengths.items():
            spin.setValue(measurements.get(key, defaults[key]))
        self.rectangle_size.setVisible(shape == 'rectangle')
        self.measured_sides.setVisible(shape == 'quadrilateral')
        self.reference_error.clear()
        self._update_class_hint()
        if reference is None:
            text = "Choose a reference run."
        elif frame is None:
            text = (
                "Mark four visible reference points in the photograph and enter "
                "their physical distances."
            )
        else:
            text = ""
        if frame is not None and not frame.has_size():
            text = "Enter the physical dimensions of the reference area."
        if disabled:
            text = (text + ' ' if text else '') + 'Not visible: ' + ', '.join(CORNER_LABELS[name] for name in CLICK_ORDER if name in disabled) + '.'
            if frame is not None and not frame.complete():
                text += ' Keep the known dimensions; use measured references to locate the unobserved corners.'
        self.frame_status.setText(text)
        self.frame_status.setVisible(bool(text))
        configured = (
            reference is not None and rig_frame_from_inputs(reference) is not None
        )
        self.use_configuration.setEnabled(
            configured
            and frame is not None
            and frame.source != FRAME_FROM_CONFIGURATION
        )
        self.copy_crop.setEnabled(frame is not None and frame.complete())
        self._update_reference_placement_controls()
        self.restore_reference.setEnabled(self._approved_reference() is not None)
        # The crop rectangle follows the reference quadrilateral.
        self._show_crop()

    def _frame_from_ui(self) -> RigFrame | None:
        reference = self.reference()
        corners = self.reference_photo_corners()
        disabled = tuple(name for row, name in enumerate(CLICK_ORDER)
            if self.corners.item(row, REFERENCE_VISIBLE_COLUMN) is not None
            and self.corners.item(row, REFERENCE_VISIBLE_COLUMN).checkState() == Qt.CheckState.Unchecked)
        if not corners and not disabled:
            return None
        shape = self.reference_shape.currentData()
        reference_geometry = None
        output_corners = dict(corners)
        width, height = self.width_m.value() or None, self.height_m.value() or None
        old_geometry = getattr(self.frame, 'reference_geometry', None)
        if len(corners) == 4:
            from .rig_reference_quadrilateral import reference_frame_geometry
            # Named rows are identities, including on rotated photographs.
            # Validate them instead of repairing an edit by renaming points.
            reference_frame_geometry(corners, measurements_m=dict(width=1., height=1.))
        if disabled and old_geometry:
            reference_geometry = dict(old_geometry, photo_corners_rc={name: list(point) for name, point in corners.items()})
        if len(corners) == 4 and not disabled and (shape == 'quadrilateral' or (width and height)):
            from .rig_reference_quadrilateral import reference_frame_geometry
            measurements = (dict(width=width, height=height) if shape == 'rectangle' else
                            {key: spin.value() for key, spin in self.side_lengths.items()})
            result = reference_frame_geometry(output_corners, shape=shape, measurements_m=measurements)
            output_corners, width, height = result['corners'], result['width_m'], result['height_m']
            reference_geometry = result['reference_geometry']
            if (old_geometry and all(np.array_equal(old_geometry.get('photo_corners_rc', {}).get(name, (np.nan, np.nan)),
                    reference_geometry['photo_corners_rc'][name]) for name in CLICK_ORDER)
                    and old_geometry.get('shape') == shape
                    and old_geometry.get('measurements_m') == reference_geometry['measurements_m']):
                reference_geometry = dict(old_geometry)
        descriptions = {name: combo.currentData() for name, combo in self.description_boxes.items()}
        if reference_geometry is None and any(value != 'rig_frame' for value in descriptions.values()):
            reference_geometry = dict(old_geometry or {}, shape=shape,
                photo_corners_rc={name: list(point) for name, point in corners.items()})
        if reference_geometry is None and len(corners) == 4:
            reference_geometry = dict(shape=shape,
                photo_corners_rc={name: list(point) for name, point in corners.items()})
        if reference_geometry is not None:
            reference_geometry['point_descriptions'] = descriptions
        source = FRAME_MARKED
        if (self.frame is not None and self.frame.source == FRAME_FROM_STATIC_REFERENCES
                and output_corners == self.frame.corners
                and width == self.frame.width_m and height == self.frame.height_m):
            # Visibility alone does not invalidate a model computed from measurements.
            source = FRAME_FROM_STATIC_REFERENCES
        return RigFrame(
            corners=output_corners,
            width_m=width,
            height_m=height,
            run_id=reference.run_id if reference is not None else "",
            source=source,
            image=str(
                self._photo_path or (reference.baseline_image if reference else "")
            ),
            image_size=self._photo_size,
            disabled_corners=disabled,
            reference_geometry=reference_geometry,
        )

    def _store_frame(self) -> None:
        try:
            frame = self._frame_from_ui()
        except ValueError as error:
            self.reference_error.setText(str(error))
            self.copy_crop.setEnabled(False)
            return
        if frame is None or not frame.run_id:
            return
        if self.frame is not None and self.frame.source == FRAME_FROM_CONFIGURATION:
            unchanged = (
                frame.corners == self.frame.corners
                and frame.width_m == self.frame.width_m
                and frame.height_m == self.frame.height_m
                and frame.disabled_corners == self.frame.disabled_corners
                and frame.reference_geometry == self.frame.reference_geometry
            )
            if unchanged:
                return
        try:
            save_rig_frame(self.project, frame, actor=default_actor())
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Could not save the rig frame", str(error))
            return
        self.frame = frame
        self._loading = True
        try:
            self._show_frame()
            self._refresh_point_pixels()
        finally:
            self._loading = False
        self._update_status()
        self._draw_overlay()
        self.changed.emit()

    def _corner_edited(self, _item: QTableWidgetItem) -> None:
        if not self._loading:
            self._set_mode('')
            self._store_frame()

    def reference_photo_corners(self) -> dict[str, tuple[float, float]]:
        result = {}
        for row, name in enumerate(CLICK_ORDER):
            point = _pixel_value(self.corners.item(row, REFERENCE_PIXEL_COLUMN))
            if point is not None:
                result[name] = point
        return result

    def _description_edited(self, name):
        if not self._loading:
            self._set_mode('')
            self._store_frame()

    def _size_edited(self) -> None:
        if not self._loading:
            self._set_mode('')
            self._store_frame()
        self._update_class_hint()

    def _reference_shape_changed(self):
        shape = self.reference_shape.currentData()
        self.rectangle_size.setVisible(shape == 'rectangle')
        self.measured_sides.setVisible(shape == 'quadrilateral')
        if not self._loading:
            self._set_mode('')
            self._store_frame()

    def _approved_reference(self):
        try:
            from .rig_point_review import load_review, reviewed_frame
            reference = self.reference()
            if reference is None or reference.baseline_image is None:
                return None
            document = load_review(self.project)
            if document is None:
                return None
            frame = reviewed_frame(document, reference.run_id)
            expected = Path(reference.baseline_image).resolve()
            if Path(frame.image).resolve() != expected:
                return None
            if self._photo_path is not None and Path(self._photo_path).resolve() != expected:
                return None
            if frame.image_size and self._photo_size and tuple(frame.image_size) != tuple(self._photo_size):
                return None
            return frame
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _restore_approved_reference(self):
        self._cancel_placement()
        frame = self._approved_reference()
        if frame is None:
            self.reference_error.setText('No current approved reference is bound to this run and photograph.')
            self.restore_reference.setEnabled(False)
            return
        try:
            save_rig_frame(self.project, frame, actor=default_actor())
        except (OSError, ValueError) as error:
            self.reference_error.setText('Could not restore the approved reference: ' + str(error))
            return
        self.frame = frame
        self._loading = True
        try:
            self._show_frame()
            self._refresh_point_pixels()
        finally:
            self._loading = False
        self._update_status()
        self._draw_overlay()
        self.changed.emit()

    # ------------------------------------------------------------ setup crop

    def _load_crop(self) -> None:
        """Read the series' Setup crop choice and the reference run's separate area."""

        self._crop_choice, self._crop_area, self._crop_error = CROP_REFERENCE_QUAD, None, ""
        reference = self.reference()
        try:
            self._crop_choice = saved_crop_choice(self.project)
            if reference is not None:
                self._crop_area = saved_crop_areas(self.project).get(reference.run_id)
        except (OSError, ValueError) as error:
            self._crop_error = f"Could not read the Setup crop: {error}"
        try:
            # The export judges the saved revision, not this tab's display frame.
            from .geometry_workspace_frame import current_configuration

            self._crop_reason = separate_crop_reason(
                self.project, configuration=current_configuration(self.project))
        except (OSError, ValueError) as error:
            self._crop_reason = f"The saved geometry revision could not be read: {error}"
        self._show_crop()

    def _crop_text(self) -> str:
        """What Setup crops to now, or what is missing for the chosen crop."""

        if self._crop_choice != CROP_SEPARATE_AREA:
            return "Setup crops to the reference quadrilateral."
        if self._crop_reason:
            return self._crop_reason
        frame, area = self.frame, self._crop_area
        if frame is None or not frame.complete():
            return "Mark the reference quadrilateral first; the crop area is marked on the same photograph."
        if area is None:
            return "Mark the four corners of the crop area on the reference photograph."
        if not area.image or not frame.image or not same_storage_path(area.image, frame.image):
            return ("The crop area was marked on another photograph than the reference "
                    "quadrilateral. Mark its corners again.")
        try:
            extent = crop_extent(frame, area)
        except ValueError as error:
            return str(error)
        return (
            f"Setup crops to x {1000 * extent.left_m:.1f} to {1000 * extent.right_m:.1f} mm and "
            f"y {1000 * extent.bottom_m:.1f} to {1000 * extent.top_m:.1f} mm "
            f"({1000 * extent.width_m:.1f} x {1000 * extent.height_m:.1f} mm). "
            "A changed crop takes effect after a new Setup export, Apply rig corners to Setup "
            "and Finish setup."
        )

    def _show_crop(self) -> None:
        """Show the crop choice, what Setup crops to and whether corners can be marked."""

        separate = self._crop_choice == CROP_SEPARATE_AREA
        (self.crop_separate if separate else self.crop_same).setChecked(True)
        # A stored separate choice stays selectable, so it can be changed back.
        self.crop_separate.setEnabled(separate or not self._crop_reason)
        self.crop_separate.setToolTip(self._crop_reason or (
            "Setup crops to the rig-aligned rectangle between four points marked on the "
            "reference photograph. Alignment and physical size still come from the "
            "reference quadrilateral."))
        self.mark_crop.setVisible(separate)
        self.crop_note.setVisible(separate)
        self.mark_crop.setEnabled(
            separate and not self._crop_reason and self.frame is not None and self.frame.complete())
        text = self._crop_text()
        self.crop_status.setText(f"{self._crop_error}\n{text}" if self._crop_error else text)

    def _crop_choice_clicked(self) -> None:
        choice = CROP_SEPARATE_AREA if self.crop_separate.isChecked() else CROP_REFERENCE_QUAD
        if self._mode == "crop":
            self._cancel_placement()
        if choice == CROP_SEPARATE_AREA and self._crop_reason:
            self._show_crop()
            return
        error = ""
        try:
            save_crop_choice(self.project, choice, actor=default_actor())
            frame = self.frame
            if (choice == CROP_SEPARATE_AREA and self._crop_area is None
                    and frame is not None and frame.complete()):
                # The separate area starts as a copy of the reference quadrilateral.
                save_crop_area(self.project, crop_area_from_frame(frame), actor=default_actor())
        except (OSError, ValueError) as exc:
            error = f"Could not save the Setup crop: {exc}"
        self._load_crop()
        if error:
            self._crop_error = error
            self._show_crop()
        self._draw_overlay()

    def _crop_photo_reason(self) -> str:
        """Why crop corners cannot be clicked on the photograph shown now ("" if they can)."""

        frame, reference = self.frame, self.reference()
        if frame is None or not frame.complete() or reference is None:
            return "Mark the reference quadrilateral first."
        if (self._photo_size is None or self._shown_preview is None
                or self.canvas.preview is not self._shown_preview):
            return "Show the reference photograph first."
        preview = self.canvas.preview or {}
        shown_run = preview.get("run_id")
        if shown_run and str(shown_run).casefold() != reference.run_id.casefold():
            return "Show the reference run's photograph first."
        shown = self._photo_path or preview.get("path") or reference.baseline_image
        if not frame.image or shown is None or not same_storage_path(shown, frame.image):
            return "Show the photograph the reference quadrilateral was marked on."
        if frame.image_size and tuple(frame.image_size) != tuple(self._photo_size):
            return "The photograph shown has another size than the reference quadrilateral's."
        return ""

    def _mark_crop_toggled(self, active: bool) -> None:
        self._crop_points = {}
        if active:
            reason = self._crop_photo_reason() if self._ensure_photo() else "Show the reference photograph first."
            if reason:
                self._set_mode("")
                self._crop_error = reason
                self._show_crop()
                self._update_status()
                return
        self._crop_error = ""
        self._set_mode("crop" if active else "")
        self._show_crop()
        self._update_status()
        self._draw_overlay()

    def _store_crop_area(self, corners: dict[str, tuple[float, float]]) -> None:
        """Save the clicked crop area if Setup can crop to it."""

        frame = self.frame
        error = ""
        try:
            area = CropArea(corners=corners, run_id=frame.run_id, image=frame.image,
                            image_size=self._photo_size)
            crop_extent(frame, area)
        except ValueError as exc:
            error = f"Crop area not saved: {exc}"
        else:
            try:
                save_crop_area(self.project, area, actor=default_actor())
            except (OSError, ValueError) as exc:
                error = f"Could not save the crop area: {exc}"
        self._load_crop()
        if error:
            self._crop_error = error
            self._show_crop()
        self._draw_overlay()

    # ------------------------------------------------------------- rig class

    def set_rig_class(self, key: str) -> None:
        """The series' rig class (a key of ``RIG_CLASSES``; "" if unset)."""

        self._rig_class = str(key or "")
        self._update_class_hint()

    def _update_class_hint(self) -> None:
        rig = RIG_CLASSES.get(self._rig_class)
        if rig is None:
            self.class_row.hide()
            return
        self.class_row.show()
        if rig.width_m is None or rig.height_m is None:
            self.class_size.setText(f"{rig.label}: no nominal size.")
            self.use_class_size.hide()
            return
        self.class_size.setText(f"{rig.label}: nominal inner size {rig.size_text()}.")
        self.class_size.setToolTip(rig.source)
        self.use_class_size.show()
        same = (
            abs(self.width_m.value() - rig.width_m) < 5.0e-5
            and abs(self.height_m.value() - rig.height_m) < 5.0e-5
        )
        self.use_class_size.setEnabled(not same)

    def apply_class_size(self) -> None:
        """Enter the class' nominal inner size as the rig size."""

        rig = RIG_CLASSES.get(self._rig_class)
        if rig is None or rig.width_m is None or rig.height_m is None:
            return
        self._loading = True
        try:
            self.width_m.setValue(float(rig.width_m))
            self.height_m.setValue(float(rig.height_m))
        finally:
            self._loading = False
        self._store_frame()
        self._update_class_hint()

    def _use_configuration(self) -> None:
        reference = self.reference()
        if reference is None:
            return
        try:
            remove_rig_frame(self.project, reference.run_id, actor=default_actor())
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Could not remove the rig frame", str(error))
            return
        self.load()
        self.changed.emit()

    def _copy_crop(self) -> None:
        if self.frame is None or not self.frame.complete():
            return
        table = self.frame.native_crop()
        if self._crop_choice == CROP_SEPARATE_AREA:
            # Copy the crop Setup uses: the separate area's rectangle.
            try:
                if self._crop_area is None:
                    raise ValueError("no crop area is marked.")
                table = crop_extent(self.frame, self._crop_area).native_crop()
            except ValueError as error:
                self.status.setText(f"No crop section copied: {error}")
                return
        lines = ["[corrections.curvature.crop]"]
        for key, value in table.items():
            if value is True:
                text = "true"
            elif isinstance(value, list):
                text = f"[{value[0]}, {value[1]}]"
            else:
                text = repr(value)
            lines.append(f"{key} = {text}")
        QGuiApplication.clipboard().setText("\n".join(lines) + "\n")
        self.status.setText("Crop section copied.")

    # ---------------------------------------------------------------- points

    def _append_point(self, point: dict[str, Any]) -> None:
        row = self.points.rowCount()
        self.points.insertRow(row)
        pixel = point.get("pixel") or {}
        values = (
            point.get("location_id") or "",
            point.get("name") or "",
            _fmt(point.get("x_m"), 4),
            _fmt(point.get("y_m"), 4),
            _fmt(pixel.get("row"), 1),
            _fmt(pixel.get("col"), 1),
            point.get("source") or "",
        )
        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            if column >= 4:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.points.setItem(row, column, item)
        self.points.item(row, 0).setData(Qt.ItemDataRole.UserRole, dict(pixel) or None)

    def _row_point(self, row: int) -> dict[str, Any]:
        def text(column: int) -> str:
            item = self.points.item(row, column)
            return item.text().strip() if item is not None else ""

        pixel = (
            self.points.item(row, 0).data(Qt.ItemDataRole.UserRole)
            if self.points.item(row, 0)
            else None
        )
        return {
            "location_id": text(0),
            "name": text(1),
            "x_m": _number(text(2)),
            "y_m": _number(text(3)),
            "pixel": (
                pixel
                if isinstance(pixel, dict) and pixel.get("row") is not None
                else None
            ),
            "source": text(6),
        }

    def current_points(self) -> list[dict[str, Any]]:
        return [self._row_point(row) for row in range(self.points.rowCount())]

    def _store_points(self) -> None:
        try:
            save_injection_points(
                self.project, self.current_points(), actor=default_actor()
            )
        except (OSError, ValueError) as error:
            self.status.setText(f"Injection points not saved: {error}")
            return
        self._update_status()
        self.changed.emit()

    def _point_edited(self, item: QTableWidgetItem) -> None:
        if self._loading:
            return
        self._set_mode('')
        row = item.row()
        if item.column() in (2, 3):
            self._loading = True
            try:
                self.points.item(row, 6).setText("typed")
                # Typed coordinates replace a picked pixel.
                self.points.item(row, 0).setData(Qt.ItemDataRole.UserRole, None)
                self._refresh_point_pixels()
            finally:
                self._loading = False
        self._store_points()
        self._draw_overlay()

    def _refresh_point_pixels(self) -> None:
        """Show each point's pixel in the reference photograph."""

        frame = self.frame
        reference = self.reference()
        for row in range(self.points.rowCount()):
            point = self._row_point(row)
            pixel = point.get("pixel")
            shown: tuple[float, float] | None = None
            usable_frame = frame is not None and frame.complete() and frame.has_size()
            if pixel and (
                reference is None or pixel.get("run_id") in ("", reference.run_id)
            ):
                shown = (pixel["row"], pixel["col"])
                if usable_frame and (point["x_m"] is None or point["y_m"] is None):
                    x_m, y_m = frame.pixel_to_rig(pixel["row"], pixel["col"])
                    self.points.item(row, 2).setText(_fmt(x_m, 4))
                    self.points.item(row, 3).setText(_fmt(y_m, 4))
            elif usable_frame and point["x_m"] is not None and point["y_m"] is not None:
                shown = frame.rig_to_pixel(point["x_m"], point["y_m"])
            self.points.item(row, 4).setText(_fmt(shown[0], 1) if shown else "")
            self.points.item(row, 5).setText(_fmt(shown[1], 1) if shown else "")

    def _add_point(self) -> None:
        used = {point["location_id"] for point in self.current_points()}
        number = 1
        while f"p{number}" in used:
            number += 1
        self._loading = True
        try:
            self._append_point(
                {
                    "location_id": f"p{number}",
                    "name": "",
                    "x_m": None,
                    "y_m": None,
                    "source": "",
                }
            )
        finally:
            self._loading = False
        self.points.selectRow(self.points.rowCount() - 1)
        self._store_points()

    def _remove_points(self) -> None:
        rows = sorted(
            {index.row() for index in self.points.selectedIndexes()}, reverse=True
        )
        for row in rows:
            self.points.removeRow(row)
        if rows:
            self._store_points()
            self._draw_overlay()

    def _load_protocol(self) -> None:
        reference = self.reference()
        path = reference.injection_protocol if reference is not None else None
        if path is None or not Path(path).is_file():
            QMessageBox.information(
                self,
                "No injection protocol",
                "The reference run has no readable injection protocol.",
            )
            return
        try:
            incoming = protocol_injection_points(Path(path))
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.warning(
                self, "Could not read the injection protocol", str(error)
            )
            return
        existing = {
            (round(point["x_m"], 6), round(point["y_m"], 6))
            for point in self.current_points()
            if point["x_m"] is not None and point["y_m"] is not None
        }
        used = {point["location_id"] for point in self.current_points()}
        added = 0
        self._loading = True
        try:
            for point in incoming:
                key = (round(point["x_m"], 6), round(point["y_m"], 6))
                if key in existing:
                    continue
                identifier = point["location_id"]
                while identifier in used:
                    identifier += "'"
                used.add(identifier)
                point["location_id"] = identifier
                self._append_point(point)
                added += 1
            self._refresh_point_pixels()
        finally:
            self._loading = False
        self.status.setText(
            f"{added} injection point(s) read from {Path(path).name}."
            if added
            else "The protocol's injection points are already listed."
        )
        if added:
            self._store_points()
            self._draw_overlay()

    def _copy_points(self) -> None:
        lines = ["location_id,location_x,location_y"]
        for point in self.current_points():
            if point["x_m"] is not None and point["y_m"] is not None:
                lines.append(
                    f"{point['location_id']},{point['x_m']:.4f},{point['y_m']:.4f}"
                )
        QGuiApplication.clipboard().setText("\n".join(lines) + "\n")
        self.status.setText("Injection-point coordinates copied.")

    # ----------------------------------------------------------------- photo

    def _choose_photo(self) -> None:
        reference = self.reference()
        from darsia.fluidflower.image_formats import photo_dialog_filter

        start = ""
        baseline = None
        if reference is not None and reference.baseline_image is not None:
            baseline = Path(reference.baseline_image)
            start = str(baseline.parent)
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Show a raw photograph",
            start,
            photo_dialog_filter(baseline),
        )
        if selected:
            self.show_photo(Path(selected))

    def show_reference_photo(self) -> None:
        reference = self.reference()
        if reference is None or reference.baseline_image is None:
            QMessageBox.information(
                self, "No reference photo", "The reference run has no baseline image."
            )
            return
        self.show_photo(Path(reference.baseline_image))

    def use_host_photo(self, size: tuple[int, int], preview: Any) -> None:
        """The host shows the reference photograph (``preview``) itself."""
        old = self._shown_preview or {}
        if self._mode and (old.get('run_id') != preview.get('run_id')
                or old.get('path') != preview.get('path') or self._photo_size != tuple(size)):
            self._cancel_placement()
        self._photo_path = None
        self._photo_size = (int(size[0]), int(size[1]))
        self._shown_preview = preview
        self._update_reference_placement_controls()
        self._draw_overlay()

    def show_photo(self, path: Path) -> bool:
        """Display a raw photograph in the canvas (reduced for display)."""

        from darsia.fluidflower.photograph_io import load_reduced

        host_show = getattr(self.page, "show_frame_photo", None)
        try:
            # A RAW of a run with a camera profile is shown on U, the grid of the frame.
            from darsia.fluidflower.layer_painting import photo_camera_u

            reference = self.reference()
            camera_u = (photo_camera_u(self.project, reference.run_id, path)
                        if reference is not None else None)
            width, height = photo_size(path, camera_u=camera_u)
            if not callable(host_show):
                factor = max(1, int(math.ceil(max(width, height) / _DISPLAY_LIMIT_PX)))
                rgb = load_reduced(path, factor, camera_u=camera_u)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Could not show the photograph", str(error))
            return False
        if callable(host_show):
            if not host_show(Path(path)):
                return False
        else:
            shape = rgb.shape[:2]
            preview = {
                "baseline": rgb,
                "labels": np.zeros(shape, dtype=np.int32),
                "facies": np.zeros(shape, dtype=np.int32),
                "depth": np.full(shape, np.nan, dtype=np.float32),
            }
            self.canvas.set_model(preview, self.page._collect_config())
        self._shown_preview = self.canvas.preview
        for name in ("background", "overlay"):
            combo = getattr(self.page, name, None)
            index = combo.findData("baseline") if combo is not None else -1
            if index >= 0:
                combo.setCurrentIndex(index)
        self._photo_path = Path(path)
        self._photo_size = (int(width), int(height))
        expected = None
        if self.frame is not None and self.frame.image_size:
            expected = tuple(self.frame.image_size)
        note = ""
        if expected and expected != self._photo_size:
            note = (
                f" The frame was marked on a {expected[0]} x {expected[1]} px photograph;"
                " use photographs of the same size."
            )
        self.photo_label.setText(
            f"Showing {Path(path).name} ({width} x {height} px).{note}"
        )
        self._update_reference_placement_controls()
        self._draw_overlay()
        return True

    # ----------------------------------------------------------- interaction

    def _host_hint(self, text: str) -> None:
        """Show what to click where the host page shows its hints ("" ends it)."""

        hint = getattr(self.page, "show_frame_hint", None)
        if callable(hint):
            hint(text)

    def _set_mode(self, mode: str) -> None:
        if mode and self.static_references is not None:
            self.static_references.cancel_interaction()
        self._mode = mode
        if mode != 'reference_point':
            self._selected_reference_name = None
        if mode != "crop":
            self._crop_points = {}
        if not mode:
            self._host_hint("")
        for button, name in (
            (self.mark_corners, "corners"),
            (self.place_reference_point, 'reference_point'),
            (self.place_point, "point"),
            (self.mark_crop, "crop"),
        ):
            button.blockSignals(True)
            button.setChecked(mode == name)
            button.blockSignals(False)
        self.canvas.place_mode = bool(mode)
        if mode:
            for name in ("place_injector", "select_layer", "draw_facies_example"):
                button = getattr(self.page, name, None)
                if button is not None and button.isChecked():
                    button.setChecked(False)
            self.canvas.place_mode = True
        shortcut = getattr(self, 'cancel_placement_shortcut', None)
        if shortcut is not None:
            shortcut.setEnabled(bool(mode))
        self._update_reference_placement_controls()

    def _cancel_placement(self):
        self._set_mode('')
        self._update_status()

    def _raw_photo_ready(self):
        if not self.isEnabled() or self._photo_size is None or self._shown_preview is None:
            return False
        if self.canvas.preview is not self._shown_preview:
            return False
        reference = self.reference()
        if reference is None:
            return False
        shown_run = (self.canvas.preview or {}).get('run_id')
        if shown_run and str(shown_run).casefold() != reference.run_id.casefold():
            return False
        shown_path = (self.canvas.preview or {}).get('path')
        if shown_path and reference.baseline_image is not None:
            if Path(shown_path).resolve() != Path(reference.baseline_image).resolve():
                return False
        if self._photo_path is not None and reference.baseline_image is not None:
            if Path(self._photo_path).resolve() != Path(reference.baseline_image).resolve():
                return False
        try:
            from .geometry_workspace_frame import frame_kind
            return frame_kind(getattr(self.page, '_workspace_frame_config', {})) == 'raw'
        except ValueError:
            return False

    def _selected_reference_row(self):
        selected = self.corners.currentRow()
        return selected if any(index.row() == selected for index in self.corners.selectedIndexes()) else -1

    def _update_reference_placement_controls(self):
        selected = self._selected_reference_row()
        valid = 0 <= selected < len(CLICK_ORDER)
        visible = self.corners.item(selected, REFERENCE_VISIBLE_COLUMN) if valid else None
        valid = valid and (visible is None or visible.checkState() != Qt.CheckState.Unchecked)
        self.place_reference_point.setEnabled(bool(valid and self._raw_photo_ready()))
        self.reference_target.setText(f'{selected+1} · {CORNER_LABELS[CLICK_ORDER[selected]]}'
            if 0 <= selected < len(CLICK_ORDER) else 'Select a reference row')

    def _reference_selection_changed(self):
        if not self._loading and self._mode:
            self._cancel_placement()
        self._update_reference_placement_controls()

    def _injection_selection_changed(self):
        if not self._loading and self._mode:
            self._cancel_placement()

    def _reference_place_toggled(self, active):
        selected = self._selected_reference_row()
        if active and (not self._raw_photo_ready() or not 0 <= selected < len(CLICK_ORDER)):
            self._cancel_placement()
            return
        self._selected_reference_name = CLICK_ORDER[selected] if active else None
        self._set_mode('reference_point' if active else '')
        self._update_status()

    def _ensure_photo(self) -> bool:
        """Clicks are mapped to the photograph this panel shows last."""

        if (
            self._shown_preview is not None
            and self.canvas.preview is self._shown_preview
        ):
            return True
        if self._photo_path is not None:
            return self.show_photo(self._photo_path)
        return self._show_default_photo()

    def _show_default_photo(self) -> bool:
        reference = self.reference()
        if reference is None or reference.baseline_image is None:
            QMessageBox.information(
                self, "No photograph", "Show a raw photograph of the rig first."
            )
            return False
        return self.show_photo(Path(reference.baseline_image))

    def _mark_corners_toggled(self, active: bool) -> None:
        if active and not self._ensure_photo():
            self._set_mode("")
            return
        self._corner_index = 0
        self._corner_rows = [row for row in range(len(CLICK_ORDER))
            if self.corners.item(row, REFERENCE_VISIBLE_COLUMN) is None
            or self.corners.item(row, REFERENCE_VISIBLE_COLUMN).checkState() != Qt.CheckState.Unchecked]
        if active and not self._corner_rows:
            self._set_mode('')
            self.status.setText('Select at least one visible corner before marking.')
            return
        self._set_mode("corners" if active else "")
        self._update_status()

    def _place_toggled(self, active: bool) -> None:
        if active:
            if self.points.currentRow() < 0:
                self._add_point()
            if not self._ensure_photo():
                self._set_mode("")
                return
        self._set_mode("point" if active else "")
        self._update_status()

    def _canvas_clicked(self, fx: float, fy: float) -> None:
        shown_run = (self.canvas.preview or {}).get('run_id')
        reference = self.reference()
        if shown_run and reference is not None and shown_run != reference.run_id:
            return
        if not self._mode or self._photo_size is None:
            return
        width, height = self._photo_size
        row = float(fy) * (height - 1)
        col = float(fx) * (width - 1)
        if self._mode == 'reference_point':
            name = self._selected_reference_name
            if name is None or not self._raw_photo_ready():
                self._cancel_placement()
                return
            self._loading = True
            try:
                self.corners.setItem(CLICK_ORDER.index(name), REFERENCE_PIXEL_COLUMN, _pixel_item(row, col))
            finally:
                self._loading = False
            self._set_mode('')
            self._store_frame()
            self._update_status()
        elif self._mode == "corners":
            index = self._corner_rows[self._corner_index]
            self._loading = True
            try:
                self.corners.setItem(index, REFERENCE_PIXEL_COLUMN, _pixel_item(row, col))
            finally:
                self._loading = False
            self._corner_index += 1
            if self._corner_index >= len(self._corner_rows):
                self._set_mode("")
                self._store_frame()
            else:
                self._draw_overlay()
            self._update_status()
        elif self._mode == "crop":
            reason = self._crop_photo_reason()
            if reason:
                self._cancel_placement()
                self._crop_error = reason
                self._show_crop()
                self._draw_overlay()
                return
            self._crop_points[CLICK_ORDER[len(self._crop_points)]] = (row, col)
            if len(self._crop_points) == len(CLICK_ORDER):
                corners = dict(self._crop_points)
                # Like a click on the button, so the page sees placement end.
                self.mark_crop.setChecked(False)
                self._store_crop_area(corners)
            else:
                self._draw_overlay()
            self._update_status()
        elif self._mode == "point":
            selected = self.points.currentRow()
            if selected < 0:
                return
            reference = self.reference()
            self._loading = True
            try:
                self.points.item(selected, 0).setData(
                    Qt.ItemDataRole.UserRole,
                    {
                        "run_id": reference.run_id if reference is not None else "",
                        "row": row,
                        "col": col,
                    },
                )
                frame = self.frame
                if frame is not None and frame.complete() and frame.has_size():
                    x_m, y_m = frame.pixel_to_rig(row, col)
                    self.points.item(selected, 2).setText(_fmt(x_m, 4))
                    self.points.item(selected, 3).setText(_fmt(y_m, 4))
                else:
                    self.points.item(selected, 2).setText("")
                    self.points.item(selected, 3).setText("")
                self.points.item(selected, 6).setText("picked in photo")
                self._refresh_point_pixels()
            finally:
                self._loading = False
            self._set_mode("")
            self._store_points()
            self._draw_overlay()

    def _update_status(self) -> None:
        if self._mode == 'reference_point':
            name = self._selected_reference_name
            text = (f'Click once to place {CORNER_LABELS[name]}. Only this reference point changes. Esc cancels.'
                    if name is not None else 'Select a reference point first.')
            self.status.setText(text)
            self._host_hint(text)
            return
        if self._mode == "corners":
            rows = self._corner_rows
            text = (
                f"Click {CORNER_LABELS[CLICK_ORDER[rows[self._corner_index]]]} "
                f"({self._corner_index + 1}/{len(rows)} visible corners). "
                "Zoom with the mouse wheel; the right button drags the "
                "view."
            )
            self.status.setText(text)
            self._host_hint(text)
            return
        if self._mode == "point":
            text = "Click the injection point in the photograph."
            self.status.setText(text)
            self._host_hint(text)
            return
        if self._mode == "crop":
            count = len(self._crop_points)
            text = (
                f"Click crop corner {count + 1}/4 ({CORNER_LABELS[CLICK_ORDER[count]]}). "
                "Zoom with the mouse wheel; the right button drags the view. Esc cancels."
            )
            self.status.setText(text)
            self._host_hint(text)
            return
        self._host_hint("")
        missing = [
            point["location_id"] or "point"
            for point in self.current_points()
            if point["x_m"] is None or point["y_m"] is None
        ]
        if missing:
            self.status.setText(
                "No rig coordinates yet for " + ", ".join(missing) + ": define the rig "
                "frame (corners, width and height)."
            )
        else:
            self.status.setText("")

    # --------------------------------------------------------------- overlay

    def set_overlay_visibility(self, *, points: bool = True, lines: bool = True) -> None:
        """Change only displayed reference/injection marks, never saved geometry."""
        self._show_reference_points = bool(points)
        self._show_reference_lines = bool(lines)
        self._draw_overlay()

    def _scene_point(self, row: float, col: float) -> QPointF | None:
        if self._photo_size is None:
            return None
        if getattr(self.canvas, 'photo_display_mapping', None) is not None:
            return self.canvas.raw_to_scene(row, col)
        bounds = self.canvas.pixmap_item.boundingRect()
        if bounds.width() <= 1 or bounds.height() <= 1:
            return None
        width, height = self._photo_size
        if (
            abs(bounds.width() / bounds.height() - width / height)
            > 0.01 * width / height
        ):
            return None
        return QPointF(
            col / max(1.0, width - 1.0) * (bounds.width() - 1.0),
            row / max(1.0, height - 1.0) * (bounds.height() - 1.0),
        )

    def _draw_overlay(self) -> None:
        scene = self.canvas.scene()
        for item in self._overlay:
            try:
                scene.removeItem(item)
            except RuntimeError:
                pass
        self._overlay = []
        shown_run = (self.canvas.preview or {}).get('run_id')
        reference = self.reference()
        if shown_run and reference is not None and shown_run != reference.run_id:
            return
        if self.canvas.preview is None:
            return
        bounds = self.canvas.pixmap_item.boundingRect()
        size = max(4.0, 0.006 * max(bounds.width(), bounds.height()))
        try:
            photo_corners = self.reference_photo_corners()
        except ValueError:
            return
        disabled = tuple(name for row, name in enumerate(CLICK_ORDER)
            if self.corners.item(row, REFERENCE_VISIBLE_COLUMN) is not None
            and self.corners.item(row, REFERENCE_VISIBLE_COLUMN).checkState() == Qt.CheckState.Unchecked)
        frame = RigFrame(corners=photo_corners, disabled_corners=disabled) if photo_corners else None
        if frame is not None:
            pen = QPen(QColor(64, 224, 208))
            pen.setCosmetic(True)
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.DashLine)
            points = [
                self._scene_point(*frame.corners[name])
                for name in CLICK_ORDER
                if name in frame.corners and name not in frame.disabled_corners
            ]
            points = [point for point in points if point is not None]
            if self._show_reference_lines and len(points) == 4:
                polygon = QGraphicsPolygonItem(QPolygonF(points))
                polygon.setPen(pen)
                polygon.setZValue(20)
                scene.addItem(polygon)
                self._overlay.append(polygon)
            elif self._show_reference_lines and points:
                for first, second in zip(CLICK_ORDER, CLICK_ORDER[1:]+CLICK_ORDER[:1]):
                    if (first in frame.corners and second in frame.corners
                            and first not in frame.disabled_corners and second not in frame.disabled_corners):
                        a, b = self._scene_point(*frame.corners[first]), self._scene_point(*frame.corners[second])
                        if a is not None and b is not None:
                            edge = QGraphicsLineItem(a.x(), a.y(), b.x(), b.y())
                            edge.setPen(pen)
                            edge.setZValue(20)
                            scene.addItem(edge)
                            self._overlay.append(edge)
            for name in CLICK_ORDER if self._show_reference_points else ():
                if name not in frame.corners or name in frame.disabled_corners:
                    continue
                point = self._scene_point(*frame.corners[name])
                if point is None:
                    continue
                dot = QGraphicsEllipseItem(
                    point.x() - size / 2, point.y() - size / 2, size, size
                )
                dot.setPen(pen)
                dot.setZValue(21)
                scene.addItem(dot)
                self._overlay.append(dot)
                number = QGraphicsSimpleTextItem(str(CLICK_ORDER.index(name)+1))
                number.setBrush(QColor(64, 224, 208))
                font = number.font()
                font.setBold(True)
                number.setFont(font)
                number.setPos(point.x()+size, point.y()-size)
                number.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
                number.setZValue(22)
                number.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                scene.addItem(number)
                self._overlay.append(number)
        if self._show_reference_lines:
            self._draw_crop_overlay(scene, size)
        if not self._show_reference_points:
            return
        marker_pen = QPen(QColor(238, 74, 83))
        marker_pen.setCosmetic(True)
        marker_pen.setWidth(2)
        for row in range(self.points.rowCount()):
            row_text = (
                self.points.item(row, 4).text() if self.points.item(row, 4) else ""
            )
            col_text = (
                self.points.item(row, 5).text() if self.points.item(row, 5) else ""
            )
            r, c = _number(row_text), _number(col_text)
            if r is None or c is None:
                continue
            point = self._scene_point(r, c)
            if point is None:
                continue
            ring = QGraphicsEllipseItem(
                point.x() - size, point.y() - size, 2 * size, 2 * size
            )
            ring.setPen(marker_pen)
            ring.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            ring.setZValue(22)
            scene.addItem(ring)
            label = QGraphicsSimpleTextItem(self.points.item(row, 0).text())
            label.setBrush(QBrush(QColor(248, 92, 99)))
            label.setPos(point.x() + 1.3 * size, point.y() - 2 * size)
            label.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
            )
            label.setZValue(22)
            scene.addItem(label)
            self._overlay += [ring, label]

    def _draw_crop_overlay(self, scene: Any, size: float) -> None:
        """Draw the marked crop area (dashed) and the rectangle Setup crops to (solid)."""

        if self._mode == "crop":
            marked = dict(self._crop_points)
        elif self._crop_choice == CROP_SEPARATE_AREA and self._crop_area is not None:
            marked = dict(self._crop_area.corners)
        else:
            return
        colour = QColor(255, 170, 0)
        pen = QPen(colour)
        pen.setCosmetic(True)
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        points = [self._scene_point(*marked[name]) for name in CLICK_ORDER if name in marked]
        points = [point for point in points if point is not None]
        items = []
        if len(points) == len(CLICK_ORDER):
            outline = QGraphicsPolygonItem(QPolygonF(points))
            outline.setPen(pen)
            items.append(outline)
        else:
            for first, second in zip(points, points[1:]):
                edge = QGraphicsLineItem(first.x(), first.y(), second.x(), second.y())
                edge.setPen(pen)
                items.append(edge)
        for point in points:
            dot = QGraphicsEllipseItem(point.x() - size / 2, point.y() - size / 2, size, size)
            dot.setPen(pen)
            items.append(dot)
        if self._mode != "crop" and self.frame is not None and self._crop_area is not None:
            try:
                extent = crop_extent(self.frame, self._crop_area)
            except ValueError:
                extent = None
            corners = [] if extent is None else [
                self._scene_point(*extent.corners[name]) for name in CLICK_ORDER]
            if corners and all(point is not None for point in corners):
                solid = QPen(colour)
                solid.setCosmetic(True)
                solid.setWidth(2)
                rectangle = QGraphicsPolygonItem(QPolygonF(corners))
                rectangle.setPen(solid)
                rectangle.setToolTip("The rectangle Setup crops to")
                items.append(rectangle)
        for item in items:
            item.setZValue(23)
            scene.addItem(item)
            self._overlay.append(item)


__all__ = ["RigFramePanel"]
