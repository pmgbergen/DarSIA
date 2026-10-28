"""Guided deviation painting using the original geometry painting canvas.

The injected typed draft owns rasterization and classification history. This
widget displays its class masks and forwards explicit human requests only;
it creates no exclusion, proof, target version, approval or adopted result.
"""
from copy import deepcopy
from collections.abc import Mapping

import numpy as np
from PySide6.QtCore import QEvent, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeySequence, QPen, QShortcut
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QGraphicsPathItem, QGridLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QVBoxLayout, QWidget,
)

from .painting_gap_review_ui import _gap_contours, _path, _project, _rgb
from .pages.geometry_facies.page import GeometryImageView


def _display_rgb(packet):
    """Retain the original comparison view's unavailable-support semantics."""
    rgb = _rgb(packet)
    support = packet.get("valid_support")
    if support is not None:
        support = np.asarray(support)
        if support.dtype != np.bool_ or support.shape != rgb.shape[:2]:
            raise ValueError("The comparison image lacks its exact boolean display support")
        rgb[~support] = 64
    return rgb


class DeviationPaintingStep(QWidget):
    """One deviation, with canonical class tokens supplied by the backend.

    ``class_choices`` contains (canonical token, English label) pairs and
    ``sand_class`` identifies the separate map-fix route. The injected draft
    exposes editable_mask, class_masks, can_undo/can_redo, paint(kind,
    points_xy_native, class_name, radius_px), undo/redo and snapshot. Class
    masks are preview classifications, never scientific layer identifiers.
    """

    photo_requested = Signal(str)
    preview_changed = Signal(object)
    requested = Signal(object)
    map_fix_requested = Signal(object)

    _COLOURS = ("#74c9ef", "#8de0a4", "#e9b14f", "#c5a5ef", "#ed9393")

    def __init__(self, *, class_choices, sand_class, parent=None):
        super().__init__(parent)
        choices = tuple(class_choices)
        if (len(choices) != 5 or any(len(row) != 2 for row in choices)
                or any(type(key) is not str or not key or type(label) is not str or not label
                       for key, label in choices)
                or len({key for key, _ in choices}) != 5
                or sand_class not in {key for key, _ in choices}):
            raise ValueError("Supply the five explicit classification choices and the sand map-fix token")
        self.class_choices, self.sand_class = choices, sand_class
        self.packet = self.draft = self._plain = None
        self._photo_role = "current"
        self._photo_pending = False
        self._requested_photo = None
        self._clearwater_available = False
        self._items = []
        self._suspended = False
        self._pending_gap_fit = None
        self._gap_fit_queued = False
        self._gap_fit_size = None
        self._gap_fit_timer = QTimer(self)
        self._gap_fit_timer.setSingleShot(True)
        self._gap_fit_timer.timeout.connect(self._fit_initial_gap)
        layout = QVBoxLayout(self)
        self.question = QLabel("What is inside the outlined area that the transferred map does not cover?")
        self.question.setWordWrap(True)
        layout.addWidget(self.question)
        instruction = QLabel("Choose what you see, then paint it with the same brush or polygon tool as the sand map. "
            "You can mark several classes in this area. Sand opens a correction to the map; it is never left out of the analysis.")
        instruction.setWordWrap(True)
        layout.addWidget(instruction)
        tools = QGridLayout()
        tools.addWidget(QLabel("What is here?"), 0, 0)
        self.classification = QComboBox()
        self.classification.addItem("Select a class to paint", None)
        for index, (key, label) in enumerate(choices):
            self.classification.addItem(label, key)
            self.classification.setItemData(index + 1, QColor(self._COLOURS[index]), Qt.ItemDataRole.ForegroundRole)
        self.classification.currentIndexChanged.connect(self._mode_changed)
        tools.addWidget(self.classification, 0, 1)
        self.mode = QComboBox()
        for label, key in (("Pan and zoom", "pan"), ("Brush", "brush"), ("Polygon", "polygon")):
            self.mode.addItem(label, key)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        tools.addWidget(self.mode, 0, 2)
        self.radius = QDoubleSpinBox()
        self.radius.setRange(.5, 200.)
        self.radius.setValue(3.)
        self.radius.setPrefix("Brush radius: ")
        self.radius.setSuffix(" image pixels")
        tools.addWidget(self.radius, 0, 3)
        self.finish = QPushButton("Finish polygon")
        self.finish.clicked.connect(lambda: self.canvas.finish_edit())
        self.undo = QPushButton("Undo")
        self.redo = QPushButton("Redo")
        self.undo.clicked.connect(lambda: self._history(False))
        self.redo.clicked.connect(lambda: self._history(True))
        tools.addWidget(self.finish, 1, 1)
        tools.addWidget(self.undo, 1, 2)
        tools.addWidget(self.redo, 1, 3)
        layout.addLayout(tools)
        self.canvas = GeometryImageView()
        self.canvas.viewport().installEventFilter(self)
        self.canvas.set_options(overlay="none", show_injectors=False)
        self.canvas.label_edit_drawn.connect(self._drawn)
        layout.addWidget(self.canvas, 1)
        photo_tools = QHBoxLayout()
        self.photo_toggle = QPushButton("Show clear-water image (Alt+I)")
        self.photo_toggle.clicked.connect(self.request_other_photo)
        photo_tools.addWidget(self.photo_toggle)
        self.photo_shortcut = QShortcut(QKeySequence("Alt+I"), self)
        self.photo_shortcut.activated.connect(self.request_other_photo)
        photo_tools.addWidget(QLabel("Brightness"))
        self.brightness = QSlider(Qt.Orientation.Horizontal)
        self.brightness.setRange(25, 300)
        self.brightness.setValue(100)
        self.brightness.valueChanged.connect(self._render)
        photo_tools.addWidget(self.brightness, 1)
        layout.addLayout(photo_tools)
        self.status = QLabel("The guided review will open the next uncovered area here.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.continue_button = QPushButton("Review these classifications")
        self.continue_button.clicked.connect(self._continue)
        layout.addWidget(self.continue_button)
        self._buttons()

    def _masks(self, *, packet=None, draft=None):
        packet = self.packet if packet is None else packet
        draft = self.draft if draft is None else draft
        masks = draft.class_masks
        if not isinstance(masks, Mapping) or set(masks) != {key for key, _ in self.class_choices}:
            raise ValueError("The classification draft has another class schema")
        editable = packet["gap_mask"]
        occupied = np.zeros(editable.shape, bool)
        result = {}
        for key, _ in self.class_choices:
            mask = np.asarray(masks[key])
            if (mask.dtype != np.bool_ or mask.shape != editable.shape
                    or np.any(mask & ~editable) or np.any(mask & occupied)):
                raise ValueError("Classification previews must be disjoint and inside this deviation")
            result[key] = mask
            occupied |= mask
        return result

    def set_context(self, packet, draft):
        """Install one admitted photo/deviation and its exact typed draft."""
        rgb = _display_rgb(packet)
        mask = np.asarray(packet["gap_mask"])
        matrix = np.asarray(packet["native_to_photo_h"], np.float64)
        if (not packet.get("run_id") or not packet.get("component_id")
                or not isinstance(packet.get("display_grid"), dict)
                or mask.dtype != np.bool_ or mask.ndim != 2 or not mask.any()
                or matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or np.linalg.det(matrix) == 0
                or packet["display_grid"].get("shape_hw") != list(mask.shape)
                or not np.array_equal(packet["display_grid"].get("native_to_photo_h"), matrix)
                or np.asarray(draft.editable_mask).dtype != np.bool_
                or not np.array_equal(draft.editable_mask, mask)):
            raise ValueError("The painting step needs the exact admitted deviation, native grid and draft")
        self._masks(packet=packet, draft=draft)
        same_context = (self.packet is not None and self.draft is draft
            and all(self.packet.get(key) == packet.get(key)
                    for key in ("run_id", "component_id", "display_grid")))
        self.packet, self.draft, self._plain = deepcopy(packet), draft, rgb
        self._photo_role = "current"
        self._photo_pending = False
        self._requested_photo = None
        self._clearwater_available = any(row.get("image_id") == "clearwater" and row.get("available") is True
            for row in packet.get("image_choices", []))
        if not same_context:
            self.classification.setCurrentIndex(0)
            self.mode.setCurrentIndex(0)
            self.canvas.cancel_edit()
        self._render(initial=not same_context)
        rows, cols = np.nonzero(mask)
        corners = _project([[cols.min()-.5, rows.min()-.5], [cols.max()+.5, rows.min()-.5],
                            [cols.max()+.5, rows.max()+.5], [cols.min()-.5, rows.max()+.5]], matrix)
        low, high = corners.min(axis=0), corners.max(axis=0)
        bounds = QRectF(float(low[0]), float(low[1]), float(high[0]-low[0]), float(high[1]-low[1]))
        margin = max(12., max(bounds.width(), bounds.height()) * .15)
        bounds.adjust(-margin, -margin, margin, margin)
        if not same_context:
            self._pending_gap_fit = bounds
            self._gap_fit_size = None
        self._queue_gap_fit()
        self.status.setText("The outline shows this uncovered area. Choose a class and paint only what you can identify.")
        self._buttons()

    def _queue_gap_fit(self):
        """Wait for the actual stacked editor viewport, once per new context."""
        if (self._pending_gap_fit is None or self._gap_fit_queued
                or not self.isVisible() or not self.canvas.viewport().isVisible()):
            return
        self._gap_fit_queued = True
        self._gap_fit_timer.start(0)

    def _fit_initial_gap(self):
        self._gap_fit_queued = False
        if (self._pending_gap_fit is None or not self.isVisible()
                or not self.canvas.viewport().isVisible()):
            return
        viewport = self.canvas.viewport()
        size = (viewport.width(), viewport.height())
        if size != self._gap_fit_size:
            self._gap_fit_size = size
            self._queue_gap_fit()  # Let the shown stack's queued layout settle.
            return
        bounds, self._pending_gap_fit = self._pending_gap_fit, None
        self.canvas.fitInView(bounds, Qt.AspectRatioMode.KeepAspectRatio)

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        self._queue_gap_fit()

    def eventFilter(self, watched, event):  # noqa: N802
        if watched is self.canvas.viewport():
            if event.type() in {QEvent.Type.Show, QEvent.Type.Resize}:
                self._queue_gap_fit()
            elif event.type() in {QEvent.Type.Wheel, QEvent.Type.MouseButtonPress}:
                # An explicit user view/edit gesture owns the view from now on.
                self._pending_gap_fit = None
                self._gap_fit_timer.stop()
                self._gap_fit_queued = False
        return super().eventFilter(watched, event)

    def set_suspended(self, suspended):
        """The owning guided review controls work/loading; no worker is created here."""
        self._suspended = bool(suspended)
        self._buttons()

    def set_brightness(self, percent):
        """Share the owning review's display slider without editing source RGB."""
        if type(percent) is not int or not 25 <= percent <= 300:
            raise ValueError("Display brightness must be between 25 and 300 percent")
        self.brightness.setValue(percent)

    def request_other_photo(self):
        if self.packet is None or self._photo_pending or self._suspended or not self._clearwater_available:
            return
        self._photo_pending = True
        self._requested_photo = "clearwater" if self._photo_role == "current" else "current"
        self._buttons()
        self.status.setText("Opening the other verified image; your zoom, position and painting are kept.")
        self.photo_requested.emit(self._requested_photo)

    def set_photo(self, packet):
        """Accept only the existing reader's photo on the unchanged display grid."""
        if self.packet is None:
            raise ValueError("Open a deviation before changing its photograph")
        rgb = _display_rgb(packet)
        role = packet.get("image_role")
        if (not self._photo_pending or role != self._requested_photo or rgb.shape != self._plain.shape
                or packet.get("display_grid") != self.packet["display_grid"]
                or not np.array_equal(packet.get("native_to_photo_h"), self.packet["native_to_photo_h"])):
            raise ValueError("The other photograph must retain this exact native display grid and transform")
        self._plain = rgb
        self._photo_role = role
        self._photo_pending = False
        self._requested_photo = None
        self._render()
        self.status.setText("Clear-water image; the same area and painting are shown." if role == "clearwater"
                            else "Experiment image; the same area and painting are shown.")
        self._buttons()

    def photo_failed(self, message):
        self._photo_pending = False
        self._requested_photo = None
        self.status.setText("The other image could not open: " + str(message))
        self._buttons()

    def _render(self, *_args, initial=False):
        if self.packet is None:
            return
        rgb = np.rint(np.clip(self._plain.astype(np.float32) * self.brightness.value()/100., 0, 255)).astype(np.uint8)
        preview = dict(baseline=rgb.astype(np.float32)/255.)
        if initial:
            self.canvas.set_model(preview, {})
        else:
            self.canvas.update_preview(preview, {})
        for item in self._items:
            self.canvas.scene().removeItem(item)
        self._items = []
        masks = self._masks()
        layers = [(self.packet["gap_mask"], QColor("magenta"), False)]
        for index, (key, _) in enumerate(self.class_choices):
            colour = QColor(self._COLOURS[index])
            colour.setAlpha(100)
            layers.append((masks[key], colour, True))
        for mask, colour, fill in layers:
            item = QGraphicsPathItem(_path(_gap_contours(mask), self.packet["native_to_photo_h"]))
            pen = QPen(QColor(colour), 2.)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setBrush(colour if fill else Qt.BrushStyle.NoBrush)
            item.setZValue(3 if fill else 4)
            self.canvas.scene().addItem(item)
            self._items.append(item)

    def _mode_changed(self, *_args):
        mode = self.mode.currentData()
        if mode != "pan" and self.classification.currentData() is None:
            self.mode.blockSignals(True)
            self.mode.setCurrentIndex(0)
            self.mode.blockSignals(False)
            mode = "pan"
            self.status.setText("First choose what you see in the area, then use brush or polygon.")
        self.canvas.set_edit_mode(mode)
        self._buttons()

    def _drawn(self, kind, points):
        if self.draft is None or self._photo_pending or self._suspended or self.classification.currentData() is None:
            return
        try:
            height, width = self._plain.shape[:2]
            photo = np.asarray(points, np.float64) * [width-1, height-1]
            native = _project(photo, np.linalg.inv(self.packet["native_to_photo_h"]))
            changed = self.draft.paint(kind=kind, points_xy_native=native.tolist(),
                class_name=self.classification.currentData(), radius_px=self.radius.value())
            if changed is False:
                self.status.setText("Paint inside the outlined uncovered area. This stroke changed nothing.")
                self._buttons()
                return
            self._render()
            self.preview_changed.emit(self.draft.snapshot())
            self.status.setText("Your draft is shown in colour. Mark another class, undo, or review these classifications.")
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            self.status.setText("The stroke could not be used: " + str(error))
        self._buttons()

    def _history(self, redo):
        if self.draft is None or self._photo_pending or self._suspended:
            return
        (self.draft.redo if redo else self.draft.undo)()
        self._render()
        self.preview_changed.emit(self.draft.snapshot())
        self._buttons()

    def _buttons(self):
        editable = self.draft is not None and not self._photo_pending and not self._suspended
        for control in (self.classification, self.mode, self.canvas):
            control.setEnabled(editable)
        self.radius.setEnabled(editable and self.mode.currentData() == "brush")
        self.finish.setEnabled(editable and self.mode.currentData() == "polygon")
        self.undo.setEnabled(editable and bool(self.draft.can_undo))
        self.redo.setEnabled(editable and bool(self.draft.can_redo))
        self.photo_toggle.setEnabled(editable and self._clearwater_available)
        self.photo_shortcut.setEnabled(editable and self._clearwater_available)
        self.photo_toggle.setText("Show experiment image (Alt+I)" if self._photo_role == "clearwater"
                                  else "Show clear-water image (Alt+I)")
        masks = self._masks() if self.draft is not None else {}
        sand = bool(masks and masks[self.sand_class].any())
        marked = any(mask.any() for mask in masks.values())
        self.continue_button.setText("Fix the sand map in this area" if sand else "Review these classifications")
        self.continue_button.setEnabled(editable and marked)

    def _continue(self):
        if not self.continue_button.isEnabled():
            return
        snapshot = self.draft.snapshot()
        if self._masks()[self.sand_class].any():
            self.map_fix_requested.emit(snapshot)
        else:
            self.requested.emit(snapshot)
