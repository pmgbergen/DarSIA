"""Explicit per-target painting over a verified gap, with the original N54 canvas.

No reference drawing is loaded, restored or changed. The pure backend owns
pixel rasterization and immutable version/approval writing. Only actual human
draw gestures modify its draft, and only Save and approve writes a version.
"""
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
from PySide6.QtCore import QThread, Qt, Signal, Slot
from PySide6.QtGui import QColor, QKeySequence, QPen, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QGraphicsPathItem, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout,
)

from .painting_gap_review_ui import _Read, _gap_contours, _path, _project, _rgb
from .pages.geometry_facies.page import GeometryImageView


class TargetCorrectionDialog(QDialog):
    saved = Signal(object)

    def __init__(self, review, run_id, component_id, *, actor, destination,
                 point_native_xy=None, parent=None, reader=None, draft_factory=None,
                 current_display_reader=None, comparison_reader=None, current_display_packet=None):
        super().__init__(parent)
        if reader is None:
            from .painted_analysis_exclusions import read_target_paint
            reader = read_target_paint
        if draft_factory is None:
            from .painted_target_corrections import TargetCorrectionDraft
            draft_factory = TargetCorrectionDraft
        if comparison_reader is None:
            from .painted_camera_jpg_display import read_comparison_preview
            comparison_reader = read_comparison_preview
        self.comparison_reader = comparison_reader
        self.current_display_reader = current_display_reader or (
            lambda review, run, component: comparison_reader(review, run, component, "current"))
        self._initial_display = deepcopy(current_display_packet)
        self._display_valid = False
        self._photo_role = "current"
        self.reader, self.draft_factory = reader, draft_factory
        self.review = deepcopy(review)
        self.run_id, self.component_id = run_id, component_id
        self.actor, self.destination = str(actor).strip(), destination
        self.point_native_xy = deepcopy(point_native_xy)
        self.packet = self.draft = self.binding = None
        self.thread = self.worker = None
        self._closing = False
        self._outlines = []
        self.setWindowTitle("Painting tab – target correction for " + run_id.upper())
        self.resize(1000, 780)
        layout = QVBoxLayout(self)
        note = QLabel("You paint only inside the selected gap area for this run. "
            "The source map, reference photograph and original drawing stay unchanged. "
            "A new target version is saved only when you click Save and approve.")
        note.setWordWrap(True)
        layout.addWidget(note)
        controls = QHBoxLayout()
        self.layer = QComboBox()
        self.layer.addItem("Choose the layer yourself…", None)
        self.layer.currentIndexChanged.connect(self._buttons)
        controls.addWidget(self.layer)
        self.mode = QComboBox()
        for label, value in (("Pan", "pan"), ("Polygon", "polygon"), ("Brush", "brush")):
            self.mode.addItem(label, value)
        self.mode.currentIndexChanged.connect(self._mode)
        controls.addWidget(self.mode)
        self.radius = QDoubleSpinBox()
        self.radius.setRange(.5, 200.)
        self.radius.setValue(1.)
        self.radius.setSuffix(" native pixels")
        controls.addWidget(self.radius)
        self.finish = QPushButton("Finish polygon")
        self.finish.clicked.connect(lambda: self.canvas.finish_edit())
        controls.addWidget(self.finish)
        self.undo = QPushButton("Undo")
        self.redo = QPushButton("Redo")
        self.undo.clicked.connect(lambda: self._history(False))
        self.redo.clicked.connect(lambda: self._history(True))
        controls.addWidget(self.undo)
        controls.addWidget(self.redo)
        layout.addLayout(controls)
        self.canvas = GeometryImageView()
        self.canvas.set_options(overlay="none", show_injectors=False)
        self.canvas.label_edit_drawn.connect(self._drawn)
        layout.addWidget(self.canvas, 1)
        self.photo_toggle = QPushButton("Show clear-water image (Alt+I)")
        self.photo_toggle.clicked.connect(self._switch_photo)
        self.photo_shortcut = QShortcut(QKeySequence("Alt+I"), self)
        self.photo_shortcut.activated.connect(self._switch_photo)
        layout.addWidget(self.photo_toggle)
        toggles = QHBoxLayout()
        self.outline = QCheckBox("Show gap outline")
        self.boundaries = QCheckBox("Show layer boundaries")
        self.brighten = QCheckBox("Brighten photograph (display only)")
        self.outline.setChecked(True)
        self.boundaries.setChecked(True)
        for control in (self.outline, self.boundaries, self.brighten):
            control.toggled.connect(self._render)
            toggles.addWidget(control)
        layout.addLayout(toggles)
        self.status = QLabel("Reading the bound run photograph and map…")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.reason = QLineEdit()
        self.reason.setPlaceholderText("Your short reason for the correction")
        self.reason.setMaxLength(512)
        self.reason.textChanged.connect(self._buttons)
        layout.addWidget(self.reason)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.save = buttons.addButton("Save new target version and approve", QDialogButtonBox.ButtonRole.ActionRole)
        self.save.setAutoDefault(False)
        self.save.clicked.connect(self._save)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        self._buttons()
        self._start("load", self._load)

    @staticmethod
    def _grid(packet):
        science = packet["target_correction_basis"]["science"]
        return dict(format="darsia-painted-review-display-grid-v1",
            native_photo=deepcopy(science["native_photo"]),
            native_photo_pixels=deepcopy(science["native_photo_pixels"]),
            model=deepcopy(science["target_model"]), shape_hw=list(packet["target_labels"].shape),
            native_to_photo_h=np.asarray(packet["native_to_photo_h"]).tolist())

    @classmethod
    def _photo_rgb(cls, packet, display, role):
        # The scientific packet stays unchanged. Only a same-grid, explicitly
        # selected camera-JPG display result may replace the visible RGB.
        from .painted_review_comparison import _same
        rgb = _rgb(display)
        if (display.get("image_role") != role or display.get("camera_jpg_display_only") is not True
                or not _same(display.get("display_grid"), cls._grid(packet))
                or not _same(np.asarray(display.get("native_to_photo_h")).tolist(),
                             np.asarray(packet["native_to_photo_h"]).tolist())
                or rgb.shape != np.asarray(packet["image_rgb"]).shape):
            raise ValueError("The selected camera JPG has another role, native grid or photograph transform")
        support = display.get("valid_support")
        if support is not None:
            support = np.asarray(support)
            if support.dtype != np.bool_ or support.shape != rgb.shape[:2]:
                raise ValueError("The selected camera JPG lacks its exact boolean display support")
            rgb[~support] = 64
        return rgb

    def _load(self):
        # Always run the original target reader, even when Root supplies a
        # previously verified visible JPG. It owns all scientific admission.
        packet = self.reader(self.review, self.run_id, self.component_id)
        if self._initial_display is None:
            display = self.current_display_reader(self.review, self.run_id, self.component_id)
        else:
            display = deepcopy(self._initial_display)
            if display.get("run_id") != self.run_id or display.get("component_id") != self.component_id:
                raise ValueError("The supplied experiment photo belongs to another run or deviation")
            from . import painted_camera_jpg_display as camera_jpg
            from .painted_review_comparison import _checked, _descriptor, _same
            from .map_adoption import native_transport
            from types import SimpleNamespace
            model = _checked(packet["target_correction_basis"]["science"]["target_model"], metadata=True)
            raw = _descriptor(model["reference_image"])
            native, shape = native_transport(SimpleNamespace(model=model, run_id=self.run_id))
            if (not _same(display.get("source_raw"), raw) or not _same(camera_jpg._companion(raw), display.get("camera_jpeg"))
                    or shape != packet["target_labels"].shape
                    or not _same(np.asarray(display.get("display_transform")).tolist(), native.tolist())):
                raise ValueError("The supplied experiment JPG differs from this exact model, exposure or placement")
            _checked(display["camera_jpeg"])
        self._photo_rgb(packet, display, "current")
        return dict(paint=packet, display=display)

    def _switch_photo(self):
        if self.draft is None or self.busy() or self._closing or self.binding is not None:
            return
        role = "clearwater" if self._photo_role == "current" else "current"
        review, run, component = deepcopy(self.review), self.run_id, self.component_id
        self.status.setText("Opening the other verified photo; your draft, zoom and position are kept.")
        read = (lambda: self.current_display_reader(review, run, component)) if role == "current" else (
            lambda: self.comparison_reader(review, run, component, role))
        self._start(("photo", role), read)

    def busy(self):
        return self.thread is not None

    def _start(self, operation, function):
        self.thread, self.worker = QThread(self), _Read(function)
        self.worker.operation = operation
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit, Qt.ConnectionType.DirectConnection)
        self.thread.finished.connect(self._retired)
        self.thread.start()
        self._buttons()

    @Slot()
    def _retired(self, expected=None):
        if expected is None:
            expected = self.sender()
        if self.thread is None or self.thread is not expected:
            return
        thread, worker = self.thread, self.worker
        thread.wait()
        self.thread = self.worker = None
        try:
            if worker.operation == "save" and not worker.error:
                # The human already clicked Save and approve. Retain that
                # successful binding before a deferred Close is completed.
                self.binding = deepcopy(worker.result)
                if not self._closing:
                    self.status.setText("The new target version is saved and approved. Check again in the findings list; nothing has been adopted.")
                self.saved.emit(dict(run_id=self.run_id, component_id=self.component_id, binding=deepcopy(self.binding)))
            elif not self._closing:
                if worker.error:
                    raise ValueError(worker.error)
                if worker.operation == "load":
                    self._loaded(worker.result["paint"], worker.result["display"])
                else:
                    role = worker.operation[1]
                    rgb = self._photo_rgb(self.packet, worker.result, role)
                    self._plain, self._photo_role, self._display_valid = rgb, role, True
                    self._render()
                    self.status.setText("Clear-water image; your map correction is unchanged." if role == "clearwater"
                                        else "Experiment image; your map correction is unchanged.")
        except (KeyError, TypeError, ValueError, OSError) as error:
            if worker.operation == "load" or isinstance(worker.operation, tuple):
                self._display_valid = False
            if not self._closing:
                self.status.setText("The correction or photo could not be opened or saved: " + str(error))
        del worker
        thread.deleteLater()
        self._buttons()
        if self._closing:
            self.close()

    def _loaded(self, packet, display):
        if packet["run_id"] != self.run_id or packet["component_id"] != self.component_id:
            raise ValueError("The paint packet belongs to another run or connected area")
        self.draft = self.draft_factory(labels=packet["target_labels"], editable_mask=packet["editable_mask"],
            allowed_ids=[row["layer_id"] for row in packet["allowed_layers"]], basis=packet["target_correction_basis"])
        self.packet, self._plain = packet, self._photo_rgb(packet, display, "current")
        self._display_valid = True
        for row in packet["allowed_layers"]:
            self.layer.addItem(f"{row['layer_id']}: {row['name']} ({row['facies']}; {row['role']})", row["layer_id"])
        self._render(initial=True)
        if self.point_native_xy is not None:
            point = _project([self.point_native_xy], packet["native_to_photo_h"])[0]
            self.canvas.centerOn(float(point[0]), float(point[1]))
        self.status.setText("Choose the layer and polygon or brush. Only your own strokes change the draft. "
                            "Everything outside the selected gap area stays unchanged.")

    def _render(self, *_args, initial=False):
        if self.packet is None or self.draft is None:
            return
        # The display copy is made once per photograph and brightening, and the
        # same array is passed on, so the canvas keeps its 8-bit copy too.
        brighten = self.brighten.isChecked()
        kept = self.__dict__.get("_baseline_kept")
        if kept is not None and kept[0] is self._plain and kept[1] == brighten:
            baseline = kept[2]
        else:
            rgb = (np.minimum(self._plain.astype(np.uint16)*2, 255).astype(np.uint8)
                   if brighten else self._plain)
            baseline = rgb.astype(np.float32)/255.
            self._baseline_kept = (self._plain, brighten, baseline)
        preview = dict(baseline=baseline)
        if initial:
            self.canvas.set_model(preview, {})
        else:
            self.canvas.update_preview(preview, {})
        for item in self._outlines:
            self.canvas.scene().removeItem(item)
        self._outlines = []
        masks = []
        if self.outline.isChecked():
            masks.append((self.packet["editable_mask"], "magenta"))
        if self.boundaries.isChecked():
            masks.append((GeometryImageView._boundaries(self.draft.labels), "white"))
        for mask, colour in masks:
            item = QGraphicsPathItem(_path(_gap_contours(mask), self.packet["native_to_photo_h"]))
            pen = QPen(QColor(colour), 2)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setBrush(Qt.BrushStyle.NoBrush)
            item.setZValue(3)
            self.canvas.scene().addItem(item)
            self._outlines.append(item)

    def _mode(self, *_args):
        mode = self.mode.currentData()
        if mode != "pan" and self.layer.currentData() is None:
            self.mode.blockSignals(True)
            self.mode.setCurrentIndex(0)
            self.mode.blockSignals(False)
            mode = "pan"
            self.status.setText("Choose the layer yourself before painting.")
        self.canvas.set_edit_mode(mode)

    def _drawn(self, kind, points):
        if (self.draft is None or not self._display_valid or self.busy() or self._closing
                or self.binding is not None or self.layer.currentData() is None):
            return
        try:
            height, width = self._plain.shape[:2]
            photo_points = np.asarray(points, np.float64) * [width-1, height-1]
            native = _project(photo_points, np.linalg.inv(self.packet["native_to_photo_h"]))
            self.draft.paint(kind=kind, points_xy=native.tolist(), label_id=self.layer.currentData(), radius_px=self.radius.value())
            self._render()
            self.status.setText("Your stroke is in the draft. Undo or redo before choosing to save and approve.")
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            self.status.setText("The stroke could not be used: " + str(error))
        self._buttons()

    def _history(self, redo):
        if self.draft is not None and self._display_valid and not self.busy() and not self._closing and self.binding is None:
            (self.draft.redo if redo else self.draft.undo)()
            self._render()
            self._buttons()

    def _buttons(self, *_args):
        idle = self.draft is not None and not self.busy() and not self._closing and self.binding is None
        editable = idle and self._display_valid
        for control in (self.layer, self.mode, self.radius, self.finish):
            control.setEnabled(editable)
        self.canvas.setEnabled(editable)
        self.undo.setEnabled(editable and bool(self.draft.can_undo))
        self.redo.setEnabled(editable and bool(self.draft.can_redo))
        self.save.setEnabled(editable and bool(self.draft.dirty) and bool(self.actor) and bool(self.reason.text().strip()))
        self.photo_toggle.setEnabled(idle)
        self.photo_shortcut.setEnabled(idle)
        self.photo_toggle.setText("Show experiment image (Alt+I)" if self._photo_role == "clearwater"
                                  else "Show clear-water image (Alt+I)")

    def _save(self):
        if not self.save.isEnabled():
            return
        actor, when, reason = self.actor, datetime.now(timezone.utc).isoformat(), self.reason.text().strip()
        frozen, destination = self.draft.snapshot(), deepcopy(self.destination)
        self._start("save", lambda: frozen.save_and_approve(destination,
                    actor=actor, approved_utc=when, reason=reason))

    def wait_for_workers(self):
        """Shutdown barrier only; interactive Close/Escape remain nonblocking."""
        self._closing = True
        self._buttons()
        thread = self.thread
        if thread is not None:
            # Consume a completed explicit Save synchronously: aboutToQuit
            # may be the last event, so its queued finished slot is insufficient.
            thread.wait()
            self._retired(thread)
        else:
            self.close()

    def reject(self):
        self._closing = True
        self._buttons()
        if self.busy():
            self.status.setText("This window closes when the current read or approved save finishes.")
            return
        super().reject()

    def closeEvent(self, event):
        self._closing = True
        self._buttons()
        if self.busy():
            self.status.setText("This window closes when the current read or approved save finishes.")
            event.ignore()
            return
        super().closeEvent(event)
