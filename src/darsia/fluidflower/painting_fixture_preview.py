"""Optional reference-photo display of the drawn/continued fixture underlay.

This preview writes nothing. Adoption uses a separately recomputed underlay
bound to an immutable approved version, never this draft display result.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QCheckBox, QGraphicsPixmapItem, QLabel, QVBoxLayout, QWidget
from shiboken6 import isValid

from . import layer_painting as lp
from . import layer_painting_fixtures as fixtures
from . import layer_painting_manual as manual


def preview_underlay(session, frame, image_size, display_shape):
    """Use the original manual renderer and continuation on the native grid."""
    polygon = lp.frame_polygon(frame, (1.0, 1.0))
    if (polygon is None or not np.isfinite(polygon).all()
            or not cv2.isContourConvex(np.asarray(polygon, np.float32))
            or abs(cv2.contourArea(np.asarray(polygon, np.float32))) <= 0):
        raise ValueError("The current rig frame must be finite, convex and noncollapsed.")
    painted = manual.manual_map(session, image_size, frame=frame)
    underlay = fixtures.fixture_underlay(session, painted.classes, frame=frame)
    if underlay is None:
        raise ValueError("No ColorChecker footprint is drawn inside the rig frame.")
    rows, cols = np.indices(painted.classes.shape, dtype=np.float64)
    forward = np.asarray(frame._forward(), dtype=np.float64)
    denominator = forward[2, 0] * cols + forward[2, 1] * rows + forward[2, 2]
    if not np.isfinite(forward).all() or np.any(denominator == 0):
        raise ValueError("The current rig frame has no finite native-pixel coordinates.")
    # Exact vector form of the existing RigFrame.pixel_to_rig height mapping.
    heights = (1.0 - (forward[1, 0] * cols + forward[1, 1] * rows + forward[1, 2])
               / denominator) * float(frame.height_m)
    roles = {row.layer_id: session.facies_of(row.layer_id).role for row in session.layers}
    result = fixtures.continue_fixture_underlay(underlay.analysis_layers, roles,
        underlay.fixture_mask, underlay.drawn_layers, heights)
    height, width = (int(value) for value in display_shape)
    if min(height, width) <= 0:
        raise ValueError("A nonempty display grid is required.")
    def projected(value):
        return cv2.resize(np.asarray(value, np.uint8), (width, height),
                          interpolation=cv2.INTER_NEAREST)
    return dict(classes=projected(result["layers"]),
        drawn=projected(result["drawn_behind"]).astype(bool),
        continued=projected(result["fixture_filled"]).astype(bool),
        unfilled=projected(result["unfilled"]).astype(bool),
        counts=dict(result["counts"]), native_shape=tuple(painted.classes.shape))


class _Compute(QObject):
    finished = Signal()

    def __init__(self, arguments):
        """Capture the private preview context and lifecycle bindings."""
        super().__init__()
        self.arguments, self.result, self.error = arguments, None, ""

    @Slot()
    def run(self):
        """Compute the immutable drawn-only preview off the GUI thread."""
        try:
            self.result = preview_underlay(*self.arguments)
        except Exception as error:
            self.error = str(error)
        self.finished.emit()


class PaintingFixturePreview(QObject):
    busy_changed = Signal(bool)

    def __init__(self, view):
        """Capture the private preview context and lifecycle bindings."""
        super().__init__(view)
        self.view, self.canvas = view, view.canvas
        self._disposed = False
        self._thread = self._worker = None
        self._loading_key = self._loaded_key = None
        self.result, self.error = None, ""
        self.widget = QWidget(view)
        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.show_underlay = QCheckBox("Show sand behind the ColorChecker", self.widget)
        self.show_underlay.setChecked(False)
        self.show_underlay.setToolTip("Drawn non-ColorChecker polygons are retained. Only undrawn cells are continued from porous layers. Display only; no version or adoption is changed.")
        self.note = QLabel(self.widget)
        self.note.setWordWrap(True)
        self.note.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.show_underlay)
        layout.addWidget(self.note)
        self.item = QGraphicsPixmapItem()
        self.item.setZValue(19.0)
        self.item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.canvas.scene().addItem(self.item)
        self.item.hide()
        self.show_underlay.toggled.connect(self.refresh)
        view.changed.connect(self.refresh)
        view.display_styles_changed.connect(self.redraw)
        view.opacity.valueChanged.connect(self.redraw)
        view.run_preview.display_changed.connect(self.refresh)
        self.canvas.destroyed.connect(self._invalidate)
        view.destroyed.connect(self._invalidate)
        self.refresh()

    def _alive(self):
        """Check the host and Qt lifetime before accessing controls."""
        return (not self._disposed and isValid(self) and isValid(self.view)
                and isValid(self.canvas) and isValid(self.widget))

    def _invalidate(self, *_args):
        # Destruction callbacks must not dereference Qt wrappers.
        """Invalidate the controller without touching destroyed Qt objects."""
        self._disposed = True
        self.item = None

    def _snapshot(self):
        """Capture the exact current reference drawing and frame."""
        if not self._alive():
            return None, "The reference canvas is unavailable."
        view = self.view
        session, image = view.session, view.image
        if session is None or image is None:
            return None, "Choose the reference photograph to preview the fixture underlay."
        if not any(entry.role == fixtures.COLORCHECKER for entry in session.facies.values()):
            return None, "Choose the ColorChecker role explicitly in the facies table; existing Excluded facies are unchanged."
        if (not view.run_preview.is_reference() or view._other_photo is not None
                or getattr(view, "_background_shown", False)
                or self.canvas.preview is not view._reference_token):
            return None, "Return to the bound reference photograph to preview its fixture underlay."
        from .geometry_workspace_frame import frame_kind
        if frame_kind(getattr(view, "_workspace_frame_config", {})) != "raw":
            return None, "The fixture preview needs the bound reference camera grid; corrected-frame adoption is checked separately."
        if self.canvas.place_mode:
            return None, "Finish placing the rig frame before previewing the fixture underlay."
        size = tuple(image.image_size)
        from .storage_paths import same_storage_path
        if (tuple(session.image_size or ()) != size or tuple(self.canvas.raw_size or ()) != size
                or not same_storage_path(session.image_path, image.path)):
            return None, "The drawing and canvas do not share the exact reference photograph and native grid."
        frame = view.current_frame()
        if (frame is None or not frame.complete() or not frame.has_size()
                or not np.isfinite([frame.width_m, frame.height_m]).all()
                or min(frame.width_m, frame.height_m) <= 0):
            return None, "Set the complete rig frame and inner size before previewing continuation."
        if ((frame.run_id and frame.run_id != session.run_id)
                or (frame.image_size is not None and tuple(frame.image_size) != size)
                or (frame.image and not same_storage_path(frame.image, image.path))):
            return None, "The rig frame belongs to another reference photograph or grid."
        display_shape = tuple(int(value) for value in image.display.shape[:2])
        payload = dict(session=session.to_dict(), frame=frame.to_dict(),
                       image_path=str(Path(image.path).resolve()), size=size,
                       display_shape=display_shape)
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False,
            separators=(",", ":")).encode()).hexdigest()
        return (key, session, frame, size, display_shape), ""

    def busy(self):
        """Report whether a preview worker still owns resources."""
        return self._thread is not None

    def can_close(self):
        """Refuse closure until the current preview worker retires."""
        return not self.busy()

    def wait_for_workers(self):
        """Wait for natural worker retirement before host teardown."""
        if self._thread is not None:
            self._retire(self._thread)

    def refresh(self, *_args):
        """Request the latest exact reference drawing when enabled."""
        if not self._alive():
            return
        if getattr(self.view, "_waiting_for_workers", False):
            self.item.hide()
            return
        try:
            snapshot, problem = self._snapshot()
        except (OSError, ValueError, TypeError, KeyError) as error:
            snapshot, problem = None, str(error)
        self.show_underlay.setEnabled(snapshot is not None)
        if snapshot is None or not self.show_underlay.isChecked():
            self.item.hide()
            self.note.setText(problem if snapshot is None else "Sand behind the ColorChecker is hidden. The photograph and saved map remain visible.")
            return
        key, session, frame, size, display_shape = snapshot
        if key == self._loaded_key and self.result is not None:
            self.redraw()
            return
        self.item.hide()
        if self.busy():
            self.note.setText("Computing the drawn and continued cells; stale preview is hidden.")
            return
        if key == self._loaded_key and self.error:
            self.note.setText(self.error)
            return
        if self.view.busy():
            self.note.setText("Wait for the current painting operation before drawing the fixture underlay.")
            return
        worker = _Compute((deepcopy(session), deepcopy(frame), size, display_shape))
        thread = QThread(self)
        self._thread, self._worker, self._loading_key = thread, worker, key
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._closed)
        self.note.setText("Computing the full-resolution drawn and continued fixture cells…")
        thread.start()
        self.busy_changed.emit(True)

    @Slot()
    def _closed(self):
        """Store worker completion without publishing stale state."""
        thread = self.sender()
        if thread is self._thread:
            self._retire(thread)

    def _retire(self, thread):
        """Wait and release the finished Qt worker on the GUI thread."""
        if thread is not self._thread:
            return
        thread.wait()
        worker, key = self._worker, self._loading_key
        self._thread = self._worker = self._loading_key = None
        if self._alive():
            try:
                snapshot, _problem = self._snapshot()
            except (OSError, ValueError, TypeError, KeyError):
                snapshot = None
            if snapshot is not None and snapshot[0] == key:
                self._loaded_key, self.result, self.error = key, worker.result, worker.error
        # Q185: join and release the worker from the GUI; no worker.deleteLater.
        del worker
        if isValid(thread):
            thread.deleteLater()
        if self._alive() and not getattr(self.view, "_waiting_for_workers", False):
            self.refresh()
        if isValid(self):
            self.busy_changed.emit(self.busy())

    def redraw(self, *_args):
        """Render only a current, verified preview result."""
        if not self._alive():
            return
        try:
            snapshot, _problem = self._snapshot()
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.item.hide()
            self.note.setText(str(error))
            return
        if (not self.show_underlay.isChecked() or snapshot is None
                or snapshot[0] != self._loaded_key or self.result is None):
            self.item.hide()
            return
        result = self.result
        rgba, shown = self.view._layer_overlay(result["classes"])
        footprint = result["drawn"] | result["continued"] | result["unfilled"]
        rgba[~footprint] = 0
        rows, cols = np.indices(footprint.shape)
        hatch = ((rows + cols) // 3) % 2 == 0
        rgba[result["continued"] & hatch & shown] = (255, 255, 255, 210)
        rgba[result["unfilled"]] = (240, 0, 200, 220)
        rgba = np.ascontiguousarray(rgba, np.uint8)
        height, width = rgba.shape[:2]
        image = QImage(rgba.data, width, height, int(rgba.strides[0]),
                       QImage.Format.Format_RGBA8888).copy()
        self.item.setPixmap(QPixmap.fromImage(image))
        self.item.show()
        count = result["counts"]
        self.note.setText(f"Drawn: {count['drawn']:,} (solid); continued: {count['continued']:,} (hatched); "
            f"unfilled: {count['unfilled']:,} (magenta). Counts use the native photograph grid. "
            "Draft display only; adoption replays the exact saved approved version. "
            "Measured-polygon agreement is unavailable until a verified polygon source is bound.")
