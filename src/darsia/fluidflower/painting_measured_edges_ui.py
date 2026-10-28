"""Passive saved measured lines and truthful camera-U/frame view fitting.

The host must include busy/can_close/wait_for_workers in its existing guards,
call context_changed after reference reload, and refresh after photo changes.
No image, painting, geometry, or display-companion file is written here.
"""
from __future__ import annotations

from copy import deepcopy
import math

from PySide6.QtCore import QEvent, QObject, QRectF, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import QCheckBox, QGraphicsPathItem, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout
from shiboken6 import isValid

from . import geometry_workspace_frame as workspace
from . import painting_measured_edges as edges
from .painting_measured_edges_display import load_measured_edge_display
from .series_geometry import native_series_link


class _Load(QObject):
    finished = Signal()

    def __init__(self, arguments):
        super().__init__()
        self.arguments, self.result, self.error = arguments, None, ""

    @Slot()
    def run(self):
        try:
            self.result = load_measured_edge_display(**self.arguments)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
        self.finished.emit()


def line_display_segment(coefficients, shape_hw):
    """A finite viewing extent along the exact saved line, not observed ends.

    Lines may lie just outside the photo; retaining that position lets the
    canvas margin show their displacement without moving a frame corner.
    """
    a, b, c = edges._line(coefficients)
    height, width = edges._shape(shape_hw, "Line display shape")
    if abs(a) >= abs(b):
        points = ((0., -c/a), (float(height-1), -(b*(height-1)+c)/a))
    else:
        points = ((-c/b, 0.), (-(a*(width-1)+c)/b, float(width-1)))
    if not all(math.isfinite(value) for point in points for value in point):
        raise ValueError("The saved line has no finite viewing extent.")
    return points


class PaintingMeasuredEdgesController(QObject):
    changed = Signal()
    busy_changed = Signal(bool)

    def __init__(self, view):
        super().__init__(view)
        self.view, self.canvas = view, view.canvas
        self._viewport = self.canvas.viewport()
        self._disposed = False
        self._thread = self._worker = None
        self._loading_key = self._loaded_key = None
        self._reload_requested = False
        self.result = None
        self._items = []
        self._error = "Load the saved measured-edge display metadata to show reference lines."
        self.widget = QGroupBox("Measured frame reference", view)
        layout = QVBoxLayout(self.widget)
        self.show_edges = QCheckBox("Show saved measured walls and bottom")
        self.show_edges.setToolTip("Three read-only saved lines; no measured top edge is invented.")
        layout.addWidget(self.show_edges)
        row = QHBoxLayout()
        self.reload_button = QPushButton("Load measured reference")
        self.fit_camera_button = QPushButton("Fit camera U")
        self.fit_frame_button = QPushButton("Fit rig frame")
        self.fit_camera_button.setToolTip("Fit the existing camera-U photograph. This is not the full pre-crop sensor.")
        self.fit_frame_button.setToolTip("Fit the current marked rig corners without changing the photograph or drawing coordinates.")
        for button in (self.reload_button, self.fit_camera_button, self.fit_frame_button):
            row.addWidget(button)
        layout.addLayout(row)
        self.note = QLabel(self._error)
        self.note.setWordWrap(True)
        self.note.setTextFormat(Qt.TextFormat.PlainText)
        self.note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.note)
        self.show_edges.toggled.connect(self.refresh)
        self.reload_button.clicked.connect(self.reload)
        self.fit_camera_button.clicked.connect(self.fit_camera_u)
        self.fit_frame_button.clicked.connect(self.fit_rig_frame)
        self.canvas.installEventFilter(self)
        self._viewport.installEventFilter(self)
        self.canvas.destroyed.connect(self._invalidate)
        self._viewport.destroyed.connect(self._invalidate)
        view.destroyed.connect(self._invalidate)
        preview = getattr(view, "run_preview", None)
        if preview is not None and hasattr(preview, "display_changed"):
            preview.display_changed.connect(self.refresh)
        self.refresh()

    def _alive(self):
        return (not self._disposed and isValid(self) and isValid(self.view)
                and isValid(self.canvas) and isValid(self._viewport) and isValid(self.widget))

    def _invalidate(self, *_args):
        # Canvas destruction can deliver a final event. No Qt access here.
        self._disposed = True
        self._items = []

    def _snapshot(self):
        if not self._alive():
            return None, "The reference canvas is unavailable."
        view = self.view
        session, image, project = (getattr(view, key, None) for key in ("session", "image", "project"))
        if session is None or image is None or project is None:
            return None, "Choose the reference photograph first."
        preview = getattr(view, "run_preview", None)
        if preview is not None and not preview.is_reference():
            return None, "Measured reference lines are hidden on another run's read-only preview."
        if (getattr(view, "_other_photo", None) is not None
                or getattr(view, "_background_shown", False)):
            return None, "Return to the reference photograph to show its measured lines."
        if getattr(self.canvas, "place_mode", False):
            return None, "Finish placing frame points before showing the measured reference."
        try:
            kind = workspace.frame_kind(getattr(view, "_workspace_frame_config", {}))
        except (AttributeError, TypeError, ValueError) as error:
            return None, str(error)
        if kind != "raw":
            return None, "The saved corrected photograph has no verified link to these camera-U lines."
        size = tuple(getattr(image, "image_size", ()) or ())
        if (len(size) != 2 or tuple(session.image_size or ()) != size
                or tuple(self.canvas.raw_size or ()) != size):
            return None, "The drawing and canvas have different photograph grids."
        reference_token = getattr(view, "_reference_token", None)
        if reference_token is None or self.canvas.preview is not reference_token:
            return None, "The canvas does not show the bound reference photograph."
        if (getattr(session, "photo_changed", False)
                or edges._path_key(session.image_path) != edges._path_key(image.path)):
            return None, "The displayed photograph differs from the drawing's reference."
        profile = getattr(image, "camera_u", None)
        if profile is None:
            return None, "This photograph has no verified camera-U profile."
        try:
            camera = {key: deepcopy(getattr(profile, key)) for key in
                      ("path", "sha256", "profile_id", "camera_wh")}
            camera["path"] = str(camera["path"])
            camera["camera_wh"] = tuple(camera["camera_wh"])
            if camera["camera_wh"] != size:
                raise ValueError("Camera-U grid differs from the displayed photograph")
            series_root = native_series_link(project).get("series_root")
            if not series_root:
                raise ValueError("The project has no native series root")
            arguments = dict(project_root=str(project.root), project_id=project.project_id,
                series_root=str(series_root), run=session.run_id,
                photograph_path=str(image.path), shape_hw=size[::-1], camera_u_binding=camera,
                photograph_stamp=session.image_stamp,
                geometry_index_path=str(getattr(preview, "_index", "")))
        except (AttributeError, TypeError, ValueError) as error:
            return None, str(error)
        key = (id(project), id(session), id(image), arguments["project_id"], str(project.root),
               str(series_root), session.run_id, str(image.path), session.image_stamp, size,
               camera["path"], camera["sha256"], camera["profile_id"], camera["camera_wh"],
               str(getattr(preview, "_index", "")))
        return (key, arguments), ""

    def busy(self):
        return self._thread is not None

    def can_close(self):
        return not self.busy()

    def wait_for_workers(self):
        if self._thread is not None:
            self._retire(self._thread)

    def context_changed(self, *_args):
        self.result, self._loaded_key = None, None
        self._error = "The reference changed; load its saved measured lines."
        return self.reload()

    def reload(self, *_args):
        if not self._alive():
            return False
        if self.busy():
            self._reload_requested = True
            return False
        snapshot, problem = self._snapshot()
        if snapshot is None:
            self._error = problem
            self.refresh()
            return False
        if getattr(self.view, "busy", lambda: False)():
            self._error = "Wait for the current painting operation before loading measured lines."
            self.refresh()
            return False
        key, arguments = snapshot
        worker, thread = _Load(arguments), QThread(self)
        self.result = self._loaded_key = None
        self._loading_key = key
        self._thread, self._worker = thread, worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._closed)
        self._error = ""
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
        worker, key = self._worker, self._loading_key
        self._thread = self._worker = self._loading_key = None
        if self._alive():
            snapshot, _ = self._snapshot()
            if snapshot is None or snapshot[0] != key or self._reload_requested:
                self._error = "The reference changed while loading; stale measured lines were discarded."
            elif worker.error:
                self._error = worker.error
            else:
                self.result, self._loaded_key = worker.result, key
                self._error = ""
        # The completed worker is released on the GUI thread, after joining.
        del worker
        if isValid(thread):
            thread.deleteLater()
        self.refresh()
        pending, self._reload_requested = self._reload_requested, False
        if pending and self._alive():
            self.reload()
        if isValid(self):
            self.busy_changed.emit(self.busy())

    def _clear_items(self):
        items, self._items = self._items, []
        for item in items:
            if isValid(item) and item.scene() is not None:
                item.scene().removeItem(item)

    def refresh(self, *_args):
        if not self._alive():
            return
        self._clear_items()
        snapshot, problem = self._snapshot()
        available = snapshot is not None
        valid = available and self.result is not None and self._loaded_key == snapshot[0]
        self.reload_button.setEnabled(available and not self.busy())
        self.show_edges.setEnabled(valid and not self.busy())
        self.fit_camera_button.setEnabled(available and not self.busy())
        self.fit_frame_button.setEnabled(available and not self.busy() and self._frame_points() is not None)
        if self.busy():
            self.note.setText("Checking saved measured lines and photograph identity…")
        elif problem:
            self.note.setText(problem)
        elif self._error or not valid:
            self.note.setText(self._error or "Load the saved measured lines for this reference.")
        else:
            confirmation = self.result.edges.provenance["edge_records"]["left"]["physical_B_confirmation"]
            self.note.setText("Saved measured left wall, bottom and right wall. No top edge is inferred. "
                              "Read only; the marked corners and adoption are unchanged. Confirmation: " + confirmation)
            if self.show_edges.isChecked():
                pen = QPen(QColor("#00ffff"), 2.)
                pen.setCosmetic(True)
                pen.setStyle(Qt.PenStyle.DashLine)
                for name, coefficients in self.result.edges.lines:
                    first, second = line_display_segment(coefficients, self.result.edges.shape_hw)
                    path = QPainterPath(self.canvas.raw_to_scene(*first))
                    path.lineTo(self.canvas.raw_to_scene(*second))
                    item = QGraphicsPathItem(path)
                    item.setPen(pen)
                    item.setZValue(32.)
                    item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                    item.setToolTip("Saved measured " + name + " line (read only)")
                    self.canvas.scene().addItem(item)
                    self._items.append(item)
        self.changed.emit()

    def _frame_points(self):
        frame = getattr(self.view, "current_frame", lambda: None)()
        if frame is None or not frame.complete():
            return None
        image = getattr(self.view, "image", None)
        size = tuple(getattr(image, "image_size", ()) or ())
        session = getattr(self.view, "session", None)
        if (getattr(frame, "run_id", "")
                and (session is None or str(frame.run_id).casefold() != str(session.run_id).casefold())):
            return None
        if getattr(frame, "image_size", None) is not None and tuple(frame.image_size) != size:
            return None
        if getattr(frame, "image", "") and edges._path_key(frame.image) != edges._path_key(image.path):
            return None
        try:
            points = [tuple(frame.corners[name]) for name in
                      ("top_left", "top_right", "bottom_right", "bottom_left")]
            if any(len(point) != 2 or not all(math.isfinite(float(v)) for v in point) for point in points):
                return None
            return points
        except (KeyError, TypeError, ValueError):
            return None

    def fit_camera_u(self):
        if self._snapshot()[0] is None or self.busy():
            return False
        self.canvas.fit_image()
        self.refresh()
        return True

    def fit_rig_frame(self):
        if self._snapshot()[0] is None or self.busy():
            return False
        points = self._frame_points()
        if points is None:
            return False
        scene = [self.canvas.raw_to_scene(*point) for point in points]
        xs, ys = [point.x() for point in scene], [point.y() for point in scene]
        bounds = QRectF(min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))
        if bounds.width() <= 0 or bounds.height() <= 0:
            return False
        self.canvas.resetTransform()
        self.canvas.fitInView(bounds, Qt.AspectRatioMode.KeepAspectRatio)
        self.canvas._user_zoomed = True
        self.canvas._update_pan_margins()
        self.canvas.refresh_detail()
        self.canvas._update_cursor()
        self.refresh()
        return True

    def eventFilter(self, watched, event):
        if (self._alive() and (watched is self.canvas or watched is self._viewport)
                and event.type() in (QEvent.Type.Resize, QEvent.Type.Wheel, QEvent.Type.Show)):
            QTimer.singleShot(0, self.refresh)
        return False
