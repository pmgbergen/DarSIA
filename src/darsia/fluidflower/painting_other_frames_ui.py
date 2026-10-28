"""Read-only outlines of other runs' own frames on the reference photograph.

The host includes this controller in its busy/close/wait guards and calls
context_changed after a reference reload. Only metadata is read by its worker;
no photograph, drawing, geometry, approval, or settings are written.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import QCheckBox, QGraphicsPathItem, QGroupBox, QLabel, QPushButton, QVBoxLayout
from shiboken6 import isValid

from . import geometry_workspace_frame as workspace
from . import layer_painting as lp
from . import painting_frame_models as frame_models
from . import painting_run_preview as preview
from . import series_geometry as geometry
from .painting_preview_frame_domain import _check_frame_polygon, _project


@dataclass(frozen=True)
class OtherFramesResult:
    guides: tuple[dict, ...]
    missing: tuple[dict, ...]
    geometry_binding: dict
    config_sources: tuple[dict, ...]


def _metadata_pin(path):
    path = Path(path)
    try:
        before = path.stat()
    except FileNotFoundError:
        return dict(path=str(path), missing=True)
    if before.st_size > 16 * 1024 * 1024:
        raise ValueError("The frame configuration exceeds the metadata size limit.")
    raw = path.read_bytes()
    after = path.stat()
    stamp = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if stamp(before) != stamp(after) or len(raw) != before.st_size:
        raise ValueError("The frame configuration changed while it was read.")
    return dict(path=str(path), size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def _series_rows(root):
    """Use the existing common/run merge and retain exact metadata identities."""
    root = Path(root)
    series_pin = _metadata_pin(root / "series.toml")
    manifest = geometry.read_series_manifest(root)
    paths, runs = manifest.get("paths") or {}, manifest.get("runs") or {}
    ids = list(runs.get("ids") or [])
    keys = [str(run).strip().casefold() for run in ids]
    if not keys or any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("The series must name each run exactly once to show frame guides.")
    configs = runs.get("config_files") or {}
    folder = str(paths.get("run_configs") or "config/runs")
    source_paths = [geometry._storage_path(paths.get("common_config", "config/common.toml"), root)]
    source_paths.extend(geometry._storage_path(configs.get(str(run)) or f"{folder}/{run}.toml", root) for run in ids)
    pins = [series_pin] + [_metadata_pin(path) for path in source_paths if path is not None]
    rows = geometry.series_run_inputs(root, check_files=False)
    if [row.run_id.strip().casefold() for row in rows] != keys:
        raise ValueError("The series run list changed while loading frame guides.")
    if any(_metadata_pin(pin["path"]) != pin for pin in pins):
        raise ValueError("The series frame configuration changed while loading guides.")
    return rows, tuple(pins)


def _frame_points(frame, run, model):
    if frame is None or not frame.complete():
        raise ValueError("No complete own rig frame is saved or configured.")
    if str(frame.run_id).strip().casefold() != run:
        raise ValueError("The saved frame belongs to a different run.")
    path = str(model.reference_image.get("path") or "")
    if not frame.image or not path or not preview._same_path(frame.image, path):
        raise ValueError("The own frame is not bound to this model's photograph.")
    shape = model.shape_hw
    if (not isinstance(shape, (tuple, list)) or len(shape) != 2
            or any(type(value) is not int or value < 1 for value in shape)):
        raise ValueError("The model has no exact native photograph grid.")
    if frame.image_size is not None and tuple(frame.image_size)[::-1] != tuple(shape):
        raise ValueError("The own frame declares a different native photograph grid.")
    points = lp.frame_polygon(frame, (1., 1.))
    _check_frame_polygon(points)
    return points


def load_other_frames(*, project, selection, reference_run, photograph_path, shape_hw, frame):
    """Metadata-worker API. Each outline comes from that run's own authority."""
    if not isinstance(selection, frame_models.FrameModelSelection):
        raise ValueError("Choose the series frame models before showing other frames.")
    captured = frame_models.capture_frame_model_selection(project,
        override_path=selection.index_path if selection.source == "user override" else "")
    if captured != selection:
        raise ValueError("The frame-model selection belongs to another project or series.")
    frame_models.verify_frame_model_selection(selection)
    models, binding = preview.read_preview_geometry(selection.index_path)
    source_run = str(reference_run).strip().casefold()
    source = models.get(source_run)
    if source is None or source.run_id != source_run:
        raise ValueError("The frame models do not include this reference run.")
    if tuple(shape_hw) != tuple(source.shape_hw or ()):
        raise ValueError("The reference drawing and model have different native grids.")
    if not preview._same_path(photograph_path, source.reference_image.get("path", "")):
        raise ValueError("The frame model belongs to another reference photograph.")
    _frame_points(frame, source_run, source)
    link = geometry.native_series_link(project)
    series_root = str(link.get("series_root") or "")
    if not series_root:
        raise ValueError("The project has no native series root for the own-frame configurations.")
    rows, config_pins = _series_rows(series_root)
    if source_run not in {row.run_id.strip().casefold() for row in rows}:
        raise ValueError("The reference run is not in this series.")
    raw_saved = link.get("rig_frames") or {}
    if not isinstance(raw_saved, dict):
        raise ValueError("The saved own-frame table is invalid.")
    saved = {}
    for key, value in raw_saved.items():
        run = str(key).strip().casefold()
        if not run or run in saved:
            raise ValueError("The saved frame table names a run more than once.")
        saved[run] = value
    guides, missing = [], []
    for row in rows:
        run = row.run_id.strip().casefold()
        if run == source_run:
            continue
        try:
            target = models.get(run)
            if target is None or target.run_id != run:
                raise ValueError("No exact measured frame model is available.")
            if run in saved:
                # A broken saved authority is never replaced by configuration.
                if not isinstance(saved[run], dict):
                    raise ValueError("The saved own frame is invalid.")
                own = geometry.RigFrame.from_dict(saved[run])
            else:
                if row.issues:
                    raise ValueError("The run configuration is unavailable: " + "; ".join(row.issues))
                own = geometry.rig_frame_from_inputs(row)
            corners = _frame_points(own, run, target)
            mapped = _project(corners, source.metric_to_grid @ np.linalg.inv(target.metric_to_grid))
            _check_frame_polygon(mapped)
            height, width = shape_hw
            outside = bool(np.any((mapped[:, 0] < 0.) | (mapped[:, 0] > width - 1.)
                                  | (mapped[:, 1] < 0.) | (mapped[:, 1] > height - 1.)))
            guides.append(dict(run_id=run, corners_rc=tuple((float(y), float(x)) for x, y in mapped),
                source=str(own.source), extends_outside_photo=outside,
                frame=deepcopy(own.to_dict())))
        except (ValueError, TypeError, KeyError, IndexError, np.linalg.LinAlgError) as error:
            missing.append(dict(run_id=run, reason=str(error)))
    if any(_metadata_pin(pin["path"]) != pin for pin in config_pins):
        raise ValueError("The own-frame configuration changed before the guides were ready.")
    model_pins = [binding["geometry_index"], *(row["model"] for row in binding["models"].values())]
    if any(_metadata_pin(pin["path"]) != pin for pin in model_pins):
        raise ValueError("The measured frame models changed before the guides were ready.")
    frame_models.verify_frame_model_selection(selection)
    return OtherFramesResult(tuple(guides), tuple(missing), binding, config_pins)


class _Load(QObject):
    finished = Signal()

    def __init__(self, arguments):
        super().__init__()
        self.arguments, self.result, self.error = arguments, None, ""

    @Slot()
    def run(self):
        try:
            self.result = load_other_frames(**self.arguments)
        except Exception as error:
            self.error = str(error)
        self.finished.emit()


class PaintingOtherFramesController(QObject):
    changed = Signal()
    busy_changed = Signal(bool)

    def __init__(self, view):
        super().__init__(view)
        self.view, self.canvas = view, view.canvas
        self._viewport = self.canvas.viewport()
        self._disposed = self._reload_requested = self._waiting = False
        self._thread = self._worker = self._loading_key = self._loaded_key = None
        self.result, self._items = None, []
        self._error = "Show the other runs' own saved or configured frames on this photograph."
        self.widget = QGroupBox("Other run frames", view)
        layout = QVBoxLayout(self.widget)
        self.show_frames = QCheckBox("Show other runs' frames")
        self.show_frames.setToolTip("Display only. Each outline uses that run's own saved or configured frame; no measured top is inferred.")
        layout.addWidget(self.show_frames)
        self.reload_button = QPushButton("Reload frame guides")
        layout.addWidget(self.reload_button)
        self.note = QLabel(self._error)
        self.note.setWordWrap(True)
        self.note.setTextFormat(Qt.TextFormat.PlainText)
        self.note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.note)
        self.show_frames.toggled.connect(self._toggled)
        self.reload_button.clicked.connect(self.reload)
        self.canvas.installEventFilter(self)
        self._viewport.installEventFilter(self)
        self.canvas.destroyed.connect(self._invalidate)
        self._viewport.destroyed.connect(self._invalidate)
        view.destroyed.connect(self._invalidate)
        run_preview = getattr(view, "run_preview", None)
        if run_preview is not None and hasattr(run_preview, "display_changed"):
            run_preview.display_changed.connect(self.refresh)
        self.refresh()

    def _alive(self):
        return (not self._disposed and isValid(self) and isValid(self.view)
                and isValid(self.canvas) and isValid(self._viewport) and isValid(self.widget))

    def _invalidate(self, *_args):
        self._disposed = True
        self._items = []

    def _snapshot(self):
        if not self._alive():
            return None, "The reference canvas is unavailable."
        if QThread.currentThread() != QObject.thread(self):
            raise RuntimeError("Frame-guide context must be captured on the GUI thread.")
        view, canvas = self.view, self.canvas
        session, image, project = (getattr(view, key, None) for key in ("session", "image", "project"))
        if session is None or image is None or project is None:
            return None, "Choose a reference photograph before showing frame guides."
        run_preview = getattr(view, "run_preview", None)
        if run_preview is None or not run_preview.is_reference():
            return None, "Return to the editable reference photograph to show other frames."
        if getattr(view, "_other_photo", None) is not None or getattr(view, "_background_shown", False):
            return None, "Return to the reference photograph to show other frames."
        if getattr(canvas, "place_mode", False):
            return None, "Finish placing frame points before showing other frames."
        try:
            if workspace.frame_kind(getattr(view, "_workspace_frame_config", {})) != "raw":
                raise ValueError("Other frames are available only on the bound raw reference photograph.")
            size = tuple(image.image_size or ())
            if (len(size) != 2 or tuple(session.image_size or ()) != size
                    or tuple(canvas.raw_size or ()) != size):
                raise ValueError("The drawing and reference canvas have different photograph grids.")
            token = getattr(view, "_reference_token", None)
            if token is None or canvas.preview is not token:
                raise ValueError("The canvas does not show the bound reference photograph.")
            if getattr(session, "photo_changed", False) or not preview._same_path(session.image_path, image.path):
                raise ValueError("The displayed photograph differs from the drawing's reference.")
            profile = getattr(image, "camera_u", None)
            if profile is None or tuple(profile.camera_wh) != size:
                raise ValueError("The reference photograph has no matching camera-U profile.")
            camera = tuple(str(getattr(profile, name)) for name in ("path", "sha256", "profile_id", "camera_wh"))
            selection = run_preview.geometry_selection()
            frame = deepcopy(view.current_frame())
            frozen_project = deepcopy(project)
            payload = dict(project=project.manifest, frame=frame.to_dict() if frame else None)
            fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
            key = (id(project), id(session), id(image), id(token), str(project.root),
                   session.run_id, str(session.image_path), str(session.image_stamp), size,
                   selection, camera, fingerprint)
            arguments = dict(project=frozen_project, selection=selection, reference_run=session.run_id,
                             photograph_path=str(session.image_path), shape_hw=size[::-1], frame=frame)
        except (AttributeError, TypeError, ValueError) as error:
            return None, str(error)
        return (key, arguments), ""

    def busy(self):
        return self._thread is not None

    def can_close(self):
        return not self.busy()

    def wait_for_workers(self):
        self._waiting = True
        self._reload_requested = False
        try:
            if self._thread is not None:
                self._retire(self._thread)
        finally:
            self._waiting = False

    def context_changed(self, *_args):
        self.result, self._loaded_key = None, None
        self._error = "The reference changed; frame guides need to be reloaded."
        if self._alive() and self.show_frames.isChecked():
            return self.reload()
        self.refresh()
        return False

    def _toggled(self, checked):
        if checked and self._alive():
            snapshot, _ = self._snapshot()
            if snapshot is not None and (self.result is None or self._loaded_key != snapshot[0]):
                self.reload()
                return
        self.refresh()

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
            self._error = "Wait for the current painting operation, then reload frame guides."
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
                self._error = "The reference changed while loading; stale frame guides were discarded."
            elif worker.error:
                self._error = worker.error
            else:
                self.result, self._loaded_key = worker.result, key
                self._error = ""
        del worker
        if isValid(thread):
            thread.deleteLater()
        pending, self._reload_requested = self._reload_requested, False
        if pending and not self._waiting and self._alive():
            self.reload()
        self.refresh()
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
        valid = snapshot is not None and self.result is not None and self._loaded_key == snapshot[0]
        self.show_frames.setEnabled(snapshot is not None and not self.busy())
        self.reload_button.setEnabled(snapshot is not None and not self.busy())
        if self.busy():
            text = "Checking other runs' own frame configurations and measured models…"
        elif problem:
            text = problem
        elif self._error or not valid:
            text = self._error or ("Reload frame guides for the current reference."
                if self.show_frames.isChecked() else "Turn on the guides to load other runs' own frames.")
        else:
            outside = sum(guide["extends_outside_photo"] for guide in self.result.guides)
            text = (f"{len(self.result.guides)} other-run frame guides. Display only; saved/configured corners, "
                    "not independently measured top edges.")
            if outside:
                text += f" {outside} guides extend outside this photograph; drawing inside the photo cannot cover those areas."
            if self.result.missing:
                text += " Unavailable: " + "; ".join(f"{row['run_id']}: {row['reason']}" for row in self.result.missing)
            if self.show_frames.isChecked():
                for index, guide in enumerate(self.result.guides):
                    corners = guide["corners_rc"]
                    path = QPainterPath(self.canvas.raw_to_scene(*corners[0]))
                    for point in corners[1:]:
                        path.lineTo(self.canvas.raw_to_scene(*point))
                    path.closeSubpath()
                    pen = QPen(QColor.fromHsv((index * 47) % 360, 170, 230), 1.)
                    pen.setCosmetic(True)
                    pen.setStyle(Qt.PenStyle.DashLine)
                    item = QGraphicsPathItem(path)
                    item.setPen(pen)
                    item.setZValue(31.)
                    item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                    item.setToolTip(guide["run_id"] + " own frame (" + guide["source"] + "), display only")
                    self.canvas.scene().addItem(item)
                    self._items.append(item)
        self.note.setText(text)
        self.changed.emit()

    def eventFilter(self, watched, event):
        if (self._alive() and (watched is self.canvas or watched is self._viewport)
                and event.type() in (QEvent.Type.Resize, QEvent.Type.Wheel, QEvent.Type.Show)):
            QTimer.singleShot(0, self.refresh)
        return False
