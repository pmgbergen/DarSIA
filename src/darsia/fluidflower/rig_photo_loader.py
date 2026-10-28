"""Finite display-only RAW reads, with at most one decoded photograph in flight."""
from PySide6.QtCore import QCoreApplication, QObject, QThread, Qt, Signal, Slot
import numpy as np
from PIL import Image


def read_static_reference_packet(reference, profile):
    """Read one registered exposure on its own lens-corrected camera-U plane."""
    from pathlib import Path
    from .photograph_io import is_camera_raw, load_reduced_rgb8
    from .native.io import sha256_file

    path = Path(reference.baseline_image) if reference.baseline_image is not None else None
    if path is None:
        raise ValueError('This run has no registered reference photograph')
    if is_camera_raw(path):
        if profile is None:
            raise ValueError('This RAW photograph needs its bound lens profile before static areas can be matched')
        from .native.raw_full_display import read_full_raw_display
        packet = read_full_raw_display(path, profile_path=profile.path,
            profile_sha256=profile.sha256, profile_id=profile.profile_id)
    else:
        before = sha256_file(path)
        rgb = load_reduced_rgb8(path, 1)
        if sha256_file(path) != before:
            raise ValueError('The registered photograph changed during the read')
        height, width = rgb.shape[:2]
        packet = dict(image_rgb=rgb, image_encoding='rgb_uint8',
            camera_u_wh=[width, height], extent_u_xywh=[0, 0, width, height],
            camera_u_to_photo_h=np.eye(3).tolist(), valid_support=np.ones((height, width), bool),
            source_raw=dict(path=str(path), sha256=before, size_bytes=path.stat().st_size),
            profile=dict(profile_id='image-pixels', sha256=before, path=str(path)))
    return dict(packet, run_id=reference.run_id, path=str(path))


def read_rig_photo(path, profile):
    from .native.raw_full_display import read_full_raw_display
    from . import layer_painting as lp

    packet = read_full_raw_display(path, profile_path=profile.path,
        profile_sha256=profile.sha256, profile_id=profile.profile_id)
    image = Image.fromarray(packet['image_rgb'])
    image.thumbnail((lp.DISPLAY_LIMIT_PX, lp.DISPLAY_LIMIT_PX), Image.Resampling.BOX)
    plain = np.asarray(image).copy()
    return dict(packet, display_plain=plain,
        display_enhanced=lp.enhance_contrast(plain.astype(np.float32) / 255.))


class _Read(QObject):
    finished = Signal()

    def __init__(self, reader, args):
        super().__init__()
        self.reader, self.args = reader, args
        self.packet, self.error = None, ''

    @Slot()
    def run(self):
        try:
            self.packet = self.reader(*self.args)
        except Exception as error:
            self.error = f'{type(error).__name__}: {error}'
        finally:
            self.finished.emit()


# A view may close while a decoder finishes. Keep only its finite reader alive;
# Qt automatically disconnects the destroyed view's result slot.
_READERS = set()


class RigPhotoLoader(QObject):
    ready = Signal(object, object, str)

    def __init__(self, *, reader=read_rig_photo):
        super().__init__()
        self.reader = reader
        self.thread = self.worker = None
        self.active = self.pending = None
        self.disposed = False
        application = QCoreApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self.wait)

    def request(self, key, path, profile):
        if self.disposed:
            return
        if self.active is not None and self.active[0] == key:
            self.pending = None
            return
        self.pending = (key, path, profile)
        if self.thread is None:
            self._start()

    def invalidate(self):
        self.pending = None

    def busy(self):
        return self.thread is not None

    def wait(self):
        """Join the finite decoder at application shutdown, also after view disposal."""
        self.pending = None
        if self.thread is not None:
            self.thread.wait()
            self._retire()

    def dispose(self):
        self.disposed = True
        self.pending = None
        if self.thread is None:
            self.deleteLater()

    def _start(self):
        self.active, self.pending = self.pending, None
        _key, path, profile = self.active
        self.thread = QThread(self)
        self.worker = _Read(self.reader, (path, profile))
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.thread.quit, Qt.ConnectionType.DirectConnection)
        self.thread.finished.connect(self._retire)
        _READERS.add(self)
        self.thread.start()

    @Slot()
    def _retire(self):
        if self.thread is None:
            return
        key = self.active[0]
        packet, error = self.worker.packet, self.worker.error
        self.thread.deleteLater()
        self.thread = self.worker = self.active = None
        if not self.disposed:
            self.ready.emit(key, packet, error)
        if self.pending is not None and not self.disposed:
            self._start()
        else:
            _READERS.discard(self)
            if self.disposed:
                self.deleteLater()
