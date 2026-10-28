"""Explicit attachment of an existing receipt; no collection or payload reads."""
from __future__ import annotations

import hashlib
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QObject, QThread, Qt, Signal, Slot

from .calibration_review_native import NativeSavedReview
from .pipeline import products_publication as publication


def _selected_receipt(path):
    """Bind only explicit selected metadata with a hard read limit."""
    path = publication._file(path)
    before = path.stat()
    if before.st_size > publication.MAX_INDEX_BYTES:
        raise ValueError("Selected collection metadata exceeds the original index size limit")
    with path.open("rb") as source:
        raw = source.read(publication.MAX_INDEX_BYTES + 1)
    if len(raw) > publication.MAX_INDEX_BYTES:
        raise ValueError("Selected collection metadata exceeds the original index size limit")
    after = path.stat()
    if publication._stamp(before) != publication._stamp(after) or len(raw) != after.st_size:
        raise ValueError("Selected collection metadata changed while being read")
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def attach_existing_collection(reader, selected_receipt_path):
    """Construct a replacement reader from an explicitly selected receipt.

    Only this user-selected metadata file gets a new descriptor. Its contents
    must retain the original index/run/attempt and every copied producer's
    path/SHA/size through the unchanged NativeSavedReview resolver. Existing
    bindings and scientific identities are retained; the live reader is never
    mutated. This function never copies products or opens an array or image.
    """
    if not isinstance(reader, NativeSavedReview):
        raise ValueError("Choose an already admitted native saved result")
    if not selected_receipt_path:
        raise ValueError("Choose an existing absolute collection/publication receipt path")
    path = Path(selected_receipt_path)
    if not path.is_absolute():
        raise ValueError("Choose an existing absolute collection/publication receipt path")
    selected = _selected_receipt(path)
    # Bound the verification independently, including a file changed during
    # descriptor acquisition. No descriptor of an old index/payload is replaced.
    publication._verified_bytes(path, publication._bound(selected), limit=publication.MAX_INDEX_BYTES)
    bindings, seen = [], set()
    for item in (*reader.collections, selected):
        original = publication._bound(item)
        normalized = dict(path=original["path"], sha256=original["sha256"], size_bytes=original["bytes"])
        key = normalized["path"], normalized["sha256"], normalized["size_bytes"]
        if key not in seen:
            seen.add(key)
            bindings.append(normalized)
    return NativeSavedReview(reader.index, current_machine_id=reader.current_machine_id,
        expected_run=reader.expected_run, expected_calibration_sha256=reader.expected_calibration_sha256,
        collections=bindings, source_paths=reader.source_paths,
        geometry_producer_sha256=reader.geometry_producer_sha256,
        expected_pipeline=reader.pipeline, photographs=reader.photographs)


class _Attachment(QThread):
    def __init__(self, generation, context, reader, path, parent):
        super().__init__(parent)
        self.generation, self.context = generation, context
        self.reader, self.path = reader, path
        self.replacement, self.error = None, ""

    def run(self):
        if self.isInterruptionRequested():
            return
        try:
            replacement, error = attach_existing_collection(self.reader, self.path), ""
        except Exception as failure:
            replacement, error = None, str(failure) or type(failure).__name__
        if not self.isInterruptionRequested():
            self.replacement, self.error = replacement, error


class NativeCollectionLookup(QObject):
    """One metadata worker, one latest pending request, and no stale delivery."""

    ready = Signal(object, object, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._generation, self._worker, self._pending = 0, None, None
        self._closed = False
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.stop)

    def busy(self):
        return self._worker is not None or self._pending is not None

    def request(self, context, reader, selected_receipt_path):
        if self._closed:
            return
        self._generation += 1
        self._pending = self._generation, context, reader, selected_receipt_path
        if self._worker is not None:
            self._worker.requestInterruption()
        else:
            self._start()

    def _start(self):
        if self._closed or self._worker is not None or self._pending is None:
            return
        worker = _Attachment(*self._pending, self)
        self._pending, self._worker = None, worker
        # This QObject slot runs on the GUI thread after QThread has finished;
        # result handling never runs in the metadata worker's run() method.
        worker.finished.connect(self._worker_finished, Qt.ConnectionType.QueuedConnection)
        worker.start()

    @Slot()
    def _worker_finished(self):
        self._finish(self.sender())

    def _finish(self, worker):
        if worker is None or worker is not self._worker:
            return
        self._worker = None
        current = not self._closed and worker.generation == self._generation and self._pending is None
        context, replacement, error = worker.context, worker.replacement, worker.error
        worker.replacement, worker.reader = None, None
        worker.deleteLater()
        if current:
            # The host may synchronously rebind/reload in this callback, so
            # busy() must already be false for the completed current request.
            self.ready.emit(context, replacement, error)
        self._start()

    @Slot()
    def stop(self):
        self._closed = True
        self._generation += 1
        self._pending = None
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.requestInterruption()
            worker.wait()
            worker.replacement, worker.reader = None, None
            worker.deleteLater()
