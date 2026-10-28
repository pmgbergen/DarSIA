"""One coalescing metadata lookup for explicitly registered native review runs."""
from pathlib import Path
from copy import deepcopy
import socket

from PySide6.QtCore import QCoreApplication, QObject, QStandardPaths, QThread, Signal

from .native_jobs_registry import JobsUIRegistry


class _Lookup(QThread):
    ready = Signal(int, object, object, object)

    def __init__(self, generation, context, registry, machine, parent):
        super().__init__(parent)
        self.generation, self.context = generation, context
        self.registry, self.machine = registry, machine
        self.extra_collections = {}

    def run(self):
        from .calibration_review_native_catalog import discover_native_reviews
        try:
            dataset, root = self.context
            rows, errors = discover_native_reviews(self.registry, root, dataset,
                current_machine_id=self.machine, cancelled=self.isInterruptionRequested,
                extra_collections=self.extra_collections)
        except Exception as error:
            rows, errors = [], [str(error)]
        if not self.isInterruptionRequested():
            self.ready.emit(self.generation, self.context, rows, errors)


class NativeReviewCatalogue(QObject):
    ready = Signal(object, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        root = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation))
        self.registry = JobsUIRegistry(root / "native_jobs")
        # Use the same receiving-host identity as the existing Jobs panel.
        # Never select a producer's identity to make its paths look local.
        self.machine = socket.gethostname()
        self._generation, self._worker, self._pending = 0, None, None
        self._pending_collections = {}
        QCoreApplication.instance().aboutToQuit.connect(self.stop)

    def is_loading(self, context):
        return self._pending == context or self._worker is not None and self._worker.context == context

    def refresh(self, dataset, root, *, extra_collections=None):
        self._generation += 1
        self._pending = str(dataset), str(Path(root).resolve())
        self._pending_collections = deepcopy(extra_collections or {})
        if self._worker is not None:
            self._worker.requestInterruption()
        else:
            self._start()

    def _start(self):
        if self._pending is None:
            return
        worker = _Lookup(self._generation, self._pending, self.registry, self.machine, self)
        worker.extra_collections = self._pending_collections
        self._pending_collections = {}
        self._pending, self._worker = None, worker
        worker.ready.connect(self._ready)
        worker.finished.connect(lambda: self._finished(worker))
        worker.start()

    def _ready(self, generation, context, rows, errors):
        if generation == self._generation:
            self.ready.emit(context, rows, errors)

    def _finished(self, worker):
        if self._worker is worker:
            self._worker = None
        worker.deleteLater()
        self._start()

    def stop(self):
        self._generation += 1
        self._pending = None
        self._pending_collections = {}
        if self._worker is not None:
            self._worker.requestInterruption()
            self._worker.wait()
