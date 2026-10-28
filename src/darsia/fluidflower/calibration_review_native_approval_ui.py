"""One bounded metadata read for the currently displayed native run approval."""
from copy import deepcopy

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot


class _Read(QThread):
    ready = Signal(int, object, object, str)

    def __init__(self, generation, context, arguments, parent):
        super().__init__(parent)
        self.generation, self.context, self.arguments = generation, context, arguments

    def run(self):
        from .calibration_review_native_approval import prepare_native_run_approval
        if self.isInterruptionRequested():
            return
        try:
            binding, error = prepare_native_run_approval(**self.arguments), ""
        except Exception as exception:
            binding, error = None, str(exception)
        if not self.isInterruptionRequested():
            self.ready.emit(self.generation, self.context, binding, error)


class NativeApprovalLookup(QObject):
    ready = Signal(object, object, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._generation, self._worker, self._pending = 0, None, None
        self._closed = False

    def request(self, context, **arguments):
        if self._closed:
            return
        self._generation += 1
        arguments = dict(arguments, registration=deepcopy(arguments["registration"]),
                         source_registrations=deepcopy(arguments.get("source_registrations", ())))
        self._pending = context, arguments
        if self._worker is not None:
            self._worker.requestInterruption()
        else:
            self._start()

    def invalidate(self):
        self._generation += 1
        self._pending = None
        if self._worker is not None:
            self._worker.requestInterruption()

    def _start(self):
        if self._pending is None or self._closed or self._worker is not None:
            return
        context, arguments = self._pending
        worker = _Read(self._generation, context, arguments, self)
        self._pending, self._worker = None, worker
        worker.ready.connect(self._ready, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(self._worker_finished, Qt.ConnectionType.QueuedConnection)
        worker.start()

    @Slot(int, object, object, str)
    def _ready(self, generation, context, binding, error):
        if not self._closed and generation == self._generation:
            self.ready.emit(context, binding, error)

    @Slot()
    def _worker_finished(self):
        self._finished(self.sender())

    def _finished(self, worker):
        if worker is None or worker is not self._worker:
            return
        self._worker = None
        worker.context, worker.arguments = None, None
        worker.deleteLater()
        self._start()

    @Slot()
    def stop(self):
        self._closed = True
        self.invalidate()
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.wait()
            worker.context, worker.arguments = None, None
            worker.deleteLater()
