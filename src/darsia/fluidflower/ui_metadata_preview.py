"""Cancellable, disposable GUI metadata reads; no jobs or mutation authority."""
from contextvars import copy_context
from threading import Event, Thread
from PySide6.QtCore import QObject, Qt, Signal, Slot


class MetadataPreview(QObject):
    """Discard obsolete results without making GUI close wait on a network read."""
    completed = Signal(object, str)
    _returned = Signal(int, object, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.generation = 0
        self.busy = False
        self.cancelled = Event()
        self._returned.connect(self._accept, Qt.ConnectionType.QueuedConnection)

    def cancel(self):
        self.cancelled.set()
        self.generation += 1
        self.busy = False

    def start(self, read):
        self.cancel()
        generation, cancelled = self.generation, Event()
        self.cancelled, self.busy = cancelled, True
        context = copy_context()
        def run():
            value, error = None, ""
            try:
                if not cancelled.is_set():
                    value = read(cancelled.is_set)
            except Exception as failure:
                error = f"{type(failure).__name__}: {failure}"
            if cancelled.is_set():
                return
            try:
                self._returned.emit(generation, value, error)
            except RuntimeError:
                pass  # Its display may already have closed; no result is installed.
        Thread(target=lambda: context.run(run), name="darsia-metadata-preview", daemon=True).start()

    @Slot(int, object, str)
    def _accept(self, generation, value, error):
        if generation != self.generation or self.cancelled.is_set():
            return
        self.busy = False
        self.completed.emit(value, error)
