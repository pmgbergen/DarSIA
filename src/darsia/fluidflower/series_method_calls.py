"""Owned finite calls for the series-choice metadata; no numerical workers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot


class SeriesMethodCalls(QObject):
    """Cache only the exact current root/selection/manifest generation.

    File reads, identity checks and explicit writes use the original finite
    reader. Readiness properties and field-change callbacks perform no I/O.
    """

    changed = Signal()
    action_finished = Signal(object)

    def __init__(self, parent=None, *, verify_approval=True):
        super().__init__(parent)
        self.verify_approval = bool(verify_approval)
        self.context = None
        self.record = None
        self.error = ""
        self.binding = None
        self.checked_context = None
        self.generation = 0
        self.thread = self.worker = self.call = None
        self.kind = None
        from .ui_metadata_preview import MetadataPreview
        self.preview = MetadataPreview(self)
        self.preview.completed.connect(self._verified_preview)
        self.preview_context = None
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._verify)

    def set_context(self, root, selection, manifest_bytes, record, error="", *, force=False):
        context = (str(root) if root is not None else None, selection, manifest_bytes)
        if context != self.context or force:
            self.preview.cancel()
            self.context = context
            self.generation += 1
            self.binding = self.checked_context = None
            self.error = error
        self.record = deepcopy(record)
        if error:
            self.error = error
        self._schedule()

    def _schedule(self):
        self.timer.stop()
        if (not self.verify_approval or self.thread is not None or self.preview.busy
                or self.context is None or self.error):
            return
        root, selection, raw = self.context
        if (root is not None and raw is not None and selection.active == "phase_model"
                and selection.model_version.strip() and selection.mass_calibration.strip()
                and self.checked_context != self.context):
            self.timer.start()

    def ready(self):
        return (self.thread is None and not self.preview.busy and self.context is not None
                and self.checked_context == self.context and self.binding is not None
                and not self.error)

    def _verify(self):
        if self.thread is not None or self.preview.busy or self.context is None:
            return
        context = self.context
        root, selection, raw = context

        def verify(cancelled):
            from .series_calibration_approval import read_approval, selection_binding, same_selection_binding
            from .series_method import ManifestChangedError

            manifest = Path(root) / "series.toml"
            if manifest.read_bytes() != raw:
                raise ManifestChangedError("The series changed while checking the selected files.")
            record = read_approval(root, verify=False)
            if cancelled():
                return None
            binding = selection_binding(root, selection)
            if record is not None and record["locked"] and not same_selection_binding(root, binding, record["binding"]):
                raise ValueError("The approved calibration or model changed. Unlock and review the choice.")
            if manifest.read_bytes() != raw:
                raise ManifestChangedError("The series changed while checking the selected files.")
            return binding

        self.preview_context = (context, self.generation)
        self.preview.start(verify)
        self.changed.emit()

    def _verified_preview(self, binding, error):
        if self.preview_context == (self.context, self.generation):
            self.checked_context = self.context
            self.binding, self.error = binding, error
            self.changed.emit()

    def start_action(self, kind, function, context):
        if self.thread is not None or self.preview.busy or context != self.context:
            return False
        self.timer.stop()
        return self._start(kind, function, context)

    def _start(self, kind, function, context):
        from .painting_gap_review_ui import _Read

        thread = QThread(self)
        worker = _Read(function)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._retired)
        thread.finished.connect(thread.deleteLater)
        self.thread, self.worker = thread, worker
        self.call = (context, self.generation)
        self.kind = kind
        thread.start()
        self.changed.emit()
        return True

    @Slot()
    def _retired(self):
        thread, worker, call, kind = self.thread, self.worker, self.call, self.kind
        if thread is None:
            return
        thread.wait()
        self.thread = self.worker = self.call = self.kind = None
        context, generation = call
        current = context == self.context and generation == self.generation
        if kind == "verify":
            if current:
                self.checked_context = context
                self.binding = None if worker.error else worker.result
                self.error = worker.error
        else:
            if kind == "method_action" and current:
                self.checked_context = context
                self.binding = None if worker.error else worker.result
                self.error = worker.error
            self.action_finished.emit(dict(kind=kind, root=context[0], current=current,
                                           result=worker.result, error=worker.error))
        self.changed.emit()
        self._schedule()

    def can_close(self):
        return self.thread is None

    def stop_pending(self):
        """Cancel disposable read-only metadata; keep explicit mutation actions owned."""
        self.timer.stop()
        self.preview.cancel()
