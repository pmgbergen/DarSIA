"""Save and seal blind evidence before the first visible phase prediction."""
from contextvars import copy_context
from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal, Slot

from ..phase_evidence_audit import read
from ..phase_review_contradictions import read_reveal_marker, seal_blind_evidence


class _Signals(QObject):
    finished = Signal(object, str, bool)


class _SealTask(QRunnable):
    def __init__(self, decisions, workspace, reviewer, write_session):
        super().__init__()
        self.signals = _Signals()
        self.context = copy_context()
        self.decisions, self.workspace = decisions, deepcopy(workspace)
        self.reviewer, self.write_session = reviewer, write_session

    def run(self):
        self.context.run(self._run)

    def _run(self):
        marker, error, invalid = None, "", False
        root = Path(self.workspace.output_dir)
        try:
            marker = seal_blind_evidence(self.decisions, root / "review_session.json",
                model_id=self.workspace.target_model_id, reviewer=self.reviewer)
            self.write_session(self.workspace, self.reviewer)
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"
            if (root / "blind_reveal.json").exists():
                try:
                    marker = read_reveal_marker(root)
                except Exception:
                    invalid = True
        self.signals.finished.emit(marker, error, invalid)


class PhasePredictionDisclosure(QObject):
    def __init__(self, window, desktop):
        super().__init__(window)
        self.window, self.desktop = window, desktop
        self.marker, self.callback, self.task = None, None, None
        self.awaiting_save = self.closing = self.seal_invalid = False
        self.save_started = False
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self._central_was_enabled = window.centralWidget().isEnabled()
        root = Path(window.workspace.output_dir)
        try:
            session = read(root / "review_session.json")
            if ((root / "blind_reveal.json").exists()
                    or session.get("blindness_status") == "revealed" or session.get("frozen_blind_snapshot")):
                self.marker = read_reveal_marker(root)
                window._workflow_blind_sealed(self.marker)
        except Exception as error:
            self.seal_invalid = True
            self._lock()
            self._error(error)

    def _status(self, text):
        self.window.statusBar().showMessage(text)

    def _error(self, error):
        self._status(f"Phase map remains hidden: {error}")

    def _lock(self):
        self.window._workflow_sealing = True
        self.window.centralWidget().setEnabled(False)
        self.window._refresh_quiz()

    def _unlock(self):
        if self.seal_invalid:
            return
        self.window._workflow_sealing = False
        if not self.closing and not getattr(self.window, "_workflow_closing", False):
            self.window.centralWidget().setEnabled(self._central_was_enabled)
        self.window._refresh_quiz()

    def request(self, callback):
        if self.closing or self.seal_invalid:
            return False
        if self.marker is not None:
            callback()
            return True
        self.callback = callback
        if self.awaiting_save or self.task is not None:
            return True
        w = self.window
        if (w.store.is_read_only or not w.reviewer.text().strip()
                or w.quiz_preparation_task is not None
                or any(not adapter.can_close() for adapter in w._workflow_adapters if adapter is not self)):
            self.callback = None
            self._error("The editable review and its current work must finish saving first.")
            return False
        self._central_was_enabled = w.centralWidget().isEnabled()
        self.awaiting_save = True
        self.save_started = False
        self._lock()
        self._status("Saving the review before showing the phase map…")
        try:
            w._save_now()
            self.save_started = True
            if w.sync_error_pending:
                raise RuntimeError("The original review save failed. Retry saving before showing the phase map.")
            self._saved_complete()
        except Exception as error:
            self.awaiting_save, self.callback = False, None
            self._unlock()
            self._error(error)
            return False
        return True

    def _saved_complete(self):
        w = self.window
        if (not self.awaiting_save or not self.save_started or self.closing or w.save_inflight
                or w.pending_save_requests or w.save_dirty or w.sync_error_pending):
            return
        self.awaiting_save = False
        self.task = _SealTask(Path(w.store.json_path), w.workspace, w.reviewer.text().strip(),
                              self.desktop.write_review_session_manifest)
        self.task.signals.finished.connect(self._finished, Qt.ConnectionType.QueuedConnection)
        self._status("Sealing blind answers before showing the phase map…")
        self.pool.start(self.task)

    @Slot(object, str, bool)
    def _finished(self, marker, error, invalid):
        self.task, self.seal_invalid = None, invalid
        callback, self.callback = self.callback, None
        if marker is not None:
            # Even a later metadata-write failure cannot undo disclosure.
            self.window._workflow_blind_sealed(marker)
        if not error:
            self.marker = marker
        self._unlock()
        if error:
            self._error(error)
        elif not self.closing and callback is not None:
            self._status("Blind answers sealed. Later edits are corrections.")
            callback()

    def render(self):
        self._saved_complete()

    def save_finished(self, generation, run, record, error):
        if self.awaiting_save and error:
            self.awaiting_save, self.callback = False, None
            self._unlock()
            self._error("The original review save failed. Retry saving before showing the phase map.")
        self._saved_complete()

    def begin_close(self):
        self.closing, self.awaiting_save, self.callback = True, False, None
        if self.task is None:
            self._unlock()

    def can_close(self):
        return not self.awaiting_save and self.task is None and self.pool.activeThreadCount() == 0

    def cancel_close(self):
        self.closing = False
        if self.task is not None or self.seal_invalid:
            self._lock()
        else:
            self._unlock()


def install(window, desktop):
    if not isinstance(window, desktop.CalibrationReviewWindow):
        raise ValueError("Prediction disclosure requires the original CalibrationReviewWindow")
    return PhasePredictionDisclosure(window, desktop)
