"""One explicit saved-FINAL-map box request on the native review surface.

The desktop owns the existing store and numerical reader context. This adapter
adds an action and retires its finite worker before the window may close. It does
not create a review session or deserialize/overwrite any answers.
"""
from __future__ import annotations

from copy import deepcopy
import json
import threading
from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import QObject, QThreadPool, QTimer, Signal, Slot
from PySide6.QtWidgets import QLabel, QPushButton, QToolBar, QWidget, QVBoxLayout

from .. import phase_review_saved_categories as saved_categories
from ..phase_review_proposal_generation import (
    ProposalPublicationGuard, ProposalSourceUnavailable,
    generate_more, prepare_saved_categorical_source,
)


class _Signals(QObject):
    progress = Signal(object, str)
    finished = Signal(object, object, str)


def _validate_request(context):
    """Recheck only saved session metadata; safe to call on the reader worker.

    The shared child already owns its editing lease on the GUI thread. Do not
    reacquire that thread-owned lease here, and never inspect a widget here.
    """
    from ..native_phase_review_session import read_current_review_session
    from .phase_model_correction import _bound_path, validate_workspace

    workspace = SimpleNamespace(**context["workspace"])
    manifest_path = Path(workspace.output_dir) / "review_session.json"
    binding_path = manifest_path.with_name("review_source_binding.json")
    if not manifest_path.is_file() or not binding_path.is_file():
        raise ValueError("The original review session or exact source binding is unavailable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if (not isinstance(manifest, dict) or manifest.get("blindness_status") == "revealed"
            or manifest.get("frozen_blind_snapshot") or manifest_path.with_name("blind_reveal.json").exists()):
        raise ValueError("Blind answers are sealed; no additional blind boxes may be published")
    reservation = read_current_review_session(context["save_root"], series=context["series"],
                                              reviewer=context["reviewer"])
    if reservation is None:
        raise ValueError("The native review session is no longer active")
    registered = {_bound_path(path, context=manifest_path) for path in reservation["manifest_paths"]}
    if _bound_path(manifest_path, context=manifest_path) not in registered:
        raise ValueError("The original review store is not attached to this active session")
    binding = validate_workspace(workspace, reviewer=context["reviewer"], reservation=reservation,
                                 model_root=context["model_root"])
    if binding != context["binding"]:
        raise ValueError("The exact saved review source binding changed during this request")
    if binding["selected_candidates"].get(context["run"]) != context["candidate_id"]:
        raise ValueError("The requested experiment is not the session's selected candidate")


def _request_context(window, actor):
    """Copy the launch binding on the GUI thread without retaining any widget."""
    context = deepcopy(getattr(window, "_workflow_request_context", None))
    if not isinstance(context, dict):
        raise ValueError("New boxes require the shared window's exact model and session binding")
    workspace = window.workspace
    captured = context["workspace"]
    current = {key: deepcopy(getattr(workspace, key)) for key in captured}
    if (actor != context["reviewer"] or str(window._workflow_model_root) != str(context["model_root"])
            or current != captured):
        raise ValueError("The current review no longer matches its original workspace, model, session or human owner")
    candidate = window.current_pair_object.candidate_a
    context.update(run=str(candidate.run).casefold(), candidate_id=candidate.candidate_id)
    if context["binding"]["selected_candidates"].get(context["run"]) != candidate.candidate_id:
        raise ValueError("Choose an experiment from this session's exact saved candidate selection")
    return context


def _task_type(desktop):
    class RequestedBoxesTask(desktop._SavedReaderContextTask):
        def __init__(self, *, key, candidate, model_id, model_root, output_dir, count, actor, existing,
                     initial_proposal_count, context):
            super().__init__()
            self.key, self.candidate, self.model_id = key, candidate, model_id
            self.model_root, self.context = model_root, context
            self.output_dir, self.count, self.actor, self.existing = output_dir, count, actor, existing
            self.initial_proposal_count = initial_proposal_count
            self.cancelled = threading.Event()
            self.publication_guard = ProposalPublicationGuard()
            self.signals = _Signals()

        def invalidate(self):
            self.publication_guard.invalidate()
            self.cancelled.set()

        def is_cancelled(self):
            return self.cancelled.is_set() or self.publication_guard.is_cancelled()

        def _run_saved_read(self):
            try:
                _validate_request(self.context)
                if desktop.review._candidate_prediction_withheld(self.output_dir, self.candidate):
                    raise PermissionError("This selected run's predictions are withheld by its original publication policy; new model-based boxes are unavailable.")
                campaign = desktop._quiz_generation_campaign(self.candidate)
                if campaign is None:
                    raise ProposalSourceUnavailable(
                        "This candidate has no unique saved FINAL-map campaign for generating new boxes. "
                        "Its existing boxes and answers remain available.")
                def report(message):
                    self.signals.progress.emit(self, message)
                report("Checking the selected model's saved FINAL phase map and rule-change regions")
                source = prepare_saved_categorical_source(campaign, self.candidate.run,
                    candidate_id=self.candidate.candidate_id, params_sha256=self.candidate.params_sha256,
                    model_id=self.model_id, model_root=self.model_root, cancelled=self.is_cancelled)
                answer = generate_more(source, self.output_dir, requested_count=self.count, actor=self.actor,
                    existing_proposals=self.existing, progress=report, cancelled=self.is_cancelled,
                    initial_proposal_count=self.initial_proposal_count,
                    publication_guard=self.publication_guard,
                    validate_request=lambda: _validate_request(self.context))
                self.signals.finished.emit(self, answer, "")
            except Exception as error:  # Visible worker boundary; no fake successful pool.
                self.signals.finished.emit(self, {}, f"{type(error).__name__}: {error}")
    return RequestedBoxesTask


class ReviewProposalGeneration(QObject):
    def __init__(self, window, desktop):
        super().__init__(window)
        self.window, self.desktop = window, desktop
        self.task = None
        self.completion = None
        self.close_requested = False
        self.pool = QThreadPool(window)
        self.pool.setMaxThreadCount(1)
        self.retirement = QTimer(window)
        self.retirement.setInterval(50)
        self.retirement.timeout.connect(self._retire)
        toolbar = QToolBar("Additional blind validation boxes", window)
        toolbar.setObjectName("phaseRequestedBoxesToolbar")
        toolbar.setMovable(False)
        body = QWidget(toolbar)
        layout = QVBoxLayout(body)
        self.button = QPushButton("Generate more blind validation boxes")
        self.button.setToolTip(
            "Generate new boxes from this candidate's saved FINAL phase boundaries and saved rule-change regions. "
            "Uses the number beside Add regions. No phase maps are recalculated.")
        self.status = QLabel("Add regions uses the saved pool. Generate more adds boxes from the saved FINAL map and available saved rule-change regions.")
        self.status.setWordWrap(True)
        layout.addWidget(self.button)
        layout.addWidget(self.status)
        toolbar.addWidget(body)
        window.addToolBar(toolbar)
        self.toolbar = toolbar
        self.button.clicked.connect(self.request)
        self.render()

    def _blind_locked(self):
        return bool(getattr(self.window, "_workflow_revealed", False)
                    or getattr(self.window, "_workflow_sealing", False))

    def render(self):
        """Called by the shared window when the review's sealing state changes."""
        self.button.setEnabled(not self.close_requested and self.task is None
                               and not self.window.store.is_read_only and not self._blind_locked())

    def blind_sealed(self):
        self.invalidate_request()
        self.render()
        if self._blind_locked():
            self.status.setText("Blind answers are sealed or being sealed. Additional blind boxes cannot be requested in this session.")

    def _key(self):
        pair = self.window.current_pair_object
        if pair is None:
            return None
        candidate = pair.candidate_a
        return (candidate.run, candidate.candidate_id, candidate.params_sha256,
                str(self.window.workspace.target_model_id), str(self.window.workspace.output_dir),
                str(getattr(self.window, "_workflow_model_root", "")),
                str(self.window.workspace.review_session_id),
                getattr(self.window, "_workflow_selection_epoch", 0))

    def invalidate_request(self):
        """Revoke publication before the shared owner changes selection or seal."""
        if self.task is not None:
            self.task.invalidate()

    def request(self):
        window = self.window
        if self.task is not None or self.close_requested:
            return
        if self._blind_locked():
            self.blind_sealed()
            return
        if window.store.is_read_only:
            self.status.setText("Frozen review: no new boxes can be added to this evidence session.")
            return
        if getattr(window, "quiz_preparation_task", None) is not None:
            self.status.setText("Wait for the existing saved-box reader to finish before generating more boxes.")
            return
        if window.workspace.review_surface != "blind_regions":
            self.status.setText("New blind boxes require the blind validation role; current labels remain unchanged.")
            return
        key = self._key()
        if key is None:
            self.status.setText("Choose a prepared experiment before requesting boxes.")
            return
        actor = window.reviewer.text().strip()
        if not actor:
            self.status.setText("Enter your reviewer name before requesting blind validation boxes.")
            return
        try:
            context = _request_context(window, actor)
            task = _task_type(self.desktop)(key=key, candidate=deepcopy(window.current_pair_object.candidate_a),
                model_id=window.workspace.target_model_id, model_root=context["model_root"],
                output_dir=Path(window.workspace.output_dir), count=window.new_round_count.value(),
                actor=actor, existing=deepcopy(window.pool), context=context,
                initial_proposal_count=window.current_pair["candidate_a"].get(
                    "phase_quiz_initial_count", len(window.pool)))
        except Exception as error:
            self.status.setText("Could not request new boxes: " + str(error))
            return
        self.task = task
        task.signals.progress.connect(self._progress)
        task.signals.finished.connect(self._finished)
        self.button.setEnabled(False)
        self.status.setText("Checking the exact saved candidate and FINAL map…")
        self.pool.start(task)

    @Slot(object, str)
    def _progress(self, task, message):
        if (not self.close_requested and task is self.task and not task.is_cancelled()
                and task.key == self._key()):
            self.status.setText(message)

    @Slot(object, object, str)
    def _finished(self, task, result, error):
        # A queued signal can arrive before QRunnable.run returns. Keep owning the
        # task/pool until activeThreadCount reaches zero; never destroy a live worker.
        if task is not self.task:
            return
        self.completion = task, result, error
        self.retirement.start()
        self._retire()

    @Slot()
    def _retire(self):
        if self.completion is None or self.pool.activeThreadCount():
            return
        task, result, error = self.completion
        if task is not self.task:
            self.completion = None
            return
        self.completion = None
        self.task = None
        self.retirement.stop()
        self.render()
        if self.close_requested or self._blind_locked():
            # The original store has already captured its final save. A completed
            # overlay can be reloaded next time; it must not enqueue a late save.
            self.status.setText("Requested box work has finished; the closing or sealed review was preserved.")
            return
        if task.is_cancelled() or task.key != self._key():
            self.status.setText("The requested selection changed or was cancelled; the current experiment was not changed.")
            return
        if error:
            self.status.setText("Could not generate more blind boxes: " + error)
            return
        window = self.window
        try:
            if window.store.is_read_only:
                raise ValueError("The original review store became read-only before these boxes could be attached")
            if _request_context(window, window.reviewer.text().strip()) != task.context:
                raise ValueError("The exact current review context changed before these boxes could be attached")
            _validate_request(task.context)
            # Publication and attachment are separate boundaries. Recheck the
            # compact current identity after actual runnable retirement, before
            # touching any live review state. This helper never reads pixels.
            saved_categories.validate_current_identity(result["payload"]["source"])
            rows = [dict(row, candidate="A") for row in result["payload"]["proposals"]]
            by_id = {row["proposal_id"]: row for row in rows}
            # Manual edits and answers can arrive while the worker runs. Check the
            # live pool instead of restoring any captured UI/save snapshot.
            if any(row["proposal_id"] not in by_id or by_id[row["proposal_id"]] != row for row in window.pool):
                raise ValueError("The live box list changed; reopen this experiment to reconcile the saved pool.")
            window.pool = rows
            candidate = window.current_pair["candidate_a"]
            candidate.update(phase_quiz_proposals=rows, phase_quiz_ready=bool(rows),
                             phase_quiz_available_count=len(rows))
            # Do not reset selected_ids, answers, annotations, rounds or quiz_index.
            # Original Add regions remains the only action that offers a new batch.
            window._refresh_quiz()
            window._queue_save()
            self.status.setText(result["message"] + " Use Add regions to offer boxes from this pool.")
        except Exception as error:
            self.status.setText("New boxes were saved, but the live review was preserved: " + str(error))

    def can_close(self):
        """Inspect retirement without cancelling work during ordinary busy checks."""
        return self.task is None and not self.pool.activeThreadCount()

    def begin_close(self):
        """Request cooperative cancellation; retain the task until it has retired."""
        self.invalidate_request()
        self.close_requested = True
        self.button.setEnabled(False)
        if self.task is not None:
            self.status.setText("Cancelling requested box generation before closing the review.")

    def cancel_close(self):
        """Resume UI actions when the original review's final save failed."""
        self.close_requested = False
        self.render()
        if self.task is not None:
            self.status.setText("The review remains open. Waiting for cancelled box work to finish.")
        else:
            self.status.setText("The review remains open; additional blind boxes can be requested again.")


def install(window, desktop):
    if getattr(window, "requested_phase_boxes", None) is not None:
        return window.requested_phase_boxes
    if not isinstance(window, desktop.CalibrationReviewWindow):
        raise ValueError("Requested blind boxes require the original CalibrationReviewWindow tools.")
    adapter = ReviewProposalGeneration(window, desktop)
    window.requested_phase_boxes = adapter
    return adapter
