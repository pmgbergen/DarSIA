"""Native process ownership and original-review admission, with no child jobs."""
import hashlib
import json
from types import SimpleNamespace as NS

import pytest
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QLabel, QLineEdit, QRadioButton, QTextEdit, QWidget

from darsia.fluidflower import native_phase_review_owner as owner_module
from darsia.fluidflower import native_phase_review_session as lifecycle
from darsia.fluidflower import phase_workflow


class Event:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def disconnect(self, callback=None):
        self.callbacks = [] if callback is None else [c for c in self.callbacks if c != callback]

    def emit(self, *args):
        for callback in list(self.callbacks):
            callback(*args)


class Process:
    ProcessChannelMode = QProcess.ProcessChannelMode
    ProcessError = QProcess.ProcessError
    ProcessState = QProcess.ProcessState
    ExitStatus = QProcess.ExitStatus
    instances = []
    raise_on_start = False

    def __init__(self, parent):
        self.parent = parent
        self.deleted = False
        self.stderr = b""
        self.command = None
        self._state = self.ProcessState.NotRunning
        for name in ("readyReadStandardOutput", "readyReadStandardError", "errorOccurred", "finished", "started"):
            setattr(self, name, Event())
        self.instances.append(self)

    def setProcessChannelMode(self, value):
        self.channel_mode = value

    def start(self, program, arguments):
        if self.raise_on_start:
            raise RuntimeError("Synthetic failed process allocation")
        self.command = (program, arguments)
        self._state = self.ProcessState.Starting

    def state(self):
        return self._state

    def errorString(self):
        return "Interpreter unavailable"

    def readAllStandardError(self):
        result, self.stderr = self.stderr, b""
        return result

    def readAllStandardOutput(self):
        return b""

    def deleteLater(self):
        self.deleted = True


@pytest.fixture
def owned(qtbot, monkeypatch, tmp_path):
    monkeypatch.setenv("DARSIA_REVIEW_ALIAS_FILES", str(tmp_path / "absent-aliases.json"))
    monkeypatch.setattr(owner_module, "QProcess", Process)
    Process.instances = []
    Process.raise_on_start = False
    steps = QWidget()
    steps.series_id = "ffac"
    qtbot.addWidget(steps)
    calls = []
    studio = NS(review_save_root=QLineEdit(str(tmp_path / "reviews"), steps),
        reviewer=QLineEdit("Reviewer", steps), review_status=QLabel(steps), bundles=QTextEdit(steps),
        training_process=None, _series_batch_active=False, review_opened=Event())
    studio._start_review_process = lambda *args, **kwargs: calls.append(("original", args, kwargs))
    studio._selected_validation_model = lambda: {"model_id": "saved-model", "root": str(tmp_path / "saved-model")}
    studio._review_command = lambda mode, **kwargs: ["interpreter", mode, kwargs["session_id"]]
    studio._refresh_review_sessions = lambda: calls.append("refresh-sessions")
    workflow = NS(steps=steps, studio=studio, refresh=lambda: calls.append("refresh-page"))
    owner = owner_module.NativePhaseReviewOwner(workflow)
    return NS(owner=owner, studio=studio, steps=steps, calls=calls,
              root=tmp_path / "reviews", identity=dict(series="ffac", reviewer="Reviewer"))


def test_failed_start_retires_process_and_retry_reuses_reservation(owned):
    s = owned
    s.owner.start("open_correction", {"candidate": "saved"})
    process = s.owner.process
    session_id = lifecycle.read_current_review_session(s.root, **s.identity)["review_session_id"]
    process.errorOccurred.emit(QProcess.ProcessError.FailedToStart)
    assert process.deleted and s.owner.process is None
    assert "reservation is retained" in s.studio.review_status.text()
    assert lifecycle.read_current_review_session(s.root, **s.identity)["review_session_id"] == session_id
    s.owner.start("blind_region_labeling", {"candidate": "saved"})
    assert s.owner.process is not process
    assert session_id in s.owner.process.command[1]
    assert "blind" in s.owner.process.command[1]
    s.owner.process.finished.emit(0, QProcess.ExitStatus.NormalExit)


def test_scratch_uses_selected_photographs_without_a_target_model(owned):
    s = owned
    s.steps.scratch_basis = QRadioButton(s.steps)
    s.steps.scratch_basis.setChecked(True)
    binding = dict(ffac_root=str(s.root.parent / "ac"),
        ffum_root=str(s.root.parent / "um"), combined_root=str(s.root.parent / "combined"),
        selected_candidates={"ac01": "photograph-source"})
    s.studio._review_command = lambda mode, **kw: ["interpreter", mode,
        "--review-session-id", kw["session_id"], "--target-model-id", "old-model"]
    s.owner.start("open_correction", binding)
    process = s.owner.process
    assert process is not None, s.studio.review_status.text()
    assert process.review_basis == "scratch"
    assert "--target-model-id" not in process.command[1]
    assert "--review-current-model-root" not in process.command[1]
    session = lifecycle.read_current_review_session(s.root, **s.identity)
    assert session["target_model_id"] == "" and session["review_basis"] == "scratch"
    assert session["source_identity"]["selected_candidates"] == binding["selected_candidates"]
    process.finished.emit(0, QProcess.ExitStatus.NormalExit)
    s.owner.start("open_correction", binding)
    assert lifecycle.read_current_review_session(s.root, **s.identity)["review_session_id"] == session["review_session_id"]
    s.owner.process.finished.emit(0, QProcess.ExitStatus.NormalExit)


def test_one_process_is_retained_until_finished_and_saved_session_remains_active(owned):
    s = owned
    s.owner.start("open_correction", {})
    process = s.owner.process
    s.owner.start("blind_region_labeling", {})
    assert s.owner.process is process and len(Process.instances) == 1
    assert "already open" in s.studio.review_status.text()
    with pytest.raises(ValueError, match="Close the phase correction"):
        s.owner.run_training(lambda: pytest.fail("Must not train while review is open"))
    process.finished.emit(0, QProcess.ExitStatus.NormalExit)
    assert s.owner.process is None and process.deleted
    assert lifecycle.read_current_review_session(s.root, **s.identity)["status"] == "active"
    assert "refresh-sessions" in s.calls


def test_queued_retired_process_callback_cannot_retire_a_new_window(owned):
    s = owned
    s.owner.start("open_correction", {})
    old = s.owner.process
    queued_finished = list(old.finished.callbacks)
    queued_error = list(old.errorOccurred.callbacks)
    old.errorOccurred.emit(QProcess.ProcessError.FailedToStart)
    s.owner.start("open_correction", {})
    current = s.owner.process
    for callback in queued_finished:
        callback(1, QProcess.ExitStatus.CrashExit)
    for callback in queued_error:
        callback(QProcess.ProcessError.FailedToStart)
    assert s.owner.process is current and not current.deleted
    current.finished.emit(0, QProcess.ExitStatus.NormalExit)


def test_synchronous_start_error_retires_allocation_and_keeps_reservation(owned):
    Process.raise_on_start = True
    owned.owner.start("open_correction", {})
    assert owned.owner.process is None
    assert Process.instances[-1].deleted
    assert lifecycle.read_current_review_session(owned.root, **owned.identity)["status"] == "active"
    assert "Synthetic failed" in owned.studio.review_status.text()


def test_blank_save_root_is_rejected_before_registry_or_child_creation(owned):
    owned.studio.review_save_root.clear()
    owned.owner.start("open_correction", {})
    assert not Process.instances
    assert not owned.root.exists()
    assert "review" in owned.studio.review_status.text().casefold()


def test_empty_original_command_is_rejected_before_flags_can_become_program(owned):
    owned.studio._review_command = lambda *args, **kwargs: []
    owned.owner.start("open_correction", {})
    assert not Process.instances
    assert owned.owner.process is None


def test_crash_exit_with_zero_code_still_reports_an_error_and_preserves_session(owned):
    owned.owner.start("open_correction", {})
    process = owned.owner.process
    process.stderr = b"Child did not complete its normal close"
    process.finished.emit(0, QProcess.ExitStatus.CrashExit)
    assert process.deleted and owned.owner.process is None
    assert "stopped with an error" in owned.studio.review_status.text()
    assert "Child did not complete" in owned.studio.review_status.text()
    assert lifecycle.read_current_review_session(owned.root, **owned.identity)["status"] == "active"


def test_nonphase_review_route_keeps_original_callback(owned):
    owned.owner.start("comparison", {"original": True}, session_id="original-session")
    assert owned.calls == [("original", ("comparison", {"original": True}), {"session_id": "original-session"})]
    assert not Process.instances and not owned.root.exists()


def selected_evidence(s, *, excluded=False):
    row = lifecycle.reserve_review_session(s.root, **s.identity, target_model_id="saved-model")
    folder = s.root / "reviewer" / "training" / row["review_session_id"]
    folder.mkdir(parents=True)
    session_path = folder / "review_session.json"
    session_path.write_text(json.dumps(dict(format=lifecycle.SESSION_FORMAT,
        review_session_id=row["review_session_id"], target_model_id="saved-model",
        reviewer="Reviewer", decision_dir=str(folder))), encoding="utf-8")
    decisions = folder / "pairwise_review_decisions.json"
    decisions.write_text('{"decisions": {"ac28": {"annotations": [{"id":"box"}]}}}', encoding="utf-8")
    manifest = s.root / "combined" / "postflight_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text('{"jobs": []}', encoding="utf-8")
    lifecycle.attach_review_manifest(s.root, **s.identity,
        review_session_id=row["review_session_id"], session_path=session_path)
    s.studio.bundles.setPlainText(str(decisions) + " | " + str(manifest))
    if excluded:
        lifecycle.discard_review_session(s.root, **s.identity,
            review_session_id=row["review_session_id"], reason="Use another saved review")
    return row, session_path, decisions, manifest


def test_training_closes_session_only_after_original_freeze_and_lock(owned):
    s = owned
    row, path, decisions, manifest = selected_evidence(s)
    snapshot = s.root / "frozen_decisions.json"
    snapshot.write_bytes(decisions.read_bytes())
    request = dict(model_id="new-model", request_path=str(s.root / "training_request.json"),
                   bundles=[dict(decisions=dict(source_path=str(decisions), snapshot_path=str(snapshot),
                       sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest()))])

    def original_prepare_and_launch():
        assert lifecycle.read_current_review_session(s.root, **s.identity)["status"] == "active"
        phase_workflow.lock_phase_review_sessions(request)
        s.owner.before_launch(request)
        s.calls.append("admitted-original-launch")

    s.owner.run_training(original_prepare_and_launch)
    assert lifecycle.read_current_review_session(s.root, **s.identity) is None
    assert lifecycle.read_review_session_registry(s.root, **s.identity)["sessions"][0]["status"] == "submitted"
    assert s.owner._training_session is None and "admitted-original-launch" in s.calls


def test_discarded_evidence_fails_before_training_callback(owned):
    selected_evidence(owned, excluded=True)
    with pytest.raises(ValueError, match="excluded from training"):
        owned.owner.run_training(lambda: pytest.fail("Excluded evidence reached original training"))
    assert owned.owner._training_session is None


def test_saved_labels_only_training_preserves_unrelated_active_review(owned, tmp_path):
    selected_evidence(owned)
    active = lifecycle.read_current_review_session(owned.root, **owned.identity)
    old_decisions, old_manifest = tmp_path / "old-labels.json", tmp_path / "old-images.json"
    old_decisions.write_text('{"decisions": {}}')
    old_manifest.write_text('{"jobs": []}')
    owned.studio.bundles.setPlainText(str(old_decisions) + " | " + str(old_manifest))
    called = []
    def launch():
        owned.owner.before_launch(dict(model_id="new-version"))
        called.append(True)
    owned.owner.run_training(launch, saved_labels_only=True)
    assert called == [True]
    assert lifecycle.read_current_review_session(owned.root, **owned.identity) == active
    assert owned.owner._training_session is None


def test_preparation_failure_retains_active_session_and_clears_launch_authority(owned):
    selected_evidence(owned)

    def fail_preparation():
        raise ValueError("Original immutable source check failed")

    with pytest.raises(ValueError, match="immutable source"):
        owned.owner.run_training(fail_preparation)
    assert owned.owner._training_session is None
    assert lifecycle.read_current_review_session(owned.root, **owned.identity)["status"] == "active"
    with pytest.raises(ValueError, match="session admission"):
        owned.owner.before_launch({"model_id": "unrelated"})


def test_original_request_boundary_refuses_discarded_session_before_any_output(owned, tmp_path):
    _row, _path, decisions, manifest = selected_evidence(owned, excluded=True)
    output, cache = tmp_path / "new-models", tmp_path / "new-cache"
    with pytest.raises(ValueError, match="excluded from training"):
        phase_workflow.prepare_phase_training_request(output_parent=output, cache_parent=cache,
            bundles=[(decisions, manifest)], actor="Reviewer", settings={})
    assert not output.exists() and not cache.exists()
