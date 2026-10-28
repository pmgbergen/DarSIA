"""Additive saved-FINAL-map requests preserve review and worker ownership.

Tests are authored for the isolated GUI suite; no science or jobs are started.
"""
from contextvars import ContextVar
from copy import deepcopy
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace as NS

import pytest
from PySide6.QtWidgets import QLineEdit, QMainWindow, QSpinBox

from darsia.fluidflower.ui import phase_review_proposal_generation as proposal_ui


class OriginalReviewWindow(QMainWindow):
    def __init__(self, root):
        super().__init__()
        self.store = NS(is_read_only=False)
        self.workspace = NS(key="training", single_model=True, review_session_id="",
            target_model_id="model-1", output_dir=root / "reviews" / "store", review_surface="blind_regions",
            combined_root=root / "campaign", ffac_root=root / "campaign", ffum_root=root / "campaign",
            selected_candidates={"ac01": "candidate-1"}, ffac_runs=None, ffum_runs=None)
        self._workflow_model_root = root / "model-1"
        self._workflow_selection_epoch = 1
        self.current_pair_object = NS(candidate_a=NS(run="ac01", candidate_id="candidate-1",
            params_sha256="a" * 64, quality_flags={}))
        self.current_pair = {"candidate_a": {"phase_quiz_initial_count": 1}}
        self.reviewer = QLineEdit("Reviewer", self)
        self.new_round_count = QSpinBox(self)
        self.new_round_count.setValue(2)
        self.pool = [{"proposal_id": "old", "candidate": "A"}]
        self.selected_ids = ["old"]
        self.answers = {"old": {"phase_class": "aqueous"}}
        self.annotations = [{"id": "manual", "phase_class": "gas"}]
        self.rounds = [{"proposal_ids": ["old"]}]
        self.quiz_index = 0
        self.refreshed = self.saves = 0

    def _refresh_quiz(self):
        self.refreshed += 1

    def _queue_save(self):
        self.saves += 1


@pytest.fixture
def review(qtbot, tmp_path, monkeypatch):
    # The original desktop resolves its series during import. Establish the
    # finite test binding first, independently of the user's registry.
    series = tmp_path / "configured_series"
    series.mkdir()
    (series / "series.toml").write_text(
        'schema_version = 1\nseries_id = "ffac"\n[paths]\n', encoding="utf8",
    )
    monkeypatch.setenv("DARSIA_FFAC_PROJECT_ROOT", str(series))
    monkeypatch.setenv("DARSIA_REVIEW_ALIAS_FILES", str(tmp_path / "absent-aliases.json"))
    from darsia.fluidflower.tools.calibration_review_desktop import _SavedReaderContextTask
    from darsia.fluidflower import native_phase_review_session as lifecycle
    from darsia.fluidflower.ui import phase_model_correction as correction

    window = OriginalReviewWindow(tmp_path)
    ws = window.workspace
    save_root = tmp_path / "reviews"
    reserved = lifecycle.reserve_review_session(save_root, series="ffac", reviewer="Reviewer", target_model_id="model-1")
    ws.review_session_id = reserved["review_session_id"]
    binding = correction.validate_workspace(ws, reviewer="Reviewer", reservation=reserved,
                                             model_root=window._workflow_model_root)
    ws.output_dir.mkdir(parents=True)
    (ws.output_dir / "review_source_binding.json").write_text(json.dumps(binding), encoding="utf8")
    manifest = dict(format=lifecycle.SESSION_FORMAT, workspace="training", review_surface="blind_regions",
        review_session_id=ws.review_session_id, target_model_id=ws.target_model_id,
        reviewer="Reviewer", reviewer_id="reviewer", decision_dir=str(ws.output_dir),
        include_in_training=True, editing_locked=False, blindness_status="blind",
        combined_root=str(ws.combined_root), ffac_root=str(ws.ffac_root), ffum_root=str(ws.ffum_root),
        selected_candidates=ws.selected_candidates, ffac_runs=None, ffum_runs=None)
    session_path = ws.output_dir / "review_session.json"
    session_path.write_text(json.dumps(manifest), encoding="utf8")
    lifecycle.attach_review_manifest(save_root, series="ffac", reviewer="Reviewer",
        review_session_id=ws.review_session_id, session_path=session_path)
    window._workflow_request_context = dict(save_root=save_root, series="ffac", reviewer="Reviewer",
        model_root=window._workflow_model_root, binding=binding, workspace=deepcopy(vars(ws)))
    qtbot.addWidget(window)
    desktop = NS(CalibrationReviewWindow=OriginalReviewWindow,
                 _SavedReaderContextTask=_SavedReaderContextTask,
                 review=NS(_candidate_prediction_withheld=lambda *args: False),
                 _quiz_generation_campaign=lambda candidate: tmp_path)
    return window, desktop


def install_fake_source(monkeypatch):
    monkeypatch.setattr(proposal_ui, "prepare_saved_categorical_source", lambda *a, **k: {"bound": "final"})
    # This adapter fixture has no scientific product. Reader identity is covered
    # by its own source fixtures; reject/ordering behavior is exercised below.
    monkeypatch.setattr(proposal_ui.saved_categories, "validate_current_identity", lambda source: None)


def result_for(existing):
    return dict(payload={"source": {"bound": "final"},
                         "proposals": [*deepcopy(existing), {"proposal_id": "new", "candidate": "A"}]},
                message="Added 1 box.")


def test_finite_worker_captures_saved_reader_context_and_keeps_live_answers(review, qtbot, monkeypatch):
    window, desktop = review
    install_fake_source(monkeypatch)
    context = ContextVar("requested-box-context", default="missing")
    entered, release = Event(), Event()
    seen = []
    def generate(source, output, **kwargs):
        seen.append((context.get(), kwargs["requested_count"], kwargs["actor"], kwargs["initial_proposal_count"]))
        entered.set()
        assert release.wait(5)
        return result_for(kwargs["existing_proposals"])
    monkeypatch.setattr(proposal_ui, "generate_more", generate)
    adapter = proposal_ui.install(window, desktop)
    assert proposal_ui.install(window, desktop) is adapter
    token = context.set("source-bound-reader-context")
    try:
        adapter.request()
    finally:
        context.reset(token)
    try:
        qtbot.waitUntil(entered.is_set)
        task = adapter.task
        assert not adapter.can_close()
        assert not task.cancelled.is_set()  # Busy inspection is not cancellation.
        window.answers["old"]["phase_class"] = "water"
        window.annotations.append({"id": "manual-during-request", "phase_class": "aqueous"})
        live = (window.selected_ids, window.answers, window.annotations, window.rounds)
    finally:
        release.set()
        qtbot.waitUntil(adapter.can_close)
    assert seen == [("source-bound-reader-context", 2, "Reviewer", 1)]
    assert [row["proposal_id"] for row in window.pool] == ["old", "new"]
    assert all(a is b for a, b in zip(live, (window.selected_ids, window.answers, window.annotations, window.rounds)))
    assert window.answers["old"]["phase_class"] == "water" and window.quiz_index == 0
    assert len(window.annotations) == 2 and window.saves == window.refreshed == 1


def test_emitted_result_is_not_retired_before_runnable_returns_and_close_avoids_late_save(review, qtbot, monkeypatch):
    window, desktop = review
    install_fake_source(monkeypatch)
    after_signal, release = Event(), Event()
    class HoldingContextTask(desktop._SavedReaderContextTask):
        def run(self):
            super().run()
            after_signal.set()
            assert release.wait(5)
    desktop._SavedReaderContextTask = HoldingContextTask
    monkeypatch.setattr(proposal_ui, "generate_more", lambda *a, **k: result_for(k["existing_proposals"]))
    adapter = proposal_ui.install(window, desktop)
    before = deepcopy((window.pool, window.answers, window.annotations))
    adapter.request()
    try:
        qtbot.waitUntil(lambda: after_signal.is_set() and adapter.completion is not None)
        assert adapter.pool.activeThreadCount() == 1 and adapter.task is not None
        assert not adapter.can_close()
        adapter.begin_close()
        assert adapter.task.cancelled.is_set() and not adapter.button.isEnabled()
    finally:
        release.set()
        qtbot.waitUntil(adapter.can_close)
    assert (window.pool, window.answers, window.annotations) == before
    assert window.saves == window.refreshed == 0
    adapter.cancel_close()
    assert adapter.button.isEnabled() and not adapter.close_requested


def test_result_for_previous_experiment_does_not_touch_current_review(review, qtbot, monkeypatch):
    window, desktop = review
    install_fake_source(monkeypatch)
    entered, release = Event(), Event()
    def generate(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return result_for(kwargs["existing_proposals"])
    monkeypatch.setattr(proposal_ui, "generate_more", generate)
    adapter = proposal_ui.install(window, desktop)
    adapter.request()
    try:
        qtbot.waitUntil(entered.is_set)
        window.current_pair_object.candidate_a.run = "ac02"
        window.pool = [{"proposal_id": "other-experiment", "candidate": "A"}]
    finally:
        release.set()
        qtbot.waitUntil(adapter.can_close)
    assert window.pool == [{"proposal_id": "other-experiment", "candidate": "A"}]
    assert window.saves == 0 and "current experiment was not changed" in adapter.status.text()


def test_final_map_action_passes_exact_model_binding_and_never_uses_mass_fallback(review, qtbot, monkeypatch):
    window, desktop = review
    window.current_pair_object.candidate_a.quality_flags["blind_review_mass_result"] = {"path": "explicit-result"}
    calls = []
    def source(campaign, run, **kwargs):
        calls.append((campaign, run, kwargs))
        raise proposal_ui.ProposalSourceUnavailable("Exact FINAL categorical source is unavailable")
    monkeypatch.setattr(proposal_ui, "prepare_saved_categorical_source", source)
    monkeypatch.setattr(proposal_ui, "generate_more", lambda *a, **k: pytest.fail("Missing final labels reached publication"))
    adapter = proposal_ui.install(window, desktop)
    adapter.request()
    qtbot.waitUntil(adapter.can_close)
    assert len(calls) == 1
    campaign, run, arguments = calls[0]
    assert campaign == desktop._quiz_generation_campaign(window.current_pair_object.candidate_a)
    assert run == "ac01" and arguments["candidate_id"] == "candidate-1"
    assert arguments["params_sha256"] == "a" * 64 and arguments["model_id"] == "model-1"
    assert arguments["model_root"] == window._workflow_model_root
    assert window.saves == 0 and "Exact FINAL categorical source" in adapter.status.text()


@pytest.mark.parametrize("flag", ["_workflow_revealed", "_workflow_sealing"])
def test_blind_sealing_disables_new_box_action_and_rejects_direct_requests(review, monkeypatch, flag):
    window, desktop = review
    adapter = proposal_ui.install(window, desktop)
    setattr(window, flag, True)
    adapter.blind_sealed()
    assert not adapter.button.isEnabled()
    adapter.request()
    assert adapter.task is None and "sealed" in adapter.status.text()
    adapter.begin_close()
    adapter.cancel_close()
    assert not adapter.button.isEnabled()


def test_candidate_publication_hold_rejects_request_before_reading_any_fields(review, qtbot, monkeypatch):
    window, desktop = review
    desktop.review._candidate_prediction_withheld = lambda *args: True
    calls = []
    monkeypatch.setattr(proposal_ui, "prepare_saved_categorical_source", lambda *a, **k: calls.append("final"))
    monkeypatch.setattr(proposal_ui, "generate_more", lambda *a, **k: calls.append("generate"))
    adapter = proposal_ui.install(window, desktop)
    adapter.request()
    qtbot.waitUntil(adapter.can_close)
    assert calls == [] and window.saves == 0
    assert "publication policy" in adapter.status.text()


def guarded_publication(output, arguments):
    """Exercise the adapter's real token and file validator at the commit seam."""
    arguments["validate_request"]()
    with arguments["publication_guard"].critical_section():
        if arguments["cancelled"]():
            raise RuntimeError("Request was cancelled before publication")
        (Path(output) / "published-test-pool.json").write_text("published", encoding="utf8")
    return result_for(arguments["existing_proposals"])


@pytest.mark.parametrize("transition", ["same_run_reload", "away_and_back", "seal", "close"])
def test_invalidated_request_cannot_publish_or_apply_after_owner_transition(review, qtbot, monkeypatch, transition):
    window, desktop = review
    install_fake_source(monkeypatch)
    entered, release = Event(), Event()

    def generate(_source, output, **arguments):
        entered.set()
        assert release.wait(5)
        return guarded_publication(output, arguments)

    monkeypatch.setattr(proposal_ui, "generate_more", generate)
    adapter = proposal_ui.install(window, desktop)
    before = deepcopy((window.pool, window.selected_ids, window.answers, window.annotations, window.rounds))
    adapter.request()
    try:
        qtbot.waitUntil(entered.is_set)
        task = adapter.task
        original_key = task.key
        if transition in {"same_run_reload", "away_and_back"}:
            # The shared-window tests exercise the original _load_run hook;
            # here its adapter contract is driven while publication is pending.
            adapter.invalidate_request()
            window._workflow_selection_epoch += 1
            if transition == "away_and_back":
                window.current_pair_object.candidate_a.run = "ac02"
                adapter.invalidate_request()
                window._workflow_selection_epoch += 1
                window.current_pair_object.candidate_a.run = "ac01"
            assert adapter._key() != original_key
        elif transition == "seal":
            adapter.invalidate_request()
            window._workflow_sealing = True
            adapter.blind_sealed()
        else:
            adapter.begin_close()
            adapter.cancel_close()  # Reopening must not revive the cancelled token.
        assert task.publication_guard.is_cancelled() and task.cancelled.is_set()
    finally:
        release.set()
        qtbot.waitUntil(adapter.can_close)
    assert not (window.workspace.output_dir / "published-test-pool.json").exists()
    assert (window.pool, window.selected_ids, window.answers, window.annotations, window.rounds) == before
    assert window.saves == window.refreshed == 0


@pytest.mark.parametrize("change", ["editing_locked", "discarded", "foreign_owner", "source_root",
    "model_root", "reveal_marker", "unattached", "retired_registry"])
def test_prepublication_validator_rechecks_original_files_without_widget_access(review, qtbot, monkeypatch, change):
    from darsia.fluidflower import native_phase_review_session as lifecycle

    window, desktop = review
    install_fake_source(monkeypatch)
    entered, release = Event(), Event()
    observed = []

    def generate(_source, output, **arguments):
        observed.append(arguments["validate_request"])
        entered.set()
        assert release.wait(5)
        return guarded_publication(output, arguments)

    monkeypatch.setattr(proposal_ui, "generate_more", generate)
    adapter = proposal_ui.install(window, desktop)
    before = deepcopy((window.pool, window.answers, window.annotations))
    adapter.request()
    try:
        qtbot.waitUntil(entered.is_set)
        root = window.workspace.output_dir
        path = root / "review_session.json"
        if change == "reveal_marker":
            # Even an incomplete marker is a refusal, never permission to append.
            (root / "blind_reveal.json").write_text("{}", encoding="utf8")
        else:
            if change == "model_root":
                path = root / "review_source_binding.json"
            elif change in {"unattached", "retired_registry"}:
                context = window._workflow_request_context
                path = lifecycle.review_session_registry_path(context["save_root"],
                    series=context["series"], reviewer=context["reviewer"])
            value = json.loads(path.read_text(encoding="utf8"))
            if change == "editing_locked":
                value["editing_locked"] = True
            elif change == "discarded":
                value["include_in_training"] = False
            elif change == "foreign_owner":
                value.update(reviewer="Another reviewer", reviewer_id="another-reviewer")
            elif change == "source_root":
                value["ffac_root"] = str(root / "other-source")
            elif change == "model_root":
                value["model_root"] = str(root / "same-id-other-model")
            elif change == "unattached":
                value["sessions"][0]["manifest_paths"] = []
            else:
                value["sessions"][0]["status"] = "discarded"
            path.write_text(json.dumps(value), encoding="utf8")
        assert not adapter.task.publication_guard.is_cancelled()  # The file guard must catch this.
    finally:
        release.set()
        qtbot.waitUntil(adapter.can_close)
    assert len(observed) == 1
    assert not (window.workspace.output_dir / "published-test-pool.json").exists()
    assert (window.pool, window.answers, window.annotations) == before
    assert window.saves == window.refreshed == 0
    assert "Could not generate" in adapter.status.text()


def test_callbacks_from_another_task_with_equal_key_are_ignored(review, qtbot, monkeypatch):
    window, desktop = review
    install_fake_source(monkeypatch)
    entered, release = Event(), Event()

    def generate(_source, _output, **arguments):
        entered.set()
        assert release.wait(5)
        return result_for(arguments["existing_proposals"])

    monkeypatch.setattr(proposal_ui, "generate_more", generate)
    adapter = proposal_ui.install(window, desktop)
    adapter.request()
    try:
        qtbot.waitUntil(entered.is_set)
        original_task = adapter.task
        stale = NS(key=original_task.key)
        status = adapter.status.text()
        adapter._progress(stale, "Stale message")
        adapter._finished(stale, {"payload": {"proposals": []}}, "Stale failure")
        assert adapter.task is original_task and adapter.completion is None
        assert adapter.status.text() == status
    finally:
        release.set()
        qtbot.waitUntil(adapter.can_close)
    assert window.saves == 1 and [row["proposal_id"] for row in window.pool] == ["old", "new"]


@pytest.mark.parametrize("missing", ["_workflow_request_context", "_workflow_model_root"])
def test_unbound_window_refuses_before_starting_worker(review, monkeypatch, missing):
    window, desktop = review
    delattr(window, missing)
    monkeypatch.setattr(proposal_ui, "_task_type", lambda *args: pytest.fail("Unbound request started a worker"))
    adapter = proposal_ui.install(window, desktop)
    adapter.request()
    assert adapter.task is None and "Could not request" in adapter.status.text()


@pytest.mark.parametrize("change", ["readonly", "workspace_source", "workspace_candidates", "workspace_flags",
    "manifest_lock", "reveal_marker", "registration_identity"])
def test_retired_result_revalidates_before_attaching_and_preserves_published_pool(review, qtbot, monkeypatch, change):
    window, desktop = review
    install_fake_source(monkeypatch)
    after_signal, release = Event(), Event()
    identity_calls = []

    class HoldingContextTask(desktop._SavedReaderContextTask):
        def run(self):
            super().run()
            after_signal.set()
            assert release.wait(5)

    desktop._SavedReaderContextTask = HoldingContextTask
    monkeypatch.setattr(proposal_ui, "generate_more", lambda _source, output, **arguments:
                        guarded_publication(output, arguments))

    def current_identity(source):
        identity_calls.append(source)
        if change == "registration_identity":
            raise ValueError("The selected model's current registration changed")

    monkeypatch.setattr(proposal_ui.saved_categories, "validate_current_identity", current_identity)
    adapter = proposal_ui.install(window, desktop)
    before = deepcopy((window.pool, window.answers, window.annotations, window.selected_ids, window.rounds))
    adapter.request()
    try:
        qtbot.waitUntil(lambda: after_signal.is_set() and adapter.completion is not None)
        published = window.workspace.output_dir / "published-test-pool.json"
        published_bytes = published.read_bytes()
        assert window.saves == 0 and adapter.pool.activeThreadCount() == 1
        assert identity_calls == []  # Recheck occurs only after actual runnable retirement.
        if change == "readonly":
            window.store.is_read_only = True
        elif change == "workspace_source":
            window.workspace.ffac_root = window.workspace.output_dir / "other-source"
        elif change == "workspace_candidates":
            window.workspace.selected_candidates["ac02"] = "other-candidate"
        elif change == "workspace_flags":
            window.workspace.single_model = False
        elif change == "manifest_lock":
            path = window.workspace.output_dir / "review_session.json"
            value = json.loads(path.read_text(encoding="utf8"))
            value["editing_locked"] = True
            path.write_text(json.dumps(value), encoding="utf8")
        elif change == "reveal_marker":
            (window.workspace.output_dir / "blind_reveal.json").write_text("{}", encoding="utf8")
    finally:
        release.set()
        qtbot.waitUntil(adapter.can_close)
    assert (window.pool, window.answers, window.annotations, window.selected_ids, window.rounds) == before
    assert window.saves == window.refreshed == 0
    assert published.read_bytes() == published_bytes
    assert "live review was preserved" in adapter.status.text()
    assert identity_calls == ([{"bound": "final"}] if change == "registration_identity" else [])


def test_attachment_checks_current_identity_after_actual_retirement_before_refresh(review, qtbot, monkeypatch):
    window, desktop = review
    install_fake_source(monkeypatch)
    checks = []

    def current_identity(source):
        adapter = window.requested_phase_boxes
        assert adapter.pool.activeThreadCount() == 0
        assert window.pool == [{"proposal_id": "old", "candidate": "A"}]
        assert window.saves == window.refreshed == 0
        checks.append(source)

    monkeypatch.setattr(proposal_ui.saved_categories, "validate_current_identity", current_identity)
    monkeypatch.setattr(proposal_ui, "generate_more", lambda *args, **arguments:
                        result_for(arguments["existing_proposals"]))
    adapter = proposal_ui.install(window, desktop)
    adapter.request()
    qtbot.waitUntil(adapter.can_close)
    assert checks == [{"bound": "final"}]
    assert window.saves == window.refreshed == 1
