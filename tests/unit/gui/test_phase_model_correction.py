"""Shared correction-window guards and graceful close, without science/jobs."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from PySide6.QtWidgets import (
    QComboBox, QLabel, QLineEdit, QMainWindow, QPushButton, QSpinBox, QTabWidget, QWidget,
)
from PySide6.QtGui import QCloseEvent

from darsia.fluidflower import native_phase_review_session as lifecycle
from darsia.fluidflower.ui import phase_model_correction as correction


def workspace(tmp_path):
    return NS(key="training", single_model=True, review_surface="blind_regions",
              review_session_id="review-current", target_model_id="saved-model",
              output_dir=tmp_path / "original-store", combined_root=tmp_path / "sources" / "combined",
              ffac_root=tmp_path / "sources" / "ffac", ffum_root=tmp_path / "sources" / "ffum",
              selected_candidates={"ac28": "saved-model-ac28"}, ffac_runs=None, ffum_runs=None)


def original_manifest(ws, **changes):
    value = dict(format=lifecycle.SESSION_FORMAT, review_session_id=ws.review_session_id,
                 target_model_id=ws.target_model_id, reviewer="Reviewer", reviewer_id="reviewer",
                 decision_dir=str(ws.output_dir), include_in_training=True, editing_locked=False,
                 workspace="training", review_surface="blind_regions",
                 combined_root=str(ws.combined_root), ffac_root=str(ws.ffac_root), ffum_root=str(ws.ffum_root),
                 selected_candidates=deepcopy(ws.selected_candidates), ffac_runs=ws.ffac_runs, ffum_runs=ws.ffum_runs)
    value.update(changes)
    ws.output_dir.mkdir(parents=True, exist_ok=True)
    path = ws.output_dir / "review_session.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def reservation(ws, **changes):
    value = dict(review_session_id=ws.review_session_id, target_model_id=ws.target_model_id,
                 status="active", manifest_paths=[])
    value.update(changes)
    return value


@pytest.mark.parametrize("changes", [
    {"editing_locked": True}, {"include_in_training": False}, {"retired_workflow": True},
    {"independent_holdout": True}, {"evidence_role": "blind_transfer_validation"},
    {"evidence_role": "independent_blind_gas_validation"}, {"format": "unrecognized"},
    {"reviewer": "Another Person", "reviewer_id": "another-person"},
    {"review_session_id": "other-session"}, {"target_model_id": "other-model"}])
def test_original_locked_foreign_or_heldout_store_rejected_without_any_write(tmp_path, changes):
    ws = workspace(tmp_path)
    path = original_manifest(ws, **changes)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        correction.validate_workspace(ws, reviewer="Reviewer", reservation=reservation(ws))
    assert path.read_bytes() == before


@pytest.mark.parametrize("changes", [
    {"status": "submitted"}, {"status": "discarding"}, {"status": "discarded"},
    {"review_session_id": "another-session"}, {"target_model_id": "another-model"}])
def test_closed_or_rebound_reservation_is_rejected_before_new_original_store(tmp_path, changes):
    ws = workspace(tmp_path)
    with pytest.raises(ValueError):
        correction.validate_workspace(ws, reviewer="Reviewer", reservation=reservation(ws, **changes))
    assert not ws.output_dir.exists()


def test_current_original_store_is_reused_without_rewriting_role_or_boxes(tmp_path):
    ws = workspace(tmp_path)
    path = original_manifest(ws, workspace="training", blindness_status="blind")
    boxes = ws.output_dir / "pairwise_review_decisions.json"
    boxes.write_text('{"decisions":{"ac28":{"annotations":[{"id":"label"}],"phase_quiz_answers":[{"id":"blind"}]}}}', encoding="utf-8")
    before = {p: p.read_bytes() for p in (path, boxes)}
    correction.validate_workspace(ws, reviewer="Reviewer", reservation=reservation(ws))
    assert all(p.read_bytes() == content for p, content in before.items())


def child_desktop(ws, root, calls):
    class App:
        @staticmethod
        def setAttribute(*args):
            pass

        def __init__(self, *args):
            pass

        def setApplicationName(self, *args):
            pass

        def setStyle(self, *args):
            pass

    class Surface:
        RenderableType = NS(OpenGL=1)
        OpenGLContextProfile = NS(CoreProfile=1)

        def setRenderableType(self, *args):
            pass

        def setVersion(self, *args):
            pass

        def setProfile(self, *args):
            pass

        def setSamples(self, *args):
            pass

        @staticmethod
        def setDefaultFormat(*args):
            pass

    def original_builder(*args, **kwargs):
        calls.append("original-writer")
        pytest.fail("An invalid store reached the original manifest writer")

    def atomic_json(path, payload):
        calls.append("atomic-binding")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    return NS(effective_reviewer_settings=lambda args: ("Reviewer", root), QApplication=App,
        Qt=NS(ApplicationAttribute=NS(AA_ShareOpenGLContexts=1)), QSurfaceFormat=Surface,
        sys=NS(argv=["review"]), local_reviewer_workspace=lambda ws, *_: ws,
        review_workspaces=lambda args: {"training": ws},
        ensure_writable_directory=lambda path: calls.append("ensure-writable"),
        atomic_json=atomic_json, build_workspace_model=original_builder)


@pytest.mark.parametrize("locked,registered", [(True, True), (False, False)])
def test_child_refuses_locked_or_unreserved_store_before_original_model_builder(tmp_path, monkeypatch, locked, registered):
    monkeypatch.setenv("DARSIA_REVIEW_ALIAS_FILES", str(tmp_path / "absent-aliases.json"))
    root = tmp_path / "reviews"
    ws = workspace(root)
    if registered:
        row = lifecycle.reserve_review_session(root, series="ffac", reviewer="Reviewer", target_model_id="saved-model")
        ws.review_session_id = row["review_session_id"]
    path = original_manifest(ws, editing_locked=locked)
    calls = []
    desktop = child_desktop(ws, root, calls)
    args = NS(workspace="training", phase_workflow_series="ffac", evidence_audit=None,
              review_current_model_root=tmp_path / "saved-model")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        correction.run_shared_main(desktop, args)
    assert calls == [] and path.read_bytes() == before


def source_receipt(ws, model_root):
    value = correction.validate_workspace(ws, reviewer="Reviewer", reservation=reservation(ws), model_root=model_root)
    ws.output_dir.mkdir(parents=True, exist_ok=True)
    path = ws.output_dir / "review_source_binding.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize("changed", ["combined_root", "ffac_root", "ffum_root", "candidate_replaced",
    "candidate_added", "candidate_removed", "ffac_runs", "ffum_runs", "model_root", "missing_model_root", "missing_receipt"])
def test_child_rejects_source_rebind_before_any_original_write(tmp_path, monkeypatch, changed):
    monkeypatch.setenv("DARSIA_REVIEW_ALIAS_FILES", str(tmp_path / "absent-aliases.json"))
    root = tmp_path / "reviews"
    ws = workspace(root)
    row = lifecycle.reserve_review_session(root, series="ffac", reviewer="Reviewer", target_model_id="saved-model")
    ws.review_session_id = row["review_session_id"]
    model_root = tmp_path / "saved-model"
    receipt = None if changed == "missing_receipt" else source_receipt(ws, model_root)
    path = original_manifest(ws)
    decisions = ws.output_dir / "pairwise_review_decisions.json"
    decisions.write_text('{"annotations":[{"id":"retained-box"}]}', encoding="utf-8")
    retained = {p: p.read_bytes() for p in (path, decisions, *([receipt] if receipt else []))}
    if changed in {"combined_root", "ffac_root", "ffum_root"}:
        setattr(ws, changed, tmp_path / ("other-" + changed))
    elif changed == "candidate_replaced":
        ws.selected_candidates["ac28"] = "another-candidate"
    elif changed == "candidate_added":
        ws.selected_candidates["ac29"] = "saved-model-ac29"
    elif changed == "candidate_removed":
        ws.selected_candidates.clear()
    elif changed in {"ffac_runs", "ffum_runs"}:
        setattr(ws, changed, ("ac28",) if changed == "ffac_runs" else ())
    elif changed == "model_root":
        model_root = tmp_path / "same-id-other-saved-model"
    elif changed == "missing_model_root":
        model_root = None
    calls = []
    args = NS(workspace="training", phase_workflow_series="ffac", evidence_audit=None,
              review_current_model_root=model_root)
    with pytest.raises(ValueError, match="binding|source folders"):
        correction.run_shared_main(child_desktop(ws, root, calls), args)
    assert calls == []
    assert all(p.read_bytes() == content for p, content in retained.items())
    assert receipt is not None or not (ws.output_dir / "review_source_binding.json").exists()


def test_first_builder_failure_retains_binding_and_same_binding_retry_does_not_rewrite_it(tmp_path, monkeypatch):
    monkeypatch.setenv("DARSIA_REVIEW_ALIAS_FILES", str(tmp_path / "absent-aliases.json"))
    root = tmp_path / "reviews"
    ws = workspace(root)
    row = lifecycle.reserve_review_session(root, series="ffac", reviewer="Reviewer", target_model_id="saved-model")
    ws.review_session_id = row["review_session_id"]
    args = NS(workspace="training", phase_workflow_series="ffac", evidence_audit=None,
              review_current_model_root=tmp_path / "saved-model")
    calls = []
    desktop = child_desktop(ws, root, calls)
    def unavailable(*args, **kwargs):
        calls.append("original-builder")
        raise OSError("Original source temporarily unavailable")
    desktop.build_workspace_model = unavailable
    receipt = ws.output_dir / "review_source_binding.json"
    for attempt in range(2):
        with pytest.raises(OSError, match="temporarily unavailable"):
            correction.run_shared_main(desktop, args)
        if attempt == 0:
            before = receipt.read_bytes()
        else:
            assert receipt.read_bytes() == before
    assert calls == ["ensure-writable", "atomic-binding", "original-builder", "ensure-writable", "original-builder"]
    assert not (ws.output_dir / "review_session.json").exists()


def test_alias_equivalent_original_roots_and_model_root_keep_immutable_binding(tmp_path, monkeypatch):
    aliases = tmp_path / "aliases.json"
    monkeypatch.setenv("DARSIA_REVIEW_ALIAS_FILES", str(aliases))
    old, current = tmp_path / "old", tmp_path / "current"
    ws = workspace(old)
    receipt = source_receipt(ws, old / "model")
    path = original_manifest(ws)
    retained = {p: p.read_bytes() for p in (path, receipt)}
    aliases.write_text(json.dumps({"format": "fluidflower-review-storage-aliases-v1", "aliases": [dict(
        source=str(old / "sources"), destination=str(current / "sources"),
        original_save_root=str(tmp_path / "reviews"), save_root=str(tmp_path / "reviews")), dict(
        source=str(old / "model"), destination=str(current / "model"),
        original_save_root=str(tmp_path / "reviews"), save_root=str(tmp_path / "reviews"))]}), encoding="utf-8")
    from darsia.fluidflower import review_storage
    review_storage.review_aliases(refresh=True)
    for key in ("combined_root", "ffac_root", "ffum_root"):
        setattr(ws, key, current / "sources" / Path(getattr(ws, key)).name)
    correction.validate_workspace(ws, reviewer="Reviewer", reservation=reservation(ws), model_root=current / "model")
    assert all(p.read_bytes() == content for p, content in retained.items())


def test_configured_storage_mapping_resolves_before_host_absolute_check(tmp_path, monkeypatch):
    from darsia.fluidflower import storage_paths
    original = storage_paths.resolve_storage_path
    legacy = r"Z:\retained-model"
    mapped = tmp_path / "saved-model"
    monkeypatch.setattr(storage_paths, "resolve_storage_path",
        lambda value: mapped if str(value) == legacy else original(value))
    assert correction._bound_path(legacy, context=tmp_path / "review_session.json") == correction._bound_path(
        mapped, context=tmp_path / "review_session.json")
    with pytest.raises(ValueError, match="absolute"):
        correction._bound_path("unbound-relative-model", context=tmp_path / "review_session.json")


def test_new_native_review_requires_explicit_candidate_selection(tmp_path):
    ws = workspace(tmp_path)
    ws.selected_candidates = {}
    with pytest.raises(ValueError, match="selected-candidate binding"):
        correction.validate_workspace(ws, reviewer="Reviewer", reservation=reservation(ws), model_root=tmp_path / "saved-model")
    assert not ws.output_dir.exists()


class Pool:
    def __init__(self):
        self.count = 0

    def activeThreadCount(self):
        return self.count


class OriginalWindow(QMainWindow):
    """The original desktop contract; workers are inert counters in this test."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.setCentralWidget(QWidget())
        self.reviewer = QLineEdit("Reviewer", self.centralWidget())
        self.quiz_title = QLabel(self.centralWidget())
        self.quiz_tabs = QTabWidget(self.centralWidget())
        self.manual_evidence_panel = QWidget(self.centralWidget())
        self.phase_buttons = {phase: QPushButton(phase, self.centralWidget())
                              for phase in ("water", "aqueous", "gas")}
        self.ignore_region = QPushButton("Ignore region", self.centralWidget())
        self.clear_answer = QPushButton("Clear answer", self.centralWidget())
        self.confidence = QComboBox(self.centralWidget())
        self.confidence.addItems(["Low", "Medium", "High"])
        self.answer_note = QLineEdit(self.centralWidget())
        self.new_round = QPushButton("Add regions", self.centralWidget())
        self.new_round_count = QSpinBox(self.centralWidget())
        self.quiz_entry_help = QLabel(self.centralWidget())
        self.answers = [{"proposal_id": "saved-box", "human_choice": "water"}]
        self.annotations = [{"id": "drawn-label", "phase": "gas"}]
        self.selected_ids = ["saved-box"]
        self.lifecycle_events = []
        self.answer_actions = []
        self.save_pool, self.quiz_preparation_pool = Pool(), Pool()
        self.cache_actions = []
        self.image_cache = NS(pool=Pool(), reference_pool=Pool(), display_pool=Pool(), persistence_pool=Pool(),
            running=set(), reference_running=set(), display_running="", persistence_running=None,
            queued={}, display_queued={}, persistence_queued=[],
            cancel_queued_reference_preparations=lambda: self.cache_actions.append("cancel-reference"),
            focus=lambda keys: self.cache_actions.append(("focus", keys)),
            prefetch_window=lambda sources: self.cache_actions.append(("prefetch", sources)))
        self.save_inflight = False
        self.pending_save_requests = {}
        self.save_tasks = {}
        self.quiz_preparation_task = None
        self.quiz_preparation_closed = False
        self.sync_error_pending = False
        self.store = NS(is_read_only=False)
        self.current_run = "ac28"
        self.saved_calls = 0
        self.original_close_called = False

    def _save_now(self):
        self.saved_calls += 1
        self.save_inflight = True

    def _load_run(self, run):
        if not run:
            return
        self.lifecycle_events.append(("original-load", run, self._workflow_selection_epoch))
        self.current_run = run.casefold()
        return "original-loaded"

    def _save_finished(self, generation, run, record, error):
        self.lifecycle_events.append(("original-save", generation, run, record, error))
        self.save_inflight = False
        self.sync_error_pending = bool(error)

    def _render_vectors(self):
        self.lifecycle_events.append("original-render")

    def _refresh_quiz(self):
        self.lifecycle_events.append("original-refresh-quiz")
        # Original refresh enables valid choices again. The shared subclass
        # must enforce the seal after every refresh, including role changes.
        for control in (*self.phase_buttons.values(), self.ignore_region, self.clear_answer,
                        self.confidence, self.answer_note, self.new_round, self.new_round_count):
            control.setEnabled(True)

    def _answer(self, *args):
        self.answer_actions.append(("answer", args))
        self.answers.append({"human_choice": args[0]})
        return "original-answer"

    def _update_existing_answer(self):
        self.answer_actions.append("update-answer")
        self.answers[0]["comment"] = self.answer_note.text()
        return "original-update"

    def _clear_answer(self):
        self.answer_actions.append("clear-answer")
        self.answers.clear()
        return "original-clear"

    def _start_new_round(self):
        self.answer_actions.append("new-round")
        self.selected_ids.append("next-box")
        return "original-round"

    def closeEvent(self, event):
        self.original_close_called = True
        event.accept()


@pytest.fixture
def window(qtbot):
    result = correction.create_window(NS(CalibrationReviewWindow=OriginalWindow), role="label")
    cleaned = False

    def cleanup(owned):
        nonlocal cleaned
        if cleaned:
            return
        # pytest-qt closes widgets before yield-fixture teardown. Retire the
        # inert test state before that close can schedule a synthetic save.
        owned._workflow_close_timer.stop()
        owned.store.is_read_only = True
        owned.current_run = ""
        owned.save_inflight = False
        owned.sync_error_pending = False
        owned._workflow_adapters = []
        for pool in (owned.save_pool, owned.quiz_preparation_pool, owned.image_cache.pool,
                     owned.image_cache.reference_pool, owned.image_cache.display_pool, owned.image_cache.persistence_pool):
            pool.count = 0
        cleaned = True

    qtbot.addWidget(result, before_close_func=cleanup)
    yield result
    cleanup(result)


def test_shared_role_switch_preserves_original_store_and_review_controls(window):
    store, manual, quiz = window.store, window.manual_evidence_panel, window.quiz_tabs
    window.workflow_role.setCurrentIndex(1)
    assert window.workflow_role.currentData() == "blind"
    window.workflow_role.setCurrentIndex(0)
    assert window.store is store and window.manual_evidence_panel is manual and window.quiz_tabs is quiz
    assert window.reviewer.isReadOnly()
    assert window.saved_calls == 0


def test_close_waits_for_original_save_and_image_workers_without_second_synchronous_save(window):
    window.image_cache.persistence_pool.count = 1
    first = QCloseEvent()
    window.closeEvent(first)
    assert not first.isAccepted() and window.saved_calls == 1
    assert window.cache_actions == ["cancel-reference", ("focus", ()), ("prefetch", ())]
    assert not window.centralWidget().isEnabled()
    window._save_finished(1, "ac28", {}, "")
    second = QCloseEvent()
    window.closeEvent(second)
    assert not second.isAccepted() and window.saved_calls == 1
    window.image_cache.persistence_pool.count = 0
    last = QCloseEvent()
    window.closeEvent(last)
    assert last.isAccepted() and window.saved_calls == 1
    assert not window.original_close_called


def test_failed_final_save_keeps_window_editable_and_session_open(window):
    adapter_actions = []
    window._workflow_adapters = [NS(begin_close=lambda: adapter_actions.append("closing"),
        can_close=lambda: True, cancel_close=lambda: adapter_actions.append("reopened"))]
    event = QCloseEvent()
    window.closeEvent(event)
    window._save_finished(1, "ac28", {}, "Storage temporarily unavailable")
    assert not window._workflow_closing
    assert not window._workflow_close_saved
    assert window.centralWidget().isEnabled() and window.workflow_role.isEnabled()
    assert not window.quiz_preparation_closed
    assert "final save failed" in window.statusBar().currentMessage()
    assert adapter_actions == ["closing", "reopened"]
    assert not window.original_close_called


def quiz_mutation_controls(window):
    return [*window.phase_buttons.values(), window.ignore_region, window.clear_answer,
            window.confidence, window.answer_note, window.new_round, window.new_round_count]


def test_sealed_blind_controls_stay_disabled_across_roles_and_original_refresh(window):
    marker = {"blindness_status": "revealed", "frozen_blind_snapshot_sha256": "verified-by-adapter"}
    before = deepcopy((window.answers, window.annotations, window.selected_ids, marker))
    window._workflow_blind_sealed(marker)
    assert window._workflow_revealed
    assert all(not control.isEnabled() for control in quiz_mutation_controls(window))
    assert "sealed" in window.workflow_role.itemText(1)
    assert "sealed" in window.workflow_notice.text()
    for role in (1, 0, 1, 0):
        window.workflow_role.setCurrentIndex(role)
        window._refresh_quiz()
        assert window._workflow_revealed
        assert all(not control.isEnabled() for control in quiz_mutation_controls(window))
    assert (window.answers, window.annotations, window.selected_ids, marker) == before
    assert window.manual_evidence_panel.isEnabled()
    assert window.reviewer.isReadOnly()
    assert "label drawing tools" in window.quiz_entry_help.text()
    assert not window.answer_actions


@pytest.mark.parametrize("method,args,expected", [
    ("_answer", ("gas",), "original-answer"),
    ("_update_existing_answer", (), "original-update"),
    ("_clear_answer", (), "original-clear"),
    ("_start_new_round", (), "original-round"),
])
def test_direct_quiz_mutations_delegate_before_seal_and_are_blocked_after(window, method, args, expected):
    action = getattr(window, method)
    assert action(*args) == expected
    assert len(window.answer_actions) == 1
    window._workflow_blind_sealed({"blindness_status": "revealed"})
    before = deepcopy((window.answers, window.annotations, window.selected_ids, window.answer_actions))
    assert action(*args) is None
    window.workflow_role.setCurrentIndex(1)
    window.workflow_role.setCurrentIndex(0)
    assert action(*args) is None
    assert (window.answers, window.annotations, window.selected_ids, window.answer_actions) == before


def test_failed_close_reopens_label_tools_but_keeps_blind_answers_sealed(window):
    window._workflow_blind_sealed({"blindness_status": "revealed"})
    before = deepcopy((window.answers, window.annotations, window.selected_ids))
    event = QCloseEvent()
    window.closeEvent(event)
    window._save_finished(1, "ac28", {}, "Review store temporarily unavailable")
    assert window.centralWidget().isEnabled() and window.manual_evidence_panel.isEnabled()
    assert window._workflow_revealed
    assert all(not control.isEnabled() for control in quiz_mutation_controls(window))
    window._answer("gas")
    window._clear_answer()
    assert (window.answers, window.annotations, window.selected_ids) == before


@pytest.mark.parametrize("error", ["", "Original save failed"])
def test_save_callbacks_receive_original_result_after_original_state_update(window, error):
    record = {"annotations": [{"id": "saved-label"}]}
    args = (17, "ac28", record, error)
    window.save_inflight = True

    def after_save(name, *received):
        assert window.save_inflight is False
        assert window.sync_error_pending is bool(error)
        assert received == args and received[2] is record
        window.lifecycle_events.append(name)

    window._workflow_adapters = [
        NS(save_finished=lambda *received: after_save("first-adapter", *received)),
        NS(),  # An unrelated adapter need not implement this callback.
        NS(save_finished=lambda *received: after_save("second-adapter", *received)),
    ]
    window._save_finished(*args)
    assert window.lifecycle_events == [("original-save", *args), "first-adapter", "second-adapter"]
    assert record == {"annotations": [{"id": "saved-label"}]}


def test_vector_callbacks_run_after_original_vectors_without_changing_evidence(window):
    before = deepcopy((window.answers, window.annotations, window.selected_ids))

    def render(name):
        assert window.lifecycle_events[0] == "original-render"
        window.lifecycle_events.append(name)

    window._workflow_adapters = [NS(render=lambda: render("first-overlay")), NS(),
                                 NS(render=lambda: render("second-overlay"))]
    window._render_vectors()
    assert window.lifecycle_events == ["original-render", "first-overlay", "second-overlay"]
    assert (window.answers, window.annotations, window.selected_ids) == before


def test_original_run_load_invalidates_before_mutation_and_advances_epoch_even_after_aba(window):
    from darsia.fluidflower.phase_review_proposal_generation import ProposalPublicationGuard

    guard = ProposalPublicationGuard()
    epoch = window._workflow_selection_epoch
    captured = (window.current_run, epoch)

    def invalidate():
        window.lifecycle_events.append(("invalidate", window.current_run, window._workflow_selection_epoch))
        guard.invalidate()

    window._workflow_adapters = [NS(invalidate_request=invalidate)]
    assert window._load_run("ac29") == "original-loaded"
    assert window._load_run("ac28") == "original-loaded"
    assert window._load_run("ac28") == "original-loaded"  # Reloading the same run is a new visit.
    assert guard.is_cancelled()
    assert window.lifecycle_events == [
        ("invalidate", "ac28", epoch), ("original-load", "ac29", epoch + 1),
        ("invalidate", "ac29", epoch + 1), ("original-load", "ac28", epoch + 2),
        ("invalidate", "ac28", epoch + 2), ("original-load", "ac28", epoch + 3),
    ]
    assert window.current_run == captured[0] and window._workflow_selection_epoch != captured[1]
    before = list(window.lifecycle_events)
    assert window._load_run("") is None
    assert window.lifecycle_events == before


@pytest.mark.parametrize("transition", ["sealing", "revealed", "sealed_callback"])
def test_shared_seal_transitions_revoke_publication_before_setting_state(window, transition):
    from darsia.fluidflower.phase_review_proposal_generation import ProposalPublicationGuard

    guard = ProposalPublicationGuard()
    seen = []

    def invalidate():
        seen.append((window._workflow_sealing, window._workflow_revealed))
        guard.invalidate()

    window._workflow_adapters = [NS(invalidate_request=invalidate)]
    if transition == "sealing":
        window._workflow_sealing = True  # Actual contradiction adapter assignment.
    elif transition == "revealed":
        window._workflow_revealed = True
    else:
        window._workflow_blind_sealed({"blindness_status": "revealed"})
    assert seen == [(False, False)] and guard.is_cancelled()
    window._workflow_sealing = False
    window.workflow_role.setCurrentIndex(1)
    window.workflow_role.setCurrentIndex(0)
    assert guard.is_cancelled()
