"""Lifecycle metadata tests. No training, scientific computation or image reads."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import pytest

from darsia.fluidflower import native_phase_review_session as lifecycle
from darsia.fluidflower.phase_workflow import lock_phase_review_sessions


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_record(store, path, *, derived=None):
    source = path.parent / "pairwise_review_decisions.json"
    captured = source.read_bytes()
    root = store / "frozen" / path.parents[3].name
    root.mkdir(parents=True)
    original = root / "original_decisions.json"
    original.write_bytes(captured)
    snapshot = root / "decisions.json"
    snapshot.write_bytes(captured if derived is None else json.dumps(derived).encode("utf-8"))
    return dict(source_path=str(source), snapshot_path=str(snapshot),
                sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(captured).hexdigest(),
                original_snapshot_path=str(original), original_snapshot_sha256=hashlib.sha256(captured).hexdigest(),
                derived_training_view=True)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DARSIA_REVIEW_ALIAS_FILES", str(tmp_path / "absent-aliases.json"))
    root = tmp_path / "reviews"
    root.mkdir()
    return root


def reserve(store, **kwargs):
    values = dict(series="ffac", reviewer="Reviewer", target_model_id="saved-model")
    values.update(kwargs)
    return lifecycle.reserve_review_session(store, **values)


def original(store, session_id, *, workspace="training", **metadata):
    root = store / "reviewer" / "campaign" / workspace / "sessions" / session_id / "manual_pairwise_review"
    value = dict(format=lifecycle.SESSION_FORMAT, reviewer="Reviewer", reviewer_id="reviewer",
                 review_session_id=session_id, target_model_id="saved-model", decision_dir=str(root),
                 workspace=workspace, combined_root=str(store / "combined"), editing_locked=False,
                 training_locks=[], selected_candidates={"ac28": "candidate"})
    value.update(metadata)
    path = root / "review_session.json"
    write(path, value)
    write(root / "pairwise_review_decisions.json", dict(decisions={"ac28": {
        "annotations": [{"id": "drawn-box", "phase": "gas", "box": [1, 2, 3, 4]}],
        "phase_quiz_answers": [{"id": "blind-box", "phase": "water"}]}}))
    write(root / "pairwise_review_history.json", {"events": [{"action": "labelled phase"}]})
    write(root / "blind_reveal.json", {"blindness_status": "revealed"})
    return path


def attach(store, row, path):
    return lifecycle.attach_review_manifest(store, series="ffac", reviewer="Reviewer",
        review_session_id=row["review_session_id"], session_path=path)


def discard(store, row):
    return lifecycle.discard_review_session(store, series="ffac", reviewer="Reviewer",
        review_session_id=row["review_session_id"], reason="Review began with the wrong model")


def test_one_session_survives_reopen_and_refuses_model_rebind(store):
    first = reserve(store)
    assert reserve(store)["review_session_id"] == first["review_session_id"]
    assert reserve(store, series="FF_AC", reviewer="reviewer")["review_session_id"] == first["review_session_id"]
    with pytest.raises(ValueError, match="Submit or discard"):
        reserve(store, target_model_id="another-model")
    assert reserve(store, reviewer="Reviewer Two")["review_session_id"] != first["review_session_id"]
    assert reserve(store, series="ffum")["review_session_id"] != first["review_session_id"]


def test_discard_preserves_all_evidence_and_closes_both_original_role_stores(store):
    row = reserve(store)
    paths = [original(store, row["review_session_id"], workspace=role) for role in ("training", "blind")]
    for path in paths:
        attach(store, row, path)
    evidence = {p: p.read_bytes() for path in paths for p in path.parent.iterdir() if p != path}
    result = discard(store, row)
    assert result["status"] == "discarded"
    assert result["history"] == [dict(action="discarded review session", person="Reviewer",
        reason="Review began with the wrong model", utc=result["history"][0]["utc"])]
    for path in paths:
        saved = read(path)
        assert saved["include_in_training"] is False
        assert saved["editing_locked"] is True
        assert saved["selected_candidates"] == {"ac28": "candidate"}
        assert saved["review_lifecycle_status"] == "discarded"
    assert all(path.read_bytes() == content for path, content in evidence.items())
    assert lifecycle.active_review_session(store, series="ffac", reviewer="Reviewer") is None
    assert discard(store, row) == result
    new = reserve(store)
    assert new["review_session_id"] != row["review_session_id"]
    registry = lifecycle.read_review_session_registry(store, series="ffac", reviewer="Reviewer")
    assert [item["status"] for item in registry["sessions"]] == ["discarded", "active"]


def test_failed_launch_reservation_requires_explicit_discard(store):
    first = reserve(store)
    assert first["manifest_paths"] == []
    assert reserve(store) == first
    discard(store, first)
    assert reserve(store)["review_session_id"] != first["review_session_id"]


def test_attachment_refuses_wrong_identity_owner_model_and_external_path(store, tmp_path):
    row = reserve(store)
    cases = [original(store, "another-session"),
             original(store, row["review_session_id"], workspace="other-person", reviewer="Someone Else"),
             original(store, row["review_session_id"], workspace="other-model", target_model_id="other-model"),
             original(tmp_path / "another-store", row["review_session_id"])]
    for path in cases:
        with pytest.raises(ValueError):
            attach(store, row, path)
    assert lifecycle.active_review_session(store, series="ffac", reviewer="Reviewer")["manifest_paths"] == []


@pytest.mark.parametrize("metadata", [{"independent_holdout": True},
    {"evidence_role": "blind_transfer_validation"},
    {"evidence_role": "independent_blind_gas_validation"}, {"include_in_training": False}])
def test_independent_and_excluded_reviews_cannot_be_attached_or_admitted(store, metadata):
    row = reserve(store)
    path = original(store, row["review_session_id"], **metadata)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        attach(store, row, path)
    with pytest.raises(ValueError):
        lifecycle.require_training_session_admission([(path.parent / "pairwise_review_decisions.json", "unused")])
    assert path.read_bytes() == before


def test_discard_is_enforced_by_existing_pending_evidence_flags_and_native_admission(store):
    row = reserve(store)
    path = original(store, row["review_session_id"])
    attach(store, row, path)
    bundles = [(path.parent / "pairwise_review_decisions.json", "unused")]
    lifecycle.require_training_session_admission(bundles)
    discard(store, row)
    with pytest.raises(ValueError, match="excluded from training"):
        lifecycle.require_training_session_admission(bundles)
    # Even an accidental future promotion of include_in_training cannot erase
    # the explicit discard. The original pending reader also sees editing_locked.
    saved = read(path)
    saved["include_in_training"] = True
    write(path, saved)
    with pytest.raises(ValueError, match="excluded from training"):
        lifecycle.require_training_session_admission(bundles)


def test_legacy_sessions_are_explicitly_discarded_without_merging(store):
    paths = [original(store, "old-" + role, workspace=role) for role in ("training", "blind")]
    sessions = [dict(read(path), session_path=str(path)) for path in paths]
    before = {path: (path.parent / "pairwise_review_decisions.json").read_bytes() for path in paths}
    with pytest.raises(ValueError, match="Existing editable reviews"):
        reserve(store, existing_sessions=sessions)
    lifecycle.discard_legacy_review_sessions(store, series="ffac", reviewer="Reviewer",
        session_paths=paths, reason="Begin a new combined review")
    assert all((path.parent / "pairwise_review_decisions.json").read_bytes() == before[path] for path in paths)
    assert all(read(path)["include_in_training"] is False for path in paths)
    refreshed = [dict(read(path), session_path=str(path)) for path in paths]
    assert reserve(store, existing_sessions=refreshed)["status"] == "active"


def test_discard_write_failure_keeps_intent_and_retry_finishes_without_new_session(store, monkeypatch):
    row = reserve(store)
    paths = [original(store, row["review_session_id"], workspace=role) for role in ("training", "blind")]
    for path in paths:
        attach(store, row, path)
    original_write = lifecycle._write

    def fail_second(path, value):
        if path == paths[1]:
            raise OSError("Review store temporarily unavailable")
        original_write(path, value)

    monkeypatch.setattr(lifecycle, "_write", fail_second)
    with pytest.raises(OSError):
        discard(store, row)
    assert lifecycle.active_review_session(store, series="ffac", reviewer="Reviewer")["status"] == "discarding"
    with pytest.raises(ValueError, match="pending discard"):
        reserve(store)
    monkeypatch.setattr(lifecycle, "_write", original_write)
    result = discard(store, row)
    assert result["status"] == "discarded"
    assert all(len(read(path)["review_lifecycle_history"]) == 1 for path in paths)


def test_training_completion_requires_original_lock_and_every_original_store(store):
    row = reserve(store)
    paths = [original(store, row["review_session_id"], workspace=role) for role in ("training", "blind")]
    for path in paths:
        attach(store, row, path)
    request = dict(model_id="new-model", request_path=str(store / "models" / "training_request.json"),
                   bundles=[dict(decisions=frozen_record(store, path))
                            for path in paths])

    def complete(value):
        return lifecycle.complete_review_session_training(store, series="ffac", reviewer="Reviewer",
            review_session_id=row["review_session_id"], request=value, reason="Train from these saved reviews")

    with pytest.raises(ValueError, match="has not locked"):
        complete(request)
    partial = deepcopy(request)
    partial["bundles"] = partial["bundles"][:1]
    lock_phase_review_sessions(partial)
    with pytest.raises(ValueError, match="All stores"):
        complete(partial)
    with pytest.raises(ValueError, match="already frozen"):
        discard(store, row)
    lock_phase_review_sessions(request)
    result = complete(request)
    assert result["status"] == "submitted"
    assert result["submitted_model_id"] == "new-model"
    assert complete(request) == result
    with pytest.raises(ValueError, match="already submitted"):
        discard(store, row)
    # Historical training inputs remain available for original retraining.
    lifecycle.require_training_session_admission([
        (path.parent / "pairwise_review_decisions.json", "unused") for path in paths])


@pytest.mark.parametrize("changed", ["before_lock", "after_lock", "snapshot", "original_snapshot", "unverified_lock"])
def test_training_completion_rejects_changed_frozen_or_live_revision_without_retiring(store, changed):
    row = reserve(store)
    path = original(store, row["review_session_id"])
    attach(store, row, path)
    record = frozen_record(store, path, derived={"decisions": {}})
    request = dict(model_id="new-model", bundles=[dict(decisions=record)])
    decisions = Path(record["source_path"])
    if changed == "before_lock":
        decisions.write_bytes(decisions.read_bytes() + b"\n")
    lock_phase_review_sessions(request)
    if changed == "after_lock":
        decisions.write_bytes(decisions.read_bytes() + b"\n")
    if changed in {"snapshot", "original_snapshot"}:
        target = Path(record[changed + "_path"])
        target.write_bytes(target.read_bytes() + b"\n")
    if changed == "unverified_lock":
        manifest = read(path)
        manifest["training_locks"][0]["source_matches_frozen_revision"] = True
        manifest["training_locks"][0]["source_decisions_sha256"] = ""
        write(path, manifest)
    originals = {item: item.read_bytes() for item in (
        path, decisions, Path(record["snapshot_path"]), Path(record["original_snapshot_path"]))}
    registry_path = lifecycle.review_session_registry_path(store, series="ffac", reviewer="Reviewer")
    registry_before = registry_path.read_bytes()
    with pytest.raises(ValueError, match="snapshot|live review"):
        lifecycle.complete_review_session_training(store, series="ffac", reviewer="Reviewer",
            review_session_id=row["review_session_id"], request=request, reason="Train from saved reviews")
    assert registry_path.read_bytes() == registry_before
    assert all(item.read_bytes() == content for item, content in originals.items())
    assert lifecycle.read_current_review_session(store, series="ffac", reviewer="Reviewer")["status"] == "active"


def test_training_completion_accepts_verified_original_capture_with_derived_training_view(store):
    row = reserve(store)
    path = original(store, row["review_session_id"])
    attach(store, row, path)
    record = frozen_record(store, path, derived={"decisions": {}})
    request = dict(model_id="new-model", bundles=[dict(decisions=record)])
    lock_phase_review_sessions(request)
    assert read(path)["training_locks"][0]["source_matches_frozen_revision"] is False
    originals = {item: item.read_bytes() for item in (
        path, Path(record["source_path"]), Path(record["snapshot_path"]), Path(record["original_snapshot_path"]))}
    result = lifecycle.complete_review_session_training(store, series="ffac", reviewer="Reviewer",
        review_session_id=row["review_session_id"], request=request, reason="Train from saved reviews")
    assert result["status"] == "submitted"
    assert all(item.read_bytes() == content for item, content in originals.items())


def test_corrupt_registry_does_not_silently_start_another_session(store):
    row = reserve(store)
    path = lifecycle.review_session_registry_path(store, series="ffac", reviewer="Reviewer")
    value = read(path)
    value["sessions"].append(dict(row, review_session_id="another-active"))
    write(path, value)
    with pytest.raises(ValueError, match="Several active"):
        reserve(store)


def test_concurrent_registry_action_fails_without_replacing_reservation(store):
    path = lifecycle.review_session_registry_path(store, series="ffac", reviewer="Reviewer")
    with lifecycle._locked(path):
        with pytest.raises(RuntimeError, match="in progress"):
            reserve(store)
    assert not path.exists()


def test_legacy_discard_refuses_registered_frozen_and_independent_stores(store):
    row = reserve(store)
    registered = original(store, row["review_session_id"])
    attach(store, row, registered)
    frozen = original(store, "old-frozen", training_locks=[{"model_id": "trained"}])
    heldout = original(store, "old-independent", independent_holdout=True)
    for path in (registered, frozen, heldout):
        before = path.read_bytes()
        with pytest.raises(ValueError):
            lifecycle.discard_legacy_review_sessions(store, series="ffac", reviewer="Reviewer",
                session_paths=[path], reason="Start another review")
        assert path.read_bytes() == before


def test_editor_lease_blocks_other_instance_actions_and_allows_own_attachment(store):
    row = reserve(store)
    path = original(store, row["review_session_id"])
    legacy = original(store, "old-session")
    before = path.read_bytes()
    with lifecycle.review_editing_lease(store, series="ffac", reviewer="Reviewer"):
        # The child attaches its original store while already holding its lease.
        attach(store, row, path)
        actions = [lambda: reserve(store), lambda: attach(store, row, path),
                   lambda: discard(store, row),
                   lambda: lifecycle.discard_legacy_review_sessions(store, series="ffac", reviewer="Reviewer",
                       session_paths=[legacy], reason="Begin another review"),
                   lambda: lifecycle.complete_review_session_training(store, series="ffac", reviewer="Reviewer",
                       review_session_id=row["review_session_id"], request={"model_id": "new-model"},
                       reason="Train with the saved evidence")]
        with ThreadPoolExecutor(max_workers=1) as executor:
            for action in actions:
                with pytest.raises(RuntimeError, match="Close the open model correction window"):
                    executor.submit(action).result()
        assert path.read_bytes() == before
    # Release is automatic and no validation files or lock files are deleted.
    assert discard(store, row)["status"] == "discarded"
    registry = lifecycle.review_session_registry_path(store, series="ffac", reviewer="Reviewer")
    assert registry.with_name(registry.stem + ".editor.lock").is_file()


def test_reservation_reloads_stale_policy_and_ignores_other_review_store(store, tmp_path):
    path = original(store, "legacy")
    stale = dict(read(path), session_path=str(path), editing_locked=True, include_in_training=False)
    with pytest.raises(ValueError, match="Existing editable reviews"):
        reserve(store, existing_sessions=[stale])
    unrelated = original(tmp_path / "another-review-store", "legacy")
    other = dict(read(unrelated), session_path=str(unrelated))
    assert reserve(store, existing_sessions=[other])["status"] == "active"


def scratch_sources(store, **changes):
    value = dict(ffac_root=str(store / "photos"), ffum_root="", combined_root=str(store / "combined"),
                 ffac_runs=["ac28", "ac01"], ffum_runs=[],
                 selected_candidates={"ac28": "candidate-28", "ac01": "candidate-01"})
    value.update(changes)
    return lifecycle.canonical_source_identity(value, series="FF_AC")


def scratch_reserve(store, **changes):
    values = dict(target_model_id="", review_basis="scratch", source_identity=scratch_sources(store))
    values.update(changes)
    return reserve(store, **values)


def scratch_original(store, row, **changes):
    values = dict(target_model_id="", review_basis="scratch", source_identity=row["source_identity"])
    values.update(changes)
    return original(store, row["review_session_id"], **values)


def test_scratch_source_identity_is_model_free_canonical_and_preserves_all_run_filter(store):
    source = scratch_sources(store, target_model_id="must-not-be-copied", model_root="not-a-source")
    assert source["format"] == lifecycle.SCRATCH_SOURCE_FORMAT
    assert source["series_id"] == "ffac"
    assert source["ffac_runs"] == ["ac01", "ac28"]
    assert list(source["selected_candidates"]) == ["ac01", "ac28"]
    assert "target_model_id" not in source and "model_root" not in source
    assert lifecycle.canonical_source_identity(source) == source
    assert scratch_sources(store, ffac_runs=None)["ffac_runs"] is None


@pytest.mark.parametrize("changes", [
    {"selected_candidates": {}}, {"selected_candidates": {"ac28": ""}},
    {"ffac_runs": ["ac28", "ac28"]}, {"ffac_runs": ["ac01"]},
    {"ffac_runs": "ac28"}, {"combined_root": "relative"},
    {"ffac_root": ""}, {"series_id": "ffum"},
])
def test_scratch_rejects_ambiguous_sources_without_reserving(store, changes):
    with pytest.raises(ValueError):
        scratch_reserve(store, source_identity=dict(scratch_sources(store), **changes))
    assert lifecycle.active_review_session(store, series="ffac", reviewer="Reviewer") is None


def test_scratch_reopen_requires_same_sources_and_cannot_turn_into_model_review(store):
    row = scratch_reserve(store)
    assert row["target_model_id"] == "" and row["review_basis"] == "scratch"
    assert scratch_reserve(store) == row
    changed = scratch_sources(store, selected_candidates={"ac01": "candidate-01", "ac28": "changed"})
    with pytest.raises(ValueError, match="Submit or discard"):
        scratch_reserve(store, source_identity=changed)
    with pytest.raises(ValueError, match="Submit or discard"):
        reserve(store)
    discard(store, row)
    assert reserve(store)["target_model_id"] == "saved-model"
    with pytest.raises(ValueError, match="Submit or discard"):
        scratch_reserve(store)


@pytest.mark.parametrize("changes", [
    {"target_model_id": "fake-model"}, {"source_identity": None}, {"review_basis": "model"},
])
def test_scratch_requires_explicit_basis_and_empty_target_model(store, changes):
    with pytest.raises(ValueError):
        scratch_reserve(store, **changes)
    assert lifecycle.active_review_session(store, series="ffac", reviewer="Reviewer") is None


def test_scratch_attach_discard_preserves_original_evidence_and_source_binding(store):
    row = scratch_reserve(store)
    path = scratch_original(store, row)
    for changed in (
        dict(read(path), source_identity=scratch_sources(store, combined_root=str(store / "other"))),
        dict(read(path), review_basis="model"),
        dict(read(path), target_model_id="pretend-model"),
    ):
        good = path.read_bytes()
        write(path, changed)
        with pytest.raises(ValueError):
            attach(store, row, path)
        path.write_bytes(good)
    attach(store, row, path)
    evidence = {p: p.read_bytes() for p in path.parent.iterdir() if p != path}
    result = discard(store, row)
    assert result["status"] == "discarded"
    assert result["target_model_id"] == "" and result["source_identity"] == row["source_identity"]
    assert read(path)["editing_locked"] and read(path)["include_in_training"] is False
    assert all(p.read_bytes() == data for p, data in evidence.items())


def test_scratch_uses_existing_training_freeze_and_submission_ownership(store):
    row = scratch_reserve(store)
    path = scratch_original(store, row)
    attach(store, row, path)
    request = dict(model_id="first-real-trained-model", bundles=[dict(decisions=frozen_record(store, path))])

    def complete():
        return lifecycle.complete_review_session_training(store, series="ffac", reviewer="Reviewer",
            review_session_id=row["review_session_id"], request=request, reason="Train from scratch labels")

    with pytest.raises(ValueError, match="has not locked"):
        complete()
    lock_phase_review_sessions(request)
    with pytest.raises(ValueError, match="already frozen"):
        discard(store, row)
    original_bytes = path.read_bytes()
    write(path, dict(read(path), source_identity=scratch_sources(store, combined_root=str(store / "changed"))))
    with pytest.raises(ValueError, match="source identity"):
        complete()
    assert lifecycle.active_review_session(store, series="ffac", reviewer="Reviewer")["status"] == "active"
    path.write_bytes(original_bytes)
    result = complete()
    assert result["status"] == "submitted"
    assert result["target_model_id"] == ""
    assert result["submitted_model_id"] == "first-real-trained-model"
    assert result["source_identity"] == row["source_identity"]


def test_scratch_editor_lease_blocks_another_instance_and_allows_own_attach(store):
    row = scratch_reserve(store)
    path = scratch_original(store, row)
    with lifecycle.review_editing_lease(store, series="ffac", reviewer="Reviewer"):
        attach(store, row, path)
        with ThreadPoolExecutor(max_workers=1) as executor:
            for action in (lambda: scratch_reserve(store), lambda: discard(store, row)):
                with pytest.raises(RuntimeError, match="Close the open"):
                    executor.submit(action).result()
    assert discard(store, row)["status"] == "discarded"


def test_preexisting_model_registry_without_review_basis_remains_reopenable(store):
    row = reserve(store)
    registry = lifecycle.review_session_registry_path(store, series="ffac", reviewer="Reviewer")
    value = read(registry)
    del value["sessions"][0]["review_basis"]
    write(registry, value)
    assert reserve(store)["review_session_id"] == row["review_session_id"]
    path = original(store, row["review_session_id"])
    attach(store, row, path)
    assert discard(store, row)["status"] == "discarded"
