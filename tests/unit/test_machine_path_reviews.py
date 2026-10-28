"""Review session registries stay readable on both systems.

A temporary folder stands in for a share opened at /Volumes/<share>. A store
attached on a computer with a machine path map is registered by its shared
path, and Windows-absolute registered paths are accepted on any system.
Metadata only: no review window, training or image read.
"""

import json
from pathlib import Path

import pytest

from darsia.fluidflower import native_phase_review_session as lifecycle
from darsia.fluidflower import storage_paths

SHARE = r"\\nas-host\Lab"
REVIEWER = "Reviewer One"


@pytest.fixture
def share(tmp_path, monkeypatch):
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "registry.toml"))
    monkeypatch.setenv("DARSIA_REVIEW_ALIAS_FILES", str(tmp_path / "absent-aliases.json"))
    local = tmp_path / "Volumes" / "Lab"
    (local / "reviews").mkdir(parents=True)
    storage_paths.set_machine_path_map([(SHARE, str(local))])
    yield local
    storage_paths.set_machine_path_map(())


def _original(store, session_id):
    root = store / "reviewer-one" / "campaign" / "training" / "sessions" / session_id / "manual_pairwise_review"
    root.mkdir(parents=True)
    path = root / "review_session.json"
    path.write_text(json.dumps(dict(
        format=lifecycle.SESSION_FORMAT, reviewer=REVIEWER, reviewer_id="reviewer-one",
        review_session_id=session_id, target_model_id="saved-model",
        decision_dir=storage_paths.machine_stored_text(root), workspace="training",
        editing_locked=False, training_locks=[], selected_candidates={"r1": "candidate"})), encoding="utf-8")
    (root / "pairwise_review_decisions.json").write_text(json.dumps({"decisions": {}}), encoding="utf-8")
    return path


def test_a_store_attached_here_is_registered_by_its_shared_path(share):
    store = share / "reviews"
    row = lifecycle.reserve_review_session(store, series="study", reviewer=REVIEWER, target_model_id="saved-model")
    path = _original(store, row["review_session_id"])
    attached = lifecycle.attach_review_manifest(store, series="study", reviewer=REVIEWER,
                                                review_session_id=row["review_session_id"], session_path=path)
    relative = path.relative_to(share).parts
    assert attached["manifest_paths"] == ["\\".join((SHARE, *relative))]
    # Attaching again recognises the shared spelling; the closing action reads it here.
    again = lifecycle.attach_review_manifest(store, series="study", reviewer=REVIEWER,
                                             review_session_id=row["review_session_id"], session_path=path)
    assert again["manifest_paths"] == attached["manifest_paths"]
    closed = lifecycle.discard_review_session(store, series="study", reviewer=REVIEWER,
                                              review_session_id=row["review_session_id"], reason="Wrong model")
    assert closed["status"] == "discarded"
    assert json.loads(path.read_text(encoding="utf-8"))["include_in_training"] is False


def test_windows_absolute_registered_paths_are_accepted_and_relative_ones_refused(tmp_path):
    registry = tmp_path / "registry.json"
    row = dict(review_session_id="s1", status="discarded", target_model_id="saved-model", history=[],
               manifest_paths=[SHARE + r"\reviews\s1\review_session.json"])
    registry.write_text(json.dumps(dict(format=lifecycle.FORMAT, series="study", reviewer_id="reviewer-one",
                                        sessions=[row])), encoding="utf-8")
    assert lifecycle._registry(registry, "study", "reviewer-one")["sessions"][0]["review_session_id"] == "s1"
    row["manifest_paths"] = ["reviews/s1/review_session.json"]
    registry.write_text(json.dumps(dict(format=lifecycle.FORMAT, series="study", reviewer_id="reviewer-one",
                                        sessions=[row])), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest path"):
        lifecycle._registry(registry, "study", "reviewer-one")
