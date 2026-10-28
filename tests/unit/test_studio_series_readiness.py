"""Generic registered series without changing the existing Studio contracts."""

import json

import pytest

from darsia.fluidflower import series_paths
from darsia.fluidflower.tools import phase_reference_readiness as readiness


def _registered(tmp_path, monkeypatch, key="ffre"):
    root = tmp_path / key
    root.mkdir()
    (root / "series.toml").write_text(
        'schema_version = 1\nseries_id = "' + key + '"\n'
        '[runs]\nids = ["own_c1", "own_c2"]\n', encoding="utf-8")
    monkeypatch.setenv("DARSIA_" + key.upper() + "_PROJECT_ROOT", str(root))
    return root


def _validation(tmp_path, dataset="ffre", run="own_c1"):
    root = tmp_path / "TESTONLY_validation"
    root.mkdir()
    binding = {"output_mode": "review", "selected_candidates": {run: "own-candidate"}}
    campaign = {"output_mode": "review", "jobs": [
        {"dataset": dataset, "run": run, "candidate_id": "own-candidate"}]}
    for name, value in (("validation_binding.json", binding), ("postflight_manifest.json", campaign)):
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    return root


def test_Studio_choices_keep_original_order_and_append_registered_series(monkeypatch, tmp_path):
    monkeypatch.setattr(series_paths, "configured_series", lambda: (
        ("ffre", "Reuse", tmp_path / "re"), ("ffac", "AC", tmp_path / "ac"),
        ("customstudy", "Another study", tmp_path / "custom")))
    assert series_paths.studio_series_keys() == ("ffum", "ffac", "ffbm", "ffre", "customstudy")


def test_Studio_without_extra_registration_has_exact_original_choices(monkeypatch):
    monkeypatch.setattr(series_paths, "configured_series", lambda: ())
    assert series_paths.studio_series_keys() == ("ffum", "ffac", "ffbm")


@pytest.mark.parametrize("key", ["ffre", "customstudy"])
def test_registered_series_defers_only_own_exact_bound_validation(tmp_path, monkeypatch, key):
    _registered(tmp_path, monkeypatch, key)
    root = _validation(tmp_path, key)
    receipt = readiness.defer_validation_references(root)
    assert receipt["status"] == "deferred"
    assert receipt["materialization_policy"] == "on-demand-exact-reference"
    assert receipt["reference_coverage"] == "not-requested"
    assert receipt["references"] == []
    assert receipt["unavailable_references"] == []
    readiness.require_reference_readiness(root, allow_pending=True)
    with pytest.raises(ValueError, match="not ready"):
        readiness.require_reference_readiness(root)


@pytest.mark.parametrize("dataset", ["ffac", "ffum", "ffbm"])
def test_original_series_branch_needs_no_new_registry_or_manifest(tmp_path, monkeypatch, dataset):
    def forbidden_lookup(_key):
        raise AssertionError("Original series must not acquire a new lookup requirement")
    monkeypatch.setattr(readiness, "series_manifest", forbidden_lookup)
    root = _validation(tmp_path, dataset, "original_run")
    assert readiness.defer_validation_references(root)["status"] == "deferred"


def test_generic_reference_refuses_unregistered_series_before_receipt(tmp_path, monkeypatch):
    monkeypatch.delenv("DARSIA_UNREGISTERED_PROJECT_ROOT", raising=False)
    monkeypatch.setattr(series_paths, "configured_series", lambda: ())
    root = _validation(tmp_path, "unregistered")
    with pytest.raises(ValueError, match="not registered"):
        readiness.defer_validation_references(root)
    assert not (root / readiness.RECEIPT).exists()


def test_generic_reference_refuses_foreign_run_and_noncanonical_series(tmp_path, monkeypatch):
    _registered(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="own series manifest"):
        readiness._require_reference_series([{"dataset": "ffre", "run": "foreign"}])
    with pytest.raises(ValueError, match="canonical registered ID"):
        readiness._require_reference_series([{"dataset": "FF_RE", "run": "own_c1"}])


def test_generic_reference_keeps_complete_campaign_candidate_guard(tmp_path, monkeypatch):
    _registered(tmp_path, monkeypatch)
    root = _validation(tmp_path)
    path = root / "postflight_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["jobs"][0]["candidate_id"] = "another-candidate"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="complete selected validation campaign"):
        readiness.defer_validation_references(root)


def test_generic_deferred_receipt_does_not_hide_changed_binding_or_failed_preparation(tmp_path, monkeypatch):
    _registered(tmp_path, monkeypatch)
    root = _validation(tmp_path)
    readiness.defer_validation_references(root)
    path = root / "validation_binding.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="another validation package"):
        readiness.require_reference_readiness(root, allow_pending=True)
    (root / readiness.RECEIPT).write_text(json.dumps(
        {"format": readiness.FORMAT, "status": "failed", "error": "TESTONLY original failure"}), encoding="utf-8")
    with pytest.raises(ValueError, match="TESTONLY original failure"):
        readiness.defer_validation_references(root)
