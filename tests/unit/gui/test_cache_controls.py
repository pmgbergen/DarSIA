"""Canonical cache-policy behaviors; no DarSIA application import.

Real namespace/lifecycle cases live once in test_cache_controls_identity_bound.
Fixture maxima and fake checkout observations are test values, not host profiles.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia_cache.policy import (
    CachePolicy, CachePolicyError, PHYSICAL_RESERVE_BYTES, budget_report,
    cache_settings, cache_status_line,
)
from darsia_cache.control import ClaimCacheBudget, failure_cleanup_due, reusable_checkout_admission
from darsia_cache.environment import derive_cache_environment

WHEN = datetime(2026, 10, 17, tzinfo=timezone.utc)
OWNER = {"pid": 1234, "process_create_time": "1792215600.125"}


def test_missing_quota_and_overlapping_durable_results_refuse(tmp_path):
    with pytest.raises(CachePolicyError, match="Configure"):
        cache_settings({}, required=True)
    doc = {"cache": {"root": str(tmp_path / "managed"), "quota_bytes": 0, "profiles": {}}}
    with pytest.raises(CachePolicyError):
        cache_settings(doc)
    doc["cache"].update(quota_bytes=100, profiles={"jobs": {
        "path": str(tmp_path / "profile.json"), "sha256": "0" * 64, "size_bytes": 1}})
    doc["stores"] = {"S": {"results": str(tmp_path / "managed" / "durable-results")}}
    with pytest.raises(CachePolicyError, match="disjoint"):
        cache_settings(doc)


def test_volume_root_store_overlap_remains_refused():
    document = dict(cache=dict(root="C:/managed-cache", quota_bytes=10,
        profiles={"jobs": dict(path="C:/proof.json", sha256="a" * 64, size_bytes=1)}),
        stores={"series": {"results": "C:/"}})
    with pytest.raises(CachePolicyError):
        cache_settings(document, required=True)
    document["cache"]["root"] = "C:/"
    document["stores"] = {}
    with pytest.raises(CachePolicyError):
        cache_settings(document, required=True)


def test_failure_retention_requires_48h_or_verified_same_gate_trigger():
    failed = dict(utc=WHEN.isoformat(), logical_gate="same-gate")
    assert not failure_cleanup_due(failed,
        now_utc=(WHEN + timedelta(hours=48, microseconds=-1)).isoformat(), retention_hours=48)
    assert failure_cleanup_due(failed, now_utc=(WHEN + timedelta(hours=48)).isoformat(), retention_hours=48)
    for hours in (0, 49):
        with pytest.raises(CachePolicyError):
            failure_cleanup_due(failed, now_utc=WHEN.isoformat(), retention_hours=hours)
    with pytest.raises(CachePolicyError):
        failure_cleanup_due(failed, now_utc=(WHEN - timedelta(seconds=1)).isoformat(), retention_hours=48)
    for logical_gate, complete in (("other-gate", True), ("same-gate", False)):
        with pytest.raises(CachePolicyError):
            failure_cleanup_due(failed, now_utc=(WHEN + timedelta(hours=2)).isoformat(), retention_hours=48,
                same_gate_success=dict(utc=(WHEN + timedelta(hours=1)).isoformat(),
                                       logical_gate=logical_gate, complete=complete))
    assert failure_cleanup_due(failed, now_utc=(WHEN + timedelta(hours=2)).isoformat(), retention_hours=48,
        same_gate_success=dict(utc=(WHEN + timedelta(hours=1)).isoformat(), logical_gate="same-gate", complete=True))
    # This pure predicate is not receipt verification; the real cleanup test
    # separately proves that a bare complete=True successor is insufficient.


def test_logical_quota_and_40gib_pending_allocation_estimate_are_separate():
    policy = CachePolicy("C:/managed-cache", 1000, 48, {})
    row = budget_report(policy, used_bytes=200, reserved_unwritten_bytes=300,
        request_bytes=500, physical_free_bytes=PHYSICAL_RESERVE_BYTES + 800)
    assert row["sufficient"] and row["post_admission_quota_headroom_bytes"] == 0
    assert row["needed_physical_bytes"] == PHYSICAL_RESERVE_BYTES + 800
    assert row["future_cache_physical_allocation_bound"] is None
    assert row["hard_filesystem_quota_verified"] is False
    assert not budget_report(policy, used_bytes=200, reserved_unwritten_bytes=300,
        request_bytes=501, physical_free_bytes=PHYSICAL_RESERVE_BYTES + 1000)["sufficient"]
    assert not budget_report(policy, used_bytes=0, reserved_unwritten_bytes=0,
        request_bytes=1, physical_free_bytes=PHYSICAL_RESERVE_BYTES,
        eligible_reclaimable_bytes=10**15)["sufficient"]
    quota = CachePolicy("C:/managed-cache", 100, 48, {})
    assert not budget_report(quota, used_bytes=90, reserved_unwritten_bytes=0,
        request_bytes=11, physical_free_bytes=PHYSICAL_RESERVE_BYTES + 100,
        eligible_reclaimable_bytes=90)["sufficient"]
    gib = 1024**3
    large = CachePolicy("C:/managed-cache", 100 * gib, 48, {})
    for pending, free, needed in ((0, 40 * gib + gib // 2, 41 * gib),
                                  (gib, 41 * gib + gib // 2, 42 * gib)):
        report = budget_report(large, used_bytes=0, reserved_unwritten_bytes=pending,
            request_bytes=gib, physical_free_bytes=free)
        assert report["needed_physical_bytes"] == needed and not report["sufficient"]
        assert report["future_cache_physical_allocation_bound"] is None


def test_preimport_environment_preserves_original_inputs_and_parent():
    token = "b" * 32
    stores = {"cache": dict(root="C:/managed-cache", quota_bytes=1,
                             profiles={"tests": {"separately": "bound"}})}
    original = dict(HOME="user-home", LOCALAPPDATA="persistent-state",
                    OMP_NUM_THREADS="32", NUMBA_DISABLE_JIT="0")
    before = dict(original)
    child = derive_cache_environment(stores, purpose="tests", token=token,
        scope_root="C:/managed-cache/tests/current/" + token, parent_environment=original)
    assert original == before and all(child[k] == v for k, v in original.items())
    assert child["NUMBA_CACHE_DIR"].endswith(token + "\\numba")
    assert child["TEMP"].endswith(token + "\\tmp")
    assert child["NUMBA_CACHE_LOCATOR_CLASSES"] == "UserProvidedCacheLocator"


def test_reusable_checkout_plan_refuses_dirty_live_foreign_or_unmanaged(tmp_path):
    policy = CachePolicy(str(tmp_path / "managed"), 100, 48, {})
    root = policy.purpose("clones") / "current"
    good = dict(managed_root=str(root), registered_as_own_reusable_checkout=True,
                clean=True, readers_retired=True, head="1" * 40)
    plan = reusable_checkout_admission(policy, checkout=root, expected_head="2" * 40, observation=good)
    assert plan["update_needed"] and not plan["new_full_tree_copy_allowed"]
    assert plan["execution_released"] is False
    for key in ("clean", "readers_retired", "registered_as_own_reusable_checkout"):
        with pytest.raises(CachePolicyError, match="preserved"):
            reusable_checkout_admission(policy, checkout=root, expected_head="2" * 40,
                                        observation={**good, key: False})
    with pytest.raises(CachePolicyError):
        reusable_checkout_admission(policy, checkout=tmp_path / "foreign-repo",
                                    expected_head="2" * 40, observation=good)
    with pytest.raises(CachePolicyError):
        reusable_checkout_admission(policy, checkout=root, expected_head="not-a-commit", observation=good)
    # These are supplied-observation rules, not a real Git update/reader lease.


def test_missing_writer_adapter_refuses_before_any_claim_or_profile_read(tmp_path):
    binding = {"path": str(tmp_path / "absent-profile.json"), "sha256": "0" * 64, "size_bytes": 1}
    runtime = SimpleNamespace(stores=SimpleNamespace(document={"cache": {
        "root": str(tmp_path / "managed"), "quota_bytes": 100, "profiles": {"jobs": binding}}}))
    with pytest.raises(CachePolicyError, match="writer/lifecycle adapter"):
        ClaimCacheBudget(runtime, SimpleNamespace(reserve=PHYSICAL_RESERVE_BYTES), OWNER)
    assert not Path(binding["path"]).exists() and not (tmp_path / "managed").exists()


def test_status_retains_measurement_time_and_requires_comparable_delta():
    original = dict(root="cache", logical_bytes=100, measured_utc=WHEN.isoformat(), semantics="logical")
    assert cache_status_line("Main", original)["delta_bytes"] == "UNMEASURED"
    current = {**original, "logical_bytes": 120, "measured_utc": (WHEN + timedelta(hours=1)).isoformat()}
    row = cache_status_line("Main", current, original)
    assert row["delta_bytes"] == 20 and row["measured_utc"] == current["measured_utc"]
    assert cache_status_line("Main", {**current, "root": "other"}, original)["delta_bytes"] == "UNMEASURED"

