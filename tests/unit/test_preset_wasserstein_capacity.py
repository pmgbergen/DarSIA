"""UNRUN source-authored admission tests; no scientific parity claim."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import socket
from types import SimpleNamespace

import pytest

from darsia.fluidflower import preset_wasserstein_capacity as preset
from darsia.fluidflower.pipeline import capacity
from darsia.fluidflower.pipeline.job_queue import _immutable_bytes


@pytest.fixture
def task(monkeypatch):
    # These tests isolate capacity from the separately tested typed catalogue.
    monkeypatch.setattr(preset, "validate_preset_wasserstein_task", lambda value: value)
    return dict(pipeline="preset-wasserstein", backend="cpu", precision="original",
        commit="a" * 40, run="comparison-time-00000000", ordered_case_ids=["comparison-time-00000000"],
        attempt_id="1" * 32, expected_comparisons=6, preset_inputs_sha256="b" * 64,
        comparison_time=dict(order=0, time_hours=2.5, tolerance_hours=0.1),
        configuration=dict(sha256="c" * 64, bytes=123))


def _administrative_estimate(task):
    scope = preset.task_scope(task, requested_time_count=3)
    return preset.PresetWassersteinEntitlement(task["run"], 1, 6, 700, 80,
        scope, {}, {}, task["commit"])


def test_scope_pins_operation_inputs_time_and_result_count(task):
    scope = preset.task_scope(task, requested_time_count=3)
    assert scope["operation_sha256"] == task["configuration"]["sha256"]
    assert scope["comparison_time"] == task["comparison_time"]
    assert scope["expected_comparisons"] == 6
    assert scope["requested_time_count"] == 3
    changed = deepcopy(task)
    changed["comparison_time"]["order"] = 3
    with pytest.raises(ValueError, match="outside"):
        preset.task_scope(changed, requested_time_count=3)


def test_operation_scope_refuses_changed_consumed_bytes(task, monkeypatch):
    monkeypatch.setattr(preset, "validate_operation", lambda value: value)
    inputs = dict(runs=[3, 1, 2], rois=["upper", "lower"],
        requested_times=[task["comparison_time"]])
    operation = dict(run=task["run"], comparison_time=task["comparison_time"], preset_inputs=inputs)
    task["preset_inputs_sha256"] = preset.digest(inputs)
    raw = _immutable_bytes(operation)
    task["configuration"] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    assert preset.operation_scope(task, operation)["expected_comparisons"] == 6
    changed = deepcopy(operation)
    changed["preset_inputs"]["runs"] = [1, 2, 3]
    with pytest.raises(ValueError, match="operation bytes"):
        preset.operation_scope(task, changed)


def test_native_and_other_host_profiles_refuse_before_evidence_reads(task):
    scope = preset.task_scope(task, requested_time_count=3)
    reader = lambda pin: pytest.fail("An inapplicable profile must not read original output paths")
    with pytest.raises(ValueError, match="no native profile reuse"):
        preset.resolve_profile(dict(format="darsia-expected-space-profile-v1"), task, scope, reader=reader)
    profile = dict(format=preset.PROFILE_FORMAT, commit=task["commit"], pipeline=task["pipeline"],
        backend="cpu", precision="original", entries=[dict(scope=scope)],
        qualification_host=socket.gethostname() + "-other", repository="C:/checkout")
    with pytest.raises(ValueError, match="another machine"):
        preset.resolve_profile(profile, task, scope, reader=reader)


def test_another_requested_time_never_scales_a_measured_entry(task):
    scope = preset.task_scope(task, requested_time_count=3)
    other = deepcopy(scope)
    other["comparison_time"]["time_hours"] = 99.0
    profile = dict(format=preset.PROFILE_FORMAT, commit=task["commit"], pipeline=task["pipeline"],
        backend="cpu", precision="original", entries=[dict(scope=other)],
        qualification_host=socket.gethostname(), repository="C:/checkout")
    with pytest.raises(ValueError, match="No measured profile"):
        preset.resolve_profile(profile, task, scope, reader=lambda pin: pytest.fail("No matching evidence"))


def test_whole_time_reservation_has_no_frame_release(task, tmp_path):
    ledger = object.__new__(capacity.CapacityLedger)
    ledger.root = tmp_path / "ledger"
    ledger.root.mkdir()
    ledger._common_binding = lambda: True
    ledger.machine_id, ledger.host, ledger.account = "worker", "physical", "sid"
    ledger.volume, ledger.remote_scope = dict(volume_id="volume"), None
    owner = dict(machine="worker", pid=123, process_create_time="12.5")
    claim = SimpleNamespace(task=task, token=task["attempt_id"])
    reserved = ledger.bind(tmp_path / "job", claim, owner, _administrative_estimate(task))
    row = json.loads((reserved / "reserved.json").read_text())
    assert row["comparison_count"] == 6 and row["case_count"] == 1
    assert "products_per_frame_estimated_bytes" not in row
    assert ledger.remaining() == 780
    ledger.progress(reserved, owner, dict(state="preset_wasserstein_progress", completed_count=5))
    assert ledger.remaining() == 780
    with pytest.raises(ValueError, match="Whole-operation"):
        ledger.progress(reserved, owner, dict(state="frame_completed", complete=True, run=task["run"]))
    ledger.release(reserved, owner)
    assert ledger.remaining() == 0


def test_forged_frame_progress_cannot_reduce_retained_preset_reservation(task, tmp_path):
    ledger = object.__new__(capacity.CapacityLedger)
    ledger.root = tmp_path / "ledger"
    ledger.root.mkdir()
    ledger._common_binding = lambda: True
    ledger.machine_id, ledger.host, ledger.account = "worker", "physical", "sid"
    ledger.volume, ledger.remote_scope = dict(volume_id="volume"), None
    owner = dict(machine="worker", pid=123, process_create_time="12.5")
    reservation = ledger.bind(tmp_path / "job", SimpleNamespace(task=task, token=task["attempt_id"]), owner, _administrative_estimate(task))
    (reservation / "p-00000001.json").write_text("{}")
    with pytest.raises(ValueError, match="cannot have frame progress"):
        ledger.remaining()


def test_first_administration_is_never_a_measured_estimate(task):
    estimate = preset.PresetWassersteinEntitlement(task["run"], 1, 6, 800, 20,
        preset.task_scope(task, requested_time_count=3), {}, {}, task["commit"])
    assert preset.require_estimate(task, estimate, allow_administration=True)
    with pytest.raises(ValueError, match="measured"):
        preset.require_estimate(task, estimate)
    assert asdict(estimate)["not_a_measured_profile"] is True


@pytest.mark.parametrize("allow_administration", [False, True])
def test_preformed_measured_estimate_is_disabled_even_with_administration_flag(task, allow_administration):
    estimate = preset.PresetWassersteinEstimate(task["run"], 1, 6, 700, 80,
        {}, {}, {}, preset.task_scope(task, requested_time_count=3), task["commit"])
    with pytest.raises(ValueError) as error:
        preset.require_estimate(task, estimate, allow_administration=allow_administration)
    assert str(error.value) == preset.MEASURED_UNAVAILABLE


@pytest.mark.parametrize("projection", [False, True])
def test_actual_cache_adapter_refuses_a_preformed_measured_estimate(task, projection):
    from darsia_cache.worker_adapter import WorkerCacheAdapter
    adapter = object.__new__(WorkerCacheAdapter)
    adapter.runtime = SimpleNamespace(backend="cpu", commit=task["commit"])
    adapter.lease = SimpleNamespace(profile={"format": "darsia-cache-import-profile-v1"},
                                   first_qualification=None)
    adapter.profiles = {}
    estimate = preset.PresetWassersteinEstimate(task["run"], 1, 6, 700, 80,
        {}, {}, {}, preset.task_scope(task, requested_time_count=3), task["commit"])
    verify = adapter.verify_profile_projection if projection else adapter.verify_profile
    with pytest.raises(ValueError) as error:
        verify(task, estimate, {}, {"format": preset.PROFILE_FORMAT}, {}, {})
    assert str(error.value) == preset.MEASURED_UNAVAILABLE
    assert adapter.profiles == {}


def test_retained_inventory_counts_metadata_without_source_path_reads(monkeypatch):
    # Byte-inventory metadata alone never qualifies writer IO or measured peaks.
    root = "C:/producer-local/attempt"
    names = ["row.json", "comparison_execution.json", "progress_00000000.json",
             "candidate_completed.json", "retirement.json"]
    files = {name: dict(path=root + "/" + name, sha256="d" * 64, size_bytes=index + 1)
             for index, name in enumerate(names)}
    measured = dict(execution_receipt=files["comparison_execution.json"], result_bytes=15,
        retained_output_inventory=dict(root=root, files=files))
    result = dict(files={"row.json": files["row.json"]})
    from darsia.fluidflower.pipeline import job_queue
    monkeypatch.setattr(job_queue, "descriptor", lambda path: pytest.fail("Ordinary replay must not visit producer output"))
    assert preset._retained_output_bytes(measured, result) == 15
    measured["result_bytes"] = 3  # Scientific row + receipt alone omit retained operations.
    with pytest.raises(ValueError, match="retained attempt"):
        preset._retained_output_bytes(measured, result)
    measured["result_bytes"] = 15
    del files["retirement.json"]
    with pytest.raises(ValueError, match="candidate and retirement"):
        preset._retained_output_bytes(measured, result)


def test_retained_inventory_only_counts_bytes_and_refuses_uninventoried_file(tmp_path):
    # This helper never establishes writer IO, lifetime peaks or retirement.
    from darsia.fluidflower.pipeline.job_queue import descriptor
    files = {}
    for name in ("row.json", "comparison_execution.json", "progress_00000000.json",
                 "candidate_completed.json", "retirement.json"):
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        files[name] = descriptor(path)
    measured = dict(execution_receipt=files["comparison_execution.json"],
        result_bytes=sum(pin["size_bytes"] for pin in files.values()),
        retained_output_inventory=dict(root=str(tmp_path), files=files))
    result = dict(files={"row.json": files["row.json"]})
    assert preset._retained_output_bytes(measured, result, verify_payloads=True) == 10
    (tmp_path / "retained-extra.bin").write_bytes(b"not ignored")
    with pytest.raises(ValueError, match="all actual original output"):
        preset._retained_output_bytes(measured, result, verify_payloads=True)
