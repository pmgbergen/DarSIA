"""UNRUN refusal cases: consistent historical declarations are not IO proof.

The original result dataclass and receipt writer create actual scientific JSON
bytes here. The control documents are synthetic; no solver, service, writer
observation, process retirement or lifetime capacity qualification is claimed.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from darsia.fluidflower import preset_wasserstein_capacity as preset
from darsia.fluidflower import preset_wasserstein_protocol as protocol
from darsia.fluidflower import preset_wasserstein_results as results
from darsia.fluidflower.pipeline.capacity import read_bound
from darsia.fluidflower.pipeline.checkpoint import retirement_document
from darsia.fluidflower.pipeline.job_queue import _immutable_bytes, descriptor
from darsia.presets.workflows.comparison.comparison_wasserstein import WassersteinDistanceResult
from .test_preset_wasserstein_protocol import task_and_job


def _put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return descriptor(path)


@pytest.fixture(params=["empty", "wrong-task", "wrong-owner"])
def historical_declarations(tmp_path, request):
    task, _job, operation = task_and_job()
    task["attempt_id"] = "a" * 32
    raw = _immutable_bytes(operation)
    reference = dict(task["configuration"], sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    # This existing typed fixture deliberately shares the one metadata reference.
    for key in ("configuration", "series", "series_manifest"):
        task[key] = deepcopy(reference)
    task["input_catalog"]["configuration"] = deepcopy(reference)
    task["input_catalog"]["inputs"][0]["reference"] = deepcopy(reference)
    scope = preset.operation_scope(task, operation)
    output = (tmp_path / "attempts" / task["attempt_id"]).resolve()
    output.mkdir(parents=True)
    rows, files = [], {}
    for expected in protocol.expected_comparisons(operation["preset_inputs"], task["comparison_time"]):
        # Explicit no-match controls retain the original missing-result schema/NaNs.
        row = WassersteinDistanceResult(run_1=expected["run_1"], run_2=expected["run_2"],
            time=expected["time_hours"], time_1=float("nan"), time_2=float("nan"),
            roi_name=expected["roi_name"], roi_exact_mass=float("nan"),
            roi_detected_mass_1=float("nan"), roi_detected_mass_2=float("nan"),
            distance=float("nan"), normalized_distance={key: float("nan")
                for key in ("plume_based", "rig_based", "mass_based")},
            computation_time=0.0, timestamp="2026-10-23T12:00:00", status="missing")
        path = row.save_to_dir(output)
        pin = descriptor(path)
        files[path.name] = pin
        rows.append(dict(expected, status="missing", file=pin))
    result = dict(format=results.RESULT_FORMAT, pipeline=protocol.PIPELINE, complete=True,
        run=task["run"], job_id=task["job_id"], comparison_time=task["comparison_time"],
        backend="cpu", precision="original", frames=0, expected_count=len(rows),
        comparison_count=len(rows), comparisons=rows, files=files, output_count=len(files),
        vtk=dict(files=[], actual_count=0, state="no_files_written"),
        worker_references_closed=True, validation=dict(state="not_scored"))
    result = results.write_preset_wasserstein_receipt(result, task=task, output_root=output)
    owner = dict(machine="fixture-cpu", pid=123, process_create_time="12.5")
    _put(output / "candidate_completed.json", dict(result, task=task,
        attempt_id=task["attempt_id"], commit=task["commit"], actual_worker=owner))
    retired = retirement_document(task, output, [owner], owner)
    if request.param == "empty":
        retired = {}
    elif request.param == "wrong-task":
        retired["run"] = "comparison-time-00000001"
    else:
        retired["supervisor"] = dict(owner, machine="unrelated-cpu", pid=999)
    _put(output / "retirement.json", retired)
    inventory = {path.relative_to(output).as_posix(): descriptor(path)
                 for path in output.rglob("*") if path.is_file()}
    metadata = tmp_path / "historical-declarations"
    namespace = _put(metadata / "namespace.json", dict(host=socket.gethostname()))
    service = _put(metadata / "service.json", dict(format="darsia-cache-import-profile-v1"))
    stores = _put(metadata / "stores.json", {"fixture": "declaration, not admission"})
    coverage = dict(format="darsia-worker-cache-service-only-qualification-v1", complete=True,
        passed=True, commit=task["commit"], pipeline=task["pipeline"], backend="cpu",
        precision="original", case_count=1, allocation_model="shared-service-only-no-claim-files-v1",
        service_profile=service, namespace_admission=namespace, stores_source=stores,
        producer_source_closure=[descriptor(Path(preset.__file__))], runtime_source_closure=[],
        shared_service_writers=["bytecode", "matplotlib", "numba", "tempfile"],
        claim_local_disposable_writers=[], unclassified_writers=[],
        actual_writer_IO_observed=True, maximum_service_cache_bytes=4096)
    retained = sum(pin["size_bytes"] for pin in inventory.values())
    measured = dict(format=preset.MEASUREMENT_FORMAT, complete=True, scope=scope,
        execution_receipt=result["execution_receipt"], result_bytes=retained,
        retained_output_inventory=dict(root=str(output), files=inventory),
        maximum_output_bytes=retained + 128, maximum_transient_bytes=0,
        maximum_observed_cache_bytes=0, worker_cache_service_coverage=coverage,
        observation=dict(format="darsia-preset-wasserstein-writer-observation-v1", complete=True,
            actual_writer_IO_observed=True, output_writers=["original-comparison-json", "original-vtk",
                "execution-receipt", "queue-progress", "queue-candidate", "worker-retirement"],
            transient_writers=[], unclassified_writers=[]))
    measurement = _put(metadata / "measurement.json", measured)
    coverage_hash = hashlib.sha256(json.dumps(coverage, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    completion = _put(metadata / "completion.json", dict(format=preset.COMPLETION_FORMAT,
        complete=True, passed=True, commit=task["commit"], scope=scope,
        measurement=measurement, execution_receipt=result["execution_receipt"],
        worker_cache_service_coverage_sha256=coverage_hash))
    receipt, _ = read_bound(result["execution_receipt"])
    repository = tmp_path / "repository"
    replay = _put(metadata / "replay.json", dict(format="darsia-preset-wasserstein-qualified-replay-v1",
        scope=scope, measurement=measurement, completion=completion,
        execution_receipt=result["execution_receipt"], receipt=receipt,
        qualification_host=socket.gethostname(), repository=str(repository),
        source_closure_sha256=preset.digest(coverage), actual_original_bytes_verified=True))
    entry = dict(scope=scope, measurement=measurement, completion=completion, replay=replay)
    profile = dict(format=preset.PROFILE_FORMAT, commit=task["commit"], pipeline=task["pipeline"],
        backend="cpu", precision="original", entries=[entry],
        qualification_host=socket.gethostname(), repository=str(repository))
    # Explicitly establish that original scientific bytes, receipt, all file hashes
    # and byte accounting are valid. This intentionally does NOT verify retirement.
    assert preset._receipt(measured, task, read_bound) == receipt
    assert result["result_statuses"] == {"missing": len(rows)}
    assert result["nonfinite_distances"] == len(rows)
    return SimpleNamespace(task=task, operation=operation, scope=scope, measured=measured,
        entry=entry, profile=profile, repository=repository, output=output,
        record=dict(task=task, operation=operation, measurement=measurement, completion=completion))


@pytest.mark.parametrize("verify_payloads", [False, True])
def test_consistent_writer_flags_and_retained_bytes_never_prove_measurement(historical_declarations, verify_payloads):
    case = historical_declarations
    with pytest.raises(ValueError) as error:
        preset._evidence(case.entry, case.task, case.scope, reader=read_bound,
                         verify_payloads=verify_payloads)
    assert str(error.value) == preset.MEASURED_UNAVAILABLE


def test_measured_builder_refuses_before_source_checks_or_any_profile_write(historical_declarations, tmp_path, monkeypatch):
    case = historical_declarations
    monkeypatch.setattr(preset, "require_repository_sources",
        lambda *args: pytest.fail("Source hashes cannot repair unavailable lifetime/retirement proof"))
    destination = tmp_path / "new-profile.json"
    before = {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")}
    with pytest.raises(ValueError) as error:
        preset.write_preset_wasserstein_profile([case.record], destination, repository=case.repository)
    assert str(error.value) == preset.MEASURED_UNAVAILABLE
    assert not destination.exists() and not list(tmp_path.glob("new-profile.entry-*.json"))
    assert {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")} == before


def test_old_host_local_measured_profile_cannot_be_reused(historical_declarations):
    case = historical_declarations
    with pytest.raises(ValueError) as error:
        preset.resolve_profile(case.profile, case.task, case.scope, reader=read_bound)
    assert str(error.value) == preset.MEASURED_UNAVAILABLE
