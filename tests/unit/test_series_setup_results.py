"""Typed operation results using synthetic files and original byte publication."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import series_setup_results as results
from darsia.fluidflower.pipeline.job_queue import descriptor
from darsia.fluidflower.pipeline.alias_job import job_metadata_reference, prepared_snapshot
from darsia.fluidflower.pipeline.products_publication import (
    publish_run_metadata, published_metadata_descriptor,
)
from darsia.fluidflower.published_jobs_snapshot import published_snapshot, _terminal

from .test_series_setup_protocol import _task, _job


def _put(path, value):
    """Write a synthetic JSON file and retain its exact byte descriptor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return descriptor(path)


def _read(path, expected=None):
    """Read only exact fixture metadata, preserving normal reader identity checks."""
    path = Path(path)
    if expected is not None and descriptor(path) != expected:
        raise ValueError("Fixture metadata changed")
    return json.loads(path.read_text(encoding="utf-8"))


def _context(tmp_path, *, operation="build"):
    """Create a producer receipt and the runtime's original claim envelope."""
    run = "setup-apply-fixture" if operation == "apply" else "run-a"
    task = dict(_task(operation, run=run), job_id="setup-job", attempt_id="a" * 32,
                execution_ready=True, runtime_support="darsia-role-io-runtime-v1")
    job = dict(_job(task), format_version=1, legacy_paths=False, series_id="campaign",
        execution_ready=True, runtime_support="darsia-role-io-runtime-v1",
        expected_runs=["setup-apply-fixture", "run-a"],
        setup_operations=[dict(run="setup-apply-fixture", operation="apply"),
                          dict(run="run-a", operation="build")])
    output = tmp_path / "local-results" / task["job_id"] / run / ".attempts" / task["attempt_id"]
    receipt = dict(format=results.RESULT_FORMAT, pipeline="series-setup", complete=True,
        run=run, operation=operation, backend="cpu", precision="float32",
        setup_plan=dict(path="Z:/unavailable/setup-plan.json", sha256="d" * 64, size_bytes=17),
        worker_references_closed=True,
        actual_controls=dict(configuration_files=[dict(path="Z:/unavailable/run.toml",
            sha256="e" * 64, size_bytes=41)], renderer_certificate=dict(
                path="Z:/unavailable/baseline_native_renderer.json", sha256="f" * 64, size_bytes=61)),
        complete_rig_model=False, production_qualification_claimed=False)
    bound = _put(output / results.RECEIPT, receipt)
    result = dict(receipt, files={results.RECEIPT: bound}, task=deepcopy(task),
                  attempt_id=task["attempt_id"], commit=task["commit"],
                  actual_worker=dict(machine="fixture-cpu", pid=123))
    return SimpleNamespace(task=task, job=job, output=output, result=result,
                           receipt=receipt, token=task["attempt_id"])


def _published(tmp_path, *, operation="build"):
    """Use the existing metadata publisher and ordinary alias done/index formats."""
    case = _context(tmp_path, operation=operation)
    case.root = tmp_path / "shared-job"
    _put(case.root / "job.json", case.job)
    _put(case.root / "results" / case.task["run"] / "task.json", case.task)
    case.lease = case.root / "leases" / (case.task["run"] + "." + case.token)
    case.owner = dict(format="darsia-alias-lease-owner-v1", token=case.token,
        task=deepcopy(case.task), actual_owner=deepcopy(case.result["actual_worker"]),
        machine="fixture-cpu", output=dict(machine_id="fixture", root_alias="results",
            relative_path="/".join((case.task["job_id"], case.task["run"], ".attempts", case.token))))
    _put(case.lease / "owner.json", case.owner)
    rows, metadata, accessible = results.extract_setup_products(case.result, case, "fixture")
    candidate = _put(case.output / "candidate_completed.json", case.result)
    case.index = publish_run_metadata(case.lease / "artifacts", job=descriptor(case.root / "job.json"),
        run=case.task["run"], attempt_id=case.token, machine_id="fixture", completed=candidate,
        products=rows, metadata_product_ids=metadata, source_paths=accessible,
        file_writers_closed=True, all_final_products_listed=True)
    published = published_metadata_descriptor(case.index)
    completed = case.root / "results" / case.task["run"] / "completed.json"
    completed.write_bytes(Path(published["path"]).read_bytes())
    case.done = dict(format="darsia-alias-done-v1", run=case.task["run"], attempt_id=case.token,
        completed=job_metadata_reference(case.root, completed),
        products_index=job_metadata_reference(case.root, Path(case.index["path"])))
    _put(case.root / "done" / (case.task["run"] + ".json"), case.done)
    return case


def _claimed(tmp_path, *, operation="apply"):
    """Retain one actual synthetic current claim and its validated owner."""
    case = _context(tmp_path, operation=operation)
    case.root = tmp_path / "shared-job"
    _put(case.root / "job.json", case.job)
    _put(case.root / "results" / case.task["run"] / "task.json", case.task)
    _put(case.root / "claimed" / "fixture-cpu" / (case.task["run"] + ".json"), case.task)
    _put(case.root / "leases" / (case.task["run"] + "." + case.token) / "owner.json",
        dict(format="darsia-alias-lease-owner-v1", token=case.token,
             task=deepcopy(case.task), machine="fixture-cpu",
             actual_owner=deepcopy(case.result["actual_worker"]),
             output=dict(machine_id="fixture", root_alias="results",
                 relative_path="/".join((case.task["job_id"], case.task["run"], ".attempts", case.token)))))
    case.log = case.root / "logs" / "fixture-cpu.log"
    case.log.parent.mkdir()
    return case


def _progress_line(case, *, timestamp="2026-01-01T10:00:00Z", **changes):
    event = dict(state="setup_progress", operation=case.task["operation"],
        run="child-run" if case.task["operation"] == "apply" else case.task["run"],
        step="verify_saved_geometry", message="Checking saved geometry metadata")
    event.update(changes.pop("event", {}))
    envelope = dict(kind="progress", run=case.task["run"], attempt_id=case.token, event=event)
    envelope.update(changes)
    return timestamp + " " + json.dumps(envelope) + "\n"


@pytest.mark.parametrize("operation", ["apply", "build"])
def test_current_setup_claim_projects_exact_service_stage_without_completion(tmp_path, operation):
    case = _claimed(tmp_path, operation=operation)
    case.log.write_text(_progress_line(case), encoding="utf-8")
    snapshot = published_snapshot(dict(root=str(case.root)), case.job, {}, reader=_read)
    row = next(row for row in snapshot["runs"] if row["run"] == case.task["run"])
    assert row["reported_progress"] == dict(task_run=case.task["run"], attempt_id=case.token,
        operation=operation, run="child-run" if operation == "apply" else case.task["run"],
        step="verify_saved_geometry", message="Checking saved geometry metadata",
        reported_utc="2026-01-01T10:00:00Z", source_log=str(case.log))
    assert snapshot["workers"][0]["last_progress_utc"] == "2026-01-01T10:00:00Z"
    assert "Last report: " + operation in snapshot["workers"][0]["state"]
    assert row["state"] == "Claimed (liveness not established)"
    assert snapshot["state"] == "Claimed" and not snapshot["complete"]
    assert snapshot["completed_operations"] == snapshot["committed"] == 0
    assert not snapshot["errors"]


@pytest.mark.parametrize("part", ["attempt", "task_run", "operation", "service", "timestamp"])
def test_setup_stage_ignores_foreign_attempt_task_operation_service_or_time(tmp_path, part):
    case = _claimed(tmp_path)
    changes = dict(
        attempt=dict(attempt_id="b" * 32), task_run=dict(run="another-task"),
        operation=dict(event=dict(operation="build")),
        service={}, timestamp=dict(timestamp="2026-01-01T10:00:00"),
    )[part]
    path = case.log if part != "service" else case.log.with_name("another-service.log")
    path.write_text(_progress_line(case, **changes), encoding="utf-8")
    snapshot = published_snapshot(dict(root=str(case.root)), case.job, {}, reader=_read)
    assert all("reported_progress" not in row for row in snapshot["runs"])
    assert snapshot["workers"][0]["last_progress_utc"] == "Not reported"
    assert not snapshot["errors"] and not snapshot["complete"]


def test_setup_stage_reads_a_bounded_tail_and_accepts_only_latest_matching_report(tmp_path):
    case = _claimed(tmp_path)
    old = _progress_line(case, event=dict(step="old-step"))
    latest = _progress_line(case, timestamp="2026-01-01T10:01:00Z", event=dict(step="apply"))
    case.log.write_text(old + "x" * (128 * 1024) + "\n" + latest
        + _progress_line(case, attempt_id="b" * 32) + "partial record", encoding="utf-8")
    snapshot = published_snapshot(dict(root=str(case.root)), case.job, {}, reader=_read)
    row = snapshot["runs"][0]
    assert row["reported_progress"]["step"] == "apply"
    assert row["reported_progress"]["reported_utc"] == "2026-01-01T10:01:00Z"
    assert len(snapshot["logs"][0]["tail"].encode()) == 128 * 1024
    assert "old-step" not in snapshot["logs"][0]["tail"]


def test_setup_result_keeps_exact_task_plan_and_operation_without_mutation(tmp_path):
    """The result joins the actual operation without inventing calibration or frames."""
    case = _context(tmp_path)
    before = deepcopy((case.result, case.task, case.job))
    assert results.validate_setup_result(case.result, case.task, case.job) is case.result
    assert (case.result, case.task, case.job) == before
    assert "calibration_sha256" not in case.result and "frames" not in case.result


@pytest.mark.parametrize("part", ["operation", "run", "precision", "plan", "files", "frames", "closure"])
def test_setup_result_refuses_changed_operation_identity(tmp_path, part):
    """A changed operation or invented image result cannot reach publication."""
    case = _context(tmp_path)
    if part == "plan":
        case.result["setup_plan"]["sha256"] = "e" * 64
    elif part == "files":
        case.result["files"]["extra.json"] = case.result["files"][results.RECEIPT]
    elif part == "frames":
        case.result["frames"] = 1
    elif part == "closure":
        case.result["worker_references_closed"] = False
    else:
        case.result[part] = dict(operation="apply", run="another-run", precision="float64")[part]
    with pytest.raises(ValueError):
        results.validate_setup_result(case.result, case.task, case.job)


def test_extractor_indexes_only_actual_receipt_and_preserves_external_descriptors(tmp_path):
    """External control/configuration/certificate paths remain passive provenance."""
    case = _context(tmp_path)
    rows, metadata, accessible = results.extract_setup_products(case.result, case, "fixture")
    assert [row["product_id"] for row in rows] == metadata == [results.RECEIPT]
    assert accessible == {results.RECEIPT: case.output / results.RECEIPT}
    assert _read(accessible[results.RECEIPT]) == case.receipt
    assert case.receipt["complete_rig_model"] is False
    assert case.receipt["production_qualification_claimed"] is False


@pytest.mark.parametrize("service,accepted", [("fixture-cpu", True), ("other-target-cpu", False)])
def test_actual_service_identity_is_bound_to_the_physical_target_and_backend(tmp_path, service, accepted):
    """Runtime service names retain the physical store id plus explicit backend."""
    case = _context(tmp_path)
    case.result["actual_worker"]["machine"] = service
    if accepted:
        rows, _, _ = results.extract_setup_products(case.result, case, "fixture")
        assert rows[0]["machine_id"] == "fixture"
    else:
        with pytest.raises(ValueError, match="runtime envelope"):
            results.extract_setup_products(case.result, case, "fixture")


@pytest.mark.parametrize("part", ["escaped_receipt", "changed_receipt", "wrong_owner", "wrong_attempt"])
def test_extractor_refuses_foreign_or_changed_attempt_products(tmp_path, part):
    """Receipt-only extraction still requires actual owned bytes and runtime identity."""
    case = _context(tmp_path)
    if part == "escaped_receipt":
        case.result["files"][results.RECEIPT] = _put(tmp_path / results.RECEIPT, case.receipt)
    elif part == "changed_receipt":
        _put(case.output / results.RECEIPT, dict(case.receipt, operation="apply"))
    elif part == "wrong_owner":
        case.result["actual_worker"]["machine"] = "other-target"
    else:
        case.result["attempt_id"] = "b" * 32
    with pytest.raises(ValueError):
        results.extract_setup_products(case.result, case, "fixture")


def test_terminal_uses_only_shared_copies_and_retains_operation_descriptors(tmp_path, monkeypatch):
    """A reader cannot stat or open the producer's original local output tree."""
    case = _published(tmp_path)
    original_stat, original_open = Path.stat, Path.open

    def guarded_stat(path, *args, **kwargs):
        if path.is_relative_to(case.output):
            pytest.fail("Terminal reader inspected the producer's local path")
        return original_stat(path, *args, **kwargs)

    def guarded_open(path, *args, **kwargs):
        if path.is_relative_to(case.output):
            pytest.fail("Terminal reader opened the producer's local path")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    monkeypatch.setattr(Path, "open", guarded_open)
    completed, counted, lineage = _terminal(case.root, case.root / "results" / case.task["run"],
        case.task, case.job, case.done, _read)
    assert completed == case.result and counted == {} and lineage == []
    assert completed["actual_controls"] == case.receipt["actual_controls"]
    snapshot = published_snapshot(dict(root=str(case.root)), case.job, {}, reader=_read)
    assert snapshot["completed_operations"] == 1 and snapshot["total_operations"] == 2
    assert snapshot["committed"] == 0 and snapshot["total_frames"] is None
    assert snapshot["runs"][1]["stage"] == "Build"
    assert snapshot["runs"][1]["case_ids"] == [] and not snapshot["errors"]


@pytest.mark.parametrize("part", ["owner", "receipt_copy", "operation_mapping", "done_attempt"])
def test_terminal_refuses_tampered_shared_identity(tmp_path, part):
    """The normal owner, done, index and receipt identities remain authoritative."""
    case = _published(tmp_path)
    if part == "owner":
        _put(case.lease / "owner.json", dict(case.owner, actual_owner=dict(machine="other-target")))
    elif part == "receipt_copy":
        _put(case.lease / "artifacts" / "metadata" / results.RECEIPT,
             dict(case.receipt, complete=False))
    elif part == "operation_mapping":
        case.job["setup_operations"][1]["operation"] = "apply"
    else:
        case.done["attempt_id"] = "b" * 32
    with pytest.raises(ValueError):
        results.setup_terminal(case.root, case.task, case.job, case.done, _read)


def test_prepared_projection_waits_for_apply_only_from_explicit_operation_map(tmp_path):
    """Future child tasks are absent until Apply; their ids never imply a stage."""
    case = _context(tmp_path, operation="apply")
    root = tmp_path / "shared-job"
    _put(root / "job.json", case.job)
    _put(root / "results" / case.task["run"] / "task.json", case.task)
    snapshot = prepared_snapshot(dict(root=str(root)), case.job, {}, reader=_read)
    assert [row["stage"] for row in snapshot["runs"]] == ["Apply", "Build"]
    assert snapshot["runs"][1]["state"] == "Waiting for Apply"
    assert snapshot["completed_operations"] == 0 and snapshot["remaining_operations"] == 2
    assert snapshot["work_unit"] == "operations" and snapshot["committed"] == 0
    legacy = dict(case.job)
    legacy.pop("setup_operations")
    snapshot = prepared_snapshot(dict(root=str(root)), legacy, {}, reader=_read)
    assert snapshot["runs"][1]["state"] == "Not queued" and snapshot["runs"][1]["stage"] is None
