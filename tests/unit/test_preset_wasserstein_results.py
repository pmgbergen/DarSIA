"""Synthetic original bytes and shared publication lineage; no solver or payload IO."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import preset_wasserstein_results as results
from darsia.fluidflower.pipeline.alias_job import job_metadata_reference, prepared_snapshot
from darsia.fluidflower.pipeline.job_inputs import catalog_order_key
from darsia.fluidflower.pipeline.job_queue import descriptor
from darsia.fluidflower.pipeline.products_publication import (
    publish_run_metadata, published_metadata_descriptor,
)
from darsia.fluidflower.pipeline.publication_runtime import extract_products, make_publication_factory
from darsia.fluidflower.published_jobs_snapshot import _terminal, published_snapshot


def _put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return descriptor(path)


def _read(path, expected=None):
    path = Path(path)
    if expected is not None and descriptor(path) != expected:
        raise ValueError("Fixture metadata changed")
    return json.loads(path.read_text(encoding="utf-8"))


def _context(tmp_path, *, nonfinite=False, integer_ids=False, single_time=False):
    run = "comparison-time-00000000"
    references = {key: dict(root_alias="configuration", relative_path=leaf,
        sha256=letter * 64, bytes=17, role="configuration")
        for key, leaf, letter in (("configuration", "operations/" + run + ".json", "a"),
            ("series", "series.toml", "b"), ("series_manifest", "manifest.json", "c"))}
    uses = dict(configuration="run.configuration", series="run.execution_series",
                series_manifest="run.series_manifest")
    inputs = [dict(reference=deepcopy(value), uses=[uses[key]]) for key, value in references.items()]
    inputs.sort(key=lambda row: catalog_order_key(row["reference"]))
    selected = dict(order=0, time_hours=2.5, tolerance_hours=0.1)
    common = dict(format_version=1, legacy_paths=False, pipeline="preset-wasserstein",
        commit="1" * 40, backend="cpu", precision="original", storage_profile="full",
        execution_ready=True, runtime_support="darsia-role-io-runtime-v1", job_id="comparison-job",
        preset_inputs_sha256="d" * 64, series=references["series"], series_manifest=references["series_manifest"])
    task = dict(common, format="darsia-alias-task-v1", run=run,
        attempt_id="a" * 32, ordered_case_ids=[run], threads=1, comparison_time=selected,
        expected_comparisons=1, configuration=references["configuration"],
        input_catalog=dict(format="darsia-role-input-catalog-v1", complete=True, run=run,
            configuration=deepcopy(references["configuration"]), inputs=inputs,
            payload_hashes_verified=False, numerical_calls_started=False))
    job = dict(common, format="darsia-alias-job-v1", expected_runs=[run, "comparison-time-00000001"],
        preset_comparison_times=[selected, dict(order=1, time_hours=3.5, tolerance_hours=0.1)],
        comparisons_per_time=1, series_id="campaign")
    if single_time:
        job.update(expected_runs=[run], preset_comparison_times=[selected])
    output = tmp_path / "local-results" / task["job_id"] / run / ".attempts" / task["attempt_id"]
    left, right, roi = (1, 2, 3) if integer_ids else ("run-a", "run-b", "whole rig")
    row = dict(run_1=left, run_2=right, time=2.5, time_1=2.45, time_2=2.55, roi_name=roi,
        roi_exact_mass=1.25, roi_detected_mass_1=1.2, roi_detected_mass_2=1.3,
        distance=float("nan") if nonfinite else 0.012345678901234567,
        normalized_distance=dict(plume_based=float("nan") if nonfinite else 0.1,
            rig_based=float("inf") if nonfinite else 0.2, mass_based=0.3),
        computation_time=0.25, timestamp="2026-10-23T12:00:00", status="success")
    declaration = dict(run_1=left, run_2=right, time_hours=2.5, roi_name=roi, status=row["status"])
    filename = results._filename(declaration)
    original = _put(output / filename, row)
    vtk = output / "vtk" / "roi" / "flux.vtu"
    vtk.parent.mkdir(parents=True)
    vtk.write_bytes(b"synthetic original VTK bytes")
    scientific = dict(format=results.RESULT_FORMAT, pipeline="preset-wasserstein", complete=True,
        run=run, job_id=task["job_id"], comparison_time=selected, backend="cpu", precision="original",
        frames=0, expected_count=1, comparison_count=1, comparisons=[dict(declaration, file=original)],
        files={filename: original, "vtk/roi/flux.vtu": descriptor(vtk)}, output_count=2,
        vtk=dict(files=["vtk/roi/flux.vtu"], actual_count=1, state="files_written"),
        source_consumption=dict(complete=True), runtime=dict(input_dtypes=["float64"]),
        wall_seconds=0.3, worker_references_closed=True, validation=dict(state="not_scored"))
    finalized = results.write_preset_wasserstein_receipt(scientific, task=task, output_root=output)
    result = dict(finalized, task=deepcopy(task), attempt_id=task["attempt_id"], commit=task["commit"],
        actual_worker=dict(machine="fixture-cpu", pid=123))
    return SimpleNamespace(task=task, job=job, output=output, result=result,
        original=original, filename=filename, scientific=scientific, token=task["attempt_id"], machine="fixture-cpu")


def _shared(case, tmp_path):
    case.root = tmp_path / "shared-job"
    _put(case.root / "job.json", case.job)
    _put(case.root / "results" / case.task["run"] / "task.json", case.task)
    case.lease = case.root / "leases" / (case.task["run"] + "." + case.token)
    case.owner = dict(format="darsia-alias-lease-owner-v1", token=case.token,
        task=deepcopy(case.task), actual_owner=deepcopy(case.result["actual_worker"]),
        machine="fixture-cpu", output=dict(machine_id="fixture", root_alias="results",
            relative_path="/".join((case.task["job_id"], case.task["run"], ".attempts", case.token))))
    _put(case.lease / "owner.json", case.owner)
    return case


def _published(tmp_path, **options):
    case = _shared(_context(tmp_path, **options), tmp_path)
    rows, metadata, accessible = extract_products(case.result, case, "fixture", producer_origin_for_attempt=None)
    candidate = _put(case.output / "candidate_completed.json", case.result)
    case.index = publish_run_metadata(case.lease / "artifacts", job=descriptor(case.root / "job.json"),
        run=case.task["run"], attempt_id=case.token, machine_id="fixture", completed=candidate,
        products=rows, metadata_product_ids=metadata, source_paths=accessible,
        file_writers_closed=True, all_final_products_listed=True)
    copied = published_metadata_descriptor(case.index)
    completed = case.root / "results" / case.task["run"] / "completed.json"
    completed.write_bytes(Path(copied["path"]).read_bytes())
    case.done = dict(format="darsia-alias-done-v1", run=case.task["run"], attempt_id=case.token,
        completed=job_metadata_reference(case.root, completed),
        products_index=job_metadata_reference(case.root, Path(case.index["path"])))
    _put(case.root / "done" / (case.task["run"] + ".json"), case.done)
    return case


@pytest.mark.parametrize("integer_ids", [False, True])
def test_receipt_preserves_original_nans_and_identity_types(tmp_path, integer_ids):
    case = _context(tmp_path, nonfinite=True, integer_ids=integer_ids)
    assert descriptor(case.output / case.filename) == case.original
    raw = (case.output / case.filename).read_bytes()
    assert b"NaN" in raw and b"Infinity" in raw
    assert "execution_receipt" not in case.scientific
    assert case.result["files"] == case.scientific["files"]
    assert case.result["nonfinite_distances"] == 1
    assert case.result["nonfinite_normalized_distances"] == 2
    assert case.result["result_statuses"] == {"success": 1}
    assert case.result["validation"]["state"] == "not_scored"
    json.dumps(case.result, allow_nan=False)
    receipt = _read(case.output / results.RECEIPT)
    json.dumps(receipt, allow_nan=False)
    assert receipt["comparisons"] == case.result["comparisons"]
    assert results.validate_preset_wasserstein_result(case.result, case.task, case.job) is case.result


def test_extraction_keeps_scientific_json_as_verbatim_scalar_table(tmp_path):
    case = _context(tmp_path, nonfinite=True)
    assert "target_machine" not in case.task  # Portable tasks acquire a target only at claim time.
    rows, metadata, accessible = results.extract_preset_wasserstein_products(case.result, case, "fixture")
    assert {row["product_id"]: row["kind"] for row in rows} == {
        case.filename: "scalar_table", "vtk/roi/flux.vtu": "intermediate", results.RECEIPT: "operational_receipt"}
    assert set(metadata) == {case.filename, results.RECEIPT}
    assert accessible[case.filename] == case.output / case.filename


def test_portable_result_is_bound_to_actual_claim_and_publication_machine(tmp_path):
    case = _context(tmp_path)
    case.machine = "another-cpu"
    with pytest.raises(ValueError, match="actual claim service"):
        results.extract_preset_wasserstein_products(case.result, case, "fixture")
    case.machine = "fixture-cpu"
    with pytest.raises(ValueError, match="actual claim service"):
        results.extract_preset_wasserstein_products(case.result, case, "another")


@pytest.mark.parametrize("change", ["original", "vtk", "receipt", "summary", "owner", "attempt", "escape", "time"])
def test_extraction_refuses_changed_bytes_or_claim_identity(tmp_path, change):
    case = _context(tmp_path)
    if change == "original":
        (case.output / case.filename).write_text("{}", encoding="utf-8")
    elif change == "vtk":
        (case.output / "vtk" / "roi" / "flux.vtu").write_bytes(b"different")
    elif change == "receipt":
        _put(case.output / results.RECEIPT, dict(_read(case.output / results.RECEIPT), complete=False))
    elif change == "summary":
        case.result["nonfinite_distances"] = 1
    elif change == "owner":
        case.result["actual_worker"]["machine"] = "other-cpu"
    elif change == "attempt":
        case.result["attempt_id"] = "b" * 32
    elif change == "time":
        case.result["comparison_time"] = dict(order=0, time_hours=99, tolerance_hours=0.1)
    else:
        foreign = _put(tmp_path / case.filename, _read(case.output / case.filename))
        case.result["files"][case.filename] = foreign
        case.result["comparisons"][0]["file"] = foreign
    with pytest.raises(ValueError):
        results.extract_preset_wasserstein_products(case.result, case, "fixture")


def test_publication_factory_accepts_finite_envelope_with_original_nonfinite_scalar_json(tmp_path):
    case = _shared(_context(tmp_path, nonfinite=True), tmp_path)
    candidate = _put(case.output / "candidate_completed.json", case.result)
    publish = make_publication_factory(case.root, case.job, "fixture", producer_origin_for_attempt=None)
    prepared = publish(case, candidate, True)
    assert set(prepared) == {"candidate", "products_index"}
    assert (case.lease / "artifacts" / "metadata" / case.filename).read_bytes() == (case.output / case.filename).read_bytes()


def test_shared_terminal_and_snapshot_never_probe_producer_and_count_records_not_frames(tmp_path, monkeypatch):
    case = _published(tmp_path, nonfinite=True)
    original_stat, original_open = Path.stat, Path.open
    def guarded_stat(path, *args, **kwargs):
        if path.is_relative_to(case.output):
            pytest.fail("Jobs inspected producer-local output")
        return original_stat(path, *args, **kwargs)
    def guarded_open(path, *args, **kwargs):
        if path.is_relative_to(case.output):
            pytest.fail("Jobs opened producer-local output")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    monkeypatch.setattr(Path, "open", guarded_open)
    completed, counted, lineage = _terminal(case.root, case.root / "results" / case.task["run"],
        case.task, case.job, case.done, _read)
    assert completed == case.result and counted == {} and lineage == []
    snapshot = published_snapshot(dict(root=str(case.root)), case.job, {}, reader=_read)
    assert snapshot["completed_comparisons"] == 1 and snapshot["total_comparisons"] == 2
    assert snapshot["completed_runs"] == 1 and snapshot["remaining_comparisons"] == 1
    assert snapshot["work_unit"] == "comparisons" and snapshot["committed"] == 0
    assert snapshot["total_frames"] is None and snapshot["frames_per_minute"] is None
    assert snapshot["nonfinite_distances"] == 1 and snapshot["result_statuses"] == {"success": 1}
    assert snapshot["runs"][0]["comparison_time"] == case.task["comparison_time"]
    assert snapshot["runs"][0]["case_ids"] == [] and not snapshot["errors"]
    from darsia.fluidflower.jobs_execution_display import execution_report_text
    assert snapshot["runs"][0]["machine"] == "fixture"
    assert execution_report_text(snapshot, "fixture").startswith("This computer reported: fixture")


@pytest.mark.parametrize("part", ["owner", "receipt_copy", "scalar_copy", "done_attempt"])
def test_terminal_rejects_shared_tampering_and_snapshot_withholds_result_counts(tmp_path, part):
    case = _published(tmp_path)
    if part == "owner":
        _put(case.lease / "owner.json", dict(case.owner, actual_owner=dict(machine="other-cpu")))
    elif part == "receipt_copy":
        _put(case.lease / "artifacts" / "metadata" / results.RECEIPT, {"complete": False})
    elif part == "scalar_copy":
        _put(case.lease / "artifacts" / "metadata" / case.filename, {"distance": 123})
    else:
        case.done["attempt_id"] = "b" * 32
        _put(case.root / "done" / (case.task["run"] + ".json"), case.done)
    with pytest.raises(ValueError):
        results.preset_wasserstein_terminal(case.root, case.task, case.job, case.done, _read)
    snapshot = published_snapshot(dict(root=str(case.root)), case.job, {}, reader=_read)
    assert snapshot["completed_comparisons"] == 0 and snapshot["errors"]
    assert snapshot["state"] == "Metadata error"


def test_prepared_entry_shows_typed_unexecuted_comparison_counts(tmp_path):
    case = _shared(_context(tmp_path), tmp_path)
    snapshot = prepared_snapshot(dict(root=str(case.root)), case.job, {}, reader=_read)
    assert snapshot["work_unit"] == "comparisons" and snapshot["completed_comparisons"] == 0
    assert snapshot["remaining_comparisons"] == 2 and snapshot["workers"] == []
    assert snapshot["result_statuses"] == {} and not snapshot["errors"]


def test_claimed_comparison_reports_verified_physical_machine_without_liveness_or_progress(tmp_path):
    from darsia.fluidflower.jobs_execution_display import execution_report_text
    case = _shared(_context(tmp_path), tmp_path)
    _put(case.root / "claimed" / "fixture-cpu" / (case.task["run"] + ".json"), case.task)
    snapshot = published_snapshot(dict(root=str(case.root)), case.job, {}, reader=_read)
    assert snapshot["completed_comparisons"] == 0 and snapshot["machines"] == []
    assert snapshot["workers"][0]["machine"] == "fixture"
    assert snapshot["workers"][0]["state"] == "Claimed (liveness not established)"
    assert "fixture-cpu" in snapshot["workers"][0]["worker"]
    assert execution_report_text(snapshot, "fixture").startswith("This computer reported: fixture")
