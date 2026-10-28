"""Synthetic queue/identity bytes only; never execute native, phase or statistics.

The bound support module supplies its already-reviewed synthetic metadata and
undecoded tiny payload bytes. These records are unit fixtures, not Main proofs.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import job_filesystem
from darsia.fluidflower.pipeline import phase1_protocol as protocol
from darsia.fluidflower.pipeline import publication_runtime, job_queue
from darsia.fluidflower.pipeline.store_runtime import ResolvedInput, RunUnavailable
from .test_staged_data import phase1_data, _closure, _write


def metadata():
    data = phase1_data.__wrapped__()
    data["run"] = "run-one"
    raw = (json.dumps(data, allow_nan=False) + "\n").encode()
    ref = dict(root_alias="series", relative_path="phase1.json", role="configuration",
               sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    task = dict(format="darsia-alias-task-v1", format_version=1, legacy_paths=False,
        execution_ready=True, pipeline=protocol.PIPELINE, run=data["run"], job_id="synthetic-phase1",
        commit="1" * 40, precision="float32", backend="cpu", threads=1,
        configuration=ref, series=deepcopy(ref), series_manifest=deepcopy(ref),
        ordered_case_ids=data["ordered_case_ids"],
        input_catalog=dict(format="darsia-role-input-catalog-v1", complete=True,
            run=data["run"], configuration=ref, runtime_support="darsia-role-io-runtime-v1",
            inputs=[dict(reference=ref, uses=["run.configuration", "run.execution_series", "run.series_manifest"])]))
    job = dict(format="darsia-alias-job-v1", format_version=1, legacy_paths=False,
        execution_ready=True, runtime_support="darsia-role-io-runtime-v1", series_id="Synthetic",
        commit=task["commit"], precision="float32", expected_runs=[task["run"]], publish=[])
    job, tasks = protocol.bind_phase1_identity(job, [task])
    return data, raw, job, tasks[0]


def scientific_fixture(claim, data):
    """Create only explicitly synthetic bytes; no producer/API numerical call."""
    closure = _closure(claim.output, data)
    task = claim.task
    config = closure["configuration"]
    manifest = dict(format=protocol.MANIFEST_FORMAT, job_id=task["job_id"], run=task["run"],
        configuration=config, code=dict(commit=task["commit"], identity_algorithm="sha256-git-commit-ascii"),
        commit_identity_digest=hashlib.sha256(task["commit"].encode("ascii")).hexdigest(),
        native_code_admission=dict(entries=[dict(commit=task["commit"], role="synthetic-not-real-code-admission")]),
        settings=dict(backend="cpu", precision="float32", threads=task["threads"]),
        stage_identity=dict(pipeline=protocol.PIPELINE, configuration=task["configuration"],
            run=task["run"], commit=task["commit"], precision=task["precision"], backend="cpu",
            threads=task["threads"], attempt_id=claim.token,
            phase1_configurations_sha256=task[protocol.MAP_SHA_FIELD]),
        frames={row["case_id"]: row for row in data["frames"]})
    manifest_asset = _write(claim.output / "manifest.json", manifest)
    base = dict(format=protocol.RESULT_FORMAT, complete=True, whole_run=True,
        pipeline=protocol.PIPELINE, run=task["run"], backend="cpu", precision="float32",
        case_count=len(task["ordered_case_ids"]), ordered_case_ids=task["ordered_case_ids"],
        phase1_completion=closure["completion"], phase1_frames=[closure["frame"]], manifest=manifest_asset,
        configuration=config, worker_references_closed=True, fitted_mass_calibration_read=False,
        mass_evaluation_calls=0)
    return closure, base


class SyntheticServerClock:
    def now_ns(self):
        return 7_000_000_000


def claimed_fixture(tmp_path):
    data, raw, job, task = metadata()
    queue = job_queue.JobQueue(tmp_path / "shared", machine_id="Main", server_clock=SyntheticServerClock())
    job_queue.write_immutable(queue.root / "job.json", job)
    source = tmp_path / "input.json"
    source.write_bytes(raw)
    queue.enqueue(task)

    def availability(value):
        return tuple(ResolvedInput(deepcopy(row["reference"]), source, tuple(row["uses"]))
                     for row in value["input_catalog"]["inputs"])

    def output_factory(value, token):
        root = tmp_path / "local-results"
        output = root / value["job_id"] / value["run"] / ".attempts" / token
        admission = job_filesystem.LocalFileAdmission(True, "synthetic-local", "unit fixture",
            "No production filesystem qualification is claimed", str(tmp_path))
        return job_queue.LocalOutputBinding("Main", root, output, admission)

    def no_prefix(_attempt):
        raise AssertionError("Fresh-only phase one must not request a prefix origin")

    publisher = publication_runtime.make_publication_factory(queue.root, job, "Main", producer_origin_for_attempt=no_prefix)
    claim = queue.claim("Main", "cpu", {"pid": 123, "test_fixture_only": True},
        availability=availability, output_factory=output_factory, publication_factory=publisher)
    closure, base = scientific_fixture(claim, data)
    result = protocol.worker_result(base, task=claim.task, job=job, commit=task["commit"],
        owner=dict(pid=123, test_fixture_only=True), token=claim.token)
    event = dict(state="phase1_frame_completed", stage=protocol.PIPELINE, complete=True,
        run=task["run"], case_id=data["ordered_case_ids"][0], frame_index=data["frames"][0]["frame_index"],
        compute_seconds=0., phase1_commit=closure["frame"])
    return SimpleNamespace(queue=queue, job=job, claim=claim, result=result, closure=closure, event=event)


def test_explicit_map_identity_contains_no_fake_calibration_and_preserves_inputs():
    _data, _raw, job, task = metadata()
    assert "calibration_sha256" not in job and "calibration_sha256" not in task
    assert protocol.configuration_map_sha(job[protocol.MAP_FIELD]) == task[protocol.MAP_SHA_FIELD]
    assert task["configuration"] == job[protocol.MAP_FIELD][task["run"]]
    job_queue.validate_task(task)


@pytest.mark.parametrize("change", ["calibration", "commit", "map", "precision", "configuration"])
def test_exact_job_task_stage_identity_rejects_cross_binding(change):
    _data, _raw, job, task = metadata()
    if change == "calibration": task["calibration_sha256"] = "a" * 64
    elif change == "commit": task["commit"] = "2" * 40
    elif change == "map": task[protocol.MAP_SHA_FIELD] = "3" * 64
    elif change == "precision": task["precision"] = "float64"
    else: task["configuration"] = dict(task["configuration"], bytes=True)
    with pytest.raises(ValueError): protocol.validate_phase1_claim(task, job)


def test_default_calibrated_task_guard_remains_required():
    _data, _raw, _job, task = metadata()
    task["pipeline"] = "native-phase-n1"
    with pytest.raises(ValueError, match="calibration_sha256"):
        job_queue.validate_task(task)
    task["calibration_sha256"] = "a" * 64
    job_queue.validate_task(task)


def test_phase1_resume_calls_original_selector_and_refuses_existing_science(monkeypatch):
    from darsia.fluidflower.pipeline import checkpoint
    calls = []
    monkeypatch.setattr(checkpoint, "resume_source_for_claim", lambda *a: calls.append(a) or dict(attempt_directory="unit-only"))
    with pytest.raises(RunUnavailable, match="preserve the prefix without replay"):
        protocol.phase1_resume_source("original-shared", "original-claim")
    assert calls == [("original-shared", "original-claim")]


def test_fresh_original_lineage_is_still_none(tmp_path):
    f = claimed_fixture(tmp_path)
    assert protocol.phase1_resume_source(f.queue, f.claim) is None


def test_progress_uses_actual_server_clock_and_exact_shared_commit_bytes(tmp_path):
    f = claimed_fixture(tmp_path)
    f.queue.progress(f.claim, f.event)
    record = job_queue.read_json(f.claim.lease / "progress_00000000.json")
    assert record["written_unix_ns"] == 7_000_000_000
    assert "published_frame" not in record
    ref = record["published_phase1_frame"]
    copied = job_queue._own_asset(f.claim.lease, ref)
    assert copied.read_bytes() == Path(f.closure["frame"]["path"]).read_bytes()
    assert copied.name == "phase1_commit.json"


def test_actual_missing_lease_fences_new_progress_before_write(tmp_path):
    f = claimed_fixture(tmp_path)
    moved = f.queue.root / "reclaimed" / "preserved-test-lease"
    job_queue.rename(f.claim.lease, moved)
    with pytest.raises(job_queue.LeaseLost): f.queue.progress(f.claim, f.event)
    assert not list(f.claim.output.glob("progress_*.json"))


@pytest.mark.parametrize("change", ["case", "index", "outside", "sha"])
def test_progress_requires_original_case_type_owner_and_bytes(tmp_path, change):
    f = claimed_fixture(tmp_path)
    event = deepcopy(f.event)
    if change == "case": event["case_id"] = "foreign-case"
    elif change == "index": event["frame_index"] = True
    elif change == "outside": event["phase1_commit"]["path"] = str(tmp_path / "foreign.json")
    else: event["phase1_commit"]["sha256"] = "0" * 64
    with pytest.raises(ValueError): protocol.phase1_frame_commit(f.claim, event)


def test_completion_indexes_every_original_stage_row_without_main_fields_or_payload_reads(tmp_path, monkeypatch):
    f = claimed_fixture(tmp_path)
    payloads = {bound["path"] for leaf, bound in f.closure["record"]["files"].items()
                if Path(leaf).suffix.lower() in {".png", ".npz", ".npy"}}
    from darsia.fluidflower.pipeline import products_publication
    original = products_publication._file
    def no_payload(path):
        assert str(path) not in payloads, "Indexing must not load native/phase/photo payloads"
        return original(path)
    monkeypatch.setattr(products_publication, "_file", no_payload)
    f.queue.complete(f.claim, f.result)
    done = job_queue.read_json(f.queue.root / "done" / "run-one.json")
    completed = job_queue.read_json(job_queue._own_asset(f.queue.root, done["completed"]))
    assert completed["format"] == protocol.RESULT_FORMAT
    assert not {"calibration_sha256", "records", "table", "frame_producers"} & set(completed)
    index = job_queue.read_json(job_queue._own_asset(f.queue.root, done["products_index"]))
    expected = {"phase1/case-one/" + leaf for leaf in f.closure["record"]["files"]}
    expected |= {"phase1/case-one/phase1_commit.json", "manifest.json", "phase1_completed.json", "completed.json"}
    assert {row["product_id"] for row in index["products"]} == expected
    assert index["file_closure"]["os_retirement_asserted"] is False


@pytest.mark.parametrize("change", ["case_count", "precision", "manifest_code", "core_order"])
def test_genuine_completion_cannot_substitute_another_core_or_producer(tmp_path, change):
    f = claimed_fixture(tmp_path)
    result = deepcopy(f.result)
    if change == "case_count": result["case_count"] = True
    elif change == "precision": result["precision"] = "float64"
    elif change == "manifest_code": result["commit"] = "2" * 40
    else: result["phase1_frames"] = []
    with pytest.raises(ValueError):
        protocol.extract_phase1_products(result, f.claim, "Main")


def test_manifest_actual_requested_threads_cannot_differ_under_valid_outer_SHA(tmp_path):
    """One own synthetic metadata mutation; no producer or OS proof is claimed."""
    f = claimed_fixture(tmp_path)
    path = Path(f.claim.output) / "manifest.json"
    before = job_queue.descriptor(path)
    assert protocol._manifest(f.claim, before)["settings"]["threads"] == f.claim.task["threads"]
    value = json.loads(path.read_bytes())
    value["settings"]["threads"] = f.claim.task["threads"] + 1
    path.write_bytes((json.dumps(value, sort_keys=True) + "\n").encode())
    actual = job_queue.descriptor(path)
    assert actual["sha256"] != before["sha256"]
    assert actual["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual["size_bytes"] == path.stat().st_size
    with pytest.raises(ValueError, match="Phase-one manifest differs from the actual claimed stage"):
        protocol._manifest(f.claim, actual)
