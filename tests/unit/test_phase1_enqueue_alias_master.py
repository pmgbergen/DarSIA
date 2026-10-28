"""Tiny metadata/protocol tests only; no campaign, source processing or worker.

The builder tests use a named catalogue protocol double: original typed DATA,
map and generic task validators and immutable queue writes are real; catalogue
payload/role traversal and real network-filesystem admission are NOT qualified.
Master tests use original local publication/index/queue methods and explicit
synthetic ownership. Fake PID123 is a fixture, never OS-retirement evidence.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import job_filesystem, published_jobs_snapshot as snapshot
from darsia.fluidflower.pipeline import alias_job, enqueue, job_queue, phase1_protocol as protocol
from darsia.fluidflower.pipeline import publication_runtime
from darsia.fluidflower.pipeline.job_inputs import catalog_order_key
from darsia.fluidflower.pipeline.store_runtime import ResolvedInput
from .test_phase1_worker_protocol import metadata, scientific_fixture, SyntheticServerClock
from .test_staged_data import _write


def read(path, expected=None):
    actual = job_queue.descriptor(path)
    if expected is not None:
        assert (actual["sha256"], actual["size_bytes"]) == (expected["sha256"], expected.get("bytes", expected.get("size_bytes")))
    value = job_queue.read_json(Path(path))
    assert job_queue.descriptor(path) == actual
    return value



def _overwrite_test_owned_json(test_root, path, value):
    """Only intentional test-fixture forgeries may overwrite owned JSON bytes.

    This is never a producer/writer policy. The immutable original _write
    remains unchanged for every ordinary new fixture creation.
    """
    root = Path(test_root).resolve(strict=True)
    target = Path(path).resolve(strict=True)
    assert root.is_dir() and target.is_file()
    assert target.is_relative_to(root) and target != root
    assert target.suffix == ".json"
    raw = (json.dumps(value, allow_nan=False) + "\n").encode("utf-8")
    target.write_bytes(raw)
    assert target.resolve(strict=True).is_relative_to(root)
    return job_queue.descriptor(target)


def owned_publication(tmp_path, *, complete=False):
    data, raw, job, task = metadata()
    queue = job_queue.JobQueue(tmp_path / "shared", machine_id="main", server_clock=SyntheticServerClock())
    job_queue.write_immutable(queue.root / "job.json", job)
    asset = tmp_path / "input.json"
    asset.write_bytes(raw)
    queue.enqueue(task)
    def availability(value):
        return tuple(ResolvedInput(deepcopy(row["reference"]), asset, tuple(row["uses"])) for row in value["input_catalog"]["inputs"])
    def output_factory(value, token):
        root = tmp_path / "local-results"
        output = root / value["job_id"] / value["run"] / ".attempts" / token
        proof = job_filesystem.LocalFileAdmission(True, "synthetic-local", "unit fixture", "No production filesystem qualification", str(tmp_path))
        return job_queue.LocalOutputBinding("main", root, output, proof)
    def no_prefix(_attempt):
        raise AssertionError("Fresh phase one must not request a prefix origin")
    publisher = publication_runtime.make_publication_factory(queue.root, job, "main", producer_origin_for_attempt=no_prefix)
    owner = dict(machine="main", pid=123, process_create_time="synthetic-unit-epoch")
    claim = queue.claim("main", "cpu", owner, availability=availability, output_factory=output_factory, publication_factory=publisher)
    closure, base = scientific_fixture(claim, data)
    result = protocol.worker_result(base, task=claim.task, job=job, commit=task["commit"], owner=owner, token=claim.token)
    event = dict(state="phase1_frame_completed", stage=protocol.PIPELINE, complete=True, run=task["run"],
        case_id=data["ordered_case_ids"][0], frame_index=data["frames"][0]["frame_index"], phase1_commit=closure["frame"])
    queue.progress(claim, event)
    if complete:
        queue.complete(claim, result)
    return SimpleNamespace(queue=queue, root=queue.root, job=job, task=claim.task, claim=claim, result=result, event=event, test_root=tmp_path)


def value(data):
    return snapshot.published_snapshot({"root": str(data.root)}, data.job, {}, reader=read)


def test_real_shared_phase1_progress_is_counted_without_mass_fields(tmp_path):
    data = owned_publication(tmp_path)
    actual = value(data)
    assert actual["committed"] == 1 and actual["complete"] is False
    assert actual["runs"][0]["case_ids"] == data.task["ordered_case_ids"]
    assert actual["errors"] == []


def test_real_phase1_terminal_index_and_original_core_are_read_without_foreign_files(tmp_path):
    data = owned_publication(tmp_path, complete=True)
    actual = value(data)
    assert actual["complete"] is True and actual["committed"] == 1
    assert actual["runs"][0]["state"] == "Complete" and actual["errors"] == []
    assert "calibration_sha256" not in data.result and "records" not in data.result


@pytest.mark.parametrize("state", ["frame_started", "input_verified", "copy_started"])
def test_phase1_nonframe_lifecycle_events_remain_ignored(tmp_path, state):
    _write = lambda path, value: _overwrite_test_owned_json(tmp_path, path, value)
    data = owned_publication(tmp_path)
    path = data.claim.lease / "progress_00000000.json"
    row = read(path)
    row["event"] = {"state": state, "run": data.task["run"]}
    _write(path, row)
    assert snapshot._events(data.claim.lease, data.task, read) == []


def test_mass_frame_event_in_positive_phase1_is_rejected_before_mass_reader(tmp_path):
    _write = lambda path, value: _overwrite_test_owned_json(tmp_path, path, value)
    data = owned_publication(tmp_path)
    path = data.claim.lease / "progress_00000000.json"
    row = read(path)
    row["event"]["state"] = "frame_completed"
    _write(path, row)
    with pytest.raises(ValueError, match="Mass frame progress"):
        snapshot._events(data.claim.lease, data.task, read)


@pytest.mark.parametrize("field,bad", [("case_id", "other"), ("frame_index", True), ("precision", "float64"), ("format", "Main")])
def test_hash_valid_shared_progress_still_needs_positive_commit_identity(tmp_path, field, bad):
    _write = lambda path, value: _overwrite_test_owned_json(tmp_path, path, value)
    data = owned_publication(tmp_path)
    progress_path = data.claim.lease / "progress_00000000.json"
    progress = read(progress_path)
    path = job_queue._own_asset(data.claim.lease, progress["published_phase1_frame"])
    record = read(path)
    record[field] = bad
    changed = _write(path, record)
    progress["published_phase1_frame"] = job_queue._own_reference(data.claim.lease, path)
    progress["event"]["phase1_commit"].update(sha256=changed["sha256"], size_bytes=changed["size_bytes"])
    _write(progress_path, progress)
    with pytest.raises(ValueError, match="Shared phase-one commit"):
        snapshot._events(data.claim.lease, data.task, read)


def test_named_commit_product_ID_cannot_be_renamed_under_valid_index_SHA(tmp_path):
    _write = lambda path, value: _overwrite_test_owned_json(tmp_path, path, value)
    data = owned_publication(tmp_path, complete=True)
    done_path = data.root / "done" / (data.task["run"] + ".json")
    done = read(done_path)
    index_path = job_queue._own_asset(data.root, done["products_index"])
    index = read(index_path)
    row = next(row for row in index["products"] if row["product_id"].endswith("/phase1_commit.json"))
    old_copy = index_path.parent / row["metadata_copy"]["relative_path"]
    row["product_id"] = row["product_id"].replace("phase1_commit.json", "renamed.json")
    row["metadata_copy"]["relative_path"] = "metadata/" + row["product_id"]
    new_copy = index_path.parent / row["metadata_copy"]["relative_path"]
    new_copy.parent.mkdir(parents=True, exist_ok=True)
    new_copy.write_bytes(old_copy.read_bytes())
    _write(index_path, index)
    done["products_index"] = job_queue._own_reference(data.root, index_path)
    _write(done_path, done)
    actual = value(data)
    assert actual["state"] == "Metadata error" and actual["complete"] is False


def builder_inputs(tmp_path, monkeypatch):
    data, _raw, _job, old = metadata()
    config = _write(tmp_path / "phase1.json", data)
    parent = _write(tmp_path / "parent.json", {"synthetic_provenance_only": True})
    manifest = tmp_path / "series.toml"
    manifest.write_bytes(b'schema_version = 1\nseries_id = "Synthetic"\n[roots]\nseries = "."\n')
    bound_manifest = job_queue.descriptor(manifest)
    calls = []
    class CatalogueProtocol:
        """Only a catalogue ABI double, never a proof of actual role traversal."""
        def __init__(self, roots, aliases=()):
            self.rows = {}
        def add(self, bound, role, use):
            actual = job_queue.descriptor(bound["path"])
            assert actual["sha256"] == bound["sha256"]
            reference = dict(root_alias="series", relative_path=Path(bound["path"]).name, role=role,
                sha256=actual["sha256"], bytes=actual["size_bytes"])
            row = self.rows.setdefault(reference["relative_path"], dict(reference=reference, uses=[]))
            row["uses"].append(use)
            return deepcopy(reference)
        def doc(self, bound, role, use):
            self.add(bound, role, use)
            return read(bound["path"], bound)
        def collect(self, bound, current):
            calls.append(deepcopy(current))
            reference = self.add(bound, "configuration", "run.configuration")
            return dict(format="darsia-role-input-catalog-v1", complete=True, run=current["run"],
                configuration=reference,
                inputs=sorted(self.rows.values(), key=lambda row: catalog_order_key(row["reference"])))
    monkeypatch.setattr(alias_job, "shared_output_admission", lambda root: object())
    job = dict(format=alias_job.JOB_FORMAT, pipeline=protocol.PIPELINE, commit=old["commit"],
        precision=old["precision"], expected_runs=[old["run"]], publish=[], created_utc="synthetic-unit-fixture",
        environment_lock={"path": "requirements.txt", "sha256": "a" * 64})
    task = {key: deepcopy(old[key]) for key in ("format", "pipeline", "run", "commit", "precision", "job_id", "backend", "threads")}
    task["configuration"] = config
    return SimpleNamespace(job=job, tasks=[task], root=tmp_path / "new-job", parent=parent,
        manifest=bound_manifest, factory=CatalogueProtocol, calls=calls, data=data)


def prepare(inputs):
    return alias_job.prepare_phase1_alias_job(inputs.job, inputs.tasks, inputs.root,
        execution_series=inputs.parent, series_manifest=inputs.manifest, catalog_factory=inputs.factory)


def test_positive_builder_keeps_supplied_typed_DATA_and_uses_original_queue_validation(tmp_path, monkeypatch):
    inputs = builder_inputs(tmp_path, monkeypatch)
    original = deepcopy(inputs.data)
    result = prepare(inputs)
    assert result["complete"] is True and result["numerical_calls_started"] is False
    assert inputs.calls == [original] and inputs.data == original
    job = read(inputs.root / "job.json")
    task = read(inputs.root / "results" / inputs.tasks[0]["run"] / "task.json")
    job_queue.validate_task(task)
    protocol.validate_phase1_claim(task, job)
    assert "calibration_sha256" not in job and "calibration_sha256" not in task


@pytest.mark.parametrize("bad", [None, {}, {"path": None, "sha256": "a" * 64}, {"path": "", "sha256": "a" * 64},
    {"path": "requirements.txt", "sha256": "A" * 64}, {"path": "../requirements.txt", "sha256": "a" * 64}])
def test_builder_bad_lock_metadata_stops_before_any_immutable_job_write(tmp_path, monkeypatch, bad):
    inputs = builder_inputs(tmp_path, monkeypatch)
    inputs.job["environment_lock"] = bad
    with pytest.raises(ValueError):
        prepare(inputs)
    assert not inputs.root.exists()


@pytest.mark.parametrize("field,bad", [("backend", "other"), ("threads", 0), ("threads", True), ("commit", "2" * 40)])
def test_builder_bad_task_metadata_stops_before_any_immutable_job_write(tmp_path, monkeypatch, field, bad):
    inputs = builder_inputs(tmp_path, monkeypatch)
    inputs.tasks[0][field] = bad
    with pytest.raises(ValueError):
        prepare(inputs)
    assert not inputs.root.exists()


def test_CLI_positive_task_has_no_fake_K_and_uses_existing_enqueue(tmp_path, monkeypatch):
    _data, _raw, job, task = metadata()
    root = tmp_path / "job"
    root.mkdir()
    _write(root / "job.json", job)
    tasks = _write(tmp_path / "tasks.json", [task])
    # This unit test deliberately does not claim a real UNC transport gate.
    monkeypatch.setattr(enqueue, "require_unc", lambda path: None)
    captured = []
    class ExistingQueueProtocol:
        def __init__(self, path): assert path == root
        def enqueue(self, value):
            job_queue.validate_task(value)
            protocol.validate_phase1_claim(value, job)
            captured.append(deepcopy(value))
            return root / "queue" / "run-one.json"
    monkeypatch.setattr(enqueue, "JobQueue", ExistingQueueProtocol)
    enqueue.main(["--job", str(root), "--tasks", tasks["path"]])
    assert len(captured) == 1 and "calibration_sha256" not in captured[0]


def rewrite_shared_index(data, update):
    """Test-only byte-valid index forgery, never a new producer receipt."""
    _write = lambda path, value: _overwrite_test_owned_json(data.test_root, path, value)
    done_path = data.root / "done" / (data.task["run"] + ".json")
    done = read(done_path)
    path = job_queue._own_asset(data.root, done["products_index"])
    index = read(path)
    update(index, path, done)
    _write(path, index)
    done["products_index"] = job_queue._own_reference(data.root, path)
    _write(done_path, done)


def test_hash_valid_other_manifest_job_id_is_not_a_phase1_completion(tmp_path):
    _write = lambda path, value: _overwrite_test_owned_json(tmp_path, path, value)
    data = owned_publication(tmp_path, complete=True)
    def forge(index, index_path, done):
        manifest = next(row for row in index["products"] if row["product_id"] == "manifest.json")
        path = index_path.parent / manifest["metadata_copy"]["relative_path"]
        document = read(path)
        document["job_id"] = "other-explicit-job"
        changed = _write(path, document)
        for bound in (manifest, manifest["metadata_copy"]):
            bound.update(sha256=changed["sha256"], bytes=changed["size_bytes"])
        completion = next(row for row in index["products"] if row["product_id"] == "completed.json")
        copied = index_path.parent / completion["metadata_copy"]["relative_path"]
        result = read(copied)
        result["manifest"].update(sha256=changed["sha256"], size_bytes=changed["size_bytes"])
        bound = _write(copied, result)
        canonical = data.root / "results" / data.task["run"] / "completed.json"
        assert _write(canonical, result)["sha256"] == bound["sha256"]
        for value in (completion, completion["metadata_copy"]):
            value.update(sha256=bound["sha256"], bytes=bound["size_bytes"])
        done["completed"] = job_queue._own_reference(data.root, canonical)
    rewrite_shared_index(data, forge)
    actual = value(data)
    assert actual["state"] == "Metadata error" and actual["complete"] is False
    assert "manifest/code/settings/configuration" in actual["errors"][0]


@pytest.mark.parametrize("part", ["job", "run", "token"])
def test_hash_valid_coherently_relocated_index_stops_before_any_original_metadata_read(tmp_path, monkeypatch, part):
    data = owned_publication(tmp_path, complete=True)
    original = str(data.claim.output)
    path = Path(original)
    if part == "token":
        replacement = path.parent / ("f" * 32)
    elif part == "run":
        replacement = path.parent.parent.parent / "other-run" / ".attempts" / path.name
    else:
        replacement = path.parent.parent.parent.parent / "other-job" / path.parent.parent.name / ".attempts" / path.name
    def forge(index, _index_path, _done):
        for row in index["products"]:
            assert row["local_path"].startswith(original)
            row["local_path"] = str(replacement) + row["local_path"][len(original):]
    rewrite_shared_index(data, forge)
    # Every indexed producer path is coherently moved and all shared byte hashes
    # remain valid. The namespace boundary must reject it BEFORE trying to read
    # or interpret a manifest/commit. This is not a new fully resealed producer.
    def forbidden(*args, **kwargs):
        raise AssertionError("Namespace rejection must precede original metadata reads")
    monkeypatch.setattr(snapshot, "_original_metadata", forbidden)
    actual = value(data)
    assert actual["state"] == "Metadata error" and actual["complete"] is False
    assert "outside its genuine job/run/attempt namespace" in actual["errors"][0]


@pytest.mark.parametrize("lock_path", ["/requirements.txt", "\\requirements.txt"])
def test_builder_drive_less_rooted_lock_stops_before_immutable_writes(tmp_path, monkeypatch, lock_path):
    inputs = builder_inputs(tmp_path, monkeypatch)
    inputs.job["environment_lock"]["path"] = lock_path
    with pytest.raises(ValueError, match="repository-relative"):
        prepare(inputs)
    assert not inputs.root.exists()


def test_CLI_original_calibrated_path_keeps_existing_key_defaulting(tmp_path, monkeypatch):
    _data, _raw, _positive_job, task = metadata()
    task = deepcopy(task)
    task.pop(protocol.MAP_SHA_FIELD)
    task["pipeline"] = "n1"
    for key in ("commit", "precision"):
        task.pop(key)
    root = tmp_path / "calibrated-job"
    root.mkdir()
    job = dict(commit="1" * 40, precision="float32", calibration_sha256="b" * 64)
    _write(root / "job.json", job)
    tasks = _write(tmp_path / "calibrated-tasks.json", [task])
    monkeypatch.setattr(enqueue, "require_unc", lambda path: None)
    captured = []
    class ExistingQueueProtocol:
        def __init__(self, path): assert path == root
        def enqueue(self, current):
            job_queue.validate_task(current)
            captured.append(deepcopy(current))
            return root / "queue" / "run-one.json"
    monkeypatch.setattr(enqueue, "JobQueue", ExistingQueueProtocol)
    enqueue.main(["--job", str(root), "--tasks", tasks["path"]])
    assert len(captured) == 1
    assert all(captured[0][key] == job[key] for key in ("commit", "precision", "calibration_sha256"))
