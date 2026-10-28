"""Actual queue claims and original metadata publication on private files.

The admitted-service and filesystem/clock observations are explicit synthetic
seams. The test result is a metadata-only operation fixture, not a scientific
producer result or native service qualification.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from darsia.fluidflower import job_filesystem
from darsia.fluidflower.pipeline import capacity, job_queue, publication_runtime, store_runtime, worker_io
from darsia.fluidflower.pipeline import series_setup_results as results
from darsia.fluidflower.pipeline.job_inputs import catalog_order_key
from darsia.fluidflower.pipeline.series_setup_sink import current_sink
from .test_series_setup_worker_admission import _runtime


class SyntheticServerClock:
    """Supply a private wall-clock observation without a server qualification."""

    def now_ns(self):
        return time.time_ns()


def _pin(path):
    """Bind an exact private metadata input without a producer call."""
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _queue_context(tmp_path, monkeypatch, *, foreign_target=False):
    """Create real rooted controls, an ordinary queue and its original publisher."""
    monkeypatch.delenv("DARSIA_CACHE_FIRST_QUALIFICATION", raising=False)
    prepared, task, config, owner, _events, _service = _runtime(tmp_path, monkeypatch)
    controls = tmp_path / "controls"
    controls.mkdir()
    rows = prepared.setup_verified_inputs[task["attempt_id"]].rows
    by_use = {use: row for row in rows for use in row.uses}
    references = {}
    fields = (("series", "run.execution_series"), ("series_manifest", "run.series_manifest"),
              ("setup_plan", "setup.plan"))
    for field, use in fields:
        old = by_use[use]
        path = controls / old.path.name
        path.write_bytes(old.path.read_bytes())
        bound = _pin(path)
        references[field] = dict(old.reference, root_alias="configuration", relative_path=path.name,
                                 sha256=bound["sha256"], bytes=bound["size_bytes"])
        if field == "setup_plan":
            config["setup_plan"] = bound
    config["metadata_output"] = dict(maximum_output_bytes=512_000, maximum_file_bytes=128_000)
    operation = controls / "operation.json"
    operation.write_text(json.dumps(config), encoding="utf-8")
    bound = _pin(operation)
    references["configuration"] = dict(task["configuration"], root_alias="configuration",
        relative_path=operation.name, sha256=bound["sha256"], bytes=bound["size_bytes"])
    task.update(deepcopy(references), execution_ready=True, execution_blocker="", threads=1)
    task["input_catalog"].update(configuration=deepcopy(task["configuration"]),
        runtime_support="darsia-role-io-runtime-v1")
    uses = dict(series="run.execution_series", series_manifest="run.series_manifest",
                setup_plan="setup.plan", configuration="run.configuration")
    task["input_catalog"]["inputs"] = sorted(
        [dict(reference=deepcopy(reference), uses=[uses[field]]) for field, reference in references.items()],
        key=lambda row: catalog_order_key(row["reference"]))
    if foreign_target:
        task["target_machine"] = "another-target"
    job = dict(prepared.job, format_version=1, legacy_paths=False, execution_ready=True,
               runtime_support="darsia-role-io-runtime-v1", publish=[],
               setup_plan=deepcopy(task["setup_plan"]))
    roots = dict(prepared.stores.roots(job["series_id"]), configuration=str(controls))
    prepared.stores.roots = lambda _series: dict(roots)
    runtime = object.__new__(worker_io.WorkerRunIO)
    runtime.__dict__.update(prepared.__dict__)
    runtime.__dict__.pop("availability")
    runtime.__dict__.pop("output_factory")
    runtime.job = job
    runtime.setup_verified_inputs, runtime.setup_admissions, runtime.setup_output_scopes = {}, {}, {}
    runtime.bindings, runtime.factories = {}, {}
    proof = job_filesystem.LocalFileAdmission(True, "synthetic-fixed-local", "private test observation",
        "Not a production filesystem or native-service qualification", roots["results"])
    monkeypatch.setattr(store_runtime, "inspect_results_root", lambda _root: proof)
    queue = job_queue.JobQueue(runtime.root, machine_id=runtime.stores.machine_id,
                              server_clock=SyntheticServerClock())
    (queue.root / "queue" / (task["run"] + ".json")).rename(queue.root / "prepared-input-task.json")
    job_queue.write_immutable(queue.root / "job.json", job)
    pending = queue.enqueue(task)
    queued = job_queue.read_json(pending)

    def no_prefix(_attempt):
        raise AssertionError("Setup cannot substitute an analysis frame prefix")

    publisher = publication_runtime.make_publication_factory(queue.root, job, runtime.stores.machine_id,
        producer_origin_for_attempt=no_prefix)
    return SimpleNamespace(runtime=runtime, queue=queue, task=queued, config=config,
                           owner=owner, pending=pending, publisher=publisher, controls=controls)


def _claim(context):
    """Use the actual atomic queue rename, finite controls and whole-cap bind."""
    runtime = context.runtime
    with runtime.capacity_claim(context.owner) as candidate:
        claim = context.queue.claim(runtime.service_id, runtime.backend, context.owner,
            availability=candidate.availability, output_factory=runtime.output_factory,
            publication_factory=context.publisher, claim_controls=candidate.claim_controls)
        candidate.bind(claim)
    return claim


def _operation_result(context, claim):
    """Write the explicitly synthetic typed receipt through its actual sink."""
    document = dict(format=results.RESULT_FORMAT, pipeline="series-setup", complete=True,
        run=claim.task["run"], operation=claim.task["operation"], backend="cpu", precision="float32",
        setup_plan=deepcopy(context.config["setup_plan"]), worker_references_closed=True,
        scientific_calls=0, production_qualification_claimed=False)
    path = claim.output / results.RECEIPT
    job_queue.write_immutable(path, document)
    return dict(document, files={results.RECEIPT: _pin(path)}, task=deepcopy(claim.task),
                attempt_id=claim.token, commit=claim.task["commit"], actual_worker=deepcopy(context.owner))


def test_real_claim_publication_and_done_share_one_bounded_metadata_allowance(tmp_path, monkeypatch):
    context = _queue_context(tmp_path, monkeypatch)
    claim = _claim(context)
    runtime, queue = context.runtime, context.queue
    helper = runtime.setup_admissions[claim.token]
    actual_owner = job_queue.read_json(claim.lease / "owner.json")
    assert actual_owner["task"] == claim.task and actual_owner["actual_owner"] == context.owner
    assert isinstance(claim.local_output, job_queue.LocalOutputBinding)
    assert not context.pending.exists() and claim.claimed.is_file()
    controls_bytes = helper.sink.written_bytes
    assert controls_bytes > 0
    with runtime.output_scope(claim) as sink:
        assert current_sink() is sink
        result = _operation_result(context, claim)
        queue.complete(claim, result)
        done = job_queue.read_json(queue.root / "done" / (claim.task["run"] + ".json"))

        def metadata_reader(path, expected=None):
            """Keep the real snapshot reader's optional exact descriptor check."""
            if expected is not None:
                assert job_queue.descriptor(path) == expected
            return job_queue.read_json(path)

        completed, frame_progress, _lineage = results.setup_terminal(
            queue.root, claim.task, runtime.job, done, metadata_reader)
        assert completed == result and frame_progress == {}
        assert sink.written_bytes > controls_bytes
        written = {Path(row["path"]) for row in sink.evidence()["files"]}
        assert claim.lease / "owner.json" in written
        assert claim.output / "candidate_completed.json" in written
        assert queue.root / "done" / (claim.task["run"] + ".json") in written
        assert not (queue.root / "failed" / (claim.task["run"] + ".json")).exists()
    runtime.release(claim)
    assert all((Path(row["path"]) / "released.json").is_file()
               for row in helper.admission.bound_reservations)
    assert helper.sink.closed and job_queue.read_json(queue.root / "done" / (claim.task["run"] + ".json")) == done


def test_actual_failure_publication_retains_caps_until_exact_owner_retirement(tmp_path, monkeypatch):
    context = _queue_context(tmp_path, monkeypatch)
    claim = _claim(context)
    runtime, queue = context.runtime, context.queue
    helper = runtime.setup_admissions[claim.token]
    with runtime.output_scope(claim):
        queue.fail(claim, "private operation failure; original partial retained", context.owner)
    runtime.release(claim, retain_capacity=True)
    assert (queue.root / "failed" / (claim.task["run"] + ".json")).is_file()
    assert not (queue.root / "done" / (claim.task["run"] + ".json")).exists()
    assert all(not (Path(row["path"]) / "released.json").exists()
               for row in helper.admission.bound_reservations)
    supervisor = dict(machine=runtime.service_id, pid=999, process_create_time="999.125")
    proof = dict(complete=True, all_absent=True, worker_os_references_closed=True,
        run=claim.task["run"], attempt_id=claim.token, supervisor=supervisor,
        identities=[dict(context.owner, pid=456)])
    with pytest.raises(ValueError, match="reservation owner"):
        capacity.release_retired_capacity(queue.root, claim.task, runtime.stores, proof)
    assert all(not (Path(row["path"]) / "released.json").exists()
               for row in helper.admission.bound_reservations)
    proof["identities"] = [deepcopy(context.owner)]
    released = capacity.release_retired_capacity(queue.root, claim.task, runtime.stores, proof)
    assert set(released) == {row["path"] for row in helper.admission.bound_reservations}
    assert (claim.output / "failed.json").is_file()


def test_other_target_skips_actual_queue_before_any_foreign_control_is_opened(tmp_path, monkeypatch):
    context = _queue_context(tmp_path, monkeypatch, foreign_target=True)
    before = context.pending.read_bytes()
    original_open = Path.open

    def guarded_open(path, mode="r", *args, **kwargs):
        if path.is_relative_to(context.controls):
            raise AssertionError("A different target must not open these physical controls")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    assert _claim(context) is None
    assert context.pending.read_bytes() == before
    assert not context.runtime.setup_admissions and not context.runtime.setup_verified_inputs
    assert not list((context.queue.root / "leases").iterdir())
