"""Live worker-close predicates over tiny owned publications and original DATA.

The scientific inputs use the original P1 verifier, genuine tiny refit and
held-out application join. Native handles, immutable slot transport and the
resource estimate factory are synthetic unit boundaries, never host proof.
No scientific producer or subprocess is started by these tests.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import common_worker_closure as closure
from darsia.fluidflower.pipeline import job_queue, publication_runtime, worker_io
from darsia.fluidflower.pipeline.run_io import current_binding
from darsia_cache import bootstrap, common_qualification as common, owned_process
from darsia_cache.policy import CachePolicyError
from .test_common_profile_paths import real_catalogue_join, real_common_claim  # noqa: F401
from .test_common_qualification import provider


def write(path, value):
    """Create one new tiny fixture artifact with its actual byte descriptor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, allow_nan=False).encode()
    with path.open("xb") as stream:
        stream.write(raw)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def publish_original_result(claim, root, job, owner, control):
    """Use the unchanged publisher/index and original terminal receipt layout."""
    claim.output = claim.local_output.path
    claim.lease = root / "leases" / (claim.task["run"] + "." + claim.token)
    task = claim.task
    settings = dict(backend=task["backend"], precision=task["precision"], threads=task["threads"])
    manifest = write(claim.output / "manifest.json", dict(run=task["run"], configuration=task["configuration"],
        code=dict(commit=task["commit"]), settings=settings))
    records, producers = [], []
    for case in task["ordered_case_ids"]:
        final = claim.output / "frames" / case
        scalar = write(final / "result.json", dict(complete=True, run=task["run"], case_id=case,
            precision=task["precision"], pixel_backend_facts=dict(opencl_n1=dict(
                control=deepcopy(control), used=False, candidates=[], selected_device=None))))
        maps = write(final / "products.npz", b"synthetic undecoded pixel bytes")
        files = {Path(pin["path"]).name: {key: item for key, item in pin.items() if key != "path"}
                 for pin in (scalar, maps)}
        commit = write(final / "commit.json", dict(complete=True, frame_id=case, commit_id="c" * 32,
            manifest_sha256=manifest["sha256"], settings=settings,
            products=dict(precision=task["precision"], backend=task["backend"]), files=files))
        record = dict(frame_id=case, nas_directory=str(final), manifest_sha256=manifest["sha256"],
            files=dict(files, **{"commit.json": {key: item for key, item in commit.items() if key != "path"}}))
        # Deliberately producer-named leaves: finalization must copy their
        # descriptors, rather than infer a second frames/ops layout.
        registration = write(claim.output / "ops" / (case + "-registered.json"),
            dict(frame_id=case, manifest_sha256=manifest["sha256"], commit=commit, files=files))
        publication = write(claim.output / "ops" / (case + "-published.json"), record)
        records.append(record)
        producers.append(dict(case_id=case, attempt_directory=str(claim.output), manifest=manifest,
            commit=commit, frame_registration=registration, publication=publication, result=scalar))
    table = write(claim.output / "mass_table.csv", b"case,mass\nsynthetic,0\n")
    result = dict(complete=True, worker_references_closed=True, run=task["run"], task=task,
        attempt_id=claim.token, actual_worker=owner, commit=task["commit"],
        calibration_sha256=task["calibration_sha256"], precision=task["precision"], backend=task["backend"],
        frames=len(records), case_count=len(records), ordered_case_ids=task["ordered_case_ids"],
        records=records, frame_producers=producers, manifest=manifest, table=table,
        wall_seconds=3., images_per_hour=1200.)
    candidate = write(claim.output / "candidate_completed.json", result)
    write(claim.lease / "owner.json", dict(format="darsia-alias-lease-owner-v1",
        machine="synthetic", token=claim.token, task=task,
        actual_owner=owner, output=dict(machine_id="synthetic", root_alias="results",
            relative_path="/".join((task["job_id"], task["run"], ".attempts", claim.token)))))
    def no_prefix(attempt):
        """Fresh unit fixtures never infer or adopt another attempt's producer."""
        pytest.fail("Unexpected prefix producer: " + str(attempt))
    publish = publication_runtime.make_publication_factory(root, job, "synthetic",
        producer_origin_for_attempt=no_prefix)
    prepared = publish(claim, candidate, True)
    copied = job_queue._own_asset(claim.lease, prepared["candidate"])
    public = write(root / "results" / task["run"] / "completed.json", copied.read_bytes())
    index = job_queue._own_asset(claim.lease, prepared["products_index"])
    write(root / "done" / (task["run"] + ".json"), dict(format="darsia-alias-done-v1", run=task["run"],
        attempt_id=claim.token, completed=job_queue._own_reference(root, Path(public["path"])),
        products_index=job_queue._own_reference(root, index)))
    return result


@pytest.fixture
def live_close(real_catalogue_join, monkeypatch):
    """Admit real DATA and keep every science/publication/RunIO predicate live."""
    value = real_catalogue_join
    claim, config, configuration = real_common_claim(value)
    root = value.root / claim.task["job_id"]
    job = dict(format="darsia-alias-job-v1", job_id=claim.task["job_id"], series_id="synthetic",
        commit=claim.task["commit"], calibration_sha256=claim.task["calibration_sha256"],
        precision=claim.task["precision"], expected_runs=[claim.task["run"]], publish=[])
    job_pin = write(root / "job.json", job)
    results = claim.local_output.results_root
    stores_raw = ('[machine]\nid="synthetic"\n[stores.synthetic]\nseries=' + json.dumps(str(value.root)) +
        '\nresults=' + json.dumps(str(results)) + '\n').encode()
    stores = write(value.root / "admitted-stores.toml", stores_raw)
    root_provider = provider()
    root_provider["scope"] = {key: deepcopy(claim.task[key]) for key in common.COMMON_SCOPE_FIELDS}
    for name in ("phase1", "phase2"):
        root_provider["stages"][name].update(run=claim.task["run"],
            ordered_case_ids=deepcopy(claim.task["ordered_case_ids"]))
    owner = dict(machine="synthetic", pid=123, process_create_time="synthetic-epoch")
    native = dict(path=str(value.root / "native-launch.json"), sha256="a" * 64, size_bytes=1)
    state = SimpleNamespace(common_mode=True, stage="phase2", task=claim.task, job=job, claim=None,
        root_provider=root_provider, provider=common.stage_provider(root_provider, "phase2"),
        stores_source=stores, binding=dict(path=str(value.root / "request.json"), sha256="b" * 64, size_bytes=1),
        provider_binding=dict(path=str(value.root / "provider.json"), sha256="c" * 64, size_bytes=1),
        request=dict(configuration=configuration, job_root=str(root), job=job_pin),
        output_entitlement=dict(maximum_output_bytes=1024**2))
    state.require_identity = MethodType(common.CommonQualification.require_identity, state)
    state.require_task = MethodType(common.CommonQualification.require_task, state)
    lease = SimpleNamespace(first_qualification=state, source=stores, profile={}, owner=owner,
        manifest=dict(native_launch_context=native),
        receipt=dict(path=str(value.root / "service.json"), sha256="d" * 64, size_bytes=1))
    monkeypatch.setattr(bootstrap, "_LEASE", lease)
    current_native = dict(pid=123, creation_filetime_100ns="100")
    monkeypatch.setattr(owned_process, "_ACTIVE_CHILD", dict(binding=native,
        job_name="synthetic-owned-family", current_native=current_native))
    events = []
    def open_job(access, inherit, name):
        """Supply only a synthetic retained native-Job handle at the OS seam."""
        assert access == 0x4 and inherit is False and name == "synthetic-owned-family"
        return 17
    def check(okay, operation):
        """Preserve the original native API's refusal on a missing handle."""
        assert okay, operation
    api = SimpleNamespace(k=SimpleNamespace(OpenJobObjectW=open_job, GetCurrentProcess=lambda: 19),
        check=check, require_job_flags=lambda handle: 0,
        member=lambda process, handle: process == 19 and handle == 17,
        identity=lambda process: deepcopy(current_native), close=lambda handle: None)
    monkeypatch.setattr(owned_process, "NativeAPI", lambda: api)
    def predecessor(view, role):
        """Transport already-bound predecessor metadata without replacing checks."""
        assert role == "scientific_evidence"
        stage = next(name for name in common.STAGES if view["scope"]["pipeline"] == common.PIPELINES[name])
        if stage == "phase1":
            return dict(phase1=dict(configuration=value.phase1["configuration"], completion=value.phase1["completion"])), {}
        assert stage == "calibration"
        return dict(calibration_join=deepcopy(value.join)), {}
    monkeypatch.setattr(common, "_slot_read", predecessor)
    monkeypatch.setattr(closure, "_slot_read", predecessor)
    runtime = object.__new__(worker_io.WorkerRunIO)
    runtime.commit, runtime.bindings, runtime.capacities = claim.task["commit"], {}, {}
    def capacity_closed(actual_claim):
        """Observe the original capacity-close method after scientific slot write."""
        assert actual_claim is claim and events == ["scientific_slot"]
        events.append("capacity_closed")
    runtime.cache_adapter = SimpleNamespace(lease=lease, close_claim=capacity_closed)
    binding = runtime.prepare(claim)
    completed = publish_original_result(claim, root, job, owner, config["opencl_acceleration"])
    def lean_estimate(stores_map, series_id, task, actual_config, *, repository, metadata_rows):
        """Supply a tiny resource-factory bound; all DATA/science checks stay real."""
        assert stores_map.source["sha256"] == stores["sha256"] and series_id == job["series_id"]
        assert task == claim.task and actual_config == config and repository == binding.repository
        assert tuple(metadata_rows) == binding.verified.rows
        return SimpleNamespace(estimate=dict(admitted_frame_bytes=1024**2, admitted_run_bytes=1024**2))
    monkeypatch.setattr("darsia.fluidflower.pipeline.capacity._phase2_lean_estimate", lean_estimate)
    written = []
    def save_slot(view, role, payload):
        """Keep exact lease, native family and RunIO active before the slot closes."""
        assert view == state.provider and role == "worker_scientific_evidence"
        assert bootstrap.current_import_lease() is lease
        assert owned_process.current_native_membership() == native
        assert current_binding() is binding and events == []
        written.append(deepcopy(payload))
        events.append("scientific_slot")
        return dict(path=str(value.root / "scientific-worker.json"), sha256="e" * 64, size_bytes=1)
    monkeypatch.setattr(closure, "_slot_write", save_slot)
    return SimpleNamespace(value=value, state=state, lease=lease, claim=claim, binding=binding,
        runtime=runtime, written=written, events=events, completed=completed)


def test_finalize_worker_stage_closes_real_source_join_and_original_producer_receipts(live_close):
    """The original science/index predicates produce one own live-worker record."""
    value = live_close
    with value.binding.activate():
        pin = closure.finalize_worker_stage(value.state, value.lease,
            claim=value.claim, binding=value.binding)
    payload, = value.written
    assert value.state.worker_scientific_evidence == pin
    assert payload["calibration_join"] == value.value.join and payload["complete"] is True
    assert payload["process_retirement_claimed"] is False
    assert payload["task"] == value.claim.task and payload["owner"] == value.lease.owner
    assert payload["resolved_inputs"] == [dict(reference=row.reference, uses=list(row.uses), path=str(row.path),
        version=list(value.binding.verified.versions[str(row.path)])) for row in value.binding.verified.rows]
    assert payload["frame_records"][0]["producer"] == value.completed["frame_producers"][0]
    assert payload["frame_records"][0]["producer"] is not value.completed["frame_producers"][0]
    assert payload["compute_evidence"]["control"]["mode"] == "auto"
    assert value.events == ["scientific_slot"] and current_binding() is None


def test_worker_runtime_finalizes_in_original_binding_before_capacity_close(live_close):
    """Actual runtime activation encloses the whole finalizer before reservation release."""
    value = live_close
    assert current_binding() is None
    pin = value.runtime.finalize_common(value.claim)
    assert value.state.worker_scientific_evidence == pin and len(value.written) == 1
    assert value.events == ["scientific_slot"] and current_binding() is None
    value.runtime.capacity_close(value.claim)
    assert value.events == ["scientific_slot", "capacity_closed"]


@pytest.mark.parametrize("changed", ("lease", "native"))
def test_worker_close_refuses_a_different_original_owner_context(live_close, monkeypatch, changed):
    """Neither another import lease nor native-family binding may emit a record."""
    value = live_close
    if changed == "lease":
        monkeypatch.setattr(bootstrap, "_LEASE", SimpleNamespace())
    else:
        monkeypatch.setitem(owned_process._ACTIVE_CHILD, "binding", dict(synthetic="another family"))
    with pytest.raises(CachePolicyError):
        value.runtime.finalize_common(value.claim)
    assert value.written == [] and value.events == [] and current_binding() is None


def test_worker_close_keeps_original_genuine_calibration_join_predicate(live_close):
    """A different predecessor C anchor cannot emit even a valid owned P2 result."""
    value = live_close
    value.value.join["anchor"]["affected_pore_volume_m3"] *= 2
    with pytest.raises(CachePolicyError, match="frozen C/application binding changed"):
        value.runtime.finalize_common(value.claim)
    assert value.written == [] and value.events == [] and current_binding() is None
