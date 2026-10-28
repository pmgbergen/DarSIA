"""Private real writer/ledger lifecycle inside the original worker IO class.

Native service admission uses the isolated seam from the admission tests.
One replay callback observes the runtime branch; it does not qualify science.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import worker_io
from darsia.fluidflower.pipeline import series_setup_worker_admission as admission
from darsia.fluidflower.pipeline.series_setup_sink import current_sink
from darsia.fluidflower.pipeline.store_runtime import InputHashMismatch
from .test_series_setup_worker_admission import _runtime, _write_claim
from .test_series_setup_protocol import _job, _task


def _bound_runtime(tmp_path, monkeypatch):
    """Bind original private output reservations before driving WorkerRunIO."""
    prepared, task, config, owner, _events, _service = _runtime(tmp_path, monkeypatch)
    runtime = object.__new__(worker_io.WorkerRunIO)
    runtime.__dict__.update(prepared.__dict__)
    runtime.bindings, runtime.factories, runtime.setup_output_scopes = {}, {}, {}
    helper = admission.setup_claim_admission(runtime, task, config, owner)
    helper.acquire()
    try:
        claim = _write_claim(helper)
        helper.bind(claim)
    finally:
        helper.unlock()
    claim.input_rows = runtime.setup_verified_inputs[claim.token].rows
    claim.local_output = SimpleNamespace(path=helper.preclaim_sink.output)
    runtime.setup_admissions = {claim.token: helper}
    return runtime, helper, claim, owner


def test_scope_spans_preparation_execution_publication_then_closes_original_caps(tmp_path, monkeypatch):
    runtime, helper, claim, owner = _bound_runtime(tmp_path, monkeypatch)
    retained = helper.admission.bound_reservations
    with runtime.output_scope(claim) as sink:
        assert current_sink() is sink is helper.sink
        runtime.bindings[claim.token], runtime.factories[claim.token] = object(), object()
        for stage in ("prepare", "execute", "publication"):
            sink.write_bytes(claim.local_output.path / (stage + ".json"), b"{}\n")
        runtime.capacity_progress(claim, owner, dict(state="frame_completed", complete=True))
        assert runtime.finalize_common(claim) is None
        with pytest.raises(ValueError, match="scope must close"):
            runtime.capacity_close(claim)
        assert all(not (Path(row["path"]) / "released.json").exists() for row in retained)
    assert current_sink() is None and sink.closed
    with pytest.raises(ValueError, match="closed"):
        sink.write_bytes(claim.local_output.path / "too-late.json", b"{}")
    runtime.release(claim)
    runtime.release(claim)  # The original successful close remains idempotent.
    assert claim.token not in runtime.bindings and claim.token not in runtime.factories
    assert claim.token not in runtime.setup_admissions
    assert all((Path(row["path"]) / "released.json").is_file() for row in retained)


def test_failed_scope_closes_writer_but_retains_all_caps_for_owned_retirement(tmp_path, monkeypatch):
    runtime, helper, claim, _owner = _bound_runtime(tmp_path, monkeypatch)
    runtime.bindings[claim.token], runtime.factories[claim.token] = object(), object()
    path = claim.local_output.path / "retained-failure.json"
    with pytest.raises(RuntimeError, match="private execution failure"):
        with runtime.output_scope(claim) as sink:
            sink.write_bytes(path, b"retained")
            raise RuntimeError("private execution failure")
    runtime.release(claim, retain_capacity=True)
    assert current_sink() is None and helper.sink.closed
    assert path.read_bytes() == b"retained"
    assert runtime.setup_admissions[claim.token] is helper
    assert claim.token not in runtime.bindings and claim.token not in runtime.factories
    assert all(not (Path(row["path"]) / "released.json").exists()
               for row in helper.admission.bound_reservations)


def test_live_stream_cannot_be_declared_closed_or_release_its_volume_caps(tmp_path, monkeypatch):
    runtime, helper, claim, _owner = _bound_runtime(tmp_path, monkeypatch)
    stream_scope = None
    try:
        with pytest.raises(ValueError, match="streams have not closed"):
            with runtime.output_scope(claim) as sink:
                stream_scope = sink.open_exclusive(claim.local_output.path / "open.json")
                stream = stream_scope.__enter__()
                stream.write(b"retained")
        assert not helper.sink.closed
        with pytest.raises(ValueError, match="streams have not closed"):
            runtime.release(claim)
        assert all(not (Path(row["path"]) / "released.json").exists()
                   for row in helper.admission.bound_reservations)
    finally:
        if stream_scope is not None:
            stream_scope.__exit__(None, None, None)
        runtime.release(claim, retain_capacity=True)


@pytest.mark.parametrize("changed", [False, True])
def test_prepare_preserves_original_input_hashes_and_skips_only_common_task_policy(tmp_path, monkeypatch, changed):
    runtime, helper, claim, _owner = _bound_runtime(tmp_path, monkeypatch)
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("A Setup operation cannot inherit measured Common task qualification")

    runtime.cache_adapter.lease.first_qualification = SimpleNamespace(
        common_mode=True, reference_build=True, stage="phase2", require_task=forbidden)
    runtime.cache_adapter.lease.profile = dict(format="darsia-common-phase-import-profile-v1")

    def replay(task, verified, local_output, *, repository, admitted_commit):
        assert current_sink() is helper.sink
        assert task == claim.task and verified.run == task["run"]
        assert verified.metadata("run.configuration") == helper.admission.configuration
        assert local_output is claim.local_output and admitted_commit == task["commit"]
        calls.append("replayed")
        return SimpleNamespace(configuration=deepcopy(helper.admission.configuration))

    monkeypatch.setattr(worker_io.RunIOBinding, "replay", staticmethod(replay))
    if changed:
        row = runtime.setup_verified_inputs[claim.token].for_use("run.configuration")
        raw = row.path.read_bytes()
        row.path.write_bytes(b"X" + raw[1:])
    with runtime.output_scope(claim):
        if changed:
            with pytest.raises(InputHashMismatch, match="SHA/bytes differ"):
                runtime.prepare(claim)
            assert calls == [] and runtime.bindings == {}
        else:
            binding = runtime.prepare(claim)
            assert runtime.bindings[claim.token] is binding and calls == ["replayed"]
            assert not hasattr(binding, "opencl_proof_released")
    runtime.release(claim, retain_capacity=True)


@pytest.mark.parametrize('machine', ['fixture', 'FiXtUrE'])
def test_setup_availability_never_selects_a_frame_resume_source(tmp_path, monkeypatch, machine):
    runtime, helper, claim, _owner = _bound_runtime(tmp_path, monkeypatch)
    from darsia.fluidflower.pipeline import checkpoint

    def forbidden(*args, **kwargs):
        raise AssertionError("Setup operations do not resume analysis frame prefixes")

    task = deepcopy(claim.task)
    task["execution_ready"] = True
    task["input_catalog"]["runtime_support"] = "darsia-role-io-runtime-v1"
    monkeypatch.setattr(checkpoint, "resume_source_for_claim", forbidden)
    monkeypatch.setattr(worker_io, "resolve_catalog", lambda *_args: claim.input_rows)
    runtime.stores.machine_id = machine
    assert worker_io.WorkerRunIO.availability(runtime, task) == claim.input_rows
    runtime.release(claim, retain_capacity=True)


@pytest.mark.parametrize("setup", [False, True])
def test_actual_service_adapter_keeps_lease_and_skips_reference_fence_only_for_typed_setup(monkeypatch, setup):
    from darsia_cache import worker_adapter
    lease = SimpleNamespace(admission={"existing": True}, first_qualification=SimpleNamespace(reference_build=True))
    seen = []

    def original_lease(stores):
        seen.append(stores)
        return lease

    monkeypatch.setattr(worker_adapter, "require_same_startup_stores", original_lease)
    factory = lambda *_args: None
    job = _job(_task()) if setup else dict(pipeline="native-phase-n1")
    runtime = SimpleNamespace(job=job, stores=object(), factory=factory)
    adapter = worker_adapter.WorkerCacheAdapter(runtime)
    assert adapter.lease is lease and seen == [runtime.stores]
    assert adapter.profiles == {} and adapter.claims == {}
    assert (runtime.factory is factory) is setup
