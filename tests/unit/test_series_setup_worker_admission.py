"""Private original output ledgers with an isolated admitted-service seam.

The service stub observes lock order and retained cache accounting only. It
does not qualify a native launch, scratch profile, RAW decoder or scientific IO.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import capacity
from darsia.fluidflower.pipeline import series_setup_admission as admission
from darsia.fluidflower.pipeline import series_setup_worker_admission as worker
from darsia.fluidflower.pipeline.job_inputs import catalog_order_key
from darsia.fluidflower.pipeline.store_runtime import ResolvedInput, verify_inputs
from .test_series_setup_admission import _admission
from .test_series_setup_protocol import _job


def _private_admission(tmp_path, monkeypatch, **kwargs):
    """Keep real series inputs apart from external capacity administration."""
    if kwargs.get("raw_cap"):
        from . import test_series_setup_admission as fixtures
        original = fixtures.prepared_raw_request.__wrapped__

        def raw_request(root):
            source = root / "scientific-inputs"
            source.mkdir(parents=True)
            return original(source)

        monkeypatch.setattr(fixtures, "prepared_raw_request", SimpleNamespace(__wrapped__=raw_request))
    return _admission(tmp_path, monkeypatch, **kwargs)


class _ServiceLedger:
    """Observe the existing service reservation without creating claim scopes."""

    def __init__(self, policy, events):
        self.policy, self.events, self.admission = policy, events, {"private": True}
        self.active, self.sufficient = False, True

    @contextmanager
    def locked(self):
        assert not self.active
        self.active = True
        self.events.append("service-locked")
        try:
            yield self
        finally:
            self.events.append("service-unlocked")
            self.active = False

    def require_reservation(self, reservation, owner, maximum):
        assert self.active and maximum == 123
        self.events.append("existing-service-reservation")

    def project(self, maximum, *, noncache_bytes):
        assert self.active and maximum == 0
        self.events.append(("existing-budget-projection", noncache_bytes))
        return dict(sufficient=self.sufficient)


class _ServiceAdapter:
    """Keep service verification separate from incompatible measured profiles."""

    def __init__(self, ledger, owner):
        self.namespace_admission = ledger.admission
        self.owner = owner
        self.lease = SimpleNamespace(ledger=ledger, policy=ledger.policy,
            reservation={"original": True}, maximum=123, manifest={"owner": {"original": True}})

    def verify_preimport(self, stores, policy):
        assert policy is self.lease.policy
        return True

    def _is_current_claim_owner(self, owner):
        return owner == self.owner

    def budget_ledger(self, result_capacity):
        return self.lease.ledger

    def register_claim_allocation(self, *args, **kwargs):
        raise AssertionError("No invented measured profile or zero claim scratch scope")


def _runtime(tmp_path, monkeypatch, *, raw_cap=0, shared=True):
    """Bind real finite input bytes and original output ledgers on private roots."""
    import darsia_cache.capacity as cache_capacity
    import darsia_cache.control as control
    import darsia_cache.worker_adapter as cache_worker
    actual = _private_admission(tmp_path, monkeypatch, metadata_cap=32_000,
                        metadata_file_cap=16_000, raw_cap=raw_cap, shared=shared, bind=False)
    data = actual.test_context
    task, config = deepcopy(data.task), deepcopy(data.config)
    rows = []
    for key, use, filename in (
        ("series", "run.execution_series", "execution-series.toml"),
        ("series_manifest", "run.series_manifest", "series-manifest.json"),
        ("setup_plan", "setup.plan", "setup-plan.json")):
        path = tmp_path / filename
        path.write_bytes(b"{}\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        task[key].update(relative_path=filename, sha256=digest, bytes=path.stat().st_size)
        if key == "setup_plan":
            config["setup_plan"] = dict(path=str(path), sha256=digest, size_bytes=path.stat().st_size)
        rows.append(ResolvedInput(deepcopy(task[key]), path, (use,)))
    path = tmp_path / "operation.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    task["configuration"].update(relative_path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  bytes=path.stat().st_size)
    rows.append(ResolvedInput(deepcopy(task["configuration"]), path, ("run.configuration",)))
    rows.sort(key=lambda row: catalog_order_key(row.reference))
    task["input_catalog"]["configuration"] = deepcopy(task["configuration"])
    task["input_catalog"]["inputs"] = [dict(reference=deepcopy(row.reference), uses=list(row.uses)) for row in rows]
    verified = verify_inputs(task, tuple(rows))
    for state in data.states.values():
        state["free_bytes"] = 10**12
    original_ledger = capacity.CapacityLedger

    def ledger_factory(stores, series_id, root):
        result = original_ledger(stores, series_id, root, volume_provider=data.volume,
                                 context_provider=lambda: dict(data.context))
        data.states[result.volume["volume_id"]]["free_bytes"] = 10**12
        result.reserve = 40 * 1024**3
        return result

    monkeypatch.setattr(capacity, "CapacityLedger", ledger_factory)
    monkeypatch.setattr(control, "CacheBudgetLedger", _ServiceLedger)
    monkeypatch.setattr(cache_worker, "WorkerCacheAdapter", _ServiceAdapter)
    monkeypatch.setattr(cache_capacity, "volume_info", data.volume)
    events = []
    service = _ServiceLedger(SimpleNamespace(root=str(tmp_path / "service-cache")), events)
    adapter = _ServiceAdapter(service, data.owner)
    root = data.metadata_root / "job"
    from darsia.fluidflower.pipeline.job_queue import JobQueue
    JobQueue(root)
    (root / "queue" / (task["run"] + ".json")).write_text(json.dumps(task), encoding="utf-8")
    output = data.metadata_root / "attempt-output"
    runtime = SimpleNamespace(root=root, job=dict(_job(task), series_id="sample-series"),
        stores=data.stores, service_id="fixture-cpu", backend="cpu", commit=task["commit"],
        cache_adapter=adapter, capacity_refusals={}, capacities={},
        setup_verified_inputs={task["attempt_id"]: verified},
        output_factory=lambda *_args: SimpleNamespace(path=output), availability=lambda _task: tuple(rows))
    return runtime, task, config, data.owner, events, service


def _write_claim(helper):
    """Use exact original control paths before binding the real claim."""
    sink = helper.preclaim_sink
    with helper.preclaim_scope(helper.task, helper.runtime.service_id):
        sink.mkdir(sink.claimed.parent, parents=True, exist_ok=True)
        sink.rename(sink.pending, sink.claimed)
        sink.mkdir(sink.lease)
        sink.mkdir(sink.output, parents=True)
        for final in sorted(sink.finals):
            temporary = final.with_name("." + final.name + "." + "a" * 32 + ".partial")
            sink.register_temporary(final, temporary)
            sink.write_bytes(temporary, b"{}\n")
            sink.rename(temporary, final)
    return SimpleNamespace(task=deepcopy(helper.task), token=helper.task["attempt_id"])


def test_claim_branch_retains_actual_whole_caps_and_one_metadata_allowance(tmp_path, monkeypatch):
    runtime, task, _config, owner, events, _service = _runtime(tmp_path, monkeypatch)
    import darsia_cache.control as control

    def forbidden(*args, **kwargs):
        raise AssertionError("Setup cannot enter measured estimate/profile selection")

    monkeypatch.setattr(capacity, "load_estimate", forbidden)
    monkeypatch.setattr(control, "ClaimCacheBudget", forbidden)
    with capacity.ClaimCapacity(runtime, owner) as candidate:
        candidate.availability(task)
        helper = candidate.setup_candidate
        claim = _write_claim(helper)
        assert helper.preclaim_sink.written_bytes == 6
        candidate.bind(claim)
        assert runtime.setup_admissions[claim.token] is helper
        assert helper.sink.written_bytes == 6
        helper.sink.write_bytes(helper.preclaim_sink.output / "result.json", b"{}\n")
        assert helper.sink.written_bytes == 9
        assert not hasattr(runtime.cache_adapter, "profiles")
    assert events[-1] == "service-unlocked"
    assert all(not (Path(row["path"]) / "released.json").exists()
               for row in helper.admission.bound_reservations)
    copied = helper.admission.task
    copied["run"] = "foreign"
    assert helper.admission.task["run"] == task["run"]


def test_preclaim_controls_refuse_unlisted_paths_and_preserve_cap_partial(tmp_path, monkeypatch):
    runtime, task, config, owner, _events, _service = _runtime(tmp_path, monkeypatch)
    helper = worker.setup_claim_admission(runtime, task, config, owner)
    helper.acquire()
    try:
        sink = helper.preclaim_sink
        with helper.preclaim_scope(task, runtime.service_id):
            sink.mkdir(sink.lease, parents=True)
            with pytest.raises(ValueError, match="finite"):
                sink.write_bytes(sink.lease / "unlisted.json", b"{}")
            final = sink.lease / "owner.json"
            temporary = final.with_name(".owner.json." + "b" * 32 + ".partial")
            sink.register_temporary(final, temporary)
            with sink.open_exclusive(temporary) as stream:
                stream.write(b"retained")
                with pytest.raises(ValueError, match="before write"):
                    stream.write(b"x" * sink.maximum_file)
            assert temporary.read_bytes() == b"retained"
        with pytest.raises(RuntimeError, match="active locked"):
            sink.write_bytes(final, b"{}")
    finally:
        helper.unlock()
    assert not helper.admission.bound_reservations
    assert not final.exists()


def test_existing_service_headroom_is_checked_without_a_new_claim_reservation(tmp_path, monkeypatch):
    runtime, task, config, owner, events, service = _runtime(tmp_path, monkeypatch, raw_cap=100)
    helper = worker.setup_claim_admission(runtime, task, config, owner)
    service.sufficient = False
    helper.acquire()
    try:
        with pytest.raises(ValueError, match="headroom"):
            helper.check()
        assert ("existing-budget-projection", 32_100) in events
        assert not helper.admission.bound_reservations
    finally:
        helper.unlock()


def test_real_bind_failure_keeps_every_successful_volume_prefix_reachable(tmp_path, monkeypatch):
    runtime, task, _config, owner, _events, _service = _runtime(tmp_path, monkeypatch, raw_cap=100, shared=False)
    with capacity.ClaimCapacity(runtime, owner) as candidate:
        candidate.availability(task)
        helper = candidate.setup_candidate
        claim = _write_claim(helper)
        ledger = helper.admission._scopes[1][0]

        def failure(*args, **kwargs):
            raise OSError("private second-volume bind failure")

        monkeypatch.setattr(ledger, "bind", failure)
        with pytest.raises(admission.SetupBindError, match="retained"):
            candidate.bind(claim)
        assert runtime.setup_admissions[claim.token] is helper
        assert len(helper.admission.bound_reservations) == 1
    prefix = Path(helper.admission.bound_reservations[0]["path"])
    assert (prefix / "reserved.json").is_file() and not (prefix / "released.json").exists()


@pytest.mark.parametrize("part", ["target", "service", "owner", "source"])
def test_foreign_runtime_is_refused_before_locks_or_control_writes(tmp_path, monkeypatch, part):
    runtime, task, config, owner, events, _service = _runtime(tmp_path, monkeypatch)
    if part == "target":
        runtime.stores.machine_id = "another-target"
    elif part == "service":
        runtime.service_id = "another-service"
    elif part == "owner":
        owner = dict(owner, pid=456)
    else:
        runtime.commit = "2" * 40
    with pytest.raises(ValueError, match="worker|owner"):
        worker.setup_claim_admission(runtime, task, config, owner)
    assert not events


def test_same_machine_with_mixed_case_reaches_original_admission(tmp_path, monkeypatch):
    runtime, task, config, owner, _events, _service = _runtime(tmp_path, monkeypatch)
    runtime.stores.machine_id = 'FiXtUrE'
    helper = worker.setup_claim_admission(runtime, task, config, owner)
    assert helper.admission.task == task


def test_folder_claim_keeps_initial_and_final_space_checks_without_repeating_each_write(tmp_path, monkeypatch):
    runtime, task, config, owner, _events, _service = _runtime(tmp_path, monkeypatch, raw_cap=100)
    helper = worker.setup_claim_admission(runtime, task, config, owner)
    # Isolate the pre-existing service-owner boundary; all claim paths, caps,
    # directory identities, immutable files and reservation writes remain real.
    import darsia_cache.adapters as adapters
    monkeypatch.setattr(adapters, "bind_actual_worker_owner", lambda *args, **kwargs: None)
    helper.folder_mode, helper.cache_ledger = True, None
    checks, original = [], helper.admission.check
    def check():
        checks.append(True)
        return original()
    monkeypatch.setattr(helper.admission, "check", check)
    helper.acquire()
    try:
        helper.check()
        claim = _write_claim(helper)
        assert len(checks) == 1
        helper.bind(claim)
        assert len(checks) == 2
        assert helper.admission.bound_reservations
    finally:
        helper.unlock()


def test_build_worker_restricts_its_original_declared_scope_to_job_and_owned_attempt(tmp_path, monkeypatch):
    runtime, task, config, owner, _events, _service = _runtime(tmp_path, monkeypatch, raw_cap=100)
    original, calls = worker.prepare_setup_admission, []
    def prepare(*args, **kwargs):
        calls.append(kwargs.get("metadata_targets"))
        return original(*args, **kwargs)
    monkeypatch.setattr(worker, "prepare_setup_admission", prepare)
    helper = worker.setup_claim_admission(runtime, task, config, owner)
    assert calls == [(runtime.root.resolve(), helper.preclaim_sink.output.parent)]
    assert helper.admission.configuration == config


def test_output_namespace_ancestry_does_not_grant_metadata_raw_authority(tmp_path, monkeypatch):
    selected = tmp_path / "case"
    selected.mkdir()
    with pytest.raises(ValueError, match="overlap.*RAW"):
        _admission(selected, monkeypatch, metadata_cap=20, raw_cap=100,
                   metadata_roots=[selected], bind=False)


def test_oversize_initial_reservation_refuses_before_open_and_retains_prefix(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, bind=False)
    ledger = actual._scopes[0][0]
    original_bind, original_open = ledger.bind, Path.open

    def bind(*args, **kwargs):
        path = original_bind(*args, **kwargs)
        (path / "reserved.json").write_bytes(b"x" * (1024**2 + 1))
        return path

    def guarded_open(path, mode="r", *args, **kwargs):
        if path.name == "reserved.json" and mode == "rb":
            raise AssertionError("Oversize reservation must be rejected before open")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(ledger, "bind", bind)
    monkeypatch.setattr(Path, "open", guarded_open)
    with actual.locked():
        actual.check()
        with pytest.raises(admission.SetupBindError, match="metadata bound"):
            actual.bind(actual.test_context.job_root, actual.test_context.claim, actual.test_context.owner)
    assert len(actual.bound_reservations) == 1


def test_changed_bound_reservation_size_refuses_before_reading_oversize_bytes(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20)
    reserved = Path(actual.bound_reservations[0]["path"]) / "reserved.json"
    reserved.write_bytes(b"x" * (1024**2 + 1))
    original = Path.open

    def guarded_open(path, mode="r", *args, **kwargs):
        if path == reserved and mode == "rb":
            raise AssertionError("A changed reservation size must fail before read")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError, match="changed before read"):
        actual.require_write(actual.test_context.metadata_root / "blocked.json")


def _retirement_case(tmp_path, monkeypatch, *, shared=False, partial_bind=False):
    """Retain real multi-volume records and a partial original output tree."""
    actual = _private_admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100,
                        shared=shared, bind=False)
    data, task, config = actual.test_context, deepcopy(actual.task), actual.configuration
    controls = Path(data.stores.roots("sample-series")["series"])
    controls.mkdir()
    operation = controls / "operation.json"
    operation.write_text(json.dumps(config), encoding="utf-8")
    previous = deepcopy(task["configuration"])
    task["configuration"].update(root_alias="series", relative_path=operation.name,
        sha256=hashlib.sha256(operation.read_bytes()).hexdigest(), bytes=operation.stat().st_size)
    task["input_catalog"]["configuration"] = deepcopy(task["configuration"])
    next(row for row in task["input_catalog"]["inputs"] if row["reference"] == previous)["reference"] = deepcopy(task["configuration"])
    task["input_catalog"]["inputs"].sort(key=lambda row: catalog_order_key(row["reference"]))
    original = capacity.CapacityLedger

    def ledger_factory(stores, series_id, root):
        return original(stores, series_id, root, volume_provider=data.volume,
                        context_provider=lambda: dict(data.context))

    monkeypatch.setattr(capacity, "CapacityLedger", ledger_factory)
    actual = admission.prepare_setup_admission(data.stores, "sample-series", task, config,
        metadata_roots=config["output_scope"]["metadata_roots"],
        scratch_roots=config["output_scope"]["scratch_roots"])
    data.job_root.mkdir()
    (data.job_root / "job.json").write_text(json.dumps(dict(_job(task), series_id="sample-series")), encoding="utf-8")
    claim = SimpleNamespace(task=task, token=task["attempt_id"])
    if partial_bind:
        def failure(*args, **kwargs):
            raise OSError("private second-volume bind failure")
        monkeypatch.setattr(actual._scopes[1][0], "bind", failure)
    with actual.locked():
        actual.check()
        if partial_bind:
            with pytest.raises(admission.SetupBindError):
                actual.bind(data.job_root, claim, data.owner)
        else:
            actual.bind(data.job_root, claim, data.owner)
    destination = Path(actual.raw_authority["destination"])
    destination.mkdir()
    (destination / "retained-partial.bin").write_bytes(b"retained original output")
    proof = dict(complete=True, all_absent=True, worker_os_references_closed=True,
        run=task["run"], attempt_id=task["attempt_id"], identities=[deepcopy(data.owner)],
        supervisor=dict(machine="fixture-cpu", pid=999, process_create_time="999.125"))
    return actual, data, task, config, operation, destination, proof


@pytest.mark.parametrize("shared,partial_bind", [(False, False), (True, False), (False, True)])
def test_owned_retirement_releases_exact_volume_records_with_partial_output_present(tmp_path, monkeypatch, shared, partial_bind):
    actual, data, task, config, _operation, destination, proof = _retirement_case(
        tmp_path, monkeypatch, shared=shared, partial_bind=partial_bind)
    request = json.loads(Path(config["raw_build_request"]["path"]).read_bytes())
    forbidden = {Path(row["path"]) for row in request["configuration_files"]}
    forbidden.update(Path(request[key]["path"]) for key in ("baseline", "profile", "shape_basis_certificate"))
    forbidden.update(Path(row["source"]["path"]) for row in request["protocol_inputs"])
    original_open = Path.open

    def guarded_open(path, mode="r", *args, **kwargs):
        if path in forbidden and "r" in mode:
            raise AssertionError("Retirement must not reopen historical scientific inputs")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    released = capacity.release_retired_capacity(data.job_root, task, data.stores, proof)
    assert len(released) == (1 if shared or partial_bind else 2)
    assert set(released) == {row["path"] for row in actual.bound_reservations}
    for path in released:
        row = json.loads((Path(path) / "released.json").read_bytes())
        assert row["basis"] == "original_owned_child_retirement" and row["retirement"] == proof
    assert (destination / "retained-partial.bin").read_bytes() == b"retained original output"


@pytest.mark.parametrize("missing", ["operation", "request"])
def test_retirement_missing_pinned_controls_keeps_every_actual_reservation(tmp_path, monkeypatch, missing):
    actual, data, task, config, operation, destination, proof = _retirement_case(tmp_path, monkeypatch)
    path = operation if missing == "operation" else Path(config["raw_build_request"]["path"])
    retained = path.with_name(path.name + ".retained")
    path.rename(retained)
    with pytest.raises(FileNotFoundError):
        capacity.release_retired_capacity(data.job_root, task, data.stores, proof)
    assert retained.is_file()
    assert all(not (Path(row["path"]) / "released.json").exists()
               for row in actual.bound_reservations)
    assert (destination / "retained-partial.bin").is_file()
