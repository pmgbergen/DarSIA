"""Synthetic J6 protocol regressions; not cache or scientific measurements.

All profiles stay in memory. No production profile files, worker, namespace API,
payload, cleanup, request environment or machine-specific path is used. These
tests are prepared as canonical pytest source and have not been run here.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia_cache import capacity as cache_capacity
from darsia_cache import control
from darsia_cache import worker_adapter as adapter_module
from darsia_cache.policy import CachePolicy, CachePolicyError, PHYSICAL_RESERVE_BYTES
from darsia_cache.worker_adapter import WorkerCacheAdapter
from darsia.fluidflower.pipeline import capacity
from darsia.fluidflower import job_filesystem


def pin(path):
    """An in-memory TEST_ONLY descriptor, never a real qualification receipt."""
    return dict(path=str(path), sha256="1" * 64, size_bytes=1)


class SyntheticCacheLedger(control.CacheBudgetLedger):
    def __init__(self, policy, identity, original_owner, reservation, trace):
        self.policy = policy
        self.admission = dict(cache_identity=identity)
        self.original_owner = deepcopy(original_owner)
        self.reservation = deepcopy(reservation)
        self.trace = trace
        self.lock_active = False
        self.owners_checked = []
        self.used, self.unwritten, self.eligible = 100, 100, 50
        self.scope_written = 100
        self.other_scope_bytes = {}
        self.free = PHYSICAL_RESERVE_BYTES + 1000
        self.fresh = True
        self.binding_present = True
        self.measure = self.observation

    @contextmanager
    def locked(self, *, create_binding=True):
        assert create_binding is False, "J6 must never initialize a cache binding"
        if not self.binding_present:
            raise CachePolicyError("Missing existing synthetic cache binding")
        assert not self.lock_active
        self.trace.append("cache-read-enter")
        self.lock_active = True
        try:
            yield self
        finally:
            self.lock_active = False
            self.trace.append("cache-read-exit")

    def observation(self, root):
        self._require_lock()
        assert root == self.policy.root
        return dict(root=root, fresh=self.fresh,
            root_identity=self.admission["cache_identity"], no_reparse_traversal=True,
            active_scope_bytes={self.reservation["token"]: self.scope_written, **self.other_scope_bytes},
            logical_bytes=self.used, free_bytes=self.free,
            measured_utc="TEST_ONLY_SYNTHETIC_TIME", eligible_reclaimable_bytes=self.eligible)

    def remaining(self, allocated_by_token):
        self._require_lock()
        assert allocated_by_token == {self.reservation["token"]: self.scope_written, **self.other_scope_bytes}
        return self.unwritten

    def require_reservation(self, reservation, owner, needed):
        self._require_lock()
        self.owners_checked.append(deepcopy(owner))
        if reservation != self.reservation or owner != self.original_owner or needed != 200:
            raise CachePolicyError("Wrong original synthetic reservation owner/binding")
        return {}, b""

    def reserve(self, *args, **kwargs):
        raise AssertionError("Read projection must not reserve")

    def close(self, *args, **kwargs):
        raise AssertionError("Read projection must not close a budget")


class SyntheticResults:
    def __init__(self, stores, common, trace):
        self.stores, self.common_root, self.trace = stores, common, trace
        self.host, self.account = "synthetic-host", "S-1-5-21-1"
        self.remote_scope = None
        self.reserve = PHYSICAL_RESERVE_BYTES
        self.binding_present = True

    @contextmanager
    def locked(self, *, create=True):
        assert create is False, "J6 must never initialize a result ledger"
        if not self.binding_present:
            raise CachePolicyError("Missing existing synthetic result binding")
        self.trace.append("result-read-enter")
        try:
            yield self
        finally:
            self.trace.append("result-read-exit")

    def remaining(self, *, exclude_job=None):
        return 100

    def fresh_volume(self):
        return dict(volume_id="synthetic-volume", free_bytes=PHYSICAL_RESERVE_BYTES + 1000)


@pytest.fixture
def projection_fixture(tmp_path, monkeypatch):
    """Accepted upstream proofs are in-memory synthetic protocol doubles only."""
    trace = []
    original = dict(pid=101, process_create_time="1")
    attached = dict(pid=202, process_create_time="2")
    stores = SimpleNamespace(source=pin(tmp_path / "stores.toml"), machine_id="synthetic-host")
    stores.roots = lambda series: dict(results=str(tmp_path / "results"))
    namespace_binding = pin(tmp_path / "namespace.json")
    service_profile = pin(tmp_path / "service-profile.json")
    reservation = dict(token="a" * 32, pool="synthetic", sha256="2" * 64)
    identity = dict(TEST_ONLY=True)
    policy = CachePolicy(str(tmp_path / "cache"), 1000, 48,
                         dict(jobs=pin(tmp_path / "jobs-profile.json")))
    ledger = SyntheticCacheLedger(policy, identity, original, reservation, trace)
    common = tmp_path / "common"
    result = SyntheticResults(stores, common, trace)
    receipt = dict(policy=stores.source, profile=service_profile,
                   manifest=dict(reservation=reservation, root=str(tmp_path / "service"), owner=original))
    lease = SimpleNamespace(policy=policy, source=stores.source, owner=attached,
        receipt=pin(tmp_path / "service-receipt.json"), package_file=__file__,
        root=tmp_path / "service", profile_binding=service_profile, reservation=reservation,
        maximum=200, ledger=ledger, admission=dict(cache_identity=identity),
        context=dict(host=result.host, service_account_sid=result.account, common_root=str(common)),
        document=dict(cache_bootstrap=dict(namespace_admission=namespace_binding)))
    runtime = SimpleNamespace(stores=stores, root=tmp_path / "job", job=dict(series_id="FF_AC"),
                              backend="cpu", commit="3" * 40, repository=tmp_path / "repository")
    task = dict(run="ac52", attempt_id="b" * 32, commit=runtime.commit,
        pipeline="native-phase-n1", precision="float32",
        ordered_case_ids=["f" + str(i) for i in range(39)])
    estimate = SimpleNamespace(products_estimated_bytes=300, transient_estimated_bytes=100)
    pipeline = runtime.repository / "src" / "darsia" / "fluidflower" / "pipeline"
    producers = [Path(adapter_module.__file__), pipeline / "worker_io.py", pipeline / "capacity.py",
        pipeline / "runner.py", pipeline / "series.py", pipeline.parent / "native" / "sidecar.py",
        pipeline.parent / "mass" / "n1_pipeline.py"]
    runtime_sources = [tmp_path / suffix for suffix in (
        "numba/core/caching.py", "matplotlib/font_manager.py", "tempfile.py", "importlib/_bootstrap_external.py")]
    closure = [pin(p) for p in producers + runtime_sources]
    lease.profile = dict(module_closure=closure)
    coverage = dict(format="darsia-worker-cache-service-only-qualification-v1", complete=True,
        passed=True, commit=runtime.commit, pipeline=task["pipeline"], backend="cpu", precision="float32",
        case_count=39, allocation_model="shared-service-only-no-claim-files-v1",
        service_profile=service_profile, namespace_admission=namespace_binding,
        stores_source=stores.source, producer_source_closure=[pin(p) for p in producers],
        runtime_source_closure=[pin(p) for p in runtime_sources],
        shared_service_writers=["bytecode", "matplotlib", "numba", "tempfile"],
        claim_local_disposable_writers=[], unclassified_writers=[],
        actual_writer_IO_observed=True, maximum_service_cache_bytes=170)
    measured = dict(complete=True, maximum_observed_cache_bytes=0, worker_cache_service_coverage=coverage)
    completion = dict(complete=True, passed=True, commit=runtime.commit,
        worker_cache_service_coverage_sha256=adapter_module._coverage_hash(coverage))
    profile = dict(format="darsia-cache-measured-profile-v1", purpose="jobs", pipeline=task["pipeline"],
        precision="float32", case_count=39, commit=runtime.commit, maximum_cache_bytes=0,
        measurement=pin(tmp_path / "measurement.json"), completion=pin(tmp_path / "completion.json"),
        result_estimate_excludes_cache=True, writer_contract="all-cache-writers-bounded-and-lifecycle-bound-v1")
    documents = {policy.profiles["jobs"]["path"]: profile,
                 profile["measurement"]["path"]: measured, profile["completion"]["path"]: completion}
    monkeypatch.setattr(cache_capacity, "read_bound", lambda bound: (documents[bound["path"]], b""))
    monkeypatch.setattr(cache_capacity, "volume_info", lambda root: result.fresh_volume())
    monkeypatch.setattr(adapter_module, "require_same_startup_stores", lambda value: lease)
    monkeypatch.setattr(adapter_module, "_exact", lambda bound, package: deepcopy(receipt))
    monkeypatch.setattr(adapter_module, "_owner", lambda: (attached, {}))
    adapter = WorkerCacheAdapter.__new__(WorkerCacheAdapter)
    adapter.runtime, adapter.lease = runtime, lease
    adapter.namespace_admission = lease.admission
    adapter.profiles, adapter.claims = {}, {}
    return SimpleNamespace(adapter=adapter, stores=stores, lease=lease, ledger=ledger,
        result=result, task=task, estimate=estimate, runtime=runtime, receipt=receipt,
        profile=profile, measured=measured, coverage=coverage, completion=completion,
        original=original, attached=attached, trace=trace)


def project(f):
    return f.adapter.project_expected_space(f.stores, "FF_AC", [f.task], [f.estimate],
                                            f.result, job_root=f.runtime.root)


def test_original_reservation_owner_differs_from_attached_child(projection_fixture):
    f = projection_fixture
    result = project(f)
    assert f.original != f.attached
    assert f.ledger.owners_checked == [f.original, f.original]
    assert result["service_reservation_original_owner"] == f.original
    assert result["attached_service_actor"] == f.attached
    assert result["required_cache_bytes"] == 200
    assert result["observed_service_peak_bytes"] == 170
    assert result["additional_cache_reservation_bytes"] == 0
    assert result["quota_headroom_bytes"] == 800
    assert result["post_admission_quota_headroom_bytes"] == 800
    assert result["eligible_reclaimable_logical_bytes"] == 50
    assert result["fresh_cache_budget"]["free_physical_bytes"] == PHYSICAL_RESERVE_BYTES + 1000
    assert result["fresh_cache_budget"]["needed_physical_bytes"] == PHYSICAL_RESERVE_BYTES + 600
    assert result["complete"] and result["sufficient"]
    assert f.adapter.profiles == {} and f.adapter.claims == {}
    assert f.trace[-4:] == ["result-read-enter", "cache-read-enter", "cache-read-exit", "result-read-exit"]


def test_wrong_original_owner_refuses_without_child_fallback(projection_fixture):
    f = projection_fixture
    f.receipt["manifest"]["owner"] = dict(pid=999, process_create_time="9")
    with pytest.raises(CachePolicyError, match="Wrong original"):
        project(f)
    assert f.ledger.owners_checked == [f.receipt["manifest"]["owner"]]
    assert f.attached not in f.ledger.owners_checked


def test_exact_v1_profile_refuses_80_without_scaling(projection_fixture):
    f = projection_fixture
    f.task["ordered_case_ids"] = ["f" + str(i) for i in range(80)]
    with pytest.raises(CachePolicyError, match="applicability differs"):
        project(f)
    assert "result-read-enter" not in f.trace
    assert f.adapter.profiles == {}


@pytest.mark.parametrize("which", ["cache", "result"])
def test_missing_existing_binding_refuses(projection_fixture, which):
    f = projection_fixture
    (f.ledger if which == "cache" else f.result).binding_present = False
    with pytest.raises(CachePolicyError, match="Missing existing"):
        project(f)
    assert f.adapter.profiles == {} and f.adapter.claims == {}


def test_stale_accounting_refuses(projection_fixture):
    f = projection_fixture
    f.ledger.fresh = False
    with pytest.raises(CachePolicyError, match="Fresh actual"):
        project(f)


def test_known_insufficient_numbers_remain_visible(projection_fixture):
    f = projection_fixture
    # Existing inactive logical cache may consume quota without changing the
    # admitted service's own100 written bytes or its100 remaining entitlement.
    f.ledger.used = 950
    result = project(f)
    assert result["complete"] and not result["sufficient"]
    assert result["quota_headroom_bytes"] == -50
    with f.ledger.locked(create_binding=False), pytest.raises(CachePolicyError, match="Admission refused"):
        f.ledger.admit(0, noncache_bytes=500)


def test_different_volumes_do_not_charge_results_to_cache(projection_fixture, monkeypatch):
    f = projection_fixture
    monkeypatch.setattr(cache_capacity, "volume_info", lambda root:
                        dict(volume_id="different-cache-volume", free_bytes=f.ledger.free))
    result = project(f)
    assert not result["same_cache_and_results_volume"]
    assert result["fresh_cache_budget"]["required_noncache_bytes"] == 0
    assert result["fresh_cache_budget"]["needed_physical_bytes"] == PHYSICAL_RESERVE_BYTES + 100


def test_positive_claim_entitlement_remains_unqualified(projection_fixture):
    f = projection_fixture
    f.profile["maximum_cache_bytes"] = 1
    with pytest.raises(CachePolicyError, match="positive claim-local"):
        project(f)


def test_whole_job_missing_provider_is_not_combined_safe(projection_fixture, monkeypatch):
    f = projection_fixture
    monkeypatch.setattr(capacity, "CapacityLedger", lambda *args: f.result)
    monkeypatch.setattr(capacity, "load_estimate", lambda *args, **kwargs: f.estimate)
    monkeypatch.setattr(capacity, "asdict", lambda row: dict(TEST_ONLY_SYNTHETIC=True))
    monkeypatch.setattr(job_filesystem, "inspect_results_root", lambda root:
                        SimpleNamespace(allowed=True, protocol="synthetic", local=True))
    result = capacity.whole_job_report(f.stores, "FF_AC", [f.task], [{}],
                                      repository=f.runtime.repository, job_root=f.runtime.root)
    assert result["results_space_sufficient"]
    assert not result["sufficient"] and not result["complete_admission_sufficient"]
    assert result["cache_policy_state"] == "MISSING_QUALIFIED_CACHE_PROVIDER"
    assert result["required_cache_bytes"] is None
    assert f.trace == []


def test_whole_job_explicit_qualified_provider_uses_real_budget(projection_fixture, monkeypatch):
    f = projection_fixture
    monkeypatch.setattr(capacity, "CapacityLedger", lambda *args: f.result)
    monkeypatch.setattr(capacity, "load_estimate", lambda *args, **kwargs: f.estimate)
    monkeypatch.setattr(capacity, "asdict", lambda row: dict(TEST_ONLY_SYNTHETIC=True))
    monkeypatch.setattr(job_filesystem, "inspect_results_root", lambda root:
                        SimpleNamespace(allowed=True, protocol="synthetic", local=True))
    result = capacity.whole_job_report(f.stores, "FF_AC", [f.task], [{}],
        repository=f.runtime.repository, job_root=f.runtime.root, cache_provider=f.adapter)
    assert result["sufficient"] and result["complete_admission_sufficient"]
    # N51: this job's result estimate excludes other result reservations, which
    # remain visible. The independently qualified cache budget stays strict.
    assert result["needed_bytes"] == PHYSICAL_RESERVE_BYTES + 400
    assert result["other_active_remaining_bytes"] == 100
    assert result["other_active_remaining_included_in_results_estimate"] is False
    assert result["results_space_policy"] == "estimate, not coordinated across hosts"
    assert result["required_cache_bytes"] == 200
    assert result["quota_headroom_bytes"] == 800
    assert result["cache"]["reservation_created"] is False
    assert result["cache"]["cleanup_performed"] is False


@pytest.mark.parametrize("changed", ["backend", "precision", "commit", "service_profile", "runtime_closure"])
def test_original_source_and_role_applicability_stays_strict(projection_fixture, changed):
    f = projection_fixture
    if changed == "backend":
        f.runtime.backend = "cuda"
    elif changed == "precision":
        f.task["precision"] = "float64"
    elif changed == "commit":
        f.task["commit"] = "4" * 40
    elif changed == "service_profile":
        f.coverage["service_profile"] = pin(Path("different-service-profile.json"))
    else:
        f.coverage["runtime_source_closure"] = f.coverage["runtime_source_closure"][:-1]
    f.completion["worker_cache_service_coverage_sha256"] = adapter_module._coverage_hash(f.coverage)
    with pytest.raises(CachePolicyError):
        project(f)
    assert "result-read-enter" not in f.trace
    assert f.adapter.profiles == {}


def test_original_claim_validation_still_records_its_binding(projection_fixture):
    f = projection_fixture
    maximum, binding, _, _, _ = control.applicable_job_cache_profile(
        f.lease.policy, f.adapter, f.task, f.estimate)
    assert maximum == 0
    assert binding == f.lease.policy.profiles["jobs"]
    assert f.adapter.profiles[f.task["attempt_id"]]["binding"] == binding


def _observed_N1_unit_profile(f):
    """A separate in-memory TESTONLY stage proof, never a production measurement."""
    f.task["pipeline"] = f.profile["pipeline"] = f.coverage["pipeline"] = "n1"
    root = f.runtime.repository / "src" / "darsia" / "fluidflower" / "pipeline"
    paths = [root / "staged_data.py", root / "job_inputs.py", root.parent / "mass" / "pixel_pipeline.py",
        root.parent / "mass" / "pixels.py", root.parent / "geometry" / "pixels.py",
        root.parent / "storage_extensive.py", root.parent / "storage_extensive_io.py",
        root.parent / "storage_n1_outputs.py", root.parent / "storage_n1_saved_products.py",
        root.parent / "storage_outputs.py", root.parent / "storage_resolution.py"]
    f.coverage["producer_source_closure"].extend(pin(path) for path in paths)
    f.lease.profile["module_closure"] = f.coverage["producer_source_closure"] + f.coverage["runtime_source_closure"]
    f.completion["worker_cache_service_coverage_sha256"] = adapter_module._coverage_hash(f.coverage)


def test_N1_profile_requires_its_own_observed_stage_and_complete_source_closure(projection_fixture):
    f = projection_fixture
    _observed_N1_unit_profile(f)
    result = project(f)
    assert result["observed_service_peak_bytes"] == 170
    assert result["additional_cache_reservation_bytes"] == 0
    assert f.adapter.profiles == {}, "Projection must never bind or create a claim"


@pytest.mark.parametrize("fault", ["old_stage_measurement", "unobserved_IO", "omitted_writer", "different_count", "positive_claim_cache"])
def test_N1_cache_profile_is_not_an_alias_for_old_native_measurements(projection_fixture, fault):
    f = projection_fixture
    _observed_N1_unit_profile(f)
    if fault == "old_stage_measurement":
        f.coverage["pipeline"] = "native-phase-n1"
    elif fault == "unobserved_IO":
        f.coverage["actual_writer_IO_observed"] = False
    elif fault == "omitted_writer":
        f.coverage["producer_source_closure"] = [row for row in f.coverage["producer_source_closure"]
            if not row["path"].endswith("storage_extensive_io.py")]
    elif fault == "different_count":
        f.coverage["case_count"] -= 1
    else:
        f.profile["maximum_cache_bytes"] = 1
    f.completion["worker_cache_service_coverage_sha256"] = adapter_module._coverage_hash(f.coverage)
    with pytest.raises(CachePolicyError):
        project(f)
    assert "result-read-enter" not in f.trace
    assert f.adapter.profiles == {}


def test_other_active_service_is_counted_without_rereserving_own_service(projection_fixture):
    f = projection_fixture
    f.ledger.other_scope_bytes = {"c" * 32: 25}
    f.ledger.used = 125
    f.ledger.unwritten = 150
    result = project(f)
    assert result["required_cache_bytes"] == 200
    assert result["additional_cache_reservation_bytes"] == 0
    assert result["quota_headroom_bytes"] == 725
    assert result["fresh_cache_budget"]["expected_cache_allocation_bytes"] == 150


def test_real_cache_read_transaction_does_not_create_missing_binding(tmp_path, monkeypatch):
    """Exercise the real new branch with in-memory namespace/lock doubles."""
    key = "d" * 64
    (tmp_path / ("mutex-" + key)).touch()  # Existing TEST_ONLY POSIX lock, not a profile.
    events = []

    @contextmanager
    def local_lock(*args):
        yield

    class EmptyNamespace:
        def __init__(self, *args):
            pass

        def names(self):
            return []

        def write_once(self, *args):
            raise AssertionError("Read-only cache binding creation attempted")

        def close(self):
            events.append("closed")

    monkeypatch.setattr(cache_capacity, "_LocalVolumeLock", local_lock)
    monkeypatch.setattr(control, "WindowsNamespace", EmptyNamespace)
    ledger = control.CacheBudgetLedger.__new__(control.CacheBudgetLedger)
    ledger.lock_active = False
    ledger.bound, ledger.key = dict(path=str(tmp_path), identity={}), key
    ledger.capacity = SimpleNamespace(host="synthetic-host")
    ledger.admission = dict(authorized_sids=["S-1-5-21-1"], cache_identity={})
    ledger.policy = CachePolicy(str(tmp_path / "cache"), 1000, 48, {})
    with pytest.raises(CachePolicyError, match="existing accepted cache binding"):
        with ledger.locked(create_binding=False):
            pytest.fail("Missing binding admitted")
    assert events == ["closed"]
    assert not (tmp_path / ("cache-" + key + ".json")).exists()


def test_real_result_read_transaction_does_not_create_missing_binding(tmp_path, monkeypatch):
    common, directory = tmp_path / "common", tmp_path / "state"
    root = directory / "pool"
    common.mkdir()
    root.mkdir(parents=True)
    key = "e" * 64
    (common / ("mutex-" + key)).touch()

    @contextmanager
    def local_lock(*args):
        yield

    def forbidden_write(*args, **kwargs):
        raise AssertionError("Read-only result binding creation attempted")

    monkeypatch.setattr(capacity, "_LocalVolumeLock", local_lock)
    monkeypatch.setattr(capacity, "write_immutable", forbidden_write)
    ledger = capacity.CapacityLedger.__new__(capacity.CapacityLedger)
    ledger.common_root, ledger.directory, ledger.root = common, directory, root
    ledger.key, ledger.host, ledger.account = key, "synthetic-host", "S-1-5-21-1"
    ledger.volume, ledger.remote_scope = dict(volume_id="synthetic-volume"), None
    ledger.binding_path = common / (key + ".json")
    with pytest.raises(ValueError, match="no authoritative host-volume binding"):
        with ledger.locked(create=False):
            pytest.fail("Missing result binding admitted")
    assert not ledger.binding_path.exists()
    assert not (root / "identity.json").exists()
