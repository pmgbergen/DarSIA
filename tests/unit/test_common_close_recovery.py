"""Retained immutable close candidates permit metadata-only recovery."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from darsia_cache import common_qualification as common
from darsia_cache import first_qualification as first
from darsia_cache import windows
from darsia_cache.policy import CachePolicyError
from .test_common_qualification import provider


def candidate_store(monkeypatch, failure):
    """Keep genuine attempted bytes while injecting a short write or failed flush."""
    contents, writes = {}, []
    class Namespace:
        def __init__(self, path, identity):
            """Keep only the declared synthetic namespace location."""
            self.path = path
        def names(self):
            """Return retained candidate names without removing any object."""
            return list(contents)
        def read_once(self, name, **kwargs):
            """Return the exact retained bytes of the selected candidate."""
            return contents[name], {}
        def write_once(self, name, raw, **kwargs):
            """Model exclusive creation and retain bytes on an injected write failure."""
            if name in contents:
                raise FileExistsError(name)
            writes.append(name)
            contents[name] = raw[:17] if failure == "short" and len(writes) == 1 else raw
            if len(writes) == 1:
                raise OSError("Injected short write" if failure == "short" else "Injected flush failure")
        def close(self):
            """Close the synthetic handle without changing retained bytes."""
            pass
    monkeypatch.setattr(windows, "WindowsNamespace", Namespace)
    return contents, writes


def actual_namespace_transport(monkeypatch, view, names):
    """Keep original WindowsNamespace methods with an explicit synthetic kernel transport."""
    namespace_type = windows.WindowsNamespace
    missing_reads = []
    def missing(parent, name, **kwargs):
        """Expose an attempted read through the actual namespace method."""
        missing_reads.append(name)
        raise windows.CacheObjectIncomplete("Synthetic missing NTFS name", dict(component=name))
    def opened(path, identity):
        """Bind original namespace methods to the synthetic kernel transport."""
        assert dict(path=path, identity=identity) == view["output_namespace"]
        instance = object.__new__(namespace_type)
        instance.api = SimpleNamespace(names=lambda handle: list(names), child=missing)
        instance.handle, instance.ancestors, instance.closed = object(), [], False
        return instance
    monkeypatch.setattr(windows, "WindowsNamespace", opened)
    return missing_reads


@pytest.mark.parametrize("failure", ("short", "flush"))
def test_close_retry_retains_failed_candidate_and_never_overwrites_or_relaunches(monkeypatch, failure):
    """A retry retains failed bytes and reuses only an identical complete result."""
    view = common.stage_provider(provider(), "phase1")
    contents, writes = candidate_store(monkeypatch, failure)
    value = dict(format="darsia-common-phase-scientific-closure-v1", complete=True, exact="original")
    with pytest.raises(OSError):
        common._slot_write(view, "scientific_evidence", value)
    kept = deepcopy(contents)
    pin = common._slot_write(view, "scientific_evidence", value)
    assert all(contents[name] == raw for name, raw in kept.items())
    assert len(writes) == (2 if failure == "short" else 1)
    assert common._slot_read(view, "scientific_evidence") == (value, pin)
    assert common._slot_write(view, "scientific_evidence", value) == pin
    assert len(writes) == (2 if failure == "short" else 1)
    with pytest.raises(CachePolicyError, match="differs"):
        common._slot_write(view, "scientific_evidence", dict(value, exact="foreign"))


def test_exhausted_short_candidates_stay_held_for_new_released_namespace(monkeypatch):
    """Exhausted finite slots cannot trigger overwrite, deletion or a scientific rerun."""
    view = common.stage_provider(provider(), "phase1")
    contents, writes = candidate_store(monkeypatch, "short")
    for name in common._close_slot_names(view, "completion"):
        contents[name] = b'{"unfinished":'
    before = deepcopy(contents)
    with pytest.raises(CachePolicyError, match="exhausted"):
        common._slot_write(view, "completion", dict(format=common.STAGE_PROOF, complete=True))
    assert contents == before and writes == []


def test_original_supervisor_recovers_only_close_before_claim_or_native_construction(monkeypatch):
    """The original entry resumes metadata only before any new claim or native launch."""
    from darsia_cache import owned_process
    state = SimpleNamespace(common_mode=True, provider=common.stage_provider(provider(), "phase1"),
        request=dict(public_command=["exact-command"]))
    actual_namespace_transport(monkeypatch, state.provider,
        [state.provider["slots"]["native_completion"].upper()])
    state.consume = lambda lease: pytest.fail("A metadata retry consumed a new scientific claim")
    lease = SimpleNamespace(first_qualification=state)
    monkeypatch.setattr(first, "validate_startup", lambda lease: None)
    monkeypatch.setattr(owned_process, "_public_command", lambda lease: ["exact-command"])
    monkeypatch.setattr(owned_process, "OwnedWorker", lambda *a, **kw: pytest.fail("A metadata retry created a native worker"))
    native = dict(passed=True, observer_native_handles_closed=True)
    monkeypatch.setattr(common, "_slot_read", lambda provider, role: (native, {}))
    closed = []
    monkeypatch.setattr(first, "_close_common_stage_after_retirement", lambda s, l, n: closed.append(n))
    assert first.supervise_first_worker(lease) == 0
    assert closed == [native]


def test_failed_close_records_held_close_only_state_without_releasing_reservation(monkeypatch):
    """A close failure reports its precise recovery action and keeps the reservation."""
    state = SimpleNamespace(common_mode=True, binding={}, provider_binding={})
    records = []
    lease = SimpleNamespace(_record=lambda name, value: records.append(value))
    monkeypatch.setattr(common, "close_stage", lambda *args: (_ for _ in ()).throw(OSError("metadata unavailable")))
    with pytest.raises(OSError):
        first._close_common_stage_after_retirement(state, lease,
            dict(passed=True, observer_native_handles_closed=True))
    assert len(records) == 1 and records[0]["retry_action"] == "metadata-close-only"
    assert records[0]["scientific_worker_rerun_allowed"] is False
    assert records[0]["reservation_closed"] is False and records[0]["no_retry"] is False


@pytest.mark.parametrize("foreign_first", (False, True))
def test_matching_candidate_never_hides_another_complete_conflicting_candidate(monkeypatch, foreign_first):
    """Every complete retained candidate participates in exact-byte conflict checks."""
    view = common.stage_provider(provider(), "phase1")
    contents, _writes = candidate_store(monkeypatch, None)
    names = common._close_slot_names(view, "completion")
    wanted = dict(format=common.STAGE_PROOF, complete=True, exact="wanted")
    foreign = dict(wanted, exact="foreign")
    for name, document in zip(names, (foreign, wanted) if foreign_first else (wanted, foreign)):
        contents[name] = first._slot_encoded(document)
    before = deepcopy(contents)
    with pytest.raises(CachePolicyError, match="differs"):
        common._slot_write(view, "completion", wanted)
    with pytest.raises(CachePolicyError, match="disagree"):
        common._slot_read(view, "completion")
    assert contents == before


def test_derived_and_declared_slot_names_cannot_alias_by_windows_case():
    """Case variants of derived names cannot alias another admitted NTFS slot."""
    root = provider()
    derived = common._close_slot_names(common.stage_provider(root, "phase1"), "completion")[1]
    root["stages"]["phase2"]["slots"]["observation"] = derived.upper()
    with pytest.raises(CachePolicyError, match="distinct"):
        common.checked_provider(root)


def test_reserved_common_slot_is_refused_before_native_admission():
    """Reserved NTFS names fail during provider validation before scientific work."""
    root = provider()
    root["stages"]["phase1"]["slots"]["scientific_evidence"] = "CON.json"
    with pytest.raises(CachePolicyError, match="non-reserved"):
        common.checked_provider(root)


def test_fresh_common_supervisor_uses_actual_namespace_presence_before_native_launch(monkeypatch):
    """An absent native slot reaches original claim/start, without replacing either slot reader."""
    from darsia_cache import owned_process
    from .test_common_qualification import protocol
    from tests.unit.gui import test_cache_first_qualification as original
    state, _, _ = protocol()
    lease = original.lease_for(state)
    observed = deepcopy(state.required_capacity_bound["volume_cap_observation"])
    lease.policy.volume_cap = SimpleNamespace(maximum_size_bytes=lease.policy.quota_bytes,
        document=lambda: deepcopy(observed["binding"]))
    report = original.report(active_cache_remaining_bytes=0)
    report.update(hard_filesystem_quota_verified=True, current_cache_logical_bytes=0,
        volume_cap_observation=observed)
    lease.ledger.project = lambda requested: deepcopy(report)
    missing_reads = actual_namespace_transport(monkeypatch, state.provider, [])
    events = []
    class NativeStartReached(RuntimeError):
        """Stop this source-regression fixture at the native process transport boundary."""
    class NativeTransport:
        """Record the native boundary after the actual claim was consumed."""
        resume_attempted = False
        info = None
        def __init__(self, *args, **kwargs):
            """Record construction after original fresh claim consumption."""
            events.append("constructed")
        def start(self, *args):
            """Expose that the actual fresh-entry branch reaches native start."""
            events.append("started")
            raise NativeStartReached("Native transport reached")
        def close(self):
            """Retain ordinary controller cleanup after the synthetic boundary stop."""
            events.append("closed")
    observer = SimpleNamespace(admit_output=lambda lease: None,
        output_document=lambda *args: {}, cache_samples=[], reserve_samples=[],
        sampled_cache_logical_high_water_bytes=0, reserve_crossed=False, stop_response=None)
    monkeypatch.setattr(first, "validate_startup", lambda lease: None)
    monkeypatch.setattr(owned_process, "_public_command", lambda lease: state.request["public_command"])
    monkeypatch.setattr(owned_process, "OwnedWorker", NativeTransport)
    monkeypatch.setattr(first, "BoundOutputObservations", lambda value: observer)
    with pytest.raises(NativeStartReached, match="transport reached"):
        first.supervise_first_worker(lease)
    assert events == ["constructed", "started", "closed"]
    assert state.started is True and len(lease.ledger.documents) == 1
    assert state.provider["slots"]["native_completion"] not in missing_reads


def test_first_namespace_uses_actual_bound_quota_without_a_fixed_host_size_list():
    """Any positive bound quota is eligible here, while mismatched binding bytes refuse."""
    from .test_common_qualification import protocol
    from tests.unit.gui import test_cache_first_qualification as original
    state, _, _ = protocol()
    lease = original.lease_for(state)
    lease.policy.quota_bytes = 173 * 1024**3
    lease.policy.volume_cap = SimpleNamespace(maximum_size_bytes=lease.policy.quota_bytes)
    assert state._namespace(lease) is None
    lease.policy.quota_bytes += 512
    with pytest.raises(CachePolicyError, match="accepted one-host cap"):
        state._namespace(lease)
