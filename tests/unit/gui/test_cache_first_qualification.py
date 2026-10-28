"""Synthetic protocol cases; never a real cache profile or host admission."""
from contextlib import contextmanager
from copy import deepcopy
import os
from types import SimpleNamespace

import pytest

from darsia_cache import first_qualification as first
from darsia_cache.policy import CachePolicy, CachePolicyError, PHYSICAL_RESERVE_BYTES, budget_report
from darsia_cache.worker_adapter import WorkerCacheAdapter


def pin(name, digit="a"):
    return dict(path="C:/TESTONLY/"+name, sha256=digit*64, size_bytes=1)


def provider():
    producers=[pin("producer.py")]
    runtime=[pin("runtime.py", "b")]
    return dict(format=first.PROVIDER, qualification_id="1"*32,
        scope=dict(commit="c"*40, pipeline="native-phase-n1", backend="cpu",
            precision="float32", run="ac52", ordered_case_ids=["ac52_f"+str(i) for i in range(39)]),
        namespace_admission=pin("namespace.json"),
        output_namespace=dict(path="C:/TESTONLY/receipts",identity=dict(TESTONLY="not admitted")),
        slots={name:name+".json" for name in first.SLOTS},
        python_executable=pin("python.exe"), startup=pin("startup.json"),
        producer_source_closure=producers,runtime_source_closure=runtime,
        module_closure=producers+runtime)


def protocol(request_name="request.json", request_digit="a", task_digit="a", job_id="TESTONLY-job"):
    value=provider(); binding=pin("provider.json"); stores=pin("stores.toml")
    request=dict(format=first.REQUEST, execution_authorized=True, authority_ids=["Q97","Q99"],
        root_release=pin("release.json"), provider=binding, stores_source=stores,
        namespace_admission=value["namespace_admission"], scope=value["scope"],
        task=pin("task.json", task_digit),
        cache_stat_period_seconds=30,
        public_command=["C:/TESTONLY/python.exe","-B","-m","darsia.fluidflower.worker"])
    release=dict(format="darsia-cache-first-qualification-root-release-v1",
        execution_authorized=True,provider=binding,stores_source=stores,scope=value["scope"],
        request_payload_sha256=first._digest({k:v for k,v in request.items() if k!="root_release"}))
    task=deepcopy(value["scope"]);task.pop("backend")
    task.update(attempt_id="2"*32,job_id=job_id)
    objects={binding["path"]:value,request["root_release"]["path"]:release,
        "C:/TESTONLY/task.json":task,
        "C:/TESTONLY/"+request_name:request}
    read=lambda descriptor:deepcopy(objects[descriptor["path"]])
    state=first.FirstQualification(pin(request_name, request_digit),provider_binding=binding,
        stores_source=stores,namespace=value["namespace_admission"],read=read)
    return state,objects


def report(**changes):
    value=dict(hard_filesystem_quota_verified=True,reserve_bytes=PHYSICAL_RESERVE_BYTES,
        quota_headroom_bytes=500,
        free_physical_bytes=PHYSICAL_RESERVE_BYTES+first.BOOTSTRAP_CONTROL_MARGIN_BYTES+400,
        active_cache_remaining_bytes=20,required_noncache_bytes=30,
        current_cache_logical_bytes=0)
    value.update(changes)
    return value


def test_predeclared_provider_is_acyclic_not_a_fake_completed_profile():
    state,objects=protocol()
    assert set(state.provider)==set(provider())
    assert not any(key in state.provider for key in ("stores_source","request","profiles","passed"))
    assert state.binding not in state.provider.values()
    assert "maximum_family_processes" not in state.provider
    assert state.request["stores_source"]==pin("stores.toml")
    assert objects["C:/TESTONLY/release.json"]["request_payload_sha256"]==first._digest(
        {k:v for k,v in state.request.items() if k!="root_release"})


@pytest.mark.parametrize("mutation",[
    lambda v:v["scope"].update(pipeline="n1"),
    lambda v:v["scope"].update(precision="float64"),
    lambda v:v["scope"].update(backend="cuda"),
    lambda v:v["scope"].update(run="ac18"),
    lambda v:v["scope"].update(ordered_case_ids=["same"]*39),
    lambda v:v["scope"].update(ordered_case_ids=["a"]),
    lambda v:v["slots"].update(completion="../foreign.json"),
    lambda v:v["slots"].update(completion=v["slots"]["observation"]),
    lambda v:v["module_closure"].pop(),
    lambda v:v.update(profiles={"TESTONLY":"forbidden cycle"}),
])
def test_finite_provider_rejects_wrong_stage_cases_sources_and_output_aliases(mutation):
    value=provider();mutation(value)
    with pytest.raises(CachePolicyError):first.checked_provider(value)


def test_administrative_entitlement_uses_actual_headroom_less_existing_reserves():
    assert first.administrative_entitlement(report())==350
    assert first.administrative_entitlement(report(quota_headroom_bytes=100))==100
    # This number is not presented as an earlier observed or measured peak.


@pytest.mark.parametrize("control_bytes,sufficient", [
    (53248, True), (1024**3, True), (1024**3 + 1, False),
])
def test_q133_real_admission_guard_keeps_margin_for_control_writes(control_bytes, sufficient):
    # Exact preclaim observations from the zero-SCI fc6713 service. The real
    # budget calculator must still refuse if later control writes exceed 1GiB.
    policy = CachePolicy("C:/TESTONLY/cache", 500 * 1024**3, 48, {})
    initial_free = 536572362752
    initial = budget_report(policy, used_bytes=117590, reserved_unwritten_bytes=0,
        request_bytes=0, physical_free_bytes=initial_free)
    initial["hard_filesystem_quota_verified"] = True  # Synthetic accepted cap observation.
    maximum = first.administrative_entitlement(initial)
    assert first.BOOTSTRAP_CONTROL_MARGIN_BYTES == 1024**3
    assert maximum == 492548947968
    later = budget_report(policy, used_bytes=117590, reserved_unwritten_bytes=maximum,
        request_bytes=0, physical_free_bytes=initial_free-control_bytes)
    assert later["sufficient"] is sufficient
    assert later["physical_reserve_guard_satisfied"] is sufficient
    assert later["reserve_bytes"] == PHYSICAL_RESERVE_BYTES == 40 * 1024**3
    assert later["quota_bytes"] == 500 * 1024**3
    if control_bytes == 53248:
        old_saturated = budget_report(policy, used_bytes=117590,
            reserved_unwritten_bytes=initial_free-PHYSICAL_RESERVE_BYTES,
            request_bytes=0, physical_free_bytes=initial_free-control_bytes)
        assert old_saturated["sufficient"] is False


@pytest.mark.parametrize("available", [0, 1024**3-1, 1024**3])
def test_q133_first_admission_refuses_when_only_control_margin_is_available(available):
    value = report(free_physical_bytes=PHYSICAL_RESERVE_BYTES+20+30+available)
    with pytest.raises(CachePolicyError, match="No actual positive"):
        first.administrative_entitlement(value)


def test_q133_first_admission_keeps_explicit_margin_and_existing_pending_charges():
    value = report(free_physical_bytes=PHYSICAL_RESERVE_BYTES+20+30+1024**3+1)
    assert first.administrative_entitlement(value) == 1


@pytest.mark.parametrize("changes",[
    dict(hard_filesystem_quota_verified=False),dict(reserve_bytes=0),
    dict(quota_headroom_bytes=0),dict(free_physical_bytes=PHYSICAL_RESERVE_BYTES),
])
def test_first_admission_keeps_real_cap_and_original_free_reserve(changes):
    with pytest.raises(CachePolicyError):first.administrative_entitlement(report(**changes))


def test_modified_request_after_root_payload_release_refuses():
    state,objects=protocol()
    objects["C:/TESTONLY/request.json"]["public_command"].append("--changed")
    with pytest.raises(CachePolicyError,match="Root-released"):
        first.FirstQualification(pin("request.json"),provider_binding=state.provider_binding,
            stores_source=state.stores_source,namespace=state.provider["namespace_admission"],
            read=lambda bound:deepcopy(objects[bound["path"]]))


class Ledger_TESTONLY:
    def __init__(self):
        self.documents={};self.control=self;self.maximums=[];self.lock_active=False
    @contextmanager
    def locked(self,**kwargs):
        if self.lock_active:raise CachePolicyError("Nested cache transaction refused")
        self.lock_active=True
        try:yield self
        finally:self.lock_active=False
    def _require_lock(self):
        if not self.lock_active:raise CachePolicyError("Same host transaction required")
    def project(self,maximum):
        assert maximum==0
        return report(active_cache_remaining_bytes=0)
    def write_once(self,name,raw):
        if name in self.documents:raise CachePolicyError("TESTONLY immutable marker already exists")
        self.documents[name]=raw
    def names(self):return list(self.documents)
    def read_once(self,name):return self.documents[name],{}
    def require_reservation(self,*args):pass
    def require_unbound_scope(self,*args):pass


def lease_for(state):
    ledger=Ledger_TESTONLY()
    return SimpleNamespace(first_qualification=state,ledger=ledger,
        document={"cache_bootstrap":{"namespace_admission":state.provider["namespace_admission"]}},
        policy=SimpleNamespace(volume_cap=SimpleNamespace(maximum_size_bytes=500*1024**3),quota_bytes=500*1024**3,
            profiles={"jobs":state.provider_binding}),
        admission={"receipts":state.provider["output_namespace"]},
        owner=dict(pid=12,process_create_time="123.0"),source=state.stores_source)


def test_once_marker_is_shared_host_authority_and_repeated_provider_cannot_retry(monkeypatch):
    state,_=protocol();lease=lease_for(state)
    monkeypatch.setattr(first,"_outputs_absent",lambda value:None)
    state.consume(lease)
    assert state.started is True and len(lease.ledger.documents)==1
    replacement,_=protocol();replacement.provider["qualification_id"]="2"*32
    with pytest.raises(CachePolicyError,match="already exists"):replacement.consume(lease)


def test_existing_output_refuses_before_native_launch(monkeypatch):
    state,_=protocol();lease=lease_for(state)
    def output_exists(value):raise CachePolicyError("existing completed/partial profile")
    monkeypatch.setattr(first,"_outputs_absent",output_exists)
    with pytest.raises(CachePolicyError,match="existing completed/partial"):state.consume(lease)
    assert not lease.ledger.documents


def test_service_entitlement_requires_original_native_context_and_once_marker(monkeypatch):
    state,_=protocol();lease=lease_for(state)
    monkeypatch.setattr(first,"_outputs_absent",lambda value:None)
    with pytest.raises(CachePolicyError,match="controlled native Job"):
        state.service_entitlement(lease)
    state.consume(lease);lease.native_launch_context=pin("actual-native-launch.json")
    assert state.service_entitlement(lease)==370
    assert state.administrative["measured_profile"] is False
    assert state.administrative["bootstrap_control_margin_bytes"] == 1024**3
    assert state.administrative["admission"]["reserve_bytes"]==PHYSICAL_RESERVE_BYTES
    assert len(lease.ledger.documents)==1  # No service reservation is duplicated here.


def test_different_spawned_owner_cannot_reuse_first_claim(monkeypatch):
    state,_=protocol();lease=lease_for(state);task=task_for(state)
    launch_for_claim_TESTONLY(state, lease, monkeypatch)
    with lease.ledger.locked():
        state.remember_claim_locked(lease,SimpleNamespace(token=task["attempt_id"],task=task))
    other,_=protocol();lease.first_qualification=other;lease.owner=dict(pid=99,process_create_time="999.0")
    with pytest.raises(CachePolicyError,match="different first claim/owner"):
        with lease.ledger.locked():
            other.remember_claim_locked(lease,SimpleNamespace(token=task["attempt_id"],task=task))


def test_exact_task_contract_binds_calibration_and_all_original_input_metadata():
    state,_=protocol();lease=lease_for(state);task=task_for(state)
    altered=deepcopy(task);altered["calibration_sha256"]="f"*64
    with pytest.raises(CachePolicyError,match="exact released"):
        state.require_task(lease,altered)
    altered=deepcopy(task);altered["input_catalog"]={"TESTONLY":"different actual RAW binding"}
    with pytest.raises(CachePolicyError,match="exact released"):
        state.require_task(lease,altered)


def task_for(state):
    task=deepcopy(state.provider["scope"]);task.pop("backend")
    task.update(attempt_id="2"*32,job_id="TESTONLY-job")
    return task


def launch_for_claim_TESTONLY(state, lease, monkeypatch):
    monkeypatch.setattr(first,"_outputs_absent",lambda value:None)
    state.consume(lease)
    lease.native_launch_context=pin("actual-native-launch.json")


def test_one_real_claim_can_bind_but_second_attempt_or_changed_cases_cannot(monkeypatch):
    state,_=protocol();lease=lease_for(state);task=task_for(state)
    assert state.require_task(lease,task) is True
    assert state.claim is None
    launch_for_claim_TESTONLY(state, lease, monkeypatch)
    with lease.ledger.locked():
        state.remember_claim_locked(lease,SimpleNamespace(token=task["attempt_id"],task=task))
    changed=deepcopy(task);changed["attempt_id"]="3"*32
    with pytest.raises(CachePolicyError):state.require_task(lease,changed)
    changed=deepcopy(task);changed["ordered_case_ids"].reverse()
    with pytest.raises(CachePolicyError,match="exact released"):state.require_task(lease,changed)


def test_first_claim_uses_existing_service_without_inventing_measured_zero_or_new_scope(monkeypatch):
    state,_=protocol();lease=lease_for(state)
    adapter=WorkerCacheAdapter.__new__(WorkerCacheAdapter)
    adapter.lease=lease;adapter.runtime=SimpleNamespace(backend="cpu",commit="c"*40)
    adapter.profiles={};adapter.claims={}
    task=task_for(state)
    launch_for_claim_TESTONLY(state, lease, monkeypatch)
    assert adapter.prepare_first_claim(task,None)==state.binding
    assert state.claim is None and len(lease.ledger.documents)==1
    proof=adapter.profiles[task["attempt_id"]]
    assert proof["first_administration"] is True and proof["coverage_sha256"] is None
    assert "maximum_observed_cache_bytes" not in proof
    lease.reservation=dict(token="4"*32,pool="5"*64,sha256="6"*64)
    records=[]
    lease._record=lambda name,value:(records.append(deepcopy(value)) or pin(name))
    monkeypatch.setattr("darsia_cache.worker_adapter._owner",lambda:(lease.owner,{}))
    monkeypatch.setattr("darsia_cache.worker_adapter.worker_identity_epoch",
                        lambda pid:lease.owner["process_create_time"])
    claim=SimpleNamespace(token=task["attempt_id"],task=task)
    with lease.ledger.locked():state.remember_claim_locked(lease,claim)
    adapter.register_claim_allocation(claim,lease.ledger,{"token":claim.token},lease.owner,0,state.binding)
    assert records[0]["format"]=="darsia-worker-cache-first-claim-service-binding-v1"
    assert records[0]["claim_scope_created"] is False
    assert records[0]["service_reservation"]==lease.reservation
    assert records[0]["writer_coverage_sha256"] is None
    assert len(lease.ledger.documents)==3  # Launch proof, actual consumption and exact claim authority.


# The Q197 worker process. Its 100 ns FILETIME count is odd and above 2**53, so
# psutil's double division and the exact division give two epoch strings.
Q197_TICKS=134369890719926969


def q197_epochs():
    """Return the exact _owner() epoch and the worker queue (psutil) epoch."""
    since_1970=Q197_TICKS-116444736000000000
    return str(since_1970/10**7),str(float(since_1970)/10000000.0)


def q197_claim_TESTONLY(monkeypatch):
    """A first claim whose live process is the Q197 one, ready to bind."""
    state,_=protocol();lease=lease_for(state)
    exact,_=q197_epochs()
    lease.owner=dict(pid=12,process_create_time=exact,creation_filetime_100ns=str(Q197_TICKS))
    adapter=WorkerCacheAdapter.__new__(WorkerCacheAdapter)
    adapter.lease=lease;adapter.runtime=SimpleNamespace(backend="cpu",commit="c"*40)
    adapter.profiles={};adapter.claims={}
    task=task_for(state)
    launch_for_claim_TESTONLY(state, lease, monkeypatch)
    adapter.prepare_first_claim(task,None)
    lease.reservation=dict(token="4"*32,pool="5"*64,sha256="6"*64)
    lease._record=lambda name,value:pin(name)
    monkeypatch.setattr("darsia_cache.worker_adapter._owner",lambda:(deepcopy(lease.owner),{}))
    created={12:float(Q197_TICKS-116444736000000000)/10000000.0}
    monkeypatch.setattr("psutil.Process",lambda pid:SimpleNamespace(create_time=lambda:created[pid]))
    claim=SimpleNamespace(token=task["attempt_id"],task=task)
    with lease.ledger.locked():state.remember_claim_locked(lease,claim)
    return state,lease,adapter,claim


def test_q197_claim_owner_in_the_worker_queue_epoch_form_binds_its_own_process(monkeypatch):
    """Q197: the queue's psutil epoch and the exact epoch name the same live process."""
    exact,queue=q197_epochs()
    assert (exact,queue)==("1792515471.992697","1792515471.9926968")
    state,lease,adapter,claim=q197_claim_TESTONLY(monkeypatch)
    owner=dict(machine="main-cpu",pid=12,process_create_time=queue)
    adapter.register_claim_allocation(claim,lease.ledger,{"token":claim.token},owner,0,state.binding)
    assert adapter.claims[claim.token]["owner"]==owner


@pytest.mark.parametrize("change",["pid","epoch","bootstrap_form","kernel_count"])
def test_q197_claim_owner_check_stays_exact_for_any_other_process(monkeypatch,change):
    """Another pid, epoch, epoch form or kernel FILETIME count is still refused."""
    _,queue=q197_epochs()
    state,lease,adapter,claim=q197_claim_TESTONLY(monkeypatch)
    owner=dict(machine="main-cpu",pid=12,process_create_time=queue)
    if change=="pid":
        owner["pid"]=13
    elif change=="epoch":
        owner["process_create_time"]=str(float(Q197_TICKS+3-116444736000000000)/10000000.0)
        assert owner["process_create_time"]!=queue
    elif change=="bootstrap_form":
        owner["process_create_time"]=lease.owner["process_create_time"]
    else:
        moved=dict(lease.owner,creation_filetime_100ns=str(Q197_TICKS+1))
        monkeypatch.setattr("darsia_cache.worker_adapter._owner",lambda:(deepcopy(moved),{}))
    with pytest.raises(CachePolicyError,match="owner binding differs"):
        adapter.register_claim_allocation(claim,lease.ledger,{"token":claim.token},owner,0,state.binding)
    assert claim.token not in adapter.claims


def test_q197_only_the_live_claim_owner_may_close_its_zero_entitlement(monkeypatch):
    """close_claim uses the same exact owner rule as the claim binding."""
    _,queue=q197_epochs()
    state,lease,adapter,claim=q197_claim_TESTONLY(monkeypatch)
    owner=dict(machine="main-cpu",pid=12,process_create_time=queue)
    adapter.register_claim_allocation(claim,lease.ledger,{"token":claim.token},owner,0,state.binding)
    moved=dict(lease.owner,creation_filetime_100ns=str(Q197_TICKS+1))
    monkeypatch.setattr("darsia_cache.worker_adapter._owner",lambda:(deepcopy(moved),{}))
    with pytest.raises(CachePolicyError,match="actual live claim owner"):
        adapter.close_claim(claim)


@pytest.mark.skipif(os.name!="nt",reason="the kernel FILETIME count is read on Windows")
def test_worker_identity_epoch_is_psutils_double_division_of_the_kernel_count():
    """worker_identity_epoch is the queue form, a function of the kernel FILETIME count."""
    import ctypes
    from ctypes import wintypes
    import psutil
    from darsia_cache.adapters import worker_identity_epoch
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
    times=[wintypes.FILETIME() for _ in range(4)]
    assert kernel.GetProcessTimes(kernel.GetCurrentProcess(),*[ctypes.byref(t) for t in times])
    since_1970=((times[0].dwHighDateTime<<32)|times[0].dwLowDateTime)-116444736000000000
    assert worker_identity_epoch(os.getpid())==str(psutil.Process(os.getpid()).create_time())
    assert worker_identity_epoch(os.getpid())==str(float(since_1970)/10000000.0)


def test_q138_preclaim_failure_allows_only_a_new_released_request(monkeypatch):
    state,_=protocol();lease=lease_for(state)
    launch_for_claim_TESTONLY(state,lease,monkeypatch)
    original=dict(lease.ledger.documents)
    assert "cache-first-"+state._once_key()+".json" not in original
    replacement,_=protocol("fresh-request.json","d","e","TESTONLY-fresh-namespace")
    lease.first_qualification=replacement
    replacement.consume(lease)
    assert state._once_key()==replacement._once_key()
    assert state._launch_marker_name()!=replacement._launch_marker_name()
    assert len(lease.ledger.documents)==2
    assert all(lease.ledger.documents[name]==raw for name,raw in original.items())
    with pytest.raises(CachePolicyError,match="already exists"):
        replacement.consume(lease)


def test_q138_preclaim_preparation_and_lost_queue_race_consume_nothing(monkeypatch):
    from darsia.fluidflower.pipeline.capacity import ClaimCapacity
    state,_=protocol();lease=lease_for(state)
    launch_for_claim_TESTONLY(state,lease,monkeypatch)
    original=dict(lease.ledger.documents)
    adapter=WorkerCacheAdapter.__new__(WorkerCacheAdapter)
    adapter.lease=lease;adapter.runtime=SimpleNamespace(backend="cpu",commit="c"*40)
    adapter.profiles={};adapter.claims={}
    assert adapter.prepare_first_claim(deepcopy(state.task),None)==state.binding
    context=ClaimCapacity.__new__(ClaimCapacity)
    context.bind(None)  # The atomic queue claim was lost; no claim-side state exists.
    assert state.claim is None and lease.ledger.documents==original


def test_q138_actual_claim_consumes_before_result_reservation_can_fail(monkeypatch):
    from darsia.fluidflower.pipeline.capacity import ClaimCapacity
    state,_=protocol();lease=lease_for(state)
    launch_for_claim_TESTONLY(state,lease,monkeypatch)
    claim=SimpleNamespace(token=state.task["attempt_id"],task=deepcopy(state.task))
    context=ClaimCapacity.__new__(ClaimCapacity)
    context.runtime=SimpleNamespace(cache_adapter=SimpleNamespace(lease=lease),root="TESTONLY-root")
    context.owner=lease.owner;context.estimates={claim.token:object()}
    def result_reservation_failed(*args):raise OSError("TESTONLY after-claim reservation failure")
    context.ledger=SimpleNamespace(bind=result_reservation_failed)
    with pytest.raises(OSError,match="after-claim reservation"):
        with lease.ledger.locked():context.bind(claim)
    once="cache-first-"+state._once_key()+".json"
    assert once in lease.ledger.documents and once[:-5]+"-claim.json" in lease.ledger.documents
    assert state.claim==claim.task
    replacement,_=protocol("fresh-request.json","d","e","TESTONLY-fresh-namespace")
    lease.first_qualification=replacement
    with pytest.raises(CachePolicyError,match="actual first claim already consumed"):
        replacement.consume(lease)


def test_q138_actual_claim_success_preserves_result_then_cache_binding_order(monkeypatch):
    from darsia.fluidflower.pipeline.capacity import ClaimCapacity
    state,_=protocol();lease=lease_for(state)
    launch_for_claim_TESTONLY(state,lease,monkeypatch)
    claim=SimpleNamespace(token=state.task["attempt_id"],task=deepcopy(state.task))
    context=ClaimCapacity.__new__(ClaimCapacity)
    context.runtime=SimpleNamespace(cache_adapter=SimpleNamespace(lease=lease),
        root="TESTONLY-root",capacities={})
    context.owner=lease.owner;estimate=object();context.estimates={claim.token:estimate}
    events=[];reservation={"TESTONLY":"original result reservation"}
    def bind_result(root,actual,owner,bound_estimate):
        assert root==context.runtime.root and actual is claim and owner==lease.owner
        assert bound_estimate is estimate
        assert "cache-first-"+state._once_key()+".json" in lease.ledger.documents
        assert state.claim==claim.task
        events.append("result");return reservation
    context.ledger=SimpleNamespace(bind=bind_result)
    context.cache=SimpleNamespace(bind=lambda actual:events.append("cache") if actual is claim else pytest.fail())
    with lease.ledger.locked():context.bind(claim)
    assert events==["result","cache"]
    assert context.runtime.capacities[claim.token]==(context.ledger,reservation,lease.owner)


def test_q138_actual_claim_requires_original_lock_and_exact_task_token(monkeypatch):
    state,_=protocol();lease=lease_for(state)
    launch_for_claim_TESTONLY(state,lease,monkeypatch)
    original=dict(lease.ledger.documents)
    claim=SimpleNamespace(token=state.task["attempt_id"],task=deepcopy(state.task))
    with pytest.raises(CachePolicyError,match="Same host transaction"):
        state.remember_claim_locked(lease,claim)
    with pytest.raises(CachePolicyError,match="claim token"):
        with lease.ledger.locked():
            state.remember_claim_locked(lease,SimpleNamespace(token="f"*32,task=claim.task))
    assert lease.ledger.documents==original


def test_q138_legacy_consumed_marker_is_never_rearmed_or_overwritten(monkeypatch):
    state,_=protocol();lease=lease_for(state)
    old_name="cache-first-"+state._once_key()+".json"
    old_bytes=b'{"TESTONLY":"retained old preclaim marker; no deletion or repair"}\n'
    lease.ledger.documents[old_name]=old_bytes
    monkeypatch.setattr(first,"_outputs_absent",lambda value:None)
    with pytest.raises(CachePolicyError,match="actual first claim already consumed"):
        state.consume(lease)
    assert lease.ledger.documents=={old_name:old_bytes}


def test_q138_claim_marker_write_failure_still_consumes_actual_claim(monkeypatch):
    state,_=protocol();lease=lease_for(state)
    launch_for_claim_TESTONLY(state,lease,monkeypatch)
    original_write=lease.ledger.write_once
    def fail_claim_record(name,raw):
        if name.endswith("-claim.json"):raise OSError("TESTONLY immutable claim record failure")
        return original_write(name,raw)
    monkeypatch.setattr(lease.ledger,"write_once",fail_claim_record)
    claim=SimpleNamespace(token=state.task["attempt_id"],task=deepcopy(state.task))
    with pytest.raises(OSError,match="claim record failure"):
        with lease.ledger.locked():state.remember_claim_locked(lease,claim)
    once="cache-first-"+state._once_key()+".json"
    assert once in lease.ledger.documents and once[:-5]+"-claim.json" not in lease.ledger.documents
    with pytest.raises(CachePolicyError,match="actual first claim already consumed"):
        state.consume(lease)


@pytest.mark.parametrize("mutation",[
    lambda marker:marker.update(format="darsia-cache-first-qualification-consumed-v1"),
    lambda marker:marker.update(request=pin("foreign-request.json")),
    lambda marker:marker.update(provider=pin("foreign-provider.json")),
    lambda marker:marker.update(stores_source=pin("foreign-stores.toml")),
    lambda marker:marker.update(scope=dict(TESTONLY="foreign scope")),
    lambda marker:marker.update(no_retry=False),
])
def test_q138_actual_claim_keeps_original_native_launch_binding_guards(monkeypatch,mutation):
    from darsia_cache.capacity import _json_bytes
    state,_=protocol();lease=lease_for(state)
    launch_for_claim_TESTONLY(state,lease,monkeypatch)
    name=state._launch_marker_name();marker=_json_bytes(lease.ledger.documents[name]);mutation(marker)
    lease.ledger.documents[name]=first._encoded(marker)
    before=dict(lease.ledger.documents)
    claim=SimpleNamespace(token=state.task["attempt_id"],task=deepcopy(state.task))
    with pytest.raises(CachePolicyError,match="one-shot first launch marker"):
        with lease.ledger.locked():state.remember_claim_locked(lease,claim)
    assert lease.ledger.documents==before


def test_q138_exact_claim_is_idempotent_but_another_released_request_cannot_claim(monkeypatch):
    state,_=protocol();lease=lease_for(state)
    launch_for_claim_TESTONLY(state,lease,monkeypatch)
    other,_=protocol("fresh-request.json","d","e","TESTONLY-fresh-namespace")
    lease.first_qualification=other;other.consume(lease)
    lease.first_qualification=state
    claim=SimpleNamespace(token=state.task["attempt_id"],task=deepcopy(state.task))
    with lease.ledger.locked():state.remember_claim_locked(lease,claim)
    original=dict(lease.ledger.documents)
    with lease.ledger.locked():state.remember_claim_locked(lease,claim)
    assert lease.ledger.documents==original
    lease.first_qualification=other
    with pytest.raises(CachePolicyError,match="different first claim/owner"):
        with lease.ledger.locked():
            other.remember_claim_locked(lease,SimpleNamespace(token=other.task["attempt_id"],task=deepcopy(other.task)))
    assert lease.ledger.documents==original


def test_sampled_peak_is_not_lifetime_peak_and_kernel_total_is_not_concurrency():
    state,_=protocol();collector=first.FiniteObservations(state)
    lease=lease_for(state);lease.policy.root="C:/TESTONLY/cache"
    rows=iter([dict(logical_bytes=90),dict(logical_bytes=12)])
    lease.measure=lambda root:next(rows)
    owned=SimpleNamespace(lease=lease,service_binding=pin("service.json"),
        observed_binding=pin("native.json"),job=1,
        api=SimpleNamespace(lifetime_memory=lambda job:dict(peak_job_committed_bytes=123)))
    collector.observe(owned,dict(active_processes=2));collector.observe(owned,dict(active_processes=0),force_cache=True)
    result=collector.document(owned,dict(accounting=dict(total_processes=5,active_processes=0),wrapper_exit_code=0),
        pin("retirement.json"),dict(TESTONLY="not production eligibility"))
    assert result["sampled_cache_logical_high_water_bytes"]==90
    assert result["sampled_cache_logical_high_water_is_lifetime_peak"] is False
    assert result["sampled_active_processes_high_water"]==2
    assert result["kernel_total_processes"]==5
    assert result["kernel_total_processes_is_concurrent_peak"] is False
    assert result["conservative_family_process_bound_from_kernel_total"]==5
    assert result["passed"] is False and result["writer_lifetime_cache_bytes_proof"] is None


def test_cache_stat_cadence_is_declared_bounded_and_preserves_every_cheap_kernel_count():
    state,_=protocol();now=[0.0]
    collector=first.FiniteObservations(state,clock=lambda:now[0])
    lease=lease_for(state);lease.policy.root="C:/TESTONLY/cache";scans=[]
    def measure(root):
        scans.append(now[0]);return dict(logical_bytes=len(scans))
    lease.measure=measure;owned=SimpleNamespace(lease=lease)
    for instant,active in [(0,2),(.2,3),(.4,4),(29.9,5),(30,6),(30.2,0)]:
        now[0]=instant;collector.observe(owned,dict(active_processes=active))
    assert scans==[0,30]
    assert collector.native_poll_count==6
    assert collector.sampled_active_processes_high_water==6
    collector.observe(owned,dict(active_processes=0),force_cache=True)
    assert scans==[0,30,30.2]
    assert collector.cache_samples[-1]["forced_terminal"] is True
    assert collector.cache_samples[1]["actual_gap_since_previous_start_seconds"]==30
    assert collector.native_poll_count==6


def test_exiting_module_inventory_preserves_unknown_sources_instead_of_inheriting_old_closure(monkeypatch):
    state,_=protocol()
    calls=[]
    monkeypatch.setattr("darsia_cache.owned_process._exact_source",lambda value:calls.append(value))
    monkeypatch.setattr(first,"sys",SimpleNamespace(modules={
        "known":SimpleNamespace(__file__="C:/TESTONLY/producer.py"),
        "unclassified":SimpleNamespace(__file__="C:/TESTONLY/unclassified.py"),
        "builtin":SimpleNamespace()}))
    result=state.loaded_modules()
    assert calls==[state.provider["producer_source_closure"][0]]
    assert result["all_file_backed_modules_predeclared_and_exact"] is False
    assert result["undeclared"][0]["name"]=="unclassified"
    assert result["actual_task"] is None
    assert result["this_is_not_process_retirement"] is True


@pytest.mark.parametrize("evidence",[None,{},dict(complete=True,passed=True),
    dict(maximum_observed_cache_bytes=1,all_writers=True),
    dict(required_capacity_bound=500*1024**3,physical_cap=True)])
def test_polling_or_fixed_cap_cannot_fabricate_a_passing_lifetime_profile(evidence):
    with pytest.raises(CachePolicyError):
        first.require_publishable(dict(format=first.OBSERVATION,complete=True),evidence)


def test_direct_old_profile_reader_is_unchanged_and_new_provider_never_falls_back(monkeypatch):
    old=dict(format="darsia-cache-import-profile-v1",TESTONLY="old reader returns exact same object")
    assert first.resolve_profile(pin("old.json"),"import",lambda binding:old) is old
    def incomplete(value,role):raise CachePolicyError("no completed provider bundle")
    monkeypatch.setattr(first,"_slot_read",incomplete)
    with pytest.raises(CachePolicyError,match="no completed"):
        first.resolve_profile(pin("new.json"),"import",lambda binding:provider())


def bound_plan_TESTONLY():
    capacity = 500 * 1024**3 - 2 * 1024**2
    return dict(format=first.CAPACITY_BOUND,
        authority_id="Q106-answer-sampled-operational-reserve-hard-bound-is-Q87-volume-no-new-seam",
        volume_cap_observation=dict(format="darsia-readonly-fixed-vhdx-cap-observation-v1",
            verified=True, filesystem="NTFS", is_loaded=True, volume_total_bytes=capacity),
        volume_capacity_bytes=capacity, free_volume_reserve_bytes=PHYSICAL_RESERVE_BYTES,
        required_capacity_bound_bytes=capacity,
        operational_admission_headroom_bytes=capacity-PHYSICAL_RESERVE_BYTES,
        reserve_watchdog_period_seconds=1,
        hard_bound_semantics="actual-fixed-volume-only-not-capacity-minus-reserve",
        reserve_semantics="operational-admission-plus-sampled-watchdog-not-hard-floor",
        exclusive_volume_writer_required=True,
        sampled_observed_highwater_is_complete=False,
        enforcement_state="PRELAUNCH_HOLD_N43_REFERENCE_OUTPUT_BUDGET_AND_PROFILE_PRODUCER")


def test_typed_capacity_bound_uses_actual_usable_capacity_not_sampled_highwater():
    value = bound_plan_TESTONLY()
    actual = first.checked_required_capacity_bound(value)
    assert actual == value and actual is not value
    assert "maximum_observed_cache_bytes" not in actual
    assert actual["enforcement_state"] == "PRELAUNCH_HOLD_N43_REFERENCE_OUTPUT_BUDGET_AND_PROFILE_PRODUCER"
    assert actual["required_capacity_bound_bytes"]==actual["volume_capacity_bytes"]
    assert actual["operational_admission_headroom_bytes"]==actual["volume_capacity_bytes"]-PHYSICAL_RESERVE_BYTES


@pytest.mark.parametrize("change", [
    lambda v: v.update(required_capacity_bound_bytes=v["required_capacity_bound_bytes"]+1),
    lambda v: v.update(free_volume_reserve_bytes=0),
    lambda v: v.update(volume_capacity_bytes=500*1024**3),
    lambda v: v.update(sampled_observed_highwater_is_complete=True),
    lambda v: v.update(enforcement_state="PASSED"),
    lambda v: v["volume_cap_observation"].update(verified=False),
    lambda v: v.update(required_capacity_bound_bytes=v["operational_admission_headroom_bytes"]),
    lambda v: v.update(reserve_watchdog_period_seconds=30),
    lambda v: v.update(exclusive_volume_writer_required=False),
    lambda v: v.update(reserve_semantics="hard-never-consumed-free-floor"),
])
def test_typed_bound_cannot_shrink_reserve_relabel_samples_or_claim_enforcement(change):
    value = bound_plan_TESTONLY(); change(value)
    with pytest.raises(CachePolicyError): first.checked_required_capacity_bound(value)


@pytest.fixture
def output_entitlement_TESTONLY():
    state, _ = protocol(); task = task_for(state)
    task_binding = pin("task.json"); footprint_binding = pin("R4-footprint.json")
    footprint = dict(format="first-qualification-predeclared-output-entitlement-source-data-not-profile-v1",
        authority_id="Q104-answer-first-qualification-produces-result-space-profile-under-predeclared-entitlement",
        run="ac52", case_count=39, ordered_case_ids=task["ordered_case_ids"], factor=2,
        not_current_measurement_or_profile=True, no_first_qualification_or_adoption=True,
        files=[dict(path="C:/TESTONLY/R4/"+case+"/"+role,size_bytes=17,case_id=case,roles=[role],
            actual_stat=dict(size_bytes=17),declared_original_sha256="a"*64)
            for case in task["ordered_case_ids"]
            for role in ("products.npz","result.json","preview.png","native_statistics.json.gz","geometry")])
    total = sum(row["size_bytes"] for row in footprint["files"])
    footprint.update(unique_output_file_count=len(footprint["files"]),
        actual_result_footprint_bytes=total, administrative_output_entitlement_bytes=2*total)
    value = dict(format="darsia-first-qualification-output-entitlement-v1",
        authority_id=footprint["authority_id"],task=task_binding,
        scope={key:task[key] for key in ("commit","pipeline","precision","run","ordered_case_ids")},
        R4_footprint=footprint_binding,administrative_output_entitlement_bytes=2*total,
        measured_profile=False,enforcement_state="PRELAUNCH_HOLD_N43_REFERENCE_OUTPUT_BUDGET_AND_PROFILE_PRODUCER")
    return task,task_binding,value,footprint


def test_output_entitlement_is_exact_Q104_factor_without_measured_profile_claim(output_entitlement_TESTONLY):
    task,binding,value,footprint=output_entitlement_TESTONLY
    actual=first.checked_output_entitlement(task,value,task_binding=binding,read=lambda _:footprint)
    assert actual==value and actual["measured_profile"] is False
    assert actual["administrative_output_entitlement_bytes"]==2*footprint["actual_result_footprint_bytes"]


@pytest.mark.parametrize("change",[
    lambda v,f:v.update(administrative_output_entitlement_bytes=35),
    lambda v,f:v.update(measured_profile=True),
    lambda v,f:v["scope"].update(commit="d"*40),
    lambda v,f:v.update(task=pin("other-task.json")),
    lambda v,f:f.update(factor=3),
    lambda v,f:f["files"].append(deepcopy(f["files"][0])),
    lambda v,f:f["files"][0]["actual_stat"].update(size_bytes=18),
    lambda v,f:f["files"][0].update(roles=["unrelated"]),
])
def test_output_entitlement_refuses_unbound_case_source_size_or_factor(output_entitlement_TESTONLY,change):
    task,binding,value,footprint=output_entitlement_TESTONLY;change(value,footprint)
    with pytest.raises(CachePolicyError):
        first.checked_output_entitlement(task,value,task_binding=binding,read=lambda _:footprint)


def test_new_bound_mode_refuses_before_consumption_or_any_native_start(monkeypatch):
    state,_=protocol();state.bound_mode=True;lease=lease_for(state)
    monkeypatch.setattr(first,"_outputs_absent",lambda _:pytest.fail("No output/namespace access before HOLD"))
    with pytest.raises(CachePolicyError,match="Bound prelaunch requires actual"):
        state.consume(lease)
    assert not lease.ledger.documents and state.started is False


def test_Q104_whole_output_entitlement_never_invents_bytes_per_frame():
    from darsia.fluidflower.pipeline.capacity import PredeclaredFirstOutputEntitlement
    value = PredeclaredFirstOutputEntitlement("ac52", 39, first.FIRST_OUTPUT_BASE_ENTITLEMENT_BYTES,
        pin("request.json"), pin("output-plan.json"), pin("all-stage.json"), dict(TESTONLY=True))
    assert value.products_estimated_bytes == 65728995330
    assert value.needed(other_remaining_bytes=17, reserve_bytes=PHYSICAL_RESERVE_BYTES) == (
        65728995330 + 17 + PHYSICAL_RESERVE_BYTES)
    assert not hasattr(value, "products_per_frame_estimated_bytes")
    assert "including preparation/ops/in-flight" in value.model


def all_stage_reference_TESTONLY(*, label="reference", reference_commit="d" * 40):
    state, _ = protocol(); task = task_for(state)
    task["backend"] = "cpu"
    task["configuration"] = dict(root_alias="series", relative_path="run.json", sha256="1" * 64, bytes=7)
    completion = pin(label + "-completed.json")
    value = dict(format="darsia-first-qualification-all-stage-reference-v1",
        candidate_commit=task["commit"], reference_commit=reference_commit,
        configuration=task["configuration"], ordered_case_ids=task["ordered_case_ids"],
        reference_completion=completion, reference_Root_acceptance=pin(label + "-Root-acceptance.json"),
        reference_native_host_evidence=pin(label + "-native-host.json"),
        reference_source_compatibility=pin(label + "-source-compatibility.json"),
        requested_native_layout={"TESTONLY": "no native admission"}, output_writers=[],
        measurement=pin(label + "-all-stage-measurement.json"), not_a_cache_profile=True)
    root = first.PureWindowsPath("C:/TESTONLY/" + label + "-attempt")
    inventory = []; objects = {}; records = []; producers = []
    def asset(relative):
        inventory.append(dict(relative_path=relative, bytes=1, sha256="b" * 64))
        return dict(path=str(root / relative), size_bytes=1, sha256="b" * 64)
    manifest = asset("manifest.json"); table = asset("mass_table.csv")
    for case in task["ordered_case_ids"]:
        stage = {name: asset("preparation/" + case + "/" + name) for name in (
            "native_log.npy", "native_valid.npy", "temporal_log.npy", "temporal_valid.npy", "phase_area.npz")}
        files = {name: {key: val for key, val in asset("frames/" + case + "/" + name).items() if key != "path"}
            for name in ("products.npz", "native_geometry.npz", "native_statistics.json.gz", "result.json", "commit.json")}
        record = dict(frame_id=case, nas_directory=str(root / "frames" / case), files=files)
        producer = dict(case_id=case, attempt_directory=str(root), manifest=manifest,
            commit=dict(path=str(root / "frames" / case / "commit.json"), **files["commit.json"]),
            result=dict(path=str(root / "frames" / case / "result.json"), **files["result.json"]),
            frame_registration=asset("ops/frame_commit_" + case + ".json"),
            publication=asset("ops/nas_copy_" + case + ".json"))
        original_files = {name: row for name, row in files.items() if name != "commit.json"}
        objects[producer["commit"]["path"]] = dict(complete=True, frame_id=case, commit_id=case,
            manifest_sha256=manifest["sha256"], files=original_files)
        objects[producer["result"]["path"]] = dict(complete=True, case_id=case,
            prediction=dict(pipeline=dict(complete=True, case_id=case,
                configuration_sha256=task["configuration"]["sha256"], files=stage)))
        objects[producer["frame_registration"]["path"]] = dict(frame_id=case,
            commit=producer["commit"], manifest_sha256=manifest["sha256"], files=original_files)
        objects[producer["publication"]["path"]] = record
        records.append(record); producers.append(producer)
    objects.update({completion["path"]: dict(complete=True, whole_run=True, worker_references_closed=True,
        commit=value["reference_commit"], newly_computed_frames=39, manifest=manifest, table=table,
        backend="cpu", actual_worker=dict(machine="main-cpu"),
        pipeline=task["pipeline"], precision=task["precision"], ordered_case_ids=task["ordered_case_ids"],
        case_count=39, frames=39, configuration=task["configuration"], reused_case_ids=[],
        records=records, frame_producers=producers),
        value["reference_Root_acceptance"]["path"]: {"TESTONLY": "not Root authorization"},
        value["reference_native_host_evidence"]["path"]: {"TESTONLY": "not actual native host proof"},
        value["reference_source_compatibility"]["path"]: {"TESTONLY": "not source qualification"},
        value["measurement"]["path"]: dict(format="darsia-first-qualification-all-stage-output-inventory-v1",
            reference_completion=completion, root=str(root), root_identity={"TESTONLY": True},
            inventory=inventory, total_logical_bytes=len(inventory),
            inventory_scope="whole-owned-result-attempt-including-preparation-and-ops",
            closed_writer_evidence=pin(label + "-closed-writers.json"), complete=True),
        "C:/TESTONLY/" + label + "-closed-writers.json": {"TESTONLY": "not actual retirement"}})
    return task, value, objects


@pytest.mark.parametrize("change", [
    lambda v, m: v.update(candidate_commit="e" * 40),
    lambda v, m: v.update(reference_Root_acceptance=None),
    lambda v, m: v.update(reference_native_host_evidence=None),
    lambda v, m: m.update(inventory_scope="final-products-only"),
    lambda v, m: m.update(complete=False),
    lambda v, m: m.update(total_logical_bytes=m["total_logical_bytes"] + 1),
    lambda v, m: m["inventory"].pop(0),
    lambda v, m: m["inventory"].append(deepcopy(m["inventory"][0])),
    lambda v, m: m["inventory"][0].update(relative_path="../unowned.npy"),
    lambda v, m: (m["inventory"][0].update(bytes=first.Q104_OUTPUT_ENTITLEMENT_BYTES),
        m.update(total_logical_bytes=first.Q104_OUTPUT_ENTITLEMENT_BYTES + len(m["inventory"]) - 1)),
])
def test_Q104_refuses_missing_all_stage_reference_or_budget_evidence(change):
    task, value, objects = all_stage_reference_TESTONLY()
    measure = objects[value["measurement"]["path"]]; change(value, measure)
    with pytest.raises(CachePolicyError):
        first.checked_all_stage_reference(task, value, read=lambda pin: deepcopy(objects[pin["path"]]))


def test_Q104_all_stage_data_checks_every_published_and_retained_file():
    task, value, objects = all_stage_reference_TESTONLY()
    # TESTONLY structural metadata, not a Root release or actual measurement.
    assert first.checked_all_stage_reference(task, value,
        read=lambda pin: deepcopy(objects[pin["path"]])) == value


@pytest.mark.parametrize("change", [
    lambda c, m: c.update(backend="cuda"),
    lambda c, m: c.update(actual_worker=None),
    lambda c, m: c["actual_worker"].update(machine="second-cpu"),
    lambda c, m: c.update(commit="e" * 40),
    lambda c, m: c.update(newly_computed_frames=38),
    lambda c, m: c["frame_producers"].reverse(),
    lambda c, m: c["records"].reverse(),
    lambda c, m: c["records"][0]["files"].pop("products.npz"),
    lambda c, m: c["records"][0]["files"]["native_geometry.npz"].update(size_bytes=2),
    lambda c, m: c["records"][0]["files"]["native_statistics.json.gz"].update(sha256="c" * 64),
    lambda c, m: c["frame_producers"][0]["result"].update(path="C:/TESTONLY/outside.json"),
    lambda c, m: c["frame_producers"][0]["publication"].update(sha256="c" * 64),
])
def test_Q104_refuses_stale_commit_prefix_reordered_producer_or_final_file_identity(change):
    task, value, objects = all_stage_reference_TESTONLY()
    complete = objects[value["reference_completion"]["path"]]
    change(complete, objects[value["measurement"]["path"]])
    with pytest.raises(CachePolicyError):
        first.checked_all_stage_reference(task, value, read=lambda pin: deepcopy(objects[pin["path"]]))


def test_physical_bound_evidence_cannot_be_used_as_old_lifetime_profile():
    # Even a purported completed observation cannot provide the mandatory exact
    # bounded record, and the ordinary consumer must never fall back to old IO.
    with pytest.raises(CachePolicyError, match="Exact bounded evidence"):
        first.require_publishable(dict(format=first.OBSERVATION, complete=True),
            dict(format=first.BOUNDED_EVIDENCE, actual_writer_IO_proven=True,
                maximum_observed_cache_bytes=7, passed=True))


def bounded_evidence_TESTONLY():
    from darsia_cache.environment import FIRST_WRITER_ENVIRONMENT_KEYS
    state, _ = protocol(); owner = dict(TESTONLY="not a process observation")
    root = "C:/TESTONLY/service"; bound = bound_plan_TESTONLY()
    bound["volume_cap_observation"]["virtual_size_bytes"] = 500 * 1024**3
    env = {key: root + "/" + key.lower() for key in FIRST_WRITER_ENVIRONMENT_KEYS}
    env["NUMBA_CACHE_LOCATOR_CLASSES"] = "UserProvidedCacheLocator"
    row = dict(owner=owner, registration=pin("registration.json"), inventory=pin("exit.json"),
        all_file_backed_modules_predeclared_and_exact=True, undeclared=[], actual_writer_environment=env,
        writer_environment_is_actual_IO_or_lifetime_proof=False)
    cache = dict(logical_bytes=7)
    observation = dict(format=first.OBSERVATION, complete=True, request=state.binding,
        provider=state.provider_binding, stores_source=state.stores_source, scope=state.provider["scope"],
        retirement=pin("retirement.json"), source_runtime_before_after_exact=True,
        actual_public_worker_exit_code=0, operational_reserve_crossed=False,
        sampled_cache_logical_high_water_is_lifetime_peak=False,
        reserve_samples_prove_no_crossing_between_samples=False,
        actual_registered_owner_exit_inventories=[row], reserve_watchdog_period_seconds=1,
        reserve_samples=[dict(utc="TESTONLY", monotonic_seconds=1.0,
            actual_gap_since_previous_start_seconds=None, operational_reserve_bytes=PHYSICAL_RESERVE_BYTES,
            at_or_below_operational_reserve=False, free_volume_observation=dict(free_bytes=PHYSICAL_RESERVE_BYTES+1))],
        sample_count=1, cache_samples=[dict(started_monotonic_seconds=1.0, elapsed_seconds=.1,
            actual_gap_since_previous_start_seconds=None, observation=cache)],
        final_sample=cache, sampled_cache_logical_high_water_bytes=7)
    reserved = dict(format="darsia-cache-budget-reservation-v2", token="1" * 32,
        pool="TESTONLY", owner=owner, maximum_bytes=100)
    value = dict(format=first.BOUNDED_EVIDENCE,
        **{key: observation[key] for key in ("request", "provider", "stores_source", "scope", "retirement")},
        namespace_admission=pin("namespace.json"), capacity_bound_plan=pin("bound.json"), capacity_bound=bound,
        service_namespace=root, administrative_service_entitlement=dict(maximum_bytes=100, measured_profile=False),
        actual_service_reservation=dict(token=reserved["token"], pool=reserved["pool"], sha256=first._digest(reserved)),
        actual_service_reservation_document=reserved, actual_volume_observation=bound["volume_cap_observation"],
        registered_owner_environments=[row], observation_sha256=first._digest(observation),
        source_runtime_before_after_exact=True, actual_writer_IO_proven=False,
        writer_lifetime_cache_bytes_proof=None, sampled_highwater_is_lifetime_peak=False,
        output_evidence=None, original_comparison=None, ordinary_use_authorization=None,
        profile_publication_state="UNQUALIFIED_BOUND_OBSERVATION", no_ordinary_profile_created=True)
    return observation, value


def test_typed_bound_record_preserves_gaps_without_minting_an_ordinary_profile():
    observation, value = bounded_evidence_TESTONLY()
    assert first.checked_bounded_evidence(observation, value) == value
    with pytest.raises(CachePolicyError, match="Bounded observation alone"):
        first.require_publishable(observation, value)


def test_bound_producer_uses_readonly_context_cap_and_loaded_receipt_not_projection():
    observation, value = bounded_evidence_TESTONLY(); state, _ = protocol()
    state.bound_mode = True; state.required_capacity_bound = value["capacity_bound"]
    state.request["capacity_bound_plan"] = value["capacity_bound_plan"]
    class ReadOnlyLedger_TESTONLY:
        volume_cap_observation = value["actual_volume_observation"]
        @contextmanager
        def locked(self, *, create_binding):
            assert create_binding is False
            yield self
        def project(self, *_):
            pytest.fail("Original observer context deliberately cannot measure producer cache")
        def require_reservation(self, reservation, owner, maximum):
            assert reservation == value["actual_service_reservation"]
            assert owner == value["actual_service_reservation_document"]["owner"]
            assert maximum == value["administrative_service_entitlement"]["maximum_bytes"]
            return value["actual_service_reservation_document"], b"TESTONLY"
    context = SimpleNamespace(ledger=ReadOnlyLedger_TESTONLY(), root=value["service_namespace"],
        reservation=value["actual_service_reservation"],
        manifest=dict(owner=value["actual_service_reservation_document"]["owner"]), maximum=100,
        receipt=pin("service-receipt.json"))  # Real context holds a descriptor.
    loaded = dict(first_qualification=dict(administrative_entitlement=value["administrative_service_entitlement"]))
    assert first.bounded_evidence(state, context, observation, service_receipt=loaded) == value


def test_forced_zero_gap_sample_is_preserved_without_claiming_continuous_coverage():
    observation, value = bounded_evidence_TESTONLY()
    terminal = deepcopy(observation["reserve_samples"][0]); terminal["actual_gap_since_previous_start_seconds"] = 0
    observation["reserve_samples"].append(terminal)
    value["observation_sha256"] = first._digest(observation)
    assert first.checked_bounded_evidence(observation, value) == value


@pytest.mark.parametrize("change", [
    lambda o, v: v.update(actual_writer_IO_proven=True),
    lambda o, v: v.update(writer_lifetime_cache_bytes_proof=pin("invented-lifetime.json")),
    lambda o, v: v.update(ordinary_use_authorization=pin("invented-authorization.json")),
    lambda o, v: v.update(actual_volume_observation={}),
    lambda o, v: v.update(service_namespace="C:/TESTONLY/../service"),
    lambda o, v: v["administrative_service_entitlement"].update(maximum_bytes=500*1024**3),
    lambda o, v: v["registered_owner_environments"][0]["actual_writer_environment"].update(TEMP="C:/outside"),
    lambda o, v: v["registered_owner_environments"][0].update(undeclared=["unknown.py"]),
    lambda o, v: o.update(operational_reserve_crossed=True),
    lambda o, v: o["reserve_samples"][0].update(actual_gap_since_previous_start_seconds=0),
    lambda o, v: o.update(sampled_cache_logical_high_water_bytes=8),
])
def test_typed_bound_reader_refuses_conflicts_missing_writers_and_lifetime_claims(change):
    observation, value = bounded_evidence_TESTONLY(); change(observation, value)
    value["observation_sha256"] = first._digest(observation)
    with pytest.raises(CachePolicyError):
        first.checked_bounded_evidence(observation, value)


def bound_output_state_TESTONLY():
    state, _ = protocol(); state.bound_mode = True
    scope = dict(format="darsia-first-q104-output-scope-v1", role="results",
        path="C:/TESTONLY/current-results", identity=dict(TESTONLY="not admitted"),
        stat_identity=dict(st_dev=1, st_ino=2, creation_ns=3), volume_id="TESTONLY",
        entry_limit=1000, sample_period_seconds=1)
    state.output_entitlement = dict(format=first.BOUND_OUTPUT_ADMINISTRATION,
        administrative_output_entitlement_bytes=first.FIRST_OUTPUT_BASE_ENTITLEMENT_BYTES,
        watched_output_scope=scope)
    state.request["output_entitlement_plan"] = pin("output-plan.json")
    return state, scope


@pytest.mark.parametrize("change", [
    lambda v: v.update(path="C:/TESTONLY/../unowned"),
    lambda v: v.update(entry_limit=0),
    lambda v: v.update(entry_limit=100001),
    lambda v: v.update(sample_period_seconds=2),
    lambda v: v.update(role="cache"),
    lambda v: v["stat_identity"].update(st_ino=0),
])
def test_bound_output_scope_requires_one_exact_current_root_and_original_watch_target(change):
    _, scope = bound_output_state_TESTONLY(); change(scope)
    with pytest.raises(CachePolicyError): first.checked_bound_output_scope(scope)


def test_output_watch_period_and_latched_crossing_never_become_recovered_PASS(monkeypatch):
    state, _ = bound_output_state_TESTONLY(); now = [0.0]; calls = []; stops = []
    watcher = first.BoundOutputObservations(state, clock=lambda: now[0])
    def snapshot(scope, *, entitlement_bytes, include_file_stats=False):
        calls.append(now[0])
        assert entitlement_bytes == first.FIRST_OUTPUT_BASE_ENTITLEMENT_BYTES
        size = entitlement_bytes + 1 if now[0] == 1 else 5
        return dict(utc="TESTONLY", logical_bytes=size, entries=1,
            volume=dict(free_bytes=PHYSICAL_RESERVE_BYTES+entitlement_bytes+1),
            observed_scan_anomalies=[], file_stats=[] if include_file_stats else None)
    monkeypatch.setattr(first, "bound_output_snapshot", snapshot)
    monkeypatch.setattr(first, "latch_bound_output_stop", lambda f, l, reason, row:
        (stops.append(reason) or dict(requested=False, state="TESTONLY_NOT_AN_ACTUAL_COMMAND")))
    owned = SimpleNamespace(lease=object())
    for instant in [0, .2, 1, 1.2, 2]:
        now[0] = instant; watcher.observe_output(owned)
    watcher.observe_output(owned, force=True)
    doc = watcher.output_document(dict(accounting=dict(active_processes=0), wrapper_exit_code=0),
        pin("retirement.json"), dict(original_retirement_verifier_accepted=True))
    assert calls == [0, 1, 2, 2]
    assert stops == ["OUTPUT_ENTITLEMENT_EXCEEDED"]
    assert doc["sampled_highwater_bytes"] == first.FIRST_OUTPUT_BASE_ENTITLEMENT_BYTES + 1
    assert doc["closed_output_logical_bytes"] == 5  # Actual closed size is not a recovery of PASS.
    assert doc["complete"] is False and doc["passed"] is False
    assert doc["sampled_highwater_is_lifetime_peak"] is False
    assert doc["state"] == "STOP_OR_INCOMPLETE_DIAGNOSTIC"


def test_output_sampling_error_preserves_primary_exception_if_stop_record_also_fails(monkeypatch):
    state, _ = bound_output_state_TESTONLY(); watcher = first.BoundOutputObservations(state)
    def snapshot(*args, **kwargs): raise OSError("TESTONLY original sampling failure")
    def stop(*args): raise RuntimeError("TESTONLY stop-record failure")
    monkeypatch.setattr(first, "bound_output_snapshot", snapshot)
    monkeypatch.setattr(first, "latch_bound_output_stop", stop)
    with pytest.raises(OSError, match="original sampling failure") as caught:
        watcher.observe_output(SimpleNamespace(lease=object()))
    assert "stop-record failure" in caught.value.__notes__[0]
    doc = watcher.output_document(None, None, None)
    assert doc["first_observation_error"]["error_type"] == "OSError"
    assert doc["complete"] is False and doc["passed"] is False
    assert doc["closed_output_logical_bytes"] is None


def test_closed_output_needs_original_retirement_and_never_mints_a_profile(monkeypatch):
    state, _ = bound_output_state_TESTONLY(); watcher = first.BoundOutputObservations(state)
    monkeypatch.setattr(first, "bound_output_snapshot", lambda *args, **kwargs:
        dict(utc="TESTONLY", logical_bytes=7, entries=1,
            volume=dict(free_bytes=PHYSICAL_RESERVE_BYTES+first.FIRST_OUTPUT_BASE_ENTITLEMENT_BYTES+1),
            observed_scan_anomalies=[], file_stats=[]))
    watcher.observe_output(SimpleNamespace(lease=object()), force=True)
    doc = watcher.output_document(dict(accounting=dict(active_processes=0), wrapper_exit_code=0),
        pin("retirement.json"), None)
    assert doc["closed_output_stat_observation"] is False
    assert doc["closed_output_logical_bytes"] is None
    assert doc["passed"] is False and doc["no_result_space_profile_created"] is True


def test_output_admission_refuses_retained_data_and_scope_overlap_before_native_start(monkeypatch):
    state, scope = bound_output_state_TESTONLY(); watcher = first.BoundOutputObservations(state)
    lease = SimpleNamespace(policy=SimpleNamespace(root="Q:/TESTONLY/cache"))
    monkeypatch.setattr(first, "bound_output_snapshot", lambda scope, **kwargs:
        dict(logical_bytes=1, observed_scan_anomalies=[],
            volume=dict(free_bytes=PHYSICAL_RESERVE_BYTES+first.FIRST_OUTPUT_BASE_ENTITLEMENT_BYTES+1)))
    with pytest.raises(CachePolicyError, match="Fresh complete output scope"):
        watcher.admit_output(lease)
    lease.policy.root = scope["path"]
    with pytest.raises(CachePolicyError, match="disjoint"):
        watcher.admit_output(lease)


def comparison_contract_TESTONLY(task, reference, objects):
    sources = dict(numeric_checker=dict(path="C:/TESTONLY/original9949.py",
        sha256="9949a184fef7e3b55c75336544fa04f52a2062f1b37cfdad1018f2243280b5ce", size_bytes=17239),
        record_extractors=dict(path="C:/TESTONLY/original95dc.py",
        sha256="95dc41bf655c5d22a91dd5bf7cf55430f0f5a511d8d7cdabe8f8e68378b6aa13", size_bytes=21587),
        adapter=pin("reviewed-current-pair.py"), read_locator=pin("locator.py"), typed_ipp_validator=pin("ipp.py"))
    binding = pin("predeclared-comparison-contract.json")
    review = pin("adapter-source-review.json")
    objects[review["path"]] = {"TESTONLY": "not actual Root review"}
    objects[binding["path"]] = dict(format="darsia-first-volume-bound-current-comparison-contract-v1",
        scope={key:task[key] for key in ("commit", "pipeline", "backend", "precision", "run", "ordered_case_ids")},
        reference_commit=reference["reference_commit"], candidate_commit=task["commit"],
        reference_completed=reference["reference_completion"], sources=sources,
        source_pair_compatibility=reference["reference_source_compatibility"], adapter_source_review=review,
        public_command=["C:/TESTONLY/python.exe", "-B", sources["adapter"]["path"]],
        report_format="TESTONLY-current-whole39-comparison", comparison_kind="same_precision")
    return binding


def whole_output_administration_TESTONLY(*, extra_retained_bytes=0):
    task, reference, objects = all_stage_reference_TESTONLY()
    measurement = objects[reference["measurement"]["path"]]
    if extra_retained_bytes:
        measurement["inventory"].append(dict(relative_path="preparation/retained.tmp",
            bytes=extra_retained_bytes, sha256="f" * 64))
        measurement["total_logical_bytes"] += extra_retained_bytes
    reference_binding = pin("whole-all-stage-reference.json")
    answers_binding = pin("Q109-Q117-Q119-answers.json")
    objects[reference_binding["path"]] = reference
    objects[answers_binding["path"]] = [dict(id=name, decided_by="coordinator") for name in (
        "Q109-answer-option-A-whole-output-entitlement-65728995330B-all-stages-count",
        "Q117-answer-yes-this-is-the-Q102-Q104-Q106-evidence-contract",
        "Q119-answer-Q109-entitlement-already-covers-first-qualification-or-twice-measured-allstage")]
    _, output_scope = bound_output_state_TESTONLY()
    value = dict(format=first.BOUND_OUTPUT_ADMINISTRATION,
        authority_ids=["Q104", "Q109", "Q117", "Q119"], answers_source=answers_binding,
        task=pin("task.json"), scope={key:task[key] for key in (
            "commit", "pipeline", "backend", "precision", "run", "ordered_case_ids")},
        all_stage_reference=reference_binding,
        comparison_contract=comparison_contract_TESTONLY(task, reference, objects),
        administrative_output_entitlement_bytes=max(first.FIRST_OUTPUT_BASE_ENTITLEMENT_BYTES,
            2 * measurement["total_logical_bytes"]),
        measured_profile=False, watched_output_scope=output_scope,
        producer_state="PREDECLARED_NOT_QUALIFIED")
    return task, value, objects


@pytest.mark.parametrize("extra", [0, 65728995330])
def test_Q119_predeclares_whole_allstage_maximum_without_inventing_per_frame_bytes(extra, monkeypatch):
    monkeypatch.setattr("darsia_cache.owned_process._exact_source", lambda pin: None)
    task, value, objects = whole_output_administration_TESTONLY(extra_retained_bytes=extra)
    result = first.checked_output_entitlement(task, value, task_binding=pin("task.json"),
        read=lambda binding: deepcopy(objects[binding["path"]]))
    measured = objects[objects[value["all_stage_reference"]["path"]]["measurement"]["path"]]
    assert result == value
    assert result["administrative_output_entitlement_bytes"] == max(
        first.FIRST_OUTPUT_BASE_ENTITLEMENT_BYTES, 2 * measured["total_logical_bytes"])
    assert result["measured_profile"] is False


@pytest.mark.parametrize("change", [
    lambda v, o: v.update(administrative_output_entitlement_bytes=27374032352),
    lambda v, o: v.update(administrative_output_entitlement_bytes=v["administrative_output_entitlement_bytes"]+1),
    lambda v, o: v.update(measured_profile=True),
    lambda v, o: v.update(producer_state="QUALIFIED"),
    lambda v, o: v["scope"].update(backend="cuda"),
    lambda v, o: o[v["answers_source"]["path"]].pop(),
    lambda v, o: o[v["answers_source"]["path"]][-1].update(decided_by="UNKNOWN"),
])
def test_Q119_refuses_old_footprint_unbound_authority_or_unreviewed_entitlement(change, monkeypatch):
    monkeypatch.setattr("darsia_cache.owned_process._exact_source", lambda pin: None)
    task, value, objects = whole_output_administration_TESTONLY(); change(value, objects)
    with pytest.raises(CachePolicyError):
        first.checked_output_entitlement(task, value, task_binding=pin("task.json"),
            read=lambda binding: deepcopy(objects[binding["path"]]))


def actual_comparison_TESTONLY():
    state, _ = protocol()
    task, reference, objects = all_stage_reference_TESTONLY()
    _, candidate, new = all_stage_reference_TESTONLY(label="candidate", reference_commit=task["commit"])
    objects.update(new); state.task = task
    refpin = pin("reference-allstage.json"); objects[refpin["path"]] = reference
    contractpin = comparison_contract_TESTONLY(task, reference, objects)
    state.output_entitlement = dict(all_stage_reference=refpin, comparison_contract=contractpin)
    contract = objects[contractpin["path"]]; rows = []
    for index, case in enumerate(task["ordered_case_ids"]):
        receipt = pin(case + "-original-comparison.json")
        objects[receipt["path"]] = dict(case_id=case, limits_passed=True, first_failure=None,
            reference_result=objects[reference["reference_completion"]["path"]]["frame_producers"][index]["result"],
            candidate_result=objects[candidate["reference_completion"]["path"]]["frame_producers"][index]["result"])
        rows.append(dict(case_id=case, passed=True, first_failure=None, receipt=receipt))
    reportpin, proofpin = pin("original-completed.json"), pin("original-os-retirement-proof.json")
    objects[reportpin["path"]] = dict(format=contract["report_format"], complete=True,
        comparison_kind="same_precision", numeric_checker=contract["sources"]["numeric_checker"],
        reference_completed=reference["reference_completion"], candidate_completed=candidate["reference_completion"],
        case_count=39, new_comparisons_count=39, ordered_case_ids=task["ordered_case_ids"],
        accepted_prior_comparisons=[], frame_add8_limits_passed=True, new_scientific_arrays_computed=False,
        comparisons=rows)
    proof = dict(format="darsia-first-volume-bound-original-comparison-v1", contract=contractpin,
        report=reportpin, actual_command=contract["public_command"], actual_exit_code=0,
        retirement=pin("original-comparator-retirement.json"),
        source_runtime_before_after=pin("original-comparison-source-post.json"), Root_acceptance=pin("comparison-Root.json"))
    objects[proofpin["path"]] = proof
    objects[proof["retirement"]["path"]] = dict(format="darsia-first-original-comparator-retirement-v1",
        actual_exit_code=0, actual_command=contract["public_command"], all_owned_retired=True,
        known_epochs=[dict(pid=7, process_create_time=123.0, exact_incarnation_absent=True)])
    objects[proof["source_runtime_before_after"]["path"]] = dict(module_closure=state.provider["module_closure"],
        source_runtime_before_after_exact=True, comparison_sources=contract["sources"])
    objects[proof["Root_acceptance"]["path"]] = {"TESTONLY": "not actual acceptance"}
    return state, candidate, proofpin, objects


def test_original_whole39_comparison_contract_preserves_distinct_actual_source_commits(monkeypatch):
    monkeypatch.setattr("darsia_cache.owned_process._exact_source", lambda pin: None)
    state, candidate, proof, objects = actual_comparison_TESTONLY()
    assert first._checked_bound_comparison(state, proof, candidate=candidate,
        read=lambda bound: deepcopy(objects[bound["path"]])) == objects[proof["path"]]
    assert objects[state.output_entitlement["comparison_contract"]["path"]]["reference_commit"] != state.task["commit"]


@pytest.mark.parametrize("change", [
    lambda p, c, r, o: p.update(actual_exit_code=False),
    lambda p, c, r, o: p.update(actual_exit_code=1),
    lambda p, c, r, o: p.update(actual_command=["unreviewed-adapter"]),
    lambda p, c, r, o: c.update(candidate_commit="e"*40),
    lambda p, c, r, o: c["sources"]["numeric_checker"].update(sha256="f"*64),
    lambda p, c, r, o: r.update(new_comparisons_count=8),
    lambda p, c, r, o: r.update(accepted_prior_comparisons=["ac52_f0"]),
    lambda p, c, r, o: r.update(new_scientific_arrays_computed=True),
    lambda p, c, r, o: r["comparisons"].reverse(),
    lambda p, c, r, o: r["comparisons"][0].update(passed=False),
    lambda p, c, r, o: o[r["comparisons"][0]["receipt"]["path"]].update(candidate_result=pin("unrelated.json")),
    lambda p, c, r, o: o[p["retirement"]["path"]]["known_epochs"][0].update(exact_incarnation_absent=False),
    lambda p, c, r, o: o[p["source_runtime_before_after"]["path"]].update(source_runtime_before_after_exact=False),
])
def test_comparison_refuses_reuse_wrong_source_missing_os_or_actual_retirement(change, monkeypatch):
    monkeypatch.setattr("darsia_cache.owned_process._exact_source", lambda pin: None)
    state, candidate, pin_value, objects = actual_comparison_TESTONLY()
    proof = objects[pin_value["path"]]; contract = objects[proof["contract"]["path"]]
    report = objects[proof["report"]["path"]]; change(proof, contract, report, objects)
    with pytest.raises(CachePolicyError):
        first._checked_bound_comparison(state, pin_value, candidate=candidate,
            read=lambda bound: deepcopy(objects[bound["path"]]))


def test_typed_bound_import_requires_actual_worker_role_namespace_and_original_cap():
    bound = bound_plan_TESTONLY(); binding = dict(TESTONLY="not actual Q87")
    bound["enforcement_state"] = "PREDECLARED_Q106_VOLUME_BOUND"
    bound["volume_cap_observation"].update(binding=binding, virtual_size_bytes=500*1024**3)
    profile = dict(format=first.BOUND_IMPORT_PROFILE, purpose="jobs", entry_kind="worker",
        role_identifier="actual_owned_jobs", namespace_admission=pin("namespace.json"),
        writer_contract=first.BOUND_WRITER_CONTRACT, sampled_highwater_is_lifetime_peak=False,
        actual_writer_IO_proven=False, capacity_bound=bound,
        operational_service_entitlement_bytes=100, maximum_cache_bytes=100)
    policy = SimpleNamespace(quota_bytes=500*1024**3,
        volume_cap=SimpleNamespace(maximum_size_bytes=500*1024**3, document=lambda: binding))
    arguments = dict(purpose="jobs", entry_kind="worker", role_identifier="actual_owned_jobs",
        policy=policy, namespace=pin("namespace.json"))
    assert first.checked_bound_import_context(profile, **arguments) == 100
    for key, value in [("purpose", "tests"), ("entry_kind", "runtime"),
            ("role_identifier", "other"), ("namespace", pin("other-namespace.json"))]:
        with pytest.raises(CachePolicyError):
            first.checked_bound_import_context(profile, **dict(arguments, **{key:value}))
    changed = deepcopy(profile); changed["actual_writer_IO_proven"] = True
    with pytest.raises(CachePolicyError): first.checked_bound_import_context(changed, **arguments)


def test_bound_publication_writes_only_fixed_profiles_then_final_completion_without_cycle(monkeypatch):
    state, _ = protocol(); state.bound_mode = True
    documents = ({"format":first.BOUND_IMPORT_PROFILE}, {"format":first.BOUND_JOB_PROFILE},
        dict(observation=pin("observation.json"), writer_evidence=pin("writer.json"),
            measurement=pin("measurement.json"), native_completion=pin("native.json"), retirement=pin("retired.json")))
    written = []
    monkeypatch.setattr(first, "_bound_profile_documents", lambda *args, **kwargs: deepcopy(documents))
    def slot(provider_value, role, value):
        written.append((role, deepcopy(value))); return pin(role + ".json")
    monkeypatch.setattr(first, "_slot_write", slot)
    assert first._publish_bound_profiles(state, {"TESTONLY":"already checked by pure constructor"}, read=lambda p:None) == pin("completion.json")
    assert [role for role, body in written] == ["import_profile", "job_profile", "completion"]
    assert all("profiles" not in body for role,body in written[:-1])
    assert written[-1][1]["profiles"] == dict(import_profile=pin("import_profile.json"), job_profile=pin("job_profile.json"))


def bound_documents_TESTONLY(monkeypatch):
    """Actual existing receipt shapes; no process, Root authority or array data."""
    state, objects = protocol(); task, admin, refs = whole_output_administration_TESTONLY()
    objects.update(refs); state.bound_mode = True; state.task = task; state.output_entitlement = admin
    state.request["output_entitlement_plan"] = pin("current-output-admin.json")
    _, candidate, generated = all_stage_reference_TESTONLY(label="candidate", reference_commit=task["commit"])
    objects.update(generated); candidatepin = pin("candidate-allstage.json"); objects[candidatepin["path"]] = candidate
    observation, evidence = bounded_evidence_TESTONLY()
    evidence["capacity_bound"]["enforcement_state"] = "PREDECLARED_Q106_VOLUME_BOUND"
    state.required_capacity_bound = evidence["capacity_bound"]
    slotpins = {role:pin(role+".json") for role in ("observation", "writer_evidence", "measurement", "native_completion")}
    terminal = dict(accounting=dict(active_processes=0, total_processes=3), wrapper_exit_code=0)
    eligibility = dict(original_retirement_verifier_accepted=True, seal=pin("seal.json"))
    observation.update(actual_claim=task, output_measurement=slotpins["measurement"], kernel_total_processes=3,
        original_kernel_terminal=terminal, original_publication_and_retirement_eligibility=eligibility,
        native_launch=pin("native-launch.json"), service_receipt=pin("actual-service.json"))
    evidence["observation_sha256"] = first._digest(observation)
    objects[eligibility["seal"]["path"]] = dict(value=dict(outcome="success"))
    actual_owner = evidence["registered_owner_environments"][0]["owner"]
    objects[observation["retirement"]["path"]] = dict(format="darsia-native-owned-worker-retirement-v1",
        native_launch=observation["native_launch"], service_receipt=observation["service_receipt"],
        whole_family_kernel_observation=True, terminal=terminal, primary=actual_owner,
        actual_primary_binding=pin("actual-primary-native-handle.json"))
    objects["C:/TESTONLY/actual-primary-native-handle.json"] = dict(actual_service_receipt=observation["service_receipt"],
        original_native_launch=observation["native_launch"], actual_primary=actual_owner,
        held_process_handle_observed_while_live=True, same_original_job_membership_observed=True)
    service = dict(format="darsia-cache-import-service-receipt-v1", policy=state.stores_source,
        profile=state.provider_binding, namespace=state.provider["namespace_admission"],
        entry_kind="worker", role_identifier="actual_jobs_role",
        manifest=dict(purpose="jobs", root=evidence["service_namespace"], reservation=evidence["actual_service_reservation"],
            owner=actual_owner, maximum_bytes=100, native_launch_context=observation["native_launch"],
            root_identity=dict(TESTONLY="not an actual file ID")),
        first_qualification=dict(request=state.binding, provider=state.provider_binding,
            administrative_entitlement=evidence["administrative_service_entitlement"]))
    from darsia_cache.environment import FIRST_WRITER_ENVIRONMENT_KEYS, FIRST_WRITER_SUBDIRECTORIES
    service["manifest"]["fresh_scope_provenance"] = dict(format="darsia-fresh-exclusive-empty-service-scope-v1",
        owner=actual_owner, token=evidence["actual_service_reservation"]["token"], purpose="jobs",
        root_identity=service["manifest"]["root_identity"], creation_disposition="NtCreateFile.FILE_CREATE",
        exclusive_root_handle=True, initial_names=[], initial_empty_observed_utc="2026-01-01T00:00:00+00:00",
        observation_precedes_writer_subdirectories_and_imports=True,
        writer_subdirectories=list(FIRST_WRITER_SUBDIRECTORIES), writer_environment_keys=list(FIRST_WRITER_ENVIRONMENT_KEYS))
    objects[observation["service_receipt"]["path"]] = service
    row = evidence["registered_owner_environments"][0]
    objects[row["registration"]["path"]] = dict(format="darsia-cache-service-lifecycle-v1", kind="owner",
        binding=dict(reservation=service["manifest"]["reservation"], policy=state.stores_source,
            profile=state.provider_binding, namespace=state.provider["namespace_admission"], manifest=service["manifest"]),
        actor=actual_owner, value=dict(owner=actual_owner, native_job_membership=observation["native_launch"]))
    objects[row["inventory"]["path"]] = dict(owner=actual_owner, first_qualification_modules=dict(
        request=state.binding, provider=state.provider_binding, all_file_backed_modules_predeclared_and_exact=True,
        undeclared=[], predeclared_module_closure_sha256=first._digest(state.provider["module_closure"]),
        actual_writer_environment=row["actual_writer_environment"], writer_environment_is_actual_IO_or_lifetime_proof=False))
    closed = objects[candidate["measurement"]["path"]]
    admin["watched_output_scope"].update(path=closed["root"], identity=closed["root_identity"])
    actual_output = dict(format="darsia-first-q104-closed-output-diagnostic-v1", request=state.binding,
        provider=state.provider_binding, stores_source=state.stores_source, scope=state.provider["scope"],
        output_plan=state.request["output_entitlement_plan"], output_scope=admin["watched_output_scope"],
        complete=True, first_stop=None, first_observation_error=None, cache_operational_reserve_crossed=False,
        whole_output_entitlement_observed_exceeded=False, actual_retirement=observation["retirement"],
        actual_terminal=terminal, entitlement_bytes=admin["administrative_output_entitlement_bytes"],
        closed_output_logical_bytes=closed["total_logical_bytes"], sampled_highwater_is_lifetime_peak=False,
        output_hard_quota=False, target_sample_period_seconds=1,
        admission=dict(logical_bytes=0, observed_scan_anomalies=[],
            volume=dict(free_bytes=PHYSICAL_RESERVE_BYTES+admin["administrative_output_entitlement_bytes"])),
        output_samples=[dict(started_monotonic_seconds=1., elapsed_seconds=.1,
            actual_gap_since_previous_start_seconds=None, sampled=True, complete_lifetime_peak=False,
            free_bytes=PHYSICAL_RESERVE_BYTES+1, logical_bytes=closed["total_logical_bytes"], forced_terminal=True)],
        sampled_highwater_bytes=closed["total_logical_bytes"],
        final_snapshot=dict(logical_bytes=closed["total_logical_bytes"],
            file_stats=[dict(relative_path=row["relative_path"], size_bytes=row["bytes"]) for row in closed["inventory"]]))
    native = dict(format="darsia-native-worker-first-qualification-completion-v1",
        **{key:observation[key] for key in ("request", "provider", "stores_source", "scope", "retirement")},
        observation=slotpins["observation"], startup=state.provider["startup"],
        module_closure=state.provider["module_closure"], complete=True, passed=True,
        observer_native_handles_closed=True, cache_profile_qualified=False, maximum_family_processes=3, eligibility=eligibility)
    slots = dict(observation=observation, writer_evidence=evidence, measurement=actual_output, native_completion=native)
    bindings = dict(candidate_all_stage=candidatepin, original_comparison=pin("independent-original-comparison.json"),
        once_marker=pin("cache-first-"+state._once_key()+".json"))
    objects[bindings["once_marker"]["path"]] = dict(format="darsia-cache-first-qualification-consumed-v1",
        **{key:observation[key] for key in ("request", "provider", "stores_source", "scope")}, no_retry=True)
    monkeypatch.setattr(first, "_slot_read", lambda provider_value, role:(deepcopy(slots[role]), slotpins[role]))
    monkeypatch.setattr(first, "_checked_bound_comparison", lambda *args, **kwargs:{"TESTONLY":"independently tested contract above"})
    monkeypatch.setattr(first, "_checked_bound_writer_compatibility", lambda *args:None)
    return state, bindings, objects, slots


def test_completed_bound_projection_uses_real_receipt_shape_and_never_lifetime_fields(monkeypatch):
    state, bindings, objects, slots = bound_documents_TESTONLY(monkeypatch)
    imported, jobs, actual = first._bound_profile_documents(state, bindings,
        read=lambda bound:deepcopy(objects[bound["path"]]))
    assert imported["format"] == first.BOUND_IMPORT_PROFILE and imported["role_identifier"] == "actual_jobs_role"
    assert imported["maximum_cache_bytes"] == 100  # Actual operational reservation, not500GiB+reserve.
    assert imported["native_supervision"]["format"] == "darsia-native-worker-supervision-profile-v1"
    assert jobs["additional_claim_cache_bytes"] == 0 and "maximum_observed_cache_bytes" not in jobs
    assert jobs["actual_writer_IO_proven"] is False and jobs["sampled_highwater_is_lifetime_peak"] is False
    assert jobs["result_profile"]["measurement"]["inventory"]
    assert actual["measurement"] == slots["observation"]["output_measurement"]


@pytest.mark.parametrize("change", [
    lambda b,o,s:s["native_completion"].update(passed=False),
    lambda b,o,s:s["native_completion"].update(observer_native_handles_closed=False),
    lambda b,o,s:o[s["observation"]["retirement"]["path"]]["terminal"]["accounting"].update(active_processes=1),
    lambda b,o,s:o[s["observation"]["service_receipt"]["path"]]["manifest"].update(purpose="tests"),
    lambda b,o,s:o[b["once_marker"]["path"]].update(no_retry=False),
    lambda b,o,s:s["measurement"].update(first_stop={"reason":"OUTPUT_ENTITLEMENT_EXCEEDED"}),
    lambda b,o,s:s["measurement"].update(output_samples=[]),
    lambda b,o,s:s["measurement"]["final_snapshot"]["file_stats"].pop(),
])
def test_native_once_namespace_or_output_stop_never_projects_ordinary_profiles(change, monkeypatch):
    state, bindings, objects, slots = bound_documents_TESTONLY(monkeypatch); change(bindings, objects, slots)
    with pytest.raises(CachePolicyError):
        first._bound_profile_documents(state, bindings, read=lambda bound:deepcopy(objects[bound["path"]]))


def test_Q106_reserve_samples_are_one_second_separate_from_full_stat_and_latch_STOP(monkeypatch):
    state,_=protocol();state.bound_mode=True;state.required_capacity_bound=bound_plan_TESTONLY()
    now=[0.0];collector=first.FiniteObservations(state,clock=lambda:now[0])
    lease=lease_for(state);lease.policy.root="C:/TESTONLY/cache";scans=[];free_samples=[];stops=[]
    lease.measure=lambda root:(scans.append(now[0]) or dict(logical_bytes=7))
    free_values=iter([PHYSICAL_RESERVE_BYTES+1,PHYSICAL_RESERVE_BYTES,PHYSICAL_RESERVE_BYTES+2])
    def volume(root):
        free_samples.append(now[0]);return dict(free_bytes=next(free_values),volume_id="TESTONLY")
    monkeypatch.setattr("darsia_cache.capacity.volume_info",volume)
    monkeypatch.setattr(first,"request_existing_frame_boundary_stop",
        lambda f,l:(stops.append((f,l)) or dict(requested=False,state="TESTONLY_UNSUPPORTED")))
    owned=SimpleNamespace(lease=lease)
    for instant in [0,.2,.8,1,1.2,2]:
        now[0]=instant;collector.observe(owned,dict(active_processes=1))
    assert scans==[0] and free_samples==[0,1,2]
    assert collector.native_poll_count==6 and len(stops)==1
    assert collector.reserve_crossed is True
    assert collector.reserve_samples[1]["at_or_below_operational_reserve"] is True
    assert collector.reserve_samples[2]["at_or_below_operational_reserve"] is False
    assert collector.reserve_samples[1]["actual_gap_since_previous_start_seconds"]==1
    assert collector.stop_response["state"]=="TESTONLY_UNSUPPORTED"


def test_Q106_existing_stop_uses_actual_master_and_preserves_current_command_CAS(monkeypatch):
    from darsia.fluidflower import job_control
    state,_=protocol();lease=lease_for(state);lease.owner["machine"]="TESTONLY-Main"
    state.request["public_command"] += ["--job","C:/TESTONLY/TESTONLY-job"]
    monkeypatch.setattr(job_control,"read_json",lambda path:dict(commit=state.task["commit"]))
    previous=pin("previous.json")
    monkeypatch.setattr(job_control,"read_control",lambda root:dict(action="run",sequence=1,command=previous))
    calls=[]
    def publish(root,action,**kwargs):
        calls.append((root,action,kwargs));return pin("stop.json")
    monkeypatch.setattr(job_control,"publish_control",publish)
    actual=first.request_existing_frame_boundary_stop(state,lease)
    assert actual["requested"] is True
    assert calls==[("C:/TESTONLY/TESTONLY-job","stop",dict(machine="TESTONLY-Main",expected_previous=previous))]


def test_Q106_foreign_master_or_failed_stop_records_STOP_without_retry_or_fallback(monkeypatch):
    from darsia.fluidflower import job_control
    state,_=protocol();lease=lease_for(state);lease.owner["machine"]="TESTONLY-Second"
    state.request["public_command"] += ["--job","C:/TESTONLY/TESTONLY-job"]
    monkeypatch.setattr(job_control,"read_json",lambda path:dict(commit=state.task["commit"]))
    monkeypatch.setattr(job_control,"read_control",lambda root:dict(action="run",sequence=0))
    calls=[]
    def refusal(*args,**kwargs):
        calls.append(kwargs);raise ValueError("Only the job's recorded master machine writes commands")
    monkeypatch.setattr(job_control,"publish_control",refusal)
    actual=first.request_existing_frame_boundary_stop(state,lease)
    assert actual["requested"] is False and actual["error_type"]=="ValueError"
    assert actual["state"]=="STOP_COMMAND_NOT_PUBLISHED"
    assert calls==[dict(machine="TESTONLY-Second",expected_previous=None)]


def test_empty_python_module_is_exact_source_not_empty_metadata():
    from darsia_cache.policy import _descriptor
    from darsia_cache.worker_adapter import _source_pins
    empty = dict(path="C:/TESTONLY/__init__.py", size_bytes=0,
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    value = provider()
    value["producer_source_closure"].append(empty)
    value["module_closure"].append(empty)
    assert first.checked_provider(value) == value
    assert _source_pins([empty]) == {'c:\\testonly\\__init__.py'}
    with pytest.raises(CachePolicyError):
        _descriptor(empty)
    for role in ("namespace_admission", "python_executable", "startup"):
        invalid = deepcopy(value)
        invalid[role] = empty
        with pytest.raises(CachePolicyError):
            first.checked_provider(invalid)


@pytest.mark.parametrize("changes", [
    dict(sha256="a"*64), dict(size_bytes=-1), dict(size_bytes=False),
    dict(size_bytes=16*1024**2+1), dict(path="C:/TESTONLY/empty.json"),
    dict(path="C:/TESTONLY/empty.pyd"), dict(path="C:/TESTONLY/python.exe"),
    dict(path=""), dict(size_bytes=0.0), dict(extra="not source identity"),
])
def test_empty_source_exception_keeps_exact_sha_type_suffix_size_and_fields(changes):
    from darsia_cache.worker_adapter import _source_pins
    value = dict(path="C:/TESTONLY/__init__.py", size_bytes=0,
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    value.update(changes)
    with pytest.raises(CachePolicyError):
        first._pins([value])
    with pytest.raises(CachePolicyError):
        _source_pins([value])


def test_module_source_descriptor_keeps_positive_executable_and_16MiB_boundary():
    from darsia_cache.policy import _descriptor, _module_source_descriptor
    for size in (1, 16*1024**2):
        value = dict(path="C:/TESTONLY/python.exe", size_bytes=size, sha256="a"*64)
        assert _module_source_descriptor(value) == _descriptor(value)
    for missing in ("path", "sha256", "size_bytes"):
        value = dict(path="C:/TESTONLY/__init__.py", size_bytes=0,
            sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        del value[missing]
        with pytest.raises(CachePolicyError):
            first._pins([value])



def test_full_14736_source_hash_is_canonical_without_widening_control_records():
    import hashlib
    import json
    from darsia_cache.control import _encoded
    sources=[dict(path=f"C:/TESTONLY/runtime/module_{i:05d}.py",
        sha256="a"*64,size_bytes=i) for i in range(14736)]
    raw=(json.dumps(sources,sort_keys=True,ensure_ascii=False,allow_nan=False,
        separators=(",",":"))+"\n").encode("utf-8")
    assert len(raw)>1024**2
    assert first._digest(sources)==hashlib.sha256(raw).hexdigest()
    with pytest.raises(CachePolicyError,match="one MiB"):_encoded(sources)
    for invalid in [float("nan"),float("inf"),float("-inf")]:
        with pytest.raises(ValueError):first._digest({"invalid":invalid})


def test_first_slot_records_large_closure_with_existing_16mib_write_bound(monkeypatch):
    import hashlib
    events=[]
    class Namespace_TESTONLY:
        def __init__(self,path,identity):events.append((path,identity))
        def write_once(self,name,raw,*,maximum):events.append((name,raw,maximum))
        def close(self):events.append("closed")
    monkeypatch.setattr("darsia_cache.windows.WindowsNamespace",Namespace_TESTONLY)
    value={"module_closure":[dict(path=f"C:/TESTONLY/m{i}.py",sha256="a"*64,size_bytes=1)
        for i in range(14736)]}
    raw=first._slot_encoded(value)
    assert 1024**2<len(raw)<16*1024**2
    actual=first._slot_write(provider(),"native_completion",value)
    assert events[1]==("native_completion.json",raw,16*1024**2)
    assert events[-1]=="closed"
    assert actual["sha256"]==hashlib.sha256(raw).hexdigest() and actual["size_bytes"]==len(raw)


def test_first_slot_exact_limit_and_oversize_refusal_before_namespace(monkeypatch):
    empty=first._slot_encoded({"padding":""})
    value={"padding":"x"*(16*1024**2-len(empty))}
    assert len(first._slot_encoded(value))==16*1024**2
    opened=[]
    monkeypatch.setattr("darsia_cache.windows.WindowsNamespace",lambda *args:opened.append(args))
    value["padding"]+="x"
    with pytest.raises(CachePolicyError,match="full samples not recorded"):
        first._slot_write(provider(),"observation",value)
    assert opened==[]


@pytest.mark.parametrize("value",[None,[],0,{"invalid":float("nan")},{"invalid":float("inf")}])
def test_first_slot_keeps_object_and_finite_contract_before_namespace(monkeypatch,value):
    opened=[]
    monkeypatch.setattr("darsia_cache.windows.WindowsNamespace",lambda *args:opened.append(args))
    with pytest.raises((CachePolicyError,ValueError)):
        first._slot_write(provider(),"observation",value)
    assert opened==[]


@pytest.mark.parametrize("failed_diagnostic",["observation","held","both"])
def test_secondary_diagnostic_failure_preserves_original_primary_exception(monkeypatch,failed_diagnostic):
    state,_=protocol();state.reference_build=True
    lease=lease_for(state);primary=RuntimeError("TESTONLY primary native failure")
    events=[]
    class Owned_TESTONLY:
        resume_attempted=False;info=None
        def __init__(self,*args,**kwargs):pass
        def start(self,*args):raise primary
        def close(self):events.append("closed")
    monkeypatch.setattr(first,"validate_startup",lambda lease:None)
    monkeypatch.setattr(state,"consume",lambda lease:None)
    monkeypatch.setattr("darsia_cache.owned_process._public_command",lambda lease:state.request["public_command"])
    monkeypatch.setattr("darsia_cache.owned_process.OwnedWorker",Owned_TESTONLY)
    def slot(*args):
        if failed_diagnostic in {"observation","both"}:raise CachePolicyError("TESTONLY slot full")
        return pin("partial.json")
    def record(*args):
        if failed_diagnostic in {"held","both"}:raise CachePolicyError("TESTONLY held full")
    monkeypatch.setattr(first,"_slot_write",slot);lease._record=record
    with pytest.raises(RuntimeError) as captured:first.supervise_first_worker(lease)
    assert captured.value is primary and events==["closed"]
    notes=" ".join(primary.__notes__)
    assert "not recorded" in notes
    if failed_diagnostic in {"observation","both"}:assert "full samples are not durable" in notes


def authority_file_TESTONLY(tmp_path,raw):
    import hashlib
    path=tmp_path/"answers.json";path.write_bytes(raw)
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))


def test_authority_list_exact_raw_source_and_original_object_reader_stays_strict(tmp_path):
    from darsia_cache.bootstrap import _exact,_exact_authority_list
    binding=authority_file_TESTONLY(tmp_path,b'[{"id":"N43","decided_by":"owner"}]\n')
    package=str(tmp_path/"source"/"darsia"/"__init__.py")
    assert _exact_authority_list(binding,package)==[{"id":"N43","decided_by":"owner"}]
    with pytest.raises(ValueError,match="JSON object"):_exact(binding,package)


@pytest.mark.parametrize("raw",[b'{}',b'[null]',b'[1]',b'[{"id":1,"id":2}]',
    b'[{"id":NaN}]',b'[{"id":Infinity}]',b'[{"id":-Infinity}]',b'[],"extra":{}'])
def test_authority_list_preserves_typed_duplicate_and_nonfinite_rejections(tmp_path,raw):
    from darsia_cache.bootstrap import _exact_authority_list
    binding=authority_file_TESTONLY(tmp_path,raw)
    with pytest.raises(ValueError):
        _exact_authority_list(binding,str(tmp_path/"source"/"darsia"/"__init__.py"))


@pytest.mark.parametrize("change",["wrong-sha","changed-bytes","negative","oversize","missing","unc"])
def test_authority_list_keeps_original_raw_descriptor_and_path_guards(tmp_path,change):
    from pathlib import Path
    from darsia_cache.bootstrap import _exact_authority_list
    binding=authority_file_TESTONLY(tmp_path,b'[{"id":"N43"}]')
    if change=="wrong-sha":binding["sha256"]="0"*64
    elif change=="changed-bytes":Path(binding["path"]).write_bytes(b'[{"id":"BAD"}]')
    elif change=="negative":binding["size_bytes"]=-1
    elif change=="oversize":binding["size_bytes"]=16*1024**2+1
    elif change=="missing":binding["path"]=str(tmp_path/"not-created.json")
    elif change=="unc":binding["path"]=r"\\TESTONLY\share\answers.json"
    with pytest.raises((ValueError,OSError)):
        _exact_authority_list(binding,str(tmp_path/"source"/"darsia"/"__init__.py"))


def test_direct_bound_import_profile_cannot_bypass_checked_provider_DAG(monkeypatch):
    fake = dict(format=first.BOUND_IMPORT_PROFILE,
        completion=pin("fake-complete-passed.json"))
    def no_slot_read(*args, **kwargs):
        raise AssertionError("Direct typed profiles must refuse before slot or allocation use")
    monkeypatch.setattr(first, "_slot_read", no_slot_read)
    with pytest.raises(CachePolicyError, match="original checked provider DAG"):
        first.resolve_profile(pin("direct-import.json"), "import", lambda binding: fake)


def test_direct_bound_job_profile_cannot_bypass_checked_provider_DAG(monkeypatch):
    fake = dict(format=first.BOUND_JOB_PROFILE,
        completion=pin("fake-complete-passed.json"))
    def no_slot_read(*args, **kwargs):
        raise AssertionError("Direct typed profiles must refuse before slot or allocation use")
    monkeypatch.setattr(first, "_slot_read", no_slot_read)
    with pytest.raises(CachePolicyError, match="original checked provider DAG"):
        first.resolve_profile(pin("direct-job.json"), "jobs", lambda binding: fake)


def test_bound_provider_still_dispatches_to_complete_checked_DAG(monkeypatch):
    actual_provider = provider(); provider_binding = pin("provider.json")
    bundle = dict(format=first.BOUND_BUNDLE); selected = {"TESTONLY": "validated dispatch result"}
    calls = []
    def slot(value, role):
        assert value == actual_provider and role == "completion"
        return bundle, pin("completion.json")
    def checked(binding, value, completed, kind, read, *, stores_source, read_authorities):
        calls.append((binding, value, completed, kind, stores_source, read_authorities))
        return selected
    monkeypatch.setattr(first, "_slot_read", slot)
    monkeypatch.setattr(first, "_resolve_bound_profile", checked)
    read = lambda binding: actual_provider
    authorities = lambda binding: []
    assert first.resolve_profile(provider_binding, "import", read,
        stores_source=pin("stores.toml"), read_authorities=authorities) is selected
    assert calls == [(provider_binding, actual_provider, bundle, "import",
        pin("stores.toml"), authorities)]


def q123_admission_TESTONLY(monkeypatch):
    import hashlib
    from pathlib import Path
    from darsia_cache.retained_qualification import RetainedColdScope
    from darsia_cache.control import _encoded
    from darsia_cache import service_context
    state=SimpleNamespace(bound_mode=True,reference_build=False)
    cap=SimpleNamespace(maximum_size_bytes=500*1024**3,document=lambda:{"TESTONLY":"same cap"})
    policy=SimpleNamespace(root="C:/TESTONLY/cache",volume_cap=cap,purpose=lambda purpose:Path("C:/TESTONLY/cache/jobs"))
    owner=dict(pid=101,process_create_time="TESTONLY-original-epoch");token="a"*32
    row=dict(format="darsia-cache-budget-reservation-v2",token=token,pool="TESTONLY",owner=owner,maximum_bytes=100)
    raw=_encoded(row);reservation=dict(token=token,pool="TESTONLY",sha256=hashlib.sha256(raw).hexdigest())
    close=_encoded(dict(owner=owner,token=token,reservation_sha256=reservation["sha256"],no_further_writes=True))
    records={"cache-r-"+token+".json":raw,"cache-c-"+token+".json":close}
    admission=dict(cache_identity={"TESTONLY":"cache ID"},host_ledger={"path":"C:/TESTONLY/ledger"})
    old=SimpleNamespace(policy=policy,reservation=reservation,ledger=SimpleNamespace(admission=admission),
        manifest=dict(owner=owner,root_identity={"TESTONLY":"old service"}))
    calls=[];value=RetainedColdScope.__new__(RetainedColdScope)
    value.value=dict(retained_cache_inventory=pin("retained.json"));value._seal=first._digest(value.value)
    value.close_record=dict(path="C:/TESTONLY/ledger/cache-c-"+token+".json",
        sha256=hashlib.sha256(close).hexdigest(),size_bytes=len(close))
    value.closed=SimpleNamespace(original=old,
        retired_after_close_locked=lambda ledger,pin:calls.append((ledger,pin)) or True)
    value.read=lambda binding:dict(logical_bytes=117590)
    monkeypatch.setattr(service_context,"verify_retained_cache_inventory",lambda document,root:document)
    monkeypatch.setattr(value,"matplotlib_root",lambda *args,**kwargs:"C:/TESTONLY/output/diagnostic/mpl")
    ledger=SimpleNamespace(control=SimpleNamespace(names=lambda:list(records),read_once=lambda name:(records[name],None)))
    lease=SimpleNamespace(context=dict(host="reference-host"),policy=policy,admission=admission,ledger=ledger)
    report=dict(current_cache_logical_bytes=117590,active_cache_remaining_bytes=0,hard_filesystem_quota_verified=True)
    return value,state,lease,report,records,calls


def test_Q123_retained_bytes_stay_used_before_new_service_reservation(monkeypatch):
    value,state,lease,report,records,calls=q123_admission_TESTONLY(monkeypatch);before=deepcopy(records)
    assert value.admit(state,lease,report)==dict(logical_bytes=117590)
    assert len(calls)==1 and records==before and report["current_cache_logical_bytes"]==117590


@pytest.mark.parametrize("change",["root","cap","identity","active","used","uncapped","close"])
def test_Q123_refuses_changed_canonical_resources_and_unaccounted_bytes(monkeypatch,change):
    value,state,lease,report,records,_=q123_admission_TESTONLY(monkeypatch)
    if change in {"root","cap","uncapped"}:
        lease.policy=SimpleNamespace(**vars(lease.policy))
        if change=="root":lease.policy.root="C:/TESTONLY/another-cache"
        if change=="cap":lease.policy.volume_cap=SimpleNamespace(document=lambda:{"TESTONLY":"different"})
        if change=="uncapped":lease.policy.volume_cap=None
    if change=="identity":lease.admission=dict(lease.admission,cache_identity={"TESTONLY":"different"})
    if change=="active":report["active_cache_remaining_bytes"]=1
    if change=="used":report["current_cache_logical_bytes"]=0
    if change=="close":records["cache-c-"+"a"*32+".json"]=b"{}"
    before=deepcopy(records)
    with pytest.raises(CachePolicyError):value.admit(state,lease,report)
    assert records==before


def test_Q123_fully_written_unclosed_prior_service_still_forbids_exclusivity(monkeypatch):
    value,state,lease,report,records,_=q123_admission_TESTONLY(monkeypatch)
    records["cache-r-"+"b"*32+".json"]=b"TESTONLY fully written live reservation"
    assert report["active_cache_remaining_bytes"]==0
    with pytest.raises(CachePolicyError,match="Every prior reservation"):value.admit(state,lease,report)


@pytest.mark.parametrize("change",[None,"old-token","nonempty","opened","owner","old-identity"])
def test_Q123_new_service_requires_original_empty_FILE_CREATE_provenance(monkeypatch,change):
    from pathlib import Path
    from darsia_cache.environment import FIRST_WRITER_SUBDIRECTORIES,FIRST_WRITER_ENVIRONMENT_KEYS
    value,state,lease,_,_,_=q123_admission_TESTONLY(monkeypatch)
    token="b"*32;owner=dict(pid=202,process_create_time="TESTONLY-new-epoch");identity={"TESTONLY":"new service"}
    lease.reservation=dict(token=token,pool="TESTONLY",sha256="c"*64);lease.directories={"":identity}
    fresh=dict(format="darsia-fresh-exclusive-empty-service-scope-v1",token=token,owner=owner,purpose="jobs",
        root_identity=identity,creation_disposition="NtCreateFile.FILE_CREATE",exclusive_root_handle=True,
        initial_names=[],initial_empty_observed_utc="2026-10-18T21:00:00+00:00",
        observation_precedes_writer_subdirectories_and_imports=True,
        writer_subdirectories=list(FIRST_WRITER_SUBDIRECTORIES),writer_environment_keys=list(FIRST_WRITER_ENVIRONMENT_KEYS))
    lease.manifest=dict(token=token,owner=owner,purpose="jobs",reservation=lease.reservation,
        root=str(Path(lease.policy.root)/"jobs"/token),root_identity=identity,fresh_scope_provenance=fresh)
    if change=="old-token":lease.reservation["token"]=value.closed.original.reservation["token"]
    if change=="nonempty":fresh["initial_names"]=["old.npy"]
    if change=="opened":fresh["creation_disposition"]="NtCreateFile.FILE_OPEN"
    if change=="owner":fresh["owner"]={"TESTONLY":"unrelated"}
    if change=="old-identity":lease.manifest["root_identity"]=value.closed.original.manifest["root_identity"]
    if change is None:assert value.require_service(state,lease) is True
    else:
        with pytest.raises(CachePolicyError,match="genuinely new empty exclusive"):
            value.require_service(state,lease)


def test_Q123_only_exact_request_bound_MPL_exception_is_readable():
    observation,evidence=bounded_evidence_TESTONLY()
    external="C:/TESTONLY/output/diagnostic/mpl";binding=pin("Q123-cold.json")
    scope=SimpleNamespace(binding=binding,matplotlib_root=lambda first:external)
    state=SimpleNamespace(q123_cold_scope=scope,request=dict(q123_cold_scope=binding))
    evidence["q123_cold_scope"]=binding
    evidence["registered_owner_environments"][0]["actual_writer_environment"]["MPLCONFIGDIR"]=external
    observation["actual_registered_owner_exit_inventories"]=evidence["registered_owner_environments"]
    evidence["observation_sha256"]=first._digest(observation)
    assert first.checked_bounded_evidence(observation,evidence,first=state)==evidence
    with pytest.raises(CachePolicyError):first.checked_bounded_evidence(observation,evidence)
    for key in ("TEMP","MPLCONFIGDIR"):
        changed=deepcopy(evidence);observed=deepcopy(observation)
        changed["registered_owner_environments"][0]["actual_writer_environment"][key]=external+"/arbitrary"
        observed["actual_registered_owner_exit_inventories"]=changed["registered_owner_environments"]
        changed["observation_sha256"]=first._digest(observed)
        with pytest.raises(CachePolicyError):first.checked_bounded_evidence(observed,changed,first=state)


def test_Q123_external_MPL_is_empty_and_counted_in_the_watched_output(monkeypatch):
    from darsia_cache.retained_qualification import RetainedColdScope
    from darsia_cache import windows
    value=RetainedColdScope.__new__(RetainedColdScope)
    value.matplotlib_namespace=dict(path="C:/TESTONLY/output/diagnostic/mpl",identity={"TESTONLY":"exact ID"})
    value.closed=SimpleNamespace(original=SimpleNamespace(policy=SimpleNamespace(root="C:/TESTONLY/cache")))
    state=SimpleNamespace(output_entitlement=dict(watched_output_scope=dict(path="C:/TESTONLY/output")))
    names=[];calls=[]
    def namespace(path,identity):
        calls.append((path,identity));return SimpleNamespace(names=lambda:list(names),close=lambda:None)
    monkeypatch.setattr(windows,"WindowsNamespace",namespace)
    assert value.matplotlib_root(state,require_empty=True)==value.matplotlib_namespace["path"]
    assert calls==[(value.matplotlib_namespace["path"],value.matplotlib_namespace["identity"])]
    names.append("prior-font.json")
    with pytest.raises(CachePolicyError,match="must be empty"):value.matplotlib_root(state,require_empty=True)
    value.matplotlib_namespace["path"]="C:/TESTONLY/cache/prior-mpl"
    with pytest.raises(CachePolicyError,match="counted diagnostic child"):value.matplotlib_root(state)
