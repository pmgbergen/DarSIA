"""TESTONLY protocol cases; no actual N43 reference, namespace or profile."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest

from darsia_cache import first_qualification as first
from darsia_cache.policy import CachePolicyError, PHYSICAL_RESERVE_BYTES
from darsia.fluidflower.pipeline import capacity


def pin_TESTONLY(name):
    return dict(path="C:/TESTONLY/"+name,sha256="a"*64,size_bytes=1)


def administration_TESTONLY(host="reference-host"):
    scope=dict(commit="a"*40,pipeline="native-phase-n1",backend="cpu",precision="float32",
        run="ac52",ordered_case_ids=first.REFERENCE_CASE_IDS)
    state=SimpleNamespace(provider={"scope":scope,"python_executable":pin_TESTONLY("python.exe")},
        stores_source=pin_TESTONLY("stores.toml"),request={"task":pin_TESTONLY("task.json"),
        "public_command":["C:/TESTONLY/python.exe","-B","-m","darsia.fluidflower.worker",
            "--workers","1","--backend","cpu","--once","--job","//TESTONLY/share/job",
            "--stores","C:/TESTONLY/stores.toml"]},
        task={"configuration":{"TESTONLY":"actual task input binding"}})
    roots=[dict(role=role,path=("//TESTONLY/share/job" if role=="job" else drive+"/TESTONLY/"+role),identity=({"format":"original-j1-shared-folder-admission-v1","protocol":"smb"}
            if role=="job" else {"TESTONLY":role}),volume_id=drive,
        stat_identity={"st_dev":1,"st_ino":2,"creation_ns":3})
        for role,drive in [("cache","Q:" if host=="reference-host" else "D:"),
                          ("results","C:" if host=="reference-host" else "D:"),("job","//TESTONLY/share")]]
    value=dict(format=first.REFERENCE_ADMINISTRATION,authority_ids=["N43","Q109","Q106"],
        answers_source=pin_TESTONLY("answers.json"),accepted_plan=dict(path="C:/TESTONLY/9dc.json",
            sha256="9dc0979d5cd48ce18df6391253ed3f178f181c0c321626cef179c25a2e59e988",size_bytes=1),
        scope=scope,task=state.request["task"],host=host,
        host_policy="main-q87-fixed500" if host=="reference-host" else "second-n43-one-run-uncapped-d",
        owned_roots=roots,entry_limit=1000,aggregate_entitlement_bytes=65728995330,
        counting_semantics="all-stages-products-intermediates-receipts-previews-caches-once",
        watchdog_period_seconds=1,operational_reserve_bytes=PHYSICAL_RESERVE_BYTES,
        minimum_family_memory_limit_bytes=20*1024**3,memory_reserve_bytes=PHYSICAL_RESERVE_BYTES,
        all_external_writers_redirected=True,exclusive_producer=True,
        output_configuration=state.task["configuration"],requested_native_layout={"TESTONLY":"actual layout"},
        output_writers=[{"TESTONLY":"closed source pins"}],producer_state="PREDECLARED_UNQUALIFIED_REFERENCE_BUILD")
    answers=[dict(id="N43-OWNER-GO-execute-plan-9dc0979d-tonight-Main-reference-and-Second-qualification",decided_by="owner"),
        dict(id="Q109-answer-option-A-whole-output-entitlement-65728995330B-all-stages-count",decided_by="coordinator"),
        dict(id="Q106-answer-sampled-operational-reserve-hard-bound-is-Q87-volume-no-new-seam",decided_by="coordinator")]
    return state,value,lambda binding:deepcopy(answers if binding==value["answers_source"] else {"TESTONLY":"read exact plan"})


def test_n43_explicit_human_authority_exact_scope_and_aggregate_are_not_profiles():
    state,value,read=administration_TESTONLY()
    actual=first.checked_reference_administration(value,state,read=read)
    assert actual["aggregate_entitlement_bytes"]==65728995330
    assert not any(key in actual for key in ("passed","measurement","maximum_observed_cache_bytes"))


@pytest.mark.parametrize("mutation",[
    lambda x:x.update(aggregate_entitlement_bytes=27374089354),
    lambda x:x.update(exclusive_producer=False),
    lambda x:x.update(watchdog_period_seconds=30),
    lambda x:x.update(operational_reserve_bytes=0),
    lambda x:x.update(minimum_family_memory_limit_bytes=1024),
    lambda x:x.update(host_policy="ordinary-uncapped"),
    lambda x:x.update(output_configuration={"TESTONLY":"different calibration/RAW config"}),
    lambda x:x["scope"].update(ordered_case_ids=list(reversed(first.REFERENCE_CASE_IDS))),
    lambda x:x["owned_roots"][1].update(path=x["owned_roots"][0]["path"]+"/nested"),
])
def test_n43_refuses_wrong_entitlement_scope_reserve_and_hidden_root_overlap(mutation):
    state,value,read=administration_TESTONLY();mutation(value)
    with pytest.raises(CachePolicyError):first.checked_reference_administration(value,state,read=read)


def test_second_host_exception_is_only_the_human_one_run_D_namespace():
    state,value,read=administration_TESTONLY("second-host")
    assert first.checked_reference_administration(value,state,read=read)["host_policy"].endswith("uncapped-d")
    value["owned_roots"][0]["path"]="C:/TESTONLY/cache"
    with pytest.raises(CachePolicyError,match="assigned D"):
        first.checked_reference_administration(value,state,read=read)


def test_wrong_or_absent_actual_human_GO_refuses():
    state,value,_=administration_TESTONLY()
    with pytest.raises(CachePolicyError,match="humanGO"):
        first.checked_reference_administration(value,state,read=lambda binding:[])


class Ledger_TESTONLY:
    def __init__(self):self.lock_active=False;self.documents={};self.control=self
    @contextmanager
    def locked(self,**kwargs):
        assert not self.lock_active;self.lock_active=True
        try:yield self
        finally:self.lock_active=False
    def names(self):return list(self.documents)
    def write_once(self,name,raw):
        assert self.lock_active
        if name in self.documents:raise AssertionError("TESTONLY no overwrite")
        self.documents[name]=raw


def stop_state_TESTONLY():
    state=SimpleNamespace(binding=pin_TESTONLY("request.json"),provider_binding=pin_TESTONLY("provider.json"),
        provider={"scope":{"TESTONLY":"actual scope"}},_once_key=lambda:"a"*64)
    return state,SimpleNamespace(ledger=Ledger_TESTONLY())


def test_budget_crossing_is_durable_even_when_later_sample_recovers(monkeypatch):
    state,lease=stop_state_TESTONLY()
    monkeypatch.setattr(first,"request_existing_frame_boundary_stop",lambda *args:{"requested":False,"state":"TESTONLY"})
    first.latch_reference_stop(state,lease,"AGGREGATE_ENTITLEMENT_EXCEEDED",{"logical_bytes":65728995331})
    assert first.reference_stopped(state,lease)
    with pytest.raises(CachePolicyError,match="persistent STOP"):
        first.require_reference_not_stopped(state,lease)
    first.latch_reference_stop(state,lease,"OPERATIONAL_RESERVE_CROSSED",{"logical_bytes":1})
    assert len(lease.ledger.documents)==1


def test_publication_delegates_same_inputs_and_result_with_fences(monkeypatch):
    events=[];result={"TESTONLY":"original exact descriptor result"}
    def fence(*args):events.append("fence")
    class Original_TESTONLY:
        def __call__(self,*args,**kwargs):events.append((args,kwargs));return result
        def publish_progress(self,*args,**kwargs):return (args,kwargs)
    monkeypatch.setattr(first,"fence_reference_publication",fence)
    monkeypatch.setattr(first,"reference_publication_binding",lambda *args:{"TESTONLY":"exact typed guard"})
    wrapped=first.ReferencePublicationFence(Original_TESTONLY(),object(),object())
    actual=wrapped(1,2,True)
    assert actual==dict(result,n43_reference_guard={"TESTONLY":"exact typed guard"})
    assert result=={"TESTONLY":"original exact descriptor result"}
    assert events==["fence",((1,2,True),{}),"fence"]
    assert wrapped.publish_progress(3)==((3,),{})


def test_observed_STOP_prevents_original_publisher_call(monkeypatch):
    called=[]
    def refuse(*args):raise CachePolicyError("TESTONLY observed STOP")
    monkeypatch.setattr(first,"fence_reference_publication",refuse)
    wrapped=first.ReferencePublicationFence(lambda *args:called.append(args),object(),object())
    with pytest.raises(CachePolicyError,match="observed STOP"):wrapped(1,2,True)
    assert called==[]


def test_observed_crossing_after_original_prepare_retains_partial_and_refuses_promotion(monkeypatch):
    calls=[]
    def fence(*args):
        calls.append("fence")
        if len(calls)==3:raise CachePolicyError("TESTONLY aggregate exceeded")
    def prepare(*args):calls.append("partial shared files retained");return {"TESTONLY":"candidate"}
    monkeypatch.setattr(first,"fence_reference_publication",fence)
    with pytest.raises(CachePolicyError,match="aggregate exceeded"):
        first.ReferencePublicationFence(prepare,object(),object())(1,2,True)
    assert calls==["fence","partial shared files retained","fence"]


def test_administrative_result_model_has_no_fake_measured_size_or_profile():
    value=capacity.PredeclaredN43ProducerEntitlement("ac52",39,65728995330,
        pin_TESTONLY("request.json"),pin_TESTONLY("admin.json"),{"TESTONLY":"one actual service token"})
    assert value.maximum_aggregate_bytes==65728995330
    assert not isinstance(value,capacity.SpaceEstimate)
    assert not hasattr(value,"measured_native_bytes_per_frame")
    assert not hasattr(value,"products_per_frame_estimated_bytes")
    assert "NOT measured profile" in value.model


def test_explicit_reference_request_binds_the_actual_admin_in_Root_payload():
    state,value,read_admin=administration_TESTONLY()
    source_pins=[pin_TESTONLY("producer.py"),pin_TESTONLY("runtime.py")]
    provider=dict(format=first.PROVIDER,qualification_id="b"*32,scope=value["scope"],
        namespace_admission=pin_TESTONLY("namespace.json"),
        output_namespace=dict(path="C:/TESTONLY/receipts",identity={"TESTONLY":"actual root"}),
        slots={role:role+".json" for role in first.SLOTS},python_executable=pin_TESTONLY("python.exe"),
        startup=pin_TESTONLY("startup.json"),module_closure=source_pins,
        producer_source_closure=source_pins[:1],runtime_source_closure=source_pins[1:])
    task=deepcopy(provider["scope"]);task.pop("backend")
    task.update(attempt_id="c"*32,job_id="TESTONLY-job",configuration=value["output_configuration"])
    request=dict(format=first.REFERENCE_REQUEST,execution_authorized=True,
        authority_ids=["N43","Q109","Q106"],root_release=pin_TESTONLY("release.json"),
        provider=pin_TESTONLY("provider.json"),stores_source=pin_TESTONLY("stores.toml"),
        namespace_admission=provider["namespace_admission"],scope=provider["scope"],
        public_command=["C:/TESTONLY/python.exe","-B","-m","darsia.fluidflower.worker",
            "--workers","1","--backend","cpu","--once","--job","//TESTONLY/share/job",
            "--stores","C:/TESTONLY/stores.toml"],
        task=pin_TESTONLY("task.json"),cache_stat_period_seconds=30,
        reference_administration=pin_TESTONLY("admin.json"))
    release=dict(format="darsia-cache-first-qualification-root-release-v1",execution_authorized=True,
        provider=request["provider"],stores_source=request["stores_source"],scope=request["scope"],
        request_payload_sha256=first._digest({k:v for k,v in request.items() if k!="root_release"}))
    objects={"request.json":request,"release.json":release,"provider.json":provider,"task.json":task,"admin.json":value}
    def read(pin):
        return deepcopy(objects[pin["path"].split("/")[-1]]) if pin["path"].split("/")[-1] in objects else read_admin(pin)
    actual=first.FirstQualification(pin_TESTONLY("request.json"),provider_binding=request["provider"],
        stores_source=request["stores_source"],namespace=request["namespace_admission"],read=read)
    assert actual.reference_build and not actual.bound_mode
    assert actual.reference_administration["aggregate_entitlement_bytes"]==65728995330
    assert actual.output_entitlement is None  # Never relabel the old Q104 product-only model.
    request["reference_administration"]=pin_TESTONLY("changed.json")
    with pytest.raises(CachePolicyError,match="Root-released"):
        first.FirstQualification(pin_TESTONLY("request.json"),provider_binding=request["provider"],
            stores_source=request["stores_source"],namespace=request["namespace_admission"],read=read)


def test_ordinary_profile_resolver_remains_direct_and_does_not_mint_reference_profile():
    actual={"format":"darsia-cache-import-profile-v1","TESTONLY":"original strict downstream guards still apply"}
    assert first.resolve_profile(pin_TESTONLY("direct.json"),"import",lambda pin:actual) is actual
    with pytest.raises(CachePolicyError):
        first.require_publishable({"format":first.OBSERVATION,"complete":True},
            {"TESTONLY":"administrative observed bytes are not lifetime profile evidence"})


@pytest.mark.parametrize("change",[
    lambda argv:argv.__setitem__(argv.index("--workers")+1,"3"),
    lambda argv:argv.append("--watchdog"),
    lambda argv:argv.remove("--once"),
])
def test_reference_build_rejects_extra_workers_and_automatic_retry(change):
    state,value,read=administration_TESTONLY();change(state.request["public_command"])
    with pytest.raises(CachePolicyError,match="one original whole-run"):
        first.checked_reference_administration(value,state,read=read)


def entitlement_runtime_TESTONLY(monkeypatch):
    state,value,_=administration_TESTONLY()
    task=dict(value["scope"],configuration=value["output_configuration"])
    task.pop("backend")
    task["attempt_id"]="a"*32
    required=["mass/n1_pipeline.py","mass/n1_statistics.py","pipeline/series.py",
        "native/sidecar.py","pipeline/runner.py"]
    value["output_writers"]=[dict(relative_path="src/darsia/fluidflower/"+path,sha256="a"*64) for path in required]
    reference=SimpleNamespace(reference_build=True,reference_administration=value,
        binding=pin_TESTONLY("request.json"),request={"reference_administration":pin_TESTONLY("admin.json")},
        require_task=lambda lease,actual:actual==task)
    lease=SimpleNamespace(first_qualification=reference,reservation={"token":"b"*32})
    stores=SimpleNamespace(roots=lambda series:{"results":"C:/TESTONLY/results"})
    runtime=SimpleNamespace(cache_adapter=SimpleNamespace(lease=lease),stores=stores,
        job={"series_id":"TESTONLY"},root="//TESTONLY/share/job",repository="C:/TESTONLY/source")
    config=dict(frames=[{"case_id":case} for case in task["ordered_case_ids"]])
    monkeypatch.setattr(capacity,"settings",lambda *args:(None,PHYSICAL_RESERVE_BYTES,pin_TESTONLY("admin.json")))
    monkeypatch.setattr(capacity,"_native_layout_for_task",lambda *args,**kwargs:value["requested_native_layout"])
    calls=[]
    monkeypatch.setattr(capacity,"_writer_identity",lambda *args:calls.append(args))
    return runtime,task,config,value,calls


def test_real_N43_claim_metadata_uses_closed_writer_layout_and_one_actual_service(monkeypatch):
    runtime,task,config,value,calls=entitlement_runtime_TESTONLY(monkeypatch)
    actual=capacity.n43_entitlement(runtime,task,config,())
    assert actual.shared_service_reservation is runtime.cache_adapter.lease.reservation
    assert actual.maximum_aggregate_bytes==65728995330 and actual.case_count==39
    assert len(calls)==len(value["output_writers"])==5
    assert all(args[1]==task["commit"] for args in calls)


@pytest.mark.parametrize("mutation",[
    lambda runtime,task,config,value:config.update(storage={"TESTONLY":"not this route"}),
    lambda runtime,task,config,value:config.update(pixel_domain={"TESTONLY":"not this route"}),
    lambda runtime,task,config,value:config.update(gas_probability_storage=None),
    lambda runtime,task,config,value:config.update(gas_probability_storage={"enabled":True}),
    lambda runtime,task,config,value:config["frames"].reverse(),
    lambda runtime,task,config,value:value["output_writers"].pop(),
    lambda runtime,task,config,value:setattr(runtime,"root","Z:/TESTONLY/foreign-job"),
])
def test_N43_typed_output_admission_rejects_extra_writers_options_and_wrong_case_or_root(monkeypatch,mutation):
    runtime,task,config,value,_=entitlement_runtime_TESTONLY(monkeypatch)
    mutation(runtime,task,config,value)
    with pytest.raises(ValueError):capacity.n43_entitlement(runtime,task,config,())


def test_N43_cache_claim_checks_same_aggregate_not_a_second_output_allocation():
    from darsia_cache.control import ClaimCacheBudget
    calls=[]
    reference=SimpleNamespace(reference_build=True,
        reference_administration={"owned_roots":[dict(role="capacity_state",path="C:/TESTONLY/state")]},
        require_task=lambda lease,task:calls.append("exact task"),
        reference_admission=lambda lease,**kwargs:dict(aggregate_counted_once=True,
            entitlement_bytes=65728995330,other=kwargs["other_remaining"]))
    lease=SimpleNamespace(first_qualification=reference,reservation={"token":"a"*32},
        maximum=65728995330,manifest={"owner":{"TESTONLY":"actual primary service"}},
        document={"capacity":{"state_directory":"C:/TESTONLY/state"}})
    ledger=SimpleNamespace(require_reservation=lambda *args:calls.append(args))
    budget=ClaimCacheBudget.__new__(ClaimCacheBudget)
    budget.adapter=SimpleNamespace(lease=lease);budget.ledger=ledger
    budget.capacity=SimpleNamespace(directory="C:/TESTONLY/state")
    actual=budget.check({"TESTONLY":"task"},object(),0)
    assert actual["aggregate_counted_once"] and actual["entitlement_bytes"]==65728995330
    assert calls[1]==(lease.reservation,lease.manifest["owner"],65728995330)


def coverage_state_TESTONLY(*, extra=True):
    state,value,_=administration_TESTONLY()
    if extra:
        value["owned_roots"] += [dict(role=role,path="C:/TESTONLY/"+role,
            identity={"TESTONLY":role},volume_id="C:",
            stat_identity={"st_dev":1,"st_ino":2,"creation_ns":3}) for role in ("receipts","capacity_state","host_ledger")]
    state.reference_administration=value
    lease=SimpleNamespace(admission={role:dict(path="C:/TESTONLY/"+role,identity={"TESTONLY":role})
            for role in ("receipts","host_ledger")},
        document={"capacity":{"state_directory":"C:/TESTONLY/capacity_state"}},
        root="Q:/TESTONLY/cache/jobs/TESTONLY-service")
    return state,lease


def test_actual_receipts_host_ledger_and_capacity_state_siblings_are_counted():
    state,lease=coverage_state_TESTONLY()
    actual=first.reference_writer_coverage(state,lease)
    assert {row["aggregate_root_role"] for row in actual}=={"receipts","host_ledger","capacity_state"}


def test_omitted_control_sibling_refuses_before_original_producer():
    state,lease=coverage_state_TESTONLY(extra=False)
    with pytest.raises(CachePolicyError,match="omitted"):
        first.reference_writer_coverage(state,lease)


def test_actual_control_identity_must_match_accepted_namespace():
    state,lease=coverage_state_TESTONLY()
    lease.admission["host_ledger"]["identity"]={"TESTONLY":"foreign actual directory"}
    with pytest.raises(CachePolicyError,match="kernel identity"):
        first.reference_writer_coverage(state,lease)


def test_nested_receipts_count_once_when_already_below_an_owned_root():
    state,lease=coverage_state_TESTONLY()
    state.reference_administration["owned_roots"]=[row for row in state.reference_administration["owned_roots"] if row["role"]!="receipts"]
    lease.admission["receipts"]["path"]="Q:/TESTONLY/cache/receipts"
    rows=first.reference_writer_coverage(state,lease)
    assert next(row for row in rows if row["writer_role"]=="receipts")["aggregate_root_role"]=="cache"


def test_actual_writer_environment_cannot_escape_inventoried_service_scope(monkeypatch):
    from darsia_cache.environment import FIRST_WRITER_ENVIRONMENT_KEYS
    state,lease=coverage_state_TESTONLY()
    for key in FIRST_WRITER_ENVIRONMENT_KEYS:
        monkeypatch.setenv(key,"UserProvidedCacheLocator" if key=="NUMBA_CACHE_LOCATOR_CLASSES"
            else str(lease.root)+"/TESTONLY-writer")
    rows=first.reference_writer_coverage(state,lease,writer_environment=True)
    assert len(rows)==3+len(FIRST_WRITER_ENVIRONMENT_KEYS)
    monkeypatch.setenv("TEMP","C:/TESTONLY/foreign-temp")
    with pytest.raises(CachePolicyError,match="escapes"):
        first.reference_writer_coverage(state,lease,writer_environment=True)


def test_shared_job_admission_uses_original_RunIO_transport_not_local_NTFS(monkeypatch):
    from darsia_cache import windows
    from darsia.fluidflower import job_filesystem
    state,value,_=administration_TESTONLY()
    state.reference_administration=value;state.reference_roots_validated=False
    state.reference_transport_observation=None
    calls=[]
    class LocalNamespace_TESTONLY:
        def __init__(self,path,identity):
            assert not str(path).startswith("//")
            calls.append((path,identity))
        def close(self):pass
    monkeypatch.setattr(windows,"WindowsNamespace",LocalNamespace_TESTONLY)
    monkeypatch.setattr(first,"_reference_directory_stat",lambda path:SimpleNamespace(st_dev=1,st_ino=2,st_ctime_ns=3))
    monkeypatch.setattr(job_filesystem,"inspect_shared_folder",lambda path:job_filesystem.SharedFolderAdmission(
        True,"smb","TESTONLY actual OS transport","TESTONLY original J1 provider"))
    first.validate_reference_roots(state)
    assert len(calls)==2 and state.reference_roots_validated
    assert state.reference_transport_observation["protocol"]=="smb"


def test_root_stat_identity_uses_creation_not_live_directory_mtime():
    value=SimpleNamespace(st_dev=1,st_ino=2,st_ctime_ns=8,st_birthtime_ns=3,st_mtime_ns=999)
    assert first._reference_stat_identity(value)==dict(st_dev=1,st_ino=2,creation_ns=3)


def test_live_snapshot_holds_no_immutable_directory_handles(monkeypatch,tmp_path):
    from darsia_cache import windows,capacity as cache_capacity
    path=tmp_path/"TESTONLY-owned";path.mkdir();(path/"one").write_bytes(b"abc")
    info=path.lstat()
    state=SimpleNamespace(reference_administration=dict(entry_limit=100,host_policy="main-q87-fixed500",
        owned_roots=[dict(role="cache",path=str(path),identity={"TESTONLY":"preaccepted separately"},
            stat_identity=first._reference_stat_identity(info),volume_id=str(info.st_dev))]),
        reference_transport_observation=None)
    def forbidden(*args):raise AssertionError("live snapshot must not retain immutable shareREAD handles")
    monkeypatch.setattr(windows,"WindowsNamespace",forbidden)
    monkeypatch.setattr(cache_capacity,"volume_info",lambda root:dict(volume_id=str(info.st_dev),free_bytes=10**12))
    monkeypatch.setattr(first,"reference_writer_coverage",lambda *args:[])
    actual=first.reference_scope_snapshot(state,object())
    assert actual["logical_bytes"]==3 and actual["observed_scan_anomalies"]==[]
    assert actual["non_atomic"] and not actual["complete_lifetime_peak"]


@pytest.mark.parametrize("entry_limit", [100, 1])
def test_live_snapshot_closes_parent_iterator_before_child_and_on_limit(monkeypatch,tmp_path,entry_limit):
    from darsia_cache import capacity as cache_capacity
    path=tmp_path/"TESTONLY-owned";path.mkdir()
    child=path/"nested";child.mkdir();(child/"one").write_bytes(b"abc")
    info=path.lstat();active=set();opened=[];closed=[];original=first.os.scandir
    class TrackedScan:
        def __init__(self,directory):
            # A child scan must begin after every ancestor scan has closed.
            assert not active
            self.directory=str(directory);self.stream=original(directory)
        def __enter__(self):
            active.add(self.directory);opened.append(self.directory)
            return self.stream
        def __exit__(self,*args):
            self.stream.close();active.remove(self.directory);closed.append(self.directory)
    monkeypatch.setattr(first.os,"scandir",TrackedScan)
    monkeypatch.setattr(cache_capacity,"volume_info",lambda root:dict(volume_id=str(info.st_dev),free_bytes=10**12))
    monkeypatch.setattr(first,"reference_writer_coverage",lambda *args:[])
    state=SimpleNamespace(reference_administration=dict(entry_limit=entry_limit,host_policy="main-q87-fixed500",
        owned_roots=[dict(role="cache",path=str(path),identity={"TESTONLY":"preaccepted separately"},
            stat_identity=first._reference_stat_identity(info),volume_id=str(info.st_dev))]),
        reference_transport_observation=None)
    if entry_limit==1:
        with pytest.raises(CachePolicyError,match="entry limit"):
            first.reference_scope_snapshot(state,object())
    else:
        actual=first.reference_scope_snapshot(state,object())
        assert actual["logical_bytes"]==3 and actual["entries"]==2
        assert actual["observed_scan_anomalies"]==[]
    assert not active and opened==closed==[str(path),str(child)]


def test_live_directory_move_is_reported_incomplete_not_a_false_lifetime_peak(monkeypatch,tmp_path):
    from darsia_cache import capacity as cache_capacity
    path=tmp_path/"TESTONLY-owned";path.mkdir();moving=path/"moving";moving.mkdir()
    root_info=path.lstat();original=first._reference_directory_stat
    def observed(directory):
        if str(directory)==str(moving):raise FileNotFoundError("TESTONLY original live atomic rename")
        return original(directory)
    state=SimpleNamespace(reference_administration=dict(entry_limit=100,host_policy="main-q87-fixed500",
        owned_roots=[dict(role="cache",path=str(path),identity={"TESTONLY":"preaccepted separately"},
            stat_identity=first._reference_stat_identity(root_info),volume_id=str(root_info.st_dev))]),
        reference_transport_observation=None)
    monkeypatch.setattr(first,"_reference_directory_stat",observed)
    monkeypatch.setattr(cache_capacity,"volume_info",lambda root:dict(volume_id=str(root_info.st_dev),free_bytes=10**12))
    monkeypatch.setattr(first,"reference_writer_coverage",lambda *args:[])
    actual=first.reference_scope_snapshot(state,object())
    assert actual["observed_bytes_are_lower_bound_if_scan_anomalies"]
    assert actual["observed_scan_anomalies"][0]["state"]=="DIRECTORY_MOVED_DURING_SAMPLE"


def test_publication_fence_refuses_incomplete_live_snapshot(monkeypatch):
    state,lease=stop_state_TESTONLY();events=[]
    monkeypatch.setattr(first,"reference_scope_snapshot",lambda *args:dict(
        observed_scan_anomalies=[{"state":"VANISHED_DURING_SAMPLE"}],logical_bytes=0,
        volumes=[{"free_bytes":10**12}]))
    monkeypatch.setattr(first,"request_existing_frame_boundary_stop",lambda *args:events.append("existing stop"))
    with pytest.raises(CachePolicyError,match="persistent STOP"):
        first.fence_reference_publication(state,lease)
    assert first.reference_stopped(state,lease) and events


def guard_context_TESTONLY(monkeypatch):
    from darsia_cache import bootstrap
    task=dict(attempt_id="a"*32,commit="b"*40,run="ac52")
    state=SimpleNamespace(reference_build=True,binding=pin_TESTONLY("request.json"),
        provider_binding=pin_TESTONLY("provider.json"),stores_source=pin_TESTONLY("stores.toml"),
        request={"task":pin_TESTONLY("task.json")},task=task,
        reference_administration={"owned_roots":[dict(role="job",path="//TESTONLY/share/job")]})
    events=[]
    state._namespace=lambda lease:events.append("actual namespace")
    state.require_task=lambda lease,value:events.append(("actual task",value))
    lease=SimpleNamespace(first_qualification=state,entry_kind="worker",purpose="jobs",source=state.stores_source,
        document={"cache_bootstrap":{"namespace_admission":pin_TESTONLY("namespace.json")}})
    monkeypatch.setattr(bootstrap,"current_import_lease",lambda:lease)
    monkeypatch.setattr(first,"fence_reference_publication",lambda *args:events.append("fresh aggregate/persistent STOP"))
    return state,lease,first.reference_publication_binding(state,lease),events


def test_typed_guard_initial_and_recovery_require_same_exact_managed_context(monkeypatch):
    state,lease,guard,events=guard_context_TESTONLY(monkeypatch)
    first.check_reference_publication_guard(guard,"//TESTONLY/share/job",state.task,success=True)
    assert events==["actual namespace",("actual task",state.task),"fresh aggregate/persistent STOP"]


@pytest.mark.parametrize("field",["request","provider","stores_source","task","namespace_admission"])
def test_typed_guard_rejects_mismatched_immutable_context_roles(monkeypatch,field):
    state,lease,guard,events=guard_context_TESTONLY(monkeypatch)
    guard[field]=pin_TESTONLY("foreign.json")
    with pytest.raises(CachePolicyError,match="differs"):
        first.check_reference_publication_guard(guard,"//TESTONLY/share/job",state.task,success=True)
    assert events==[]


def test_guarded_recovery_never_treats_missing_managed_context_as_ordinary(monkeypatch):
    from darsia_cache import bootstrap
    state,lease,guard,_=guard_context_TESTONLY(monkeypatch)
    def missing():raise CachePolicyError("TESTONLY no actual managed context")
    monkeypatch.setattr(bootstrap,"current_import_lease",missing)
    with pytest.raises(CachePolicyError,match="no actual managed context"):
        first.check_reference_publication_guard(guard,"//TESTONLY/share/job",state.task,success=True)


def test_actual_N43_context_rejects_missing_record_guard(monkeypatch):
    state,lease,guard,_=guard_context_TESTONLY(monkeypatch)
    monkeypatch.setenv(first.REQUEST_ENV,"TESTONLY actual Root-bound first request")
    with pytest.raises(CachePolicyError,match="missing its typed guard"):
        first.check_reference_publication_guard(None,"//TESTONLY/share/job",state.task,success=True)


def test_ordinary_publication_does_not_open_or_invent_a_managed_context(monkeypatch):
    from darsia_cache import bootstrap
    monkeypatch.delenv(first.REQUEST_ENV,raising=False)
    def forbidden():raise AssertionError("ordinary record must retain original route")
    monkeypatch.setattr(bootstrap,"current_import_lease",forbidden)
    assert first.check_reference_publication_guard(None,"TESTONLY ordinary root",{},success=True) is None


def test_failure_diagnostic_preserves_primary_traceback_despite_latched_STOP(monkeypatch):
    primary={"complete":False,"traceback":"TESTONLY exact original primary traceback"}
    def forbidden(*args):raise AssertionError("STOP must not prevent original failed diagnostic logging")
    monkeypatch.setattr(first,"fence_reference_publication",forbidden)
    monkeypatch.setattr(first,"reference_publication_binding",lambda *args:{"TESTONLY":"same typed guard"})
    actual=first.ReferencePublicationFence(lambda *args:primary,object(),object())(1,2,False)
    assert actual["traceback"]==primary["traceback"] and primary=={
        "complete":False,"traceback":"TESTONLY exact original primary traceback"}


def alias_queue_TESTONLY(monkeypatch,tmp_path):
    from darsia.fluidflower.pipeline import job_queue as queue_module
    queue=queue_module.JobQueue(tmp_path/"TESTONLY-job")
    token="a"*32;task=dict(format="darsia-alias-task-v1",run="ac52",attempt_id=token)
    directory=queue.root/"publishing"/"ac52";directory.mkdir()
    queue_module.write_immutable(directory/"owner.json",dict(task=task,token=token))
    queue_module.write_immutable(directory/"candidate.json",dict(run="ac52",complete=True,
        worker_references_closed=True,TESTONLY="unchanged original result bytes"))
    queue_module.write_immutable(directory/"index.json",dict(TESTONLY="original index validators covered by old tests"))
    publication=dict(format="darsia-alias-publication-v1",run="ac52",attempt_id=token,success=True,
        candidate=queue_module._own_reference(directory,directory/"candidate.json"),
        products_index=queue_module._own_reference(directory,directory/"index.json"),
        n43_reference_guard={"TESTONLY":"guard validation separately tested"})
    monkeypatch.setattr(queue,"_validate_alias_index",lambda *args:None)
    monkeypatch.setattr(queue,"_validate_alias_payload",lambda *args:None)
    return queue_module,queue,directory,publication


def test_original_recovery_calls_the_same_guard_before_completed_and_done(monkeypatch,tmp_path):
    module,queue,directory,publication=alias_queue_TESTONLY(monkeypatch,tmp_path)
    calls=[]
    monkeypatch.setattr(first,"check_reference_publication_guard",lambda *args,**kwargs:calls.append((args,kwargs)))
    module.write_immutable(directory/"publication.json",publication)
    queue.recover_publications()
    assert len(calls)==2 and all(row[1]=={"success":True} for row in calls)
    assert (queue.root/"results"/"ac52"/"completed.json").read_bytes()==(directory/"candidate.json").read_bytes()
    assert (queue.root/"done"/"ac52.json").is_file()


def test_recovery_observed_STOP_keeps_candidates_and_refuses_canonical_publication(monkeypatch,tmp_path):
    module,queue,directory,publication=alias_queue_TESTONLY(monkeypatch,tmp_path)
    def stopped(*args,**kwargs):raise CachePolicyError("TESTONLY latched aggregate STOP")
    monkeypatch.setattr(first,"check_reference_publication_guard",stopped)
    module.write_immutable(directory/"publication.json",publication)
    with pytest.raises(CachePolicyError,match="latched aggregate STOP"):queue.recover_publications()
    assert (directory/"candidate.json").is_file() and (directory/"publication.json").is_file()
    assert not (queue.root/"results"/"ac52"/"completed.json").exists()
    assert not (queue.root/"done"/"ac52.json").exists()


def test_later_STOP_after_completed_keeps_it_diagnostic_and_refuses_done(monkeypatch,tmp_path):
    module,queue,directory,publication=alias_queue_TESTONLY(monkeypatch,tmp_path)
    calls=[]
    def check(*args,**kwargs):
        calls.append(True)
        if len(calls)>1:raise CachePolicyError("TESTONLY later observed STOP")
    monkeypatch.setattr(first,"check_reference_publication_guard",check)
    with pytest.raises(CachePolicyError,match="later observed STOP"):
        queue._finish_alias_publication(directory,publication)
    assert (queue.root/"results"/"ac52"/"completed.json").read_bytes()==(directory/"candidate.json").read_bytes()
    assert not (queue.root/"done"/"ac52.json").exists()


def test_explicit_null_guard_is_not_an_ordinary_record(monkeypatch,tmp_path):
    module,queue,directory,publication=alias_queue_TESTONLY(monkeypatch,tmp_path)
    publication["n43_reference_guard"]=None
    with pytest.raises(ValueError,match="exact typed"):
        queue._finish_alias_publication(directory,publication)
    assert not (queue.root/"done"/"ac52.json").exists()


def initial_alias_claim_TESTONLY(monkeypatch,tmp_path):
    module,queue,directory,publication=alias_queue_TESTONLY(monkeypatch,tmp_path)
    task=module.read_json(directory/"owner.json")["task"];token=task["attempt_id"]
    lease=queue.root/"leases"/("ac52."+token);directory.rename(lease)
    claimed=queue.root/"claimed"/"TESTONLY-main"/"ac52.json"
    module.write_immutable(claimed,task)
    output=tmp_path/"TESTONLY-output";output.mkdir()
    def prepare(claim,candidate,success):
        original=Path_TESTONLY(candidate["path"]).read_bytes()
        target=claim.lease/("original_completed.json" if success else "original_failed.json")
        target.write_bytes(original)
        prepared=dict(candidate=module._own_reference(claim.lease,target),
            n43_reference_guard={"TESTONLY":"exact guard checked independently"})
        if success:prepared["products_index"]=module._own_reference(claim.lease,claim.lease/"index.json")
        return prepared
    claim=SimpleNamespace(task=task,token=token,lease=lease,claimed=claimed,output=output,
        publication_factory=prepare)
    return module,queue,claim


# Explicit alias avoids confusing the real Path use in fixture publication
# with the exact immutable producer path descriptor passed by the old queue.
from pathlib import Path as Path_TESTONLY


def test_original_initial_promotion_persists_typed_guard_and_checks_all_seams(monkeypatch,tmp_path):
    module,queue,claim=initial_alias_claim_TESTONLY(monkeypatch,tmp_path)
    calls=[]
    monkeypatch.setattr(first,"check_reference_publication_guard",lambda *args,**kwargs:calls.append(kwargs["success"]))
    result=dict(complete=True,run="ac52",worker_references_closed=True,TESTONLY="original scientific bytes")
    assert queue.complete(claim,result) is result
    assert calls==[True,True,True,True]
    # The persisted record stays in the in-place lease that the run marker names.
    assert module.publication_directory(queue.root,"ac52")==claim.lease and claim.lease.is_dir()
    record=module.read_json(claim.lease/"publication.json")
    assert record["n43_reference_guard"]=={"TESTONLY":"exact guard checked independently"}
    assert module.read_json(queue.root/"results"/"ac52"/"completed.json")==result
    assert (queue.root/"done"/"ac52.json").is_file()


def test_initial_guard_STOP_preserves_original_candidate_without_ownership_move(monkeypatch,tmp_path):
    module,queue,claim=initial_alias_claim_TESTONLY(monkeypatch,tmp_path)
    def stopped(*args,**kwargs):raise CachePolicyError("TESTONLY observed entitlement STOP")
    monkeypatch.setattr(first,"check_reference_publication_guard",stopped)
    result=dict(complete=True,run="ac52",worker_references_closed=True)
    with pytest.raises(CachePolicyError,match="entitlement STOP"):queue.complete(claim,result)
    assert module.read_json(claim.output/"candidate_completed.json")==result
    assert claim.lease.is_dir() and not (queue.root/"done"/"ac52.json").exists()


def test_original_failed_json_is_diagnostic_and_keeps_exact_primary_traceback(monkeypatch,tmp_path):
    module,queue,claim=initial_alias_claim_TESTONLY(monkeypatch,tmp_path)
    calls=[]
    monkeypatch.setattr(first,"check_reference_publication_guard",lambda *args,**kwargs:calls.append(kwargs["success"]))
    primary="TESTONLY primary traceback retained verbatim after sampled budget STOP"
    actual=queue.fail(claim,primary,{"TESTONLY":"actual owned actor"})
    assert actual["traceback"]==primary and actual["complete"] is False
    assert module.read_json(queue.root/"failed"/"ac52.json")["traceback"]==primary
    assert calls==[False,False,False]
    assert not (queue.root/"done"/"ac52.json").exists()


def test_N43_diagnostic_cache_reuses_actual_aggregate_without_old_measure_or_token_claim():
    now=[10.0]
    state=SimpleNamespace(reference_build=True,request={"cache_stat_period_seconds":30})
    collector=first.FiniteObservations(state,clock=lambda:now[0])
    aggregate=dict(utc="TESTONLY actual aggregate sample time",started_monotonic_seconds=9.8,
        roots=[dict(role="cache",path="Q:/TESTONLY/cache",logical_bytes=123)],observed_scan_anomalies=[])
    def observe_reserve(*args,**kwargs):collector.last_aggregate=deepcopy(aggregate)
    collector.observe_reserve=observe_reserve
    def forbidden(*args,**kwargs):raise AssertionError("N43 diagnostics must not invoke original cache handle walk")
    lease=SimpleNamespace(policy=SimpleNamespace(root="Q:/TESTONLY/cache"),measure=forbidden,
        ledger=SimpleNamespace(locked=forbidden))
    collector.observe(SimpleNamespace(lease=lease),{"active_processes":2})
    assert collector.last["logical_bytes"]==123
    assert collector.last["source_aggregate_monotonic_seconds"]==9.8
    assert collector.last["actual_sample_age_seconds"]==pytest.approx(.2)
    assert collector.last["per_token_accounting_provided"] is False
    assert collector.last["ordinary_cache_ledger_measurement"] is False
    assert "active_scope_bytes" not in collector.last
    assert collector.sampled_cache_logical_high_water_bytes==123


def test_live_snapshot_refuses_actual_hardlink(monkeypatch,tmp_path):
    import os
    from darsia_cache import capacity as cache_capacity
    path=tmp_path/"TESTONLY-owned";path.mkdir()
    original=path/"one";original.write_bytes(b"abc")
    os.link(original,path/"same-inode")
    assert original.lstat().st_nlink==2
    info=path.lstat()
    state=SimpleNamespace(reference_administration=dict(entry_limit=100,host_policy="main-q87-fixed500",
        owned_roots=[dict(role="cache",path=str(path),identity={"TESTONLY":"preaccepted separately"},
            stat_identity=first._reference_stat_identity(info),volume_id=str(info.st_dev))]),
        reference_transport_observation=None)
    monkeypatch.setattr(cache_capacity,"volume_info",lambda root:dict(volume_id=str(info.st_dev),free_bytes=10**12))
    monkeypatch.setattr(first,"reference_writer_coverage",lambda *args:[])
    with pytest.raises(CachePolicyError,match="file type/hardlink refused"):
        first.reference_scope_snapshot(state,object())



def test_typed_authority_callback_is_used_only_for_exact_answers_role():
    state,value,old_read=administration_TESTONLY();events=[]
    def read(binding):
        assert binding!=value["answers_source"]
        events.append(("object",binding));return old_read(binding)
    def authorities(binding):
        assert binding==value["answers_source"]
        events.append(("authority",binding));return old_read(binding)
    assert first.checked_reference_administration(value,state,read=read,
        read_authorities=authorities)==value
    assert events==[("authority",value["answers_source"]),("object",value["accepted_plan"])]


def test_direct_old_profile_never_uses_typed_authority_callback():
    value={"format":"darsia-cache-import-profile-v1","TESTONLY":"original object"}
    def forbidden(binding):raise AssertionError("Unrelated ordinary profile used authority reader")
    assert first.resolve_profile(pin_TESTONLY("old.json"),"import",lambda binding:value,
        read_authorities=forbidden) is value
