"""Tiny allocation/reservation metadata; no science, arrays, NAS or worker run."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import pytest
from darsia.fluidflower.pipeline import capacity as cap
from darsia.fluidflower.pipeline.store_runtime import RunUnavailable
from darsia.fluidflower import job_filesystem as fs
from darsia.fluidflower.store_map import capacity_settings,StoreMapError


def native_layout():
    """Explicit tiny metadata fixture; no native arrays or actual profile."""
    return dict(format="darsia-native-phase-n1-measured-layout-v1",pipeline="native-phase-n1",
        precision="float32",native_shape_hw=[2,4],phase_shape_hw=[3,6],native_shape_chw=[3,2,4],
        feature_rgb_dtype="float32",source_box_xywh=[1,1,4,2],
        native_configuration_sha256="a"*64,phase_configuration_sha256="b"*64)


def native_metadata(tmp_path):
    """The original native JSON roles, all synthetic and owned by this test."""
    import hashlib,json
    tmp_path.mkdir(parents=True,exist_ok=True)
    rows=[]
    def put(use,value):
        path=tmp_path/(use+".json");raw=json.dumps(value,sort_keys=True).encode();path.write_bytes(raw)
        bound=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
        rows.append(SimpleNamespace(path=path,reference=dict(root_alias="series",relative_path=path.name,
            sha256=bound["sha256"],bytes=len(raw)),uses=(use,)))
        return bound
    native=dict(photometry=dict(native_shape_chw=[3,2,4],native_shape_hw=[2,4]),
        source=dict(native_shape_hw=[2,4]),extension=dict(native_shape_hw=[2,4]),support=dict(native_shape_hw=[2,4]))
    phase=dict(parameters=dict(geometry=dict(native_shape_hw=[2,4],source_box_xywh=[1,1,4,2]),
        scientific=dict(native_shape_hw=[2,4]),features=dict(alignment=dict(rgb_dtype="float32"))))
    config=dict(format="darsia-native-phase-n1-run-v1",run="ac52",
        native=dict(configuration=put("native.configuration",native)),
        phase=dict(configuration=put("phase.configuration",phase)),frames=[])
    for case in ["ac52_f1","ac52_f2"]:
        config["frames"].append(dict(case_id=case,native_task=dict(kind="rig",
            geometry=put("frame."+case+".native_geometry",dict(native_shape_hw=[2,4]))),
            geometry=put("frame."+case+".rig_geometry",dict(source_native_shape_hw=[2,4],phase_shape_hw=[3,6])),
            mass_spec=dict(original_domain=dict(native_shape_hw=[2,4]),photo_top=dict(shape_hw=[3,6]))))
    layout=native_layout();layout.update(native_configuration_sha256=config["native"]["configuration"]["sha256"],
        phase_configuration_sha256=config["phase"]["configuration"]["sha256"])
    return config,tuple(rows),layout


def measured():
    ids=["ac52_f1","ac52_f2"]
    task=dict(run="ac52",pipeline="native-phase-n1",precision="float32",ordered_case_ids=ids,commit="a"*40,attempt_id="1"*32)
    config=dict(run="ac52",precision="float32",frames=[dict(case_id=x) for x in ids])
    profile=dict(format=cap.PROFILE_FORMAT,pipeline=task["pipeline"],precision="float32",measured_commit="a"*40,measured_case_ids=ids)
    profile["measured_native_layout"]=native_layout()
    inventory=[dict(relative_path=".attempts/old/frames/ac52_f1/products.npz",bytes=100),
               dict(relative_path=".attempts/old/frames/ac52_f2/products.npz",bytes=140),dict(relative_path="manifest.json",bytes=10)]
    measurement=dict(inventory=inventory,total_logical_bytes=250)
    completion=dict(complete=True,passed=True,actual_case_count=2,actual_commit="a"*40,precision="float32")
    return task,config,profile,measurement,completion


def estimate(values=None,**kw):
    task,config,profile,measurement,completion=values or measured()
    kw.setdefault("requested_native_layout",deepcopy(profile.get("measured_native_layout")))
    return cap.estimate_run(task,config,profile,measurement=measurement,completion=completion,**kw)


def test_native_uses_observed_total_divided_by_actual_case_count():
    value=estimate()
    assert value.measured_native_bytes_per_frame==125
    assert value.native_expected_bytes==250 and value.coarse_uncompressed_bytes==0
    assert value.header_metadata_copy_estimated_bytes==0 and value.transient_estimated_bytes==140
    assert value.needed()==250+140+40*1024**3
    assert "expected space" in value.model and "no assumed compression" in value.model


def bound_output_profile_TESTONLY():
    from darsia_cache.first_qualification import BOUND_JOB_PROFILE
    task, config, profile, measurement, completion = measured()
    ids = ["ac52_f" + str(index) for index in range(1, 40)]
    task.update(backend="cpu", ordered_case_ids=ids,
        configuration=dict(root_alias="series", relative_path="run.json", sha256="b"*64, bytes=7))
    config["frames"] = [dict(case_id=case) for case in ids]
    profile["measured_case_ids"] = ids
    writers = [dict(relative_path=path, sha256="c"*64) for path in (
        "src/darsia/fluidflower/mass/n1_pipeline.py", "src/darsia/fluidflower/mass/n1_statistics.py",
        "src/darsia/fluidflower/pipeline/series.py", "src/darsia/fluidflower/native/sidecar.py",
        "src/darsia/fluidflower/pipeline/runner.py")]
    inventory = [dict(relative_path=directory+"/"+case+"/"+filename, bytes=size)
        for case in ids for directory,filename,size in (
            ("preparation", "native_log.npy", 101), ("frames", "products.npz", 53))]
    inventory.append(dict(relative_path="manifest.json", bytes=7))
    measurement.update(inventory=inventory, total_logical_bytes=sum(row["bytes"] for row in inventory))
    completion["actual_case_count"] = 39
    actual = dict(profile, writers=writers, measurement=measurement, completion=completion)
    wrapped = dict(format=BOUND_JOB_PROFILE,
        scope={key:task[key] for key in ("commit", "pipeline", "backend", "precision", "run", "ordered_case_ids")},
        configuration=task["configuration"], result_profile=actual)
    return task, config, wrapped


def test_bound_output_reader_uses_original_estimate_arithmetic_for_all_retained_stages(monkeypatch):
    task, config, wrapped = bound_output_profile_TESTONLY(); actual = wrapped["result_profile"]
    checked = []
    monkeypatch.setattr(cap, "_profile_writer_identity", lambda repository, measured, current, row:
        checked.append((measured, current, row["relative_path"])))
    monkeypatch.setattr(cap, "_native_layout_for_task", lambda *args: native_layout())
    binding = dict(path="TESTONLY/provider.json", sha256="a"*64, size_bytes=1)
    result = cap._bound_result_estimate(None, "FF_AC", task, config, wrapped, binding,
        repository="TESTONLY", metadata_rows=None)
    expected = cap.estimate_run(task, config, actual, measurement=actual["measurement"],
        completion=actual["completion"], requested_native_layout=native_layout(), profile_identity=binding)
    assert result == expected
    assert len(checked) == 5 and all(a == b == task["commit"] for a,b,_ in checked)
    assert result.measured_native_bytes_per_frame == (actual["measurement"]["total_logical_bytes"]+38)//39


@pytest.mark.parametrize("change", [
    # Another run's own configuration is covered (N188 D2): test_bound_profile_other_runs.
    lambda t,c,p:c.update(storage={"TESTONLY":"not measured by this profile"}),
    lambda t,c,p:c.update(gas_probability_storage={"TESTONLY":"not measured"}),
    lambda t,c,p:p["result_profile"]["writers"].pop(),
    lambda t,c,p:p["result_profile"].update(measured_commit="e"*40),
    lambda t,c,p:p["result_profile"]["completion"].update(passed=False),
])
def test_bound_output_profile_refuses_unqualified_config_source_layout_or_completion(change, monkeypatch):
    task, config, wrapped = bound_output_profile_TESTONLY()
    # The immutable task and resulting profile are separate parsed documents.
    wrapped = deepcopy(wrapped); change(task, config, wrapped)
    monkeypatch.setattr(cap, "_profile_writer_identity", lambda *args: None)
    monkeypatch.setattr(cap, "_native_layout_for_task", lambda *args: native_layout())
    with pytest.raises(ValueError):
        cap._bound_result_estimate(None, "FF_AC", task, config, wrapped, {},
            repository="TESTONLY", metadata_rows=None)


@pytest.mark.parametrize("field,value",[("precision","float64"),("pipeline","native-phase-statistics"),("measured_commit","not-a-commit")])
def test_profile_applicability_cannot_be_assumed(field,value):
    data=measured();data[2][field]=value
    with pytest.raises(ValueError):estimate(data)


@pytest.mark.parametrize("field,value",[("passed",False),("complete",False),("actual_case_count",True),
    ("actual_case_count",3),("actual_commit","b"*40),("precision","float64")])
def test_incomplete_or_historical_gate_cannot_be_relabelled_current(field,value):
    data=measured();data[4][field]=value
    with pytest.raises(ValueError):estimate(data)


@pytest.mark.parametrize("change",["total","duplicate","missing_case","foreign_path"])
def test_observed_inventory_requires_exact_distinct_case_coverage(change):
    data=measured();m=data[3]
    if change=="total":m["total_logical_bytes"]+=1
    elif change=="duplicate":m["inventory"].append(dict(m["inventory"][0]));m["total_logical_bytes"]+=100
    elif change=="missing_case":m["inventory"][1]["relative_path"]="other/products.npz"
    else:m["inventory"][0]["relative_path"]="../foreign/products.npz"
    with pytest.raises(ValueError):estimate(data)


def test_coarse_keeps_native_and_adds_grid_bytes_without_compression_gain():
    data=measured();data[1]["storage"]=dict(format="darsia-storage-output-request-v1",resolution_mm=1,
        gas_score_encoding="uint8-la-q255-v1",extent_m=[0,.004,0,.002])
    layout=dict(format="darsia-storage-frame-v1",complete=True,precision="float32",
        coarse=dict(grid=dict(native_shape=[2,4]),arrays={"concentration_aqueous":dict(dtype="<f4")}),files=[])
    value=estimate(data,layout=layout,layout_bytes=50)
    assert value.native_expected_bytes==250
    assert value.coarse_uncompressed_bytes==(2*4*4+2*(2*4+1))*2
    assert value.header_metadata_copy_estimated_bytes>100
    assert value.transient_estimated_bytes==140+(value.coarse_uncompressed_bytes+value.header_metadata_copy_estimated_bytes)//2


def test_requested_storage_without_layout_is_explicitly_unavailable():
    data=measured();data[1]["storage"]=dict(format="darsia-storage-output-request-v1",gas_score_encoding="uint8-la-q255-v1",resolution_mm="native")
    with pytest.raises(ValueError,match="layout"):estimate(data)


def test_N51_results_threshold_is_own_estimate_plus_reserve_other_history_informative():
    value = estimate()
    assert value.needed(other_remaining_bytes=31, reserve_bytes=40) == 250+140+31+40
    needed = 250+140+40
    result = cap.report(value, {"free_bytes": needed}, 31, 40)
    assert result["sufficient"] is True and result["needed_bytes"] == needed
    assert result["other_active_remaining_bytes"] == 31
    assert result["other_active_remaining_included_in_results_estimate"] is False
    assert result["results_space_policy"] == "estimate, not coordinated across hosts"
    assert cap.report(value, {"free_bytes": needed-1}, 31, 40)["sufficient"] is False


def settings_doc(tmp_path):
    return dict(capacity=dict(state_directory=str(tmp_path/"capacity"),profiles={"FF_AC":dict(path=str(tmp_path/"profile.json"),sha256="a"*64,size_bytes=1)}))


def test_settings_default_and_explicit_reserve(tmp_path):
    doc=settings_doc(tmp_path)
    assert capacity_settings(doc)["reserve_bytes"]==40*1024**3
    doc["capacity"]["reserve_bytes"]=7
    assert capacity_settings(doc)["reserve_bytes"]==7
    assert capacity_settings({}) is None


@pytest.mark.parametrize("value",[-1,True,1.5,"40"])
def test_invalid_reserve_is_not_silently_coerced(tmp_path,value):
    doc=settings_doc(tmp_path);doc["capacity"]["reserve_bytes"]=value
    with pytest.raises(StoreMapError):capacity_settings(doc)


def test_capacity_state_cannot_be_a_nas_lock_folder(tmp_path):
    doc=settings_doc(tmp_path);doc["capacity"]["state_directory"]=r"\\not-a-real-host\share\locks"
    with pytest.raises(StoreMapError,match="local"):capacity_settings(doc)


def test_missing_profile_has_no_universal_default(tmp_path):
    stores=SimpleNamespace(document={},machine_id="main")
    with pytest.raises(RunUnavailable,match="explicit applicable"):cap.settings(stores,"FF_AC")


def ledger_fixture(tmp_path,monkeypatch,*,free=10000):
    roots=dict(series=str(tmp_path/"series"),results=str(tmp_path/"results"))
    stores=SimpleNamespace(document=settings_doc(tmp_path),machine_id="main",roots=lambda _:roots)
    monkeypatch.setattr(fs,"inspect_local_file",lambda p, **kwargs:fs.LocalFileAdmission(True,"fixture-fixed","synthetic","fixture",str(tmp_path)))
    state=dict(volume_id="fixture-volume",volume_anchor="fixture",free_bytes=free,checked_container=str(tmp_path))
    common=dict(host="physical-fixture",service_account_sid="S-1-5-21-100",common_root=str(tmp_path/"common-host-bindings"))
    ledger=cap.CapacityLedger(stores,"FF_AC",roots["results"],volume_provider=lambda _:dict(state),context_provider=lambda:dict(common))
    return ledger,stores,state


def claim_fixture(job="one",run="ac52",token="1"*32):
    task=measured()[0];task.update(job_id=job,run=run,attempt_id=token)
    return SimpleNamespace(task=task,token=token)


def owner(pid=123,machine="main-cpu"):
    return dict(machine=machine,pid=pid,process_create_time="12345.125")


def test_two_services_and_jobs_share_volume_and_count_once(tmp_path,monkeypatch):
    a,stores,state=ledger_fixture(tmp_path,monkeypatch)
    b=cap.CapacityLedger(stores,"FF_AC",tmp_path/"other-result-directory",volume_provider=lambda _:dict(state),context_provider=a.context_provider)
    assert a.key==b.key
    with a.locked():r1=a.bind(tmp_path/"job1",claim_fixture(),owner(),estimate())
    with b.locked():
        assert b.remaining()==390
        r2=b.bind(tmp_path/"job2",claim_fixture(job="two",token="2"*32),owner(124,"main-cuda"),estimate())
        assert b.remaining()==780 and b.remaining(exclude_job=tmp_path/"job1")==390
        b.progress(r2,owner(124,"main-cuda"),dict(state="frame_completed",complete=True,run="ac52",case_id="ac52_f1"))
        assert b.remaining()==655
    assert r1!=r2 and (r1/"reserved.json").exists()


def test_own_release_preserves_all_history(tmp_path,monkeypatch):
    ledger,_,_=ledger_fixture(tmp_path,monkeypatch)
    with ledger.locked():
        path=ledger.bind(tmp_path/"job",claim_fixture(),owner(),estimate())
        raw=(path/"reserved.json").read_bytes()
        ledger.progress(path,owner(),dict(state="frame_completed",complete=True,run="ac52",case_id="ac52_f1"))
        ledger.release(path,owner());ended=(path/"released.json").read_bytes()
        ledger.release(path,owner())
        assert ledger.remaining()==0 and (path/"reserved.json").read_bytes()==raw
        assert (path/"released.json").read_bytes()==ended and (path/"p-00000001.json").exists()


@pytest.mark.parametrize("change",["owner","case","not_committed"])
def test_progress_cannot_fabricate_remaining_capacity(tmp_path,monkeypatch,change):
    ledger,_,_=ledger_fixture(tmp_path,monkeypatch)
    with ledger.locked():
        path=ledger.bind(tmp_path/"job",claim_fixture(),owner(),estimate())
        event=dict(state="frame_completed",complete=True,run="ac52",case_id="ac52_f1");actual=owner()
        if change=="owner":actual=owner(999)
        elif change=="case":event["case_id"]="not-in-this-task"
        else:event["complete"]=False
        with pytest.raises(ValueError):ledger.progress(path,actual,event)
        assert ledger.remaining()==390 and not list(path.glob("p-*.json"))


def test_supervisor_retirement_binds_owner_epoch_and_token(tmp_path,monkeypatch):
    ledger,_,_=ledger_fixture(tmp_path,monkeypatch)
    with ledger.locked():
        path=ledger.bind(tmp_path/"job",claim_fixture(),owner(),estimate());supervisor=owner(998)
        retirement=dict(complete=True,all_absent=True,worker_os_references_closed=True,
            supervisor=supervisor,run="ac52",attempt_id="1"*32,identities=[owner()])
        wrong=deepcopy(retirement);wrong["identities"][0]["process_create_time"]="12345.126"
        with pytest.raises(ValueError,match="retirement"):ledger.release(path,supervisor,retirement=wrong)
        assert not (path/"released.json").exists()
        ledger.release(path,supervisor,retirement=retirement)
        assert ledger.remaining()==0


def test_fresh_free_space_and_changed_volume(tmp_path,monkeypatch):
    ledger,_,state=ledger_fixture(tmp_path,monkeypatch)
    assert ledger.fresh_volume()["free_bytes"]==10000
    state["free_bytes"]=3
    assert ledger.fresh_volume()["free_bytes"]==3
    state["volume_id"]="different"
    with pytest.raises(RunUnavailable,match="changed"):ledger.fresh_volume()


def test_original_input_identity_errors_keep_original_behavior():
    runtime=SimpleNamespace(capacity_refusals={},availability=lambda task:(_ for _ in ()).throw(ValueError("original identity STOP")))
    context=cap.ClaimCapacity(runtime,owner());context.ledger=object()
    with pytest.raises(ValueError,match="original identity STOP") as found:context.availability(dict(run="ac52"))
    assert type(found.value) is ValueError


def test_foreign_actor_cannot_release_an_original_reservation(tmp_path,monkeypatch):
    ledger,_,_=ledger_fixture(tmp_path,monkeypatch)
    with ledger.locked():
        path=ledger.bind(tmp_path/"job",claim_fixture(),owner(),estimate())
        with pytest.raises(ValueError,match="original owner"):ledger.release(path,owner(999))
        assert ledger.remaining()==390 and not (path/"released.json").exists()


def test_changed_exact_metadata_sha_refuses_without_a_measurement_fallback(tmp_path):
    import hashlib
    path=tmp_path/"profile.json";path.write_bytes(b'{"literal":1}')
    bound=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size_bytes=path.stat().st_size)
    assert cap.read_bound(bound)[0]=={"literal":1}
    path.write_bytes(b'{"literal":2}')
    with pytest.raises(ValueError,match="SHA/size/version"):cap.read_bound(bound)


def test_profile_sha_requires_a_literal_string_not_a_numeric_lookalike():
    with pytest.raises(ValueError,match="SHA/size"):cap._bound(dict(path="/declared.json",sha256=int("1"*64),size_bytes=1))


def test_whole_job_uses_same_estimates_and_one_reserve(tmp_path,monkeypatch):
    value=estimate();task,config,*_=measured()
    task2=deepcopy(task);task2.update(run="ac53",ordered_case_ids=["ac53_f1","ac53_f2"])
    config2=deepcopy(config);config2.update(run="ac53",frames=[dict(case_id=key) for key in task2["ordered_case_ids"]])
    seen=[]
    def load(stores,series,actual_task,actual_config,**kw):
        seen.append((actual_task,actual_config))
        return replace(value,run=actual_task["run"])
    monkeypatch.setattr(cap,"load_estimate",load)
    monkeypatch.setattr(fs,"inspect_results_root",lambda _:SimpleNamespace(allowed=True,protocol="smb",local=False))
    monkeypatch.setattr(cap,"CapacityLedger",lambda *a:SimpleNamespace(
        reserve=40,remaining=lambda **kw:31,fresh_volume=lambda:dict(free_bytes=9999),remote_scope=None))
    stores=SimpleNamespace(machine_id="main",roots=lambda _:dict(results=str(tmp_path/"results")))
    before=list(tmp_path.iterdir())
    report=cap.whole_job_report(stores,"FF_AC",[task,task2],[config,config2],repository=tmp_path)
    assert report["needed_bytes"]==2*(250+140)+40
    assert report["other_active_remaining_bytes"]==31
    assert report["results_space_policy"]=="estimate, not coordinated across hosts"
    assert len(seen)==2 and report["results_local"] is False and report["results_protocol"]=="smb"
    assert list(tmp_path.iterdir())==before and "native" not in str(report.get("locality",""))


def test_configuration_for_another_run_cannot_reuse_profile_case_order():
    values=measured();values[1]["run"]="another-run"
    with pytest.raises(ValueError,match="differs"):estimate(values)


@pytest.mark.parametrize("changed_executor",[False,True])
def test_runner_reporting_provenance_does_not_waive_executor_identity(tmp_path,monkeypatch,changed_executor):
    import ast,hashlib
    functions=["_phase_outputs","_inventory","validate_frame_artifacts","_restore_predecessor","scalar_table","run_series","_run_series"]
    original=("\n".join("def "+name+"(): return 1" for name in functions)+"\ndef prepare_job(): return 1\n").encode()
    current=original.replace(b"def prepare_job(): return 1",b"def prepare_job(): return 2")
    if changed_executor:current=current.replace(b"def _run_series(): return 1",b"def _run_series(): return 9")
    row=dict(relative_path="src/darsia/fluidflower/pipeline/runner.py",
        measured_sha256=hashlib.sha256(original).hexdigest(),current_sha256=hashlib.sha256(current).hexdigest(),
        functions={node.name:hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
                   for node in ast.parse(original).body if isinstance(node,ast.FunctionDef) and node.name in functions})
    monkeypatch.setattr(cap,"_writer_identity",lambda *args:None)  # The separate whole-file checker owns disk/Git admission.
    monkeypatch.setattr(cap.subprocess,"check_output",lambda args:original if args[-1].startswith("a"*40+":") else current)
    if changed_executor:
        with pytest.raises(ValueError,match="executor/output body changed"):cap._orchestration_identity(tmp_path,"a"*40,"b"*40,row)
    else:cap._orchestration_identity(tmp_path,"a"*40,"b"*40,row)


def test_writer_current_disk_and_canonical_git_sha_are_both_required(tmp_path,monkeypatch):
    import hashlib
    path=tmp_path/"src/darsia/fluidflower/writer.py";path.parent.mkdir(parents=True)
    original=b"# synthetic writer identity only\n";path.write_bytes(original)
    monkeypatch.setattr(cap.subprocess,"check_output",lambda args:original)
    row=dict(relative_path="src/darsia/fluidflower/writer.py",sha256=hashlib.sha256(original).hexdigest())
    cap._writer_identity(tmp_path,"a"*40,row)
    path.write_bytes(original+b"# changed\n")
    with pytest.raises(ValueError,match="identity differs"):cap._writer_identity(tmp_path,"a"*40,row)


def test_public_prepare_reports_machine_estimate_without_persisting_machine_paths(tmp_path,monkeypatch):
    import json
    from .test_alias_job_metadata import assembly_fixture
    from darsia.fluidflower.pipeline import runner
    root,manifest,source,config,value=assembly_fixture(tmp_path,monkeypatch)
    destination=tmp_path/"prepared-job"
    machine=SimpleNamespace(machine_id="main")
    def project(stores,series,tasks,configs,**kw):
        assert stores is machine and not destination.exists()
        assert [task["run"] for task in tasks]==["ac52"] and configs==[config]
        return dict(machine_id="main",results_root="/machine-private/result-root",needed_bytes=1234,free_bytes=1233,sufficient=False)
    monkeypatch.setattr(cap,"whole_job_report",project)
    result=runner.prepare_job(source["path"],destination,expected_sha256=source["sha256"],series_manifest=manifest,capacity_stores=[machine])
    assert result["capacity_reports"][0]["needed_bytes"]==1234
    assert result["capacity_reports"][0]["free_bytes"]==1233
    assert result["capacity_reports"][0]["sufficient"] is False
    for path in [destination/"job.json",destination/"queue/ac52.json"]:
        text=path.read_bytes().decode()
        assert "capacity_reports" not in text and "/machine-private" not in text


def test_one_host_volume_refuses_split_state_directories(tmp_path,monkeypatch):
    first,stores,state=ledger_fixture(tmp_path,monkeypatch)
    alternate=deepcopy(stores.document);alternate["capacity"]["state_directory"]=str(tmp_path/"split-state")
    other_stores=SimpleNamespace(document=alternate,machine_id=stores.machine_id,roots=stores.roots)
    second=cap.CapacityLedger(other_stores,"FF_AC",tmp_path/"results",volume_provider=lambda _:dict(state),context_provider=first.context_provider)
    assert first.key==second.key and first.binding_path==second.binding_path
    with first.locked():first.bind(tmp_path/"job",claim_fixture(),owner(),estimate())
    before=first.binding_path.read_bytes()
    with pytest.raises(RunUnavailable,match="another configured directory"):
        with second.locked():pytest.fail("A split ledger must never be admitted")
    assert not (tmp_path/"split-state").exists()
    assert first.binding_path.read_bytes()==before and first.remaining()==390


def test_other_windows_service_account_cannot_create_a_second_pool(tmp_path,monkeypatch):
    first,stores,state=ledger_fixture(tmp_path,monkeypatch)
    facts=first.context_provider();facts["service_account_sid"]="S-1-5-21-200"
    other=cap.CapacityLedger(stores,"FF_AC",tmp_path/"results",volume_provider=lambda _:dict(state),context_provider=lambda:dict(facts))
    assert first.key==other.key
    with first.locked():pass
    before=first.binding_path.read_bytes()
    with pytest.raises(RunUnavailable,match="service-account SID"):
        with other.locked():pytest.fail("Another account must not bypass the bound pool")
    assert first.binding_path.read_bytes()==before


def test_logical_map_cpu_cuda_aliases_do_not_split_physical_pool(tmp_path,monkeypatch):
    first,stores,state=ledger_fixture(tmp_path,monkeypatch)
    alias=SimpleNamespace(document=stores.document,machine_id="another-map-alias",roots=stores.roots)
    other=cap.CapacityLedger(alias,"FF_AC",tmp_path/"results",volume_provider=lambda _:dict(state),context_provider=first.context_provider)
    assert first.key==other.key
    with first.locked():first.bind(tmp_path/"job1",claim_fixture(),owner(),estimate())
    with other.locked():
        assert other.remaining()==390
        other.bind(tmp_path/"job2",claim_fixture(job="two",token="2"*32),owner(124,"alias-cuda"),estimate())
        assert other.remaining()==780
    assert first.remaining()==780


def test_inaccessible_machine_common_binding_refuses_without_private_fallback(tmp_path,monkeypatch):
    ledger,stores,state=ledger_fixture(tmp_path,monkeypatch)
    original=cap._read_local
    with ledger.locked():pass
    def forbidden(path):
        if path==ledger.binding_path:raise PermissionError("literal synthetic common-binding access denied")
        return original(path)
    monkeypatch.setattr(cap,"_read_local",forbidden)
    with pytest.raises(PermissionError,match="access denied"):
        with ledger.locked():pytest.fail("No per-user fallback may be created")


@pytest.mark.parametrize("writer",["src/darsia/fluidflower/pipeline/series.py","src/darsia/fluidflower/native/sidecar.py"])
@pytest.mark.parametrize("defect",["missing","changed"])
def test_actual_external_native_writer_is_mandatory_and_fresh(tmp_path,monkeypatch,writer,defect):
    import ast,hashlib,json
    task,config,profile,measurement,completion=measured()
    config,metadata_rows,profile["measured_native_layout"]=native_metadata(tmp_path/"native-layout")
    from darsia.fluidflower.pipeline import store_runtime
    monkeypatch.setattr(store_runtime,"resolve_catalog",lambda *a:metadata_rows)
    repo=tmp_path/"repo"
    required=["src/darsia/fluidflower/mass/n1_pipeline.py","src/darsia/fluidflower/mass/n1_statistics.py",
        "src/darsia/fluidflower/pipeline/series.py","src/darsia/fluidflower/native/sidecar.py"]
    source={key:b"# Tiny synthetic identity fixture, never scientific code.\n" for key in required}
    functions=["_phase_outputs","_inventory","validate_frame_artifacts","_restore_predecessor","scalar_table","run_series","_run_series"]
    runner="src/darsia/fluidflower/pipeline/runner.py"
    source[runner]=("\n".join("def "+key+"(): return None" for key in functions)+"\n").encode()
    for key,raw in source.items():
        path=repo/key;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    def git(args):
        commit,relative=args[-1].split(":",1)
        assert commit=="a"*40 and relative in source
        return source[relative]
    monkeypatch.setattr(cap.subprocess,"check_output",git)
    def put(path,value):
        raw=json.dumps(value,sort_keys=True).encode();path.write_bytes(raw)
        return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
    row=dict(relative_path=runner,measured_sha256=hashlib.sha256(source[runner]).hexdigest(),current_sha256=hashlib.sha256(source[runner]).hexdigest(),
        functions={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in ast.parse(source[runner]).body})
    profile.update(measurement=put(tmp_path/"measurement.json",measurement),completion=put(tmp_path/"completion.json",completion),
        writers=[dict(relative_path=key,sha256=hashlib.sha256(source[key]).hexdigest()) for key in required],orchestration=[row],layout_receipt=None)
    document=settings_doc(tmp_path);document["capacity"]["profiles"]["FF_AC"]=put(tmp_path/"profile.json",profile)
    stores=SimpleNamespace(document=document)
    assert cap.load_estimate(stores,"FF_AC",task,config,repository=repo).native_expected_bytes==250
    if defect=="missing":
        profile["writers"]=[r for r in profile["writers"] if r["relative_path"]!=writer]
        document["capacity"]["profiles"]["FF_AC"]=put(tmp_path/"profile.json",profile)
        message="omits an actual applicable output writer"
    else:
        (repo/writer).write_bytes(source[writer]+b"# literal source changed after valid profile SHA\n")
        message="writer identity differs"
    with pytest.raises(ValueError,match=message):cap.load_estimate(stores,"FF_AC",task,config,repository=repo)


@pytest.mark.parametrize("free,cache_refuses,accepted", [(430, False, True), (429, False, False), (430, True, False)])
def test_N51_actual_claim_keeps_basic_free_boundary_and_original_cache_checks(tmp_path, monkeypatch, free, cache_refuses, accepted):
    ledger, stores, state = ledger_fixture(tmp_path, monkeypatch, free=free)
    ledger.reserve = 40
    with ledger.locked():
        ledger.bind(tmp_path / "other-job", claim_fixture(job="other", token="2" * 32), owner(124), estimate())
    task = claim_fixture().task
    runtime = SimpleNamespace(stores=stores, job=dict(series_id="FF_AC"),
        capacity_refusals={}, availability=lambda actual: (), capacities={})
    check = cap.ClaimCapacity(runtime, owner()); check.ledger = ledger
    observed = []
    class CacheBoundary:
        def prepare(self, actual, value): observed.append(("prepare", actual, value))
        def acquire(self): observed.append(("acquire",))
        def check(self, actual, value, other):
            observed.append(("check", other))
            if cache_refuses: raise RunUnavailable("original cache admission refused")
            return dict(TEST_ONLY_ORIGINAL_CACHE_BOUNDARY=True)
        def unlock(self): pass
    check.cache = CacheBoundary()
    monkeypatch.setattr(cap, "configuration_for_task", lambda *args: {})
    monkeypatch.setattr(cap, "load_estimate", lambda *args, **kw: estimate())
    if accepted:
        try: assert check.availability(task) == ()
        finally: check._unlock()
    else:
        with pytest.raises(RunUnavailable, match="original cache admission refused" if cache_refuses else "leave this whole run pending"):
            check.availability(task)
        assert check.lock is None
    assert ("check", 390) in observed
    assert len(list(ledger.root.glob("r-*/reserved.json"))) == 1
