"""Ordinary synthetic metadata assembly/guards; no original executor runs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from darsia.fluidflower.pipeline import alias_job, series, job_queue, worker
from darsia.fluidflower.jobs_panel import JobsPanel
from darsia.fluidflower.jobs_metadata import JobSnapshotReader
from darsia.fluidflower.store_map import StoreMapError


def put(path, raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(raw)
    return {"path":str(path),"sha256":hashlib.sha256(raw).hexdigest(),"size_bytes":len(raw)}


def doc(path,value):
    return put(path,json.dumps(value,sort_keys=True).encode())


def assembly_fixture(tmp_path,monkeypatch):
    root=tmp_path/"source_series";root.mkdir()
    manifest=put(root/"series.toml",b'schema_version=1\nseries_id="Synthetic"\n[roots]\nseries="."\n')
    def data(key): return put(root/(key+".npy"),b"payload never decoded")
    def metadata(key): return doc(root/(key+".json"),{})
    full={"native_centre":{"log_linear_rgb":data("native_log"),"valid_mask":data("native_valid")},"ring_windows":[]}
    frame={"case_id":"ac52_f1","current":full,"baseline":deepcopy(full),
           **{key:data(key) for key in ("labels","classes","area","photo_top")},
           "original_domain":{"parent_labels_map":put(root/"original_domain.png", b"opaque source labels"),
                              "native_shape_hw":[2,2],"frame_index":0}}
    config={"format":"darsia-n1-run-v1","run":"ac52","frames":[frame],
        "calibration":metadata("calibration"),"physical_correction":metadata("correction"),"boxes":{},
        "shared":{"physical_reference":data("physical"),"floor_curve":data("floor"),"depth_points":[],
                  "reference_matrix":[],"clearwater":{"rgb":data("clear_rgb"),"valid":data("clear_valid")},
                  "optics":{"metadata":metadata("optics"),"color_paths":{},"signals":{},"flash":metadata("flash")}},
        "provenance":{"path":"C:/historical/not-consumed","sha256":"a"*64}}
    bound=doc(root/"config.json",config)
    source=put(root/"execution.toml",b'# inert original-admission fixture\n')
    value={"job":{"commit":"a"*40,"precision":"float32","calibration_sha256":"b"*64,
                  "environment_lock":{"path":"requirements.lock","sha256":"c"*64}},
           "job_id":"synthetic","runs":[{"run":"ac52","configuration":bound,"resources":{"backend":"cpu","threads":1}}]}
    # Scope: tests the J2 wiring below the original admitted-series boundary.
    # No configuration is actually admitted for numerical execution by this stub.
    monkeypatch.setattr(series,"read_series",lambda *a,**k:(deepcopy(value),[deepcopy(config)],dict(source)))
    monkeypatch.setattr(series,"require_unc",lambda _:None)
    monkeypatch.setattr(alias_job,"require_unc",lambda _:None)
    from darsia.fluidflower import job_filesystem
    monkeypatch.setattr(job_filesystem,"inspect_shared_folder",lambda path:
        job_filesystem.SharedFolderAdmission(True,"smb","authored unit transport","no real FS qualification"))
    return root,manifest,source,config,value


def test_all_portable_metadata_before_creation_and_input_bytes_unchanged(tmp_path,monkeypatch):
    root,manifest,source,_config,_value=assembly_fixture(tmp_path,monkeypatch)
    before={str(p):p.read_bytes() for p in root.iterdir() if p.is_file()}
    destination=tmp_path/"shared_job_fixture"
    result=alias_job.prepare_alias_job(source["path"],destination,expected_sha256=source["sha256"],series_manifest=manifest)
    assert result["complete"] and result["execution_ready"] and not result["numerical_calls_started"]
    job=json.loads((destination/"job.json").read_bytes())
    task=json.loads((destination/"queue/ac52.json").read_bytes())
    assert job["legacy_paths"] is False and task["legacy_paths"] is False
    assert job["publish"] == ["film", "lossless_export"]
    assert job["runtime_support"]==task["input_catalog"]["runtime_support"]=="darsia-role-io-runtime-v1"
    assert task["output_directory"]=="results/ac52"
    for value in (job,task):
        encoded=json.dumps(value)
        assert str(root) not in encoded and "historical/not-consumed" not in encoded
        assert "configuration" not in value or "path" not in value["configuration"]
    assert {str(p):p.read_bytes() for p in root.iterdir() if p.is_file()}==before
    snapshot=JobSnapshotReader()({"root":str(destination)})
    assert snapshot["committed"]==0 and snapshot["frames_per_minute"] is None and snapshot["eta_seconds"] is None


def test_unknown_actual_role_root_fails_before_any_job_mkdir(tmp_path,monkeypatch):
    root,manifest,source,config,_value=assembly_fixture(tmp_path,monkeypatch)
    config["frames"][0]["labels"]={"path":str(tmp_path/"unlisted"/"labels.png"),"sha256":"a"*64}
    destination=tmp_path/"must_not_exist"
    with pytest.raises(StoreMapError,match="No declared root"):
        alias_job.prepare_alias_job(source["path"],destination,expected_sha256=source["sha256"],series_manifest=manifest)
    assert not destination.exists()


def test_missing_manifest_never_falls_back_to_new_absolute_path_job(tmp_path,monkeypatch):
    from darsia.fluidflower import job_filesystem
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda _path:
        job_filesystem.SharedFolderAdmission(True, "smb", "isolated unit transport", "not a real FS qualification"))
    with pytest.raises(ValueError,match="explicit bound"):
        alias_job.prepare_alias_job(tmp_path/"plan",tmp_path/"never_created",
                                    expected_sha256="a"*64,series_manifest=None)
    assert not (tmp_path/"never_created").exists()


def test_alias_tasks_require_explicit_runtime_bindings_before_claim(tmp_path,monkeypatch):
    _root,manifest,source,_config,_value=assembly_fixture(tmp_path,monkeypatch)
    destination=tmp_path/"shared_job_fixture"
    alias_job.prepare_alias_job(source["path"],destination,expected_sha256=source["sha256"],series_manifest=manifest)
    unbound=job_queue.JobQueue(destination)
    with pytest.raises(RuntimeError,match="explicit physical store machine_id"):
        unbound.claim("Main","cpu",{"pid":1,"process_create_time":"1"})
    assert (destination/"queue/ac52.json").is_file()
    assert not list((destination/"claimed").glob("*/*.json"))
    queue=job_queue.JobQueue(destination,machine_id="Main")
    with pytest.raises(RuntimeError,match="requires availability"):
        queue.claim("Main","cpu",{"pid":1,"process_create_time":"1"})
    assert (destination/"queue/ac52.json").is_file()
    assert not list((destination/"claimed").glob("*/*.json"))


def test_j2_worker_start_refuses_before_git_or_any_executor(monkeypatch,tmp_path):
    from darsia.fluidflower import job_filesystem
    monkeypatch.setattr(worker,"require_unc",lambda _:None)
    monkeypatch.setattr(job_filesystem,"inspect_shared_folder",lambda _:job_filesystem.SharedFolderAdmission(True,"smb","fixture","fixture"))
    # Fixed old J2 wire fixture: it must never be promoted by opening/starting.
    pending=b'{"format":"darsia-alias-job-v1","format_version":1,"legacy_paths":false,"execution_ready":false,"execution_blocker":"Alias job metadata is prepared; J3 runtime integration is not yet available"}'
    monkeypatch.setattr(worker,"read_json",lambda _:json.loads(pending))
    monkeypatch.setattr(worker.subprocess,"check_output",lambda *_a,**_k:pytest.fail("J2 runtime must refuse before original Git checks"))
    with pytest.raises(RuntimeError,match="J3 runtime"):
        worker.verify_startup(tmp_path)


def test_explicit_ui_start_is_blocked_before_control_or_launcher(qtbot,monkeypatch):
    panel=JobsPanel(local_machine="Main")
    qtbot.addWidget(panel)
    panel.current={"root":"not-a-real-job","job_facts":{"format":alias_job.JOB_FORMAT,"execution_blocker":"J3 pending"}}
    errors=[];monkeypatch.setattr(panel,"_error",errors.append)
    monkeypatch.setattr(panel,"_read_control_snapshot",lambda *_:pytest.fail("No control/launcher lookup for J2 pending job"))
    assert not panel._start_allowed("Main") and errors==["J3 pending"]


def test_portable_ui_binding_uses_named_fields_without_historical_path_copies(tmp_path,monkeypatch):
    root,manifest,source,_config,value=assembly_fixture(tmp_path,monkeypatch)
    raw=put(root/"camera.jpg",b"camera payload never decoded")
    proof=doc(root/"protocol.json",{})
    binding={"execution_plan":source,"series_id":"Synthetic","series_root":str(root),
        "runs":[{"run":"ac52","manifest":manifest,"source_metadata":[proof],
                 "configuration":value["runs"][0]["configuration"],
                 "cases":[{"case_id":"ac52_f1","elapsed_h":0.1,"b4_input_index":1,
                           "source_path":raw["path"],"original_camera_source":raw}]}]}
    result=alias_job.portable_selection_binding(binding,series_manifest=manifest)
    assert result["format"].endswith("-v2")
    assert str(root) not in json.dumps(result) and "source_path" not in result["runs"][0]["cases"][0]
    assert result["runs"][0]["cases"][0]["original_camera_source"]["sha256"]==raw["sha256"]


def test_original_series_validation_can_use_explicit_rooted_metadata_reader(tmp_path):
    source={"source_sha256":"a"*64,"previous_source_sha256":"b"*64,"run":"ac52",
            "selected":True,"selected_index":1,"elapsed_h":0.1}
    calibration={"path":"unchanged","sha256":"c"*64}
    config={"format":series.RUN_FORMAT,"run":"ac52",
        "frames":[{"case_id":"ac52_f1","frame_index":1,"elapsed_h":0.1,"source":source,
                   "native_task":{"kind":"rig"},"geometry":{},"phase_roles":{"area":{"role":"selected","factor":1}},"mass_spec":{}}],
        "native":{},"phase":{"initial_previous":{"source":{"source_sha256":"b"*64}}},
        "mass":{"calibration":calibration,"template":{"calibration":calibration}},
        "table_fields":["total_mass_g"],"history_policy":{},"selection_index_field":"selected_index"}
    selection={"complete":True,"runs":[{"run":"ac52","ordered_case_ids":["ac52_f1"]}]}
    boundconfig=doc(tmp_path/"config.json",config);boundselection=doc(tmp_path/"selection.json",selection)
    text=('format="darsia-series-v1"\nadopted_for_execution=true\n'
          '[selection]\npath='+json.dumps(boundselection["path"])+'\nsha256="'+boundselection["sha256"]+'"\n'
          '[job]\ncalibration_sha256="'+"c"*64+'"\n'
          '[[runs]]\nrun="ac52"\n[runs.configuration]\npath='+json.dumps(boundconfig["path"])+'\nsha256="'+boundconfig["sha256"]+'"\n')
    bound=put(tmp_path/"execution.toml",text.encode())
    seen=[]
    class Reader:
        def json(self,item):
            seen.append(item["path"])
            return deepcopy(config if item["path"]==boundconfig["path"] else selection)
    _value,configs,asset=series.read_series(bound["path"],expected_sha256=bound["sha256"],reader_factory=lambda aliases:Reader())
    assert configs==[config] and asset==bound and seen==[boundselection["path"],boundconfig["path"]]


def test_preparation_is_idempotent_without_reenqueuing_an_existing_claim(tmp_path,monkeypatch):
    _root,manifest,source,_config,_value=assembly_fixture(tmp_path,monkeypatch)
    destination=tmp_path/"shared_job_fixture"
    first=alias_job.prepare_alias_job(source["path"],destination,expected_sha256=source["sha256"],series_manifest=manifest)
    pending=destination/"queue/ac52.json";original=pending.read_bytes()
    claimed=destination/"claimed/Main/ac52.json";claimed.parent.mkdir(parents=True)
    pending.rename(claimed)
    second=alias_job.prepare_alias_job(source["path"],destination,expected_sha256=source["sha256"],series_manifest=manifest)
    assert not first["tasks"][0]["already_enqueued"]
    assert second["tasks"][0]["already_enqueued"]
    assert claimed.read_bytes()==original and not pending.exists()


@pytest.mark.parametrize("publish",[[r"\\rooted"],["C:/machine/path"],["../escape"],["a//b"]])
def test_publish_names_cannot_smuggle_a_machine_path_before_job_creation(tmp_path,monkeypatch,publish):
    _root,manifest,source,_config,value=assembly_fixture(tmp_path,monkeypatch)
    value["job"]["publish"]=publish
    destination=tmp_path/"must_not_exist"
    with pytest.raises(ValueError,match="Publish must name relative products"):
        alias_job.prepare_alias_job(source["path"],destination,expected_sha256=source["sha256"],series_manifest=manifest)
    assert not destination.exists()


def test_new_master_own_metadata_reference_is_relative_and_hash_bound(tmp_path):
    root=tmp_path/"shared_job";root.mkdir()
    target=doc(root/"job.json",{"format":alias_job.JOB_FORMAT})
    ref=alias_job.job_metadata_reference(root,Path(target["path"]))
    assert ref==dict(relative_path="job.json",sha256=target["sha256"],bytes=target["size_bytes"])
    assert str(root) not in json.dumps(ref)
    changed=doc(root/"job.json",{"format":alias_job.JOB_FORMAT,"changed":True})
    assert alias_job.job_metadata_reference(root,Path(target["path"]))["sha256"]==changed["sha256"]
    with pytest.raises(ValueError,match="outside"):
        alias_job.job_metadata_reference(root,tmp_path/"foreign.json")


@pytest.mark.parametrize("legacy",[True,False])
def test_master_view_explicitly_distinguishes_original_absolute_path_jobs(qtbot,monkeypatch,legacy):
    panel=JobsPanel(local_machine="Main");qtbot.addWidget(panel)
    monkeypatch.setattr(panel,"_refresh_targets",lambda:None)
    monkeypatch.setattr(panel,"_job_requests",lambda:[])
    panel._show_snapshot(dict(root="not-opened",job_facts={"legacy_paths":legacy},
                             name="Fixture",state="Waiting",master_machine="Main"))
    assert "legacy_paths="+str(legacy).lower() in panel.master_label.text()
