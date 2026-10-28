"""Local-filesystem behavioral tests for whole-series lifecycle public APIs.

UNC routing is replaced only at the storage boundary. Content admission, queue
reservation, physical geometry, sidecar publication and chronological state
execute normally on authored tiny fixtures. RAW/model operators are explicit
test doubles; these tests do not claim a scientific equivalence gate.
"""

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import queue
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from darsia.fluidflower.mass import n1_pipeline as mass
from darsia.fluidflower.pipeline import checkpoint, job_queue, runner, series, worker
from darsia.fluidflower.pipeline import alias_job, job_inputs


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return job_queue.descriptor(path)


def _array(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
    return job_queue.descriptor(path)


@pytest.fixture
def local_storage(monkeypatch, tmp_path):
    """Replace network-location policy only; all real byte/hash checks remain."""
    def admit(path):
        resolved = Path(path).resolve()
        assert resolved == tmp_path or tmp_path in resolved.parents
    monkeypatch.setattr(runner, "require_unc", admit)
    monkeypatch.setattr(series, "require_unc", admit)
    monkeypatch.setattr(job_queue, "require_unc", admit)
    monkeypatch.setattr(mass, "_is_unc", lambda path: True)
    return tmp_path


@pytest.fixture
def portable_prepare_storage(local_storage, monkeypatch):
    """Real original metadata checks; tiny role graph is an explicit test double.

    This authored fixture intentionally has no RAW/model graph. Full named-input
    closure is covered independently in test_alias_job_inputs.
    """
    original_catalog = job_inputs.InputCatalog
    class TinyPreparationCatalog(original_catalog):
        def collect(self, bound, config):
            reference = self.add(bound, "configuration", "run.configuration")
            self.add(config["mass"]["calibration"], "calibration", "mass.calibration")
            return dict(format=job_inputs.CATALOG_FORMAT, complete=True,
                run=config["run"], configuration=reference,
                inputs=[self.entries[key] for key in sorted(self.entries)], payload_hashes_verified=False,
                numerical_calls_started=False)
    monkeypatch.setattr(alias_job, "InputCatalog", TinyPreparationCatalog)
    monkeypatch.setattr(alias_job, "require_unc", lambda path: None)
    from darsia.fluidflower import job_filesystem
    monkeypatch.setattr(job_filesystem,"inspect_shared_folder",lambda path:
        job_filesystem.SharedFolderAdmission(True,"smb","authored unit transport","no real FS qualification"))
    return local_storage


def _run_config(calibration):
    source = dict(run="tiny", frame_index=1, elapsed_h=1.0,
        source_sha256="a"*64, previous_source_sha256="b"*64,
        selected=True, studio_anchor=False,
        source_binding=dict(source_sha256="a"*64))
    return dict(format=series.RUN_FORMAT, run="tiny",
        selection_index_field="frame_index",
        frames=[dict(case_id="tiny_f1", frame_index=1, elapsed_h=1.0,
            source=source, native_task=dict(kind="rig"), geometry={},
            phase_roles=dict(area=dict(role="selected", factor=1.0)),
            mass_spec=dict(nominal_injected_mass_g=1.5))],
        native={}, phase=dict(initial_previous=dict(source=dict(source_sha256="b"*64))),
        mass=dict(calibration=calibration, template=dict(calibration=calibration)),
        table_fields=["case_id", "total_mass_g"], history_policy="camera chronology")


def _prepared_series(root):
    calibration = _json(root/"calibration.json", dict(gain=2.0))
    config = _run_config(calibration)
    config_asset = _json(root/"run.json", config)
    selection = _json(root/"selection.json",
        dict(complete=True, runs=[dict(run="tiny", ordered_case_ids=["tiny_f1"])]))
    def inline(value):
        return "{ " + ", ".join(key+" = "+json.dumps(item) for key,item in value.items())+" }"
    text = "\n".join([
        'format = "darsia-series-v1"', "adopted_for_execution = true",
        'job_id = "tiny-job"', "selection = "+inline(selection),
        "[job]", 'commit = "'+"d"*40+'"', 'precision = "float32"',
        'calibration_sha256 = "'+calibration["sha256"]+'"',
        'environment_lock = {path = "requirements-worker.lock", sha256 = "'+"e"*64+'"}',
        "[[runs]]", 'run = "tiny"', "configuration = "+inline(config_asset),
        '[runs.resources]', 'backend = "cpu"', "threads = 1", ""
    ])
    path=root/"execution.toml"
    path.write_text(text, encoding="utf-8")
    (root/"series.toml").write_text(
        'schema_version=1\nseries_id="unit"\n[roots]\nseries="."\n',
        encoding="utf-8")
    return path, job_queue.descriptor(path), config


def test_prepare_job_reserves_once_and_preserves_claimed_task(portable_prepare_storage):
    local_storage=portable_prepare_storage
    root=local_storage
    path, bound, _ = _prepared_series(root)
    job=root/"job"
    first=runner.prepare_job(path, job, expected_sha256=bound["sha256"],
        series_manifest=job_queue.descriptor(local_storage/"series.toml"))
    pending=job/"queue"/"tiny.json"
    initial=pending.read_bytes()
    token=json.loads(initial)["attempt_id"]
    # Simulate an already claimed reservation; preparation must not re-enqueue.
    claimed=job/"claimed"/"machine"/"tiny.json"
    claimed.parent.mkdir(parents=True)
    pending.rename(claimed)
    second=runner.prepare_job(path, job, expected_sha256=bound["sha256"],
        series_manifest=job_queue.descriptor(local_storage/"series.toml"))
    assert first["numerical_calls_started"] is False
    assert first["tasks"][0]["already_enqueued"] is False
    assert second["tasks"][0]["already_enqueued"] is True
    assert claimed.read_bytes() == initial and not pending.exists()
    assert json.loads((job/"results"/"tiny"/"task.json").read_bytes())["attempt_id"] == token


@pytest.mark.parametrize("conflict", ["job", "run"])
def test_prepare_job_rejects_existing_metadata_without_republishing(portable_prepare_storage, conflict):
    local_storage=portable_prepare_storage
    path, bound, _ = _prepared_series(local_storage)
    job=local_storage/"job"
    runner.prepare_job(path,job,expected_sha256=bound["sha256"],
        series_manifest=job_queue.descriptor(local_storage/"series.toml"))
    pending=job/"queue"/"tiny.json"
    original=pending.read_bytes()
    location=job/("job.json" if conflict=="job" else "results/tiny/task.json")
    data=json.loads(location.read_bytes())
    data["commit"]="f"*40
    _json(location,data)
    with pytest.raises(ValueError,match="differs"):
        runner.prepare_job(path,job,expected_sha256=bound["sha256"],
        series_manifest=job_queue.descriptor(local_storage/"series.toml"))
    assert pending.read_bytes()==original


def test_prepare_cli_emits_bound_result_and_missing_required_option_stops(portable_prepare_storage, capsys):
    local_storage=portable_prepare_storage
    path,bound,_=_prepared_series(local_storage)
    runner.main(["--series",str(path),"--series-sha256",bound["sha256"],
        "--series-manifest",str(local_storage/"series.toml"),
        "--job",str(local_storage/"job")])
    value=json.loads(capsys.readouterr().out)
    assert value["job"]==job_queue.descriptor(local_storage/"job/job.json")
    assert value["tasks"][0]["run"]=="tiny" and not value["numerical_calls_started"]
    with pytest.raises(SystemExit) as failure:
        runner.main(["--series",str(path)])
    assert failure.value.code==2


def test_prepare_cli_rejects_undeclared_source_root_before_preparation(tmp_path,monkeypatch):
    from darsia.fluidflower import job_filesystem
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda _path:
        job_filesystem.SharedFolderAdmission(True, "smb", "isolated unit transport", "not a real FS qualification"))
    registered=tmp_path/"registered";registered.mkdir()
    manifest=registered/"series.toml"
    manifest.write_text('schema_version=1\nseries_id="unit"\n[roots]\nseries="."\n',
                        encoding="utf-8")
    with pytest.raises(ValueError,match="No declared root"):
        runner.main(["--series",str(tmp_path/"absent.toml"),
            "--series-sha256","0"*64,"--series-manifest",str(manifest),
            "--job",str(tmp_path/"job")])
    assert not (tmp_path/"job").exists()


def test_load_run_admits_actual_hash_and_rejects_other_calibration_and_legacy_task(local_storage):
    path,bound,config=_prepared_series(local_storage)
    config_asset=job_queue.descriptor(local_storage/"run.json")
    task=dict(pipeline=series.PIPELINE,configuration=config_asset,run="tiny",
        calibration_sha256=config["mass"]["calibration"]["sha256"])
    assert series.load_run(task)==config
    with pytest.raises(ValueError,match="run/calibration"):
        series.load_run(dict(task,calibration_sha256="f"*64))
    with pytest.raises(ValueError,match="explicit series"):
        series.load_run(dict(task,pipeline="n1"))
    (local_storage/"run.json").write_text("{}",encoding="utf-8")
    with pytest.raises(ValueError,match="size|SHA"):
        series.load_run(task)


def _frame_inventory(root):
    attempt=root/"attempt"; case="tiny_f1"
    stage=attempt/"preparation"/case
    final=attempt/"frames"/case
    final.mkdir(parents=True)
    source=_run_config({})["frames"][0]["source"]
    spec=dict(case_id=case,source=source)
    inventory={}
    for key in ("labels","classes","area","operator_masks"):
        bound=_array(stage/(key+".npy"),np.array([[1,2]],np.uint8))
        inventory[Path(bound["path"]).name]=bound
    temporal_log=_array(stage/"temporal_log.npy",np.zeros((3,1,2),np.float32))
    temporal_valid=_array(stage/"temporal_valid.npy",np.ones((1,2),bool))
    full=dict(contract="authored source data",source_sha256=source["source_sha256"])
    history=_json(stage/"history_input.json",dict(complete=True,source=source,full=full,
        current_h=np.eye(3).tolist(),temporal_log=temporal_log,temporal_valid=temporal_valid))
    audit=_json(stage/"native_audit.json",dict(source_sha256=source["source_sha256"]))
    qa=_json(stage/"qa.json",dict(area_cm2=.02))
    inventory.update({Path(b["path"]).name:b for b in (temporal_log,temporal_valid,history,audit,qa)})
    phase={k:inventory[k+".npy"] for k in ("labels","classes","area","operator_masks")}
    phase.update(source_sha256=source["source_sha256"],qa=qa,history_input=history)
    product=dict(complete=True,format="darsia-series-frame-v1",source=source,
        case_id=case,configuration_sha256="c"*64,files=inventory,
        native=dict(source_sha256=source["source_sha256"],audit=audit,full=full),phase=phase)
    result=dict(prediction=dict(pipeline=product),source_binding=source["source_binding"],
        bindings=dict(phase_sha256=phase["labels"]["sha256"]))
    return spec,result,final,dict(_configuration_sha256="c"*64)


def test_frame_artifact_admission_checks_original_files_and_source(local_storage):
    args=_frame_inventory(local_storage)
    spec,result,final,config=args
    assert runner.validate_frame_artifacts(*args) is result["prediction"]["pipeline"]
    bound=result["prediction"]["pipeline"]["phase"]["labels"]
    Path(bound["path"]).write_bytes(b"changed after publication")
    with pytest.raises(ValueError,match="size|SHA"):
        runner.validate_frame_artifacts(*args)


@pytest.mark.parametrize("change",["source","configuration","escaped","phase","history","mass"])
def test_frame_artifact_admission_rejects_cross_attempt_and_cross_stage_links(local_storage,change):
    spec,result,final,config=_frame_inventory(local_storage)
    product=result["prediction"]["pipeline"]
    if change=="source":
        product["native"]["source_sha256"]="e"*64
    elif change=="configuration":
        product["configuration_sha256"]="e"*64
    elif change=="escaped":
        key=next(iter(product["files"]))
        product["files"][key]=dict(product["files"][key],path=str(local_storage/key))
    elif change=="phase":
        product["phase"]["classes"]=dict(product["phase"]["classes"],sha256="e"*64)
    elif change=="history":
        history=product["phase"]["history_input"]
        value=json.loads(Path(history["path"]).read_bytes())
        value["full"]=dict(another_acquisition=True)
        new=_json(Path(history["path"]),value)
        product["phase"]["history_input"]=new
        product["files"][Path(new["path"]).name]=new
    else:
        result["bindings"]["phase_sha256"]="e"*64
    with pytest.raises(ValueError):
        runner.validate_frame_artifacts(spec,result,final,config)


@pytest.mark.parametrize("raises",[False,True])
def test_run_series_limits_real_threads_and_restores_after_success_or_failure(monkeypatch,raises):
    previous=cv2.getNumThreads()
    seen=[]
    def execute(task,output,backend,precision,callback,runtime):
        seen.append(runtime)
        assert cv2.getNumThreads()==1
        assert runtime["requested_threads"]==1
        assert all(pool["num_threads"]==1 for pool in runtime["threadpools"])
        if raises:
            raise RuntimeError("authored stage failure")
        return dict(complete=True,callback=callback)
    monkeypatch.setattr(runner,"_run_series",execute)
    callback=object()
    if raises:
        with pytest.raises(RuntimeError,match="authored stage failure"):
            runner.run_series(dict(threads=1),"unused","cpu","float32",callback)
    else:
        result=runner.run_series(dict(threads=1),"unused","cpu","float32",callback)
        assert result["complete"] and result["callback"] is callback
    assert len(seen)==1 and cv2.getNumThreads()==previous


@pytest.mark.parametrize("value",[0,-1,True,1.5])
def test_run_series_bad_thread_count_never_enters_stages(monkeypatch,value):
    monkeypatch.setattr(runner,"_run_series",lambda *args:pytest.fail("stage entered"))
    previous=cv2.getNumThreads()
    with pytest.raises(ValueError,match="Positive internal"):
        runner.run_series(dict(threads=value),"unused","cpu","float32")
    assert cv2.getNumThreads()==previous


GEOMETRY=dict(determinant_floor=1e-14,point_horizon_floor=1e-12,
    area_horizon_floor=1e-10,observation_horizon_floor=1e-10,
    raster_scale=256,raster_shift=8,canvas_margin=.5,square_mm_per_cm2=100,
    nominal_pixel_area_mm2=.25,
    contracts=dict(rig_context="tiny rig",independent_context="tiny independent"),
    fields=dict(reference_polygon="polygon",reference_fixtures="fixtures",
        nominal_area_field="nominal_area",reference_metric_field="metric",
        reference_transform_field="reference_to_frame"))


def _geometry_owner(root):
    owner=series.SeriesInputs.__new__(series.SeriesInputs)
    owner.reader=mass.DataReader()
    owner.parameters=dict(geometry=dict(rig_origin_xy=[0,0],
        native_shape_hw=[2,3],source_box_xywh=[0,0,3,2]),
        scientific=dict(contract="tiny scientific",storage="NPY authored windows"))
    owner.phase=dict(geometry_parameters=GEOMETRY,
        geometry_fields=dict(reference_to_metric="metric",reference_to_frame="frame"))
    top=np.ones((2,3),bool);top[0,0]=False
    top_asset=_array(root/"top.npy",top)
    row=dict(phase_shape_hw=[2,3],frame_photo_top_keep=top_asset,
        area_parameters=dict(metric=np.diag([2.,2.,1.]).tolist(),frame=np.eye(3).tolist()),
        reference_rig=dict(polygon_reference_native=[[0,0],[2,0],[2,1],[0,1]],
            metric_pixel_area_mm2=.25),
        H_reference_rig_to_frame_native=np.eye(3).tolist(),
        frame_fixture_polygons_native={},fixtures_verified_none=True)
    audit=dict(renderer=dict(review_to_camera_h=np.eye(3).tolist(),camera_jpeg_wh=[2,2]))
    return owner,row,audit,top


def test_series_geometry_separates_photographic_support_from_measured_area(tmp_path):
    owner,row,audit,top=_geometry_owner(tmp_path)
    valid=np.ones((2,3),bool)
    valid[1,1]=False
    valid[1,2]=False  # No native feature and no camera observation: unobserved fill.
    actual,observed,admitted_top=owner.geometry(row,valid,audit)
    assert np.array_equal(observed,np.array([[True,True,False],[True,True,False]]))
    assert np.array_equal(admitted_top,top)
    assert np.array_equal(actual.rig_domain,top)
    assert actual.modelled[1,1]==False
    assert actual.filled_observed[1,1]==True
    assert actual.filled_unobserved[1,2]==True
    assert actual.pixel_area_mm2.dtype==np.float32
    assert np.array_equal(actual.pixel_area_mm2,np.ones((2,3),np.float32))
    # Bad full-canvas top DATA must not silently resize or reinterpret.
    bad=_array(tmp_path/"bad_top.npy",np.ones((1,3),bool))
    with pytest.raises(ValueError,match="unchanged complete canvas"):
        owner.geometry(dict(row,frame_photo_top_keep=bad),valid,audit)


def test_series_constructor_keeps_reference_and_current_history_distinct(local_storage,monkeypatch):
    from darsia.fluidflower.pipeline import phase_inputs as loaders
    from darsia.fluidflower.phase.inputs import MaskedNativeInput,TemporalNativeInput
    root=local_storage
    ref_log=np.zeros((3,2,3),np.float32)
    old_log=np.full((3,2,3),np.float32(np.log(2)),np.float32)
    valid=np.ones((2,3),bool)
    ref_center=_array(root/"ref.npy",ref_log)
    old_center=_array(root/"previous.npy",old_log)
    ref=dict(native_centre=dict(log_linear_rgb=ref_center),role="reference")
    old=dict(native_centre=dict(log_linear_rgb=old_center),role="previous")
    row_asset=_json(root/"native_row.json",dict(original_source=True))
    phase=_json(root/"phase.json",dict(parameters=dict(scientific={},inputs={},
        native=dict(block_rows=1),features=dict(flatfield=dict(natural_matrix=np.eye(3).tolist())))))
    native=_json(root/"native.json",dict(source={},extended_ring={}))
    physical=_json(root/"physical_alias.json",dict(value=1))
    logical=dict(physical,path=r"C:\historical\metadata.json")
    cache=SimpleNamespace(clear=lambda:None)
    monkeypatch.setattr(series.SeriesInputs,"_models",
        lambda self:dict(native=dict(fixed_baseline_decode_cache=cache)))
    calls=[]
    def compose(value,**kwargs):
        calls.append(value["role"])
        return (ref_log if value["role"]=="reference" else old_log),valid
    monkeypatch.setattr(loaders,"compose_extended_scientific",compose)
    monkeypatch.setattr(loaders,"original_native_input",
        lambda row,scientific_native_log,**kw:MaskedNativeInput(scientific_native_log,valid,dict(role="loaded")))
    monkeypatch.setattr(loaders,"original_temporal_input",
        lambda native,row,**kw:TemporalNativeInput(native.log,native.valid,(0,0),dict(role="temporal")))
    monkeypatch.setattr(loaders,"temporal_policy",lambda row,**kw:dict(pad_px=0))
    config=dict(data_aliases=[dict(logical=logical,physical=physical)],
        native=dict(configuration=native),
        phase=dict(configuration=phase,reference=dict(native_row=row_asset,full=ref),
            initial_previous=dict(native_row=row_asset,full=old,source=dict(source_sha256="b"*64),
                current_h=np.eye(3).tolist())))
    before=deepcopy(config)
    owner=series.SeriesInputs(config,root/"output")
    assert calls==["reference","previous"]
    assert owner.mass_baseline==ref and owner.previous["source"]["source_sha256"]=="b"*64
    assert np.array_equal(owner.model_bundle["baseline"]["native_input"].log,ref_log)
    assert np.array_equal(owner.previous["full_log"],old_log)
    flat=owner.model_bundle["native"]["flatfield_reference"]
    assert np.array_equal(flat.linear_rgb,np.ones((2,3,3),np.float32))
    assert not flat.linear_rgb.flags.writeable
    assert owner.native_parameters["source"]["asset_aliases"]=={logical["path"]:physical["path"]}
    assert config==before
    with pytest.raises(ValueError,match="size|SHA"):
        Path(phase["path"]).write_text("{}",encoding="utf-8")
        series.SeriesInputs(config,root/"another_output")


@pytest.mark.parametrize("kind",["rig","source"])
def test_series_native_publishes_actual_arrays_and_preserves_processor_arguments(tmp_path,monkeypatch,kind):
    from darsia.fluidflower.native import source,extension
    owner,rig_row,audit,top=_geometry_owner(tmp_path)
    log=np.zeros((3,2,3),np.float32);log[:,1,1]=np.nan
    native_row=dict(frame_index=17)
    native_geometry=_json(tmp_path/"native_geometry.json",native_row)
    rig_geometry=_json(tmp_path/"rig_geometry.json",rig_row)
    constructor_calls=[];correct_calls=[];extension_calls=[]
    class Processor:
        def __init__(self,*args,**kwargs):
            constructor_calls.append((args,kwargs))
        def correct(self,*args,**kwargs):
            correct_calls.append((args,kwargs))
            if kind=="rig":
                return log,audit
            return SimpleNamespace(log_linear_rgb=log,
                certificate=dict(render=audit["renderer"],photo=dict(frozen_fit=True)))
        def identity(self):
            return dict(actual_test_processor=True)
    monkeypatch.setattr(source,"RigSourceProcessor",Processor)
    monkeypatch.setattr(source,"SourceProcessor",Processor)
    def extend(actual,actual_audit,processor,eligible,**kwargs):
        extension_calls.append((actual,actual_audit,processor,eligible.copy(),kwargs))
        return actual.copy(),np.isfinite(actual).all(0),dict(windows=[dict(canvas_xyxy=[0,0,1,1])])
    monkeypatch.setattr(extension,"extend_prepared_source",extend)
    owner.processors={}
    owner.native_parameters=dict(extension=dict(reference_to_frame_h_key="carried_h"))
    owner.config=dict(native=dict(source_certificate_fields=dict(renderer="render",photometry="photo")))
    manifest=_json(tmp_path/"manifest.json",dict(frozen_manifest=True))
    recipe=dict(native_manifest=manifest)
    task=dict(kind=kind,geometry=native_geometry,processor=recipe,
        parent_frame_index=9,parent_reference=True)
    if kind=="source":
        recipe.clear()
        recipe.update(source_row=_json(tmp_path/"source_row.json",dict(source_id="authored")),
            calibration_manifest=manifest,photo_aggregate=manifest,
            enclosed_hole_policy="authored hole",display_mode="authored display",
            isolated_invalid_policy="authored invalid")
        args={name:_json(tmp_path/(name+".json"),dict(authored=True))
            for name in ("camera_jpeg","camera_arw","review_jpeg")}
        task.update(arguments=dict(args,frame_index=17),reference_to_frame_h=np.eye(3).tolist())
    spec=dict(case_id="tiny_f1",source=dict(source_sha256="a"*64),
        native_task=task,geometry=rig_geometry)
    result=owner.native(spec,tmp_path/"stage")
    assert np.array_equal(result["log"],log,equal_nan=True)
    assert result["valid"].dtype==bool and result["valid"][1,1]==False
    stage=tmp_path/"stage"
    assert np.array_equal(np.load(stage/"native_log.npy"),log,equal_nan=True)
    assert np.array_equal(np.load(stage/"ring_0.npy"),log[:,:1,:1],equal_nan=True)
    assert result["native_geometry_row"]==native_row
    assert result["geometry_row"]==rig_row
    assert np.array_equal(extension_calls[0][3],top & result["observed"])
    assert result["record"]["full"]["binding"]["geometry"]==rig_row
    assert job_queue.descriptor(stage/"native.json")==result["record_asset"]
    assert len(constructor_calls)==len(correct_calls)==1
    if kind=="rig":
        assert correct_calls[0]==((native_row,),dict(parent_frame_index=9,parent_reference=True))
    else:
        assert correct_calls[0][1]["frame_index"]==17
        assert all(isinstance(correct_calls[0][1][name],Path)
            for name in ("camera_jpeg","camera_arw","review_jpeg"))
        assert extension_calls[0][2].processor is next(iter(owner.processors.values()))
        assert extension_calls[0][1]["carried_h"]==np.eye(3).tolist()
    # A fresh scientific directory is required even when the processor is cached.
    with pytest.raises(FileExistsError):
        owner.native(spec,stage)


def test_series_advance_preserves_probe_history_and_close_releases_owners():
    owner=series.SeriesInputs.__new__(series.SeriesInputs)
    previous=dict(source=dict(source_sha256="before"),full_log=object())
    owner.previous=previous
    current=dict(source=dict(selected=False,baseline=False,source_sha256="new"),
        full_log=object(),full_valid=object(),current_h=object(),temporal_input=object())
    owner.advance(current)
    assert owner.previous is previous
    current["source"]["selected"]=True
    owner.advance(current)
    assert owner.previous["full_log"] is current["full_log"]
    assert set(owner.previous)=={"source","full_log","full_valid","current_h","temporal_input"}
    cleared=[]
    cache=SimpleNamespace(clear=lambda:cleared.append("cache"))
    owner.model_bundle=dict(native=dict(fixed_baseline_decode_cache=cache),baseline=object())
    owner.processors={"source":object()}
    owner.close()
    assert cleared==["cache"] and owner.previous is None
    assert not owner.model_bundle and not owner.processors


def test_retirement_document_binds_original_receipts_and_rejects_foreign_attempt(tmp_path):
    attempt=tmp_path/"attempt"
    final=attempt/"frames"/"tiny_f1"
    commit=_json(final/"commit.json",dict(commit_id="authored",complete=True))
    manifest=_json(attempt/"manifest.json",dict(run="tiny"))
    registration=_json(attempt/"ops/frame_commit_authored.json",dict(frame_id="tiny_f1"))
    publication=_json(attempt/"ops/nas_copy_authored.json",
        dict(frame_id="tiny_f1",nas_directory=str(final),files={"commit.json":commit}))
    identities=[dict(pid=123,exact_incarnation_absent=True)]
    supervisor=dict(actual_test_proof=True)
    result=checkpoint.retirement_document(dict(run="tiny"),attempt,identities,supervisor)
    assert result["manifest"]==manifest and result["identities"] is identities
    assert result["supervisor"] is supervisor and result["all_absent"] is True
    assert result["frame_receipts"]==[dict(frame_id="tiny_f1",
        registration=registration,publication=publication)]
    foreign=tmp_path/"foreign"/"frames"/"tiny_f1"
    data=json.loads(Path(publication["path"]).read_bytes())
    data["nas_directory"]=str(foreign)
    _json(Path(publication["path"]),data)
    with pytest.raises(ValueError,match="escapes"):
        checkpoint.retirement_document(dict(run="tiny"),attempt,identities,supervisor)


def test_retirement_document_rejects_changed_original_commit(tmp_path):
    attempt=tmp_path/"attempt"
    final=attempt/"frames"/"tiny_f1"
    commit=_json(final/"commit.json",dict(commit_id="authored"))
    _json(attempt/"ops/nas_copy_authored.json",
        dict(frame_id="tiny_f1",nas_directory=str(final),files={"commit.json":commit}))
    Path(commit["path"]).write_text('{"commit_id":"modified"}',encoding="utf-8")
    with pytest.raises(ValueError,match="differs"):
        checkpoint.retirement_document(dict(run="tiny"),attempt,[],{})


def test_queue_stream_print_flush_and_nonterminal_behavior():
    messages=queue.Queue()
    stream=worker.QueueStream(messages,"stdout")
    assert stream.encoding=="utf-8" and stream.isatty() is False
    print("µ output",file=stream,flush=True)
    first=messages.get_nowait();second=messages.get_nowait()
    assert first==dict(kind="console",stream="stdout",text="µ output")
    assert second==dict(kind="console",stream="stdout",text="\n")
    assert stream.flush() is None and messages.empty()
    assert stream.write("")==0 and messages.empty()
