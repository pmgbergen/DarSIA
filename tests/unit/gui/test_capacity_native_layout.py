"""Literal native JSON layout admission; no arrays, real profiles or job claims."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import pytest

from darsia.fluidflower.pipeline import capacity as cap
from .test_capacity import measured, native_layout, native_metadata


def actual_request(tmp_path):
    task,_,profile,measurement,completion=measured()
    config,rows,expected=native_metadata(tmp_path)
    profile["measured_native_layout"]=expected
    return task,config,profile,measurement,completion,rows


def request_layout(data):
    task,config,_,_,_,rows=data
    return cap._native_layout_for_task(None,"FF_AC",task,config,rows)


def request_estimate(data):
    task,config,profile,measurement,completion,_=data
    return cap.estimate_run(task,config,profile,measurement=measurement,completion=completion,
        requested_native_layout=request_layout(data))


def rewrite(data,use,edit):
    """An explicit owned metadata mutation with valid new outer SHA/size."""
    _,config,_,_,_,rows=data
    row=next(row for row in rows if row.uses==(use,))
    value=json.loads(row.path.read_bytes());edit(value)
    raw=json.dumps(value,sort_keys=True).encode();row.path.write_bytes(raw)
    row.reference.update(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    if use=="native.configuration":bound=config["native"]["configuration"]
    elif use=="phase.configuration":bound=config["phase"]["configuration"]
    else:
        _,case,kind=use.split(".");frame=next(frame for frame in config["frames"] if frame["case_id"]==case)
        bound=frame["native_task"]["geometry"] if kind=="native_geometry" else frame["geometry"]
    bound.update(sha256=row.reference["sha256"],size_bytes=len(raw))


def test_genuine_native_precision_is_task_owned_without_mutating_config(tmp_path):
    data=actual_request(tmp_path);before=json.dumps(data[1],sort_keys=True)
    assert "precision" not in data[1]
    assert request_layout(data)==data[2]["measured_native_layout"]
    value=request_estimate(data)
    assert value.native_expected_bytes==250 and value.transient_estimated_bytes==140
    assert json.dumps(data[1],sort_keys=True)==before and "precision" not in data[1]


def test_layout_reads_only_the_six_declared_json_roles_not_unknown_binary(tmp_path,monkeypatch):
    data=actual_request(tmp_path);original=cap.read_bound;seen=[]
    def read(bound):seen.append(bound["path"]);return original(bound)
    monkeypatch.setattr(cap,"read_bound",read)
    unknown=SimpleNamespace(path=tmp_path/"MUST_NOT_OPEN.npz",reference={},uses=("frame.ac52_f1.image",))
    data=(*data[:-1],(*data[-1],unknown))
    assert request_layout(data)==data[2]["measured_native_layout"]
    assert len(seen)==6 and set(seen)=={str(row.path) for row in data[-1] if row is not unknown}
    assert all(Path(path).suffix==".json" for path in seen) and not unknown.path.exists()


@pytest.mark.parametrize("key,value",[
    ("native_shape_hw",[3,4]),("native_shape_hw",[1,4]),
    ("phase_shape_hw",[4,6]),("phase_shape_hw",[2,6]),
    ("native_shape_chw",[4,2,4]),("source_box_xywh",[2,1,4,2]),
    ("native_configuration_sha256","c"*64),("phase_configuration_sha256","c"*64),
    ("feature_rgb_dtype","float64"),("precision","float64")])
def test_different_layout_or_parameter_identity_has_no_smaller_or_larger_fallback(key,value):
    measured_layout=native_layout();requested=deepcopy(measured_layout);requested[key]=value
    with pytest.raises(ValueError):cap._check_measured_native_layout({"measured_native_layout":measured_layout},requested)


@pytest.mark.parametrize("key,value",[
    ("format","unknown"),("pipeline","native-phase-statistics"),("precision","unknown"),
    ("native_shape_hw",[True,4]),("native_shape_hw",[0,4]),("phase_shape_hw",[-1,6]),
    ("native_shape_chw",[3,2]),("source_box_xywh",[1,1,True,2]),
    ("native_configuration_sha256","A"*64)])
def test_equal_but_malformed_layout_objects_do_not_validate_each_other(key,value):
    value_layout=native_layout();value_layout[key]=value
    with pytest.raises(ValueError):cap._check_measured_native_layout({"measured_native_layout":value_layout},deepcopy(value_layout))


def test_missing_measured_layout_is_a_refusal_not_a_historical_default():
    task,config,profile,measurement,completion=measured();profile.pop("measured_native_layout")
    with pytest.raises(ValueError,match="closed"):
        cap.estimate_run(task,config,profile,measurement=measurement,completion=completion,
            requested_native_layout=native_layout())


@pytest.mark.parametrize("defect",["native_height","source_shape","phase_canvas","mixed_canvas","channels","feature_dtype"])
def test_hash_valid_requested_json_still_requires_exact_native_and_canvas_layout(tmp_path,defect):
    data=actual_request(tmp_path)
    if defect=="native_height":
        rewrite(data,"frame.ac52_f1.native_geometry",lambda v:v.update(native_shape_hw=[3,4]))
    elif defect=="source_shape":
        rewrite(data,"native.configuration",lambda v:v["source"].update(native_shape_hw=[1,4]))
    elif defect=="phase_canvas":
        for case in ("ac52_f1","ac52_f2"):
            rewrite(data,"frame."+case+".rig_geometry",lambda v:v.update(phase_shape_hw=[4,6]))
            next(f for f in data[1]["frames"] if f["case_id"]==case)["mass_spec"]["photo_top"]["shape_hw"]=[4,6]
    elif defect=="mixed_canvas":
        rewrite(data,"frame.ac52_f2.rig_geometry",lambda v:v.update(phase_shape_hw=[4,6]))
        data[1]["frames"][1]["mass_spec"]["photo_top"]["shape_hw"]=[4,6]
    elif defect=="channels":
        rewrite(data,"native.configuration",lambda v:v["photometry"].update(native_shape_chw=[4,2,4]))
    else:
        rewrite(data,"phase.configuration",lambda v:v["parameters"]["features"]["alignment"].update(rgb_dtype="float64"))
    with pytest.raises(ValueError):request_estimate(data)


@pytest.mark.parametrize("defect",["missing","duplicate"])
def test_one_explicit_catalogue_use_is_required_before_read(tmp_path,monkeypatch,defect):
    data=actual_request(tmp_path);rows=list(data[-1]);first=next(row for row in rows if row.uses==("phase.configuration",))
    rows.remove(first) if defect=="missing" else rows.append(first)
    data=(*data[:-1],tuple(rows));monkeypatch.setattr(cap,"read_bound",lambda _:pytest.fail("No ambiguous role read"))
    with pytest.raises(ValueError,match="One existing layout metadata role"):request_layout(data)


@pytest.mark.parametrize("defect",["sha","size","boolean_size"])
def test_catalogue_and_original_descriptor_must_agree_before_read(tmp_path,monkeypatch,defect):
    data=actual_request(tmp_path);row=next(row for row in data[-1] if row.uses==("phase.configuration",))
    if defect=="sha":row.reference["sha256"]="0"*64
    elif defect=="size":row.reference["bytes"]+=1
    else:row.reference["bytes"]=True
    monkeypatch.setattr(cap,"read_bound",lambda _:pytest.fail("No mismatched layout read"))
    with pytest.raises(ValueError,match="SHA/size differs"):request_layout(data)


def test_source_native_processor_does_not_borrow_the_measured_rig_profile(tmp_path):
    data=actual_request(tmp_path);data[1]["frames"][0]["native_task"]["kind"]="source"
    with pytest.raises(ValueError,match="Unmeasured native processor"):request_estimate(data)


def test_future_phase_one_cannot_relabel_a_full_pipeline_measurement():
    task,config,profile,measurement,completion=measured()
    task["pipeline"]=profile["pipeline"]="native-phase-statistics"
    with pytest.raises(ValueError,match="separately measured"):
        cap.estimate_run(task,config,profile,measurement=measurement,completion=completion)


def test_an_explicit_native_config_precision_mismatch_still_refuses(tmp_path):
    data=actual_request(tmp_path);data[1]["precision"]="float64"
    with pytest.raises(ValueError,match="precision"):request_estimate(data)


def test_other_pipeline_precision_requirement_is_unchanged():
    task,config,profile,measurement,completion=measured();task["pipeline"]=profile["pipeline"]="n1"
    config.pop("precision")
    with pytest.raises(ValueError,match="precision"):
        cap.estimate_run(task,config,profile,measurement=measurement,completion=completion)
