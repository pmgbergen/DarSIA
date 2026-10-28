"""Small metadata-only negative tests; no scientific images or inference."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from darsia.fluidflower import phase_saved_review_metadata as metadata

_COMBINED = {'class_key': 'combined_floor', 'exports': {}, 'literals': {'_saved_combined_floor_verify_heads': [{'python_type': 'NoneType', 'value': None}, {'python_type': 'NoneType', 'value': None}, {'python_type': 'str', 'value': 'Combined candidate changed its frozen head inventory'}, {'python_type': 'str', 'value': 'sha256'}, {'python_type': 'str', 'value': 'Combined candidate changed a declared frozen head'}, {'python_type': 'str', 'value': 'Combined candidate head escaped its isolated namespace'}]}}

def _asset(path):
    data=path.read_bytes()
    return {"path":str(path),"sha256":hashlib.sha256(data).hexdigest()}
def test_saved_head_guard_rejects_changed_inventory_and_bytes(tmp_path):
    guard=metadata._SavedCombinedBindings(copy.deepcopy(_COMBINED))
    path=tmp_path/"head.json";path.write_text("{}",encoding="utf-8")
    bound=_asset(path);expected={"head":bound["sha256"]}
    guard._verify_heads({"head":bound},expected)
    with pytest.raises(ValueError):
        guard._verify_heads({"head":bound,"extra":bound},expected)
    path.write_text('{"changed":true}',encoding="utf-8")
    with pytest.raises(ValueError):
        guard._verify_heads({"head":bound},expected)

_JOINT = {'class_key': 'joint_floor', 'exports': {'FEATURE_NAMES': {'python_type': 'tuple', 'value': ['ff_log_r', 'ff_log_g', 'ff_log_b', 'ff_log_y', 'new_lamp_indicator']}}, 'literals': {'_saved_joint_floor_verify_pixel_model': [{'python_type': 'str', 'value': 'T4b joint pixel model changed'}, {'python_type': 'str', 'value': 'fit'}, {'python_type': 'str', 'value': 'variant'}, {'python_type': 'str', 'value': 'G3'}, {'python_type': 'str', 'value': 'training_revision'}, {'python_type': 'str', 'value': 'G3-T4b'}, {'python_type': 'str', 'value': 'feature_names'}, {'python_type': 'str', 'value': 'lamp_shift_only'}, {'python_type': 'bool', 'value': True}, {'python_type': 'str', 'value': 'converged'}, {'python_type': 'bool', 'value': True}, {'python_type': 'str', 'value': 'backend'}, {'python_type': 'str', 'value': 'cpu'}, {'python_type': 'int', 'value': 5}, {'python_type': 'str', 'value': 'mean'}, {'python_type': 'str', 'value': 'scale'}, {'python_type': 'str', 'value': 'coefficients'}, {'python_type': 'str', 'value': 'mean'}, {'python_type': 'str', 'value': 'scale'}, {'python_type': 'str', 'value': 'coefficients'}, {'python_type': 'str', 'value': 'scale'}, {'python_type': 'int', 'value': 0}, {'python_type': 'str', 'value': 'intercept'}, {'python_type': 'str', 'value': 'T4b requires one converged five-feature shift-only pixel head'}]}}

def test_saved_joint_head_rejects_nonfinite_and_wrong_shape(tmp_path):
    guard=metadata._SavedJointBindings(copy.deepcopy(_JOINT))
    model={"variant":"G3","training_revision":"G3-T4b",
           "feature_names":list(guard._export("FEATURE_NAMES")),
           "lamp_shift_only":True,"fit":{"converged":True,"backend":"cpu",
           "mean":[0.]*5,"scale":[1.]*5,"coefficients":[0.]*5,"intercept":0.}}
    path=tmp_path/"pixel.json"
    def publish(value):
        path.write_text(json.dumps(value),encoding="utf-8")
        return _asset(path)["sha256"]
    guard.verify_pixel_model(path,publish(model))
    model["fit"]["coefficients"][0]=float("nan")
    with pytest.raises(ValueError):guard.verify_pixel_model(path,publish(model))
    model["fit"]["coefficients"]=[0.]*4
    with pytest.raises(ValueError):guard.verify_pixel_model(path,publish(model))
