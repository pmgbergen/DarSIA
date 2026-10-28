"""Original test_n1_pipeline tiny saved-model fixture; functions copied verbatim.

PRIVATE validation helper only. No campaign data and no native/phase inference.
"""
from pathlib import Path
import json
import numpy as np
import pytest
import darsia
from darsia.fluidflower.mass import n1_pipeline as pipeline

def assert_same(first, second):
    """Compare complete numerical structures, including native NaN patterns."""
    if isinstance(first,np.ndarray):
        assert first.dtype == second.dtype
        np.testing.assert_array_equal(first,second)
    elif isinstance(first,dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_same(first[key],second[key])
    elif isinstance(first,(list,tuple)):
        assert len(first) == len(second)
        for left,right in zip(first,second):
            assert_same(left,right)
    else:
        assert first == second


@pytest.fixture
def tiny_config(tmp_path):
    """Build a genuine tiny saved-model dataset with no fitted parameters."""
    def save(name, value, kind="json"):
        path=tmp_path/name
        if kind == "json":
            path.write_text(json.dumps(value),encoding="utf-8")
        elif kind == "npy":
            np.save(path,value,allow_pickle=False)
        elif kind == "npz":
            np.savez(path,**value)
        elif kind == "png":
            from PIL import Image
            Image.fromarray(value).save(path)
        return pipeline.asset(path)
    shape=(3,3)
    domain=np.ones(shape,bool)
    facies=np.ones(shape,np.uint8)
    facies[:,2]=2
    labels=np.ones(shape,np.uint8)
    labels[1,1]=2
    labels[2,0]=0
    physical=save("physical.npz",dict(facies=facies,domain=domain,porosity=np.full(shape,.3),depth_m=np.full(shape,.01)),"npz")
    clear=save("clear.npy",np.ones((*shape,3),np.float32),"npy")
    valid=save("valid.npy",domain,"npy")
    phase=save("phase.png",labels,"png")
    old_domain=save("old.png",np.zeros(shape,np.uint8),"png")
    classes=save("classes.npz",dict(rig_domain=domain,modelled=domain,filled_observed=domain,filled_unobserved=np.zeros(shape,bool)),"npz")
    area=save("area.npz",dict(pixel_area_mm2=np.ones(shape)),"npz")
    top=save("top.npz",dict(keep=domain),"npz")
    floor=save("floor.npy",np.full(3,2.),"npy")
    def raw(name, rgb):
        value=np.broadcast_to(np.asarray(rgb,np.float32),(*shape,3)).copy()
        logs=np.moveaxis(np.log(value*np.float32(65535)),2,0).astype(np.float32)
        bound=save(name,logs,"npy")
        return dict(phase_shape_hw=list(shape),native_origin_xy=[0,0],
            native_centre=dict(native_shape_hw=list(shape),log_linear_rgb=bound,valid_mask=valid),
            ring_windows=[],binding=dict(geometry={"H_ac28_to_frame_native":np.eye(3).tolist()}))
    current,baseline=raw("current.npy",[.8,.7,.6]),raw("baseline.npy",[1.,1.,1.])
    colors,signals={},{}
    flash_bounds=dict(min_value_aq=0.,max_value_aq=1.,min_value_g=0.,max_value_g=1.)
    for label in (1,2):
        base=tmp_path/("color_"+str(label))
        path=darsia.ColorPath(colors=[np.array([1.,1.,1.]),np.array([.8,.7,.6])])
        darsia.ColorPathInterpolation(path,darsia.ColorMode.RELATIVE,[0.,1.]).save(base)
        colors[str(label)]=pipeline.asset(base.with_suffix(".json"))
        base=tmp_path/("signal_"+str(label))
        darsia.PWTransformation([0.,1.],[0.,1.]).save(base)
        signals[str(label)]=pipeline.asset(base.with_suffix(".csv"))
    base=tmp_path/"flash"
    darsia.SimpleFlash(**flash_bounds).save(base)
    metadata=save("optics.json",dict(label_ids=[1,2],ignore_labels=[2],basis="facies",flash_bounds=flash_bounds))
    config=dict(format=pipeline.CONFIG_FORMAT,run="experiment",frames=[],
        shared=dict(physical_reference=physical,reference_matrix=[[1.,0.,0.],[0.,-1.,3.],[0.,0.,1.]],
            floor_curve=floor,floor_origin_xy=[0,0],depth_points=[[0,1,.01],[3,1,.01],[0,3,.01],[3,3,.01]],
            porosity_by_facies={"1":.3,"2":.3},swi_by_facies={"1":.2,"2":.25},unquantified_facies=[2],
            clearwater=dict(rgb=clear,valid=valid,H_reference_to_grid=np.eye(3).tolist()),
            optics=dict(metadata=metadata,color_paths=colors,signals=signals,flash=pipeline.asset(base.with_suffix(".json")))),
        calibration=save("calibration.json",dict(parameters=dict(candidate="N1",gain=1.2,beta=.3))),
        physical_correction=save("correction.json",dict(correction_matrix=np.eye(3).tolist())),
        anchor=dict(elapsed_h=.5,affected_pore_volume_m3=1e-8),recipe=dict(alkalinity_M=.001,btb_M=.0001),
        dimensions_m=[1.,1.],boxes=dict(box1=[[0.,0.],[1.,1.]],box2=[[0.,0.],[.5,.5]]),
        summary=dict(baseline_gain=1.1,intervals=32768,max_column_distance_m=.02,max_depth_m=.0155,max_porosity=.45),
        bindings=dict(amendment_sha256="a"*64,round4_spec_sha256="b"*64),
        contracts=dict(result="saved-products",sandmap="saved-sandmap"))
    for index in (1,3):
        config["frames"].append(dict(case_id="sample_"+str(index),frame_index=index,elapsed_h=float(index),selected_target_h=float(index),
            source_binding=dict(source_sha256="c"*64),current=current,baseline=baseline,labels=phase,classes=classes,
            area=area,photo_top=top,original_domain=dict(parent_labels_map=old_domain,native_shape_hw=list(shape),
                H_ac28_to_frame_native=np.eye(3).tolist(),frame_index=0),inner_polygon_native=[[0,0],[2,0],[2,2],[0,2]],
            atmosphere=dict(pressure_bar=1.,temperature_c=20.,pressure_gradient_bar_per_m=0.,temperature_gradient_c_per_m=0.),
            nominal_injected_mass_g=1.,light_epoch="measured"))
    return config,save("config.json",config)

