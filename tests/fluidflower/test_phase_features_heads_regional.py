"""Tiny source-arithmetic regression fixtures for the extracted phase core."""
from __future__ import annotations
import ast
import copy
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import sys
import subprocess
from functools import lru_cache
import types
from typing import Any, Mapping, Protocol, Iterator, Sequence
import cv2
import numpy as np
import pytest
from scipy.special import expit
from darsia.fluidflower.phase import features, heads, regional

PARAMETERS = json.loads('{"features": {"native_decoder": {"denominator": 65535, "invalid_fill": 0, "operation": "exp(float32 channel,out=float32), divide np.float32(denominator)"}, "rig_decoder": {"denominator": 65535.0, "operation": "exp(maskedlog.astype(float64))/Pythonfloat denominator, assignment into float32"}, "optical": {"texture_sigma": 2.0, "spatial_context_fraction": 0.0015, "review_denominator": 255, "fraction_epsilon": 0.0001, "ratio_floor": 0.0001, "relative_grain_floor": 0.05, "relative_grain_min": -1, "relative_grain_max": 4, "block_rows_native": 128, "dynamic_available_column": true, "generic_contracts": "Exact source constant tables and original frozen head feature_contract; version identifiers are data", "review_encoding": "review-uint8", "unit_encoding": "v5-unit-float32", "block_rows": 128, "dynamic_unavailable_columns": 7}, "alignment": {"homography_min_determinant": 1e-09, "homography_min_scale": 1e-09, "support_tolerance": 1e-06, "interpolation": "cv2.INTER_LINEAR", "border": "cv2.BORDER_CONSTANT", "border_rgb": [0.0, 0.0, 0.0], "origin_translation_dtype": "float64", "rgb_dtype": "float32", "homography_floor": 1e-09, "coverage_floor": 1e-06}, "flatfield": {"box_size": 5, "halo_rows": 2, "denominator_floor": 1e-06, "denominator_dtype": "float32", "native_border": "cv2.BORDER_REFLECT_101", "masked_border": "cv2.BORDER_CONSTANT", "natural_matrix": [[0.00022422812891554153, -5.6491789785883e-05, 7.379563203284224e-06], [-4.387919269217012e-05, 0.0001407937073266365, -7.83856015712351e-05], [-2.6284823779787712e-05, -3.1023545081590044e-05, 0.0005425952594693075]], "natural_matrix_dtype": "float64", "luminance": [0.2126, 0.7152, 0.0722], "luminance_work_dtype": "float32", "ordering": "calibrate original baseline before warp; native difference-of-logs path and masked ratio-then-log path remain distinct", "feature_count": 4, "alignment_h_key": "H_v5_B_to_frame_native"}, "contract": {"schema": 1, "versions": {"corrected-rgb-texture-dynamic-v1": [["current_r", "current_g", "current_b", "delta_r", "delta_g", "delta_b", "current_intensity", "delta_intensity", "current_red_fraction", "current_green_fraction", "current_blue_fraction", "delta_red_fraction", "delta_green_fraction", "delta_blue_fraction", "log_r_over_g", "log_g_over_b", "log_r_over_b", "delta_log_r_over_g", "delta_log_g_over_b", "delta_log_r_over_b", "texture_highpass", "delta_texture_highpass", "rgb_change_norm"], ["dynamic_available", "dynamic_delta_r", "dynamic_delta_g", "dynamic_delta_b", "dynamic_delta_intensity", "dynamic_texture_delta", "dynamic_rgb_change_norm"]], "corrected-rgb-texture-dynamic-v2": [["current_r", "current_g", "current_b", "delta_r", "delta_g", "delta_b", "current_intensity", "delta_intensity", "current_red_fraction", "current_green_fraction", "current_blue_fraction", "delta_red_fraction", "delta_green_fraction", "delta_blue_fraction", "log_r_over_g", "log_g_over_b", "log_r_over_b", "delta_log_r_over_g", "delta_log_g_over_b", "delta_log_r_over_b", "texture_highpass", "delta_texture_highpass", "rgb_change_norm", "local_rgb_change_mean", "local_rgb_change_std", "local_rgb_change_coherence"], ["dynamic_available", "dynamic_delta_r", "dynamic_delta_g", "dynamic_delta_b", "dynamic_delta_intensity", "dynamic_texture_delta", "dynamic_rgb_change_norm", "local_dynamic_rgb_change_mean"]], "baseline-relative-texture-dynamic-v3": [["delta_r", "delta_g", "delta_b", "delta_intensity", "delta_red_fraction", "delta_green_fraction", "delta_blue_fraction", "delta_log_r_over_g", "delta_log_g_over_b", "delta_log_r_over_b", "delta_texture_highpass", "abs_delta_texture_highpass", "rgb_change_norm", "local_rgb_change_mean", "local_rgb_change_std", "local_rgb_change_coherence"], ["dynamic_available", "dynamic_delta_r", "dynamic_delta_g", "dynamic_delta_b", "dynamic_delta_intensity", "dynamic_texture_delta", "dynamic_rgb_change_norm", "local_dynamic_rgb_change_mean"]], "baseline-aware-bright-grain-v4": [["delta_r", "delta_g", "delta_b", "delta_intensity", "delta_red_fraction", "delta_green_fraction", "delta_blue_fraction", "delta_log_r_over_g", "delta_log_g_over_b", "delta_log_r_over_b", "delta_texture_highpass", "abs_delta_texture_highpass", "rgb_change_norm", "local_rgb_change_mean", "local_rgb_change_std", "local_rgb_change_coherence", "baseline_intensity", "baseline_signed_local_contrast", "current_signed_local_contrast", "delta_signed_local_contrast", "persistent_bright_contrast", "new_bright_contrast"], ["dynamic_available", "dynamic_delta_r", "dynamic_delta_g", "dynamic_delta_b", "dynamic_delta_intensity", "dynamic_texture_delta", "dynamic_rgb_change_norm", "local_dynamic_rgb_change_mean"]], "baseline-relative-bright-grain-v5": [["delta_r", "delta_g", "delta_b", "delta_intensity", "delta_red_fraction", "delta_green_fraction", "delta_blue_fraction", "delta_log_r_over_g", "delta_log_g_over_b", "delta_log_r_over_b", "delta_texture_highpass", "abs_delta_texture_highpass", "rgb_change_norm", "local_rgb_change_mean", "local_rgb_change_std", "local_rgb_change_coherence", "baseline_intensity", "baseline_signed_local_contrast", "current_signed_local_contrast", "delta_signed_local_contrast", "persistent_bright_contrast", "new_bright_contrast", "baseline_relative_local_contrast", "current_relative_local_contrast", "delta_relative_local_contrast", "persistent_relative_bright_contrast", "new_relative_bright_contrast", "relative_bright_contrast_agreement"], ["dynamic_available", "dynamic_delta_r", "dynamic_delta_g", "dynamic_delta_b", "dynamic_delta_intensity", "dynamic_texture_delta", "dynamic_rgb_change_norm", "local_dynamic_rgb_change_mean"]]}, "input_color_space": "radiometrically-corrected-rgb-0-1", "change_detectors": {"plain": "concentration-analysis-plain-background-difference-v1", "none": "none"}, "change_feature_names": ["delta_r", "delta_g", "delta_b", "rgb_change_norm", "local_rgb_change_mean", "local_rgb_change_std", "local_rgb_change_coherence", "dynamic_delta_r", "dynamic_delta_g", "dynamic_delta_b", "dynamic_rgb_change_norm", "local_dynamic_rgb_change_mean"], "chemistry_adapters": {"none": [], "btb-ratiometric-v1": ["btb_acidification", "delta_btb_acidification"], "btb-ratiometric-relative-v2": ["delta_btb_acidification"]}, "reference_adapters": {"none": [], "clearwater-projection-v1": ["clearwater_reference_reliability", "clearwater_projection", "clearwater_orthogonal_distance", "clearwater_alignment", "clearwater_projection_score", "clearwater_distance_from_baseline", "clearwater_distance_to_endpoint", "clearwater_texture_difference"]}, "bright_version": "baseline-aware-bright-grain-v4", "relative_bright_version": "baseline-relative-bright-grain-v5", "packed_versions": ["baseline-relative-texture-dynamic-v3", "baseline-aware-bright-grain-v4", "baseline-relative-bright-grain-v5"], "defaults": {"schema": 1, "generic_version": "baseline-aware-bright-grain-v4", "input_color_space": "radiometrically-corrected-rgb-0-1", "change_detector": "concentration-analysis-plain-background-difference-v1", "chemistry_adapter": "none", "reference_adapter": "none", "include_dynamic": true}}}, "heads": {"encoding": {"labelled": "pixel.astype(float16); regional resized INTER_LINEAR beforefloat16; decode eachfloat32", "compact_scale": 254, "invalid_code": 255, "compact_dtype": "uint8", "rounding": "np.rint nearest-even", "compact_decode_denominator": 254, "compact_decode_dtype": "float32", "regional_own_grid_dtype": "float16", "regional_decoded_dtype": "float32"}, "default_threshold": 0.5, "variants": {"current": ["G0prime", "G0\'", "G0\\u2032", "G1"], "flatfield_only": "G3", "combined": "G2"}, "regional_feature_names": ["fine_presence_mean", "fine_gas_joint_mean", "fine_positive_density", "fine_strong_density", "fine_red_fraction_change", "fine_green_fraction_change", "fine_blue_fraction_change", "fine_relative_intensity_change", "fine_chromatic_change_magnitude", "fine_density_times_change", "coarse_presence_mean", "coarse_gas_joint_mean", "coarse_positive_density", "coarse_strong_density", "coarse_red_fraction_change", "coarse_green_fraction_change", "coarse_blue_fraction_change", "coarse_relative_intensity_change", "coarse_chromatic_change_magnitude", "coarse_density_times_change"], "flat_feature_names": ["ff_log_r", "ff_log_g", "ff_log_b", "ff_log_y"], "flat_feature_count": 4, "score_chunk_rows": 128, "gas_head_name": "gas_within_co2", "epoch": {"values": ["rig_light", "new_lamps"], "positive_value": "new_lamps"}}, "regional": {"settings_from_original_models": true, "optical_default_settings": {"working_spacing_m": 0.001, "fine_sigma_m": 0.00075, "coarse_sigma_m": 0.002, "support_threshold": 0.5, "seed_threshold": 0.8, "minimum_seed_density": 0.05, "change_noise_multiplier": 3.0, "minimum_chromatic_change": 0.008}, "pooling": {"coverage_threshold": 0.999, "coverage_division_floor": 1e-06, "positive_probability": 0.5, "strong_probability": 0.9, "quiet_probability": 0.01, "quiet_min_count": 32, "rgb_fraction_floor": 0.01, "quiet_empty_drift_dtype": "numpy zeros defaultfloat64", "conditional_gas_columns": [1, 11], "scale_order": ["fine", "coarse"], "multiply_before_float32_pooling": true, "division_floor": 1e-06, "chromatic_floor": 0.01}, "original_profile_guards": {"regional_gas_threshold": 0.6, "island_min_area_mm2": 2.5, "co2_connectivity": 8, "gas_connectivity": 4, "keep_original_profile_selection": true}, "spacing_conventions": "inherited optical axes and G3-export axes boundseparately; nativebaseline/previous omittedmask conventions vs extended boundmasks retained; only share independentlybitidentical alignedarrays", "settings": {"working_spacing_m": 0.001, "fine_sigma_m": 0.00075, "coarse_sigma_m": 0.002, "support_threshold": 0.5, "seed_threshold": 0.8, "minimum_seed_density": 0.05, "change_noise_multiplier": 3.0, "minimum_chromatic_change": 0.008}, "settings_fields": ["working_spacing_m", "fine_sigma_m", "coarse_sigma_m", "support_threshold", "seed_threshold", "minimum_seed_density", "change_noise_multiplier", "minimum_chromatic_change"], "bounds": {"maximum_scale_m": 0.01, "maximum_guard_area_mm2": 25}, "feature_names": ["fine_presence_mean", "fine_gas_joint_mean", "fine_positive_density", "fine_strong_density", "fine_red_fraction_change", "fine_green_fraction_change", "fine_blue_fraction_change", "fine_relative_intensity_change", "fine_chromatic_change_magnitude", "fine_density_times_change", "coarse_presence_mean", "coarse_gas_joint_mean", "coarse_positive_density", "coarse_strong_density", "coarse_red_fraction_change", "coarse_green_fraction_change", "coarse_blue_fraction_change", "coarse_relative_intensity_change", "coarse_chromatic_change_magnitude", "coarse_density_times_change"], "conditional_columns": [1, 11], "model_format": "baseline-relative-optical-regions-v1", "formats": {"gas": "learned-regional-gas-confirmation-v1", "output": "regional-final-domain-islands-v1", "free_gas": "native-free-gas-minimum-v1"}, "profile_keys": {"gas": "regional_gas", "output": "output_guards", "free_gas": "free_gas_cleanup"}, "partition": {"presence_threshold": 0.5, "gas_threshold": 0.6}, "weighted_guard": {"minimum_area_mm2": 2.5, "connectivity": 8}, "guard_defaults": {"gas": {"format": "learned-regional-gas-confirmation-v1", "threshold": 0.6}, "output": {"format": "regional-final-domain-islands-v1", "minimum_island_area_mm2": 2.5}, "free_gas": {"format": "native-free-gas-minimum-v1", "minimum_area_mm2": 2.5}}}}')
BASE = Path(__file__).resolve().parents[2]/"src/darsia/fluidflower"


@lru_cache(maxsize=None)
def original_source(filename):
    """Fixed pre-extraction Git blobs remain references after facade switches."""
    return subprocess.check_output(['git','show','fb39a2bc0cf00e9d5d4f577715ecfaa1860412b8:src/darsia/fluidflower/'+filename],cwd=BASE.parents[2])


def original_namespace(filename, names=None, **extra):
    """Execute only original declared math definitions on synthetic inputs."""
    source=original_source(filename)
    parsed=ast.parse(source)
    namespace=dict(__name__='original_synthetic_phase',np=np,cv2=cv2,expit=expit,dataclass=dataclass,asdict=asdict,Any=Any,Mapping=Mapping,Protocol=Protocol,Iterator=Iterator,Sequence=Sequence,**extra)
    sys.modules.setdefault(namespace['__name__'],types.ModuleType(namespace['__name__']))
    selected=[]
    for n in parsed.body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and (names is None or n.name in names):selected.append(n)
        elif isinstance(n,(ast.Assign,ast.AnnAssign)):
            ids=[x.id for x in (n.targets if isinstance(n,ast.Assign) else [n.target]) if isinstance(x,ast.Name)]
            if ids and ids[0] in extra:continue
            if filename=='phase_features.py' and ids and ids[0] not in ('CHANGE_EVIDENCE_PROVIDERS','__all__'):selected.append(n)
    exec(compile(ast.Module(body=selected,type_ignores=[]),filename,'exec'),namespace)
    return types.SimpleNamespace(**namespace)


@pytest.fixture
def p():
    value=copy.deepcopy(PARAMETERS)
    value['regional']['pooling']['review_denominator']=255
    return value


def models(p):
    fp=p['features'];defaults=fp['contract']['defaults'];contract=features.OpticalFeatureContract(**defaults,parameters=fp)
    count=len(contract.feature_names)
    payload=dict(feature_contract=contract.to_dict(),feature_mean=np.linspace(-.2,.3,count).tolist(),feature_scale=np.linspace(.4,1.2,count).tolist(),coefficients=np.linspace(-.3,.7,count).tolist(),intercept=.17)
    model=heads.HierarchicalPhaseModel.from_mappings({'co2_presence':payload,'gas_within_co2':dict(payload,intercept=-.2)},adapter='generic',feature_parameters=fp,parameters=p['heads'])
    old=original_namespace('phase_features.py')
    oldcontract=old.OpticalFeatureContract.from_dict(contract.to_dict())
    oldheads=[types.SimpleNamespace(**dict(vars(h),feature_contract=oldcontract)) for h in (model.co2_presence,model.gas_within_co2)]
    oldmodel=types.SimpleNamespace(co2_presence=oldheads[0],gas_within_co2=oldheads[1],feature_contract=oldcontract)
    return model,oldmodel,old


def assert_exact(left,right):
    assert left.dtype==right.dtype and left.shape==right.shape
    assert np.array_equal(left,right,equal_nan=True)


def test_sensor_decoders_keep_distinct_original_rounding(p):
    oldnative=original_namespace('phase_ffac_selected_native_centre.py',{'decode_native_log'})
    oldrig=original_namespace('phase_rig_domain_inference.py',{'log_to_unit_rgb'})
    rng=np.random.default_rng(42);log=np.log(rng.uniform(3,60000,(3,5,7))).astype(np.float32);valid=np.ones((5,7),bool);valid[1,2]=False;log[:,1,2]=np.nan
    assert_exact(features.decode_native_log(log,valid,parameters=p['features']),oldnative.decode_native_log(log,valid))
    got,support=features.log_to_unit_rgb(log,valid,parameters=p['features']);expected,oldmask=oldrig.log_to_unit_rgb(log,valid)
    assert_exact(got,expected);assert np.array_equal(support,oldmask)
    with pytest.raises(ValueError):features.decode_native_log(log,np.ones_like(valid),parameters=p['features'])
    with pytest.raises(KeyError):features.decode_native_log(log,valid,parameters={})


@pytest.mark.parametrize('dynamic',[False,True])
def test_cpu_features_and_both_head_operations_exact(p,dynamic):
    model,oldmodel,oldfeatures=models(p)
    old=original_namespace('tools/phase_inference_benchmark_backends.py',{'packed_parameters','CudaFeaturePredictor','CpuFeaturePredictor'},OpticalFeatureContract=oldfeatures.OpticalFeatureContract,relative_bright_grain_columns=oldfeatures.relative_bright_grain_columns,GENERIC_FEATURE_VERSION=oldfeatures.GENERIC_FEATURE_VERSION,RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION=oldfeatures.RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION,BASELINE_RELATIVE_FEATURE_VERSION=oldfeatures.BASELINE_RELATIVE_FEATURE_VERSION,SPATIAL_CONTEXT_FRACTION=oldfeatures.SPATIAL_CONTEXT_FRACTION)
    rng=np.random.default_rng(8);baseline=rng.uniform(.02,.6,(9,11,3)).astype(np.float32);current=(baseline*.9).astype(np.float32);previous=(baseline*.97).astype(np.float32) if dynamic else None
    current_predictor=heads.CpuFeaturePredictor(model,baseline,parameters=p['heads'],feature_parameters=p['features'],block_rows=3,texture_sigma=2.,input_encoding=p['features']['optical']['unit_encoding'])
    old_predictor=old.CpuFeaturePredictor(oldmodel,baseline,block_rows=3,texture_sigma=2.,input_encoding=p['features']['optical']['unit_encoding'])
    for a,b in zip(current_predictor.feature_blocks(current_predictor.prepare_rgb(current),current_predictor.baseline if previous is None else current_predictor.prepare_rgb(previous),dynamic_available=dynamic),old_predictor.feature_blocks(old_predictor.prepare_rgb(current),old_predictor.baseline if previous is None else old_predictor.prepare_rgb(previous),dynamic_available=dynamic)):
        assert a[:2]==b[:2];assert_exact(a[2],b[2])
        assert_exact(heads.score_packed_heads(a[2],current_predictor.weights,current_predictor.bias),expit(b[2]@old_predictor.weights+old_predictor.bias))
        assert_exact(heads.score_packed_presence(a[2],current_predictor.weights,current_predictor.bias),expit(b[2]@old_predictor.weights[:,0]+old_predictor.bias[0]))
    got=current_predictor.predict(current,previous);expected=old_predictor.predict(current,previous)
    for key in got:assert_exact(got[key],expected[key])


def test_feature_contract_and_model_parsing_reject_changed_order(p):
    model,_,_=models(p);doc=model.feature_contract.to_dict();doc['feature_names']=list(reversed(doc['feature_names']))
    with pytest.raises(ValueError):features.OpticalFeatureContract.from_dict(doc,parameters=p['features'])
    with pytest.raises(ValueError):features.OpticalFeatureContract.from_dict(dict(doc,schema=99),parameters=p['features'])
    payload=dict(feature_contract=model.feature_contract.to_dict(),feature_mean=[0],feature_scale=[1],coefficients=[1],intercept=0)
    with pytest.raises(ValueError):heads.SerializedPhaseHead.from_mapping(payload,name='tiny',feature_parameters=p['features'],parameters=p['heads'])


def test_temporal_support_keeps_exact_old_warp_and_stops_extrapolation(p):
    old=original_namespace('tools/ffac_v5_temporal_alignment.py')
    image=np.linspace(.01,.5,8*10*3,dtype=np.float32).reshape(8,10,3);mask=np.ones((8,10),bool);mask[3,4]=False;H=np.array([[1.,0.,.25],[0.,1.,.5],[0.,0.,1.]])
    got=features.warp_reference_to_current(image,H,source_valid_mask=mask,parameters=p['features'], source_origin_xy=(0, 0));expected=old.warp_reference_to_current(image,H,source_valid_mask=mask)
    assert_exact(got[0],expected[0]);assert np.array_equal(got[1],expected[1])
    with pytest.raises(ValueError):features.warp_reference_to_current(image,H,required_valid_mask=np.ones_like(mask),source_valid_mask=mask,parameters=p['features'], source_origin_xy=(0, 0))
    a=features.align_temporal_inputs(baseline=image,current=image,previous=image,h_baseline_to_current=np.eye(3),h_baseline_to_previous=np.eye(3),parameters=p['features'], baseline_origin_xy=(0, 0), previous_origin_xy=(0, 0));b=old.align_temporal_inputs(baseline=image,current=image,previous=image,h_baseline_to_current=np.eye(3),h_baseline_to_previous=np.eye(3))
    assert_exact(a[0],b[0]);assert_exact(a[1],b[1]);assert a[2]==b[2]


def test_flatfield_tiles_preserve_halo_and_baseline_before_warp(p):
    temporal=original_namespace('tools/ffac_v5_temporal_alignment.py');fp=p['features']
    display=types.SimpleNamespace(M_NATURAL=np.asarray(fp['flatfield']['natural_matrix'],np.float64))
    old=original_namespace('phase_gas_flatfield.py',display=display,_h=temporal._h,BOX_SIZE=5,DENOMINATOR_FLOOR=np.float32(1e-6),_BOX_KERNEL=np.ones((5,5),np.uint8),_LUMINANCE=np.asarray(fp['flatfield']['luminance'],np.float32))
    rng=np.random.default_rng(1);log=np.log(rng.uniform(100,4000,(3,11,13))).astype(np.float32);valid=np.ones((11,13),bool);valid[5,6]=False
    a=features.prepare_flatfield_reference(log,source_valid_mask=valid,tile_rows=3,parameters=fp);b=old.prepare_flatfield_reference(log,source_valid_mask=valid,tile_rows=3)
    assert_exact(a.linear_rgb,b.linear_rgb);assert np.array_equal(a.valid,b.valid)
    H=np.array([[1.,0.,.25],[0.,1.,.125],[0.,0.,1.]])
    ar=features.register_flatfield_reference(a,H,target_shape_hw=valid.shape,parameters=fp, source_origin_xy=(0, 0), target_origin_xy=(0, 0));br=old.register_flatfield_reference(b,H,target_shape_hw=valid.shape)
    for key in ('rgb_box5','luminance_box5','valid','box_valid'):assert_exact(getattr(ar,key),getattr(br,key))
    aa=features.extract_flatfield_features(log,ar,current_valid_mask=valid,tile_rows=3,parameters=fp);bb=old.extract_flatfield_features(log,br,current_valid_mask=valid,tile_rows=3)
    assert_exact(aa[0],bb[0]);assert np.array_equal(aa[1],bb[1])
    with pytest.raises(ValueError):features.prepare_flatfield_reference(log,tile_rows=0,parameters=fp)


def test_probability_encodings_retain_ties_invalid_and_float16(p):
    old=original_namespace('phase_ffac_selected_native_centre.py',{'encode_decode_probabilities'})
    pixel=np.asarray([[.5/254,1.5/254,.7,np.nan],[2.5/254,.9999,0.,1.]],np.float32);valid=np.isfinite(pixel);region=np.asarray([[.59999,.60001]],np.float32)
    for labelled in (False,True):
        got=heads.encode_decode_probabilities(pixel,region,valid,labelled=labelled,parameters=p['heads']);expected=old.encode_decode_probabilities(pixel,region,valid,labelled=labelled)
        assert_exact(got[0],expected[0]);assert_exact(got[1],expected[1]);assert got[2]==expected[2]
    with pytest.raises(ValueError):heads.encode_decode_probabilities(pixel.astype(np.float64),region,valid,labelled=False,parameters=p['heads'])


def test_float64_gas_and_float32_regional_heads_keep_order(p):
    rng=np.random.default_rng(44);x=rng.normal(size=(6,4)).astype(np.float32);fit=dict(mean=[.1,.2,.3,.4],scale=[.2,.7,1.,1.3],coefficients=[.2,-.4,.5,1.3],intercept=.12);model=dict(variant=p['heads']['variants']['flatfield_only'],feature_names=p['heads']['flat_feature_names'],fit=fit)
    old=original_namespace('tools/train_ffac_gas_flatfield_variants.py',{'gas_probability'})
    assert_exact(heads.gas_probability(model,x,x,parameters=p['heads']),old.gas_probability(model,x,x))
    hp=dict(p['heads'],regional_feature_names=['a','b','c','d'])
    ref=original_namespace('phase_optical_regions.py',{'probability'},FEATURE_NAMES=('a','b','c','d'))
    assert_exact(heads.regional_probability(x,fit,parameters=hp),ref.probability(x,fit))
    model['fit']=dict(fit,scale=[0,1,1,1])
    with pytest.raises(ValueError):heads.gas_probability(model,x,x,parameters=p['heads'])


def test_logit_shift_and_epoch_projection_preserve_original_model(p):
    old=original_namespace('phase_gas_t4c_placements.py',{'pixel_logits','shifted_probability','_bool_grid'})
    joint=dict(feature_names=[*p['heads']['flat_feature_names'],'indicator'],fit=dict(mean=[.1]*5,scale=[.7]*5,coefficients=[.2,-.1,.3,.5,.4],intercept=.3));before=copy.deepcopy(joint)
    ref=original_namespace('phase_gas_t4b_regional_inputs.py',{'epoch_pixel_model'},deepcopy=copy.deepcopy,shared=types.SimpleNamespace(FLATFIELD_FEATURE_NAMES=p['heads']['flat_feature_names']))
    for epoch in p['heads']['epoch']['values']:
        a=heads.epoch_pixel_model(joint,epoch,parameters=p['heads']);b=ref.epoch_pixel_model(joint,epoch);assert a==b
        x=np.arange(5*7*4,dtype=np.float32).reshape(5,7,4)/100
        aa=heads.pixel_logits(a,x,chunk_rows=2,parameters=p['heads']);bb=old.pixel_logits(b,x,chunk_rows=2);assert_exact(aa,bb)
        valid=np.ones((5,7),bool);valid[2,3]=False
        assert_exact(heads.shifted_probability(aa,1.14,valid),old.shifted_probability(bb,1.14,valid))
    assert joint==before
    with pytest.raises(ValueError):heads.epoch_pixel_model(joint,'unknown',parameters=p['heads'])


def test_refit_keeps_non_gas_head_and_settings(p):
    frozen=dict(settings={'x':1},feature_names=['x'],heads={'co2_presence':{'value':1},'gas_within_co2':{'value':2}});changed=copy.deepcopy(frozen);changed['heads']['gas_within_co2']['value']=3
    heads.validate_refit(frozen,changed,parameters=p['heads'])
    changed['heads']['co2_presence']['value']=2
    with pytest.raises(ValueError):heads.validate_refit(frozen,changed,parameters=p['heads'])


@pytest.mark.parametrize('spacing',[.001,.0004])
def test_regional_preparation_and_shared_context_exact(p,spacing):
    rp=p['regional'];old=original_namespace('phase_optical_regions.py',VERSION=rp['model_format'],FEATURE_NAMES=tuple(rp['feature_names']))
    rng=np.random.default_rng(25);base=rng.uniform(.01,.7,(11,13,3)).astype(np.float32);cur=(base*.9).astype(np.float32);presence=np.full((11,13),.001,np.float32);presence[3:7,4:9]=.75;gas=np.full_like(presence,.61);domain=np.ones_like(presence,bool);domain[5,6]=False
    a=regional.prepare(cur,base,presence,gas,domain,(spacing,spacing),parameters=rp);b=old.prepare(cur,base,presence,gas,domain,(spacing,spacing),old.Settings(**rp['settings']))
    for key in a:
        if isinstance(a[key],np.ndarray):assert_exact(a[key],b[key])
        else:assert a[key]==b[key]
    legacy=original_namespace('phase_gas_regional_context.py',regional=old,GAS_DEPENDENT_COLUMNS=(1,11))
    ca=regional.build_shared_regional_context(cur,base,presence,domain,(spacing,spacing),parameters=rp);cb=legacy.build_shared_regional_context(cur,base,presence,domain,(spacing,spacing),old.Settings(**rp['settings']))
    ga=regional.prepare_conditional_gas(ca,gas,parameters=rp);gb=legacy.prepare_conditional_gas(cb,gas)
    assert_exact(ga['features'],gb['features']);assert_exact(ga['features'],b['features'])


def test_regional_reconstruction_and_original_guards_exact(p):
    rp=p['regional'];old=original_namespace('phase_optical_regions.py',VERSION=rp['model_format'],FEATURE_NAMES=tuple(rp['feature_names']))
    shape=(9,11);domain=np.ones(shape,bool);domain[0]=False;current=np.full((*shape,3),.4,np.float32);base=current*.9;presence=np.full(shape,.7,np.float32);gas=np.full(shape,.65,np.float32)
    fit=dict(mean=[0]*20,scale=[1]*20,coefficients=[0]*20,intercept=2.);model=dict(format=rp['model_format'],feature_names=rp['feature_names'],settings=rp['settings'],heads={'co2_presence':fit,'gas_within_co2':fit})
    a=regional.prepare(current,base,presence,gas,domain,(.001,.001),parameters=rp);b=old.prepare(current,base,presence,gas,domain,(.001,.001),old.Settings(**rp['settings']));raw=np.where(presence>=.5,np.where(gas>=.6,2,1),0).astype(np.uint8)
    got=regional.predict(a,model,raw,domain,parameters=rp);expected=old.predict(b,model,raw,domain);assert_exact(got[0],expected[0]);assert got[1]==expected[1]
    for filename,name,cls,key in [('phase_regional_gas.py','confirm_gas','GasSettings','gas'),('phase_region_output_guards.py','guard_labels','OutputSettings','output'),('phase_free_gas_output_guard.py','clean_labels','FreeGasSettings','free_gas')]:
        legacy=original_namespace(filename,{'Settings',name},VERSION=rp['formats'][key]);settings=legacy.Settings(**rp['guard_defaults'][key])
        args=(raw,np.full((4,5),.59,np.float32),domain) if key=='gas' else (raw,domain,(.001,.001)) if key=='output' else (raw,(.001,.001))
        got=getattr(regional,name)(*args,parameters=rp);expected=getattr(legacy,name)(*args,settings)
        assert_exact(got[0],expected[0])
        if isinstance(got[1],np.ndarray):assert_exact(got[1],expected[1])
        else:assert got[1]==expected[1]


def test_public_core_imports_and_docstrings():
    for module in (features,heads,regional):
        tree=ast.parse(Path(module.__file__).read_bytes())
        for n in tree.body:
            if isinstance(n,ast.FunctionDef) and not n.name.startswith('_'):assert ast.get_docstring(n),n.name
            if isinstance(n,ast.ImportFrom):assert not any(x in (n.module or '') for x in ('tools','studio','project','phase_training'))


def test_normalized_mask_filters_keep_constant_colour_without_hardware(p):
    old=original_namespace('phase_rig_domain_inference.py',{'normalized_gaussian'})
    data=np.full((11,13,3),.3,np.float32);valid=np.ones((11,13),bool);valid[4:7,5:8]=False
    got=features.normalized_gaussian(data,valid,1.7,parameters=p['features']);expected=old.normalized_gaussian(data,valid,1.7)
    assert_exact(got,expected)
    mask=np.zeros((11,13),bool)
    assert not features.normalized_gaussian(data,mask,1.7,parameters=p['features']).any()
    with pytest.raises(ValueError):features.normalized_gaussian(data,np.ones((3,4),bool),1.7,parameters=p['features'])


def test_weighted_islands_use_measured_area_and_inclusive_minimum(p):
    labels=np.zeros((7,9),np.uint8);labels[2:4,2:4]=1;labels[5,6]=2;labels[0,2:5]=1
    area=np.full(labels.shape,.625,np.float64);area[5,6]=2.4999;area[0]=10.
    result,audit=regional.weighted_co2_output_guard(labels,np.ones(labels.shape,bool),area,parameters=p['regional'])
    assert (result[2:4,2:4]==1).all()
    assert result[5,6]==0 and not result[0].any()
    assert audit['minimum_island_area_mm2']==2.5 and audit['area_weighted']
    assert labels[5,6]==2


def test_serialized_unpacked_head_preserves_original_sign_split(p):
    model,_,_=models(p);head=model.co2_presence
    old=original_namespace('phase_learning.py',{'SerializedPhaseHead'},OpticalFeatureContract=features.OpticalFeatureContract)
    oldhead=old.SerializedPhaseHead(head.name,head.feature_contract,head.mean,head.scale,head.coefficients,head.intercept,head.threshold,head.radiometry_profile)
    x=np.linspace(-100.,100.,6*len(head.mean),dtype=np.float32).reshape(6,-1)
    assert_exact(head.probability(x),oldhead.probability(x))


def test_original_co2_composition_preserves_all_profile_guards(p):
    rp=p['regional'];old=original_namespace('phase_optical_regions.py',VERSION=rp['model_format'],FEATURE_NAMES=tuple(rp['feature_names']))
    refs={}
    for key,filename,func in [('gas','phase_regional_gas.py','confirm_gas'),('output','phase_region_output_guards.py','guard_labels'),('free_gas','phase_free_gas_output_guard.py','clean_labels')]:
        refs[key]=original_namespace(filename,{'Settings',func},VERSION=rp['formats'][key],PROFILE_KEY=rp['profile_keys'][key])
    # The reference is the untouched original body with only imports injected from these exact source definitions.
    parsed=ast.parse(original_source('phase_ffac_selected_native_centre.py'));body=copy.deepcopy(next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='original_co2_labels'));body.body=[n for n in body.body if not isinstance(n,ast.ImportFrom)]
    env=dict(np=np,regional=old,gas=refs['gas'],guards=refs['output'],cleanup=refs['free_gas'])
    exec(compile(ast.Module(body=[body],type_ignores=[]),'<original-composition>','exec'),env)
    shape=(11,13);domain=np.ones(shape,bool);domain[0]=False
    base=np.full((*shape,3),.4,np.float32);current=base*.85;presence=np.full(shape,.8,np.float32);gas=np.full(shape,.7,np.float32)
    fit=dict(mean=[0]*20,scale=[1]*20,coefficients=[0]*20,intercept=2.)
    model=dict(format=rp['model_format'],feature_names=rp['feature_names'],settings=rp['settings'],heads={'co2_presence':fit,'gas_within_co2':fit})
    profile={rp['profile_keys'][key]:value for key,value in rp['guard_defaults'].items()}
    a=regional.original_co2_labels(current,base,presence,gas,domain,(.001,.001),model,profile,parameters=rp);b=env['original_co2_labels'](current,base,presence,gas,domain,(.001,.001),model,profile)
    assert_exact(a[0],b[0]);assert a[1]==b[1]
    assert (a[0][~domain]==255).all()


def test_masked_flatfield_retains_calibrate_before_warp_and_exact_ratio_log(p):
    fp=p['features'];temporal=original_namespace('tools/ffac_v5_temporal_alignment.py')
    display=types.SimpleNamespace(M_NATURAL=np.asarray(fp['flatfield']['natural_matrix'],np.float64))
    flat=original_namespace('phase_gas_flatfield.py',display=display,_h=temporal._h,BOX_SIZE=5,DENOMINATOR_FLOOR=np.float32(1e-6),_BOX_KERNEL=np.ones((5,5),np.uint8),_LUMINANCE=np.asarray(fp['flatfield']['luminance'],np.float32))
    parsed=ast.parse(original_source('phase_rig_domain_inference.py'));body=copy.deepcopy(next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='masked_flatfield_features'));body.body=[n for n in body.body if not isinstance(n,ast.ImportFrom)]
    env=dict(np=np,cv2=cv2,flat=flat,M_NATURAL=display.M_NATURAL);exec(compile(ast.Module(body=[body],type_ignores=[]),'<original-masked-flatfield>','exec'),env)
    rng=np.random.default_rng(10);log=np.log(rng.uniform(100.,4000.,(3,11,13))).astype(np.float32);valid=np.ones((11,13),bool);valid[[0,-1]]=False;valid[:,[0,-1]]=False
    H=np.eye(3)
    a=features.masked_flatfield_features(log,log,valid,source_baseline_log=log,h_baseline_to_current=H,source_baseline_valid=np.ones_like(valid),parameters=fp);b=env['masked_flatfield_features'](log,log,valid,source_baseline_log=log,h_baseline_to_current=H,source_baseline_valid=np.ones_like(valid))
    assert_exact(a[0],b[0]);assert np.array_equal(a[1],b[1])
