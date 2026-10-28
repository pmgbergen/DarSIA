"""Pure metadata contracts for the separate FF_AC camera/rig revision.

An old top-domain alias permits only a mask change. This contract instead
records every new source, frozen source asset and own geometric certificate.
Builders do not read labels, write files, publish candidates or approve mass.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

import numpy as np

SOURCE_KEY = "ffac_rig_camera_sources"
SOURCE_CONTRACT = "ffac-rig-camera-native-sources-v1"
INPUT_CONTRACT = "ffac-rig-camera-source-bindings-v1"
PROFILE_FORMAT = "ffac-rig-camera-physics-v1"
REVISION_CONTRACT = "ffac-rig-domain-camera-clock-v1"
QA_CONTRACT = "ffac-rig-camera-qa-v1"
ACCEPTANCE_CONTRACT = "ffac-rig-camera-geometry-acceptance-v1"
RIG_CONTRACT = "ffac-rig-domain-revision-v1"
SCIENTIFIC_CONTRACT = "ffac-camera-clock-full-native-scientific-v1"
EXTENDED_SCIENTIFIC_CONTRACT = "ffac-rig-frozen-raw-camera-ring-v1"
SOURCE_SHAPE = [2979,5472]
SOURCE_BOX = [416,496,5472,2979]
CANVAS_WH = [6304,3546]
ORIGIN_XY = [-416,-496]
CONFIGURATION = dict(regional_threshold=.6,pixel_threshold=.7,minimum_area_mm2=2.5,connectivity=4)
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SCIENTIFIC_FIELDS = (
    "v5_log_linear_rgb_path","v5_log_linear_rgb_sha256",
    "v5_log_linear_valid_mask_path","v5_log_linear_valid_mask_sha256",
    "v5_extended_ring_path","v5_extended_ring_sha256","v5_extended_pad_px","v5_extended_origin_xy",
    "v5_frame_receipt_path","v5_frame_receipt_sha256","v5_raw_path","v5_raw_sha256",
    "native_jpeg_sha256","v5_display_modes","v5_raw_detail_jpeg_path","v5_raw_detail_jpeg_sha256",
)


def require(condition,message):
    if not condition:
        raise ValueError(message)


def asset(value):
    require(isinstance(value,dict) and isinstance(value.get("path"),str) and bool(value["path"])
            and isinstance(value.get("sha256"),str) and _SHA.fullmatch(value["sha256"]),
            "Revision assets require an exact path and SHA256")
    return deepcopy(value)


def homography(value):
    h=np.asarray(value,np.float64)
    require(h.shape==(3,3) and np.isfinite(h).all() and abs(np.linalg.det(h))>1e-12
            and abs(h[2,2])>1e-12,"Invalid source-bound rig homography")
    return (h/h[2,2]).tolist()


def phase_grid(phase_shape_hw=None):
    """Declare the phase raster independently of the unchanged photograph canvas.

    Accepted geometry may append bottom rows. Changing the origin, width or
    source crop would need a different source contract and is rejected here.
    """
    shape=list(CANVAS_WH[::-1] if phase_shape_hw is None else phase_shape_hw)
    require(len(shape)==2 and all(type(value) is int for value in shape)
            and shape[0]>=CANVAS_WH[1] and shape[1]==CANVAS_WH[0],
            "Rig phase grid permits explicit bottom padding only")
    return dict(shape_hw=shape,native_origin_xy=deepcopy(ORIGIN_XY),
                photograph_canvas_wh=deepcopy(CANVAS_WH),
                source_native_shape_hw=deepcopy(SOURCE_SHAPE),
                source_content_box_xywh=deepcopy(SOURCE_BOX))


def validate_phase_grid(value):
    require(isinstance(value,dict) and value==phase_grid(value.get("shape_hw")),
            "Phase grid must bind its exact raster, origin and unchanged photographic crop")
    return deepcopy(value)


def validate_rig_descriptor(value):
    require(isinstance(value,dict) and value.get("contract")==RIG_CONTRACT
            and value.get("native_origin_xy")==ORIGIN_XY,
            "Rig classes/area must declare the full canvas and its exact native origin")
    shape=value.get("phase_shape_hw",CANVAS_WH[::-1])
    grid=phase_grid(shape)
    require(value.get("canvas_wh")==CANVAS_WH,
            "Rig descriptor changed its fixed photographic canvas")
    require(shape==CANVAS_WH[::-1] or value.get("phase_shape_hw")==shape,
            "Padded rig descriptors need an explicit phase_shape_hw")
    require(value.get("fixed_area_mm2")==.027053,"Rig revision changed the historical AC28 area key")
    for name in ("area_map","classes"):
        asset(value.get(name))
    require(value["area_map"].get("array_key","pixel_area_mm2")=="pixel_area_mm2",
            "Rig area map has another array key")
    require(value.get("mass_eligible") is False,"Rig area descriptor cannot approve mass")
    return deepcopy(value)


def validate_frame_binding(row,record,*,run,index,reference=False):
    require(isinstance(row,dict) and row.get("run")==run
            and row.get("frame_index")==index,"Source binding belongs to another frame")
    require(row.get("source_sha256")==record.get("source_sha256")
            and row.get("source_path")==record.get("source_path")
            and float(row.get("elapsed_h",math.nan))==float(record.get("elapsed_h",math.nan)),
            "Source binding changed its camera image or elapsed time")
    require(math.isfinite(float(record["elapsed_h"])) and isinstance(record["source_sha256"],str)
            and _SHA.fullmatch(record["source_sha256"]),"Source timeline has invalid time/SHA")
    asset(row.get("geometry_receipt"))
    homography(row.get("H_ac28_to_frame_native"))
    if not reference:
        homography(row.get("H_v5_B_to_frame_native"))
        require(type(row.get("source_changed")) is bool,"Source binding lacks explicit source-change state")
        old=row.get("parent_source_frame_index")
        require((row["source_changed"] and old is None and row.get("own_source_geometry") is True)
                or (not row["source_changed"] and type(old) is int and old>=0),
                "Changed sources need their own geometry; unchanged sources need an exact parent index")
    grid=validate_phase_grid(row.get("phase_grid"))
    require(row.get("rig_domain_revision")==record.get("rig_domain_revision"),
            "Source and area/class bindings differ")
    validate_rig_descriptor(row["rig_domain_revision"])
    require(row["rig_domain_revision"].get("phase_shape_hw",CANVAS_WH[::-1])==grid["shape_hw"],
            "Source binding and rig descriptor disagree on the phase raster")
    full=record.get("full_native_scientific")
    require(full==row.get("full_native_scientific") and isinstance(full,dict)
            and full.get("contract")==SCIENTIFIC_CONTRACT and full.get("source_sha256")==record["source_sha256"]
            and full.get("native_shape_hw")==SOURCE_SHAPE and full.get("unit_denominator")==65535,
            "Full-native scientific source needs its own exact source/correction contract")
    for name in ("log_linear_rgb","valid_mask","receipt"):
        asset(full.get(name))
    extended=record.get("full_extended_scientific")
    require(extended==row.get("full_extended_scientific"),
            "Source and extended RAW-ring bindings differ")
    if extended is not None:
        require(isinstance(extended,dict) and extended.get("contract")==EXTENDED_SCIENTIFIC_CONTRACT
                and extended.get("complete") is True and extended.get("source_sha256")==record["source_sha256"]
                and extended.get("phase_shape_hw")==grid["shape_hw"] and extended.get("native_origin_xy")==ORIGIN_XY
                and extended.get("native_centre")==full and extended.get("unit_denominator")==65535
                and extended.get("storage")=="exact-native-centre-plus-disjoint-frozen-raw-ring-windows",
                "Extended RAW ring lost its source, raster, correction or exact native centre")
        for window in extended.get("ring_windows",[]):asset(window.get("log_linear_rgb"))
    cache=full.get("parent_cache")
    if reference or row.get("source_changed") is False:
        require(isinstance(cache,dict) and cache.get("log_linear_rgb")==dict(
            path=record["v5_log_linear_rgb_path"],sha256=record["v5_log_linear_rgb_sha256"])
            and cache.get("valid_mask")==dict(path=record["v5_log_linear_valid_mask_path"],
                sha256=record["v5_log_linear_valid_mask_sha256"]),
            "Expanded scientific source lost its exact frozen parent cache")
    else:
        require(cache is None,"New camera acquisition cannot claim another source's parent cache")
    photos=row.get("photographs") or {}
    require(set(photos)=={"natural","lift1","lift2","change","raw_detail","original"},
            "Source revision requires all six explicit photographic views")
    for mode,value in photos.items():
        asset(value)
    require(photos["original"]==dict(path=record["source_path"],sha256=record["source_sha256"]),
            "Original display is not the bound camera JPEG")
    prefix="v5_baseline" if reference else "v5_frame"
    for prefix_key in ("v5_log_linear_rgb","v5_log_linear_valid_mask",prefix+"_receipt"):
        asset(dict(path=record.get(prefix_key+"_path"),sha256=record.get(prefix_key+"_sha256")))
    require((record.get("v5_display_modes") or {}).get("natural",{}).get("jpeg_sha256")
            ==photos["natural"]["sha256"],"Natural photograph differs from the scientific source record")
    for mode in ("natural","lift1","lift2"):
        declaration=(record.get("v5_display_modes") or {}).get(mode) or {}
        require(declaration.get("jpeg_path")==photos[mode]["path"]
                and declaration.get("jpeg_sha256")==photos[mode]["sha256"],
                "Photographic mode differs from its bound source declaration")
    require(record.get("v5_raw_detail_jpeg_path")==photos["raw_detail"]["path"]
            and record.get("v5_raw_detail_jpeg_sha256")==photos["raw_detail"]["sha256"],
            "RAW detail differs from its bound source declaration")
    return deepcopy(row)


def build_native_metadata(parent_native,camera_run,records,bindings,reference_binding,*,input_manifest):
    """Return a new native document; inputs and all old label/source records stay intact."""
    run=str(parent_native.get("run","")).casefold()
    require(run==camera_run.get("run") and run.startswith("ac"),"Camera plan belongs to another run")
    require(parent_native.get("canvas_wh")==CANVAS_WH
            and parent_native.get("analysis_image_shape_hw")==SOURCE_SHAPE
            and parent_native.get("content_box_xywh")==SOURCE_BOX,"Parent source geometry changed")
    selected=camera_run.get("frames") or []
    require(len(records)==len(bindings)==len(selected)==camera_run.get("frame_count"),
            "Source metadata does not cover the reselected camera timeline")
    result=deepcopy(parent_native)
    for old_marker in ("v5_native_source_alias","v5_track1_photo_geometry","v5_frame_geometry"):
        result.pop(old_marker,None)
    result["frame_sources"]=deepcopy(records)
    result["frame_count"]=len(records)
    for index,(new,selected_row,binding) in enumerate(zip(records,selected,bindings)):
        require(new.get("frame_index")==index and new.get("source_sha256")==selected_row["source_sha256"]
                and new.get("source_path")==selected_row["source_path"]
                and float(new.get("elapsed_h",math.nan))==float(selected_row["elapsed_h"]),
                "Materialized frame changed the camera-clock selection")
        old_index=selected_row.get("retained_old_frame_index")
        if old_index is not None:
            old=parent_native["frame_sources"][old_index]
            require(old["source_sha256"]==new["source_sha256"],"Retained source is not the same camera image")
            require(all(new.get(key)==old.get(key) for key in _SCIENTIFIC_FIELDS),
                    "Unchanged camera source must retain all frozen scientific/photographic assets")
            require(binding.get("parent_source_frame_index")==old_index
                    and binding.get("source_changed") is False,"Retained source lost its original index")
            require(binding.get("temporal_input_policy")=="retain-parent-predecessor-by-source-sha"
                    and binding.get("temporal_predecessor")==selected_row.get("frozen_temporal_predecessor"),
                    "Retained source changed its original source-specific temporal predecessor")
        else:
            require(binding.get("source_changed") is True,"New acquisition was incorrectly declared unchanged")
            previous=records[index-1] if index>1 else None
            predecessor=binding.get("temporal_predecessor")
            require(binding.get("temporal_input_policy")=="camera-clock-prior-source"
                    and ((previous is None and predecessor is None) or
                         (isinstance(predecessor,dict) and predecessor.get("source_sha256")==previous["source_sha256"]
                          and predecessor.get("source_path")==previous["source_path"]
                          and predecessor.get("full_native_scientific")==previous["full_native_scientific"])),
                    "New acquisition changed its declared camera-clock temporal predecessor")
        result["frame_sources"][index]["temporal_input_policy"]=binding["temporal_input_policy"]
        result["frame_sources"][index]["temporal_predecessor"]=deepcopy(binding["temporal_predecessor"])
        validate_frame_binding(binding,new,run=run,index=index)
    reference=deepcopy(parent_native.get("reference") or {})
    reference["rig_domain_revision"]=deepcopy(reference_binding["rig_domain_revision"])
    reference["full_native_scientific"]=deepcopy(reference_binding["full_native_scientific"])
    if "full_extended_scientific" in reference_binding:
        reference["full_extended_scientific"]=deepcopy(reference_binding["full_extended_scientific"])
    result["reference"]=reference
    validate_frame_binding(reference_binding,reference,run=run,index=None,reference=True)
    grids=[validate_phase_grid(row.get("phase_grid")) for row in [reference_binding,*bindings]]
    require(all(grid==grids[0] for grid in grids),"One camera run contains different phase rasters")
    result[SOURCE_KEY]=dict(contract=SOURCE_CONTRACT,input_manifest=asset(input_manifest),
        source_native_shape_hw=SOURCE_SHAPE,source_content_box_xywh=SOURCE_BOX,
        phase_grid=grids[0],labels_source_owned=True,inference_authorized=False)
    return result


def build_source_bindings(parent_native_asset,camera_run_asset,frames,reference,*,spec_sha256):
    """Input certificate is written before the child native SHA, avoiding hash cycles."""
    require(_SHA.fullmatch(str(spec_sha256)),"Source bindings lack the exact combined specification SHA")
    return dict(contract=INPUT_CONTRACT,spec_sha256=spec_sha256,
        parent_native_manifest=asset(parent_native_asset),camera_timeline_run=asset(camera_run_asset),
        frames=deepcopy(frames),reference=deepcopy(reference),labels_source_owned=True,
        source_native_grid_unchanged=True,old_native_sources_modified=False)


def validate_campaign_bindings(sources):
    """Require all 43 runs/864 unique contiguous selected frame positions."""
    require(isinstance(sources,list) and len(sources)==43,"Rig/camera revision requires 43 source runs")
    keys=set();runs=set()
    for source in sources:
        run=source.get("run")
        require(run not in runs and isinstance(run,str) and re.fullmatch(r"ac\d+",run),"Duplicate or invalid source run")
        runs.add(run)
        for index,row in enumerate(source.get("frames") or []):
            require(row.get("run")==run and row.get("frame_index")==index,"Camera source sequence is not contiguous")
            keys.add((run,index))
    require(len(keys)==864,"Rig/camera revision requires all 864 selected sources")
    return keys


def public_profile(profile):
    """Serialize public physics only; private tuple-key caches are derived at read time."""
    return {key:deepcopy(value) for key,value in profile.items() if not key.startswith("_")}


def build_public_profile(parent_identity,parent_profile,revision_manifest,*,spec_sha256,heads,phase_shape_hw=None):
    require(isinstance(parent_identity,list) and len(parent_identity)==2
            and _SHA.fullmatch(parent_identity[1]),"Missing exact parent model identity")
    require(set(heads)=={"co2","g3","regional"},"Rig/camera revision requires the exact three frozen heads")
    for value in heads.values():
        asset(value)
    require((parent_profile.get("approved_g3_gas") or {}).get("settings",{}).get("pixel_threshold")==.7
            and parent_profile.get("gas_barrier_rule_b") and parent_profile.get("photo_sand_top_revision"),
            "Combined revision requires the approved G3/top/Rule B parent")
    return dict(format=PROFILE_FORMAT,spec_sha256=spec_sha256,parent_model_identity=deepcopy(parent_identity),
        inherited_public_profile=public_profile(parent_profile),frozen_heads=deepcopy(heads),
        configuration=deepcopy(CONFIGURATION),rig_domain_revision=dict(format=REVISION_CONTRACT,
            manifest=asset(revision_manifest),phase_grid=phase_grid(phase_shape_hw),source_contract=SOURCE_CONTRACT),
        publication_scope="studio_validation_only",mass_eligible=False,scientific_acceptance=False,
        deployment_allowed=False,automatic_activation=False,training=False,labels_changed=False)


def validate_geometry_acceptance(value,*,geometry_sha256,spec_sha256):
    require(value.get("contract")==ACCEPTANCE_CONTRACT and value.get("accepted") is True
            and value.get("geometry_manifest_sha256")==geometry_sha256
            and value.get("spec_sha256")==spec_sha256 and value.get("run_count")==43
            and value.get("frame_count")==864 and value.get("visible_edges_within_3px") is True
            and type(value.get("physical_scale_approved")) is bool
            and value.get("relative_geometry_certified") is True
            and value.get("reference_correction_status") in ("applied","not-required","provisional-diagnostic"),
            "Geometry is not accepted for this exact complete rig/camera revision")
    asset(value.get("rig_check_measurement"))
    asset(value.get("relative_geometry_certificate"))
    return deepcopy(value)
