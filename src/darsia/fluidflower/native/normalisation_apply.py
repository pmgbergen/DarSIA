"""Historical keep/raw lighting application, only for isolated B1 reproduction.

The scientific operations follow Track1V4R5SourceProcessor.correct in the
preserved source SHA 341b239d9d1933259a34952d10532e3973f981546b2a3e6fec10585fc30aa598.
Current SourceProcessor still validates the complete input cohort under its
own runtime policy; this separate calculation never calls or alters its correct
method. Presentation and production processing are outside this replay.
"""

import json
from typing import NamedTuple

import cv2
import numpy as np

from .io import sha256_file, resolve_asset_path
from .photometry import correct_frame_native
from .registration import registered_edge_aware_sensor_linear, register_chart
from .source import source_homography, chart_baseline_coordinates
from .support import source_photo_grid, project_source_photo_safe_core


class ReproducedFrame(NamedTuple):
    log_linear_rgb: np.ndarray
    required_valid_mask: np.ndarray


def apply_historical_frame(processor, *, run, frame_id, camera_jpeg, camera_arw,
                           review_jpeg, source_manifest_sha256, content_shape_hw,
                           content_box_xywh):
    """Apply the original equations to a validated cohort and reproduced fits.

    IPP is required, never enabled here. The caller owns the dedicated B1 child
    process and must restore its current policy before any other calculation.
    """
    if not cv2.ipp.useIPP():
        raise RuntimeError("Historical B1 application requires IPP in its separate process.")
    config = processor.parameters
    source = config["source"]
    keep = source["policy_ids"]["keep"]
    if (processor.enclosed_hole_policy != keep or processor.isolated_invalid_policy != keep
            or processor.display_mode != source["policy_ids"]["raw_display"]):
        raise ValueError("Historical B1 application requires the saved keep/raw choices.")
    if (run != processor.run or source_manifest_sha256 != processor.row["source_manifest_sha256"]
            or tuple(content_shape_hw) != tuple(source["native_shape_hw"])
            or tuple(content_box_xywh) != tuple(source["content_box_xywh"])):
        raise ValueError("V4-R5 source/geometry binding changed")
    if not frame_id.startswith("frame_"):
        raise ValueError("Historical B1 application requires a selected timeline frame.")
    frame = processor.cohort.frame(int(frame_id.split("_")[1]))
    path = lambda value: resolve_asset_path(value, source["asset_aliases"])
    expected = dict(arw_sha256=sha256_file(path(camera_arw)),
        camera_jpeg_sha256=sha256_file(path(camera_jpeg)),
        review_native_jpeg_sha256=sha256_file(path(review_jpeg)))
    if (expected["camera_jpeg_sha256"] != frame["camera_jpeg_sha256"]
            or expected["review_native_jpeg_sha256"] != frame["native_review_jpeg_sha256"]):
        raise ValueError("Timeline source differs from photo/H receipt")
    safe_grid = source_photo_grid(path(frame["photo_sand_reference_mask_path"]),
        frame["photo_sand_reference_mask_sha256"], parameters=config["support"])
    h_ref_frame = source_homography(frame[source["symbols"]["reference_to_frame_h_key"]], parameters=config)
    h_b_frame = source_homography(frame["H_v5_B_to_frame_native"], parameters=config)
    h_ref_b = source_homography(np.linalg.inv(h_b_frame) @ h_ref_frame, parameters=config)
    safe = project_source_photo_safe_core(safe_grid, h_ref_frame,
        shape_hw=tuple(content_shape_hw), enclosed_hole_policy=keep,
        parameters=config["support"], guard_px=source["mask_guard_px"])
    linear, renderer = registered_edge_aware_sensor_linear(camera_arw, camera_jpeg, review_jpeg,
        expected_source_hashes=expected, content_box=content_box_xywh,
        parameters=config["registration"], source_id=source["renderer_source_ids"][run],
        profile_id=source["profiles"][run])
    exact_b = (expected["arw_sha256"] == frame["v5_B_arw_sha256"]
               and expected["camera_jpeg_sha256"] == frame["v5_B_camera_jpeg_sha256"])
    xy = usable = None
    if not exact_b:
        certificate = json.loads(path(frame["v5_B_registration_certificate_path"]).read_text(encoding="utf-8-sig"))
        if certificate.get("sources", {}).get("review_native_jpeg_sha256") != frame["v5_B_native_jpeg_sha256"]:
            raise ValueError("Baseline chart registration belongs to another B")
        xy, usable, _ = register_chart(
            baseline_camera_jpeg=path(frame["v5_B_camera_jpeg_path"]),
            baseline_camera_sha256=frame["v5_B_camera_jpeg_sha256"],
            frame_camera_jpeg=camera_jpeg, frame_camera_sha256=expected["camera_jpeg_sha256"],
            baseline_chart_xy8=chart_baseline_coordinates(processor.fit, processor.grid, h_ref_b, parameters=config),
            baseline_chart_log_rgb=processor.fit.CB,
            baseline_review_to_camera_h=np.asarray(certificate["review_to_camera_h"], np.float64),
            frame_review_to_camera_h=np.asarray(renderer["review_to_camera_h"], np.float64),
            frame_native_linear_rgb=linear, parameters=config["registration"])
    corrected, _ = correct_frame_native(linear, processor.fit, processor.reference, processor.grid,
        h_ref_frame, safe, exact_baseline_source=exact_b, frame_chart_xy8=xy,
        frame_chart_usable=usable, old_field_ref8=processor.old_field, parameters=config["photometry"])
    # The original keep/raw output intersects optical and finite support without
    # filling, clipping, resampling, or converting the float32 scientific array.
    effective = safe & np.isfinite(corrected).all(axis=0)
    scientific = corrected.copy()
    scientific[:, ~effective] = np.nan
    if not effective.any():
        raise ValueError("V4-R5 has no source-supported corrected sand")
    if np.nanmax(scientific) > np.log(source["signal_denominator"]):
        raise ValueError("V4-R5 corrected linear signal exceeds fixed denominator")
    return ReproducedFrame(scientific, effective)
