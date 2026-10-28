"""Per-acquisition geometry from an actual Build and its approved map fields."""
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from ..geometry.context import RigFrameContext, homography, photographic_observation_mask, rig_plane_area_map
from ..mass.native_optical_source import _same
from ..mass.n1_pipeline import _write_json, _write_npz, asset

FORMAT = "darsia-build-frame-geometry-v1"
MASKS = ("rig_domain", "modelled", "filled_observed", "filled_unobserved", "hardware_excluded")


def _arrays(inside, excluded, valid, matrix, frame_to_metric, renderer, phase):
    parameters = phase["geometry_parameters"]
    geometry = phase["parameters"]["geometry"]
    origin = tuple(geometry["rig_origin_xy"])
    box = tuple(geometry["source_box_xywh"])
    height, width = phase["phase_shape_hw"]
    shape = (height, width)
    if (valid.dtype != bool or list(valid.shape) != geometry["native_shape_hw"]
            or box[2:] != tuple(reversed(valid.shape)) or origin != (-box[0], -box[1])
            or min(box[:2]) < 0 or box[0] + box[2] > width or box[1] + box[3] > height):
        raise ValueError("The original phase canvas differs from the measured RAW frame")
    canvas = np.eye(3)
    canvas[:2, 2] = -np.asarray(origin)
    transform = homography(canvas @ matrix, parameters=parameters)

    def warp(mask):
        return cv2.warpPerspective(np.asarray(mask, np.uint8), transform, (width, height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)

    outer = warp(inside)
    by_name = {name: warp(mask) & outer for name, mask in excluded.items()}
    hardware = np.zeros(shape, bool)
    for mask in by_name.values():
        hardware |= mask
    domain = outer & ~hardware
    observed = photographic_observation_mask(renderer["native_to_camera_u_h"],
        renderer["camera_u_wh"], [width, height], native_origin_xy=origin, parameters=parameters)
    supported = np.zeros(shape, bool)
    x, y, w, h = box
    supported[y:y + h, x:x + w] = valid
    observed |= supported
    values = dict(rig_domain=domain, modelled=domain & supported,
        filled_observed=domain & observed & ~supported,
        filled_unobserved=domain & ~observed, hardware_excluded=hardware,
        observed=observed, top_keep=outer.copy(), native_valid=valid)
    area = rig_plane_area_map(frame_to_metric, shape, origin_xy=origin,
        metric_pixel_area_mm2=1., parameters=parameters)
    return values, by_name, area


def prepare_build_frame_geometry(build, native_frame, *, map_product, output):
    """Produce own-frame masks and metric area without borrowing another image.

    The approved map's independent ``inside`` field defines the physical frame;
    its explicitly excluded facies define hardware/occlusion. Material label zero
    inside that field remains reported as unknown. Photographic observation and
    native support come from this acquisition's original RAW producer.
    """
    from .local_build import LocalBuildInputs
    if not isinstance(build, LocalBuildInputs):
        raise TypeError("Actual local Build inputs are required")
    reader = build.reader
    source = reader.archive(map_product, ("array", "inside"))
    if (source["inside"].dtype != bool or source["inside"].shape != build.context.labels.shape
            or not np.array_equal(source["array"], build.context.labels)):
        raise ValueError("The approved map differs from the actual saved Build")
    receipt = reader.json(native_frame.receipt)
    native = reader.json(native_frame.native_task["geometry"])
    from .current_build import _physical_context_equal
    if (not _physical_context_equal(receipt["physical_context"], build.context.provenance)
            or not _same(receipt["native_frame"], native_frame.native_task["geometry"])
            or native["run"] != build.context.run):
        raise ValueError("The RAW frame does not belong to the selected actual Build")
    valid = reader.archive(native_frame.products, ("valid_mask",))["valid_mask"]
    matrix = homography(receipt["build_to_frame_native_h"], parameters=build.geometry_parameters)
    metric = np.diag([1000., 1000., 1.]) @ build.build_to_m @ np.linalg.inv(matrix)
    excluded = {"facies_" + str(key): source["inside"] & (build.context.facies == key)
        for key, role in build.roles.items()
        if role["analysis_role"] in {"excluded", "colorchecker"}}
    phase = build.phase
    masks, by_name, area = _arrays(source["inside"], excluded, valid, matrix, metric,
                                  receipt["correction"]["renderer"], phase)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    fields = _write_npz(destination / "geometry.npz", dict(masks,
        **{"excluded_" + key: value for key, value in by_name.items()}, pixel_area_mm2=area))
    gp = phase["parameters"]["geometry"]
    references = phase["geometry_fields"]
    row = dict(format=FORMAT, run=native["run"], raw_source=deepcopy(native["acquisition"]),
        native_origin_xy=deepcopy(gp["rig_origin_xy"]),
        source_native_shape_hw=deepcopy(gp["native_shape_hw"]),
        source_content_box_xywh=deepcopy(gp["source_box_xywh"]),
        phase_shape_hw=list(area.shape), frame_photo_top_keep=dict(fields, array_key="top_keep"),
        **{references["reference_to_frame"]: deepcopy(native["reference_to_native_h"])},
        area_parameters={references["reference_to_frame"]: deepcopy(native["reference_to_native_h"]),
                         references["reference_to_metric"]: (metric @ np.asarray(native["reference_to_native_h"])).tolist()},
        build_geometry=dict(operation_result=deepcopy(build.operation_result),
            metric_model=deepcopy(build.metric_model), map_product=deepcopy(map_product),
            bridge_frame=deepcopy(native_frame.receipt), native_frame=deepcopy(native_frame.native_task["geometry"]),
            fields=fields, excluded_names=list(by_name), frame_to_metric_mm=metric.tolist(),
            source_to_frame_native=matrix.tolist(),
            producer=asset(Path(__file__)), physical_qualification_claimed=False))
    for bound in (map_product, native_frame.receipt, native_frame.products, build.metric_model):
        reader.path(bound)
    return _write_json(destination / "geometry.json", row)


def load_build_frame_geometry(row, native_valid, audit, phase, reader):
    """Load this acquisition's generated geometry for the original phase loop."""
    if row.get("format") != FORMAT:
        raise ValueError("Unknown Build frame geometry")
    from ..storage_paths import resolve_storage_path
    from .series import same_raw_source
    saved = row["build_geometry"]
    receipt = reader.json(saved["bridge_frame"])
    native = reader.json(saved["native_frame"])
    if (not _same(native["acquisition"], row["raw_source"])
            or not same_raw_source(audit["sources"]["camera_arw"], row["raw_source"], resolve_storage_path)
            or not _same(receipt["correction"]["renderer"], audit["renderer"])):
        raise ValueError("Build geometry differs from this RAW acquisition/renderer")
    fields = reader.archive(saved["fields"], (*MASKS, "observed", "top_keep", "native_valid", "pixel_area_mm2",
        *("excluded_" + name for name in saved["excluded_names"])))
    shape = tuple(row["phase_shape_hw"])
    if (not np.array_equal(fields["native_valid"], native_valid)
            or any(fields[key].dtype != bool or fields[key].shape != shape for key in (*MASKS, "observed", "top_keep"))
            or fields["pixel_area_mm2"].dtype != np.float32
            or fields["pixel_area_mm2"].shape != shape
            or not np.isfinite(fields["pixel_area_mm2"]).all()
            or np.any(fields["pixel_area_mm2"] <= 0)
            or np.any(fields["modelled"] & fields["filled_observed"])
            or np.any(fields["modelled"] & fields["filled_unobserved"])
            or np.any(fields["filled_observed"] & fields["filled_unobserved"])
            or np.any(fields["hardware_excluded"] & fields["rig_domain"])
            or not np.array_equal(fields["rig_domain"], fields["modelled"] | fields["filled_observed"] | fields["filled_unobserved"])):
        raise ValueError("Build geometry lost its original RAW support/physical partition")
    gp = phase["parameters"]["geometry"]
    context = RigFrameContext(*(fields[key] for key in MASKS),
        {name: fields["excluded_" + name] for name in saved["excluded_names"]},
        tuple(gp["rig_origin_xy"]), tuple(gp["native_shape_hw"]), tuple(gp["source_box_xywh"]),
        fields["pixel_area_mm2"], dict(format=FORMAT, source=deepcopy(saved)))
    return context, fields["observed"], fields["top_keep"]
