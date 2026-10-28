"""Change photograph geometry coordinates using two exact RAW renderer audits.

This producer writes the existing SeriesInputs geometry-row and top_keep DATA
formats. It neither measures missing walls/hardware nor qualifies any physical,
optical, phase, training or mass domain. Historical QA and uncertainty remain
literal historical DATA. A matching array size grants no coordinate authority.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from ..geometry.photographic_data import homography, top_curve_mask, transform_points
from ..mass.n1_pipeline import DataReader, _write_json, _write_npz, asset
from ..mass.native_optical_source import _bound, _same
from ..native import raw_positioning, raw_rig_source, source
from .raw_acquisition import FORMAT as RAW_ACQUISITIONS


def prepare_geometry_data(*, old_geometry, old_raw_audit, old_acquisition_collection, old_native_configuration,
                          new_bridge_frame, native_configuration, phase_configuration,
                          phase_shape_hw, output, reader=None):
    """Produce current-grid DATA only, refusing an unproved RAW coordinate join.

    Every argument naming a file is its original path/SHA/size descriptor.
    ``old_geometry`` already declares independently sourced rigid-wall geometry,
    photograph-derived top curve and photographed hardware. ``old_raw_audit``
    is the completed original RAW correction of that same acquisition on the
    geometry's old native grid; a JPEG renderer is never a substitute.
    ``old_acquisition_collection`` is the original raw_acquisition.prepare
    collection, joining that audit to its consumed configuration and geometry.
    ``new_bridge_frame`` is the original BuildNativeBridge frame receipt.
    The complete reporting canvas is explicit DATA, not inferred from a crop.

    No RAW acquisition or RGB/physical raster is decoded here. Only the bound
    one-dimensional photographic top curve is loaded. Existing source receipts
    remain unchanged; the output retains their scope and the exact coordinate
    transport separately. The caller still owns scientific qualification.
    """
    context = _coordinate_context(old_geometry=old_geometry, old_raw_audit=old_raw_audit,
        old_acquisition_collection=old_acquisition_collection, old_native_configuration=old_native_configuration,
        new_bridge_frame=new_bridge_frame, native_configuration=native_configuration,
        phase_configuration=phase_configuration, phase_shape_hw=phase_shape_hw, reader=reader)
    reader, inputs, row = (context[key] for key in ("reader", "inputs", "row"))
    reference, shape, origin, transform = (context[key] for key in ("reference", "shape", "origin", "transform"))
    native = context["native"]
    top, curve_binding = _photographic_inputs(context)
    inputs["top_curve"] = deepcopy(curve_binding)
    curve = reader.array(curve_binding)
    keep, _ = top_curve_mask(curve, reference, shape, origin)
    result = _transported_row(context)
    # Recheck actual consumed source descriptors before the first output. The
    # RAW itself and old/new image rasters are not consumed by this DATA adapter.
    consumed_paths = [reader.path(binding) for binding in inputs.values()]
    output = Path(output)
    from ..job_filesystem import inspect_local_file
    from ..rig_raw_write import _no_links
    if not output.is_absolute() or output.exists() or not inspect_local_file(output / ".geometry-data-location").allowed:
        raise ValueError("Geometry DATA needs one fresh positively local output folder")
    _no_links(output.parent)
    protected_paths = consumed_paths + [Path(binding["path"]) for binding in inputs.values()]
    if any(output.resolve() == path.resolve().parent
           or output.resolve().is_relative_to(path.resolve().parent)
           for path in protected_paths):
        raise ValueError("Geometry DATA output overlaps its source folders")
    output.mkdir()
    produced_top = _write_npz(output / "photo_top.npz", dict(top_keep=keep))
    result["frame_photo_top_keep"] = dict(top, **produced_top, array_key="top_keep", shape_hw=shape)
    result["coordinate_transport"] = dict(inputs=inputs,
        old_native_to_new_native_h=transform.tolist(), producer=asset(Path(__file__)),
        scope="same-acquisition coordinate DATA only; original independent QA and uncertainty unchanged")
    for binding in inputs.values():
        reader.path(binding)
    geometry = _write_json(output / "geometry.json", result)
    return dict(geometry=geometry, frame_photo_top_keep=result["frame_photo_top_keep"],
                phase_roles=dict(current_h=deepcopy(native["baseline_to_native_h"]),
                                 reference_to_current_h=deepcopy(native["reference_to_native_h"])))


def _coordinate_context(*, old_geometry, old_raw_audit, old_acquisition_collection,
                        old_native_configuration, new_bridge_frame, native_configuration,
                        phase_configuration, phase_shape_hw, reader=None):
    """Re-admit the original RAW coordinate graph without reading image/curve bodies."""
    reader = DataReader() if reader is None else reader
    inputs = {}

    def document(role, binding, *, historical=False):
        # Historical row leaves keep their original descriptor schema, including
        # an absent size; DataReader verifies their declared SHA without repair.
        inputs[role] = deepcopy(binding if historical else _bound(binding))
        return reader.json(binding)

    row = document("old_geometry", old_geometry)
    old = document("old_raw_audit", old_raw_audit)
    previous = document("old_native_configuration", old_native_configuration)
    current = document("native_configuration", native_configuration)
    phase = document("phase_configuration", phase_configuration)
    receipt = document("new_bridge_frame", new_bridge_frame)
    if receipt.get("format") != "darsia-build-native-frame-v1":
        raise ValueError("The original completed Build native-frame receipt is required")
    bridge = document("bridge", receipt["bridge"])
    native = document("native_frame", receipt["native_frame"])
    if (bridge.get("format") != "darsia-build-native-bridge-v1"
            or not _same(bridge["native_configuration"], native_configuration)
            or not _same(receipt["raw_source_binding"], bridge["raw_source_binding"])):
        raise ValueError("Current native configuration and bridge differ")
    old_binding = document("old_raw_binding", bridge["photometry_source"])
    new_binding = document("raw_source_binding", receipt["raw_source_binding"])
    collection = document("old_acquisition_collection", old_acquisition_collection)
    configuration = document("old_acquisition_configuration", collection["configuration"])
    records = [item for item in collection["records"]
               if _same(item["full"]["binding"]["producer_audit"], old_raw_audit)]
    specifications = [item for item in configuration["acquisitions"]
                      if _same(item["geometry"], old_geometry)]
    if (collection.get("format") != RAW_ACQUISITIONS or collection.get("run") != row["run"]
            or configuration.get("format") != RAW_ACQUISITIONS
            or configuration.get("run") != collection["run"]
            or not _same(configuration["native"]["configuration"], old_native_configuration)
            or len(records) != 1 or len(specifications) != 1
            or not _same(records[0]["full"]["binding"]["geometry"], row)
            or not _same(specifications[0]["native_task"]["processor"]["raw_binding"], bridge["photometry_source"])):
        raise ValueError("The original RAW acquisition producer does not bind this old audit, geometry and configuration")
    if (native.get("contract") != raw_rig_source.FRAME_CONTRACT
            or old_binding.get("contract") != raw_rig_source.CONTRACT
            or new_binding.get("contract") != raw_rig_source.CONTRACT
            or not _same(old["processor"]["source_binding"],
                         {key: bridge["photometry_source"][key] for key in ("path", "sha256")})):
        raise ValueError("Old RAW audit does not belong to the bridge's exact photometry source")
    from ..storage_paths import resolve_storage_path
    from .series import same_raw_source
    run = native["run"]
    acquisition = native["acquisition"]
    now = receipt["correction"]
    for audit in (old, now):
        if (audit.get("contract") != raw_rig_source.CONTRACT
                or audit.get("jpeg_inputs") is not False
                or audit.get("source_identity") != dict(kind="raw-acquisition", raw_sha256=acquisition["sha256"])
                or not same_raw_source(audit["sources"]["camera_arw"], acquisition, resolve_storage_path)):
            raise ValueError("Both RAW renderer audits must name the exact same acquisition")
    if (row.get("run") != run or old_binding.get("run") != run or new_binding.get("run") != run
            or row["raw_source"]["sha256"] != acquisition["sha256"]
            or not _same(old_binding["models"], new_binding["models"])
            or not _same(old_binding["baseline_raw"], new_binding["baseline_raw"])):
        raise ValueError("Geometry, RAW baseline or complete frozen photometry models differ")
    profile = old_binding["profile_id"]
    if (new_binding["profile_id"] != profile
            or old["processor"]["profile_id"] != profile
            or now["processor"]["profile_id"] != profile
            or not _same(previous["registration"]["common"], current["registration"]["common"])
            or previous["source"]["profiles"].get(run) != profile
            or current["source"]["profiles"].get(run) != profile):
        raise ValueError("The full calibrated camera-U basis and profile must remain exact")
    for audit, binding, config, descriptor in (
            (old, old_binding, previous, bridge["photometry_source"]),
            (now, new_binding, current, receipt["raw_source_binding"])):
        if (not _same(audit["processor"]["source_binding"], {key: descriptor[key] for key in ("path", "sha256")})
                or not _same(audit["processor"]["models"], binding["models"])
                or not _same(audit["processor"]["execution_identity"], binding["execution_identity"])
                or audit["processor"]["native_shape_hw"] != binding["native_shape_hw"]
                or config["source"]["native_shape_hw"] != binding["native_shape_hw"]):
            raise ValueError("The original RAW audit and complete frozen source bindings differ")
    parameters = current
    before, after = old["renderer"], now["renderer"]
    for renderer, binding in ((before, old_binding), (after, new_binding)):
        if (renderer.get("contract") != raw_positioning.CONTRACT
                or renderer.get("source_kind") != "raw"
                or not _same(renderer["raw_acquisition"], acquisition)
                or renderer["profile_id"] != profile
                or renderer["native_shape_hw"] != binding["native_shape_hw"]
                or renderer["camera_u_wh"] != current["registration"]["common"]["sensor"]["camera_wh"]):
            raise ValueError("Original RAW renderer source, profile or native/camera-U grid differs")
    if not _same(before["sensor_metadata"], after["sensor_metadata"]):
        raise ValueError("The same-acquisition camera sensor metadata changed")
    native_shape = after["native_shape_hw"]
    gp = phase["parameters"]["geometry"]
    box, origin = gp["source_box_xywh"], gp["rig_origin_xy"]
    shape = list(phase_shape_hw)
    photograph = phase["parameters"]["scientific"]["photograph_canvas_wh"]
    if (len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape)
            or len(photograph) != 2 or any(type(v) is not int or v < 1 for v in photograph)
            or photograph[0] > shape[1] or photograph[1] > shape[0]
            or len(box) != 4 or any(type(v) is not int for v in box)
            or gp["native_shape_hw"] != native_shape
            or phase["parameters"]["scientific"]["native_shape_hw"] != native_shape
            or box[2:] != list(reversed(native_shape)) or origin != [-box[0], -box[1]]
            or min(box[:2]) < 0 or box[0] + box[2] > shape[1] or box[1] + box[3] > shape[0]
            or row["source_native_shape_hw"] != before["native_shape_hw"]):
        raise ValueError("Explicit current canvas, origin, content box and native grids differ")
    reference_field = phase["geometry_fields"]["reference_to_frame"]
    homography(row["area_parameters"][phase["geometry_fields"]["reference_to_metric"]])
    if (not _same(row["area_parameters"][reference_field], row[reference_field])
            or not source._same_h(homography(row[reference_field]),
                          homography(old[reference_field]), parameters=parameters)):
        raise ValueError("Old photograph geometry is not on the old RAW renderer's reference basis")
    transform = homography(np.linalg.inv(homography(after["native_to_camera_u_h"]))
                           @ homography(before["native_to_camera_u_h"]))
    reference = homography(transform @ homography(row[reference_field]))
    if (not source._same_h(reference, homography(native["reference_to_native_h"]), parameters=parameters)
            or not source._same_h(reference, homography(now[reference_field]), parameters=parameters)
            or not source._same_h(homography(native["baseline_to_native_h"]),
                                  homography(now["H_v5_B_to_frame_native"]), parameters=parameters)):
        raise ValueError("Coordinate-only geometry and measured current RAW reference transforms disagree")
    return dict(reader=reader, inputs=inputs, document=document, row=row, old=old, now=now, native=native,
        previous=previous, current=current, phase=phase, transform=transform, reference=reference,
        reference_field=reference_field, shape=shape, origin=origin, box=box, native_shape=native_shape)


def _photographic_inputs(context):
    """Retain the original independent observation checks, without curve decoding."""
    row, document = context["row"], context["document"]
    # No inferred geometry or old safe-sand complement may supply missing walls,
    # fixtures or the independently photographed surface.
    if (not isinstance(row.get("boundary_evidence"), str) or not row["boundary_evidence"]
            or type(row.get("edge_gate_verified_on_own_photograph")) is not bool
            or not isinstance(row.get("reference_rig"), dict)
            or not isinstance(row.get("frame_fixture_polygons_native"), dict)
            or not row["frame_fixture_polygons_native"]):
        raise ValueError("Independent wall observations and explicit photographed fixture polygons are required")
    for points in (row["reference_rig"].get("polygon_reference_native"), row.get("frame_inner_polygon_native"),
                   *row["frame_fixture_polygons_native"].values()):
        if len(transform_points(points, np.eye(3))) < 3:
            raise ValueError("Explicit photographed polygons require at least three vertices")
    hardware = document("hardware_pose", row["hardware_pose_receipt"], historical=True)
    if (hardware.get("phase_labels_used") is not False or hardware.get("grain_registration_used") is not False
            or not _same(hardware["public_source"], row["source_photo"])
            or not _same(hardware["polygons_native_xy"], row["frame_fixture_polygons_native"])):
        raise ValueError("Photographed hardware must retain its independent source and exact outlines")
    top = row["frame_photo_top_keep"]
    evidence = document("top_photo_evidence", top["reference_photo_evidence"], historical=True)
    curve_binding = top["curve_source"]
    if (top.get("phase_or_labels_used") is not False
            or not _same(evidence["curve"], curve_binding)
            or not isinstance(evidence.get("photographic_definition"), str)):
        raise ValueError("The original independently photographed top curve is required")
    return top, curve_binding


def _transported_row(context):
    """Keep the original producer's exact independent coordinate updates."""
    row, reference_field, reference, transform = (context[key] for key in
        ("row", "reference_field", "reference", "transform"))
    origin, shape, native_shape, box, phase = (context[key] for key in
        ("origin", "shape", "native_shape", "box", "phase"))
    result = deepcopy(row)
    result[reference_field] = reference.tolist()
    result["area_parameters"][reference_field] = reference.tolist()
    result["H_reference_rig_to_frame_native"] = (transform @ homography(row["H_reference_rig_to_frame_native"])).tolist()
    result["frame_inner_polygon_native"] = transform_points(row["frame_inner_polygon_native"], transform).tolist()
    result["frame_fixture_polygons_native"] = {
        name: transform_points(points, transform).tolist()
        for name, points in row["frame_fixture_polygons_native"].items()}
    result.update(native_origin_xy=deepcopy(origin), phase_shape_hw=shape,
                  source_native_shape_hw=deepcopy(native_shape), source_content_box_xywh=deepcopy(box),
                  photograph_canvas_wh=deepcopy(phase["parameters"]["scientific"]["photograph_canvas_wh"]))
    return result


def _read_geometry_transport(geometry, *, reader=None):
    """Check saved coordinate DATA against the same original metadata-only joins.

    This shared producer input reader never decodes the RAW, top curve/mask or
    physical rasters. It grants no new photographic or physical qualification.
    """
    reader = DataReader() if reader is None else reader
    row = reader.json(_bound(geometry))
    transport = row["coordinate_transport"]
    declared = transport["inputs"]
    names = ("old_geometry", "old_raw_audit", "old_acquisition_collection", "old_native_configuration",
             "new_bridge_frame", "native_configuration", "phase_configuration")
    context = _coordinate_context(**{key: declared[key] for key in names},
        phase_shape_hw=row["phase_shape_hw"], reader=reader)
    _photographic_inputs(context)
    expected = _transported_row(context)
    if (not _same(transport["old_native_to_new_native_h"], context["transform"].tolist())
            or not _same({key: value for key, value in row.items()
                          if key not in ("coordinate_transport", "frame_photo_top_keep")},
                         {key: value for key, value in expected.items() if key != "frame_photo_top_keep"})):
        raise ValueError("Saved geometry differs from its original admitted RAW coordinate transport")
    top = row["frame_photo_top_keep"]
    original_top = expected["frame_photo_top_keep"]
    replaced = ("path", "sha256", "size_bytes", "array_key", "shape_hw")
    if (not _same({key: value for key, value in top.items() if key not in replaced},
                  {key: value for key, value in original_top.items() if key not in replaced})
            or top.get("array_key") != "top_keep" or top.get("shape_hw") != context["shape"]):
        raise ValueError("Saved geometry changed its original photograph-top source metadata")
    for key, binding in context["inputs"].items():
        if not _same(binding, declared.get(key)):
            raise ValueError("Saved geometry source graph differs from its original RAW inputs")
    original = context["row"]
    inherited = dict(hardware_pose=original["hardware_pose_receipt"],
        top_photo_evidence=original_top["reference_photo_evidence"], top_curve=original_top["curve_source"])
    if (set(declared) != set(context["inputs"]) | set(inherited)
            or any(not _same(declared[key], value) for key, value in inherited.items())):
        raise ValueError("Saved geometry changed its original independent source bindings")
    # Recheck consumed metadata only. The saved geometry already retains every
    # curve/mask descriptor; downstream area/static producers do not consume
    # those array bodies and must not decode or hash them to re-admit coordinates.
    context["inputs"].update(geometry=deepcopy(geometry), geometry_producer=deepcopy(transport["producer"]))
    for binding in context["inputs"].values():
        reader.path(binding)
    context["geometry"] = row
    return context
