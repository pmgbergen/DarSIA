"""Produce a current-grid measured-parent area pair from retained source DATA.

The historical area is regenerated from its unscaled metric basis, never
warped or read from an old raster. The independently recorded factor is then
applied once using the original measured-parent float64/float32 recipe.
Existing width-only measurement scope and negative qualification stay literal.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from ..geometry.areas import apply_measured_area_revision
from ..geometry.context import homography, rig_plane_area_map
from ..mass.n1_pipeline import _write_json, _write_npz, asset
from ..mass.native_optical_source import _same
from ..native import source
from .build_geometry_data import _read_geometry_transport

MEASURED_CONTRACT = "ffac-measured-pixel-area-revision-v1"


def _read_measured_area_context(*, geometry, area, measurement, physical_scale_plan,
                                physical_scale_receipt, measured_frame_receipt,
                                historical_frame_receipt, reader=None):
    """Replay only the original metadata checks; never decode or write area arrays."""
    context = _read_geometry_transport(geometry, reader=reader)
    reader, inputs = context["reader"], context["inputs"]

    def document(name, binding):
        inputs[name] = deepcopy(binding)
        return reader.json(binding)

    revision = document("measurement", measurement)
    plan = document("physical_scale_plan", physical_scale_plan)
    scale = document("physical_scale_receipt", physical_scale_receipt)
    measured = document("measured_frame_receipt", measured_frame_receipt)
    historical = document("historical_frame_receipt", historical_frame_receipt)
    binding = plan["binding"]
    if (area.get("role") != "measured_parent"
            or revision.get("contract") != MEASURED_CONTRACT
            or binding.get("contract") != MEASURED_CONTRACT
            or scale.get("contract") != MEASURED_CONTRACT or scale.get("complete") is not True
            or measured.get("contract") != MEASURED_CONTRACT or measured.get("complete") is not True
            or historical.get("contract") != "ffac-rig-camera-inference-frame-v1"
            or historical.get("complete") is not True):
        raise ValueError("The original complete measured-parent area lineage is required")
    if (scale["source_plan_sha256"] != physical_scale_plan["sha256"]
            or scale["pipeline_sha256"] != plan["pipeline_sha256"]
            or not _same(scale["model_identity"], plan["model_identity"])
            or not _same(binding["measurement"], measurement)
            or not _same(scale["measurement"], measurement)
            or not _same(scale["parent_seal"], binding["parent_seal"])
            or not _same(scale["parent_registration"], binding["parent_registration"])):
        raise ValueError("The original scale receipt, plan and measurement bindings differ")
    parent = document("parent_seal", binding["parent_seal"])
    registration = document("parent_registration", binding["parent_registration"])
    if (parent.get("complete") is not True
            or not _same(parent["model_identity"], binding["parent_identity"])
            or not _same([registration["model_id"], registration["model_sha256"]], binding["parent_identity"])
            or historical["binding"]["pipeline_sha256"] != parent["pipeline_sha256"]):
        raise ValueError("The historical producer differs from the original registered parent")

    old = context["row"]
    run, frame = old["run"], old["frame_index"]
    for collection, receipt in ((scale, measured_frame_receipt), (parent, historical_frame_receipt)):
        matches = [item for item in collection["frames"]
                   if item["run"] == run and item["frame_index"] == frame]
        if len(matches) != 1 or not _same(matches[0]["receipt"], receipt):
            raise ValueError("Area receipts do not select one exact original photograph frame")
    original_source = historical["binding"]["source"]
    if (not _same(original_source["rig_geometry"], old)
            or original_source["run"] != run or original_source["frame_index"] != frame
            or original_source["elapsed_h"] != old["elapsed_h"]
            or original_source["scientific_task_id"] != old["scientific_task_id"]
            or original_source["source_path"] != old["source_photo"]["path"]
            or original_source["source_sha256"] != old["source_photo"]["sha256"]
            or old["source_sha256"] != old["source_photo"]["sha256"]
            or measured["run"] != run or measured["frame_index"] != frame
            or measured["elapsed_h"] != old["elapsed_h"]
            or measured["source_sha256"] != original_source["source_sha256"]
            or not _same(measured["source_producer_receipt"], historical_frame_receipt)):
        raise ValueError("Measured and historical areas differ from the same original source geometry")
    original_descriptor, measured_descriptor = historical["rig_domain_revision"], measured["rig_domain_revision"]
    if (not _same(area["historical"], original_descriptor["area_map"])
            or not _same(area["historical"], measured["numerical_scale_qa"]["source"])
            or not _same(area["measured"], measured_descriptor["area_map"])
            or area["historical"].get("array_key") != "pixel_area_mm2"
            or area["measured"].get("array_key") != "pixel_area_mm2"
            or not _same(area["revision"], revision)
            or not _same(measured["measurement"], revision)
            or not _same(measured_descriptor["physical_scale_revision"], revision)
            or not _same(measured_descriptor["parameters"]["physical_scale_revision"], revision)
            or area["factor"] != revision["area_factor"]):
        raise ValueError("Area arrays, literal factor or original measurement revision differ")
    if (any(revision.get(key) is not False for key in
            ("vertical_scale_independently_verified", "mass_eligible", "bottom_mass_allowed"))
            or any(scale.get(key) is not False or measured.get(key) is not False for key in
                   ("mass_eligible", "bottom_mass_allowed"))
            or original_descriptor.get("mass_eligible") is not False
            or measured_descriptor.get("mass_eligible") is not False):
        raise ValueError("Original width-only measurement and negative mass qualifications must remain unchanged")

    phase, native = context["phase"], context["current"]
    reference_field = phase["geometry_fields"]["reference_to_frame"]
    metric_field = phase["geometry_fields"]["reference_to_metric"]
    old_area = old["area_parameters"]
    new_area = context["geometry"]["area_parameters"]
    original_parameters = original_descriptor["parameters"]
    measured_parameters = measured_descriptor["parameters"]
    old_metric = homography(old_area[metric_field], parameters=phase["geometry_parameters"]) @ np.linalg.inv(
        homography(old_area[reference_field], parameters=phase["geometry_parameters"]))
    annotated_metric = homography(original_parameters["H_frame_native_to_metric_pixel"],
                                  parameters=phase["geometry_parameters"])
    annotated_measured = annotated_metric.copy()
    annotated_measured[:2] *= revision["linear_factor"]
    fixed = old_area["fixed_area_mm2"]
    if (old_area.get("physical_scale_approved") is not False
            or original_parameters.get("physical_scale_approved") is not False
            or "physical_scale_revision" in original_parameters
            or not source._same_h(old_metric, annotated_metric, parameters=native)
            or not source._same_h(annotated_measured,
                homography(measured_parameters["H_frame_native_to_metric_pixel"], parameters=phase["geometry_parameters"]),
                parameters=native)
            or not _same(new_area[metric_field], old_area[metric_field])
            or any(value != fixed for value in (new_area["fixed_area_mm2"],
                original_descriptor["fixed_area_mm2"], original_parameters["metric_pixel_area_mm2"],
                measured_descriptor["fixed_area_mm2"], revision["nominal_historical_pixel_area_mm2_preserved"]))
            or original_descriptor["phase_shape_hw"] != old["phase_shape_hw"]
            or original_descriptor["native_origin_xy"] != old["native_origin_xy"]
            or measured_descriptor["phase_shape_hw"] != original_descriptor["phase_shape_hw"]
            or measured_descriptor["native_origin_xy"] != original_descriptor["native_origin_xy"]):
        raise ValueError("The original unscaled metric basis, grid or recorded scale annotation differs")

    context["area_fixed"] = fixed
    return context


def prepare_measured_area_data(*, geometry, area, measurement, physical_scale_plan,
                               physical_scale_receipt, measured_frame_receipt,
                               historical_frame_receipt, output, reader=None):
    """Write two fresh existing-format area assets, preserving measured_parent.

    File arguments are original path/SHA descriptors. ``area`` is the original
    frame's phase_roles.area mapping and must agree with the independently
    bound original scale revision/receipts. Original frame identity comes from
    the checked photograph geometry, not a selected RAW frame's logical index.

    Geometry's existing same-RAW admission is replayed without reading the RAW,
    top masks/curves or old area bodies. This coordinate DATA operation neither
    chooses a physical scale nor establishes physical/mass qualification.
    """
    context = _read_measured_area_context(geometry=geometry, area=area, measurement=measurement,
        physical_scale_plan=physical_scale_plan, physical_scale_receipt=physical_scale_receipt,
        measured_frame_receipt=measured_frame_receipt, historical_frame_receipt=historical_frame_receipt,
        reader=reader)
    reader, inputs = context["reader"], context["inputs"]
    phase = context["phase"]
    new_area = context["geometry"]["area_parameters"]
    metric_field = phase["geometry_fields"]["reference_to_metric"]
    reference_field = phase["geometry_fields"]["reference_to_frame"]
    fixed = context["area_fixed"]

    # No old area is opened: evaluate the current native grid through the exact
    # original common-reference metric basis, retaining the double inversion.
    frame_to_metric = homography(new_area[metric_field], parameters=phase["geometry_parameters"]) @ np.linalg.inv(
        homography(new_area[reference_field], parameters=phase["geometry_parameters"]))
    original = rig_plane_area_map(frame_to_metric, context["shape"], origin_xy=context["origin"],
        metric_pixel_area_mm2=fixed, parameters=phase["geometry_parameters"])
    # Validate the bound rounding parameters before any output is created.
    # The source descriptor in this first audit is replaced after the new
    # historical NPZ exists; it never names an old raster as the new source.
    corrected, numerical = apply_measured_area_revision(original, area["factor"], source_binding=area["historical"],
                                                       parameters=phase["area_parameters"])
    consumed_paths = [reader.path(descriptor) for descriptor in inputs.values()]
    output = Path(output)
    from ..job_filesystem import inspect_local_file
    from ..rig_raw_write import _no_links
    if not output.is_absolute() or output.exists() or not inspect_local_file(output / ".area-data-location").allowed:
        raise ValueError("Area DATA needs one fresh positively local output folder")
    _no_links(output.parent)
    # Retain logical old-area protection without opening/hashing those rasters,
    # and protect actual consumed alias locations as well as logical metadata.
    protected = consumed_paths + [Path(item["path"]) for item in
        list(inputs.values()) + [area["historical"], area["measured"]]]
    if any(output.resolve() == path.resolve().parent
           or output.resolve().is_relative_to(path.resolve().parent) for path in protected):
        raise ValueError("Area DATA output overlaps its source folders")
    output.mkdir()
    historical_asset = dict(_write_npz(output / "historical.area.npz", dict(pixel_area_mm2=original)),
                            array_key="pixel_area_mm2")
    measured_asset = dict(_write_npz(output / "measured.area.npz", dict(pixel_area_mm2=corrected)),
                          array_key="pixel_area_mm2")
    # Only audit source identity changes; retain every numeric audit field from
    # the single original rounding operation already performed above.
    numerical["source"] = historical_asset
    result = dict(deepcopy(area), historical=historical_asset, measured=measured_asset)
    for descriptor in inputs.values():
        reader.path(descriptor)
    provenance = _write_json(output / "area_data.json", dict(area=result, inputs=inputs,
        numerical_scale_qa=numerical, producer=asset(Path(__file__)),
        scope="same-acquisition coordinate DATA; original width-only revision and negative qualification unchanged"))
    return dict(area=result, provenance=provenance)
