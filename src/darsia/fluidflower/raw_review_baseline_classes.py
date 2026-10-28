"""Baseline geometry classes for RAW Change, without phase inference.

Use the baseline's saved rig geometry, photographic surface and hardware,
its RAW camera footprint, and the same transported floor as phase inference.
These masks describe geometric support only. No model or phase label is read.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .geometry import context as rig_geometry
from .phase import geometry as phase_geometry
from .phase_saved_assets import VerifiedAssetReader, asset

CONTRACT = "darsia-raw-baseline-display-classes-v1"


def _array(reader, descriptor):
    """Read a verified array or copy the named array from a closed archive."""
    value = reader.array(descriptor)
    if isinstance(value, np.lib.npyio.NpzFile):
        with value as archive:
            return archive[descriptor["array_key"]].copy()
    return value


def baseline_rig_classes(reference_record: Mapping, phase_configuration: Mapping, *, reader=None):
    """Return exact geometric baseline classes using existing geometry helpers.

    The record must retain its own ``full.binding.geometry`` and RAW producer
    audit. The phase configuration supplies its original geometry parameters
    and floor descriptor. A missing binding is an error, never a reason to
    reuse a later frame's classes or omit the classifier's floor cutoff.
    """
    reader = reader or VerifiedAssetReader()
    full = reference_record["full"]
    binding = full.get("binding") or {}
    row = binding.get("geometry") or {}
    native_row = reader.json(reference_record["native_row"])
    audit_descriptor = native_row["raw_producer_audit"]
    if (binding.get("producer_audit", {}).get("sha256") != audit_descriptor.get("sha256")
            or row.get("reference") is not True
            or row.get("raw_source", {}).get("sha256") != full["source_sha256"]):
        raise ValueError("Baseline Change needs its own bound RAW rig geometry and audit")
    audit = reader.json(audit_descriptor)
    if (audit.get("jpeg_inputs") is not False
            or audit.get("source_identity", {}).get("raw_sha256") != full["source_sha256"]):
        raise ValueError("Baseline rig geometry belongs to another RAW acquisition")
    if (list(row["phase_shape_hw"]) != list(full["phase_shape_hw"])
            or list(row["native_origin_xy"]) != list(full["native_origin_xy"])):
        raise ValueError("Baseline rig geometry differs from its saved RAW canvas")
    phase = phase_configuration
    gp, constants = phase["parameters"]["geometry"], phase["geometry_parameters"]
    if (list(gp["rig_origin_xy"]) != list(full["native_origin_xy"])
            or list(gp["native_shape_hw"]) != list(full["native_centre"]["native_shape_hw"])):
        raise ValueError("Baseline phase geometry differs from its saved RAW grid")
    renderer = audit["renderer"]
    observed = rig_geometry.photographic_observation_mask(
        renderer["native_to_camera_u_h"], renderer["camera_u_wh"],
        list(reversed(full["phase_shape_hw"])),
        native_origin_xy=tuple(gp["rig_origin_xy"]), parameters=constants)
    fields, area = phase["geometry_fields"], row["area_parameters"]
    metric = (np.asarray(area[fields["reference_to_metric"]])
              @ np.linalg.inv(np.asarray(area[fields["reference_to_frame"]])))
    context = rig_geometry.independent_rig_frame_context(
        row["reference_rig"], row["H_reference_rig_to_frame_native"],
        tuple(gp["native_shape_hw"]),
        feature_valid_mask=_array(reader, full["native_centre"]["valid_mask"]),
        canvas_wh=list(reversed(full["phase_shape_hw"])),
        content_box_xywh=gp["source_box_xywh"],
        frame_photo_top_keep=_array(reader, row["frame_photo_top_keep"]),
        frame_fixture_polygons_native=row["frame_fixture_polygons_native"],
        fixtures_verified_none=row.get("fixtures_verified_none", False),
        photographic_observed_mask=observed, H_frame_native_to_metric_pixel=metric,
        parameters=constants)
    floor = phase_geometry.transported_floor(
        _array(reader, phase["chain_assets"]["floor"]),
        H_source_to_target=row[fields["reference_to_frame"]],
        source_origin_xy=gp["floor_source_origin_xy"],
        target_origin_xy=gp["rig_origin_xy"], target_shape_hw=context.rig_domain.shape,
        parameters=gp)
    below = context.rig_domain & (np.arange(context.rig_domain.shape[0])[:, None] > floor[None, :])
    corrected = context.rig_domain & ~below
    result = {name: np.asarray(getattr(context, name)) & corrected
              for name in ("modelled", "filled_observed", "filled_unobserved")}
    result.update(rig_domain=corrected, hardware_excluded=context.hardware_excluded,
                  below_sand_floor=below)
    if not np.array_equal(sum(result[name].astype(np.uint8) for name in
                              ("modelled", "filled_observed", "filled_unobserved")),
                          corrected.astype(np.uint8)):
        raise ValueError("Baseline geometry classes do not partition the rig domain")
    return result


def write_baseline_rig_classes(path: str | Path, reference_record: Mapping,
                               phase_configuration: Mapping, *, reader=None) -> dict:
    """Write a fresh display-only class asset; preserve every existing file."""
    values = baseline_rig_classes(reference_record, phase_configuration, reader=reader)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)
    return asset(path)


__all__ = ["CONTRACT", "baseline_rig_classes", "write_baseline_rig_classes"]
