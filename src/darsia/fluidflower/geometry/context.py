"""Measured physical geometry on an unchanged complete review canvas.

These functions evaluate supplied geometry only; they do not fit boundaries,
select scientific fields, infer phases, or read campaign metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import cv2
import numpy as np


def _raster_polygon(points, shape_hw, *, raster_scale, raster_shift):
    """Rasterize already placed points with the configured fixed-point rule."""
    mask = np.zeros(shape_hw, np.uint8)
    cv2.fillPoly(
        mask,
        [np.rint(points * raster_scale).astype(np.int32)],
        1,
        shift=raster_shift,
    )
    return mask.astype(bool)


def homography(value, *, parameters: Mapping):
    """Admit a finite, nonsingular 3-by-3 projective coordinate transform.

    The determinant floor is explicit parameter DATA; matrix evaluation is float64."""
    matrix = np.asarray(value, np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Rig transform must be a finite 3x3 homography")
    if abs(np.linalg.det(matrix)) < parameters["determinant_floor"]:
        raise ValueError("Rig transform is singular")
    return matrix


def transform_points(points, matrix, *, parameters: Mapping):
    """Project finite x/y positions through a physical coordinate transform.

    The supplied horizon tolerance rejects points whose homogeneous divisor vanishes."""
    points = np.asarray(points, np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or (not np.isfinite(points).all()):
        raise ValueError("Rig points must be finite Nx2 native coordinates")
    projected = (
        homography(matrix, parameters=parameters)
        @ np.column_stack((points, np.ones(len(points)))).T
    )
    if np.any(np.abs(projected[2]) < parameters["point_horizon_floor"]):
        raise ValueError("Rig transform crosses a projective horizon")
    return (projected[:2] / projected[2]).T


def inverse_homography_area_map(
    reference_to_frame_h,
    shape_hw,
    *,
    origin_xy,
    fixed_area_mm2,
    reference_to_metric_h=None,
    parameters: Mapping,
):
    """Evaluate physical pixel areas in mm2 from the inverse reference-to-frame map.

    The optional reference-to-metric map composes on the left. Pixel origins and
    reference pixel area are explicit; the projective Jacobian is evaluated in
    float64 before the original float32 output conversion."""
    height, width = map(int, shape_hw)
    if min(height, width) < 1 or not np.isfinite(fixed_area_mm2) or fixed_area_mm2 <= 0:
        raise ValueError("Invalid physical area or image shape")
    matrix = np.linalg.inv(homography(reference_to_frame_h, parameters=parameters))
    if reference_to_metric_h is not None:
        matrix = homography(reference_to_metric_h, parameters=parameters) @ matrix
    x = np.arange(width, dtype=np.float64)[None, :] + float(origin_xy[0])
    y = np.arange(height, dtype=np.float64)[:, None] + float(origin_xy[1])
    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    if (
        np.any(np.abs(denominator) < parameters["area_horizon_floor"])
        or np.min(denominator) * np.max(denominator) <= 0
    ):
        raise ValueError("Physical area map crosses a projective horizon")
    area = fixed_area_mm2 * abs(np.linalg.det(matrix)) / np.abs(denominator) ** 3
    if not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("Physical pixel areas must be finite and positive")
    return area.astype(np.float32)


def rig_plane_area_map(
    H_frame_native_to_metric_pixel,
    shape_hw,
    *,
    origin_xy,
    metric_pixel_area_mm2,
    parameters: Mapping,
):
    """Evaluate mm2 areas from an independently measured frame-to-metric transform.

    This route retains the original pair of inversions, separately from texture
    registration. The metric reference pixel area and pixel origin are required."""
    return inverse_homography_area_map(
        np.linalg.inv(
            homography(H_frame_native_to_metric_pixel, parameters=parameters)
        ),
        shape_hw,
        origin_xy=origin_xy,
        fixed_area_mm2=metric_pixel_area_mm2,
        parameters=parameters,
    )


def photographic_observation_mask(
    H_native_to_camera, camera_wh, canvas_wh, *, native_origin_xy, parameters: Mapping
):
    """Mark review-canvas pixel centres genuinely visible in the recorded camera.

    Coordinates use the explicit native origin, inclusive camera bounds, and
    configured projective-horizon tolerance. This is observation, not feature validity.
    """
    camera_width, camera_height = map(int, camera_wh)
    width, height = map(int, canvas_wh)
    if min(camera_width, camera_height, width, height) < 1:
        raise ValueError("Invalid camera or canvas dimensions")
    matrix = homography(H_native_to_camera, parameters=parameters)
    x = np.arange(width, dtype=float)[None, :] + float(native_origin_xy[0])
    y = np.arange(height, dtype=float)[:, None] + float(native_origin_xy[1])
    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    safe = np.abs(denominator) > parameters["observation_horizon_floor"]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denominator
        v = (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denominator
    return (
        safe
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0)
        & (u <= camera_width - 1)
        & (v >= 0)
        & (v <= camera_height - 1)
    )


@dataclass(frozen=True)
class RigFrameContext:
    """Physical review-canvas domain, support classes, fixtures, and mm2 pixel areas.

    Arrays retain the unchanged complete canvas; origin and content box record
    the placement of native measurements. Parameters contain the resulting audit."""

    rig_domain: np.ndarray
    modelled: np.ndarray
    filled_observed: np.ndarray
    filled_unobserved: np.ndarray
    hardware_excluded: np.ndarray
    excluded_by_name: Mapping[str, np.ndarray]
    origin_xy: tuple[int, int]
    native_shape_hw: tuple[int, int]
    content_box_xywh: tuple[int, int, int, int]
    pixel_area_mm2: np.ndarray
    parameters: dict


def rig_frame_context(
    reference,
    reference_to_frame_h,
    native_shape_hw,
    *,
    feature_valid_mask,
    canvas_wh,
    content_box_xywh,
    photographic_observed_mask=None,
    parameters: Mapping,
):
    """Rasterize a measured reference rig polygon and explicit hardware fixtures.

    The feature-valid native mask and optional photographed footprint determine
    modelled, observed-fill, and unobserved-fill regions. Required parameter DATA
    provides field names, fixed-point raster settings, and physical area units."""
    width, height = map(int, canvas_wh)
    x0, y0, native_width, native_height = map(int, content_box_xywh)
    if tuple(map(int, native_shape_hw)) != (native_height, native_width):
        raise ValueError("Native shape and canvas content box disagree")
    if x0 < 0 or y0 < 0 or x0 + native_width > width or (y0 + native_height > height):
        raise ValueError("Native content is outside the unchanged review canvas")
    valid = np.asarray(feature_valid_mask)
    if valid.dtype != np.bool_ or valid.shape != tuple(native_shape_hw):
        raise ValueError(
            "Feature validity must be a boolean mask on the exact native grid"
        )
    exclusions = reference.get(parameters["fields"]["reference_fixtures"])
    if not isinstance(exclusions, Mapping) or (
        not exclusions and (not reference.get("fixtures_verified_none"))
    ):
        raise ValueError(
            "Rig geometry lacks explicit photograph-verified fixture exclusions"
        )
    matrix = homography(reference_to_frame_h, parameters=parameters)
    polygon = transform_points(
        reference[parameters["fields"]["reference_polygon"]],
        matrix,
        parameters=parameters,
    ) + (x0, y0)
    if np.any(polygon.min(axis=0) < -parameters["canvas_margin"]) or np.any(
        polygon.max(axis=0)
        > np.array(
            [width - parameters["canvas_margin"], height - parameters["canvas_margin"]]
        )
    ):
        raise ValueError(
            f"Physical rig domain exceeds review canvas; required native bounds {polygon.min(axis=0) - (x0, y0)} to {polygon.max(axis=0) - (x0, y0)}"
        )

    def raster(points):
        return _raster_polygon(
            points, (height, width), raster_scale=parameters["raster_scale"],
            raster_shift=parameters["raster_shift"],
        )

    outer = raster(polygon)
    by_name = {
        str(name): raster(
            transform_points(points, matrix, parameters=parameters) + (x0, y0)
        )
        & outer
        for name, points in exclusions.items()
    }
    excluded = np.zeros_like(outer)
    for mask in by_name.values():
        excluded |= mask
    domain = outer & ~excluded
    if photographic_observed_mask is None:
        observed = np.zeros_like(domain)
        observed[y0 : y0 + native_height, x0 : x0 + native_width] = True
        observed_definition = "native scientific content box only; caller has not supplied camera footprint"
    else:
        observed = np.asarray(photographic_observed_mask)
        if observed.dtype != np.bool_ or observed.shape != domain.shape:
            raise ValueError(
                "Photographic observation mask must match the complete canvas"
            )
        observed_definition = (
            "source-bound camera pixel footprint on unchanged review canvas"
        )
    support = np.zeros_like(domain)
    support[y0 : y0 + native_height, x0 : x0 + native_width] = valid
    observed = observed | support
    modelled = domain & support
    filled_observed = domain & observed & ~support
    filled_unobserved = domain & ~observed
    area = inverse_homography_area_map(
        matrix,
        (height, width),
        origin_xy=(-x0, -y0),
        fixed_area_mm2=reference.get(
            parameters["fields"]["nominal_area_field"],
            parameters["nominal_pixel_area_mm2"],
        ),
        reference_to_metric_h=reference.get(
            parameters["fields"]["reference_metric_field"]
        ),
        parameters=parameters,
    )
    context_parameters = dict(
        format=parameters["contracts"]["rig_context"],
        **{parameters["fields"]["reference_transform_field"]: matrix.tolist()},
        origin_xy=[-x0, -y0],
        canvas_wh=[width, height],
        content_box_xywh=[x0, y0, native_width, native_height],
        **{
            parameters["fields"]["nominal_area_field"]: reference.get(
                parameters["fields"]["nominal_area_field"],
                parameters["nominal_pixel_area_mm2"],
            )
        },
        **{
            parameters["fields"]["reference_metric_field"]: reference.get(
                parameters["fields"]["reference_metric_field"]
            )
        },
        observed_definition=observed_definition,
        domain_area_cm2=float(
            area[domain].sum(dtype=np.float64) / parameters["square_mm_per_cm2"]
        ),
        area_cm2_by_class={
            name: float(
                area[mask].sum(dtype=np.float64) / parameters["square_mm_per_cm2"]
            )
            for name, mask in (
                ("modelled", modelled),
                ("filled_observed", filled_observed),
                ("filled_unobserved", filled_unobserved),
            )
        },
        excluded_area_cm2={
            name: float(
                area[mask].sum(dtype=np.float64) / parameters["square_mm_per_cm2"]
            )
            for name, mask in by_name.items()
        },
    )
    return RigFrameContext(
        domain,
        modelled,
        filled_observed,
        filled_unobserved,
        excluded,
        by_name,
        (-x0, -y0),
        tuple(native_shape_hw),
        tuple(content_box_xywh),
        area,
        context_parameters,
    )


def independent_rig_frame_context(
    reference_rig,
    H_reference_rig_to_frame_native,
    native_shape_hw,
    *,
    feature_valid_mask,
    canvas_wh,
    content_box_xywh,
    frame_photo_top_keep,
    frame_fixture_polygons_native,
    fixtures_verified_none=False,
    photographic_observed_mask=None,
    H_frame_native_to_metric_pixel=None,
    parameters: Mapping,
):
    """Combine independently measured rig walls, photographic sand top, and fixtures.

    The full-canvas top mask may remove only each column's upper prefix. Native
    hardware polygons remain independent of wall registration; an optional
    frame-to-metric transform changes physical pixel area without fitting geometry."""
    matrix = homography(H_reference_rig_to_frame_native, parameters=parameters)
    if not isinstance(frame_fixture_polygons_native, Mapping):
        raise ValueError("Native photograph-verified hardware polygons are required")
    reference = dict(
        **{
            parameters["fields"]["reference_polygon"]: reference_rig[
                "polygon_reference_native"
            ]
        },
        **{
            parameters["fields"]["reference_fixtures"]: {
                name: transform_points(
                    points, np.linalg.inv(matrix), parameters=parameters
                )
                for name, points in frame_fixture_polygons_native.items()
            }
        },
        fixtures_verified_none=bool(fixtures_verified_none),
        **{
            parameters["fields"]["nominal_area_field"]: reference_rig.get(
                "metric_pixel_area_mm2", parameters["nominal_pixel_area_mm2"]
            )
        },
    )
    context = rig_frame_context(
        reference,
        matrix,
        native_shape_hw,
        feature_valid_mask=feature_valid_mask,
        canvas_wh=canvas_wh,
        content_box_xywh=content_box_xywh,
        photographic_observed_mask=photographic_observed_mask,
        parameters=parameters,
    )
    top = np.asarray(frame_photo_top_keep)
    if top.dtype != bool or top.shape != context.rig_domain.shape:
        raise ValueError(
            "Photographic sand top mask must be bool on the unchanged complete canvas"
        )
    if not np.all(top[-1]) or np.any(top[:-1] & ~top[1:]):
        raise ValueError(
            "Photographic sand top mask must remove only a prefix above each column's surface"
        )
    area = context.pixel_area_mm2
    if H_frame_native_to_metric_pixel is not None:
        area = rig_plane_area_map(
            H_frame_native_to_metric_pixel,
            area.shape,
            origin_xy=context.origin_xy,
            metric_pixel_area_mm2=reference[parameters["fields"]["nominal_area_field"]],
            parameters=parameters,
        )
    domain = context.rig_domain & top
    modelled = context.modelled & top
    observed = context.filled_observed & top
    unobserved = context.filled_unobserved & top
    by_name = {name: mask & top for name, mask in context.excluded_by_name.items()}
    excluded = context.hardware_excluded & top
    context_parameters = dict(
        context.parameters,
        format=parameters["contracts"]["independent_context"],
        H_reference_rig_to_frame_native=matrix.tolist(),
        H_frame_native_to_metric_pixel=(
            None
            if H_frame_native_to_metric_pixel is None
            else homography(
                H_frame_native_to_metric_pixel, parameters=parameters
            ).tolist()
        ),
        reference_rig_identity=reference_rig.get("identity"),
        wall_transform_role="independent rigid-plane registration",
        top_geometry_role="frame-specific photograph-registered sand surface",
        fixture_geometry_role="explicit per-frame native photograph polygons",
        metric_pixel_area_mm2=reference[parameters["fields"]["nominal_area_field"]],
        **{
            parameters["fields"]["nominal_area_field"]: parameters[
                "nominal_pixel_area_mm2"
            ]
        },
        physical_scale_approved=False,
        domain_area_cm2=float(
            area[domain].sum(dtype=np.float64) / parameters["square_mm_per_cm2"]
        ),
        area_cm2_by_class={
            name: float(
                area[mask].sum(dtype=np.float64) / parameters["square_mm_per_cm2"]
            )
            for name, mask in (
                ("modelled", modelled),
                ("filled_observed", observed),
                ("filled_unobserved", unobserved),
            )
        },
        excluded_area_cm2={
            name: float(
                area[mask].sum(dtype=np.float64) / parameters["square_mm_per_cm2"]
            )
            for name, mask in by_name.items()
        },
    )
    context_parameters.pop(parameters["fields"]["reference_transform_field"], None)
    context_parameters.pop(parameters["fields"]["reference_metric_field"], None)
    return replace(
        context,
        rig_domain=domain,
        modelled=modelled,
        filled_observed=observed,
        filled_unobserved=unobserved,
        hardware_excluded=excluded,
        excluded_by_name=by_name,
        pixel_area_mm2=area,
        parameters=context_parameters,
    )
