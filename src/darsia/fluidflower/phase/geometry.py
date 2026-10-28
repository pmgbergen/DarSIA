"""Existing reference warps, curve transport and categorical zone placement.

All source matrices, raster dimensions, origins and zone identities are
supplied explicitly. These helpers do not estimate registration or geometry.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import cv2
import numpy as np

from .features import log_to_unit_rgb, warp_reference_to_current


def align_scientific(
    signal,
    valid,
    transform,
    *,
    target_shape_hw,
    parameters: Mapping,
    feature_parameters: Mapping,
):
    """Warp a linear scientific reference, retaining the original log dtype.

    Decode and bilinear support use the existing feature helpers. Reconstruction
    multiplies in float64 before assigning CHW float32; current source pixels
    are never passed through this reference operation.
    """
    rgb, supported = log_to_unit_rgb(signal, valid, parameters=feature_parameters)
    aligned, supported = warp_reference_to_current(
        rgb,
        transform,
        target_shape_hw=tuple(target_shape_hw),
        source_origin_xy=tuple(parameters["source_origin_xy"]),
        source_valid_mask=supported,
        parameters=feature_parameters,
    )
    result = np.full((3, *aligned.shape[:2]), np.nan, np.float32)
    supported &= (aligned > 0).all(-1)
    result[:, supported] = np.log(
        aligned[supported].astype(np.float64) * parameters["unit_denominator"]
    ).T.astype(np.float32)
    return result, supported


def _project(H, x, y, *, parameters: Mapping):
    H = np.asarray(H, dtype=np.float64)
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    den = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    if (
        H.shape != (3, 3)
        or np.any(~np.isfinite(den))
        or np.any(np.abs(den) < parameters["curve_homography_floor"])
    ):
        raise ValueError("Bottom report metric homography is invalid")
    return (
        (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / den,
        (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / den,
    )


def transport_curve(
    curve,
    *,
    H_source_to_target,
    source_origin_xy,
    target_origin_xy,
    target_width,
    parameters: Mapping,
):
    """Transport observed segments only, without bridging missing evidence."""
    good = np.isfinite(curve)
    result = np.full(target_width, np.nan)
    cuts = np.flatnonzero(np.diff(np.r_[False, good, False].astype(np.int8)))
    for a, b in zip(cuts[::2], cuts[1::2]):
        xs = np.arange(a, b, dtype=float) + source_origin_xy[0]
        tx, ty = _project(
            H_source_to_target,
            xs,
            curve[a:b] + source_origin_xy[1],
            parameters=parameters,
        )
        tx, ty = tx - target_origin_xy[0], ty - target_origin_xy[1]
        order = np.argsort(tx)
        tx, ty = tx[order], ty[order]
        if len(tx) > 1 and np.all(np.diff(tx) > 0):
            low, high = max(0, math.ceil(tx[0])), min(
                target_width - 1, math.floor(tx[-1])
            )
            result[low : high + 1] = np.interp(np.arange(low, high + 1), tx, ty)
        elif len(tx) == 1 and 0 <= round(tx[0]) < target_width:
            result[round(tx[0])] = ty[0]
    return result


def transported_floor(
    curve,
    *,
    H_source_to_target,
    source_origin_xy,
    target_origin_xy,
    target_shape_hw,
    parameters: Mapping,
):
    """Transport the frozen floor and hold only its supported endpoint values.

    Holding the pooled line beyond its ends is the existing domain convention.
    An interior missing segment remains an error rather than invented evidence.
    """
    result = transport_curve(
        curve,
        H_source_to_target=np.asarray(H_source_to_target, float),
        source_origin_xy=source_origin_xy,
        target_origin_xy=target_origin_xy,
        target_width=target_shape_hw[1],
        parameters=parameters,
    )
    good = np.flatnonzero(np.isfinite(result))
    if not len(good):
        raise ValueError("Transported sand floor has no valid columns")
    result[: good[0]] = result[good[0]]
    result[good[-1] + 1 :] = result[good[-1]]
    if not np.isfinite(result).all():
        raise ValueError("Transported pooled floor contains an interior gap")
    return result


def _warp(mask, H, shape, *, parameters: Mapping):
    return cv2.warpPerspective(
        mask.astype(np.uint8),
        H,
        (shape[1], shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=parameters["mask_border_value"],
    ).astype(bool)


def native_l1(
    zones,
    *,
    H_source_to_target,
    native_shape_hw,
    source_box_xywh,
    phase_shape_hw,
    zone_id,
    parameters: Mapping,
):
    """Place a frozen zone mask and extend its last native row below the box."""
    H = np.asarray(H_source_to_target, float)
    native = _warp(zones == zone_id, H, native_shape_hw, parameters=parameters)
    full = np.zeros(phase_shape_hw, bool)
    x, y, width, height = source_box_xywh
    full[y : y + height, x : x + width] = native
    full[y + height :, x : x + width] = native[-1]
    return full
