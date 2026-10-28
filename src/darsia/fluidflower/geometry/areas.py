"""Distinct physical-area scaling recipes retained from their measured roles."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def apply_measured_area_revision(
    original, factor, *, source_binding, parameters: Mapping
):
    """Apply a separately measured physical-area factor to a positive float32 raster.

    This parent-field recipe explicitly upcasts to float64 before multiplication
    and returns float32 areas plus the original rounding and min/max audit.
    The admitted source binding and rounding tolerance are supplied as DATA."""
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Area scale must be positive and finite")
    if (
        original.dtype != np.float32
        or original.ndim != 2
        or (not np.isfinite(original).all())
        or (original <= 0).any()
    ):
        raise ValueError("Area source must be a positive finite float32 raster")
    corrected = (original.astype(np.float64) * factor).astype(np.float32)
    expected = original.astype(np.float64) * factor
    relative = float(np.max(np.abs(corrected.astype(np.float64) / expected - 1)))
    if relative > parameters["maximum_relative_rounding_error"]:
        raise ValueError("Scaled area exceeds float32 rounding tolerance")
    return (
        corrected,
        dict(
            source=source_binding,
            shape_hw=list(original.shape),
            dtype="float32",
            maximum_relative_rounding_error=relative,
            all_pixels_scaled=True,
            minimum_before_mm2=float(original.min()),
            maximum_before_mm2=float(original.max()),
            minimum_after_mm2=float(corrected.min()),
            maximum_after_mm2=float(corrected.max()),
        ),
    )


def apply_selected_area_scale(original_area, area_factor):
    """Scale an ordinary selected-frame area raster in its original NumPy dtype.

    This route intentionally differs from the measured parent-field recipe:
    multiplication uses the native array dtype before the same-dtype cast.
    The caller supplies the independently recorded area factor, not a value
    inferred from a linear coordinate scale.
    """
    return (original_area * float(area_factor)).astype(original_area.dtype)
