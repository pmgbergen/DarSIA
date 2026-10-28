"""Original baseline-only old-light residual with explicit method parameters.

The arithmetic is retained from the saved producer's
``ffac_v5_local_old_residual.py`` (SHA-256 f2f11c66394baaf585223e2295cb3173df6c00283fec23ca0c30feecbdc56380).
Inputs are already bound optical fitting signals, never phase predictions.
"""

import cv2
import numpy as np

from . import photometry


def residual_clean_mask(fit, consensus, *, parameters, photometry_parameters):
    """Reproduce the original baseline-only residual-colour exclusion mask."""
    keep = fit.sand & photometry._same_fac(fit.fac, parameters=photometry_parameters)
    colour = photometry._colour(fit.logB, fit.Minv, parameters=photometry_parameters)
    signal = colour - (fit.F if fit.F is not None else 0)
    delta = (signal[2] - signal[1]) - (consensus[2] - consensus[1])
    finite = keep & np.isfinite(delta)
    num = cv2.GaussianBlur(np.where(finite, delta, 0).astype(np.float32),
                           (0, 0), parameters["colour_sigma"])
    den = cv2.GaussianBlur(finite.astype(np.float32), (0, 0), parameters["colour_sigma"])
    delta = np.where(finite, num / np.maximum(den, parameters["division_floor"]), np.nan)
    centred = keep & np.isfinite(delta)
    if centred.sum() > parameters["minimum_centring_pixels"]:
        delta[fit.fac == 1] -= np.median(delta[centred])
    clean = keep & np.isfinite(delta) & (delta > -parameters["residual_colour_limit"])
    fraction = float(1 - clean.sum() / max(keep.sum(), 1))
    if abs(fraction - fit.residual_exclusion_fraction) > parameters["fraction_tolerance"]:
        raise ValueError(f"{fit.run}: source B RESCOL mask differs from fit")
    return clean


def estimate_old_B_field(fit, reference, consensus, corrected_B, *, parameters,
                         photometry_parameters):
    """Estimate the saved achromatic Gaussian residual from the baseline only."""
    shape = fit.sand.shape
    if (reference.sand.shape != shape or consensus.shape != (3, *shape)
            or corrected_B.shape != (3, *shape)):
        raise ValueError("Old-light B and virtual consensus geometry differ")
    clean = residual_clean_mask(fit, consensus, parameters=parameters,
                                photometry_parameters=photometry_parameters)
    support = (clean & reference.sand & fit.sand & ~fit.chart &
               np.isfinite(consensus[1]) & np.isfinite(corrected_B[1]))
    if int(support.sum()) < parameters["minimum_support_pixels"]:
        raise ValueError(f"{fit.run}: insufficient source-clean old-light sand")
    residual = consensus[1] - corrected_B[1]
    den = cv2.GaussianBlur(support.astype(np.float32), (0, 0),
                           parameters["sigma_reference_cells"])
    num = cv2.GaussianBlur(np.where(support, residual, 0).astype(np.float32),
                           (0, 0), parameters["sigma_reference_cells"])
    field = num / np.maximum(den, parameters["division_floor"])
    field[den < parameters["minimum_smooth_support"]] = 0
    field *= fit.taper * (~fit.chart)
    if not np.all(np.isfinite(field)):
        raise ArithmeticError("Non-finite old-light B residual field")
    return field.astype(np.float32), {
        "method": "source_B_G_residual_normalized_gaussian_scalar_all_RGB",
        "sigma_reference_cells": parameters["sigma_reference_cells"],
        "minimum_smooth_support": parameters["minimum_smooth_support"],
        "source_clean_sand_cells": int(support.sum()),
        "fitted_rescol_exclusion_fraction": fit.residual_exclusion_fraction,
        "chart_cells_zero": bool(np.all(field[fit.chart] == 0)),
        "same_field_all_channels": True,
    }


def apply_old_field(corrected, field):
    """Add the original scalar residual equally to all three log-RGB channels."""
    rgb = np.asarray(corrected)
    light = np.asarray(field)
    if rgb.dtype != np.float32 or light.dtype != np.float32:
        raise TypeError("Old-light correction requires float32 log-linear RGB and scalar field")
    if rgb.ndim != 3 or rgb.shape[0] != 3 or rgb.shape[1:] != light.shape:
        raise ValueError("Old-light field geometry differs from RGB")
    return (rgb + light[None]).astype(np.float32)
