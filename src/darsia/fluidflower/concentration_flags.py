"""Per-pixel flags for dissolved-CO2 products: saturated and not quantified.

Two honest states that a concentration map must carry instead of a number:

* **saturated** – the indicator has used (almost) its whole colour range, so
  the colour no longer measures the dissolved CO2. The dissolved mass is then
  an interval: at least the value at the saturation threshold, at most the
  solubility at the lab temperature.
* **not quantified** – the pixel has no colour path (free water, excluded
  areas, outside the rig frame) or holds free gas, so the colour does not
  measure dissolved CO2 at all. Reported as a mask, never as zero.

The functions are pure numpy and series agnostic. The defaults (threshold
``f_min = 0.05``) are proposals awaiting the project owner's decision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from darsia.fluidflower.titration import TitrationModel

__all__ = [
    "DEFAULT_SATURATION_BLUE_FRACTION",
    "DissolvedInterval",
    "saturated_mask",
    "dissolved_interval",
    "not_quantified_mask",
    "flag_fractions",
]

DEFAULT_SATURATION_BLUE_FRACTION = 0.05
"""Blue fraction below which a pixel counts as saturated (95 % of the colour
range used; about 5 mM DIC for the FF_RE recipe)."""


def saturated_mask(blue_fraction, f_min: float = DEFAULT_SATURATION_BLUE_FRACTION):
    """True where the indicator is saturated (blue fraction below f_min).
    NaN blue fractions are never saturated (they are not quantified)."""
    f = np.asarray(blue_fraction, dtype=float)
    with np.errstate(invalid="ignore"):
        return (f < f_min) & np.isfinite(f)


@dataclass(frozen=True)
class DissolvedInterval:
    """Lower and upper bound of the dissolved CO2 (mol/L) per pixel, with the
    saturation mask that explains where the two differ."""

    lower: np.ndarray
    upper: np.ndarray
    saturated: np.ndarray


def dissolved_interval(blue_fraction, model: TitrationModel,
                       f_min: float = DEFAULT_SATURATION_BLUE_FRACTION,
                       p_co2_atm: float = 1.0, valid_mask=None) -> DissolvedInterval:
    """Interval of the dissolved CO2 per pixel.

    Below the threshold the colour is only a lower bound: ``lower`` is the DIC
    at ``f_min`` and ``upper`` the solubility at ``p_co2_atm``. Elsewhere both
    bounds equal the closed-form DIC of the blue fraction. Invalid colour
    (NaN, outside [0, 1], or False in ``valid_mask``) gives NaN in both bounds
    and is never saturated. The bounds are ordered by construction; a
    threshold whose DIC exceeds the solubility is rejected."""
    if not 0.0 < f_min < 1.0:
        raise ValueError("f_min must lie in (0, 1)")
    if not (np.isfinite(p_co2_atm) and p_co2_atm > 0.0):
        raise ValueError("p_co2_atm must be a finite positive pressure")
    f = np.asarray(blue_fraction, dtype=float)
    valid = np.isfinite(f) & (f >= 0.0) & (f <= 1.0)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    f = np.where(valid, f, np.nan)
    sat = saturated_mask(f, f_min)
    threshold_dic = float(model.dic_from_blue_fraction(f_min))
    solubility = float(model.saturation_dic(p_co2_atm))
    if not (np.isfinite(threshold_dic) and np.isfinite(solubility) and solubility > 0.0):
        raise ValueError("the model gives no finite positive threshold and solubility for these inputs")
    if solubility < threshold_dic:
        raise ValueError("saturation threshold f_min lies above the solubility at p_co2_atm")
    value = model.dic_from_blue_fraction(np.where(sat, f_min, f))
    upper = np.where(sat, solubility, value)
    upper = np.where(valid, np.maximum(upper, value), np.nan)
    return DissolvedInterval(lower=value, upper=upper, saturated=sat)


def not_quantified_mask(labels, quantified_labels, gas_mask=None, outside_mask=None,
                        invalid_colour_mask=None):
    """True where the colour does not measure dissolved CO2: labels without a
    colour path (anything not in ``quantified_labels``), free gas, pixels
    outside the rig frame, and pixels whose colour input is invalid (NaN or
    rejected by the caller's validity test)."""
    labels = np.asarray(labels)
    quantified = np.isin(labels, np.asarray(list(quantified_labels)))
    mask = ~quantified
    for extra in (gas_mask, outside_mask, invalid_colour_mask):
        if extra is not None:
            mask |= np.asarray(extra, dtype=bool)
    return mask


def flag_fractions(saturated, not_quantified, sand_mask) -> dict:
    """Area fractions for the result table: saturated and not-quantified shares
    of the sand area (the area that is meant to be quantified)."""
    sand = np.asarray(sand_mask, dtype=bool)
    n = int(sand.sum())
    if n == 0:
        return {"saturated_fraction": float("nan"), "not_quantified_fraction": float("nan"), "sand_pixels": 0}
    sat = np.asarray(saturated, dtype=bool) & sand
    nq = np.asarray(not_quantified, dtype=bool) & sand
    return {
        "saturated_fraction": float(sat.sum() / n),
        "not_quantified_fraction": float(nq.sum() / n),
        "sand_pixels": n,
    }
