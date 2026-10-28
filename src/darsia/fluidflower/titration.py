"""Titration model for a bromothymol-blue (BTB) / NaOH indicator solution.

The model turns the colour state of the indicator into dissolved inorganic
carbon (DIC) in the pore water and back, with the solution recipe as input
instead of a fitted curve. It is the physical link behind

* the background CO2 at the start of an injection cycle (from the colour of
  the cycle baseline),
* the saturation flag: where the colour has used almost its whole range the
  dissolved mass is only bounded from below,
* a physical check of a colour-to-mass calibration.

Chemistry (zero ionic strength, mM solutions): the species are H+, OH-,
dissolved CO2 (H2CO3*), HCO3-, CO3 2- and the indicator in its yellow (HIn-)
and blue (In 2-) forms. The blue fraction is f = Ka_I / (H + Ka_I). The
charge balance for the sodium salt of BTB reads

    A + H = Kw/H + [HCO3-] + 2 [CO3 2-] + f * B_T,

with A the NaOH concentration and B_T the total indicator. For the free acid
of BTB the sulfonic group consumes one OH- per molecule, so A = [NaOH] - B_T.
DIC -> colour solves the charge balance for H (monotone, bisection).
Colour -> DIC is closed form: H = Ka_I (1 - f) / f and

    DIC = (A + H - Kw/H - f B_T) * (H^2 + K1 H + K1 K2) / (K1 H + 2 K1 K2).

Reference values for the FF_RE recipe (1 g NaOH and 9.04 g BTB per 20 L) are
in ``tests/fluidflower/test_titration.py``. Every default in this module is a
proposal awaiting the project owner's decision; none of it changes
the existing FF_AC method.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

__all__ = [
    "IndicatorRecipe",
    "TitrationConstants",
    "TitrationModel",
    "FFRE_RECIPE",
    "M_CO2_G_PER_MOL",
]

M_CO2_G_PER_MOL = 44.01
_M_NAOH = 39.997
_M_BTB = {"sodium_salt": 646.36, "free_acid": 624.38}


@dataclass(frozen=True)
class IndicatorRecipe:
    """Indicator solution recipe in grams per litre of water."""

    naoh_g_per_l: float
    btb_g_per_l: float
    form: str = "sodium_salt"

    def __post_init__(self) -> None:
        if self.form not in _M_BTB:
            raise ValueError(f"unknown BTB form {self.form!r}; use 'sodium_salt' or 'free_acid'")
        if self.naoh_g_per_l < 0 or self.btb_g_per_l < 0:
            raise ValueError("recipe amounts must be non-negative")

    @property
    def sodium_m(self) -> float:
        """NaOH concentration, mol/L."""
        return self.naoh_g_per_l / _M_NAOH

    @property
    def btb_total_m(self) -> float:
        """Total indicator concentration, mol/L."""
        return self.btb_g_per_l / _M_BTB[self.form]

    @property
    def alkalinity_m(self) -> float:
        """Alkalinity available to CO2 and the indicator, mol/L."""
        if self.form == "free_acid":
            return self.sodium_m - self.btb_total_m
        return self.sodium_m


# FF_RE recipe (project owner, 18 Oct 2026): 1 g NaOH and 9.04 g BTB per 20 L; the lye
# wash uses the same 1 g NaOH per 20 L. The BTB form is assumed to be the sodium
# salt: the free acid would give a green solution at pH 7.5 without any CO2.
FFRE_RECIPE = IndicatorRecipe(naoh_g_per_l=1.0 / 20.0, btb_g_per_l=9.04 / 20.0)


@dataclass(frozen=True)
class TitrationConstants:
    """Equilibrium constants at a temperature, zero ionic strength."""

    temperature_c: float
    k1: float
    k2: float
    kw: float
    k_indicator: float
    henry_mol_per_l_atm: float

    @classmethod
    def at(cls, temperature_c: float = 25.0, pka_indicator: float = 7.10) -> "TitrationConstants":
        """Carbonate constants after Plummer & Busenberg (1982), Kw after
        Harned & Owen, Henry solubility by van 't Hoff from 0.034 mol/(L atm)
        at 25 C. The BTB pKa (7.10) is taken as temperature independent."""
        t = temperature_c + 273.15
        log_k1 = -356.3094 - 0.06091964 * t + 21834.37 / t + 126.8339 * math.log10(t) - 1684915.0 / t**2
        log_k2 = -107.8871 - 0.03252849 * t + 5151.79 / t + 38.92561 * math.log10(t) - 563713.9 / t**2
        pkw = 4470.99 / t - 6.0875 + 0.01706 * t
        henry = 0.034 * math.exp(2400.0 * (1.0 / t - 1.0 / 298.15))
        return cls(temperature_c, 10**log_k1, 10**log_k2, 10**-pkw, 10**-pka_indicator, henry)


class TitrationModel:
    """Colour state <-> dissolved CO2 for one recipe and temperature."""

    def __init__(self, recipe: IndicatorRecipe = FFRE_RECIPE, temperature_c: float = 25.0,
                 pka_indicator: float = 7.10, blue_fraction_floor: float = 1e-9,
                 clip_negative_dic: bool = True) -> None:
        """``blue_fraction_floor`` keeps the closed form finite at f = 0 and
        f = 1 (proposal: 1e-9); ``clip_negative_dic`` maps the small negative
        DIC that noise above the fresh-solution colour would give to 0
        (proposal: True). Both are policies for the project owner's decision, not physics.
        Invalid inputs (NaN, negative DIC, f outside [0, 1]) give NaN."""
        if not 0.0 < blue_fraction_floor < 0.5:
            raise ValueError("blue_fraction_floor must lie in (0, 0.5)")
        self.recipe = recipe
        self.constants = TitrationConstants.at(temperature_c, pka_indicator)
        self.alkalinity = recipe.alkalinity_m
        self.btb_total = recipe.btb_total_m
        self.blue_fraction_floor = float(blue_fraction_floor)
        self.clip_negative_dic = bool(clip_negative_dic)

    # -- building blocks -----------------------------------------------------
    def blue_fraction_from_h(self, h):
        """Fraction of the indicator in the blue (In 2-) form at proton
        concentration h (mol/L)."""
        h = np.asarray(h, dtype=float)
        return self.constants.k_indicator / (h + self.constants.k_indicator)

    def blue_fraction_from_ph(self, ph):
        return self.blue_fraction_from_h(10.0 ** -np.asarray(ph, dtype=float))

    def charge_residual(self, h, dic):
        """A + H - (OH- + HCO3- + 2 CO3 2- + f B_T); positive where the pH is
        too low for the given DIC (bisection helper)."""
        c = self.constants
        h = np.asarray(h, dtype=float)
        dic = np.asarray(dic, dtype=float)
        d = h * h + c.k1 * h + c.k1 * c.k2
        hco3 = dic * c.k1 * h / d
        co3 = dic * c.k1 * c.k2 / d
        return self.alkalinity + h - (c.kw / h + hco3 + 2.0 * co3 + self.blue_fraction_from_h(h) * self.btb_total)

    # -- DIC -> colour --------------------------------------------------------
    def ph_from_dic(self, dic, iterations: int = 100):
        """pH of the solution holding DIC (mol/L); vectorised bisection on
        [2, 13]. DIC may be any array shape; NaN or negative DIC gives NaN."""
        dic = np.asarray(dic, dtype=float)
        valid = np.isfinite(dic) & (dic >= 0.0)
        safe = np.where(valid, dic, 0.0)
        lo = np.full(dic.shape, 2.0)
        hi = np.full(dic.shape, 13.0)
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            positive = self.charge_residual(10.0**-mid, safe) > 0.0
            lo = np.where(positive, mid, lo)
            hi = np.where(positive, hi, mid)
        return np.where(valid, 0.5 * (lo + hi), np.nan)

    def blue_fraction_from_dic(self, dic):
        return self.blue_fraction_from_ph(self.ph_from_dic(dic))

    # -- colour -> DIC --------------------------------------------------------
    def dic_from_blue_fraction(self, f):
        """Closed-form DIC (mol/L) from the blue fraction f in [0, 1].
        f outside [0, 1] or NaN gives NaN; f at the ends is kept finite by
        ``blue_fraction_floor``; negative results are clipped to 0 when
        ``clip_negative_dic`` is set."""
        c = self.constants
        f = np.asarray(f, dtype=float)
        valid = np.isfinite(f) & (f >= 0.0) & (f <= 1.0)
        fl = self.blue_fraction_floor
        fs = np.clip(np.where(valid, f, 0.5), fl, 1.0 - fl)
        h = c.k_indicator * (1.0 - fs) / fs
        d = h * h + c.k1 * h + c.k1 * c.k2
        dic = (self.alkalinity + h - c.kw / h - fs * self.btb_total) * d / (c.k1 * h + 2.0 * c.k1 * c.k2)
        if self.clip_negative_dic:
            dic = np.maximum(dic, 0.0)
        return np.where(valid, dic, np.nan)

    # -- reference states -----------------------------------------------------
    def fresh_ph(self) -> float:
        """pH of the fresh solution without any CO2."""
        return float(self.ph_from_dic(0.0))

    def saturation_dic(self, p_co2_atm: float = 1.0) -> float:
        """DIC at equilibrium with CO2 gas at p_co2_atm (essentially all as
        dissolved CO2 at the resulting low pH), mol/L."""
        c = self.constants
        co2 = c.henry_mol_per_l_atm * p_co2_atm
        ph = self._ph_open(co2)
        h = 10.0**-ph
        return float(co2 * (1.0 + c.k1 / h + c.k1 * c.k2 / h**2))

    def air_equilibrium(self, p_co2_atm: float = 4.2e-4) -> tuple[float, float, float]:
        """(pH, DIC mol/L, blue fraction) of the solution fully equilibrated
        with air at the given CO2 partial pressure."""
        c = self.constants
        co2 = c.henry_mol_per_l_atm * p_co2_atm
        ph = self._ph_open(co2)
        h = 10.0**-ph
        dic = co2 * (1.0 + c.k1 / h + c.k1 * c.k2 / h**2)
        return float(ph), float(dic), float(self.blue_fraction_from_h(h))

    def _ph_open(self, co2_aq: float, iterations: int = 100) -> float:
        """pH when the dissolved CO2 is fixed by a gas phase (open system)."""
        c = self.constants
        lo, hi = 2.0, 13.0
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            h = 10.0**-mid
            dic = co2_aq * (1.0 + c.k1 / h + c.k1 * c.k2 / h**2)
            if self.charge_residual(h, dic) > 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # -- mass ----------------------------------------------------------------
    @staticmethod
    def mass_per_pixel_g(dic_mol_per_l, porosity, depth_m, pixel_area_m2):
        """Dissolved CO2 mass per pixel in grams: DIC (mol/L = 1000 mol/m3)
        times the pore volume of the pixel times the molar mass. Every
        argument may be a scalar or an array that broadcasts with the others
        (native per-pixel area arrays included)."""
        dic = np.asarray(dic_mol_per_l, dtype=float)
        return dic * 1000.0 * np.asarray(porosity, dtype=float) * np.asarray(depth_m, dtype=float) \
            * np.asarray(pixel_area_m2, dtype=float) * M_CO2_G_PER_MOL
