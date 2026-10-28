"""Flash calculation for two-phase flow analysis."""

import json
import logging
from pathlib import Path
from warnings import warn

import numpy as np

import darsia

logger = logging.getLogger(__name__)


class Flash:
    """Flash calculation."""

    def __init__(self, s_g_max: float = 1, s_g_cutoff: float = 0) -> None:
        """Constructor.

        Parameters
        ----------
        s_g_max : float
            Maximum saturation in gas phase.
        s_g_cutoff : float
            Cutoff saturation in gas phase.
        """
        self.s_g_max = s_g_max
        """Maximum saturation in gas phase."""
        self.s_g_cutoff = s_g_cutoff
        """Cutoff saturation in gas phase."""

    def __call__(
        self, c_g: darsia.Image, c_aq: darsia.Image
    ) -> tuple[darsia.Image, darsia.Image, darsia.Image, darsia.Image]:
        """Flash calculation.

        Parameters
        ----------
        c_g : Image
            Numerical concentration in gas phase.
        c_aq : Image
            Numerical concentration in aqueous phase.

        Returns
        -------
        tuple
            Volumetric concentration in gas phase, volumetric concentration in aqueous
            phase, saturation in gas phase, saturation in aqueous phase.
        """

        if np.max(c_g.img) > 1 + 1e-6:
            warn("Concentration of CO2 in gas phase has to be normalized.", UserWarning)

        if np.max(c_aq.img) > 1 + 1e-6:
            warn(
                "Concentration of CO2 in aqueous phase has to be normalized.",
                UserWarning,
            )

        # Extract physically compatible saturation values, taking into account residual
        # saturation, make sure that the saturation is not larger than the maximum saturation
        s_g = c_g.copy()
        s_g.img = self.s_g_max * np.clip(s_g.img, 0, 1)

        # Physically meaningful clean up
        cutoff_ind = c_g.img < self.s_g_cutoff
        s_g.img[cutoff_ind] = 0.0

        # Volume conservative saturation for acquous phase
        s_aq = s_g.copy()
        s_aq.img = 1 - s_g.img

        # Assume maximal volumetric concentration in gas phase
        chi_g = s_g.copy()  # * 1.

        # Assume c_aq indicates the concentration in the aqueous phase
        chi_aq = s_aq.copy()
        chi_aq.img[cutoff_ind] *= c_aq.img[cutoff_ind]

        return chi_g, chi_aq, s_g, s_aq


class AdvancedFlash(Flash):
    """Flash with a saturation cutoff and optional restoration of the result."""

    def __init__(
        self,
        s_g_max: float = 1.0,
        s_g_cutoff: float = 0.0,
        restoration=None,
    ) -> None:
        super().__init__(s_g_max, s_g_cutoff)
        """Flash object."""

        self.restoration = restoration
        """Restoration object."""

    def __call__(
        self, c_g: darsia.Image, c_aq: darsia.Image
    ) -> tuple[darsia.Image, darsia.Image, darsia.Image, darsia.Image]:
        """Flash calculation with restoration.

        Parameters
        ----------
        c_g : Image
            Numerical concentration in gas phase.
        c_aq : Image
            Numerical concentration in aqueous phase.

        Returns
        -------
        tuple
            Volumetric concentration in gas phase, volumetric concentration in aqueous
            phase, saturation in gas phase, saturation in aqueous phase.
        """
        # Flash
        chi_g, chi_aq, s_g, s_aq = super().__call__(c_g, c_aq)

        # Restoration
        if self.restoration is not None:
            chi_g = self.restoration(chi_g)
            chi_aq = self.restoration(chi_aq)
            s_g = self.restoration(s_g)
            s_aq = self.restoration(s_aq)

        return chi_g, chi_aq, s_g, s_aq


class SimpleFlash:
    """Flash splitting a signal into aqueous and gas fractions by fixed bounds."""

    def __init__(
        self,
        min_value_aq: float,
        max_value_aq: float,
        min_value_g: float,
        max_value_g: float,
        restoration=None,
    ) -> None:
        """Constructor."""

        self.min_value_aq = min_value_aq
        """Minimum value."""
        self.max_value_aq = max_value_aq
        """Maximum value."""
        self.min_value_g = min_value_g
        """Minimum value."""
        self.max_value_g = max_value_g
        """Maximum value."""
        self.restoration = restoration
        """Restoration object."""

    def update(
        self,
        min_value_aq: float | None = None,
        max_value_aq: float | None = None,
        min_value_g: float | None = None,
        max_value_g: float | None = None,
    ) -> None:
        """Update of internal parameters.

        Parameters
        ----------
        min_value_aq : float | None
            Minimum value for aqueous phase.
        max_value_aq : float | None
            Maximum value for aqueous phase.
        min_value_g : float | None
            Minimum value for gas phase.
        max_value_g : float | None
            Maximum value for gas phase.
        """
        self.min_value_aq = min_value_aq or self.min_value_aq
        self.max_value_aq = max_value_aq or self.max_value_aq
        self.min_value_g = min_value_g or self.min_value_g
        self.max_value_g = max_value_g or self.max_value_g

    @darsia.timing_decorator
    def __call__(self, signal: darsia.Image) -> tuple[darsia.Image, darsia.Image]:
        """Simple flash calculation.

        Parameters
        ----------
        signal : Image
            Numerical signal.

        Returns
        -------
        tuple
            Volumetric concentration in aqueous phase, saturation in gas phase.
        """
        # if np.isclose(self.cut_off, 0):
        #    c_aq = darsia.full_like(signal, 0.0)
        # else:
        #    c_aq = darsia.full_like(
        #        signal, np.clip(signal.img, 0, self.cut_off) / self.cut_off
        #    )
        # if self.max_value is None:
        #    s_g = darsia.full_like(signal, np.clip(signal.img - self.cut_off, 0, None))
        # elif np.isclose(self.max_value, self.cut_off):
        #    s_g = darsia.full_like(signal, signal.img >= self.cut_off)
        # else:
        #    s_g = darsia.full_like(
        #        signal,
        #        (np.clip(signal.img, self.cut_off, self.max_value) - self.cut_off)
        #        / (self.max_value - self.cut_off),
        #    )
        # if self.restoration is not None:
        #    c_aq = self.restoration(c_aq)
        #    s_g = self.restoration(s_g)

        # return c_aq, s_g

        c_aq = darsia.full_like(
            signal,
            (
                np.clip(signal.img, self.min_value_aq, self.max_value_aq)
                - self.min_value_aq
            )
            / (self.max_value_aq - self.min_value_aq),
        )
        s_g = darsia.full_like(
            signal,
            (np.clip(signal.img, self.min_value_g, self.max_value_g) - self.min_value_g)
            / (self.max_value_g - self.min_value_g),
        )
        if self.restoration is not None:
            c_aq = self.restoration(c_aq)
            s_g = self.restoration(s_g)
        return c_aq, s_g

    def to_dict(self) -> dict:
        """Convert the SimpleFlash parameters to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the SimpleFlash parameters.
        """
        return {
            "min_value_aq": self.min_value_aq,
            "max_value_aq": self.max_value_aq,
            "min_value_g": self.min_value_g,
            "max_value_g": self.max_value_g,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimpleFlash":
        """Create a SimpleFlash from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary representation of the SimpleFlash parameters.
        """
        return cls(
            min_value_aq=data["min_value_aq"],
            max_value_aq=data.get("max_value_aq"),
            min_value_g=data.get("min_value_g"),
            max_value_g=data.get("max_value_g"),
        )

    def save(self, path: Path) -> None:
        """Save the SimpleFlash parameters to a file.

        Parameters
        ----------
        path : Path
            The path to the file where the parameters will be saved.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(self.to_dict(), f)
        logger.info(f"Saved SimpleFlash parameters to {path}.")

    @classmethod
    def load(cls, path: Path) -> "SimpleFlash":
        """Load the SimpleFlash parameters from a file.

        Parameters
        ----------
        path : Path
            The path to the file from which the parameters will be loaded.

        Returns
        -------
        SimpleFlash
            The loaded SimpleFlash instance.
        """
        with open(path.with_suffix(".json"), "r") as f:
            data = json.load(f)
        flash_cls = TitrationFlash if cls is SimpleFlash and data.get("model") == "titration" else cls
        flash = flash_cls.from_dict(data)
        logger.info(f"Loaded SimpleFlash parameters from {path}.")
        return flash


def _titration_yellow_fraction(
    dic_M: np.ndarray,
    alkalinity_M: float,
    btb_M: float,
    pKa1: float,
    pKa2: float,
    pKaB: float,
) -> np.ndarray:
    """Yellow (protonated, HIn) fraction of a BTB pH-indicator solution as a function of
    dissolved inorganic carbon (DIC), from carbonate + indicator equilibrium.

    Physics: adding CO2 (a weak acid) titrates the strong-base alkalinity set by NaOH.
    The pH follows from the charge/proton balance

        [Na+] + [H+] = [OH-] + [HCO3-] + 2[CO3=] + [In-]

    and the BTB colour is the protonation state HIn/(HIn+In-) = [H+]/([H+]+Ka_BTB),
    i.e. the yellow fraction. Solved for [H+] by bisection per DIC value (vectorised loop;
    this is built once into a small lookup table, never per pixel).
    """
    Kw = 1e-14
    Ka1 = 10.0 ** (-pKa1)
    Ka2 = 10.0 ** (-pKa2)
    KaB = 10.0 ** (-pKaB)
    dic_M = np.asarray(dic_M, dtype=np.float64)
    out = np.empty_like(dic_M)
    for i, CT in enumerate(dic_M):
        lo, hi = 0.5, 13.5  # pH bracket
        for _ in range(60):  # bisection -> ~1e-18 in [H+]
            mid = 0.5 * (lo + hi)
            H = 10.0 ** (-mid)
            OH = Kw / H
            denom = H * H + Ka1 * H + Ka1 * Ka2
            f = (alkalinity_M + H - OH
                 - CT * Ka1 * H / denom - 2.0 * CT * Ka1 * Ka2 / denom
                 - btb_M * KaB / (KaB + H))
            if f > 0.0:  # too acidic side of balance -> raise pH
                lo = mid
            else:
                hi = mid
        H = 10.0 ** (-0.5 * (lo + hi))
        out[i] = H / (H + KaB)
    return out


def _alkalinity_from_initial_ph(
    initial_ph: float,
    btb_m: float,
    pkab: float,
) -> float:
    """Return effective alkalinity reproducing a zero-DIC initial pH."""

    resolved_ph = float(initial_ph)
    resolved_btb = float(btb_m)
    resolved_pkab = float(pkab)
    if not np.isfinite(resolved_ph) or not 0.5 < resolved_ph < 13.5:
        raise ValueError("initial_pH must be finite and between 0.5 and 13.5")
    if not np.isfinite(resolved_btb) or resolved_btb < 0.0:
        raise ValueError("btb_M must be finite and non-negative")
    hydrogen = 10.0**-resolved_ph
    hydroxide = 1.0e-14 / hydrogen
    kab = 10.0**-resolved_pkab
    deprotonated_btb = resolved_btb * kab / (kab + hydrogen)
    return float(hydroxide + deprotonated_btb - hydrogen)


def carbonate_dic_from_ph(
    ph: float | np.ndarray,
    *,
    alkalinity_M: float,
    btb_M: float = 0.0,
    pKa1: float = 6.35,
    pKa2: float = 10.33,
    pKaB: float = 7.10,
) -> np.ndarray:
    """Infer DIC from pH using the charge balance used by ``TitrationFlash``.

    The returned concentration is in mol/litre. Negative concentrations caused by
    pH values above the zero-DIC reference state are clipped to zero. Temperature
    and pressure dependence must be represented by reviewed pKa values upstream;
    this function does not silently substitute an environmental correction.
    """

    ph_array = np.asarray(ph, dtype=np.float64)
    if np.any(~np.isfinite(ph_array)) or np.any((ph_array <= 0.5) | (ph_array >= 13.5)):
        raise ValueError("pH must be finite and between 0.5 and 13.5")
    if not np.isfinite(alkalinity_M) or alkalinity_M < 0.0:
        raise ValueError("alkalinity_M must be finite and non-negative")
    if not np.isfinite(btb_M) or btb_M < 0.0:
        raise ValueError("btb_M must be finite and non-negative")

    hydrogen = np.power(10.0, -ph_array)
    hydroxide = 1.0e-14 / hydrogen
    ka1 = 10.0**-float(pKa1)
    ka2 = 10.0**-float(pKa2)
    kab = 10.0**-float(pKaB)
    denominator = hydrogen * hydrogen + ka1 * hydrogen + ka1 * ka2
    alpha1 = ka1 * hydrogen / denominator
    alpha2 = ka1 * ka2 / denominator
    deprotonated_btb = float(btb_M) * kab / (kab + hydrogen)
    charge_per_dic = np.maximum(alpha1 + 2.0 * alpha2, 1.0e-15)
    dic = (
        float(alkalinity_M)
        + hydrogen
        - hydroxide
        - deprotonated_btb
    ) / charge_per_dic
    return np.maximum(dic, 0.0)


def carbonate_species_from_ph(
    ph: float | np.ndarray,
    *,
    alkalinity_M: float,
    btb_M: float = 0.0,
    pKa1: float = 6.35,
    pKa2: float = 10.33,
    pKaB: float = 7.10,
) -> dict[str, np.ndarray]:
    """Return DIC and carbonate species for a pH value or array."""

    ph_array = np.asarray(ph, dtype=np.float64)
    dic = carbonate_dic_from_ph(
        ph_array,
        alkalinity_M=alkalinity_M,
        btb_M=btb_M,
        pKa1=pKa1,
        pKa2=pKa2,
        pKaB=pKaB,
    )
    hydrogen = np.power(10.0, -ph_array)
    ka1 = 10.0**-float(pKa1)
    ka2 = 10.0**-float(pKa2)
    denominator = hydrogen * hydrogen + ka1 * hydrogen + ka1 * ka2
    alpha0 = hydrogen * hydrogen / denominator
    alpha1 = ka1 * hydrogen / denominator
    alpha2 = ka1 * ka2 / denominator
    return {
        "dic_M": dic,
        "co2_aqueous_M": dic * alpha0,
        "bicarbonate_M": dic * alpha1,
        "carbonate_M": dic * alpha2,
        "alpha0": alpha0,
        "alpha1": alpha1,
        "alpha2": alpha2,
    }


class TitrationFlash(SimpleFlash):
    """SimpleFlash whose AQUEOUS branch uses the physically-derived BTB/carbonate
    titration curve instead of a linear ramp. The gas branch (s_g) is unchanged.

    Why: the linear ``c_aq(signal)`` ramp assigns aqueous mass roughly proportional to
    colour-path position, which over-counts the dilute plume fringe. A pixel at the
    colour-transition midpoint reads ``c_aq ~ 0.5`` under the linear ramp (-> ~50 % of
    CO2 solubility), but the measured indicator chemistry (e.g. 9.07 g BTB + 1.00 g NaOH
    per 20 L) puts the transition midpoint at dissolved CO2 ~1 mM, i.e. only ~3 % of
    solubility. Mapping the normalised signal through the inverse titration curve makes
    transition pixels carry their true (small) DIC, removing the spreading-area
    over-detection while leaving the well-behaved saturated/free-gas regime (which is why
    ff_um worked) essentially untouched.

    Signal convention: the normalised position ``(signal - min_value_aq) /
    (max_value_aq - min_value_aq)`` in [0, 1] is treated as the normalised yellow fraction
    (0 = blue baseline, 1 = full yellow). ``min_value_aq`` / ``max_value_aq`` thus keep
    their meaning (the signal window of the colour transition) and stay optimisable; only
    the SHAPE between them is replaced by the fixed physical curve. ``c_aq`` is returned
    normalised to [0, 1] (= DIC / CO2 solubility), exactly as the linear flash, so the
    downstream ``solubility * c_aq`` mass term is unchanged.
    """

    def __init__(
        self,
        min_value_aq: float,
        max_value_aq: float,
        min_value_g: float,
        max_value_g: float,
        restoration=None,
        alkalinity_M: float = 1.25e-3,
        btb_M: float = 0.726e-3,
        pKa1: float = 6.35,
        pKa2: float = 10.33,
        pKaB: float = 7.10,
        co2_sat_M: float = 0.034,
        n_lut: int = 512,
        aqueous_blend: float = 0.0,
        initial_pH: float | None = None,
    ) -> None:
        if initial_pH is not None:
            alkalinity_M = _alkalinity_from_initial_ph(initial_pH, btb_M, pKaB)
        super().__init__(min_value_aq, max_value_aq, min_value_g, max_value_g, restoration)
        self.n_lut = int(n_lut)
        self.aqueous_blend = float(aqueous_blend)
        self.initial_pH = None if initial_pH is None else float(initial_pH)
        if not 0.0 <= self.aqueous_blend <= 1.0:
            raise ValueError("aqueous_blend must be between 0 and 1")
        self.titration_params = {
            "alkalinity_M": alkalinity_M,
            "btb_M": btb_M,
            "pKa1": pKa1,
            "pKa2": pKa2,
            "pKaB": pKaB,
            "co2_sat_M": co2_sat_M,
        }
        self._build_lut(self.n_lut)

    def _build_lut(self, n_lut: int) -> None:
        """Precompute the (normalised yellow fraction -> normalised c_aq) lookup table.

        DIC grid is dense near 0 (the steep transition holds most of the colour change)
        and sparse toward saturation. Built once at construction; __call__ is a cheap
        np.interp over this table.
        """
        tp = self.titration_params
        sat = tp["co2_sat_M"]
        half = max(8, n_lut // 2)
        CT = np.concatenate([
            np.linspace(0.0, 0.1 * sat, half, endpoint=False),
            np.linspace(0.1 * sat, sat, n_lut - half),
        ])
        y = _titration_yellow_fraction(
            CT, tp["alkalinity_M"], tp["btb_M"], tp["pKa1"], tp["pKa2"], tp["pKaB"]
        )
        # normalise yellow fraction to [0, 1] over [blue(0 DIC), full yellow(saturation)]
        y = (y - y[0]) / (y[-1] - y[0])
        caq = CT / sat  # normalised aqueous concentration (DIC / solubility)
        # ensure strictly increasing x for np.interp
        order = np.argsort(y)
        y = y[order]
        caq = caq[order]
        keep = np.concatenate([[True], np.diff(y) > 0])
        self._lut_y = y[keep]
        self._lut_caq = caq[keep]

    @darsia.timing_decorator
    def __call__(self, signal: darsia.Image) -> tuple[darsia.Image, darsia.Image]:
        denom = (self.max_value_aq - self.min_value_aq) or 1.0
        y_norm = np.clip((signal.img - self.min_value_aq) / denom, 0.0, 1.0)
        physical_concentration = np.interp(y_norm, self._lut_y, self._lut_caq)
        c_aq = darsia.full_like(
            signal,
            (1.0 - self.aqueous_blend) * physical_concentration
            + self.aqueous_blend * y_norm,
        )
        s_g = darsia.full_like(
            signal,
            (np.clip(signal.img, self.min_value_g, self.max_value_g) - self.min_value_g)
            / (self.max_value_g - self.min_value_g),
        )
        if self.restoration is not None:
            c_aq = self.restoration(c_aq)
            s_g = self.restoration(s_g)
        return c_aq, s_g

    @classmethod
    def from_simple(cls, flash: "SimpleFlash", **titration_kwargs) -> "TitrationFlash":
        """Build a TitrationFlash carrying an existing SimpleFlash's value window and
        restoration, overriding only the aqueous transfer shape."""
        return cls(
            min_value_aq=flash.min_value_aq,
            max_value_aq=flash.max_value_aq,
            min_value_g=flash.min_value_g,
            max_value_g=flash.max_value_g,
            restoration=getattr(flash, "restoration", None),
            **titration_kwargs,
        )

    def to_dict(self) -> dict:
        """Persist both the flash window and the physical titration recipe."""
        data = super().to_dict()
        data.update(
            {
                "model": "titration",
                "titration_params": dict(self.titration_params),
                "n_lut": self.n_lut,
                "aqueous_blend": self.aqueous_blend,
                "initial_pH": self.initial_pH,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TitrationFlash":
        """Restore a saved titration flash while accepting legacy value keys."""
        titration_params = dict(data.get("titration_params") or {})
        return cls(
            min_value_aq=data["min_value_aq"],
            max_value_aq=data.get("max_value_aq"),
            min_value_g=data.get("min_value_g"),
            max_value_g=data.get("max_value_g"),
            n_lut=int(data.get("n_lut", 512)),
            aqueous_blend=float(data.get("aqueous_blend", 0.0)),
            initial_pH=data.get("initial_pH"),
            **titration_params,
        )
