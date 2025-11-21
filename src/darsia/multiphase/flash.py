"""Flash calculation for two-phase flow analysis."""

from warnings import warn

import numpy as np
from pathlib import Path
import json

import darsia
import logging

logger = logging.getLogger(__name__)


class Flash:
    """Flash calculation."""

    def __init__(self, s_g_max: float = 1, s_g_cutoff: float = 0) -> None:
        """Constructor.

        Args:
            s_g_max (float): maximum saturation in gas phase
            s_g_cutoff (float): cutoff saturation in gas phase

        """
        self.s_g_max = s_g_max
        """Maximum saturation in gas phase."""
        self.s_g_cutoff = s_g_cutoff
        """Cutoff saturation in gas phase."""

    def __call__(
        self, c_g: darsia.Image, c_aq: darsia.Image
    ) -> tuple[darsia.Image, darsia.Image, darsia.Image, darsia.Image]:
        """Flash calculation.

        Args:
            c_g (Image): numerical concentration in gas phase
            c_aq (Image): numerical concentration in aqueous phase

        Returns:
            tuple: volumetric concentration in gas phase, volumetric concentration in aqueous
                phase, saturation in gas phase, saturation in aqueous phase

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

        Args:
            c_g (Image): numerical concentration in gas phase
            c_aq (Image): numerical concentration in aqueous phase

        Returns:
            tuple: volumetric concentration in gas phase, volumetric concentration in aqueous
                phase, saturation in gas phase, saturation in aqueous phase

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
    def __init__(
        self,
        cut_off: float,
        max_value: float | None = None,
        restoration=None,
    ) -> None:
        """Constructor."""

        self.cut_off = cut_off
        """Cut-off value."""
        self.max_value = max_value
        """Maximum value."""
        self.restoration = restoration
        """Restoration object."""

    def update(
        self, cut_off: float | None = None, max_value: float | None = None
    ) -> None:
        """Update of internal parameters.

        Args:
            cut_off (float, optional): cut-off value
            max_value (float, optional): maximum value

        """
        if cut_off is not None:
            self.cut_off = cut_off

        if max_value is not None:
            self.max_value = max_value

    @darsia.timing_decorator
    def __call__(self, signal: darsia.Image) -> tuple[darsia.Image, darsia.Image]:
        """Simple flash calculation.

        Args:
            signal (Image): numerical signal

        Returns:
            tuple: volumetric concentration in aqueous phase, saturation in gas phase

        """
        if np.isclose(self.cut_off, 0):
            c_aq = darsia.full_like(signal, 0.0)
        else:
            c_aq = darsia.full_like(
                signal, np.clip(signal.img, 0, self.cut_off) / self.cut_off
            )
        if self.max_value is None:
            s_g = darsia.full_like(signal, np.clip(signal.img - self.cut_off, 0, None))
        elif np.isclose(self.max_value, self.cut_off):
            s_g = darsia.full_like(signal, signal.img >= self.cut_off)
        else:
            s_g = darsia.full_like(
                signal,
                (np.clip(signal.img, self.cut_off, self.max_value) - self.cut_off)
                / (self.max_value - self.cut_off),
            )
        if self.restoration is not None:
            c_aq = self.restoration(c_aq)
            s_g = self.restoration(s_g)

        return c_aq, s_g

    def to_dict(self) -> dict:
        """Convert the SimpleFlash parameters to a dictionary.

        Returns:
            dict: Dictionary representation of the SimpleFlash parameters.

        """
        return {
            "cut_off": self.cut_off,
            "max_value": self.max_value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimpleFlash":
        """Create a SimpleFlash from a dictionary.

        Args:
            data (dict): Dictionary representation of the SimpleFlash parameters.

        """
        return cls(
            cut_off=data["cut_off"],
            max_value=data.get("max_value"),
        )

    def save(self, path: Path) -> None:
        """Save the SimpleFlash parameters to a file.

        Args:
            path (Path): The path to the file where the parameters will be saved.

        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(self.to_dict(), f)
        logger.info(f"Saved SimpleFlash parameters to {path}.")

    @classmethod
    def load(cls, path: Path) -> "SimpleFlash":
        """Load the SimpleFlash parameters from a file.

        Args:
            path (Path): The path to the file from which the parameters will be loaded.

        Returns:
            SimpleFlash: The loaded SimpleFlash instance.

        """
        with open(path.with_suffix(".json"), "r") as f:
            data = json.load(f)
        flash = cls.from_dict(data)
        logger.info(f"Loaded SimpleFlash parameters from {path}.")
        return flash
