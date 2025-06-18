"""Flash calculation for two-phase flow analysis."""

from warnings import warn

import darsia
import numpy as np


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
