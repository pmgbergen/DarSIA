"""Module with tools for CO2 mass analysis."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

import darsia


class CO2MassAnalysis:
    """Interpolation routines, based on NIST data."""

    def __init__(
        self,
        baseline: darsia.Image,
        atmospheric_pressure: float = 1.010,
        temperature: float = 23.0,
        log: Optional[Path] = None,
    ) -> None:
        """Initialization of mass analysis.

        Args:
            baseline (Image): baseline image
            atmospheric_pressure (float): atmospheric pressure in bar
            temperature (float): temperature in Celsius
            max_solubility_co2 (float): maximum solubility of CO2 in water
            log (Path): path to log file

        """
        self.baseline = baseline
        """Baseline image."""
        self.atmospheric_pressure = atmospheric_pressure
        """Atmospheric pressure in bar."""
        self.temperature = temperature
        """Temperature in Celsius."""

        self.setup_20_degrees_celsius()
        self.setup_23_degrees_celsius()

        # Setup density of gaseous CO2
        self.setup_density_gaseous_co2()

        self.log(log)

    def log(self, log: Optional[Path]) -> None:
        """Plot density, solubility, hydrostatic pressure, tempreature."""
        if log:
            plt.figure("density")
            plt.imshow(self.density_gaseous_co2)
            plt.colorbar()
            plt.title(
                f"density gaseous CO2 - {self.atmospheric_pressure}"
                + f" bar - {self.temperature} deg Celsius"
            )
            plt.savefig(log / "density_gaseous_co2.png")
            plt.close()

            plt.figure("solubility")
            plt.imshow(self.solubility_co2)
            plt.colorbar()
            plt.title(
                f"solubility CO2 - {self.atmospheric_pressure}"
                + f" bar - {self.temperature} deg Celsius"
            )
            plt.savefig(log / "solubility_co2.png")
            plt.close()

    def setup_20_degrees_celsius(self) -> None:
        self.water_density_20 = 998.21  # kg/m^3 at 20 deg Celsius
        """Water derisity at 20 degrees Celsius."""

        # NIST data for co2 density at fixed temperature and varying pressure
        self.data_NIST_20 = (
            # pressure (bar), density (kg/m^3); at 20 deg Celsius
            [0.90 + 0.01 * i for i in range(61)],
            [
                1.6328,
                1.6510,
                1.6692,
                1.6875,
                1.7057,
                1.7239,
                1.7422,
                1.7604,
                1.7787,
                1.7969,
                1.8152,
                1.8334,
                1.8517,
                1.8699,
                1.8882,
                1.9064,
                1.9247,
                1.9429,
                1.9612,
                1.9795,
                1.9977,
                2.0160,
                2.0343,
                2.0526,
                2.0708,
                2.0891,
                2.1074,
                2.1257,
                2.1439,
                2.1622,
                2.1805,
                2.1988,
                2.2171,
                2.2354,
                2.2537,
                2.2720,
                2.2903,
                2.3086,
                2.3269,
                2.3452,
                2.3635,
                2.3818,
                2.4001,
                2.4186,
                2.4367,
                2.4550,
                2.4734,
                2.4917,
                2.5100,
                2.5283,
                2.5467,
                2.5650,
                2.5833,
                2.6016,
                2.6200,
                2.6383,
                2.6566,
                2.6750,
                2.6933,
                2.7117,
                2.7300,
            ],
        )

        self.solubility_co2_20 = (
            # pressure (bar), solubility (kg/m^3); at 20 deg Celsius
            [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            [1.53, 1.7, 1.87, 2.04, 2.21, 2.37, 2.54],
        )

    def setup_23_degrees_celsius(self) -> None:
        self.water_density_23 = 997.62  # kg/m^3 at 23 deg Celsius

        # NIST data for co2 density at fixed temperature and varying pressure
        self.data_NIST_23 = (
            # pressure (bar), density (kg/m^3); at 23 deg Celsius
            [0.90 + 0.01 * i for i in range(61)],
            [
                1.6160,
                1.6340,
                1.6521,
                1.6701,
                1.6882,
                1.7062,
                1.7242,
                1.7423,
                1.7604,
                1.7784,
                1.7965,
                1.8145,
                1.8326,
                1.8506,
                1.8687,
                1.8868,
                1.9048,
                1.9229,
                1.9410,
                1.9590,
                1.9771,
                1.9952,
                2.0133,
                2.0314,
                2.0494,
                2.0675,
                2.0856,
                2.1037,
                2.1218,
                2.1399,
                2.1580,
                2.1761,
                2.1942,
                2.2123,
                2.2304,
                2.2485,
                2.2666,
                2.2847,
                2.3028,
                2.3209,
                2.3390,
                2.3571,
                2.3752,
                2.3934,
                2.4115,
                2.4296,
                2.4477,
                2.4658,
                2.4840,
                2.5021,
                2.5202,
                2.5384,
                2.5565,
                2.5746,
                2.5928,
                2.6109,
                2.6291,
                2.6472,
                2.6653,
                2.6835,
                2.7016,
            ],
        )

        self.solubility_co2_23 = (
            # pressure (bar), solubility (kg/m^3); at 23 deg Celsius
            [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            # [1.44, 1.6, 1.76, 1.92, 2.08],
            [1.30, 1.45, 1.60, 1.75, 1.90, 2.05, 2.18],
        )

    def setup_density_gaseous_co2(self) -> None:
        """Setup density of gaseous CO2 taking into account pressure variation.

        NOTE: Fixed temperature at 23 deg Celsius.

        """

        # Hydrostatic pressure distribution
        height_map = np.linspace(
            0, self.baseline.dimensions[0], self.baseline.num_voxels[0]
        )[:, None] * np.ones(self.baseline.num_voxels)
        g = 9.81  # m/s^2
        pa2bar = 1e-5  # Conversion factor from Pascal to bar

        # Interpolate water density depending on temperature
        water_density = self.water_density_20 * (23 - self.temperature) / 3 + (
            self.water_density_23 * (self.temperature - 20) / 3
        )

        # Determine hydrostatic pressure
        hydrostatic_pressure = (
            self.atmospheric_pressure + water_density * g * height_map * pa2bar
        )

        # Interpolate NIST data: pressure -> density
        self.co2_density_interpolator_20 = interpolate.interp1d(
            self.data_NIST_20[0], self.data_NIST_20[1], kind="linear"
        )

        # Interpolate NIST data: pressure -> density
        self.co2_density_interpolator_23 = interpolate.interp1d(
            self.data_NIST_23[0], self.data_NIST_23[1], kind="linear"
        )

        self.co2_solubility_interpolator_20 = interpolate.interp1d(
            self.solubility_co2_20[0], self.solubility_co2_20[1], kind="linear"
        )

        self.co2_solubility_interpolator_23 = interpolate.interp1d(
            self.solubility_co2_23[0], self.solubility_co2_23[1], kind="linear"
        )

        # Define density of gaseous CO2 at hydrostatic pressure
        density_gaseous_co2_20 = self.co2_density_interpolator_20(hydrostatic_pressure)
        density_gaseous_co2_23 = self.co2_density_interpolator_23(hydrostatic_pressure)
        solubility_co2_20 = self.co2_solubility_interpolator_20(hydrostatic_pressure)
        solubility_co2_23 = self.co2_solubility_interpolator_23(hydrostatic_pressure)

        # Interpolate density of gaseous CO2 depending on temperature
        self.density_gaseous_co2 = (
            density_gaseous_co2_20 * (23 - self.temperature) / 3
            + density_gaseous_co2_23 * (self.temperature - 20) / 3
        )

        self.solubility_co2 = (
            solubility_co2_20 * (23 - self.temperature) / 3
            + solubility_co2_23 * (self.temperature - 20) / 3
        )

        if False:
            self.solubility_co2[:, :] = 1.8
            warn("constant solubility?")

    def density_co2(self, temperature: float, pressure: float) -> float:
        density_co2_20 = self.co2_density_interpolator_20(pressure)
        density_co2_23 = self.co2_density_interpolator_23(pressure)
        return (
            density_co2_20 * (23 - temperature) / 3
            + density_co2_23 * (temperature - 20) / 3
        )

    def hydrostatic_pressure(self, depth: float) -> np.ndarray:
        # Hydrostatic pressure distribution
        height_map = np.linspace(
            0, self.baseline.dimensions[0], self.baseline.num_voxels[0]
        )[:, None]
        g = 9.81  # m/s^2
        pa2bar = 1e-5  # Conversion factor from Pascal to bar

        # Interpolate water density depending on temperature
        water_density = self.water_density_20 * (23 - self.temperature) / 3 + (
            self.water_density_23 * (self.temperature - 20) / 3
        )

        # Determine hydrostatic pressure
        hydrostatic_pressure = (
            self.atmospheric_pressure + water_density * g * height_map * pa2bar
        )

        # Convert depth to index
        index = int(depth / self.baseline.dimensions[0] * self.baseline.num_voxels[0])

        return hydrostatic_pressure[index]

    def __call__(self, chi_g: darsia.Image, chi_aq: darsia.Image) -> darsia.Image:
        """Analyze mass of CO2, given maps for dissolved and gaseous CO2.

        Args:
            chi_g (Image): volumetric concentration in gas phase
            chi_aq (Image): volumetric concentration in aqueous

        Returns:
            mass map of CO2

        """
        # Allocate mass map
        mass = darsia.zeros_like(chi_aq, mode="voxels", dtype=np.float32)
        mass_g = darsia.zeros_like(chi_aq, mode="voxels", dtype=np.float32)
        mass_aq = darsia.zeros_like(chi_aq, mode="voxels", dtype=np.float32)

        # Calculate mass of dissolved CO2
        mass_aq.img = chi_aq.img * self.solubility_co2

        # Calculate mass of gas
        mass_g.img = chi_g.img * self.density_gaseous_co2

        mass.img = mass_g.img + mass_aq.img

        return mass, mass_g, mass_aq


class AdvancedCO2MassAnalysis:
    def __init__(
        self,
        concentration_analysis_g,
        concentration_analysis_aq,
        restoration,
        flash,
        mass_analysis,
    ) -> None:
        self.concentration_analysis_g = concentration_analysis_g
        """Concentration analysis object for gaseous CO2."""
        self.concentration_analysis_aq = concentration_analysis_aq
        """Concentration analysis object for dissolved CO2."""
        self.restoration = restoration
        """Restoration object."""
        self.flash = flash
        """Flash object."""
        self.mass_analysis = mass_analysis
        """Mass analysis object."""

    def __call__(
        self, img: darsia.Image
    ) -> tuple[darsia.Image, darsia.Image, darsia.Image, darsia.Image, darsia.Image]:
        """Analyze mass of CO2, given maps for dissolved and gaseous CO2.

        Args:
            img (Image): input image

        Returns:
            tuple: mass of CO2, mass map of CO2

        """
        c_g = self.concentration_analysis_g(img)
        c_aq = self.concentration_analysis_aq(img)
        # TODO integrate in concentration analysis!?
        # TODO any other expert knowledge somewhere?
        c_g.img = np.clip(c_g.img, 0, 1)
        c_aq.img = np.clip(c_aq.img, 0, 1)
        if self.restoration is not None:
            c_g = self.restoration(c_g)
            c_aq = self.restoration(c_aq)
        chi_g, chi_aq, s_g, s_aq = self.flash(c_g, c_aq)
        mass = self.mass_analysis(chi_g, chi_aq)
        return mass, chi_g, chi_aq, s_g, s_aq

    def mass(self, img: darsia.Image) -> darsia.Image:
        """Analyze mass of CO2, given maps for dissolved and gaseous CO2.

        Args:
            img (Image): input image

        Returns:
            Image: mass of CO2

        """
        return self(img)[0]

    def ndofs(self) -> int:
        """Return number of degrees of freedom of the mass analysis.

        Returns:
            int: number of degrees of freedom

        """
        return (
            self.concentration_analysis_g.ndofs()
            + self.concentration_analysis_aq.ndofs()
        )

    def update_parameters(self, params: np.ndarray) -> None:
        """Update parameters of the mass analysis.

        Args:
            params (np.ndarray): parameters

        """
        ndofs_g = self.concentration_analysis_g.ndofs()
        ndofs_aq = self.concentration_analysis_aq.ndofs()
        self.concentration_analysis_g.update_parameters(params[:ndofs_g])
        self.concentration_analysis_aq.update_parameters(
            params[ndofs_g : ndofs_g + ndofs_aq]
        )


@dataclass
class MassAnalysisResults:
    """Summary object."""

    mass: darsia.Image
    """Total mass of phase."""
    mass_g: darsia.Image
    """Mass of gaseous phase."""
    mass_aq: darsia.Image
    """Mass of aqueous phase."""
    chi_g: darsia.Image
    """Volumetric concentration of gaseous phase."""
    chi_aq: darsia.Image
    """Volume concentration of aqueous phase."""
    saturation_g: darsia.Image
    """Saturation of gaseous phase."""
    saturation_aq: darsia.Image
    """Saturation of aqueous phase."""
    normalized_signal_g: darsia.Image
    """Normalized signal of gaseous phase."""
    normalized_signal_aq: darsia.Image
    """Normalized signal of aqueous phase."""
    signal_g: darsia.Image
    """Signal of gaseous phase."""
    signal_aq: darsia.Image
    """Signal of aqueous phase."""
    concentration_co2_aq: darsia.Image
    """Concentration of CO2 in aqueous phase."""

    def subregion(self, roi: darsia.CoordinateArray) -> "MassAnalysisResults":
        return MassAnalysisResults(
            mass=self.mass.subregion(roi),
            mass_g=self.mass_g.subregion(roi),
            mass_aq=self.mass_aq.subregion(roi),
            chi_g=self.chi_g.subregion(roi),
            chi_aq=self.chi_aq.subregion(roi),
            saturation_g=self.saturation_g.subregion(roi),
            saturation_aq=self.saturation_aq.subregion(roi),
            normalized_signal_g=self.normalized_signal_g.subregion(roi),
            normalized_signal_aq=self.normalized_signal_aq.subregion(roi),
            signal_g=self.signal_g.subregion(roi),
            signal_aq=self.signal_aq.subregion(roi),
            concentration_co2_aq=self.concentration_co2_aq.subregion(roi),
        )


@dataclass
class ThresholdAnalysisResults:
    co2: darsia.Image
    """Thresholded image of CO2."""
    co2_g: darsia.Image
    """Thresholded image of gaseous phase."""

    def subregion(self, roi: darsia.CoordinateArray) -> "ThresholdAnalysisResults":
        return ThresholdAnalysisResults(
            co2=self.co2.subregion(roi), co2_g=self.co2_g.subregion(roi)
        )
