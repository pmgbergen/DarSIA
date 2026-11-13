"""Module with tools for CO2 mass analysis."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import darsia

EPSILON = 1e-12

GRAVITY = 9.81
"""Gravitational acceleration in m/s^2."""

PA2BAR = 1e-5
"""Conversion factor from Pascal to bar."""


class HydrostaticState:
    """Pressure and temperature fields in the rig, shared by all fluid models.

    The temperature varies with height by its gradient. The pressure is the
    atmospheric pressure at the top plus the hydrostatic pressure of the aqueous phase.
    """

    def __init__(
        self,
        baseline: darsia.Image,
        pressure: float,
        temperature: float,
        pressure_gradient: float = 0.0,
        temperature_gradient: float = 0.0,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        baseline : Image
            Baseline image, defining the geometry.
        pressure : float
            Atmospheric pressure in bar.
        temperature : float
            Temperature in Celsius.
        pressure_gradient : float
            Pressure gradient in bar per unit height.
        temperature_gradient : float
            Temperature gradient in Celsius per unit height.
        """
        self.baseline = baseline
        self.pressure = pressure
        self.temperature = temperature
        self.pressure_gradient = pressure_gradient
        self.temperature_gradient = temperature_gradient

    @property
    def height_map(self) -> np.ndarray:
        """Height of each voxel (along the first axis of the baseline)."""
        return np.linspace(0, self.baseline.dimensions[0], self.baseline.num_voxels[0])[
            :, None
        ] * np.ones(self.baseline.num_voxels)

    def temperature_map(self) -> np.ndarray:
        """Temperature in Celsius in each voxel, due to the gradient."""
        return np.ones_like(self.height_map) * self.temperature + (
            self.temperature_gradient * self.height_map
        )

    def top_pressure(self) -> float:
        """Atmospheric pressure in bar at the top of the rig."""
        return self.pressure + (self.pressure_gradient * self.baseline.dimensions[0])

    def pressure_map(self, system: darsia.FluidSystem) -> np.ndarray:
        """Hydrostatic pressure in bar in each voxel.

        The aqueous density is taken at the top pressure and the local temperature.

        Parameters
        ----------
        system : darsia.FluidSystem
            Fluid system providing the density of the aqueous phase.
        """
        water_density = system.density_aqueous(
            self.top_pressure(), self.temperature_map()
        )
        return self.top_pressure() + water_density * GRAVITY * self.height_map * PA2BAR


class CO2MassAnalysis:
    """Mass analysis of CO2 in water, with properties from the fluid system."""

    def __init__(
        self,
        baseline: darsia.Image,
        atmospheric_pressure: float = 1.010,
        atmospheric_temperature: float = 23.0,
        atmospheric_pressure_gradient: float = 0.0,
        atmospheric_temperature_gradient: float = 0.0,
        system: darsia.FluidSystem | None = None,
    ) -> None:
        """Initialization of mass analysis.

        Parameters
        ----------
        baseline : Image
            Baseline image.
        atmospheric_pressure : float
            Atmospheric pressure in bar.
        atmospheric_temperature : float
            Temperature in Celsius.
        atmospheric_pressure_gradient : float
            Atmospheric pressure gradient in bar per unit height.
        atmospheric_temperature_gradient : float
            Temperature gradient in Celsius per unit height.
        system : darsia.FluidSystem, optional
            Fluids and their properties; default is water and CO2.
        """
        self.system = darsia.FluidSystem.two_phase() if system is None else system
        """Fluid system providing densities and solubility."""
        self.baseline = baseline
        """Baseline image."""
        self.atmospheric_pressure = atmospheric_pressure
        """Atmospheric pressure in bar."""
        self.atmospheric_temperature = atmospheric_temperature
        """Temperature in Celsius."""
        self.atmospheric_pressure_gradient = atmospheric_pressure_gradient
        """Atmospheric pressure gradient in bar."""
        self.atmospheric_temperature_gradient = atmospheric_temperature_gradient
        """Temperature gradient in Celsius."""

        # Setup density of gaseous CO2 and solubility
        self.setup_density_gaseous_co2()

    def update_state(
        self,
        atmospheric_pressure: float | None = None,
        atmospheric_temperature: float | None = None,
        atmospheric_pressure_gradient: float | None = None,
        atmospheric_temperature_gradient: float | None = None,
    ) -> None:
        """Update atmospheric pressure and temperature.

        Parameters
        ----------
        atmospheric_pressure : float
            Atmospheric pressure in bar.
        atmospheric_temperature : float
            Temperature in Celsius.
        """
        if atmospheric_pressure is not None:
            self.atmospheric_pressure = atmospheric_pressure
        if atmospheric_temperature is not None:
            self.atmospheric_temperature = atmospheric_temperature
        if atmospheric_pressure_gradient is not None:
            self.atmospheric_pressure_gradient = atmospheric_pressure_gradient
        if atmospheric_temperature_gradient is not None:
            self.atmospheric_temperature_gradient = atmospheric_temperature_gradient

        # Setup density of gaseous CO2
        self.setup_density_gaseous_co2()

    @property
    def hydrostatic_state(self) -> HydrostaticState:
        """Pressure and temperature fields of the current state."""
        return HydrostaticState(
            self.baseline,
            self.atmospheric_pressure,
            self.atmospheric_temperature,
            self.atmospheric_pressure_gradient,
            self.atmospheric_temperature_gradient,
        )

    def atmospheric_temperature_map(self) -> np.ndarray:
        """Return atmospheric temperature map due to gradient."""
        return self.hydrostatic_state.temperature_map()

    def top_atmospheric_pressure(self) -> float:
        return self.hydrostatic_state.top_pressure()

    @property
    def max_density(self) -> float:
        """Largest mass per volume of a phase in kg/m3 (for example for plot ranges).

        This is the larger one of the maximal solubility and gas density; non-finite
        values are ignored.
        """
        values = np.concatenate(
            [np.ravel(self.solubility_co2), np.ravel(self.density_gaseous_co2)]
        ).astype(float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("Solubility and gas density contain no finite values.")
        return float(np.max(finite))

    @property
    def height_map(self) -> np.ndarray:
        return self.hydrostatic_state.height_map

    def log(self, path: Path) -> None:
        """Plot density, solubility, hydrostatic pressure, temperature.

        Parameters
        ----------
        path : Path
            Path to save the plots.
        """
        plt.figure("density")
        plt.imshow(self.density_gaseous_co2)
        plt.colorbar()
        plt.title(
            f"density gaseous CO2 - {self.atmospheric_pressure}"
            + f" bar - {self.atmospheric_temperature} deg Celsius"
        )
        plt.savefig(path / "density_gaseous_co2.png")
        plt.close()

        plt.figure("solubility")
        plt.imshow(self.solubility_co2)
        plt.colorbar()
        plt.title(
            f"solubility CO2 - {self.atmospheric_pressure}"
            + f" bar - {self.atmospheric_temperature} deg Celsius"
        )
        plt.savefig(path / "solubility_co2.png")
        plt.close()

    def setup_density_gaseous_co2(self) -> None:
        """Setup density of gaseous CO2 and CO2 solubility under hydrostatic pressure.

        The temperature varies with height due to the temperature gradient. The pressure
        is the atmospheric pressure at the top plus the hydrostatic pressure of the
        aqueous phase. Density and solubility are looked up in the fluid system.
        """

        state = self.hydrostatic_state
        atmospheric_temperature_map = state.temperature_map()
        hydrostatic_pressure = state.pressure_map(self.system)

        # Density of gaseous CO2 and CO2 solubility at hydrostatic pressure
        self.density_gaseous_co2 = self.system.density_gas(
            hydrostatic_pressure, atmospheric_temperature_map
        )
        self.solubility_co2 = self.system.solubility(
            hydrostatic_pressure, atmospheric_temperature_map
        )

    def __call__(
        self,
        chi_g: darsia.Image,
        chi_aq: darsia.Image,
    ) -> Tuple[darsia.Image, darsia.Image, darsia.Image]:
        """Determine mass of CO2, given maps for dissolved and gaseous CO2.

        Formulas are:
        - Mass of gaseous CO2: ( m_g = chi_g x rho_co2_g )
        - Mass of dissolved CO2: ( m_aq = chi_aq x solubility_co2 )
        - Total mass of CO2: ( m = m_g + m_aq )

        Parameters
        ----------
        chi_g : Image
            Volumetric concentration in gas phase.
        chi_aq : Image
            Volumetric concentration in aqueous.

        Returns
        -------
        Tuple[Image, Image, Image]
            Mass map of CO2, mass map of gaseous CO2, mass map of
            aqueous CO2.
        """
        # Allocate mass maps
        mass = darsia.zeros_like(chi_aq, mode="voxels", dtype=np.float32)
        mass_g = mass.copy()
        mass_aq = mass.copy()

        # Calculate mass of gaseous, dissolved and total CO2
        mass_g.img = chi_g.img * self.density_gaseous_co2
        mass_aq.img = chi_aq.img * self.solubility_co2
        mass.img = mass_g.img + mass_aq.img

        return mass, mass_g, mass_aq

    def mass_analysis(
        self,
        c_aq: darsia.Image,
        s_g: darsia.Image,
    ) -> None:  # darsia.SimpleMassAnalysisResults:
        """Determine mass of CO2, given maps for dissolved and gaseous CO2."""
        mass_g_array = self.density_gaseous_co2 * s_g.img
        mass_aq_array = self.solubility_co2 * c_aq.img * np.clip(1 - s_g.img, 0, None)
        return darsia.SimpleMassAnalysisResults(
            name=c_aq.name,  # TODO exclude?
            date=c_aq.date,  # TODO exclude?
            time=c_aq.time,  # TODO exclude?
            mass=darsia.full_like(c_aq, mass_g_array + mass_aq_array),
            mass_g=darsia.full_like(c_aq, mass_g_array),
            mass_aq=darsia.full_like(c_aq, mass_aq_array),
            saturation_g=s_g,  # TODO exclude?
            concentration_aq=c_aq,  # TODO exclude?
            color_signal=None,  # Exclude! TODO
        )

    def inverse_mass_analysis(
        self, mass: darsia.Image
    ) -> "darsia.SimpleMassAnalysisResults":
        """Determine phase maps from a total mass map and recompute components.

        Parameters
        ----------
        mass : darsia.Image
            Total CO2 mass map.

        Returns
        -------
        darsia.SimpleMassAnalysisResults
            Reconstructed concentrations, saturation,
            and component masses.
        """
        solubility = self.solubility_co2
        density = self.density_gaseous_co2

        c_aq_array = np.divide(
            mass.img,
            solubility,
            out=np.zeros_like(mass.img, dtype=float),
            where=np.abs(solubility) > EPSILON,
        )
        c_aq_array = np.clip(c_aq_array, 0.0, 1.0)

        numerator = np.clip(mass.img - solubility, 0.0, None)
        denominator = density - solubility
        s_g_array = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(mass.img, dtype=float),
            where=np.abs(denominator) > EPSILON,
        )
        s_g_array = np.clip(s_g_array, 0.0, 1.0)

        mass_g_array = density * s_g_array
        mass_aq_array = solubility * c_aq_array * np.clip(1 - s_g_array, 0.0, None)

        return darsia.SimpleMassAnalysisResults(
            name=mass.name,
            date=mass.date,
            time=mass.time,
            mass=darsia.full_like(mass, mass_g_array + mass_aq_array),
            mass_g=darsia.full_like(mass, mass_g_array),
            mass_aq=darsia.full_like(mass, mass_aq_array),
            saturation_g=darsia.full_like(mass, s_g_array),
            color_signal=None,
            concentration_aq=darsia.full_like(mass, c_aq_array),
        )


class TracerMassAnalysis:
    """Mass analysis of a single-phase tracer: there is no gas phase.

    The concentration is relative to the injected solution. The mass is a density per
    pore volume (like for CO2); integrating over the geometry yields the total mass of
    the solution, i.e. the volume times the water density.

    Like for CO2, the density depends on pressure and temperature, including their
    gradients (`HydrostaticState`), so it may vary in space.
    """

    def __init__(
        self,
        baseline: darsia.Image | None = None,
        atmospheric_pressure: float = 1.010,
        atmospheric_temperature: float = 20.0,
        atmospheric_pressure_gradient: float = 0.0,
        atmospheric_temperature_gradient: float = 0.0,
        system: darsia.FluidSystem | None = None,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        baseline : Image, optional
            Baseline image, defining the geometry of the pressure and temperature
            fields. Without it, the density is uniform at the atmospheric state.
        atmospheric_pressure : float
            Atmospheric pressure in bar.
        atmospheric_temperature : float
            Temperature in Celsius.
        atmospheric_pressure_gradient : float
            Atmospheric pressure gradient in bar per unit height.
        atmospheric_temperature_gradient : float
            Temperature gradient in Celsius per unit height.
        system : darsia.FluidSystem, optional
            Fluids and their properties; default is single-phase water.
        """
        self.system = darsia.FluidSystem.single_phase() if system is None else system
        """Fluid system providing the density of the aqueous phase."""
        self.baseline = baseline
        """Baseline image (geometry of the pressure and temperature fields)."""
        self.atmospheric_pressure = atmospheric_pressure
        """Atmospheric pressure in bar."""
        self.atmospheric_temperature = atmospheric_temperature
        """Temperature in Celsius."""
        self.atmospheric_pressure_gradient = atmospheric_pressure_gradient
        """Atmospheric pressure gradient in bar per unit height."""
        self.atmospheric_temperature_gradient = atmospheric_temperature_gradient
        """Temperature gradient in Celsius per unit height."""
        self.setup_density()

    @property
    def hydrostatic_state(self) -> HydrostaticState | None:
        """Pressure and temperature fields of the current state (None without baseline)."""
        if self.baseline is None:
            return None
        return HydrostaticState(
            self.baseline,
            self.atmospheric_pressure,
            self.atmospheric_temperature,
            self.atmospheric_pressure_gradient,
            self.atmospheric_temperature_gradient,
        )

    def setup_density(self) -> None:
        """Setup the mass per pore volume at relative concentration 1."""
        self.scale = float(
            self.system.density_aqueous(
                self.atmospheric_pressure, self.atmospheric_temperature
            )
        )
        """Reference density (kg/m3) of the solution at the atmospheric state, e.g. to
        convert an injected volume to a mass."""

        state = self.hydrostatic_state
        if state is None:
            self.density_aqueous = np.asarray(self.scale)
        else:
            self.density_aqueous = np.asarray(
                self.system.density_aqueous(
                    state.pressure_map(self.system), state.temperature_map()
                ),
                dtype=float,
            )
        """Mass per pore volume (kg/m3) at relative concentration 1 in each voxel."""

    def update_state(
        self,
        atmospheric_pressure: float | None = None,
        atmospheric_temperature: float | None = None,
        atmospheric_pressure_gradient: float | None = None,
        atmospheric_temperature_gradient: float | None = None,
    ) -> None:
        """Update atmospheric pressure and temperature, and their gradients.

        Parameters
        ----------
        atmospheric_pressure : float, optional
            Atmospheric pressure in bar.
        atmospheric_temperature : float, optional
            Temperature in Celsius.
        atmospheric_pressure_gradient : float, optional
            Atmospheric pressure gradient in bar per unit height.
        atmospheric_temperature_gradient : float, optional
            Temperature gradient in Celsius per unit height.
        """
        if atmospheric_pressure is not None:
            self.atmospheric_pressure = atmospheric_pressure
        if atmospheric_temperature is not None:
            self.atmospheric_temperature = atmospheric_temperature
        if atmospheric_pressure_gradient is not None:
            self.atmospheric_pressure_gradient = atmospheric_pressure_gradient
        if atmospheric_temperature_gradient is not None:
            self.atmospheric_temperature_gradient = atmospheric_temperature_gradient
        self.setup_density()

    @property
    def max_density(self) -> float:
        """Largest mass per volume of a phase in kg/m3 (for example for plot ranges)."""
        values = np.ravel(self.density_aqueous).astype(float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("The aqueous density contains no finite values.")
        return float(np.max(finite))

    def log(self, path: Path) -> None:
        """Plot the density of the solution.

        Parameters
        ----------
        path : Path
            Path to save the plot.
        """
        plt.figure("density")
        plt.imshow(np.atleast_2d(self.density_aqueous))
        plt.colorbar()
        plt.title(
            f"density aqueous - {self.atmospheric_pressure}"
            + f" bar - {self.atmospheric_temperature} deg Celsius"
        )
        plt.savefig(path / "density_aqueous.png")
        plt.close()

    def mass_analysis(
        self, c_aq: darsia.Image, s_g: darsia.Image | None = None
    ) -> "darsia.SimpleMassAnalysisResults":
        """Mass of the solution; `s_g` is ignored (there is no gas phase)."""
        mass_aq_array = self.density_aqueous * c_aq.img
        return darsia.SimpleMassAnalysisResults(
            name=c_aq.name,
            date=c_aq.date,
            time=c_aq.time,
            mass=darsia.full_like(c_aq, mass_aq_array),
            mass_g=darsia.zeros_like(c_aq),
            mass_aq=darsia.full_like(c_aq, mass_aq_array),
            saturation_g=darsia.zeros_like(c_aq),
            color_signal=None,
            concentration_aq=c_aq,
        )

    def inverse_mass_analysis(
        self, mass: darsia.Image
    ) -> "darsia.SimpleMassAnalysisResults":
        """Concentration from a (rescaled) mass map; never creates gas."""
        density = np.broadcast_to(self.density_aqueous, mass.img.shape)
        c_aq_array = np.divide(
            mass.img,
            density,
            out=np.zeros_like(mass.img, dtype=float),
            where=np.abs(density) > EPSILON,
        )
        c_aq_array = np.clip(c_aq_array, 0.0, 1.0)
        return self.mass_analysis(darsia.full_like(mass, c_aq_array))


class AdvancedCO2MassAnalysis:
    """CO2 mass analysis combining gas/aqueous concentration analyses and a flash."""

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

        Parameters
        ----------
        img : Image
            Input image.

        Returns
        -------
        tuple
            Mass of CO2, mass map of CO2.
        """
        c_g = self.concentration_analysis_g(img)
        c_aq = self.concentration_analysis_aq(img)
        # TODO integrate in concentration analysis!?
        # TODO any other expert knowledge somewhere?
        # TODO add logging that stores to a file?
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

        Parameters
        ----------
        img : Image
            Input image.

        Returns
        -------
        Image
            Mass of CO2.
        """
        return self(img)[0]

    def ndofs(self) -> int:
        """Return number of degrees of freedom of the mass analysis.

        Returns
        -------
        int
            Number of degrees of freedom.
        """
        return (
            self.concentration_analysis_g.ndofs()
            + self.concentration_analysis_aq.ndofs()
        )

    def update_parameters(self, params: np.ndarray) -> None:
        """Update parameters of the mass analysis.

        Parameters
        ----------
        params : np.ndarray
            Parameters.
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

    name: str
    """Name for the mass analysis result, e.g., name of raw image."""
    date: Optional[datetime]
    """Date of the mass analysis result."""
    time: Optional[float]
    """Relative time of the mass analysis result."""
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
            name=self.name,
            date=self.date,
            time=self.time,
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
