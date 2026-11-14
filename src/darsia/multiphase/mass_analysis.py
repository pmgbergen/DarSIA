"""Module with tools for mass analysis of fluids (CO2, tracer)."""

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


class FluidMassAnalysis:
    """Mass analysis of a fluid system: one formula for all fluids.

    The mass of the injected component per pore volume is

        m = rho_g s_g + chi_aq c_aq (1 - s_g),

    with the gas density rho_g and the aqueous capacity chi_aq, the mass per aqueous
    volume at c_aq = 1: the solubility for a gas (CO2), the density of the injected
    solution for a single-phase tracer. Both depend on pressure and temperature,
    including their gradients (`HydrostaticState`). A single-phase system has no gas,
    rho_g = s_g = 0.
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
            fields. Without it, the properties are uniform at the atmospheric state.
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
        """Fluid system providing densities and the aqueous capacity."""
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
        self._state_key: tuple | None = None
        """State of the current properties, to skip recomputing an unchanged state."""
        self.setup_state()

    @property
    def has_gas(self) -> bool:
        """Whether the system has a gas phase."""
        return self.system.is_multiphase

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

    def _state(self) -> tuple:
        return (
            self.atmospheric_pressure,
            self.atmospheric_temperature,
            self.atmospheric_pressure_gradient,
            self.atmospheric_temperature_gradient,
        )

    def setup_state(self) -> None:
        """Evaluate the fluid properties of the current state.

        The temperature varies with height due to its gradient; the pressure is the
        atmospheric pressure at the top plus the hydrostatic pressure of the aqueous
        phase. The properties are looked up in the fluid system.
        """
        self.scale = float(
            self.system.density_aqueous(
                self.atmospheric_pressure, self.atmospheric_temperature
            )
        )
        """Reference density (kg/m3) of the aqueous phase at the atmospheric state, e.g.
        to convert an injected volume to a mass."""

        state = self.hydrostatic_state
        if state is None:
            pressure = self.atmospheric_pressure
            temperature = self.atmospheric_temperature
        else:
            pressure = state.pressure_map(self.system)
            temperature = state.temperature_map()

        self.density_gas = np.asarray(
            self.system.density_gas(pressure, temperature), dtype=float
        )
        """Gas density (kg/m3) in each voxel; zero without a gas phase."""
        self.aqueous_capacity = np.asarray(
            self.system.aqueous_capacity(pressure, temperature), dtype=float
        )
        """Mass per aqueous volume (kg/m3) at c_aq = 1 in each voxel."""
        self._state_key = self._state()

    def update_state(
        self,
        atmospheric_pressure: float | None = None,
        atmospheric_temperature: float | None = None,
        atmospheric_pressure_gradient: float | None = None,
        atmospheric_temperature_gradient: float | None = None,
    ) -> None:
        """Update atmospheric pressure and temperature, and their gradients.

        The properties are only evaluated again if the state changed.

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
        if self._state() != self._state_key:
            self.setup_state()

    @property
    def max_density(self) -> float:
        """Largest mass per volume of a phase in kg/m3 (for example for plot ranges).

        This is the larger one of the maximal aqueous capacity and gas density;
        non-finite values are ignored.
        """
        values = np.concatenate(
            [np.ravel(self.aqueous_capacity), np.ravel(self.density_gas)]
        ).astype(float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(
                "Aqueous capacity and gas density contain no finite values."
            )
        return float(np.max(finite))

    def log(self, path: Path) -> None:
        """Plot the aqueous capacity and, with a gas phase, the gas density.

        Parameters
        ----------
        path : Path
            Path to save the plots.
        """
        state = (
            f" - {self.atmospheric_pressure} bar"
            + f" - {self.atmospheric_temperature} deg Celsius"
        )
        plots = [("aqueous_capacity", self.aqueous_capacity)]
        if self.has_gas:
            plots.append(("density_gas", self.density_gas))
        for name, values in plots:
            plt.figure(name)
            plt.imshow(np.atleast_2d(values))
            plt.colorbar()
            plt.title(name.replace("_", " ") + state)
            plt.savefig(path / f"{name}.png")
            plt.close()

    def mass_analysis(
        self, c_aq: darsia.Image, s_g: darsia.Image | None = None
    ) -> "darsia.SimpleMassAnalysisResults":
        """Mass of the injected component, given the fields c_aq and s_g.

        Parameters
        ----------
        c_aq : Image
            Relative concentration in the aqueous phase.
        s_g : Image, optional
            Gas saturation; ignored (zero) without a gas phase.
        """
        if s_g is None or not self.has_gas:
            s_g = darsia.zeros_like(c_aq)
        mass_g_array = self.density_gas * s_g.img
        mass_aq_array = self.aqueous_capacity * c_aq.img * np.clip(1 - s_g.img, 0, None)
        return darsia.SimpleMassAnalysisResults(
            name=c_aq.name,
            date=c_aq.date,
            time=c_aq.time,
            mass=darsia.full_like(c_aq, mass_g_array + mass_aq_array),
            mass_g=darsia.full_like(c_aq, mass_g_array),
            mass_aq=darsia.full_like(c_aq, mass_aq_array),
            saturation_g=s_g,
            concentration_aq=c_aq,
            color_signal=None,
        )

    def inverse_mass_analysis(
        self, mass: darsia.Image
    ) -> "darsia.SimpleMassAnalysisResults":
        """Fields c_aq and s_g from a (rescaled) total mass map; recompute the masses.

        Up to the aqueous capacity, the mass is dissolved (c_aq = m / chi_aq); the excess
        is gas (s_g = (m - chi_aq) / (rho_g - chi_aq)). Without a gas phase, s_g = 0.

        Parameters
        ----------
        mass : darsia.Image
            Total mass map.

        Returns
        -------
        darsia.SimpleMassAnalysisResults
            Reconstructed concentration, saturation, and component masses.
        """
        capacity = np.broadcast_to(self.aqueous_capacity, mass.img.shape)
        c_aq_array = np.divide(
            mass.img,
            capacity,
            out=np.zeros_like(mass.img, dtype=float),
            where=np.abs(capacity) > EPSILON,
        )
        c_aq_array = np.clip(c_aq_array, 0.0, 1.0)
        if not self.has_gas:
            return self.mass_analysis(darsia.full_like(mass, c_aq_array))

        density = self.density_gas
        numerator = np.clip(mass.img - capacity, 0.0, None)
        denominator = density - capacity
        s_g_array = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(mass.img, dtype=float),
            where=np.abs(denominator) > EPSILON,
        )
        s_g_array = np.clip(s_g_array, 0.0, 1.0)

        mass_g_array = density * s_g_array
        mass_aq_array = capacity * c_aq_array * np.clip(1 - s_g_array, 0.0, None)

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


class CO2MassAnalysis(FluidMassAnalysis):
    """Mass analysis of CO2 in water (two-phase `FluidMassAnalysis`).

    Kept for compatibility: CO2-specific names of the generic properties.
    """

    def __init__(
        self,
        baseline: darsia.Image,
        *args,
        system: darsia.FluidSystem | None = None,
        **kwargs,
    ) -> None:
        """Constructor, see `FluidMassAnalysis`; default system is water and CO2."""
        system = darsia.FluidSystem.two_phase() if system is None else system
        super().__init__(baseline, *args, system=system, **kwargs)

    @property
    def density_gaseous_co2(self) -> np.ndarray:
        """Density of gaseous CO2 (alias of `density_gas`)."""
        return self.density_gas

    @density_gaseous_co2.setter
    def density_gaseous_co2(self, value: np.ndarray) -> None:
        self.density_gas = value

    @property
    def solubility_co2(self) -> np.ndarray:
        """Solubility of CO2 in water (alias of `aqueous_capacity`)."""
        return self.aqueous_capacity

    @solubility_co2.setter
    def solubility_co2(self, value: np.ndarray) -> None:
        self.aqueous_capacity = value

    def setup_density_gaseous_co2(self) -> None:
        """Evaluate the properties of the current state (alias of `setup_state`)."""
        self.setup_state()

    def atmospheric_temperature_map(self) -> np.ndarray:
        """Return atmospheric temperature map due to gradient."""
        return self.hydrostatic_state.temperature_map()

    def top_atmospheric_pressure(self) -> float:
        return self.hydrostatic_state.top_pressure()

    @property
    def height_map(self) -> np.ndarray:
        return self.hydrostatic_state.height_map

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


class TracerMassAnalysis(FluidMassAnalysis):
    """Mass analysis of a single-phase tracer (single-phase `FluidMassAnalysis`).

    The concentration is relative to the injected solution; integrating the mass over
    the geometry yields the volume times the density of the solution. Kept for
    compatibility: tracer-specific names of the generic properties.
    """

    @property
    def density_aqueous(self) -> np.ndarray:
        """Density of the solution in each voxel (alias of `aqueous_capacity`)."""
        return self.aqueous_capacity

    @density_aqueous.setter
    def density_aqueous(self, value: np.ndarray) -> None:
        self.aqueous_capacity = value

    def setup_density(self) -> None:
        """Evaluate the properties of the current state (alias of `setup_state`)."""
        self.setup_state()


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
