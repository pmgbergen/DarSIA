"""Fluid systems: single-phase (aqueous only) and two-phase (aqueous and gas)."""

from __future__ import annotations

import numpy as np

from darsia.pvt import data
from darsia.pvt.fluid import Fluid, get_fluid
from darsia.pvt.table import PropertyTable

_SOLUBILITY_TABLES: dict[tuple[str, str], PropertyTable] = {
    ("water", "co2"): PropertyTable(
        temperatures=data.TEMPERATURES,
        pressures=data.CO2_SOLUBILITY_PRESSURES,
        values=(data.CO2_SOLUBILITY_20, data.CO2_SOLUBILITY_23),
        name="co2 solubility in water",
    )
}
"""Solubility of the gas in the aqueous fluid (kg/m3) by (aqueous, gas) names."""


class FluidSystem:
    """Fluids present in an experiment: an aqueous phase and optionally a gas phase.

    Single-phase systems have only the aqueous phase (for example a tracer in water);
    two-phase systems add a gas that dissolves in the aqueous phase. The aqueous
    phase is always water, the only supported gas is CO2.

    Units: pressure in bar, temperature in degrees Celsius, density in kg/m3.
    """

    def __init__(
        self,
        aqueous: Fluid,
        gas: Fluid | None = None,
        solubility: PropertyTable | None = None,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        aqueous : Fluid
            Fluid of the aqueous phase (liquid).
        gas : Fluid, optional
            Fluid of the gas phase; None for a single-phase system.
        solubility : PropertyTable, optional
            Mass of dissolved gas per aqueous volume at saturation in kg/m3; required
            if and only if there is a gas phase.
        """
        if aqueous.phase != "liquid":
            raise ValueError(f"Aqueous fluid must be a liquid, got {aqueous}.")
        if gas is not None and gas.phase != "gas":
            raise ValueError(f"Gas fluid must be a gas, got {gas}.")
        if (gas is None) != (solubility is None):
            raise ValueError("A solubility is needed if and only if there is a gas.")
        self.aqueous = aqueous
        """Fluid of the aqueous phase."""
        self.gas = gas
        """Fluid of the gas phase, None for a single-phase system."""
        self._solubility = solubility

    @classmethod
    def single_phase(cls, aqueous: str = "water") -> "FluidSystem":
        """System with the aqueous phase only (for example a tracer)."""
        return cls(get_fluid(aqueous))

    @classmethod
    def two_phase(cls, aqueous: str = "water", gas: str = "co2") -> "FluidSystem":
        """System with an aqueous and a gas phase (for example water and CO2)."""
        try:
            solubility = _SOLUBILITY_TABLES[(aqueous, gas)]
        except KeyError:
            raise KeyError(
                f"No solubility of '{gas}' in '{aqueous}' available, use one of "
                f"{sorted(_SOLUBILITY_TABLES)}."
            ) from None
        return cls(get_fluid(aqueous), get_fluid(gas), solubility)

    @property
    def is_multiphase(self) -> bool:
        """Whether there is a gas phase."""
        return self.gas is not None

    def density_aqueous(
        self, pressure: float | np.ndarray, temperature: float | np.ndarray
    ) -> float | np.ndarray:
        """Density of the aqueous fluid in kg/m3."""
        return self.aqueous.density(pressure, temperature)

    def density_gas(
        self, pressure: float | np.ndarray, temperature: float | np.ndarray
    ) -> float | np.ndarray:
        """Density of the gas in kg/m3; zero for a single-phase system."""
        if self.gas is None:
            return np.zeros(np.broadcast(pressure, temperature).shape)
        return self.gas.density(pressure, temperature)

    def solubility(
        self, pressure: float | np.ndarray, temperature: float | np.ndarray
    ) -> float | np.ndarray:
        """Dissolved gas per aqueous volume at saturation in kg/m3; zero if no gas."""
        if self._solubility is None:
            return np.zeros(np.broadcast(pressure, temperature).shape)
        return self._solubility(pressure, temperature)

    def __repr__(self) -> str:
        gas = None if self.gas is None else self.gas.name
        return f"FluidSystem(aqueous={self.aqueous.name!r}, gas={gas!r})"
