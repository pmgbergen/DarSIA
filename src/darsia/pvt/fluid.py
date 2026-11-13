"""Fluids with a standardized density API, realized by lookup tables."""

from __future__ import annotations

import numpy as np

from darsia.pvt import data
from darsia.pvt.table import PropertyTable


class Fluid:
    """A pure fluid with density as function of pressure and temperature.

    Units: pressure in bar, temperature in degrees Celsius, density in kg/m3.
    """

    PHASES = ("liquid", "gas")
    """Supported phases."""

    def __init__(self, name: str, phase: str, density_table: PropertyTable) -> None:
        """Constructor.

        Parameters
        ----------
        name : str
            Name of the fluid.
        phase : str
            Phase in which the fluid appears, "liquid" or "gas".
        density_table : PropertyTable
            Density in kg/m3.
        """
        if phase not in self.PHASES:
            raise ValueError(f"Unknown phase '{phase}', use one of {self.PHASES}.")
        self.name = name
        """Name of the fluid."""
        self.phase = phase
        """Phase of the fluid, "liquid" or "gas"."""
        self._density = density_table

    def density(
        self, pressure: float | np.ndarray, temperature: float | np.ndarray
    ) -> float | np.ndarray:
        """Density in kg/m3.

        Parameters
        ----------
        pressure : float or np.ndarray
            Pressure in bar.
        temperature : float or np.ndarray
            Temperature in degrees Celsius.

        Returns
        -------
            Density in the shape of the broadcast inputs.
        """
        return self._density(pressure, temperature)

    def __repr__(self) -> str:
        return f"Fluid(name={self.name!r}, phase={self.phase!r})"


def _water() -> Fluid:
    table = PropertyTable(
        temperatures=data.TEMPERATURES,
        values=data.WATER_DENSITY,
        name="water density",
    )
    return Fluid("water", "liquid", table)


def _co2() -> Fluid:
    table = PropertyTable(
        temperatures=data.TEMPERATURES,
        pressures=data.CO2_DENSITY_PRESSURES,
        values=(data.CO2_DENSITY_20, data.CO2_DENSITY_23),
        name="co2 density",
    )
    return Fluid("co2", "gas", table)


FLUIDS: dict[str, Fluid] = {"water": _water(), "co2": _co2()}
"""Supported fluids by name."""


def get_fluid(name: str) -> Fluid:
    """Fluid by name.

    Parameters
    ----------
    name : str
        Name of the fluid, "water" or "co2".

    Returns
    -------
        The fluid.
    """
    try:
        return FLUIDS[name]
    except KeyError:
        raise KeyError(
            f"Unknown fluid '{name}', use one of {sorted(FLUIDS)}."
        ) from None
