"""Lookup tables of material properties as function of pressure and temperature."""

from __future__ import annotations

import numpy as np


class PropertyTable:
    """Property given on a (temperature, pressure) grid, interpolated linearly.

    Inside the pressure range the property is linear in pressure; in temperature it is
    piecewise linear and extrapolated linearly beyond the first and last temperature.
    Pressure outside of the table range raises an error.

    Units: pressure in bar, temperature in degrees Celsius.
    """

    def __init__(
        self,
        temperatures: tuple[float, ...] | np.ndarray,
        values: tuple | np.ndarray,
        pressures: tuple[float, ...] | np.ndarray | None = None,
        name: str = "property",
    ) -> None:
        """Constructor.

        Parameters
        ----------
        temperatures : array-like
            Increasing temperatures (at least two) of the table in degrees Celsius.
        values : array-like
            Values of shape (num_temperatures,) for a pressure-independent property, or
            of shape (num_temperatures, num_pressures).
        pressures : array-like, optional
            Increasing pressures of the table in bar; None for a property that does
            not depend on pressure.
        name : str
            Name of the property used in error messages.
        """
        self.name = name
        """Name of the property."""
        self.temperatures = np.asarray(temperatures, dtype=float)
        """Temperatures of the table in degrees Celsius."""
        self.values = np.asarray(values, dtype=float)
        """Table values."""
        self.pressures = None if pressures is None else np.asarray(pressures, float)
        """Pressures of the table in bar, None if pressure-independent."""

        if self.temperatures.ndim != 1 or len(self.temperatures) < 2:
            raise ValueError(f"{name}: need at least two temperatures.")
        if np.any(np.diff(self.temperatures) <= 0):
            raise ValueError(f"{name}: temperatures must be increasing.")
        expected = (len(self.temperatures),)
        if self.pressures is not None:
            if np.any(np.diff(self.pressures) <= 0):
                raise ValueError(f"{name}: pressures must be increasing.")
            expected = (len(self.temperatures), len(self.pressures))
        if self.values.shape != expected:
            raise ValueError(
                f"{name}: values have shape {self.values.shape}, expected {expected}."
            )

    def __call__(
        self,
        pressure: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Evaluate the property.

        Parameters
        ----------
        pressure : float or np.ndarray
            Pressure in bar.
        temperature : float or np.ndarray
            Temperature in degrees Celsius; broadcast against the pressure.

        Returns
        -------
            Property values in the shape of the broadcast inputs (float for scalars).
        """
        pressure, temperature = np.broadcast_arrays(
            np.asarray(pressure, dtype=float), np.asarray(temperature, dtype=float)
        )

        # Values at the table temperatures, interpolated in pressure.
        if self.pressures is None:
            columns = np.stack([np.full(pressure.shape, v) for v in self.values])
        else:
            self._check_pressure_range(pressure)
            columns = np.stack(
                [np.interp(pressure, self.pressures, row) for row in self.values]
            )

        # Piecewise linear in temperature, linear extrapolation at the ends.
        lower = np.clip(
            np.searchsorted(self.temperatures, temperature, side="right") - 1,
            0,
            len(self.temperatures) - 2,
        )
        t_lower = self.temperatures[lower]
        t_upper = self.temperatures[lower + 1]
        v_lower = np.take_along_axis(columns, lower[None], axis=0)[0]
        v_upper = np.take_along_axis(columns, lower[None] + 1, axis=0)[0]
        result = v_lower * (t_upper - temperature) / (t_upper - t_lower) + (
            v_upper * (temperature - t_lower) / (t_upper - t_lower)
        )
        return float(result) if result.ndim == 0 else result

    def _check_pressure_range(self, pressure: np.ndarray) -> None:
        """Raise a clear error for pressures outside of the table."""
        low, high = float(self.pressures[0]), float(self.pressures[-1])
        if np.any(pressure < low) or np.any(pressure > high):
            raise ValueError(
                f"{self.name}: pressure outside of the table range "
                f"[{low}, {high}] bar (got min {float(np.min(pressure))}, "
                f"max {float(np.max(pressure))})."
            )
