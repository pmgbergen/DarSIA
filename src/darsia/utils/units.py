"""Unit conversion and auto-scaling for mass/time values shown in plots.

All mass values in this codebase are computed in kilograms (density in
kg/m^3 integrated over a meter-based geometry); all time values reaching
plots are already converted to hours upstream. These are therefore the
canonical internal units that every conversion factor below is relative
to.
"""

import numpy as np

MASS_UNIT_FACTORS: dict[str, float] = {
    "g": 1e3,
    "kg": 1.0,
    "t": 1e-3,
    "kt": 1e-6,
    "Mt": 1e-9,
    "Gt": 1e-12,
}
"""Multiplier from kg to each named mass unit."""

TIME_UNIT_FACTORS: dict[str, float] = {
    "s": 3600.0,
    "min": 60.0,
    "h": 1.0,
    "day": 1.0 / 24.0,
}
"""Multiplier from hours to each named time unit."""


def convert_mass(value_kg, unit: str):
    """Convert a mass value (scalar, list, or array), given in kg, to `unit`."""
    return np.multiply(value_kg, MASS_UNIT_FACTORS[unit])


def convert_time(value_h, unit: str):
    """Convert a time value (scalar, list, or array), given in hours, to `unit`."""
    return np.multiply(value_h, TIME_UNIT_FACTORS[unit])


def mass_label(unit: str) -> str:
    """Axis label for a mass value expressed in `unit`."""
    return f"Mass ({unit})"


def time_label(unit: str) -> str:
    """Axis label for a time value expressed in `unit`."""
    return f"Time ({unit})"


def choose_unit(value: float, factors: dict[str, float], default: str) -> str:
    """Pick the unit (key of `factors`) that keeps `value` closest to 1.

    `value` is expressed in the canonical unit (kg or h) that `factors` is
    relative to. The chosen unit is whichever makes the converted
    magnitude closest to 1 in log-space, i.e. keeps the displayed number
    as close as possible to a single/double digit.
    """
    if value == 0 or not np.isfinite(value):
        return default
    value = abs(value)
    return min(factors, key=lambda unit_key: abs(np.log10(value * factors[unit_key])))


def choose_mass_unit(value_kg: float) -> str:
    """Pick the mass unit that keeps `value_kg` closest to 1 in log-space."""
    return choose_unit(value_kg, MASS_UNIT_FACTORS, default="kg")


def choose_time_unit(value_h: float) -> str:
    """Pick the time unit that keeps `value_h` closest to 1 in log-space."""
    return choose_unit(value_h, TIME_UNIT_FACTORS, default="h")
