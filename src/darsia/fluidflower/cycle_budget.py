"""Mass budget per injection cycle, series agnostic.

For every injection cycle (one injection and the waiting time until the next
one, or until the end of imaging) the budget closes

    injected = new dissolved (in the quantified sand) + free gas in the rig
               + left through the overflow + rest,

where

* ``injected`` comes from the injection protocol (MFC rate, duration and a
  gas density; the density's reference conditions are a declared input),
* ``new dissolved`` is the image product of the cycle relative to the cycle
  baseline, as a lower and an upper bound (saturated pixels, see
  ``concentration_flags``),
* ``free gas`` comes from an external scale record where one exists: the
  water displaced from the rig equals the free-gas volume, which times the gas
  density gives the mass. Runs without a scale (FF_UM, FF_RE e0) get NaN and
  the rest absorbs it,
* ``left`` is dissolved CO2 that left with the overflow water (two-way
  overflow runs only; a declared estimate),
* ``rest`` is what the images and the scale cannot see: free water, the open
  top, outside the frame, and measurement error. It is reported, never hidden.

Everything is in grams. The functions take plain numbers and tables so that
they can be used from the pipeline, from a notebook and from tests alike.
Defaults are proposals awaiting the project owner's decision.

Several injection ports (FF_UM ports A and B, FF_BM's two ports with changing
rates) are described by ``PortSchedule``: an id, an optional rig position
(x, y) in metres and ordered, non-overlapping segments of constant MFC rate
(mL/min at the MFC's reference conditions; times in minutes on one clock for
all ports of a run). The injected mass up to a time t sums ``injected_mass_g``
over the elapsed part of every segment, with the density at the MFC reference
temperature, which every call names (no default: the MFC reference
conditions are the experimenter's choice). ``cycle_budget_from_ports`` closes
a cycle on the total of all ports with ``cycle_budget`` (same rules) and
keeps the mass per port. A run without an overflow balance (FF_UM, FF_RE e0)
passes ``scale_term_available=False``: its free-gas and overflow terms are
then absent, NaN in the result and never 0.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

__all__ = [
    "GAS_DENSITY_KG_PER_M3",
    "injected_mass_g",
    "free_gas_mass_g",
    "CycleBudgetInputs",
    "CycleBudget",
    "cycle_budget",
    "budget_rows",
    "RateSegment",
    "PortSchedule",
    "injected_mass_by_port_g",
    "total_injected_mass_g",
    "PortCycleBudget",
    "cycle_budget_from_ports",
]

GAS_DENSITY_KG_PER_M3 = {
    # CO2 at 1 atm; the MFC reference conditions decide which one applies
    "0C": 1.977,
    "15C": 1.872,
    "20C": 1.842,
    "25C": 1.811,
}


def injected_mass_g(rate_ml_per_min: float, duration_min: float, gas_density_kg_per_m3: float) -> float:
    """Injected mass from the MFC setpoint: mL/min x min x kg/m3 -> g."""
    volume_ml = rate_ml_per_min * duration_min
    return volume_ml * 1e-6 * gas_density_kg_per_m3 * 1e3


def free_gas_mass_g(displaced_water_g: float, gas_density_kg_per_m3: float,
                    water_density_kg_per_m3: float = 997.0) -> float:
    """Free gas in the rig from the water it displaced onto the scale.
    NaN in gives NaN out (no scale record)."""
    displaced_water_g = _as_term(displaced_water_g)
    if math.isnan(displaced_water_g):
        return float("nan")
    if not math.isfinite(displaced_water_g) or not (0.0 < water_density_kg_per_m3 < math.inf) \
            or not (0.0 < gas_density_kg_per_m3 < math.inf):
        raise ValueError("displaced water and densities must be finite and the densities positive")
    volume_m3 = displaced_water_g * 1e-3 / water_density_kg_per_m3
    return volume_m3 * gas_density_kg_per_m3 * 1e3


def _as_term(value) -> float:
    """Input contract of the budget: a term is a float; ``None`` means unknown
    and becomes NaN here, so callers may pass either."""
    if value is None:
        return float("nan")
    return float(value)


@dataclass(frozen=True)
class CycleBudgetInputs:
    cycle_run: str
    injected_g: float
    dissolved_new_lower_g: float
    dissolved_new_upper_g: float
    free_gas_g: float = float("nan")
    left_overflow_g: float = float("nan")
    background_dissolved_g: float = float("nan")
    quality_flags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name in ("injected_g", "dissolved_new_lower_g", "dissolved_new_upper_g"):
            v = _as_term(getattr(self, name))
            if not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number")
            object.__setattr__(self, name, v)
        for name in ("free_gas_g", "left_overflow_g", "background_dissolved_g"):
            v = _as_term(getattr(self, name))
            if math.isinf(v):
                raise ValueError(f"{name} must be finite or unknown (NaN / None), not infinite")
            object.__setattr__(self, name, v)
        object.__setattr__(self, "quality_flags", tuple(self.quality_flags))


@dataclass(frozen=True)
class CycleBudget:
    cycle_run: str
    injected_g: float
    dissolved_new_lower_g: float
    dissolved_new_upper_g: float
    free_gas_g: float
    left_overflow_g: float
    background_dissolved_g: float
    rest_lower_g: float
    rest_upper_g: float
    dissolved_fraction_lower: float
    dissolved_fraction_upper: float
    free_gas_known: bool
    overflow_known: bool
    quality_flags: tuple[str, ...]

    def as_row(self) -> dict:
        row = asdict(self)
        row["quality_flags"] = " | ".join(self.quality_flags)
        return row


def _nan_to_zero(x: float) -> float:
    return 0.0 if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def cycle_budget(inputs: CycleBudgetInputs) -> CycleBudget:
    """Close the budget. Unknown terms (NaN) count as zero in the rest, and the
    flags ``free_gas_known`` / ``overflow_known`` say so."""
    if inputs.injected_g < 0:
        raise ValueError("injected mass must be non-negative")
    if inputs.dissolved_new_upper_g < inputs.dissolved_new_lower_g:
        raise ValueError("upper bound of the dissolved mass below the lower bound")
    free_gas_known = not math.isnan(inputs.free_gas_g)
    overflow_known = not math.isnan(inputs.left_overflow_g)
    gas = _nan_to_zero(inputs.free_gas_g)
    left = _nan_to_zero(inputs.left_overflow_g)
    rest_upper = inputs.injected_g - inputs.dissolved_new_lower_g - gas - left
    rest_lower = inputs.injected_g - inputs.dissolved_new_upper_g - gas - left
    inj = inputs.injected_g
    return CycleBudget(
        cycle_run=inputs.cycle_run,
        injected_g=inj,
        dissolved_new_lower_g=inputs.dissolved_new_lower_g,
        dissolved_new_upper_g=inputs.dissolved_new_upper_g,
        free_gas_g=inputs.free_gas_g,
        left_overflow_g=inputs.left_overflow_g,
        background_dissolved_g=inputs.background_dissolved_g,
        rest_lower_g=rest_lower,
        rest_upper_g=rest_upper,
        dissolved_fraction_lower=inputs.dissolved_new_lower_g / inj if inj > 0 else float("nan"),
        dissolved_fraction_upper=inputs.dissolved_new_upper_g / inj if inj > 0 else float("nan"),
        free_gas_known=free_gas_known,
        overflow_known=overflow_known,
        quality_flags=tuple(inputs.quality_flags),
    )


def budget_rows(inputs: list[CycleBudgetInputs]) -> list[dict]:
    """One table row per cycle, in the order given."""
    return [cycle_budget(x).as_row() for x in inputs]


# --- several injection ports with piecewise-constant rates (FF_UM, FF_BM) ---


def _finite(name: str, value) -> float:
    """A required number: finite; None, NaN and infinities are rejected."""
    v = _as_term(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number")
    return v


def _reference_density(reference_temperature: str) -> float:
    """CO2 density at the MFC reference temperature (a key of GAS_DENSITY_KG_PER_M3)."""
    if reference_temperature not in GAS_DENSITY_KG_PER_M3:
        raise ValueError(f"unknown MFC reference temperature {reference_temperature!r}; "
                         f"use one of {', '.join(GAS_DENSITY_KG_PER_M3)}")
    return GAS_DENSITY_KG_PER_M3[reference_temperature]


@dataclass(frozen=True)
class RateSegment:
    """Constant MFC rate from ``start_min`` to ``end_min`` (minutes on the run's
    clock), in mL/min at the MFC's reference conditions."""

    start_min: float
    end_min: float
    rate_ml_per_min: float

    def __post_init__(self) -> None:
        for name in ("start_min", "end_min", "rate_ml_per_min"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if not self.start_min < self.end_min:
            raise ValueError("a rate segment must start before it ends")
        if self.rate_ml_per_min < 0.0:
            raise ValueError("rate_ml_per_min must be non-negative")


@dataclass(frozen=True)
class PortSchedule:
    """Injection schedule of one port: an id, an optional rig position (x, y)
    in metres and its segments in time order, without overlap (a segment may
    start where the previous one ends). Segments are ``RateSegment`` objects
    or ``(start_min, end_min, rate_ml_per_min)`` tuples."""

    port_id: str
    segments: tuple[RateSegment, ...]
    position_m: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.port_id, str) or not self.port_id.strip():
            raise ValueError("port_id must be a non-empty string")
        segments = tuple(s if isinstance(s, RateSegment) else RateSegment(*s)
                         for s in self.segments)
        for previous, current in zip(segments, segments[1:]):
            if current.start_min < previous.start_min:
                raise ValueError(f"port {self.port_id}: segments must be in time order")
            if current.start_min < previous.end_min:
                raise ValueError(f"port {self.port_id}: segments overlap")
        object.__setattr__(self, "segments", segments)
        if self.position_m is not None:
            position = tuple(self.position_m)
            if len(position) != 2:
                raise ValueError("position_m must be (x, y) in metres")
            object.__setattr__(self, "position_m",
                               tuple(_finite("position_m", c) for c in position))


def _ports(ports) -> tuple[PortSchedule, ...]:
    ports = tuple(ports)
    if not ports:
        raise ValueError("at least one port schedule is required")
    if not all(isinstance(p, PortSchedule) for p in ports):
        raise TypeError("ports must be PortSchedule objects")
    ids = [p.port_id for p in ports]
    if len(set(ids)) != len(ids):
        raise ValueError(f"port ids must be unique, got {ids}")
    return ports


def _port_mass_g(port: PortSchedule, since: float, until: float, density: float) -> float:
    masses = []
    for seg in port.segments:
        elapsed_min = min(seg.end_min, until) - max(seg.start_min, since)
        if elapsed_min > 0.0:
            masses.append(injected_mass_g(seg.rate_ml_per_min, elapsed_min, density))
    return math.fsum(masses)


def injected_mass_by_port_g(ports, t_min: float, *, reference_temperature: str,
                            since_min: float | None = None) -> dict[str, float]:
    """CO2 mass (g) injected through each port up to ``t_min``, or from
    ``since_min`` to ``t_min``, keyed by port id in the order given. Every
    segment adds ``injected_mass_g`` over its elapsed part, with the density at
    the MFC ``reference_temperature`` (a key of GAS_DENSITY_KG_PER_M3, always
    given by the caller)."""
    ports = _ports(ports)
    density = _reference_density(reference_temperature)
    until = _finite("t_min", t_min)
    since = -math.inf if since_min is None else _finite("since_min", since_min)
    if since > until:
        raise ValueError("since_min must not lie after t_min")
    return {p.port_id: _port_mass_g(p, since, until, density) for p in ports}


def total_injected_mass_g(ports, t_min: float, *, reference_temperature: str,
                          since_min: float | None = None) -> float:
    """Sum over the ports of ``injected_mass_by_port_g``."""
    by_port = injected_mass_by_port_g(
        ports, t_min, reference_temperature=reference_temperature, since_min=since_min)
    return math.fsum(by_port.values())


@dataclass(frozen=True)
class PortCycleBudget:
    """One cycle fed by one or more ports. ``budget`` is ``cycle_budget`` of the
    total over the ports (unchanged rules), ``injected_by_port_g`` the share of
    every port. ``scale_term_available`` is False for a run without an overflow
    balance, whose free-gas and overflow terms are then NaN."""

    budget: CycleBudget
    injected_by_port_g: dict[str, float]
    start_min: float
    end_min: float
    reference_temperature: str
    scale_term_available: bool

    def as_row(self) -> dict:
        row = self.budget.as_row()
        row.update(start_min=self.start_min, end_min=self.end_min,
                   reference_temperature=self.reference_temperature,
                   scale_term_available=self.scale_term_available)
        for port_id, mass_g in self.injected_by_port_g.items():
            row[f"injected_{port_id}_g"] = mass_g
        return row


def cycle_budget_from_ports(cycle_run: str, ports, *, start_min: float, end_min: float,
                            reference_temperature: str, dissolved_new_lower_g: float,
                            dissolved_new_upper_g: float, scale_term_available: bool,
                            free_gas_g: float | None = None,
                            left_overflow_g: float | None = None,
                            background_dissolved_g: float | None = None,
                            quality_flags: tuple[str, ...] = ()) -> PortCycleBudget:
    """Budget of the cycle from ``start_min`` to ``end_min``, with the injected
    mass integrated from the port schedules and closed by ``cycle_budget``.

    ``scale_term_available`` (no default) says whether the run has an overflow
    balance. Without one (FF_UM, FF_RE e0) the free-gas and overflow terms are
    absent: a number for them, 0 included, is rejected; they are NaN in the
    result, never 0, and the rest absorbs them as in ``cycle_budget``. With one,
    both follow the existing rule (finite, or None/NaN for a cycle without a
    record)."""
    if not isinstance(scale_term_available, bool):
        raise TypeError("scale_term_available must be True or False")
    if not scale_term_available:
        for name, value in (("free_gas_g", free_gas_g), ("left_overflow_g", left_overflow_g)):
            if not math.isnan(_as_term(value)):
                raise ValueError(f"{name} given for a run without an overflow balance; "
                                 "leave it out: it is unknown, not 0")
    start = _finite("start_min", start_min)
    end = _finite("end_min", end_min)
    by_port = injected_mass_by_port_g(ports, end, reference_temperature=reference_temperature,
                                      since_min=start)
    budget = cycle_budget(CycleBudgetInputs(
        cycle_run,
        injected_g=math.fsum(by_port.values()),
        dissolved_new_lower_g=dissolved_new_lower_g,
        dissolved_new_upper_g=dissolved_new_upper_g,
        free_gas_g=free_gas_g,
        left_overflow_g=left_overflow_g,
        background_dissolved_g=background_dissolved_g,
        quality_flags=quality_flags,
    ))
    return PortCycleBudget(budget, by_port, start, end, reference_temperature,
                           scale_term_available)
