"""Per-run measured gas-volume anchors and explicit pressure-matched estimates."""
from copy import deepcopy
import math

import numpy as np

POLICY = "own-measurement-or-pressure-matched-phase-split-v1"
TARGET_FORMAT = "darsia-overflow-phase-split-target-v1"


class TargetUnavailable(ValueError):
    """No own measurement or comparable donor establishes this frame's target."""


def _number(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError("A finite " + ("positive " if positive else "") + name + " is required")
    return float(value)


def injection_phase(stages, time):
    """Match injection/rest phase and pulse count without crossing a pulse edge."""
    time = _number(time, "elapsed time")
    if not isinstance(stages, list) or not stages:
        raise ValueError("The run's recorded injection intervals are required")
    intervals = []
    for row in stages:
        start, end = _number(row["start_h"], "injection start"), _number(row["end_h"], "injection end")
        if end <= start:
            raise ValueError("An injection interval must have positive duration")
        intervals.append((start, end))
    intervals = sorted(set(intervals))
    active = sum(start <= time < end for start, end in intervals)
    completed = sum(end <= time for start, end in intervals)
    return ("injecting" if active else "rest", completed, active)


def frozen_tables():
    """The mass analysis's frozen water-density, CO2-density and solubility tables.

    The same constants every mass analysis sets up; no image is read.
    """
    from types import SimpleNamespace
    import darsia
    tables = SimpleNamespace()
    darsia.CO2MassAnalysis.setup_20_degrees_celsius(tables)
    darsia.CO2MassAnalysis.setup_23_degrees_celsius(tables)
    return tables


def phase_split_state(*, run, elapsed_h, pressure_bar, temperature_c, injected_mass_g,
                      injection_stages, adopted_revision_sha256, recipe, math, tables):
    """One image's state for its own target and for pressure-matched donors.

    The gas density is the hydrostatic CO2 density at the image's atmospheric
    pressure and temperature at zero height, from ``tables`` (frozen_tables or
    a prepared frame's physics) with the working pixel ``math``.
    """
    thermodynamics = math.hydrostatic_fields(
        temperature=math.pixels(np.array([temperature_c])),
        height=math.pixels(np.array([0.])), top_pressure=math.pixels(np.array([pressure_bar])),
        water_density_20=tables.water_density_20, water_density_23=tables.water_density_23,
        density_20=tables.data_NIST_20, density_23=tables.data_NIST_23,
        solubility_20=tables.solubility_co2_20, solubility_23=tables.solubility_co2_23)
    return dict(run=run, elapsed_h=float(elapsed_h), pressure_bar=float(pressure_bar),
        temperature_c=float(temperature_c), gas_density_kg_m3=float(thermodynamics["density"][0]),
        injected_mass_g=injected_mass_g, stage=injection_phase(injection_stages, elapsed_h),
        compatibility=dict(adopted_revision_sha256=adopted_revision_sha256, recipe=deepcopy(recipe)))


def own_target(curve, *, run, elapsed_h, source):
    """Interpolate this run's accepted measured volume; never pool or shrink it."""
    time = _number(elapsed_h, "elapsed time")
    if not isinstance(run, str) or not run or not isinstance(source, dict) or not source:
        raise ValueError("The exact run and reviewed measurement source are required")
    times, volumes = np.asarray(curve["t"], np.float64), np.asarray(curve["vC"], np.float64)
    if (times.ndim != 1 or len(times) < 2 or volumes.shape != times.shape
            or not np.isfinite(times).all() or not np.isfinite(volumes).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("The reviewed measured volume curve must have ordered finite observations")
    if not times[0] <= time <= times[-1]:
        raise TargetUnavailable("The image lies outside this run's reviewed overflow measurement period")
    high = int(np.searchsorted(times, time, side="left"))
    low = high if times[high] == time else high - 1
    volume_ml = float(np.interp(time, times, volumes))
    if volume_ml < 0:
        raise TargetUnavailable("This run's reviewed overflow curve gives a negative gas volume at the image time")
    return dict(format=TARGET_FORMAT, policy=POLICY, run=run, elapsed_h=time,
        target_volume_m3=volume_ml * 1e-6, kind="own_measurement", estimated=False,
        source=deepcopy(source), interpolation=dict(
            lower_time_h=float(times[low]), upper_time_h=float(times[high]),
            lower_volume_ml=float(volumes[low]), upper_volume_ml=float(volumes[high])),
        pooled=False, shrunk=False)


def distribute_target(population, target_volume_m3):
    """Honor the target over the admitted CO2 pore support, not a binary gas mask.

    Stored positive gas scores retain their existing ordering. If their support
    is insufficient, the measured excess occupies the remaining admitted CO2
    pore volume uniformly; a zero model score cannot veto a measured phase split.
    """
    from .overflow_steering import GasPopulation, redistribute_gas
    if not isinstance(population, GasPopulation):
        raise TypeError("The original checked gas population is required")
    target = _number(target_volume_m3, "gas-volume target")
    if target < 0:
        raise ValueError("The gas-volume target must be nonnegative")
    capacity = population.pore_volume_m3 * population.maximum_saturation
    physical = population.eligible & (capacity > 0)
    maximum = float(np.sum(capacity[physical], dtype=np.float64))
    if target > maximum:
        raise ValueError("The measured/estimated gas volume exceeds the admitted physical CO2 pore capacity; the target was not capped")
    scored = physical & (population.probability > 0)
    scored_capacity = float(np.sum(capacity[scored], dtype=np.float64))
    result = redistribute_gas(population, target_volume_m3=min(target, scored_capacity))
    if target > scored_capacity:
        unscored = physical & ~scored
        remaining = float(np.sum(capacity[unscored], dtype=np.float64))
        saturation = result["saturation_free_gas"].copy()
        saturation[unscored] = population.maximum_saturation[unscored] * np.float32((target-scored_capacity)/remaining)
        saturation.setflags(write=False)
        result.update(saturation_free_gas=saturation, threshold=None, mode="physical_CO2_capacity_extension",
            actual_volume_m3=float(np.sum(population.pore_volume_m3*saturation, dtype=np.float64)))
    result.update(requested_volume_m3=target, target_volume_m3=target, available_volume_m3=maximum,
        capped_volume_m3=0., float32_volume_residual_m3=result["actual_volume_m3"]-target,
        extended_zero_score_pixels=int(np.count_nonzero(physical & ~scored)) if target > scored_capacity else 0)
    return result


def _donor_at(rows, recipient):
    ordered = sorted(rows, key=lambda row: row["elapsed_h"])
    time = recipient["elapsed_h"]
    exact = [row for row in ordered if row["elapsed_h"] == time]
    if exact:
        return dict(exact[0], interpolation_sources=[deepcopy(exact[0]["source"])])
    lower = [row for row in ordered if row["elapsed_h"] < time]
    upper = [row for row in ordered if row["elapsed_h"] > time]
    if not lower or not upper:
        return None
    a, b = lower[-1], upper[0]
    if a["stage"] != recipient["stage"] or b["stage"] != recipient["stage"]:
        return None
    weight = (time-a["elapsed_h"])/(b["elapsed_h"]-a["elapsed_h"])
    return dict(a, elapsed_h=time,
        pressure_bar=(1-weight)*a["pressure_bar"]+weight*b["pressure_bar"],
        gas_fraction=(1-weight)*a["gas_fraction"]+weight*b["gas_fraction"],
        interpolation_sources=[deepcopy(a["source"]), deepcopy(b["source"])])


def pressure_matched_target(recipient, donors, *, fit_runs):
    """Use nearest-pressure comparable fit donors; this is explicitly an estimate.

    Donor gas amounts are normalized by their own cumulative injected mass.
    The recipient's own P/T gas density converts the estimated amount to volume.
    The caller supplies actual recorded stage and compatibility identities.
    """
    time = _number(recipient["elapsed_h"], "elapsed time")
    pressure = _number(recipient["pressure_bar"], "recipient pressure", positive=True)
    dose = _number(recipient["injected_mass_g"], "recipient injected mass", positive=True)
    density = _number(recipient["gas_density_kg_m3"], "recipient gas density", positive=True)
    _number(recipient["temperature_c"], "recipient temperature")
    if not recipient.get("compatibility") or not recipient.get("stage"):
        raise ValueError("The recipient needs its physical/chemical context and injection phase")
    fit = set(fit_runs)
    groups = {}
    for row in donors:
        if (row.get("run") not in fit or row.get("run") == recipient["run"]
                or row.get("kind") != "own_measurement" or row.get("estimated") is not False
                or row.get("compatibility") != recipient["compatibility"]
                or row.get("stage") != recipient["stage"]):
            continue
        p = _number(row["pressure_bar"], "donor pressure", positive=True)
        d = _number(row["injected_mass_g"], "donor injected mass", positive=True)
        rho = _number(row["gas_density_kg_m3"], "donor gas density", positive=True)
        volume = _number(row["target_volume_m3"], "donor measured gas volume")
        if volume < 0:
            continue
        fraction = volume * rho * 1000 / d
        if not 0 <= fraction <= 1:
            continue
        groups.setdefault(row["run"], []).append(dict(row, pressure_bar=p, gas_fraction=fraction))
    matched = [value for rows in groups.values() if (value := _donor_at(rows, recipient)) is not None]
    if not matched:
        raise TargetUnavailable("No fit-run overflow measurement has a comparable injection phase, time coverage and physical context")
    nearest = min(abs(row["pressure_bar"]-pressure) for row in matched)
    selected = [row for row in matched if abs(row["pressure_bar"]-pressure) == nearest]
    fraction = sum(row["gas_fraction"] for row in selected)/len(selected)
    return dict(format=TARGET_FORMAT, policy=POLICY, run=recipient["run"], elapsed_h=time,
        target_volume_m3=fraction*dose/(density*1000), kind="pressure_matched_estimate", estimated=True,
        recipient_pressure_bar=pressure, recipient_temperature_c=recipient["temperature_c"],
        recipient_gas_density_kg_m3=density, recipient_injected_mass_g=dose,
        estimated_gas_fraction_of_injected_mass=fraction, pressure_difference_bar=nearest,
        donors=[dict(run=row["run"], pressure_bar=row["pressure_bar"], gas_fraction=row["gas_fraction"],
            weight=1/len(selected), source=deepcopy(row["interpolation_sources"])) for row in selected],
        pooled=False, shrunk=False)
