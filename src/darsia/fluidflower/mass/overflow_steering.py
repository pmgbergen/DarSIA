"""Overflow-governed gas on an unchanged, explicitly admitted phase grid.

These numerical functions do not admit a source, calibration or job. Their
caller must first use the original P1 completion, probability-product and
physical-geometry readers. No existing F-off evaluator calls this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GasPopulation:
    """Working float32 gas capacities with original float64 pore volumes."""

    probability: np.ndarray
    maximum_saturation: np.ndarray
    baseline_saturation: np.ndarray
    pore_volume_m3: np.ndarray
    eligible: np.ndarray
    phase_gas: np.ndarray
    certain_without_probability: np.ndarray


def gas_population(arrays, labels, eligible, pore_volume_m3, swi):
    """Retain stored probabilities; unscored/zero-probability gas becomes p=1.

    ``labels`` is the frame's phase map or its PhaseExpertKnowledge; the CO2
    and free-gas regions come from that expert knowledge.

    Positive probability on an aqueous-labelled CO2 pixel may receive gas.
    Background, excluded and physically unknown pixels cannot receive it.
    The original probability validator retains its exact dtype/support rules.
    """
    from ..phase.probability_storage import validate_arrays

    from .phase_expert_knowledge import PhaseExpertKnowledge
    expert = labels if isinstance(labels, PhaseExpertKnowledge) else PhaseExpertKnowledge(np.asarray(labels))
    labels, eligible = expert.labels, np.asarray(eligible)
    volume, swi = np.asarray(pore_volume_m3), np.asarray(swi)
    validate_arrays(arrays, labels=labels)
    shape = arrays["pixel_probability"].shape
    if (labels.dtype != np.uint8 or labels.shape != shape
            or eligible.dtype != bool or eligible.shape != shape
            or volume.dtype != np.float64 or volume.shape != shape
            or swi.dtype != np.float64 or swi.shape != shape):
        raise ValueError("Exact native labels, boolean support and float64 physical grids required")
    if (not np.isfinite(volume[eligible]).all() or np.any(volume[eligible] < 0)
            or not np.isfinite(swi[eligible]).all()
            or np.any((swi[eligible] < 0) | (swi[eligible] > 1))):
        raise ValueError("Admitted pore volumes and Swi must retain their physical bounds")
    keep = eligible & expert.co2 & (volume > 0)
    gas = keep & expert.free_gas
    scored = arrays["branch_support"] != 0
    p = np.zeros(shape, np.float32)
    p[keep & scored] = arrays["pixel_probability"][keep & scored]
    certain = gas & (~scored | (p == 0))
    p[certain] = np.float32(1)
    maximum = np.zeros(shape, np.float32)
    # The original N1 float32 gas branch casts Swi before subtracting it.
    maximum[keep] = np.float32(1) - swi[keep].astype(np.float32)
    baseline = np.where(gas, maximum, np.float32(0))
    physical = np.where(keep, volume, np.float64(0))
    for value in (p, maximum, baseline, physical, keep, gas, certain):
        value.setflags(write=False)
    return GasPopulation(p, maximum, baseline, physical, keep, gas, certain)


def _threshold(probability, capacity, target):
    """Solve sum(capacity * min(1,p/threshold)) without quantizing p."""
    order = np.argsort(probability, kind="stable")
    p, w = probability[order], capacity[order]
    prefix_w = np.cumsum(w, dtype=np.float64)
    prefix_wp = np.cumsum(w * p, dtype=np.float64)
    total = float(prefix_w[-1])
    ends = np.r_[np.flatnonzero(p[1:] != p[:-1]), len(p) - 1]
    for index in ends:
        saturated = max(0., total - float(prefix_w[index]))
        denominator = target - saturated
        if denominator <= 0:
            continue
        threshold = float(prefix_wp[index]) / denominator
        upper = float(p[index + 1]) if index + 1 < len(p) else math.inf
        if float(p[index]) <= threshold <= upper:
            return threshold
    raise ValueError("No finite probability threshold matches the admitted target")


def redistribute_gas(population, *, target_volume_m3):
    """Solve the approved probability law and report any unreachable target.

    A requested volume above the available support is capped to that support.
    The actual float32 saturation is retained; its float64 volume residual is
    reported rather than being hidden by rescaling a field or geometry.
    """
    if not isinstance(population, GasPopulation):
        raise TypeError("An explicitly checked gas population is required")
    if (isinstance(target_volume_m3, bool) or not isinstance(target_volume_m3, (int, float))
            or not math.isfinite(target_volume_m3) or target_volume_m3 < 0):
        raise ValueError("A finite nonnegative gas-volume target is required")
    p = population.probability
    capacity = population.pore_volume_m3 * population.maximum_saturation
    active = population.eligible & (p > 0) & (capacity > 0)
    maximum = float(np.sum(capacity[active], dtype=np.float64))
    requested = float(target_volume_m3)
    target = min(requested, maximum)
    saturation = np.zeros(p.shape, np.float32)
    threshold, mode = None, "zero"
    if target > 0:
        if target == maximum:
            threshold, mode = float(np.min(p[active])), "full_available_support"
            fraction = np.ones(int(active.sum()), np.float32)
        else:
            threshold, mode = _threshold(p[active].astype(np.float64), capacity[active], target), "probability"
            if not 0 < threshold <= np.finfo(np.float32).max:
                raise ValueError("The probability threshold is not representable in float32")
            threshold = float(np.float32(threshold))
            fraction = np.minimum(np.float32(1), p[active] / np.float32(threshold))
        saturation[active] = population.maximum_saturation[active] * fraction
    achieved = float(np.sum(population.pore_volume_m3 * saturation, dtype=np.float64))
    baseline = float(np.sum(population.pore_volume_m3 * population.baseline_saturation,
                            dtype=np.float64))
    saturation.setflags(write=False)
    return dict(saturation_free_gas=saturation, threshold=threshold, mode=mode,
                requested_volume_m3=requested, target_volume_m3=target,
                available_volume_m3=maximum, actual_volume_m3=achieved,
                baseline_volume_m3=baseline, capped_volume_m3=requested-target,
                float32_volume_residual_m3=achieved-target,
                certain_without_probability_pixels=int(population.certain_without_probability.sum()),
                labels_modified=False, geometry_modified=False)


def shrink_run_levels(observations, *, fit_runs, runs):
    """Precision-weighted fit-only group and method-of-moments shrinkage.

    Each observation is an already source-bound run ratio and measurement SD.
    This routine never estimates uncertainty from between-run variation, or
    includes a held-out run in the group. Cochran's Q moment equation removes
    the known measurement contribution before estimating between-run variance.
    """
    fit_runs, runs = tuple(fit_runs), tuple(runs)
    if (not runs or len(set(runs)) != len(runs) or not fit_runs
            or len(set(fit_runs)) != len(fit_runs) or not set(fit_runs) <= set(runs)
            or any(not isinstance(run, str) or not run.strip() for run in runs)
            or type(observations) is not dict or not set(observations) <= set(runs)):
        raise ValueError("Exact distinct run roles and named overflow observations required")
    for row in observations.values():
        if (type(row) is not dict or set(row) != {"ratio", "sigma"}
                or any(type(row[key]) not in (int, float) or not math.isfinite(row[key])
                       for key in row) or row["ratio"] < 0 or row["sigma"] <= 0):
            raise ValueError("Nonnegative measured ratios and positive measurement uncertainty required")
        if not math.sqrt(np.finfo(np.float64).tiny) <= row["sigma"] <= math.sqrt(np.finfo(np.float64).max):
            raise ValueError("Measurement uncertainty is not representable for shrinkage")
    pool = tuple(run for run in fit_runs if run in observations)
    if len(pool) < 2:
        raise ValueError("The group needs at least two usable fit-run measurements")
    ratio = np.array([observations[run]["ratio"] for run in pool], np.float64)
    with np.errstate(over="ignore", divide="ignore", under="ignore"):
        variance = np.square(np.array([observations[run]["sigma"] for run in pool], np.float64))
        precision = 1 / variance
    if not np.isfinite(variance).all() or np.any(variance <= 0) or not np.isfinite(precision).all():
        raise ValueError("Measurement uncertainty is not representable for shrinkage")
    sum_precision = float(precision.sum())
    if not math.isfinite(sum_precision):
        raise ValueError("Measurement precision cannot be summed finitely")
    group = float(np.dot(precision, ratio) / sum_precision)
    q = float(np.dot(precision, np.square(ratio - group)))
    denominator = sum_precision - float(np.dot(precision, precision)) / sum_precision
    if not math.isfinite(group) or not math.isfinite(q) or not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("The fit-run measurement moment is not finite and positive")
    tau_squared = max(0., (q - (len(pool) - 1)) / denominator)
    if not math.isfinite(tau_squared):
        raise ValueError("The between-run measurement variance is not finite")
    result = {}
    for run in runs:
        row = observations.get(run)
        total_variance = None if row is None else tau_squared + row["sigma"]**2
        if total_variance is not None and not math.isfinite(total_variance):
            raise ValueError("The run and between-run measurement variance cannot be summed finitely")
        weight = 0. if row is None else tau_squared / total_variance
        value = group if row is None else weight * row["ratio"] + (1 - weight) * group
        if not math.isfinite(weight) or not 0 <= weight <= 1 or not math.isfinite(value) or value < 0:
            raise ValueError("The derived run weight and gas level must be finite and physical")
        result[run] = dict(ratio=None if row is None else float(row["ratio"]),
                           sigma=None if row is None else float(row["sigma"]),
                           weight=float(weight), uncapped_level=float(value),
                           level=min(1., float(value)), capped_level=max(0., float(value)-1),
                           group_fallback=row is None,
                           ratio_above_one=row is not None and row["ratio"] > 1)
    return dict(group_ratio=group, group_level=min(1., group),
                between_run_variance=tau_squared, group_fit_runs=list(pool), runs=result)


def apply_gas_fields(original, distribution, *, population, solubility, gas_density):
    """Use the original concentration and operation order with only s_g changed.

    Freed water in an original gas pixel retains its original local-solubility
    concentration. A pixel originally labelled aqueous retains its original
    concentration. Phase labels and the gain/beta response are unchanged.
    """
    gas = distribution["saturation_free_gas"]
    sol, rho = np.asarray(solubility), np.asarray(gas_density)
    required = {"aqueous", "gas", "concentration", "mass_total", "mass_aqueous", "mass_gas"}
    if (not isinstance(population, GasPopulation)
            or not isinstance(original, dict) or not required <= original.keys()
            or gas.dtype != np.float32 or sol.dtype != np.float32 or rho.dtype != np.float32
            or sol.shape != gas.shape or rho.shape != gas.shape
            or any(not isinstance(original[key], np.ndarray)
                   or original[key].dtype != np.float32 or original[key].shape != gas.shape
                   for key in required)):
        raise ValueError("Original same-grid float32 N1 fields and thermodynamic arrays required")
    keep = population.eligible
    if (keep.shape != gas.shape or not np.isfinite(gas).all() or np.any((gas < 0) | (gas > 1))
            or np.any(gas > population.maximum_saturation)
            or not np.isfinite(sol[keep]).all() or np.any(sol[keep] < 0)
            or not np.isfinite(rho[keep]).all() or np.any(rho[keep] < 0)
            or not np.isfinite(original["aqueous"][keep]).all()):
        raise ValueError("Finite physical gas saturation and thermodynamic fields required")
    if not np.array_equal(original["gas"][keep], population.baseline_saturation[keep]):
        raise ValueError("Original gas fields differ from the admitted phase/physical population")
    # Preserve every byte outside admitted main support, including NaN payloads
    # and the original diagnostic excluded fields.
    changed = {name: original[name].copy() for name in ("gas", "mass_total", "mass_aqueous", "mass_gas")}
    changed["gas"][keep] = gas[keep]
    raw_gas = rho[keep] * gas[keep]
    raw_aqueous = sol[keep] * original["aqueous"][keep] * np.clip(np.float32(1)-gas[keep], np.float32(0), None)
    changed["mass_total"][keep] = np.asarray(raw_gas+raw_aqueous, dtype=np.float32)
    changed["mass_aqueous"][keep] = np.asarray(raw_aqueous, dtype=np.float32)
    changed["mass_gas"][keep] = np.asarray(raw_gas, dtype=np.float32)
    return dict(original, **changed)


def gas_report(population, distribution, *, gas_density):
    """Signed gas change and non-gas-labelled allocation, in mass and shares."""
    density = np.asarray(gas_density)
    if (density.dtype != np.float32 or density.shape != population.probability.shape
            or not np.isfinite(density[population.eligible]).all()
            or np.any(density[population.eligible] < 0)):
        raise ValueError("Original same-grid float32 gas density required")
    volume = population.pore_volume_m3
    original = density * population.baseline_saturation
    changed = density * distribution["saturation_free_gas"]
    keep = population.eligible
    before = float(np.sum(volume[keep] * original[keep], dtype=np.float64)) * 1000
    after = float(np.sum(volume[keep] * changed[keep], dtype=np.float64)) * 1000
    extra = keep & ~population.phase_gas
    allocated = float(np.sum(volume[extra] * changed[extra],
                             dtype=np.float64)) * 1000
    return dict(baseline_gas_mass_g=before, gas_mass_g=after,
                gas_change_g=after-before,
                gas_change_share=None if after == 0 else (after-before)/after,
                gas_on_non_gas_label_g=allocated,
                gas_on_non_gas_label_share=None if after == 0 else allocated/after,
                capped_volume_m3=distribution["capped_volume_m3"])
