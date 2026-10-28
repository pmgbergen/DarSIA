"""Reference-normalized chromatic concentration on fixed phase pore volumes.

This restores the September 14 optical equation, not its old gas policy.
DarSIA supplies geometry, gas density and aqueous solubility; the empirical
optical power curve is fitted to mass observations. No mass projection or
phase inference occurs here. Clear water is a reference, not a saturation
standard, and fitted gas occupancy is not an independent gas measurement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import minimize

from .phase_mass_joint_optics import (
    JointReference, joint_optical_signals,
)

VERSION = "joint-chromatic-od-mass-calibration-v1"
DEFAULT_GAS_BOUNDS = (.5, 1.)


@dataclass(frozen=True)
class JointOptimizationSettings:
    """Explicit broader search budget; omitted settings retain the legacy fit."""

    start_count: int = 12
    maxiter: int = 360
    maxfev: int = 20000
    xtol: float = 1e-6
    ftol: float = 1e-8

    def __post_init__(self):
        for name in ("start_count", "maxiter", "maxfev"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError("Optimization counts must be positive integers")
            object.__setattr__(self, name, int(value))
        if not 3 <= self.start_count <= 64:
            raise ValueError("Optimization start_count must be between 3 and 64")
        if (isinstance(self.xtol, bool) or isinstance(self.ftol, bool)
                or not np.isfinite([self.xtol, self.ftol]).all()
                or not 0 < self.xtol < 1 or not 0 < self.ftol < 1):
            raise ValueError("Optimization tolerances must be finite and between zero and one")
        object.__setattr__(self, "xtol", float(self.xtol))
        object.__setattr__(self, "ftol", float(self.ftol))


@dataclass(frozen=True)
class JointConcentrationParameters:
    endpoint: float
    amplitude: float = 1.
    gamma: float = 1.
    gas_saturation: float = .5
    beta: float = 0.

    def __post_init__(self):
        if (not np.isfinite(list(asdict(self).values())).all()
                or self.endpoint <= 0 or not 0 < self.amplitude <= 1
                or self.gamma <= 0 or not 0 <= self.gas_saturation <= 1
                or self.beta < 0):
            raise ValueError("Invalid joint concentration parameters")


@dataclass(frozen=True)
class DilutionAnchor:
    elapsed_h: float
    affected_pore_volume_m3: float

    def __post_init__(self):
        if (not np.isfinite([self.elapsed_h, self.affected_pore_volume_m3]).all()
                or self.affected_pore_volume_m3 <= 0):
            raise ValueError("Dilution anchor requires positive observed pore volume")


def _parameters(parameters):
    return (parameters if isinstance(parameters, JointConcentrationParameters)
            else JointConcentrationParameters(**parameters))


def _anchor(anchor):
    return anchor if isinstance(anchor, DilutionAnchor) else DilutionAnchor(
        elapsed_h=float(anchor["elapsed_h"]),
        affected_pore_volume_m3=float(anchor["affected_pore_volume_m3"]))


def dilution_factor(elapsed_h, affected_volume, anchor, beta):
    """Volume dilution after observed injection end, never future maximum area."""
    anchor = _anchor(anchor)
    if not np.isfinite([elapsed_h, affected_volume, beta]).all() or min(affected_volume, beta) < 0:
        raise ValueError("Invalid dilution state")
    if elapsed_h <= anchor.elapsed_h:
        return 1.
    return float(min(1., max(affected_volume / anchor.affected_pore_volume_m3, 1.) ** -beta))


def concentration_field(signal, parameters, factor, solubility):
    """Historical power response in g/L; computation and output are float32."""
    p = _parameters(parameters)
    signal = np.asarray(signal, np.float32)
    solubility = np.asarray(solubility, np.float32)
    if (not np.isfinite(signal).all() or not np.isfinite(solubility).all()
            or np.any(solubility < 0) or not np.isfinite(factor) or not 0 <= factor <= 1):
        raise ValueError("Invalid concentration signal or solubility")
    return (p.amplitude * np.clip(signal / p.endpoint, 0, 1) ** p.gamma
            * factor * solubility).astype(np.float32)


def _physical_fields(signal, av, gv, solubility, density):
    signal = np.asarray(signal, np.float32)
    arrays = [np.asarray(value, np.float32) for value in (av, gv, solubility, density)]
    if signal.ndim != 2 or not np.isfinite(signal).all():
        raise ValueError("Joint signal must be a finite two-dimensional field")
    if any(a.shape != signal.shape or not np.isfinite(a).all() or np.any(a < 0) for a in arrays):
        raise ValueError("Physical fields must be finite, nonnegative and share the signal grid")
    return (signal, *arrays)


def evaluate_signal_fields(signal, av, gv, solubility, density, parameters, *, elapsed_h, anchor):
    """Materialize concentrations and phase-split masses on immutable support."""
    p = _parameters(parameters)
    signal, av, gv, solubility, density = _physical_fields(signal, av, gv, solubility, density)
    affected = float(av.sum(dtype=np.float64) + gv.sum(dtype=np.float64))
    factor = dilution_factor(elapsed_h, affected, anchor, p.beta)
    concentration = concentration_field(signal, p, factor, solubility)
    residual = float(np.sum(gv.astype(float) * solubility) * 1000 * (1 - p.gas_saturation))
    aqueous = float(np.sum(concentration.astype(float) * av) * 1000) + residual
    gas = float(np.sum(gv.astype(float) * density) * 1000 * p.gas_saturation)
    return dict(concentration_g_l=concentration), dict(
        elapsed_h=float(elapsed_h), affected_pore_volume_m3=affected,
        dilution_factor=factor, aqueous_mass_g=aqueous, gas_mass_g=gas,
        total_mass_g=aqueous + gas, residual_liquid_in_gas_g=residual,
        gas_saturation=p.gas_saturation)


def evaluate_joint_fields(rgb, reference: JointReference, av, gv, solubility, density,
                          parameters, *, elapsed_h, anchor):
    """Convert registered RGB response into aqueous and gas mass on fixed pore volumes.

    Volumes are in m3, solubility and gas density in g/L, and elapsed time in
    hours. The optical reference fixes the signal scale; explicit concentration
    parameters and an observed injection-end anchor control the mass law.
    """
    if not reference.usable:
        raise ValueError("Joint optical reference has no reliable positive span")
    signal = joint_optical_signals(rgb, reference)["joint_chromatic_od"]
    arrays, record = evaluate_signal_fields(signal, av, gv, solubility, density,
        parameters, elapsed_h=elapsed_h, anchor=anchor)
    arrays["joint_signal"] = signal
    return arrays, record


def summarize_frame(signal, av, gv, solubility, density, *, run, elapsed_h, injected_g,
                    bins, max_signal, **metadata):
    """Compact capacity-weighted input; exact zero is separate from positive bins.

    The final positive bin contains the tail above ``max_signal``; its capacity
    is reported, so a clipped endpoint cannot masquerade as an observed one.
    """
    signal, av, gv, solubility, density = _physical_fields(signal, av, gv, solubility, density)
    if bins < 16 or not np.isfinite(max_signal) or max_signal <= 0:
        raise ValueError("Invalid histogram grid")
    if (not run or not np.isfinite([elapsed_h, injected_g]).all() or injected_g < 0):
        raise ValueError("Run, time and nonnegative injection observation are required")
    capacity = av.astype(float) * solubility * 1000
    positive = signal > 0
    edges = np.linspace(0, max_signal, bins + 1)
    values = np.r_[0., .5 * (edges[:-1] + edges[1:])]
    weights = np.r_[capacity[~positive].sum(), np.histogram(
        np.minimum(signal[positive], np.nextafter(max_signal, 0.)),
        bins=edges, weights=capacity[positive])[0]]
    return dict(metadata, format=VERSION, run=str(run), elapsed_h=float(elapsed_h),
        nominal_injected_mass_g=float(injected_g), signal_values=values.tolist(),
        aqueous_capacity_g=weights.tolist(),
        gas_aqueous_capacity_g=float(np.sum(gv.astype(float) * solubility) * 1000),
        gas_capacity_g=float(np.sum(gv.astype(float) * density) * 1000),
        affected_pore_volume_m3=float(av.sum(dtype=np.float64) + gv.sum(dtype=np.float64)),
        signal_quantization_half_width=float(max_signal / bins / 2),
        signal_tail_capacity_g=float(capacity[signal >= max_signal].sum()),
        signal_maximum=float(signal.max(initial=0)), quantification_complete=True,
        unknown_mass_upper_g=0.)


def _validate_summary(summary):
    values = np.asarray(summary["signal_values"], float)
    weights = np.asarray(summary["aqueous_capacity_g"], float)
    keys = ("gas_aqueous_capacity_g", "gas_capacity_g", "affected_pore_volume_m3",
            "nominal_injected_mass_g")
    if (summary.get("format") != VERSION or values.ndim != 1 or values.size < 2
            or weights.shape != values.shape or not np.isfinite(values).all()
            or not np.isfinite(weights).all() or np.any(values < 0) or np.any(weights < 0)
            or np.any(np.diff(values) <= 0) or values[0] != 0
            or not np.isfinite([summary[key] for key in (*keys, "elapsed_h")]).all()
            or any(summary[key] < 0 for key in keys)):
        raise ValueError("Invalid joint concentration histogram")
    return values, weights


def endpoint_from_summaries(summaries, indices):
    """Choose the capacity-weighted positive optical endpoint from training rows.

    The dimensionless endpoint excludes the exact-zero signal bin and uses
    only the supplied observation indices, preserving stable quantile order.
    """
    selected = [summaries[int(i)] for i in indices]
    if not selected:
        raise ValueError("Endpoint requires training observations")
    nodes, weights = [], []
    for summary in selected:
        x, w = _validate_summary(summary)
        nodes.append(x[1:]); weights.append(w[1:])
    x, w = np.concatenate(nodes), np.concatenate(weights)
    if w.sum() <= 0:
        raise ValueError("No positive training aqueous signal")
    order = np.argsort(x, kind="stable")
    index = min(len(x) - 1, int(np.searchsorted(np.cumsum(w[order]), .99 * w.sum())))
    endpoint = max(.02, float(x[order[index]]))
    return endpoint


def anchor_from_summaries(summaries):
    """Locate the first observed end of injection on a strictly ordered timeline.

    Dose is measured in grams, time in hours, and affected pore volume in m3.
    A declared final dose must be consistent and reached by an observation.
    """
    if not summaries or any(b["elapsed_h"] <= a["elapsed_h"] for a, b in zip(summaries, summaries[1:])):
        raise ValueError("Injection timeline must be strictly ordered")
    declared = {float(row["expected_final_injected_mass_g"]) for row in summaries
                if row.get("expected_final_injected_mass_g") is not None}
    if len(declared) > 1 or (declared and (not np.isfinite(list(declared)).all() or min(declared) <= 0)):
        raise ValueError("Expected final injection dose must be consistent and positive")
    maximum = (next(iter(declared)) if declared else
               max(row["nominal_injected_mass_g"] for row in summaries))
    if maximum <= 0:
        raise ValueError("Dilution requires a positive final injection dose")
    completed = [i for i, row in enumerate(summaries) if row["nominal_injected_mass_g"] >= .9999 * maximum]
    if not completed:
        raise ValueError("No observation reaches the declared final injection dose")
    index = completed[0]
    row = summaries[index]
    return dict(frame_index=index, **asdict(DilutionAnchor(
        row["elapsed_h"], row["affected_pore_volume_m3"])))


def evaluate_joint_summary(summary, parameters, anchor):
    """Evaluate phase masses and histogram quantization bounds in grams.

    Explicit optical parameters map capacity-weighted signal bins to aqueous
    concentration; the anchor supplies observed injection-end time and volume.
    Gas occupancy also accounts for the residual aqueous liquid in gas pores.
    """
    values, weights = _validate_summary(summary)
    p = _parameters(parameters)
    factor = dilution_factor(summary["elapsed_h"], summary["affected_pore_volume_m3"], anchor, p.beta)
    fraction = p.amplitude * np.clip(values / p.endpoint, 0, 1) ** p.gamma
    residual = (1 - p.gas_saturation) * summary["gas_aqueous_capacity_g"]
    aqueous = float(weights @ fraction * factor + residual)
    gas = float(p.gas_saturation * summary["gas_capacity_g"])
    # Monotonic interval around every histogram bin gives an explicit numerical
    # error bound, including the low-signal power curve when gamma < 1.
    half = float(summary.get("signal_quantization_half_width", 0.))
    lo, hi = np.maximum(values - half, 0), values + half
    lo[0] = hi[0] = 0
    hi[-1] = max(hi[-1], float(summary.get("signal_maximum", hi[-1])))
    lower = float(weights @ (p.amplitude * np.clip(lo / p.endpoint, 0, 1) ** p.gamma) * factor + residual + gas)
    upper = float(weights @ (p.amplitude * np.clip(hi / p.endpoint, 0, 1) ** p.gamma) * factor + residual + gas)
    return dict(run=summary["run"], elapsed_h=summary["elapsed_h"],
        nominal_injected_mass_g=summary["nominal_injected_mass_g"],
        aqueous_mass_g=aqueous, gas_mass_g=gas, total_mass_g=aqueous + gas,
        total_mass_lower_g=aqueous + gas, total_mass_upper_g=aqueous + gas,
        numerical_lower_g=lower, numerical_upper_g=upper,
        summary_mass_error_bound_g=max(aqueous + gas - lower, upper - aqueous - gas),
        residual_liquid_in_gas_g=float(residual), dilution_factor=factor,
        quantification_complete=True, unknown_mass_upper_g=0.)


def predict_joint_summaries(summaries_by_run, parameters_by_run, anchors_by_run):
    """Evaluate ordered frame summaries using each run's parameters and anchor.

    Each returned row reports aqueous, gas, and total mass in grams together
    with the numerical bounds from the unchanged histogram representation.
    """
    return {run: [evaluate_joint_summary(row, parameters_by_run[run], anchors_by_run[run])
                  for row in rows] for run, rows in summaries_by_run.items()}


def joint_mass_metrics(predictions, indices=None):
    """Measure mass error against positive injection observations.

    RMSE uses grams; normalized errors use the observed dose with the existing
    final-dose floor. Optional indices select observations independently per run.
    """
    by_run, all_errors, all_normalized = {}, [], []
    for run, rows in predictions.items():
        chosen = list(range(len(rows))) if indices is None else list(indices.get(run, ()))
        chosen = [i for i in chosen if rows[i]["nominal_injected_mass_g"] > 0]
        if not chosen:
            continue
        final = max(row["nominal_injected_mass_g"] for row in rows)
        errors = np.array([rows[i]["total_mass_g"] - rows[i]["nominal_injected_mass_g"] for i in chosen])
        normalized = errors / np.array([max(rows[i]["nominal_injected_mass_g"], .05 * final) for i in chosen])
        by_run[run] = dict(observations=len(chosen), rmse_g=float(np.sqrt(np.mean(errors ** 2))),
            nmae=float(np.mean(abs(normalized))), nrmse=float(np.sqrt(np.mean(normalized ** 2))))
        all_errors.extend(errors); all_normalized.extend(normalized)
    if not all_errors:
        return dict(observations=0, by_run=by_run)
    return dict(observations=len(all_errors), rmse_g=float(np.sqrt(np.mean(np.square(all_errors)))),
        nmae=float(np.mean(np.abs(all_normalized))), nrmse=float(np.sqrt(np.mean(np.square(all_normalized)))),
        mean_run_nmae=float(np.mean([r["nmae"] for r in by_run.values()])), by_run=by_run)


def _broader_starts(legacy_starts, bounds, count):
    """Keep known starts, then cover the bounded box with an unscrambled Halton sequence."""
    def radical_inverse(index, base):
        value, scale = 0., 1. / base
        while index:
            index, digit = divmod(index, base)
            value += digit * scale
            scale /= base
        return value

    starts = [tuple(start) for start in legacy_starts]
    for index in range(1, count - len(starts) + 1):
        unit = [radical_inverse(index, base) for base in (2, 3, 5, 7)]
        starts.append(tuple(lo + value * (hi - lo) for value, (lo, hi) in zip(unit, bounds)))
    return starts


def _compile_broader_objective(compiled, summaries_by_run, endpoints):
    """Pool matching training grids and discard exactly zero-capacity columns.

    No histogram is rebuilt, and no held-out observations enter this objective.
    Run-level losses retain equal weights even with unequal observation counts.
    """
    grouped = {}
    original_columns = 0
    for run, (_normalized, weights, residual, gas, log_ratio, target, denominator) in compiled.items():
        nodes = np.clip(np.asarray(summaries_by_run[run][0]["signal_values"], float) / endpoints[run], 0, 1)
        key = (nodes.shape, nodes.tobytes())
        group = grouped.setdefault(key, dict(nodes=nodes, rows=[]))
        group["rows"].append((weights, residual, gas, log_ratio, target, denominator))
        original_columns += weights.shape[1]
    groups, retained_columns = [], 0
    for group in grouped.values():
        rows = group["rows"]
        lengths = [row[0].shape[0] for row in rows]
        offsets = np.r_[0, np.cumsum(lengths)]
        weights, residual, gas, log_ratio, target, denominator = [
            np.concatenate([row[index] for row in rows], axis=0) for index in range(6)]
        active = np.any(weights != 0, axis=0)
        groups.append((group["nodes"][active], np.ascontiguousarray(weights[:, active]),
                       residual, gas, log_ratio, target, denominator, offsets))
        retained_columns += int(active.sum())

    def objective(x):
        amplitude, gamma, sg, beta = x
        losses = []
        for nodes, weights, residual, gas, log_ratio, target, denominator, offsets in groups:
            mass = (weights @ (amplitude * nodes ** gamma) * np.exp(-beta * log_ratio)
                    + (1 - sg) * residual + sg * gas)
            error = (mass - target) / denominator
            for begin, end in zip(offsets[:-1], offsets[1:]):
                local = error[begin:end]
                losses.append(np.mean(abs(local)) + .25 * np.sqrt(np.mean(local ** 2)))
        return float(np.mean(losses))

    return objective, dict(training_runs=len(compiled), response_grid_groups=len(groups),
        original_histogram_columns=original_columns, retained_response_columns=retained_columns,
        histograms_rebuilt=False, zero_capacity_columns_removed=True)


def _fit_broader_objective(objective, legacy_starts, bounds, settings):
    starts = _broader_starts(legacy_starts, bounds, settings.start_count)
    options = {name: getattr(settings, name) for name in ("xtol", "ftol", "maxiter", "maxfev")}
    names = ("amplitude", "gamma", "gas_saturation", "beta")
    records, fits = [], []
    for index, start in enumerate(starts):
        initial_objective = objective(start)
        fit = minimize(objective, start, method="Powell", bounds=bounds, options=options)
        x = np.asarray(fit.x, float)
        feasible = (x.shape == (4,) and np.isfinite(x).all()
                    and all(lo <= value <= hi for value, (lo, hi) in zip(x, bounds)))
        value = objective(x) if feasible else float("nan")
        valid = bool(feasible and np.isfinite(value))
        records.append(dict(start_index=index, initial_parameters=dict(zip(names, map(float, start))),
            initial_objective=float(initial_objective),
            parameters=dict(zip(names, map(float, x))) if feasible else None,
            objective=float(value) if valid else None,
            reported_objective=float(fit.fun) if np.isfinite(fit.fun) else None,
            success=bool(fit.success) and valid, feasible_finite=bool(valid),
            status=int(fit.status), message=str(fit.message), evaluations=int(fit.nfev),
            iterations=int(fit.nit)))
        fits.append(fit)
    eligible = [row for row in records if row["feasible_finite"]]
    if not eligible:
        raise RuntimeError("No optimizer start produced a finite solution within the physical bounds")
    selected = min(eligible, key=lambda row: row["objective"])
    best = fits[selected["start_index"]]
    # Re-evaluate the selected objective on the compiled training data rather
    # than relying on a solver's potentially incomplete status payload.
    best.fun = selected["objective"]
    agreement = max(1e-6, abs(best.fun) * 1e-3)
    near = [row for row in eligible if row["objective"] - best.fun <= agreement]
    converged = [row for row in eligible if row["success"]]
    normalized_spans = {name: (float((max(row["parameters"][name] for row in near)
        - min(row["parameters"][name] for row in near)) / (hi - lo)) if hi > lo else 0.)
        for name, (lo, hi) in zip(names, bounds)}
    bound_diagnostics = {name: dict(value=float(value), lower=float(lo), upper=float(hi),
        lower_distance_fraction=float((value - lo) / (hi - lo)) if hi > lo else 0.,
        upper_distance_fraction=float((hi - value) / (hi - lo)) if hi > lo else 0.,
        active=bool(min(value - lo, hi - value) <= 1e-3 * (hi - lo)))
        for name, value, (lo, hi) in zip(names, best.x, bounds)}
    diagnostics = dict(settings=asdict(settings), start_design="three legacy starts plus unscrambled Halton bases 2,3,5,7",
        starts=records, selected_start_index=selected["start_index"],
        feasible_finite_starts=len(eligible), converged_starts=len(converged),
        selected_converged=bool(selected["success"]),
        best_converged_objective=min(row["objective"] for row in converged) if converged else None,
        objective_spread=float(max(row["objective"] for row in eligible) - best.fun),
        near_best_objective_tolerance=agreement, near_best_starts=len(near),
        near_best_converged_starts=sum(row["success"] for row in near),
        near_best_parameter_span_fraction=normalized_spans,
        bounds=bound_diagnostics,
        solver_evaluation_budget=settings.start_count * settings.maxfev,
        diagnostic_evaluations=settings.start_count + len(eligible),
        convergence_is_global_optimality_proof=False)
    warnings = []
    if not selected["success"]:
        warnings.append("Best optimizer solution did not converge; treat this calibration as provisional")
    if sum(row["success"] for row in near) < 2:
        warnings.append("Best objective lacks agreement from two converged starts; global search agreement is unresolved")
    uncertain = [name for name, span in normalized_spans.items() if span > .1]
    if uncertain:
        warnings.append("Near-equal objective values have different parameter estimates: " + ", ".join(uncertain))
    return best, fits, diagnostics, warnings


def fit_joint_summaries(summaries_by_run, *, scope="individual", gas_bounds=DEFAULT_GAS_BOUNDS,
                        training_indices_by_run=None, optimization_settings=None,
                        overflow_mode="off", overflow_references=(), overflow_weight):
    """Fit common response parameters with equal experiment objective weights.

    Each run retains its reference images and observed injection-end anchor.
    Shared fitting uses one pooled training endpoint with equal positive
    capacity weight for every experiment, and one common parameter vector.
    Pass ``JointOptimizationSettings`` or its field mapping to request broader
    deterministic search and detailed convergence diagnostics. ``None`` keeps
    the previous three-start algorithm and numerical objective unchanged.
    """
    if (scope not in ("individual", "shared") or not summaries_by_run
            or (scope == "individual" and len(summaries_by_run) != 1)
            or (scope == "shared" and len(summaries_by_run) < 2)):
        raise ValueError("Fit scope must match the number of experiments")
    if len(gas_bounds) != 2 or not 0 <= gas_bounds[0] < gas_bounds[1] <= 1:
        raise ValueError("Explicit increasing gas saturation bounds in [0,1] required")
    if overflow_mode not in ("off", "compare", "constrain", "hard"):
        raise ValueError("Unknown overflow mode")
    if not np.isfinite(overflow_weight) or not 0 < overflow_weight <= 10:
        raise ValueError("Overflow weight must be finite and in (0, 10]")
    references = list(overflow_references)
    if overflow_mode == "off" and references:
        raise ValueError("Overflow references require compare or constrain mode")
    if overflow_mode != "off" and not references:
        raise ValueError("Overflow mode requires time-matched gas references")
    reference_keys = set()
    for ref in references:
        run, index = ref.get("run"), ref.get("frame_index")
        if (run not in summaries_by_run or type(index) is not int
                or not 0 <= index < len(summaries_by_run[run])
                or (run, index) in reference_keys
                or not np.isfinite([ref.get("mass_g", np.nan), ref.get("sigma_g", np.nan),
                                    ref.get("time_h", np.nan)]).all()
                or ref["mass_g"] < 0 or ref["sigma_g"] <= 0
                or abs(float(ref["time_h"]) - summaries_by_run[run][index]["elapsed_h"])
                    > ref.get("match_tolerance_h", .03)
                or ref.get("mode") not in ("two-sided-interval", "gas-upper-bound")
                or not ref.get("source")):
            raise ValueError("Gas reference must be unique, sourced, uncertain and time-matched")
        if overflow_mode in ("constrain", "hard") and ref["source"] != "measured-run-specific":
            raise ValueError("Only measured run-specific overflow may constrain the joint fit")
        reference_keys.add((run, index))
    if optimization_settings is not None and not isinstance(optimization_settings, JointOptimizationSettings):
        if not isinstance(optimization_settings, Mapping):
            raise ValueError("optimization_settings must be a field mapping or JointOptimizationSettings")
        try:
            optimization_settings = JointOptimizationSettings(**optimization_settings)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid optimization_settings: " + str(error)) from error
    selected, endpoints, anchors, compiled = {}, {}, {}, {}
    for run, rows in summaries_by_run.items():
        for row in rows:
            _validate_summary(row)
            if row["run"] != run:
                raise ValueError("Summary identity differs from experiment")
        chosen = list(range(len(rows))) if training_indices_by_run is None else list(training_indices_by_run.get(run, ()))
        if (len(set(chosen)) != len(chosen)
                or any(not isinstance(i, (int, np.integer)) or i < 0 or i >= len(rows) for i in chosen)):
            raise ValueError("Training indices must be unique and in range")
        chosen = [i for i in chosen if rows[i]["nominal_injected_mass_g"] > 0]
        if len(chosen) < 4:
            raise ValueError(f"{run} needs at least four positive training observations")
        selected[run] = chosen
        endpoints[run] = endpoint_from_summaries(rows, chosen)
        anchors[run] = anchor_from_summaries(rows)
        nodes = np.asarray(rows[0]["signal_values"], float)
        if any(not np.array_equal(row["signal_values"], nodes) for row in rows):
            raise ValueError("One experiment must use one histogram grid")
        observations = [rows[i] for i in chosen]
        times = np.array([row["elapsed_h"] for row in observations])
        ratio = np.maximum(np.array([row["affected_pore_volume_m3"] for row in observations])
                           / anchors[run]["affected_pore_volume_m3"], 1.)
        log_ratio = np.where(times > anchors[run]["elapsed_h"], np.log(ratio), 0.)
        target = np.array([row["nominal_injected_mass_g"] for row in observations])
        compiled[run] = (nodes / endpoints[run],
            np.array([row["aqueous_capacity_g"] for row in observations]),
            np.array([row["gas_aqueous_capacity_g"] for row in observations]),
            np.array([row["gas_capacity_g"] for row in observations]),
            log_ratio, target, np.maximum(target, .05 * max(row["nominal_injected_mass_g"] for row in rows)))

    if scope == "shared":
        pooled_nodes, pooled_weights = [], []
        for run, rows in summaries_by_run.items():
            nodes = np.asarray(rows[0]["signal_values"], float)[1:]
            weights = np.sum([np.asarray(rows[i]["aqueous_capacity_g"], float)[1:]
                              for i in selected[run]], axis=0)
            pooled_nodes.append(nodes)
            pooled_weights.append(weights / weights.sum())
        nodes, weights = np.concatenate(pooled_nodes), np.concatenate(pooled_weights)
        order = np.argsort(nodes, kind="stable")
        index = min(len(nodes) - 1, int(np.searchsorted(np.cumsum(weights[order]), .99 * weights.sum())))
        endpoint = max(.02, float(nodes[order[index]]))
        for run in summaries_by_run:
            normalized, *rest = compiled[run]
            compiled[run] = (normalized * endpoints[run] / endpoint, *rest)
            endpoints[run] = endpoint

    # A fold may use only its training observations. In particular, the gas
    # targets of a held-out run or time must never enter parameter fitting.
    training_references = {run: [ref for ref in references if ref["run"] == run
                                      and ref["frame_index"] in selected[run]]
                           for run in summaries_by_run}

    # A hard overflow observation constrains the one physically meaningful
    # shared gas-saturation parameter. Only training-time, measured targets
    # participate; a held-out measurement is used solely for comparison.
    hard_constraints = []
    hard_gas_bounds = tuple(map(float, gas_bounds))
    if overflow_mode == "hard":
        lower_sg, upper_sg = hard_gas_bounds
        for run, refs in training_references.items():
            for ref in refs:
                capacity = float(summaries_by_run[run][ref["frame_index"]]["gas_capacity_g"])
                lower_mass = (0. if ref["mode"] == "gas-upper-bound" else
                              max(0., float(ref["mass_g"]) - float(ref["sigma_g"])))
                upper_mass = float(ref["mass_g"]) + float(ref["sigma_g"])
                if capacity <= 0.:
                    if lower_mass > 0.:
                        raise ValueError(
                            f"Hard overflow infeasible for {run} at {ref['time_h']:.3f} h: "
                            f"measured gas lower bound {lower_mass:.6g} g exceeds zero "
                            "visible gas capacity; this fit cannot fabricate hidden gas.")
                    saturation_interval = (0., 1.)
                else:
                    saturation_interval = (lower_mass / capacity, upper_mass / capacity)
                hard_constraints.append(dict(run=run, frame_index=ref["frame_index"],
                    time_h=float(ref["time_h"]), gas_capacity_g=capacity,
                    mass_interval_g=[lower_mass, upper_mass],
                    saturation_interval=list(saturation_interval), mode=ref["mode"]))
                lower_sg = max(lower_sg, saturation_interval[0])
                upper_sg = min(upper_sg, saturation_interval[1])
        if lower_sg > upper_sg:
            conflicts = ", ".join(
                f"{row['run']} {row['time_h']:.3f} h "
                f"[{row['saturation_interval'][0]:.3f}, {row['saturation_interval'][1]:.3f}]"
                for row in hard_constraints)
            raise ValueError(
                f"Hard overflow infeasible: required gas-saturation lower bound "
                f"{lower_sg:.6g} exceeds upper bound {upper_sg:.6g} "
                f"within physical bounds {list(hard_gas_bounds)}. Targets: {conflicts}. "
                "The fixed visible-gas phase support cannot satisfy these measurements; "
                "no hidden gas is fabricated.")
        hard_gas_bounds = (lower_sg, upper_sg)

    def overflow_penalty(saturation):
        if overflow_mode != "constrain":
            return 0.
        losses = []
        for run, refs in training_references.items():
            if not refs:
                continue
            rows = summaries_by_run[run]
            final_dose = max(row["nominal_injected_mass_g"] for row in rows)
            deviations = []
            for ref in refs:
                estimate = saturation * rows[ref["frame_index"]]["gas_capacity_g"]
                lower = 0. if ref["mode"] == "gas-upper-bound" else ref["mass_g"] - ref["sigma_g"]
                upper = ref["mass_g"] + ref["sigma_g"]
                outside = max(lower - estimate, estimate - upper, 0.)
                deviations.append(outside / max(ref["sigma_g"], .05 * final_dose))
            losses.append(float(np.mean(deviations)))
        return float(overflow_weight * np.mean(losses)) if losses else 0.

    def objective(x):
        amplitude, gamma, sg, beta = x
        losses = []
        for normalized, weights, residual_capacity, gas_capacity, log_ratio, target, denominator in compiled.values():
            response = amplitude * np.clip(normalized, 0, 1) ** gamma
            mass = weights @ response * np.exp(-beta * log_ratio) + (1 - sg) * residual_capacity + sg * gas_capacity
            error = (mass - target) / denominator
            losses.append(np.mean(abs(error)) + .25 * np.sqrt(np.mean(error ** 2)))
        return float(np.mean(losses) + overflow_penalty(sg))

    bounds = [(.05, 1.), (.4, 3.), hard_gas_bounds, (0., 2.)]
    gas_mid = .5 * sum(hard_gas_bounds)
    starts = [(.7, 1., gas_mid, .7), (.9, 2., gas_mid, 1.),
              (.4, .6, hard_gas_bounds[0], .3)]
    search_diagnostics, search_warnings = None, []
    if optimization_settings is None:
        fits = [minimize(objective, start, method="Powell", bounds=bounds,
            options=dict(xtol=1e-6, ftol=1e-7, maxiter=180, maxfev=6500)) for start in starts]
        best = min(fits, key=lambda fit: float(fit.fun))
    else:
        efficient_base, efficiency = _compile_broader_objective(compiled, summaries_by_run, endpoints)
        efficient_objective = lambda x: efficient_base(x) + overflow_penalty(x[2])
        best, fits, search_diagnostics, search_warnings = _fit_broader_objective(
            efficient_objective, starts, bounds, optimization_settings)
        search_diagnostics["histogram_efficiency"] = efficiency
    shared = dict(zip(("amplitude", "gamma", "gas_saturation", "beta"), map(float, best.x)))
    if overflow_mode == "hard":
        saturation = shared["gas_saturation"]
        if not hard_gas_bounds[0] - 1e-10 <= saturation <= hard_gas_bounds[1] + 1e-10:
            raise RuntimeError("Hard overflow optimizer left the feasible gas-saturation interval")
        for constraint in hard_constraints:
            estimate = saturation * constraint["gas_capacity_g"]
            lo, hi = constraint["mass_interval_g"]
            if estimate < lo - 1e-10 or estimate > hi + 1e-10:
                raise RuntimeError(
                    f"Hard overflow result violates {constraint['run']} "
                    f"at {constraint['time_h']:.3f} h")
    parameters = {run: dict(endpoint=endpoints[run], **shared) for run in summaries_by_run}
    predictions = predict_joint_summaries(summaries_by_run, parameters, anchors)
    comparisons = [dict(ref, predicted_gas_mass_g=predictions[ref["run"]][ref["frame_index"]]["gas_mass_g"],
                        used_for_fit=bool(overflow_mode in ("constrain", "hard") and
                                          ref["frame_index"] in selected[ref["run"]]))
                   for ref in references]
    warnings = ["Mass consistency does not independently validate local concentrations or the free-gas split", *search_warnings]
    active = [name for name, value, (lo, hi) in zip(shared, best.x, bounds)
              if min(value - lo, hi - value) <= 1e-3 * (hi - lo)]
    if active:
        warnings.append("Parameters close to bounds: " + ", ".join(active))
    if any(row.get("signal_tail_capacity_g", 0) > 0 for rows in summaries_by_run.values() for row in rows):
        warnings.append("Positive optical histogram tail is clipped; inspect endpoint and numerical error bounds")
    return dict(format=VERSION, scope=scope, shared_parameters=shared,
        parameters_by_run=parameters, endpoints_by_run=endpoints, anchors_by_run=anchors,
        predictions=predictions, training_indices_by_run=selected,
        metrics=joint_mass_metrics(predictions, selected), warnings=warnings,
        overflow_mode=overflow_mode, overflow_comparisons=comparisons,
        diagnostics=dict(objective=float(best.fun), optimizer="deterministic bounded multistart Powell",
            optimizer_success=bool(best.success), optimizer_message=str(best.message),
            evaluations=sum(int(fit.nfev) for fit in fits), active_bounds=active,
            parameter_bounds=dict(zip(shared, bounds)), gas_bounds=list(gas_bounds),
            endpoint_policy=("positive training aqueous-capacity weighted 99th percentile, pooled with equal experiment weight"
                             if scope == "shared" else "positive training aqueous-capacity weighted 99th percentile"),
            shared_scope="endpoint, amplitude, gamma, gas_saturation and beta shared; observed reference and anchor per experiment",
            loss="equal-run mean of NMAE + 0.25 NRMSE; normalization=max(current injected mass, 5% final dose)",
            mass_projection=False, contact_or_temporal_penalty=False,
            overflow_loss=(
                "weighted equal-run mean of uncertainty-interval gas mass residuals; training observations only"
                if overflow_mode == "constrain" else "none"),
            overflow_hard=(dict(original_gas_bounds=list(map(float, gas_bounds)),
                feasible_gas_bounds=list(hard_gas_bounds),
                training_constraints=hard_constraints,
                held_out_references=len(references)-len(hard_constraints),
                policy="measured training-time gas intervals intersected with physical saturation bounds; no penalty or hidden gas")
                if overflow_mode == "hard" else None),
            overflow_weight=float(overflow_weight),
            **({"optimization": search_diagnostics} if search_diagnostics is not None else {})))
