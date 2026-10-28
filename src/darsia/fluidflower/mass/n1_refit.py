"""Refit of the N1 gain and dilution exponent from per-frame estimate rows.

The rows are the per-frame ``darsia-n1-estimates-v1`` summaries that phase-one
production writes (``n1_statistics``). Every prediction comes from the
canonical ``n1_statistics.evaluate_summary``; this module only selects frames,
weights them, optimises and audits. Its arithmetic is that of the earlier
common-mass fit, with every series value taken from a plan (:class:`RefitRules`):

* stage 1 fits the gain on the injection frames with dilution off;
* the stage-2 sample is frozen at that gain and the baseline beta (W/d at most
  the width limit), then beta is fitted with the gain fixed; every run has
  equal weight, split equally over its occupied post-injection time bins;
* each stage gets a wide and a local grid profile of the identical residual,
  and a grid point with a lower objective requires a stop;
* leave-one-run-out repeats the whole fit without one fit run and predicts
  that run's frames.

Nothing is fitted in the gas support, no map is rescaled and overflow is not in
the objective. Variant F (overflow-governed free gas) is a plan switch that is
refused until its decisions exist.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

FIT_FORMAT = "darsia-n1-refit-fit-v1"
RULES_FORMAT = "darsia-n1-refit-rules-v1"
VARIANT_F_REFUSAL = "Using overflow as the free-gas target requires reviewed overflow inputs and the overflow-target calibration workflow"


@dataclass(frozen=True)
class RefitRules:
    """Series values of one refit; every value comes from the plan.

    Times are hours after injection start. ``stage1_window_h`` is closed;
    stage 2 takes frames from ``stage2_start_h``. ``post_bin_edges_h`` split
    stage 2 into ``[start, e1]``, ``(e1, e2]``, ..., ``(last edge, end]``.
    ``target_times_h``, when given, limits the fit to frames selected at
    those exact target times (for example an earlier, coarser time grid).
    The anchor of each run is its frame nearest ``anchor_target_h`` within
    ``anchor_tolerance_h``.
    """

    fit_runs: tuple
    stage1_window_h: tuple
    stage2_start_h: float
    post_bin_edges_h: tuple
    width_limit: float
    anchor_target_h: float
    anchor_tolerance_h: float
    baseline: dict
    gain_starts: tuple
    beta_starts: tuple
    gain_grid: dict
    beta_grid: dict
    require_frame_list_evidence: bool
    variant_f: dict
    target_times_h: tuple | None = None

    @classmethod
    def from_plan(cls, plan):
        """Admit the ``rules`` and run roles of a refit plan."""
        rules = plan["rules"]
        if rules.get("format") != RULES_FORMAT:
            raise ValueError("Explicit refit rules are required")
        roles = plan["runs"]
        fit = tuple(roles["fit"])
        if not fit or len(set(fit)) != len(fit):
            raise ValueError("The refit needs distinct fit runs")
        if set(fit) & (set(roles.get("heldout", ())) | set(roles.get("excluded", ()))):
            raise ValueError("A fit run cannot also be held out or excluded")
        baseline = dict(rules["baseline"])
        if baseline.get("candidate") != "N1" or not (baseline["gain"] > 0 and baseline["beta"] >= 0):
            raise ValueError("The baseline must be an N1 gain and beta")
        window = tuple(float(v) for v in rules["stage1_window_h"])
        edges = tuple(float(v) for v in rules["post_bin_edges_h"])
        if (len(window) != 2 or window[0] >= window[1] or float(rules["stage2_start_h"]) <= window[1]
                or list(edges) != sorted(edges) or any(e <= float(rules["stage2_start_h"]) for e in edges)):
            raise ValueError("Stage windows and post-injection bins must be ordered")
        variant_f = dict(plan.get("variant_f", {"enabled": False}))
        times = rules.get("target_times_h")
        if times is not None and (not times or len(set(map(float, times))) != len(times)):
            raise ValueError("Target times must be a non-empty list of distinct hours")
        return cls(fit_runs=fit, stage1_window_h=window, stage2_start_h=float(rules["stage2_start_h"]),
                   post_bin_edges_h=edges, width_limit=float(rules["width_limit"]),
                   anchor_target_h=float(rules["anchor_target_h"]),
                   anchor_tolerance_h=float(rules["anchor_tolerance_h"]), baseline=baseline,
                   gain_starts=tuple(float(v) for v in rules["gain_starts"]),
                   beta_starts=tuple(float(v) for v in rules["beta_starts"]),
                   gain_grid=dict(rules["gain_grid"]), beta_grid=dict(rules["beta_grid"]),
                   require_frame_list_evidence=bool(rules["require_frame_list_evidence"]),
                   variant_f=variant_f,
                   target_times_h=None if times is None else tuple(float(value) for value in times))

    def require_variant_f_off(self):
        """Refuse variant F until its decisions and inputs exist."""
        if self.variant_f.get("enabled", False) is not False:
            raise ValueError(VARIANT_F_REFUSAL)

    def stage(self, target_h):
        """'stage1', 'stage2' or None for a selected target time."""
        if self.target_times_h is not None and float(target_h) not in self.target_times_h:
            return None
        low, high = self.stage1_window_h
        if low <= target_h <= high:
            return "stage1"
        if target_h >= self.stage2_start_h:
            return "stage2"
        return None

    def post_bin(self, target_h):
        """The post-injection bin of a stage-2 target time, as a label."""
        if target_h < self.stage2_start_h:
            return None
        lower = self.stage2_start_h
        for edge in self.post_bin_edges_h:
            if target_h <= edge:
                return f"{lower:g}-{edge:g}h"
            lower = edge
        return f"after{lower:g}h"


def parameters(gain, beta):
    """The N1 parameter mapping that ``evaluate_summary`` takes."""
    return dict(candidate="N1", gain=float(gain), beta=float(beta))


def evaluate(row, gain, beta, anchor):
    """One canonical per-frame evaluation."""
    from .n1_statistics import evaluate_summary
    return evaluate_summary(row, parameters(gain, beta), anchor)


def compile_frame(row, gain):
    """Compile one row at a fixed gain so that a beta evaluation is logarithmic.

    This is ``evaluate_summary`` rewritten exactly for a fixed gain. Most pairs
    have two diluted aqueous estimates and reduce to two sums. A pair with
    undiluted gas-water above it has one crossing; sorted crossings and prefix
    sums keep every per-pixel min/max. The tests require agreement with
    ``evaluate_summary``.
    """
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("One positive common gain is required")
    paired = row["paired_estimates"]
    own, column, ratio, kind, weight = (np.asarray(paired[key], np.float64 if key != "kind" else np.int64)
                                        for key in ("own_native_relative", "column_native_relative",
                                                    "solubility_ratio", "kind", "aqueous_capacity_g"))
    first = np.clip(own * gain, 0, 1)
    second = ratio * np.clip(column * gain, 0, 1)
    has1, has2, gas2 = (kind & 1) != 0, (kind & 2) != 0, (kind & 4) != 0
    if np.any(~has1 & ~has2):
        raise ValueError("Every pair needs an own or a column estimate, as the statistics write them")
    both_aq = has1 & has2 & ~gas2
    only_aq = (has1 & ~has2) | (~has1 & has2 & ~gas2)
    single = np.where(has1, first, second)
    lo_aq = float(np.sum(weight[both_aq] * np.minimum(first[both_aq], second[both_aq]))
                  + np.sum(weight[only_aq] * single[only_aq]))
    hi_aq = float(np.sum(weight[both_aq] * np.maximum(first[both_aq], second[both_aq]))
                  + np.sum(weight[only_aq] * single[only_aq]))
    only_gas = ~has1 & has2 & gas2
    constant = row["estimate_gas_constant_g"] + float(np.sum(weight[only_gas] * ratio[only_gas]))
    cross = has1 & has2 & gas2
    a, c = first[cross] * weight[cross], ratio[cross] * weight[cross]
    zero = a == 0
    constant_hi = float(c[zero].sum())
    a, c = a[~zero], c[~zero]
    cut = c / a
    order = np.argsort(cut, kind="stable")
    cut, a, c = cut[order], a[order], c[order]
    histogram = row["main_aqueous_histogram"]
    main_coefficient = float(np.dot(np.asarray(histogram["capacity_g"], np.float64),
                                    np.clip(np.asarray(histogram["native_relative"], np.float64) * gain, 0, 1)))
    compiled = dict(row=row, gain=float(gain), main_coefficient_g=main_coefficient,
                lo_aqueous_coefficient_g=lo_aq, hi_aqueous_coefficient_g=hi_aq,
                constant_g=constant, constant_hi_g=constant_hi, crossings=cut,
                prefix_a=np.r_[0., np.cumsum(a)], prefix_c=np.r_[0., np.cumsum(c)],
                total_a=float(a.sum()), total_c=float(c.sum()))
    if row.get("analysis_scope") is not None:
        compiled["analysis_scope"] = deepcopy(row["analysis_scope"])
    return compiled


def evaluate_compiled(compiled, beta, anchor):
    """Interval, width and residual of a compiled row at one beta."""
    from .concentration_joint import dilution_factor
    from .n1_statistics import interval_residual
    from .n1_arithmetic import scoped_result
    row = compiled["row"]
    factor = dilution_factor(row["elapsed_h"], row["affected_pore_volume_m3"], anchor, beta)
    index = int(np.searchsorted(compiled["crossings"], factor, side="right"))
    a, c = compiled["prefix_a"][index], compiled["prefix_c"][index]
    cross_lo = c + (compiled["total_a"] - a) * factor
    cross_hi = a * factor + compiled["total_c"] - c
    lo = compiled["constant_g"] + compiled["lo_aqueous_coefficient_g"] * factor + cross_lo
    hi = compiled["constant_g"] + compiled["hi_aqueous_coefficient_g"] * factor + cross_hi + compiled["constant_hi_g"]
    main = compiled["main_coefficient_g"] * factor + row["main_gas_mass_g"] + row["main_gas_water_mass_g"]
    upper = row["urest_g"]
    width = None if upper is None else hi - lo + upper
    dose = row["injected_mass_g"]
    residual = None if width is None or dose is None or dose <= 0 else interval_residual(main + lo, dose, width)
    result = dict(main_mass_g=main, total_mass_g=main, E_lo_g=lo, E_hi_g=hi, U_rest_g=upper, W_g=width,
                W_over_d=None if width is None or dose is None or dose <= 0 else width / dose,
                signed_interval_residual=residual, delta=None if residual is None else abs(residual),
                dilution_factor=factor)
    return scoped_result(result, compiled.get("analysis_scope") or row.get("analysis_scope"))


def validate_rows(rows, rules):
    """Require fit-run rows with unique frames, targets and source bindings."""
    if not rows or any(row.get("run") not in rules.fit_runs for row in rows):
        raise ValueError("Only the plan's fit runs enter the refit")
    identities = [(row["run"], row["frame_index"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate source frames cannot receive repeated weight")
    if any(row.get("selected_target_h") is None or not row.get("source_binding") for row in rows):
        raise ValueError("Each row needs its selected target time and source binding")


def anchors_by_run(rows, rules):
    """Each run's dilution anchor: its frame nearest the injection-end target."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["run"]].append(row)
    anchors = {}
    for run, values in groups.items():
        if any(row.get("analysis_scope") is not None for row in values):
            values = [row for row in values if row.get("analysis_scope") is None]
            if not values:
                raise ValueError(f"No source-bound injection-end volume anchor for {run}")
        anchor = min(values, key=lambda row: (abs(row["elapsed_h"] - rules.anchor_target_h), row["frame_index"]))
        if (abs(anchor["elapsed_h"] - rules.anchor_target_h) > rules.anchor_tolerance_h
                or anchor["affected_pore_volume_m3"] <= 0):
            raise ValueError(f"No source-bound injection-end volume anchor for {run}")
        anchors[run] = {key: anchor[key] for key in
                        ("elapsed_h", "affected_pore_volume_m3", "frame_index", "source_binding")}
    return anchors


def frame_weights(rows, rules, *, post=False):
    """Equal weight per run; after injection also per occupied time bin, then per frame."""
    counts = Counter(row["run"] for row in rows)
    if not counts:
        return np.zeros(0)
    if not post:
        return np.array([1 / np.sqrt(len(counts) * counts[row["run"]]) for row in rows], np.float64)
    bins = defaultdict(set)
    bin_counts = Counter((row["run"], rules.post_bin(row["selected_target_h"])) for row in rows)
    for run, time_bin in bin_counts:
        bins[run].add(time_bin)
    return np.array([1 / np.sqrt(len(counts) * len(bins[row["run"]])
                                 * bin_counts[(row["run"], rules.post_bin(row["selected_target_h"]))])
                     for row in rows], np.float64)


def select_frames(rows, rules, *, excluded_run=None, stage1_gain=None, anchors=None):
    """Split rows into the two stages and record why each frame is in or out."""
    validate_rows(rows, rules)
    if excluded_run is not None and excluded_run not in rules.fit_runs:
        raise ValueError("A leave-one-run-out fold must omit one of the fit runs")
    stages, decisions = {"stage1": [], "stage2": []}, []
    baseline_beta = float(rules.baseline["beta"])
    for row in rows:
        stage = rules.stage(row["selected_target_h"])
        if stage is None:
            continue
        reasons = []
        if row["run"] == excluded_run:
            reasons.append("leave_one_run_out")
        if rules.require_frame_list_evidence:
            frame_list = row.get("frame_list_evidence") or {}
            if frame_list.get("accepted") is not True or not frame_list.get("source_binding"):
                reasons.append("frame_list_rejected_or_unbound")
        no_loss = row.get("no_outflow_evidence") or {}
        if no_loss.get("no_co2_left_rig") is not True or not (no_loss.get("source") or no_loss.get("source_binding")):
            reasons.append("no_loss_proxy_not_passed_or_unbound")
        dose = row.get("injected_mass_g")
        if dose is None or not np.isfinite(dose) or dose <= 0:
            reasons.append("no_positive_dose")
        if row.get("urest_g") is None or row.get("analysis_scope") is not None:
            reasons.append("estimate_geometry_undefined")
        baseline = None
        if stage == "stage2":
            if stage1_gain is None or anchors is None:
                reasons.append("stage1_not_yet_frozen")
            else:
                baseline = evaluate_compiled(compile_frame(row, stage1_gain), baseline_beta, anchors[row["run"]])
                if baseline["W_over_d"] is None or baseline["W_over_d"] > rules.width_limit:
                    reasons.append("baseline_width_exceeds_limit")
        if not reasons:
            stages[stage].append(row)
        decision = dict(
            run=row["run"], frame_index=row["frame_index"], selected_target_h=row["selected_target_h"],
            elapsed_h=row["elapsed_h"], stage=stage, included=not reasons, omitted_reasons=reasons,
            baseline_eligibility_gain=stage1_gain if stage == "stage2" else None,
            baseline_eligibility_beta=baseline_beta if stage == "stage2" else None,
            baseline_W_over_d=None if baseline is None else baseline["W_over_d"],
            source_binding=row["source_binding"])
        if row.get("analysis_scope") is not None:
            decision["analysis_scope"] = deepcopy(row["analysis_scope"])
        decisions.append(decision)
    return stages, decisions


def multistart(function, starts, bounds):
    """Least squares from every start; keep the lowest cost and record each start."""
    from scipy.optimize import least_squares
    results = [least_squares(function, [start], bounds=bounds, max_nfev=600, ftol=1e-10, xtol=1e-10, gtol=1e-10)
               for start in starts]
    finite = [result for result in results if result.success and np.isfinite(result.cost)]
    if not finite:
        raise ValueError("The least-squares refit failed from every start")
    minimum = min(result.cost for result in finite)
    chosen = next(result for result in finite if result.cost <= minimum + max(1e-12, abs(minimum) * 1e-8))
    return chosen, dict(
        multistart=[dict(initial=float(start), solution=result.x.tolist(), cost=float(result.cost),
                         success=bool(result.success), nfev=int(result.nfev)) for start, result in zip(starts, results)],
        jacobian_rank=int(np.linalg.matrix_rank(chosen.jac)),
        jacobian_singular_values=np.linalg.svd(chosen.jac, compute_uv=False).tolist())


def grid_values(grid, fitted):
    """The audited parameter values: wide segments, extra points and a local profile."""
    segments = [np.geomspace(*segment[:2], int(segment[2])) if grid.get("spacing") == "geometric"
                else np.linspace(*segment[:2], int(segment[2])) for segment in grid["wide"]]
    local = grid["local"]
    if grid.get("spacing") == "geometric":
        segments.append(np.geomspace(max(fitted * local["factor_low"], local["floor"]), fitted * local["factor_high"],
                                     int(local["points"])))
    else:
        segments.append(np.linspace(max(local["floor"], fitted - local["half_width"]), fitted + local["half_width"],
                                    int(local["points"])))
    return np.unique(np.r_[tuple(segments) + (np.asarray(grid.get("extra", ()), np.float64), [fitted])])


def audit_grid(function, fitted, grid, *, kind):
    """Profile the identical residual over the grid; a lower grid objective requires a stop."""
    values = grid_values(grid, fitted)
    logarithmic = kind == "gain"
    coordinates = np.log(values) if logarithmic else values
    objectives = np.array([.5 * float(np.dot(value := function([coordinate]), value)) for coordinate in coordinates])
    residual = function([np.log(fitted) if logarithmic else fitted])
    fitted_objective = .5 * float(np.dot(residual, residual))
    index = int(np.argmin(objectives))
    tolerance = max(1e-12, abs(fitted_objective) * 1e-8)
    superior = bool(objectives[index] < fitted_objective - tolerance)
    return dict(parameter=kind, values=values.tolist(), objectives=objectives.tolist(), fitted_value=float(fitted),
                fitted_objective=fitted_objective, grid_minimum_value=float(values[index]),
                grid_minimum_objective=float(objectives[index]), grid_finds_lower_objective=superior,
                stop_required=superior, numerical_comparison_tolerance=tolerance,
                same_residual_and_weights=True, shape_profile_wide_and_local=True)


def fit_candidate(rows, rules, *, excluded_run=None, progress=None):
    """Fit the gain, freeze the late sample, then beta only; optionally without one fit run."""
    rules.require_variant_f_off()
    validate_rows(rows, rules)
    anchors = anchors_by_run(rows, rules)
    report = progress or (lambda **_: None)
    first, _ = select_frames(rows, rules, excluded_run=excluded_run)
    first = first["stage1"]
    if not first:
        raise ValueError("No source-bound selected injection frames are eligible")
    weights1 = frame_weights(first, rules)

    def residual1(vector):
        gain = float(np.exp(vector[0]))
        return weights1 * np.array([evaluate_compiled(compile_frame(row, gain), 0., anchors[row["run"]])
                                    ["signed_interval_residual"] for row in first], np.float64)

    report(step="stage1", frames=len(first))
    result1, audit1 = multistart(residual1, [math.log(start) for start in rules.gain_starts], ([-700.], [700.]))
    gain = float(np.exp(result1.x[0]))
    report(step="stage1_grid", gain=gain)
    gain_grid = audit_grid(residual1, gain, rules.gain_grid, kind="gain")
    stages, decisions = select_frames(rows, rules, excluded_run=excluded_run, stage1_gain=gain, anchors=anchors)
    second = stages["stage2"]
    common = dict(format=FIT_FORMAT, calibration_runs=[run for run in rules.fit_runs if run != excluded_run],
                  heldout_fit_run=excluded_run, anchors_by_run=anchors, frame_decisions=decisions)
    if gain_grid["stop_required"] or not second:
        return dict(common, stop_required=True,
                    reason="gain_grid_lower_objective" if gain_grid["stop_required"] else "no_eligible_late_frames",
                    parameters=parameters(gain, rules.baseline["beta"]), grid_audits=dict(stage1=gain_grid),
                    optimizer_audits=dict(stage1=audit1))
    weights2 = frame_weights(second, rules, post=True)
    compiled = [compile_frame(row, gain) for row in second]

    def residual2(vector):
        beta = float(vector[0])
        return weights2 * np.array([evaluate_compiled(value, beta, anchors[value["row"]["run"]])
                                    ["signed_interval_residual"] for value in compiled], np.float64)

    report(step="stage2", frames=len(second))
    result2, audit2 = multistart(residual2, list(rules.beta_starts), ([0.], [np.inf]))
    beta = float(result2.x[0])
    report(step="stage2_grid", beta=beta)
    beta_grid = audit_grid(residual2, beta, rules.beta_grid, kind="beta")
    for decision in decisions:
        if decision["included"] and decision["stage"] == "stage2":
            row = next(row for row in second
                       if row["run"] == decision["run"] and row["frame_index"] == decision["frame_index"])
            decision["fitted_beta_W_over_d"] = evaluate_compiled(compile_frame(row, gain), beta,
                                                                 anchors[row["run"]])["W_over_d"]
    return dict(common, parameters=parameters(gain, beta),
                stages={name: dict(frames=len(values), runs=sorted({row["run"] for row in values}),
                                   targets=dict(Counter(str(row["selected_target_h"]) for row in values)))
                        for name, values in stages.items()},
                grid_audits=dict(stage1=gain_grid, stage2=beta_grid),
                optimizer_audits=dict(stage1=audit1, stage2=audit2),
                stop_required=beta_grid["stop_required"],
                reason="beta_grid_lower_objective" if beta_grid["stop_required"] else None,
                stage2_eligibility_policy="frozen at this fit's stage-one gain and the baseline beta",
                equal_run_weight=True, equal_occupied_time_bin_weight=True, equal_frame_weight_within_bin=True,
                post_time_bins=sorted({rules.post_bin(row["selected_target_h"]) for row in second}),
                gas_branch_fitted=False, map_mass_rescaling=False, overflow_in_objective=False,
                final_heldout_used=False)


def predict_run(rows, fit, run):
    """Predictions of one run's frames with a fit's parameters and anchors."""
    anchor = fit["anchors_by_run"][run]
    gain, beta = fit["parameters"]["gain"], fit["parameters"]["beta"]
    predictions = []
    for row in sorted((row for row in rows if row["run"] == run), key=lambda row: row["frame_index"]):
        value = evaluate(row, gain, beta, anchor)
        prediction = {key: value[key] for key in (
            "run", "frame_index", "elapsed_h", "total_mass_g", "aqueous_mass_g", "gas_mass_g", "E_lo_g", "E_hi_g",
            "U_rest_g", "W_g", "W_over_d", "injected_mass_g", "interval_lower_g", "interval_upper_g",
            "signed_interval_residual", "delta", "dilution_factor")} | dict(selected_target_h=row["selected_target_h"])
        if value.get("analysis_scope") is not None:
            from .n1_arithmetic import public_analysis_scope
            prediction.update({key: item for key, item in value.items()
                               if key == "analysis_scope" or key.startswith("quantified_")})
            prediction = public_analysis_scope(prediction)
        predictions.append(prediction)
    return predictions


def leave_one_run_out(rows, rules, *, progress=None):
    """One fit without each fit run, and that run's predictions from its fold."""
    folds = {}
    for index, run in enumerate(rules.fit_runs):
        if progress:
            progress(step="loro", fold=index + 1, folds=len(rules.fit_runs), fit_run=run)
        fold = fit_candidate(rows, rules, excluded_run=run)
        folds[run] = dict(fit=fold, predictions=None if fold["stop_required"] else predict_run(rows, fold, run))
    return folds
