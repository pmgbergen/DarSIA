"""F fit control with the unchanged original N1 optimisation kernels.

The private control body is a literal source port of ``n1_refit.fit_candidate``
after its OFF-only gate. Every numerical kernel remains the original function.
The public numerical result is explicitly separate and grants no source,
owned-preparation, calibration, job or phase-two admission.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
import numpy as np

from .n1_refit import (FIT_FORMAT, RefitRules, validate_rows, anchors_by_run,
    select_frames, frame_weights, compile_frame, evaluate_compiled, multistart,
    audit_grid, parameters, predict_run)
from .overflow_statistics import STATISTICS_FORMAT

NUMERIC_FORMAT = "darsia-overflow-refit-numerical-result-v1"
VARIANT_NAME = "Overflow corrected"
REFERENCE_NAME = "Reference without overflow"


def _require_adjusted_rows(rows, rules):
    if (type(rules) is not RefitRules or rules.variant_f != {"enabled": True}
            or type(rows) is not list or not rows
            or any(type(row) is not dict or row.get("contract") != STATISTICS_FORMAT
                   or type(row.get("overflow_steering")) is not dict
                   or row["overflow_steering"].get("format") != "darsia-overflow-statistics-derivation-v1"
                   for row in rows)):
        raise ValueError("The separate F numerical controller requires explicit F rules and derived F rows")


def _fit_control(rows, rules, *, excluded_run=None, progress=None):
    """Fit the gain, freeze the late sample, then beta only; optionally without one fit run."""
    _require_adjusted_rows(rows, rules)
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


def fit_adjusted_rows(rows, rules, *, excluded_run=None, progress=None):
    """Numerical F control only; original owned source admission is separate."""
    fitted = _fit_control(rows, rules, excluded_run=excluded_run, progress=progress)
    return dict(format=NUMERIC_FORMAT, variant_name=VARIANT_NAME,
        comparison_variant_name=REFERENCE_NAME, numeric_fit=fitted,
        fitted_parameters=deepcopy(fitted["parameters"]), stop_required=fitted["stop_required"],
        heldout_fit_run=excluded_run, source_admitted=False,
        owned_preparation_or_fit_verified=False, calibration_admitted=False,
        production_admitted=False)
