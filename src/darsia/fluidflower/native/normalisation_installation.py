"""Source-bound donor/flat iteration for saved image normalisation.

Numerical order follows ``fit_multirun_observed_sand_v4r5.py`` with SHA-256
819d783521e91d5e3cae6118b2b1932ec724a9cadff7855221076b735057a2d0.
This module takes explicit input arrays and method data; it neither chooses
sources, changes a series, nor starts any external job.
"""

from copy import deepcopy
from contextlib import nullcontext
import warnings

import numpy as np

from . import normalisation_residual as residual
from . import photometry

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


def median_masked(arrays):
    """Retain the original stacked float32 nanmedian operation."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.stack(arrays), axis=0).astype(np.float32)


def fit_installation(*, baseline_rgb, fit_support, chart_xy, chart_log_rgb,
                     chart_clipped, grid, old_donors, new_donors, new_light_runs,
                     reference_run, matrix, common_old, common_new, parameters,
                     convergence, residual_parameters, photometry_parameters,
                     on_iteration=None, check_cancelled=None):
    """Reproduce the original donor iteration on already bound fitting inputs.

    Every run is refitted using the same current donor state. Only the explicit
    donor lists update that state; other runs never contribute to its median.
    The caller must verify the RAW/source and parameter bindings before entry
    and compare the returned assets with the exact saved assets before use.
    """
    runs = tuple(baseline_rgb)
    old_donors, new_donors = tuple(old_donors), tuple(new_donors)
    if not runs or len(set(old_donors + new_donors)) != len(old_donors + new_donors):
        raise ValueError("The original donor lists must be nonempty and distinct.")
    if not old_donors or not new_donors or not set(old_donors + new_donors) <= set(runs):
        raise ValueError("Every original donor needs a bound baseline.")
    if reference_run not in runs or not set(new_light_runs) <= set(runs):
        raise ValueError("Reference and light-class assignments must belong to the bound runs.")
    if set(chart_xy) != set(runs) or set(chart_log_rgb) != set(runs) or set(chart_clipped) != set(runs) or set(fit_support) != set(runs):
        raise ValueError("Every baseline needs its own chart and fitting support.")
    for run in runs:
        if baseline_rgb[run].dtype != np.float32 or baseline_rgb[run].shape != (3, *grid.shape_hw):
            raise ValueError("Baseline RGB must retain its float32 reference-grid representation.")
        if fit_support[run].dtype != np.int16 or fit_support[run].shape != grid.shape_hw:
            raise ValueError("Fitting support must retain its int16 reference-grid representation.")
    if any(mask.dtype != bool or mask.shape != grid.shape_hw for mask in (common_old, common_new)):
        raise ValueError("Common donor support must be boolean on the reference grid.")
    if np.any(common_new & ~common_old):
        raise ValueError("New-light donor support must lie inside old-light support.")

    def cancelled():
        if check_cancelled is not None:
            check_cancelled()

    def estimate(fit, reference, consensus, corrected):
        return residual.estimate_old_B_field(fit, reference, consensus, corrected,
            parameters=residual_parameters, photometry_parameters=photometry_parameters)

    virtual_fac = common_old.astype(np.int16)
    consensus = median_masked([
        np.where(fit_support[run][None] == 1, baseline_rgb[run], np.nan)
        for run in old_donors
    ])
    consensus[:, ~common_old] = np.nan
    flat = np.zeros_like(consensus)
    flat[:, ~common_old] = 0

    def fit_ref_and_runs(current_cons, current_flat):
        cancelled()
        xy = chart_xy[reference_run] + np.array(parameters["chart_offset_xy"])
        reference = photometry.fit_observed_baseline(
            run=parameters["virtual_reference_name"], logB=current_cons,
            facies=virtual_fac, chart_xy=xy, chart_log_rgb=chart_log_rgb[reference_run],
            ref_chart_log_rgb=chart_log_rgb[reference_run],
            chart_usable=~chart_clipped[reference_run], matrix=np.eye(3),
            rig_consensus=current_cons, grid=grid, parameters=photometry_parameters)
        fitted = {}
        context = threadpool_limits(limits=1) if threadpool_limits else nullcontext()
        with context:
            for run in runs:
                cancelled()
                fitted[run] = photometry.fit_observed_baseline(
                    run=run, logB=baseline_rgb[run], facies=fit_support[run],
                    chart_xy=chart_xy[run] + np.array(parameters["chart_offset_xy"]),
                    chart_log_rgb=chart_log_rgb[run], ref_chart_log_rgb=chart_log_rgb[reference_run],
                    chart_usable=~(chart_clipped[run] | chart_clipped[reference_run]),
                    matrix=matrix if run in new_light_runs else np.eye(3),
                    rig_consensus=current_cons, flat=current_flat if run in new_light_runs else None,
                    grid=grid, parameters=photometry_parameters)
        return reference, fitted

    def update(current_cons, current_flat):
        reference, fitted = fit_ref_and_runs(current_cons, current_flat)
        corrected, without_flat, local = {}, {}, {}
        for run in old_donors + new_donors:
            cancelled()
            fit = fitted[run]
            corrected[run] = photometry.apply_grid(baseline_rgb[run], fit, reference, grid,
                                                   parameters=photometry_parameters)
            if run in old_donors:
                local[run], _ = estimate(fit, reference, current_cons, corrected[run])
                corrected[run] = residual.apply_old_field(corrected[run], local[run])
            saved = fit.F
            fit.F = None
            try:
                without_flat[run] = photometry.apply_grid(baseline_rgb[run], fit, reference, grid,
                                                          parameters=photometry_parameters)
            finally:
                fit.F = saved
            if run in old_donors:
                without_flat[run] = residual.apply_old_field(without_flat[run], local[run])
        next_cons = median_masked([
            np.where(fit_support[run][None] == 1, without_flat[run], np.nan)
            for run in old_donors
        ])
        next_cons[:, ~common_old] = np.nan
        next_flat = photometry.make_local_flat([
            np.where(fit_support[run][None] == 1, without_flat[run], np.nan)
            for run in new_donors
        ], next_cons, reference, common_new, grid, parameters=photometry_parameters)
        return reference, fitted, corrected, next_cons, next_flat

    history, prior = [], None
    converged = False
    for iteration in range(1, parameters["maximum_asset_iterations"] + 1):
        reference, fitted, corrected, next_cons, next_flat = update(consensus, flat)
        valid = reference.sand
        fd = np.abs(next_flat[:, valid] - flat[:, valid])
        cd = np.abs(next_cons[:, valid] - consensus[:, valid])
        medians, p95s = [], []
        if prior is not None:
            for run in old_donors + new_donors:
                mask = fitted[run].sand
                if int(mask.sum()) < convergence["facies_minimum_pixels"]:
                    continue
                difference = np.abs(corrected[run][:, mask] - prior[run][:, mask])
                medians.append(float(np.nanmedian(difference)))
                p95s.append(float(np.nanpercentile(difference, 95)))
        row = dict(iteration=iteration,
            raw_flat_median_abs_log=float(np.nanmedian(fd)),
            raw_flat_p95_abs_log=float(np.nanpercentile(fd, 95)),
            raw_consensus_median_abs_log=float(np.nanmedian(cd)),
            max_run_facies_corrected_change_median_abs_log=max(medians, default=None),
            max_run_facies_corrected_change_p95_abs_log=max(p95s, default=None))
        history.append(row)
        if on_iteration is not None:
            on_iteration(deepcopy(row))
        consensus = (parameters["previous_weight"] * consensus + parameters["next_weight"] * next_cons).astype(np.float32)
        flat = (parameters["previous_weight"] * flat + parameters["next_weight"] * next_flat).astype(np.float32)
        prior = corrected
        converged = (
            iteration >= convergence["minimum_iteration"]
            and row["raw_flat_median_abs_log"] <= convergence["raw_flat_fixedpoint_residual_median_abs_log_over_donor_sand_all_channels"]
            and row["raw_flat_p95_abs_log"] <= convergence["raw_flat_fixedpoint_residual_p95_abs_log_over_donor_sand_all_channels"]
            and row["raw_consensus_median_abs_log"] <= convergence["raw_consensus_fixedpoint_residual_median_abs_log_over_donor_sand_all_channels"]
            and row["max_run_facies_corrected_change_median_abs_log"] is not None
            and row["max_run_facies_corrected_change_median_abs_log"] <= convergence["donor_corrected_baseline_change_median_abs_log_per_run_and_facies"]
            and row["max_run_facies_corrected_change_p95_abs_log"] <= convergence["donor_corrected_baseline_change_p95_abs_log_per_run_and_facies"]
        )
        if converged:
            break
    if not converged:
        raise ValueError("Observed-sand donor installation did not converge; no asset saved")

    final_ref, final_fits = fit_ref_and_runs(consensus, flat)
    local_fields, local_details = {}, {}
    for run in runs:
        cancelled()
        if run in new_light_runs:
            continue
        base = photometry.apply_grid(baseline_rgb[run], final_fits[run], final_ref, grid,
                                     parameters=photometry_parameters)
        local_fields[run], local_details[run] = estimate(final_fits[run], final_ref, consensus, base)
    return dict(reference=final_ref, fits=final_fits, consensus=consensus, flat=flat,
                old_fields=local_fields, old_field_details=local_details,
                iteration_history=history, convergence_pass=True)
