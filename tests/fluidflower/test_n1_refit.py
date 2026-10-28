"""Tests of the configuration-driven N1 refit on synthetic estimate rows."""
from __future__ import annotations

import numpy as np
import pytest

from darsia.fluidflower.mass import n1_refit
from darsia.fluidflower.mass.n1_statistics import CONTRACT

TRUE_GAIN, TRUE_BETA = 23.0, 0.7
STAGE1 = [0.5 + 0.125 * i for i in range(16)]
STAGE2 = [2.5, 4.0, 8.0, 12.0, 24.0, 48.0, 72.0]


def small_grids():
    """Grid definitions shaped like the production ones, with fewer points."""
    return dict(gain_grid=dict(spacing="geometric", wide=[[1e-2, 1e3, 61]],
                               local=dict(factor_low=.5, factor_high=2., floor=1e-8, points=41)),
                beta_grid=dict(spacing="linear", wide=[[0., 4., 81], [4., 16., 13]], extra=[128.],
                               local=dict(half_width=.25, floor=0., points=41)))


def make_plan(fit_runs, *, variant_f=False, baseline=None, **rules):
    """A refit plan with the production rule values and small grids."""
    value = dict(format=n1_refit.RULES_FORMAT, stage1_window_h=[.5, 2.375], stage2_start_h=2.5,
                 post_bin_edges_h=[8., 48.], width_limit=.05, anchor_target_h=2.247, anchor_tolerance_h=.13,
                 baseline=baseline or dict(candidate="N1", gain=24.0, beta=0.6),
                 gain_starts=[1., 20., 24.], beta_starts=[.6, 0., 1., 3.],
                 require_frame_list_evidence=False, **small_grids())
    value.update(rules)
    return dict(runs=dict(fit=list(fit_runs), heldout=["held"], excluded=[]), rules=value,
                variant_f=dict(enabled=variant_f))


def make_row(run, frame_index, target_h, *, scale=1.0, dose=0.4, urest=1e-7, paired=None, gas=0.01):
    """One estimate row whose N1 mass meets the dose at the true gain and beta."""
    volume_end = 1e-3
    volume = volume_end * (1 + 0.2 * max(target_h - 2.375, 0.0))
    factor = (volume / volume_end) ** -TRUE_BETA
    relative = np.array([.001, .004, .01, .02])
    weights = np.array([1., 2., 3., 1.5])
    total = (dose * scale - gas) / (TRUE_GAIN * factor)
    capacity = weights / float(np.dot(weights, relative)) * total
    paired = paired or dict(own_native_relative=[], column_native_relative=[], solubility_ratio=[], kind=[],
                            aqueous_capacity_g=[])
    return dict(contract=CONTRACT, run=run, frame_index=frame_index, elapsed_h=target_h, selected_target_h=target_h,
                injected_mass_g=dose, source_binding=dict(source_sha256="0" * 64),
                no_outflow_evidence=dict(no_co2_left_rig=True, source=dict(sha256="1" * 64)),
                main_aqueous_histogram=dict(native_relative=relative.tolist(), capacity_g=capacity.tolist()),
                main_gas_mass_g=gas * .9, main_gas_water_mass_g=gas * .1, affected_pore_volume_m3=volume,
                estimate_fixed_gas_mass_g=0., estimate_fixed_gas_water_mass_g=0., estimate_gas_constant_g=0.,
                estimate_geometry_extended_area_cm2=0., urest_g=urest, excluded_parts={},
                paired_estimates=dict(paired, aqueous_capacity_total_g=float(np.sum(paired["aqueous_capacity_g"])),
                                      ratio_scale=1.0))


def make_rows(runs, scales=None):
    rows = []
    for run in runs:
        for index, target in enumerate(STAGE1 + STAGE2):
            rows.append(make_row(run, index + 1, target, scale=(scales or {}).get(run, 1.0)))
    return rows


def test_compiled_evaluation_equals_the_canonical_summary_evaluation():
    rng = np.random.default_rng(3)
    n = 400
    paired = dict(own_native_relative=rng.uniform(0, .05, n).tolist(),
                  column_native_relative=rng.uniform(0, .05, n).tolist(),
                  solubility_ratio=rng.uniform(.5, 1.5, n).tolist(), kind=rng.choice([1, 2, 3, 5, 6, 7], n).tolist(),
                  aqueous_capacity_g=rng.uniform(0, 1e-4, n).tolist())
    row = make_row("r1", 1, 6.0, paired=paired)
    anchor = dict(elapsed_h=2.25, affected_pore_volume_m3=1e-3)
    for gain in (3., 23., 80.):
        compiled = n1_refit.compile_frame(row, gain)
        for beta in (0., .4, 1.3, 5.):
            canonical = n1_refit.evaluate(row, gain, beta, anchor)
            fast = n1_refit.evaluate_compiled(compiled, beta, anchor)
            for key in ("E_lo_g", "E_hi_g", "W_g", "total_mass_g", "W_over_d"):
                assert fast[key] == pytest.approx(canonical[key], rel=1e-12, abs=1e-18)
            assert fast["signed_interval_residual"] == pytest.approx(canonical["signed_interval_residual"],
                                                                     rel=1e-12, abs=1e-15)
    paired["kind"][0] = 4  # neither estimate: the statistics never write such a pair
    with pytest.raises(ValueError, match="own or a column estimate"):
        n1_refit.compile_frame(make_row("r1", 1, 6.0, paired=paired), 23.)


def test_two_stage_refit_recovers_a_known_gain_and_beta():
    runs = ["r1", "r2", "r3"]
    rules = n1_refit.RefitRules.from_plan(make_plan(runs))
    events = []
    fit = n1_refit.fit_candidate(make_rows(runs), rules, progress=lambda **event: events.append(event["step"]))
    assert fit["stop_required"] is False
    assert fit["parameters"]["gain"] == pytest.approx(TRUE_GAIN, rel=1e-4)
    assert fit["parameters"]["beta"] == pytest.approx(TRUE_BETA, rel=1e-3)
    assert fit["stages"]["stage1"]["frames"] == 3 * 16 and fit["stages"]["stage2"]["frames"] == 3 * 7
    assert set(fit["grid_audits"]) == {"stage1", "stage2"}
    assert events == ["stage1", "stage1_grid", "stage2", "stage2_grid"]
    assert fit["anchors_by_run"]["r1"]["frame_index"] == STAGE1.index(2.25) + 1


def test_selection_records_every_omission_reason_and_weights_runs_equally():
    runs = ["r1", "r2"]
    rules = n1_refit.RefitRules.from_plan(make_plan(runs))
    rows = make_rows(runs)
    rows[0]["injected_mass_g"] = 0.
    rows[1]["urest_g"] = None
    rows[2]["no_outflow_evidence"] = dict(no_co2_left_rig=False, source=dict(sha256="1" * 64))
    rows[20]["urest_g"] = 1.0  # far wider than 5 % of the dose
    stages, decisions = n1_refit.select_frames(rows, rules, excluded_run="r2", stage1_gain=TRUE_GAIN,
                                               anchors=n1_refit.anchors_by_run(rows, rules))
    reasons = {(d["run"], d["frame_index"]): d["omitted_reasons"] for d in decisions}
    assert reasons[("r1", 1)] == ["no_positive_dose"]
    assert reasons[("r1", 2)] == ["estimate_geometry_undefined"]
    assert reasons[("r1", 3)] == ["no_loss_proxy_not_passed_or_unbound"]
    assert reasons[("r1", 21)] == ["baseline_width_exceeds_limit"]
    assert all(r == ["leave_one_run_out"] for (run, _), r in reasons.items() if run == "r2")
    weights = n1_refit.frame_weights(stages["stage2"], rules, post=True)
    by_bin = {}
    for row, weight in zip(stages["stage2"], weights):
        by_bin.setdefault(rules.post_bin(row["selected_target_h"]), []).append(weight ** 2)
    assert sorted(by_bin) == ["2.5-8h", "8-48h", "after48h"]
    assert np.allclose([sum(v) for v in by_bin.values()], 1 / 3)


def test_a_grid_point_below_the_fitted_objective_requires_a_stop():
    def residual(vector):
        x = float(vector[0])
        return np.array([min((x - 1.0) ** 2 + .1, (x - 3.0) ** 2)])
    grid = small_grids()["beta_grid"]
    assert n1_refit.audit_grid(residual, 1.0, grid, kind="beta")["stop_required"] is True
    assert n1_refit.audit_grid(residual, 3.0, grid, kind="beta")["stop_required"] is False


def test_leave_one_run_out_fits_without_each_run_and_predicts_it():
    runs = ["r1", "r2", "r3"]
    rules = n1_refit.RefitRules.from_plan(make_plan(runs))
    rows = make_rows(runs)
    folds = n1_refit.leave_one_run_out(rows, rules)
    assert sorted(folds) == runs
    for run, fold in folds.items():
        assert fold["fit"]["heldout_fit_run"] == run and run not in fold["fit"]["calibration_runs"]
        assert [p["frame_index"] for p in fold["predictions"]] == list(range(1, 24))
        assert {p["run"] for p in fold["predictions"]} == {run}


def test_variant_f_is_refused_until_its_decisions_exist():
    rules = n1_refit.RefitRules.from_plan(make_plan(["r1"], variant_f=True))
    with pytest.raises(ValueError, match="Variant F waits"):
        n1_refit.fit_candidate(make_rows(["r1"]), rules)


def test_target_times_limit_the_fit_to_an_earlier_time_grid():
    runs = ["r1", "r2"]
    times = STAGE1[::2] + STAGE2[:4]  # every second injection time, 0.5 to 2.25 h
    rules = n1_refit.RefitRules.from_plan(make_plan(runs, target_times_h=times))
    stages, decisions = n1_refit.select_frames(make_rows(runs), rules)
    assert len(stages["stage1"]) == 2 * 8
    assert {decision["selected_target_h"] for decision in decisions} == set(times)
    for invalid in ([], [1.0, 1.0]):
        with pytest.raises(ValueError, match="Target times"):
            n1_refit.RefitRules.from_plan(make_plan(runs, target_times_h=invalid))


def test_rules_need_ordered_windows_and_distinct_roles():
    with pytest.raises(ValueError, match="ordered"):
        n1_refit.RefitRules.from_plan(make_plan(["r1"], stage2_start_h=2.0))
    plan = make_plan(["r1", "held"])
    with pytest.raises(ValueError, match="held out"):
        n1_refit.RefitRules.from_plan(plan)
    with pytest.raises(ValueError, match="Only the plan's fit runs"):
        n1_refit.validate_rows(make_rows(["other"]), n1_refit.RefitRules.from_plan(make_plan(["r1"])))
