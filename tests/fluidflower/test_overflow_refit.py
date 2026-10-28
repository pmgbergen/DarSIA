"""Finite numerical F control; no completed cohort, claim or fitted source seal.

The existing synthetic N1 rows retain their exact mass/interval arithmetic.
Only the explicit derived-row contract is authored here. Operational admission
must independently reproduce actual indexed preparation/source bindings.
"""
from copy import deepcopy

import pytest

from darsia.fluidflower.mass import n1_refit, overflow_refit
from darsia.fluidflower.mass.overflow_statistics import STATISTICS_FORMAT
from tests.fluidflower.test_n1_refit import make_plan, make_rows


def adjusted(runs):
    rows = deepcopy(make_rows(runs))
    for row in rows:
        row["contract"] = STATISTICS_FORMAT
        row["overflow_steering"] = dict(format="darsia-overflow-statistics-derivation-v1")
    return rows


def test_explicit_f_control_uses_the_original_fit_and_grid_arithmetic():
    runs = ["r1", "r2"]
    original = n1_refit.fit_candidate(make_rows(runs),
        n1_refit.RefitRules.from_plan(make_plan(runs)))
    rows = adjusted(runs)
    before = deepcopy(rows)
    fitted = overflow_refit.fit_adjusted_rows(rows,
        n1_refit.RefitRules.from_plan(make_plan(runs, variant_f=True)))
    assert fitted["numeric_fit"] == original
    assert fitted["fitted_parameters"] == original["parameters"]
    assert "parameters" not in fitted
    assert fitted["format"] != original["format"]
    assert fitted["stop_required"] is False
    assert fitted["variant_name"] == "Overflow corrected"
    assert fitted["comparison_variant_name"] == "Reference without overflow"
    assert all(fitted[field] is False for field in
        ("source_admitted", "owned_preparation_or_fit_verified", "calibration_admitted", "production_admitted"))
    assert rows == before
    with pytest.raises(ValueError, match="Variant F waits"):
        n1_refit.fit_candidate(rows, n1_refit.RefitRules.from_plan(make_plan(runs, variant_f=True)))


def test_loro_reuses_the_same_rows_and_original_anchor_and_selection_rules():
    runs = ["r1", "r2"]
    rows = adjusted(runs)
    expected = n1_refit.fit_candidate(make_rows(runs),
        n1_refit.RefitRules.from_plan(make_plan(runs)), excluded_run="r2")
    result = overflow_refit.fit_adjusted_rows(rows,
        n1_refit.RefitRules.from_plan(make_plan(runs, variant_f=True)), excluded_run="r2")
    assert result["numeric_fit"] == expected
    assert result["heldout_fit_run"] == "r2"
    assert result["numeric_fit"]["calibration_runs"] == ["r1"]
    assert n1_refit.predict_run(rows, result["numeric_fit"], "r2") == n1_refit.predict_run(
        make_rows(runs), expected, "r2")


@pytest.mark.parametrize("fault", ["off-rules", "ordinary-row", "missing-derivation"])
def test_f_controller_cannot_inherit_ordinary_or_implicit_f_admission(fault):
    rows = adjusted(["r1"])
    rules = n1_refit.RefitRules.from_plan(make_plan(["r1"], variant_f=fault != "off-rules"))
    if fault == "ordinary-row":
        rows[0]["contract"] = make_rows(["r1"])[0]["contract"]
    elif fault == "missing-derivation":
        rows[0].pop("overflow_steering")
    with pytest.raises(ValueError, match="explicit F rules and derived F rows"):
        overflow_refit.fit_adjusted_rows(rows, rules)
