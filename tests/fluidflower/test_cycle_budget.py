"""Per-cycle mass budget."""
import math

import pytest

from darsia.fluidflower.cycle_budget import (
    GAS_DENSITY_KG_PER_M3,
    CycleBudgetInputs,
    budget_rows,
    cycle_budget,
    free_gas_mass_g,
    injected_mass_g,
)


def test_injected_mass_ffre_protocol():
    # 10 mL/min for 20 min at 1.872 kg/m3 = 0.3744 g (the provisional FF_RE value)
    assert injected_mass_g(10.0, 20.0, GAS_DENSITY_KG_PER_M3["15C"]) == pytest.approx(0.3744, rel=1e-6)
    assert injected_mass_g(10.0, 20.0, GAS_DENSITY_KG_PER_M3["0C"]) == pytest.approx(0.3954, rel=1e-3)
    assert injected_mass_g(10.0, 20.0, GAS_DENSITY_KG_PER_M3["20C"]) == pytest.approx(0.3684, rel=1e-3)


def test_free_gas_from_scale():
    # 111.9 g of displaced water (e2 cycle 1 at the end of injection) is 112.2 mL of gas = 0.210 g at 15 C
    assert free_gas_mass_g(111.9, GAS_DENSITY_KG_PER_M3["15C"]) == pytest.approx(0.2101, rel=1e-3)
    assert math.isnan(free_gas_mass_g(float("nan"), 1.872))


def test_budget_closes_with_all_terms():
    b = cycle_budget(CycleBudgetInputs("e2_c1", injected_g=0.3744, dissolved_new_lower_g=0.10,
                                       dissolved_new_upper_g=0.12, free_gas_g=0.21, left_overflow_g=0.0))
    assert b.rest_lower_g == pytest.approx(0.3744 - 0.12 - 0.21)
    assert b.rest_upper_g == pytest.approx(0.3744 - 0.10 - 0.21)
    assert b.free_gas_known and b.overflow_known
    assert b.dissolved_fraction_lower == pytest.approx(0.10 / 0.3744)


def test_budget_without_scale_marks_unknown_terms():
    b = cycle_budget(CycleBudgetInputs("um_run6_a", injected_g=4.4, dissolved_new_lower_g=3.0,
                                       dissolved_new_upper_g=3.5, quality_flags=("no overflow balance",)))
    assert not b.free_gas_known and not b.overflow_known
    assert math.isnan(b.free_gas_g)
    assert b.rest_lower_g == pytest.approx(4.4 - 3.5)
    assert b.rest_upper_g == pytest.approx(4.4 - 3.0)
    row = b.as_row()
    assert row["quality_flags"] == "no overflow balance"


def test_budget_rejects_inconsistent_bounds():
    with pytest.raises(ValueError):
        cycle_budget(CycleBudgetInputs("x", 1.0, 0.5, 0.4))
    with pytest.raises(ValueError):
        cycle_budget(CycleBudgetInputs("x", -1.0, 0.0, 0.0))


def test_budget_rows_keep_order():
    rows = budget_rows([CycleBudgetInputs("a", 1.0, 0.1, 0.2), CycleBudgetInputs("b", 2.0, 0.5, 0.5)])
    assert [r["cycle_run"] for r in rows] == ["a", "b"]
    assert rows[1]["rest_lower_g"] == pytest.approx(1.5)


def test_none_means_unknown_and_required_terms_must_be_finite():
    b = cycle_budget(CycleBudgetInputs("x", 1.0, 0.1, 0.2, free_gas_g=None, left_overflow_g=None))
    assert not b.free_gas_known and not b.overflow_known and math.isnan(b.free_gas_g)
    assert math.isnan(free_gas_mass_g(None, 1.872))
    with pytest.raises(ValueError):
        CycleBudgetInputs("x", None, 0.1, 0.2)
    with pytest.raises(ValueError):
        CycleBudgetInputs("x", float("nan"), 0.1, 0.2)


def test_infinite_terms_are_rejected():
    with pytest.raises(ValueError):
        CycleBudgetInputs("x", float("inf"), 0.1, 0.2)
    with pytest.raises(ValueError):
        CycleBudgetInputs("x", 1.0, 0.1, 0.2, free_gas_g=float("inf"))
    with pytest.raises(ValueError):
        free_gas_mass_g(float("inf"), 1.872)
    with pytest.raises(ValueError):
        free_gas_mass_g(10.0, 0.0)
