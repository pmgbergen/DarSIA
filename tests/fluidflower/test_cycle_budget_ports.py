"""Port schedules and the multi-port cycle budget.

Hand values: grams = mL x density in kg/m3 x 1e-3; at 15 C (1.872 kg/m3)
100 mL is 0.1872 g. The schedules are synthetic.
"""
import math

import pytest

from darsia.fluidflower.cycle_budget import (
    GAS_DENSITY_KG_PER_M3,
    CycleBudgetInputs,
    PortSchedule,
    RateSegment,
    cycle_budget,
    cycle_budget_from_ports,
    injected_mass_by_port_g,
    injected_mass_g,
    total_injected_mass_g,
)

# FF_UM: port A is lab port 4.6, port B lab port 9.5 (rig coordinates in metres)
UM_A = PortSchedule("A", [(0.0, 120.0, 10.0)], position_m=(0.9, 0.3))
UM_B = PortSchedule("B", [(60.0, 180.0, 10.0)], position_m=(1.9, 0.5))
# FF_BM-like ports whose rates change: 5 then 10 mL/min, a pause, then 2 mL/min
BM_1 = PortSchedule("1", [(0.0, 60.0, 5.0), (60.0, 120.0, 10.0), (150.0, 180.0, 2.0)])
BM_2 = PortSchedule("2", [RateSegment(20.0, 50.0, 3.0), RateSegment(50.0, 80.0, 6.0)])


def _budget(ports, **kwargs):
    args = dict(start_min=0.0, end_min=1440.0, reference_temperature="15C",
                dissolved_new_lower_g=1.0, dissolved_new_upper_g=1.2)
    args.update(kwargs)
    return cycle_budget_from_ports("c1", ports, **args)


def _same_row(a, b):
    def same(x, y):
        return x == y or (isinstance(x, float) and math.isnan(x) and math.isnan(y))
    return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)


@pytest.mark.parametrize("segment", [
    (10.0, 10.0, 1.0),  # zero length
    (20.0, 10.0, 1.0),  # ends before it starts
    (0.0, 10.0, -0.5),  # negative rate
    (float("nan"), 10.0, 1.0),
    (0.0, float("inf"), 1.0),
    (0.0, 10.0, float("nan")),
    (0.0, 10.0, None),
])
def test_invalid_segments_are_rejected(segment):
    with pytest.raises(ValueError):
        RateSegment(*segment)
    with pytest.raises(ValueError):
        PortSchedule("A", [segment])


def test_port_schedule_rejects_overlap_disorder_and_bad_metadata():
    with pytest.raises(ValueError, match="overlap"):
        PortSchedule("A", [(0.0, 60.0, 1.0), (30.0, 90.0, 1.0)])
    with pytest.raises(ValueError, match="overlap"):
        PortSchedule("A", [(0.0, 100.0, 1.0), (10.0, 20.0, 1.0)])  # nested
    with pytest.raises(ValueError, match="time order"):
        PortSchedule("A", [(60.0, 120.0, 1.0), (0.0, 60.0, 1.0)])
    for bad_id in ("", "  ", None, 4.6):
        with pytest.raises(ValueError):
            PortSchedule(bad_id, [(0.0, 1.0, 1.0)])
    for bad_position in ((0.9, float("nan")), (float("inf"), 0.3), (0.9, 0.3, 0.0), (0.9,)):
        with pytest.raises(ValueError):
            PortSchedule("A", [(0.0, 1.0, 1.0)], position_m=bad_position)
    # a segment may start where the previous ends; zero rates and unused ports are valid
    port = PortSchedule("A", [(0.0, 60.0, 5.0), (60.0, 90.0, 0.0), (90.0, 120.0, 10.0)])
    assert all(isinstance(s, RateSegment) for s in port.segments)
    assert UM_A.position_m == (0.9, 0.3) and BM_1.position_m is None
    unused = PortSchedule("B", [])
    assert total_injected_mass_g([unused], 100.0, reference_temperature="15C") == 0.0


@pytest.mark.parametrize("key", list(GAS_DENSITY_KG_PER_M3))
def test_single_port_reproduces_injected_mass_g_exactly(key):
    old = injected_mass_g(10.0, 20.0, GAS_DENSITY_KG_PER_M3[key])
    port = PortSchedule("p2", [(0.0, 20.0, 10.0)], position_m=(0.658, 0.155))
    assert total_injected_mass_g([port], 20.0, reference_temperature=key) == old
    assert total_injected_mass_g([port], 1440.0, reference_temperature=key) == old
    assert injected_mass_by_port_g([port], 20.0, reference_temperature=key) == {"p2": old}
    # a later cycle, injected 24 h after t = 0, in its own window
    later = PortSchedule("p2", [(1440.0, 1460.0, 10.0)])
    assert total_injected_mass_g([later], 2880.0, reference_temperature=key,
                                 since_min=1440.0) == old


def test_single_port_budget_reproduces_cycle_budget():
    port = PortSchedule("p2", [(0.0, 20.0, 10.0)])
    new = cycle_budget_from_ports(
        "e2_c1", [port], start_min=0.0, end_min=20.0, reference_temperature="15C",
        dissolved_new_lower_g=0.10, dissolved_new_upper_g=0.12, scale_term_available=True,
        free_gas_g=0.21, left_overflow_g=0.0)
    old = cycle_budget(CycleBudgetInputs(
        "e2_c1", injected_mass_g(10.0, 20.0, GAS_DENSITY_KG_PER_M3["15C"]), 0.10, 0.12,
        free_gas_g=0.21, left_overflow_g=0.0))
    assert _same_row(new.budget.as_row(), old.as_row())
    assert new.budget.injected_g == pytest.approx(0.3744, rel=1e-12)
    assert new.scale_term_available and new.budget.free_gas_known
    assert new.injected_by_port_g == {"p2": old.injected_g}


def test_two_ports_sum_correctly():
    # at 90 min: A 10 mL/min x 90 min = 900 mL, B 10 mL/min x 30 min = 300 mL
    by_port = injected_mass_by_port_g([UM_A, UM_B], 90.0, reference_temperature="15C")
    assert list(by_port) == ["A", "B"]
    assert by_port["A"] == pytest.approx(1.6848, rel=1e-12)
    assert by_port["B"] == pytest.approx(0.5616, rel=1e-12)
    total = total_injected_mass_g([UM_A, UM_B], 90.0, reference_temperature="15C")
    assert total == pytest.approx(2.2464, rel=1e-12)  # 1200 mL
    # after both ends: 1200 mL each
    total = total_injected_mass_g([UM_A, UM_B], 1440.0, reference_temperature="15C")
    assert total == pytest.approx(4.4928, rel=1e-12)


@pytest.mark.parametrize("t_min, grams", [
    (-10.0, 0.0),  # before the first segment
    (0.0, 0.0),  # at its start
    (30.0, 0.2808),  # inside: 5 mL/min x 30 min = 150 mL
    (60.0, 0.5616),  # at the rate change: 300 mL
    (90.0, 1.1232),  # inside the 10 mL/min segment: 300 + 300 mL
    (120.0, 1.6848),  # at its end: 900 mL
    (135.0, 1.6848),  # in the pause
    (150.0, 1.6848),  # at the start of the last segment
    (165.0, 1.74096),  # inside it: 900 + 2 x 15 = 930 mL
    (180.0, 1.79712),  # at the end: 960 mL
    (1440.0, 1.79712),  # long after the end
])
def test_time_varying_rates_integrate_piecewise(t_min, grams):
    got = total_injected_mass_g([BM_1], t_min, reference_temperature="15C")
    assert got == pytest.approx(grams, rel=1e-12)


def test_window_and_reference_temperature():
    # from 30 to 165 min: 5 x 30 + 10 x 60 + 2 x 15 = 780 mL
    assert total_injected_mass_g([BM_1], 165.0, reference_temperature="15C", since_min=30.0) \
        == pytest.approx(1.46016, rel=1e-12)
    assert total_injected_mass_g([BM_1], 165.0, reference_temperature="15C",
                                 since_min=165.0) == 0.0
    with pytest.raises(ValueError):
        total_injected_mass_g([BM_1], 30.0, reference_temperature="15C", since_min=165.0)
    # 960 mL at the other reference temperatures of the density table
    assert total_injected_mass_g([BM_1], 180.0, reference_temperature="0C") \
        == pytest.approx(1.89792, rel=1e-12)
    assert total_injected_mass_g([BM_1], 180.0, reference_temperature="25C") \
        == pytest.approx(1.73856, rel=1e-12)
    with pytest.raises(TypeError):
        total_injected_mass_g([BM_1], 180.0)  # no default reference temperature
    for bad in ("17C", 15, None):
        with pytest.raises(ValueError):
            total_injected_mass_g([BM_1], 180.0, reference_temperature=bad)


def test_absent_scale_term_is_nan_with_flag():
    b = _budget([UM_A, UM_B], scale_term_available=False,
                quality_flags=("no overflow balance",))
    assert b.scale_term_available is False
    assert math.isnan(b.budget.free_gas_g) and math.isnan(b.budget.left_overflow_g)
    assert not b.budget.free_gas_known and not b.budget.overflow_known
    assert b.budget.injected_g == pytest.approx(4.4928, rel=1e-12)
    # the rest holds the unknown terms, as in cycle_budget
    assert b.budget.rest_lower_g == pytest.approx(4.4928 - 1.2, rel=1e-12)
    assert b.budget.rest_upper_g == pytest.approx(4.4928 - 1.0, rel=1e-12)
    row = b.as_row()
    assert row["scale_term_available"] is False and math.isnan(row["free_gas_g"])
    assert math.isnan(row["left_overflow_g"]) and row["quality_flags"] == "no overflow balance"
    assert row["injected_A_g"] == pytest.approx(2.2464, rel=1e-12)
    assert row["injected_B_g"] == pytest.approx(2.2464, rel=1e-12)
    assert row["reference_temperature"] == "15C"
    assert (row["start_min"], row["end_min"]) == (0.0, 1440.0)
    assert set(cycle_budget(CycleBudgetInputs("c1", 1.0, 0.1, 0.2)).as_row()) <= set(row)


@pytest.mark.parametrize("term", ["free_gas_g", "left_overflow_g"])
def test_absent_scale_term_is_never_zero(term):
    for value in (0.0, 0.21, float("inf")):
        with pytest.raises(ValueError):
            _budget([UM_A, UM_B], scale_term_available=False, **{term: value})
    for absent in (None, float("nan")):
        b = _budget([UM_A, UM_B], scale_term_available=False, **{term: absent})
        assert math.isnan(getattr(b.budget, term))


def test_scale_available_keeps_the_existing_term_rules():
    b = _budget([UM_A], scale_term_available=True)  # no scale record for this cycle
    assert b.scale_term_available and not b.budget.free_gas_known
    assert math.isnan(b.budget.free_gas_g)
    assert _budget([UM_A], scale_term_available=True, free_gas_g=0.0).budget.free_gas_known
    with pytest.raises(ValueError):
        _budget([UM_A], scale_term_available=True, free_gas_g=float("inf"))
    with pytest.raises(ValueError):
        _budget([UM_A], scale_term_available=True, dissolved_new_lower_g=None)
    with pytest.raises(ValueError):
        _budget([UM_A], scale_term_available=True, dissolved_new_lower_g=1.3)  # above upper
    with pytest.raises(TypeError):
        _budget([UM_A], scale_term_available=None)
    with pytest.raises(TypeError):
        _budget([UM_A])  # the flag has no default


def test_conservation_total_is_the_sum_over_ports():
    ports = [BM_1, BM_2, UM_A, UM_B]
    for t_min in (-5.0, 0.0, 33.3, 60.0, 89.9, 120.0, 1e4):
        by_port = injected_mass_by_port_g(ports, t_min, reference_temperature="20C")
        total = total_injected_mass_g(ports, t_min, reference_temperature="20C")
        assert total == math.fsum(by_port.values())
        assert total == pytest.approx(sum(by_port.values()), rel=1e-12)
    # consecutive windows add up to the cumulative mass
    whole = total_injected_mass_g(ports, 165.0, reference_temperature="20C")
    first = total_injected_mass_g(ports, 70.0, reference_temperature="20C")
    second = total_injected_mass_g(ports, 165.0, reference_temperature="20C", since_min=70.0)
    assert first + second == pytest.approx(whole, rel=1e-12)
    # FF_BM cycle from 30 to 165 min: port 1 780 mL, port 2 3 x 20 + 6 x 30 = 240 mL
    b = _budget([BM_1, BM_2], start_min=30.0, end_min=165.0, scale_term_available=True)
    assert b.injected_by_port_g["1"] == pytest.approx(1.46016, rel=1e-12)
    assert b.injected_by_port_g["2"] == pytest.approx(0.44928, rel=1e-12)
    assert b.budget.injected_g == math.fsum(b.injected_by_port_g.values())
    assert b.budget.injected_g == pytest.approx(1.90944, rel=1e-12)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), None])
def test_non_finite_times_are_rejected(bad):
    with pytest.raises(ValueError):
        total_injected_mass_g([BM_1], bad, reference_temperature="15C")
    with pytest.raises(ValueError):
        _budget([BM_1], start_min=bad, scale_term_available=True)
    with pytest.raises(ValueError):
        _budget([BM_1], end_min=bad, scale_term_available=True)
    if bad is not None:  # None means from the first segment
        with pytest.raises(ValueError):
            total_injected_mass_g([BM_1], 100.0, reference_temperature="15C", since_min=bad)


def test_port_sets_must_be_non_empty_and_unique():
    with pytest.raises(ValueError):
        total_injected_mass_g([], 10.0, reference_temperature="15C")
    with pytest.raises(ValueError):
        injected_mass_by_port_g([UM_A, PortSchedule("A", [])], 10.0,
                                reference_temperature="15C")
    with pytest.raises(TypeError):
        total_injected_mass_g([(0.0, 10.0, 1.0)], 10.0, reference_temperature="15C")
    with pytest.raises(ValueError):
        _budget([BM_1], start_min=200.0, end_min=100.0, scale_term_available=True)
