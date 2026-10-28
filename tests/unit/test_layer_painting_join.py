"""Joining an outline never moves the neighbour or mutates an input."""

import copy
import random

import pytest
from shapely.geometry import LineString, Polygon

from darsia.fluidflower import layer_painting_join as join
from darsia.fluidflower.layer_painting_join import (
    JoinEdgesError,
    JoinedStretch,
    JoinPreview,
    join_stretches,
    join_to_neighbour,
    verified_working_distance,
)

# A thin layer whose top dips to 3 px above its bottom, 2 px above the
# polygon below: from about 5.5 px the dip is joined too and the result
# would cross itself along the shared edge.
THIN = [(10, 5), (10, 28), (22, 40), (10, 52), (10, 75), (25, 75), (25, 5)]
BELOW = [(27, 2), (27, 78), (55, 78), (55, 2)]
DIP = (22.0, 40.0)


def rectangle(top, left, bottom, right):
    return [(top, left), (top, right), (bottom, right), (bottom, left)]


def _works(target, master, distance):
    """Whether the full join there joins a stretch without a refusal."""
    try:
        return bool(join_to_neighbour(target, master, distance).stretches)
    except JoinEdgesError:
        return False


@pytest.mark.parametrize("bottom", [8.0, 12.0])
def test_gap_and_overlap_follow_the_master_without_moving_it(bottom):
    target = rectangle(0, 0, bottom, 20)
    master = [(10, 0), (10, 8), (10, 15), (10, 20), (30, 20), (30, 0)]
    original = copy.deepcopy((target, master))
    plan = join_to_neighbour(target, master, 3)
    assert (target, master) == original
    assert len(plan.stretches) == 1
    assert Polygon(plan.target).is_valid
    assert Polygon(plan.target).intersection(Polygon(master)).area == 0
    assert LineString([*plan.target, plan.target[0]]).intersection(
        LineString([*master, master[0]])
    ).length == pytest.approx(20)
    assert tuple(master) == plan.neighbour
    assert (10, 8) in plan.target and (10, 15) in plan.target


def test_mid_edge_endpoints_are_inserted_in_original_order_at_most_twice():
    target = rectangle(0, 6, 8, 14)
    master = rectangle(10, 0, 30, 20)
    plan = join_to_neighbour(target, master, 3)
    assert len(plan.neighbour) == len(master) + 2
    assert [p for p in plan.neighbour if p in master] == master
    assert plan.neighbour[:4] == ((10, 0), (10, 6), (10, 14), (10, 20))
    assert Polygon(plan.neighbour).equals(Polygon(master))
    assert (10, 6) in plan.target and (10, 14) in plan.target


def test_far_outline_and_zero_distance_leave_every_point_unchanged():
    target, master = rectangle(0, 0, 8, 20), rectangle(20, 0, 40, 20)
    for distance in (0, 5):
        plan = join_to_neighbour(target, master, distance)
        assert plan.target == tuple(target)
        assert plan.neighbour == tuple(master)
        assert not plan.stretches


def test_already_shared_outline_is_not_an_edit():
    target, master = rectangle(0, 0, 10, 20), rectangle(10, 0, 30, 20)
    plan = join_to_neighbour(target, master, 3)
    assert not plan.stretches
    assert plan.target == tuple(target)
    assert plan.neighbour == tuple(master)


def test_a_join_crossing_the_first_vertex_is_one_continuous_stretch():
    target = [(8, 0), (0, 0), (0, 20), (8, 20)]
    master = rectangle(10, 0, 30, 20)
    plan = join_to_neighbour(target, master, 3)
    assert len(plan.stretches) == 1
    assert Polygon(plan.target).is_valid
    assert Polygon(plan.target).intersection(Polygon(master)).area == 0


def test_cannot_collapse_the_whole_polygon_onto_the_neighbour():
    with pytest.raises(JoinEdgesError, match="whole outline"):
        join_to_neighbour(rectangle(0, 0, 1, 1), rectangle(0, 0, 2, 2), 10)


@pytest.mark.parametrize("which", [0, 1])
def test_self_intersection_is_refused_with_its_location(which):
    inputs = [rectangle(0, 0, 8, 20), rectangle(10, 0, 30, 20)]
    inputs[which] = [(0, 0), (10, 10), (0, 10), (10, 0)]
    before = copy.deepcopy(inputs)
    with pytest.raises(JoinEdgesError, match="Self-intersection") as caught:
        join_to_neighbour(*inputs, 3)
    assert caught.value.location == (5, 5)
    assert inputs == before


@pytest.mark.parametrize("distance", [-1, float("nan"), float("inf")])
def test_invalid_distance_is_refused(distance):
    with pytest.raises(JoinEdgesError, match="finite"):
        join_to_neighbour(rectangle(0, 0, 8, 20), rectangle(10, 0, 30, 20), distance)


def test_join_that_would_cross_another_part_of_the_target_is_refused():
    target = [(7.28, 15.31), (10.27, 9.01), (9.92, 1.37), (18.61, -1.35),
              (-7.14, -8.10), (-14.97, -3.25), (-15.70, 6.53), (-1.93, 1.09),
              (-12.19, 7.42), (-11.89, 10.04), (-6.28, 11.96), (-0.84, 6.84)]
    master = [(-30, 10), (30, 10), (30, 40), (-30, 40)]
    assert Polygon(target).is_valid and Polygon(master).is_valid
    before = copy.deepcopy((target, master))
    with pytest.raises(JoinEdgesError, match="Joined polygon: Self-intersection") as caught:
        join_to_neighbour(target, master, 7)
    assert caught.value.location is not None
    assert (target, master) == before


def test_original_master_vertices_are_retained_even_if_two_coincide():
    master = [(10, 0), (10, 0), (10, 20), (30, 20), (30, 0)]
    plan = join_to_neighbour(rectangle(0, 6, 8, 14), master, 3)
    assert [p for p in plan.neighbour if p in master] == master
    assert len(plan.neighbour) == len(master) + 2


def test_transverse_crossing_is_not_collapsed_into_one_shared_point():
    # Almost perpendicular outlines: their projections are only 0.12 px
    # apart, below the existing 0.5 px shared-point tolerance.
    target = [(0, 0), (0, 100), (10, 100), (10, 0)]
    master = [(-10, 40), (30, 40.2), (30, 60.2), (-10, 60)]
    plan = join_to_neighbour(target, master, 3)
    assert not plan.stretches
    assert plan.target == tuple(target)
    assert plan.neighbour == tuple(master)


def test_refusals_name_the_refused_outline_and_its_reason():
    crossed = [(0, 0), (10, 10), (0, 10), (10, 0)]
    with pytest.raises(JoinEdgesError) as caught:
        join_to_neighbour(crossed, rectangle(10, 0, 30, 20), 3)
    assert caught.value.outline == "Selected polygon"
    assert caught.value.reason.startswith("Self-intersection")
    with pytest.raises(JoinEdgesError) as caught:
        join_to_neighbour(THIN, BELOW, 8)
    assert caught.value.outline == "Joined polygon"
    assert "Self-intersection" in caught.value.reason
    assert caught.value.location is not None
    with pytest.raises(JoinEdgesError, match="whole outline") as caught:
        join_to_neighbour(rectangle(0, 0, 1, 1), rectangle(0, 0, 2, 2), 10)
    assert caught.value.outline is None and caught.value.reason is None


def test_the_stretches_of_a_refused_join_are_listed_in_preview_order():
    before = copy.deepcopy((THIN, BELOW))
    listed = join_stretches(THIN, BELOW, 8)
    assert (THIN, BELOW) == before
    assert len(listed) == 2 and all(isinstance(row, JoinedStretch) for row in listed)
    assert [DIP in row.before for row in listed].count(True) == 1
    assert join_stretches(THIN, BELOW, 4) == join_to_neighbour(THIN, BELOW, 4).stretches
    with pytest.raises(JoinEdgesError, match="whole outline"):
        join_stretches(rectangle(0, 0, 1, 1), rectangle(0, 0, 2, 2), 10)


def test_a_subset_is_joined_alone_or_refused_by_the_same_check():
    listed = join_stretches(THIN, BELOW, 8)
    dip = next(number for number, row in enumerate(listed) if DIP in row.before)
    bottom = 1 - dip
    plan = join_to_neighbour(THIN, BELOW, 8, include={bottom})
    assert plan.stretches == (listed[bottom],)
    assert DIP in plan.target
    assert Polygon(plan.target).is_valid
    assert Polygon(plan.target).intersection(Polygon(BELOW)).area == pytest.approx(0.0)
    assert len(plan.neighbour) == len(BELOW) + 2
    with pytest.raises(JoinEdgesError, match="Joined polygon: Self-intersection") as caught:
        join_to_neighbour(THIN, BELOW, 8, include={dip})
    assert caught.value.location is not None
    nothing = join_to_neighbour(THIN, BELOW, 8, include=())
    assert nothing.stretches == ()
    assert nothing.target == tuple(THIN) and nothing.neighbour == tuple(BELOW)


def test_each_stretch_of_a_notched_outline_can_be_joined_alone():
    notched = [(5, 15), (5, 50), (28, 50), (28, 38), (20, 38), (20, 27), (28, 27), (28, 15)]
    master = rectangle(30, 5, 55, 70)
    listed = join_stretches(notched, master, 5)
    both = join_to_neighbour(notched, master, 5)
    assert len(listed) == 2 and both.stretches == listed
    for number in range(2):
        plan = join_to_neighbour(notched, master, 5, include=[number])
        assert plan.stretches == (listed[number],)
        assert Polygon(plan.target).is_valid
        assert len(plan.neighbour) == len(master) + 2
        assert plan.target != both.target


@pytest.mark.parametrize("include", [{2}, {-1}, {True}, {0.0}, ["0"]])
def test_unknown_stretch_numbers_are_refused(include):
    with pytest.raises(JoinEdgesError, match="not one of the 2 stretches"):
        join_to_neighbour(THIN, BELOW, 8, include=include)


def test_a_refused_distance_gets_a_verified_working_distance_next_to_a_failing_one():
    assert not _works(THIN, BELOW, 8)
    found = verified_working_distance(THIN, BELOW, 8, 0.05)
    assert found is not None and found < 8
    assert _works(THIN, BELOW, found)
    assert abs(found / 0.05 - round(found / 0.05)) < 1e-9
    assert not _works(THIN, BELOW, found + 0.05)
    plan = join_to_neighbour(THIN, BELOW, found)
    assert len(plan.stretches) == 1 and DIP in plan.target


def test_a_working_chosen_distance_is_returned_as_it_is():
    assert verified_working_distance(THIN, BELOW, 4.0, 0.05) == 4.0


@pytest.mark.parametrize(
    "target, master, distance",
    [
        # Refused from 2 px, and joins nothing below: no distance works.
        ([(10, 10), (10, 10.6), (10.5, 10.3)], rectangle(12, 0, 30, 30), 5.0),
        (rectangle(0, 0, 8, 20), rectangle(20, 0, 40, 20), 5.0),
        (THIN, BELOW, 0.04),
    ],
)
def test_no_working_distance_gives_none(target, master, distance):
    assert not _works(target, master, distance)
    assert verified_working_distance(target, master, distance, 0.05) is None


@pytest.mark.parametrize("tolerance", [0.0, -0.05, float("nan"), float("inf")])
def test_invalid_tolerance_is_refused(tolerance):
    with pytest.raises(JoinEdgesError, match="Tolerance"):
        verified_working_distance(THIN, BELOW, 8, tolerance)


def test_invalid_polygons_are_refused_before_any_search():
    with pytest.raises(JoinEdgesError, match="Selected polygon"):
        verified_working_distance([(0, 0), (10, 10), (0, 10), (10, 0)], BELOW, 8, 0.05)


def test_real_joins_never_yield_a_failing_distance():
    notched = [(5, 15), (5, 50), (28, 50), (28, 38), (20, 38), (20, 27), (28, 27), (28, 15)]
    pairs = [
        (THIN, BELOW),
        (notched, rectangle(30, 5, 55, 70)),
        (rectangle(0, 0, 8, 20), [(10, 0), (10, 8), (10, 15), (10, 20), (30, 20), (30, 0)]),
        ([(10, 10), (10, 10.6), (10.5, 10.3)], rectangle(12, 0, 30, 30)),
        (rectangle(0, 0, 1, 1), rectangle(0, 0, 2, 2)),
    ]
    offered = 0
    for target, master in pairs:
        for chosen in (1.0, 3.0, 5.6, 8.0, 12.0, 20.0):
            found = verified_working_distance(target, master, chosen, 0.25)
            if found is not None:
                offered += 1
                assert found <= chosen and _works(target, master, found)
    assert offered >= 10


def _fake_join(pattern, calls):
    """A stand-in join whose outcome at each distance is ``pattern(distance)``."""

    def fake(target, master, distance, include=None):
        calls.append(distance)
        outcome = pattern(distance)
        if outcome == "refused":
            raise JoinEdgesError("Joined polygon: Self-intersection[1 1]", (1.0, 1.0))
        stretch = JoinedStretch(((0.0, 0.0), (0.0, 1.0)), ((1.0, 0.0), (1.0, 1.0)), 1.0)
        stretches = (stretch,) if outcome == "works" else ()
        return JoinPreview(tuple(target), tuple(master), stretches)

    return fake


def test_non_monotone_outcomes_give_a_checked_working_distance(monkeypatch):
    # Works on [2, 3) and again on [6, 6.5): the search may end next to
    # either window, which is why its result is never called the largest.
    def pattern(distance):
        if distance < 2.0:
            return "nothing"
        if distance < 3.0 or 6.0 <= distance < 6.5:
            return "works"
        return "refused"

    calls = []
    monkeypatch.setattr(join, "join_to_neighbour", _fake_join(pattern, calls))
    target, master = rectangle(0, 0, 8, 20), rectangle(10, 0, 30, 20)
    found = verified_working_distance(target, master, 10.0, 0.05)
    assert found is not None and pattern(found) == "works"
    assert pattern(found + 0.05) != "works"
    assert found < 6.0  # a working distance, though a larger one works too
    assert calls[-1] == found  # checked once more just before it is returned


def test_random_outcome_patterns_never_yield_a_failing_distance(monkeypatch):
    rng = random.Random(4321)
    target, master = rectangle(0, 0, 8, 20), rectangle(10, 0, 30, 20)
    offered = 0
    for _ in range(300):
        edges = sorted(rng.uniform(0.0, 12.0) for _ in range(rng.randint(1, 8)))
        states = [rng.choice(("works", "nothing", "refused")) for _ in range(len(edges) + 1)]

        def pattern(distance, edges=edges, states=states):
            return states[sum(distance >= edge for edge in edges)]

        calls = []
        monkeypatch.setattr(join, "join_to_neighbour", _fake_join(pattern, calls))
        chosen = rng.uniform(0.0, 12.0)
        tolerance = rng.choice((0.05, 0.1, 0.37))
        found = verified_working_distance(target, master, chosen, tolerance)
        if found is not None:
            offered += 1
            assert found <= chosen and pattern(found) == "works"
            assert calls[-1] == found
    assert offered >= 100


def test_a_distance_that_fails_its_final_check_is_not_returned(monkeypatch):
    tried = set()

    def pattern(distance):
        # Below 3 a distance works only the first time it is tried.
        if distance >= 3.0 or distance in tried:
            return "refused"
        tried.add(distance)
        return "works"

    monkeypatch.setattr(join, "join_to_neighbour", _fake_join(pattern, []))
    target, master = rectangle(0, 0, 8, 20), rectangle(10, 0, 30, 20)
    assert verified_working_distance(target, master, 5.0, 0.05) is None
    assert len(tried) > 1


def test_a_cancelled_search_stops_before_its_next_join(monkeypatch):
    calls = []
    refused = _fake_join(lambda distance: "refused", calls)
    monkeypatch.setattr(join, "join_to_neighbour", refused)
    target, master = rectangle(0, 0, 8, 20), rectangle(10, 0, 30, 20)
    assert verified_working_distance(target, master, 5.0, 0.05, cancelled=lambda: True) is None
    assert calls == []
    answers = iter([False, False, True])
    assert verified_working_distance(
        target, master, 5.0, 0.05, cancelled=lambda: next(answers, True)
    ) is None
    assert len(calls) == 2
