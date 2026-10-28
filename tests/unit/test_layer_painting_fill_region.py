"""Vector gap fills preserve neighbours and never change their inputs."""

import copy

import pytest
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from darsia.fluidflower.layer_painting_fill_region import (
    FillRegionError,
    fill_region,
)


def rectangle(top, left, bottom, right):
    return [(top, left), (top, right), (bottom, right), (bottom, left)]


FRAME = rectangle(0, 0, 30, 40)


def _checked(plan, before, frame):
    result = Polygon(plan.polygon_points)
    assert result.is_valid and not result.interiors
    assert Polygon(frame).covers(result)
    assert result.area == pytest.approx(plan.area_px2)
    covered = unary_union([Polygon(points) for points in before.values()])
    assert result.intersection(covered).area == pytest.approx(0)
    for update in plan.updates:
        assert update.before == tuple(before[update.index])
        assert Polygon(update.before).equals(Polygon(update.after))
        # Every original point, including a repeated one, retains its order.
        remaining = iter(update.after)
        for point in update.before:
            assert any(candidate == point for candidate in remaining)
    return result


def test_gap_between_layers_and_frame_is_exact_and_keeps_collinear_points():
    above = [(0, 0), (0, 40), (10, 40), (10, 23), (10, 7), (10, 0)]
    below = [(20, 0), (20, 13), (20, 40), (30, 40), (30, 0)]
    polygons = {2: above, 8: below}
    before = copy.deepcopy((polygons, FRAME))
    plan = fill_region(polygons, FRAME, (15, 15), layer_id=3)
    assert plan.layer_id == 3
    result = _checked(plan, polygons, FRAME)
    assert result.equals(Polygon(rectangle(10, 0, 20, 40)))
    assert all(point in plan.polygon_points for point in [(10, 23), (10, 7), (20, 13)])
    assert not plan.updates
    assert (polygons, FRAME) == before


def test_nonrectangular_frame_edges_and_raw_half_pixel_origin_are_preserved():
    frame = [(-.5, -.5), (1.5, 19.5), (15.5, 16.5), (17.5, 2.5)]
    plan = fill_region({}, frame, (8, 8), layer_id=1)
    assert set(plan.polygon_points) == set(frame)
    assert _checked(plan, {}, frame).equals(Polygon(frame))
    assert not plan.updates


def test_only_the_clicked_disconnected_gap_is_filled():
    polygons = {0: rectangle(0, 15, 30, 20)}
    left = fill_region(polygons, FRAME, (5, 5), layer_id=2)
    right = fill_region(polygons, FRAME, (5, 25), layer_id=2)
    assert _checked(left, polygons, FRAME).equals(Polygon(rectangle(0, 0, 30, 15)))
    assert _checked(right, polygons, FRAME).equals(Polygon(rectangle(0, 20, 30, 40)))
    assert set(left.polygon_points).isdisjoint({(0, 20), (30, 20)})


def test_new_frame_intersections_insert_shared_points_without_moving_neighbour():
    # A donor crosses the frame. Its intersection with the frame is a new
    # vertex in the gap, and must also become a vertex of the original donor.
    donor = [(-5, 20), (15, 10), (35, 20), (35, 45), (-5, 45)]
    polygons = {7: donor}
    before = copy.deepcopy(polygons)
    plan = fill_region(polygons, FRAME, (15, 4), layer_id=2)
    _checked(plan, polygons, FRAME)
    assert len(plan.updates) == 1 and plan.updates[0].index == 7
    inserted = set(plan.updates[0].after) - set(donor)
    assert inserted == {(0, 17.5), (30, 17.5)}
    assert inserted.issubset(set(plan.polygon_points))
    assert polygons == before


def test_cut_is_a_barrier_and_its_endpoints_are_shared_with_both_neighbours():
    polygons = {0: rectangle(0, 0, 10, 40), 1: rectangle(20, 0, 30, 40)}
    cut = [(8, 17), (22, 17)]
    original = copy.deepcopy((polygons, FRAME, cut))
    plan = fill_region(polygons, FRAME, (15, 30), layer_id=4, cut_line=cut)
    assert _checked(plan, polygons, FRAME).equals(Polygon(rectangle(10, 17, 20, 40)))
    assert len(plan.updates) == 2
    for update, endpoint in zip(plan.updates, [(10, 17), (20, 17)]):
        assert endpoint in update.after and endpoint in plan.polygon_points
    assert (polygons, FRAME, cut) == original


def test_cut_follows_a_polyline_and_is_not_replaced_by_a_box():
    cut = [(-1, 10), (12, 10), (18, 20), (31, 20)]
    plan = fill_region({}, FRAME, (10, 30), layer_id=1, cut_line=cut)
    region = _checked(plan, {}, FRAME)
    assert region.equals(Polygon([(0, 10), (12, 10), (18, 20), (30, 20), (30, 40), (0, 40)]))
    assert region.area != region.envelope.area
    assert (12, 10) in plan.polygon_points and (18, 20) in plan.polygon_points


def test_cut_is_restricted_to_the_clicked_gap():
    polygons = {0: rectangle(0, 17, 30, 22)}
    plan = fill_region(polygons, FRAME, (20, 30), layer_id=1,
                       cut_line=[(12, -5), (12, 45)])
    assert _checked(plan, polygons, FRAME).equals(Polygon(rectangle(12, 22, 30, 40)))


def test_hole_is_refused_instead_of_covering_a_smaller_island():
    polygons = {0: rectangle(10, 10, 20, 20), 1: rectangle(12, 12, 15, 15)}
    original = copy.deepcopy(polygons)
    with pytest.raises(FillRegionError, match="enclosed polygon"):
        fill_region(polygons, FRAME, (5, 5), layer_id=2)
    assert polygons == original


def test_cut_across_an_island_can_select_an_exact_hole_free_part():
    polygons = {0: rectangle(10, 10, 20, 20)}
    plan = fill_region(polygons, FRAME, (5, 5), layer_id=2,
                       cut_line=[(15, -1), (15, 41)])
    region = _checked(plan, polygons, FRAME)
    expected = Polygon([(0, 0), (0, 40), (15, 40), (15, 20),
                        (10, 20), (10, 10), (15, 10), (15, 0)])
    assert region.equals(expected)
    assert len(plan.updates) == 1
    assert {(15, 10), (15, 20)}.issubset(plan.updates[0].after)


def test_overlaps_and_nested_polygons_are_all_covered_not_donors_to_move():
    polygons = {0: rectangle(0, 0, 12, 40), 2: rectangle(5, 5, 15, 30),
                4: rectangle(6, 7, 8, 10)}
    plan = fill_region(polygons, FRAME, (22, 5), layer_id=1)
    result = _checked(plan, polygons, FRAME)
    expected = Polygon(FRAME).difference(unary_union([Polygon(p) for p in polygons.values()]))
    assert result.equals(expected)
    assert Point(7, 8).within(result) is False


@pytest.mark.parametrize("point", [(-1, 3), (0, 3), (31, 3), (3, 40)])
def test_outside_and_frame_boundary_clicks_are_refused(point):
    with pytest.raises(FillRegionError, match="inside the rig frame"):
        fill_region({}, FRAME, point, layer_id=1)


@pytest.mark.parametrize("point", [(5, 5), (10, 5)])
def test_covered_or_donor_boundary_click_is_refused(point):
    with pytest.raises(FillRegionError, match="already covered"):
        fill_region({0: rectangle(0, 0, 10, 40)}, FRAME, point, layer_id=1)


@pytest.mark.parametrize("cut", [[(10, 10), (20, 10)], [(-1, 10), (20, 10)],
                                  [(-1, 50), (31, 50)]])
def test_dangling_or_irrelevant_cut_does_not_silently_fill_the_whole_gap(cut):
    with pytest.raises(FillRegionError, match="does not divide"):
        fill_region({}, FRAME, (15, 30), layer_id=1, cut_line=cut)


def test_click_on_the_cut_is_refused():
    with pytest.raises(FillRegionError, match="away from the cut"):
        fill_region({}, FRAME, (10, 10), layer_id=1, cut_line=[(-1, 10), (31, 10)])


@pytest.mark.parametrize("cut", [[], [(1, 1)], [(1, 1), (1, 1)],
                                  [(1, 1), (5, 5), (1, 5), (5, 1)],
                                  [(1, 1), (1, 5), (5, 5), (1, 1)]])
def test_invalid_cut_is_refused(cut):
    with pytest.raises(FillRegionError, match="cut line|Cut line"):
        fill_region({}, FRAME, (15, 30), layer_id=1, cut_line=cut)


@pytest.mark.parametrize("bad", [[], [(0, 0), (1, 1)],
                                  [(0, 0), (10, 10), (0, 10), (10, 0)],
                                  [(0, 0), (1, float("nan")), (2, 0)]])
@pytest.mark.parametrize("which", ["frame", "donor"])
def test_invalid_outlines_are_refused_without_repair(bad, which):
    polygons, frame = ({0: bad}, FRAME) if which == "donor" else ({}, bad)
    original = copy.deepcopy((polygons, frame))
    with pytest.raises(FillRegionError):
        fill_region(polygons, frame, (15, 30), layer_id=1)
    assert repr((polygons, frame)) == repr(original)


@pytest.mark.parametrize("layer_id", [0, -1, True, 1.0, "1"])
def test_layer_must_be_an_actual_selected_layer(layer_id):
    with pytest.raises(FillRegionError, match="Choose a layer"):
        fill_region({}, FRAME, (15, 30), layer_id=layer_id)


def test_repeated_existing_vertex_is_preserved_and_plan_is_immutable():
    donor = [(0, 0), (0, 40), (10, 40), (10, 40), (10, 0)]
    plan = fill_region({3: donor}, FRAME, (15, 30), layer_id=2,
                       cut_line=[(8, 17), (31, 17)])
    _checked(plan, {3: donor}, FRAME)
    assert plan.updates[0].after.count((10, 40)) == 2
    with pytest.raises(AttributeError):
        plan.layer_id = 8


def test_small_real_gap_is_not_closed_by_a_pixel_tolerance():
    width = .0001
    polygons = {0: rectangle(0, 0, 10, 40), 1: rectangle(10 + width, 0, 30, 40)}
    plan = fill_region(polygons, FRAME, (10 + width / 2, 20), layer_id=3)
    assert _checked(plan, polygons, FRAME).area == pytest.approx(width * 40)
    assert not plan.updates


def test_result_boundaries_include_the_exact_shared_cut_endpoints():
    polygons = {0: [(-5, 0), (8, 0), (18, 40), (-5, 40)]}
    cut = [(0, 12), (35, 12)]
    plan = fill_region(polygons, FRAME, (25, 20), layer_id=2, cut_line=cut)
    region = _checked(plan, polygons, FRAME)
    assert (11, 12) in plan.polygon_points
    assert (11, 12) in plan.updates[0].after
    assert region.boundary.intersection(LineString(cut)).length == pytest.approx(19)
