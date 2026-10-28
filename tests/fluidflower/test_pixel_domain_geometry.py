"""Physical rig, observation and column-donor rules on small explicit grids."""

import numpy as np
import pytest

from darsia.fluidflower.geometry.pixels import (
    column_co2_donors,
    extend_column_geometry,
    polygon_pixel_domain,
    rig_pixel_domain,
    visible_pixels,
)


def _heights(shape, values=None):
    if values is None:
        values = np.arange(shape[0] - 1, -1, -1, dtype=np.float64)
    return np.broadcast_to(np.asarray(values)[:, None], shape).copy()


def test_native_polygons_use_one_origin_translation_and_original_rasterizer():
    """Native negative coordinates reach top/bottom rows without a photo cut."""
    import cv2

    inner = np.asarray([[-2., -1.], [2., -1.], [2., 2.], [-2., 2.]])
    fitting = np.asarray([[-1., 0.], [0., 0.], [0., 1.], [-1., 1.]])
    domain, inside, fixtures = polygon_pixel_domain(
        inner, {"injector": fitting}, (4, 5), native_origin_xy=(-2, -1),
        raster_scale=16, raster_shift=4)
    expected = np.zeros((4, 5), np.uint8)
    cv2.fillPoly(expected, [np.rint((fitting + (2, 1)) * 16).astype(np.int32)],
                 1, shift=4)
    assert inside.all()
    assert domain[0].all() and domain[-1].all()
    np.testing.assert_array_equal(fixtures["injector"], expected.astype(bool))
    np.testing.assert_array_equal(domain, ~expected.astype(bool))
    np.testing.assert_array_equal(inner[0], [-2., -1.])
    empty, _, _ = polygon_pixel_domain(
        inner, {}, (4, 5), native_origin_xy=(-2, -1),
        raster_scale=16, raster_shift=4)
    assert empty.all()


@pytest.mark.parametrize("change", [
    {"shape_hw": (0, 5)}, {"native_origin_xy": (np.nan, 0)},
    {"raster_scale": 2}, {"raster_shift": 17},
    {"inner_polygon": [[0, 0], [1, 1]]},
    {"fixture_polygons": {"": [[0, 0], [1, 0], [1, 1]]}},
])
def test_polygon_domain_rejects_missing_or_ambiguous_geometry(change):
    values = dict(inner_polygon=[[0, 0], [2, 0], [2, 2]], fixture_polygons={},
                  shape_hw=(3, 3), native_origin_xy=(0, 0),
                  raster_scale=16, raster_shift=4)
    values.update(change)
    with pytest.raises(ValueError):
        polygon_pixel_domain(**values)


def test_domain_is_only_rig_interior_minus_all_hardware():
    inside = np.array([[False, True, True], [True, True, True]], bool)
    fitting = np.array([[True, True, False], [False, False, False]], bool)
    chart = np.array([[False, True, False], [False, True, False]], bool)
    original = inside.copy()
    domain = rig_pixel_domain(inside, {"fitting": fitting, "chart": chart})
    np.testing.assert_array_equal(domain, [[False, False, True], [True, False, True]])
    np.testing.assert_array_equal(inside, original)
    assert not np.shares_memory(domain, inside)
    np.testing.assert_array_equal(rig_pixel_domain(inside, {}), inside)


def test_visibility_uses_actual_source_validity_even_outside_old_photo_sand():
    domain = np.array([[True, True, False], [True, True, True]], bool)
    source = np.array([[False, True, True], [True, False, True]], bool)
    np.testing.assert_array_equal(
        visible_pixels(domain, source), [[False, True, False], [True, False, True]]
    )


def test_nearest_facies_uses_physical_column_distance_and_keeps_originals():
    facies = np.zeros((5, 2), np.int16)
    facies[0, 0], facies[4, 0], facies[1, 1] = 1, 2, 3
    defined = facies != 0
    domain = np.ones(facies.shape, bool)
    before = facies.copy()
    result = extend_column_geometry(
        facies, defined, domain, _heights(facies.shape, [.10, .09, .08, .02, 0.]),
        porosity_by_facies={"1": .41, "2": .44, "3": .45},
        swi_by_facies={1: .32, 2: .12, 3: .14},
    )
    np.testing.assert_array_equal(result.facies[:, 0], [1, 1, 1, 2, 2])
    np.testing.assert_array_equal(result.donor_rows[:, 0], [0, 0, 0, 4, 4])
    np.testing.assert_array_equal(result.facies[:, 1], [3, 3, 3, 3, 3])
    np.testing.assert_array_equal(result.porosity[:, 0], [.41, .41, .41, .44, .44])
    np.testing.assert_array_equal(result.swi[:, 0], [.32, .32, .32, .12, .12])
    np.testing.assert_array_equal(result.facies[defined], facies[defined])
    np.testing.assert_array_equal(result.geometry_filled, ~defined)
    assert not result.geometry_unknown.any()
    assert result.facies.dtype == facies.dtype
    assert result.porosity.dtype == result.swi.dtype == np.float64
    np.testing.assert_array_equal(facies, before)


def test_column_fill_tie_selects_higher_and_never_borrows_from_another_column():
    facies = np.array([[1, 0], [0, 0], [0, 0], [0, 0], [2, 0]], np.uint8)
    result = extend_column_geometry(
        facies, facies != 0, np.ones(facies.shape, bool), _heights(facies.shape),
        porosity_by_facies={1: .4, 2: .5}, swi_by_facies={1: .3, 2: .1},
    )
    assert result.facies[2, 0] == 1
    assert result.donor_rows[2, 0] == 0
    assert result.geometry_unknown[:, 1].all()
    assert np.isnan(result.porosity[:, 1]).all()
    assert np.isnan(result.swi[:, 1]).all()
    assert (result.donor_rows[:, 1] == -1).all()
    assert np.isnan(result.donor_distance_m[:, 1]).all()
    assert not result.geometry_filled[:, 1].any()


def test_fixture_or_undefined_property_never_becomes_a_donor():
    facies = np.array([[1, 8], [0, 0], [2, 0]], np.uint8)
    defined = facies != 0
    domain = np.ones(facies.shape, bool)
    domain[0, 0] = False
    result = extend_column_geometry(
        facies, defined, domain, _heights(facies.shape),
        porosity_by_facies={2: .42}, swi_by_facies={2: .12},
    )
    assert result.facies[0, 0] == 1  # outside-domain source is preserved
    assert result.donor_rows[1, 0] == 2
    assert result.facies[1, 0] == 2
    assert result.geometry_unknown[:, 1].all()
    assert result.facies[0, 1] == 8  # defined but unmeasured is not relabelled
    assert not result.geometry_unknown[0, 0]
    assert np.isnan(result.porosity[0, 0])


def test_missing_water_saturation_stays_unknown_without_invalidating_porosity():
    facies = np.array([[7], [0]], np.uint8)
    result = extend_column_geometry(
        facies, facies != 0, np.ones(facies.shape, bool), _heights(facies.shape),
        porosity_by_facies={7: .44}, swi_by_facies={},
    )
    np.testing.assert_array_equal(result.facies, [[7], [7]])
    np.testing.assert_array_equal(result.porosity, [[.44], [.44]])
    assert np.isnan(result.swi).all()
    assert not result.geometry_unknown.any()
    assert "depth_m" not in vars(result)  # depth belongs to target coordinates


def test_filled_facies_keeps_depth_at_target_coordinates_not_the_donor():
    from darsia.fluidflower.mass.n1_geometry import measured_geometry

    facies = np.array([[1, 1], [0, 0]], np.uint8)
    x = np.array([[0., 1.], [0., 1.]])
    y = np.array([[1., 1.], [0., 0.]])
    result = extend_column_geometry(
        facies, facies != 0, np.ones(facies.shape, bool), y,
        porosity_by_facies={1: .44}, swi_by_facies={1: .12},
    )
    measured = measured_geometry(
        x, y, result.facies,
        [[0., 0., .010], [1., 0., .012], [0., 1., .014], [1., 1., .016]],
        porosity_by_facies={1: .44}, swi_by_facies={1: .12},
    )
    np.testing.assert_array_equal(result.donor_rows[1], [0, 0])
    np.testing.assert_array_equal(measured["depth_m"], [[.014, .016], [.010, .012]])
    assert np.all(measured["depth_m"][1] != measured["depth_m"][0])


def test_column_donor_is_strictly_above_and_the_metric_limit_is_inclusive():
    donors = np.array([[True, False], [False, False], [False, False], [True, False]], bool)
    y = _heights(donors.shape, [.04, .03, .02, .01])
    rows, available, distance = column_co2_donors(donors, y, max_distance_m=.020)
    np.testing.assert_array_equal(rows[:, 0], [-1, 0, 0, -1])
    np.testing.assert_array_equal(available[:, 0], [False, True, True, False])
    assert distance[2, 0] == .020
    assert not available[:, 1].any()
    assert np.isnan(distance[:, 1]).all()


def test_column_distance_does_not_add_a_hidden_boundary_tolerance():
    y = np.array([[0.], [-np.nextafter(.020, np.inf)]])
    rows, available, distance = column_co2_donors(
        np.array([[True], [False]]), y, max_distance_m=.020
    )
    assert not available.any()
    assert (rows == -1).all()
    assert np.isnan(distance).all()


@pytest.mark.parametrize("bad", [np.ones((2, 2), np.uint8), np.ones(2, bool), np.ones((0, 2), bool)])
def test_domain_rejects_non_boolean_or_non_grid_masks(bad):
    with pytest.raises(ValueError):
        rig_pixel_domain(bad, {})


def test_visibility_and_fixtures_reject_different_grids():
    with pytest.raises(ValueError):
        visible_pixels(np.ones((2, 2), bool), np.ones((1, 2), bool))
    with pytest.raises(ValueError):
        rig_pixel_domain(np.ones((2, 2), bool), {"fitting": np.ones((2, 1), bool)})
    with pytest.raises(TypeError):
        rig_pixel_domain(np.ones((2, 2), bool), [])


@pytest.mark.parametrize("heights", [[[1.], [1.]], [[0.], [1.]], [[np.nan], [0.]]])
def test_column_operations_reject_nonfinite_or_non_descending_heights(heights):
    with pytest.raises(ValueError):
        column_co2_donors(np.ones((2, 1), bool), heights, max_distance_m=.020)
    with pytest.raises(ValueError):
        extend_column_geometry(
            np.ones((2, 1), np.uint8), np.ones((2, 1), bool), np.ones((2, 1), bool),
            heights, porosity_by_facies={1: .4}, swi_by_facies={1: .1},
        )


@pytest.mark.parametrize("distance", [0., -.01, np.inf, np.nan, True])
def test_column_donors_require_an_explicit_finite_positive_distance(distance):
    with pytest.raises(ValueError):
        column_co2_donors(np.ones((2, 1), bool), [[1.], [0.]], max_distance_m=distance)


@pytest.mark.parametrize("porosity", [{1: 0.}, {1: 1.1}, {1: np.nan}, {1: True}])
def test_geometry_rejects_nonphysical_porosity(porosity):
    with pytest.raises(ValueError):
        extend_column_geometry(
            np.ones((2, 1), np.uint8), np.ones((2, 1), bool), np.ones((2, 1), bool),
            [[1.], [0.]], porosity_by_facies=porosity, swi_by_facies={1: .1},
        )
def _hidden_adjacency_case(x, y, label=1, visible=True):
    from darsia.fluidflower.geometry.pixels import hidden_boundary_adjacency
    return hidden_boundary_adjacency(
        np.array([[label]], np.uint8), np.array([[visible]], bool),
        np.array([[x]], np.float64), np.array([[y]], np.float64),
        inner_polygon_m=[[-.2, -.2], [.2, -.2], [.2, .2], [-.2, .2]],
        fixture_polygons_m={
            "left": [[0, 0], [.1, 0], [.1, .1], [0, .1]],
            "right": [[.05, 0], [.15, 0], [.15, .1], [.05, .1]],
            "mount": [[.4, .4], [.5, .4], [.5, .5], [.4, .5]],
        },
    )


@pytest.mark.parametrize("label,visible", [(0, True), (255, True), (1, False), (2, False)])
def test_n16_requires_visible_positive_phase_near_hidden_boundary(label, visible):
    result = _hidden_adjacency_case(.075, -.01, label, visible)
    assert result["co2_adjacent"] is False
    assert result["touching_length_mm"] == 0


@pytest.mark.parametrize("label", [1, 2])
def test_n16_union_boundary_touch_length_is_not_counted_twice(label):
    result = _hidden_adjacency_case(.075, -.01, label)
    assert result["co2_adjacent"] is True
    assert result["touching_length_mm"] == pytest.approx(2 * np.sqrt(.02 ** 2 - .01 ** 2) * 1000)


def test_n16_inclusive_20mm_tangency_has_zero_length_but_forces_upper_bound():
    result = _hidden_adjacency_case(.075, -.020)
    assert result["co2_adjacent"] is True
    assert result["touching_length_mm"] == 0
    outside = _hidden_adjacency_case(.075, -.020000001)
    assert outside["co2_adjacent"] is False
    assert outside["touching_length_mm"] == 0


def test_q52_hidden_physical_geometry_uses_original_visible_column_donors():
    from darsia.fluidflower.geometry.pixels import hidden_physical_geometry
    inside = np.ones((3, 1), bool)
    hidden = np.array([[False], [True], [False]])
    labels = np.array([[1], [9], [2]], np.uint8)
    result = hidden_physical_geometry(labels, inside, inside, {"chart": hidden},
        np.array([[.03], [.02], [.01]]),
        porosity_by_facies={1: .3, 2: .4, 9: .9}, swi_by_facies={1: .1, 2: .2, 9: .9})
    columns = result["geometry"]
    assert columns.facies[1, 0] == 1
    assert columns.porosity[1, 0] == .3
    assert columns.donor_rows[1, 0] == 0
    np.testing.assert_array_equal(result["analysis_mask"], ~hidden)
    np.testing.assert_array_equal(result["hidden_mask"], hidden)
    np.testing.assert_array_equal(result["physical_mask"], inside)
    assert result["physical_geometry_admitted"] is False
    np.testing.assert_array_equal(labels, [[1], [9], [2]])


def test_n16_below_photo_strip_is_estimated_or_zero_and_never_unknown():
    from darsia.fluidflower.geometry.pixels import below_photo_column_support
    labels = np.array([[1, 0], [255, 255], [255, 255]], np.uint8)
    strip = np.array([[False, False], [True, True], [True, True]])
    visible = ~strip
    result = below_photo_column_support(strip, visible, labels,
        np.array([[.03, .03], [.022, .022], [.001, .001]]))
    np.testing.assert_array_equal(result["estimated_mask"], [[False, False], [True, False], [False, False]])
    np.testing.assert_array_equal(result["zero_mask"], [[False, False], [False, True], [True, True]])
    assert not result["unknown_mask"].any()
    assert result["donor_rows"][1, 0] == 0
    assert result["phase_labels_modified"] is False
    np.testing.assert_array_equal(labels, [[1, 0], [255, 255], [255, 255]])
