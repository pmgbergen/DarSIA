import copy

import numpy as np
import pytest

from darsia.fluidflower.rig_reference_points import (
    REFERENCE_POINT_NAMES,
    point_description,
    reference_point_coordinates_mm,
)
from darsia.fluidflower.rig_reference_quadrilateral import reference_frame_geometry
from darsia.fluidflower.series_geometry import RigFrame


def irregular_frame():
    metric = np.array([[-.08, .46], [.84, .54], [.90, 0.], [0., 0.]])
    photo = np.column_stack((1000. - metric[:, 1] * 1500., 200. + metric[:, 0] * 1000.))
    sides = np.linalg.norm(np.roll(metric, -1, axis=0) - metric, axis=1)
    measures = dict(zip(("top", "right", "bottom", "left"), sides))
    measures["diagonal_ac"] = float(np.linalg.norm(metric[2] - metric[0]))
    return RigFrame(**reference_frame_geometry(photo, shape="quadrilateral", measurements_m=measures))


def rectangle_frame(**changes):
    values = {
        "corners": dict(zip(REFERENCE_POINT_NAMES, ((10., 20.), (10., 919.), (500., 919.), (500., 20.)))),
        "width_m": .899,
        "height_m": .490,
    }
    return RigFrame(**{**values, **changes})


@pytest.mark.parametrize("stored_metric", [True, False])
def test_irregular_references_use_observed_bottom_left_not_bounding_box(stored_metric):
    frame = irregular_frame()
    if not stored_metric:
        del frame.reference_geometry["metric_corners_xy"]
    before = copy.deepcopy(frame.to_dict())
    result = reference_point_coordinates_mm(frame)
    expected = [[-80., 460.], [840., 540.], [900., 0.], [0., 0.]]
    np.testing.assert_allclose([result[name] for name in REFERENCE_POINT_NAMES], expected, atol=1e-8)
    assert frame.to_dict() == before
    assert frame.width_m == pytest.approx(.980)
    assert frame.pixel_to_rig(*frame.corners["bottom_left"]) == pytest.approx((0., 0.))


def test_legacy_rectangle_uses_physical_mm_and_never_rewrites_saved_frame():
    frame = rectangle_frame()
    before = frame.to_dict()
    result = reference_point_coordinates_mm(frame)
    np.testing.assert_allclose([result[name] for name in REFERENCE_POINT_NAMES],
                               [[0., 490.], [899., 490.], [899., 0.], [0., 0.]], atol=1e-8)
    assert frame.to_dict() == before


@pytest.mark.parametrize("width,height", [(None, None), (.899, None), (None, .490), (float("nan"), .490), (.899, 0.)])
def test_missing_or_invalid_dimensions_do_not_invent_reference_positions(width, height):
    frame = rectangle_frame(width_m=width, height_m=height)
    assert reference_point_coordinates_mm(frame) == {
        "top_left": None, "top_right": None, "bottom_right": None, "bottom_left": [0., 0.],
    }


def test_unavailable_bottom_left_means_no_shared_origin_and_no_coordinates():
    frame = rectangle_frame()
    del frame.corners["bottom_left"]
    assert reference_point_coordinates_mm(frame) == dict.fromkeys(REFERENCE_POINT_NAMES)


def test_incomplete_legacy_rectangle_can_display_its_known_physical_points():
    frame = rectangle_frame(disabled_corners=("top_left", "top_right"))
    del frame.corners["top_right"]
    assert reference_point_coordinates_mm(frame) == {
        "top_left": [0., 490.], "top_right": None,
        "bottom_right": [899., 0.], "bottom_left": [0., 0.],
    }


def test_invalid_stored_metric_falls_back_to_observed_points_without_mutation():
    frame = irregular_frame()
    frame.reference_geometry["metric_corners_xy"]["top_left"] = [float("inf"), 0.]
    result = reference_point_coordinates_mm(frame)
    np.testing.assert_allclose(result["top_left"], [-80., 460.], atol=1e-8)
    assert np.isinf(frame.reference_geometry["metric_corners_xy"]["top_left"][0])


def test_descriptions_are_per_point_metadata_not_inferred_from_estimated_height():
    frame = rectangle_frame(reference_geometry={
        "estimated_height_uncertainty_m": .005,
        "point_descriptions": {"top_left": "top_of_free_water", "top_right": "custom_reference"},
    })
    before = copy.deepcopy(frame.to_dict())
    assert point_description(frame, "top_left") == "top_of_free_water"
    assert point_description(frame, "top_right") == "custom_reference"
    assert point_description(frame, "bottom_left") == "rig_frame"
    assert point_description(rectangle_frame(reference_geometry={"estimated_height_uncertainty_m": .005}), "top_left") == "rig_frame"
    assert point_description(rectangle_frame(), "top_left") == "rig_frame"
    assert frame.to_dict() == before
