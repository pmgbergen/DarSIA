import copy

import numpy as np
import pytest

from darsia.fluidflower.rig_reference_quadrilateral import reference_frame_geometry
from darsia.fluidflower.series_geometry import RigFrame


NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")
METRIC = np.array([[-.08, .46], [.84, .54], [.90, 0.0], [0.0, 0.0]])


def project(points, matrix):
    mapped = np.column_stack((points, np.ones(len(points)))) @ np.asarray(matrix).T
    return mapped[:, :2] / mapped[:, 2, None]


def measures(points=METRIC):
    sides = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    result = dict(zip(("top", "right", "bottom", "left"), sides))
    result["diagonal_ac"] = float(np.linalg.norm(points[2] - points[0]))
    return result


def photo(points=METRIC, matrix=None):
    if matrix is None:
        matrix = [[4000., 300., 500.], [120., -3800., 2600.], [.1, .08, 1.]]
    return project(points, matrix)[:, ::-1]


def test_rectangle_preserves_the_four_clicked_points_and_metric_scale():
    points = photo(np.array([[0., .490], [.899, .490], [.899, 0.], [0., 0.]]))
    result = reference_frame_geometry(points, measurements_m={"width": .899, "height": .490})
    assert list(result["corners"].values()) == list(map(tuple, points))
    frame = RigFrame(**result)
    assert frame.width_m == .899 and frame.height_m == .490
    for name, rc, xy in zip(NAMES, points, [[0., .490], [.899, .490], [.899, 0.], [0., 0.]]):
        np.testing.assert_allclose(frame.pixel_to_rig(*rc), xy, atol=1e-12)
        assert result["reference_geometry"]["photo_corners_rc"][name] == list(rc)
    assert frame.native_crop()["height"] == .490
    sides = {"top": .899, "bottom": .899, "left": .490, "right": .490}
    assert reference_frame_geometry(points, measurements_m=sides)["width_m"] == .899


def test_measured_quad_recovers_each_side_and_diagonal_without_rectangular_stretch():
    points = photo()
    result = reference_frame_geometry(points, shape="quadrilateral", measurements_m=measures())
    frame = RigFrame(**result)
    metric = np.array(list(result["reference_geometry"]["metric_corners_xy"].values()))
    expected = METRIC - METRIC.min(axis=0)
    np.testing.assert_allclose(metric, expected, atol=1e-12)
    recovered = np.array([frame.pixel_to_rig(*rc) for rc in points])
    np.testing.assert_allclose(recovered, expected, atol=1e-12)
    for key, length in measures(recovered).items():
        assert length == pytest.approx(measures()[key], abs=1e-12)
    assert frame.width_m == pytest.approx(.98)
    assert frame.height_m == pytest.approx(.54)
    assert not np.allclose(list(frame.corners.values()), points)
    crop = frame.native_crop()
    for name in NAMES:
        assert crop[name] == [round(v) for v in frame.corners[name]]
    for x, y in [[0., 0.], [.12, .13], [.98, .54], [.68, .21]]:
        np.testing.assert_allclose(frame.pixel_to_rig(*frame.rig_to_pixel(x, y)), [x, y], atol=1e-12)


def test_reference_metadata_roundtrips_and_legacy_shape_stays_unchanged():
    frame = RigFrame(**reference_frame_geometry(photo(), shape="quadrilateral", measurements_m=measures()))
    frame.reference_geometry["estimated_height_uncertainty_m"] = .005
    record = frame.to_dict()
    assert RigFrame.from_dict(record).to_dict() == record
    record["reference_geometry"]["measurements_m"]["top"] = 99.
    assert frame.reference_geometry["measurements_m"]["top"] != 99.
    original = {"corners": {name: list(value) for name, value in zip(NAMES, [[0., 0.], [0., 10.], [20., 10.], [20., 0.]])},
                "width_m": .899, "height_m": .490, "run_id": "ac17", "source": "marked in geometry workspace", "image": "", "image_size": None}
    assert RigFrame.from_dict(original).to_dict() == original
    hidden = copy.deepcopy(original)
    hidden["disabled_corners"] = ["top_left", "top_right"]
    assert RigFrame.from_dict(hidden).to_dict() == hidden
    with pytest.raises(ValueError, match="measurement record"):
        RigFrame.from_dict({**original, "reference_geometry": []})


@pytest.mark.parametrize("points", [
    [[0., 0.], [0., 10.], [0., 20.], [0., 30.]],
    [[0., 0.], [10., 10.], [0., 10.], [10., 0.]],
    [[0., 0.], [0., 10.], [3., 3.], [10., 0.]],
    [[0., 0.], [0., 10.], [10., 10.], [np.nan, 0.]],
    [[0., 0.], [0., 10.], [10., 10.]],
])
def test_crossed_concave_or_nonfinite_image_reference_is_rejected(points):
    with pytest.raises(ValueError):
        reference_frame_geometry(points, measurements_m={"width": .899, "height": .490})


@pytest.mark.parametrize("shape,lengths", [
    ("rectangle", {"width": -.899, "height": .490}),
    ("rectangle", {"width": .899, "height": float("inf")}),
    ("rectangle", {"top": .899, "bottom": .9, "left": .490, "right": .490}),
    ("quadrilateral", {"top": .9, "bottom": .9, "left": .49, "right": .49}),
    ("quadrilateral", {"top": .9, "bottom": .9, "left": .49, "right": .49, "diagonal_ac": 1.5}),
    ("quadrilateral", {"top": .2, "bottom": .9, "left": .49, "right": .1, "diagonal_ac": 1.0}),
])
def test_missing_inconsistent_or_impossible_physical_measurements_are_rejected(shape, lengths):
    with pytest.raises(ValueError):
        reference_frame_geometry(photo(), shape=shape, measurements_m=lengths)


def test_domain_crossing_perspective_horizon_is_rejected():
    # The measured convex region is visible, but its enclosing rectangular
    # corner lies across the camera plane. Exporting that rectangle is invalid.
    points = photo(matrix=[[100., 0., 100.], [0., -100., 100.], [1., 1., .02]])
    with pytest.raises(ValueError, match="horizon"):
        reference_frame_geometry(points, shape="quadrilateral", measurements_m=measures())
