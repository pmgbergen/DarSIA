import copy

import numpy as np
import pytest

from darsia.fluidflower.native.raw_camera import CameraIdentity
from darsia.fluidflower.phase.ffum_raw_input import UmRelativeMapping, validate_um_camera_identity


def profile():
    return dict(candidate="R_only", raw_offset_xy=[16, 8], center_jpg_xy=[4127.5, 2751.5],
                radius_px=4961.23855503845, green_preserved=True,
                coefficients_px={"R": [1.610066890659688, -.8642695466479476],
                                 "G": [0., 0.], "B": [0., 0.]})


def test_all_sensor_edges_fit_canvas_and_green_anchor_is_unchanged():
    mapping = UmRelativeMapping(profile())
    receipt = mapping.validate_boundary()
    assert receipt["sensor_boundary_points_per_channel"] == 27616
    points = np.array([[0., 0.], [8287., 5519.], [4127.5, 2751.5]])
    np.testing.assert_array_equal(mapping.sensor_to_display(points, "G"), points + [3, 3])
    np.testing.assert_array_equal(mapping.sensor_to_display(points, "B"), points + [3, 3])


def test_red_inverse_covers_complete_field():
    mapping = UmRelativeMapping(profile())
    x, y = np.meshgrid(np.linspace(0, 8287, 79), np.linspace(0, 5519, 53))
    sensor = np.stack((x, y), -1)
    actual = mapping.display_to_sensor(mapping.sensor_to_display(sensor, "R"), "R")
    np.testing.assert_allclose(actual, sensor, rtol=0, atol=1e-8)


def test_different_sensor_canvas_and_common_geometry_are_rejected():
    with pytest.raises(ValueError, match="validated Nikon"):
        UmRelativeMapping(profile(), sensor_wh=(8256, 5504))
    with pytest.raises(ValueError, match="validated Nikon"):
        UmRelativeMapping(profile(), origin_jpg_xy=(-16, -8))
    changed = copy.deepcopy(profile()); changed["coefficients_px"]["G"][0] = .01
    with pytest.raises(ValueError, match="validated Nikon"):
        UmRelativeMapping(changed)


def test_camera_identity_and_orientation_are_bound_before_profile_application():
    assert validate_um_camera_identity(CameraIdentity("NIKON CORPORATION", "NIKON Z 7_2", 1)).key == "nikon-z-7-2"
    with pytest.raises(ValueError, match="Nikon Z7II"):
        validate_um_camera_identity(CameraIdentity("SONY", "ILCE-7RM3", 1))
    with pytest.raises(ValueError, match="unrotated"):
        validate_um_camera_identity(CameraIdentity("NIKON CORPORATION", "NIKON Z 7_2", 6))
