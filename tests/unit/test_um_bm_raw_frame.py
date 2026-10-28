from copy import deepcopy

import numpy as np
import pytest

from darsia.fluidflower.native.raw_positioning import Sensor
from darsia.fluidflower.phase.um_bm_raw_frame import rectify_sensor


def fixture_frame(*, distortion=(0., 0., 0.), ca=(0., 0., 0.)):
    yy, xx = np.mgrid[:24, :32]
    rgb = np.stack((xx + 100, yy + 200, xx + 2 * yy + 300), axis=-1).astype(np.uint16)
    parameters = {"common": {
        "sensor": {"camera_key": "nikon-z-7-2", "raw_wh": [32, 24],
                   "camera_wh": [26, 20], "tile_rows": 7},
        "lens": {"raw_offset_xy": [3., 2.], "knots": [0., .5, 1.],
                 "distortion_denominator": 1., "ca_denominator": 1.},
        "profiles": {"test": {"coefficients": {
            "cu": 12.5, "cv": 9.5, "tx": 0., "ty": 0., "R": 20., "amp": 1.},
            "distortion": list(distortion)}}}}
    sensor = Sensor(rgb, {"path": "fixture.raw", "sha256": "a" * 64},
                    {"white_level": 16383, "black_level": 1008},
                    distortion, ca, ca, "test")
    return sensor, parameters


def test_restores_sensor_content_outside_camera_crop_without_scaling():
    sensor, parameters = fixture_frame()
    original = deepcopy(parameters)
    frame = rectify_sensor(sensor, parameters=parameters, series="FF_UM")
    left, top, width, height = frame.certificate["extent_u_xywh"]
    # A RAW pixel discarded by the old camera crop is retained at identical DN.
    x, y = 1 - 3 - left, 1 - 2 - top
    assert frame.valid_support[y, x]
    np.testing.assert_array_equal(frame.linear_rgb_chw[:, y, x], sensor.rgb[1, 1])
    assert frame.valid_support.sum() == 32 * 24
    # Even the last sensor centres survive; a NaN interpolation border loses them.
    np.testing.assert_array_equal(
        frame.linear_rgb_chw[:, 23 - 2 - top, 31 - 3 - left], sensor.rgb[-1, -1])
    assert width > 32 and height > 24
    assert parameters == original
    assert frame.certificate["rig_crop_applied"] is False
    assert frame.certificate["white_range_dn"] == 15375


def test_nonlinear_lens_boundary_is_enclosed_and_coordinates_roundtrip():
    sensor, parameters = fixture_frame(distortion=(0., -.06, -.12))
    frame = rectify_sensor(sensor, parameters=parameters, series="FF_UM")
    mapping = frame.mapping
    boundary = mapping.sensor_boundary_xy()
    corrected = mapping.sensor_to_camera_u_xy(boundary)
    returned = mapping.camera_u_to_sensor_xy(corrected)
    np.testing.assert_allclose(returned, boundary, atol=5e-6)
    display = mapping.camera_u_to_display_xy(corrected)
    assert np.all(display >= 0)
    assert np.all(display[:, 0] < frame.valid_support.shape[1])
    assert np.all(display[:, 1] < frame.valid_support.shape[0])
    # Correcting barrel distortion needs more canvas, not a new inner crop.
    assert frame.valid_support.shape[1] > sensor.rgb.shape[1]


def test_undefined_border_is_masked_and_not_fabricated_black():
    sensor, parameters = fixture_frame(distortion=(0., -.05, -.1), ca=(0., .01, .02))
    frame = rectify_sensor(sensor, parameters=parameters, series="FF_UM")
    assert frame.valid_support.any() and not frame.valid_support.all()
    assert np.isfinite(frame.linear_rgb_chw[:, frame.valid_support]).all()
    assert np.isnan(frame.linear_rgb_chw[:, ~frame.valid_support]).any(axis=0).all()
    assert np.all(frame.linear_rgb_chw[:, frame.valid_support] > 0)


def test_fixed_canvas_may_not_crop_the_sensor():
    sensor, parameters = fixture_frame()
    with pytest.raises(ValueError, match="crops"):
        rectify_sensor(sensor, parameters=parameters, series="FF_UM", extent=(0, 0, 26, 20))


def test_same_renderer_supports_bm_and_excludes_other_series():
    sensor, parameters = fixture_frame()
    parameters["common"]["sensor"]["camera_key"] = "sony-ilce-7rm3"
    assert rectify_sensor(sensor, parameters=parameters, series="FF_BM").valid_support.any()
    for series in ("FF_AC", "FF_UM", "other"):
        with pytest.raises(ValueError, match="restricted"):
            rectify_sensor(sensor, parameters=parameters, series=series)


def test_folded_lens_model_is_rejected():
    sensor, parameters = fixture_frame(distortion=(0., -.5, -.9))
    with pytest.raises(ValueError, match="not invertible"):
        rectify_sensor(sensor, parameters=parameters, series="FF_UM")


def test_wrong_profile_is_rejected():
    sensor, parameters = fixture_frame()
    parameters["common"]["profiles"]["test"]["distortion"] = [0., .1, .2]
    with pytest.raises(ValueError, match="differs"):
        rectify_sensor(sensor, parameters=parameters, series="FF_UM")
