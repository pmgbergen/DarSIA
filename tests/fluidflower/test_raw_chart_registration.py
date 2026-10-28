"""Actual SIFT chart translation using only bound in-memory RAW views."""

from hashlib import sha256

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower.native import raw_chart, raw_positioning, registration


def test_actual_chart_matcher_uses_RAW_views_and_retains_native_saturation_gate(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("Chart matching opened a companion photograph")
    monkeypatch.setattr(Image, "open", refuse)
    monkeypatch.setattr(registration, "gray_roi", refuse)
    parameters = dict(common=dict(sensor=dict(camera_wh=[96, 96], content_wh=[64, 64])),
        chart=dict(swatches=6, grid_scale=1, center_offset=0., template_margin=7,
            minimum_template_size=20, search_margin=7, rng_seed=0, sift_features=500,
            ratio_threshold=.75, minimum_ratio_matches=4, ransac_threshold=2.,
            minimum_inliers=4, maximum_p95=2., patch_lower=1, patch_upper=2,
            saturation_threshold=15100, minimum_usable=5,
            certificate=dict(contract="fixture-chart", saturation_description="test")))
    def sensor(name):
        path = tmp_path / (name + ".ARW")
        path.write_bytes(name.encode())
        return raw_positioning.Sensor(np.zeros((96, 96, 3), np.uint16),
            dict(path=str(path), sha256=sha256(path.read_bytes()).hexdigest()),
            dict(white_level=16383, black_level=1008), (0,), (0,), (0,), "camera")
    baseline, frame = sensor("B"), sensor("frame")
    camera = np.random.default_rng(193).uniform(100, 10000, (3, 96, 96)).astype(np.float32)
    moved = np.roll(camera, (2, 3), axis=(1, 2))
    xy = np.float32([[12, 12], [24, 12], [36, 12], [12, 36], [24, 36], [48, 48]])
    native = np.full((3, 64, 64), 1000, np.float32)
    native[:, 14, 15] = 15100
    result, usable, cert = raw_chart.register(
        baseline_sensor=baseline, frame_sensor=frame, baseline_camera_u=camera, frame_camera_u=moved,
        baseline_chart_xy8=xy, baseline_chart_log_rgb=np.ones((3, 6), np.float32),
        baseline_native_to_u_h=np.eye(3), frame_native_to_u_h=np.eye(3),
        frame_native_linear_rgb=native, parameters=parameters)
    np.testing.assert_allclose(result, xy + [3, 2], rtol=0, atol=.3)
    assert usable.tolist() == [False, True, True, True, True, True]
    assert cert["baseline_raw"] == dict(baseline.source) and cert["frame_raw"] == dict(frame.source)
    assert cert["jpeg_inputs"] is False and "frame_camera_jpeg" not in cert
