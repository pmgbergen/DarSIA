"""Synthetic source, geometry and channel-order checks for registration."""
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import sys

import cv2
import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower.native import registration as reg


@pytest.fixture
def parameters():
    return {
        "common": {
            "sensor": {"content_wh": [4, 4], "camera_wh": [4, 4],
                       "raw_wh": [8, 8], "cfa_pattern": [[0, 1], [3, 2]],
                       "color_description": "RGBG", "black_levels": [800] * 4,
                       "black_level": 800, "white_maximum": 65535,
                       "plane_stride": 2, "green_mean_weight": .5, "tile_rows": 2},
            "lens": {"main_page": 0, "sub_page": 0, "distortion_tag": 0x7037,
                     "ca_tag": 0x7035, "distortion_header": 11,
                     "distortion_min_length": 12, "ca_header": 22,
                     "ca_min_length": 23, "distortion_slice": [1, 12],
                     "red_slice": [1, 12], "blue_slice": [12, 23],
                     "ca_red": [0] * 11, "ca_blue": [0] * 11,
                     "knots": np.linspace(0., 1., 11).tolist(),
                     "distortion_denominator": 16384., "ca_denominator": float(2**21),
                     "raw_offset_xy": [0, 0]},
            "profiles": {"fixture": {"coefficients": dict(cu=0., cv=0., tx=0.,
                                      ty=0., R=10., amp=1.), "distortion": [0] * 11}},
            "matching": {"sigma_small": 1.2, "sigma_large": 6.,
                         "minimum_template_std": 1e-4},
        },
        "raw": {"channel_phases": [[0., 0.], [.5, .5], [1., 1.]],
                "certificate": {"contract": "fixture-linear", "eligibility_reason": "test"}},
        "edge_aware": {"demosaic_code": "COLOR_BayerRG2BGR_EA",
                       "order_fixture": [[10000, 4000] * 3, [4000, 1000] * 3] * 3,
                       "order_check_xy": [2, 2], "order_expected_rgb": [10000, 4000, 1000],
                       "certificate": {"contract": "fixture-edge", "white_balance_reason": "test"}},
        "raw_grid": {"shape_hw": [2, 2], "black_level": 800., "blur_sigma": 1.8,
                     "grid_scale": 2, "center_offset": .5, "ca_denominator": 2**21,
                     "channel_phases": {"R": [0., 0.], "G": [.5, .5], "B": [1., 1.]},
                     "geometry_coefficients_key": "coefficients",
                     "geometry_distortion_key": "distortion", "geometry_ca_key": "ca",
                     "signal_minimum": 2, "signal_maximum": 15100},
    }


def test_lens_metadata_is_exact_and_rejects_changed_channel_table(parameters, monkeypatch):
    lens = parameters["common"]["lens"]
    tags = {lens["distortion_tag"]: SimpleNamespace(value=[11] + [0] * 11),
            lens["ca_tag"]: SimpleNamespace(value=[22] + [0] * 22)}
    page = SimpleNamespace(tags=tags)
    class Tiff:
        pages = [SimpleNamespace(pages=[page])]
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(reg.tifffile, "TiffFile", lambda _: Tiff())
    assert reg.read_lens_tables(Path("fixture"), parameters=lens) == ((0,) * 11,) * 3
    tags[lens["ca_tag"]].value[1] = 1
    with pytest.raises(ValueError, match="channel table"):
        reg.read_lens_tables(Path("fixture"), parameters=lens)


def test_bandpass_and_patch_matches_detect_known_translation(parameters):
    policy = parameters["common"]["matching"]
    constant = np.full((48, 48), 30., np.float32)
    filtered = reg.bandpass(constant, parameters=policy)
    assert filtered.dtype == np.float32
    assert np.ptp(filtered) == 0
    same_scale = dict(policy, sigma_large=policy["sigma_small"])
    assert np.array_equal(reg.bandpass(constant, parameters=same_scale), np.zeros_like(filtered))
    impulse = np.zeros((65, 65), np.float32)
    impulse[32, 32] = 1
    response = reg.bandpass(impulse, parameters=policy)
    assert response[32, 32] > 0 and np.min(response) < 0
    a = np.random.default_rng(8).normal(size=(64, 64)).astype(np.float32)
    b = cv2.warpAffine(a, np.float32([[1, 0, 2], [0, 1, -1]]), (64, 64))
    rows = reg.match_rows(a, b, [(20, 20), (35, 35), (0, 0)], template_half=4,
                          search_half=5, guess=lambda x, y: (x, y),
                          min_correlation=.9, parameters=policy)
    assert rows.shape == (2, 5) and rows.dtype == np.float64
    np.testing.assert_allclose(rows[:, 2:4] - rows[:, :2], [[2, -1]] * 2, atol=.05)
    assert reg.match_rows(constant, constant, [(20, 20)], template_half=4,
                          search_half=5, guess=lambda x, y: (x, y),
                          min_correlation=.9, parameters=policy).shape == (0, 5)


def test_match_metrics_empty_and_exact_scaled_displacement():
    assert reg.match_metrics(np.empty((0, 5)), scale=1.0) == {
        "n": 0, "p50_px": None, "p95_px": None, "max_px": None}
    result = reg.match_metrics(np.array([[0, 0, 3, 4, .5], [2, 2, 5, 6, 1.]]), scale=2.)
    assert result == dict(n=2, p50_px=10., p95_px=10., max_px=10.,
                          median_match_correlation=.75)


def test_review_homography_uses_independent_regions(parameters):
    common = parameters["common"]
    common["review"] = dict(fit_y_range=[20, 80, 10], fit_x_range=[20, 80, 10],
        template_half=4, search_half=7, guess_offset_xy=[0, 0], min_correlation=.8,
        minimum_matches=20, ransac_threshold=1., minimum_inlier_fraction=.9,
        orientation_y=[0, 48, 95], orientation_x=[0, 48, 95],
        heldout_zones=[["independent", [18, 18, 80, 80], 16]],
        heldout_template_half=3, heldout_search_half=5,
        heldout_min_correlation=.8, heldout_minimum_matches=9)
    image = np.random.default_rng(9).uniform(20, 220, (96, 96)).astype(np.float32)
    moved = cv2.warpAffine(image, np.float32([[1, 0, 2], [0, 1, 3]]), (96, 96))
    h, audit = reg.review_to_camera(image, moved, parameters=common)
    np.testing.assert_allclose(h, [[1, 0, 2], [0, 1, 3], [0, 0, 1]], atol=.04)
    assert audit["sampled_jacobian_min"] > 0
    assert audit["heldout"]["independent"]["n"] >= 9
    common["review"]["minimum_matches"] = 1000
    with pytest.raises(ValueError, match="Insufficient"):
        reg.review_to_camera(image, moved, parameters=common)


def test_raw_coordinates_preserve_float64_lens_math_and_channel_scale(parameters):
    common = parameters["common"]
    u, v = np.mgrid[:3, :4].astype(np.float32)
    p = common["profiles"]["fixture"]["coefficients"]
    x, y = reg.raw_coordinates(u, v, np.eye(3), p, [0] * 11,
                               parameters=common["lens"])
    assert x.dtype == y.dtype == np.float64
    assert np.array_equal(x, u) and np.array_equal(y, v)
    x, y = reg.raw_coordinates(u, v, np.eye(3), p, [0] * 11, [2**20] * 11,
                               parameters=common["lens"])
    assert np.array_equal(x, u.astype(np.float64) * 1.5)
    assert np.array_equal(y, v.astype(np.float64) * 1.5)


def test_green_residual_keeps_the_independent_two_by_two_reference(parameters):
    common = parameters["common"]
    common["residual"] = dict(shape_hw=[40, 40], sample_scale=2, sample_offset=.5,
        native_crop_rows=80, zones=[["inside", [8, 8, 32, 32], 6]],
        template_half=2, search_half=3, min_correlation=.9, minimum_matches=9)
    reference = np.random.default_rng(10).uniform(30, 220, (40, 40)).astype(np.float32)
    native = np.repeat(np.repeat(reference, 2, axis=0), 2, axis=1)
    audit = reg.raw_review_residual(reference + 1., native, np.eye(3),
        common["profiles"]["fixture"]["coefficients"], [0] * 11, parameters=common)
    assert audit["inside"]["n"] >= 9
    shifted = cv2.warpAffine(reference + 1., np.float32([[1, 0, 1], [0, 1, 0]]), (40, 40))
    displaced = reg.raw_review_residual(shifted, native, np.eye(3),
        common["profiles"]["fixture"]["coefficients"], [0] * 11, parameters=common)
    assert displaced["inside"]["p50_px"] > audit["inside"]["p50_px"] + 1
    with pytest.raises(ValueError, match="Insufficient direct"):
        reg.raw_review_residual(np.ones((40, 40), np.float32), np.zeros((80, 80), np.float32),
            np.eye(3), common["profiles"]["fixture"]["coefficients"], [0] * 11,
            parameters=common)


def test_demosaic_control_rejects_a_red_blue_swap(parameters):
    reg.check_demosaic_order(parameters=parameters)
    parameters["edge_aware"]["order_expected_rgb"].reverse()
    with pytest.raises(ValueError, match="output order"):
        reg.check_demosaic_order(parameters=parameters)


def test_gray_camera_region_is_copied_and_size_is_bound(tmp_path, parameters):
    a = np.arange(144, dtype=np.uint8).reshape(12, 12)
    path = tmp_path / "camera.png"
    Image.fromarray(a).save(path)
    sensor = parameters["common"]["sensor"]
    sensor["camera_wh"] = [12, 12]
    result = reg.gray_roi(path, (2, 3, 8, 9), parameters=sensor)
    assert np.array_equal(result, a[3:9, 2:8]) and result.flags.owndata
    sensor["camera_wh"] = [13, 12]
    with pytest.raises(ValueError, match="dimensions"):
        reg.gray_roi(path, (2, 3, 8, 9), parameters=sensor)


@pytest.fixture
def sensor_files(tmp_path, parameters, monkeypatch):
    arw = tmp_path / "source.arw"
    camera = tmp_path / "source.jpg"
    review = tmp_path / "review.jpg"
    arw.write_bytes(b"synthetic sensor")
    Image.fromarray(np.full((4, 4, 3), 100, np.uint8)).save(camera)
    Image.fromarray(np.full((4, 4, 3), 100, np.uint8)).save(review)
    mosaic = np.empty((8, 8), np.uint16)
    mosaic[0::2, 0::2] = 4800
    mosaic[0::2, 1::2] = mosaic[1::2, 0::2] = 2800
    mosaic[1::2, 1::2] = 1800
    class Raw:
        raw_image_visible = mosaic
        raw_pattern = np.array([[0, 1], [3, 2]])
        color_desc = b"RGBG"
        black_level_per_channel = [800] * 4
        white_level = 16383
        camera_whitebalance = [2., 1., 1.5, 1.]
        rgb_xyz_matrix = np.eye(3)
        def __enter__(self): return self
        def __exit__(self, *args): pass
    calls = []
    def decode(path): calls.append(path); return Raw()
    monkeypatch.setitem(sys.modules, "rawpy", SimpleNamespace(imread=decode))
    monkeypatch.setattr(reg, "review_to_camera", lambda *a, **k: (np.eye(3), {"fixture": True}))
    monkeypatch.setattr(reg, "read_lens_tables", lambda *a, **k: ((0,) * 11,) * 3)
    monkeypatch.setattr(reg, "raw_review_residual", lambda *a, **k: {"fixture": True})
    hashes = {k: sha256(p.read_bytes()).hexdigest() for k, p in [
        ("arw_sha256", arw), ("camera_jpeg_sha256", camera), ("review_native_jpeg_sha256", review)]}
    return arw, camera, review, hashes, calls


@pytest.mark.parametrize("function", [reg.registered_sensor_linear,
                                    reg.registered_edge_aware_sensor_linear])
def test_renderers_keep_black_subtraction_basis_and_fail_on_source_change(
        function, sensor_files, parameters):
    arw, camera, review, hashes, calls = sensor_files
    result, cert = function(arw, camera, review, expected_source_hashes=hashes,
        content_box=(0, 0, 4, 4), parameters=parameters, source_id="fixture", profile_id="fixture")
    assert result.shape == (3, 4, 4) and result.dtype == np.float32
    assert np.array_equal(result[:, 2, 2], np.array([4000, 2000, 1000], np.float32))
    assert cert["sources"] == hashes and cert["run"] == "fixture"
    if function is reg.registered_edge_aware_sensor_linear:
        assert cert["white_balance"]["applied"] is False
        assert cert["camera_to_xyz"]["applied"] is False
        assert cert["sensor_saturated_count"] == 0
    count = len(calls)
    arw.write_bytes(b"changed")
    with pytest.raises(ValueError, match="[Ss][Hh][Aa]|hash"):
        function(arw, camera, review, expected_source_hashes=hashes,
            content_box=(0, 0, 4, 4), parameters=parameters, source_id="fixture", profile_id="fixture")
    assert len(calls) == count


def test_chart_registration_projects_translation_and_rejects_saturated_support(
        tmp_path, parameters, monkeypatch):
    camera = tmp_path / "camera.png"
    Image.fromarray(np.zeros((96, 96), np.uint8)).save(camera)
    parameters["common"]["sensor"].update(camera_wh=[96, 96], content_wh=[64, 64])
    parameters["chart"] = dict(swatches=6, grid_scale=1, center_offset=0.,
        template_margin=5, minimum_template_size=20, search_margin=5,
        rng_seed=0, sift_features=30, ratio_threshold=.75, minimum_ratio_matches=4,
        ransac_threshold=2., minimum_inliers=4, maximum_p95=2., patch_lower=1,
        patch_upper=2, saturation_threshold=15100, minimum_usable=5,
        certificate={"contract": "fixture-chart", "saturation_description": "test"})
    points = np.float32([[8, 8], [16, 8], [24, 8], [8, 16], [16, 16], [24, 16]])
    class Features:
        count = 0
        def detectAndCompute(self, *args):
            p = points if self.count == 0 else points + [8, 7]
            self.count += 1
            return [SimpleNamespace(pt=x) for x in p], np.ones((6, 8), np.float32)
    monkeypatch.setattr(reg.cv2, "SIFT_create", lambda **kwargs: Features())
    pairs = [[SimpleNamespace(queryIdx=i, trainIdx=i, distance=.1),
              SimpleNamespace(distance=1.)] for i in range(6)]
    monkeypatch.setattr(reg.cv2, "BFMatcher", lambda: SimpleNamespace(knnMatch=lambda *a, **k: pairs))
    xy = np.float32([[12, 12], [24, 12], [36, 12], [12, 36], [24, 36], [48, 48]])
    native = np.ones((3, 64, 64), np.float32) * 1000
    native[:, 14, 15] = 15100
    kwargs = dict(baseline_camera_jpeg=camera, baseline_camera_sha256=sha256(camera.read_bytes()).hexdigest(),
        frame_camera_jpeg=camera, frame_camera_sha256=sha256(camera.read_bytes()).hexdigest(),
        baseline_chart_xy8=xy, baseline_chart_log_rgb=np.ones((3, 6), np.float32),
        baseline_review_to_camera_h=np.eye(3), frame_review_to_camera_h=np.eye(3),
        frame_native_linear_rgb=native, parameters=parameters)
    moved, usable, cert = reg.register_chart(**kwargs)
    assert moved.dtype == np.float32 and usable.dtype == bool
    np.testing.assert_array_equal(moved, xy + [3, 2])
    assert usable.tolist() == [False, True, True, True, True, True]
    assert cert["usable_patches"] == 5
    native[:, 14, 27] = np.nan
    with pytest.raises(ValueError, match="Fewer"):
        reg.register_chart(**kwargs)


def test_prewarp_sensor_grid_log_retains_float32_and_original_signal_bounds(
        sensor_files, parameters):
    arw, _, _, _, _ = sensor_files
    p = parameters["common"]["profiles"]["fixture"]["coefficients"]
    geometry = {"coefficients": {"fixture": p}, "distortion": {"fixture": [0] * 11},
                "ca": {"R": [0] * 11, "B": [0] * 11}}
    result = reg.registered_sensor_grid_log(arw, np.eye(3), "fixture", geometry, parameters=parameters)
    assert result.shape == (3, 2, 2) and result.dtype == np.float32
    np.testing.assert_array_equal(result[:, 1, 1], np.log(np.float32([4000, 2000, 1000])))
    original = result
    parameters["raw_grid"]["signal_maximum"] = 3000
    result = reg.registered_sensor_grid_log(arw, np.eye(3), "fixture", geometry, parameters=parameters)
    assert np.isnan(result[0]).all()
    assert np.array_equal(result[1:], original[1:], equal_nan=True)
    with pytest.raises(ValueError, match="Unknown geometry profile"):
        reg.registered_sensor_grid_log(arw, np.eye(3), "unbound", geometry, parameters=parameters)
