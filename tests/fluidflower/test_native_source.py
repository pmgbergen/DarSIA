"""Source-bound camera math and sparse windows on small synthetic grids."""

from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest

from darsia.fluidflower.native import extension, source


@pytest.fixture
def parameters():
    return {
        "source": {
            "homography_min_scale": 1e-12,
            "receipt_homography_min_scale": 1e-9,
            "homography_match_atol": 1e-7,
            "chart_shape": [2, 2], "cell_scale": 2, "cell_center": .5,
            "asset_aliases": {}, "source_id_pattern": r"r[0-9]+",
            "symbols": {"baseline_assets_contract": "bound-calibration"},
            "policy_ids": {"keep": "keep", "natural_display": "natural_modes_v1",
                           "photographic_display": "photographic_v1"},
            "default_policies": {"isolated_invalid_policy": "keep"},
        },
        "extension": {
            "native_shape_hw": [2, 3], "minimum_canvas_hw": [4, 5],
            "native_origin_xy": [-1, -1], "contract": "camera-ring",
            "allowed_light_degrees": [1, 2], "light_cell_center": 1.5,
            "light_cell_scale": 4, "tile_rows": 2,
        },
        "sidecar": {"raw_detail_floor": 2, "raw_detail_ceiling": 100,
                    "raw_detail_exponent": 1., "raw_detail_invalid": 1.},
    }


def test_two_homography_contracts_remain_distinct(parameters):
    matrix = 2 * np.eye(3)
    assert np.array_equal(source.source_homography(matrix, parameters=parameters), np.eye(3))
    assert np.array_equal(source.normalize_homography(matrix, parameters=parameters), np.eye(3))
    singular = np.diag([0., 1., 1.])
    assert np.array_equal(source.source_homography(singular, parameters=parameters), singular)
    with pytest.raises(ValueError, match="homography is invalid"):
        source.normalize_homography(singular, parameters=parameters)


def test_chart_centres_preserve_float32_coordinates(parameters):
    fit = SimpleNamespace(xy=np.array([[0., 0.], [2., 3.]], np.float64))
    grid = SimpleNamespace(origin_cell_xy=(-1, 2))
    transformed = source.chart_baseline_coordinates(fit, grid, np.eye(3), parameters=parameters)
    assert transformed.dtype == np.float32
    assert np.array_equal(transformed, np.array([[-1., 2.], [1., 5.]], np.float32))


def test_parent_cache_exact_reuse_and_new_pixels(parameters):
    full = np.full((3, 2, 2), 9., np.float32)
    parent = np.full_like(full, 3.)
    valid = np.array([[True, False], [False, True]])
    result, audit = source.retain_parent_scientific_pixels(full, parent, valid, parameters=parameters)
    assert np.array_equal(result[:, valid], parent[:, valid])
    assert np.array_equal(result[:, ~valid], np.full((3, 2), 9., np.float32))
    assert audit["freshly_decoded_max_abs_log_difference"] == 6.
    with pytest.raises(ValueError, match="exact native float geometry"):
        source.retain_parent_scientific_pixels(full.astype(np.float64), parent, valid, parameters=parameters)


def test_scientific_support_is_separate_from_display(parameters):
    values = np.full((3, 2, 2), np.log(10), np.float32)
    values[0, 0, 1] = np.nan
    before = values.copy()
    safe = np.array([[True, True], [False, True]])
    rgb, science, effective = source.source_outputs(values, safe, display_mode="raw_log", parameters=parameters)
    assert np.array_equal(values, before, equal_nan=True)
    assert np.array_equal(effective, np.array([[True, False], [False, True]]))
    assert np.isnan(science[:, ~effective]).all()
    assert np.array_equal(science[:, effective], before[:, effective])
    assert np.array_equal(rgb[0, 1], np.ones(3, np.float32))


def test_baseline_assets_reject_mutually_consistent_replacement(tmp_path, parameters):
    rows = []
    for identity in ("r1", "virtual_reference"):
        baseline = tmp_path / (identity + "_baseline.npz")
        sidecar = baseline.with_suffix(".json")
        baseline.write_bytes(b"baseline")
        sidecar.write_bytes(b"{}")
        rows.append({"run": identity, "baseline_path": str(baseline),
                     "baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
                     "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest()})
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(json.dumps({"results": rows}).encode())
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    result = source.verify_calibration_baselines(manifest, digest, "r1", parameters=parameters)
    assert result["verified"] is True and len(result["assets"]) == 4
    (tmp_path / "r1_baseline.npz").write_bytes(b"replacement")
    (tmp_path / "r1_baseline.json").write_bytes(b'{"replacement":true}')
    with pytest.raises(ValueError, match="parameter asset changed"):
        source.verify_calibration_baselines(manifest, digest, "r1", parameters=parameters)


def test_sparse_rectangles_are_disjoint_and_minimal(parameters):
    mask = np.zeros((4, 5), bool)
    mask[0, 2] = mask[3, 1] = mask[1, 0] = mask[2, 4] = True
    rectangles = extension.peripheral_rectangles(mask, (2, 3), (-1, -1), parameters=parameters)
    assert rectangles == [(2, 0, 3, 1), (1, 3, 2, 4), (0, 1, 1, 2), (4, 2, 5, 3)]


def test_empty_ring_preserves_centre_and_invalid_exterior(parameters):
    centre = np.arange(18, dtype=np.float32).reshape(3, 2, 3)
    result, valid, audit = extension.extend_prepared_source(
        centre, {}, None, np.zeros((4, 5), bool), parameters=parameters
    )
    assert np.array_equal(result[:, 1:3, 1:4], centre)
    expected_valid = np.zeros((4, 5), bool)
    expected_valid[1:3, 1:4] = True
    assert np.array_equal(valid, expected_valid)
    assert np.isnan(result[:, ~valid]).all()
    assert audit["new_photometry_fit"] is False
    assert audit["finite_ring_pixels"] == 0


def test_recorded_zero_light_requires_no_refit(parameters):
    grid = SimpleNamespace(shape_hw=(2, 3), origin_cell_xy=(0, 0))
    assert extension.frozen_light_from_receipt(
        {"mode": "exact-source-bound-B", "zero_field": True}, grid, parameters=parameters
    ) is None
    with pytest.raises(ValueError, match="certified frame-light"):
        extension.frozen_light_from_receipt({}, grid, parameters=parameters)


def test_current_code_binding_cannot_accept_absent_or_partial_fields():
    digest = "a" * 64
    expected = {"processor": digest, "renderer": digest}
    source.verify_execution_identity(expected, expected, ["processor", "renderer"])
    for invalid in ({"unknown": None}, {"processor": digest}, {}):
        with pytest.raises(ValueError, match="exact current"):
            source.verify_execution_identity(expected, invalid, ["processor", "renderer"])
    with pytest.raises(ValueError, match="execution binding: renderer"):
        source.verify_execution_identity({"processor": digest}, expected, ["processor", "renderer"])


def test_relative_frame_transform_uses_recorded_baseline_maps(parameters):
    cohort = source.PhotoCohort.__new__(source.PhotoCohort)
    cohort.parameters = parameters
    previous = np.array([[1., 0., 2.], [0., 1., -3.], [0., 0., 1.]])
    current = np.array([[1., 0., 7.], [0., 1., 1.], [0., 0., 1.]])
    cohort.frame = lambda index: {"H_v5_B_to_frame_native": (previous if index == 0 else current).tolist()}
    assert np.array_equal(cohort.baseline_to_frame(1), current)
    assert np.array_equal(cohort.previous_to_frame(0, 1), np.array([[1., 0., 5.], [0., 1., 4.], [0., 0., 1.]]))
