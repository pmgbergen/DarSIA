"""Physical area examples that distinguish area from pixel-count filtering."""

import numpy as np
import pytest

from darsia.fluidflower.phase import metrics


@pytest.mark.parametrize("zero_parent", [False, True])
def test_categorical_replay_preserves_original_exact_area_table(zero_parent):
    import ast
    from pathlib import Path
    import subprocess

    source = subprocess.check_output(
        [
            "git",
            "show",
            "f0d338aa2be5e6ee8be6a2db0bd8f6fb70e9bd5a:src/darsia/fluidflower/phase_ffac_selected_phase.py",
        ],
        cwd=Path(__file__).resolve().parents[2],
    )
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "replay_difference"
    )
    env = dict(np=np, CO2_REPLAY_LIMIT=0.001, GAS_REPLAY_LIMIT=0.01)
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "preserved replay", "exec"),
        env,
    )
    parent = np.zeros((19, 23), np.uint8)
    if not zero_parent:
        parent[2:14, 4:19] = 1
        parent[5:10, 7:13] = 2
    parent[-1, -1] = 255
    current = parent.copy()
    current[5, 7] = 2
    current[3, 4] = 0
    current[-1, -1] = 0
    area = np.linspace(0.01, 0.7, parent.size, dtype=np.float32).reshape(parent.shape)
    identity = dict(run="test", frame_index=1, source_sha256="a" * 64, elapsed_h=0.17)
    params = dict(
        allowed_labels=[0, 1, 2, 255],
        co2_replay_limit=0.001,
        gas_replay_limit=0.01,
        square_millimetres_per_square_centimetre=100,
    )
    actual = metrics.categorical_replay(
        parent, current, area, identity=identity, parameters=params
    )
    expected = env["replay_difference"](parent, current, area, **identity)
    assert actual == expected
    assert actual["support_changed_pixels"] == 1
    if zero_parent:
        assert actual["gas"]["changed_area_fraction"] is None
        assert actual["gas"]["pass_gate"] is False


def test_array_identity_binds_values_shape_and_dtype_independently_of_layout():
    values = np.array([[1, 2], [3, 4]], np.uint8)
    digest = metrics.array_identity(values)
    assert digest == metrics.array_identity(np.asfortranarray(values))
    assert digest != metrics.array_identity(values.reshape(1, 4))
    assert digest != metrics.array_identity(values.view(np.int8))
    changed = values.copy()
    changed[0, 0] = 5
    assert digest != metrics.array_identity(changed)
    assert np.array_equal(values, np.array([[1, 2], [3, 4]], np.uint8))


def test_pair_identity_distinguishes_signed_zero_and_reports_first_index():
    actual = np.zeros((2, 3), np.float32)
    expected = actual.copy()
    expected[0, 2] = expected[1, 0] = -0.0
    assert np.array_equal(actual, expected)
    result = metrics.array_pair_identity(actual, expected)
    assert result["bit_identical"] is False
    assert result["changed_elements"] == 2
    assert result["first_different_index"] == [0, 2]
    assert result["actual_array_sha256"] != result["expected_array_sha256"]


def test_pair_identity_compares_nan_payload_bits_and_not_numeric_nan_equality():
    actual = np.array([0x7FC00001, 0x7FC00002], np.uint32).view(np.float32)
    expected = np.array([0x7FC00001, 0x7FC00003], np.uint32).view(np.float32)
    assert np.isnan(actual).all() and np.isnan(expected).all()
    result = metrics.array_pair_identity(actual, expected)
    assert not result["bit_identical"] and result["changed_elements"] == 1
    assert result["first_different_index"] == [1]
    same = metrics.array_pair_identity(actual, actual.copy())
    assert same["bit_identical"] and same["changed_elements"] == 0
    assert same["first_different_index"] is None


@pytest.mark.parametrize("mismatch", ["dtype", "shape"])
def test_pair_identity_rejects_grid_mismatch_even_when_raw_byte_hashes_match(mismatch):
    actual = np.arange(4, dtype=np.uint8).reshape(2, 2)
    expected = actual.view(np.int8) if mismatch == "dtype" else actual.reshape(1, 4)
    result = metrics.array_pair_identity(actual, expected)
    assert not result["bit_identical"]
    assert result["changed_elements"] is None
    assert result["first_different_index"] is None
    assert result["actual_array_sha256"] == result["expected_array_sha256"]
    assert result["actual_shape"] == [2, 2]
    assert result["actual_dtype"] == "uint8"


def test_area_validation_retains_the_original_floating_grid():
    values = np.full((2, 3), 0.125, np.float32)
    assert metrics.validate_pixel_area(values, (2, 3)) is values
    assert metrics.validate_pixel_area(np.float32(0.125), (2, 3)) == 0.125
    for invalid in (np.zeros((2, 3)), np.ones((2, 3), int), np.ones((3, 2))):
        with pytest.raises(ValueError, match="grid"):
            metrics.validate_pixel_area(invalid, (2, 3))
    with pytest.raises(ValueError, match="positive"):
        metrics.validate_pixel_area(float("nan"), (2, 3))


def test_area_reductions_use_float64_weights_and_explicit_units():
    mask = np.array([[True, False, True], [False, True, False]])
    area = np.array([[1, 20, 3], [40, 5, 60]], np.float32)
    assert metrics.sum_pixel_area(mask, area) == 9
    assert metrics.sum_pixel_area(mask, area, unit="cm2") == 0.09
    assert metrics.sum_pixel_area(mask, 2, unit="m2") == 6e-6
    with pytest.raises(ValueError, match="unit"):
        metrics.sum_pixel_area(mask, area, unit="unknown")


@pytest.mark.parametrize(
    "function", [metrics.filter_components_by_area, metrics.area_filtered_gas]
)
def test_component_cutoff_uses_weighted_area_and_includes_the_boundary(function):
    mask = np.zeros((4, 6), bool)
    mask[0, :2] = True
    mask[3, 2:5] = True
    area = np.ones(mask.shape, np.float32)
    area[0, :2] = 1.25
    area[3, 2:5] = 0.6
    result, audit = function(mask, area, minimum_area_mm2=2.5, connectivity=4)
    assert result[0, :2].all()
    assert not result[3, 2:5].any()
    assert audit["withdrawn_components"] == 1
    assert audit["withdrawn_pixels"] == 3
    assert np.array_equal(mask[3, 2:5], np.ones(3, bool))
    with pytest.raises(ValueError, match="four"):
        function(mask, area, minimum_area_mm2=2.5, connectivity=8)


def test_diagonal_gas_pixels_are_separate_and_zero_cutoff_returns_a_copy():
    mask = np.eye(3, dtype=bool)
    result, _ = metrics.filter_components_by_area(
        mask, 1.0, minimum_area_mm2=1.5, connectivity=4
    )
    assert not result.any()
    result, _ = metrics.area_filtered_gas(mask, 1.0, minimum_area_mm2=0, connectivity=4)
    assert np.array_equal(result, mask) and not np.shares_memory(result, mask)
