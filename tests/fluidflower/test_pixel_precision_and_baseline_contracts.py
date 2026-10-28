"""Local behavioral cases for declared pixel arithmetic and baseline rebinding."""
import copy

import numpy as np
import pytest

from darsia.fluidflower.mass.production_precision import PixelMath
from darsia.fluidflower.phase import features
from tests.fluidflower.test_phase_features_heads_regional import PARAMETERS


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_relative_rgb_modes_preserve_sign_and_declared_precision(precision):
    math = PixelMath(precision)
    current = math.pixels([[[2.0, 0.5, 1.0]]])
    baseline = math.pixels([[[1.0, 1.0, 1.0]]])
    expected = {
        "plain": [[[1.0, -0.5, 0.0]]],
        "positive": [[[1.0, 0.0, 0.0]]],
        "negative": [[[0.0, 0.5, 0.0]]],
        "absolute": [[[1.0, 0.5, 0.0]]],
    }
    before = current.copy(), baseline.copy()
    for mode, values in expected.items():
        got = math.relative_rgb(current, baseline, option=mode)
        assert got.dtype == math.dtype
        np.testing.assert_array_equal(got, math.pixels(values))
    np.testing.assert_array_equal(current, before[0])
    np.testing.assert_array_equal(baseline, before[1])
    with pytest.raises(ValueError, match="Unknown native"):
        math.relative_rgb(current, baseline, option="changed")
    with pytest.raises(ValueError, match="grids differ"):
        math.relative_rgb(current, baseline[:, :0])
    with pytest.raises(TypeError, match="declared working dtype"):
        math.relative_rgb(current.astype(np.int32), baseline)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_integrate_excludes_outside_domain_and_uses_float64_weights(precision):
    math = PixelMath(precision)
    value = math.pixels([[1.0, 2.0], [4.0, 8.0]])
    weight = np.asarray([[0.5, 2.0], [3.0, 0.25]], np.float64)
    domain = np.asarray([[True, False], [True, True]])
    before = value.copy(), weight.copy(), domain.copy()
    result = math.integrate(value, weight, domain)
    assert result.dtype == np.float64
    assert result == 14.5
    for actual, original in zip((value, weight, domain), before):
        np.testing.assert_array_equal(actual, original)
    with pytest.raises(ValueError, match="domain mask"):
        math.integrate(value, weight, domain.astype(np.uint8))
    with pytest.raises(ValueError, match="domain mask"):
        math.integrate(value, weight, domain[:, :1])


def test_relative_grain_evidence_is_scale_invariant_and_separates_new_contrast():
    parameters = {"optical": {"relative_grain_floor": 0.25,
                              "relative_grain_min": -1.0,
                              "relative_grain_max": 1.0}}
    baseline = np.asarray([[1.5, 0.5]], np.float32)
    current = np.asarray([[2.0, 0.75]], np.float32)
    smooth = np.ones_like(baseline)
    actual = features.relative_bright_grain_columns(
        baseline, smooth, current, smooth, parameters=parameters)
    expected = [[[0.5, -0.5]], [[1.0, -0.25]], [[0.5, 0.25]],
                [[0.5, 0.0]], [[0.5, 0.0]], [[0.2, 0.0]]]
    for output, values in zip(actual, expected):
        assert output.dtype == np.float32
        np.testing.assert_array_equal(output, np.asarray(values, np.float32))
    rescaled = features.relative_bright_grain_columns(
        baseline * np.float32(0.5), smooth * np.float32(0.5),
        current * np.float32(0.5), smooth * np.float32(0.5),
        parameters=parameters)
    for first, second in zip(actual, rescaled):
        np.testing.assert_array_equal(first, second)
    with pytest.raises(KeyError):
        features.relative_bright_grain_columns(
            baseline, smooth, current, smooth, parameters={})


def test_baseline_rebinding_clears_temporal_cache_and_keeps_corrected_pixels():
    parameters = copy.deepcopy(PARAMETERS["features"])
    contract = features.OpticalFeatureContract(
        **parameters["contract"]["defaults"], parameters=parameters)
    baseline = np.full((3, 4, 3), 0.5, np.float32)
    replacement = np.full_like(baseline, 0.25)
    builder = features.FeatureBuilder(
        contract, baseline, parameters=parameters,
        input_encoding=parameters["optical"]["unit_encoding"],
        block_rows=2, texture_sigma=0.5)
    builder.previous, builder.previous_key = object(), "prior-grid"
    builder.set_baseline(replacement)
    np.testing.assert_array_equal(builder.baseline["rgb"], replacement)
    assert builder.previous is None and builder.previous_key is None
    np.testing.assert_array_equal(baseline, np.full_like(baseline, 0.5))
    np.testing.assert_array_equal(replacement, np.full_like(replacement, 0.25))
    with pytest.raises(ValueError, match="shape changed"):
        builder.set_baseline(replacement[:2])
    review = features.FeatureBuilder(
        contract, np.full(baseline.shape, 64, np.uint8), parameters=parameters,
        input_encoding=parameters["optical"]["review_encoding"],
        block_rows=2, texture_sigma=0.5)
    with pytest.raises(ValueError, match="optical-only"):
        review.set_baseline(replacement)
