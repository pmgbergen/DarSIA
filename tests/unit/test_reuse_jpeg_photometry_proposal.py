"""Synthetic contracts for the unwired JPEG proposal; no camera fit or images."""

import numpy as np
import pytest

from darsia.fluidflower.reuse_jpeg_photometry_proposal import FixedSeriesJpegProposal


def proposal(**changes):
    options = dict(
        series_id="synthetic-series",
        encoded_knots=(np.array([10., 20., 40.]), np.array([5., 25.]), np.array([0., 50.])),
        linear_knots=(np.array([1., 3., 7.]), np.array([2., 6.]), np.array([0., 5.])),
        flatfield_rgb=np.array([[[2., 4., 8.], [4., 8., 16.]]]),
        flatfield_support=np.ones((1, 2, 3), dtype=bool),
        normalization_rgb=np.array([2., 2., 4.]),
        interpolation="piecewise_linear", outside_support="nan", output_dtype="float64",
    )
    options.update(changes)
    return FixedSeriesJpegProposal(**options)


def apply(model, values, mask=None):
    values = np.asarray(values)
    return model.apply(values, series_id="synthetic-series",
                       input_valid=np.ones(values.shape, dtype=bool) if mask is None else mask)


def test_measured_channel_arithmetic_and_explicit_flat_normalization():
    result = apply(proposal(), [[[15., 15., 25.], [30., 5., 50.]]])
    np.testing.assert_allclose(result.linear_rgb, [[[2., 4., 2.5], [5., 2., 5.]]])
    np.testing.assert_allclose(result.corrected_rgb, [[[2., 2., 1.25], [2.5, .5, 1.25]]])
    assert result.channel_valid.all() and result.valid_pixels.all()


def test_measured_endpoints_are_supported_without_synthetic_black_white_anchors():
    result = apply(proposal(), [[[10., 5., 0.], [40., 25., 50.]]])
    np.testing.assert_array_equal(result.linear_rgb, [[[1., 2., 0.], [7., 6., 5.]]])
    assert result.channel_valid.all()


def test_outside_measured_support_is_nan_with_channel_and_pixel_masks():
    result = apply(proposal(), [[[9., 15., 25.], [30., 26., 50.]]])
    np.testing.assert_array_equal(result.out_of_calibration, [[[1, 0, 0], [0, 1, 0]]])
    assert np.isnan(result.corrected_rgb[0, 0, 0])
    assert np.isnan(result.corrected_rgb[0, 1, 1])
    assert np.isfinite(result.corrected_rgb[0, 0, 1:]).all()
    assert not result.valid_pixels.any()


def test_outside_policy_can_explicitly_refuse_instead_of_extrapolating():
    with pytest.raises(ValueError, match="outside measured"):
        apply(proposal(outside_support="raise"), [[[9., 15., 25.], [30., 5., 50.]]])


def test_nonfinite_and_caller_excluded_samples_remain_distinct_from_support():
    values = np.array([[[np.nan, np.inf, 25.], [30., 5., 50.]]])
    mask = np.ones(values.shape, bool)
    mask[0, 1, 0] = False
    result = apply(proposal(), values, mask)
    assert result.invalid_input[0, 0, :2].all() and result.invalid_input[0, 1, 0]
    assert not result.out_of_calibration.any()
    assert np.isnan(result.corrected_rgb[0, 0, :2]).all()
    assert np.isnan(result.linear_rgb[0, 1, 0])


def test_flatfield_support_is_fixed_and_does_not_replace_invalid_values_by_zero():
    support = np.ones((1, 2, 3), bool)
    support[0, 0, 1] = False
    flat = np.array([[[2., np.nan, 8.], [4., 8., 16.]]])
    result = apply(proposal(flatfield_support=support, flatfield_rgb=flat),
                   [[[15., 15., 25.], [30., 5., 50.]]])
    assert result.flatfield_invalid[0, 0, 1]
    assert np.isnan(result.corrected_rgb[0, 0, 1])
    assert result.linear_rgb[0, 0, 1] == 4.
    np.testing.assert_array_equal(result.valid_pixels, [[False, True]])


def test_same_fixed_proposal_is_reused_without_fit_or_state_update():
    model = proposal()
    a = [[[15., 15., 25.], [30., 5., 50.]]]
    before = model.flatfield_rgb.copy()
    first = apply(model, a)
    apply(model, [[[10., 5., 0.], [40., 25., 50.]]])
    np.testing.assert_array_equal(first.corrected_rgb, apply(model, a).corrected_rgb)
    np.testing.assert_array_equal(model.flatfield_rgb, before)


def test_calibration_copies_caller_arrays_and_is_readonly():
    flat = np.ones((1, 2, 3))
    x = np.array([10., 20., 40.])
    model = proposal(flatfield_rgb=flat, encoded_knots=(x, np.array([5., 25.]), np.array([0., 50.])))
    flat[:] = 100.
    x[:] = -1.
    assert np.all(model.flatfield_rgb == 1.)
    np.testing.assert_array_equal(model.encoded_knots[0], [10., 20., 40.])
    with pytest.raises(ValueError):
        model.normalization_rgb[0] = 99.


def test_encoded_units_are_not_implicitly_divided_by_255():
    result = apply(proposal(), np.array([[[10, 5, 0], [40, 25, 50]]], dtype=np.uint8))
    np.testing.assert_array_equal(result.linear_rgb, [[[1., 2., 0.], [7., 6., 5.]]])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_output_precision_requires_explicit_choice(dtype):
    result = apply(proposal(output_dtype=dtype), [[[15., 15., 25.], [30., 5., 50.]]])
    assert result.corrected_rgb.dtype == np.dtype(dtype)
    assert result.linear_rgb.dtype == np.dtype(dtype)


def test_unrepresentable_output_is_invalid_nan_after_requested_cast():
    model = proposal(output_dtype="float32", normalization_rgb=np.full(3, 1e300))
    result = apply(model, [[[15., 15., 25.], [30., 5., 50.]]])
    assert result.invalid_result.all() and not result.channel_valid.any()
    assert np.isnan(result.corrected_rgb).all()


@pytest.mark.parametrize("changes, message", [
    ({"series_id": ""}, "series_id"),
    ({"interpolation": "spline"}, "piecewise_linear"),
    ({"outside_support": "clip"}, "outside_support"),
    ({"output_dtype": "uint8"}, "output_dtype"),
    ({"encoded_knots": (np.array([1., 2.]),)}, "Exactly three"),
    ({"encoded_knots": (np.array([10., 10., 40.]), np.array([5., 25.]), np.array([0., 50.]))}, "strictly increasing"),
    ({"linear_knots": (np.array([1., .5, 7.]), np.array([2., 6.]), np.array([0., 5.]))}, "nonnegative and monotone"),
    ({"linear_knots": (np.array([1., np.nan, 7.]), np.array([2., 6.]), np.array([0., 5.]))}, "finite"),
    ({"flatfield_rgb": np.zeros((1, 2, 3))}, "positive"),
    ({"flatfield_support": np.ones((1, 2, 3))}, "boolean"),
    ({"normalization_rgb": np.array([1., 0., 1.])}, "finite positive"),
    ({"normalization_rgb": np.ones((1, 3))}, "three finite"),
])
def test_invalid_calibration_is_refused_without_repair(changes, message):
    with pytest.raises(ValueError, match=message):
        proposal(**changes)


def test_series_grid_and_input_mask_must_match_explicitly():
    model = proposal()
    with pytest.raises(ValueError, match="different series"):
        model.apply(np.ones((1, 2, 3)), series_id="other", input_valid=np.ones((1, 2, 3), bool))
    with pytest.raises(ValueError, match="grid"):
        apply(model, np.ones((2, 1, 3)))
    with pytest.raises(ValueError, match="boolean"):
        apply(model, np.ones((1, 2, 3)), np.ones((1, 2), bool))
