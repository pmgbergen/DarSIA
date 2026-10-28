"""Exact floating self-alignment outside feature conversion's supported range."""

import numpy as np
import pytest

from darsia.corrections.shape.translation import TranslationEstimator


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("value", [1.25, -1.25])
def test_identical_outside_range_is_exact_zero_without_conversion(dtype, value, monkeypatch):
    """Equal finite RGB arrays need no clipped or normalized feature image."""
    image = np.full((8, 9, 3), value, dtype=dtype)
    before = image.copy()
    estimator = TranslationEstimator()

    def unused(*args, **kwargs):
        """Refuse accidental feature matching for the exact unsupported identity."""
        pytest.fail("exact out-of-range identity reached feature matching")

    monkeypatch.setattr(estimator, "_find_matching_transformation", unused)
    matrix, success = estimator.find_effective_translation(image, image.copy())
    np.testing.assert_array_equal(matrix, [[1., 0., 0.], [0., 1., 0.]])
    assert success is True
    np.testing.assert_array_equal(image, before)
    assert image.dtype == dtype


@pytest.mark.parametrize("implicit_destination", [False, True])
def test_same_offset_roi_keeps_absolute_zero(implicit_destination):
    """Equal ROIs at the same nonzero origin do not introduce an offset."""
    image = np.full((9, 10, 3), 1.5, dtype=np.float32)
    roi = (slice(2, 7), slice(3, 9))
    matrix, success = TranslationEstimator().find_effective_translation(
        image, image.copy(), roi, None if implicit_destination else roi)
    np.testing.assert_array_equal(matrix, [[1., 0., 0.], [0., 1., 0.]])
    assert success is True


def _matcher_calls(estimator, monkeypatch):
    """Record that the original matcher runs, without replacing its feature images."""
    calls = []
    original = estimator._find_matching_transformation

    def record(*args, **kwargs):
        calls.append(kwargs["transformation_type"])
        return original(*args, **kwargs)

    monkeypatch.setattr(estimator, "_find_matching_transformation", record)
    return calls


def test_unequal_outside_range_uses_the_matcher_with_values_read_as_white(monkeypatch):
    """A differing pixel prevents the shortcut; feature images read floats on [0, 1]."""
    source = np.full((8, 9, 3), 1.5, dtype=np.float32)
    destination = source.copy()
    destination[-1, -1, -1] = 1.6
    estimator = TranslationEstimator()
    calls = _matcher_calls(estimator, monkeypatch)
    # Both read as uniform white: no features, so no translation and no error.
    assert estimator.find_effective_translation(source, destination) == (None, False)
    assert calls == ["homography", "partial_affine"]


def test_equal_selected_roi_does_not_hide_a_different_full_array(monkeypatch):
    """The identity case requires equality outside the selected ROI as well."""
    source = np.full((8, 9, 3), 1.5, dtype=np.float32)
    destination = source.copy()
    destination[-1, -1, -1] = 1.6
    roi = (slice(1, 5), slice(1, 5))
    estimator = TranslationEstimator()
    calls = _matcher_calls(estimator, monkeypatch)
    assert estimator.find_effective_translation(source, destination, roi, roi) == (None, False)
    assert calls == ["homography", "partial_affine"]


@pytest.mark.parametrize("case", ["different_roi", "mask", "dtype", "nonfinite", "valid_roi"])
def test_other_inputs_retain_original_matcher_path(case, monkeypatch):
    """ROI, mask, dtype, finite and selected-range restrictions preserve the old path."""
    source = np.full((8, 9, 3), 1.5, dtype=np.float32)
    destination = source.copy()
    roi_src = roi_dst = None
    mask = None
    if case == "different_roi":
        roi_src, roi_dst = (slice(1, 6), slice(1, 6)), (slice(2, 7), slice(1, 6))
    elif case == "mask":
        mask = np.ones(source.shape[:2], dtype=bool)
    elif case == "dtype":
        destination = destination.astype(np.float64)
    elif case == "nonfinite":
        source[-1, -1, -1] = destination[-1, -1, -1] = np.inf
    else:
        source[:4, :4] = destination[:4, :4] = .5
        roi_src = roi_dst = (slice(0, 4), slice(0, 4))
    estimator = TranslationEstimator()
    calls = []

    def original_path(*args, **kwargs):
        """Record both original transform attempts without replacing their caller."""
        calls.append(kwargs["transformation_type"])
        return None, False, None

    monkeypatch.setattr(estimator, "_find_matching_transformation", original_path)
    assert estimator.find_effective_translation(source, destination, roi_src, roi_dst,
        mask_src=mask, mask_dst=mask) == (None, False)
    assert calls == ["homography", "partial_affine"]


@pytest.mark.parametrize("dtype,value", [(np.float32, .5), (np.uint8, 128)])
def test_valid_domain_identical_rgb_still_uses_original_orb_path(dtype, value, monkeypatch):
    """Normal supported floating images retain both original feature attempts."""
    image = np.full((8, 9, 3), value, dtype=dtype)
    estimator = TranslationEstimator()
    calls = []

    def original_path(*args, **kwargs):
        """Observe that valid-domain identity still enters the original matcher."""
        calls.append(kwargs["transformation_type"])
        return None, False, None

    monkeypatch.setattr(estimator, "_find_matching_transformation", original_path)
    assert estimator.find_effective_translation(image, image.copy()) == (None, False)
    assert calls == ["homography", "partial_affine"]


def test_valid_domain_real_matcher_failure_remains_unchanged():
    """A featureless supported image retains the original unsuccessful result."""
    image = np.full((8, 9, 3), .5, dtype=np.float32)
    assert TranslationEstimator().find_effective_translation(image, image.copy()) == (None, False)
