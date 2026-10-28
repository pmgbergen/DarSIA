"""RAW positioning measures saved drift on the sRGB view its JPEG-built base expects.

The saved DriftCorrection bases come from camera JPEGs (sRGB encoded) and the
feature matcher works on 8-bit values. A dark linear RAW view finds too few
features there; its sRGB view finds the shift, and the shift warps the linear
values. Synthetic arrays only; no photograph is read.
"""

import cv2
import numpy as np
import pytest

import darsia
from darsia.fluidflower.native import raw_positioning as raw

ROI = (slice(20, 140), slice(20, 180))
SHIFT = np.float32([[1, 0, 3], [0, 1, -2]])


def scene_TESTONLY(peak=0.08):
    """A dark linear texture like a chart ROI, and the same scene shifted."""
    small = np.random.default_rng(3).random((20, 25)).astype(np.float32)
    linear = cv2.resize(small, (200, 160), interpolation=cv2.INTER_NEAREST) * peak + 0.002
    rgb = np.stack([linear] * 3, -1)
    moved = cv2.warpAffine(rgb, SHIFT, (200, 160), borderMode=cv2.BORDER_REFLECT)
    return rgb, moved


def drift_TESTONLY(rgb):
    return darsia.DriftCorrection(base=raw.drift_view(rgb), config={"roi": ROI, "active": True})


def test_drift_view_is_the_srgb_transfer_clipped_to_the_unit_range():
    linear = np.array([-0.5, 0.0, 0.0031308, 0.01, 0.18, 1.0, 1.3], np.float32)
    expected = np.where(np.clip(linear, 0, 1) <= 0.0031308, 12.92 * np.clip(linear, 0, 1),
                        1.055 * np.clip(linear, 0, 1) ** (1 / 2.4) - 0.055)
    view = raw.drift_view(linear)
    assert view.dtype == np.float32
    np.testing.assert_allclose(view, expected, rtol=1e-6, atol=1e-7)
    assert view[0] == 0 and view[-1] == view[-2] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(view) >= 0)


def test_a_dark_linear_roi_finds_no_drift_but_its_srgb_view_finds_the_shift():
    rgb, moved = scene_TESTONLY()
    estimator = drift_TESTONLY(rgb).translation_estimator
    base = raw.drift_view(rgb)
    _translation, linear_ok = estimator.find_effective_translation(moved[ROI], base, None, ROI)
    translation, srgb_ok = estimator.find_effective_translation(raw.drift_view(moved[ROI]), base, None, ROI)
    assert not linear_ok
    assert srgb_ok
    np.testing.assert_allclose(translation[:, :2], np.eye(2), atol=1e-12)
    np.testing.assert_allclose(translation[:, 2], [-3, 2], atol=0.5)


def test_native_review_measures_drift_on_the_srgb_view_and_warps_linear_values(monkeypatch):
    rgb, moved = scene_TESTONLY()
    correction = drift_TESTONLY(rgb)
    original = correction.translation_estimator.find_effective_translation
    seen = []

    def spy(img_src, img_dst, roi_src=None, roi_dst=None, *args):
        result = original(img_src, img_dst, roi_src, roi_dst, *args)
        seen.append((img_src, roi_src, roi_dst, result))
        return result

    monkeypatch.setattr(correction.translation_estimator, "find_effective_translation", spy)
    scale = 15580
    camera = np.ascontiguousarray(np.moveaxis(moved, -1, 0) * np.float32(scale), np.float32)
    review = raw.native_review_rgb(camera, [correction], white_range_dn=scale, shape_hw=(160, 200),
                                   image_metadata=dict(height=1., width=1.))
    (view, roi_src, roi_dst, (translation, intact)), = seen
    assert intact and roi_src is None and roi_dst == ROI
    np.testing.assert_array_equal(view, raw.drift_view((camera / np.float32(scale)).transpose(1, 2, 0)[ROI]))
    linear = np.moveaxis(camera, 0, -1) / np.float32(scale)
    expected = np.moveaxis(cv2.warpAffine(linear, translation, (200, 160)), -1, 0) * np.float32(scale)
    np.testing.assert_array_equal(review, expected.astype(np.float32))


def test_values_above_white_in_the_roi_no_longer_raise():
    rgb, moved = scene_TESTONLY(peak=1.3)  # Demosaic overshoot above white.
    correction = drift_TESTONLY(np.clip(rgb, 0, 1))
    # The 8-bit feature conversion reads linear values above white as white.
    correction.translation_estimator.find_effective_translation(moved[ROI], correction.base, None, ROI)
    image = darsia.OpticalImage(moved.copy(), color_space="RGB", height=1., width=1.)
    raw._apply_drift(image, correction)
    assert image.img.shape == moved.shape and np.isfinite(image.img).all()


def test_linear_drift_base_measures_frames_with_values_above_white():
    """A RAW-built (linear) base sees linear frame values; overshoot reads as white."""
    rgb, moved = scene_TESTONLY(peak=1.3)
    correction = darsia.DriftCorrection(base=np.clip(rgb, 0, 1),
                                        config={"roi": ROI, "active": True, "linear_input": True})
    assert moved[ROI].max() > 1
    image = darsia.OpticalImage(moved.copy(), color_space="RGB", height=1., width=1.)
    raw._apply_drift(image, correction)
    expected, intact = correction.translation_estimator.find_effective_translation(
        np.clip(moved[ROI], 0, 1), correction.base, None, ROI)
    assert intact
    np.testing.assert_allclose(expected[:, 2], [-3, 2], atol=0.5)
    np.testing.assert_array_equal(image.img, cv2.warpAffine(moved, expected, (200, 160)))


def test_an_inactive_drift_leaves_the_linear_values_unchanged():
    rgb, moved = scene_TESTONLY()
    correction = darsia.DriftCorrection(base=raw.drift_view(rgb), config={"roi": ROI, "active": False})
    camera = np.ascontiguousarray(np.moveaxis(moved, -1, 0) * np.float32(1000), np.float32)
    review = raw.native_review_rgb(camera, [correction], white_range_dn=1000, shape_hw=(160, 200),
                                   image_metadata=dict(height=1., width=1.))
    np.testing.assert_allclose(review, camera, rtol=1e-6)
