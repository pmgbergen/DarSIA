"""Feature extraction reads float images on [0, 1], like an 8-bit photograph."""
import numpy as np
import skimage

from darsia.utils.features import FeatureDetection


def _texture(low=0.0, high=1.0):
    rng = np.random.default_rng(3)
    coarse = rng.uniform(low, high, (30, 40, 3))
    return np.kron(coarse, np.ones((4, 4, 1))).astype(np.float32)


def _same(left, right):
    (kps_a, descs_a), found_a = left
    (kps_b, descs_b), found_b = right
    assert found_a and found_b
    assert [kp.pt for kp in kps_a] == [kp.pt for kp in kps_b]
    np.testing.assert_array_equal(descs_a, descs_b)


def test_linear_values_above_one_are_read_as_white():
    image = _texture()
    image[40:70, 50:90] = 1.25  # Saturated linear values with demosaic overshoot.
    _same(FeatureDetection.extract_features(image),
          FeatureDetection.extract_features(np.clip(image, 0, 1)))


def test_float_images_within_range_convert_exactly_as_before():
    image = _texture(-0.2, 1.0)
    _same(FeatureDetection.extract_features(image),
          FeatureDetection.extract_features(skimage.img_as_ubyte(image)))


def test_input_image_is_not_modified():
    image = _texture()
    image[0, 0] = 2.0
    before = image.copy()
    FeatureDetection.extract_features(image, roi=(slice(0, 60), slice(0, 80)))
    np.testing.assert_array_equal(image, before)
