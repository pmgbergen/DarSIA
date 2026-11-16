"""Unit tests for darsia.Resize."""

import cv2
import numpy as np
import pytest

import darsia


def test_equalize_voxel_size_default():
    """Test equalization of voxel size for image with unequal voxel size."""

    # Create image with shape and dimensions not complying with each other.
    array = np.ones((10, 10))
    image = darsia.Image(array, width=2, height=1, scalar=True)

    # Check that voxel size unequal
    original_voxel_size = image.voxel_size
    assert not np.isclose(original_voxel_size[0], original_voxel_size[1])

    # Transform to same voxel size (using the default option)
    resized_image = darsia.equalize_voxel_size(image)
    resized_voxel_size = resized_image.voxel_size
    assert np.isclose(resized_voxel_size[0], resized_voxel_size[1])

    # Check whether the dimensions are still the same
    original_dimensions = image.dimensions
    resized_dimensions = resized_image.dimensions
    assert np.allclose(original_dimensions, resized_dimensions)


def test_equalize_voxel_size():
    """Test equalization of voxel size for image with unequal voxel size."""

    # Create image with shape and dimensions not complying with each other.
    array = np.ones((10, 10))
    image = darsia.Image(array, width=2, height=1, scalar=True)

    # Check that voxel size unequal
    original_voxel_size = image.voxel_size
    assert not np.isclose(original_voxel_size[0], original_voxel_size[1])

    # Transform to same voxel size (using the default option)
    resized_image = darsia.equalize_voxel_size(image, voxel_size=0.05)
    resized_voxel_size = resized_image.voxel_size
    assert np.isclose(resized_voxel_size[0], resized_voxel_size[1])

    # Check whether the dimensions are still the same
    original_dimensions = image.dimensions
    resized_dimensions = resized_image.dimensions
    assert np.allclose(original_dimensions, resized_dimensions)


def test_resize_extensive_image_default_is_conservative():
    """Test default resize behavior for extensive images."""
    array = np.arange(1, 17, dtype=float).reshape((4, 4))
    image = darsia.ExtensiveImage(array, dimensions=[1, 1], scalar=True)

    resized_image = darsia.resize(image, shape=(2, 2))

    assert isinstance(resized_image, darsia.ExtensiveImage)
    assert np.isclose(np.sum(resized_image.img), np.sum(image.img))


def _per_channel_resize(array, dsize, interpolation):
    """Reference: each channel resized on its own (the original implementation)."""
    shape = array.shape
    channels = np.reshape(array, (*shape[:2], -1))
    resized = [
        (
            cv2.resize(channel, dsize)
            if interpolation is None
            else cv2.resize(channel, dsize, interpolation=interpolation)
        )
        for channel in cv2.split(channels)
    ]
    merged = cv2.merge(resized)
    return np.reshape(merged, (*merged.shape[:2], *shape[2:]))


FLAGS = {
    None: None,
    "inter_area": cv2.INTER_AREA,
    "inter_linear": cv2.INTER_LINEAR,
    "inter_nearest": cv2.INTER_NEAREST,
}

# Image sizes and targets: the result of OpenCV can depend on the size (vector widths)
SIZES = [(h, w) for h in range(5, 40, 7) for w in range(5, 60, 9)]

LAST_DIGIT = 1e-7
"""Accepted difference for float32 images with 3 or 4 channels and linear interpolation
(one step in the last digit of float32 for values in [0, 1] is 6e-8)."""


def _assert_resized_as_expected(resized, expected, interpolation, context=None):
    """Byte-identical, except for the accepted float32 case (see `_single_call`)."""
    assert resized.dtype == expected.dtype
    assert resized.shape == expected.shape
    accepted = (
        resized.dtype == np.float32
        and resized.ndim == 3
        and resized.shape[2] in (3, 4)
        and interpolation in (None, "inter_linear")
    )
    if accepted:
        difference = np.abs(resized.astype(float) - expected.astype(float))
        assert difference.max() <= LAST_DIGIT, context
    else:
        assert resized.tobytes() == expected.tobytes(), context


@pytest.mark.parametrize("channels", [None, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.uint8, np.uint16])
@pytest.mark.parametrize("interpolation", list(FLAGS))
def test_resize_equals_resizing_each_channel(channels, dtype, interpolation):
    """Some images are resized in one OpenCV call (see `_single_call` in
    `restoration/resize.py`). This must give the result of resizing each channel (up to
    the accepted float32 case); the test fails on an OpenCV build where it does not."""
    rng = np.random.default_rng(0)
    for rows, cols in SIZES:
        full_shape = (rows, cols) if channels is None else (rows, cols, channels)
        array = rng.random(full_shape)
        if np.issubdtype(dtype, np.integer):
            array = np.round(array * np.iinfo(dtype).max)
        array = array.astype(dtype)
        for shape in [
            (2 * rows + 3, 3 * cols + 1),
            (max(1, rows // 3), max(1, cols // 2)),
        ]:
            expected = _per_channel_resize(
                array, tuple(reversed(shape)), FLAGS[interpolation]
            )
            resized = darsia.Resize(shape=shape, interpolation=interpolation)(array)
            _assert_resized_as_expected(
                resized, expected, interpolation, (rows, cols, shape)
            )


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.uint8])
def test_resize_of_images_and_views(dtype):
    """Images and views which are not contiguous give the result of a contiguous copy
    (which the resizing used to make first)."""
    rng = np.random.default_rng(1)
    array = rng.random((61, 83, 3))
    if dtype is np.uint8:
        array = np.round(array * 255)
    array = array.astype(dtype)
    for interpolation in FLAGS:
        for shape in [(120, 170), (20, 30)]:
            dsize = tuple(reversed(shape))
            resize = darsia.Resize(shape=shape, interpolation=interpolation)
            image = darsia.Image(array, dimensions=[1, 1])
            expected = _per_channel_resize(array, dsize, FLAGS[interpolation])
            _assert_resized_as_expected(resize(image).img, expected, interpolation)
            view = array[::2, ::2]
            expected_view = _per_channel_resize(
                np.ascontiguousarray(view), dsize, FLAGS[interpolation]
            )
            _assert_resized_as_expected(resize(view), expected_view, interpolation)


def test_resize_to_same_shape_returns_a_copy():
    """Without resizing, the result must not share the data of the input."""
    array = np.arange(24, dtype=float).reshape((2, 4, 3))
    image = darsia.Image(array, dimensions=[1, 1])
    for result in (
        darsia.Resize(shape=(2, 4))(array),
        darsia.resize(image, shape=(2, 4)).img,
    ):
        assert np.array_equal(result, array)
        assert not np.shares_memory(result, array)
    # Resizing never modifies the input
    darsia.resize(image, shape=(4, 8))
    assert np.array_equal(image.img, np.arange(24, dtype=float).reshape((2, 4, 3)))


def test_resize_object_extensive_image_default_is_conservative():
    """Test default Resize object behavior for extensive images."""
    array = np.arange(1, 17, dtype=float).reshape((4, 4))
    image = darsia.ExtensiveImage(array, dimensions=[1, 1], scalar=True)

    resized_image = darsia.Resize(shape=(2, 2))(image)

    assert isinstance(resized_image, darsia.ExtensiveImage)
    assert np.isclose(np.sum(resized_image.img), np.sum(image.img))
