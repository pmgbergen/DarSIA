"""Unit tests for darsia.Resize."""

import numpy as np

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
