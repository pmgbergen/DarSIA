"""Module for testing the patches."""

import numpy as np

import darsia


def test_patches():
    """Test patch extraction and assembly."""
    # Create image.
    array = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]])
    image = darsia.Image(array, space_dim=2)

    # Extract patches
    patches = darsia.Patches(image, num_patches=(2, 4))

    # Check that the patches are correct
    assert np.allclose(patches(0, 0).img, np.array([[1, 2]]))
    assert np.allclose(patches(0, 1).img, np.array([[3, 4]]))
    assert np.allclose(patches(0, 2).img, np.array([[5, 6]]))
    assert np.allclose(patches(0, 3).img, np.array([[7, 8]]))
    assert np.allclose(patches(1, 0).img, np.array([[9, 10]]))
    assert np.allclose(patches(1, 1).img, np.array([[11, 12]]))
    assert np.allclose(patches(1, 2).img, np.array([[13, 14]]))
    assert np.allclose(patches(1, 3).img, np.array([[15, 16]]))

    # Check that assemble works
    assert np.allclose(patches.assemble().img, array)
