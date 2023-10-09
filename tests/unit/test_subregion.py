"""Module testing functionlity to extract subregions in space and time.

"""

import numpy as np

import darsia


def test_time_interval_scalar_image_2d():
    """Test whether one can extract time intervals correctly from space-time images."""

    # Construct space-time image
    meta = {
        "space_dim": 2,
        "series": True,
        "time": [0, 1, 2, 3, 4],
    }
    arr = np.random.rand(3, 4, 5)
    image = darsia.ScalarImage(arr, **meta)

    # Construct subimage
    sub_image = image.time_interval(slice(1, 3))

    # Test whether the dimensions of the image array are correct
    assert np.allclose(sub_image.shape, (3, 4, 2))

    # Test whether the data is correct
    assert np.allclose(sub_image.img, arr[:, :, 1:3])

    # Test whether the time is restricted correctly
    assert np.allclose(sub_image.time, [1, 2])


def test_time_interval_scalar_image_2d_reset_time():
    """Test whether one can extract time intervals correctly from space-time images."""

    # Construct space-time image
    meta = {
        "space_dim": 2,
        "series": True,
        "time": [0, 1, 2, 3, 4],
    }
    arr = np.random.rand(3, 4, 5)
    image = darsia.ScalarImage(arr, **meta)

    # Construct subimage
    sub_image = image.time_interval(slice(1, 3))
    sub_image.reset_reference_time()

    # Test whether the dimensions of the image array are correct
    assert np.allclose(sub_image.shape, (3, 4, 2))

    # Test whether the data is correct
    assert np.allclose(sub_image.img, arr[:, :, 1:3])

    # Test whether the time is restricted correctly
    assert np.allclose(sub_image.time, [0, 1])


def test_time_interval_advanced():
    """Test whether one can extract time intervals correctly from space-time images."""

    # Construct space-time image
    meta = {
        "space_dim": 3,
        "series": True,
        "scalar": False,
        "time": [0, 1, 2, 3, 4],
    }
    arr = np.random.rand(3, 4, 5, 6, 7)
    image = darsia.Image(arr, **meta)

    # Construct subimage
    sub_image = image.time_interval(slice(1, 3))

    # Test whether the dimensions of the image array are correct
    assert np.allclose(sub_image.shape, (3, 4, 5, 2, 7))

    # Test whether the data is correct
    assert np.allclose(sub_image.img, arr[:, :, :, 1:3, :])

    # Test whether the time is restricted correctly
    assert np.allclose(sub_image.time, [1, 2])
