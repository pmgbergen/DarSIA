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


def test_eval_scalar_image_voxel():
    """Test eval method with a scalar 2D image and Voxel input."""
    arr = np.arange(12, dtype=float).reshape(3, 4)
    image = darsia.ScalarImage(arr, dimensions=[1, 1], space_dim=2)

    # Single Voxel
    voxel = darsia.Voxel([1, 2])
    result = image.eval(voxel)
    assert result == arr[1, 2]


def test_eval_scalar_image_voxel_array():
    """Test eval method with a scalar 2D image and VoxelArray input."""
    arr = np.arange(12, dtype=float).reshape(3, 4)
    image = darsia.ScalarImage(arr, dimensions=[1, 1], space_dim=2)

    # VoxelArray with two points
    voxels = darsia.VoxelArray([[0, 1], [2, 3]])
    result = image.eval(voxels)
    assert np.allclose(result, [arr[0, 1], arr[2, 3]])


def test_eval_scalar_image_coordinate():
    """Test eval method with a scalar 2D image and Coordinate input."""
    arr = np.arange(12, dtype=float).reshape(3, 4)
    image = darsia.ScalarImage(arr, dimensions=[1, 1], space_dim=2)

    # With dimensions=[1,1] and space_dim=2, voxel_size_x=0.25 (4 cols) and
    # voxel_size_y=1/3 (3 rows). The y-axis is reversed (origin at [0,1]).
    # Coordinate x=0.6 falls in column 2 (x in [0.5, 0.75)), and
    # y=0.5 falls in row 1 (y in (0.333, 0.667] in the reversed y-axis).
    coord = darsia.Coordinate(np.array([0.6, 0.5]))
    result = image.eval(coord)
    assert result == arr[1, 2]


def test_eval_scalar_image_coordinate_array():
    """Test eval method with a scalar 2D image and CoordinateArray input."""
    arr = np.arange(12, dtype=float).reshape(3, 4)
    image = darsia.ScalarImage(arr, dimensions=[1, 1], space_dim=2)

    # Use coordinates clearly inside target voxels:
    # [0.35, 0.8] -> voxel [0, 1], [0.85, 0.1] -> voxel [2, 3]
    coords = darsia.CoordinateArray(np.array([[0.35, 0.8], [0.85, 0.1]]))
    result = image.eval(coords)
    assert np.allclose(result, [arr[0, 1], arr[2, 3]])


def test_eval_nonscalar_image_voxel():
    """Test eval method with a non-scalar (e.g., RGB) 2D image and Voxel input."""
    arr = np.arange(36, dtype=float).reshape(3, 4, 3)
    image = darsia.Image(arr, scalar=False, dimensions=[1, 1], space_dim=2)

    voxel = darsia.Voxel([1, 2])
    result = image.eval(voxel)
    assert np.allclose(result, arr[1, 2])


def test_eval_nonscalar_image_voxel_array():
    """Test eval method with a non-scalar 2D image and VoxelArray input."""
    arr = np.arange(36, dtype=float).reshape(3, 4, 3)
    image = darsia.Image(arr, scalar=False, dimensions=[1, 1], space_dim=2)

    voxels = darsia.VoxelArray([[0, 1], [2, 3]])
    result = image.eval(voxels)
    assert np.allclose(result, arr[[0, 2], [1, 3]])


def test_eval_clipping():
    """Test eval method clips out-of-bounds voxel indices to valid range."""
    arr = np.arange(12, dtype=float).reshape(3, 4)
    image = darsia.ScalarImage(arr, dimensions=[1, 1], space_dim=2)

    # Voxel exceeding bounds should be clipped
    voxel = darsia.Voxel([10, 20])
    result = image.eval(voxel)
    assert result == arr[2, 3]
