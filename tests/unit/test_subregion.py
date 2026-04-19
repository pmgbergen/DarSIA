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


# ! ---- Tests for linear interpolation mode


def test_eval_linear_at_voxel_center_matches_nearest():
    """Linear interpolation at an exact voxel center must equal the nearest result."""
    arr = np.arange(12, dtype=float).reshape(3, 4)
    image = darsia.ScalarImage(arr, dimensions=[1, 1], space_dim=2)

    # Coordinate that maps exactly to fractional voxel [1, 2]:
    #   frac_col = x / 0.25          =>  x = 2 * 0.25 = 0.5
    #   frac_row = (1 - y) * 3 = 1   =>  y = 1 - (1/3) = 2/3
    coord = darsia.Coordinate(np.array([0.5, 2.0 / 3.0]))
    result_nearest = image.eval(coord, interpolation="nearest")
    result_linear = image.eval(coord, interpolation="linear")
    assert np.isclose(result_nearest, arr[1, 2])
    assert np.isclose(result_linear, arr[1, 2])


def test_eval_linear_midpoint_two_columns():
    """Linear interpolation at mid-point between two column-neighbours."""
    arr = np.arange(12, dtype=float).reshape(3, 4)
    image = darsia.ScalarImage(arr, dimensions=[1, 1], space_dim=2)

    # Fractional voxel [1, 2.5]:
    #   frac_col = 2.5  =>  x = 2.5 * 0.25 = 0.625
    #   frac_row = 1    =>  y = 1 - (1/3) = 2/3
    coord = darsia.Coordinate(np.array([0.625, 2.0 / 3.0]))
    result = image.eval(coord, interpolation="linear")
    expected = 0.5 * arr[1, 2] + 0.5 * arr[1, 3]
    assert np.isclose(result, expected)


def test_eval_linear_midpoint_two_rows():
    """Linear interpolation at mid-point between two row-neighbours."""
    arr = np.arange(12, dtype=float).reshape(3, 4)
    image = darsia.ScalarImage(arr, dimensions=[1, 1], space_dim=2)

    # Fractional voxel [1.5, 2]:
    #   x = 2 * 0.25 = 0.5,  y = 1 - 1.5/3 = 0.5
    coord = darsia.Coordinate(np.array([0.5, 0.5]))
    result = image.eval(coord, interpolation="linear")
    expected = 0.5 * arr[1, 2] + 0.5 * arr[2, 2]
    assert np.isclose(result, expected)


def test_eval_linear_midpoint_four_voxels():
    """Linear interpolation at the centre of four neighbouring voxels."""
    arr = np.arange(12, dtype=float).reshape(3, 4)
    image = darsia.ScalarImage(arr, dimensions=[1, 1], space_dim=2)

    # Fractional voxel [1.5, 2.5]:
    #   x = 2.5 * 0.25 = 0.625,  y = 1 - 1.5/3 = 0.5
    coord = darsia.Coordinate(np.array([0.625, 0.5]))
    result = image.eval(coord, interpolation="linear")
    expected = 0.25 * (arr[1, 2] + arr[1, 3] + arr[2, 2] + arr[2, 3])
    assert np.isclose(result, expected)


def test_eval_linear_coordinate_array():
    """Linear interpolation with a CoordinateArray (multiple points)."""
    arr = np.arange(12, dtype=float).reshape(3, 4)
    image = darsia.ScalarImage(arr, dimensions=[1, 1], space_dim=2)

    # Two query points: midpoint of cols 2-3 at row 1, and exact center of voxel [0,0]
    #   Point 1: frac [1, 2.5]  -> x=0.625, y=2/3
    #   Point 2: frac [0, 0]    -> x=0,     y=1
    coords = darsia.CoordinateArray(np.array([[0.625, 2.0 / 3.0], [0.0, 1.0]]))
    result = image.eval(coords, interpolation="linear")
    expected = np.array([0.5 * arr[1, 2] + 0.5 * arr[1, 3], arr[0, 0]], dtype=float)
    assert np.allclose(result, expected)


def test_eval_linear_voxel_input_uses_nearest():
    """Voxel inputs must always use nearest-neighbour even when interpolation='linear'."""
    arr = np.arange(12, dtype=float).reshape(3, 4)
    image = darsia.ScalarImage(arr, dimensions=[1, 1], space_dim=2)

    voxel = darsia.Voxel([1, 2])
    result = image.eval(voxel, interpolation="linear")
    assert result == arr[1, 2]


def test_eval_linear_voxel_array_uses_nearest():
    """VoxelArray inputs use nearest-neighbour even when interpolation='linear'."""
    arr = np.arange(12, dtype=float).reshape(3, 4)
    image = darsia.ScalarImage(arr, dimensions=[1, 1], space_dim=2)

    voxels = darsia.VoxelArray([[0, 1], [2, 3]])
    result = image.eval(voxels, interpolation="linear")
    assert np.allclose(result, [arr[0, 1], arr[2, 3]])


def test_eval_linear_out_of_bounds():
    """Out-of-bounds coordinates are clipped to the valid range before interpolation."""
    arr = np.arange(12, dtype=float).reshape(3, 4)
    image = darsia.ScalarImage(arr, dimensions=[1, 1], space_dim=2)

    # Coordinate far outside the domain - should clip to corner voxel [2, 3]
    # x >> 1 clips frac_col to 3; y << 0 clips frac_row to 2
    coord = darsia.Coordinate(np.array([5.0, -1.0]))
    result = image.eval(coord, interpolation="linear")
    assert np.isclose(result, arr[2, 3])


def test_eval_linear_nonscalar_image():
    """Linear interpolation works correctly for non-scalar (multi-channel) images."""
    arr = np.arange(36, dtype=float).reshape(3, 4, 3)
    image = darsia.Image(arr, scalar=False, dimensions=[1, 1], space_dim=2)

    # Fractional voxel [1, 2.5] -> mid-point between cols 2 and 3 at row 1
    #   x = 0.625,  y = 2/3
    coord = darsia.Coordinate(np.array([0.625, 2.0 / 3.0]))
    result = image.eval(coord, interpolation="linear")
    expected = 0.5 * arr[1, 2] + 0.5 * arr[1, 3]
    assert np.allclose(result, expected)
