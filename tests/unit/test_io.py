"""Test I/O capabilities in darsia."""

from pathlib import Path

import cv2
import numpy as np
import pytest

import darsia


def test_imread_from_numpy():
    """Test imread for numpy images."""

    # Generate Image from random array
    shape = (10, 20)
    array = np.random.rand(*shape)
    path = Path("random_distribution.npy")
    np.save(path, array)

    # Read numpy image
    np_image = darsia.imread(path, dim=2, width=2, height=1)

    # Compare arrays.
    assert np.allclose(np_image.img, array)

    # Clean up
    path.unlink()


@pytest.mark.parametrize("shape", [(10, 20, 3), (10, 20)])
def test_imread_from_bytes(shape):
    """Test imread for bytes images, for RGB and grayscale type images."""

    # Generate array equivalent with 255 values
    array = 255 * np.ones(shape, dtype=np.uint8)

    # Save array as jpg using cv2
    path = Path("random_distribution.jpg")
    cv2.imwrite(str(path), array)

    # Read numpy image
    with open(path, "rb") as file:
        byte_str = file.read()
        file.close()
        bytes_image = darsia.imread_from_bytes(byte_str, dim=2, width=2, height=1)

    # Compare arrays.
    assert np.allclose(bytes_image.img, array)

    # Clean up
    path.unlink()


def test_imread_from_vtu():
    """Test imread for single and space-time vtu files."""

    vtu_2d_path = Path("../../examples/images/fracture_flow_2.vtu")
    if not vtu_2d_path.exists():
        pytest.xfail("Image required for test not available.")

    # Single time-slab
    vtu_image_2d = darsia.imread(vtu_2d_path, key="c", shape=(200, 200), vtu_dim=2)

    # Space-time image
    space_time_vtu_image_2d = darsia.imread(
        [vtu_2d_path, vtu_2d_path], time=[0, 1], key="c", shape=(200, 200), vtu_dim=2
    )

    # Compare
    assert space_time_vtu_image_2d.time_num == 2
    assert np.allclose(vtu_image_2d.origin, space_time_vtu_image_2d.origin)
    assert np.allclose(vtu_image_2d.dimensions, space_time_vtu_image_2d.dimensions)
    assert not vtu_image_2d.series
    assert space_time_vtu_image_2d.series

    slice_0 = space_time_vtu_image_2d.time_slice(0)
    slice_1 = space_time_vtu_image_2d.time_slice(1)
    assert not slice_0.series
    assert np.allclose(slice_0.origin, vtu_image_2d.origin)
    assert np.allclose(slice_0.dimensions, vtu_image_2d.dimensions)

    assert np.allclose(slice_0.img, vtu_image_2d.img)
    assert np.allclose(slice_1.img, vtu_image_2d.img)
