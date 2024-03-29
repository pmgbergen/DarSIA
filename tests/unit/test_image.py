"""
Module testing the initialization of images
and the access to its attributes.

"""

import os

import cv2
import numpy as np
from deepdiff import DeepDiff

import darsia


def test_initialize_image_without_meta():
    #############################################################################
    # ! ---- Define image array
    path = f"{os.path.dirname(__file__)}/../../examples/images/baseline.jpg"
    array = cv2.imread(path)

    # ! ---- Initialize darsia image

    image = darsia.Image(img=array)

    assert hasattr(image, "img")
    assert hasattr(image, "coordinatesystem")
    assert image.space_dim == 2
    assert image.indexing == "ij"


def test_initialize_general_image():
    #############################################################################
    # ! ---- Define image array
    path = f"{os.path.dirname(__file__)}/../../examples/images/baseline.jpg"
    array = cv2.imread(path)

    # ! ---- Define some metadata corresponding to the input array
    info = {
        "scalar": False,
        "series": False,
        "space_dim": 2,
        "indexing": "ij",
        "dimensions": [1.5, 2.8],
    }

    # ! ---- Initialize darsia image

    image = darsia.Image(img=array, **info)

    assert hasattr(image, "img"), "image array not defined"
    assert hasattr(image, "coordinatesystem"), "coordinatesystem not defined"
    assert not image.scalar, "scalar not correct"
    assert not image.series, "series not correct"
    assert image.space_dim == 2, "space_dim not correct"
    assert image.time_dim == 0, "time_dim not correct"
    assert image.range_dim == 1, "range_dim not correct"
    assert image.range_num == 3, "range_num not correct"
    assert image.indexing == "ij", "indexing not correct"
    assert np.allclose(image.dimensions, np.array([1.5, 2.8])), "dimensions not correct"
    assert np.allclose(image.origin, np.array([0.0, 1.5])), "origin not correct"
    assert np.allclose(image.num_voxels, [1788, 3180]), "num_voxels not correct"
    assert np.allclose(
        image.voxel_size, [1.5 / 1788, 2.8 / 3180]
    ), "voxel_size not correct"


def test_initialize_optical_image():
    # ! ---- Define image array
    path = f"{os.path.dirname(__file__)}/../../examples/images/baseline.jpg"
    array = cv2.imread(path)

    # ! ---- Define some metadata corresponding to the input array
    info = {
        "series": False,
        "scalar": False,
        "space_dim": 2,
        "indexing": "ij",
        "dimensions": [1.5, 2.8],
    }

    optical_info = {
        "series": False,
        "dimensions": [1.5, 2.8],
    }

    # ! ---- Initialize darsia image

    # Create the color correction and apply it at initialization of image class
    image = darsia.Image(img=array, **info)
    optical_image = darsia.OpticalImage(img=array, **optical_info)

    assert np.allclose(image.img, optical_image.img)
    assert hasattr(image, "coordinatesystem")
    assert not image.scalar
    assert not image.series
    assert optical_image.space_dim == 2
    assert optical_image.time_dim == 0
    assert optical_image.range_dim == 1
    assert optical_image.range_num == 3
    assert optical_image.indexing == "ij"
    assert np.allclose(optical_image.dimensions, np.array([1.5, 2.8]))
    assert np.allclose(optical_image.origin, np.array([0.0, 1.5]))


def test_monochromatic_optical_images():
    # ! ---- Define image array
    path = f"{os.path.dirname(__file__)}/../../examples/images/baseline.jpg"
    array = cv2.imread(path)

    # ! ---- Define some metadata corresponding to the input array
    optical_info = {
        "series": False,
        "dimensions": [1.5, 2.8],
    }

    # ! ---- Initialize darsia image

    # Create the color correction and apply it at initialization of image class
    optical_image = darsia.OpticalImage(img=array, **optical_info)

    # ! ---- Monochromatic versions
    red_image = optical_image.to_monochromatic("red")
    green_image = optical_image.to_monochromatic("green")
    blue_image = optical_image.to_monochromatic("blue")

    assert np.allclose(red_image.img, optical_image.img[:, :, 0])
    assert np.allclose(green_image.img, optical_image.img[:, :, 1])
    assert np.allclose(blue_image.img, optical_image.img[:, :, 2])

    assert red_image.scalar
    assert not red_image.series
    assert red_image.space_dim == 2
    assert red_image.time_dim == 0
    assert red_image.range_dim == 0
    assert red_image.range_num == 1
    assert red_image.indexing == "ij"
    assert np.allclose(red_image.dimensions, np.array([1.5, 2.8]))
    assert np.allclose(red_image.origin, np.array([0.0, 1.5]))


def test_io():
    # Test whether the image can be saved and loaded.

    # ! ---- Define image array
    path = f"{os.path.dirname(__file__)}/../../examples/images/baseline.jpg"
    array = cv2.imread(path)
    dimensions = [1.5, 2.8]
    image = darsia.Image(img=array, dimensions=dimensions)
    metadata = image.metadata()

    # ! ---- Save image
    image.save("test_image.npz")

    # ! ---- Load image
    loaded_image = darsia.imread("test_image.npz")
    loaded_metadata = loaded_image.metadata()

    # ! ---- Check equality
    assert np.allclose(image.img, loaded_image.img)
    assert DeepDiff(metadata, loaded_metadata) == {}

    # ! ---- Remove test file
    os.remove("test_image.npz")
