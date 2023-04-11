"""
Module testing the initialization of images
and the access to its attributes.

"""

import os

import cv2
import numpy as np

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
        "dim": 2,
        "indexing": "ij",
        "dimensions": [1.5, 2.8],
        "origin": [0.0, 1.5],
    }

    # ! ---- Initialize darsia image

    image = darsia.Image(img=array, **info)

    assert hasattr(image, "img")
    assert hasattr(image, "coordinatesystem")
    assert not image.scalar
    assert not image.series
    assert image.space_dim == 2
    assert image.time_dim == 0
    assert image.range_dim == 1
    assert image.range_num == 3
    assert image.indexing == "ij"
    assert np.allclose(image.dimensions, np.array([1.5, 2.8]))
    assert np.allclose(image.origin, np.array([0.0, 1.5]))


def test_initialize_optical_image():

    # ! ---- Define image array
    path = f"{os.path.dirname(__file__)}/../../examples/images/baseline.jpg"
    array = cv2.imread(path)

    # ! ---- Define some metadata corresponding to the input array
    info = {
        "series": False,
        "scalar": False,
        "dim": 2,
        "indexing": "ij",
        "dimensions": [1.5, 2.8],
        "origin": [0.0, 1.5],
    }

    optical_info = {
        "series": False,
        "dimensions": [1.5, 2.8],
        "origin": [0.0, 1.5],
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
        "origin": [0.0, 1.5],
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
