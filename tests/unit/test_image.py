"""
Module testing the initialization of images
and the access to its attributes.

"""

import os

import cv2
import numpy as np

import darsia


def test_initialize_image():

    # Define path to image
    path = f"{os.path.dirname(__file__)}/../../examples/images/baseline.jpg"

    # ! ---- Setup the manual color correction

    # Need to specify the pixel coordines in (x,y), i.e., (col,row) format, of the
    # marks on the color checker.
    roi_cc = np.array(
        [
            [154, 176],
            [222, 176],
            [222, 68],
            [154, 68],
        ]
    )
    color_correction = darsia.ColorCorrection(
        roi=roi_cc,
    )

    # Create the color correction and apply it at initialization of image class
    image = darsia.Image(
        path,
        color_correction=color_correction,
        width=2.8,
        height=1.5,
    )

    assert hasattr(image, "img")
    assert hasattr(image, "coordinatesystem")


def test_initialize_simple_general_image():

    #############################################################################
    # ! ---- Define image array
    path = f"{os.path.dirname(__file__)}/../../examples/images/baseline.jpg"
    array = cv2.imread(path)

    # ! ---- Initialize darsia image

    # Create the color correction and apply it at initialization of image class
    image = darsia.GeneralImage(img=array)

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

    # Create the color correction and apply it at initialization of image class
    image = darsia.GeneralImage(img=array, **info)

    assert hasattr(image, "img")
    assert hasattr(image, "coordinatesystem")
    assert not image.scalar
    assert not image.series
    assert image.space_dim == 2
    assert image.time_dim == 0
    assert image.range_dim == 1
    assert image.range_num == 3
    assert image.indexing == "ij"
    assert np.all(np.isclose(image.dimensions, np.array([1.5, 2.8])))
    assert np.all(np.isclose(image.origin, np.array([0.0, 1.5])))


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
    image = darsia.GeneralImage(img=array, **info)
    optical_image = darsia.OpticalImage(img=array, **optical_info)

    assert np.all(np.isclose(image.img, optical_image.img))
    assert hasattr(image, "coordinatesystem")
    assert not image.scalar
    assert not image.series
    assert optical_image.space_dim == 2
    assert optical_image.time_dim == 0
    assert optical_image.range_dim == 1
    assert optical_image.range_num == 3
    assert optical_image.indexing == "ij"
    assert np.all(np.isclose(optical_image.dimensions, np.array([1.5, 2.8])))
    assert np.all(np.isclose(optical_image.origin, np.array([0.0, 1.5])))


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

    assert np.all(np.isclose(red_image.img, optical_image.img[:, :, 0]))
    assert np.all(np.isclose(green_image.img, optical_image.img[:, :, 1]))
    assert np.all(np.isclose(blue_image.img, optical_image.img[:, :, 2]))

    assert red_image.scalar
    assert not red_image.series
    assert red_image.space_dim == 2
    assert red_image.time_dim == 0
    assert red_image.range_dim == 0
    assert red_image.range_num == 1
    assert red_image.indexing == "ij"
    assert np.all(np.isclose(red_image.dimensions, np.array([1.5, 2.8])))
    assert np.all(np.isclose(red_image.origin, np.array([0.0, 1.5])))
