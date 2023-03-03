"""
Module testing the initialization of images
and the access to its attributes.

"""

from pathlib import Path
import os
import numpy as np
import darsia
import cv2


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


def test_initialize_general_image():

    #############################################################################
    # ! ---- Define image array
    path = f"{os.path.dirname(__file__)}/../../examples/images/baseline.jpg"
    array = cv2.imread(path)

    # ! ---- Define some metadata corresponding to the input array
    info = {
        "dim": 2,
        "orientation": "ij",
    }

    #############################################################################
    # Transformation.

    # ! ---- Setup color correction

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

    # ! ---- Order of transformations

    transformations = [
        color_correction,
    ]

    # ! ---- Initialize darsia image

    # Create the color correction and apply it at initialization of image class
    image = darsia.GeneralImage(img=array, transformations=transformations, **info)

    assert hasattr(image, "img")
    assert hasattr(image, "coordinatesystem")
    assert image.space_dim == 2
    assert image.orientation == "ij"
