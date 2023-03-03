from pathlib import Path
import os
import numpy as np
import darsia
import cv2

# def test_color_correction():
#
#    # Define path to image
#    image = f"{os.path.dirname(__file__)}/../examples/images/baseline.jpg"
#
#    # ! ---- Setup the manual color correction
#
#    # Need to specify the pixel coordines in (x,y), i.e., (col,row) format, of the
#    # marks on the color checker.
#    roi_cc = np.array(
#        [
#            [154, 176],
#            [222, 176],
#            [222, 68],
#            [154, 68],
#        ]
#    )
#    color_correction = darsia.ColorCorrection(
#        roi=roi_cc,
#    )
#
#    # Create the color correction and apply it at initialization of image class
#    corrected_image = darsia.Image(
#        image,
#        color_correction=color_correction,
#        width=2.8,
#        height=1.5,
#    )
#
#    # Load reference image
#    reference_image = np.load(
#        "./reference/color_corrected_baseline.npy", allow_pickle=True
#    )
#
#    # Make a direct comparison
#    assert np.all(np.isclose(reference_image, corrected_image.img))


def test_color_correction():

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

    # ! ---- Compare corrected image with reference

    # Load reference image
    reference_image = np.load(
        "../reference/color_corrected_baseline.npy", allow_pickle=True
    )

    # Make a direct comparison
    assert np.all(np.isclose(reference_image, image.img))
