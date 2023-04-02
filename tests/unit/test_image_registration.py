"""
Module testing the image registration and local warping of images.

"""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytest

import darsia


def read_test_image(img_id: str) -> tuple[Optional[np.ndarray], Optional[dict], bool]:
    """Centralize reading of test image.

    Returns:
        array: image array in RGB format read from jpg.
        dict: metadata

    """

    # ! ---- Define image array in RGB format
    path = f"{os.path.dirname(__file__)}/../../examples/images/{img_id}.jpg"
    if Path(path).exists():
        array = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        # ! ---- Define some metadata corresponding to the input array
        info = {
            "dim": 2,
            "orientation": "ij",
        }

        return array, info, True
    else:
        return None, None, False


def test_image_registration():
    """Test the local patchwise affine correction."""

    # ! ---- Fetch image arrays
    array_dst, info, success_dst = read_test_image("fine/Baseline")
    array_src, _, success_src = read_test_image("fine/pulse1")

    if not (success_dst and success_src):
        pytest.xfail("Files required for test not available.")

    # ! ---- Transformations

    # Setup curvature correction (here only cropping)
    config = {
        "crop": {
            # Define the pixel values (x,y) of the corners of the ROI.
            # Start at top left corner and then continue counterclockwise.
            "pts_src": [[52, 0], [64, 4429], [7896, 4429], [7891, 0]],
            # Specify the true dimensions of the reference points
            "width": 2.8,
            "height": 1.5,
        }
    }
    curvature_correction = darsia.CurvatureCorrection(config)

    # Setup drift correction taking care of moving camera in between taking photos.
    # Use the color checker as reference in both images, and make the src image
    # the anker.
    roi = (slice(0, 600), slice(0, 600))
    drift_correction = darsia.DriftCorrection(base=array_src, config={"roi": roi})

    # ! ---- Corrected images

    img_src = darsia.Image(
        img=array_src,
        transformations=[
            drift_correction,
            curvature_correction,
        ],
        dimensions=[1.5, 2.8],
        origin=[0.0, 1.5],
    )

    img_dst = darsia.Image(
        img=array_dst,
        transformations=[
            drift_correction,
            curvature_correction,
        ],
        dimensions=[1.5, 2.8],
        origin=[0.0, 1.5],
    )

    # Extract ROI to cut away the color palette. Use pixel ranges to crop the image.
    roi_crop = (slice(470, img_src.img.shape[0]), slice(60, 7940))
    da_img_src = img_src.subregion(voxels=roi_crop)
    da_img_dst = img_dst.subregion(voxels=roi_crop)

    # ! ---- Setup image registration
    # Define image registration tool
    config["image registration"] = {
        # Define the number of patches in x and y directions
        "N_patches": [10, 20],
        # Define a relative overlap.
        "rel_overlap": 0.1,
        # Add some tuning parameters for the feature detection (these are actually
        # the default values and could be also omitted.
        "max_features": 200,
        "tol": 0.05,
    }

    # ! ---- Image registration

    image_registration = darsia.ImageRegistration(
        da_img_dst, **config["image registration"]
    )
    da_new_image = image_registration(da_img_src)

    # ! --- Compare reference
    ref_img = np.load("../reference/image_registration.npy", allow_pickle=True)
    assert np.all(np.isclose(ref_img, da_new_image.img))

    # ! ---- Evaluate deformation in patches

    box = np.array([[0.05, 0.6], [1.2, 1.1]])  # two coordinate-pairs
    img_box = da_img_src.subregion(coordinates=box)
    patched_box = darsia.Patches(img_box, [3, 5])
    deformation_patch_centers = image_registration.evaluate(patched_box)

    # ! ---- Compare with reference

    reference_deformation = np.array(
        [
            [
                [6.0347117003798296e-05, -0.01703448285839166],
                [0.00012434595037347882, -0.016906950613570997],
                [8.030563791059885e-05, -0.012484352531479302],
                [0.0015754651382501482, -0.012268694374474686],
                [0.0008378176672299626, -0.01097122986760357],
            ],
            [
                [6.585765517047654e-05, -0.013220317002590529],
                [8.026821250043336e-05, -0.013459138660144325],
                [0.0001926195259653129, -0.009916008955465279],
                [0.0008098157753585142, -0.008293671811481499],
                [0.0004426604970706542, -0.009092187934429676],
            ],
            [
                [7.384206208350228e-05, -0.01067977940474071],
                [0.00012197954791242723, -0.010401261630327941],
                [0.0002285961592267303, -0.007916737316360068],
                [0.0005695112760614377, -0.0065475789514414864],
                [0.00024251334847527218, -0.00496866802155904],
            ],
        ]
    )
    assert np.all(np.isclose(reference_deformation, deformation_patch_centers))
