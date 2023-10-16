import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

import darsia


def read_test_image(img_id: str) -> tuple[np.ndarray, dict]:
    """Centralize reading of test image.

    Returns:
        array: image array in RGB format read from jpg.
        dict: metadata

    """

    # ! ---- Define image array in RGB format
    path = f"{os.path.dirname(__file__)}/../../examples/images/{img_id}.jpg"
    array = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    # ! ---- Define some metadata corresponding to the input array
    info = {
        "space_dim": 2,
        "indexing": "ij",
    }

    success = array is not None

    return array, info, success


def test_color_correction():
    """Test color correction, effectively converting from BGR to RGB."""

    # ! ---- Fetch test image
    array, info, success = read_test_image("baseline")

    if not success:
        pytest.xfail("Image required for test not available.")

    # ! ---- Setup color correction

    # Need to specify the pixel coordines in (x,y), i.e., (col,row) format, of the
    # marks on the color checker.
    config = {
        "roi": np.array(
            [
                [154, 176],
                [222, 176],
                [222, 68],
                [154, 68],
            ]
        )
    }
    color_correction = darsia.ColorCorrection(**config)

    # ! ---- Define corrected image

    image = darsia.Image(img=array, transformations=[color_correction], **info)

    # ! ---- Compare corrected image with reference

    # Load reference image
    reference_path = "../reference/color_corrected_baseline.npy"
    if not Path(reference_path).exists():
        pytest.xfail("Image required for test not available.")
    reference_image = np.load(reference_path, allow_pickle=True)

    # Make a direct comparison
    assert np.allclose(reference_image, image.img)


def test_curvature_correction():
    """Test of curvature correction applied to a numpy array. The correction
    routine contains all relevant operations, incl. bulging, stretching, and
    cropping."""

    # ! ---- Fetch test image
    array, info, success = read_test_image("co2_2")

    if not success:
        pytest.xfail("Image required for test not available.")

    # ! ---- Setup correction

    # Fetch config file, holding info to several correction routines.
    config_path = f"{os.path.dirname(__file__)}/../../examples/images/config.json"
    with open(config_path, "r") as openfile:
        config = json.load(openfile)

    # Define curvature correction object, initiated with config file
    curvature_correction = darsia.CurvatureCorrection(config=config["curvature"])

    # ! ---- Define corrected image

    image = darsia.Image(img=array, transformations=[curvature_correction], **info)

    # ! ---- Compare corrected image with reference

    reference_path = "../reference/curvature_corrected_co2_2.npy"
    if not Path(reference_path).exists():
        pytest.xfail("Image required for test not available.")
    reference_image = np.load(reference_path, allow_pickle=True)

    # Make a direct comparison
    assert np.allclose(reference_image, image.img)


def test_drift_correction():
    """Test the relative aligning of images via a drift."""

    # ! ---- Fetch test images
    original_array, info, success = read_test_image("baseline")

    if not success:
        pytest.xfail("Image required for test not available.")

    original_image = darsia.Image(img=original_array, **info)

    # ! ---- Define drift correction
    roi = (slice(0, 600), slice(0, 600))
    drift_correction = darsia.DriftCorrection(base=original_image, config={"roi": roi})

    # ! ---- Apply affine transformation
    affine_matrix = np.array([[1, 0, 10], [0, 1, -6]]).astype(np.float32)
    translated_array = cv2.warpAffine(
        original_array, affine_matrix, tuple(reversed(original_array.shape[:2]))
    )
    corrected_image = darsia.Image(
        img=translated_array, transformations=[drift_correction], **info
    )

    # ! ---- Compare original and corrected image, but remove the boundary.
    assert np.allclose(
        original_image.img[10:-10, 10:-10], corrected_image.img[10:-10, 10:-10]
    )


def test_rotation():
    """Test rotation of images."""

    # Generate simple image with white line
    img_array = np.zeros((10, 10))
    img_array[4:6, 2:8] = 1

    info = {
        "space_dim": 2,
        "indexing": "ij",
    }

    # Rotation
    rotation = darsia.RotationCorrection(dim=2, anchor=[5, 5], rotations=[0.5 * np.pi])

    image = darsia.ScalarImage(img=img_array, transformations=[rotation], **info)

    image_ref = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    assert np.allclose(image.img, image_ref)
