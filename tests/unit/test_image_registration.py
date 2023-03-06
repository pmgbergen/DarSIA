"""
Module testing the image registration and local warping of images.

"""

import os

import cv2
import numpy as np

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
        "dim": 2,
        "orientation": "ij",
    }

    return array, info


def test_image_registration():
    """Test the local patchwise affine correction."""

    # ! ---- Fetch image arrays
    array_dst, info = read_test_image("fine/Baseline")
    array_src, _ = read_test_image("fine/pulse1")

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
    drift_correction = darsia.DriftCorrection(base=array_src, roi=roi)

    # ! ---- Corrected images

    img_src = darsia.GeneralImage(
        img=array_src,
        transformations=[
            drift_correction,
            curvature_correction,
        ],
        dimensions=[1.5, 2.8],
        origin=[0.0, 1.5],
    )

    img_dst = darsia.GeneralImage(
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
    da_img_src = darsia.extractROIPixel(img_src, roi_crop)
    da_img_dst = darsia.extractROIPixel(img_dst, roi_crop)

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

    box = np.array([[0.0, 1.2], [1.1, 0.6]])
    img_box = darsia.extractROI(da_img_src, box)
    patched_box = darsia.GeneralPatches(img_box, [3, 5])
    deformation_patch_centers = image_registration.evaluate(patched_box)

    # ! ---- Compare with reference
    print(deformation_patch_centers)

    reference_deformation = np.array(
        [
            [
                [-4.93091415e-05, -1.93677716e-02],
                [4.60359794e-05, -1.97569753e-02],
                [-1.60386595e-04, -1.67867149e-02],
                [7.15840122e-04, -1.45076868e-02],
                [1.56396856e-03, -1.55768394e-02],
            ],
            [
                [-4.01095481e-05, -1.41275267e-02],
                [6.56553419e-05, -1.42395756e-02],
                [7.89620574e-05, -1.19017095e-02],
                [5.72394208e-04, -9.23458292e-03],
                [1.24946312e-03, -1.01292181e-02],
            ],
            [
                [1.55575358e-04, -1.09349365e-02],
                [3.20850540e-05, -1.13698501e-02],
                [1.79619414e-04, -8.78894088e-03],
                [5.37285985e-04, -7.11758117e-03],
                [4.46936144e-04, -5.99474517e-03],
            ],
        ]
    )
    print(reference_deformation)
    assert np.all(np.isclose(reference_deformation, deformation_patch_centers))
