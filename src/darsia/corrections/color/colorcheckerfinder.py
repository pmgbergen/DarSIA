"""Module for finding a colorchecker in an image."""

from typing import Literal
from warnings import warn

import colour_checker_detection
import numpy as np
import skimage

import darsia


def _reorient_colorchecker(
    img: np.ndarray,
    local_voxels: np.array,
) -> tuple[np.ndarray, np.ndarray]:
    """Reorient the colorchecker such that the brown swatch is in the top left corner.

    Args:
        img (np.ndarray): Image of the colorchecker.
        local_voxels (np.ndarray): Voxels of the colorchecker (do not need to be sorted
            in any way).

    Returns:
        reoriented_img_cc (np.ndarray): Reoriented image of the colorchecker.
        reoriented_local_voxels (np.ndarray): Reoriented voxels of the colorchecker.

    """
    # Assert that the image is in uint8 or uint16 format
    if img.dtype in [np.uint8, np.uint16]:
        img = skimage.img_as_float(img)
    else:
        img = img / np.max(img)

    # Expected colors in the corner swatches in RGB colors
    brown_swatch = np.array([175, 130, 110]) / 255
    white_swatch = np.array([250, 250, 250]) / 255
    black_swatch = np.array([60, 60, 60]) / 255
    turquoise_swatch = np.array([175, 235, 225]) / 255
    expected_swatches = [
        brown_swatch,
        white_swatch,
        black_swatch,
        turquoise_swatch,
    ]

    # Determine the indices closest to the expected colors
    expected_swatch_indices = [
        np.where(
            np.logical_or(
                np.linalg.norm(img - swatch, axis=-1)
                < 1.05 * np.min(np.linalg.norm(img - swatch, axis=-1)),
                np.isclose(
                    np.linalg.norm(img - swatch, axis=-1),
                    1.05 * np.min(np.linalg.norm(img - swatch, axis=-1)),
                ),
            )
        )
        for swatch in expected_swatches
    ]

    # Determine the centers of the swatches as median of the indices
    expected_swatches_centers = [
        np.median(indices, axis=-1).astype(int) for indices in expected_swatch_indices
    ]

    # Mark corners in anti-clockwise direction starting at the top left corner
    corners = np.array(
        [
            [0, 0],
            [img.shape[0] - 1, 0],
            [img.shape[0] - 1, img.shape[1] - 1],
            [0, img.shape[1] - 1],
        ]
    )

    # Determine which swatch is closest to each corner
    closest_swatches = [
        np.argmin(np.linalg.norm(corners - center, axis=-1))
        for center in expected_swatches_centers
    ]

    # Arange local voxels such that they follow clock-wise sorting
    local_voxels = darsia.sort_quad(local_voxels)

    # Check whether the indices 0, 1, 2, 3 are in the closest swatches
    success = np.all(np.sort(closest_swatches) == np.arange(4))
    if not success:
        warn("Colorchecker orientation not found.")
        return img, local_voxels

    # Reorient the local voxels in an anti-clockwise direction starting at the top left
    # corner. First shift them to the top left corner
    shifted_local_voxels = local_voxels - np.min(local_voxels, axis=0)

    # Then sort them in anti-clockwise direction
    sorting_local_voxels = [
        np.argmin(np.linalg.norm(corners - shifted_local_voxel, axis=-1))
        for shifted_local_voxel in shifted_local_voxels
    ]

    # Reorient the local voxels by concatenating the two sortings, where closets_swatches
    # needs to be inverted, such that the map after all maps from voxels to swatches
    # (via the corners).
    inverted_closest_swatches = np.zeros(4, dtype=int)
    inverted_closest_swatches[np.array(closest_swatches)] = np.arange(4, dtype=int)
    orientation = inverted_closest_swatches[sorting_local_voxels]

    # Reorient the local voxels accordingly
    reoriented_local_voxels = local_voxels[orientation]

    # Reorient the image. Count how many positions need to be shifted to the left for 0
    # to be at the first position.
    shift = np.where(orientation == 0)[0][0]

    # Turn the image shift-many times by 90 degrees
    reoriented_img_cc = np.rot90(img, k=shift)

    return reoriented_img_cc, reoriented_local_voxels


def find_colorchecker(
    img: darsia.Image, strategy: Literal["upper_right", "upper_left"]
):
    """Detect colorchecker in corner.

    Search for position and colors.

    Args:
        img (darsia.Image): Image to search in.
        strategy (str, optional): Strategy to use. Defaults to "upper_right".

    Returns:
        colorchecker (darsia.CustomColorChecker): Colorchecker.
        global_voxels (np.ndarray): Voxels of the colorchecker.

    """

    # Preproccess image array (required by the algorthms provided by colour-science)
    # EOTF requires ubyte input
    eotf = darsia.EOTF()
    arr = eotf.adjust(img.img_as(np.uint8).img)
    shape = arr.shape

    def detect_colorchecker(arr):
        """Colour-based routine to detect the location and swatches of a colorchecker."""

        # Detect swatches using colour science algorithm
        colorcheckers = colour_checker_detection.detect_colour_checkers_segmentation(
            arr, additional_data=True
        )
        success = len(colorcheckers) == 1

        if success:
            print("Colorchecker detected.")
            colorchecker = colorcheckers[0]
            swatches = colorchecker.swatch_colours
            detection_data = colour_checker_detection.segmenter_default(
                arr, additional_data=True
            )

            # Reshape to original image size to retrieve the correct position
            detection_shape = detection_data.segmented_image.shape
            input_shape = arr.shape
            max_detection_coarsening_rate = max(
                [input_shape[i] / detection_shape[i] for i in range(2)]
            )
            # Colour uses reverse matrix indexing
            coarse_detection_voxels = darsia.make_voxel(
                detection_data.rectangles[0], matrix_indexing=False
            )
            # Determine the voxels of the colorchecker
            voxels = darsia.make_voxel(
                max_detection_coarsening_rate * coarse_detection_voxels.copy()
            )
        else:
            warn(f"{len(colorcheckers)} colorcheckers detected.")
            swatches = None
            voxels = None

        return success, swatches, voxels

    # Define strategy
    if strategy == "upper_right":
        target_corner = np.array([0, shape[1]])
        start_corner = np.array([shape[0], 0])
        update = 0.8
    elif strategy == "upper_left":
        target_corner = np.array([0, 0])
        start_corner = np.array([shape[0], shape[1]])
        update = 0.8
    else:
        raise NotImplementedError(f"Strategy {strategy} not implemented.")

    # Iterate by decreasing a window for searching the colorchecker
    success = False
    current = 1
    while not success:
        current_corner = (
            current * start_corner + (1 - current) * target_corner
        ).astype(int)
        roi = (
            slice(*np.sort([current_corner[0], target_corner[0]]).astype(int)),
            slice(*np.sort([current_corner[1], target_corner[1]]).astype(int)),
        )

        # Do not use swatches as detected by colour, just use the location.
        success, _, local_voxels = detect_colorchecker(arr[roi])
        if success:
            break
        current *= update
        if current < update**20:
            assert False, "No color checker found."

    # Extract the color checker directly from the image (more robust than colour).
    # Yet, need the image to be oriented correctly. Thus, first find the orientation
    # such that the brown swatch is in the top left corner, and then extract the
    # colorchecker.
    img_cc = img.subregion(roi).subregion(local_voxels)
    oriented_img_cc, oriented_local_voxels = _reorient_colorchecker(
        img_cc.img, local_voxels
    )
    colorchecker = darsia.CustomColorChecker(image=oriented_img_cc)

    # Map to global voxels - colour_checker_detection uses coarsening
    global_voxels = img.coordinatesystem.voxel(
        img.subregion(roi).coordinatesystem.coordinate(oriented_local_voxels)
    )

    return colorchecker, global_voxels
