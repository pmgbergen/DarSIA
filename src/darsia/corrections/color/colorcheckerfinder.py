"""Module for finding a colorchecker in an image."""

from typing import Literal

import numpy as np
import colour
from colour_checker_detection import (
    colour_checkers_coordinates_segmentation,
    detect_colour_checkers_segmentation,
)

import darsia


def find_colorchecker(img: darsia.Image, strategy=Literal["upper_right"]):
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
    eotf = darsia.EOTF()
    arr = eotf.adjust(img.img)
    shape = arr.shape

    def detect_colorchecker(arr):
        """Colour-based routine to detect the location and swatches of a colorchecker."""

        # Detect swatches using colour science algorithm
        swatches = detect_colour_checkers_segmentation(arr)
        success = len(swatches) > 0
        if success:
            # Also find position
            detection_data = colour_checkers_coordinates_segmentation(arr, True)

            # Reshape to original image size to retrieve the correct position
            detection_shape = detection_data.segmented_image.shape
            input_shape = arr.shape
            max_detection_coarsening_rate = max(
                [input_shape[i] / detection_shape[i] for i in range(2)]
            )
            # Colour uses reverse matrix indexing
            coarse_detection_voxels = darsia.make_voxel(
                detection_data.colour_checkers[0], matrix_indexing=False
            )
            # Determine the voxels of the colorchecker
            voxels = darsia.make_voxel(
                max_detection_coarsening_rate * coarse_detection_voxels.copy()
            )
        else:
            voxels = None

        return success, swatches, voxels

    # Define strategy
    if strategy == "upper_right":
        target_corner = np.array([0, shape[1]])
        start_corner = np.array([shape[0], 0])
        update = 0.8
    else:
        raise NotImplementedError

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

    # Extract the color checker directly from the image (more robust than colour)
    img_cc = img.subregion(voxels=roi).subregion(voxels=local_voxels)
    colorchecker = darsia.CustomColorChecker(image=img_cc.img)

    # Map to global voxels - colour_checker_detection uses coarsening
    # Resort in anti-clockwise direction starting at the brown swatch
    local_voxels = local_voxels[np.array([1, 0, 3, 2])]
    global_voxels = img.coordinatesystem.voxel(
        img.subregion(roi).coordinatesystem.coordinate(local_voxels)
    )

    return colorchecker, global_voxels
