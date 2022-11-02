"""
Module containing analysis tools for segmented images. This includes
measuring lengths of contours, weighted sums (generalized mass analysis).
"""
from __future__ import annotations

from typing import Optional, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import skimage

import daria


def contour_length(
    img: daria.Image,
    roi: Optional[np.ndarray] = None,
    values_of_interest: Optional[Union[int, list[int]]] = None,
    fill_holes: bool = True,
    verbosity: bool = False,
) -> float:
    """
    Calculation of the contour length of a segmented region.

    Args:
        img (daria.Image): segmented image with boolean or integer values.
        roi (np.ndarray): set of points, for which a bounding box defines a ROI.
        values_of_interest (int or list of int): only active if integer-valued
            image provided; defines the values of interest, i.e., which part
            of the image is treated as active.
        fill_holes (bool): flag controlling whether holes in the determined mask
            are filled before the contour length is computed; if not, holes are
            treated as contour; default is True.
        verbosity (bool): flag controlling whether intermediate results are plotted;
            default is False.

    Returns:
        float: contour length in metric units based on the coordinate system of the
            input image.
    """
    # Make copy of image and restrict to region of interest
    img_roi = (
        img.copy() if roi is None else cast(daria.Image, daria.extractROI(img, roi))
    )

    # Extract boolean mask covering pixels of interest.
    if img_roi.img.dtype == bool:
        if values_of_interest is None:
            mask: np.ndarray = img_roi.img
        else:
            mask = np.zeros_like(img_roi.img, dtype=bool)
            if isinstance(values_of_interest, int):
                mask[img_roi.img == values_of_interest]
            elif isinstance(values_of_interest, list):
                for value in values_of_interest:
                    mask[img_roi.img == value] = True
    elif img_roi.img.dtype in [np.uint8, np.int32, np.int64]:
        assert values_of_interest is not None
        mask = np.zeros(img_roi.img.shape[:2], dtype=bool)
        if isinstance(values_of_interest, int):
            mask[img_roi.img == values_of_interest] = True
        elif isinstance(values_of_interest, list):
            for value in values_of_interest:
                mask[img_roi.img == value] = True
    else:
        raise ValueError(f"Images with dtype {img_roi.img.dtype} not supported.")

    # Fill all holes
    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    # Obtain connected and covered regions and region properties
    label_img, num_labels = skimage.measure.label(mask, return_num=True)
    props = skimage.measure.regionprops(label_img)

    # Determine the length of all contours and sum up
    contour_length = 0.0
    for counter in range(num_labels):
        # Add the perimeter for the region with label l+1 (0 is ignored in regionprops).
        contour_length += props[counter].perimeter

    # Convert contour length from pixel units to metric units
    metric_contour_length = cast(
        float, img_roi.coordinatesystem.pixelsToLength(contour_length)
    )

    # Plot masks and print contour length if requested.
    if verbosity:
        plt.figure()
        plt.imshow(img.img)
        plt.figure()
        plt.imshow(img_roi.img)
        plt.figure()
        plt.imshow(mask)
        plt.show()
        print(f"The contour length is {metric_contour_length}.")

    return metric_contour_length
