"""
Module containing auxiliary methods to extract ROIs from daria Images.
"""
import numpy as np

import daria


def extractROI(
    img: daria.Image, x: list, y: list, return_roi: bool = False
) -> daria.Image:
    """Extracts region of interest based on physical coordinates.

    Arguments:
        x (list): list with two elements and containst the start and end point in x-direction;
            points in metric units.
        y (list): list with two elements and containst the start and end point in y-direction;
            points in metric units.

    Returns:
        daria.Image: image object restricted to the ROI.
    """

    # Assume that x and y are in increasing order.
    assert x[0] < x[1] and y[0] < y[1]

    # Convert metric units to number of pixels, and define top-left and bottom-right
    # corners of the roi, towards addressing the image with matrix indexing
    # of x and y coordinates.
    top_left_coordinate = [x[0], y[1]]
    bottom_right_coordinate = [x[1], y[0]]
    top_left_pixel = img.coordinatesystem.coordinateToPixel(top_left_coordinate)
    bottom_right_pixel = img.coordinatesystem.coordinateToPixel(bottom_right_coordinate)

    # Define the ROI in terms of pixels, using matrix indexing, i.e., the (row,col) format
    roi = (
        slice(top_left_pixel[0], bottom_right_pixel[0]),
        slice(top_left_pixel[1], bottom_right_pixel[1]),
    )

    # Define metadata (all quantities in metric units)
    origo = [x[0], y[0]]
    width = x[1] - x[0]
    height = y[1] - y[0]

    # Construct and return image corresponding to ROI
    if return_roi:
        return (
            daria.Image(img=img.img[roi], origo=origo, width=width, height=height),
            roi,
        )
    else:
        return daria.Image(img=img.img[roi], origo=origo, width=width, height=height)


def extractROIPixel(img: daria.Image, roi: tuple) -> daria.Image:
    """Extracts region of interest based on pixel info.

    Arguments:
        roi (tuple of slices): to be used straight away to extract a region of interest;
            using the conventional matrix indexing, i.e., (row,col).

    Returns:
        daria.Image: image object restricted to the ROI.
    """
    # Define metadata; Note that img.origo uses a Cartesian indexing, while the
    # roi uses the conventional matrix indexing
    origo = img.origo + np.array([roi[1].start * img.dx, roi[0].stop * img.dy])
    height = (roi[0].stop - roi[0].start) * img.dy
    width = (roi[1].stop - roi[1].start) * img.dx

    # Construct and return image corresponding to ROI
    return daria.Image(img=img.img[roi], origo=origo, width=width, height=height)
