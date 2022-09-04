"""Apply homography to tilted and stretched image with known physical dimensions."""

from __future__ import annotations

import cv2
import numpy as np


def homography_correction(
    img_src: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Homography correction for known corner points of a square (default) object.

    Args:
        kwargs (optional keyword arguments):
            width (int or float): width of the physical object
            height (int or float): height of the physical object
            in meters (boolean): controlling whether width and height are float and
                are meant as in meters; number of pixels otherwise
            pts_src (array): N points with (x,y) pixel coordinates, N>=4
            pts_dst (array, optional): N points with (x,y) pixel coordinates, N>=4
    """

    # Determine original and target size
    height, width = img_src.shape[:2]
    target_width = kwargs.pop("width", width)
    target_height = kwargs.pop("height", height)

    # Fetch corner points in the provided image
    pts_src = kwargs.pop("pts_src")
    if isinstance(pts_src, list):
        pts_src = np.array(pts_src)

    # Allow 'width' and 'height' to be provided in meters.
    # Then aim at comparably many pixels as in the provided
    # image, modulo the ratio
    if kwargs.pop("in meters", False):

        # The goal aspect ratio
        aspect_ratio = target_width / target_height

        # Try to keep this aspect ratio, but do not use more pixels than before.
        # Convert to number of pixels
        target_width = min(width, int(aspect_ratio * float(height)))
        target_height = min(height, int(1.0 / aspect_ratio * float(width)))

    # Assign corner points as destination points if none are provided.
    if "pts_dst" not in kwargs:
        # Assume implicitly that corner points have been provided,
        # and that their orientation is mathematically positive,
        # starting with the top left corner.
        assert pts_src.shape[0] == 4
        pts_dst = np.array(
            [
                [0, 0],
                [0, target_height - 1],
                [target_width - 1, target_height - 1],
                [target_width - 1, 0],
            ]
        )
    else:
        pts_dst = kwargs.pop("pts_dst")
        if isinstance(pts_dst, list):
            pts_dst = np.array(pts_dst)

    homography, _ = cv2.findHomography(pts_src, pts_dst)

    # Warp source image
    img_dst = cv2.warpPerspective(img_src, homography, (target_width, target_height))

    return img_dst
