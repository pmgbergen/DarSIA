"""
Module containing auxiliary methods to extract ROIs from darsia Images.
"""

from typing import Literal

import cv2
import numpy as np

IndexingOption = Literal["matrix", "reverse matrix"]
"""Indexing option consi    else:dered in the following conversion."""


def to_reverse_matrix_indexing(
    pixel: np.ndarray, indexing: IndexingOption = "reverse matrix"
) -> np.ndarray:
    """Convert pixel coordinates to reverse matrix indexing format.

    Args:
        pixel (np.ndarray): pixel coordinates
        indexing (IndexingOption): indexing of pixel

    Returns:
        pixel converted to 'reverse matrix' indexing

    Raises:
        NotImplementedError: if dimension not 2
        NotImplementedError: if indexing not among IndexingOption

    """
    if not len(pixel.shape) == 2:
        raise NotImplementedError

    if indexing == "reverse matrix":
        return pixel
    elif indexing == "matrix":
        return np.fliplr(np.atleast_2d(pixel))
    else:
        raise NotImplementedError


InterpolationOption = Literal["inter_nearest", "inter_linear", "inter_area"]
"""Interpolation options considered in below warp."""


def extract_quadrilateral_ROI(
    img_src: np.ndarray,
    pts_src,
    indexing: IndexingOption = "reverse matrix",
    interpolation: InterpolationOption = "inter_linear",
    **kwargs
) -> np.ndarray:
    """
    Extract quadrilateral ROI using a perspective transform,
    given known corner points of a square (default) object.

    Args:
        kwargs (optional keyword arguments):
            width (int or float): width of the physical object
            height (int or float): height of the physical object
            pts_src (array): N points with pixel coordinates in (col,row) format, N>=4
            pts_dst (array, optional): N points with pixels in (col, row) format, N>=4
    """

    # FIXME: Implementation hardcoded for 2d.

    # ! ---- Properties of image

    scalar = len(img_src.shape) == 2

    # ! ---- Size of new image

    # Determine current dimensions and use as default size.
    original_shape = img_src.shape[:2]

    if "width" in kwargs and "height" in kwargs:
        # Determine target image dimensions based on physical units.
        target_width = kwargs.get("width")
        target_height = kwargs.get("height")

        # Aim at comparably many pixels as in the provided
        # image, modulo the ratio.
        aspect_ratio = target_width / target_height

        # Try to keep this aspect ratio, but do not use more pixels than before.
        # Convert to number of pixels
        original_height, original_width = original_shape
        width = min(original_width, int(aspect_ratio * float(original_height)))
        height = min(original_height, int(1.0 / aspect_ratio * float(original_width)))

    else:
        # Determine target image dimensions based on voxel numbers.
        target_shape = kwargs.get("shape", original_shape)
        height, width = target_shape

    # ! ---- Mapping

    # Fetch corner points in the provided image
    if isinstance(pts_src, list):
        pts_src = np.array(pts_src)
    pts_src = to_reverse_matrix_indexing(pts_src, indexing)

    # Assign corner points as destination points if none are provided.
    if "pts_dst" in kwargs:

        pts_dst: np.ndarray = to_reverse_matrix_indexing(
            np.array(kwargs.get("pts_dst")), indexing
        )

    else:

        # Assume implicitly that corner points have been provided,
        # and that their orientation is mathematically positive,
        # starting with the top left corner.
        # Furthermore, use reverse matrix indexing, i.e., (col,row).
        pts_dst = np.array(
            [
                [0, 0],
                [0, height - 1],
                [width - 1, height - 1],
                [width - 1, 0],
            ]
        )

    assert pts_src.shape[0] == pts_dst.shape[0]
    P = cv2.getPerspectiveTransform(
        pts_src.astype(np.float32), pts_dst.astype(np.float32)
    )

    # ! ---- Warp image

    # Take care of data type - cv2 requires np.float32 objects.
    # However, when using input images with integer dtype, it is
    # intended that it remains like this. One may indeed loose
    # some information. However, since data type changes are
    # challenging to keep track of, the intention is that this
    # routine returns arrays of same dtype again.
    dtype = img_src.dtype

    interpolation_flag = None
    if interpolation == "inter_nearest":
        interpolation_flag = cv2.INTER_NEAREST
    elif interpolation == "inter_linear":
        interpolation_flag = cv2.INTER_LINEAR
    elif interpolation == "inter_area":
        interpolation_flag == cv2.INTER_AREA
    else:
        raise NotImplementedError

    # Warp source image. Warping may convert a 3-tensor to a 2-tensor.
    # Force to use a 3-tensor structure.
    img_dst = np.atleast_3d(
        cv2.warpPerspective(
            img_src.astype(np.float32),
            P,
            (width, height),
            flags=interpolation_flag,
        )
    ).astype(dtype)

    # Reduce to scalar image array, if input provided in such format.
    if scalar:
        img_dst = img_dst[:, :, 0]

    return img_dst
