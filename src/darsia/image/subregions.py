"""
Module containing auxiliary methods to extract ROIs from darsia Images.
"""
import warnings
from typing import Union

import cv2
import numpy as np

import darsia


def extractROI(
    img: Union[darsia.Image, darsia.GeneralImage],
    pts: Union[np.ndarray, list],
    return_roi: bool = False,
) -> Union[darsia.Image, tuple[darsia.Image, tuple[slice, slice]]]:  # TODO GeneralImage
    """Extracts region of interest based on physical coordinates.

    Args:
        img (darsia.Image): image to be cropped.
        pts (np.ndarray or list): coordinates (x,y) with metric units implicitly defining a
            bounding box, which again defines a ROI.
        return_roi (bool): flag controlling whether the determined ROI should be returned;
            default is False.

    Returns:
        darsia.Image: image object restricted to the ROI.
    """
    # Convert coordinates to array
    if isinstance(pts, list):
        pts = np.array(pts)

    # Convert metric units to number of pixels, and define top-left and bottom-right
    # corners of the roi, towards addressing the image with matrix indexing
    # of x and y coordinates.
    top_left_coordinate = [np.min(pts[:, 0]), np.max(pts[:, 1])]
    bottom_right_coordinate = [np.max(pts[:, 0]), np.min(pts[:, 1])]
    if isinstance(img, darsia.Image):
        top_left_pixel = img.coordinatesystem.coordinateToPixel(top_left_coordinate)
        bottom_right_pixel = img.coordinatesystem.coordinateToPixel(
            bottom_right_coordinate
        )
    elif isinstance(img, darsia.GeneralImage):
        top_left_pixel = img.coordinatesystem.voxel(top_left_coordinate)
        bottom_right_pixel = img.coordinatesystem.voxel(bottom_right_coordinate)

    # Define the ROI in terms of pixels, using matrix indexing, i.e., the (row,col) format
    roi = (
        slice(max(0, top_left_pixel[0]), min(img.img.shape[0], bottom_right_pixel[0])),
        slice(max(0, top_left_pixel[1]), min(img.img.shape[1], bottom_right_pixel[1])),
    )

    if (
        min(top_left_pixel[0], top_left_pixel[1]) < 0
        or bottom_right_pixel[0] > img.img.shape[0]
        or bottom_right_pixel[1] > img.img.shape[1]
    ):
        warnings.warn("Provided coordinates lie outside image.")

    # Construct and return image corresponding to ROI
    if isinstance(img, darsia.Image):
        # Define metadata (all quantities in metric units)
        origin = [np.min(pts[:, 0]), np.min(pts[:, 1])]
        width = np.max(pts[:, 0]) - np.min(pts[:, 0])
        height = np.max(pts[:, 1]) - np.min(pts[:, 1])

        if return_roi:
            return (
                darsia.Image(
                    img=img.img[roi],
                    origin=origin,
                    width=width,
                    height=height,
                    color_space=img.colorspace,
                ),
                roi,
            )
        else:
            return darsia.Image(
                img=img.img[roi],
                origin=origin,
                width=width,
                height=height,
                color_space=img.colorspace,
            )
    elif isinstance(img, darsia.GeneralImage):
        raise NotImplementedError("Use internal functionality.")

def extractROIPixel(
    img: Union[darsia.Image, darsia.GeneralImage], roi: tuple
) -> Union[darsia.Image, darsia.GeneralImage]:
    """Extracts region of interest based on pixel info.

    Arguments:
        roi (tuple of slices): to be used straight away to extract a region of interest;
            using the conventional matrix indexing, i.e., (row,col).

    Returns:
        darsia.Image: image object restricted to the ROI.

    """
    if isinstance(img, darsia.Image):
        # Define metadata; Note that img.origin uses a Cartesian indexing, while the
        # roi uses the conventional matrix indexing
        origin = img.origin + np.array(
            [roi[1].start * img.dx, (img.num_pixels_height - roi[0].stop) * img.dy]
        )
        height = (roi[0].stop - roi[0].start) * img.dy
        width = (roi[1].stop - roi[1].start) * img.dx

        # Construct and return image corresponding to ROI
        return darsia.Image(
            img=img.img[roi],
            origin=origin,
            width=width,
            height=height,
            color_space=img.colorspace,
        )

    elif isinstance(img, darsia.GeneralImage):
        raise NotImplementedError("Use internal functionality.")


def extract_quadrilateral_ROI(img_src: np.ndarray, **kwargs) -> np.ndarray:
    """
    Extract quadrilateral ROI using a perspective transform,
    given known corner points of a square (default) object.

    Args:
        kwargs (optional keyword arguments):
            width (int or float): width of the physical object
            height (int or float): height of the physical object
            pts_src (array): N points with pixels in (col,row) format, N>=4
            pts_dst (array, optional): N points with pixels in (col, row) format, N>=4
    """

    # Determine original and target size
    height, width = img_src.shape[:2]
    target_width = kwargs.get("width", width)
    target_height = kwargs.get("height", height)

    # Fetch corner points in the provided image
    pts_src = kwargs.get("pts_src")
    if isinstance(pts_src, list):
        pts_src = np.array(pts_src)

    # Aim at comparably many pixels as in the provided
    # image, modulo the ratio.
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
        # Further more use reversed matrix indexing, i.e., (col,row).
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
        pts_dst = kwargs.get("pts_dst")
        if isinstance(pts_dst, list):
            pts_dst = np.array(pts_dst)

    P = cv2.getPerspectiveTransform(
        pts_src.astype(np.float32), pts_dst.astype(np.float32)
    )

    # Take care of data type - cv2 requires np.float32 objects.
    # However, when using input images with integer dtype, it is
    # intended that it remains like this. One may indeed loose
    # some information. However, since data type changes are
    # challenging to keep track of, the intention is that this
    # routine returns arrays of same dtype again.
    dtype = img_src.dtype

    # Warp source image. Warping may convert a 3-tensor to a 2-tensor.
    # Force to use a 3-tensor structure.
    img_dst = np.atleast_3d(
        cv2.warpPerspective(
            img_src.astype(np.float32),
            P,
            (target_width, target_height),
            flags=cv2.INTER_LINEAR,
        )
    ).astype(dtype)

    return img_dst
