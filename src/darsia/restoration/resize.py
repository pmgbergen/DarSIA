"""
Module containing wrappers to resize routines
from skimage and cv2. Access is given through
objects.

"""

from typing import Optional, Union

import cv2
import numpy as np

import darsia


class Resize:
    """
    Object for resizing 2d arrays.

    Example:

    import darsia
    options = {
        "example resize x": 0.2,
        "example resize y": 0.5,
        "example interpolation": "inter_nearest",
    }
    resizer = darsia.Resize(key = "example ", **options)
    img_small = resizer(img_large)

    """

    def __init__(
        self,
        dsize: Optional[tuple[int]] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        interpolation: Optional[str] = None,
        dtype=None,
        key: str = "",
        **kwargs,
    ) -> None:
        """
        Args:
            dsize (tuple of int, optional): desired number of col and row
                after transformation.
            fx (float, optional): resize factor in x-dimension.
            fy (float, optional): resize factor in y-dimension.
            interpolation (str, optional): interpolation method, default: None, invoking
                the default option in cv2.resize.
            dtype: conversion dtype before resizing; noting happens if None

        """

        # Cache parameters
        self.dsize = kwargs.get(key + "resize dsize", None) if dsize is None else dsize
        general_f = kwargs.get(key + "resize", None)
        self.fx = kwargs.get(key + "resize x", general_f) if fx is None else fx
        self.fy = kwargs.get(key + "resize y", general_f) if fy is None else fy
        self.dtype = kwargs.get(key + "resize dtype", None) if dtype is None else dtype

        # Safety checks - double check resize options
        if self.dsize is None:
            self.fx = 1 if self.fx is None else self.fx
            self.fy = 1 if self.fy is None else self.fy

        # Convert to CV2 format
        # NOTE: The default value for interpolation in cv2 is depending on whether
        # the image is shrinked or enlarged. In case of shrinking, cv2.INTER_NEAREST
        # is chosen, as opposed to cv2.INTER_LINEAR in the case of enlarging.
        interpolation_pre = (
            kwargs.get(key + "resize interpolation", None)
            if interpolation is None
            else None
        )
        if interpolation_pre is None:
            self.interpolation = None
        elif interpolation_pre == "inter_area":
            self.interpolation = cv2.INTER_AREA
        elif interpolation_pre == "inter_linear":
            self.interpolation = cv2.INTER_LINEAR
        elif interpolation_pre == "inter_nearest":
            self.interpolation = cv2.INTER_NEAREST
        else:
            raise NotImplementedError(
                f"Interpolation option {interpolation_pre} is not implemented."
            )

        # Check for conservative rescaling
        self.is_conservative = kwargs.get(key + "resize conservative", False)

    def __call__(
        self, img: Union[np.ndarray, darsia.Image, darsia.GeneralImage]
    ) -> Union[np.ndarray, darsia.Image, darsia.GeneralImage]:
        """
        Wrapper to cv2.resize.

        Args:
            img (np.ndarray, or darsia.Image): image

        Returns:
            np.ndarray, or darsia.Image: resized image, same format as input

        """
        # Extract original image
        input_is_image = isinstance(img, darsia.Image) or isinstance(
            img, darsia.GeneralImage
        )
        img_array = img.img.copy() if input_is_image else img.copy()

        # Convert data type
        img_array = (
            img_array.copy() if self.dtype is None else img_array.astype(self.dtype)
        )

        # Apply resizing
        if self.interpolation is None:
            resized_img_array = cv2.resize(
                img_array,
                dsize=self.dsize,
                fx=self.fx,
                fy=self.fy,
            )
        else:
            resized_img_array = cv2.resize(
                img_array,
                dsize=self.dsize,
                fx=self.fx,
                fy=self.fy,
                interpolation=self.interpolation,
            )

        # Conserve the (weighted) sum
        if self.is_conservative:
            resized_img_array *= np.prod(img_array.shape[:2]) / np.prod(
                resized_img_array.shape[:2]
            )

        # Return resized image
        if input_is_image:
            # Update metadata of the darsia.Image
            resized_image = img.copy()
            resized_image.img = resized_img_array
            if isinstance(img, darsia.Image):
                resized_image.dx *= img_array.shape[1] / resized_img_array.shape[1]
                resized_image.dy *= img_array.shape[0] / resized_img_array.shape[0]
            elif isinstance(img, darsia.GeneralImage):
                for i in range(2):
                    resized_image.voxel_size[i] *= (
                        img_array.shape[i] / resized_img_array.shape[i]
                    )

            return resized_image

        else:
            return resized_img_array
