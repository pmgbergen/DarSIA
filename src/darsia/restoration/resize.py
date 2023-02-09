"""
Module containing wrappers to resize routines
from skimage and cv2. Access is given through
objects.

"""

from typing import Optional

import cv2
import numpy as np


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

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Wrapper to cv2.resize.

        Args:
            img (np.ndarray): image

        Returns:
            np.ndarray: resized image

        """
        # Convert data type
        img = img if self.dtype is None else img.astype(self.dtype)

        # Apply resizing
        if self.interpolation is None:
            return cv2.resize(
                img,
                dsize=self.dsize,
                fx=self.fx,
                fy=self.fy,
            )
        else:
            return cv2.resize(
                img,
                dsize=self.dsize,
                fx=self.fx,
                fy=self.fy,
                interpolation=self.interpolation,
            )
