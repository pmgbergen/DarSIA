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

    """

    def __init__(
        self,
        dsize: Optional[tuple[int]] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        interpolation: Optional[str] = None,
    ) -> None:
        """
        Args:
            dsize (tuple of int, optional): desired number of col and row
                after transformation.
            fx (float, optional): resize factor in x-dimension.
            fy (float, optional): resize factor in y-dimension.
            interpolation (str, optional): interpolation method, default: None, invoking
                the default option in cv2.resize.

        """
        # Cache parameters
        self.dsize = dsize
        self.fx = fx
        self.fy = fy

        # Convert to CV2 format
        if interpolation is None:
            self.interpolation = None
        elif interpolation == "inter_area":
            self.interpolation = cv2.INTER_AREA
        elif interpolation == "inter_linear":
            self.interpolation = cv2.INTER_LINEAR
        elif interpolation == "inter_nearest":
            self.interpolation = cv2.INTER_NEAREST
        else:
            raise NotImplementedError(
                f"Interpolation option {interpolation} is not implemented."
            )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Wrapper to cv2.resize.

        Args:
            img (np.ndarray): image

        Returns:
            np.ndarray: resized image

        """
        if self.interpolation is None:
            return cv2.resize(
                img.astype(np.float32),
                dsize=self.dsize,
                fx=self.fx,
                fy=self.fy,
            )
        else:
            return cv2.resize(
                img.astype(np.float32),
                dsize=self.dsize,
                fx=self.fx,
                fy=self.fy,
                interpolation=self.interpolation,
            )
