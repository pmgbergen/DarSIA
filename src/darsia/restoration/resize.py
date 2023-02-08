"""
Module containing wrappers to resize routines
from skimage and cv2. Access is given through
objects.

"""

from typing import Optional

import cv2
import numpy as np


class Resize:
    def __init__(
        self,
        factor: Optional[float] = None,
        shape: Optional[tuple[int]] = None,
        interpolation: str = "inter_area",
    ) -> None:
        self.factor = factor
        self.shape = shape
        self.interpolation = interpolation

    def __call__(self, img: np.ndarray) -> np.ndarray:

        if self.interpolation == "inter_area":
            return cv2.resize(
                img,
                self.shape,
                fx=self.factor,
                fy=self.factor,
                interpolation=cv2.INTER_AREA,
            )
        else:
            raise NotImplementedError(
                f"Interpolation option {self.interpolation} is not implemented."
            )
