"""Module containing a base implementation of an abstract correction."""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

import darsia


class BaseCorrection(ABC):
    def __call__(
        self,
        image: Union[np.ndarray, darsia.Image, darsia.GeneralImage],
        return_image: bool = False,
    ) -> Union[np.ndarray, darsia.Image, darsia.GeneralImage]:
        """
        Manager for color correction depending on the input type.

        Args:
            image (array or Image): image
            return_image (bool): flag controlling whether the image is returned,
                only relevant for input of type Image.

        """
        if isinstance(image, np.ndarray):
            return self.correct_array(image)
        else:
            image.img = self.correct_array(image.img)
            if return_image:
                return image

    @abstractmethod
    def correct_array(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """Correction routine on array level.

        Args:
            image (array): image array.

        Returns:
            array: corrected image array.

        """
        pass
