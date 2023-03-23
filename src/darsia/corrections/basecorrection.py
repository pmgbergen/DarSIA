"""Module containing a base implementation of an abstract correction."""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

import darsia


class BaseCorrection(ABC):
    def __call__(
        self,
        image: Union[np.ndarray, darsia.Image],
        return_image: bool = False,
    ) -> Union[np.ndarray, darsia.Image]:
        """
        Manager for color correction depending on the input type.

        Args:
            image (array or Image): image
            return_image (bool): flag controlling whether the image is returned,
                only relevant for input of type Image.

        """
        if isinstance(image, np.ndarray):
            return self.correct_array(image)
        elif isinstance(image, darsia.Image):
            if image.series and hasattr(self, "correct_array_series"):
                # Apply transformation to entrie space time image
                image.img = self.correct_array_series(image.img)
            elif image.series:
                # Use external data container for shape altering corrections
                corrected_slices = []

                # Consider each time slice separately
                for time_index in range(image.time_num):
                    if image.scalar:
                        # Apply transformation to single time slices for scalar data
                        corrected_slices.append(
                            self.correct_array(image.img[..., time_index])
                        )
                    else:
                        # Apply transformation to single time slices for vectorial data
                        corrected_slices.append(
                            self.correct_array(image.img[..., time_index, :])
                        )

                # Stack slices together again
                image.img = np.stack(corrected_slices, axis=image.space_dim)

            else:
                # Apply transformation to single image
                image.img = self.correct_array(image.img)

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
