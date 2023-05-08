"""Module containing a base implementation of an abstract correction."""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

import darsia


class BaseCorrection(ABC):
    """Abstract base correction, providing template for tailored corrections."""

    def __call__(
        self,
        image: Union[np.ndarray, darsia.Image],
        overwrite: bool = False,
    ) -> Union[np.ndarray, darsia.Image]:
        """Workflow for any correction routine.

        Args:
            image (array or Image): image
            overwrite (bool): flag controlling whether the original image is
                overwritten or the correction is applied to a copy.

        Returns:
            array or Image: corrected image, data type depends on input.

        """
        if isinstance(image, np.ndarray):
            if overwrite:
                # Overwrite original array
                image = self.correct_array(image)
                return image
            else:
                # Return corrected copy of array
                return self.correct_array(image.copy())

        elif isinstance(image, darsia.Image):
            img = image.img if overwrite else image.img.copy()

            if image.series and hasattr(self, "correct_array_series"):
                # Apply transformation to entrie space time image
                img = self.correct_array_series(img)
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
                img = np.stack(corrected_slices, axis=image.space_dim)

            else:
                # Apply transformation to single image
                img = self.correct_array(img)

            if overwrite:
                # Overwrite original image
                image.img = img
                return image
            else:
                # Return corrected copy of image
                meta = image.metadata()
                return type(image)(img, **meta)

    @abstractmethod
    def correct_array(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """Correction routine on array level, to be specified for tailored correction.

        Args:
            image (array): image array.

        Returns:
            array: corrected image array.

        """
        pass
