"""Module containing a base implementation of an abstract correction."""

from abc import ABC, abstractmethod
from pathlib import Path
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

        Parameters
        ----------
        image : array or Image
            Image.
        overwrite : bool
            Flag controlling whether the original image is overwritten
            or the correction is applied to a copy. This option has to be used with
            case.

        Returns
        -------
        array or Image
            Corrected image, data type depends on input.
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

            # Apply corrections to metadata
            meta_update = self.correct_metadata(image.metadata())

            if overwrite:
                # Overwrite original image
                image.img = img
                image.update_metadata(meta_update)
                return image
            else:
                # Return corrected copy of image
                meta = image.metadata()
                meta.update(meta_update)
                return type(image)(img, **meta)

    @abstractmethod
    def correct_array(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """Correction routine on array level, to be specified for tailored correction.

        Parameters
        ----------
        image : array
            Image array.

        Returns
        -------
        array
            Corrected image array.
        """
        pass

    def correct_metadata(self, metadata: dict = {}) -> dict:
        """Correction routine on metadata level.

        Parameters
        ----------
        metadata : dict
            Metadata dictionary.

        Returns
        -------
        dict
            Corrected metadata dictionary.
        """
        return {}

    # ! ---- I/O ----

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the correction to a file.

        The method should store a npz file, continaing the class name and
        required data for loading the correction from file.

        Parameters
        ----------
        path : str
            Path to the file.
        """
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load the correction from a file.

        The method should load a npz file, containing the class name and
        required data for loading the correction from file.

        Parameters
        ----------
        path : str
            Path to the file.
        """
        ...
