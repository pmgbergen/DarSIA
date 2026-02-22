"""Module with tools for volume averaging."""

import logging
from typing import Optional, Union, overload

import numpy as np
import scipy.ndimage

import darsia

logger = logging.getLogger(__name__)


class REV:
    """Class defining an representative elementary volume."""

    def __init__(self, size: Union[float, tuple[float]], img: darsia.Image) -> None:
        """Initialization of a REV.

        Args:
            size (float or tuple of float): size of REV in length units
            img (Image): image, used for determining the number of voxels

        """
        if isinstance(size, float):
            size = [size] * img.coordinatesystem.dim
        self.size: int = max(
            [
                img.coordinatesystem.num_voxels(size[i], axis=["x", "y", "z"][i])
                for i in range(img.coordinatesystem.dim)
            ]
        )
        """Size of the REV in number of voxels."""


class VolumeAveraging:

    def __init__(
        self, rev: REV, mask: darsia.Image, labels: Optional[darsia.Image] = None
    ) -> None:
        """Constructor.

        Args:
            rev (REV): representative elementary volume
            mask (Image): mask
            labels (Image): labels; activating heterogeneous analysis, if not None

        """
        self.rev_size = rev.size
        """Size of the REV."""
        self.mask = mask
        """Mask."""
        self.labels = labels
        """Labels."""
        self.mean_pore_volume = self._heterogeneous_uniform_filter(
            self.mask.astype(float).img
        )
        """Mean pore volume."""
        tol = 1e-12
        self.zero_indices = np.where(self.mean_pore_volume < tol)
        """Zero indices in the mean pore volume."""

        # User output
        logger.info(
            f"""Number of zero indices in pore volume identification: """
            f"""{len(self.zero_indices[0])}"""
        )

    @overload  # type: ignore [override]
    def __call__(self, img: np.ndarray) -> np.ndarray: ...

    @overload  # type: ignore [override]
    def __call__(self, img: darsia.Image) -> darsia.Image: ...

    def __call__(
        self, img: Union[np.ndarray, darsia.Image]
    ) -> Union[np.ndarray, darsia.Image]:
        """Application of volume averaging.

        Args:
            img (np.ndarray or Image): image

        Returns:
            np.ndarray or Image: volume averaged image (same type as input)

        """
        if isinstance(img, np.ndarray):
            return self._average_array(img)
        elif isinstance(img, darsia.Image):
            result = img.copy()
            result.img = self._average_array(img.img)
            return result

    def _average_array(self, arr: np.ndarray) -> np.ndarray:
        """Application of volume averaging to numpy array.

        Apply averaging to each channel of a 3D array.

        Args:
            arr (np.ndarray): array

        Returns:
            np.ndarray: volume averaged array

        """
        if arr.ndim == 2:
            return self._average_array_single(arr)
        elif arr.ndim == 3:
            return np.stack(
                [self._average_array_single(arr[..., i]) for i in range(arr.shape[-1])],
                axis=-1,
            )
        else:
            raise ValueError("Only 2D and 3D arrays are supported.")

    def _average_array_single(self, arr: np.ndarray) -> np.ndarray:
        """Application of volume averaging to two-dimensional numpy array.

        Args:
            arr (np.ndarray): two-dimensional array

        Returns:
            np.ndarray: two-dimensional volume averaged array

        """
        masked_data = np.multiply(arr, self.mask.img)
        mean_masked_data = self._heterogeneous_uniform_filter(masked_data)
        result = np.divide(mean_masked_data, self.mean_pore_volume)
        result[self.zero_indices] = 0
        return result

    def _heterogeneous_uniform_filter(self, data: np.ndarray) -> np.ndarray:
        """Application of a uniform filter to heterogeneous data.

        Args:
            data (np.ndarray): data

        Returns:
            np.ndarray: filtered data

        """
        if self.labels is None:
            return scipy.ndimage.uniform_filter(data, size=self.rev_size)
        else:
            raise NotImplementedError("Heterogeneous analysis is not yet implemented.")


def volume_average(img: darsia.Image, mask: darsia.Image, size: float) -> darsia.Image:
    """Fast-access function for volume averaging.

    Note: For repeated calls, it is recommended to create a VolumeAveraging object.

    Args:
        img (Image): image
        mask (Image): mask
        size (float): size of the REV in length units

    Returns:
        Image: volume averaged image

    """
    return VolumeAveraging(rev=REV(size=size, img=img), mask=mask)(img)
