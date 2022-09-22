"""
Module that contains a class which provides the capabilities to
analyze concentrations/saturation profiles based on image comparison.
"""

from typing import Optional, Union

import numpy as np
import skimage

import daria


class ConcentrationAnalysis:
    """
    Class providing the capabilities to determine concentration/saturation
    profiles based on image comparison, and tuning of concentration-intensity
    maps.

    Attributes:

    """

    def __init__(self, img_base: Optional[np.ndarray], **kwargs) -> None:
        """
        Constructor of ConcentrationAnalysis.

        Args:
            img_base (np.ndarray): baseline image
            kwargs: Optional keyword arguments
                tvd_parameter (float): tuning parameter of the TVD algorithm
        """
        # TODO switch to daria images
        self._check_img_compatibility(img_base)
        self.baseline = img_base
        self.scaling_factor = 1.0
        self.offset = 0.0

        self.tvd_parameter = kwargs.pop("tvd_parameter", 0.1)

    def update_baseline(self, img: np.ndarray) -> None:
        """
        Update of the baseline image.

        Args:
            img (np.ndarray): image array
        """
        self._check_img_compatibility(img)
        self.baseline = img

    def _check_img_compatibility(self, img: np.ndarray):
        """Check whether the image is a mono-colored image."""
        assert len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)

    def __call__(self, img: np.ndarray, resize_factor: float) -> np.ndarray:
        """Extract concentration based on a reference image and rescaling.

        Args:
            img (np.ndarray): probing image
            resize_factor (float): factor for resizing images

        Returns:
            np.ndarray: concentration
        """
        # Check compatibility of the input image
        self._check_img_compatibility(img)

        # Take (unsigned) difference
        diff = skimage.util.compare_images(img, self.baseline, method="diff")

        # Resize the image
        diff = daria.utils.resolution.resize(diff, resize_factor * 100)

        # Apply smoothing filter
        # TODO rm
        # diff = diff[:,:,0] if len(diff.shape) == 3 else diff
        diff = skimage.restoration.denoise_tv_chambolle(diff, weight=self.tvd_parameter)

        # TODO Calibration needed. For now the images are transformed to float data
        # and stretched onto the interval [0,1].
        diff = self.convert_signal_to_concentration(diff)

        return diff

    def tune_signal_concentration_map(
        self, images: Union[daria.Image, list[daria.Image]], target: float
    ) -> None:
        """
        Tune the scaling factor and offset used in the signal-concentration conversion routine.

        Args:
            images (single or list of daria images): daria images
            target (float): slope of the total concentration over time if multiple images
                are prescibed, otherwise the total concentration associated with a single
                image

        """
        pass

    def convert_signal_to_concentration(self, img: np.ndarray) -> np.ndarray:
        """
        Make sure that the data is in float format an lies in the range [0,1]

        Args:
            img (np.ndarray): data array

        Returns:
            np.ndarray: Rescaled concentration in the interval [0,1].
        """
        return np.clip(
            skimage.util.img_as_float(self.scaling_factor * img + self.offset), 0, 1
        )

    def total_concentration(
        self, concentration: np.ndarray, weight: Union[float, np.ndarray] = 1.0
    ) -> float:
        """
        Determine the total concentration of a spatial concentration map.

        Args:
            concentration (np.ndarray): concentration data.
            weight (float or np.ndarray): Optional, possibly heterogeneous scalar, local
                weight. If an array is passed, the dimensions have to be compatible with
                those of concentration.

        Returns:
            float: The integral over the spatial, weighted concentration map.
        """
        if isinstance(weight, np.ndarray):
            assert concentration.shape == weight.shape

        # Integral of locally weighted concentration values
        return np.sum(np.multiply(weight, concentration))
