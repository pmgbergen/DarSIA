"""
Module containing a linear (affine) conversion from signals to data.
"""

from typing import Optional

import numpy as np

import darsia


class LinearModel(darsia.Model):
    """
    Linear model, applying an affine conversion for signals to data,
    with additional possibility for thresholding.

    """

    def __init__(
        self,
        scaling: float = 1,
        offset: float = 0,
        # TODO add possibility for None.
        threshold_lower: float = 0,
        threshold_upper: float = 1,
    ):
        self._scaling = scaling
        self._offset = offset
        self._threshold_lower = threshold_lower
        self._threshold_upper = threshold_upper

        self.volumes = None

    def update(
        self,
        scaling: Optional[float] = None,
        offset: Optional[float] = None,
        threshold_lower: Optional[float] = None,
        threshold_upper: Optional[float] = None,
    ) -> None:
        """
        Update of internal parameters.

        Args:
            scaling (float, optional): slope
            offset (float, optional): offset
            threshold_lower (float, optional): lower threshold parameter, effective after scaling
            threshold_upper (float, optional): upper threshold parameter, effective after scaling

        """

        if scaling is not None:
            self._scaling = scaling

        if offset is not None:
            self._offset = offset

        if threshold_lower is not None:
            self._threshold_lower = threshold_lower

        if threshold_upper is not None:
            self._threshold_upper = threshold_upper

    def update_model_parameters(self, parameters: np.ndarray) -> None:
        """
        Short cut to update scaling and offset parameters using a
        general function signature.

        Args:
            parameters (np.ndarray): 2-array containing scaling and offset values.

        """
        self.update(scaling=parameters[0], offset=parameters[1])

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Application of linear model and additional thresholding.

        Args:
            img (np.ndarray): image

        Returns:
            np.ndarray: converted signal

        """
        return np.clip(
            self._scaling * img + self._offset,
            self._threshold_lower,
            self._threshold_upper,
        )

    def calibrate(self, images: list[np.ndarray], option: str = "rate", **kwargs):
        pass # TODO
