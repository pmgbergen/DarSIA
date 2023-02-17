"""
Module containing a linear (affine) conversion from signals to data.
"""

from typing import Optional

import numpy as np

import darsia


class LinearModel(darsia.Model):
    """
    Linear model, applying an affine conversion for signals to data.

    """

    def __init__(
        self,
        key: str = "",
        **kwargs,
    ) -> None:

        self._scaling = kwargs.get(key + "scaling", 1.0)
        self._offset = kwargs.get(key + "offset", 0.0)

        self.volumes = None

    def update(
        self,
        scaling: Optional[float] = None,
        offset: Optional[float] = None,
    ) -> None:
        """
        Update of internal parameters.

        Args:
            scaling (float, optional): slope
            offset (float, optional): offset

        """

        if scaling is not None:
            self._scaling = scaling

        if offset is not None:
            self._offset = offset

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
        Application of linear model.

        Args:
            img (np.ndarray): image

        Returns:
            np.ndarray: converted signal

        """
        return self._scaling * img + self._offset

    def calibrate(self, images: list[np.ndarray], option: str = "rate", **kwargs):
        pass  # TODO
