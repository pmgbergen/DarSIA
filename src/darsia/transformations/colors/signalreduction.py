"""
Module defining a plain signal reduction

"""

import numpy as np


class SignalReduction:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Method defining the conversion to a scalar signal.
        By default it is assumed, that the input is scalar.
        If not, overwritting the method is needed.

        Args:
            img (np.ndarray): (here scalar) signal

        Returns:
            np.ndarray: scalar signal

        """
        return img
