"""
Module defining a plain signal reduction

"""

import numpy as np


class SignalReduction:
    """Base class reducing a (possibly multi-channel) signal to a scalar array.

    The default ``__call__`` is the identity, assuming a scalar input; subclasses
    override it.
    """

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Reduce the input to a scalar signal.

        Args:
            img (np.ndarray): (here scalar) signal

        Returns:
            np.ndarray: scalar signal

        """
        return img
