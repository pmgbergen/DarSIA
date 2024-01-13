"""Standard kernels accompanied by abstract base class.

Provided:
    * Linear kernel
    * Gaussian kernel

"""

from abc import ABC

import numpy as np


class BaseKernel(ABC):
    """Abstract base class for kernel."""

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute kernel between two arrays.

        Args:
            x (np.ndarray): first array
            y (np.ndarray): second array

        Returns:
            np.ndarray: kernel between x and y

        """
        pass


class LinearKernel(BaseKernel):
    """Linear kernel.

    NOTE: Allows to to avoid singularities by shifting the kernel by a constant.

    """

    def __init__(self, a: float = 0):
        self.a = a
        """Shift of the kernel."""

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute kernel between two arrays.

        Args:
            x (np.ndarray): first array
            y (np.ndarray): second array

        Returns:
            np.ndarray: kernel between x and y

        """
        return np.sum(np.multiply(x, y), axis=-1) + self.a


class GaussianKernel(BaseKernel):
    """Gaussian kernel."""

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma
        """Gamma parameter of the kernel."""

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute kernel between two arrays.

        Args:
            x (np.ndarray): first array
            y (np.ndarray): second array

        Returns:
            np.ndarray: kernel between x and y
        """
        return np.exp(-self.gamma * np.sum(np.multiply(x - y, x - y), axis=-1))
