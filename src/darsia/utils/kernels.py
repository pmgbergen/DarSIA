"""Standard kernels accompanied by abstract base class.

Provided:
    * Linear kernel
    * Gaussian kernel

"""

from abc import ABC, abstractmethod

import numba
import numpy as np


class BaseKernel(ABC):
    """Abstract base class for kernel."""

    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute kernel between two arrays.

        Args:
            x (np.ndarray): first array
            y (np.ndarray): second array

        Returns:
            np.ndarray: kernel between x and y

        """
        pass

    def linear_combination(self, signal: np.ndarray, supports, interpolation_weights):
        num_supports = len(supports)
        if num_supports == 0:
            return np.zeros_like(signal)
        # NOTE: Shape is not clear at input as it may be used via advenced indexing
        from time import time

        tic = time()
        output = interpolation_weights[0] * self.__call__(signal, supports[0])
        print("interpolation call", time() - tic)
        for n in range(1, num_supports):
            tic = time()
            output += interpolation_weights[n] * self.__call__(signal, supports[n])
            print("interpolation call", time() - tic)

        return output


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

    def linear_combination(self, signal: np.ndarray, supports, interpolation_weights):
        # Make numba version from linear_combination for GaussianKernel
        @numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
        def _linear_combination_numba(
            signal: np.ndarray, supports, interpolation_weights, gamma
        ):
            num_supports = len(supports)
            diff = signal - supports[0]
            output = interpolation_weights[0] * np.exp(
                -gamma * np.sum(np.multiply(diff, diff), axis=-1)
            )
            for n in range(1, num_supports):
                diff = signal - supports[n]
                output += interpolation_weights[n] * np.exp(
                    -gamma * np.sum(np.multiply(diff, diff), axis=-1)
                )
            return output

        return _linear_combination_numba(
            signal, supports, interpolation_weights, self.gamma
        )
