"""
Contains several functions for taking derivatives of images in single-color space. All functions are based on numpy.diff().
"""
from __future__ import annotations

from typing import Optional
import numpy as np

class Derivatives:
    """
    Class for taking derivatives of images in single-color space. All functions are based on numpy.diff().
    """
    @staticmethod
    def array_slice(a, axis, start, end, step=1):
        """
        Slice array along axis.

        Sugestion found in the thread:
        https://stackoverflow.com/questions/24398708/slicing-a-numpy-array-along-a-dynamically-specified-axis
        """
        return a[(slice(None),) * (axis % a.ndim) + (slice(start, end, step),)]

    @classmethod
    def backward_diff(cls, im: np.ndarray, axis: int) -> np.ndarray:
        """
        Backward difference of image matrix in direction of axis.

        Arguments:
            im (np.ndarray): image array in single-color space
            axis (int): axis along which the difference is taken


        Returns:
            np.ndarray: backward difference image matrix
        """
        # dimension = im.ndim
        assert axis < im.ndim, "axis must be smaller than dimension"
        return np.diff(im, axis=axis, append=cls.array_slice(im, axis, -1, None, 1))

    @classmethod
    def forward_diff(cls, im: np.ndarray, axis: int) -> np.ndarray:
        """
        Forward difference of image matrix in direction of axis.

        Arguments:
            im (np.ndarray): image array
            axis (int): axis along which the difference is taken

        Returns:
            np.ndarray: forward difference image matrix
        """
        assert axis < im.ndim, "axis must be smaller than dimension"
        return np.diff(im, axis=axis, prepend=cls.array_slice(im, axis, 0, 1, 1))

    @classmethod
    def laplace(cls, im: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """
        Laplace operator with homogeneous boundary conditions of image matrix
        in direction of axis.

        Arguments:
            im (np.ndarray): image array
            axis (int): axis along which the difference is taken


        Returns:
            np.ndarray: horizontal Laplace image matrix
        """
        dimension = im.ndim
        if axis is None:
            laplace = 0
            for ax in range(dimension):
                laplace += 0.5*(cls.backward_diff(cls.forward_diff(im, ax),ax) + cls.forward_diff(cls.backward_diff(im, ax),ax))

        else:
            assert axis < dimension, "axis must be smaller than dimension"
            laplace = 0.5*(cls.backward_diff(cls.forward_diff(im, axis), axis) + cls.forward_diff(cls.backward_diff(im, axis), axis))

        return laplace
