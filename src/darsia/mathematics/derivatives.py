"""
Contains several functions for taking derivatives of images in single-color space. All functions are based on numpy.diff(), and are very hardcoded with respect to dimension and axes.
"""
from __future__ import annotations

from typing import Optional
import numpy as np


def backward_diff(im: np.ndarray, axis: int) -> np.ndarray:
    """
    Backward difference of image matrix in direction of axis.

    Arguments:
        im (np.ndarray): image array in single-color space
        axis (int): axis along which the difference is taken


    Returns:
        np.ndarray: backward difference image matrix
    """
    dimension = im.ndim
    assert axis < dimension, "axis must be smaller than dimension"

    if dimension == 2:
        if axis == 0:
            return np.diff(im, axis=0, append=im[-1:, :])
        elif axis == 1:
            return np.diff(im, axis=1, append=im[:, -1:])

    elif dimension == 3:
        if axis == 0:
            return np.diff(im, axis=0, append=im[-1:, :, :])
        elif axis == 1:
            return np.diff(im, axis=1, append=im[:, -1:, :])
        elif axis == 2:
            return np.diff(im, axis=2, append=im[:, :, -1:])

    else:
        raise NotImplementedError("Only 2 and 3 dimensional images are supported")


def forward_diff(im: np.ndarray, axis: int) -> np.ndarray:
    """
    Forward difference of image matrix in direction of axis.

    Arguments:
        im (np.ndarray): image array
        axis (int): axis along which the difference is taken

    Returns:
        np.ndarray: forward difference image matrix
    """
    dimension = im.ndim
    assert axis < dimension, "axis must be smaller than dimension"

    if dimension == 2:
        if axis == 0:
            return np.diff(im, axis=0, prepend=im[:1, :])
        elif axis == 1:
            return np.diff(im, axis=1, prepend=im[:, :1])

    elif dimension == 3:
        if axis == 0:
            return np.diff(im, axis=0, prepend=im[:1, :, :])
        elif axis == 1:
            return np.diff(im, axis=1, prepend=im[:, :1, :])
        elif axis == 2:
            return np.diff(im, axis=2, prepend=im[:, :, :1])

    else:
        raise NotImplementedError("Only 2 and 3 dimensional images are supported")



def laplace(im: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
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
            laplace += 0.5*(backward_diff(forward_diff(im, ax),ax) + forward_diff(backward_diff(im, ax),ax))

    else:
        assert axis < dimension, "axis must be smaller than dimension"
        laplace = 0.5*(backward_diff(forward_diff(im, axis), axis) + forward_diff(backward_diff(im, axis), axis))

    return laplace
