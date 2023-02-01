"""
Contains several functions for taking derivatives of images in single-color space
"""
from __future__ import annotations

import numpy as np


def backward_diff(im: np.ndarray, axis: int, dimension: int = 2) -> np.ndarray:
    """
    Backward difference of image matrix in direction of axis.

    Arguments:
        im (np.ndarray): image array
        axis (int): axis along which the difference is taken
        dimension (int): dimension of the image array (default: 2), 
                         only 2,3 allowed

    Returns:
        np.ndarray: backward difference image matrix
    """
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


def forward_diff(im: np.ndarray, axis: int, dimension: int = 2) -> np.ndarray:
    """
    Forward difference of image matrix in direction of axis.

    Arguments:
        im (np.ndarray): image array
        axis (int): axis along which the difference is taken
        dimension (int): dimension of the image array (default: 2),
                            only 2,3 allowed
    Returns:
        np.ndarray: forward difference image matrix
    """
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





def laplace_x(im: np.ndarray) -> np.ndarray:
    """
    Laplace operator with homogeneous boundary conditions of image matrix
    in horizontal direction.

    Arguments:
        im (np.ndarray): image array

    Returns:
        np.ndarray: horizontal Laplace image matrix
    """
    return 0.5 * (
        forward_diff_x(backward_diff_x(im)) + backward_diff_x(forward_diff_x(im))
    )


def laplace_y(im: np.ndarray) -> np.ndarray:
    """
    Laplace operator with homogeneous boundary conditions of image matrix
    in vertical direction.

    Arguments:
        im (np.ndarray): image array

    Returns:
        np.ndarray: vertical Laplace image matrix
    """
    return 0.5 * (
        forward_diff_y(backward_diff_y(im)) + backward_diff_y(forward_diff_y(im))
    )


def laplace(im: np.ndarray) -> np.ndarray:
    """
    Laplace operator with homogeneous boundary conditions of image matrix.

    Arguments:
        im (np.ndarray): image array

    Returns:
        np.ndarray: Laplace image matrix
    """
    return laplace_x(im) + laplace_y(im)



def partial_difference_forward(im: np.ndarray, axis: int) -> np.ndarray:
    """
    Partial difference of image matrix in given direction.

    Arguments:
        im (np.ndarray): image array
        axis (int): direction of partial difference

    Returns:
        np.ndarray: partial difference image matrix
    """
    return np.diff(im, axis=axis, prepend=im[:1, :])
