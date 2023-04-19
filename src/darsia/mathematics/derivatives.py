"""
Contains several functions for taking derivatives of images in one color space
"""
from __future__ import annotations

import numpy as np


def backward_diff_x(im: np.ndarray) -> np.ndarray:
    """
    Backward difference of image matrix in horizontal direction.

    Arguments:
        im (np.ndarray): image array

    Returns:
        np.ndarray: backward difference image matrix
    """
    return np.diff(im, axis=1, append=im[:, -1:])


def forward_diff_x(im: np.ndarray) -> np.ndarray:
    """
    Forward difference of image matrix in horizontal direction.

    Arguments:
        im (np.ndarray): image array

    Returns:
        np.ndarray: forward difference image matrix
    """
    return np.diff(im, axis=1, prepend=im[:, :1])


def backward_diff_y(im: np.ndarray) -> np.ndarray:
    """
    Backward difference of image matrix in vertical direction.

    Arguments:
        im (np.ndarray): image array

    Returns:
        np.ndarray: backward difference image matrix
    """
    return np.diff(im, axis=0, append=im[-1:, :])


def forward_diff_y(im: np.ndarray) -> np.ndarray:
    """
    Forward difference of image matrix in vertical direction.

    Arguments:
        im (np.ndarray): image array

    Returns:
        np.ndarray: forward difference image matrix
    """
    return np.diff(im, axis=0, prepend=im[:1, :])


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
