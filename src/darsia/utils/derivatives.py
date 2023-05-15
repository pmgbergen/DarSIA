"""
Contains several functions for taking derivatives of images in single-color
space. All functions are based on numpy.diff().
"""
from __future__ import annotations

from typing import Optional

import numpy as np

import darsia as da


def backward_diff(
    im: np.ndarray, axis: int, dim: int = 2, h: Optional[float] = None
) -> np.ndarray:
    """
    Backward difference of image matrix in direction of axis.
    Args:
        im (np.ndarray): image array in single-color space
        axis (int): axis along which the difference is taken
        dim (int): dimension of image array
        h (Optional[float]): grid spacing
    Returns:
        np.ndarray: backward difference image matrix
    """
    assert axis < dim, "axis must be smaller than dimension"
    if h is None:
        return np.diff(im, axis=axis, append=da.array_slice(im, axis, -1, None, 1))
    else:
        return np.diff(im, axis=axis, append=da.array_slice(im, axis, -1, None, 1)) / h


def forward_diff(
    im: np.ndarray, axis: int, dim: int = 2, h: Optional[float] = None
) -> np.ndarray:
    """
    Forward difference of image matrix in direction of axis.
    Args:
        im (np.ndarray): image array
        axis (int): axis along which the difference is taken
        dim (int): dimension of image array
        h (Optional[float]): grid spacing
    Returns:
        np.ndarray: forward difference image matrix
    """
    assert axis < dim, "axis must be smaller than dimension"
    if h is None:
        return np.diff(im, axis=axis, prepend=da.array_slice(im, axis, 0, 1, 1))
    else:
        return np.diff(im, axis=axis, prepend=da.array_slice(im, axis, 0, 1, 1)) / h


def laplace(
    im: np.ndarray, axis: Optional[int] = None, dim: int = 2, h: Optional[float] = None
) -> np.ndarray:
    """
    Laplace operator with homogeneous boundary conditions of image matrix
    in direction of axis.
    Args:
        im (np.ndarray): image array
        axis (int): axis along which the difference is taken
        dim (int): dimension of image array
        h (Optional[float]): grid spacing
    Returns:
        np.ndarray: horizontal Laplace image matrix
    """
    if axis is None:
        laplace: np.ndarray = np.zeros_like(im)
        for ax in range(dim):
            laplace += 0.5 * (
                backward_diff(
                    im=forward_diff(im=im, axis=ax, dim=dim, h=h), axis=ax, dim=dim, h=h
                )
                + forward_diff(
                    im=backward_diff(im=im, axis=ax, dim=dim, h=h),
                    axis=ax,
                    dim=dim,
                    h=h,
                )
            )
    else:
        assert axis < dim, "axis must be smaller than dimension"
        laplace = 0.5 * (
            backward_diff(
                im=forward_diff(im=im, axis=axis, dim=dim, h=h), axis=axis, dim=dim, h=h
            )
            + forward_diff(
                im=backward_diff(im=im, axis=axis, dim=dim, h=h),
                axis=axis,
                dim=dim,
                h=h,
            )
        )
    return laplace
