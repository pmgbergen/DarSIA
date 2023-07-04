"""Functionality for taking derivatives of single-channeled images.

All functions are based on numpy.diff().

"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

import darsia as da


def backward_diff(
    img: np.ndarray, axis: int, dim: int = 2, h: Optional[float] = None
) -> np.ndarray:
    """Backward difference of image matrix in direction of axis.

    Args:
        img (np.ndarray): image array in single-color space
        axis (int): axis along which the difference is taken
        dim (int): dimension of image array
        h (Optional[float]): grid spacing

    Returns:
        np.ndarray: backward difference image matrix

    """
    assert axis < dim, "axis must be smaller than dimension"
    if h is None:
        return np.diff(img, axis=axis, append=da.array_slice(img, axis, -1, None, 1))
    else:
        return (
            np.diff(img, axis=axis, append=da.array_slice(img, axis, -1, None, 1)) / h
        )


def forward_diff(
    img: np.ndarray, axis: int, dim: int = 2, h: Optional[float] = None
) -> np.ndarray:
    """Forward difference of image matrix in direction of axis.

    Args:
        img (np.ndarray): image array
        axis (int): axis along which the difference is taken
        dim (int): dimension of image array
        h (Optional[float]): grid spacing

    Returns:
        np.ndarray: forward difference image matrix

    """
    assert axis < dim, "axis must be smaller than dimension"
    if h is None:
        return np.diff(img, axis=axis, prepend=da.array_slice(img, axis, 0, 1, 1))
    else:
        return np.diff(img, axis=axis, prepend=da.array_slice(img, axis, 0, 1, 1)) / h


def laplace(
    img: np.ndarray,
    axis: Optional[int] = None,
    dim: int = 2,
    h: Optional[float] = None,
    diffusion_coeff: Union[np.ndarray, float] = 1,
) -> np.ndarray:
    """Laplace operator with homogeneous boundary conditions.

    Considers image matrix in direction of axis.

    Args:
        img (np.ndarray): image array
        axis (int): axis along which the difference is taken
        dim (int): dimension of image array
        h (Optional[float]): grid spacing
        diffision_coeff (Optional[np.ndarray]): diffusion coefficient

    Returns:
        np.ndarray: horizontal Laplace image matrix

    """

    if isinstance(diffusion_coeff, np.ndarray):
        assert diffusion_coeff.shape == img.shape

    if axis is None:
        laplace: np.ndarray = np.zeros_like(img)
        for ax in range(dim):
            laplace += 0.5 * (
                backward_diff(
                    img=diffusion_coeff * forward_diff(img=img, axis=ax, dim=dim, h=h),
                    axis=ax,
                    dim=dim,
                    h=h,
                )
                + forward_diff(
                    img=diffusion_coeff * backward_diff(img=img, axis=ax, dim=dim, h=h),
                    axis=ax,
                    dim=dim,
                    h=h,
                )
            )
    else:
        assert axis < dim, "axis must be smaller than dimension"
        laplace = 0.5 * (
            backward_diff(
                img=diffusion_coeff * forward_diff(img=img, axis=axis, dim=dim, h=h),
                axis=axis,
                dim=dim,
                h=h,
            )
            + forward_diff(
                img=diffusion_coeff * backward_diff(img=img, axis=axis, dim=dim, h=h),
                axis=axis,
                dim=dim,
                h=h,
            )
        )
    return laplace
