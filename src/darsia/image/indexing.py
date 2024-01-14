"""Auxiliary interpreters for indexing of axes used for images.

Conventions. For unified use in the code base,
we introduce conventions of coordinate systems.

In 2d: The Cartesian coordinate system is
defined as x-y-coordinate system. E.g.
for an image with ij-matrix indexing the first
component corresponds to the y-axis with reversed
indexing, while the second axis corresponds
to the standard x-axis.

In 3d: The Cartesian coordinate system is
defines as xyz coordinate system. E.g.
for an image with ijk-matrix indexing, the first
component corresponds to the z axis with
reversed indexing, the second component corresponds
to the y-axis, and the third component corresponds to
the x-axis.

"""
from __future__ import annotations

from typing import Union

import numpy as np


def to_matrix_indexing(axis: Union[str, int], indexing: str) -> str:
    """Conversion of single axis in Cartesian to matrix indexing.

    Args:
        axis (str or int): input axis in Cartesian indexing sense.
        indexing (str): reference Cartesian indexing, also identifying
            the dimension.

    Returns:
        str: converted axis in matrix indexing sense.

    """
    assert indexing in "xy", "xyz"

    # Convert numeric axis description
    if isinstance(axis, int):
        axis = "xyz"[axis]

    if indexing == "xy":
        if axis == "x":
            return "j"
        elif axis == "y":
            return "i"
    elif indexing == "xyz":
        if axis == "x":
            return "k"
        elif axis == "y":
            return "j"
        elif axis == "z":
            return "i"

    raise ValueError


def to_cartesian_indexing(axis: Union[str, int], indexing: str) -> str:
    """Conversion of single axis in matrix indexing to Cartesian indexing.

    Args:
        axis (str or int): input axis in matrix indexing sense.

    Returns:
        str: converted axis in Cartesian indexing sense.

    """
    # Convert numeric axis description
    if isinstance(axis, int):
        axis = "ijk"[axis]

    if indexing == "ij":
        if axis == "i":
            return "y"
        elif axis == "j":
            return "x"
    elif indexing == "ijk":
        if axis == "i":
            return "z"
        elif axis == "j":
            return "y"
        elif axis == "k":
            return "x"

    raise ValueError


def interpret_indexing(axis: str, indexing: str) -> tuple[int, bool]:
    """Interpretation of axes and their indexing.

    Args:
        axis (str): target axis, e.g., "x"
        indexing (str): indexing of an image, e.g., "ijk"

    Returns:
        int: component corresponding to the axis. Covered: "x", "y", "z", "i", "j", "k".
        bool: flag controlling whether the axis has to be reverted when converting.
            Covered: "xyz", "ijk", and reduced cases in 1d and 2d.

    Raises:
        ValueError: if not supported combination used as input.

    """

    # Consider all possible combinations.

    # ! ---- 1D Cartesian indexing

    if indexing == "x":
        if axis == "x":
            return 0, False
        elif axis == "i":
            return 0, False

    # ! ---- 1D matrix indexing

    elif indexing == "i":
        if axis == "x":
            return 0, False
        elif axis == "i":
            return 0, False

    # ! ---- 2D Cartesian indexing

    elif indexing == "xy":
        if axis == "x":
            return 0, False
        elif axis == "y":
            return 1, False
        elif axis == "i":
            return 1, True
        elif axis == "j":
            return 0, False

    # ! ---- 2D matrix indexing

    elif indexing == "ij":
        if axis == "x":
            return 1, False
        elif axis == "y":
            return 0, True
        elif axis == "i":
            return 0, False
        elif axis == "j":
            return 1, False

    # ! ---- 3D Cartesian indexing

    if indexing == "xyz":
        if axis == "x":
            return 0, False
        elif axis == "y":
            return 1, False
        elif axis == "z":
            return 2, False
        elif axis == "i":
            return 2, True
        elif axis == "j":
            return 0, False
        elif axis == "k":
            return 1, True

    # ! ---- 3D matrix indexing

    elif indexing == "ijk":
        if axis == "x":
            return 1, False
        elif axis == "y":
            return 2, True
        elif axis == "z":
            return 0, True
        elif axis == "i":
            return 0, False
        elif axis == "j":
            return 1, False
        elif axis == "k":
            return 2, False

    # If the method reaches this point, something went wrong.
    # This fact is used as safety check.
    raise ValueError


def matrixToCartesianIndexing(img: np.ndarray, dim: int = 2) -> np.ndarray:
    """
    Reordering data indexing converting from (row,col) to (x,y) indexing.

    The conventional matrix indexing uses the (row, col) format, such
    that the top left corner is the (0,0) pixel. On the other hand,
    Cartesian indexing uses the (x,y) order and thereby identifies
    the lower left corner by (0,0). This routine is in particular useful
    when communicating image data to conventional simulators, which use
    the Cartesian indexing.

    Arguments:
        np.ndarray: image array with matrix indexing
        dim (int): dimension of the image, default is 2

    Returns:
        np.ndarray: image array with Cartesian indexing
    """
    if dim == 1:
        pass
    elif dim == 2:
        # Two operations are require: Swapping axis and flipping the vertical axis.
        # Exchange first and second component, to change from (row,col) to (x,y) format.
        img = np.swapaxes(img, 0, 1)
        # Flip the orientation of the 2nd axis, such that later y=0 corresonds to the bottom.
        img = np.flip(img, 1)
    elif dim == 3:
        # Need to convert from i,j,k to x,y,z and flip the z-axis and x-axis.
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 0, 1)
        img = np.flip(img, 1)
        img = np.flip(img, 2)
    else:
        raise ValueError("Only 1d, 2d, and 3d images are supported.")

    return img


def cartesianToMatrixIndexing(img: np.ndarray) -> np.ndarray:
    """
    Reordering data indexing, converting from (x,y) to (row,col) indexing.

    Inverse to matrixToCartesianIndexing.

    NOTE: Assumes 2d images.

    Arguments:
        np.ndarray: image array with Cartesian indexing

    Returns:
        np.ndarray: image array with matrix indexing
    """
    # Two operations are require: Swapping axis and flipping the vertical axis.

    # Flip the orientation of the second axis, such that later row=0 is located at the top.
    img = np.flip(img, 1)

    # Exchange first and second component, to change from (x,y) to (row,col) format.
    img = np.swapaxes(img, 0, 1)

    return img
