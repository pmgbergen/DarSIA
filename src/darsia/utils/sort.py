"""Module collecting utilities for sorting."""

from typing import Union

import numpy as np

import darsia


def sort_quad(
    pts: Union[darsia.VoxelArray, darsia.CoordinateArray]
) -> Union[darsia.VoxelArray, darsia.CoordinateArray]:
    """Sorts the four points of a quadrilateral in a clockwise fashion.

    Args:
        pts (array): 4x2 array of points

    Returns:
        array: 4x2 array of points sorted clockwise
    """
    if isinstance(pts, darsia.CoordinateArray):
        raise NotImplementedError("CoordinateArray not implemented yet.")
        # Need to take care of orentation

    # Sort from lowest to highest 0-component
    pts_top_to_bottom = pts[np.argsort(pts[:, 0])]
    top_pts = pts_top_to_bottom[:2]
    bottom_pts = pts_top_to_bottom[2:]

    # Sort top_pts from lowest to highest 1-component
    top_pts = top_pts[np.argsort(top_pts[:, 1])]

    # Sort bottom_pts from lowest to highest 1-component
    bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 1])]

    # Sort pts_top_to_bottom such that:
    # - the first point is among the first two points and has lower 1 component
    # - the second point is among the second two points and has lower 1 component
    # - the third point is among the second two points and has higher 1 component
    # - the fourth point is among the first two points and has higher 1 component
    sorted_pts = np.array(
        [
            top_pts[0],
            bottom_pts[0],
            bottom_pts[1],
            top_pts[1],
        ]
    )
    return sorted_pts
