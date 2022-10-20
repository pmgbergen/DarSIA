from typing import Optional

import numpy as np


def bounding_box(
    coords: np.ndarray, padding: int = 0, max_size: Optional[list] = None
) -> tuple:
    """
    Determine bounding box for a set of given coordinates.

    Args:
        coords (np.ndarray): coordinate array of size N x dim.
        padding (Optional[int]): padding for the bounding box
        max_size (Optional[list]): max size of bounding box.
            Provided as list with max size in each dimension.

    Returns:
        tuple of slices: slices with ranges from min to max value
            per dimension.
    """
    bounding_box = ()

    for dim in range(coords.shape[1]):
        min_value = max(np.min(coords[:, dim]) - padding, 0)
        max_value = (
            min(np.max(coords[:, dim]) + padding, max_size[dim])
            if max_size is not None
            else np.max(coords[:, dim]) + padding
        )

        bounding_box = *bounding_box, slice(min_value, max_value)

    return bounding_box


def bounding_box_inverse(bounding_box: tuple) -> np.ndarray:
    """
    Determine bounding box for a set of given coordinates.

    Args:
        tuple of slices: slices with ranges from min to max value
            per dimension.

    Returns:
        coords (np.ndarray): coordinate array of size N x dim.

    """
    coords = np.array(
        [
            [bounding_box[0].start, bounding_box[1].start],
            [bounding_box[0].stop, bounding_box[1].start],
            [bounding_box[0].stop, bounding_box[1].stop],
            [bounding_box[0].start, bounding_box[1].stop],
        ]
    )

    return coords
