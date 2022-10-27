from typing import Optional, Union

import numpy as np


def bounding_box(
    coords: np.ndarray, padding: int = 0, max_size: Optional[list] = None
) -> tuple:
    """
    Determine bounding box for a set of given coordinates.

    Args:
        coords (np.ndarray): coordinate array of size N x dim, using matrix indexing in 2d.
        padding (int): padding to create a slightly larger bounding box. Might
            be of interest if the area that is prescribed in coords cover slightly less than
            strictly needed. Default is 0.
        max_size (Optional[list]): max size of bounding box.
            Provided as list with max size in each dimension.

    Returns:
        tuple of slices: slices with ranges from min to max value
            per dimension.
    """
    bounding_box = ()

    for dim in range(coords.shape[1]):
        # Padding is added to the bounding box while making sure that it never extends out
        # of the scope of the 0 and max_size (where max size should be externally provided)
        # and should maximally be the maximal size of the image in each dimension.
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
    Returns an array that would produce the same bounding box from the bounding_box()
    function above.

    Args:
        tuple of slices: slices with ranges from min to max value
            per dimension.

    Returns:
        coords (np.ndarray): coordinate array of size N x dim, using matrix indexing in 2d.

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


def perimeter(box: Union[tuple, np.ndarray]) -> Union[int, float]:
    """
    Returns the perimeter of a box. Accepts both tuples of slices
    as well as arrays of coordinates as input.

    Args:
        box (tuple of slices or np.ndarray): definition of a box,
            either as tuple of slices, or coordinates (can also
            use metric units)

    Returns:
        float or int (depending on input): perimeter
    """
    if isinstance(box, np.ndarray):
        min_x = np.min(box[:, 0])
        max_x = np.max(box[:, 0])
        min_y = np.min(box[:, 1])
        max_y = np.max(box[:, 1])
        perimeter = 2 * (max_x - min_x) + 2 * (max_y - min_y)

    elif isinstance(box, tuple):
        raise NotImplementedError("This is currently not supported.")

    return perimeter
