from __future__ import annotations
import numpy as np
import daria as da
from typing import Union


def cv2ToSkimage(img: np.ndarray) -> np.ndarray:
    """Conversion between opencv (cv2) which has BGR-formatting and scikit-image (skimage)
    which has RGB. The same command works in both directions.

    Arguments:
        img (np.ndarray): input image

    Returns:
        np.ndarray: converted image.
    """
    return img[:, :, ::-1]


def standardToPhysicalPixelOrdering(img: np.ndarray) -> np.ndarray:
    """Reordering of pixels finally using a physical pixel ordering, starting from a
    standard one.

    The standard ordering of pixels for images is that the top left
    corner is the (0,0) pixel. With increasing x- and y-pixels, one
    moves towards the lower right corner. In addition, the first and
    second components correspond to the y and x pixels, respectively.
    This is inconsistent with the physical coordinate system,
    considering a standard photograph.

    Therefore, we introduce the term 'physical ordering' which identifies
    the lower left corner as (0,0) pixel. And with increasing x-
    and y-pixels, one moves towards the top right corner. In addition,
    first and second components correspond to x and y.

    Arguments:
        np.ndarray: image array with standard pixel ordering

    Returns:
        np.ndarray: image array with physical pixel ordering
    """
    if len(img.shape) > 3:
        raise NotImplementedError("Pixel conversion only implemented for 2d images.")

    # Reorder the y-pixels, corresponding to the first component
    img = np.flip(img, 0)

    # Flip first and second component
    img = np.swapaxes(img, 0, 1)

    return img


def physicalToStandardPixelOrdering(img: np.ndarray) -> np.ndarray:
    """Reordering of pixels finally using a standard pixel ordering, starting from a
    physical one.

    Inverse to standardToPhysicalPixelOrdering. It is the same operations, but in
    reverse order.

    Arguments:
        np.ndarray: image array with physical pixel ordering

    Returns:
        np.ndarray: image array with physical pixel ordering
    """
    if len(img.shape) > 3:
        raise NotImplementedError("Pixel conversion only implemented for 2d images.")

    # Flip first and second component
    img = np.swapaxes(img, 0, 1)

    # Reorder the y-pixels, corresponding to the first component
    img = np.flip(img, 0)

    return img


def standardToPhysicalPixel(
    img: "Image", pixel: Union[tuple[int], list[int]]
) -> tuple[int]:
    """Translate a standard to a physical pixel.

    The conversion is based on the width and height of the corresponding image.

    Arguments:
        img (Image): image object
        pixel (list of int): standard pixel

    Returns:
        list of int: physical pixel
    """
    # TODO
    pass


def physicalToStandardPixel(
    img: "Image", pixel: Union[tuple[int], Union[tuple[int], list[int]]]
) -> tuple[int]:
    """Translate a physica pixel to a standard pixel.

    Inversie of standardToPhysicalPixel.

    Arguments:
        img (Image): image object
        pixel (list of int): physical pixel

    Returns:
        list of int: standard pixel
    """
    return img.shape[1] - pixel[1], pixel[0]
