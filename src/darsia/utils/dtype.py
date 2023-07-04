"""Conversion utilities for data types."""
from warnings import warn

import numpy as np
import skimage


def convert_dtype(img: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Convert image to the specified dtype.

    Args:
        img (np.ndarray): image
        img_dtype (np.dtype): dtype to convert to

    Returns:
        np.ndarray: converted image

    """
    if dtype == np.uint8:
        return skimage.img_as_ubyte(img)
    elif dtype == np.uint16:
        return skimage.img_as_uint(img)
    elif dtype == np.float32:
        return skimage.img_as_float32(img)
    elif dtype == np.float64:
        return skimage.img_as_float64(img)
    else:
        warn("{dtype} is not a supported dtype. Returning {img.dtype} image.")
        return img
