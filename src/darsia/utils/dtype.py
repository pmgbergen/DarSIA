"""Conversion utilities for data types."""

from warnings import warn

import numpy as np
import skimage


def convert_dtype(img: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Convert image to the specified dtype.

    Parameters
    ----------
    img : np.ndarray
        Image.
    dtype : np.dtype
        Dtype to convert to.

    Returns
    -------
    np.ndarray
        Converted image.
    """
    if dtype == np.uint8:
        return skimage.img_as_ubyte(img)  # type: ignore[attr-defined]
    elif dtype == np.uint16:
        return skimage.img_as_uint(img)  # type: ignore[attr-defined]
    elif dtype == np.float32:
        return skimage.img_as_float32(img)  # type: ignore[attr-defined]
    elif dtype == np.float64:
        return skimage.img_as_float64(img)  # type: ignore[attr-defined]
    else:
        warn("{dtype} is not a supported dtype. Returning {img.dtype} image.")
        return img
