"""Compiled kernel converting unsigned integer images to float64.

`skimage.img_as_float` converts an unsigned integer image by one multiplication with
``1 / max`` (``np.multiply(image, 1.0 / imax, dtype=np.float64)``) on one core. For
images of tens of megapixels this is a notable part of reading an image. The kernel
does the same multiplication on all cores; the values are identical.

The kernel requires numba; without it `NUMBA_AVAILABLE` is False and `to_float64`
returns None (callers use skimage).

"""

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

__all__ = ["NUMBA_AVAILABLE", "to_float64"]

UNSIGNED_TYPES = (np.uint8, np.uint16)
"""Integer types converted by the kernel."""

if NUMBA_AVAILABLE:

    @njit(cache=True, parallel=True, nogil=True)
    def _scale_kernel(values, scale, out):  # pragma: no cover
        for i in prange(values.shape[0]):
            out[i] = values[i] * scale


def to_float64(array: np.ndarray) -> np.ndarray | None:
    """Unsigned integer array as floats in [0, 1], as `skimage.img_as_float`.

    Parameters
    ----------
    array : np.ndarray
        Array of 8-bit or 16-bit unsigned integers.

    Returns
    -------
    np.ndarray | None
        float64 array of the same shape (C order), or None if the kernel does not
        apply (other type, or numba not available).

    """
    if not NUMBA_AVAILABLE or array.dtype.type not in UNSIGNED_TYPES:
        return None
    values = np.ascontiguousarray(array).reshape(-1)
    out = np.empty(values.shape, dtype=np.float64)
    _scale_kernel(values, 1.0 / np.iinfo(array.dtype).max, out)
    return out.reshape(array.shape)
