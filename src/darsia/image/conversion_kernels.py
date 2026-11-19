"""Compiled kernels converting images between unsigned integers and floats.

`skimage.img_as_float` converts an unsigned integer image by one multiplication with
``1 / max`` (``np.multiply(image, 1.0 / imax, dtype=np.float64)``) on one core. For
images of tens of megapixels this is a notable part of reading an image. The kernel
does the same multiplication on all cores; the values are identical.

The other direction, `to_ubyte`, clips float images to [0, 1] and converts them to
8-bit as ``skimage.img_as_ubyte(np.clip(image, 0, 1))``: multiplication with 255 in the
precision of the image, rounding half to even, conversion; NaN gives 0 (as numpy's
conversion). In one pass, without the temporary arrays of the image size.

The kernels require numba; without it `NUMBA_AVAILABLE` is False and `to_float64` and
`to_ubyte` return None (callers use skimage).

"""

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

__all__ = ["NUMBA_AVAILABLE", "to_float64", "to_ubyte"]

UNSIGNED_TYPES = (np.uint8, np.uint16)
"""Integer types converted by the kernel."""

FLOAT_TYPES = (np.float32, np.float64)
"""Float types converted to 8-bit by the kernel."""

if NUMBA_AVAILABLE:

    @njit(cache=True, parallel=True, nogil=True)
    def _scale_kernel(values, scale, out):  # pragma: no cover
        for i in prange(values.shape[0]):
            out[i] = values[i] * scale

    @njit(cache=True, parallel=True, nogil=True)
    def _ubyte_kernel(values, scale, out):  # pragma: no cover
        # Bounds in the type of the values, so that the arithmetic stays in it
        zero = scale - scale
        one = scale / scale
        for i in prange(values.shape[0]):
            x = values[i]
            if x != x:
                out[i] = 0
                continue
            if x < zero:
                x = zero
            if x > one:
                x = one
            out[i] = np.uint8(np.rint(x * scale))


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


def to_ubyte(array: np.ndarray) -> np.ndarray | None:
    """Float array clipped to [0, 1] as 8-bit, as ``skimage.img_as_ubyte(np.clip(...))``.

    Parameters
    ----------
    array : np.ndarray
        Array of float32 or float64.

    Returns
    -------
    np.ndarray | None
        uint8 array of the same shape (C order), or None if the kernel does not apply
        (other type, or numba not available).

    """
    if not NUMBA_AVAILABLE or array.dtype.type not in FLOAT_TYPES:
        return None
    values = np.ascontiguousarray(array).reshape(-1)
    out = np.empty(values.shape, dtype=np.uint8)
    _ubyte_kernel(values, array.dtype.type(255), out)
    return out.reshape(array.shape)
