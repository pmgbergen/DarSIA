"""Compiled kernels for resampling images at transformed coordinates.

Shape corrections provide, for each pixel of the corrected image, its position in the
original image. The kernels interpolate the original image bilinearly at these positions
in a single parallel pass. They reproduce `scipy.ndimage.map_coordinates` with `order=1`
and `mode="constant"` (zero outside of the image), including the rounding for integer
images.

The kernels require numba; without it `NUMBA_AVAILABLE` is False and callers use scipy.

"""

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

__all__ = ["NUMBA_AVAILABLE", "supports", "bilinear"]

UNSIGNED_DTYPES = (np.uint8, np.uint16)
"""Integer image types of the kernels (results are rounded as scipy does)."""

SUPPORTED_DTYPES = UNSIGNED_DTYPES + (np.float32, np.float64)
"""Image types for which the kernels reproduce scipy."""


if NUMBA_AVAILABLE:

    @njit(cache=True, error_model="numpy", inline="always")
    def _interpolate(img, y, x, channel):
        """Bilinear interpolation of one channel at (y, x); zero outside the image.

        The four neighbors are summed in the order and with the weights of scipy
        (weight along the rows first), so that the result agrees to the last digit.
        """
        ny = img.shape[0]
        nx = img.shape[1]
        if not (y >= 0.0 and y <= ny - 1 and x >= 0.0 and x <= nx - 1):
            return 0.0
        y0 = int(np.floor(y))
        x0 = int(np.floor(x))
        wy = y - y0
        wx = x - x0
        value = 0.0
        value += img[y0, x0, channel] * (1.0 - wy) * (1.0 - wx)
        if x0 + 1 < nx:
            value += img[y0, x0 + 1, channel] * (1.0 - wy) * wx
        if y0 + 1 < ny:
            value += img[y0 + 1, x0, channel] * wy * (1.0 - wx)
            if x0 + 1 < nx:
                value += img[y0 + 1, x0 + 1, channel] * wy * wx
        return value

    @njit(cache=True, error_model="numpy", parallel=True, nogil=True)
    def _bilinear_unsigned(img, rows, cols, out):
        """For unsigned integer images of any width (compiled per type)."""
        for i in prange(out.shape[0]):
            y = np.float64(rows[i])
            x = np.float64(cols[i])
            for channel in range(img.shape[2]):
                # Rounding to the nearest integer, as scipy does for integer images.
                # The value lies within the range of the image (it is an average).
                out[i, channel] = int(np.floor(_interpolate(img, y, x, channel) + 0.5))

    @njit(cache=True, error_model="numpy", parallel=True, nogil=True)
    def _bilinear_float(img, rows, cols, out):
        for i in prange(out.shape[0]):
            y = np.float64(rows[i])
            x = np.float64(cols[i])
            for channel in range(img.shape[2]):
                out[i, channel] = _interpolate(img, y, x, channel)


def supports(img: np.ndarray, grid: np.ndarray, order: int) -> bool:
    """Whether `bilinear` applies to an image, a coordinate grid and an order.

    Parameters
    ----------
    img : np.ndarray
        Image array.
    grid : np.ndarray
        Positions (row, column) in the image, shape (2, N).
    order : int
        Interpolation order.

    Returns
    -------
    bool
        True for bilinear interpolation of uint8 and float images, if numba is
        available.

    """
    return (
        NUMBA_AVAILABLE
        and order == 1
        and img.dtype.type in SUPPORTED_DTYPES
        and img.ndim in (2, 3)
        and grid.ndim == 2
        and grid.shape[0] == 2
        and np.issubdtype(grid.dtype, np.floating)
    )


def bilinear(img: np.ndarray, grid: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Resample an image bilinearly at given positions.

    Parameters
    ----------
    img : np.ndarray
        Image array of shape (rows, cols) or (rows, cols, channels); see `supports`.
    grid : np.ndarray
        Positions (row, column) in the image for each pixel of the result (row-major),
        shape (2, N).
    shape : tuple
        Shape (rows, cols) of the result, with rows * cols = N.

    Returns
    -------
    np.ndarray
        Resampled image of shape (rows, cols, channels), in the type of the image;
        zero where the position lies outside of the image.

    """
    img = np.ascontiguousarray(np.atleast_3d(img))
    rows = np.ascontiguousarray(grid[0])
    cols = np.ascontiguousarray(grid[1])
    out = np.empty((rows.shape[0], img.shape[2]), dtype=img.dtype)
    if img.dtype.type in UNSIGNED_DTYPES:
        _bilinear_unsigned(img, rows, cols, out)
    else:
        _bilinear_float(img, rows, cols, out)
    return out.reshape((*shape, img.shape[2]))
