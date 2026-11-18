"""Compiled kernels for the color range and the color spectra of calibration images.

Both are computed from relative colors (image minus baseline) at full resolution. The
array-based code builds a full difference image per image and copies of the pixels of
each label; the kernels go once over the pixels instead. They give the same numbers:

- `relative_color_range`: per channel the smallest and largest relative color in a
  mask (a NaN gives NaN, as `np.min` and `np.max`);
- `label_color_histograms`: per label the counts of the relative colors on a regular
  grid in color space, with the binning of `np.histogramdd` (`searchsorted` with
  ``side="right"`` on the same edges, the right edge counted in the last bin, colors
  outside the edges dropped); pixels outside the mask count as color 0.

The difference is taken in the type of the images, so both must have the same float
type (checked by `exact_for`). The kernels require numba; without it
`NUMBA_AVAILABLE` is False and callers use the array-based implementation.

"""

import os

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

__all__ = [
    "NUMBA_AVAILABLE",
    "exact_for",
    "histogram_edges",
    "relative_color_range",
    "label_indices",
    "label_color_histograms",
]

FLOAT_TYPES = (np.float32, np.float64)
"""Image types for which the kernels compute the difference as numpy does."""


def exact_for(image: np.ndarray, baseline: np.ndarray, mask: np.ndarray) -> bool:
    """Whether the kernels give exactly the result of the array-based code.

    Parameters
    ----------
    image, baseline : np.ndarray
        Color images (rows, cols, 3).
    mask : np.ndarray
        Boolean mask (rows, cols).

    Returns
    -------
    bool
        True if numba is available, both images have the same float type and shape
        with three channels, and the mask matches them.

    """
    return (
        NUMBA_AVAILABLE
        and image.dtype == baseline.dtype
        and image.dtype.type in FLOAT_TYPES
        and image.ndim == 3
        and image.shape[2] == 3
        and image.shape == baseline.shape
        and mask.dtype == np.bool_
        and mask.shape == image.shape[:2]
    )


def histogram_edges(bins: tuple[int, ...], color_range) -> list[np.ndarray]:
    """Bin edges per channel, as `np.histogramdd` builds them for `bins` and `range`.

    Parameters
    ----------
    bins : tuple[int, ...]
        Number of bins per channel.
    color_range : sequence
        (smallest, largest) value per channel.

    Returns
    -------
    list[np.ndarray]
        Edges per channel (float64).

    """
    _, edges = np.histogramdd(np.zeros((0, len(bins))), bins=bins, range=color_range)
    return [np.ascontiguousarray(e, dtype=np.float64) for e in edges]


def _chunks(rows: int) -> int:
    return max(1, min(rows, 2 * (os.cpu_count() or 1)))


if NUMBA_AVAILABLE:

    @njit(cache=True, parallel=True, nogil=True)
    def _range_kernel(image, baseline, mask, low, high, nan):  # pragma: no cover
        rows, cols = mask.shape
        for i in prange(rows):
            for j in range(cols):
                if not mask[i, j]:
                    continue
                for c in range(3):
                    value = image[i, j, c] - baseline[i, j, c]
                    if value != value:
                        nan[i, c] = True
                    else:
                        if value < low[i, c]:
                            low[i, c] = value
                        if value > high[i, c]:
                            high[i, c] = value

    @njit(cache=True, nogil=True)
    def _bin(value, edges):  # pragma: no cover
        """Core bin of a value (as `np.histogramdd`), or -1 if it is dropped."""
        n = edges.shape[0]  # bins + 1
        if value != value:
            return -1
        # Number of edges <= value (searchsorted, side="right")
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if edges[mid] <= value:
                lo = mid + 1
            else:
                hi = mid
        if value == edges[n - 1]:
            lo -= 1
        if lo < 1 or lo > n - 1:
            return -1
        return lo - 1

    @njit(cache=True, parallel=True, nogil=True)
    def _histogram_kernel(
        image, baseline, mask, label_idx, edges_r, edges_g, edges_b, counts
    ):  # pragma: no cover
        rows, cols = mask.shape
        chunks = counts.shape[0]
        for chunk in prange(chunks):
            start = chunk * rows // chunks
            stop = (chunk + 1) * rows // chunks
            for i in range(start, stop):
                for j in range(cols):
                    # Difference in the type of the images, then exactly widened
                    if mask[i, j]:
                        r = np.float64(image[i, j, 0] - baseline[i, j, 0])
                        g = np.float64(image[i, j, 1] - baseline[i, j, 1])
                        b = np.float64(image[i, j, 2] - baseline[i, j, 2])
                    else:
                        r = 0.0
                        g = 0.0
                        b = 0.0
                    br = _bin(r, edges_r)
                    if br < 0:
                        continue
                    bg = _bin(g, edges_g)
                    if bg < 0:
                        continue
                    bb = _bin(b, edges_b)
                    if bb < 0:
                        continue
                    counts[chunk, label_idx[i, j], br, bg, bb] += 1


def relative_color_range(
    image: np.ndarray, baseline: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Smallest and largest relative color (image - baseline) per channel in a mask.

    Parameters
    ----------
    image, baseline : np.ndarray
        Color images of the same float type (see `exact_for`).
    mask : np.ndarray
        Boolean mask with at least one pixel.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Smallest and largest values per channel (type of the images).

    """
    rows = mask.shape[0]
    dtype = image.dtype
    low = np.full((rows, 3), np.inf, dtype=dtype)
    high = np.full((rows, 3), -np.inf, dtype=dtype)
    nan = np.zeros((rows, 3), dtype=np.bool_)
    _range_kernel(image, baseline, mask, low, high, nan)
    smallest, largest = low.min(axis=0), high.max(axis=0)
    has_nan = nan.any(axis=0)
    smallest[has_nan] = np.nan
    largest[has_nan] = np.nan
    return smallest, largest


def label_indices(labels: np.ndarray, label_values: np.ndarray) -> np.ndarray:
    """Index of each pixel's label in `label_values`.

    Parameters
    ----------
    labels : np.ndarray
        Label of each pixel (rows, cols).
    label_values : np.ndarray
        Sorted labels containing all values of `labels` (e.g. `np.unique(labels)`).

    Returns
    -------
    np.ndarray
        Indices (rows, cols).

    """
    return np.searchsorted(label_values, labels).astype(np.intp)


def label_color_histograms(
    image: np.ndarray,
    baseline: np.ndarray,
    mask: np.ndarray,
    label_idx: np.ndarray,
    num_labels: int,
    edges: list[np.ndarray],
) -> np.ndarray:
    """Counts of the relative colors per label on a grid in color space.

    Parameters
    ----------
    image, baseline : np.ndarray
        Color images of the same float type (see `exact_for`).
    mask : np.ndarray
        Boolean mask; pixels outside count as relative color 0.
    label_idx : np.ndarray
        Index of each pixel's label (`label_indices`).
    num_labels : int
        Number of labels.
    edges : list[np.ndarray]
        Bin edges per channel (`histogram_edges`).

    Returns
    -------
    np.ndarray
        Counts (labels, bins R, bins G, bins B), as `np.histogramdd` of the relative
        colors of each label (int64).

    """
    shape = tuple(len(e) - 1 for e in edges)
    chunks = _chunks(mask.shape[0])
    counts = np.zeros((chunks, num_labels, *shape), dtype=np.int64)
    _histogram_kernel(
        image, baseline, mask, label_idx, edges[0], edges[1], edges[2], counts
    )
    return counts.sum(axis=0)
