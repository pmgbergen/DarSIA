"""Compiled kernels for the evaluation of colors along color paths.

A color is assigned the closest point on a color path (piecewise linear in color space),
and the position on the path is converted to a value by piecewise linear interpolation.
The kernels do this in a single pass over the pixels, instead of several array passes
per segment as in :meth:`darsia.ColorPath.fit`. The arithmetic follows that method, so
results agree up to rounding in the last digit (the order of summation in the scalar
products differs).

Colors without a signal are not fitted and keep the position 0 on the path. They are
identified either by the norm of the color (`MASK_NORM`) or by the membership in a
discrete color spectrum, e.g. the colors of the baseline (`MASK_SPECTRUM`).

The kernels require numba; without it `NUMBA_AVAILABLE` is False and callers use the
array-based implementation.

"""

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

__all__ = [
    "NUMBA_AVAILABLE",
    "NORM_THRESHOLD",
    "MASK_NONE",
    "MASK_NORM",
    "MASK_SPECTRUM",
    "PathData",
    "evaluate_colors",
    "evaluate_labeled_image",
]

NORM_THRESHOLD = 1e-1
"""Colors with a smaller norm are not fitted (no color change against the baseline)."""

IMAGE_DTYPES = (np.uint8, np.float32, np.float64)
"""Types of images which `evaluate_labeled_image` reads without converting the image."""

MASK_NONE = 0
"""All colors are fitted."""
MASK_NORM = 1
"""Colors with a norm up to `NORM_THRESHOLD` are not fitted."""
MASK_SPECTRUM = 2
"""Colors in the bins of a discrete color spectrum are not fitted."""


@dataclass
class PathData:
    """Data of one color path and its values for the kernels."""

    supports: np.ndarray
    """Key colors of the path, shape (segments + 1, 3)."""
    distances: np.ndarray
    """Positions of the key colors along the path, increasing."""
    values: np.ndarray
    """Values at the key colors."""
    mask: int = MASK_NONE
    """Which colors are not fitted: `MASK_NONE`, `MASK_NORM` or `MASK_SPECTRUM`."""
    spectrum: np.ndarray | None = None
    """Boolean bins of the spectrum (for `MASK_SPECTRUM`)."""
    edges: list | None = None
    """Bin edges of the spectrum for each color channel."""
    offset: np.ndarray | None = None
    """Shift from the colors to the frame of the spectrum."""


def _tables(paths: list[PathData]) -> tuple:
    """Collect paths of different lengths (and spectra) in arrays of a common shape."""
    num_paths = len(paths)
    num_keys = max(len(path.distances) for path in paths)
    sup = np.zeros((num_paths, num_keys, 3), dtype=np.float64)
    dist = np.zeros((num_paths, num_keys), dtype=np.float64)
    val = np.zeros((num_paths, num_keys), dtype=np.float64)
    num_segments = np.zeros(num_paths, dtype=np.int64)
    mask = np.zeros(num_paths, dtype=np.int64)

    shapes = [
        path.spectrum.shape if path.mask == MASK_SPECTRUM else (1, 1, 1)
        for path in paths
    ]
    max_shape = tuple(max(shape[i] for shape in shapes) for i in range(3))
    spectrum = np.zeros((num_paths, *max_shape), dtype=np.bool_)
    edges = np.zeros((num_paths, 3, max(max_shape) + 1), dtype=np.float64)
    bins = np.ones((num_paths, 3), dtype=np.int64)
    offset = np.zeros((num_paths, 3), dtype=np.float64)

    for i, path in enumerate(paths):
        n = len(path.distances)
        sup[i, :n] = np.asarray(path.supports, dtype=np.float64)
        dist[i, :n] = np.asarray(path.distances, dtype=np.float64)
        val[i, :n] = np.asarray(path.values, dtype=np.float64)
        num_segments[i] = n - 1
        mask[i] = path.mask
        if path.mask == MASK_SPECTRUM:
            shape = path.spectrum.shape
            spectrum[i, : shape[0], : shape[1], : shape[2]] = path.spectrum
            bins[i] = shape
            for axis in range(3):
                edges[i, axis, : shape[axis] + 1] = path.edges[axis]
            offset[i] = path.offset
    return sup, dist, val, num_segments, mask, spectrum, edges, bins, offset


if NUMBA_AVAILABLE:

    @njit(cache=True, error_model="numpy", inline="always")
    def _bin(x, edges, path, axis, n):
        """Bin of x as in np.histogramdd (-1 outside of the range)."""
        low = edges[path, axis, 0]
        high = edges[path, axis, n]
        if not (x >= low and x <= high and high > low):
            return -1
        k = int((x - low) / (high - low) * n)
        if k > n - 1:
            k = n - 1
        while k > 0 and x < edges[path, axis, k]:
            k -= 1
        while k < n - 1 and x >= edges[path, axis, k + 1]:
            k += 1
        return k

    @njit(cache=True, error_model="numpy", inline="always")
    def _fitted(c0, c1, c2, path, mask, threshold, spectrum, edges, bins, offset):
        """Whether a color carries a signal and is fitted to the path."""
        kind = mask[path]
        if kind == 0:
            return True
        if kind == 1:
            return np.sqrt(c0 * c0 + c1 * c1 + c2 * c2) > threshold
        k0 = _bin(c0 + offset[path, 0], edges, path, 0, bins[path, 0])
        if k0 < 0:
            return True
        k1 = _bin(c1 + offset[path, 1], edges, path, 1, bins[path, 1])
        if k1 < 0:
            return True
        k2 = _bin(c2 + offset[path, 2], edges, path, 2, bins[path, 2])
        if k2 < 0:
            return True
        return not spectrum[path, k0, k1, k2]

    @njit(cache=True, error_model="numpy", inline="always")
    def _evaluate(c0, c1, c2, sup, dist, val, n, path, fitted):
        """Value of one color for the path with index `path`."""
        # Position on the path; colors which are not fitted stay at 0.
        p = 0.0
        if fitted:
            best = np.inf
            for s in range(n):
                a0 = sup[path, s, 0]
                a1 = sup[path, s, 1]
                a2 = sup[path, s, 2]
                d0 = sup[path, s + 1, 0] - a0
                d1 = sup[path, s + 1, 1] - a1
                d2 = sup[path, s + 1, 2] - a2
                left = dist[path, s]
                right = dist[path, s + 1]
                length = right - left

                # Orthogonal projection onto the segment; the first and last segments
                # extend beyond the path.
                q = left + length * (
                    (c0 - a0) * d0 + (c1 - a1) * d1 + (c2 - a2) * d2
                ) / (d0 * d0 + d1 * d1 + d2 * d2)
                if s == 0:
                    if q > right:
                        q = right
                elif s == n - 1:
                    if q < left:
                        q = left
                else:
                    if q < left:
                        q = left
                    if q > right:
                        q = right

                # Distance (1-norm) between the color and its projection
                w = (q - left) / length
                e = (
                    abs(c0 - (a0 + w * d0))
                    + abs(c1 - (a1 + w * d1))
                    + abs(c2 - (a2 + w * d2))
                )
                if e < best:
                    best = e
                    p = q
            if np.isnan(p):
                p = 0.0

        # Piecewise linear interpolation of the values
        result = 0.0
        for s in range(n):
            left = dist[path, s]
            right = dist[path, s + 1]
            if s == 0:
                inside = p <= right
            elif s == n - 1:
                inside = p >= left
            else:
                inside = (p >= left) and (p <= right)
            if inside:
                result = val[path, s] + ((p - left) / (right - left)) * (
                    val[path, s + 1] - val[path, s]
                )
        return result

    @njit(cache=True, error_model="numpy", parallel=True)
    def _evaluate_colors(
        colors,
        sup,
        dist,
        val,
        num_segments,
        mask,
        spectrum,
        edges,
        bins,
        offset,
        threshold,
        out,
    ):
        for i in prange(colors.shape[0]):
            c0 = colors[i, 0]
            c1 = colors[i, 1]
            c2 = colors[i, 2]
            fitted = _fitted(
                c0, c1, c2, 0, mask, threshold, spectrum, edges, bins, offset
            )
            out[i] = _evaluate(c0, c1, c2, sup, dist, val, num_segments[0], 0, fitted)

    @njit(cache=True, error_model="numpy", parallel=True)
    def _evaluate_labeled_image(
        image,
        labels,
        label_offset,
        lookup,
        sup,
        dist,
        val,
        num_segments,
        mask,
        spectrum,
        edges,
        bins,
        offset,
        threshold,
        scale,
        out,
    ):
        rows, cols = labels.shape
        for i in prange(rows):
            for j in range(cols):
                key = labels[i, j] - label_offset
                path = -1
                if key >= 0 and key < lookup.shape[0]:
                    path = lookup[key]
                if path < 0:
                    out[i, j] = 0.0
                else:
                    c0 = image[i, j, 0] * scale
                    c1 = image[i, j, 1] * scale
                    c2 = image[i, j, 2] * scale
                    fitted = _fitted(
                        c0, c1, c2, path, mask, threshold, spectrum, edges, bins, offset
                    )
                    out[i, j] = _evaluate(
                        c0, c1, c2, sup, dist, val, num_segments[path], path, fitted
                    )


if NUMBA_AVAILABLE:

    @njit(cache=True, error_model="numpy", parallel=True)
    def _evaluate_labeled_difference(
        image,
        base,
        labels,
        label_offset,
        lookup,
        sup,
        dist,
        val,
        num_segments,
        mask,
        spectrum,
        edges,
        bins,
        offset,
        threshold,
        scale,
        out,
    ):
        """As `_evaluate_labeled_image`, for the colors `image * scale - base`."""
        rows, cols = labels.shape
        for i in prange(rows):
            for j in range(cols):
                key = labels[i, j] - label_offset
                path = -1
                if key >= 0 and key < lookup.shape[0]:
                    path = lookup[key]
                if path < 0:
                    out[i, j] = 0.0
                else:
                    c0 = image[i, j, 0] * scale - base[i, j, 0]
                    c1 = image[i, j, 1] * scale - base[i, j, 1]
                    c2 = image[i, j, 2] * scale - base[i, j, 2]
                    fitted = _fitted(
                        c0, c1, c2, path, mask, threshold, spectrum, edges, bins, offset
                    )
                    out[i, j] = _evaluate(
                        c0, c1, c2, sup, dist, val, num_segments[path], path, fitted
                    )


def evaluate_colors(
    colors: np.ndarray, path: PathData, threshold: float = NORM_THRESHOLD
) -> np.ndarray:
    """Values of colors along one color path.

    Parameters
    ----------
    colors : np.ndarray
        Colors, shape (N, 3), in the color mode of the key colors of the path.
    path : PathData
        Color path, values and the colors which are not fitted.
    threshold : float
        Threshold on the norm of the colors (for `MASK_NORM`).

    Returns
    -------
    np.ndarray
        Values, shape (N,).

    """
    out = np.empty(colors.shape[0], dtype=np.float64)
    _evaluate_colors(
        np.ascontiguousarray(colors, dtype=np.float64),
        *_tables([path]),
        threshold,
        out,
    )
    return out


def evaluate_labeled_image(
    image: np.ndarray,
    labels: np.ndarray,
    paths: dict[int, PathData],
    threshold: float = NORM_THRESHOLD,
    base: np.ndarray | None = None,
) -> np.ndarray:
    """Values of an image of colors, with one color path per label.

    Parameters
    ----------
    image : np.ndarray
        Colors, shape (rows, cols, 3). Images of the types in `IMAGE_DTYPES` are
        converted to float64 pixel by pixel, exactly like a conversion of the image
        beforehand (8-bit values are multiplied by 1/255, as `skimage.img_as_float`);
        other types are converted as a whole.
    labels : np.ndarray
        Integer labels, shape (rows, cols).
    paths : dict[int, PathData]
        Color path, values and the colors which are not fitted, for each label. Pixels
        of other labels get the value 0.
    threshold : float
        Threshold on the norm of the colors (for `MASK_NORM`).
    base : np.ndarray, optional
        Baseline of the shape of the image; if given, the colors `image - base` are
        evaluated (without forming the difference image).

    Returns
    -------
    np.ndarray
        Values, shape (rows, cols).

    """
    path_labels = [int(label) for label in paths]
    label_offset = min(path_labels)
    lookup = np.full(max(path_labels) - label_offset + 1, -1, dtype=np.int64)
    for index, label in enumerate(path_labels):
        lookup[label - label_offset] = index
    out = np.empty(labels.shape, dtype=np.float64)
    if image.dtype.type in IMAGE_DTYPES:
        image = np.ascontiguousarray(image)
    else:
        image = np.ascontiguousarray(image, dtype=np.float64)
    # Multiplying a float by 1 is exact, so one kernel serves all types.
    scale = 1.0 / 255.0 if image.dtype == np.uint8 else 1.0
    tables = _tables(list(paths.values()))
    if base is None:
        _evaluate_labeled_image(
            image,
            np.ascontiguousarray(labels),
            label_offset,
            lookup,
            *tables,
            threshold,
            scale,
            out,
        )
    else:
        _evaluate_labeled_difference(
            image,
            np.ascontiguousarray(base, dtype=np.float64),
            np.ascontiguousarray(labels),
            label_offset,
            lookup,
            *tables,
            threshold,
            scale,
            out,
        )
    return out
