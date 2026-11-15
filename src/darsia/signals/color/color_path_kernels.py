"""Compiled kernels for the evaluation of colors along color paths.

A color is assigned the closest point on a color path (piecewise linear in color space),
and the position on the path is converted to a value by piecewise linear interpolation.
The kernels do this in a single pass over the pixels, instead of several array passes
per segment as in :meth:`darsia.ColorPath.fit`. The arithmetic follows that method, so
results agree up to rounding in the last digit (the order of summation in the scalar
products differs).

The kernels require numba; without it `NUMBA_AVAILABLE` is False and callers use the
array-based implementation.

"""

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

__all__ = [
    "NUMBA_AVAILABLE",
    "NORM_THRESHOLD",
    "path_tables",
    "evaluate_colors",
    "evaluate_labeled_image",
]

NORM_THRESHOLD = 1e-1
"""Colors with a smaller norm are not fitted (no color change against the baseline)."""


def path_tables(
    supports: list, distances: list, values: list
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect color paths of different lengths in arrays of a common shape.

    Parameters
    ----------
    supports : list
        Key colors of each path, arrays of shape (segments + 1, 3).
    distances : list
        Positions of the key colors along each path, increasing.
    values : list
        Values at the key colors of each path.

    Returns
    -------
    tuple
        Key colors (paths, max. keys, 3), positions and values (paths, max. keys), and
        the number of segments of each path.

    """
    num_keys = max(len(d) for d in distances)
    num_paths = len(distances)
    support_table = np.zeros((num_paths, num_keys, 3), dtype=np.float64)
    distance_table = np.zeros((num_paths, num_keys), dtype=np.float64)
    value_table = np.zeros((num_paths, num_keys), dtype=np.float64)
    num_segments = np.zeros(num_paths, dtype=np.int64)
    for i in range(num_paths):
        n = len(distances[i])
        support_table[i, :n] = np.asarray(supports[i], dtype=np.float64)
        distance_table[i, :n] = np.asarray(distances[i], dtype=np.float64)
        value_table[i, :n] = np.asarray(values[i], dtype=np.float64)
        num_segments[i] = n - 1
    return support_table, distance_table, value_table, num_segments


if NUMBA_AVAILABLE:

    @njit(cache=True, error_model="numpy", inline="always")
    def _evaluate(c0, c1, c2, sup, dist, val, n, path, use_mask, threshold):
        """Value of one color for the path with index `path`."""
        # Position on the path; colors without a color change stay at 0.
        p = 0.0
        if (not use_mask) or np.sqrt(c0 * c0 + c1 * c1 + c2 * c2) > threshold:
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
    def _evaluate_colors(colors, sup, dist, val, n, use_mask, threshold, out):
        for i in prange(colors.shape[0]):
            out[i] = _evaluate(
                colors[i, 0],
                colors[i, 1],
                colors[i, 2],
                sup,
                dist,
                val,
                n,
                0,
                use_mask,
                threshold,
            )

    @njit(cache=True, error_model="numpy", parallel=True)
    def _evaluate_labeled_image(
        image,
        labels,
        offset,
        lookup,
        sup,
        dist,
        val,
        num_segments,
        use_mask,
        threshold,
        out,
    ):
        rows, cols = labels.shape
        for i in prange(rows):
            for j in range(cols):
                key = labels[i, j] - offset
                path = -1
                if key >= 0 and key < lookup.shape[0]:
                    path = lookup[key]
                if path < 0:
                    out[i, j] = 0.0
                else:
                    out[i, j] = _evaluate(
                        image[i, j, 0],
                        image[i, j, 1],
                        image[i, j, 2],
                        sup,
                        dist,
                        val,
                        num_segments[path],
                        path,
                        use_mask[path],
                        threshold,
                    )


def evaluate_colors(
    colors: np.ndarray,
    supports: np.ndarray,
    distances: np.ndarray,
    values: np.ndarray,
    use_mask: bool,
    threshold: float = NORM_THRESHOLD,
) -> np.ndarray:
    """Values of colors along one color path.

    Parameters
    ----------
    colors : np.ndarray
        Colors, shape (N, 3), in the color mode of the supports.
    supports : np.ndarray
        Key colors of the path, shape (segments + 1, 3).
    distances : np.ndarray
        Positions of the key colors along the path.
    values : np.ndarray
        Values at the key colors.
    use_mask : bool
        Whether colors with a norm below `threshold` are not fitted (position 0).
    threshold : float
        Threshold on the norm of the colors.

    Returns
    -------
    np.ndarray
        Values, shape (N,).

    """
    sup, dist, val, num_segments = path_tables([supports], [distances], [values])
    out = np.empty(colors.shape[0], dtype=np.float64)
    _evaluate_colors(
        np.ascontiguousarray(colors, dtype=np.float64),
        sup,
        dist,
        val,
        num_segments[0],
        use_mask,
        threshold,
        out,
    )
    return out


def evaluate_labeled_image(
    image: np.ndarray,
    labels: np.ndarray,
    path_labels: list[int],
    supports: list,
    distances: list,
    values: list,
    use_mask: list[bool],
    threshold: float = NORM_THRESHOLD,
) -> np.ndarray:
    """Values of an image of colors, with one color path per label.

    Parameters
    ----------
    image : np.ndarray
        Colors, shape (rows, cols, 3).
    labels : np.ndarray
        Integer labels, shape (rows, cols).
    path_labels : list[int]
        Labels with a color path; pixels of other labels get the value 0.
    supports, distances, values : list
        Key colors, their positions and values for each label in `path_labels`.
    use_mask : list[bool]
        For each label, whether colors with a norm below `threshold` are not fitted.
    threshold : float
        Threshold on the norm of the colors.

    Returns
    -------
    np.ndarray
        Values, shape (rows, cols).

    """
    sup, dist, val, num_segments = path_tables(supports, distances, values)
    offset = int(min(path_labels))
    lookup = np.full(int(max(path_labels)) - offset + 1, -1, dtype=np.int64)
    for index, label in enumerate(path_labels):
        lookup[int(label) - offset] = index
    out = np.empty(labels.shape, dtype=np.float64)
    _evaluate_labeled_image(
        np.ascontiguousarray(image, dtype=np.float64),
        np.ascontiguousarray(labels),
        offset,
        lookup,
        sup,
        dist,
        val,
        num_segments,
        np.asarray(use_mask, dtype=np.bool_),
        threshold,
        out,
    )
    return out
