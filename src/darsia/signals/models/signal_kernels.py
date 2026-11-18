"""Compiled kernel for piecewise linear signal functions on labelled images.

Each label has a piecewise linear function through values at supports
(:class:`darsia.PWTransformation`). The kernel clips the argument to a common domain and
evaluates the function of the label of each pixel, in a single parallel pass. It
reproduces `np.clip` followed by `scipy.interpolate.interp1d(kind="linear")` (which
evaluates like `np.interp`) for arguments within the supports; change both together.

The kernel requires numba; without it `NUMBA_AVAILABLE` is False and callers use the
array-based implementation.

"""

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

__all__ = ["NUMBA_AVAILABLE", "evaluate_labeled_image"]


if NUMBA_AVAILABLE:

    @njit(cache=True, error_model="numpy", parallel=True, nogil=True)
    def _evaluate_labeled_image(
        signal, labels, label_offset, lookup, xp, fp, num, low, high, out
    ):
        rows, cols = labels.shape
        for i in prange(rows):
            for j in range(cols):
                key = labels[i, j] - label_offset
                f = -1
                if key >= 0 and key < lookup.shape[0]:
                    f = lookup[key]
                if f < 0:
                    out[i, j] = 0.0
                    continue

                # Clip to the common domain of the functions (as np.clip)
                x = signal[i, j]
                if x < low:
                    x = low
                if x > high:
                    x = high

                # Piecewise linear interpolation (as np.interp)
                n = num[f]
                if np.isnan(x):
                    out[i, j] = np.nan
                elif x > xp[f, n - 1]:
                    out[i, j] = fp[f, n - 1]
                elif x < xp[f, 0]:
                    out[i, j] = fp[f, 0]
                else:
                    k = 0
                    while k < n - 1 and xp[f, k + 1] <= x:
                        k += 1
                    if k == n - 1 or xp[f, k] == x:
                        out[i, j] = fp[f, k]
                    else:
                        slope = (fp[f, k + 1] - fp[f, k]) / (xp[f, k + 1] - xp[f, k])
                        value = slope * (x - xp[f, k]) + fp[f, k]
                        if np.isnan(value):
                            value = slope * (x - xp[f, k + 1]) + fp[f, k + 1]
                            if np.isnan(value) and fp[f, k] == fp[f, k + 1]:
                                value = fp[f, k]
                        out[i, j] = value


def evaluate_labeled_image(
    signal: np.ndarray,
    labels: np.ndarray,
    functions: dict[int, tuple],
    low: float,
    high: float,
) -> np.ndarray:
    """Evaluate the piecewise linear function of each pixel's label.

    Parameters
    ----------
    signal : np.ndarray
        Arguments, shape (rows, cols).
    labels : np.ndarray
        Integer labels, shape (rows, cols).
    functions : dict[int, tuple]
        Supports (strictly increasing) and values of the function of each label. Pixels
        of other labels get the value 0.
    low, high : float
        Bounds to which the arguments are clipped first.

    Returns
    -------
    np.ndarray
        Values, shape (rows, cols).

    """
    function_labels = [int(label) for label in functions]
    label_offset = min(function_labels)
    lookup = np.full(max(function_labels) - label_offset + 1, -1, dtype=np.int64)
    num_keys = max(len(supports) for supports, _ in functions.values())
    xp = np.zeros((len(functions), num_keys), dtype=np.float64)
    fp = np.zeros((len(functions), num_keys), dtype=np.float64)
    num = np.zeros(len(functions), dtype=np.int64)
    for index, (label, (supports, values)) in enumerate(functions.items()):
        lookup[int(label) - label_offset] = index
        num[index] = len(supports)
        xp[index, : len(supports)] = np.asarray(supports, dtype=np.float64)
        fp[index, : len(supports)] = np.asarray(values, dtype=np.float64)
    out = np.empty(labels.shape, dtype=np.float64)
    _evaluate_labeled_image(
        np.ascontiguousarray(signal, dtype=np.float64),
        np.ascontiguousarray(labels),
        label_offset,
        lookup,
        xp,
        fp,
        num,
        float(low),
        float(high),
        out,
    )
    return out
