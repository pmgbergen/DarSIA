"""Highlight a region of an image: everything outside turns (mostly) gray."""

from __future__ import annotations

import numpy as np

LUMINANCE = (0.299, 0.587, 0.114)
"""Weights of the red, green and blue channel for a perceived gray."""


def highlight_region(
    image: np.ndarray, outside: np.ndarray, weights: tuple | None = None
) -> np.ndarray:
    """Image with the pixels outside a region blended towards gray (70 % gray).

    Parameters
    ----------
    image : np.ndarray
        Color image (rows, cols, channels), float.
    outside : np.ndarray
        Boolean mask (rows, cols) of the pixels to gray out.
    weights : tuple, optional
        Channel weights of the gray (e.g. `LUMINANCE`); default is the channel mean.

    Returns
    -------
    np.ndarray
        Highlighted copy of the image.

    """
    if weights is None:
        gray = np.mean(image, axis=-1, keepdims=True)
    else:
        gray = np.sum(
            image * np.asarray(weights)[None, None, :], axis=-1, keepdims=True
        )
    return np.where(outside[..., np.newaxis], 0.7 * gray + 0.3 * image, image)
