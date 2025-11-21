"""Utility functions for arithmetic operations."""

import numpy as np


def array_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the product of two arrays.

    Args:
        a (np.ndarray): array a
        b (np.ndarray): array b

    Returns:
        np.ndarray: product

    """
    if len(a.shape) == len(b.shape) + 1 and a.shape[:-1] == b.shape:
        return a * b[..., np.newaxis]
    elif len(a.shape) == len(b.shape) - 1 and a.shape == b.shape[:-1]:
        return a[..., np.newaxis] * b
    elif len(a.shape) == len(b.shape) and a.shape == b.shape:
        return a * b
    else:
        raise ValueError("Shapes not compatible.")
