"""
Contains norms and inner products. Might be natural to make a class of this at some point.
So far only the Frobenius norms/inner-product is included, using the numpy.tensordot function.
"""

from __future__ import annotations

from math import sqrt

import numpy as np


def im_product(im1: np.ndarray, im2: np.ndarray) -> float:
    """
    L2-type inner product of two images

    Arguments:
        im1 (np.ndarray): first image factor
        im2 (np.ndarray): second image factor

    Returns:
        float: L2-type inner product of the two images
    """
    return float(np.tensordot(im1, im2))


def frobenius_norm(im: np.ndarray) -> float:
    """
    Frobenius-type norm induced by im_product

    Arguments:
        im (np.ndarray): image matrix

    Returns:
        float: norm of image
    """
    return sqrt(im_product(im, im))
