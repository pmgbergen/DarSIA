from __future__ import annotations

from math import sqrt

import numpy as np

"""
Contains norms and inner products. Might be natural to make a class of this at some point.
So far only the Frobenius norms/inner-product is included, using the numpy.tensordot function.
"""


def im_product(im1: np.ndarray, im2: np.ndarray) -> float:
    return float(np.tensordot(im1, im2))


def frobenius_norm(im: np.ndarray) -> float:
    return sqrt(im_product(im, im))
