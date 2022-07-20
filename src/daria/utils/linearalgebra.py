import numpy as np
import cv2
from math import sqrt


# TODO: make proper stopping criteria


def im_product(im1: np.ndarray, im2: np.ndarray) -> float:
    return np.tensordot(im1, im2)


def frobenius_norm(im: np.ndarray) -> float:
    return sqrt(im_product(im, im))


def richardson(
    operator: function,
    rhs: np.ndarray,
    im0: np.ndarray,
    tol: float,
    omega: float,
    norm: function,
    max_iter: int = 1000,
):
    res = rhs - operator(im0)
    im = im0
    nr = norm(res)
    iteration = 0
    while nr > tol and iteration < max_iter:
        im = im + omega * res
        res = rhs - operator(im)
        nr = norm(res)
        print(nr)
        iteration += 1
    print(iteration)

    return im


def cg(
    operator: function,
    rhs: np.ndarray,
    im0: np.ndarray,
    tol: float,
    norm: function,
    max_iter: int = 1000,
):
    im = im0
    res = operator(im0) - rhs
    p = -res
    for _ in range(max_iter):
        po = operator(p)
        alpha = -im_product(res, p) / im_product(po, p)
        im = im + alpha * p
        res = operator(im) - rhs
        beta = im_product(res, po) / im_product(po, p)
        p = -res + beta * p
        print(norm(res))
        if norm(res) < tol:
            break
    alpha = -im_product(res, p) / im_product(operator(p), p)
    im = im + alpha * p
    return im
