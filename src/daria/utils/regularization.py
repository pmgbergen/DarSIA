from __future__ import annotations

from typing import Callable
import daria as da
import numpy as np


def tv_denoising(
    img: da.Image,
    mu: float,
    l: float = 1,
    solver: Callable = da.cg,
    tol: float = 0.01,
    max_iter: int = 1000,
    norm: Callable = da.frobenius_norm,
    verbose: bool = False,
) -> da.Image:
    """
    Anisotropic TV denoising using the Bregman split from Goldstein and Osher: min_u = |dxu|+|dyu|+mu/2||u-f||^2_2

    Arguments:
        img: The DarIA image that should be regularized
        mu: Regularization coefficient
        l: Penalty coefficient from Goldstein and Osher's algorithm
        solver: Linear solver
    """

    # Make a copy of th original image for return
    reg_image = img.copy()
    rhs = img.img
    im = reg_image.img
    dx = np.zeros(rhs.shape)
    dy = np.zeros(rhs.shape)
    bx = np.zeros(rhs.shape)
    by = np.zeros(rhs.shape)

    def lhsoperator(x: np.ndarray):
        return mu * x - l * da.laplace(x)

    def rhsoperator(rhs, dx, dy, bx, by):
        return mu * rhs + l * (da.forward_diff_x(dx - bx) + da.backward_diff_y(dy - by))

    def shrink(x):
        n = norm(x)
        return x / n * max(n - 1 / l, 0)

    for i in range(max_iter):
        im_old = np.ndarray.copy(im)
        im = solver(
            lhsoperator,
            rhsoperator(rhs, dx, dy, bx, by),
            im,
            tol,
            norm,
            verbose=verbose,
        )
        dx = shrink(da.backward_diff_x(im) + bx)
        dy = shrink(da.backward_diff_y(im) + by)
        bx = bx + da.backward_diff_x(im) - dx
        by = by + da.backward_diff_y(im) - dy
        nr = norm(im - im_old) / norm(im)
        if nr < tol:
            if verbose:
                print("Final relative increment: ", nr)
                print("Total number of Bregman split iterations:", i)
            break
        if verbose:
            print("Relative increment norm: ", nr)
            print("Iteration number: ", i)
    reg_image.img = im.astype(np.uint8)
    return reg_image
