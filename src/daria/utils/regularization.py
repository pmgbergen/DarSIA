from __future__ import annotations

from typing import Callable
import daria as da
import numpy as np

from daria.utils.linearalgebra import frobenius_norm


def tv_denoising(
    img: da.Image,
    mu: float,
    l: float,
    solver: da.Solver = da.CG(da.StoppingCriterion(0.01, 100, da.frobenius_norm)),
    stoppingCriterion: da.StoppingCriterion = da.StoppingCriterion(
        0.01, 100, da.frobenius_norm
    ),
    verbose: bool = False,
) -> da.Image:
    """
    Anisotropic TV denoising using the Bregman split from Goldstein and Osher: min_u = |dxu|+|dyu|+mu/2||u-f||^2_2

    Arguments:
        img (daria.Image): Image that should be regularized
        mu (float): Regularization coefficient
        l (float): Penalty coefficient from Goldstein and Osher's algorithm
        solver (daria.Solver): Linear solver, default is Conjugate Gradients
        stoppingCriterion (daria.StoppingCriterion): stopping criterion containing information about tolerance, maximum number of iterations and norm
        verbose (bool): Set to true
    """

    # Set verbosity of stopping criterion
    stoppingCriterion.verbose = verbose
    solver.stoppingCriterion.verbose = verbose

    # Make a copy of the original image for return
    reg_image = img.copy()

    # Extract the two images
    rhs = img.img
    im = reg_image.img

    # Create algorithm specific functionality
    dx = np.zeros(rhs.shape)
    dy = np.zeros(rhs.shape)
    bx = np.zeros(rhs.shape)
    by = np.zeros(rhs.shape)

    # Left-hand-side operator
    def lhsoperator(x: np.ndarray):
        return mu * x - l * da.laplace(x)

    # Right-hand-side operator
    def rhsoperator(rhs, dx, dy, bx, by):
        return mu * rhs + l * (da.forward_diff_x(dx - bx) + da.backward_diff_y(dy - by))

    def shrink(x):
        n = da.frobenius_norm(x)
        return x / n * max(n - 1 / l, 0)

    iterations = 0
    increment = im

    while not (stoppingCriterion.check_relative(increment, im, iterations)):
        im_old = np.ndarray.copy(im)
        im = solver.apply(lhsoperator, rhsoperator(rhs, dx, dy, bx, by), im)
        dx = shrink(da.backward_diff_x(im) + bx)
        dy = shrink(da.backward_diff_y(im) + by)
        bx = bx + da.backward_diff_x(im) - dx
        by = by + da.backward_diff_y(im) - dy
        increment = im - im_old
        iterations += 1
    reg_image.img = im.astype(np.uint8)
    return reg_image
