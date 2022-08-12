from __future__ import annotations

from typing import Callable
import daria as da
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import LinearOperator
import skimage

from daria.utils.norms import frobenius_norm

def tv_denoising(
    img: da.Image,
    mu: float,
    l: float,
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
        stoppingCriterion (daria.StoppingCriterion): stopping criterion containing information about tolerance, maximum number of iterations and norm
        verbose (bool): Set to true
    """

    # Set verbosity of stopping criterion
    stoppingCriterion.verbose = verbose

    # Extract the two images
    rhs = skimage.util.img_as_float(img.img)
    im = skimage.util.img_as_float(img.img)

    # Create algorithm specific functionality
    dx = np.zeros(rhs.shape)
    dy = np.zeros(rhs.shape)
    bx = np.zeros(rhs.shape)
    by = np.zeros(rhs.shape)

    # Left-hand-side operator defined as a linear operator acting on flat images
    def mv(x: np.ndarray) -> np.ndarray:
        # Since the Laplace operator acts on 2d images, need to reshape first # FIXME try to rewrite laplace?
        x = np.reshape(x, im.shape[:2])
        return (mu * x - l * da.laplace(x)).flatten()
    im_size = im.shape[0] * im.shape[1]
    lhsoperator = LinearOperator((im_size, im_size), matvec=mv) 

    # Right-hand-side operator
    def rhsoperator(rhs, dx, dy, bx, by):
        return mu * rhs + l * (da.forward_diff_x(dx - bx) + da.backward_diff_y(dy - by))

    def shrink(x):
        n = da.frobenius_norm(x)
        return x / n * max(n - 1. / l, 0.)

    iterations: int = 0
    increment: np.ndarray = im.copy()

    # TVD algorithm from Goldstein and Osher, cf doi:10.1137/080725891
    while not (stoppingCriterion.check_relative(increment, im, iterations)):
        im_old = np.ndarray.copy(im)
        im = np.reshape(
            sps.linalg.cg(
                lhsoperator,
                rhsoperator(rhs, dx, dy, bx, by).flatten(),
                im.flatten(),
                tol = 1e-2,
                maxiter = 100
            )[0],
            im.shape[:2]
        )
        dx = shrink(da.backward_diff_x(im) + bx)
        dy = shrink(da.backward_diff_y(im) + by)
        bx = bx + da.backward_diff_x(im) - dx
        by = by + da.backward_diff_y(im) - dy
        increment = im - im_old
        iterations += 1

    # Convert to correct format
    img.img = skimage.util.img_as_ubyte(im)

    return img
