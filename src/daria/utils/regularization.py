from __future__ import annotations

from typing import Callable
import daria as da
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import LinearOperator
import skimage

def tv_denoising(
    img: da.Image,
    mu: Union[float, np.ndarray],
    l: float,
    tvd_stoppingCriterion: da.StoppingCriterion = da.StoppingCriterion(
        1e-2, 100
    ),
    cg_stoppingCriterion: da.StoppingCriterion = da.StoppingCriterion(
        1e-2, 100
    ),
    verbose: bool = False,
) -> da.Image:
    """
    Anisotropic TV denoising using the Bregman split from Goldstein and Osher: min_u = |dxu|+|dyu|+mu/2||u-f||^2_2

    NOTE: In contrast to skimage.restoration.denoise_tv_bregman, pixel-wise definition of the regularization parameter mu is allowed.

    Arguments:
        img (daria.Image): Image that should be regularized
        mu (float or array): Regularization coefficient / matrix
        l (float): Penalty coefficient from Goldstein and Osher's algorithm
        tvd_stoppingCriterion (daria.StoppingCriterion): stopping criterion for the Bregman split containing information about tolerance, maximum number of iterations and norm
        cg_stoppingCriterion (daria.StoppingCriterion): stopping criterion for the inner CG solve containing information about tolerance, maximum number of iterations and norm
        verbose (bool): Set to true for verbosity; default value is False
    """

    # Set verbosity of stopping criterion
    tvd_stoppingCriterion.verbose = verbose

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
        return (np.multiply(mu, x) - l * da.laplace(x)).flatten()
    im_size = im.shape[0] * im.shape[1]
    lhsoperator = LinearOperator((im_size, im_size), matvec=mv) 

    # Right-hand-side operator
    def rhsoperator(rhs, dx, dy, bx, by):
        return np.multiply(mu, rhs) + l * (da.forward_diff_x(dx - bx) + da.backward_diff_y(dy - by))

    def shrink(x):
        n = np.linalg.norm(x, ord='fro')
        return x / n * max(n - 1. / l, 0.)

    iterations: int = 0
    increment: np.ndarray = im.copy()

    # TVD algorithm from Goldstein and Osher, cf doi:10.1137/080725891
    while not (tvd_stoppingCriterion.check_relative(increment, im, iterations)):
        im_old = np.ndarray.copy(im)
        im = np.reshape(
            sps.linalg.cg(
                lhsoperator,
                rhsoperator(rhs, dx, dy, bx, by).flatten(),
                im.flatten(),
                tol = cg_stoppingCriterion.tolerance,
                maxiter = cg_stoppingCriterion.max_iterations,
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
