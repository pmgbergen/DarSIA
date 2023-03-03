from __future__ import annotations

from typing import Union

import numpy as np
import scipy.sparse as sps
import skimage
from scipy.sparse.linalg import LinearOperator
import matplotlib.pyplot as plt

import darsia as da


def tv_denoising(
    img: da.Image,
    mu: Union[float, np.ndarray],
    ell: float,
    tvd_stoppingCriterion: da.StoppingCriterion = da.StoppingCriterion(1e-2, 100),
    cg_stoppingCriterion: da.StoppingCriterion = da.StoppingCriterion(1e-2, 100),
    dim: int = 2,
    verbose: bool = False,
) -> da.Image:
    """
    Anisotropic TV denoising using the Bregman split from Goldstein and Osher:
        min_u = sum_i|d_iu|+mu/2||u-f||^2_2

    NOTE: In contrast to skimage.restoration.denoise_tv_bregman, pixel-wise definition of
          the regularization parameter mu is allowed.

    Arguments:
        img (darsia.Image): Image that should be regularized
        mu (float or array): Regularization coefficient / matrix
        ell (float): Penalty coefficient from Goldstein and Osher's algorithm
        tvd_stoppingCriterion (darsia.StoppingCriterion): stopping criterion for the Bregman
                                                         split containing information about
                                                         tolerance, maximum number of
                                                         iterations and norm
        cg_stoppingCriterion (darsia.StoppingCriterion): stopping criterion for the inner CG
                                                        solve containing information about
                                                        tolerance, maximum number of
                                                        iterations and norm
        verbose (bool): Set to true for verbosity; default value is False
    """

    # Set verbosity of stopping criterion
    tvd_stoppingCriterion.verbose = verbose

    # Copy the input image
    img = img.copy()

    # Extract the two images
    rhs = skimage.img_as_float(img.img)
    im = skimage.img_as_float(img.img)
    plt.imshow(im)
    plt.show()
    # Initialize the anisotropic derivatives
    ani_deriv_top = [np.zeros(rhs.shape) for _ in range(dim)]
    ani_deriv_bot = [np.zeros(rhs.shape) for _ in range(dim)]

    # Left-hand-side operator defined as a linear operator acting on flat images
    def mv(x: np.ndarray) -> np.ndarray:
        # Since the Laplace operator acts on 2d images, need to reshape
        # first
        x = np.reshape(x, im.shape[:dim])
        return (np.multiply(mu, x) - ell * da.laplace(x)).flatten()

    im_size = np.prod(im.shape)
    lhsoperator = LinearOperator((im_size, im_size), matvec=mv)

    # Right-hand-side operator
    def rhsoperator(rhs, dt, db):
        return np.multiply(mu, rhs) + ell * sum([da.backward_diff(dt[i]-db[i],i) for i in range(dim)])

    def shrink(x):
        n = np.linalg.norm(x, ord="fro")
        return x / n * max(n - 1.0 / ell, 0.0)

    iterations: int = 0
    increment: np.ndarray = im.copy()

    # TVD algorithm from Goldstein and Osher, cf doi:10.1137/080725891
    while not (tvd_stoppingCriterion.check_relative(increment, im, iterations)):
        im_old = np.ndarray.copy(im)
        im = np.reshape(
            sps.linalg.cg(
                lhsoperator,
                rhsoperator(rhs, ani_deriv_top, ani_deriv_bot).flatten(),
                im.flatten(),
                tol=cg_stoppingCriterion.tolerance,
                maxiter=cg_stoppingCriterion.max_iterations,
            )[0],
            im.shape[:dim],
        )

        for i in range(dim):
            ani_deriv_top[i] = shrink(da.backward_diff(im,i)-ani_deriv_bot[i])
            ani_deriv_bot[i] += da.backward_diff(im,i)-ani_deriv_top[i]
        increment = im - im_old
        iterations += 1

    # Convert to correct format
    plt.imshow(im)
    plt.show()
    if img.original_dtype == np.uint8:
        img.img = skimage.img_as_ubyte(im)
    elif img.original_dtype == np.uint16:
        img.img = skimage.img_as_uint(im)
    else:
        raise Exception(f"Conversion back to {img.original_dtype} is unknown.")

    return img
