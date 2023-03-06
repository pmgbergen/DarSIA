from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sps
import skimage
from scipy.sparse.linalg import LinearOperator

import darsia


def heterogeneous_tv_denoising(
    img: np.ndarray,
    weight: Optional[float, np.ndarray] = 0.1,
    omega: Optional[float, np.ndarray] = 1.0,
    penalty: float = 1,
    tvd_stopping_criterion: darsia.StoppingCriterion = darsia.StoppingCriterion(
        1e-4, 100
    ),
    cg_stopping_criterion: darsia.StoppingCriterion = darsia.StoppingCriterion(
        1e-4, 100
    ),
    verbosity: int = 0,
) -> np.ndarray:
    """
    Anisotropic TV denoising using the split Bregman from Goldstein and Osher:
        min_u = weight * (|dxu|+|dyu|) +  1 / 2 * ||u-f||^{2, omega}_2.

    NOTE: In contrast to skimage.restoration.denoise_tv_bregman, pixel-wise definition of
          the effective regularization parameter mu = omega / weight is allowed.

    Arguments:
        img (array): image that should be regularized
        weight (float or array): regularization
        omega (float or array): pore space indicator
        penalty (float): Penalty coefficient from Goldstein and Osher's algorithm
        tvd_stopping_criterion (darsia.StoppingCriterion): stopping criterion for the Bregman
                                                         split containing information about
                                                         tolerance, maximum number of
                                                         iterations and norm
        cg_stopping_criterion (darsia.StoppingCriterion): stopping criterion for the inner CG
                                                        solve containing information about
                                                        tolerance, maximum number of
                                                        iterations and norm
        verbosity (int): Set to true for verbosity
    """

    # Set verbosity of stopping criterion
    tvd_stopping_criterion.verbose = verbosity == 0

    # Combine regularization and pore space
    if omega is None:
        mu = 1.0 / weight
    else:
        mu = omega / weight

    # Extract the two images
    rhs = skimage.img_as_float(img)
    im = skimage.img_as_float(img)

    # Initialize
    dx = np.zeros(rhs.shape)
    dy = np.zeros(rhs.shape)
    bx = np.zeros(rhs.shape)
    by = np.zeros(rhs.shape)

    # Create algorithm specific functionality
    def mv(x: np.ndarray) -> np.ndarray:
        """
        Left-hand-side operator defined as a linear operator acting on flat images

        Args:
            x (np.ndarray): 1d (flat) image

        Returns:
            np.ndarray: 1d response of mu * Id - lambda * laplace

        """
        # Make image 2d (Laplace operator acts on 2d images)
        x = np.reshape(x, im.shape[:2])

        # Apply 'mu * Id - penalty * laplace'
        lhs = np.multiply(mu, x) - penalty * darsia.laplace(x)

        # Flatten response
        return np.ravel(lhs)

    # Cast mv as a linear operator
    im_size = np.prod(im.shape[:2])
    lhsoperator = LinearOperator((im_size, im_size), matvec=mv)

    # Right-hand-side operator
    def rhsoperator(
        rhs: np.ndarray, dx: np.ndarray, dy: np.ndarray, bx: np.ndarray, by: np.ndarray
    ) -> np.ndarray:
        """
        Right hand side operator of split Bregman algorithm.

        Args:
            rhs: 2d reference image
            dx: 2d approximation of grad_x smooth_img
            dy: 2d approximation of grad_y smooth_img
            bx: 2d approximation b_x in split Bregman algorithm
            by: 2d approximation b_y in split Bregman algorithm

        Returns:
            np.ndarray: 1d flat response of the rhs

        """
        # Two-dimensional varian
        rhs = np.multiply(mu, rhs) + penalty * (
            darsia.forward_diff_x(dx - bx) + darsia.backward_diff_y(dy - by)
        )
        # Return flat 1d version
        return np.ravel(rhs)

    def shrink(x):
        n = np.linalg.norm(x, ord="fro")
        return x / n * max(n - 1.0 / penalty, 0.0)

    iterations: int = 0
    increment: np.ndarray = im.copy()

    # TVD algorithm from Goldstein and Osher, cf doi:10.1137/080725891
    while not (tvd_stopping_criterion.check_relative(increment, im, iterations)):
        # Cache the previous step for tracking the increment
        im_old = np.copy(im)

        # Solve (mu * Id - penalty * laplace) * im = mu * img_ref + penalty * grad(d - b)
        # using unpreconditioned CG.
        im = np.reshape(
            sps.linalg.cg(
                lhsoperator,
                rhsoperator(rhs, dx, dy, bx, by),
                np.ravel(im),
                tol=cg_stopping_criterion.tolerance,
                maxiter=cg_stopping_criterion.max_iterations,
            )[0],
            im.shape[:2],
        )

        # Shrink operations to update d
        dx = shrink(darsia.backward_diff_x(im) + bx)
        dy = shrink(darsia.backward_diff_y(im) + by)

        # Update of b (Bregman)
        bx = bx + darsia.backward_diff_x(im) - dx
        by = by + darsia.backward_diff_y(im) - dy

        # Track convergence
        increment = im - im_old
        iterations += 1

    return im
