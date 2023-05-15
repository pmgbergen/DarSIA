"""
L2 regularization of images.
"""

from __future__ import annotations

from typing import Optional
from warnings import warn

import numpy as np
import skimage

import darsia as da


def L2_regularization(
    im: np.ndarray,
    mu: float,
    omega: float = 1.0,
    color: bool = False,
    dim: int = 2,
    maxiter: int = 100,
    tol: Optional[float] = None,
    solver_type: str = "jacobi",
    mg_depth: int = 3,
    verbose=False,
) -> np.ndarray:
    """
    L2 regularization of images.

    min_u 1/2 ||omega(u - im)||_2^2 + mu/2 ||nabla u||_2^2

    Args:
        im (np.ndarray): image to regularize
        mu (float): regularization parameter
        omega (Optional[float]): weighting of the image
            term (Should account for denoising effects).
        color (bool): if True, the image is assumed to
            be in some trichromatic color space. Default
            is False.
        dim (int): dimension of the image. Default is 2.
        maxiter (int): number of iterations. Default is 100.
        tol (Optional[float]): tolerance for convergence
        solver_type (str): solver to use. Default is "jacobi".
            Options are "jacobi", "cg", and "mg".
        mg_depth (int): depth of the multigrid solver. Default is 3.
        verbose (bool): print convergence information

    output:
        np.ndarray: regularized image
    """

    im_dtype = im.dtype

    # convert input image to float
    im = skimage.img_as_float(im)

    # Setup the solver
    if solver_type == "jacobi":
        solver: da.Solver = da.Jacobi(
            mass_coeff=omega,
            diffusion_coeff=mu,
            dim=dim,
            maxiter=maxiter,
            tol=tol,
            verbose=verbose,
        )
    elif solver_type == "cg":
        solver = da.CG(
            mass_coeff=omega,
            diffusion_coeff=mu,
            dim=dim,
            maxiter=maxiter,
            tol=tol,
            verbose=verbose,
        )

    elif solver_type == "mg":
        solver = da.MG(
            mass_coeff=omega,
            diffusion_coeff=mu,
            depth=mg_depth,
            dim=dim,
            maxiter=maxiter,
            tol=tol,
            verbose=verbose,
        )

    # Solve the minimization problem for each color channel
    # (or just once in case of monochromatic image)
    if color:
        out_im = np.zeros_like(im)
        for i in range(3):
            out_im[:, :, i] = solver(x0=im[:, :, i], rhs=omega * im[:, :, i])
    else:
        out_im = solver(x0=im, rhs=omega * im)

    if im_dtype == np.uint8:
        print(np.max(out_im), np.min(out_im))
        np.clip(out_im, 0, 1, out=out_im)
        out_im = skimage.img_as_ubyte(out_im)
        im = skimage.img_as_ubyte(im)
    elif im_dtype == np.uint16:
        out_im = skimage.img_as_uint(out_im)
        im = skimage.img_as_uint(im)
    elif im_dtype == np.float32:
        out_im = skimage.img_as_float32(out_im)
        im = skimage.img_as_float32(im)
    elif im_dtype == np.float64:
        out_im = skimage.img_as_float64(out_im)
        im = skimage.img_as_float64(im)
    else:
        warn("{im_dtype} is not a supported dtype. Returning {out_im.dtype} image.")
    return out_im
