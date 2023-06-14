"""TV denoising for numpy arrays."""

from typing import Optional, Union
from warnings import warn

import numpy as np
import skimage

import darsia as da


def split_bregman_anisotropic_tvd(
    im: np.ndarray,
    mu: Union[float, np.ndarray],
    omega: Union[float, np.ndarray] = 1.0,
    ell: Optional[Union[float, np.ndarray]] = None,
    x0: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    dim: int = 2,
    maxiter: int = 100,
    tol: Optional[float] = None,
    solver_type: str = "jacobi",
    solver_iter: int = 1,
    solver_tol: Optional[float] = None,
    mg_depth: int = 3,
    verbose: Union[bool, int] =False,
) -> np.ndarray:
    """Split Bregman algorithm for anisotropic TV denoising.

    Args:
        im (array): image array
        mu (float or array): TV penalization parameter
        omega (float or array): mass penalization parameter
        ell (float or array): diffusion penalization parameter
        x0 (tuple of arrays, optional): initial guess for image and split Bregman
            variables
        dim (int): spatial dimension of the image
        maxiter (int): maximum number of iterations
        tol (float): tolerance for relative increment of the energy functional
        solver_type (str): type of solver to use
        solver_iter (int): number of iterations of the inner solver
        solver_tol (float): tolerance of the inner solver
        mg_depth (int): depth of the inner multigrid solver
        verbose (bool, int): verbosity (frequency if int)

    Returns:
        array: denoised image

    """

    # Define penalization parameter ell
    if ell is None:
        ell = 2 * mu

    # Define solver
    if solver_type == "jacobi":
        solver: da.Solver = da.Jacobi(
            mass_coeff=omega,
            diffusion_coeff=ell,
            dim=dim,
            maxiter=solver_iter,
            tol=solver_tol,
            verbose=False,
        )
    elif solver_type == "mg":
        solver = da.MG(
            mass_coeff=omega,
            diffusion_coeff=ell,
            dim=dim,
            maxiter=solver_iter,
            tol=solver_tol,
            depth=mg_depth,
            verbose=False,
        )
    elif solver_type == "cg":
        solver = da.CG(
            mass_coeff=omega,
            diffusion_coeff=ell,
            dim=dim,
            maxiter=solver_iter,
            tol=solver_tol,
            verbose=False,
        )
    else:
        raise ValueError("Solver type not recognized.")

    # Get data type of input image
    im_dtype = im.dtype

    # Transform image to float, and define all variables, incl. split Bregman variables
    im = skimage.img_as_float(im.copy())
    d = [np.zeros(im.shape) for _ in range(dim)]
    b = [np.zeros(im.shape) for _ in range(dim)]

    # Store input image
    im_ref = im.copy()

    # Define initial guess if provided, otherwise start with input image
    if x0 is not None:
        im0, d0, b0 = x0
        im = skimage.img_as_float(im0)
        d = d0
        b = b0

    # store the norm of im_ref if tol is not None
    if tol is not None:
        im_ref_nrm = np.linalg.norm(im_ref)

    # Define energy functional for verbose
    def _functional(x: np.ndarray) -> float:
        return 0.5 * np.linalg.norm(omega * (x - im_ref)) ** 2 + sum(
            [
                np.sum(np.abs(mu * da.backward_diff(im=x, axis=j, dim=dim)))
                for j in range(dim)
            ]
        )

    if verbose if isinstance(verbose, bool) else verbose > 0:
        print(f"The energy functional starts at {_functional(im)}")

    # Define shrinkage operator (shrinks element-wise by k)
    def _shrink(x: np.ndarray, k: Union[float, np.ndarray]) -> np.ndarray:
        return np.maximum(np.abs(x) - k, 0) * np.sign(x)

    # Define right hand side function
    def _rhs_function(dt: list[np.ndarray], bt: list[np.ndarray]) -> np.ndarray:
        return omega * im_ref + ell * sum(
            [da.forward_diff(im=bt[i] - dt[i], axis=i, dim=dim) for i in range(dim)]
        )

    # Bregman iterations
    for iteration in range(maxiter):

        # First step
        im_new = solver(x0=im, rhs=_rhs_function(d, b))

        # Second step.
        for j in range(dim):
            dub = da.backward_diff(im=im_new, axis=j, dim=dim) + b[j]
            d[j] = _shrink(dub, mu / ell)
            b[j] = dub - d[j]

        # Convergence check.
        if tol is not None:
            nrm = np.linalg.norm(im_new - im) / im_ref_nrm
            if verbose if isinstance(verbose, bool) else iter % verbose == 0:
                print(f"The relative increment at iteration {iteration} is {nrm}")
            if nrm < tol:
                break

        # Update of result
        im = im_new

        # Output
        if verbose if isinstance(verbose, bool) else iter % verbose == 0:
            print(f"The the energy functional iteration {iter} is {_functional(im)}")

    # Transform image back to original dtype
    if im_dtype == np.uint8:
        im = skimage.img_as_ubyte(im)
    elif im_dtype == np.uint16:
        im = skimage.img_as_uint(im)
    elif im_dtype == np.float32:
        im = skimage.img_as_float32(im)
    elif im_dtype == np.float64:
        im = skimage.img_as_float64(im)
    else:
        warn("{im_dtype} is not a supported dtype. Returning {im.dtype} image.")

    return im
