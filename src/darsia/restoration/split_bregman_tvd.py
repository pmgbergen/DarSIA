from typing import Optional, Union
from warnings import warn

import numpy as np
import skimage

import darsia as da


def split_bregman_anisotropic_tvd(
    im: np.ndarray,
    mu: Union[float, np.ndarray],
    omega: Union[float, np.ndarray] = 1.0,
    ell: Union[float, np.ndarray] = -100,
    dim: int = 2,
    maxiter: int = 100,
    tol: Optional[float] = None,
    solver_type: str = "jacobi",
    solver_iter: int = 1,
    mg_depth: int = 3,
    verbose=False,
) -> np.ndarray:
    """
    Split Bregman algorithm for anisotropic TV denoising
    """

    # Define penalization parameter ell
    if ell == -100:
        ell = 2 * omega

    # Define solver
    if solver_type == "jacobi":
        solver: da.Solver = da.Jacobi(
            mass_coeff=omega,
            diffusion_coeff=ell,
            dim=dim,
            maxiter=solver_iter,
            tol=tol,
            verbose=False,
        )
    elif solver_type == "mg":
        solver = da.MG(
            mass_coeff=omega,
            diffusion_coeff=ell,
            dim=dim,
            maxiter=solver_iter,
            tol=tol,
            depth=mg_depth,
            verbose=False,
        )
    elif solver_type == "cg":
        solver = da.CG(
            mass_coeff=omega,
            diffusion_coeff=ell,
            dim=dim,
            maxiter=solver_iter,
            tol=tol,
            verbose=False,
        )
    else:
        "Solver type not recognized."

    # Get data type of input image
    im_dtype = im.dtype

    # Transform image to float
    im = skimage.img_as_float(im)

    # Store input image
    im0 = im.copy()

    # store the norm of im0 if tol is not None
    if tol is not None:
        im0_nrm = np.linalg.norm(im0)

    # Define energy functional for verbose
    def _functional(x: np.ndarray) -> float:
        return 0.5 * omega * np.linalg.norm(x - im0) ** 2 + sum(
            [
                np.sum(np.abs(mu * da.backward_diff(im=x, axis=j, dim=dim)))
                for j in range(dim)
            ]
        )

    if verbose:
        print(f"The energy functional starts at {_functional(im)}")

    # Define shrinkage operator (shrinks element-wise by k)
    def _shrink(x: np.ndarray, k: Union[float, np.ndarray]) -> np.ndarray:
        return np.maximum(np.abs(x) - k, 0) * np.sign(x)

    # Define split Bregman variables
    d = [np.zeros(im.shape) for _ in range(dim)]
    b = [np.zeros(im.shape) for _ in range(dim)]

    # Define right hand side function
    def _rhs_function(dt: list[np.ndarray], bt: list[np.ndarray]) -> np.ndarray:
        return omega * im0 + ell * sum(
            [da.forward_diff(im=bt[i] - dt[i], axis=i, dim=dim) for i in range(dim)]
        )

    # Bregman iterations
    for i in range(maxiter):
        im_new = solver(x0=im, rhs=_rhs_function(d, b))
        for j in range(dim):
            dub = da.backward_diff(im=im_new, axis=j, dim=dim) + b[j]
            d[j] = _shrink(dub, mu / ell)
            b[j] = dub - d[j]
        if tol is not None:
            nrm = np.linalg.norm(im_new - im) / im0_nrm
            if verbose:
                print(f"The relative increment at iteration {i} is {nrm}")
            if nrm < tol:
                break
        im = im_new
        if verbose:
            print(f"The the energy functional iteration {i} is {_functional(im)}")

    # Might need to put a clip here to avoid numerical issues
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
