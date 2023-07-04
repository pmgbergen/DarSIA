"""TV denoising for numpy arrays allowing for heterogeneous weights."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import skimage

import darsia as da


def split_bregman_tvd(
    img: np.ndarray,
    mu: Union[float, np.ndarray] = 1.0,
    omega: Union[float, np.ndarray] = 1.0,
    ell: Optional[Union[float, np.ndarray]] = None,
    dim: int = 2,
    max_num_iter: int = 100,
    eps: Optional[float] = None,
    x0: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    isotropic: bool = False,
    verbose: Union[bool, int] = False,
    solver: da.Solver = da.Jacobi(),
) -> np.ndarray:
    """Split Bregman algorithm for anisotropic TV denoising.

    The Bregman iteration introduces a regularization term to the TV denoising allowing
    for a split of the problem into two subproblems. The first subproblem is a
    diffusion equation with a mass penalization term, the second subproblem is a
    shrinkage step. The regularization term is weighted by the parameter ell.

    Args:
        img (array): image array
        mu (float or array): TV penalization parameter
        omega (float or array): mass penalization parameter
        ell (float or array): regularization parameter
        dim (int): spatial dimension of the image
        max_num_iter (int): maximum number of iterations
        eps (float): tolerance for relative increment of the energy functional
        x0 (tuple of arrays, optional): initial guess for image and split Bregman
            variables
        isotropic (bool): whether to use isotropic TV denoising
        verbose (bool, int): verbosity (frequency if int)
        solver (da.Solver): solver to use for the inner linear system

    Returns:
        array: denoised image

    """
    # Keep track of input type and convert input image to float for further calculations
    img_dtype = img.dtype

    # Store input image and its norm for convergence check
    img_nrm = np.linalg.norm(img)

    # Feed the solver with parameters, follow the suggestion to use double the weight
    # for the diffusion coefficient if no value is provided.
    if ell is None:
        ell = 2 * mu
    solver.update_params(
        mass_coeff=omega,
        diffusion_coeff=ell,
        dim=dim,
    )

    # Define energy functional for verbose
    def _functional(x: np.ndarray) -> float:
        return 0.5 * np.linalg.norm(omega * (x - img)) ** 2 + sum(
            [
                np.sum(np.abs(mu * da.backward_diff(img=x, axis=j, dim=dim)))
                for j in range(dim)
            ]
        )

    # Define shrinkage operator (shrinks element-wise by k)
    def _shrink(x: np.ndarray, k: Union[float, np.ndarray]) -> np.ndarray:
        return np.maximum(np.abs(x) - k, 0) * np.sign(x)

    # Define right hand side function
    def _rhs_function(dt: list[np.ndarray], bt: list[np.ndarray]) -> np.ndarray:
        return omega * img + ell * sum(
            [da.forward_diff(img=bt[i] - dt[i], axis=i, dim=dim) for i in range(dim)]
        )

    # Define initial guess if provided, otherwise start with input image and allovate
    # zero arrays for the split Bregman variables.
    if x0 is not None:
        img0, d0, b0 = x0
        img_iter = skimage.img_as_float(img0)
        d = d0
        b = b0
    else:
        img_iter = skimage.img_as_float(img.copy())
        d = [np.zeros(img.shape) for _ in range(dim)]
        b = [np.zeros(img.shape) for _ in range(dim)]

    if verbose if isinstance(verbose, bool) else verbose > 0:
        print(f"The energy functional starts at {_functional(img)}")

    # Bregman iterations
    for iter in range(max_num_iter):
        # First step - solve the stabilized diffusion system.
        img_new = solver(x0=img_iter, rhs=_rhs_function(d, b))

        # Second step - shrinkage.
        if isotropic:
            raise NotImplementedError("Isotropic TV denoising not implemented yet.")
        else:
            for j in range(dim):
                dub = da.backward_diff(img=img_new, axis=j, dim=dim) + b[j]
                d[j] = _shrink(dub, mu / ell)
                b[j] = dub - d[j]

        # Monitor performance
        relative_increment = np.linalg.norm(img_new - img_iter) / img_nrm
        if verbose if isinstance(verbose, bool) else iter % verbose == 0:
            print(
                f"""Split Bregman iteration {iter} - """
                f"""relative increment: {relative_increment}, """
                f"""energy functional: {_functional(img_iter)}"""
            )

        # Update of result
        img_iter = img_new.copy()

        # Convergence check
        if eps is not None:
            if relative_increment < eps:
                break

    return da.convert_dtype(img_iter, img_dtype)
