"""TV denoising for numpy arrays allowing for heterogeneous weights."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import skimage

import darsia as da


def split_bregman_tvd(
    img: Union[np.ndarray, da.Image],
    mu: Union[float, np.ndarray, list] = 1.0,
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
    """Split Bregman algorithm for TV denoising.

    The Bregman iteration introduces a regularization term to the TV denoising allowing
    for a split of the problem into two subproblems. The first subproblem is a
    diffusion equation with a mass penalization term, the second subproblem is a
    shrinkage step. The regularization term is weighted by the parameter ell.

    Args:
        img (array, da.Image): image
        mu (float or array or list): inverse TV penalization parameter, if list
            it must have the lenght of the dimension and each entry corresponds
            to the penalization in the respective direction.
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

    # Controll that mu has correct length if its a list
    if isinstance(mu, list):
        assert len(mu) == dim, "mu must be a list of length dim"

    # Feed the solver with parameters, follow the suggestion to use double the weight
    # for the mass coefficient if no value is provided.
    if ell is None:
        if isinstance(omega, float):
            ell = 2 * omega
        elif isinstance(omega, np.ndarray):
            ell = 2 * omega.copy()
            ell[ell == 0] = 1

    solver.update_params(
        mass_coeff=omega,
        diffusion_coeff=ell,
        dim=dim,
    )

    # Define energy functional for verbose
    def _functional(x: np.ndarray) -> float:
        if isinstance(mu, list):
            return 0.5 * np.linalg.norm(omega * (x - img)) ** 2 + sum(
                [
                    np.sum(np.abs(mu[j] * da.backward_diff(img=x, axis=j, dim=dim)))
                    for j in range(dim)
                ]
            )
        else:
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
    def _rhs_function(dt: np.ndarray, bt: np.ndarray) -> np.ndarray:
        return omega * img + ell * sum(
            [
                da.forward_diff(img=bt[..., i] - dt[..., i], axis=i, dim=dim)
                for i in range(dim)
            ]
        )

    # Define initial guess if provided, otherwise start with input image and allocate
    # zero arrays for the split Bregman variables.
    if x0 is not None:
        img0, d0, b0 = x0
        img_iter = skimage.img_as_float(img0)
        d = d0
        b = b0
    else:
        img_iter = skimage.img_as_float(img.copy())
        d = np.zeros((*img.shape, dim), dtype=img.dtype)
        b = np.zeros((*img.shape, dim), dtype=img.dtype)

    if verbose if isinstance(verbose, bool) else verbose > 0:
        print(f"The energy functional starts at {_functional(img)}")

    # Bregman iterations
    for iter in range(max_num_iter):
        # First step - solve the stabilized diffusion system.
        img_new = solver(x0=img_iter, rhs=_rhs_function(d, b))

        # Second step - shrinkage.
        if isotropic:
            dub = b.copy()
            for j in range(dim):
                dub[..., j] += da.backward_diff(img=img_new, axis=j, dim=dim)
            s = np.linalg.norm(dub, 2, axis=-1)
            shrinkage_factor = np.maximum(s - mu / ell, 0) / (s + 1e-18)
            d = dub * shrinkage_factor[..., None]
            b = dub - d
        elif isinstance(mu, list):
            for j in range(dim):
                dub = da.backward_diff(img=img_new, axis=j, dim=dim) + b[..., j]
                d[..., j] = _shrink(dub, mu[j] / ell)
                b[..., j] = dub - d[..., j]
        else:
            for j in range(dim):
                dub = da.backward_diff(img=img_new, axis=j, dim=dim) + b[..., j]
                d[..., j] = _shrink(dub, mu / ell)
                b[..., j] = dub - d[..., j]

        # Monitor performance
        relative_increment = np.linalg.norm(img_new - img_iter) / img_nrm

        # Update of result
        img_iter = img_new.copy()

        if verbose if isinstance(verbose, bool) else iter % verbose == 0:
            print(
                f"""Split Bregman iteration {iter} - """
                f"""relative increment: {round(relative_increment,5)}, """
                f"""energy functional: {_functional(img_iter)}"""
            )

        # Convergence check
        if eps is not None:
            if relative_increment < eps:
                break

    return da.convert_dtype(img_iter, img_dtype)
