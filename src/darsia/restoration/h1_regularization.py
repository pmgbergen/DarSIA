"""H1 regularization of images.

"""

from __future__ import annotations

from typing import Union

import numpy as np
import skimage

import darsia as da


def _H1_regularization_array(
    img: np.ndarray,
    mu: float,
    omega: float = 1.0,
    dim: int = 2,
    solver: da.Solver = da.Jacobi(),
) -> np.ndarray:
    """
    H1 regularization of numpy arrays.

    min_u 1/2 ||u - img||_{2,omega}^2 + 1/2 ||nabla u||_{2,mu}^2

    with ||u||_{2,omega}^2 = int omega |u|^2 dx, and similary for the second term.

    Parameters
    ----------
    img : np.ndarray
        Image to regularize.
    mu : float
        Regularization parameter.
    omega : float
        Weighting of the image term (Should account for denoising
        effects).
    dim : int
        Dimension of the image. Default is 2.
    solver : da.Solver
        Solver to use. Default is da.Jacobi().

    Returns
    -------
    np.ndarray
        Regularized image.
    """

    # Keep track of input type and convert input image to float for further calculations
    img_dtype = img.dtype
    img_float = skimage.img_as_float(img)

    # Setup the solver (L2-diffusion equation)
    solver.update_params(
        mass_coeff=omega,
        diffusion_coeff=mu,
        dim=dim,
    )

    # Solve the minimization problem for each channel separately
    if len(img.shape) != dim:
        # Iterate over all components of the image and apply the regularization
        # separately.
        img_reordered = np.moveaxis(
            img_float, np.arange(dim).tolist(), np.arange(-dim, 0).tolist()
        )
        out_img_reordered = np.zeros_like(img_reordered)
        for idx in np.ndindex(img_reordered.shape[:-dim]):
            out_img_reordered[idx] = solver(
                x0=img_reordered[idx], rhs=omega * img_reordered[idx]
            )
        out_img = np.moveaxis(
            out_img_reordered, np.arange(-dim, 0).tolist(), np.arange(dim).tolist()
        )

    else:
        out_img = solver(x0=img, rhs=omega * img)

    # Convert output image to the same type as the input image
    return da.convert_dtype(out_img, img_dtype)


def _H1_regularization_image(
    img: da.Image,
    mu: float,
    omega: float = 1.0,
    dim: int = 2,
    solver: da.Solver = da.Jacobi(),
) -> da.Image:
    """H1 regularization of darsia.Image.

    Parameters
    ----------
    img : darsia.Image
        Image.
    mu : float
        Regularization parameter.
    omega : float
        Weighting of the image term (Should account for denoising
        effects).
    dim : int
        Dimension of the image. Default is 2.
    solver : da.Solver
        Solver to use. Default is da.Jacobi().

    Returns
    -------
    darsia.Image
        Regularized image.
    """
    regularized_img = img.copy()
    regularized_img.img = _H1_regularization_array(
        img=img.img,
        mu=mu,
        omega=omega,
        dim=dim,
        solver=solver,
    )
    return regularized_img


def H1_regularization(
    img: Union[np.ndarray, da.Image],
    mu: float,
    omega: float = 1.0,
    dim: int = 2,
    solver: da.Solver = da.Jacobi(),
) -> Union[np.ndarray, da.Image]:
    """Inline application of H1 regularization.

    Parameters
    ----------
    img : np.ndarray or Image
        Image.
    mu : float
        Regularization parameter.
    omega : float
        Weighting of the image term (Should account for denoising
        effects).
    dim : int
        Dimension of the image. Default is 2.
    solver : da.Solver
        Solver to use. Default is da.Jacobi().

    Returns
    -------
    np.ndarray or Image
        Regularized image (same type as input).
    """
    if isinstance(img, np.ndarray):
        return _H1_regularization_array(
            img=img,
            mu=mu,
            omega=omega,
            dim=dim,
            solver=solver,
        )
    elif isinstance(img, da.Image):
        return _H1_regularization_image(
            img=img,
            mu=mu,
            omega=omega,
            dim=dim,
            solver=solver,
        )
    else:
        raise TypeError(f"Type {type(img)} not supported.")
