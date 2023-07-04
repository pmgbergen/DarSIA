"""L2 regularization of images.

"""

from __future__ import annotations

import numpy as np
import skimage

import darsia as da


def L2_regularization(
    img: np.ndarray,
    mu: float,
    omega: float = 1.0,
    dim: int = 2,
    solver: da.Solver = da.Jacobi(),
) -> np.ndarray:
    """
    L2 regularization of images.

    min_u 1/2 ||u - img||_{2,omega}^2 + 1/2 ||nabla u||_{2,mu}^2

    with ||u||_{2,omega}^2 = \int omega |u|^2 dx, and similary for the second term.

    Args:
        img (np.ndarray): image to regularize
        mu (float): regularization parameter
        omega (Optional[float]): weighting of the image term (Should account for
            denoising effects).
        dim (int): dimension of the image. Default is 2.
        solver (da.Solver): solver to use. Default is da.Jacobi().

    Returns:
        np.ndarray: regularized image

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
