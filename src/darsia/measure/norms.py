"""Module with interface to compute L1, L2 and H1 distances between two images."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import scipy.sparse as sps

import darsia


def _difference(img_1: darsia.Image, img_2: darsia.Image) -> tuple[darsia.Grid, np.ndarray]:
    """Common setup for the norm-based distances below.

    Parameters
    ----------
    img_1 : darsia.Image
        Image 1.
    img_2 : darsia.Image
        Image 2.

    Returns
    -------
    darsia.Grid
        Grid generated from img_1.
    np.ndarray
        Flat (Fortran-ordered) difference img_2 - img_1.
    """
    assert img_1.scalar and img_2.scalar
    assert img_1.img.shape == img_2.img.shape

    grid = darsia.generate_grid(img_1)
    diff = np.ravel(img_2.img - img_1.img, "F")

    return grid, diff


def _weighted_cell_mass(grid: darsia.Grid, weight: Optional[darsia.Image]) -> sps.spmatrix:
    """Cell mass matrix, optionally scaled by a cell-wise weight."""
    mass_matrix_cells = darsia.FVMass(grid).mat
    if weight is None:
        return mass_matrix_cells
    return sps.diags(np.ravel(weight.img, "F")) @ mass_matrix_cells


def _face_weight(grid: darsia.Grid, weight: Optional[darsia.Image]) -> np.ndarray:
    """Cell-wise weight averaged to faces (ones if no weight is given)."""
    if weight is None:
        return np.ones(grid.num_faces, dtype=float)
    return darsia.cell_to_face_average(grid, weight.img, "arithmetic")


def l1_distance(
    img_1: darsia.Image, img_2: darsia.Image, weight: Optional[darsia.Image] = None
) -> float:
    """L1 distance between two images with the same grid.

    Parameters
    ----------
    img_1 : darsia.Image
        Image 1.
    img_2 : darsia.Image
        Image 2.
    weight : darsia.Image, optional
        Cell-wise weight.

    Returns
    -------
    float
        L1 distance between img_1 and img_2.
    """
    grid, diff = _difference(img_1, img_2)
    cell_volumes = _weighted_cell_mass(grid, weight).diagonal()
    return float(np.abs(diff) @ cell_volumes)


def l2_distance(
    img_1: darsia.Image, img_2: darsia.Image, weight: Optional[darsia.Image] = None
) -> float:
    """L2 distance between two images with the same grid.

    Parameters
    ----------
    img_1 : darsia.Image
        Image 1.
    img_2 : darsia.Image
        Image 2.
    weight : darsia.Image, optional
        Cell-wise weight.

    Returns
    -------
    float
        L2 distance between img_1 and img_2.
    """
    grid, diff = _difference(img_1, img_2)
    mass_matrix_cells = _weighted_cell_mass(grid, weight)
    return float(np.sqrt(diff @ mass_matrix_cells.dot(diff)))


def h1_semi_norm(
    img_1: darsia.Image, img_2: darsia.Image, weight: Optional[darsia.Image] = None
) -> float:
    """H1 seminorm (L2 norm of the gradient) of the difference of two images.

    Reuses the existing finite-volume divergence operator: its transpose maps cell
    values to a face quantity proportional to the jump across the face (as already
    used for the broken-Darcy block in :mod:`darsia.measure.beckmann_problem`),
    scaled by the face area (``face_vol``). Dividing out the (geometric, unweighted)
    face mass -- itself equal to the cell volume, ``face_vol * voxel_size`` -- turns
    that jump back into an actual finite-difference gradient contribution before
    squaring and integrating.

    Parameters
    ----------
    img_1 : darsia.Image
        Image 1.
    img_2 : darsia.Image
        Image 2.
    weight : darsia.Image, optional
        Cell-wise weight.

    Returns
    -------
    float
        H1 seminorm of the difference img_2 - img_1.
    """
    grid, diff = _difference(img_1, img_2)
    face_jump = darsia.FVDivergence(grid).mat.T.dot(diff)
    mass_matrix_faces_diag = darsia.FVMass(grid, "faces", lumping=True).mat.diagonal()
    face_weight = _face_weight(grid, weight)
    seminorm_sq = np.sum(face_weight * face_jump**2 / mass_matrix_faces_diag)
    return float(np.sqrt(seminorm_sq))


def h1_distance(
    img_1: darsia.Image, img_2: darsia.Image, weight: Optional[darsia.Image] = None
) -> float:
    """Full H1 distance (L2 distance and H1 seminorm combined) between two images.

    Parameters
    ----------
    img_1 : darsia.Image
        Image 1.
    img_2 : darsia.Image
        Image 2.
    weight : darsia.Image, optional
        Cell-wise weight.

    Returns
    -------
    float
        H1 distance between img_1 and img_2.
    """
    l2 = l2_distance(img_1, img_2, weight)
    semi_norm = h1_semi_norm(img_1, img_2, weight)
    return float(np.sqrt(l2**2 + semi_norm**2))


def distance(
    img_1: darsia.Image,
    img_2: darsia.Image,
    method: Literal["l1", "l2", "h1", "h1_semi_norm", "wasserstein"] = "wasserstein",
    weight: Optional[darsia.Image] = None,
    **kwargs,
) -> float | tuple[float, dict]:
    """Unified access to distance computation between two images.

    Parameters
    ----------
    img_1 : darsia.Image
        Image 1.
    img_2 : darsia.Image
        Image 2.
    method : Literal["l1", "l2", "h1", "h1_semi_norm", "wasserstein"]
        Method to use.
    weight : darsia.Image, optional
        Cell-wise weight.
    **kwargs
        Additional arguments, forwarded to :func:`wasserstein_distance` for
        ``method="wasserstein"``.

    Returns
    -------
    float | tuple[float, dict]
        Distance between img_1 and img_2 (or (distance, info) for "wasserstein").
    """
    match method:
        case "l1":
            return l1_distance(img_1, img_2, weight)
        case "l2":
            return l2_distance(img_1, img_2, weight)
        case "h1":
            return h1_distance(img_1, img_2, weight)
        case "h1_semi_norm":
            return h1_semi_norm(img_1, img_2, weight)
        case "wasserstein":
            wasserstein_method = kwargs.pop("wasserstein_method", "newton")
            return darsia.wasserstein_distance(
                img_1, img_2, method=wasserstein_method, weight=weight, **kwargs
            )
        case _:
            raise NotImplementedError(f"Method {method} not implemented.")
