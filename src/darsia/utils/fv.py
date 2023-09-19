"""Finite volume utilities."""

import numpy as np
import scipy.sparse as sps

import darsia

# ! ---- Finite volume operators ----


class FVDivergence:
    """Finite volume divergence operator."""

    def __init__(self, grid: darsia.Grid) -> None:
        # Define sparse divergence operator, integrated over elements.
        # Note: The global direction of the degrees of freedom is hereby fixed for all
        # faces. In 2d, fluxes across vertical faces go from left to right, fluxes
        # across horizontal faces go from bottom to top. To oppose the direction of the
        # outer normal, the sign of the divergence is flipped for one side of cells for
        # all faces. Analogously, in 3d.
        div_shape = (grid.num_cells, grid.num_faces)
        div_data = np.concatenate(
            [
                grid.face_vol[d] * np.tile([1, -1], grid.num_inner_faces[d])
                for d in range(grid.dim)
            ]
        )
        div_row = np.concatenate(
            [
                np.ravel(grid.connectivity[grid.flat_inner_faces[d]])
                for d in range(grid.dim)
            ]
        )
        div_col = np.repeat(np.arange(grid.num_faces, dtype=int), 2)
        div = sps.csc_matrix(
            (div_data, (div_row, div_col)),
            shape=div_shape,
        )

        # Cache
        self.mat = div


class FVMass:
    """Finite volume mass matrix.

    The mass matrix can be formulated for cell and face quantities. For cell
    quantities, the mass matrix is diagonal and has the volume of the cells on the
    diagonal. For face quantities, the mass matrix is diagonal and has half the cell
    volumes on the diagonal, taking into accout itegration over te faces as lower
    dimensional entities, and the double occurrence of the faces in the integration
    over the cells.

    """

    def __init__(
        self, grid: darsia.Grid, mode: str = "cells", lumping: bool = True
    ) -> None:
        # Define sparse mass matrix on cells: flat_mass -> flat_mass
        if mode == "cells":
            mass_matrix = sps.diags(
                np.prod(grid.voxel_size) * np.ones(grid.num_cells, dtype=float)
            )
        elif mode == "faces":
            if lumping:
                mass_matrix = 0.5 * sps.diags(
                    np.prod(grid.voxel_size) * np.ones(grid.num_faces, dtype=float)
                )
            else:
                raise NotImplementedError

                # Define true RT0 mass matrix on faces: flat fluxes -> flat fluxes
                num_inner_cells_with_vertical_faces = len(
                    grid.inner_cells_with_vertical_faces
                )
                num_inner_cells_with_horizontal_faces = len(
                    grid.inner_cells_with_horizontal_faces
                )
                mass_matrix_shape = (grid.num_faces, grid.num_faces)
                mass_matrix_data = np.prod(grid.voxel_size) * np.concatenate(
                    (
                        2 / 3 * np.ones(grid.num_faces, dtype=float),  # all faces
                        1
                        / 6
                        * np.ones(
                            num_inner_cells_with_vertical_faces, dtype=float
                        ),  # left faces
                        1
                        / 6
                        * np.ones(
                            num_inner_cells_with_vertical_faces, dtype=float
                        ),  # right faces
                        1
                        / 6
                        * np.ones(
                            num_inner_cells_with_horizontal_faces, dtype=float
                        ),  # top faces
                        1
                        / 6
                        * np.ones(
                            num_inner_cells_with_horizontal_faces, dtype=float
                        ),  # bottom faces
                    )
                )
                mass_matrix_row = np.concatenate(
                    (
                        np.arange(grid.num_faces, dtype=int),
                        grid.connectivity_cell_to_vertical_face[
                            grid.inner_cells_with_vertical_faces, 0
                        ],
                        grid.connectivity_cell_to_vertical_face[
                            grid.inner_cells_with_vertical_faces, 1
                        ],
                        grid.connectivity_cell_to_horizontal_face[
                            grid.inner_cells_with_horizontal_faces, 0
                        ],
                        grid.connectivity_cell_to_horizontal_face[
                            grid.inner_cells_with_horizontal_faces, 1
                        ],
                    )
                )
                mass_matrix_col = np.concatenate(
                    (
                        np.arange(grid.num_faces, dtype=int),
                        grid.connectivity_cell_to_vertical_face[
                            grid.inner_cells_with_vertical_faces, 1
                        ],
                        grid.connectivity_cell_to_vertical_face[
                            grid.inner_cells_with_vertical_faces, 0
                        ],
                        grid.connectivity_cell_to_horizontal_face[
                            grid.inner_cells_with_horizontal_faces, 1
                        ],
                        grid.connectivity_cell_to_horizontal_face[
                            grid.inner_cells_with_horizontal_faces, 0
                        ],
                    )
                )

                # Define mass matrix in faces
                mass_matrix = sps.csc_matrix(
                    (
                        mass_matrix_data,
                        (mass_matrix_row, mass_matrix_col),
                    ),
                    shape=mass_matrix_shape,
                )

        # Cache
        self.mat = mass_matrix


class FVFaceAverage:
    def __init__(self, grid: darsia.Grid) -> None:
        # Operator for averaging fluxes on orthogonal, neighboring faces
        orthogonal_face_average_shape = (grid.num_faces, grid.num_faces)
        orthogonal_face_average_data = 0.25 * np.concatenate(
            (
                np.ones(
                    2 * len(grid.top_row_vertical_faces)
                    + 4 * len(grid.inner_vertical_faces)
                    + 2 * len(grid.bottom_row_vertical_faces)
                    + 2 * len(grid.left_col_horizontal_faces)
                    + 4 * len(grid.inner_horizontal_faces)
                    + 2 * len(grid.right_col_horizontal_faces),
                    dtype=float,
                ),
            )
        )
        orthogonal_face_average_rows = np.concatenate(
            (
                np.tile(grid.top_row_vertical_faces, 2),
                np.tile(grid.inner_vertical_faces, 4),
                np.tile(grid.bottom_row_vertical_faces, 2),
                np.tile(grid.left_col_horizontal_faces, 2),
                np.tile(grid.inner_horizontal_faces, 4),
                np.tile(grid.right_col_horizontal_faces, 2),
            )
        )
        orthogonal_face_average_cols = np.concatenate(
            (
                # top row: left cell -> bottom face
                grid.connectivity_cell_to_horizontal_face[
                    grid.connectivity[grid.top_row_vertical_faces, 0], 1
                ],
                # top row: vertical face -> right cell -> bottom face
                grid.connectivity_cell_to_horizontal_face[
                    grid.connectivity[grid.top_row_vertical_faces, 1], 1
                ],
                # inner rows: vertical face -> left cell -> top face
                grid.connectivity_cell_to_horizontal_face[
                    grid.connectivity[grid.inner_vertical_faces, 0], 0
                ],
                # inner rows: vertical face -> left cell -> bottom face
                grid.connectivity_cell_to_horizontal_face[
                    grid.connectivity[grid.inner_vertical_faces, 0], 1
                ],
                # inner rows: vertical face -> right cell -> top face
                grid.connectivity_cell_to_horizontal_face[
                    grid.connectivity[grid.inner_vertical_faces, 1], 0
                ],
                # inner rows: vertical face -> right cell -> bottom face
                grid.connectivity_cell_to_horizontal_face[
                    grid.connectivity[grid.inner_vertical_faces, 1], 1
                ],
                # bottom row: vertical face -> left cell -> top face
                grid.connectivity_cell_to_horizontal_face[
                    grid.connectivity[grid.bottom_row_vertical_faces, 0], 0
                ],
                # bottom row: vertical face -> right cell -> top face
                grid.connectivity_cell_to_horizontal_face[
                    grid.connectivity[grid.bottom_row_vertical_faces, 1], 0
                ],
                # left column: horizontal face -> top cell -> right face
                grid.connectivity_cell_to_vertical_face[
                    grid.connectivity[grid.left_col_horizontal_faces, 0], 1
                ],
                # left column: horizontal face -> bottom cell -> right face
                grid.connectivity_cell_to_vertical_face[
                    grid.connectivity[grid.left_col_horizontal_faces, 1], 1
                ],
                # inner columns: horizontal face -> top cell -> left face
                grid.connectivity_cell_to_vertical_face[
                    grid.connectivity[grid.inner_horizontal_faces, 0], 0
                ],
                # inner columns: horizontal face -> top cell -> right face
                grid.connectivity_cell_to_vertical_face[
                    grid.connectivity[grid.inner_horizontal_faces, 0], 1
                ],
                # inner columns: horizontal face -> bottom cell -> left face
                grid.connectivity_cell_to_vertical_face[
                    grid.connectivity[grid.inner_horizontal_faces, 1], 0
                ],
                # inner columns: horizontal face -> bottom cell -> right face
                grid.connectivity_cell_to_vertical_face[
                    grid.connectivity[grid.inner_horizontal_faces, 1], 1
                ],
                # right column: horizontal face -> top cell -> left face
                grid.connectivity_cell_to_vertical_face[
                    grid.connectivity[grid.right_col_horizontal_faces, 0], 0
                ],
                # right column: horizontal face -> bottom cell -> left face
                grid.connectivity_cell_to_vertical_face[
                    grid.connectivity[grid.right_col_horizontal_faces, 1], 0
                ],
            )
        )
        orthogonal_face_average = sps.csc_matrix(
            (
                orthogonal_face_average_data,
                (orthogonal_face_average_rows, orthogonal_face_average_cols),
            ),
            shape=orthogonal_face_average_shape,
        )

        # Cache
        self.mat = orthogonal_face_average


# ! ---- Finite volume projection operators ----


def face_to_cell(grid: darsia.Grid, flat_flux: np.ndarray) -> np.ndarray:
    """Reconstruct the vector fluxes on the cells from normal fluxes on the faces.

    Use the Raviart-Thomas reconstruction of the fluxes on the cells from the fluxes
    on the faces, and use arithmetic averaging of the fluxes on the faces,
    equivalent with the L2 projection of the fluxes on the faces to the fluxes on
    the cells.

    Matrix-free implementation.

    Args:
        grid (darsia.Grid): grid
        flat_flux (np.ndarray): flat fluxes (normal fluxes on the faces)

    Returns:
        np.ndarray: cell-based vectorial fluxes

    """
    # Reshape fluxes - use duality of faces and normals
    horizontal_fluxes = flat_flux[: grid.num_faces_axis[0]].reshape(
        grid.vertical_faces_shape
    )
    vertical_fluxes = flat_flux[grid.num_faces_axis[0] :].reshape(
        grid.horizontal_faces_shape
    )

    # Determine a cell-based Raviart-Thomas reconstruction of the fluxes, projected
    # onto piecewise constant functions.
    cell_flux = np.zeros((*grid.shape, grid.dim), dtype=float)
    # Horizontal fluxes
    cell_flux[:, :-1, 0] += 0.5 * horizontal_fluxes
    cell_flux[:, 1:, 0] += 0.5 * horizontal_fluxes
    # Vertical fluxes
    cell_flux[:-1, :, 1] += 0.5 * vertical_fluxes
    cell_flux[1:, :, 1] += 0.5 * vertical_fluxes

    return cell_flux


def cell_to_face(grid: darsia.Grid, cell_qty: np.ndarray, mode: str) -> np.ndarray:
    """Project scalar cell quantity to scalar face quantity.

    Allow for arithmetic or harmonic averaging of the cell quantity to the faces. In
    the harmonic case, the averaging is regularized to avoid division by zero.
    Matrix-free implementation.

    Args:
        grid (darsia.Grid): grid
        cell_qty (np.ndarray): scalar-valued cell-based quantity
        mode (str): mode of projection, either "arithmetic" or "harmonic"
            (averaging)

    Returns:
        np.ndarray: face-based quantity

    """

    # NOTE: No impact of Grid here, so far! Everything is implicit. This should/could
    # change. In particular when switching to 3d!

    raise NotImplementedError

    # Determine the fluxes on the faces
    if mode == "arithmetic":
        # Employ arithmetic averaging
        horizontal_face_qty = 0.5 * (cell_qty[:, :-1] + cell_qty[:, 1:])
        vertical_face_qty = 0.5 * (cell_qty[:-1, :] + cell_qty[1:, :])
    elif mode == "harmonic":
        # Employ harmonic averaging
        arithmetic_avg_horizontal = 0.5 * (cell_qty[:, :-1] + cell_qty[:, 1:])
        arithmetic_avg_vertical = 0.5 * (cell_qty[:-1, :] + cell_qty[1:, :])
        # Regularize to avoid division by zero
        regularization = 1e-10
        arithmetic_avg_horizontal = (
            arithmetic_avg_horizontal
            + (2 * np.sign(arithmetic_avg_horizontal) + 1) * regularization
        )
        arithmetic_avg_vertical = (
            0.5 * arithmetic_avg_vertical
            + (2 * np.sign(arithmetic_avg_vertical) + 1) * regularization
        )
        product_horizontal = np.multiply(cell_qty[:, :-1], cell_qty[:, 1:])
        product_vertical = np.multiply(cell_qty[:-1, :], cell_qty[1:, :])

        # Determine the harmonic average
        horizontal_face_qty = product_horizontal / arithmetic_avg_horizontal
        vertical_face_qty = product_vertical / arithmetic_avg_vertical
    else:
        raise ValueError(f"Mode {mode} not supported.")

    # Reshape the fluxes - hardcoding the connectivity here
    face_qty = np.concatenate([horizontal_face_qty.ravel(), vertical_face_qty.ravel()])

    return face_qty


# NOTE: Currently not in use. TODO rm?
#    def face_restriction(self, cell_flux: np.ndarray) -> np.ndarray:
#        """Restrict vector-valued fluxes on cells to normal components on faces.
#
#        Matrix-free implementation. The fluxes on the faces are determined by
#        arithmetic averaging of the fluxes on the cells in the direction of the normal
#        of the face.
#
#        Args:
#            cell_flux (np.ndarray): cell-based fluxes
#
#        Returns:
#            np.ndarray: face-based fluxes
#
#        """
#        # Determine the fluxes on the faces through arithmetic averaging
#        horizontal_fluxes = 0.5 * (cell_flux[:, :-1, 0] + cell_flux[:, 1:, 0])
#        vertical_fluxes = 0.5 * (cell_flux[:-1, :, 1] + cell_flux[1:, :, 1])
#
#        # Reshape the fluxes
#        flat_flux = np.concatenate(
#            [horizontal_fluxes.ravel(), vertical_fluxes.ravel()], axis=0
#        )
#
#        return flat_flux
