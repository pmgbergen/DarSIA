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


class FVTangentialReconstruction:
    """Projection of normal fluxes on grid onto tangential components.

    The tangential components are defined as the components of the fluxes that are
    orthogonal to the normal of the face. The tangential components are determined
    through averaging the fluxes on the faces that are orthogonal to the face of
    interest.

    The resulting tangential flux has co-dimension 1, i.e. it is a scalar quantity in
    2d and a 2-valued vector quantity in 3d.

    """

    def __init__(self, grid: darsia.Grid) -> None:
        """Initialize the average operator.

        Args:
            grid (darsia.Grid): grid

        """

        # Operator for averaging fluxes on orthogonal, neighboring faces
        shape = ((grid.dim - 1) * grid.num_faces, grid.num_faces)

        # Each interior inner face has four neighboring faces with normal direction
        # and all oriented in the same direction. In three dimensions, two such normal
        # directions exist. For outer inner faces, there are only two such neighboring
        # faces.
        data = np.tile(
            0.25
            * np.ones(
                2 * np.array(grid.exterior_inner_faces).size
                + 4 * np.array(grid.interior_inner_faces).size,
                dtype=float,
            ),
            grid.dim - 1,
        )

        # The rows correspond to the faces for which the tangential fluxes are
        # determined times the component of the tangential fluxes.
        rows_outer = np.concatenate(
            [np.repeat(grid.exterior_inner_faces[d], 2) for d in range(grid.dim)]
        )

        rows_inner = np.concatenate(
            [np.repeat(grid.interior_inner_faces[d], 4) for d in range(grid.dim)]
        )

        # The columns correspond to the (orthogonal) faces contributing to the average
        # of the tangential fluxes. The main idea is for each face to follow the
        # connectivity. First, we consider the outer inner faces. For each face, we

        # Consider outer inner faces. For each face, we consider the two neighboring
        # faces with normal direction and all oriented in the same direction. Need to
        # exclude true exterior faces.

        def interleaf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.ravel(np.vstack([a, b]).T)

        # Consider all close-by faces for each outer inner face which are orthogonal
        pre_cols_outer = np.concatenate(
            [
                np.ravel(
                    grid.reverse_connectivity[
                        d_perp,
                        np.ravel(grid.connectivity[grid.exterior_inner_faces[d]]),
                    ]
                )
                for d in range(grid.dim)
                for d_perp in np.delete(range(grid.dim), d)
            ]
        )
        # Clean up - remove true exterior faces
        pre_cols_outer = pre_cols_outer[pre_cols_outer != -1]

        # Same for interior inner faces,
        pre_cols_inner = np.concatenate(
            [
                np.ravel(
                    grid.reverse_connectivity[
                        d_perp,
                        np.ravel(grid.connectivity[grid.interior_inner_faces[d]]),
                    ]
                )
                for d in range(grid.dim)
                for d_perp in np.delete(range(grid.dim), d)
            ]
        )
        assert np.count_nonzero(pre_cols_inner == -1) == 0

        # Collect all rows and columns
        rows = np.concatenate((rows_outer, rows_inner))
        cols = np.concatenate((pre_cols_outer, pre_cols_inner))

        # Construct and cache the sparse projection matrix
        self.mat = sps.csc_matrix(
            (
                data,
                (rows, cols),
            ),
            shape=shape,
        )


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
