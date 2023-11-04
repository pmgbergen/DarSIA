"""Finite volume utilities."""

from typing import Optional

import numpy as np
import scipy.sparse as sps
from scipy.stats import hmean

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
                grid.face_vol[d] * np.tile([1, -1], grid.num_faces_per_axis[d])
                for d in range(grid.dim)
            ]
        )
        div_row = np.concatenate(
            [np.ravel(grid.connectivity[grid.faces[d]]) for d in range(grid.dim)]
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


class FVTangentialFaceReconstruction:
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
        shape = (grid.num_faces, grid.num_faces)

        # Each interior inner face has four neighboring faces with normal direction
        # and all oriented in the same direction. In three dimensions, two such normal
        # directions exist. For outer inner faces, there are only two such neighboring
        # faces.
        data = 0.25 * np.ones(4 * grid.num_faces, dtype=float)

        # The rows correspond to the faces for which the tangential fluxes (later to be
        # tiled for addressing the right amount of vectorial components).
        rows = np.concatenate([np.repeat(grid.faces[d], 4) for d in range(grid.dim)])

        # The columns correspond to the (orthogonal) faces contributing to the average
        # of the tangential fluxes. The main idea is for each face to follow the
        # connectivity.
        cols = [
            np.concatenate(
                [
                    np.ravel(
                        grid.reverse_connectivity[
                            d_perp,
                            np.ravel(grid.connectivity[grid.faces[d]]),
                        ]
                    )
                    for d in range(grid.dim)
                    for d_perp in [np.delete(range(grid.dim), d)[i]]
                ]
            )
            for i in range(grid.dim - 1)
        ]

        # Construct and cache the sparse projection matrix, need to extract those faces
        # which are not inner faces.
        self.mat = [
            sps.csc_matrix(
                (
                    data[col != -1],
                    (rows[col != -1], col[col != -1]),
                ),
                shape=shape,
            )
            for col in cols
        ]

        # Cache some informatio
        self.num_tangential_directions = grid.dim - 1
        self.grid = grid

    def __call__(self, normal_flux: np.ndarray, concatenate: bool = True) -> np.ndarray:
        """Apply the operator to the normal fluxes.

        Args:
            normal_flux (np.ndarray): normal fluxes
            concatenate (bool, optional): whether to concatenate the tangential fluxes

        Returns:
            np.ndarray: tangential fluxes

        """
        # Apply the operator to the normal fluxes
        tangential_flux = [
            self.mat[d].dot(normal_flux) for d in range(self.num_tangential_directions)
        ]
        if concatenate:
            tangential_flux = np.concatenate(tangential_flux, axis=0)

        return tangential_flux


class FVFullFaceReconstruction:
    def __init__(self, grid: darsia.Grid) -> None:
        self.grid = grid
        self.tangential_reconstruction = FVTangentialFaceReconstruction(grid)

    def __call__(self, normal_flux: np.ndarray) -> np.ndarray:
        """Reconstruct the full fluxes from the normal and tangential fluxes.

        Args:
            normal_flux (np.ndarray): normal fluxes

        Returns:
            np.ndarray: full fluxes

        """
        # Apply the operator to the normal fluxes
        tangential_fluxes = self.tangential_reconstruction(normal_flux, False)

        # Reconstruct the full fluxes
        dim = self.grid.dim
        full_flux = np.zeros((self.grid.num_faces, dim), dtype=float)
        for d in range(dim):
            full_flux[self.grid.faces[d], d] = normal_flux[self.grid.faces[d]]
            for i, d_perp in enumerate(np.delete(range(dim), d)):
                full_flux[self.grid.faces[d], d_perp] = tangential_fluxes[i][
                    self.grid.faces[d]
                ]

        return full_flux


# ! ---- Finite volume projection operators ----


def face_to_cell(
    grid: darsia.Grid, flat_flux: np.ndarray, pt: Optional[np.ndarray] = None
) -> np.ndarray:
    """Reconstruct the vector fluxes on the cells from normal fluxes on the faces.

    Use the Raviart-Thomas reconstruction of the fluxes on the cells from the fluxes
    on the faces, and use arithmetic averaging of the fluxes on the faces,
    equivalent with the L2 projection of the fluxes on the faces to the fluxes on
    the cells.

    Matrix-free implementation.

    Args:
        grid (darsia.Grid): grid
        flat_flux (np.ndarray): flat fluxes (normal fluxes on the faces)
        pt (np.ndarray, optional): points at which to evaluate the fluxes, relative to
            the reference cell [0,1]**dim, in matrix-indexing. Uses Defaults to None.
            Then the center of the reference cell is used.

    Returns:
        np.ndarray: cell-based vectorial fluxes

    """
    # Initialize the cell-based fluxes
    cell_flux = np.zeros((*grid.shape, grid.dim), dtype=float)

    # Pick the cell center if no pt provided
    if pt is None:
        pt = np.ones(grid.dim) / 2

    # Make pt an array - needed for 1d case
    if grid.dim == 1:
        pt = np.array([pt])

    if grid.dim >= 1:
        cell_flux[:-1, ..., 0] += pt[0] * flat_flux[grid.faces[0]].reshape(
            grid.faces_shape[0], order="F"
        )
        cell_flux[1:, ..., 0] += (1 - pt[0]) * flat_flux[grid.faces[0]].reshape(
            grid.faces_shape[0], order="F"
        )
    if grid.dim >= 2:
        cell_flux[:, :-1, ..., 1] += pt[1] * flat_flux[grid.faces[1]].reshape(
            grid.faces_shape[1], order="F"
        )
        cell_flux[:, 1:, ..., 1] += (1 - pt[1]) * flat_flux[grid.faces[1]].reshape(
            grid.faces_shape[1], order="F"
        )
    if grid.dim >= 3:
        cell_flux[:, :, :-1, ..., 2] += pt[2] * flat_flux[grid.faces[2]].reshape(
            grid.faces_shape[2], order="F"
        )
        cell_flux[:, :, 1:, ..., 2] += (1 - pt[2]) * flat_flux[grid.faces[2]].reshape(
            grid.faces_shape[2], order="F"
        )
    if grid.dim > 3:
        raise NotImplementedError(f"Dimension {grid.dim} not supported.")

    return cell_flux


def cell_to_face_average(
    grid: darsia.Grid, cell_qty: np.ndarray, mode: str
) -> np.ndarray:
    """Project scalar cell quantity to scalar face quantity via averaging.

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
    # Collect cell quantities for each face
    neighbouring_cell_values = np.zeros((grid.num_faces, 2), dtype=float)
    face_qty = np.zeros(grid.num_faces, dtype=float)

    # Flatten cell quantities
    flat_cell_qty = cell_qty.ravel("F")

    # Iterate over normal directions
    for orientation in range(grid.dim):
        # Fetch faces
        faces = grid.faces[orientation]

        # Fetch neighbouring cells
        neighbouring_cells = grid.connectivity[faces]

        # Fetch cell quantities
        neighbouring_cell_values[faces, 0] = flat_cell_qty[neighbouring_cells[:, 0]]
        neighbouring_cell_values[faces, 1] = flat_cell_qty[neighbouring_cells[:, 1]]

    # Perform averaging
    if mode == "arithmetic":
        face_qty = 0.5 * np.sum(neighbouring_cell_values, axis=1)
    elif mode == "harmonic":
        face_qty = hmean(neighbouring_cell_values, axis=1)
    else:
        raise ValueError(f"Mode {mode} not supported.")

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
