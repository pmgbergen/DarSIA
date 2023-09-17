"""Finite volume utilities."""

import numpy as np
import scipy.sparse as sps

import darsia


class FVDivergence:
    """Finite volume divergence operator."""

    def __init__(self, grid: darsia.Grid) -> None:
        # Define sparse divergence operator, integrated over elements.
        # Note: The global direction of the degrees of freedom is hereby fixed for all
        # faces. Fluxes across vertical faces go from left to right, fluxes across
        # horizontal faces go from bottom to top. To oppose the direction of the outer
        # normal, the sign of the divergence is flipped for one side of cells for all
        # faces.
        div_shape = (grid.num_cells, grid.num_faces)
        div_data = np.concatenate(
            (
                grid.voxel_size[0] * np.ones(grid.num_vertical_faces, dtype=float),
                grid.voxel_size[1] * np.ones(grid.num_horizontal_faces, dtype=float),
                -grid.voxel_size[0] * np.ones(grid.num_vertical_faces, dtype=float),
                -grid.voxel_size[1] * np.ones(grid.num_horizontal_faces, dtype=float),
            )
        )
        div_row = np.concatenate(
            (
                grid.connectivity[
                    grid.flat_vertical_faces, 0
                ],  # vertical faces, cells to the left
                grid.connectivity[
                    grid.flat_horizontal_faces, 0
                ],  # horizontal faces, cells to the top
                grid.connectivity[
                    grid.flat_vertical_faces, 1
                ],  # vertical faces, cells to the right (opposite normal)
                grid.connectivity[
                    grid.flat_horizontal_faces, 1
                ],  # horizontal faces, cells to the bottom (opposite normal)
            )
        )
        div_col = np.tile(np.arange(grid.num_faces, dtype=int), 2)
        div = sps.csc_matrix(
            (div_data, (div_row, div_col)),
            shape=div_shape,
        )

        # Cache
        self.mat = div


class FVMass:
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
