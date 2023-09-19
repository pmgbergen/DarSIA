"""Grid utilities for tensor grids."""

from typing import Union

import numpy as np

import darsia

# TODO make nested lits to arrays for faster access.


class Grid:
    """Tensor grid.

    Attributes:
        shape: Shape of grid.
        ndim: Number of dimensions.
        size: Number of grid points.

    """

    def __init__(self, shape: tuple, voxel_size: Union[float, list] = 1.0):
        """Initialize grid."""

        # Cache grid info
        self.dim = len(shape)
        """int: Number of dimensions."""

        self.shape = shape
        """tuple: Shape of grid, using matrix/tensor indexing."""

        self.voxel_size = (
            np.array(voxel_size)
            if isinstance(voxel_size, list)
            else voxel_size * np.ones(self.dim)
        )
        """np.ndarray: Size of voxels in each dimension."""

        self.face_vol = [
            np.prod(self.voxel_size[np.delete(np.arange(self.dim), d)])
            for d in range(self.dim)
        ]
        """list: Volume of faces in each dimension."""

        assert len(self.voxel_size) == self.dim

        # Define cell and face numbering
        self._setup()

    def _setup(self) -> None:
        """Define cell and face numbering."""

        # ! ---- Grid management ----

        # Define dimensions of the problem and indexing of cells, from here one start
        # counting rows from left to right, from top to bottom.
        self.num_cells = np.prod(self.shape)
        """int: Number of cells."""

        # TODO rename -> cell_index
        self.numbering_cells = np.arange(self.num_cells, dtype=int).reshape(self.shape)
        """np.ndarray: Numbering of cells."""

        # Consider only inner faces; implicitly define indexing of faces (first
        # vertical, then horizontal). The counting of vertical faces starts from top to
        # bottom and left to right. The counting of horizontal faces starts from left to
        # right and top to bottom.
        # vertical_faces_shape = (self.shape[0], self.shape[1] - 1)
        # horizontal_faces_shape = (self.shape[0] - 1, self.shape[1])
        # num_vertical_faces = np.prod(vertical_faces_shape)
        # num_horizontal_faces = np.prod(horizontal_faces_shape)
        # num_faces_axis = [
        #     num_vertical_faces,
        #     num_horizontal_faces,
        # ]
        # num_faces = np.sum(num_faces_axis)

        # Determine number of inner faces in each axis
        self.inner_faces_shape = [
            tuple(np.array(self.shape) - np.eye(self.dim, dtype=int)[d])
            for d in range(self.dim)
        ]
        """list: Shape of inner faces in each axis."""

        self.num_inner_faces = [np.prod(s) for s in self.inner_faces_shape]
        """list: Number of inner faces in each axis."""

        self.num_faces = np.sum(self.num_inner_faces)
        """int: Number of faces."""

        # Define flat indexing of faces, and order of faces, sorted by orientation.
        # vertical faces first, then horizontal faces
        # flat_vertical_faces = np.arange(num_vertical_faces, dtype=int)
        # flat_horizontal_faces = num_vertical_faces + np.arange(
        #     num_horizontal_faces, dtype=int
        # )
        # vertical_faces = flat_vertical_faces.reshape(vertical_faces_shape)
        # horizontal_faces = flat_horizontal_faces.reshape(horizontal_faces_shape)

        # Define indexing and ordering of inner faces. Horizontal -> vertical -> depth.
        # TODO replace with slices
        self.flat_inner_faces = [
            sum(self.num_inner_faces[:d])
            + np.arange(self.num_inner_faces[d], dtype=int)
            for d in range(self.dim)
        ]

        self.inner_faces = [
            self.flat_inner_faces[d].reshape(self.inner_faces_shape[d])
            for d in range(self.dim)
        ]

        # # Identify vertical faces on top, inner and bottom
        # self.top_row_vertical_faces = np.ravel(vertical_faces[0, :])
        # self.inner_vertical_faces = np.ravel(vertical_faces[1:-1, :])
        # self.bottom_row_vertical_faces = np.ravel(vertical_faces[-1, :])
        # # Identify horizontal faces on left, inner and right
        # self.left_col_horizontal_faces = np.ravel(horizontal_faces[:, 0])
        # self.inner_horizontal_faces = np.ravel(horizontal_faces[:, 1:-1])
        # self.right_col_horizontal_faces = np.ravel(horizontal_faces[:, -1])

        # Identify inner faces (full cube)
        if self.dim == 1:
            self.interior_inner_faces = [
                np.ravel(self.inner_faces[0][1:-1]),
            ]
        elif self.dim == 2:
            self.interior_inner_faces = [
                np.ravel(self.inner_faces[0][:, 1:-1]),
                np.ravel(self.inner_faces[1][1:-1, :]),
            ]
        elif self.dim == 3:
            self.interior_inner_faces = [
                np.ravel(self.inner_faces[0][:, 1:-1, 1:-1]),
                np.ravel(self.inner_faces[1][1:-1, :, 1:-1]),
                np.ravel(self.inner_faces[2][1:-1, 1:-1, :]),
            ]
        else:
            raise NotImplementedError(f"Grid of dimension {self.dim} not implemented.")

        # Identify all faces on the outer boundary of the grid. Need to use hardcoded
        # knowledge of the orientation of axes and grid indexing.
        if self.dim == 1:
            self.exterior_inner_faces = [
                np.ravel(self.inner_faces[0][np.array([0, -1])])
            ]
        elif self.dim == 2:
            self.exterior_inner_faces = [
                np.ravel(self.inner_faces[0][:, np.array([0, -1])]),
                np.ravel(self.inner_faces[1][np.array([0, -1]), :]),
            ]
        elif self.dim == 3:
            # TODO
            raise NotImplementedError
            self.outer_faces = []
        else:
            raise NotImplementedError(f"Grid of dimension {self.dim} not implemented.")

        # ! ---- Connectivity ----

        self.connectivity = np.zeros((self.num_faces, 2), dtype=int)
        """np.ndarray: Connectivity (and direction) of faces to cells."""
        if self.dim >= 1:
            self.connectivity[self.flat_inner_faces[0], 0] = np.ravel(
                self.numbering_cells[:-1, ...]
            )
            self.connectivity[self.flat_inner_faces[0], 1] = np.ravel(
                self.numbering_cells[1:, ...]
            )
        if self.dim >= 2:
            self.connectivity[self.flat_inner_faces[1], 0] = np.ravel(
                self.numbering_cells[:, :-1, ...]
            )
            self.connectivity[self.flat_inner_faces[1], 1] = np.ravel(
                self.numbering_cells[:, 1:, ...]
            )
        if self.dim >= 3:
            self.connectivity[self.flat_inner_faces[2], 0] = np.ravel(
                self.numbering_cells[:, :, -1, ...]
            )
            self.connectivity[self.flat_inner_faces[2], 1] = np.ravel(
                self.numbering_cells[:, :, 1:, ...]
            )
        if self.dim > 3:
            raise NotImplementedError(f"Grid of dimension {self.dim} not implemented.")

        ## Vertical faces to left cells
        # connectivity[: num_faces_axis[0], 0] = np.ravel(numbering_cells[:, :-1])
        ## Vertical faces to right cells
        # connectivity[: num_faces_axis[0], 1] = np.ravel(numbering_cells[:, 1:])
        ## Horizontal faces to top cells
        # connectivity[num_faces_axis[0] :, 0] = np.ravel(numbering_cells[:-1, :])
        ## Horizontal faces to bottom cells
        # connectivity[num_faces_axis[0] :, 1] = np.ravel(numbering_cells[1:, :])

        self.reverse_connectivity = -np.ones((self.dim, self.num_cells, 2), dtype=int)
        """np.ndarray: Reverse connectivity (and direction) of cells to faces."""

        # NOTE: The first components addresses the cell, the second the axis, the third
        # the direction of the relative position of the face wrt the cell (0: left/up,
        # 1: right/down, using matrix indexing in 2d - analogously in 3d).

        if self.dim >= 1:
            self.reverse_connectivity[
                0, np.ravel(self.numbering_cells[1:, ...]), 0
            ] = self.flat_inner_faces[0]
            self.reverse_connectivity[
                0, np.ravel(self.numbering_cells[:-1, ...]), 1
            ] = self.flat_inner_faces[0]

        if self.dim >= 2:
            self.reverse_connectivity[
                1, np.ravel(self.numbering_cells[:, 1:, ...]), 0
            ] = self.flat_inner_faces[1]
            self.reverse_connectivity[
                1, np.ravel(self.numbering_cells[:, :-1, ...]), 1
            ] = self.flat_inner_faces[1]

        if self.dim >= 3:
            self.reverse_connectivity[
                2, np.ravel(self.numbering_cells[:, :, 1:, ...]), 0
            ] = self.flat_inner_faces[2]
            self.reverse_connectivity[
                2, np.ravel(self.numbering_cells[:, :, :-1, ...]), 1
            ] = self.flat_inner_faces[2]

        ## Define reverse connectivity. Cell to vertical faces
        # self.connectivity_cell_to_vertical_face = -np.ones((self.num_cells, 2), dtype=int)
        ## Left vertical face of cell
        # self.connectivity_cell_to_vertical_face[
        #    np.ravel(numbering_cells[:, 1:]), 0
        # ] = flat_vertical_faces
        ## Right vertical face of cell
        # self.connectivity_cell_to_vertical_face[
        #    np.ravel(numbering_cells[:, :-1]), 1
        # ] = flat_vertical_faces
        ## Define reverse connectivity. Cell to horizontal faces
        # self.connectivity_cell_to_horizontal_face = np.zeros((self.num_cells, 2), dtype=int)
        ## Top horizontal face of cell
        # self.connectivity_cell_to_horizontal_face[
        #    np.ravel(numbering_cells[1:, :]), 0
        # ] = flat_horizontal_faces
        ## Bottom horizontal face of cell
        # self.connectivity_cell_to_horizontal_face[
        #    np.ravel(numbering_cells[:-1, :]), 1
        # ] = flat_horizontal_faces

        # Info about inner cells
        # TODO rm?
        # self.inner_cells_with_vertical_faces = np.ravel(numbering_cells[:, 1:-1])
        # self.inner_cells_with_horizontal_faces = np.ravel(numbering_cells[1:-1, :])
        self.inner_cells_with_inner_faces = (
            [] + [np.ravel(self.numbering_cells[1:-1, ...])]
            if self.dim >= 1
            else [] + [np.ravel(self.numbering_cells[:, 1:-1, ...])]
            if self.dim >= 2
            else [] + [np.ravel(self.numbering_cells[:, :, 1:-1, ...])]
            if self.dim >= 3
            else []
        )


def generate_grid(image: darsia.Image) -> Grid:
    """Get grid object."""
    shape = image.num_voxels
    voxel_size = image.voxel_size
    return Grid(shape, voxel_size)
