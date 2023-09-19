"""Grid utilities for tensor grids."""

from typing import Union

import numpy as np

import darsia

# TODO make nested lits to arrays for faster access.


class Grid:
    """Tensor grid.

    The current implmentatio does nt take into accout true boundary faces assuming
    the boundar values are not of interest.

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

        self.cell_index = np.arange(self.num_cells, dtype=int).reshape(self.shape)
        """np.ndarray: cell indices."""

        # Determine number of inner faces in each axis
        # Flip order of axes to get the correct shape
        if self.dim == 1:
            order = [np.array([0])]
        elif self.dim == 2:
            order = [np.array([0, 1]), np.array([0, 1])]
        elif self.dim == 3:
            order = [
                np.array([0, 1, 2]),
                np.array([1, 0, 2]),
                np.array([2, 0, 1]),
            ]
        self.inner_faces_shape = [
            # tuple(np.roll(np.array(self.shape) - np.eye(self.dim, dtype=int)[d], -d))
            # tuple(np.array(self.shape) - np.eye(self.dim, dtype=int)[d])
            tuple((np.array(self.shape) - np.eye(self.dim, dtype=int)[d])[order[d]])
            for d in range(self.dim)
        ]
        """list: Shape of inner faces in each axis."""

        self.num_faces_per_axis = [np.prod(s) for s in self.inner_faces_shape]
        """list: Number of inner faces in each axis."""

        self.num_faces = np.sum(self.num_faces_per_axis)
        """int: Number of faces."""

        # Define indexing and ordering of inner faces. Horizontal -> vertical -> depth.
        # TODO replace with slices
        self.faces = [
            sum(self.num_faces_per_axis[:d])
            + np.arange(self.num_faces_per_axis[d], dtype=int)
            for d in range(self.dim)
        ]
        """list: Indices of inner faces in each axis."""

        self.face_index = [
            self.faces[d].reshape(self.inner_faces_shape[d]) for d in range(self.dim)
        ]
        """list: Indices of inner faces in each axis, using matrix indexing."""

        # Identify inner faces (full cube)
        self.interior_faces = []
        """list: Indices of interior faces."""

        if self.dim == 1:
            self.interior_faces = [
                np.ravel(self.face_index[0][1:-1]),
            ]
        elif self.dim == 2:
            self.interior_faces = [
                np.ravel(self.face_index[0][:, 1:-1]),
                np.ravel(self.face_index[1][1:-1, :]),
            ]
        elif self.dim == 3:
            self.interior_faces = [
                np.ravel(self.face_index[0][:, 1:-1, 1:-1]),
                np.ravel(self.face_index[1][:, 1:-1, 1:-1]),
                np.ravel(self.face_index[2][:, 1:-1, 1:-1]),
            ]
        else:
            raise NotImplementedError(f"Grid of dimension {self.dim} not implemented.")

        # Identify all faces on the outer boundary of the grid. Need to use hardcoded
        # knowledge of the orientation of axes and grid indexing.
        self.exterior_faces = []
        """list: Indices of exterior faces (inner faces of boundary cells)."""

        if self.dim == 1:
            self.exterior_faces = [np.ravel(self.face_index[0][np.array([0, -1])])]
        elif self.dim == 2:
            self.exterior_faces = [
                np.ravel(self.face_index[0][:, np.array([0, -1])]),
                np.ravel(self.face_index[1][np.array([0, -1]), :]),
            ]
        elif self.dim == 3:
            # Extract boundary elements of each axis
            self.exterior_faces = [
                np.sort(
                    np.concatenate(
                        (
                            np.ravel(self.face_index[d][:, np.array([0, -1]), :]),
                            np.ravel(self.face_index[d][:, 1:-1, np.array([0, -1])]),
                        )
                    )
                )
                for d in range(self.dim)
            ]
        else:
            raise NotImplementedError(f"Grid of dimension {self.dim} not implemented.")

        # ! ---- Connectivity ----

        self.connectivity = np.zeros((self.num_faces, 2), dtype=int)
        """np.ndarray: Connectivity (and direction) of faces to cells."""

        if self.dim >= 1:
            self.connectivity[self.faces[0], 0] = np.ravel(self.cell_index[:-1, ...])
            self.connectivity[self.faces[0], 1] = np.ravel(self.cell_index[1:, ...])
        if self.dim >= 2:
            self.connectivity[self.faces[1], 0] = np.ravel(self.cell_index[:, :-1, ...])
            self.connectivity[self.faces[1], 1] = np.ravel(self.cell_index[:, 1:, ...])
        if self.dim >= 3:
            self.connectivity[self.faces[2], 0] = np.ravel(
                self.cell_index[:, :, :-1, ...]
            )
            self.connectivity[self.faces[2], 1] = np.ravel(
                self.cell_index[:, :, 1:, ...]
            )
        if self.dim > 3:
            raise NotImplementedError(f"Grid of dimension {self.dim} not implemented.")

        self.reverse_connectivity = -np.ones((self.dim, self.num_cells, 2), dtype=int)
        """np.ndarray: Reverse connectivity (and direction) of cells to faces."""

        # NOTE: The first components addresses the cell, the second the axis, the third
        # the direction of the relative position of the face wrt the cell (0: left/up,
        # 1: right/down, using matrix indexing in 2d - analogously in 3d).

        if self.dim >= 1:
            self.reverse_connectivity[
                0, np.ravel(self.cell_index[1:, ...]), 0
            ] = self.faces[0]
            self.reverse_connectivity[
                0, np.ravel(self.cell_index[:-1, ...]), 1
            ] = self.faces[0]

        if self.dim >= 2:
            self.reverse_connectivity[
                1, np.ravel(self.cell_index[:, 1:, ...]), 0
            ] = self.faces[1]
            self.reverse_connectivity[
                1, np.ravel(self.cell_index[:, :-1, ...]), 1
            ] = self.faces[1]

        if self.dim >= 3:
            self.reverse_connectivity[
                2, np.ravel(self.cell_index[:, :, 1:, ...]), 0
            ] = self.faces[2]
            self.reverse_connectivity[
                2, np.ravel(self.cell_index[:, :, :-1, ...]), 1
            ] = self.faces[2]

        # Info about inner cells
        # TODO rm?
        self.inner_cells_with_inner_faces = (
            [] + [np.ravel(self.cell_index[1:-1, ...])]
            if self.dim >= 1
            else [] + [np.ravel(self.cell_index[:, 1:-1, ...])]
            if self.dim >= 2
            else [] + [np.ravel(self.cell_index[:, :, 1:-1, ...])]
            if self.dim >= 3
            else []
        )
        """list: Indices of inner cells with inner faces."""


def generate_grid(image: darsia.Image) -> Grid:
    """Get grid object."""
    shape = image.num_voxels
    voxel_size = image.voxel_size
    return Grid(shape, voxel_size)
