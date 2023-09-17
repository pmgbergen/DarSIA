"""Grid utilities."""

from typing import Union

import numpy as np

import darsia


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
        self.shape = shape
        self.voxel_size = (
            np.array(voxel_size)
            if isinstance(voxel_size, list)
            else voxel_size * np.ones(self.dim)
        )
        assert len(self.voxel_size) == self.dim

        # Define cell and face numbering
        self._setup()

    def _setup(self) -> None:
        """Define cell and face numbering."""

        # ! ---- Grid management ----

        # Define dimensions of the problem and indexing of cells, from here one start
        # counting rows from left to right, from top to bottom.
        num_cells = np.prod(self.shape)
        flat_numbering_cells = np.arange(num_cells, dtype=int)
        numbering_cells = flat_numbering_cells.reshape(self.shape)

        # Consider only inner faces; implicitly define indexing of faces (first
        # vertical, then horizontal). The counting of vertical faces starts from top to
        # bottom and left to right. The counting of horizontal faces starts from left to
        # right and top to bottom.
        vertical_faces_shape = (self.shape[0], self.shape[1] - 1)
        horizontal_faces_shape = (self.shape[0] - 1, self.shape[1])
        num_vertical_faces = np.prod(vertical_faces_shape)
        num_horizontal_faces = np.prod(horizontal_faces_shape)
        num_faces_axis = [
            num_vertical_faces,
            num_horizontal_faces,
        ]
        num_faces = np.sum(num_faces_axis)

        # Define flat indexing of faces: vertical faces first, then horizontal faces
        flat_vertical_faces = np.arange(num_vertical_faces, dtype=int)
        flat_horizontal_faces = num_vertical_faces + np.arange(
            num_horizontal_faces, dtype=int
        )
        vertical_faces = flat_vertical_faces.reshape(vertical_faces_shape)
        horizontal_faces = flat_horizontal_faces.reshape(horizontal_faces_shape)

        # Identify vertical faces on top, inner and bottom
        self.top_row_vertical_faces = np.ravel(vertical_faces[0, :])
        self.inner_vertical_faces = np.ravel(vertical_faces[1:-1, :])
        self.bottom_row_vertical_faces = np.ravel(vertical_faces[-1, :])
        # Identify horizontal faces on left, inner and right
        self.left_col_horizontal_faces = np.ravel(horizontal_faces[:, 0])
        self.inner_horizontal_faces = np.ravel(horizontal_faces[:, 1:-1])
        self.right_col_horizontal_faces = np.ravel(horizontal_faces[:, -1])

        # ! ---- Connectivity ----

        # Define connectivity and direction of the normal on faces
        connectivity = np.zeros((num_faces, 2), dtype=int)
        # Vertical faces to left cells
        connectivity[: num_faces_axis[0], 0] = np.ravel(numbering_cells[:, :-1])
        # Vertical faces to right cells
        connectivity[: num_faces_axis[0], 1] = np.ravel(numbering_cells[:, 1:])
        # Horizontal faces to top cells
        connectivity[num_faces_axis[0] :, 0] = np.ravel(numbering_cells[:-1, :])
        # Horizontal faces to bottom cells
        connectivity[num_faces_axis[0] :, 1] = np.ravel(numbering_cells[1:, :])

        # Define reverse connectivity. Cell to vertical faces
        self.connectivity_cell_to_vertical_face = -np.ones((num_cells, 2), dtype=int)
        # Left vertical face of cell
        self.connectivity_cell_to_vertical_face[
            np.ravel(numbering_cells[:, 1:]), 0
        ] = flat_vertical_faces
        # Right vertical face of cell
        self.connectivity_cell_to_vertical_face[
            np.ravel(numbering_cells[:, :-1]), 1
        ] = flat_vertical_faces
        # Define reverse connectivity. Cell to horizontal faces
        self.connectivity_cell_to_horizontal_face = np.zeros((num_cells, 2), dtype=int)
        # Top horizontal face of cell
        self.connectivity_cell_to_horizontal_face[
            np.ravel(numbering_cells[1:, :]), 0
        ] = flat_horizontal_faces
        # Bottom horizontal face of cell
        self.connectivity_cell_to_horizontal_face[
            np.ravel(numbering_cells[:-1, :]), 1
        ] = flat_horizontal_faces

        # Info about inner cells
        # TODO rm?
        self.inner_cells_with_vertical_faces = np.ravel(numbering_cells[:, 1:-1])
        self.inner_cells_with_horizontal_faces = np.ravel(numbering_cells[1:-1, :])

        # ! ---- Cache ----
        # TODO reduce
        self.num_faces = num_faces
        self.num_cells = num_cells
        self.num_vertical_faces = num_vertical_faces
        self.num_horizontal_faces = num_horizontal_faces
        self.numbering_cells = numbering_cells
        self.num_faces_axis = num_faces_axis
        self.vertical_faces_shape = vertical_faces_shape
        self.horizontal_faces_shape = horizontal_faces_shape
        self.connectivity = connectivity
        self.flat_vertical_faces = flat_vertical_faces
        self.flat_horizontal_faces = flat_horizontal_faces


def generate_grid(image: darsia.Image) -> Grid:
    """Get grid object."""
    shape = image.num_voxels
    voxel_size = image.voxel_size
    return Grid(shape, voxel_size)
