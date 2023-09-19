"""Unit tests for the grid module."""

import numpy as np

import darsia


def test_grid_2d():
    grid = darsia.Grid(shape=(4, 5), voxel_size=[0.5, 0.25])

    # Check basic attributes
    assert np.allclose(grid.shape, (4, 5))
    assert grid.dim == 2
    assert np.allclose(grid.voxel_size, [0.5, 0.25])

    # Probe cell numbering
    assert grid.num_cells == 20
    assert grid.cell_index[0, 0] == 0
    assert grid.cell_index[0, 4] == 4
    assert grid.cell_index[3, 0] == 15
    assert grid.cell_index[3, 4] == 19

    # Check face volumes
    assert np.allclose(grid.face_vol, [0.25, 0.5])

    # Check face shape
    assert np.allclose(grid.inner_faces_shape[0], (3, 5))
    assert np.allclose(grid.inner_faces_shape[1], (4, 4))

    # Check face numbering
    assert grid.num_inner_faces[0] == 15
    assert grid.num_inner_faces[1] == 16
    assert grid.num_faces == 15 + 16

    # Check indexing of faces in flat format
    assert np.allclose(grid.flat_inner_faces[0], np.arange(0, 15))
    assert np.allclose(grid.flat_inner_faces[1], np.arange(15, 31))

    # Check indexing of faces in 2d format
    assert np.allclose(grid.inner_faces[0][0], np.arange(0, 5))
    assert np.allclose(grid.inner_faces[0][1], np.arange(5, 10))
    assert np.allclose(grid.inner_faces[0][2], np.arange(10, 15))
    assert np.allclose(grid.inner_faces[1][0], np.arange(15, 19))
    assert np.allclose(grid.inner_faces[1][1], np.arange(19, 23))
    assert np.allclose(grid.inner_faces[1][2], np.arange(23, 27))

    # Check identification of interior inner faces
    assert np.allclose(grid.interior_inner_faces[0], [1, 2, 3, 6, 7, 8, 11, 12, 13])
    assert np.allclose(grid.interior_inner_faces[1], [19, 20, 21, 22, 23, 24, 25, 26])

    # Check identification of exterior inner faces
    assert np.allclose(grid.exterior_inner_faces[0], [0, 4, 5, 9, 10, 14])
    assert np.allclose(grid.exterior_inner_faces[1], [15, 16, 17, 18, 27, 28, 29, 30])

    # Check connectivity: face to cells with positive orientation
    assert np.allclose(grid.connectivity[0], [0, 5])
    assert np.allclose(grid.connectivity[4], [4, 9])
    assert np.allclose(grid.connectivity[10], [10, 15])
    assert np.allclose(grid.connectivity[14], [14, 19])
    assert np.allclose(grid.connectivity[15], [0, 1])
    assert np.allclose(grid.connectivity[18], [3, 4])
    assert np.allclose(grid.connectivity[27], [15, 16])
    assert np.allclose(grid.connectivity[30], [18, 19])

    # Check reverse connectivity: cell to faces with positive orientation
    # For corner cells
    assert np.allclose(grid.reverse_connectivity[0, 0], [-1, 0])
    assert np.allclose(grid.reverse_connectivity[1, 0], [-1, 15])
    assert np.allclose(grid.reverse_connectivity[0, 4], [-1, 4])
    assert np.allclose(grid.reverse_connectivity[1, 4], [18, -1])
    assert np.allclose(grid.reverse_connectivity[0, 15], [10, -1])
    assert np.allclose(grid.reverse_connectivity[1, 15], [-1, 27])
    assert np.allclose(grid.reverse_connectivity[0, 19], [14, -1])
    assert np.allclose(grid.reverse_connectivity[1, 19], [30, -1])

    # For interior cells
    assert np.allclose(grid.reverse_connectivity[0, 6], [1, 6])
    assert np.allclose(grid.reverse_connectivity[1, 6], [19, 20])
