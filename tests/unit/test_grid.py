"""Unit tests for the grid module."""

import numpy as np
import pytest

import darsia


def test_grid_2d():
    """Test basic indexing for 2d tensor grids."""

    # Fetch grid
    grid = darsia.Grid(shape=(3, 4), voxel_size=[0.5, 0.25])

    # Check basic attributes
    assert np.allclose(grid.shape, (3, 4))
    assert grid.dim == 2
    assert np.allclose(grid.voxel_size, [0.5, 0.25])

    # Probe cell numbering
    assert grid.num_cells == 12
    assert grid.cell_index[0, 0] == 0
    assert grid.cell_index[2, 0] == 2
    assert grid.cell_index[0, 3] == 9
    assert grid.cell_index[2, 3] == 11

    # Check face volumes
    assert np.allclose(grid.face_vol, [0.25, 0.5])

    # Check face shape
    assert np.allclose(grid.faces_shape[0], (2, 4))
    assert np.allclose(grid.faces_shape[1], (3, 3))

    # Check face numbering
    assert grid.num_faces_per_axis[0] == 8
    assert grid.num_faces_per_axis[1] == 9
    assert grid.num_faces == 17

    # Check indexing of faces in flat format
    assert np.allclose(grid.faces[0], np.arange(0, 8))
    assert np.allclose(grid.faces[1], np.arange(8, 17))

    # Check indexing of faces in 2d format
    assert np.allclose(grid.face_index[0][:, 0], np.arange(0, 2))
    assert np.allclose(grid.face_index[0][:, 1], np.arange(2, 4))
    assert np.allclose(grid.face_index[0][:, 2], np.arange(4, 6))
    assert np.allclose(grid.face_index[1][:, 0], np.arange(8, 11))
    assert np.allclose(grid.face_index[1][:, 1], np.arange(11, 14))
    assert np.allclose(grid.face_index[1][:, 2], np.arange(14, 17))

    # Check identification of interior inner faces
    assert np.allclose(grid.interior_faces[0], [2, 3, 4, 5])
    assert np.allclose(grid.interior_faces[1], [9, 12, 15])

    # Check identification of exterior inner faces
    assert np.allclose(grid.exterior_faces[0], [0, 1, 6, 7])
    assert np.allclose(grid.exterior_faces[1], [8, 10, 11, 13, 14, 16])


def test_grid_connectivity_2d():
    """Test connectivity for 2d tensor grids."""

    # Fetch grid
    grid = darsia.Grid(shape=(3, 4))

    # Check connectivity: face to cells with positive orientation
    # Horizontal faces
    assert np.allclose(grid.connectivity[0], [0, 1])
    assert np.allclose(grid.connectivity[1], [1, 2])
    assert np.allclose(grid.connectivity[2], [3, 4])
    assert np.allclose(grid.connectivity[3], [4, 5])
    assert np.allclose(grid.connectivity[4], [6, 7])
    assert np.allclose(grid.connectivity[5], [7, 8])
    assert np.allclose(grid.connectivity[6], [9, 10])
    assert np.allclose(grid.connectivity[7], [10, 11])
    # Vertical faces
    assert np.allclose(grid.connectivity[8], [0, 3])
    assert np.allclose(grid.connectivity[9], [1, 4])
    assert np.allclose(grid.connectivity[10], [2, 5])
    assert np.allclose(grid.connectivity[11], [3, 6])
    assert np.allclose(grid.connectivity[12], [4, 7])
    assert np.allclose(grid.connectivity[13], [5, 8])
    assert np.allclose(grid.connectivity[14], [6, 9])
    assert np.allclose(grid.connectivity[15], [7, 10])
    assert np.allclose(grid.connectivity[16], [8, 11])

    # Check reverse connectivity: cell to faces with positive orientation
    # For corner cells
    assert np.allclose(grid.reverse_connectivity[0, 0], [-1, 0])
    assert np.allclose(grid.reverse_connectivity[1, 0], [-1, 8])
    assert np.allclose(grid.reverse_connectivity[0, 2], [1, -1])
    assert np.allclose(grid.reverse_connectivity[1, 2], [-1, 10])
    assert np.allclose(grid.reverse_connectivity[0, 9], [-1, 6])
    assert np.allclose(grid.reverse_connectivity[1, 9], [14, -1])
    assert np.allclose(grid.reverse_connectivity[0, 11], [7, -1])
    assert np.allclose(grid.reverse_connectivity[1, 11], [16, -1])

    # For interior cells
    assert np.allclose(grid.reverse_connectivity[0, 4], [2, 3])
    assert np.allclose(grid.reverse_connectivity[1, 4], [9, 12])


def test_grid_3d():
    """Test basic indexing for 2d tensor grids."""

    # Fetch grid
    grid = darsia.Grid(shape=(3, 4, 5), voxel_size=[0.5, 0.25, 2])

    # Check basic attributes
    assert np.allclose(grid.shape, (3, 4, 5))
    assert grid.dim == 3
    assert np.allclose(grid.voxel_size, [0.5, 0.25, 2])

    # Probe cell numbering
    assert grid.num_cells == 60
    assert grid.cell_index[0, 0, 0] == 0
    assert grid.cell_index[2, 0, 0] == 2
    assert grid.cell_index[0, 3, 0] == 9
    assert grid.cell_index[2, 3, 0] == 11
    assert grid.cell_index[0, 0, 4] == 48
    assert grid.cell_index[2, 0, 4] == 50
    assert grid.cell_index[0, 3, 4] == 57
    assert grid.cell_index[2, 3, 4] == 59

    # Check face volumes
    assert np.allclose(grid.face_vol, [0.5, 1, 0.125])

    # Check face shape
    assert np.allclose(grid.faces_shape[0], (2, 4, 5))
    assert np.allclose(grid.faces_shape[1], (3, 3, 5))
    assert np.allclose(grid.faces_shape[2], (3, 4, 4))

    # Check face numbering
    assert grid.num_faces_per_axis[0] == 40
    assert grid.num_faces_per_axis[1] == 45
    assert grid.num_faces_per_axis[2] == 48
    assert grid.num_faces == 40 + 45 + 48

    # Check indexing of faces in flat format
    assert np.allclose(grid.faces[0], np.arange(0, 40))
    assert np.allclose(grid.faces[1], np.arange(40, 40 + 45))
    assert np.allclose(grid.faces[2], np.arange(40 + 45, 40 + 45 + 48))

    # Check indexing of faces in 2d format
    assert np.allclose(grid.face_index[0][:, 0, 0], np.arange(0, 2))
    assert np.allclose(grid.face_index[0][:, 1, 0], np.arange(2, 4))
    assert np.allclose(grid.face_index[0][:, 2, 0], np.arange(4, 6))
    assert np.allclose(grid.face_index[0][:, 0, 4], np.arange(32, 34))
    assert np.allclose(grid.face_index[0][:, 1, 4], np.arange(34, 36))
    assert np.allclose(grid.face_index[0][:, 2, 4], np.arange(36, 38))
    assert np.allclose(grid.face_index[1][:, 0, 0], np.arange(40, 43))
    assert np.allclose(grid.face_index[1][:, 1, 0], np.arange(43, 46))
    assert np.allclose(grid.face_index[1][:, 2, 0], np.arange(46, 49))

    # Check identification of interior inner faces
    assert np.allclose(
        grid.interior_faces[0],
        [
            10,
            11,
            12,
            13,
            18,
            19,
            20,
            21,
            26,
            27,
            28,
            29,
        ],
    )
    assert np.allclose(grid.interior_faces[1], [50, 53, 56, 59, 62, 65, 68, 71, 74])
    assert np.allclose(grid.interior_faces[2], [89, 92, 101, 104, 113, 116, 125, 128])

    # Check identification of exterior inner faces
    for d in range(grid.dim):
        assert np.allclose(
            np.sort(grid.exterior_faces[d]),
            np.sort(
                np.array(
                    list(
                        set(grid.faces[d].tolist())
                        - set(grid.interior_faces[d].tolist())
                    )
                )
            ),
        )


def test_grid_connectivity_3d():
    """Test connectivity for 3d tensor grids."""

    # Fetch grid
    grid = darsia.Grid(shape=(3, 4, 5), voxel_size=[0.5, 0.25, 2])

    # Check connectivity: face to cells with positive orientation
    # Horizontal faces
    assert np.allclose(grid.connectivity[0], [0, 1])
    assert np.allclose(grid.connectivity[1], [1, 2])
    assert np.allclose(grid.connectivity[2], [3, 4])
    assert np.allclose(grid.connectivity[3], [4, 5])
    assert np.allclose(grid.connectivity[4], [6, 7])
    assert np.allclose(grid.connectivity[5], [7, 8])
    assert np.allclose(grid.connectivity[6], [9, 10])
    assert np.allclose(grid.connectivity[7], [10, 11])
    # Vertical faces
    assert np.allclose(grid.connectivity[40], [0, 3])
    assert np.allclose(grid.connectivity[41], [1, 4])
    assert np.allclose(grid.connectivity[42], [2, 5])
    assert np.allclose(grid.connectivity[43], [3, 6])
    assert np.allclose(grid.connectivity[44], [4, 7])
    assert np.allclose(grid.connectivity[45], [5, 8])
    assert np.allclose(grid.connectivity[46], [6, 9])
    assert np.allclose(grid.connectivity[47], [7, 10])
    assert np.allclose(grid.connectivity[48], [8, 11])
    # Table faces
    assert np.allclose(grid.connectivity[85], [0, 12])
    assert np.allclose(grid.connectivity[86], [1, 13])
    assert np.allclose(grid.connectivity[87], [2, 14])
    assert np.allclose(grid.connectivity[88], [3, 15])
    assert np.allclose(grid.connectivity[89], [4, 16])
    assert np.allclose(grid.connectivity[90], [5, 17])
    assert np.allclose(grid.connectivity[91], [6, 18])
    assert np.allclose(grid.connectivity[92], [7, 19])
    assert np.allclose(grid.connectivity[93], [8, 20])

    # Check reverse connectivity: cell to faces with positive orientation
    # For corner cells
    assert np.allclose(grid.reverse_connectivity[0, 0], [-1, 0])
    assert np.allclose(grid.reverse_connectivity[1, 0], [-1, 40])
    assert np.allclose(grid.reverse_connectivity[2, 0], [-1, 85])
    assert np.allclose(grid.reverse_connectivity[0, 9], [-1, 6])
    assert np.allclose(grid.reverse_connectivity[1, 9], [46, -1])
    assert np.allclose(grid.reverse_connectivity[2, 9], [-1, 94])
    assert np.allclose(grid.reverse_connectivity[0, 57], [-1, 38])
    assert np.allclose(grid.reverse_connectivity[1, 57], [82, -1])
    assert np.allclose(grid.reverse_connectivity[2, 57], [130, -1])
    assert np.allclose(grid.reverse_connectivity[0, 59], [39, -1])
    assert np.allclose(grid.reverse_connectivity[1, 59], [84, -1])
    assert np.allclose(grid.reverse_connectivity[2, 59], [132, -1])

    # For interior cells
    assert np.allclose(grid.reverse_connectivity[0, 16], [10, 11])
    assert np.allclose(grid.reverse_connectivity[1, 16], [50, 53])
    assert np.allclose(grid.reverse_connectivity[2, 16], [89, 101])


@pytest.mark.parametrize("shape", [(3, 4), (3, 4, 5)])
def test_compatibility(shape):
    """Compatibility of connectivity, reverse connectivity and interior faces.

    Make sure that inner faces really only connect to two cells which have inner faces
    in all directions again.

    """
    grid = darsia.Grid(shape=shape)
    assert (
        np.count_nonzero(
            np.concatenate(
                [
                    np.ravel(
                        grid.reverse_connectivity[
                            d_perp,
                            np.ravel(grid.connectivity[grid.interior_faces[d]]),
                        ]
                    )
                    for d in range(grid.dim)
                    for d_perp in np.delete(range(grid.dim), d)
                ]
            )
            == -1
        )
        == 0
    )
