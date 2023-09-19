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
    assert grid.num_faces_per_axis[0] == 15
    assert grid.num_faces_per_axis[1] == 16
    assert grid.num_faces == 15 + 16

    # Check indexing of faces in flat format
    assert np.allclose(grid.faces[0], np.arange(0, 15))
    assert np.allclose(grid.faces[1], np.arange(15, 31))

    # Check indexing of faces in 2d format
    assert np.allclose(grid.face_index[0][0], np.arange(0, 5))
    assert np.allclose(grid.face_index[0][1], np.arange(5, 10))
    assert np.allclose(grid.face_index[0][2], np.arange(10, 15))
    assert np.allclose(grid.face_index[1][0], np.arange(15, 19))
    assert np.allclose(grid.face_index[1][1], np.arange(19, 23))
    assert np.allclose(grid.face_index[1][2], np.arange(23, 27))

    # Check identification of interior inner faces
    assert np.allclose(grid.interior_faces[0], [1, 2, 3, 6, 7, 8, 11, 12, 13])
    assert np.allclose(grid.interior_faces[1], [19, 20, 21, 22, 23, 24, 25, 26])

    # Check identification of exterior inner faces
    assert np.allclose(grid.exterior_faces[0], [0, 4, 5, 9, 10, 14])
    assert np.allclose(grid.exterior_faces[1], [15, 16, 17, 18, 27, 28, 29, 30])

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


def test_grid_3d():
    grid = darsia.Grid(shape=(3, 4, 5), voxel_size=[0.5, 0.25, 2])

    # Check basic attributes
    assert np.allclose(grid.shape, (3, 4, 5))
    assert grid.dim == 3
    assert np.allclose(grid.voxel_size, [0.5, 0.25, 2])

    # Probe cell numbering
    assert grid.num_cells == 60
    assert grid.cell_index[0, 0, 0] == 0
    assert grid.cell_index[2, 0, 0] == 40
    assert grid.cell_index[0, 3, 0] == 15
    assert grid.cell_index[2, 3, 0] == 55
    assert grid.cell_index[0, 0, 4] == 4
    assert grid.cell_index[2, 0, 4] == 44
    assert grid.cell_index[0, 3, 4] == 19
    assert grid.cell_index[2, 3, 4] == 59

    # Check face volumes
    assert np.allclose(grid.face_vol, [0.5, 1, 0.5 * 0.25])

    # Check face shape
    assert np.allclose(grid.inner_faces_shape[0], (2, 4, 5))
    assert np.allclose(grid.inner_faces_shape[1], (3, 3, 5))
    assert np.allclose(grid.inner_faces_shape[2], (4, 3, 4))

    # Check face numbering
    assert grid.num_faces_per_axis[0] == 40
    assert grid.num_faces_per_axis[1] == 45
    assert grid.num_faces_per_axis[2] == 48
    assert grid.num_faces == 40 + 45 + 48

    # Check indexing of faces in flat format
    assert np.allclose(grid.faces[0], np.arange(0, 40))
    assert np.allclose(grid.faces[1], np.arange(40, 40 + 45))
    assert np.allclose(grid.faces[2], np.arange(40 + 45, 40 + 45 + 48))

    # Check indexing of faces in 3d format
    assert np.allclose(grid.face_index[0][0, 0], np.arange(0, 5))
    assert np.allclose(grid.face_index[0][1, 0], np.arange(20, 25))
    assert np.allclose(grid.face_index[0][0, 1], np.arange(5, 10))
    assert np.allclose(grid.face_index[0][1, 1], np.arange(25, 30))
    # ...
    assert np.allclose(grid.face_index[1][0, 0], np.arange(40, 45))
    assert np.allclose(grid.face_index[1][1, 0], np.arange(55, 60))
    assert np.allclose(grid.face_index[1][2, 0], np.arange(70, 75))
    assert np.allclose(grid.face_index[1][0, 1], np.arange(45, 50))
    # ...
    assert np.allclose(grid.face_index[2][0, 0], np.arange(85, 89))
    assert np.allclose(grid.face_index[2][1, 0], np.arange(97, 101))
    assert np.allclose(grid.face_index[2][2, 0], np.arange(109, 113))
    assert np.allclose(grid.face_index[2][0, 1], np.arange(89, 93))
    # ...

    # Check identification of interior inner faces
    assert np.allclose(
        grid.interior_faces[0], [6, 7, 8, 11, 12, 13, 26, 27, 28, 31, 32, 33]
    )
    assert np.allclose(grid.interior_faces[1], [46, 47, 48, 61, 62, 63, 76, 77, 78])
    assert np.allclose(grid.interior_faces[2], [90, 91, 102, 103, 114, 115, 126, 127])

    # Check identification of exterior inner faces
    assert np.allclose(
        grid.exterior_faces[0],
        [
            0,
            1,
            2,
            3,
            4,
            5,
            9,
            10,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            29,
            30,
            34,
            35,
            36,
            37,
            38,
            39,
        ],
    )
    assert np.allclose(
        grid.exterior_faces[1],
        [
            40,
            41,
            42,
            43,
            44,
            45,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            79,
            80,
            81,
            82,
            83,
            84,
        ],
    )
    assert np.allclose(
        grid.exterior_faces[2],
        [
            85,
            86,
            87,
            88,
            89,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
            123,
            124,
            125,
            128,
            129,
            130,
            131,
            132,
        ],
    )

    # Check connectivity: face to cells with positive orientation
    # across 0-normal:
    assert np.allclose(grid.connectivity[0], [0, 20])
    assert np.allclose(grid.connectivity[39], [39, 59])
    # across 1-normal:
    assert np.allclose(grid.connectivity[40], [0, 5])
    assert np.allclose(grid.connectivity[84], [54, 59])
    # across 2-normal:
    assert np.allclose(grid.connectivity[85], [0, 1])
    assert np.allclose(grid.connectivity[132], [58, 59])

    # Check reverse connectivity: cell to faces with positive orientation
    # For corner cells
    assert np.allclose(grid.reverse_connectivity[0, 0], [-1, 0])
    assert np.allclose(grid.reverse_connectivity[1, 0], [-1, 40])
    assert np.allclose(grid.reverse_connectivity[2, 0], [-1, 85])
    assert np.allclose(grid.reverse_connectivity[0, 19], [-1, 19])
    assert np.allclose(grid.reverse_connectivity[1, 19], [54, -1])
    assert np.allclose(grid.reverse_connectivity[2, 19], [100, -1])
    assert np.allclose(grid.reverse_connectivity[0, 59], [39, -1])
    assert np.allclose(grid.reverse_connectivity[1, 59], [84, -1])
    assert np.allclose(grid.reverse_connectivity[2, 59], [132, -1])

    # For interior cells
    assert np.allclose(grid.reverse_connectivity[0, 26], [6, 26])
    assert np.allclose(grid.reverse_connectivity[1, 26], [56, 61])
    assert np.allclose(grid.reverse_connectivity[2, 26], [105, 106])
