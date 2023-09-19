"""Unit tests for finite volume utilities."""

import numpy as np

import darsia


def test_divergence_2d():
    # Create divergence matrix
    grid = darsia.Grid(shape=(4, 5), voxel_size=[0.5, 0.25])
    divergence = darsia.FVDivergence(grid).mat.todense()

    # Check shape
    assert np.allclose(divergence.shape, (grid.num_cells, grid.num_faces))

    # Check values in corner cells
    assert np.allclose(np.nonzero(divergence[0])[1], [0, 15])
    assert np.allclose(divergence[0, np.array([0, 15])], [0.25, 0.5])

    assert np.allclose(np.nonzero(divergence[4])[1], [3, 15, 19])
    assert np.allclose(divergence[4, np.array([3, 15, 19])], [0.25, -0.5, 0.5])

    assert np.allclose(np.nonzero(divergence[16])[1], [12, 27])
    assert np.allclose(divergence[16, np.array([12, 27])], [0.25, -0.5])

    assert np.allclose(np.nonzero(divergence[19])[1], [14, 30])
    assert np.allclose(divergence[19, np.array([14, 30])], [-0.25, -0.5])

    # Check value for interior cell
    assert np.allclose(np.nonzero(divergence[6])[1], [4, 5, 17, 21])
    assert np.allclose(
        divergence[6, np.array([4, 5, 17, 21])], [-0.25, 0.25, -0.5, 0.5]
    )


def test_divergence_3d():
    # Create divergence matrix
    grid = darsia.Grid(shape=(3, 4, 5), voxel_size=[0.5, 0.25, 2])
    divergence = darsia.FVDivergence(grid).mat.todense()

    # Check shape
    assert np.allclose(divergence.shape, (grid.num_cells, grid.num_faces))

    # Check values in corner cells
    assert np.allclose(np.nonzero(divergence[0])[1], [0, 40, 85])
    assert np.allclose(divergence[0, np.array([0, 40, 85])], [0.5, 1, 0.125])

    assert np.allclose(np.nonzero(divergence[11])[1], [7, 48, 96])
    assert np.allclose(divergence[11, np.array([7, 48, 96])], [-0.5, -1, 0.125])

    assert np.allclose(np.nonzero(divergence[59])[1], [39, 84, 132])
    assert np.allclose(divergence[59, np.array([39, 84, 132])], [-0.5, -1, -0.125])

    # Check value for interior cell
    assert np.allclose(np.nonzero(divergence[16])[1], [10, 11, 50, 53, 89, 101])
    assert np.allclose(
        divergence[16, np.array([10, 11, 50, 53, 89, 101])],
        [-0.5, 0.5, -1, 1, -0.125, 0.125],
    )


def test_mass_2d():
    grid = darsia.Grid(shape=(4, 5), voxel_size=[0.5, 0.25])
    mass = darsia.FVMass(grid).mat.todense()

    # Check shape
    assert np.allclose(mass.shape, (grid.num_cells, grid.num_cells))

    # Check diagonal structure
    assert np.linalg.norm(mass - np.diag(np.diag(mass))) < 1e-10

    # Check diagonal values
    assert len(np.unique(np.diag(mass))) == 1
    assert np.isclose(mass[0, 0], 0.125)


def test_mass_3d():
    grid = darsia.Grid(shape=(3, 4, 5), voxel_size=[0.5, 0.25, 2])
    mass = darsia.FVMass(grid).mat.todense()

    # Check shape
    assert np.allclose(mass.shape, (grid.num_cells, grid.num_cells))

    # Check diagonal structure
    assert np.linalg.norm(mass - np.diag(np.diag(mass))) < 1e-10

    # Check diagonal values
    assert len(np.unique(np.diag(mass))) == 1
    assert np.isclose(mass[0, 0], 0.25)


def test_mass_face_2d():
    grid = darsia.Grid(shape=(4, 5), voxel_size=[0.5, 0.25])
    mass = darsia.FVMass(grid, mode="faces", lumping=True).mat.todense()

    # Check shape
    assert np.allclose(mass.shape, (grid.num_faces, grid.num_faces))

    # Check diagonal structure
    assert np.linalg.norm(mass - np.diag(np.diag(mass))) < 1e-10

    # Check diagonal values
    assert len(np.unique(np.diag(mass))) == 1
    assert np.isclose(mass[0, 0], 0.5 * 0.125)


def test_mass_face_3d():
    grid = darsia.Grid(shape=(3, 4, 5), voxel_size=[0.5, 0.25, 2])
    mass = darsia.FVMass(grid, mode="faces", lumping=True).mat.todense()

    # Check shape
    assert np.allclose(mass.shape, (grid.num_faces, grid.num_faces))

    # Check diagonal structure
    assert np.linalg.norm(mass - np.diag(np.diag(mass))) < 1e-10

    # Check diagonal values
    assert len(np.unique(np.diag(mass))) == 1
    assert np.isclose(mass[0, 0], 0.5 * 0.25)


def test_tangential_reconstruction_2d_1():
    grid = darsia.Grid(shape=(2, 3), voxel_size=[0.5, 0.25])
    tangential_reconstruction = darsia.FVTangentialReconstruction(grid).mat.todense()

    # Check shape
    assert np.allclose(
        tangential_reconstruction.shape, (grid.num_faces, grid.num_faces)
    )

    # Check values - first for exterior faces
    assert np.allclose(tangential_reconstruction[0], [0, 0, 0, 0.25, 0.25, 0, 0])
    assert np.allclose(tangential_reconstruction[4], [0.25, 0.25, 0, 0, 0, 0, 0])

    # Check values - then for interior faces
    assert np.allclose(tangential_reconstruction[1], [0, 0, 0, 0.25, 0.25, 0.25, 0.25])


def test_tangential_reconstruction_2d_2():
    grid = darsia.Grid(shape=(3, 4), voxel_size=[0.5, 0.25])
    tangential_reconstruction = darsia.FVTangentialReconstruction(grid).mat.todense()

    # Check shape
    assert np.allclose(
        tangential_reconstruction.shape, (grid.num_faces, grid.num_faces)
    )

    # Check values - first for exterior faces
    assert np.allclose(np.nonzero(tangential_reconstruction[0])[1], [8, 9])
    assert np.allclose(np.nonzero(tangential_reconstruction[7])[1], [15, 16])

    # Check values - then for interior faces
    assert np.allclose(np.nonzero(tangential_reconstruction[2])[1], [8, 9, 11, 12])
    assert np.allclose(np.nonzero(tangential_reconstruction[15])[1], [4, 5, 6, 7])

    # Apply once and prove values
    tangential_reconstruction_sparse = darsia.FVTangentialReconstruction(grid).mat
    normal_flux = np.arange(grid.num_faces)
    tangential_flux = tangential_reconstruction_sparse.dot(normal_flux)
    assert np.allclose(
        tangential_flux[np.array([0, 1, 4, 8, 12, 16])], [4.25, 4.75, 13, 0.5, 3.5, 3]
    )


def test_face_to_cell_2d():
    grid = darsia.Grid(shape=(3, 4), voxel_size=[0.5, 0.25])
    num_faces = grid.num_faces
    flat_flux = np.arange(num_faces)
    cell_flux = darsia.face_to_cell(grid, flat_flux)

    # Check shape
    assert np.allclose(cell_flux.shape, (*grid.shape, grid.dim))

    # Check values
    print(cell_flux)
    assert np.allclose(cell_flux[0, 0], [0, 4])
    assert np.allclose(cell_flux[2, 3], [3.5, 8])
    assert np.allclose(cell_flux[1, 1], [2.5, 10.5])


def test_face_to_cell_3d():
    grid = darsia.Grid(shape=(3, 4, 5), voxel_size=[0.5, 0.25, 2])
    num_faces = grid.num_faces
    flat_flux = np.arange(num_faces)
    cell_flux = darsia.face_to_cell(grid, flat_flux)

    # Check shape
    assert np.allclose(cell_flux.shape, (*grid.shape, grid.dim))

    # Check values
    assert np.allclose(cell_flux[0, 0, 0], [0, 20, 42.5])
    assert np.allclose(cell_flux[2, 3, 4], [19.5, 42, 66])
    assert np.allclose(cell_flux[1, 1, 1], [10.5, 51.5, 95])
