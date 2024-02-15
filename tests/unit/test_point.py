"""Unit tests for Point class, i.e., Voxel, Coordinate etc.."""

import numpy as np

import darsia


def test_make_coordinate():
    # Single coordinate - Coordinate
    coord = darsia.Coordinate([1.0, 2.0])
    assert isinstance(coord, darsia.Coordinate)
    assert np.allclose(coord, [1.0, 2.0])

    # Single coordinate - make_coordinate
    coord = darsia.make_coordinate([1.0, 2.0])
    assert isinstance(coord, darsia.Coordinate)
    assert np.allclose(coord, [1.0, 2.0])

    # Multiple coordinates - CoordinateArray
    coords = darsia.CoordinateArray([[1.0, 2.0], [3.0, 4.0]])
    assert isinstance(coords, darsia.CoordinateArray)
    assert np.allclose(coords, [[1.0, 2.0], [3.0, 4.0]])

    # Multiple coordinates - make_coordinate
    coords = darsia.make_coordinate([[1.0, 2.0], [3.0, 4.0]])
    assert isinstance(coords, darsia.CoordinateArray)
    assert np.allclose(coords, [[1.0, 2.0], [3.0, 4.0]])


def test_make_voxel():
    # Single voxel - Voxel
    voxel = darsia.Voxel([1, 2])
    assert isinstance(voxel, darsia.Voxel)
    assert np.allclose(voxel, [1, 2])

    # Single voxel - make_voxel
    voxel = darsia.make_voxel([1, 2])
    assert isinstance(voxel, darsia.Voxel)
    assert np.allclose(voxel, [1, 2])

    # Multiple voxels - VoxelArray
    voxels = darsia.VoxelArray([[1, 2], [3.2, 4.8]])
    assert isinstance(voxels, darsia.VoxelArray)
    assert np.allclose(voxels, [[1, 2], [3, 4]])

    # Multiple voxels - make_voxel
    voxels = darsia.make_voxel([[1, 2], [3.2, 4.8]])
    assert isinstance(voxels, darsia.VoxelArray)
    assert np.allclose(voxels, [[1, 2], [3, 4]])


def test_make_voxel_center():
    # Single voxel center - VoxelCenter
    voxel_center = darsia.VoxelCenter([1, 2])
    assert isinstance(voxel_center, darsia.VoxelCenter)
    assert np.allclose(voxel_center, [1.5, 2.5])

    # Single voxel center - make_voxel_center
    voxel_center = darsia.make_voxel_center([1, 2])
    assert isinstance(voxel_center, darsia.VoxelCenter)
    assert np.allclose(voxel_center, [1.5, 2.5])

    # Multiple voxel centers - VoxelCenterArray
    voxel_centers = darsia.VoxelCenterArray([[1, 2], [3.2, 4.8]])
    assert isinstance(voxel_centers, darsia.VoxelCenterArray)
    assert np.allclose(voxel_centers, [[1.5, 2.5], [3.5, 4.5]])

    # Multiple voxel centers - make_voxel_center
    voxel_centers = darsia.make_voxel_center([[1, 2], [3.2, 4.8]])
    assert isinstance(voxel_centers, darsia.VoxelCenterArray)
    assert np.allclose(voxel_centers, [[1.5, 2.5], [3.5, 4.5]])


def test_make_voxel_reverse_matrix_indexing():
    # Single voxel - Voxel
    voxel = darsia.Voxel([1, 2], matrix_indexing=False)
    assert isinstance(voxel, darsia.Voxel)
    assert np.allclose(voxel, [2, 1])

    # Single voxel - make_voxel
    voxel = darsia.make_voxel([1, 2], matrix_indexing=False)
    assert isinstance(voxel, darsia.Voxel)
    assert np.allclose(voxel, [2, 1])

    # Multiple voxels - VoxelArray
    voxels = darsia.VoxelArray([[1, 2], [3.2, 4.8]], matrix_indexing=False)
    assert isinstance(voxels, darsia.VoxelArray)
    assert np.allclose(voxels, [[2, 1], [4, 3]])

    # Multiple voxels - make_voxel
    voxels = darsia.make_voxel([[1, 2], [3.2, 4.8]], matrix_indexing=False)
    assert isinstance(voxels, darsia.VoxelArray)
    assert np.allclose(voxels, [[2, 1], [4, 3]])


def test_coordinatearray_getitem():
    # Define CoordinateArray
    coords = darsia.CoordinateArray([[1.0, 2.0], [3.0, 4.0]])

    # Getitem - single coordinate
    coord = coords[0]
    assert isinstance(coord, darsia.Coordinate)
    assert np.allclose(coord, [1.0, 2.0])

    # Iterator
    for coord in coords:
        assert isinstance(coord, darsia.Coordinate)


def test_voxelarray_getitem():
    # Define VoxelArray
    voxels = darsia.VoxelArray([[1, 2], [3, 4]])

    # Getitem - single voxel
    voxel = voxels[0]
    assert isinstance(voxel, darsia.Voxel)
    assert np.allclose(voxel, [1, 2])

    # Iterator
    for voxel in voxels:
        assert isinstance(voxel, darsia.Voxel)


def text_voxelcenterarray_getitem():
    # Define VoxelCenterArray
    voxel_centers = darsia.VoxelCenterArray([[1, 2], [3, 4]])

    # Getitem - single voxel center
    voxel_center = voxel_centers[0]
    assert isinstance(voxel_center, darsia.VoxelCenter)
    assert np.allclose(voxel_center, [1.5, 2.5])

    # Iterator
    for voxel_center in voxel_centers:
        assert isinstance(voxel_center, darsia.VoxelCenter)


def test_point_to_coordinate():
    # Define reference image
    image = darsia.Image(
        img=np.zeros((3, 4), dtype=float), dimensions=[1, 1], space_dim=2
    )

    # From Coordinate
    coord = darsia.Coordinate([0.6, 0.2])
    to_coord = coord.to_coordinate(image.coordinatesystem)
    assert isinstance(to_coord, darsia.Coordinate)
    assert np.allclose(to_coord, [0.6, 0.2])

    to_coord = coord.to(darsia.Coordinate)
    assert isinstance(to_coord, darsia.Coordinate)
    assert np.allclose(to_coord, [0.6, 0.2])

    # From Voxel
    voxel = darsia.Voxel([2, 1])
    to_coord = voxel.to_coordinate(image.coordinatesystem)
    assert isinstance(to_coord, darsia.Coordinate)
    assert np.allclose(to_coord, [0.25, 1 / 3])

    to_coord = voxel.to(darsia.Coordinate, image.coordinatesystem)
    assert isinstance(to_coord, darsia.Coordinate)
    assert np.allclose(to_coord, [0.25, 1 / 3])

    # From VoxelCenter
    voxel_center = darsia.VoxelCenter([2, 1])
    to_coord = voxel_center.to_coordinate(image.coordinatesystem)
    assert isinstance(to_coord, darsia.Coordinate)
    assert np.allclose(to_coord, [0.375, 1 / 6])

    to_coord = voxel_center.to(darsia.Coordinate, image.coordinatesystem)
    assert isinstance(to_coord, darsia.Coordinate)
    assert np.allclose(to_coord, [0.375, 1 / 6])


def test_point_to_voxel():
    # Define reference image
    image = darsia.Image(
        img=np.zeros((3, 4), dtype=float), dimensions=[1, 1], space_dim=2
    )

    # From Coordinate
    coord = darsia.Coordinate([0.6, 0.2])
    to_voxel = coord.to_voxel(image.coordinatesystem)
    assert isinstance(to_voxel, darsia.Voxel)
    assert np.allclose(to_voxel, [2, 2])

    to_voxel = coord.to(darsia.Voxel, image.coordinatesystem)
    assert isinstance(to_voxel, darsia.Voxel)
    assert np.allclose(to_voxel, [2, 2])

    # From Voxel
    voxel = darsia.Voxel([2, 1])
    to_voxel = voxel.to_voxel()
    assert isinstance(to_voxel, darsia.Voxel)
    assert np.allclose(to_voxel, [2, 1])

    to_voxel = voxel.to(darsia.Voxel)
    assert isinstance(to_voxel, darsia.Voxel)
    assert np.allclose(to_voxel, [2, 1])

    # From VoxelCenter
    voxel_center = darsia.VoxelCenter([2, 1])
    to_voxel = voxel_center.to_voxel()
    assert isinstance(to_voxel, darsia.Voxel)
    assert np.allclose(to_voxel, [2, 1])

    to_voxel = voxel_center.to(darsia.Voxel)
    assert isinstance(to_voxel, darsia.Voxel)
    assert np.allclose(to_voxel, [2, 1])


def test_point_to_voxel_center():
    # Define reference image
    image = darsia.Image(
        img=np.zeros((3, 4), dtype=float), dimensions=[1, 1], space_dim=2
    )

    # From Coordinate
    coord = darsia.Coordinate([0.6, 0.2])
    to_voxel_center = coord.to_voxel_center(image.coordinatesystem)
    assert isinstance(to_voxel_center, darsia.VoxelCenter)
    assert np.allclose(to_voxel_center, [2.5, 2.5])

    to_voxel_center = coord.to(darsia.VoxelCenter, image.coordinatesystem)
    assert isinstance(to_voxel_center, darsia.VoxelCenter)
    assert np.allclose(to_voxel_center, [2.5, 2.5])

    # From Voxel
    voxel = darsia.Voxel([2, 1])
    to_voxel_center = voxel.to_voxel_center()
    assert isinstance(to_voxel_center, darsia.VoxelCenter)
    assert np.allclose(to_voxel_center, [2.5, 1.5])

    to_voxel_center = voxel.to(darsia.VoxelCenter)
    assert isinstance(to_voxel_center, darsia.VoxelCenter)
    assert np.allclose(to_voxel_center, [2.5, 1.5])

    # From VoxelCenter
    voxel_center = darsia.VoxelCenter([2, 1])
    to_voxel_center = voxel_center.to_voxel_center()
    assert isinstance(to_voxel_center, darsia.VoxelCenter)
    assert np.allclose(to_voxel_center, [2.5, 1.5])

    to_voxel_center = voxel_center.to(darsia.VoxelCenter)
    assert isinstance(to_voxel_center, darsia.VoxelCenter)
    assert np.allclose(to_voxel_center, [2.5, 1.5])


def test_point_array_to_coordinate():
    # Define reference image
    image = darsia.Image(
        img=np.zeros((3, 4), dtype=float), dimensions=[1, 1], space_dim=2
    )

    # From CoordinateArray
    coords = darsia.CoordinateArray([[0.6, 0.2], [0.1, 0.9]])
    to_coords = coords.to_coordinate(image.coordinatesystem)
    assert isinstance(to_coords, darsia.CoordinateArray)
    assert np.allclose(to_coords, [[0.6, 0.2], [0.1, 0.9]])

    to_coords = coords.to(darsia.CoordinateArray)
    assert isinstance(to_coords, darsia.CoordinateArray)
    assert np.allclose(to_coords, [[0.6, 0.2], [0.1, 0.9]])

    # From VoxelArray
    voxels = darsia.VoxelArray([[2, 1], [1, 3]])
    to_coords = voxels.to_coordinate(image.coordinatesystem)
    assert isinstance(to_coords, darsia.CoordinateArray)
    assert np.allclose(to_coords, [[0.25, 1 / 3], [0.75, 2 / 3]])

    to_coords = voxels.to(darsia.CoordinateArray, image.coordinatesystem)
    assert isinstance(to_coords, darsia.CoordinateArray)
    assert np.allclose(to_coords, [[0.25, 1 / 3], [0.75, 2 / 3]])

    # From VoxelCenterArray
    voxel_centers = darsia.VoxelCenterArray([[2, 1], [1, 3]])
    to_coords = voxel_centers.to_coordinate(image.coordinatesystem)
    assert isinstance(to_coords, darsia.CoordinateArray)
    assert np.allclose(to_coords, [[0.375, 1 / 6], [0.875, 3 / 6]])

    to_coords = voxel_centers.to(darsia.CoordinateArray, image.coordinatesystem)
    assert isinstance(to_coords, darsia.CoordinateArray)
    assert np.allclose(to_coords, [[0.375, 1 / 6], [0.875, 3 / 6]])


def test_point_array_to_voxel():
    # Define reference image
    image = darsia.Image(
        img=np.zeros((3, 4), dtype=float), dimensions=[1, 1], space_dim=2
    )

    # From CoordinateArray
    coords = darsia.CoordinateArray([[0.6, 0.2], [0.1, 0.9]])
    to_voxels = coords.to_voxel(image.coordinatesystem)
    assert isinstance(to_voxels, darsia.VoxelArray)
    assert np.allclose(to_voxels, [[2, 2], [0, 0]])

    to_voxels = coords.to(darsia.VoxelArray, image.coordinatesystem)
    assert isinstance(to_voxels, darsia.VoxelArray)
    assert np.allclose(to_voxels, [[2, 2], [0, 0]])

    # From VoxelArray
    voxels = darsia.VoxelArray([[2, 1], [1, 3]])
    to_voxels = voxels.to_voxel(image.coordinatesystem)
    assert isinstance(to_voxels, darsia.VoxelArray)
    assert np.allclose(to_voxels, [[2, 1], [1, 3]])

    to_voxels = voxels.to(darsia.VoxelArray)
    assert isinstance(to_voxels, darsia.VoxelArray)
    assert np.allclose(to_voxels, [[2, 1], [1, 3]])

    # From VoxelCenterArray
    voxel_centers = darsia.VoxelCenterArray([[2, 1], [1, 3]])
    to_voxels = voxel_centers.to_voxel(image.coordinatesystem)
    assert isinstance(to_voxels, darsia.VoxelArray)
    assert np.allclose(to_voxels, [[2, 1], [1, 3]])

    to_voxels = voxel_centers.to(darsia.VoxelArray)
    assert isinstance(to_voxels, darsia.VoxelArray)
    assert np.allclose(to_voxels, [[2, 1], [1, 3]])


def test_point_array_to_voxel_center():
    # Define reference image
    image = darsia.Image(
        img=np.zeros((3, 4), dtype=float), dimensions=[1, 1], space_dim=2
    )

    # From CoordinateArray
    coords = darsia.CoordinateArray([[0.6, 0.2], [0.1, 0.9]])
    to_voxel_centers = coords.to_voxel_center(image.coordinatesystem)
    assert isinstance(to_voxel_centers, darsia.VoxelCenterArray)
    assert np.allclose(to_voxel_centers, [[2.5, 2.5], [0.5, 0.5]])

    to_voxel_centers = coords.to(darsia.VoxelCenterArray, image.coordinatesystem)
    assert isinstance(to_voxel_centers, darsia.VoxelCenterArray)
    assert np.allclose(to_voxel_centers, [[2.5, 2.5], [0.5, 0.5]])

    # From VoxelArray
    voxels = darsia.VoxelArray([[2, 1], [1, 3]])
    to_voxel_centers = voxels.to_voxel_center(image.coordinatesystem)
    assert isinstance(to_voxel_centers, darsia.VoxelCenterArray)
    assert np.allclose(to_voxel_centers, [[2.5, 1.5], [1.5, 3.5]])

    to_voxel_centers = voxels.to(darsia.VoxelCenterArray, image.coordinatesystem)
    assert isinstance(to_voxel_centers, darsia.VoxelCenterArray)
    assert np.allclose(to_voxel_centers, [[2.5, 1.5], [1.5, 3.5]])

    # From VoxelCenterArray
    voxel_centers = darsia.VoxelCenterArray([[2, 1], [1, 3]])
    to_voxel_centers = voxel_centers.to_voxel_center(image.coordinatesystem)
    assert isinstance(to_voxel_centers, darsia.VoxelCenterArray)
    assert np.allclose(to_voxel_centers, [[2.5, 1.5], [1.5, 3.5]])

    to_voxel_centers = voxel_centers.to(darsia.VoxelCenterArray, image.coordinatesystem)
    assert isinstance(to_voxel_centers, darsia.VoxelCenterArray)
    assert np.allclose(to_voxel_centers, [[2.5, 1.5], [1.5, 3.5]])
