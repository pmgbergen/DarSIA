"""Module testing the coordinate system of Image.

"""

import numpy as np

import darsia


def test_init_unit_coordinatesystem_2d():
    """Test the definition of all attributes."""

    # Construct a 2d image with 3 x 4 pixels and unit dimensions
    image = darsia.Image(
        np.ones((3, 4), dtype=float),
        dimensions=[1, 1],
        space_dim=2,
    )

    # Fetch coordinate system
    cs = image.coordinatesystem

    # Check all attributes defined in CoordinateSystem.__init__()
    assert cs.dim == 2, "space dimension not correct"
    assert cs.shape == (3, 4), "shape not correct"
    assert np.allclose(cs.dimensions, [1, 1]), "dimensions not correct"
    assert cs.axes == "xy", "axes not correct"
    assert np.allclose(
        [cs.voxel_size["x"], cs.voxel_size["y"]], [1 / 4, 1 / 3]
    ), "voxel_size not correct"
    assert np.allclose(cs._coordinate_of_origin_voxel, [0, 1]), "origin not correct"
    assert np.allclose(cs._coordinate_of_opposite_voxel, [1, 0]), "opposite not correct"
    assert np.isclose(cs.domain["xmin"], 0), "xmin not correct"
    assert np.isclose(cs.domain["xmax"], 1), "xmax not correct"
    assert np.isclose(cs.domain["ymin"], 0), "ymin not correct"
    assert np.isclose(cs.domain["ymax"], 1), "ymax not correct"


def test_voxels_unit_coordinatesystem_2d():
    """Test CoordinateSystem.voxels()."""

    # Construct a 2d image with 3 x 4 pixels and unit dimensions
    image = darsia.Image(
        np.ones((3, 4), dtype=float),
        dimensions=[1, 1],
        space_dim=2,
    )

    # Fetch coordinate system
    cs = image.coordinatesystem

    # Check all attributes defined in CoordinateSystem.__init__()
    assert np.allclose(
        cs.voxels,
        np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [0, 2],
                [1, 2],
                [2, 2],
                [0, 3],
                [1, 3],
                [2, 3],
            ]
        ),
    ), "voxels not correct"


def test_coordinates_unit_coordinatesystem_2d():
    """Test CoordinateSystem.coordinates()."""

    # Construct a 2d image with 3 x 4 pixels and unit dimensions
    image = darsia.Image(
        np.ones((3, 4), dtype=float),
        dimensions=[1, 1],
        space_dim=2,
    )

    # Fetch coordinate system
    cs = image.coordinatesystem

    # Check all attributes defined in CoordinateSystem.__init__()
    assert np.allclose(
        cs.coordinates,
        np.array(
            [
                [0, 1],
                [0, 2 / 3],
                [0, 1 / 3],
                [1 / 4, 1],
                [1 / 4, 2 / 3],
                [1 / 4, 1 / 3],
                [2 / 4, 1],
                [2 / 4, 2 / 3],
                [2 / 4, 1 / 3],
                [3 / 4, 1],
                [3 / 4, 2 / 3],
                [3 / 4, 1 / 3],
            ]
        ),
    ), "coordinates not correct"


def test_length_unit_coordinatesystem_2d():
    """Test CoordinateSystem.length()."""

    # Construct a 2d image with 3 x 4 pixels and unit dimensions
    image = darsia.Image(
        np.ones((3, 4), dtype=float),
        dimensions=[1, 1],
        space_dim=2,
    )

    # Fetch coordinate system
    cs = image.coordinatesystem

    for i in range(4):
        assert np.isclose(cs.length(i, "x"), i / 4), "length not correct in x"
    for i in range(3):
        assert np.isclose(cs.length(i, "y"), i / 3), "length not correct"


def test_num_voxels_unit_coordinatesystem_2d():
    """Test CoordinateSystem.num_voxels()."""

    # Construct a 2d image with 3 x 4 pixels and unit dimensions
    image = darsia.Image(
        np.ones((3, 4), dtype=float),
        dimensions=[1, 1],
        space_dim=2,
    )

    # Fetch coordinate system
    cs = image.coordinatesystem

    assert np.allclose(cs.num_voxels(1, "x"), 4), "num_voxels not correct in x"
    assert np.allclose(cs.num_voxels(1, "y"), 3), "num_voxels not correct in y"
    assert np.allclose(cs.num_voxels(0.7, "x"), 3), "num_voxels not correct in x"
    assert np.allclose(cs.num_voxels(0.6, "y"), 2), "num_voxels not correct in y"
    assert np.allclose(cs.num_voxels(0.8, "x"), 4), "num_voxels not correct in x"
    assert np.allclose(cs.num_voxels(0.7, "y"), 3), "num_voxels not correct in y"


def test_coordinate_unit_coordinatesystem_2d():
    """Test CoordinateSystem.coordinate()."""

    # Construct a 2d image with 3 x 4 pixels and unit dimensions
    image = darsia.Image(
        np.ones((3, 4), dtype=float),
        dimensions=[1, 1],
        space_dim=2,
    )

    # Fetch coordinate system
    cs = image.coordinatesystem

    # Define test for single voxel
    voxel_list = [1, 2]

    def check_voxel(voxel):
        coordinate = cs.coordinate(voxel)
        assert isinstance(coordinate, darsia.Coordinate), "coordinate type not correct"
        assert np.allclose(coordinate, [0.5, 2 / 3]), "coordinate values not correct"

    # Input: np.ndarray
    voxel = np.array(voxel_list)
    check_voxel(voxel)

    # Input: list
    voxel = voxel_list
    check_voxel(voxel)

    # Input: tuple
    voxel = tuple(voxel_list)

    # Input: Voxel
    voxel = darsia.Voxel(voxel_list)
    check_voxel(voxel)

    # Define test for multiple voxels
    voxels_list = [[1, 2], [2, 3]]

    def check_voxels(voxels):
        coordinates = cs.coordinate(voxels)
        assert isinstance(
            coordinates, darsia.CoordinateArray
        ), "coordinate type not correct"
        assert np.allclose(
            coordinates, [[0.5, 2 / 3], [0.75, 1 / 3]]
        ), "coordinate values not correct"

    # Input: np.ndarray
    voxels = np.array(voxels_list)
    check_voxels(voxels)

    # Input: list
    voxels = voxels_list
    check_voxels(voxels)

    # Input: VoxelArray
    voxels = darsia.VoxelArray(voxels_list)
    check_voxels(voxels)


def test_voxel_unit_coordinatesystem_2d():
    """Test CoordinateSystem.voxel()."""

    # Construct a 2d image with 3 x 4 pixels and unit dimensions
    image = darsia.Image(
        np.ones((3, 4), dtype=float),
        dimensions=[1, 1],
        space_dim=2,
    )

    # Fetch coordinate system
    cs = image.coordinatesystem

    # Define test for single coordinate
    coordinate_list = [0.5, 2 / 3]

    def check_coordinate(coordinate):
        voxel = cs.voxel(coordinate)
        assert isinstance(voxel, darsia.Voxel), "voxel type not correct"
        assert np.allclose(voxel, [1, 2]), "voxel values not correct"

    # Input: np.ndarray
    coordinate = np.array(coordinate_list)
    check_coordinate(coordinate)

    # Input: list
    coordinate = coordinate_list
    check_coordinate(coordinate)

    # Input: tuple
    coordinate = tuple(coordinate_list)

    # Input: Coordinate
    coordinate = darsia.Coordinate(coordinate_list)
    check_coordinate(coordinate)

    # Define test for multiple coordinates
    coordinates_list = [[0.5, 2 / 3], [0.75, 1 / 3], [0.6, 0.9]]

    def check_coordinates(coordinates):
        voxels = cs.voxel(coordinates)
        assert isinstance(voxels, darsia.VoxelArray), "voxel type not correct"
        assert np.allclose(voxels, [[1, 2], [2, 3], [0, 2]]), "voxel values not correct"

    # Input: np.ndarray
    coordinates = np.array(coordinates_list)
    check_coordinates(coordinates)

    # Input: list
    coordinates = coordinates_list
    check_coordinates(coordinates)

    # Input: CoordinateArray
    coordinates = darsia.CoordinateArray(coordinates_list)
    check_coordinates(coordinates)


def test_coordinate_vector_unit_coordinatesystem_2d():
    """Test CoordinateSystem.coordinate_vector()."""

    # Construct a 2d image with 3 x 4 pixels and unit dimensions
    image = darsia.Image(
        np.ones((3, 4), dtype=float),
        dimensions=[1, 1],
        space_dim=2,
    )

    # Fetch coordinate system
    cs = image.coordinatesystem

    # Input: np.ndarray for single vector
    voxel_vector = np.array([1, 2])
    coordinate_vector = cs.coordinate_vector(voxel_vector)
    assert np.allclose(
        coordinate_vector, [0.5, -1 / 3]
    ), "coordinate_vector values not correct"

    # Input: np.ndarray for multiple vectors
    voxel_vectors = np.array([[1, 2], [2, 2]])
    coordinate_vectors = cs.coordinate_vector(voxel_vectors)
    assert np.allclose(
        coordinate_vectors, [[0.5, -1 / 3], [0.5, -2 / 3]]
    ), "coordinate_vector values not correct"
