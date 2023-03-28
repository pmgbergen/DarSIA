"""Collection of unit tests for integration over geometries."""

import numpy as np

import darsia


def test_geometry_2d():
    """Test integration over simple two-dimensional geometry and compatible data."""

    space_dim = 2
    num_voxels = (2, 4)  # rows x cols
    dimensions = [0.5, 1.0]  # height x width
    geometry = darsia.Geometry(
        space_dim=space_dim, num_voxels=num_voxels, dimensions=dimensions
    )

    # Check voxel volume which should have height and width equal to 0.25
    assert geometry.voxel_volume == 0.25**2

    # Hardcoded data of compatible size
    data = np.zeros(num_voxels, dtype=float)
    data[0, 2] = 0.5
    data[0, 3] = 0.75
    data[1, 0] = -0.2
    data[1, 1] = 0.6
    data[1, 2] = 0.1

    # Test integration
    integral = geometry.integrate(data)
    assert np.isclose(integral, 0.25**2 * (0.5 + 0.75 - 0.2 + 0.6 + 0.1))

def test_geometry_2d_incompatible_data():
    """Test integration over simple two-dimensional geometry and incompatible data."""

    space_dim = 2
    num_voxels = (2, 4)  # rows x cols
    dimensions = [0.5, 1.0]  # height x width
    geometry = darsia.Geometry(
        space_dim=space_dim, num_voxels=num_voxels, dimensions=dimensions
    )

    # Hardcoded data of incompatible size (double)
    data = np.zeros(tuple(2 * num_voxels[i] for i in range(2)), dtype=float)
    data[0, 4] = 0.5
    data[0, 5] = 0.5
    data[1, 4] = 0.5
    data[1, 5] = 0.5

    data[0, 6] = 0.75
    data[0, 7] = 0.75
    data[1, 6] = 0.75
    data[1, 7] = 0.75

    data[2, 0] = -0.2
    data[2, 1] = -0.2
    data[3, 0] = -0.2
    data[3, 1] = -0.2

    data[2, 2] = 0.6
    data[2, 3] = 0.6
    data[3, 2] = 0.6
    data[3, 3] = 0.6

    data[2, 4] = 0.1
    data[2, 5] = 0.1
    data[3, 4] = 0.1
    data[3, 5] = 0.1

    # Test integration
    integral = geometry.integrate(data)
    assert np.isclose(integral, 0.25**2 * (0.5 + 0.75 - 0.2 + 0.6 + 0.1))

def test_extruded_geomety_fixed():
    """Test integration over extruded two-dimensional geometry with fixed depth
    and compatible data.

    """
    space_dim = 2
    num_voxels = (2, 4)  # rows x cols
    dimensions = [0.5, 1.0]  # height x width
    depth = 0.2 # fixed depth
    geometry = darsia.ExtrudedGeometry(
        depth, space_dim=space_dim, num_voxels=num_voxels, dimensions=dimensions
    )

    # Check voxel volume which should have height and width equal to 0.25
    assert geometry.effective_voxel_volume == 0.25**2 * 0.2

    # Hardcoded data of compatible size
    data = np.zeros(num_voxels, dtype=float)
    data[0, 2] = 0.5
    data[0, 3] = 0.75
    data[1, 0] = -0.2
    data[1, 1] = 0.6
    data[1, 2] = 0.1

    # Test integration
    integral = geometry.integrate(data)
    assert np.isclose(integral, 0.25**2 * 0.2 * (0.5 + 0.75 - 0.2 + 0.6 + 0.1))
