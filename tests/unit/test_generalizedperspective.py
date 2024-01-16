"""Unit tests for geometric shape transformations via generalized perspective."""

import numpy as np

import darsia


def test_generalized_perspective_identity_2d_coordinate():
    """Test generalized perspetive transformation corresponding to identity."""

    # Define image to obtain some coordinate system
    img = darsia.Image(
        np.zeros((3, 4)),
        dimensions=[1, 1],
        space_dim=2,
    )
    coordinatesystem = img.coordinatesystem

    # Define coordinates
    coordinates_src = darsia.make_coordinate([[0, 0], [2, 2]])
    coordinates_dst = darsia.make_coordinate([[0, 0], [2, 2]])

    # Define generalized perspective transformation
    generalized_perspective = darsia.GeneralizedPerspectiveTransformation()
    generalized_perspective.fit(
        coordinates_src,
        coordinates_dst,
        {
            "coordinatesystem_dst": coordinatesystem,
        },
    )

    # Check whether the map is the identity
    assert np.allclose(generalized_perspective.A, np.eye(2))
    assert np.allclose(generalized_perspective.b, 0)
    assert np.allclose(generalized_perspective.c, 0)
    assert np.allclose(generalized_perspective.stretch_factor, 0)
    assert np.allclose(generalized_perspective.stretch_center_off, 0)
    assert np.allclose(generalized_perspective.bulge_factor, 0)
    assert np.allclose(generalized_perspective.bulge_center_off, 0)


def test_generalized_perspective_identity_2d_voxel():
    """Test generalized perspetive transformation corresponding to identity."""

    # Define image to obtain some coordinate system
    img = darsia.Image(
        np.zeros((3, 4)),
        dimensions=[1, 1],
        space_dim=2,
    )
    coordinatesystem = img.coordinatesystem

    # Define voxels
    voxels_src = darsia.make_voxel([[0, 0], [2, 2]])
    voxels_dst = darsia.make_voxel([[0, 0], [2, 2]])

    # Define generalized perspective transformation
    generalized_perspective = darsia.GeneralizedPerspectiveTransformation()
    generalized_perspective.fit(
        voxels_src,
        voxels_dst,
        {
            "coordinatesystem_dst": coordinatesystem,
        },
    )

    # Check whether the map is the identity
    assert np.allclose(generalized_perspective.A, np.eye(2))
    assert np.allclose(generalized_perspective.b, 0)
    assert np.allclose(generalized_perspective.c, 0)
    assert np.allclose(generalized_perspective.stretch_factor, 0)
    assert np.allclose(generalized_perspective.stretch_center_off, 0)
    assert np.allclose(generalized_perspective.bulge_factor, 0)
    assert np.allclose(generalized_perspective.bulge_center_off, 0)


def test_generalized_perspective_identity_2d_voxel_center():
    """Test generalized perspetive transformation corresponding to identity."""

    # Define image to obtain some coordinate system
    img = darsia.Image(
        np.zeros((3, 4)),
        dimensions=[1, 1],
        space_dim=2,
    )
    coordinatesystem = img.coordinatesystem

    # Define voxel centers
    voxels_center_src = darsia.make_voxel_center([[0, 0], [2, 2]])
    voxels_center_dst = darsia.make_voxel_center([[0, 0], [2, 2]])

    # Define generalized perspective transformation
    generalized_perspective = darsia.GeneralizedPerspectiveTransformation()
    generalized_perspective.fit(
        voxels_center_src,
        voxels_center_dst,
        {
            "coordinatesystem_dst": coordinatesystem,
        },
    )

    # Check whether the map is the identity
    assert np.allclose(generalized_perspective.A, np.eye(2))
    assert np.allclose(generalized_perspective.b, 0)
    assert np.allclose(generalized_perspective.c, 0)
    assert np.allclose(generalized_perspective.stretch_factor, 0)
    assert np.allclose(generalized_perspective.stretch_center_off, 0)
    assert np.allclose(generalized_perspective.bulge_factor, 0)
    assert np.allclose(generalized_perspective.bulge_center_off, 0)


def test_generalized_perspective_underdetermined_translation_coordinate():
    """Test generalized perspective for translation."""

    # Define image to obtain some coordinate system
    img = darsia.Image(
        np.zeros((3, 4)),
        dimensions=[1, 1],
        space_dim=2,
    )
    coordinatesystem = img.coordinatesystem

    coordinates_src = darsia.make_coordinate([[0, 0], [1, 2]])
    coordinates_dst = darsia.make_coordinate([[1, 0], [2, 2]])

    # Define generalized perspective transformation
    generalized_perspective = darsia.GeneralizedPerspectiveTransformation()
    generalized_perspective.fit(
        coordinates_src,
        coordinates_dst,
        {
            "coordinatesystem_dst": coordinatesystem,
            "strategy": ["perspective"],
        },
    )

    assert np.allclose(
        generalized_perspective.A, [[0.47281042, 0.23439567], [-0.05920453, 0.97120403]]
    )
    assert np.allclose(generalized_perspective.b, [-0.47281041, 0.05920452])
    assert np.allclose(generalized_perspective.c, [-0.02558076, -0.00361835])
    assert np.allclose(generalized_perspective.stretch_factor, 0)
    assert np.allclose(generalized_perspective.stretch_center_off, 0)
    assert np.allclose(generalized_perspective.bulge_factor, 0)
    assert np.allclose(generalized_perspective.bulge_center_off, 0)


def test_generalized_perspective_translation_coordinate():
    """Test generalized perspective for translation."""

    # Define image to obtain some coordinate system
    img = darsia.Image(
        np.zeros((3, 4)),
        dimensions=[1, 1],
        space_dim=2,
    )
    coordinatesystem = img.coordinatesystem

    coordinates_src = darsia.make_coordinate([[0, 0], [1, 2], [2, 0], [3, 2]])
    coordinates_dst = darsia.make_coordinate([[1, 0], [2, 2], [3, 0], [4, 2]])

    # Define generalized perspective transformation
    generalized_perspective = darsia.GeneralizedPerspectiveTransformation()
    generalized_perspective.fit(
        coordinates_src,
        coordinates_dst,
        {
            "coordinatesystem_dst": coordinatesystem,
            "strategy": ["perspective"],
        },
    )

    # Check whether the affine map is the inverse translation
    assert np.allclose(generalized_perspective.A, np.eye(2))
    assert np.allclose(generalized_perspective.b, [-1, 0], atol=1e-5)
    assert np.allclose(generalized_perspective.c, 0)
    assert np.allclose(generalized_perspective.stretch_factor, 0)
    assert np.allclose(generalized_perspective.stretch_center_off, 0)
    assert np.allclose(generalized_perspective.bulge_factor, 0)
    assert np.allclose(generalized_perspective.bulge_center_off, 0)


def test_generalized_perspective_affine_coordinate():
    """Test generalized perspective for translation."""

    # Define image to obtain some coordinate system
    img = darsia.Image(
        np.zeros((3, 4)),
        dimensions=[1, 1],
        space_dim=2,
    )
    coordinatesystem = img.coordinatesystem

    coordinates_src = darsia.make_coordinate([[0, 0], [1, 2], [2, 0], [3, 2]])
    coordinates_dst = darsia.make_coordinate([[0, 0], [2, -2], [4, 0], [6, -2]])

    # Define generalized perspective transformation
    generalized_perspective = darsia.GeneralizedPerspectiveTransformation()
    generalized_perspective.fit(
        coordinates_src,
        coordinates_dst,
        {
            "coordinatesystem_dst": coordinatesystem,
            "strategy": ["perspective"],
        },
    )

    # Check whether the affine map is the inverse translation
    assert np.allclose(generalized_perspective.A, [[0.5, 0], [0, -1]], atol=1e-6)
    assert np.allclose(generalized_perspective.b, 0, atol=1e-6)
    assert np.allclose(generalized_perspective.c, 0, atol=1e-6)
    assert np.allclose(generalized_perspective.stretch_factor, 0)
    assert np.allclose(generalized_perspective.stretch_center_off, 0)
    assert np.allclose(generalized_perspective.bulge_factor, 0)
    assert np.allclose(generalized_perspective.bulge_center_off, 0)


def test_generalized_perspective_coordinate():
    """Test generalized perspective for translation."""

    # Define image to obtain some coordinate system
    img = darsia.Image(
        np.zeros((3, 4)),
        dimensions=[1, 1],
        space_dim=2,
    )
    coordinatesystem = img.coordinatesystem

    coordinates_src = darsia.make_coordinate(
        [[0, 0], [1, 2], [2, 0], [3, 2], [4, 0], [5, 2], [6, 0], [7, 2]]
    )
    coordinates_dst = darsia.make_coordinate(
        [
            [0.1, 0.2],
            [2.1, 2.3],
            [4.1, 0.3],
            [6.1, 2.2],
            [8.1, 0.2],
            [10.1, 2.3],
            [12.1, 0.3],
            [14.1, 2.2],
        ]
    )

    # Define generalized perspective transformation
    generalized_perspective = darsia.GeneralizedPerspectiveTransformation()
    generalized_perspective.fit(
        coordinates_src,
        coordinates_dst,
        {
            "coordinatesystem_dst": coordinatesystem,
            "maxiter": 2000,
            "strategy": ["all"],
        },
    )

    # The values have been created once from a fixed implementation to have some benchmark
    assert np.allclose(
        generalized_perspective.A,
        [[0.50669216, -0.01916554], [-0.01124939, 0.97051429]],
        atol=1e-6,
    )
    assert np.allclose(generalized_perspective.b, [-0.04515077, -0.10718482], atol=1e-6)
    assert np.allclose(generalized_perspective.c, [0.00352531, 0.00488361], atol=1e-6)
    assert np.allclose(
        generalized_perspective.stretch_factor, [-0.00362438, -0.00223337], atol=1e-6
    )
    assert np.allclose(
        generalized_perspective.stretch_center_off,
        [-7.57650567, -0.62730542],
        atol=1e-6,
    )
    assert np.allclose(
        generalized_perspective.bulge_factor,
        [-3.66546709e-04, 6.25126767e-01],
        atol=1e-6,
    )
    assert np.allclose(
        generalized_perspective.bulge_center_off, [-12.21249625, 1.4888217], atol=1e-6
    )
