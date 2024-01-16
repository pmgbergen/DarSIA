"""Unit tests for geometric shape transformations via generalized perspective."""

import numpy as np

import darsia


def test_generalized_perspective_identity_2d_coordinate():
    """Test affine transformation corresponding to identity."""

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

    # Define affine transformation
    generalized_perspective = darsia.GeneralizedPerspectiveTransformation()
    generalized_perspective.fit(
        coordinates_src,
        coordinates_dst,
        {
            "coordinatesystem_dst": coordinatesystem,
        },
    )

    # Check whether the affine map is the identity
    assert np.allclose(generalized_perspective.A, np.eye(2))
    assert np.allclose(generalized_perspective.b, 0)
    assert np.allclose(generalized_perspective.c, 0)
    assert np.allclose(generalized_perspective.stretch_factor, 0)
    assert np.allclose(generalized_perspective.stretch_center_off, 0)
    assert np.allclose(generalized_perspective.bulge_factor, 0)
    assert np.allclose(generalized_perspective.bulge_center_off, 0)


def test_generalized_perspective_identity_2d_voxel():
    """Test affine transformation corresponding to identity."""

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

    # Define affine transformation
    generalized_perspective = darsia.GeneralizedPerspectiveTransformation()
    generalized_perspective.fit(
        voxels_src,
        voxels_dst,
        {
            "coordinatesystem_dst": coordinatesystem,
        },
    )

    # Check whether the affine map is the identity
    assert np.allclose(generalized_perspective.A, np.eye(2))
    assert np.allclose(generalized_perspective.b, 0)
    assert np.allclose(generalized_perspective.c, 0)
    assert np.allclose(generalized_perspective.stretch_factor, 0)
    assert np.allclose(generalized_perspective.stretch_center_off, 0)
    assert np.allclose(generalized_perspective.bulge_factor, 0)
    assert np.allclose(generalized_perspective.bulge_center_off, 0)


def test_generalized_perspective_identity_2d_voxel_center():
    """Test affine transformation corresponding to identity."""

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

    # Define affine transformation
    generalized_perspective = darsia.GeneralizedPerspectiveTransformation()
    generalized_perspective.fit(
        voxels_center_src,
        voxels_center_dst,
        {
            "coordinatesystem_dst": coordinatesystem,
        },
    )

    # Check whether the affine map is the identity
    assert np.allclose(generalized_perspective.A, np.eye(2))
    assert np.allclose(generalized_perspective.b, 0)
    assert np.allclose(generalized_perspective.c, 0)
    assert np.allclose(generalized_perspective.stretch_factor, 0)
    assert np.allclose(generalized_perspective.stretch_center_off, 0)
    assert np.allclose(generalized_perspective.bulge_factor, 0)
    assert np.allclose(generalized_perspective.bulge_center_off, 0)
