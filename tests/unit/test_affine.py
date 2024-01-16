"""Unit tests for geometric transformations."""

import numpy as np
import pytest

import darsia


def test_affine_transformation_identity_2d_coordinate():
    """Test affine transformation corresponding to identity."""

    coordinates_src = darsia.make_coordinate([[0, 0], [2, 2]])
    coordinates_dst = darsia.make_coordinate([[0, 0], [2, 2]])

    # Define affine transformation
    affine_transformation = darsia.AffineTransformation(2)
    affine_transformation.fit(
        coordinates_src,
        coordinates_dst,
        {},
    )

    # Check whether the affine map is the identity
    assert np.allclose(affine_transformation.translation, 0)
    assert np.allclose(affine_transformation.rotation, np.eye(2))
    assert np.allclose(affine_transformation.scaling, 1)


def test_affine_transformation_identity_2d_voxel():
    """Test affine transformation corresponding to identity."""

    voxels_src = darsia.make_voxel([[0, 0], [2, 2]])
    voxels_dst = darsia.make_voxel([[0, 0], [2, 2]])

    # Define affine transformation
    affine_transformation = darsia.AffineTransformation(2)
    affine_transformation.fit(
        voxels_src,
        voxels_dst,
        {},
    )

    # Check whether the affine map is the identity
    assert np.allclose(affine_transformation.translation, 0)
    assert np.allclose(affine_transformation.rotation, np.eye(2))
    assert np.allclose(affine_transformation.scaling, 1)


def test_affine_transformation_identity_2d_voxel_center():
    """Test affine transformation corresponding to identity."""

    voxels_center_src = darsia.make_voxel_center([[0, 0], [2, 2]])
    voxels_center_dst = darsia.make_voxel_center([[0, 0], [2, 2]])

    # Define affine transformation
    affine_transformation = darsia.AffineTransformation(2)
    affine_transformation.fit(
        voxels_center_src,
        voxels_center_dst,
        {},
    )

    # Check whether the affine map is the identity
    assert np.allclose(affine_transformation.translation, 0)
    assert np.allclose(affine_transformation.rotation, np.eye(2))
    assert np.allclose(affine_transformation.scaling, 1)


@pytest.mark.parametrize("isometry", [False, True])
def test_affine_transformation_translation_coordinate(isometry):
    """Test affine transformation for translation."""

    coordinates_src = darsia.make_coordinate([[0, 0], [1, 2]])
    coordinates_dst = darsia.make_coordinate([[1, 0], [2, 2]])

    # Define affine transformation
    affine_transformation = darsia.AffineTransformation(2)
    affine_transformation.fit(
        coordinates_src,
        coordinates_dst,
        fit_options={
            "tol": 1e-6,
            "maxiter": 10000,
            "isometry": isometry,
        },
    )

    assert np.allclose(affine_transformation.translation, [1, 0])
    assert np.allclose(affine_transformation.rotation, np.eye(2))
    assert np.allclose(affine_transformation.scaling, 1)


@pytest.mark.parametrize("isometry", [False, True])
def test_affine_transformation_translation_voxel(isometry):
    """Test affine transformation for translation."""

    voxels_src = darsia.make_voxel([[0, 0], [1, 2]])
    voxels_dst = darsia.make_voxel([[1, 0], [2, 2]])

    # Define affine transformation
    affine_transformation = darsia.AffineTransformation(2)
    affine_transformation.fit(
        voxels_src,
        voxels_dst,
        fit_options={
            "tol": 1e-6,
            "maxiter": 10000,
            "isometry": isometry,
        },
    )

    assert np.allclose(affine_transformation.translation, [1, 0])
    assert np.allclose(affine_transformation.rotation, np.eye(2))
    assert np.allclose(affine_transformation.scaling, 1)


@pytest.mark.parametrize("isometry", [False, True])
def test_affine_transformation_translation_voxel_center(isometry):
    """Test affine transformation for translation."""

    voxels_center_src = darsia.make_voxel_center([[0, 0], [1, 2]])
    voxels_center_dst = darsia.make_voxel_center([[1, 0], [2, 2]])

    # Define affine transformation
    affine_transformation = darsia.AffineTransformation(2)
    affine_transformation.fit(
        voxels_center_src,
        voxels_center_dst,
        fit_options={
            "tol": 1e-6,
            "maxiter": 10000,
            "isometry": isometry,
        },
    )

    assert np.allclose(affine_transformation.translation, [1, 0])
    assert np.allclose(affine_transformation.rotation, np.eye(2))
    assert np.allclose(affine_transformation.scaling, 1)


@pytest.mark.parametrize("preconditioning", [False, True])
def test_affine_transformation_rotation_coordinate(preconditioning):
    """Test affine transformation for rotation."""

    coordinates_src = darsia.make_coordinate([[1, 0], [1, 3]])
    coordinates_dst = darsia.make_coordinate([[0, 1], [3, 1]])

    # Define affine transformation
    affine_transformation = darsia.AffineTransformation(2)
    affine_transformation.fit(
        coordinates_src,
        coordinates_dst,
        fit_options={
            "tol": 1e-4,
            "maxiter": 1000,
            "preconditioning": preconditioning,
            "isometry": False,
        },
    )

    assert np.allclose(affine_transformation.translation, [0, 2])
    assert np.allclose(affine_transformation.scaling, 1)
    assert np.allclose(affine_transformation.rotation, np.array([[0, 1], [-1, 0]]))


@pytest.mark.parametrize("preconditioning", [False, True])
def test_affine_transformation_rotation_voxel(preconditioning):
    """Test affine transformation for rotation."""

    voxels_src = darsia.make_voxel([[1, 0], [1, 3]])
    voxels_dst = darsia.make_voxel([[0, 1], [3, 1]])

    # Define affine transformation
    affine_transformation = darsia.AffineTransformation(2)
    affine_transformation.fit(
        voxels_src,
        voxels_dst,
        fit_options={
            "tol": 1e-4,
            "maxiter": 1000,
            "preconditioning": preconditioning,
            "isometry": False,
        },
    )

    assert np.allclose(affine_transformation.translation, [0, 2])
    assert np.allclose(affine_transformation.scaling, 1)
    assert np.allclose(affine_transformation.rotation, np.array([[0, 1], [-1, 0]]))


@pytest.mark.parametrize("preconditioning", [False, True])
def test_affine_transformation_rotation_voxel_center(preconditioning):
    """Test affine transformation for rotation."""

    voxels_center_src = darsia.make_voxel_center([[1, 0], [1, 3]])
    voxels_center_dst = darsia.make_voxel_center([[0, 1], [3, 1]])

    # Define affine transformation
    affine_transformation = darsia.AffineTransformation(2)
    affine_transformation.fit(
        voxels_center_src,
        voxels_center_dst,
        fit_options={
            "tol": 1e-4,
            "maxiter": 1000,
            "preconditioning": preconditioning,
            "isometry": False,
        },
    )

    assert np.allclose(affine_transformation.translation, [0, 3])
    assert np.allclose(affine_transformation.scaling, 1)
    assert np.allclose(affine_transformation.rotation, np.array([[0, 1], [-1, 0]]))
