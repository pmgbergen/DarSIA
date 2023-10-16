"""Module testing coordinate transformation capabilities for
images with incompatible coordinate systems.

"""

import numpy as np
import pytest

import darsia


def test_coordinate_transformation_identity_2d():
    """Test coordinate transformation corresponding to identity."""

    # Define image to be transformed
    arr_src = np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
        ]
    )
    image_src = darsia.Image(
        arr_src,
        dimensions=[3, 4],
    )

    # Define image representative for target canvas
    arr_dst = np.zeros((3, 4), dtype=float)
    image_dst = darsia.Image(
        arr_dst,
        dimensions=[3, 4],
    )

    voxels_src = [[0, 0], [2, 2]]
    voxels_dst = [[0, 0], [2, 2]]

    # Define coordinate transformation
    coordinate_transformation = darsia.CoordinateTransformation(
        image_src.coordinatesystem,
        image_dst.coordinatesystem,
        voxels_src,
        voxels_dst,
    )

    # Check whether coordinate transform generates the same image
    transformed_image = coordinate_transformation(image_src)
    assert np.allclose(transformed_image.img, image_src.img)

    meta_tra = transformed_image.metadata()
    meta_src = image_src.metadata()
    assert np.allclose(meta_tra["origin"], meta_src["origin"])
    assert np.allclose(meta_tra["dimensions"], meta_src["dimensions"])


def test_coordinate_transformation_change_meta_2d():
    """Test coordinate transformation corresponding to embedding with change in
    metadata.

    """

    # Define image to be transformed
    arr_src = np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
        ]
    )
    image_src = darsia.Image(
        arr_src,
        dimensions=[3, 4],
    )

    # Define image representative for target canvas
    arr_dst = np.zeros((3, 4), dtype=float)
    image_dst = darsia.Image(
        arr_dst,
        dimensions=[30, 40],
        origin=[0, 2],
    )

    voxels_src = [[0, 0], [2, 2]]
    voxels_dst = [[0, 0], [2, 2]]

    # Define coordinate transformation
    coordinate_transformation = darsia.CoordinateTransformation(
        image_src.coordinatesystem,
        image_dst.coordinatesystem,
        voxels_src,
        voxels_dst,
    )

    # Check whether coordinate transform generates the same image
    transformed_image = coordinate_transformation(image_src)
    assert np.allclose(transformed_image.img, image_src.img)

    meta_tra = transformed_image.metadata()
    meta_dst = image_dst.metadata()
    assert np.allclose(meta_tra["origin"], meta_dst["origin"])
    assert np.allclose(meta_tra["dimensions"], meta_dst["dimensions"])


@pytest.mark.parametrize("isometry", [False, True])
def test_coordinate_transformation_translation(isometry):
    """Test coordinate transformation corresponding to embedding with change in
    metadata.

    """

    # Define image to be transformed
    arr_src = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 0, 0],
        ]
    )
    image_src = darsia.Image(
        arr_src,
        dimensions=[1e-3, 1e-3],
    )

    # Define image representative for target canvas
    arr_dst = np.zeros((4, 4), dtype=float)
    image_dst = darsia.Image(
        arr_dst,
        dimensions=[1e-3, 1e-3],
    )

    voxels_src = [[0, 0], [1, 2]]
    voxels_dst = [[1, 0], [2, 2]]

    # Define coordinate transformation
    coordinate_transformation = darsia.CoordinateTransformation(
        image_src.coordinatesystem,
        image_dst.coordinatesystem,
        voxels_src,
        voxels_dst,
        fit_options={
            "tol": 1e-6,
            "maxiter": 10000,
        },
        isometry=isometry,
    )

    # Check whether coordinate transform generates the same image
    transformed_image = coordinate_transformation(image_src)
    reference_arr = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ]
    )
    assert np.allclose(transformed_image.img, reference_arr)

    meta_tra = transformed_image.metadata()
    meta_dst = image_dst.metadata()
    assert np.allclose(meta_tra["origin"], meta_dst["origin"])
    assert np.allclose(meta_tra["dimensions"], meta_dst["dimensions"])


@pytest.mark.parametrize("preconditioning", [False, True])
def test_coordinate_transformation_rotation(preconditioning):
    """Test coordinate transformation corresponding to embedding with change in
    metadata.

    """

    # Define image to be transformed
    arr_src = np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    image_src = darsia.Image(
        arr_src,
        dimensions=[1e-1, 1e-1],
    )

    # Define image representative for target canvas
    arr_dst = np.zeros((4, 4), dtype=float)
    image_dst = darsia.Image(
        arr_dst,
        dimensions=[1e-1, 1e-1],
    )

    voxels_src = [[1, 0], [1, 3]]
    voxels_dst = [[0, 1], [3, 1]]

    # Define coordinate transformation
    coordinate_transformation = darsia.CoordinateTransformation(
        image_src.coordinatesystem,
        image_dst.coordinatesystem,
        voxels_src,
        voxels_dst,
        fit_options={"tol": 1e-4, "maxiter": 1000, "preconditioning": preconditioning},
        isometry=False,
    )

    # Check whether coordinate transform generates the same image
    transformed_image = coordinate_transformation(image_src)
    reference_arr = np.array(
        [
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ]
    )
    assert np.allclose(transformed_image.img, reference_arr)

    meta_tra = transformed_image.metadata()
    meta_dst = image_dst.metadata()
    assert np.allclose(meta_tra["origin"], meta_dst["origin"])
    assert np.allclose(meta_tra["dimensions"], meta_dst["dimensions"])
