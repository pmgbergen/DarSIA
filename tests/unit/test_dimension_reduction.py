"""Module testing the dimension reduction functionality.

"""

import numpy as np

import darsia


def test_axis_averaging_x():
    """Test dimension reduction from 3d to 2d via axis averaging over the x-axis."""

    image_3d = darsia.Image(
        np.ones((3, 4, 5), dtype=float),
        dim=3,
        dimensions=[2, 3, 4],
        series=False,
        scalar=True,
    )

    averaging_axis = darsia.AxisAveraging(axis="x", dim=3)
    averaging_index = darsia.AxisAveraging(axis=2, dim=3)

    image_2d_via_axis = averaging_axis(image_3d)
    image_2d_via_index = averaging_index(image_3d)

    assert np.allclose(image_2d_via_axis.img, image_2d_via_index.img)
    assert np.allclose(image_2d_via_axis.img, 5 * np.ones((3, 4)))

    assert np.allclose(image_2d_via_axis.dimensions, image_2d_via_index.dimensions)
    assert np.allclose(image_2d_via_axis.dimensions, [2, 3])

    assert np.allclose(image_2d_via_axis.origin, image_2d_via_index.origin)
    assert np.allclose(image_2d_via_axis.origin, [0, 2])


def test_axis_averaging_y():
    """Test dimension reduction from 3d to 2d via axis averaging over the y-axis."""

    image_3d = darsia.Image(
        np.ones((3, 4, 5), dtype=float),
        dim=3,
        dimensions=[2, 3, 4],
        series=False,
        scalar=True,
    )

    averaging_axis = darsia.AxisAveraging(axis="y", dim=3)
    averaging_index = darsia.AxisAveraging(axis=1, dim=3)

    image_2d_via_axis = averaging_axis(image_3d)
    image_2d_via_index = averaging_index(image_3d)

    assert np.allclose(image_2d_via_axis.img, image_2d_via_index.img)
    assert np.allclose(image_2d_via_axis.img, 4 * np.ones((3, 5)))

    assert np.allclose(image_2d_via_axis.dimensions, image_2d_via_index.dimensions)
    assert np.allclose(image_2d_via_axis.dimensions, [2, 4])

    assert np.allclose(image_2d_via_axis.origin, image_2d_via_index.origin)
    assert np.allclose(image_2d_via_axis.origin, [0, 2])


def test_axis_averaging_z():
    """Test dimension reduction from 3d to 2d via axis averaging over the z-axis."""

    image_3d = darsia.Image(
        np.ones((3, 4, 5), dtype=float),
        dim=3,
        dimensions=[2, 3, 4],
        series=False,
        scalar=True,
    )

    averaging_axis = darsia.AxisAveraging(axis="z", dim=3)
    averaging_index = darsia.AxisAveraging(axis=0, dim=3)

    image_2d_via_axis = averaging_axis(image_3d)
    image_2d_via_index = averaging_index(image_3d)

    assert np.allclose(image_2d_via_axis.img, image_2d_via_index.img)
    assert np.allclose(image_2d_via_axis.img, 3 * np.ones((4, 5)))

    assert np.allclose(image_2d_via_axis.dimensions, image_2d_via_index.dimensions)
    assert np.allclose(image_2d_via_axis.dimensions, [3, 4])

    assert np.allclose(image_2d_via_axis.origin, image_2d_via_index.origin)
    assert np.allclose(image_2d_via_axis.origin, [0, 3])


def test_axis_averaging_series_x():
    """Test dimension reduction from 4d to 3d via axis averaging over the x-axis.

    NOTE: This test, together with test_axis_averging_x, ... test_axis_averaging_z
    is implicitly covering all axes for series."""

    image_3d = darsia.Image(
        np.ones((3, 4, 5, 6), dtype=float),
        dim=3,
        dimensions=[2, 3, 4],
        series=True,
        scalar=True,
        time=[0, 1, 2, 3, 4, 5],
    )

    averaging_axis = darsia.AxisAveraging(axis="x", dim=3)
    averaging_index = darsia.AxisAveraging(axis=2, dim=3)

    image_2d_via_axis = averaging_axis(image_3d)
    image_2d_via_index = averaging_index(image_3d)

    assert np.allclose(image_2d_via_axis.img, image_2d_via_index.img)
    assert np.allclose(image_2d_via_axis.img, 5 * np.ones((3, 4, 6)))

    assert np.allclose(image_2d_via_axis.dimensions, image_2d_via_index.dimensions)
    assert np.allclose(image_2d_via_axis.dimensions, [2, 3])

    assert np.allclose(image_2d_via_axis.origin, image_2d_via_index.origin)
    assert np.allclose(image_2d_via_axis.origin, [0, 2])
