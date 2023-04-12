"""Module testing the arithmetics capabilities for images.

"""

import numpy as np

import darsia


def test_scalar_weight_3d():

    array = np.ones((3, 4, 5), dtype=float)
    image = darsia.Image(array, dim=3)
    weight = 2.0
    weighted_image = darsia.weight(image, weight)
    assert np.allclose(weighted_image.img, 2 * np.ones((3, 4, 5)))


def test_array_weight_2d():

    image = darsia.Image(np.ones((3, 4), dtype=float), dim=2)
    weight = darsia.Image(np.random.rand(3, 4), dim=2)
    weighted_image = darsia.weight(image, weight)
    assert np.allclose(weighted_image.img, weight.img)


def test_incompatible_weight_2d():

    image = darsia.Image(np.ones((6, 8), dtype=float), dim=2)
    weight = darsia.Image(2 * np.ones((3, 4)), dim=2)

    weighted_image = darsia.weight(image, weight)
    assert np.allclose(weighted_image.img, 2 * np.ones((6, 8)))


def test_array_weight_3d():

    image = darsia.Image(np.ones((3, 4, 5), dtype=float), dim=3)
    weight = darsia.Image(np.random.rand(3, 4, 5), dim=3)
    weighted_image = darsia.weight(image, weight)
    assert np.allclose(weighted_image.img, weight.img)
