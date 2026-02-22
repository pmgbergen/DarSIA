"""Module testing the arithmetics capabilities for images.

"""

import numpy as np

import darsia


def test_scalar_weight_3d():

    image = darsia.Image(np.ones((3, 4, 5), dtype=float), space_dim=3)
    weight = 2.0
    weighted_image = darsia.weight(image, weight)
    assert np.allclose(weighted_image.img, 2 * np.ones((3, 4, 5)))


def test_array_weight_2d():

    image = darsia.Image(np.ones((3, 4), dtype=float), space_dim=2)
    weight = darsia.Image(np.random.rand(3, 4), space_dim=2)
    weighted_image = darsia.weight(image, weight)
    assert np.allclose(weighted_image.img, weight.img)


def test_incompatible_weight_2d():

    image = darsia.Image(np.ones((6, 8), dtype=float), space_dim=2, dimensions=[6, 8])
    weight = darsia.Image(2 * np.ones((3, 4)), space_dim=2, dimensions=[6, 8])

    weighted_image = darsia.weight(image, weight)
    assert np.allclose(weighted_image.img, 2 * np.ones((6, 8)))


def test_array_weight_3d():

    image = darsia.Image(np.ones((3, 4, 5), dtype=float), space_dim=3)
    weight = darsia.Image(np.random.rand(3, 4, 5), space_dim=3)
    weighted_image = darsia.weight(image, weight)
    assert np.allclose(weighted_image.img, weight.img)


def test_superposition_2d():

    image1 = darsia.ScalarImage(np.ones((3, 4), dtype=float), space_dim=2)
    image2 = darsia.ScalarImage(2 * np.ones((3, 4), dtype=float), space_dim=2)
    image3 = darsia.ScalarImage(3 * np.ones((3, 4), dtype=float), space_dim=2)
    superposed_image = darsia.superpose([image1, image2, image3])
    assert np.allclose(superposed_image.img, 6 * np.ones((3, 4)))


def test_superposition_2d_spacetime():

    meta = {
        "space_dim": 2,
        "series": True,
        "time": [0, 1, 2, 3, 4],
    }
    image1 = darsia.ScalarImage(np.ones((3, 4, 5), dtype=float), **meta)
    image2 = darsia.ScalarImage(2 * np.ones((3, 4, 5), dtype=float), **meta)
    image3 = darsia.ScalarImage(3 * np.ones((3, 4, 5), dtype=float), **meta)
    superposed_image = darsia.superpose([image1, image2, image3])

    assert np.allclose(superposed_image.img, 6 * np.ones((3, 4, 5)))
    superposed_meta = superposed_image.metadata()
    for key, value in meta.items():
        assert np.allclose(value, superposed_meta[key])
