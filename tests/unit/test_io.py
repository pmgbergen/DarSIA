"""Test I/O capabilities in darsia."""

from pathlib import Path

import numpy as np

import darsia


def test_imread_from_numpy():
    # Generate Image from random array
    shape = (10, 20)
    array = np.random.rand(*shape)
    path = Path("random_distribution.npy")
    np.save(path, array)

    # Read numpy image
    np_image = darsia.imread(path, dim=2, width=2, height=1)

    # Compare arrays.
    assert np.all(np.isclose(np_image.img, array))

    # Clean up
    path.unlink()
