"""Collection of unit tests for EMD."""

import numpy as np

import darsia


def test_emd_2d_single_pixel():
    """Test simple EMD for 2d distributions with physical dimenions."""

    # Create two mass distributions with identical mass, equal to 1
    mass1_array = np.zeros((10, 20), dtype=float)
    mass2_array = np.zeros((10, 20), dtype=float)
    mass1_array[2, 2] = 1
    mass2_array[6, 12] = 1

    # Convert the arrays to actual DarSIA Images
    mass1 = darsia.Image(mass1_array, width=2, height=1, scalar=True)
    mass2 = darsia.Image(mass2_array, width=2, height=1, scalar=True)

    # Setup EMD object
    emd = darsia.EMD()

    # Determine the EMD
    distance = emd(mass1, mass2)

    # Test whether the distance is essentially the Euclidean distance
    euclidean_distance = ((0.4 * 1) ** 2 + (0.5 * 2) ** 2) ** 0.5
    assert np.isclose(distance, euclidean_distance)


def test_emd_2d_resize():
    """Test simple EMD for 2d distributions with physical dimensions."""

    # Create two mass distributions with identical mass, equal to 1
    mass1_array = np.zeros((10, 20), dtype=float)
    mass2_array = np.zeros((10, 20), dtype=float)
    mass1_array[2, 2] = 1
    mass2_array[6, 12] = 1

    # Convert the arrays to actual DarSIA Images
    mass1 = darsia.Image(mass1_array, width=2, height=1, scalar=True)
    mass2 = darsia.Image(mass2_array, width=2, height=1, scalar=True)

    # Setup EMD object, including a resize routine (needed for cv2.EMD)
    resize = darsia.Resize(
        **{
            "resize dsize": (20, 40),  # rows, cols
            "resize interpolation": "inter_area",
            "resize conservative": True,
        }
    )
    emd = darsia.EMD(resize)

    # Determine the EMD
    distance = emd(mass1, mass2)

    # Test whether the distance is essentially the Euclidean distance
    euclidean_distance = ((0.4 * 1) ** 2 + (0.5 * 2) ** 2) ** 0.5
    assert np.isclose(distance, euclidean_distance)
