"""Unit tests for finite volume utilities."""

import numpy as np

import darsia


def test_divergence_2d():
    # Create divergence matrix
    grid = darsia.Grid(shape=(4, 5), voxel_size=[0.5, 0.25])
    divergence = darsia.FVDivergence(grid).mat.todense()

    # Check shape
    assert np.allclose(divergence.shape, (grid.num_cells, grid.num_faces))

    # Check values in corner cells
    assert np.allclose(np.nonzero(divergence[0])[1], [0, 15])
    assert np.allclose(divergence[0, np.array([0, 15])], [0.25, 0.5])

    assert np.allclose(np.nonzero(divergence[4])[1], [4, 18])
    assert np.allclose(divergence[4, np.array([4, 18])], [0.25, -0.5])

    assert np.allclose(np.nonzero(divergence[15])[1], [10, 27])
    assert np.allclose(divergence[15, np.array([10, 27])], [-0.25, 0.5])

    assert np.allclose(np.nonzero(divergence[19])[1], [14, 30])
    assert np.allclose(divergence[19, np.array([14, 30])], [-0.25, -0.5])

    # Check value for interior cell
    assert np.allclose(np.nonzero(divergence[6])[1], [1, 6, 19, 20])
    assert np.allclose(
        divergence[6, np.array([1, 6, 19, 20])], [-0.25, 0.25, -0.5, 0.5]
    )
