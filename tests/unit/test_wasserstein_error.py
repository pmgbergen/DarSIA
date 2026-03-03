"""
Tests to compare Wasserstein error to an analytical solution. This tests
the transport of two blocks of mass in 2D with different positions.
"""

import numpy as np
import pytest

import darsia

"""
Analytical solution to the Wasserstein distance problem for
two rectangular blocks of mass in 2D
(Given that they do not overlap).
"""


def analytic_solution(block1, block2, factor):
    """
    Parameters
    ----------
    block1: tuple containing (x_center, y_center, half_width) of block1
    block2: tuple containing (x_center, y_center, half_width) of block2
    factor: int refinement factor for the grid (1/factor is the voxel size)

    Returns
    -------
    distance: float, the analytical Wasserstein distance between the two blocks
    transport_map: np.ndarray of shape (factor*8, factor*8, 2), the transport
    map from block1 to block2
    """
    shape = (factor * 8, factor * 8)
    voxel_size = [1 / factor, 1 / factor]
    y, x = np.meshgrid(
        voxel_size[0] * (0.5 + np.arange(shape[0] - 1, -1, -1)),
        voxel_size[1] * (0.5 + np.arange(shape[1])),
        indexing="ij",
    )
    x_res = np.zeros_like(x, dtype=float)
    y_res = np.zeros_like(y, dtype=float)
    ones_x = np.ones_like(x, dtype=float)
    ones_y = np.ones_like(y, dtype=float)

    dx = block2[0] - block1[0]
    dy = block2[1] - block1[1]
    analytic_distance = np.sqrt(dx**2 + dy**2) * (2 * block1[2]) ** 2

    # Check that the blocks are the same size
    assert block1[2] == block2[2]
    width = block1[2]

    cos = dx / (np.sqrt(dx**2 + dy**2))
    sin = dy / (np.sqrt(dx**2 + dy**2))
    if cos == 0:
        cos = 1e-16
    if sin == 0:
        sin = 1e-16

    # Mask to denote the
    mask1 = (
        (block1[0] - width <= x)
        & (x <= block1[0] + width)
        & (block1[1] - width <= y)
        & (y <= block1[1] + width)
    )
    mask2 = (
        (block2[0] - width <= x)
        & (x <= block2[0] + width)
        & (block2[1] - width <= y)
        & (y <= block2[1] + width)
    )

    scaling1 = np.minimum(
        np.abs(x[mask1] - (block1[0] - np.sign(dx) * width)) / abs(cos),
        np.abs(y[mask1] - (block1[1] - np.sign(dy) * width)) / abs(sin),
    )
    x_res[mask1] = cos * scaling1
    y_res[mask1] = sin * scaling1
    scaling2 = np.minimum(
        np.abs(x[mask2] - (block2[0] + np.sign(dx) * width)) / abs(cos),
        np.abs(y[mask2] - (block2[1] + np.sign(dy) * width)) / abs(sin),
    )
    x_res[mask2], y_res[mask2] = cos * scaling2, sin * scaling2

    d = abs(cos) * width * np.abs(1 - np.abs(sin / cos))
    e = 2 * width * min(abs(cos), abs(sin))

    if dx == 0:
        dist = np.abs(x - block1[0])  # The blocks are horizontally aligned
    else:
        a = dy / dx
        b = block1[1] - a * block1[0]
        dist = np.abs(a * x - y + b) * abs(cos)
    mask3 = (
        np.logical_not(mask1)
        & np.logical_not(mask2)
        & (dist <= d + e)
        & (x >= min(block1[0] - width, block2[0] - width))
        & (x <= max(block1[0] + width, block2[0] + width))
        & (y >= min(block1[1] - width, block2[1] - width))
        & (y <= max(block1[1] + width, block2[1] + width))
    )
    mask3a = mask3 & (dist < d)
    mask3b = mask3 & (dist >= d)

    scaling3a = 2 / max(abs(cos), abs(sin))
    x_res[mask3a], y_res[mask3a] = (
        cos * scaling3a * ones_x[mask3a],
        sin * scaling3a * ones_y[mask3a],
    )
    scaling3b = scaling3a * 1 / e
    x_res[mask3b], y_res[mask3b] = cos * scaling3b * (
        d + e - dist[mask3b]
    ), sin * scaling3b * (d + e - dist[mask3b])

    return analytic_distance, np.stack([x_res, y_res], axis=2)


def L_dist(x, n=1):
    return (np.sum(np.abs(x) ** n)) ** (1 / n) / (x.shape[0] * x.shape[1])


angles = [0, np.pi / 4, 3 * np.pi / 8, np.pi / 2]
factors = [5, 20]

distances_angles = []
distances_factors = []


def block_test(factor, block1, block2, options, weights=None, method="newton"):
    dim = 2
    shape = (factor * 8, factor * 8)
    voxel_size = [1 / factor, 1 / factor]

    mass1_array = np.zeros(shape, dtype=float)
    mass2_array = np.zeros(shape, dtype=float)
    block1_help = [
        int(round(factor * (8 - block1[1] - block1[2]))),
        int(round(factor * (8 - block1[1] + block1[2]))),
        int(round(factor * (block1[0] - block1[2]))),
        int(round(factor * (block1[0] + block1[2]))),
    ]
    block2_help = [
        int(round(factor * (8 - block2[1] - block2[2]))),
        int(round(factor * (8 - block2[1] + block2[2]))),
        int(round(factor * (block2[0] - block2[2]))),
        int(round(factor * (block2[0] + block2[2]))),
    ]
    mass1_array[block1_help[0] : block1_help[1], block1_help[2] : block1_help[3]] = 1
    mass2_array[block2_help[0] : block2_help[1], block2_help[2] : block2_help[3]] = 1

    width = shape[1] * voxel_size[1]
    height = shape[0] * voxel_size[0]
    mass1 = darsia.Image(
        mass1_array,
        width=width,
        height=height,
        scalar=True,
        dim=dim,
        series=False,
    )

    mass2 = darsia.Image(
        mass2_array,
        width=width,
        height=height,
        scalar=True,
        dim=dim,
        series=False,
    )

    if weights is not None:
        weights = darsia.Image(
            weights,
            width=width,
            height=height,
            scalar=True,
            dim=dim,
            series=False,
        )

    distance, info = darsia.wasserstein_distance(
        mass1,
        mass2,
        weight=weights,
        method=method,
        options=options,
        plot_solution=True,
        return_solution=True,
    )
    if False:
        plot_options = {
            "resolution": 1,
            "save": False,
            "name": "squares",
            "dpi": 800,
        }
        darsia.plotting.plot_2d_wasserstein_distance(info, **plot_options)
        # darsia.plotting.to_vtk('test', flux)

    flux = info["flux"]

    flux_restructured = np.copy(flux)
    flux_restructured[:, :, 0] = -flux[:, :, 1]
    flux_restructured[:, :, 1] = flux[:, :, 0]

    return distance, flux_restructured


L = 1e-2
scaling = 3e1
regularization = 1e-16
num_iter = int(3e3)
tol = 1e-16

options = {
    "L": L,
    "num_iter": num_iter,
    "tol": tol,
    "tol_distance": 1e-2,
    "tol_increment": 1e-2,
    "tol_residual": 1e5,
    "regularization": regularization,
    "scaling": scaling,
    "depth": 0,
    "verbose": False,
    "return_info": True,
}
results = np.array(
    [
        [0.00416437, 0.0039679],
        [0.00474455, 0.00259308],
        [0.0048696, 0.00369112],
        [0.00416437, 0.0039679],
    ]
)


@pytest.mark.parametrize("a_key", range(len(angles)))
@pytest.mark.parametrize("f_key", range(len(factors)))
def test_wasserstein_block_transport(a_key, f_key):
    angle = angles[a_key]
    factor = factors[f_key]

    cos_approx = np.round(2 * np.cos(angle) * factor) / factor
    sin_approx = np.round(2 * np.sin(angle) * factor) / factor
    block1 = [4 - cos_approx, 4 - sin_approx, 1]
    block2 = [4 + cos_approx, 4 + sin_approx, 1]
    true_distance, true_flux = analytic_solution(block1, block2, factor)
    computed_distance, computed_flux = block_test(
        factor, block1, block2, options, method="newton"
    )
    error_distance = np.abs(true_distance - computed_distance) / true_distance

    assert np.isclose(error_distance, results[a_key, f_key], rtol=1e-6)


def make_wall(factor, L=6, K=10):
    shape = (factor * 8, factor * 8)
    voxel_size = [1 / factor, 1 / factor]
    y, x = np.meshgrid(
        voxel_size[0] * (0.5 + np.arange(shape[0] - 1, -1, -1)),
        voxel_size[1] * (0.5 + np.arange(shape[1])),
        indexing="ij",
    )

    wall = np.ones_like(x)

    # Assume the wall is vertical at x=4
    dx = x[0, 1] - x[0, 0]  # Assuming uniform spacing
    wall[(np.abs(x - 4) <= dx) & (np.abs(y - 4) <= L / 2)] = 0.5 * K / dx
    return wall


wall_results = [27.6990888314, 26.4370564953]


@pytest.mark.parametrize("f_key", range(len(factors)))
def test_wall(f_key):
    factor = factors[f_key]

    wall_array = make_wall(factor)

    block1 = [2, 4, 1]
    block2 = [6, 4, 1]

    dist, flux = block_test(factor, block1, block2, options, weights=wall_array)
    assert np.isclose(dist, wall_results[f_key], rtol=1e-6)
