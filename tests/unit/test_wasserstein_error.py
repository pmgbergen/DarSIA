"""
Tests to compare Wasserstein error to an analytical solution. This tests the transport of two blocks of mass in 2D
with different positions.
"""

import numpy as np
import pytest
import darsia
from darsia.utils.wasserstein_analytical_sol_block import analytic_solution





def L_dist(x, n=1):
    return (np.sum(np.abs(x)**n))**(1/n)/(x.shape[0]*x.shape[1])

angles = [0, np.pi/4, 3*np.pi/8, np.pi/2]
factors = [5, 20]

distances_angles = []
distances_factors = []

def block_test(factor, block1, block2, options, weights=None, method="newton"):
    dim = 2
    shape = (factor * 8, factor * 8)
    voxel_size = [1 / factor, 1 / factor]

    mass1_array = np.zeros(shape, dtype=float)
    mass2_array = np.zeros(shape, dtype=float)
    block1_help = [int(round(factor * (8-block1[1]-block1[2]))), int(round(factor * (8-block1[1]+block1[2]))), int(round(factor * (block1[0]-block1[2]))), int(round(factor * (block1[0]+block1[2])))]
    block2_help = [int(round(factor * (8-block2[1]-block2[2]))), int(round(factor * (8-block2[1]+block2[2]))), int(round(factor * (block2[0]-block2[2]))), int(round(factor * (block2[0]+block2[2])))]
    mass1_array[block1_help[0]:block1_help[1], block1_help[2]:block1_help[3]] = 1
    mass2_array[block2_help[0]:block2_help[1], block2_help[2]:block2_help[3]] = 1

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


L=1e-2
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
    "return_info": True
}
results = np.array([[ 0.00416437,  0.0039679 ],
 [ 0.00474455,  0.00259308],
 [ 0.0048696,   0.00369112],
 [ 0.00416437,  0.0039679 ],])

@pytest.mark.parametrize('a_key', range(len(angles)))
@pytest.mark.parametrize('f_key', range(len(factors)))
def test_wasserstein_block_transport(a_key, f_key):
    angle = angles[a_key]
    factor = factors[f_key]

    cos_approx = np.round(2 * np.cos(angle) * factor) / factor
    sin_approx = np.round(2 * np.sin(angle) * factor) / factor
    block1 = [4 - cos_approx, 4 - sin_approx, 1]
    block2 = [4 + cos_approx, 4 + sin_approx, 1]
    true_distance, true_flux = analytic_solution(block1, block2, factor)
    computed_distance, computed_flux = block_test(factor, block1, block2, options, method="newton")
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
    dx = x[0, 1] - x[0, 0] # Assuming uniform spacing
    wall[(np.abs(x - 4) <= dx) & (np.abs(y - 4) <= L/2)] = 0.5*K/dx
    return wall
wall_results = [27.6990888314, 26.4370564953]

@pytest.mark.parametrize('f_key', range(len(factors)))
def test_wall(f_key):
    factor = factors[f_key]
    dim = 2
    shape = (factor * 8, factor * 8)
    voxel_size = [1 / factor, 1 / factor]

    wall_array = make_wall(factor)

    block1 = [2, 4, 1]
    block2 = [6, 4, 1]

    dist, flux = block_test(factor, block1, block2, options, weights=wall_array)
    assert np.isclose(dist, wall_results[f_key], rtol=1e-6)
