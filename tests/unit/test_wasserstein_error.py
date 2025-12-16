"""
Tests to compare Wasserstein error to an analytical solution. This tests the transport of two blocks of mass in 2D
with different positions.
"""

import numpy as np
import pytest
import darsia

from src.darsia.utils.wasserstein_analytical_sol_block import analytic_solution


def L_dist(x, n=1):
    return (np.sum(np.abs(x)**n))**(1/n)/(x.shape[0]*x.shape[1])

angles = [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2]
factors = [5, 10, 20, 40, 80]

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
    "tol_distance": 1e-3,
    "tol_increment": 1e-3,
    "tol_residual": 1e5,
    "regularization": regularization,
    "scaling": scaling,
    "depth": 0,
    "verbose": True,
    "return_info": True
}

def test_wasserstein_block_transport(angle, factor):
    cos_approx = np.round(2 * np.cos(angle) * factor) / factor
    sin_approx = np.round(2 * np.sin(angle) * factor) / factor
    block1 = [4 - cos_approx, 4 - sin_approx, 1]
    block2 = [4 + cos_approx, 4 + sin_approx, 1]
    true_distance, true_flux = analytic_solution(block1, block2, factor)
    computed_distance, computed_flux = block_test(factor, block1, block2, options, method="newton")
    error_distance = np.abs(true_distance - computed_distance) / true_distance
    error_flux = L_dist(true_flux - computed_flux, n=1) / true_distance
    print(f"block 1: {block1}, block2: {block2}")
    print(f"True distance: {true_distance}, computed distance: {computed_distance}")
    print(f"Error distance: {error_distance}")
    print(f"Error flux: {error_flux}")
    return

test_wasserstein_block_transport(angles[2], factors[2])
