"""Example for Wasserstein computations moving a square to another location."""

import numpy as np
import pytest

import darsia

# ! ---- 2d version ----

# Coarse src image
rows = 10
cols = rows
src_square_2d = np.zeros((rows, cols), dtype=float)
src_square_2d[2:5, 2:5] = 1
meta_2d = {"width": 1, "height": 1, "space_dim": 2, "scalar": True}
src_image_2d = darsia.Image(src_square_2d, **meta_2d)

# Coarse dst image
dst_squares_2d = np.zeros((rows, cols), dtype=float)
dst_squares_2d[1:3, 1:2] = 1
dst_squares_2d[4:7, 7:9] = 1
dst_image_2d = darsia.Image(dst_squares_2d, **meta_2d)

# Rescale
shape_meta_2d = src_image_2d.shape_metadata()
geometry_2d = darsia.Geometry(**shape_meta_2d)
src_image_2d.img /= geometry_2d.integrate(src_image_2d)
dst_image_2d.img /= geometry_2d.integrate(dst_image_2d)

# Reference value for comparison
true_distance_2d = 0.379543951823

# ! ---- 3d version ----

# Coarse src image
pages = 1
src_square_3d = np.zeros((rows, cols, pages), dtype=float)
src_square_3d[2:5, 2:5, 0] = 1
meta_3d = {"dimensions": [1, 1, 1], "space_dim": 3, "series": False, "scalar": True}
src_image_3d = darsia.Image(src_square_3d, **meta_3d)

# Coarse dst image
dst_squares_3d = np.zeros((rows, cols, pages), dtype=float)
dst_squares_3d[1:3, 1:2, 0] = 1
dst_squares_3d[4:7, 7:9, 0] = 1
dst_image_3d = darsia.Image(dst_squares_3d, **meta_3d)

# Rescale
shape_meta_3d = src_image_3d.shape_metadata()
geometry_3d = darsia.Geometry(**shape_meta_3d)
src_image_3d.img /= geometry_3d.integrate(src_image_3d)
dst_image_3d.img /= geometry_3d.integrate(dst_image_3d)

# Reference value for comparison
true_distance_3d = 0.379543951823

# ! ---- Data set ----
src_image = {
    2: src_image_2d,
    3: src_image_3d,
}

dst_image = {
    2: dst_image_2d,
    3: dst_image_3d,
}

true_distance = {
    2: true_distance_2d,
    3: true_distance_3d,
}

# ! ---- Solver options ----

# Linearization
newton_options = {
    # Scheme
    "L": 1e9,
}
bregman_std_options = {
    # Scheme
    "L": 1,
}
bregman_adaptive_options = {
    # Scheme
    "L": 1,
    "bregman_update": lambda iter: iter % 20 == 0,
}
linearizations = {
    "newton": [newton_options],
    "bregman": [
        bregman_std_options,
        bregman_adaptive_options,
    ],
}

# Acceleration
off_aa = {
    # Nonlinear solver
    "aa_depth": 0,
    "aa_restart": None,
}
on_aa = {
    # Nonlinear solver
    "aa_depth": 5,
    "aa_restart": 5,
}
accelerations = [off_aa, on_aa]

# Linear solver
lu_options = {
    # Linear solver
    "linear_solver": "direct",
}
amg_options = {
    "linear_solver": "amg",
    "linear_solver_options": {
        "tol": 1e-8,
    },
}

ksp_options_amg = {
    "linear_solver": "ksp",
    "linear_solver_options": {
        "tol": 1e-8,
    "prec_schur": "hypre"
    },
}

ksp_options_direct = {
    "linear_solver": "ksp",
    "linear_solver_options": {
        "tol": 1e-8,
    },
    "prec_schur": "lu"
}

formulations = ["full", "pressure"]

solvers = [ksp_options_amg, ksp_options_direct]

# ! ---- Sinkhorn options ----
sinkhorn_methods = [
    "sinkhorn",
    "sinkhorn_log", 
    "sinkhorn_stabilized",
    "geomloss_sinkhorn_samples",
    #"geomloss_sinkhorn"
]
sinkhorn_regs = [1e0, 1e-1, 1e-2]



# General options
options = {
    # Method definition
    "l1_mode": "constant_cell_projection",
    "mobility_mode": "face_based",
    # Performance control
    "num_iter": 400,
    "tol_residual": 1e-10,
    "tol_increment": 1e-6,
    "tol_distance": 1e-10,
    "return_info": True,
    "verbose": False,
}

# ! ---- Tests ----


@pytest.mark.parametrize("a_key", range(len(accelerations)))
@pytest.mark.parametrize("s_key", range(len(solvers)))
@pytest.mark.parametrize("dim", [2, 3])
def test_newton(a_key, s_key, dim):
    """Test all combinations for Newton."""
    options.update(newton_options)
    options.update(accelerations[a_key])
    options.update(solvers[s_key])
    options.update({"formulation": formulations[0]})
    distance, info = darsia.wasserstein_distance(
        src_image[dim],
        dst_image[dim],
        options=options,
        method="newton",
    )
    assert np.isclose(distance, true_distance[dim], atol=1e-5)
    assert info["converged"]


@pytest.mark.parametrize("a_key", range(len(accelerations)))
@pytest.mark.parametrize("s_key", range(len(solvers)))
@pytest.mark.parametrize("dim", [2, 3])
def test_std_bregman(a_key, s_key, dim):
    """Test all combinations for std Bregman."""
    options.update(bregman_std_options)
    options.update(accelerations[a_key])
    options.update(solvers[s_key])
    options.update({"formulation": formulations[0]})
    distance, info = darsia.wasserstein_distance(
        src_image[dim],
        dst_image[dim],
        options=options,
        method="bregman",
    )
    assert np.isclose(distance, true_distance[dim], atol=1e-2)
    assert info["converged"]


@pytest.mark.parametrize("a_key", range(len(accelerations)))
@pytest.mark.parametrize("s_key", range(len(solvers)))
@pytest.mark.parametrize("dim", [2, 3])
def test_adaptive_bregman(a_key, s_key, dim):
    """Test all combinations for adaptive Bregman."""
    options.update(bregman_adaptive_options)
    options.update(accelerations[a_key])
    options.update(solvers[s_key])
    options.update({"formulation": formulations[0]})
    distance, info = darsia.wasserstein_distance(
        src_image[dim],
        dst_image[dim],
        options=options,
        method="bregman",
    )
    assert np.isclose(distance, true_distance[dim], atol=1e-5)
    assert info["converged"]

@pytest.mark.parametrize("method_key", range(len(sinkhorn_methods)))
@pytest.mark.parametrize("reg_key", range(len(sinkhorn_regs)))
@pytest.mark.parametrize("dim", [2, 3])
def test_sinkhorn(method_key, reg_key, dim):
    """Test all combinations for Newton."""
    options.update({"sinkhorn_algorithm": sinkhorn_methods[method_key]})
    options.update({"sinkhorn_regularization": sinkhorn_regs[reg_key]})
    options.update({"only_non_zeros": True})
    options.update({"num_iter": 1000})
    distance, info = darsia.wasserstein_distance(
        src_image[dim],
        dst_image[dim],
        options=options,
        method="sinkhorn",
    )
    eps = options["sinkhorn_regularization"]
    relative_err=abs(distance-true_distance[dim])/true_distance[dim]
    print(f"{sinkhorn_methods[method_key]} {eps=:.1e} {distance=:.2e} {true_distance[dim]=:.2e} {relative_err=:.1e} {info['niter']=}")
    assert info["converged"]
    assert np.isclose(distance, true_distance[dim], atol=sinkhorn_regs[reg_key])

if __name__ == "__main__":
    """Test all combinations for Newton."""
    dim = 3
    for i in [3]:
        print(f"Method: {sinkhorn_methods[i]}")
        for j in range(len(sinkhorn_regs)):
            test_sinkhorn(i, j, dim)