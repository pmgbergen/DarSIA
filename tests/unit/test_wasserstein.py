"""Example for Wasserstein computations moving a square to another location."""

import numpy as np
import pytest

import darsia

try:
    import petsc4py

    HAVE_PETSC = True
except ImportError:
    HAVE_PETSC = False

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
    "formulation": "full",
}
amg_options = {
    "linear_solver": "amg",
    "linear_solver_options": {
        "atol": 1e-8,
    },
    "formulation": "pressure",
}
ksp_direct_options = {
    "linear_solver": "ksp",
    "linear_solver_options": {
        "approach": "direct",
    },
    "formulation": "pressure",
}
ksp_krylov_options = {
    "linear_solver": "ksp",
    "linear_solver_options": {
        "rtol": 1e-8,
        "atol": 1e-9,
        "approach": "gmres",
        "petsc_options": {
            # "ksp_view": None,
            "pc_type": "hypre",
        },
    },
    "formulation": "flux_reduced",
}

ksp_block_krylov_options = {
    "linear_solver": "ksp",
    "linear_solver_options": {
        "rtol": 1e-8,
        "approach": "gmres",
        "pc_type": "hypre",
    },
    "formulation": "full",
}


solvers = (
    [lu_options, amg_options]
    + [
        ksp_direct_options,
        ksp_krylov_options,
        ksp_block_krylov_options,
    ]
    if HAVE_PETSC
    else []
)


# General options
options = {
    # Method definition
    "l1_mode": darsia.L1Mode.CONSTANT_CELL_PROJECTION,
    "mobility_mode": darsia.MobilityMode.FACE_BASED,
    # Performance control
    "num_iter": 400,
    "tol_residual": 1e-10,
    "tol_increment": 1e-6,
    "tol_distance": 1e-10,
    "return_info": True,
    "regularization": 1e-15,
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
    distance, info = darsia.wasserstein_distance(
        src_image[dim],
        dst_image[dim],
        options=options,
        method="bregman",
    )
    assert np.isclose(distance, true_distance[dim], atol=1e-5)
    assert info["converged"]
