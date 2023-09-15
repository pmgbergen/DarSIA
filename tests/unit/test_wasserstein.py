"""Example for Wasserstein computations moving a square to another location."""

import numpy as np
import pytest

import darsia

# Coarse src image
rows = 10
cols = rows
src_square = np.zeros((rows, cols), dtype=float)
src_square[2:5, 2:5] = 1
meta = {"width": 1, "height": 1, "space_dim": 2, "scalar": True}
src_image = darsia.Image(src_square, **meta)

# Coarse dst image
dst_squares = np.zeros((rows, cols), dtype=float)
dst_squares[1:3, 1:2] = 1
dst_squares[4:7, 7:9] = 1
dst_image = darsia.Image(dst_squares, **meta)

# Rescale
shape_meta = src_image.shape_metadata()
geometry = darsia.Geometry(**shape_meta)
src_image.img /= geometry.integrate(src_image)
dst_image.img /= geometry.integrate(dst_image)

# Reference value for comparison
true_distance = 0.379543951823

# Linearization
newton_options = {
    # Scheme
    "L": 1e-9,
}
bregman_options = {
    # Scheme
    "L": 1,
    "bregman_update_cond": lambda iter: iter % 20 == 0,
}
linearizations = {
    "newton": newton_options,
    "bregman": bregman_options,
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
    "linear_solver": "lu"
}
amg_options = {
    "linear_solver": "amg-potential",
    "linear_solver_tol": 1e-6,
}
solvers = [lu_options, amg_options]

# General options
options = {
    # Solver parameters
    "regularization": 1e-16,
    # Scheme
    "lumping": True,
    # Performance control
    "num_iter": 400,
    "tol_residual": 1e-10,
    "tol_increment": 1e-6,
    "tol_distance": 1e-10,
    # Output
    "verbose": False,
}


@pytest.mark.parametrize("a_key", range(len(accelerations)))
@pytest.mark.parametrize("s_key", range(len(solvers)))
@pytest.mark.parametrize("method", ["newton", "bregman"])
def test_newton(a_key, s_key, method):
    """Test all combinations."""
    options.update(linearizations[method])
    options.update(accelerations[a_key])
    options.update(solvers[s_key])
    distance, _, _, _, status = darsia.wasserstein_distance(
        src_image, dst_image, options=options, method=method, return_solution=True
    )
    assert np.isclose(distance, true_distance, atol=1e-5)
    assert status["converged"]
