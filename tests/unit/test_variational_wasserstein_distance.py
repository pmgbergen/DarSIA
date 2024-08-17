"""Unit test of VariationalWassersteinDistance metods."""

import numpy as np

import darsia

# ! ---- 2d version ----

# Coarse src image
rows = 2
cols = rows
src_square_2d = np.zeros((rows, cols), dtype=float)
src_square_2d[0, 0] = 1
meta_2d = {"width": 1, "height": 1, "space_dim": 2, "scalar": True}
src_image_2d = darsia.Image(src_square_2d, **meta_2d)

# Coarse dst image
dst_squares_2d = np.zeros((rows, cols), dtype=float)
dst_squares_2d[1, 1] = 1
dst_image_2d = darsia.Image(dst_squares_2d, **meta_2d)

# Rescale
shape_meta_2d = src_image_2d.shape_metadata()
geometry_2d = darsia.Geometry(**shape_meta_2d)
src_image_2d.img /= geometry_2d.integrate(src_image_2d)
dst_image_2d.img /= geometry_2d.integrate(dst_image_2d)

options = {
    # Method definition
    "l1_mode": "raviart_thomas",
}

grid = darsia.generate_grid(dst_image_2d)
w1 = darsia.VariationalWassersteinDistance(grid, options=options)
flat_flux = np.zeros(grid.num_faces, dtype=float)
flat_flux[grid.faces[0]] = 1
flat_flux[grid.faces[1]] = 2


def test_vector_face_flux_norm_cell_based():
    """Compare with the manually determined exact cell based face mobility."""
    # NOTE the coarse tolerance due to such large quadrature error
    assert np.allclose(
        w1.vector_face_flux_norm(flat_flux, "cell_based"), 4 * [1.1865], atol=1e-1
    )


def test_vector_face_flux_norm_subcell_based():
    """Compare with the manually determined exact subcell based face mobility."""
    assert np.allclose(
        w1.vector_face_flux_norm(flat_flux, "subcell_based"),
        2 * [4 / (2 / 1 + 2 / 5**0.5)] + 2 * [4 / (2 / 2 + 2 / 5**0.5)],
    )


def test_vector_face_flux_norm_face_based():
    """Compare with the manually determined exact face based face mobility."""
    assert np.allclose(
        w1.vector_face_flux_norm(flat_flux, "face_based"),
        2 * [2**0.5] + 2 * [(0.5**2 + 2**2) ** 0.5],
        atol=1e-3,
    )
