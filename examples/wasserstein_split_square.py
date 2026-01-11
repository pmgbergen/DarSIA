"""Example for Wasserstein computations moving a square to another location."""

import darsia
import numpy as np
import time

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

# Refine
lvl = 4
resize = darsia.Resize(
    **{
        "resize shape": (rows * 2**lvl, cols * 2**lvl),
        "resize interpolation": "inter_nearest",
    }
)
src_image = resize(src_image)
dst_image = resize(dst_image)

if False:
    src_image.show()
    dst_image.show()

## Some values to test Newton
# L_values = [1e-9]
# depths = [0]
# restart = [0]
# linear_solver = ["lu"]

# def L_index(case):
#    return case % len(L_values)
#
#
# def depth_index(case):
#    return int(((case - L_index(case)) / len(L_values)) % len(depths))
#
#
# def restart_index(case):
#    return int(
#        (
#            (case - L_index(case) - len(L_values) * depth_index(case))
#            / (len(L_values) * len(depths))
#        )
#        % len(restart)
#    )
#
#
# def lumping_index(case):
#    return int(
#        (
#            (
#                case
#                - L_index(case)
#                - len(L_values) * depth_index(case)
#                - len(L_values) * len(depths) * restart_index(case)
#            )
#            / (len(L_values) * len(depths) * len(restart))
#        )
#        % len(lumping)
#    )

options = {
    # Performance control
    "num_iter": 100,
    "tol_residual": 1e-10,
    "tol_increment": 1e-6,
    "tol_distance": 1e-10,
    "bregman_update": lambda iter: iter in [0, 20],
    # Linear solver
    "linear_solver": "direct",
    "linear_solver_options": {"tol": 1e-6},
    # Nonlinear solver
    # "aa_depth": 10,
    # "aa_restart": 10,
    # Output
    "verbose": True,
    "return_info": True,
    "formulation": "flux_reduced",
}

kwargs = {
    "method": "newton",
    "options": options,
}

# CV2 Earth Mover's Distance
tic = time.time()
# cv2_distance = darsia.wasserstein_distance(
#    src_image, dst_image, **{"method": "cv2.emd"}
# )
# print(f"CV2.EMD distance: {cv2_distance}")
# print(f"CV2.EMD elapsed time: {time.time() - tic}")

# DarSIA Wasserstein distance
distance, info = darsia.wasserstein_distance(src_image, dst_image, **kwargs)

# Monitor performance
print(f"Distance: {distance}")
print(f"Peak memory consumption: {info['peak_memory_consumption']}")
print(f"Elapsed time assembly: {info['timings']['assemble']}")
print(f"Elapsed time setup: {info['timings']['setup']}")
print(f"Elapsed time solve: {info['timings']['solve']}")
print(f"Elapsed time acceleration: {info['timings']['acceleration']}")
print(f"Elapsed time total: {info['timings']['total']}")
print()

# Plot solution - optimized for refinement_level = 4
grid = info["grid"]
mass_diff = info["mass_diff"]
flux = info["flux"]
pressure = info["pressure"]
transport_density = info["transport_density"]
plot_options = {
    "resolution": 1,
    "save": False,
    "name": "squares",
    "dpi": 800,
}
darsia.plotting.plot_2d_wasserstein_distance(info, **plot_options)
darsia.wasserstein_distance_to_vtk("squares", info)
