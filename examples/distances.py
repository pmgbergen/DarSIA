"""
Example script showcasing the computation of the Earth
Mover's distance between darsia Images.

"""

import matplotlib.pyplot as plt
import numpy as np

import darsia

# Create two mass distributions with identical mass, equal to 1
mass1_array = np.zeros((100, 200), dtype=float)
mass2_array = np.zeros((100, 200), dtype=float)

mass1_array[20:30, 20:30] = 1
mass1_array /= np.sum(mass1_array)
mass2_array[60:80, 120:140] = 1
mass2_array *= np.sum(mass1_array) / np.sum(mass2_array)

assert np.sum(mass1_array) == np.sum(mass2_array)

# Convert the arrays to actual DarSIA Images
mass1 = darsia.Image(mass1_array, width=20, height=10, scalar=True, dim=2, series=False)
mass2 = darsia.Image(mass2_array, width=20, height=10, scalar=True, dim=2, series=False)

# Plot if requested
if False:
    plt.figure("Mass 1")
    plt.imshow(mass1.img)
    plt.figure("Mass 2")
    plt.imshow(mass2.img)
    plt.show()

# ! ---- Compute EMD using CV2

# NOTE cv2.EMD has strict memory limitations. Coarsening is required.

# Setup EMD object, including a resize routine (needed for cv2.EMD)
# Determine distance between different masses
resize = darsia.Resize(
    **{
        "resize shape": (75, 140),
        "resize interpolation": "inter_area",
        "resize conservative": True,
    }
)
emd = darsia.EMD(resize)

# Determine the EMD
distance = emd(mass1, mass2)
print(f"The cv2.emd distance between the two mass distributions is: {distance} meters.")
print()


# ! ---- Compute EMD using Newton

options = {
    "L": 1e-2,
    "num_iter": 1000,
    "tol": 1e-8,
    "tol_distance": 1e-5,
    "regularization": 1e-12,
    "scaling": 30,
    "depth": 10,
    "verbose": False,
}

distance_newton, _, _, _, status_newton = darsia.wasserstein_distance(
    mass1,
    mass2,
    method="Newton",
    options=options,
    plot_solution=True,
    return_solution=True,
)
print(
    f"""The Wasserstein distance (computed using the Newton method) between the two """
    f"""mass distributions is: {distance_newton} meters."""
)
print("Newton status:", status_newton, "\n")


# ! ---- Compute EMD using Bregman solver

options = {
    "L": 1e2,
    "num_iter": 1000,
    "tol": 1e-8,
    "regularization": 1e-12,
    "scaling": 30,
    "depth": 1,  # TODO increase depth.
    "verbose": False,
    "update_l": True,
    "tol_distance": 1e-6,
    "max_iter_increase_diff": 20,
    "l_factor": 2,
    "L_max": 1e6,
}
distance_bregman, _, _, _, status_bregman = darsia.wasserstein_distance(
    mass1,
    mass2,
    method="Bregman",
    options=options,
    plot_solution=True,
    return_solution=True,
)

print(
    f"""The Wasserstein distance (computed using Bregman split) between the two """
    f"""mass distributions is: {distance} meters."""
)
print("Bregman status:", status_bregman)
