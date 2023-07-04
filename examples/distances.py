"""
Example script showcasing the computation of the Earth
Mover's distance between darsia Images.

"""

import matplotlib.pyplot as plt
import numpy as np

import darsia

# ! ---- Create two mass distributions

# Create two mass distributions with identical mass, equal to 1
mass1_array = np.zeros((80, 140), dtype=float)
mass2_array = np.zeros((80, 140), dtype=float)

mass1_array[20:30, 20:30] = 1
mass2_array[60:80, 120:140] = 1
mass1_array /= np.sum(mass1_array)
mass2_array *= np.sum(mass1_array) / np.sum(mass2_array)
mass1_array *= 2
mass2_array *= 2

# NOTE: The mass is not 1, neither in the sum, nor in the integrated sense.

# Check that the mass is the same
assert np.sum(mass1_array) == np.sum(mass2_array)

# Convert the arrays to actual DarSIA Images. Here the size of the pixel is equal in x
# and y, and the pixel size is 0.5 (length units, here interpreted as meters).
mass1 = darsia.Image(mass1_array, width=70, height=40, scalar=True, dim=2, series=False)
mass2 = darsia.Image(mass2_array, width=70, height=40, scalar=True, dim=2, series=False)

# Plot if requested
if False:
    plt.figure("Mass 1")
    plt.imshow(mass1.img)
    plt.figure("Mass 2")
    plt.imshow(mass2.img)
    plt.show()

# ! ---- Compute EMD using CV2

# Determine the EMD. NOTE: cv2.EMD suffers from memory issues for large images.
# The images here are chosen such that no issue should arise.
distance_cv2 = darsia.wasserstein_distance(
    mass1,
    mass2,
    method="cv2.emd",
)
print(
    f"The cv2.emd distance between the two mass distributions is: {distance_cv2} meters."
)
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
    "lumping": True,
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
    "num_iter": 10000,
    "tol": 1e-8,
    "regularization": 1e-12,
    "scaling": 30,
    "depth": 1,
    "verbose": False,
    "update_l": True,
    "tol_distance": 1e-6,
    "max_iter_increase_diff": 20,
    "l_factor": 1.2,
    "L_max": 1e8,
    "lumping": True,
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
    f"""mass distributions is: {distance_bregman} meters."""
)
print("Bregman status:", status_bregman)
