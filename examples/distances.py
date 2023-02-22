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
mass1 = darsia.Image(mass1_array, width=2, height=1)

mass2 = darsia.Image(mass2_array, width=2, height=1)

# Plot if requested
if False:
    plt.figure("Mass 1")
    plt.imshow(mass1.img)
    plt.figure("Mass 2")
    plt.imshow(mass2.img)
    plt.show()

# Setup EMD object, including a resize routine (needed for cv2.EMD)
# Determine distance between different masses
resize = darsia.Resize(
    **{
        "resize dsize": (140, 75),  # rows, cols
        "resize interpolation": "inter_area",
        "resize conservative": True,
    }
)
emd = darsia.EMD(resize)

# Determine the EMD
distance = emd(mass1, mass2)
print(f"The distance between the two mass distributions is: {distance} meters.")
