"""Example demonstrating the I/O capabilities of
DarSIA, and in particular the reading functionality
to read multiple images from file, with various
file types.

"""

import os

import matplotlib.pyplot as plt
import numpy as np

import darsia

# ! ---- DICOM images

# Provide folder with dicom images
folder = f"{os.path.dirname(__file__)}/images"
dicom_paths = folder + "/dicom"

# Read spacetime dicom image
dicom_image = darsia.imread(dicom_paths)

# Print some of the specs
print(f"The dimensions of the space time dicom image are: {dicom_image.dimensions}")

# Display each time slice, integrated over the z-axis
for time_index in range(dicom_image.time_num):
   dicom_slice: darsia.OpticalImage = dicom_image.time_slice(time_index)
   z_axis, _ = darsia.interpret_indexing(axis="z", indexing="ijk")
   int_img = np.sum(dicom_slice.img, axis=z_axis)
   plt.imshow(int_img)
   plt.show()
