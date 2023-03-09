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

# Read dicom images as 4d space-time, or list of 3d images.
dicom_image_4d = darsia.imread(dicom_paths, series=True)
dicom_images_3d = darsia.imread(dicom_paths, series=False)

# Print some of the specs
print(f"The dimensions of the space time dicom image are: {dicom_image_4d.dimensions}")

# Display each time slice, integrated over the z-axis
for time_index in range(dicom_image_4d.time_num):
   dicom_slice: darsia.OpticalImage = dicom_image_4d.time_slice(time_index)
   z_axis, _ = darsia.interpret_indexing(axis="z", indexing="ijk")
   int_img = np.sum(dicom_slice.img, axis=z_axis)
   plt.imshow(int_img)
   plt.show()
