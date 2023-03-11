"""Example demonstrating the I/O capabilities of
DarSIA, and in particular the reading functionality
to read multiple images from file, with various
file types.

"""

import os

import darsia

# ! ---- DICOM images

# Provide folder with dicom images
folder = f"{os.path.dirname(__file__)}/images"
dicom_paths = folder + "/dicom"

# Read dicom images as 4d space-time, or list of 3d images.
dicom_image_4d = darsia.imread(dicom_paths, dim=3)

# Print some of the specs
print(f"The dimensions of the space time dicom image are: {dicom_image_4d.dimensions}")

# Restrict to relevant ROI in z direction.
roi_z = slice(90, 120)
dicom_image_roi: darsia.OpticalImage = dicom_image_4d.subregion(
    voxels=(roi_z, slice(0, None), slice(0, None))
)

# Apply vertical averaging
vertical_averaging = darsia.AxisAveraging(axis="z", dim=3)
dicom_image_flat: darsia.OpticalImage = vertical_averaging(dicom_image_roi)

# Display each time slice, integrated over the z-axis
for time_index in range(dicom_image_flat.time_num):
    dicom_slice: darsia.OpticalImage = dicom_image_flat.time_slice(time_index)
    dicom_slice.show(time_index, 3)
