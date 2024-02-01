"""Example demonstrating the I/O capabilities of DarSIA, and in particular the reading
functionality to read multiple images from file, with various file types. In addition,
the possibilities for plotting 2d and 3d data via DarSIA is demonstrated.

"""

import os

import darsia

# ! ---- Read 4d DICOM images

# Provide folder with dicom images
folder = f"{os.path.dirname(__file__)}/images"
dicom_paths = folder + "/dicom"

# Read dicom images as 4d space-time, or list of 3d images.
dicom_image_4d = darsia.imread(dicom_paths, dim=3)

# Print some of the specs
print(f"The dimensions of the space time dicom image are: {dicom_image_4d.dimensions}")
print(f"The time series holds {dicom_image_4d.time_num} images.")

# ! ---- Create various reduced versions

# Extract 3d, one time slice
dicom_image_3d_slice = dicom_image_4d.time_slice(9)

# Extract 3d, time interval (just with two elements for simplicity).
dicom_image_4d_interval = dicom_image_4d.time_interval(slice(8, 10))

# Retrict 4d image to ROI in z-direction
roi_z = slice(90, 120)
dicom_image_4d_interval_roi: darsia.OpticalImage = dicom_image_4d_interval.subregion(
    (roi_z, slice(0, None), slice(0, None))
)

# Extract 3d series, with the z-direction vertically averaged, i.e., 2d in space.
vertical_averaging = darsia.AxisReduction(axis="z", dim=3)
dicom_image_3d_series = vertical_averaging(dicom_image_4d_interval_roi)

# ! ---- Demonstrate various ways of how to plot the 3d and 4d images.

# Consider four plotting examples:
#   1. For single 3d slice, plot 3d scatter plot, and 2d scatter plot side views, with
#      matplotlib.
#   2. For single 3d slive, plot 3d voxel plot, and 2d voxel plot side views (obtained
#      through integration in normal direction), with matplotlib.
#   3. Repeat mix of example 1 and 2, but for a time series, and use plotly.
#   4. For a time series of 2d images, plot the canonical voxel (pixel) images, with
#      matplotlib.

# Define fixed optione. For illustration purposes, the 3d data will have to be
# thresholded. Use relative threshold values here. Also add scaling options (only
# meaningful for scatter plots, and omitted otherwise).
options: dict = {
    "threshold": 0.05,
    "relative": True,
    "scaling": 10,
    # "duration": 180, # for testing purposes meaningful
}

## Example 1.
# dicom_image_3d_slice.show(
#    mode="matplotlib", view="scatter", side_view="scatter", **options
# )
#
## Example 2.
# dicom_image_3d_slice.show(mode="matplotlib", view="voxel", side_view="voxel", **options)

## Example 3.
# dicom_image_4d_interval.show(
#    mode="plotly", view="scatter", side_view="voxel", **options
# )

# Example 4.
dicom_image_3d_series.show(mode="matplotlib")
# dicom_image_3d_series.show(mode = "plotly")

# ! ---- Tailored corrections


# Define tailored "correction" routine, providing a 3d space-time, flat dicom
# image in darsia format.
def tailored_transformation(image_4d: darsia.ScalarImage) -> darsia.Image:
    """Correction routine tailored to block A.

    Args:
        image (darsia.Image): scalar 4d image

    Returns:
        darsia.Image: 3d flat scalar image

    """
    # 1. Rotation. Rotate normal to y-axis.
    rotation = darsia.RotationCorrection(
        anchor=[77, 0, 10], rotations=[(-0.03666321317606485, "y")], dim=3
    )
    rotated_image_4d = rotation(image_4d)

    # 2. Crop. Restrict to relevant ROI in z direction.
    roi_z = slice(90, 120)
    roi_rotated_image_4d: darsia.ScalarImage = rotated_image_4d.subregion(
        (roi_z, slice(0, None), slice(0, None))
    )

    # 3. Flatten image. Apply vertical averaging.
    vertical_averaging = darsia.AxisReduction(axis="z", dim=3)
    flat_image_3d: darsia.ScalarImage = vertical_averaging(roi_rotated_image_4d)

    return flat_image_3d


transformed_image = tailored_transformation(dicom_image_3d_slice)
transformed_image.show()
