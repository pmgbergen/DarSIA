"""
Determine compaction of FluidFlower by comparing two different images.
"""

import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia

# ! ----- Preliminaries - prepare two images for compaction analysis

# Paths to two images of interest. NOTE: These images are not part of the GH repo.
path_src = "/home/jakub/images/ift/benchmark/baseline/original/Baseline.jpg"
path_dst = "/home/jakub/images/ift/benchmark/well_test/from_description/pulse1.jpg"

# Setup curvature correction (here only cropping)
config = {
    "crop": {
        # Define the pixel values (x,y) of the corners of the ROI.
        # Start at top left corner and then continue counterclockwise.
        "pts_src": [[52, 0], [64, 4429], [7896, 4429], [7891, 0]],
        # Specify the true dimensions of the reference points
        "width": 2.8,
        "height": 1.5,
    }
}
curvature_correction = darsia.CurvatureCorrection(config)

# Setup drift correction taking care of moving camera in between taking photos.
# Use the color checker as reference in both images, and make the src image
# the anker.
roi_cc = (slice(0, 600), slice(0, 600))
drift_correction = darsia.DriftCorrection(base=path_src, roi=roi_cc)

# Create darsia images with integrated cropping. Note: the drift correction
# applied to img_src is without effect.
img_src = darsia.Image(
    img=path_src,
    drift_correction=drift_correction,
    curvature_correction=curvature_correction,
)
img_dst = darsia.Image(
    img=path_dst,
    drift_correction=drift_correction,
    curvature_correction=curvature_correction,
)

# Extract ROI to cut away the color palette. Use pixel ranges to crop the image.
roi_crop = (slice(470, img_src.img.shape[0]), slice(60, 7940))
da_img_src = darsia.extractROIPixel(img_src, roi_crop)
da_img_dst = darsia.extractROIPixel(img_dst, roi_crop)

# ! ----- Actual analysis: Determine the compaction between img_dst and aligned_img_src

# Define compaction analysis tool
config["compaction"] = {
    # Define the number of patches in x and y directions
    "N_patches": [20, 10],
    # Define a relative overlap, this makes it often slightly easier for the feature detection.
    "rel_overlap": 0.1,
    # Add some tuning parameters for the feature detection (these are actually the default
    # values and could be also omitted.
    "max_features": 200,
    "tol": 0.05,
}
compaction_analysis = darsia.CompactionAnalysis(da_img_src, **config["compaction"])

# Apply compaction analysis, providing the deformed image matching the baseline image.
# Also plot the deformation as vector field.
da_new_image, patch_translation = compaction_analysis(
    da_img_dst, reverse=True, plot_patch_translation=True, return_patch_translation=True
)

print("The centers of the 20 x 10 patches are translated:")
print(patch_translation)

# Plot the differences between the two original images and after the transformation.
fig, ax = plt.subplots(1, num=1)
ax.imshow(skimage.util.compare_images(da_img_src.img, da_img_dst.img, method="blend"))
fig, ax = plt.subplots(1, num=2)
ax.imshow(skimage.util.compare_images(da_img_src.img, da_new_image.img, method="blend"))
plt.show()

# It is also possible to evaluate the compaction approximation in arbitrary points.
# For instance, consider 4 points in metric coordinates (provided in x, y format):
pts = np.array(
    [
        [0.2, 1.4],
        [0.5, 0.5],
        [0.5, 1.2],
        [1.2, 0.75],
        [2.3, 1.1],
    ]
)
print("Consider the points:")
print(pts)

deformation = compaction_analysis.evaluate(pts)
print("Deformation evaluated:")
print(deformation)

# One can also use a patched ROI and evaluate the deformation in the patch centers.
# For this, we start extracting a roi, here we choose a box, which in the end
# corresponds to box B from the benchmark analysis. This it is sufficient to
# define two corner points of the box:
box_B = np.array([[0.0, 1.2], [1.1, 0.6]])

# and extract the corresponding ROI as darsia.Image (based on da_img_src):
img_box_B = darsia.extractROI(da_img_src, box_B)

# To double check the box, we plot the resulting box.
plt.figure("Box B")
plt.imshow(img_box_B.img)
plt.show()

# Now we patch box B, the number of patches is arbitrary (here chosen to be 5 x 3):
patches_box_B = darsia.Patches(img_box_B, 5, 3)

# The patch centers can be accessed:
patch_centers_box_B = patches_box_B.global_centers_cartesian

# The deformation in the centers of these points can be obtained by evaluating the
# deformation map in a similar fashion as above. The results comes in a matrix.
# Entry [row,col] is associated to patch with coordinate [row,col] using the
# conventional matrix indexing.
deformation_patch_centers_box_B = compaction_analysis.evaluate(patch_centers_box_B)
print("The deformation in the centers of the patches of Box B:")
print(deformation_patch_centers_box_B)
