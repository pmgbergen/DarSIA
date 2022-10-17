"""
Determine compaction of FluidFlower by comparing two different images.
"""

import matplotlib.pyplot as plt
import numpy as np
import skimage

import daria

# ! ----- Preliminaries - prepare two images for compaction analysis

# Paths to two images of interest. NOTE: These images are not part of the GH repo.
path_src = "/home/jakub/images/ift/benchmark/baseline/original/Baseline.jpg"
path_dst = "/home/jakub/images/ift/benchmark/well_test/from_description/pulse1.jpg"

# Setup config for cropping, and define geometry correction object
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
curvature_correction = daria.CurvatureCorrection(config)

# Create daria images with integrated cropping
img_src = daria.Image(img=path_src, curvature_correction=curvature_correction)
img_dst = daria.Image(img=path_dst, curvature_correction=curvature_correction)

# Scrutinze the color palette and align both images respectively. For this,
# define (inaccurate) ROIs in terms of pixel ranges (y and x) in which the
# color palette is contained.
translation_estimator = daria.TranslationEstimator()
roi_cc = (slice(0, 600), slice(0, 600))
translation_estimator.match_roi(
    img_src=img_src, img_dst=img_dst, roi_src=roi_cc, roi_dst=roi_cc
)

# Cut away the color palette - again define the ROI in terms of pixel ranges (y and x)
roi_crop = (slice(470, img_src.img.shape[0]), slice(60, 7940))
da_img_src = daria.extractROIPixel(img_src, roi_crop)
da_img_dst = daria.extractROIPixel(img_dst, roi_crop)


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
compaction_analysis = daria.CompactionAnalysis(da_img_src, **config["compaction"])

# Apply compaction analysis, providing the deformed image matching the baseline image.
# Also plot the deformation as vector field.
da_new_image = compaction_analysis(da_img_dst, plot=True, reverse=True)

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

deformation = compaction_analysis.evaluate(pts)
print("Deformation evaluated:")
print(deformation)
