"""
Determine compaction of FluidFlower by comparing two different images.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import skimage

import daria

# ! ----- Preliminaries - prepare two images for compaction analysis

# Paths to two images of interest
path_src = "/home/jakub/images/ift/benchmark/baseline/original/Baseline.jpg"
path_dst = "/home/jakub/images/ift/benchmark/well_test/pulse1.jpg"

# Read images from file
img_src = cv2.imread(str(Path(path_src)))
img_dst = cv2.imread(str(Path(path_dst)))

# Convert to RGB space
img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
img_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2RGB)

# Initialize the object which will be responsible for aligning patches, and
# evenutally determining a compaction map.
translationEstimator = daria.TranslationEstimator(max_features=200, tol=0.05)

# ! ----- Step 1: Prepare images

# ! ----- Step 1a: Prepare imagesAlign both images wrt color palette

# Define (inaccurate) ROIs in which the color palette is contained - same here.
roi_cc = (slice(0, 600), slice(0, 600))

# Scrutinze the color palette and align both images respectively.
aligned_img_src = translationEstimator.match_roi(
    img_src=img_src, img_dst=img_dst, roi_src=roi_cc, roi_dst=roi_cc
)

# Plot the original two images and the corrected ones - only for demonstration.
if False:
    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(skimage.util.compare_images(img_src, img_dst, method="blend"))
    ax[1].imshow(skimage.util.compare_images(aligned_img_src, img_dst, method="blend"))
    plt.show()

# ! ----- Step 1b: Extract a quarilateral ROI

# Extract quad ROI with known physical size
config_crop = {
    "pts_src": [[52, 0], [64, 4429], [7896, 4429], [7891, 0]],
    # Specify the true dimensions of the reference points - known as they are points on
    # the laser grid
    "width": 2.8,
    "height": 1.5,
    "in meters": True,
}

img_cropped_src = daria.extract_quadrilateral_ROI(aligned_img_src, **config_crop)
img_cropped_dst = daria.extract_quadrilateral_ROI(img_dst, **config_crop)

# ! ----- Step 1c: Cut away the color palette
final_height = 1.5 * (img_cropped_src.shape[0] - 470) / img_cropped_src.shape[0]
img_cropped_src = img_cropped_src[470:, 60:7940]
img_cropped_dst = img_cropped_dst[470:, 60:7940]

# ! ----- Step 2: Determine the compaction between img_dst and aligned_img_src

# Make daria images
da_img_src = daria.Image(img_cropped_src, width=2.8, height=final_height)
da_img_dst = daria.Image(img_cropped_dst, width=2.8, height=final_height)

# Define compaction analysis tool
N_patches = [20, 10]
rel_overlap = 0.1
translationAnalysis = daria.TranslationAnalysis(
    da_img_src,
    N_patches=N_patches,
    rel_overlap=rel_overlap,
    translationEstimator=translationEstimator,
)

# Determine translation in patch centers
translationAnalysis.find_translation(da_img_dst, units=["metric", "pixel"])
translationAnalysis.translate_centers(da_img_dst, plot_translation=True)

# ! ---- Step 3: Warp dst image back onto the baseline image

# Determine translation and translate the dst image
translationAnalysis.find_translation(da_img_dst, units=["pixel", "pixel"])
da_new_image = translationAnalysis.translate_image(
    da_img_dst, reverse=True, plot_transformation=True
)

# Plot the differences between the two original images and after the transformation.
if True:
    fig, ax = plt.subplots(1, num=1)
    ax.imshow(
        skimage.util.compare_images(da_img_src.img, da_img_dst.img, method="blend")
    )
    fig, ax = plt.subplots(1, num=2)
    ax.imshow(
        skimage.util.compare_images(da_img_src.img, da_new_image.img, method="blend")
    )
    fig, ax = plt.subplots(1, num=3)
    ax.imshow(da_img_src.img)
    fig, ax = plt.subplots(1, num=4)
    ax.imshow(da_img_dst.img)
    fig, ax = plt.subplots(1, num=5)
    ax.imshow(da_new_image.img)
    plt.show()
