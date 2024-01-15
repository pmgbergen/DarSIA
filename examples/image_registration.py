"""
Determine deformation of FluidFlower by comparing two different images.

The images correpsond to the baseline image of the official well test
performed under the benchmark, and one of the other baseline images,
most likely close to C1. Between these two images, settling/sedimentation
has occurred, i.e., to most degree the sand sunk from the src (well test)
to dst (C1 like) scenarios.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia

# ! ----- Preliminaries - prepare two images for image registration

# Paths to two images of interest. NOTE: These images are not part of the GH repo.
path_src = "/home/jakub/images/ift/more/welltest/pulse1.jpg"
path_dst = "/home/jakub/images/ift/benchmark/baseline/original/Baseline.jpg"

# Find corresponding arrays in RGB color channels
array_src = cv2.cvtColor(cv2.imread(path_src), cv2.COLOR_BGR2RGB)
array_dst = cv2.cvtColor(cv2.imread(path_dst), cv2.COLOR_BGR2RGB)

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
roi = darsia.make_voxel([[0, 0], [600, 600]])
drift_correction = darsia.DriftCorrection(base=array_src, config={"roi": roi})

# Create darsia images with integrated cropping. Note: the drift correction
# applied to img_src is without effect.
img_src = darsia.Image(
    img=array_src,
    transformations=[
        drift_correction,
        curvature_correction,
    ],
    width=2.8,
    height=1.5,
)

img_dst = darsia.Image(
    img=array_dst,
    transformations=[
        drift_correction,
        curvature_correction,
    ],
    width=2.8,
    height=1.5,
)

plt.figure("src")
plt.imshow(img_src.img)
plt.figure("dst")
plt.imshow(img_dst.img)
plt.show()

# Extract ROI to cut away the color palette. Use pixel ranges to crop the image.
roi_crop = (slice(470, img_src.img.shape[0]), slice(0, 7940))
da_img_src = img_src.subregion(voxels=roi_crop)
da_img_dst = img_dst.subregion(voxels=roi_crop)

# ! ----- Actual analysis: Determine the deformation between img_dst and aligned_img_src

# Define image registration tool
config["image registration"] = {
    # Define the number of patches in matrix indexin
    "N_patches": [10, 20],
    # Define a relative overlap, this makes it often slightly easier for the feature detection.
    "rel_overlap": 0.1,
    # Add some tuning parameters for the feature detection (these are actually the default
    # values and could be also omitted.
    "max_features": 200,
    "tol": 0.05,
}

construction_specialized = True

if construction_specialized:
    # One can directly address the special class with slightly more specialized routines.
    image_registration = darsia.DiffeomorphicImageRegistration(
        da_img_dst, **config["image registration"]
    )

    # Apply image registration, providing the deformed image matching the baseline image,
    # as well as the required translations on each patch, characterizing the total
    # deformation. Also plot the deformation as vector field.
    da_new_image, patch_translation = image_registration.call_with_output(
        da_img_src, plot_patch_translation=True, return_patch_translation=True
    )
    print("The centers of the 10 x 20 patches are translated:")
    print(patch_translation)

else:
    # Using the general interface:
    image_registration = darsia.ImageRegistration(
        da_img_dst, **config["image registration"]
    )

    # Apply image registration, providing the deformed image matching the baseline image,
    # as well as the required translations on each patch, characterizing the total
    # deformation. Also plot the deformation as vector field.
    da_new_image = image_registration(da_img_src)

# One can also apply the learned displacement, here to the input,
# but any other field can also be applied.
da_new_src_image = image_registration.apply(da_img_src)

# Plot the differences between the two original images and after the transformation.
fig, ax = plt.subplots(1, num=1)
ax.imshow(skimage.util.compare_images(da_img_src.img, da_img_dst.img, method="blend"))
fig, ax = plt.subplots(1, num=2)
ax.imshow(skimage.util.compare_images(da_img_dst.img, da_new_image.img, method="blend"))
fig, ax = plt.subplots(1, num=3)
ax.imshow(
    skimage.util.compare_images(da_new_src_image.img, da_new_image.img, method="blend")
)
plt.show()

## Plot the displacement - same as when calling the image registration
# image_registration.plot(scaling=1.0)

# It is also possible to evaluate the deformation approximation in arbitrary points.
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

deformation = image_registration.evaluate(pts)
print("Deformation evaluated:")
print(deformation)

# One can also use a patched ROI and evaluate the deformation in the patch centers.
# For this, we start extracting a roi, here we choose a box, which in the end
# corresponds to box B from the benchmark analysis. This it is sufficient to
# define two corner points of the box.
box_B = np.array([[0.0, 0.6], [1.2, 1.1]])

# and extract the corresponding ROI as darsia.Image (based on da_img_src):
img_box_B = da_img_src.subregion(coordinates=box_B)

# To double check the box, we plot the resulting box.
plt.figure("Box B")
plt.imshow(img_box_B.img)
plt.show()

# Now we patch box B, the number of patches is arbitrary (here chosen to be 5 x 3):
patched_box_B = darsia.Patches(img_box_B, [3, 5])

# The patch centers can be accessed - actually not required here, but these correspond
# to the subsequent deformations.
patch_centers_box_B = patched_box_B.global_centers_cartesian

# The deformation in the centers of these points can be obtained by evaluating the
# deformation map in a patch object. The result uses conventional matrix indexing.
# Entry [row,col] is associated to patch with coordinate [row,col], with [0,0]
# denoting the top left corner of the image/patch.
deformation_patch_centers_box_B = image_registration.evaluate(patched_box_B)

print("The deformation in the centers of the patches of Box B:")
print(deformation_patch_centers_box_B)
