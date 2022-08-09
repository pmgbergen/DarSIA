"""
The objective of this example is to showcase how to use DaRIA to identify sandlayer
in the FluidFlower Baseline image.

For this we use resizing without aliasing (to transform the fine scale grains to noise),
apply some denoising/filtering, and k-means clustering.

Clustering, based on https://www.thepythoncode.com/article/kmeans-for-image-segmentation-opencv-python
"""

import daria as da
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.restoration import denoise_tv_chambolle

# Color image of the baseline fluidflower (benchmark)
image_color = cv2.imread("../../images/fluidflower/Baseline.jpg")
image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)

# ------- Preprocessing

# Consider 1d image, either gray or green channel (blue and red channels did not lead to any reasonable results in short time)
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# Neutralize Color checker
roi_cc = (slice(100, 600), slice(100, 600))
num_bdry_pixels = (600 - 100) * 2 + (600 - 100) * 2
avg_color_boundary = np.uint8(
    (
        np.sum(image[roi_cc[0],100], axis=0)
        + np.sum(image[roi_cc[0],600-1], axis=0)
        + np.sum(image[100, roi_cc[1]], axis=0)
        + np.sum(image[600-1, roi_cc[1]], axis=0)
    ) / num_bdry_pixels
)
image[roi_cc[0], roi_cc[1]] = avg_color_boundary

# Restrict the image to the visible reservoir.
roi = (slice(0,4400), slice(140,7850))
image = image[roi]

# Apply some basic median filtering (alternatively TV denoising)
#image = skimage.restoration.denoise_tv_chambolle(image, weight=0.2, channel_axis=-1)
image = skimage.filters.rank.median(skimage.util.img_as_ubyte(image), skimage.morphology.disk(20))
image = skimage.util.img_as_ubyte(image)

# ----- Iterate through sand layers

def single_iteration(img: np.ndarray, gamma: float, k: int, active_set: np.ndarray, deactivate=0) -> np.ndarray:

    img_work = img.copy()

    # Apply gamma correction
    img_work = skimage.exposure.adjust_gamma(img_work, gamma)

    # Flatten the image and restrict to active set
    pixel_values = img_work.flatten()[active_set.flatten()]

    # Convert format (required for kmeans routine
    pixel_values = np.float32(pixel_values)

    # number of clusters (K) - optimally number of sands, start with two trying to identify the dominant ESF sand layer
    compactness, labels, (centers) = cv2.kmeans(
        pixel_values,
        k,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.1),
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )

    # Update active set - deactive where labels have value 0
    deactivated_labels = (labels==deactivate)[:,0]
    active_set_flat = active_set.flatten()
    active_ids_flat = np.where(active_set_flat)[0]
    active_set_flat[active_ids_flat[deactivated_labels]] = False
    active_set = active_set_flat.reshape(active_set.shape)

    # Highlight newly marked zone
    marked_set_flat = np.zeros_like(active_set_flat, dtype=bool)
    marked_set_flat[active_ids_flat[deactivated_labels]] = True
    marked_set = marked_set_flat.reshape(active_set.shape)

    return img, active_set, marked_set

# ------- Detect several layers

# Initialize active set and facies
active_set = np.ones(image.shape, dtype=bool) # pixels which have not yet been assigned to any sand type
facies = np.zeros(image_color.shape[:2], dtype=int)

# Retrieve the dark water on top (set low gamma, and deactivate first cluster)
image, active_set, marked_set = single_iteration(image, gamma=0.2, k=2, active_set = active_set, deactivate=0)
facies[roi][marked_set] = 60

fig, ax = plt.subplots(3,3, num=1)
ax[0,0].imshow(image)
ax[1,0].imshow(facies)
ax[2,0].imshow(skimage.color.label2rgb(facies, image_color))

# Retrieve the light ESF sand (set high gamma, and deactivate last cluster)
image, active_set, marked_set = single_iteration(image, gamma=3, k=2, active_set = active_set, deactivate=1)
facies[roi][marked_set] = 120

ax[0,1].imshow(image)
ax[1,1].imshow(facies)
ax[2,1].imshow(skimage.color.label2rgb(facies, image_color))

# Retrieve the next light areas (set high gamma, and deactivate last cluster)
image, active_set, marked_set = single_iteration(image, gamma=3, k=2, active_set = active_set, deactivate=1)
facies[roi][marked_set] = 180

ax[0,2].imshow(image)
ax[1,2].imshow(facies)
ax[2,2].imshow(skimage.color.label2rgb(facies, image_color))

plt.show()

# Plot only the facies
plt.imshow(skimage.color.label2rgb(facies, image_color))
plt.show()

# ------- Detect the ESF layer

# Initialize active set and facies
active_set = np.ones(image.shape, dtype=bool) # pixels which have not yet been assigned to any sand type
facies = np.zeros(image_color.shape[:2], dtype=int)

# Retrieve the light ESF sand (set high gamma, and deactivate last cluster)
image, active_set, marked_set = single_iteration(image, gamma=3, k=2, active_set = active_set, deactivate=1)
facies[roi][marked_set] = 120

# Sub task - fetch the lower ESF sand layer
facies[:2530] = 0
plt.imshow(skimage.color.label2rgb(facies, image_color))
plt.show()
