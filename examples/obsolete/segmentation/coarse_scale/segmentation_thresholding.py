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
image_color = cv2.imread("../../../images/fluidflower/Baseline.jpg")
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

# Restrict the image to the lower part of the visible reservoir.
roi = (slice(2500,4400), slice(140,7850))
image_roi = image[roi]

# Apply gamma correction to highlight the bright ESF sand
image_roi = skimage.exposure.adjust_gamma(image_roi, 3)

# Apply some basic median filtering (alternatively TV denoising)
#image_roi = skimage.restoration.denoise_tv_chambolle(image_roi, weight=0.2, channel_axis=-1)
image_roi = skimage.filters.rank.median(skimage.util.img_as_ubyte(image_roi), skimage.morphology.disk(20))
image_roi = skimage.util.img_as_ubyte(image_roi)

# Determine the histogram of the intensity values
hist = np.histogram(image_roi.flatten(), bins=256)[0]

# Determine a good thresholding value using the OTSU method, and some fine tuned version
thresh_otsu = skimage.filters.threshold_otsu(image_roi)

# Map esf layer back to the original image
esf_otsu = image_roi > thresh_otsu
esf_image_otsu = np.zeros(image.shape[:2], dtype=np.uint8)
esf_image_otsu[roi][esf_otsu] = 255

esf_tuned = image_roi > thresh_otsu - 1
esf_image_tuned = np.zeros(image.shape[:2], dtype=np.uint8)
esf_image_tuned[roi][esf_tuned] = 255

# Plot only the facies
fig, ax = plt.subplots(2,1)
ax[0].imshow(skimage.color.label2rgb(esf_image_otsu, image_color))
ax[1].imshow(skimage.color.label2rgb(esf_image_tuned, image_color))
plt.show()
