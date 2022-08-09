"""
Apply various filters and denoising algorithms to the FluidFlower Baseline image.
"""
# Clustering, based on https://www.thepythoncode.com/article/kmeans-for-image-segmentation-opencv-python

import daria as da
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.restoration import denoise_tv_chambolle
from scipy import ndimage as ndi

# Color image
image_color = cv2.imread("../../../images/fluidflower/Baseline.jpg")

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
image_ubyte = skimage.util.img_as_ubyte(image)

# Watershed segmentation as on https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_marked_watershed.html#sphx-glr-auto-examples-segmentation-plot-marked-watershed-py
denoised = skimage.filters.rank.median(image_ubyte, skimage.morphology.disk(20))
denoised = skimage.util.img_as_ubyte(skimage.transform.rescale(denoised, 0.1, anti_aliasing = True))

# find continuous region, i.e., areas with low local gradient
markers_basis = skimage.filters.rank.gradient(denoised, skimage.morphology.disk(10))
markers_basis_hist = np.histogram(markers_basis.flatten(), bins=100)[0] # reason for choosing 0.3 below
markers = markers_basis < 0.3 * np.max(markers_basis)
markers = ndi.label(markers)[0]

# find edges (here given ylocal gradient (disk(2) is used to keep edges thin)
gradient = skimage.filters.rank.gradient(denoised, skimage.morphology.disk(2))

# process the watershed and resize to the original size
labels_roi = skimage.util.img_as_ubyte(skimage.segmentation.watershed(gradient, markers))
labels_roi2 = skimage.util.img_as_ubyte(skimage.transform.resize(labels_roi, image_color[roi].shape[:2]))
labels = np.zeros(image_color.shape[:2], dtype=np.uint8)
labels[roi] = labels_roi2

# plot
fig, ax = plt.subplots(3,1, num=1)
ax[0].imshow(gradient)
ax[1].plot(markers_basis_hist)
ax[1].vlines(30, ymin=0, ymax=np.max(markers_basis_hist), colors='m')
ax[2].imshow(skimage.color.label2rgb(labels, image_color))
#ax[2].imshow(labels_roi2)
plt.show()

# NOTE two regions remain which need to be further segmented. They have labels: 3 and 4

# Segmentation of subregion with label 3
subregion = labels_roi == 3
sub_image = np.zeros_like(denoised, dtype=np.uint8)
sub_image[subregion] = denoised[subregion]

# Nonlinear transformation to span a lager color space and gain more information
p2, p98 = np.percentile(sub_image[subregion], (2,98))
sub_image = skimage.exposure.rescale_intensity(sub_image, in_range=(p2,p98))
#sub_image = skimage.filters.rank.median(sub_image, skimage.morphology.disk(5))
sub_image = skimage.img_as_ubyte(skimage.restoration.denoise_tv_bregman(sub_image, weight=1, channel_axis=-1))

# Gradient analysis
markers_basis = skimage.filters.rank.gradient(sub_image, skimage.morphology.disk(5))
markers_basis_hist = np.histogram(markers_basis.flatten(), bins=100)[0] # reason for choosing 0.4 below
markers = markers_basis < 0.1 * np.max(markers_basis)
markers = ndi.label(markers)[0]

# find edges (here given ylocal gradient (disk(2) is used to keep edges thin)
gradient = skimage.filters.rank.gradient(sub_image, skimage.morphology.disk(2))

# process the watershed and resize to the original size
sub_labels_roi = skimage.util.img_as_ubyte(skimage.segmentation.watershed(gradient, markers)) + int(np.max(labels_roi))
sub_labels_roi2 = skimage.util.img_as_ubyte(skimage.transform.resize(sub_labels_roi, image_color[roi].shape[:2]))
sub_labels = np.zeros(image_color.shape[:2], dtype=np.uint8)
sub_labels[roi] = sub_labels_roi2

# plot
fig, ax = plt.subplots(3,1, num=2)
ax[0].imshow(gradient)
ax[1].imshow(sub_image)
ax[2].imshow(sub_labels_roi2)
plt.show()
