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
import statistics

# Plotting aux
num = 0

# Color image
image_color = cv2.imread("../images/fluidflower/whitesands/smallrig.jpg")

# Consider 1d image, either gray or green channel (blue and red channels did not lead to any reasonable results in short time)
image = cv2.cvtColor(
    image_color,
    cv2.COLOR_BGR2GRAY,
)

#plt.imshow(image)
#plt.show()

# Restrict to reservoir with proper size
base_image = image[400:4200, 300:6300]
#base_image = image[1700:2400, 5000:5800]

def largest_grain_in_rev(b_img, iy, ix, dy, dx) -> float:
    img = skimage.util.img_as_ubyte(b_img[max(0,iy-dy):min(iy+dy,b_img.shape[0]), max(0,ix-dx):min(ix+dx, b_img.shape[1])])

    # Watershed segmentation as on https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_marked_watershed.html#sphx-glr-auto-examples-segmentation-plot-marked-watershed-py
    markers = skimage.filters.rank.gradient(img, skimage.morphology.disk(1)) < 20
    markers = ndi.label(markers)[0]
    gradient = skimage.filters.rank.gradient(img, skimage.morphology.disk(1))
    labels = skimage.segmentation.watershed(gradient, markers)
    
    # analysis - estimate sizes of grains
    num_labels = len(set(list(labels.flatten())))
    num_pixels = np.size(img)
    grain_sizes = np.zeros(num_labels, dtype=float)
    for label in range(num_labels):
        num_active_pixels = np.count_nonzero(labels==label+1)
        grain_sizes[label] = float(num_active_pixels) / float(num_pixels)

    # analysis - average color of each grain
    grain_colors = np.zeros(num_labels, dtype=float)
    for label in range(num_labels):
        roi = labels==label+1
        num_pixels_roi = np.count_nonzero(roi)
        grain_colors[label] = np.uint8(np.sum(img[roi]) / float(num_pixels_roi))
    
    # pick 10 brightest grains
    sorted_grains = np.argsort(grain_colors)[::-1]
    brightest_grains = sorted_grains[:20]

    # Pick the grain with largest size (among the brightes)
    characteristic_grain = brightest_grains[np.argsort(grain_sizes[brightest_grains])[-1]]
    roi_characteristic_grain = labels == characteristic_grain + 1

    # Highlight the characteristic grain
    #img_copy = img.copy()
    #img_copy[roi_characteristic_grain] = 255
    #plt.imshow(img_copy)
    #plt.show()

    # Retur grain size of the characteristic grain
    return grain_sizes[characteristic_grain]

def median_grain_in_rev(b_img, iy, ix, dy, dx) -> float:
    img = skimage.util.img_as_ubyte(b_img[max(0,iy-dy):min(iy+dy,b_img.shape[0]), max(0,ix-dx):min(ix+dx, b_img.shape[1])])

    # Watershed segmentation as on https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_marked_watershed.html#sphx-glr-auto-examples-segmentation-plot-marked-watershed-py
    markers = skimage.filters.rank.gradient(img, skimage.morphology.disk(1)) < 20
    markers = ndi.label(markers)[0]
    gradient = skimage.filters.rank.gradient(img, skimage.morphology.disk(1))
    labels = skimage.segmentation.watershed(gradient, markers)
    
    # analysis - estimate sizes of grains
    num_labels = len(set(list(labels.flatten())))
    num_pixels = np.size(img)
    grain_sizes = np.zeros(num_labels, dtype=float)
    for label in range(num_labels):
        num_active_pixels = np.count_nonzero(labels==label+1)
        grain_sizes[label] = float(num_active_pixels) / float(num_pixels)

    # Return median grain size
    return np.max(grain_sizes)

rev_size = 50 # to the left and right

ny = int(base_image.shape[0] / rev_size) + 1
nx = int(base_image.shape[1] / rev_size) + 1
upscaled = np.zeros((ny,nx), dtype=float)
for i in range(ny):
    for j in range(nx):
        #upscaled[i,j] = largest_grain_in_rev(base_image, i * rev_size, j * rev_size, rev_size, rev_size)
        upscaled[i,j] = median_grain_in_rev(base_image, i * rev_size, j * rev_size, rev_size, rev_size)

# Remove outliers
#upscaled = skimage.filters.rank.median(upscaled, skimage.morphology.disk(1))

num+=1
fig, ax = plt.subplots(2,1, num=num)
ax[0].imshow(upscaled)
ax[1].imshow(base_image)
plt.show()

# Continuum-like approach
grid_size = 10
ny = int(base_image.shape[0] / grid_size) + 1
nx = int(base_image.shape[1] / grid_size) + 1
upscaled = np.zeros((ny,nx), dtype=float)
for i in range(ny):
    for j in range(nx):
        upscaled[i,j] = largest_grain_inle_rev(base_image, i * grid_size, j * grid_size, rev_size, rev_size)

# Remove outliers
#upscaled = skimage.filters.rank.median(upscaled, skimage.morphology.disk(1))

num+=1
fig, ax = plt.subplots(2,1, num=num)
ax[0].imshow(upscaled)
ax[1].imshow(base_image)
plt.show()
