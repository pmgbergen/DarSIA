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
#base_image = image[400:4200, 300:6300]
base_image = image[1700:2400, 5000:5800]

def single_rev(b_img, iy, ix, dy, dx) -> float:
    img = skimage.util.img_as_ubyte(b_img[iy:iy+dy, ix:ix+dx])

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
   
    # TODO make one loop

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
    largest_grain = brightest_grains[np.argsort(grain_sizes[brightest_grains])[-2]]

    # Highlight the largest grain
    img[labels==largest_grain+1] = 255
    plt.imshow(img)
    plt.show()

    # Take the largest grain of these
    return grain_sizes[largest_grain]

#print("test", single_rev(base_image, 2600, 2750, 100, 100))
#print("test", single_rev(base_image, 3000, 3050, 100, 100))
#
#base_image_color = image_color[400:4200, 300:6300]
#plt.imshow(base_image_color[3000:3100, 3050:3150])
#plt.show()
#plt.imshow(cv2.cvtColor(base_image_color, cv2.COLOR_BGR2RGB))
#plt.show()

rev_size = 100
ny = int(base_image.shape[0] / rev_size)
nx = int(base_image.shape[1] / rev_size)
upscaled = np.zeros((ny,nx), dtype=float)
for i in range(ny):
    for j in range(nx):
        upscaled[i,j] = single_rev(base_image, i * rev_size, j * rev_size, rev_size, rev_size)

plt.imshow(upscaled)
plt.show()





# Restrict the image to the visible reservoir.
#image = base_image[2600:2700, 2750:2850]
image = base_image[3000:3100, 3050:3150]

#image = image[300:4300, 400:6600]
#image = image[3000:3100, 3050:3150]
#image = image[3050:3150, 2950:3050]
#plt.imshow(image)

# Resize - do not require the high resolution for finding sand layers
#image = cv2.resize(image, (0,0), fx = 0.1, fy = 0.1)

# Preprocessing
image_float = skimage.util.img_as_float(image)
image_ubyte = skimage.util.img_as_ubyte(image)
denoised = image_ubyte
#denoised = np.uint8(255.0 * skimage.restoration.denoise_tv_chambolle(image_ubyte, weight=0.05, channel_axis=-1))
#denoised = skimage.filters.rank.median(image_ubyte, skimage.morphology.disk(20))
#denoised = cv2.resize(denoised, (0,0), fx=0.1, fy=0.1)

# Plotting aux
num = 0

# Watershed segmentation as on https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_marked_watershed.html#sphx-glr-auto-examples-segmentation-plot-marked-watershed-py

# find continuous region (low gradient -
# where less than 10 for this image) --> markers
# disk(5) is used here to get a more smooth image
markers = skimage.filters.rank.gradient(denoised, skimage.morphology.disk(1)) < 10
markers = ndi.label(markers)[0]

# local gradient (disk(2) is used to keep edges thin)
gradient = skimage.filters.rank.gradient(denoised, skimage.morphology.disk(1))

# process the watershed
labels = skimage.segmentation.watershed(gradient, markers)

# plot
#num +=1 
#fig, ax = plt.subplots(5,1, num=num)
#ax[0].imshow(image)
#ax[1].imshow(denoised)
#ax[2].imshow(gradient)
#ax[3].imshow(markers)
#ax[4].imshow(labels)

# analysis - estimate sizes of grains
num_labels = len(set(list(labels.flatten())))
num_pixels = np.size(image)
grain_sizes = np.zeros(num_labels, dtype=float)
for l in range(num_labels):
    num_active_pixels = np.count_nonzero(labels==l+1)
    grain_sizes[l] = float(num_active_pixels) / float(num_pixels)

# analysis - determine the average color of each grain
grain_colors = np.zeros(num_labels, dtype=float)
for l in range(num_labels):
    roi = labels==l+1
    num_pixels_roi = np.count_nonzero(roi)
    grain_colors[l] = np.uint8(np.sum(denoised[roi]) / float(num_pixels_roi))
#grain_colors.astype("uint8")
#plt.plot(np.sort(grain_colors))

grain_colors_img = np.zeros_like(denoised, dtype=np.uint8)
for l in range(num_labels):
    roi = labels==l+1
    grain_colors_img[roi] = grain_colors[l]
#plt.imshow(grain_colors_img)

# pick 10 brightest grains
sorted_grains = np.argsort(grain_colors)[::-1]
grain_colors_img = np.zeros_like(denoised, dtype=np.uint8)
for l in list(sorted_grains[:10]):
    roi = labels==l+1
    grain_colors_img[roi] = grain_colors[l]
plt.imshow(grain_colors_img)

# Take the largest grain of these
print("largest grain size among the 10 brightest grains", np.max(grain_sizes[sorted_grains[:10]]))

# TODO Define a 0.1 * 0.1 resized version of the image with 100 x 100 pixels defining an REV.
# Do the same analysis for eac REV and try to isegment the image by that.

plt.show()
