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
image_color = cv2.imread("../../images/fluidflower/whitesands/smallrig.jpg")

#image_color = skimage.util.img_as_ubyte(skimage.exposure.adjust_gamma(image_color, 2.2))

# Consider 1d image, either gray or green channel (blue and red channels did not lead to any reasonable results in short time)
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# Restrict the image to the visible reservoir.
#roi = (slice(400,4200), slice(300,6300))
roi = (slice(400,1000), slice(1000,3000))
#roi = (slice(400,4200), slice(800,6300))
image_roi = skimage.util.img_as_ubyte(image[roi])

def patch_grain_analysis(b_img, iy, ix, dy, dx) -> float:
    img = skimage.util.img_as_ubyte(b_img[max(0,iy-dy):min(iy+dy,b_img.shape[0]), max(0,ix-dx):min(ix+dx, b_img.shape[1])])

    img = skimage.filters.rank.enhance_contrast(img, skimage.morphology.disk(5))

    thresh_otsu = skimage.filters.threshold_otsu(img)
    grains = img > thresh_otsu
    distance = ndi.distance_transform_edt(grains)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=grains)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=grains)

    #fig, ax = plt.subplots(3,1)
    #ax[0].imshow(labels)
    #ax[1].imshow(img)
    #ax[2].imshow(distance)
    #plt.show()

    # Statistical grain size analysis
    regions = skimage.measure.regionprops(labels)
    grain_area = np.array([regions[i]["area"] for i in range(labels.max())])

    return grain_area.mean(), grain_area.max()


#lb = np.zeros((5,12), dtype=np.uint8)
#lb[1,2] = 1
#lb[4,5] = 2
#lb = skimage.segmentation.expand_labels(lb, distance = 10)
#plt.imshow(lb)
#plt.show()
#raise ValueError

# Watershed segmentation as on https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_marked_watershed.html#sphx-glr-auto-examples-segmentation-plot-marked-watershed-py
denoised = image_roi

denoised = skimage.filters.rank.enhance_contrast(denoised, skimage.morphology.disk(5))


#denoised = skimage.transform.rescale(denoised, 0.05, anti_aliasing = False)
#denoised = skimage.transform.rescale(denoised, 20, anti_aliasing = False)

denoised = skimage.restoration.denoise_bilateral(denoised)
#plt.imshow(denoised)
#plt.show()
denoised = skimage.filters.rank.minimum(denoised, skimage.morphology.disk(20))
denoised = skimage.filters.rank.minimum(denoised, skimage.morphology.disk(2))
denoised = skimage.filters.rank.enhance_contrast(denoised, skimage.morphology.disk(5))
denoised = skimage.filters.gaussian(denoised, 10)
#plt.imshow(denoised)
#plt.show()
plt.plot(np.histogram(denoised.flatten(), bins=256)[0])
plt.show()



denoised_rescaled = skimage.util.img_as_float(denoised)
dn_min = np.min(denoised_rescaled)
dn_max = np.max(denoised_rescaled)
denoised_rescaled = (denoised_rescaled - dn_min) / (dn_max - dn_min)
denoised_rescaled = skimage.util.img_as_ubyte(denoised_rescaled)
#plt.imshow(denoised_rescaled)
#plt.show()

labels = np.zeros(denoised_rescaled.shape, dtype=np.uint8)
mask = np.logical_and(denoised_rescaled > 45, denoised_rescaled < 64)
labels[mask] = 1

mask = np.logical_and(denoised_rescaled > 157, denoised_rescaled < 173)
labels[mask] = 2

mask = np.logical_and(denoised_rescaled > 205, denoised_rescaled < 228)
labels[mask] = 3

labels = skimage.segmentation.expand_labels(labels, distance = 100)

mask = labels == 1
mask = ndi.binary_fill_holes(mask)
labels[mask] = 1

mask = labels == 2
mask = ndi.binary_fill_holes(mask)
labels[mask] = 2

mask = labels == 3
mask = ndi.binary_fill_holes(mask)
labels[mask] = 3


plt.imshow(skimage.color.label2rgb(labels, image_roi))
plt.show()

raise ValueError


#denoised = skimage.util.img_as_ubyte(skimage.filters.difference_of_gaussians(denoised, 10))
#denoised = skimage.filters.gaussian(denoised, 10)
#plt.imshow(denoised)
#plt.show()
denoised = skimage.img_as_ubyte(skimage.restoration.denoise_tv_bregman(denoised, weight=0.01, channel_axis=-1))
plt.imshow(denoised)
plt.show()
#denoised = skimage.util.img_as_ubyte(skimage.filters.difference_of_gaussians(denoised, 10))
#plt.imshow(denoised)
#plt.show()

#denoised = skimage.util.img_as_ubyte(denoised)
#denoised = skimage.exposure.equalize_adapthist(denoised, clip_limit=0.03)

# find continuous region, i.e., areas with low local gradient
markers_basis = skimage.filters.rank.gradient(denoised, skimage.morphology.disk(5))
#markers_basis_hist = np.histogram(markers_basis.flatten(), bins=100)[0] # reason for choosing 0.3 below
markers = markers_basis < 10 #0.3 * np.max(markers_basis)
markers = ndi.label(markers)[0]

# find edges (here given ylocal gradient (disk(2) is used to keep edges thin)
gradient = skimage.filters.rank.gradient(denoised, skimage.morphology.disk(10))

# process the watershed and resize to the original size
labels_roi = skimage.util.img_as_ubyte(skimage.segmentation.watershed(gradient, markers))
labels_roi2 = skimage.util.img_as_ubyte(skimage.transform.resize(labels_roi, image_color[roi].shape[:2]))
labels = np.zeros(image_color.shape[:2], dtype=np.uint8)
labels[roi] = labels_roi2

# plot
fig, ax = plt.subplots(3,1, num=1)
ax[0].imshow(gradient)
ax[1].imshow(labels)
#ax[1].plot(markers_basis_hist)
#ax[1].vlines(30, ymin=0, ymax=np.max(markers_basis_hist), colors='m')
ax[2].imshow(skimage.color.label2rgb(labels, image_color))
#ax[2].imshow(labels_roi2)
plt.show()

raise ValueError

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
