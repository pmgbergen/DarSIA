"""
Apply various filters and denoising algorithms to the FluidFlower Baseline image.
"""
# Clustering, based on https://www.thepythoncode.com/article/kmeans-for-image-segmentation-opencv-python

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage

# Plotting aux
num = 0

# Color image
image_color = cv2.imread("../../images/fluidflower/whitesands/smallrig.jpg")

# Consider 1d image, either gray or green channel (blue and red channels did not lead to any reasonable results in short time)
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)

# Restrict to reservoir with proper size
base_image = image[400:4200, 300:6300]

def patch_grain_analysis(b_img, iy, ix, dy, dx) -> float:
    img = skimage.util.img_as_ubyte(b_img[max(0,iy-dy):min(iy+dy,b_img.shape[0]), max(0,ix-dx):min(ix+dx, b_img.shape[1])])

    img = skimage.exposure.adjust_gamma(img, 2.2)
    segments_quick = skimage.segmentation.quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)

    fig, ax = plt.subplots(2,1)
    ax[0].imshow(skimage.color.label2rgb(segments_quick, img))
    ax[1].imshow(img)
    plt.show()

    # Statistical grain size analysis
    regions = skimage.measure.regionprops(segments_quick)
    grain_area = np.array([regions[i]["area"] for i in range(segments_quick.max())])

    return grain_area.mean()

rev_size = 100 # to the left and right

ny = int(base_image.shape[0] / rev_size) + 1
nx = int(base_image.shape[1] / rev_size) + 1
upscaled = np.zeros((ny,nx), dtype=float)

# Test the grain size analysis for different patches for differnt sand types - only works properly for the coarsest grains
patch_grain_analysis(base_image, 200, 3000, rev_size, rev_size)
patch_grain_analysis(base_image, 500, 2500, rev_size, rev_size)
patch_grain_analysis(base_image, 1800, 1700, rev_size, rev_size)
patch_grain_analysis(base_image, 1000, 3700, rev_size, rev_size)
