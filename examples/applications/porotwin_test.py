"""
Example script for simple image analysis. Somethign like this will be used in the PoroTwin project.
"""

import cv2
import daria as da
import numpy as np
import skimage
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from daria.corrections.color.colorchecker import ColorCorrection

# !----- Auxiliary routines 

def correction(img):
    """Standard curvature and color correction. Return ROI."""

    # Preprocessing. Transform to RGB space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Curvature correction
    img = da.curvature_correction(img)

    # Color correction
    roi_cc = (slice(0, 600), slice(0, 700))
    colorcorrection = ColorCorrection()
    img = colorcorrection.adjust(img, roi_cc, verbosity=False, whitebalancing=True)

    # Extract relevant ROI
    img = img[849:4466, 167:7831]

    # TODO calibration

    return img

def determine_tracer(img, base):
    """Extract tracer based on a reference image"""
    # Take (unsigned) difference
    diff = skimage.util.compare_images(img, base, method='diff')
    
    # Apply smoothing filter
    #diff = skimage.filters.rank.median(diff, skimage.morphology.disk(20))
    diff = skimage.filters.median(diff)

    return diff
    
def postprocessing(img):
    """Apply simple postprocessing"""
    # Make images smaller
    img = skimage.transform.rescale(img, 0.2, anti_aliasing = True)

    # Transform to BGR space
    return img
    #return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# !----- Main routine

# Read in baseline figure and apply correction once
baseline= cv2.imread("../../images/fluidflower/porotwin_test/DSC04407.JPG")
baseline = correction(baseline)

# Example 1: Determine tracer concentration for first time step
image1 = cv2.imread("../../images/fluidflower/porotwin_test/DSC04408.JPG")
image1 = correction(image1)
tracer1 = determine_tracer(image1, baseline)
tracer1 = postprocessing(tracer1)
cv2.imwrite("./out/tracer1.jpg", tracer1)

# Example 2: Determine tracer concentration for some later time step
image2 = cv2.imread("../../images/fluidflower/porotwin_test/DSC04425.JPG")
image2 = correction(image2)
tracer2 = determine_tracer(image2, baseline)
tracer2 = postprocessing(tracer2)
cv2.imwrite("./out/tracer2.jpg", tracer2)

# Plot
fig, ax = plt.subplots(2,1)
ax[0].imshow(diff_2)
ax[1].imshow(diff_3)
plt.show()
