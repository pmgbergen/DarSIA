"""
Example script for simple image analysis. By comparison of images
of the same well test, a tracer concentration can be determined.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import skimage

import daria


def preprocessing(img):
    """Standard curvature and color correction. Return ROI."""

    # Curvature correctio
    img = daria.curvature_correction(img)

    # Preprocessing. Transform to RGB space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.show()

    # Color correction
    roi_cc = (slice(0, 600), slice(0, 700))
    colorcorrection = daria.ColorCorrection()
    img = colorcorrection.adjust(img, roi_cc, verbosity=False, whitebalancing=True)

    # Transform to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img


def determine_tracer(img, base):
    """Extract tracer based on a reference image"""
    # Take (unsigned) difference
    diff = skimage.util.compare_images(img, base, method="diff")

    # Apply smoothing filter
    diff = skimage.filters.rank.median(diff, skimage.morphology.disk(20))

    # TODO: User input. Require threshold values for identifying both 0 and 1 concentrations.
    # plt.imshow(diff)
    # plt.show()
    thresh_min = 7
    thresh_max = 40

    # Calibrated thresholding
    tracer_min_mask = diff <= thresh_min
    diff[tracer_min_mask] = 0
    diff[~tracer_min_mask] -= thresh_min
    tracer_max_mask = diff >= thresh_max - thresh_min
    diff[tracer_max_mask] = thresh_max - thresh_min

    # Rescale image to range [0,255]
    diff = skimage.exposure.rescale_intensity(diff)

    # Transform to float data type for simpler conversion to tracer concentration
    diff = skimage.util.img_as_float(diff)

    return diff


# !----- Main routine

# Read in baseline figure and apply correction once
baseline = cv2.imread(str(Path("./images/tracer_0.jpg")))
baseline = preprocessing(baseline)

# Read in test figure and apply correction
# tracer_image = cv2.imread("./images/tracer_1.jpg")
# tracer_image = cv2.imread("./images/tracer_2.jpg")
tracer_image = cv2.imread("./images/tracer_3.jpg")
tracer_image = preprocessing(tracer_image)

# Determin tracer
tracer = determine_tracer(tracer_image, baseline)

# Plot
plt.imshow(tracer)
plt.show()

# Blend with baseline
fig, ax = plt.subplots(2, 1)
ax[0].imshow(skimage.util.compare_images(tracer, baseline, method="blend"))
ax[1].imshow(tracer_image)
plt.show()

plt.imshow(skimage.util.compare_images(tracer, tracer_image, method="blend"))
plt.show()
