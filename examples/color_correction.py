from pathlib import Path

import os
import cv2
import matplotlib.pyplot as plt

import daria

# -------- Convert the image into linear RGB color space

# Need to specify a ROI which contains the color checker
roi_cc = (slice(0, 240), slice(0, 240))

# Create the color correction and apply it at initialization of image class
color_correction = daria.ColorCorrection(ROI=roi_cc)
baseline_corrected = daria.Image(f"{os.path.dirname(__file__)}/images/baseline.jpg", color_correction = color_correction)

# Create an uncorrected image for comparison
baseline_uncorrected = daria.Image(f"{os.path.dirname(__file__)}/images/baseline.jpg")

# -------- Plot corrected and uncorrected images

fig, ax = plt.subplots(1, num=1)
ax.imshow(baseline_corrected.img)
fig, ax = plt.subplots(1, num=2)
ax.imshow(baseline_uncorrected.img)
plt.show(block = False)
# Pause longer if it is desired to keep the images on the screen
plt.pause(3)
plt.close()