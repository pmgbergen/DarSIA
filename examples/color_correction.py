from pathlib import Path

import os
import cv2
import matplotlib.pyplot as plt

import daria

# -------- Convert the image into linear RGB color space

# Fetch image, in BGR
img_BGR = cv2.imread(str(Path(f"{os.path.dirname(__file__)}/images/baseline.jpg")))

# Convert to RGB (int)
img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

# -------- Apply colour correction based on color checker

# Need to specify a ROI which contains the color checker
roi_cc = (slice(0, 240), slice(0, 240))

# Apply color correction
color_correction = daria.ColorCorrection()
corrected_baseline_RGB = color_correction(img_RGB, roi_cc, verbosity=False)

# NOTE: For some reason, when choosing verbosity=True the colour checker is displayed,
# but no other images can be properly displayed afterwards anymore.

# -------- Plot images pre and post correction, and store both images

fig, ax = plt.subplots(1, num=1)
ax.imshow(img_RGB)
fig, ax = plt.subplots(1, num=2)
ax.imshow(corrected_baseline_RGB)
plt.show(block = False)
# Pause longer if it is desired to keep the images on the screen
plt.pause(3)
plt.close()