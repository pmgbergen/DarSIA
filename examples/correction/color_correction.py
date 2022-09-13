from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from daria.corrections.color import ColorCorrection

# -------- Convert the image into linear RGB color space

# Fetch image, in BGR
img_BGR = cv2.imread(str(Path("./baseline.jpg")))

# Convert to RGB (int)
img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

# -------- Apply colour correction based on color checker

# Need to specify a ROI which contains the color checker
roi_cc = (slice(0, 240), slice(0, 240))

# Apply color correction
colorcorrection = ColorCorrection()
corrected_baseline_RGB = colorcorrection.adjust(
    img_RGB, roi_cc, verbosity=False, whitebalancing=True
)

# NOTE: For some reason, when choosing verbosity=True the colour checker is displayed,
# but no other images can be properly displayed afterwards anymore.

# -------- Plot images pre and post correction, and store both images

fig, ax = plt.subplots(1, num=1)
ax.imshow(img_RGB)
fig, ax = plt.subplots(1, num=2)
ax.imshow(corrected_baseline_RGB)
plt.show()
