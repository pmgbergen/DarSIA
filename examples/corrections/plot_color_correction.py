"""
Colour correction with a colour checker
=======================================

Two ways to calibrate recorded colours against a Classic X-Rite colour checker
visible in the image: a machine-learning detector that locates the chart
automatically, and a manual variant where the chart corners are given explicitly.
"""

import os

import cv2
import matplotlib.pyplot as plt

import darsia

image_path = os.path.join("..", "images", "baseline.jpg")
image_array = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

uncorrected = darsia.Image(image_array, width=2.8, height=1.5)

# %%
# Automatic detection: give a region of interest that contains the colour
# checker and let :class:`~darsia.ExperimentalColorCorrection` find it.

experimental_roi = (slice(0, 240), slice(0, 240))
experimental_correction = darsia.ExperimentalColorCorrection(roi=experimental_roi)
experimental_corrected = darsia.Image(
    image_array,
    transformations=[experimental_correction],
    width=2.8,
    height=1.5,
)

# %%
# Manual variant: pass the pixel coordinates of the four white L-marks on the
# chart (counter-clockwise from the mark nearest the brown swatch).

config = {"roi": darsia.make_voxel([[154, 176], [222, 176], [222, 68], [154, 68]])}
color_correction = darsia.ColorCorrection(base=uncorrected, config=config)
corrected = darsia.Image(
    image_array,
    transformations=[color_correction],
    width=2.8,
    height=1.5,
)

# %%
# Uncorrected, auto-corrected and manually corrected, side by side.

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, img, title in zip(
    axes,
    (uncorrected.img, experimental_corrected.img, corrected.img),
    ("uncorrected", "auto colour checker", "manual colour checker"),
):
    ax.imshow(img)
    ax.set_title(title)
    ax.set_axis_off()
fig.tight_layout()
plt.show()
