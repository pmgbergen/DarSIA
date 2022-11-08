"""
Segmentation is an important part of te image analysis of multi-layered media.
This example showcases the workflow implemented in DarSIA which is based on a
watershed segmentation approach.
"""

import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import darsia

# Define path to image folder
path = Path(f"{os.path.dirname(__file__)}/images/baseline.jpg")
image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

# Restrict image to ROI - only done here to simplify the situation and focus on the
# workflow. The input image has shadows on the left and right sides. These are simply
# cropped here.
image = image[:, 55:3140]

# ! ---- Gradient-based approach

# Define suitable config with tuning parameters for the segmentation.
config = {
    "median disk radius": 20,
    "rescaling factor": 0.3,
    "markers disk radius": 10,
    "threshold": 20,
    "gradient disk radius": 2,
}

# Determine labels. To better understand the choice of the tuning parameters (and in
# practical situations for finding choices for these) set verboisty to True. Essentially
# relevant quantities are plotted.
labels = darsia.segment(
    image,
    markers_method="gradient_based",
    edges_method="gradient_based",
    verbosity=False,
    **config,
)

# Plot the gradient-based segmentation
plt.figure("Labels")
plt.imshow(labels)
plt.figure("Image and labels")
plt.imshow(image)
plt.imshow(labels, alpha=0.3)

# ! ---- Supervised approach

# Deactivate color checker
mask = np.ones(image.shape[:2], dtype=bool)
color_checker = (slice(0, 220), slice(0, 200))
mask[color_checker] = False

config = {
    "marker_points": np.array(
        [
            [100, 1600],  # water
            [400, 1600],  # top esf
            [560, 760],  # c sand, left
            [620, 1260],  # c sand, center
            [550, 1970],  # c sand, right
            [630, 365],  # d sand, left
            [690, 1225],  # d sand, centre
            [620, 1990],  # d sand, right
            [560, 710],  # e sand, left
            [775, 1810],  # e sand, right
            [615, 1610],  # large perm fault
            [830, 570],  # f sand left
            [940, 1320],  # f sand centre
            [890, 1775],  # f sand right
            [940, 360],  # c sand, left of heterog. fault
            [1000, 240],  # d sand, left of heterog. fault
            [1130, 165],  # e sand, left of heterog. fault
            [1325, 170],  # esf sand, left of heterog. fault
            [1010, 680],  # e sand right of heterog. fault
            [1020, 2650],  # e sand top of lower esf
            [1075, 1530],  # lower right esf
            [1325, 405],  # part of heterog. fault
            [1400, 385],  # part of heterog. fault
            [1350, 1320],  # layer below lowest esf
        ]
    ),
    "region_size": 20,
    "mask scharr": mask,
}

labels_supervised = darsia.segment(
    image, markers_method="supervised", edges_method="scharr", **config
)

# Plot the result of the supervised segmentation
plt.figure("Labels supervised")
plt.imshow(labels_supervised)
plt.figure("Image and labels supervised")
plt.imshow(image)
plt.imshow(labels_supervised, alpha=0.3)

plt.show(block=False)
# Pause longer if it is desired to keep the images on the screen
plt.pause(3)
plt.close()
