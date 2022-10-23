"""
Segmentation is an important part of te image analysis of multi-layered media.
This example showcases the workflow implemented in DarIA which is based on a
watershed segmentation approach.
"""

import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

import daria

# Define path to image folder
path = Path(f"{os.path.dirname(__file__)}/images/baseline.jpg")
image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

# Restrict image to ROI - only done here to simplify the situation and focus on the
# workflow. The input image has shadows on the left and right sides. These are simply
# cropped here.
image = image[:, 55:3140]

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
labels = daria.segment(image, verbosity=False, **config)

# plot
plt.figure("Labels")
plt.imshow(labels)
plt.figure("Image and labels")
plt.imshow(image)
plt.imshow(labels, alpha=0.3)
plt.show(block=False)
# Pause longer if it is desired to keep the images on the screen
plt.pause(3)
plt.close()
