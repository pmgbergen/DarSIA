"""
Segmentation of the white sands small rig geometry using unsupervised texture segmentation using Gabor filters,
similar to 10.1016/0031-3203(91)90143-S, and inspired by the skimage example: 
https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_gabor.html#sphx-glr-auto-examples-features-detection-plot-gabor-py
"""

# TODO apply the saem to the Baseline benchmark image, with mere objective to detect the visibly most obvious sand layers (need some edge/gradient based detection in the more complicated zones).

import daria as da
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage

# White sand mid rig image
image_color = cv2.imread("../../images/fluidflower/whitesands/smallrig.jpg")

## Restrict the image to some relevant ROI (without the color checker for simplicity).
#image_color = image_color[280:4400, 380:6500]
image_color = image_color[500:1200, 4900:5500]

# Work in the following on the gray version of the image
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

plt.imshow(image)
plt.show()




