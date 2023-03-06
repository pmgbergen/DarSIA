import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import darsia

# Control
use_general_images = False

# ! ---- Read uncorrected image

# Define path to image folder
image = f"{os.path.dirname(__file__)}/images/baseline.jpg"

# Create an uncorrected image for comparison
if use_general_images:
    image_array = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    uncorrected_baseline = darsia.GeneralImage(
        image_array, dimensions=[1.5, 2.8], origin=[0, 1.5]
    )
else:
    uncorrected_baseline = darsia.Image(image, width=2.8, height=1.5, color_space="BGR")

# ! ---- Setup the machine learning based color correction

# Need to specify a ROI which contains the color checker
experimental_roi_cc = (slice(0, 240), slice(0, 240))
experimental_color_correction = darsia.ExperimentalColorCorrection(
    roi=experimental_roi_cc,
    # verbosity = True,
)

# NOTE: Setting the flag verbosity to True allows to debug some
# of the details/progress of the color checker.

# Create the color correction and apply it at initialization of image class
if use_general_images:
    experimental_corrected_baseline = darsia.GeneralImage(
        image_array,
        transformations=[experimental_color_correction],
        dimensions=[1.5, 2.8],
        origin=[0.0, 1.5],
    )
else:
    experimental_corrected_baseline = darsia.Image(
        image,
        color_correction=experimental_color_correction,
        width=2.8,
        height=1.5,
    )

# ! ---- Setup the manual color correction

# Need to specify the pixel coordines in (x,y), i.e., (col,row) format, of the
# marks on the color checker. The first coordinate is associated to the mark
# closest to the brown swatch. Continue in counter-clockwise direction.
# NOTE: That this example uses a crudely coarsened image. Thus, the marks
# are not very obvious. They are small white L's.
roi_cc = np.array(
    [
        [154, 176],
        [222, 176],
        [222, 68],
        [154, 68],
    ]
)
color_correction = darsia.ColorCorrection(
    roi=roi_cc,
    # verbosity = True,
)

# NOTE: Setting the flag verbosity to True allows to debug some
# of the details/progress of the color checker. In particular,
# the final color checker after calibration is displayed.

# Create the color correction and apply it at initialization of image class
if use_general_images:
    corrected_baseline = darsia.GeneralImage(
        image_array,
        transformations=[color_correction],
        dimensions=[1.5, 2.8],
        origin=[0.0, 1.5],
    )
else:
    corrected_baseline = darsia.Image(
        image,
        color_correction=color_correction,
        width=2.8,
        height=1.5,
    )

# -------- Plot corrected and uncorrected images
plt.figure()
plt.imshow(uncorrected_baseline.img)
plt.figure()
plt.imshow(experimental_corrected_baseline.img)
plt.figure()
plt.imshow(corrected_baseline.img)
plt.show(block=False)
# Pause longer if it is desired to keep the images on the screen
plt.pause(10)
plt.close()
