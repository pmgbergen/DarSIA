import os

import matplotlib.pyplot as plt
import numpy as np

import daria

# ! ---- Read uncorrected image

# Define path to image folder
image = f"{os.path.dirname(__file__)}/images/baseline.jpg"

# Create an uncorrected image for comparison
uncorrected_baseline = daria.Image(image, width=2.8, height=1.5)

# ! ---- Setup the machine learning based color correction

# Need to specify a ROI which contains the color checker
experimental_roi_cc = (slice(0, 240), slice(0, 240))
experimental_color_correction = daria.ExperimentalColorCorrection(
    roi=experimental_roi_cc,
    # verbosity = True,
)

# NOTE: Setting the flag verbosity to True allows to debug some
# of the details/progress of the color checker.

# Create the color correction and apply it at initialization of image class
experimental_corrected_baseline = daria.Image(
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
color_correction = daria.ColorCorrection(
    roi=roi_cc,
    # verbosity = True,
)

# NOTE: Setting the flag verbosity to True allows to debug some
# of the details/progress of the color checker. In particular,
# the final color checker after calibration is displayed.

# Create the color correction and apply it at initialization of image class
corrected_baseline = daria.Image(
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
