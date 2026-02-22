import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import darsia

# ! ---- Read uncorrected image

# Define path to image folder
image = f"{os.path.dirname(__file__)}/images/baseline.jpg"

# Create an uncorrected image for comparison
image_array = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
uncorrected_baseline = darsia.Image(
    image_array,
    width=2.8,
    height=1.5,
)

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
experimental_corrected_baseline = darsia.Image(
    image_array,
    transformations=[experimental_color_correction],
    width=2.8,
    height=1.5,
)

# ! ---- Setup the manual color correction

# Need to specify the pixel coordines in (x,y), i.e., (col,row) format, of the
# marks on the color checker. The first coordinate is associated to the mark
# closest to the brown swatch. Continue in counter-clockwise direction.
# NOTE: That this example uses a crudely coarsened image. Thus, the marks
# are not very obvious. They are small white L's.
config = {
    "roi": darsia.make_voxel(
        [
            [154, 176],
            [222, 176],
            [222, 68],
            [154, 68],
        ]
    )
}
color_correction = darsia.ColorCorrection(
    base = uncorrected_baseline,
    config = config
)

# NOTE: Setting the flag verbosity to True allows to debug some
# of the details/progress of the color checker. In particular,
# the final color checker after calibration is displayed.

# Create the color correction and apply it at initialization of image class
corrected_baseline = darsia.Image(
    image_array,
    transformations=[color_correction],
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

# To enable the example as test, the plots are closed after short time.
# Pause longer if it is desired to keep the images on the screen.
print("Warning: The plot is closed after short time to enable testing.")
plt.show(block=False)
plt.pause(5)
plt.close()
