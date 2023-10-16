from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia

# ! ---- DATA MANAGEMENT ---- !

folder = Path("images")
baseline_path = folder / Path("20220914-142404.TIF")
image_path = folder / Path("20220914-151727.TIF")


# Setup DarSIA curvature correction (here only cropping)
curvature_correction = darsia.CurvatureCorrection(
    config={
        "crop": {
            # Define the pixel values (x,y) of the corners of the ROI.
            # Start at top left corner and then continue counterclockwise.
            "pts_src": [[300, 600], [300, 4300], [7600, 4300], [7600, 600]],
            # Specify the true dimensions of the reference points
            "width": 0.92,
            "height": 0.5,
        }
    }
)
transformations = [curvature_correction]

# Read-in images and restrict to a small sub region for demonstration purpose
baseline = darsia.imread(baseline_path, transformations=transformations).subregion(
    voxels=(slice(2300, 2500), slice(2200, 5200))
)
image = darsia.imread(image_path, transformations=transformations).subregion(
    voxels=(slice(2300, 2500), slice(2200, 5200))
)

# ! ---- MAIN CONCENTRATION ANALYSIS AND CALIBRATION ---- !
"""
Before using kernel interpolation to transform the given image to a concentration image,
the support data on which the interpolation depends has to be defined.
In the thesis these were predefined samples of the smoothed and preprocessed image and
corresponding concentrations.
Now it is possible to select these samples in a GUI using darsia.BoxSelectionAssistant().
In these samples of the image, the most common colour was identified using a histogram
analysis, which is now available through darsia.extract_characteristic_data().
"""

# Chosen as in thesis:
"""samples = [
    (slice(50, 150), slice(100, 200)),
    (slice(50, 150), slice(400, 500)),
    (slice(50, 150), slice(600, 700)),
    (slice(50, 150), slice(800, 900)),
    (slice(50, 150), slice(1000, 1100)),
    (slice(50, 150), slice(1200, 1300)),
    (slice(50, 150), slice(1400, 1500)),
    (slice(50, 150), slice(1600, 1700)),
    (slice(50, 150), slice(2700, 2800)),
]
n = len(samples)
concentrations = np.append(np.linspace(1, 0.99, n - 1), 0)"""


# Same but under the use of DarSIA
# Ask user to provide characteristic regions with expected concentration values
assistant = darsia.BoxSelectionAssistant(image)
samples = assistant()
n = len(samples)
concentrations = np.append(np.linspace(1, 0.99, n - 1), 0)

# Predefine concentration analysis for now without any model (to be defined later).
# Extract characteristic colors from calibration image in relative colors.
analysis = darsia.ConcentrationAnalysis(
    base=baseline.img_as(float),
    restoration=darsia.TVD(
        weight=0.025, eps=1e-4, max_num_iter=100, method="isotropic Bregman"
    ),
    **{"diff option": "plain"}
)
smooth_RGB = analysis(image.img_as(float)).img
colours_RGB = darsia.extract_characteristic_data(
    signal=smooth_RGB, samples=samples, verbosity=True, surpress_plot=True
)

# Now add kernel interpolation as model trained by the extracted information.
"""
comments:
-   here, the support points as defined above are used to build a Kernel Interpolation
-   This Kernel interpolation is then used as the model in the darsia.ConcentrationAnalysis
-   use plain difference to keep all information (no cut off or norm)
    this is the advantage of the kernel interpolation - it can handle negative colours
-   finding the right kernel parameters is part of the modeling
"""
analysis.model = darsia.KernelInterpolation(
    darsia.GaussianKernel(gamma=9.73), colours_RGB, concentrations
)

# Finally, apply the (full) concentration analysis to analyze the test image
concentration_image = analysis(image.img_as(float)).img

# ! ----- VISUALISATION ---- !

plt.figure("concentration average along one dimension")
plt.plot(np.average(concentration_image, axis=0))
plt.xlabel("horizontal pixel")
plt.ylabel("concentration")

concentration_image[
    concentration_image > 1
] = 1  # für visualisierung von größer 1 values
concentration_image[concentration_image < 0] = 0
fig = plt.figure()
fig.suptitle("original image and resulting concentration")
ax = plt.subplot(212)
ax.imshow(concentration_image)
ax.set_ylabel("vertical pixel")
ax.set_xlabel("horizontal pixel")
ax = plt.subplot(211)
ax.imshow(skimage.img_as_ubyte(image.img))
ax.set_ylabel("vertical pixel")
ax.set_xlabel("horizontal pixel")

# plt.figure("indicator")
# indicator = np.arange(101) / 100
# plt.axis("off")
# plt.imshow([indicator, indicator, indicator, indicator, indicator])
plt.show()
