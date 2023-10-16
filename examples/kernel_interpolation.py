from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia

# ! ---- DATA MANAGEMENT ---- !

# The example images originate from a water tracer experiment with a multi-colored
# indicator. Green color corresponds to concentrations close to 100%, while blue
# color corresponds to concentrations ~99-0%. The images have been cropped and resized
# mainly for data storage reasons. It is recommended to use full resolution images.
baseline = darsia.imread(Path("images/kernel_interpolation_example_base.npz"))
image = darsia.imread(Path("images/kernel_interpolation_example_test.npz"))

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

# Predefine concentration analysis for now without any model (to be defined later).
analysis = darsia.ConcentrationAnalysis(
    base=baseline.img_as(float),
    restoration=darsia.TVD(
        weight=0.025, eps=1e-4, max_num_iter=100, method="isotropic Bregman"
    ),
    **{"diff option": "plain"}
)

# The goal is to define ome ROIs for which physical information is known.
# One possibility is to use a GUI for interactive use. This option can
# be activated on demand. For testing purposes this example by default
# uses a pre-defined sample selection.
if True:
    samples = [
        (slice(15, 40), slice(20, 45)),
        (slice(15, 40), slice(220, 245)),
        (slice(15, 40), slice(420, 445)),
        (slice(15, 40), slice(720, 745)),
    ]
else:
    # Same but under the use of a graphical user interface.
    # Ask user to provide characteristic regions with expected concentration values
    assistant = darsia.BoxSelectionAssistant(image)
    samples = assistant()

# For simplicity the concentrations are pre-defined. These could also be defined
# by the user.
n = len(samples)
concentrations = np.append(np.linspace(1, 0.99, n - 1), 0)

# Now add kernel interpolation as model trained by the extracted information.
"""
comments:
-   Here, the support points as defined above are used to build a Kernel Interpolation.
    For this, the 'concentration' without any model is used, correpsonding to the difference
    to the baseline.
-   This Kernel interpolation is then used as the model in the darsia.ConcentrationAnalysis
-   use plain difference to keep all information (no cut off or norm)
    this is the advantage of the kernel interpolation - it can handle negative colours
-   finding the right kernel parameters is part of the modeling
"""
smooth_RGB = analysis(image.img_as(float)).img
colours_RGB = darsia.extract_characteristic_data(
    signal=smooth_RGB, samples=samples, verbosity=True, surpress_plot=True
)
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

# To enable the example as test, the plots are closed after short time.
# Pause longer if it is desired to keep the images on the screen.
print("Warning: The plot is closed after short time to enable testing.")
plt.show(block=False)
plt.pause(5)
plt.close()
