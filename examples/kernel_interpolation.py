import string
from abc import ABC
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia

# ! ---- DATA MANAGEMENT ---- !

folder = Path("images")
baseline_path = folder / Path("20220914-142404.TIF")
image_path = folder / Path("20220914-151727.TIF")

# Setup curvature correction (here only cropping)
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

# Read-in images
baseline = darsia.imread(baseline_path, transformations=transformations).subregion(
    voxels=(slice(2300, 2500), slice(2200, 5200))
)
image = darsia.imread(image_path, transformations=transformations).subregion(
    voxels=(slice(2300, 2500), slice(2200, 5200))
)

# ! ---- CALIBRATION ---- !

# NOTE: Samples can also be defined using a graphical assistant.
if False:
    # Chosen as in thesis
    samples = [
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
    concentrations = np.append(np.linspace(1, 0.99, n - 1), 0)

else:
    # Same but under the use of DarSIA
    # Ask user to provide characteristic regions with expected concentration values
    assistant = darsia.BoxSelectionAssistant(image)
    samples = assistant()
    n = len(samples)
    concentrations = np.append(np.linspace(1, 0.99, n - 1), 0)


def extract_characteristic_data(signal, samples, verbosity=False, show_plot=True):
    """Assistant to extract representative colors from input image for given patches.

    Args:
        signal (np.ndarray): input signal, assumed to have the structure of a 2d,
            colored image.
        samples (list of slices): list of 2d regions of interest
        show_plot (boolean): flag controlling whether plots are displayed.

    Returns:
        np.ndarray: characteristic colors.

    """

    # Alphabet useful for labeling in plots
    letters = list(string.ascii_uppercase)

    # visualise patches
    fig, ax = plt.subplots()
    ax.imshow(np.abs(signal))  # visualise abs colours, because relative cols are neg
    ax.set_xlabel("horizontal pixel")
    ax.set_ylabel("vertical pixel")

    # double check number of patches
    n = np.shape(samples)[0]  # number of patches
    if verbosity:
        print("Number of support patches: " + str(n))

    # init colour vector
    colours = np.zeros((n, 3))
    # enumerate through all patches
    for i, p in enumerate(samples):
        # visualise patches on image
        rect = patches.Rectangle(
            (p[1].start, p[0].start),
            p[1].stop - p[1].start,
            p[0].stop - p[0].start,
            linewidth=1,
            edgecolor="w",
            facecolor="none",
        )
        ax.text(p[1].start + 5, p[0].start + 95, letters[i], fontsize=12, color="white")
        ax.add_patch(rect)

        # histo analysis
        patch = signal[p]
        flat_image = np.reshape(patch, (-1, 3))  # all pixels in one dimension
        # patch visualisation
        # plt.figure("patch" + letters[i])
        # plt.imshow(np.abs(patch))
        H, edges = np.histogramdd(
            flat_image, bins=100, range=[(-1, 1), (-1, 1), (-1, 1)]
        )
        index = np.unravel_index(H.argmax(), H.shape)
        col = [
            (edges[0][index[0]] + edges[0][index[0] + 1]) / 2,
            (edges[1][index[1]] + edges[1][index[1] + 1]) / 2,
            (edges[2][index[2]] + edges[2][index[2] + 1]) / 2,
        ]
        colours[i] = col

    if verbosity:
        c = np.abs(colours)
        plt.figure()
        ax = plt.axes(projection="3d")
        ax.set_xlabel("R*")
        ax.set_ylabel("G*")
        ax.set_zlabel("B*")
        ax.scatter(colours[:, 0], colours[:, 1], colours[:, 2], c=c)
        for i, c in enumerate(colours):
            ax.text(c[0], c[1], c[2], letters[i])
        if show_plot:
            plt.show()

        print("Characteristic colours: " + str(colours))
    return colours


# Extract characteristic colors from calibration image in relative colors.
pre_analysis = darsia.ConcentrationAnalysis(
    base=baseline.img_as(float),
    restoration=darsia.TVD(
        weight=0.025, eps=1e-4, max_num_iter=100, method="isotropic Bregman"
    ),
    **{"diff option": "plain"}
)
smooth_RGB = pre_analysis(image.img_as(float)).img
colours_RGB = extract_characteristic_data(
    signal=smooth_RGB, samples=samples, verbosity=True, show_plot=False
)

# ! ---- MAIN CONCENTRATION ANALYSIS ---- !


class KernelInterpolation:
    """General kernel-based interpolation."""

    def __init__(self, kernel, colours, concentrations):

        self.kernel = kernel

        # signal is rgb, transofrm to lab space because it is uniform and therefore
        # makes sense to interpolate in
        # signal = skimage.color.rgb2lab(signal)
        # colours = skimage.color.rgb2lab(colours)

        num_data = colours.shape[0]
        assert len(concentrations) == num_data, "Input data not compatible."

        x = np.array(colours)  # data points / control points / support points
        y = np.array(concentrations)  # goal points
        X = np.ones((num_data, num_data))  # kernel matrix
        for i in range(num_data):
            for j in range(num_data):
                X[i, j] = self.kernel(x[i], x[j])

        alpha = np.linalg.solve(X, y)

        # Cache
        self.x = x
        self.alpha = alpha

    def __call__(self, signal: np.ndarray):
        """Apply interpolation."""
        ph_image = np.zeros(signal.shape[:2])
        for n in range(self.alpha.shape[0]):
            ph_image += self.alpha[n] * self.kernel(signal, self.x[n])
        return ph_image


# ! ---- KERNEL MANAGEMENET -----


class BaseKernel(ABC):
    def __call__(self, x, y):
        pass


# define linear kernel shifted to avoid singularities
class LinearKernel(BaseKernel):
    def __init__(self, a=0):
        self.a = a

    def __call__(self, x, y):
        return np.sum(np.multiply(x, y), axis=-1) + self.a


# define gaussian kernel
class GaussianKernel(BaseKernel):
    def __init__(self, gamma=1):
        self.gamma = gamma

    def __call__(self, x, y):
        return np.exp(-self.gamma * np.sum(np.multiply(x - y, x - y), axis=-1))


# ! ---- MAIN ROUTINE ---- !

if False:
    # Reference solution

    # Convert a discrete ph stripe to a numeric pH indicator.
    color_to_concentration = KernelInterpolation(
        GaussianKernel(gamma=9.73), colours_RGB, concentrations
    )  # rgb: 9.73 , lab: 24.22
    ph_image = color_to_concentration(smooth_RGB)
    # gamma=10 value retrieved from ph analysis kernel calibration was best for c = 0.95
    # which also is physically meaningful

elif True:
    # Same but under the use of DarSIA
    analysis = darsia.ConcentrationAnalysis(
        base=baseline.img_as(float),
        restoration=darsia.TVD(
            weight=0.025, eps=1e-4, max_num_iter=100, method="isotropic Bregman"
        ),
        model=KernelInterpolation(
            GaussianKernel(gamma=9.73), colours_RGB, concentrations
        ),
        **{"diff option": "plain"}
    )
    ph_image = analysis(image.img_as(float)).img

plt.figure("cut ph val")
plt.plot(np.average(ph_image, axis=0))
plt.xlabel("horizontal pixel")
plt.ylabel("concentration")

ph_image[ph_image > 1] = 1  # für visualisierung von größer 1 values
ph_image[ph_image < 0] = 0
fig = plt.figure()
fig.suptitle("evolution of signal processing in a subregion")
ax = plt.subplot(212)
ax.imshow(ph_image)
# calibration patch for the kernel calibration visualization if needed
# rect = patches.Rectangle(
#     (1000, 80),
#     40,
#     40,
#     linewidth=1,
#     edgecolor="black",
#     facecolor="none",
# )
# ax.add_patch(rect)
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
