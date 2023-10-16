import numpy as np
import darsia
import skimage
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as patches
import string

# import matplotlib.patches as patches

folder = Path("images")
baseline_path = folder / Path("20220914-142404.TIF")
image_path = folder / Path("20220914-151727.TIF")
# image_path = Path(".data/test/singletracer.JPG")

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


# baseline_full = darsia.imread(baseline_path, transformations=transformations)
# image_full = darsia.imread(image_path, transformations=transformations)


# LAB
diff_LAB = (
    (skimage.color.rgb2lab(image.img) - skimage.color.rgb2lab(baseline.img))
    + [0, 128, 128]
) / [100, 255, 255]

# RGB
diff_RGB = skimage.img_as_float(image.img) - skimage.img_as_float(baseline.img)

# HSV
# diff = skimage.color.rgb2hsv(baseline.img) - skimage.color.rgb2hsv(image.img)

# Regularize
smooth_RGB = skimage.restoration.denoise_tv_bregman(
    diff_RGB, weight=0.025, eps=1e-4, max_num_iter=100, isotropic=True
)
smooth_LAB = skimage.restoration.denoise_tv_bregman(
    diff_LAB, weight=0.025, eps=1e-4, max_num_iter=100, isotropic=True
)

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

letters = list(string.ascii_uppercase)


def extract_support_points(signal, samples):
    # visualise patches
    fig, ax = plt.subplots()
    ax.imshow(np.abs(signal))  # visualise abs colours, because relative cols are neg
    ax.set_xlabel("horizontal pixel")
    ax.set_ylabel("vertical pixel")

    # double check number of patches
    n = np.shape(samples)[0]  # number of patches
    print("number of support patches: " + str(n))

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
        # ax.text(
        #     p[1].start + 130, p[0].start + 100, letters[i], fontsize=15, color="white"
        # )
        ax.text(p[1].start + 5, p[0].start + 95, letters[i], fontsize=12, color="white")
        ax.add_patch(rect)

        # histo analysis
        patch = signal[p]
        flat_image = np.reshape(patch, (10000, 3))  # all pixels in one dimension
        # patch visualisation
        # plt.figure("patch" + letters[i])
        # plt.imshow(np.abs(patch))
        H, edges = np.histogramdd(
            flat_image, bins=100, range=[(-1, 1), (-1, 1), (-1, 1)]
        )
        print(H.shape)
        index = np.unravel_index(H.argmax(), H.shape)
        col = [
            (edges[0][index[0]] + edges[0][index[0] + 1]) / 2,
            (edges[1][index[1]] + edges[1][index[1] + 1]) / 2,
            (edges[2][index[2]] + edges[2][index[2] + 1]) / 2,
        ]
        colours[i] = col

    c = np.abs(colours)
    plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("R*")
    ax.set_ylabel("G*")
    ax.set_zlabel("B*")
    ax.scatter(colours[:, 0], colours[:, 1], colours[:, 2], c=c)
    for i, c in enumerate(colours):
        ax.text(c[0], c[1], c[2], letters[i])

    print("characteristic colours: " + str(colours))
    return n, colours


n, colours_RGB = extract_support_points(signal=smooth_RGB, samples=samples)
n, colours_LAB = extract_support_points(signal=smooth_LAB, samples=samples)
concentrations = np.append(np.linspace(1, 0.99, n - 1), 0)


def color_to_concentration(
    k, colours, concentrations, signal: np.ndarray
) -> np.ndarray:
    # signal is rgb, transofrm to lab space because it is uniform and therefore
    # makes sense to interpolate in
    # signal = skimage.color.rgb2lab(signal)
    # colours = skimage.color.rgb2lab(colours)

    x = np.array(colours)  # data points / control points / support points
    y = np.array(concentrations)  # goal points
    X = np.ones((x.shape[0], x.shape[0]))  # kernel matrix
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            X[i, j] = k(x[i], x[j])

    alpha = np.linalg.solve(X, y)

    # Estimator / interpolant
    def estim(signal):
        sum = 0
        for n in range(alpha.shape[0]):
            sum += alpha[n] * k(signal, x[n])
        return sum

    # estim = scipy.interpolate.LinearNDInterpolator(colours, concentrations, 0)

    ph_image = np.zeros(signal.shape[:2])
    for i in range(signal.shape[0]):
        for j in range(signal.shape[1]):
            ph_image[i, j] = estim(signal[i, j])
    return ph_image


# define linear kernel shifted to avoid singularities
def k_lin(x, y, a=0):
    return np.inner(x, y) + a


# define gaussian kernel
def k_gauss(x, y, gamma=9.73):  # rgb: 9.73 , lab: 24.22
    return np.exp(-gamma * np.inner(x - y, x - y))


# Convert a discrete ph stripe to a numeric pH indicator.
ph_image = color_to_concentration(
    k_gauss, colours_RGB, concentrations, smooth_RGB
)  # gamma=10 value retrieved from ph analysis kernel calibration war bester punk für c=0.95
# was
# physikalisch am meisten sinn ergibt

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
