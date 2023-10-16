"""Module to extract characteristic data from input image for given patches."""

import string

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def extract_characteristic_data(
    signal: np.ndarray,
    samples: list[tuple[slice]],
    verbosity: bool = False,
    surpress_plot: bool = False,
) -> np.ndarray:
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

        # histogramm analysis
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
        if not surpress_plot:
            plt.show()

        print("Characteristic colours: " + str(colours))

    return colours
