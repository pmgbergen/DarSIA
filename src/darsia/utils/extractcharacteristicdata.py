"""Module to extract characteristic data from input image for given patches."""

import string

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def extract_characteristic_data(
    signal: np.ndarray,
    samples: list[tuple[slice]],
    show_plot: bool = False,
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
    # Init color vector
    num_samples = len(samples)  # number of patches
    colors = np.zeros((num_samples, 3))

    # Alphabet useful for labeling in plots
    letters = list(string.ascii_uppercase)

    # Visualise patches
    if show_plot:
        _, ax = plt.subplots()
        ax.imshow(np.abs(signal))  # visualise abs colors, because relative cols are neg
        ax.set_xlabel("horizontal pixel")
        ax.set_ylabel("vertical pixel")

    # Analyze patches separately
    for i, p in enumerate(samples):
        # Control patch dimension
        assert len(p) == 2, "Patch must be 2d"

        # Histogramm analysis for picking the dominant color
        patch = signal[p]
        flat_image = np.reshape(patch, (-1, 3))
        H, edges = np.histogramdd(
            flat_image, bins=100, range=[(-1, 1), (-1, 1), (-1, 1)]
        )
        index = np.unravel_index(H.argmax(), H.shape)
        col = [
            edges[0][index[0]],
            edges[1][index[1]],
            edges[2][index[2]],
        ]
        colors[i] = col

        # Visualise patches on image
        if show_plot:
            rect = patches.Rectangle(
                (p[1].start, p[0].start),
                p[1].stop - p[1].start,
                p[0].stop - p[0].start,
                linewidth=1,
                edgecolor="w",
                facecolor="none",
            )
            ax.text(
                int(p[1].start + 0.05 * (p[1].stop - p[1].start)),
                int(p[0].start + 0.95 * (p[0].stop - p[0].start)),
                letters[i],
                fontsize=12,
                color="white",
            )
            ax.add_patch(rect)

    if show_plot:
        # 3d plot?
        c = np.abs(colors)
        plt.figure("Relative colors")
        ax = plt.axes(projection="3d")
        ax.set_xlabel("R")
        ax.set_ylabel("G")
        ax.set_zlabel("B")
        ax.scatter(colors[:, 0], colors[:, 1], colors[:, 2], c=c)
        for i, c in enumerate(colors):
            ax.text(c[0], c[1], c[2], letters[i])
        plt.show()

    return colors
