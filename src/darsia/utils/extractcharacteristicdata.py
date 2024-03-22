"""Module to extract characteristic data from input image for given patches."""

import string
from typing import Optional
from warnings import warn

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def extract_characteristic_data(
    signal: np.ndarray,
    mask: Optional[np.ndarray] = None,
    samples: Optional[list[tuple[slice]]] = None,
    filter: callable = lambda x: x,
    num_clusters: int = 5,
    num_attempts: int = 100,
    num_iterations: int = 200,
    eps: float = 1e-1,
    show_plot: bool = False,
) -> np.ndarray:
    """Assistant to extract representative colors from input image for given patches.

    Args:
        signal (np.ndarray): input signal, assumed to have the structure of a 2d,
            colored image.
        mask (np.ndarray): boolean array flagging pixels of interest; by default all pixels
            considered.
        samples (list of slices): list of 2d regions of interest; if None, full region
            considered
        filter (callable): function to preprocess the signal before analysis, e.g.,
            Gaussian filter.
        num_clusters (int): number of clusters to be extracted from data.
        num_attempts (int): number of attempts to be performed to find the best clusters.
        num_iterations (int): max number of iterations in iterative procedure.
        eps (float): tolerance for stopping the iterative procedure.
        show_plot (boolean): flag controlling whether plots are displayed.

    Returns:
        np.ndarray: characteristic colors for chosen samples.

    """
    # Define default inputs if not provided
    if samples is None:
        samples = [(slice(0,None), slice(0, None))]

    # Init data vector
    data_dim = signal.shape[-1]
    if data_dim not in [1, 3]:
        data_dim = 1
        warn("Implicitly assume that the data is scalar.")
    num_samples = len(samples)  # number of patches
    data_clusters = np.zeros((num_samples, data_dim))

    # Alphabet useful for labeling in plots
    letters = list(string.ascii_uppercase) + list(string.ascii_lowercase)

    # Visualise patches
    if show_plot:
        _, ax = plt.subplots()
        ax.imshow(np.abs(signal))  # visualise data
        ax.set_xlabel("horizontal pixel")
        ax.set_ylabel("vertical pixel")

    # Preprocess signal - TODO obsolete
    if False:
        signal = filter(signal).copy()

    # Analyze patches separately
    for i, p in enumerate(samples):
        # Control patch dimension
        assert len(p) == 2, "Patch must be 2d"

        # Cluster analysis for extracting dominant data/colors
        if False:
            patch = signal[p]
        else:
            patch = filter(signal[p])
        flat_image = np.reshape(patch, (-1, data_dim))

        # Reduce to active pixels
        if mask is not None:
            flat_mask = np.ravel(mask[p])
            flat_image = flat_image[flat_mask]

        pixels = np.float32(flat_image)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            num_iterations,
            eps,
        )
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(
            pixels, num_clusters, None, criteria, num_attempts, flags
        )
        _, counts = np.unique(labels, return_counts=True)
        data_clusters[i] = palette[np.argmax(counts)]

        # Visualise patches on image
        if show_plot:
            if None not in [p[0].start, p[0].stop, p[1].start, p[1].stop]:
                rect = patches.Rectangle(
                    (p[1].start, p[0].start),
                    p[1].stop - p[1].start,
                    p[0].stop - p[0].start,
                    # linewidth=1,
                    # edgecolor="w",
                    linewidth=2,
                    edgecolor=[1, 100.0 / 255.0, 160.0 / 255.0],
                    facecolor="none",
                )
                # ax.text(
                #    int(p[1].start + 0.05 * (p[1].stop - p[1].start)),
                #    int(p[0].start + 0.95 * (p[0].stop - p[0].start)),
                #    letters[i],
                #    fontsize=12,
                #    color="white",
                # )
                ax.add_patch(rect)

    if show_plot:

        if mask is not None:
            plt.imshow(mask, alpha = 0.2)

        if data_dim == 3:
            warn("Assuming data is color data and using RGB as axes.")
            c = np.clip(np.abs(data_clusters), 0, 1)
            plt.figure("Relative dominant colors")
            ax = plt.axes(projection="3d")
            ax.set_xlabel("R")
            ax.set_ylabel("G")
            ax.set_zlabel("B")
            # ax.set_xlim(0.57, 0.88)
            # ax.set_ylim(0.42, 0.78)
            # ax.set_zlim(0.17, 0.53)
            ax.scatter(
                data_clusters[:, 0], data_clusters[:, 1], data_clusters[:, 2], c=c
            )
            #            for i, c in enumerate(data_clusters):
            #                ax.text(c[0], c[1], c[2], letters[i])
            # plt.savefig("rgb_cube.png", dpi=800, transparent=True)
        plt.show()

    return data_clusters
