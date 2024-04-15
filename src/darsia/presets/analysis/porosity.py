"""Helper to understand which labels can be grouped based on the color."""

from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np

import darsia
from darsia.presets.analysis.multichromatictracer import MultichromaticTracerAnalysis


class PorosityAnalysis(MultichromaticTracerAnalysis):

    def __init__(
        self,
        baseline: darsia.Image,
        labels: Optional[darsia.Image] = None,
        mode: Literal["full", "random_samples"] = "full",
        debug: bool = False,
        **kwargs
    ):

        # If no segmentation is provided, create an empty one
        if labels is not None:
            labels = darsia.zeros_like(baseline, mode="voxels", dtype=np.uint8)

        # Initialize the analysis object
        super().__init__(
            baseline,
            labels=labels,
            relative=False,
            show_plot=False,
            use_tvd=False,
            **kwargs
        )

        # Initialize the analysis object and clip values
        baseline.img = np.clip(baseline.img, 0, 1)

        # Hardcoded values - assume colored images
        # Tuning parameter: number of clusters
        num_clusters = kwargs.get("num_clusters", 5)
        num_attempts = kwargs.get("num_attempts", 10)
        num_iterations = kwargs.get("num_iterations", 100)
        eps = kwargs.get("eps", 1e-2)

        # Collect characteristic colors for each label
        # one_data = []
        # zero_data = []
        colors = []
        concentrations = []
        for mask in darsia.Masks(labels):

            # React to mode
            if mode == "full":
                samples = [(slice(0, None), slice(0, None))]
            elif mode == "random_samples":
                width = kwargs.get("sample_width", 200)
                num_patches = kwargs.get("sample_num", 10)
                samples = darsia.random_patches(
                    mask.img, width=width, num_patches=num_patches
                )

            # Determine labels and palette for each sample
            labels_collection, palette_collection = darsia.extract_characteristic_data(
                baseline.img,
                mask=mask.img,
                samples=samples,
                num_clusters=num_clusters,
                num_attempts=num_attempts,
                num_iterations=num_iterations,
                eps=eps,
                mode="all",
                show_plot=False,
            )

            for i in range(len(labels_collection)):
                labels = labels_collection[i]
                palette = palette_collection[i]

                # Determine most common color
                _, counts = np.unique(labels, return_counts=True)
                common_color = palette[np.argmax(counts)]
                least_common_color = palette[np.argmin(counts)]

                # Find the two points furthest away from each other.
                # Build a distance matrix.
                distance_matrix = np.zeros((num_clusters, num_clusters))
                for i in range(num_clusters):
                    for j in range(num_clusters):
                        distance_matrix[i, j] = np.linalg.norm(palette[i] - palette[j])

                # Reduce to colors at least as bright as the common color
                for i in range(num_clusters):
                    if np.linalg.norm(palette[i]) < np.linalg.norm(common_color):
                        distance_matrix[i, :] *= 0.0
                        distance_matrix[:, i] *= 0.0

                # Check if common color is the brightest
                if np.allclose(distance_matrix, 0):

                    one_color = common_color
                    zero_color = least_common_color

                else:

                    # Find max entry
                    ind = np.unravel_index(
                        np.argmax(distance_matrix, axis=None), distance_matrix.shape
                    )
                    i_max = ind[0]
                    j_max = ind[1]

                    # Determine which color is more common
                    if counts[i_max] > counts[j_max]:
                        one_color = palette[i_max]
                        zero_color = palette[j_max]
                    else:
                        one_color = palette[j_max]
                        zero_color = palette[i_max]

                colors.append(np.vstack((one_color, zero_color)))
                concentrations.append([1, 0])

            # Plot all data points to get an impression of the clustering - for debugging.
            if debug:
                c = np.clip(np.abs(palette), 0, 1)
                plt.figure("Relative dominant colors")
                ax = plt.axes(projection="3d")
                ax.set_xlabel("R")
                ax.set_ylabel("G")
                ax.set_zlabel("B")
                ax.scatter(
                    palette[:, 0],
                    palette[:, 1],
                    palette[:, 2],
                    c=c,
                )
                ax.scatter(
                    zero_color[0], zero_color[1], zero_color[2], c="r", alpha=0.5
                )
                ax.scatter(one_color[0], one_color[1], one_color[2], c="g", alpha=0.5)
                ax.scatter(
                    common_color[0], common_color[1], common_color[2], c="b", alpha=0.5
                )
                plt.figure("img")
                plt.imshow(baseline.img)
                plt.imshow(mask.img, alpha=0.3)
                plt.figure("img origial")
                plt.imshow(baseline.img)
                plt.figure("labels")
                plt.imshow(labels.img)
                plt.show()

        self.calibrate(colors, concentrations)

    def cut_off_small_values(
        self, porosity: darsia.Image, threshold: float = 0.5
    ) -> darsia.Image:
        """Cut off small values in the porosity image.

        Args:
            porosity (darsia.Image): porosity image
            threshold (float): threshold value

        Returns:
            darsia.Image: porosity image

        """
        porosity_copy = porosity.copy()
        porosity_copy.img = np.clip(porosity_copy.img, 0, 1)
        porosity_copy.img[porosity_copy.img < threshold] = 0
        return porosity_copy
