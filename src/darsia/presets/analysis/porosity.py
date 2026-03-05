"""Helper to understand which labels can be grouped based on the color."""

from typing import Literal, Optional
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia
from darsia.presets.analysis.multichromatictracer import MultichromaticTracerAnalysis


class PorosityAnalysis(MultichromaticTracerAnalysis):
    def __init__(
        self,
        baseline: darsia.Image,
        labels: Optional[darsia.Image] = None,
        mode: Literal["full", "random_samples", "custom"] = "full",
        num_clusters: int = 5,
        num_attempts: int = 10,
        num_iterations: int = 100,
        eps: float = 1e-2,
        debug: bool = False,
        **kwargs,
    ):
        """Constructor.

        Args:
            baseline (Image): baseline image
            labels (Image, optional): labeled image
            mode (str, optional): mode for the analysis
            num_clusters (int, optional): number of clusters
            num_attempts (int, optional): number of attempts
            num_iterations (int, optional): number of iterations
            eps (float, optional): epsilon
            debug (bool, optional): debug flag
            kwargs: other keyword arguments

        """

        # If no segmentation is provided, create an empty one
        if labels is None:
            labels = darsia.zeros_like(baseline, mode="voxels", dtype=np.uint8)

        # Initialize the analysis object
        super().__init__(
            baseline,
            labels=labels,
            relative=False,
            show_plot=False,
            use_tvd=False,
            **kwargs,
        )

        # Initialize the analysis object and clip values
        baseline.img = np.clip(baseline.img, 0, 1)

        # Determine the gradient modulus of the image
        gradient = np.sqrt(
            sum(
                [
                    np.gradient(baseline.img[..., i], axis=j) ** 2
                    for i in range(3)
                    for j in range(2)
                ]
            )
        )

        # Collect characteristic colors for each label
        colors = []
        concentrations = []
        for mask in darsia.Masks(labels):
            # Determine size of mask
            mask_size = np.sum(mask.img)

            # Determine the gradient of the masked image
            gradient_mask = np.sum(gradient[mask.img]) / mask_size

            # Initialize colors and concentrations for mask
            colors_mask = []
            concentrations_mask = []

            # React to mode
            if mode == "full":
                samples = [(slice(0, None), slice(0, None))]
                # Raise warning that the full mode is time-consuming
                warn(
                    "Full mode is time-consuming. Consider using random_samples mode.",
                    RuntimeWarning,
                )
            elif mode == "random_samples":
                width = kwargs.get("sample_width", 200)
                num_patches = kwargs.get("sample_num", 10)
                samples = darsia.random_patches(
                    mask.img, width=width, num_patches=num_patches
                )
            elif mode == "custom":
                width = kwargs.get("sample_width", 200)
                assistant = darsia.BoxSelectionAssistant(
                    baseline.img, background=mask, width=width
                )
                samples = assistant()

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
                show_plot=debug,
            )

            for i in range(len(labels_collection)):
                label = labels_collection[i]
                palette = palette_collection[i]

                # Determine most common color
                _, counts = np.unique(label, return_counts=True)

                if False:
                    palette_with_counts = np.hstack((palette, counts.reshape(-1, 1)))
                    print(palette_with_counts)

                    plt.figure("Relative dominant colors")
                    c = np.clip(np.abs(palette), 0, 1)
                    c_counts = counts
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

                    plt.figure("Relative dominant colors 2")
                    c = np.clip(np.abs(palette), 0, 1)
                    c_counts = counts
                    ax = plt.axes(projection="3d")
                    ax.set_xlabel("R")
                    ax.set_ylabel("G")
                    ax.set_zlabel("B")
                    ax.scatter(
                        palette[:, 0],
                        palette[:, 1],
                        palette[:, 2],
                        c=c_counts,
                    )
                    plt.show()

                common_color = palette[np.argmax(counts)]
                # least_common_color = palette[np.argmin(counts)]
                # colors_mask.append(np.vstack((common_color, least_common_color)))
                # concentrations_mask.append([1, 0])

                # Determine the three most common colors
                # common_colors = []

                # Determine the most common color using a cluster analysis

                brightest_color = palette[np.argmax(np.linalg.norm(palette, axis=1))]
                darkest_color = palette[np.argmin(np.linalg.norm(palette, axis=1))]
                # median_dark_color = palette[
                #    np.argsort(np.linalg.norm(palette, axis=1))[len(palette) // 2]
                # ]

                # colors_mask.append(
                #    np.vstack((median_dark_color, brightest_color, darkest_color))
                # )
                # concentrations_mask.append([1, 0, 0])

                # colors_mask.append(
                #     np.vstack((common_color, brightest_color, darkest_color))
                # )
                # concentrations_mask.append([1, 0, 0])

                if True:
                    # Works OK
                    distance_to_bright = np.linalg.norm(brightest_color - common_color)
                    distance_to_dark = np.linalg.norm(darkest_color - common_color)

                    # print(
                    #     labels.img[mask.img].flatten()[0],
                    #     gradient_mask,
                    #     distance_to_bright,
                    #     distance_to_dark,
                    # )

                    # print(darkest_color, common_color, brightest_color)

                    # Determine which is further away from the common color
                    # Default paramter based on FF-UM analysis
                    tol_color_distance = kwargs.get("tol_color_distance", 0.1)
                    tol_color_gradient = kwargs.get("tol_color_gradient", 0.02)
                    if (
                        max(distance_to_bright, distance_to_dark) < tol_color_distance
                        or gradient_mask < tol_color_gradient
                    ):
                        colors_mask.append(
                            np.vstack((common_color, brightest_color, darkest_color))
                        )
                        concentrations_mask.append([1, 1, 1])
                    elif distance_to_bright > distance_to_dark:
                        colors_mask.append(np.vstack((common_color, brightest_color)))
                        concentrations_mask.append([1, 0])
                    else:
                        colors_mask.append(np.vstack((common_color, darkest_color)))
                        concentrations_mask.append([1, 0])

                # common_color_is_brightest = np.allclose(
                #     common_color, palette[brightest_color]
                # )
                # common_color_is_darkest = np.allclose(
                #     common_color, palette[darkest_color]
                # )

                ## Find the two points furthest away from each other.
                ## Build a distance matrix.
                # distance_matrix = np.zeros((num_clusters, num_clusters))
                # for i in range(num_clusters):
                #    for j in range(num_clusters):
                #        distance_matrix[i, j] = np.linalg.norm(palette[i] - palette[j])

                ## Reduce to colors at least as bright as the common color
                # for i in range(num_clusters):
                #    if np.linalg.norm(palette[i]) < np.linalg.norm(common_color):
                #        distance_matrix[i, :] *= 0.0
                #        distance_matrix[:, i] *= 0.0

                ## Check if common color is the brightest
                # if np.allclose(distance_matrix, 0):
                #    one_color = common_color
                #    zero_color = least_common_color

                # else:

                #    # Find max entry
                #    ind = np.unravel_index(
                #        np.argmax(distance_matrix, axis=None), distance_matrix.shape
                #    )
                #    i_max = ind[0]
                #    j_max = ind[1]

                #    # Determine which color is more common
                #    if counts[i_max] > counts[j_max]:
                #        one_color = palette[i_max]
                #        zero_color = palette[j_max]
                #    else:
                #        one_color = palette[j_max]
                #        zero_color = palette[i_max]

                # colors_mask.append(np.vstack((one_color, zero_color)))
                # concentrations_mask.append([1, 0])

                # Plot all data points to get an impression of the clustering - for debugging.
                if debug and False:
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
                    # ax.scatter(
                    #    zero_color[0], zero_color[1], zero_color[2], c="r", alpha=0.5
                    # )
                    # ax.scatter(
                    #    one_color[0], one_color[1], one_color[2], c="g", alpha=0.5
                    # )
                    ax.scatter(
                        common_color[0],
                        common_color[1],
                        common_color[2],
                        c="b",
                        alpha=0.5,
                    )

                    plt.figure("Masked baseline")
                    alpha = np.clip(mask.img.astype(float), 0.5, 1)
                    masked_baseline = np.dstack(
                        (skimage.img_as_float(baseline.img), alpha)
                    )
                    plt.imshow(masked_baseline)
                    plt.figure("Original baseline")
                    plt.imshow(baseline.img)
                    plt.figure("Labels")
                    plt.imshow(labels.img)
                    plt.show()

            # Append colors and concentrations for mask
            colors.append(np.vstack(colors_mask))
            concentrations.append(np.hstack(concentrations_mask))

        self.calibrate(colors, concentrations)

    def __call__(self, img: darsia.Image) -> darsia.Image:
        """Apply the analysis to an image.

        Args:
            img (darsia.Image): image

        Returns:
            darsia.Image: porosity image

        """
        porosity = super().__call__(img)
        # plt.figure("porosity pre clip")
        # plt.imshow(porosity.img)
        porosity.img = np.clip(porosity.img, 0, 1)
        # plt.figure("porosity post clip")
        # plt.imshow(porosity.img)
        return porosity

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


def patched_porosity_analysis(
    baseline: darsia.Image,
    patches: tuple[int, int] = (1, 1),
    labels: Optional[darsia.Image] = None,
    mode: Literal["full", "random_samples", "custom"] = "full",
    num_clusters: int = 5,
    num_attempts: int = 10,
    num_iterations: int = 100,
    eps: float = 1e-2,
    debug: bool = False,
    **kwargs,
) -> darsia.Image:
    """Patched approach to porosity analysis.

    Args:
        patches (tuple[int, int]): number of patches
        otherwise: see PorosityAnalysis

    Returns:
        darsia.Image: porosity image based on baseline

    """
    porosity = darsia.zeros_like(baseline, mode="voxels", dtype=float)
    patch_size = (np.ceil(np.array(baseline.num_voxels) / np.array(patches))).astype(
        int
    )
    for i in range(patches[0]):
        for j in range(patches[1]):
            subregion = (
                slice(i * patch_size[0], (i + 1) * patch_size[0]),
                slice(j * patch_size[1], (j + 1) * patch_size[1]),
            )
            sub_baseline = baseline.subregion(subregion)
            sub_labels = labels.subregion(subregion)
            if np.any(sub_baseline.shape == 0):
                continue

            try:
                porosity_analysis = PorosityAnalysis(
                    baseline=sub_baseline,
                    labels=sub_labels,
                    mode=mode,
                    num_clusters=num_clusters,
                    num_attempts=num_attempts,
                    num_iterations=num_iterations,
                    eps=eps,
                    debug=debug,
                    **kwargs,
                )
                porosity.img[subregion] = porosity_analysis(sub_baseline).img
            except Exception as e:
                warn(
                    f"Porosity analysis failed for subregion {subregion}: {e}",
                    UserWarning,
                )
                porosity.img[subregion] = 1.0

    return porosity
