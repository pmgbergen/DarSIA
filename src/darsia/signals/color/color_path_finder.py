"""Provide tools for defining color paths."""

import copy
import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.manifold import LocallyLinearEmbedding
from dataclasses import dataclass
from time import time
from functools import wraps
from warnings import warn

import darsia

logger = logging.getLogger(__name__)


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        # Do not only use the name of the function, but also the class it belongs to
        logger.info(
            f"{func.__module__}.{func.__name__} executed in {end - start:.3f} seconds"
        )
        return result

    return wrapper


# ! ---- AUXILIARY FUNCTIONS ----


def _get_mean_color(
    image: darsia.Image, mask: darsia.Image | np.ndarray | None = None
) -> np.ndarray:
    """Calculate the mean color of an image, optionally masked by a boolean mask.

    Args:
        image (darsia.Image): The image from which to calculate the mean color.
        mask (darsia.Image | np.ndarray | None): Optional mask to apply on the image.
            If `None`, the entire image is used.

    Returns:
        np.ndarray: The mean color of the image, calculated as the average of RGB values.

    """
    if isinstance(mask, darsia.Image):
        subimage = image.img[mask.img]
    elif isinstance(mask, np.ndarray):
        subimage = image.img[mask]
    else:
        subimage = image.img
    return np.mean(subimage.reshape(-1, 3), axis=0)


# ! ---- INTERACTIVE COLOR PATH DEFINITION ----


def define_color_path(image: darsia.Image, mask: darsia.Image) -> darsia.ColorPath:
    """Interactive setup of a color path based on an image.

    Args:
        image (darsia.Image): The image to define the color path from.
        mask (darsia.Image): The mask to apply on the image.

    Returns:
        darsia.ColorPath: The defined color path with selected colors.

    """
    # Sanity checks
    assert mask.img.dtype == bool, "Mask must be a boolean mask."

    colors = []
    while True:
        assistant = darsia.RectangleSelectionAssistant(image)

        # Pick a box in the image and convert it to a mask
        box: tuple[slice, slice] = assistant()
        boxed_mask = darsia.zeros_like(mask, dtype=bool)
        boxed_mask.img[box] = mask.img[box]

        # Determine the mean color in the box
        mean_color = _get_mean_color(image, mask=boxed_mask)
        colors.append(mean_color)

        add_more = (
            input("Do you want to add another color to the path? (y/n) ")
            .strip()
            .lower()
        )
        if add_more != "y":
            break

    # Create a global color path
    return darsia.ColorPath(
        colors=colors,
        base_color=colors[0],
        mode="rgb",
    )


# ! ---- ALGORITHMIC COLOR PATH DEFINITION ----


# TODO ColorSpectrumRegression? ColorSpectrumAnalysis?
@dataclass
class ColorSpectrum:
    """Data structure to hold a color spectrum."""

    base_color: np.ndarray
    """Reference color for the spectrum."""
    spectrum: np.ndarray
    """Distribution of colors in the spectrum."""
    histogram: np.ndarray
    """Histogram of color occurrences."""


class ColorPathRegression:
    """Algorithmic definition of relative color paths through regression."""

    def __init__(
        self,
        labels: darsia.Image,
        mask: darsia.Image,
        ignore_labels: list[int] = [],
    ) -> None:
        self.labels = labels
        self.mask = mask
        self.ignore_labels = ignore_labels
        self.range = 1.5

    @timing_decorator
    def get_base_colors(
        self, image: darsia.Image, verbose: bool = False
    ) -> dict[int, np.ndarray]:
        """Get the base colors for each label in the image.

        Args:
            image (darsia.Image): The image from which to extract base colors.
            verbose (bool): Whether to print additional information.

        Returns:
            dict[int, np.ndarray]: A dictionary mapping each label to its base color.

        """
        base_colors = {}
        for mask, label in darsia.Masks(self.labels, return_label=True):
            if label in self.ignore_labels:
                base_colors[label] = np.zeros(3)
                continue
            base_colors[label] = _get_mean_color(image, mask=(self.mask.img & mask.img))

        if verbose:
            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot(111, projection="3d")
            for label in np.unique(self.labels.img):
                if label in self.ignore_labels:
                    continue
                color = base_colors[label]
                print(f"Label {label}: {color}")
                ax.scatter(color[0], color[1], color[2], c=color)
            ax.set_xlabel("R")
            ax.set_ylabel("G")
            ax.set_zlabel("B")
            ax.set_title("Base colors in RGB space")
            plt.show()

        return base_colors

    @timing_decorator
    def get_mean_base_color(self, image: darsia.Image) -> np.ndarray:
        """Get the mean base color across all labels in the image.

        Args:
            image (darsia.Image): The image from which to extract base colors.

        Returns:
            np.ndarray: The mean base color across all labels.
        """
        base_colors = self.get_base_colors(image)
        return np.mean(np.array(list(base_colors.values())), axis=0)

    @timing_decorator
    def base_color_image(self, image: darsia.Image) -> darsia.Image:
        """Create an image where each label is colored by its base color.

        Args:
            image (darsia.Image): The image from which to extract base colors.

        Returns:
            darsia.Image: An image where each label is colored by its base color.
        """
        base_color_image = image.copy()
        base_colors = self.get_base_colors(image)
        for mask, label in darsia.Masks(self.labels, return_label=True):
            base_color_image.img[mask.img, :] = base_colors[label]
        return base_color_image

    @timing_decorator
    def get_color_spectrum(
        self,
        images: list[darsia.Image],
        baseline: darsia.Image | None = None,
        resolution: tuple[int, int, int] = (11, 11, 11),
        ignore_color_spectrum: dict[int, ColorSpectrum] | None = None,
        threshold_zero: float = 0.0,
        threshold_significant: float = 0.0,
        verbose: bool = False,
    ) -> dict[int, ColorSpectrum]:
        """Get the color spectrum for each label in the image.

        The color spectrum is calculated by analyzing the distribution of
        colors within each label across the provided images. This involves
        creating a 3D histogram of color occurrences, which is then normalized
        to identify significant colors.

        Args:
            images (list[darsia.Image]): The images to analyze.
            baseline (darsia.Image | None): The baseline image for comparison.
            resolution (tuple[int, int, int]): The resolution of the color histogram.
            ignore_color_spectrum (dict[int, tuple[np.ndarray, np.ndarray]] | None): Colors to ignore in the spectrum.
            threshold_zero (float): The threshold for zeroing out insignificant colors.
            threshold_significant (float): The threshold for significant colors.
            verbose (bool): Whether to print verbose output.

        Returns:
            dict[int, ColorSpectrum]: The color spectrum for each label.

        """
        # TODO introduce a mask, and remove labels.

        # Get base colors for each label
        if baseline is None:
            base_colors = {label: np.zeros(3) for label in np.unique(self.labels.img)}
        else:
            base_colors = self.get_base_colors(baseline)

        # Prepare analysis and result
        color_spectrum = dict(
            (
                label,
                ColorSpectrum(
                    base_color=base_color,
                    histogram=np.zeros(resolution, dtype=float),
                    spectrum=np.zeros(resolution, dtype=bool),
                ),
            )
            for label, base_color in base_colors.items()
        )

        # Loop over all images
        for image in images:
            # Get relative image
            relative_image = image.copy()
            if baseline is not None:
                relative_image.img -= baseline.img
            relative_image.img[~self.mask.img] = 0.0

            for label in np.unique(self.labels.img):
                # Fetch relative colors in the label
                relative_colors = relative_image.img[self.labels.img == label].reshape(
                    (-1, 3)
                )

                # Remove zero data
                if not np.isclose(threshold_zero, 0.0):
                    relative_colors = relative_colors[
                        np.linalg.norm(relative_colors, axis=1) > threshold_zero
                    ]

                # Put the data into a 3D histogram
                single_hist, _ = np.histogramdd(
                    relative_colors,
                    bins=resolution,
                    range=(
                        (-self.range, self.range),
                        (-self.range, self.range),
                        (-self.range, self.range),
                    ),
                )

                # Ignore colors
                if ignore_color_spectrum is not None:
                    ignore = ignore_color_spectrum[label].spectrum
                    assert ignore.shape == resolution
                    assert ignore.dtype == bool
                    single_hist[ignore] = 0.0

                color_spectrum[label].histogram += single_hist

        # Normalize the histogram, and identify the significant colors
        for label in color_spectrum.keys():
            color_spectrum[label].histogram = color_spectrum[label].histogram / np.sum(
                color_spectrum[label].histogram
            )
            color_spectrum[label].spectrum = (
                color_spectrum[label].histogram > threshold_significant
            )

        # Plot the histogram as a 3D voxel plot
        if verbose:
            # Prepare reusable data
            _, edges = np.histogramdd(
                np.zeros((1, 3)),
                bins=resolution,
                range=(
                    (-self.range, self.range),
                    (-self.range, self.range),
                    (-self.range, self.range),
                ),
            )
            # Define the respective meshgrid for the histogram
            x_edges = edges[0][:-1] + 0.5 * (edges[0][1] - edges[0][0])
            y_edges = edges[1][:-1] + 0.5 * (edges[1][1] - edges[1][0])
            z_edges = edges[2][:-1] + 0.5 * (edges[2][1] - edges[2][0])
            x_mesh, y_mesh, z_mesh = np.meshgrid(
                x_edges, y_edges, z_edges, indexing="ij"
            )

            for label in np.unique(self.labels.img):
                if not np.any(color_spectrum[label].spectrum):
                    logger.info(f"Skip plotting color spectrum for label {label}.")
                    continue
                ax = plt.figure().add_subplot(projection="3d")
                ax.set_title(f"Color Spectrum for Label {label}")

                # Add colors to each mesh point
                c_mesh = np.clip(
                    color_spectrum[label].base_color
                    + np.vstack(
                        (x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten())
                    ).T,
                    0.0,
                    1.0,
                ).reshape(resolution + (3,))

                # Plot the significant boxes
                ax.voxels(color_spectrum[label].spectrum, facecolors=c_mesh)
                # Mark the ignored colors
                if ignore_color_spectrum is not None:
                    ax.voxels(
                        ignore_color_spectrum[label].spectrum,
                        facecolors="red",
                        alpha=0.1,
                    )
                # Highlight the origin
                origin = np.zeros_like(color_spectrum[label].spectrum, dtype=bool)
                origin[
                    origin.shape[0] // 2, origin.shape[1] // 2, origin.shape[2] // 2
                ] = True
                ax.voxels(origin, facecolors="red")
                plt.show()

        logger.info("Done. Color spectrum analysis.")
        return color_spectrum

    @timing_decorator
    def expand_color_spectrum(
        self,
        color_spectrum: dict[int, ColorSpectrum],
        min_points: int = 6,
        verbose: bool = False,
    ) -> dict[int, ColorSpectrum]:
        """Expand the color spectrum through linear regression.

        Args:
            color_spectrum (dict[int, ColorSpectrum]): The color spectrum to expand.
            verbose (bool): Whether to print additional information.
            min_points (int): Minimum number of significant points to perform regression.

        Returns:
            dict[int, ColorSpectrum]: The expanded color spectrum.

        """
        # Prepare the result
        expanded_color_spectrum = copy.deepcopy(color_spectrum)

        for label, data in color_spectrum.items():
            spectrum = data.spectrum
            x_points, y_points, z_points = np.where(spectrum)

            # Step 0: Add 8 neighbors to significant voxels
            spectrum[1:-1, 1:-1, 1:-1] |= (
                spectrum[:-2, 1:-1, 1:-1]
                | spectrum[2:, 1:-1, 1:-1]
                | spectrum[1:-1, :-2, 1:-1]
                | spectrum[1:-1, 2:, 1:-1]
                | spectrum[1:-1, 1:-1, :-2]
                | spectrum[1:-1, 1:-1, 2:]
                | spectrum[:-2, :-2, 1:-1]
                | spectrum[2:, 2:, 1:-1]
                | spectrum[:-2, 1:-1, :-2]
                | spectrum[2:, 1:-1, 2:]
                | spectrum[1:-1, :-2, :-2]
                | spectrum[1:-1, 2:, 2:]
                | spectrum[:-2, :-2, :-2]
                | spectrum[2:, 2:, 2:]
                | spectrum[:-2, 2:, :-2]
                | spectrum[2:, :-2, 2:]
                | spectrum[1:-1, :-2, 2:]
                | spectrum[1:-1, 2:, :-2]
                | spectrum[:-2, 1:-1, 2:]
                | spectrum[2:, 1:-1, :-2]
                | spectrum[:-2, 2:, 1:-1]
                | spectrum[2:, :-2, 1:-1]
                | spectrum[1:-1, :-2, 1:-1]
                | spectrum[1:-1, 2:, 1:-1]
                | spectrum[:-2, 1:-1, 2:]
                | spectrum[2:, 1:-1, :-2]
                | spectrum[:-2, :-2, 2:]
                | spectrum[2:, 2:, :-2]
                | spectrum[:-2, 2:, 2:]
                | spectrum[2:, :-2, 2:]
                | spectrum[1:-1, :-2, :-2]
                | spectrum[1:-1, 2:, 2:]
                | spectrum[:-2, 1:-1, :-2]
                | spectrum[2:, 1:-1, 2:]
                | spectrum[:-2, :-2, 1:-1]
                | spectrum[2:, 2:, 1:-1]
                | spectrum[:-2, 2:, 1:-1]
                | spectrum[2:, :-2, 1:-1]
            )

            # Step 1: Extract significant points and convert to relative colors in the range [-1.5, 1.5]
            relative_colors = (
                np.vstack((x_points, y_points, z_points)).T
                * 2
                * self.range
                / spectrum.shape[0]
                - self.range
            )
            num_points = relative_colors.shape[0]
            if num_points <= min_points:
                continue

            # Step 2: Reduce to 1D using Locally Linear Embedding
            lle = LocallyLinearEmbedding(
                n_neighbors=min(10, num_points - 1), n_components=1
            )
            embedding = lle.fit_transform(relative_colors)

            # Step 3: Sort embedding and colors
            sorted_indices = np.argsort(embedding[:, 0])
            sorted_embedding = embedding[sorted_indices]
            sorted_relative_colors = relative_colors[sorted_indices]

            # Prepare weights for linear regression. Use the norm of the relative colors
            # to give more weight to colors further away from the base color.
            weights = np.linalg.norm(relative_colors, axis=1)
            weights /= np.sum(weights)
            sorted_weights = weights[sorted_indices]
            sorted_weights *= num_points  # Scale weights to number of points

            # Step 4: Fit line segment
            model = LinearRegression().fit(
                sorted_embedding, sorted_relative_colors, sorted_weights
            )

            resolution = 10 * spectrum.shape[0]
            coef = 1.5 / np.mean(model.coef_) / resolution * model.coef_
            expanded_relative_colors = np.empty((0, 3))
            for i in range(-2 * resolution, 2 * resolution + 1):
                shifted_colors = relative_colors + i * coef.flatten()
                expanded_relative_colors = np.vstack(
                    (expanded_relative_colors, shifted_colors)
                )

            # Same histogram as before
            expanded_histogram, _ = np.histogramdd(
                expanded_relative_colors,
                bins=spectrum.shape,
                range=(
                    (-self.range, self.range),
                    (-self.range, self.range),
                    (-self.range, self.range),
                ),
            )
            expanded_histogram = expanded_histogram / np.sum(expanded_histogram)
            expanded_spectrum = expanded_histogram > 0.0

            expanded_color_spectrum[label].histogram = expanded_histogram
            expanded_color_spectrum[label].spectrum = expanded_spectrum

            # Add same verbose plotting as for get_color_spectrum
            if verbose:
                # Prepare reusable data
                _, edges = np.histogramdd(
                    np.zeros((1, 3)),
                    bins=spectrum.shape,
                    range=(
                        (-self.range, self.range),
                        (-self.range, self.range),
                        (-self.range, self.range),
                    ),
                )
                # Define the respective meshgrid for the histogram
                x_edges = edges[0][:-1] + 0.5 * (edges[0][1] - edges[0][0])
                y_edges = edges[1][:-1] + 0.5 * (edges[1][1] - edges[1][0])
                z_edges = edges[2][:-1] + 0.5 * (edges[2][1] - edges[2][0])
                x_mesh, y_mesh, z_mesh = np.meshgrid(
                    x_edges, y_edges, z_edges, indexing="ij"
                )

                ax = plt.figure().add_subplot(projection="3d")
                ax.set_title(f"Expanded Color Spectrum for Label {label}")

                # Add colors to each mesh point
                c_mesh = np.clip(
                    color_spectrum[label].base_color
                    + np.vstack(
                        (x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten())
                    ).T,
                    0.0,
                    1.0,
                ).reshape(spectrum.shape + (3,))

                # Plot the significant boxes
                ax.voxels(expanded_spectrum, facecolors=c_mesh)
                # Highlight the origin
                origin = np.zeros_like(expanded_spectrum, dtype=bool)
                origin[
                    origin.shape[0] // 2,
                    origin.shape[1] // 2,
                    origin.shape[2] // 2,
                ] = True
                ax.voxels(origin, facecolors="red")
                plt.show()

        logger.info("Done. Expanded color spectrum analysis.")

        return expanded_color_spectrum

    @timing_decorator
    def find_relative_color_path(
        self,
        spectrum: ColorSpectrum,
        base_color_spectrum: ColorSpectrum | None = None,
        num_segments: int = 1,
        verbose: bool = False,
        **kwargs,
    ) -> darsia.ColorPath:
        """Find a relative color path through the significant boxes.

        Args:
            spectrum (ColorSpectrum): The color spectrum to analyze.
            base_color_spectrum (ColorSpectrum | None): The baseline color spectrum to ignore.
            num_segments (int): The number of segments for the color path.
            verbose (bool): Whether to print additional information.

        Returns:
            darsia.ColorPath: The relative color path through the significant boxes.

        """
        # Step 0: Prepare
        num_dofs = num_segments + 1

        # Step 1: Extract spectrum and convert to relative colors in the range [-1.5, 1.5]
        x_points, y_points, z_points = np.where(spectrum.spectrum)
        relative_colors = (
            np.vstack((x_points, y_points, z_points)).T
            * 2
            * self.range
            / spectrum.spectrum.shape[0]
            - self.range
        )
        absolute_colors = spectrum.base_color + relative_colors
        num_points = relative_colors.shape[0]

        # Check for empty color path - need at least two points to define a path
        if num_points <= 1:
            logger.info("""Empty color or path found. Returning default color path.""")
            return darsia.ColorPath(
                colors=None,
                relative_colors=num_dofs * [np.zeros(3)],
                base_color=spectrum.base_color,
                values=np.linspace(0.0, 1.0, num_dofs).tolist(),
                mode="rgb",
            )

        # Add origin to relative colors and use high weight 1.
        origin = np.zeros(3)
        relative_colors = np.vstack((relative_colors, origin))
        absolute_colors = np.vstack((absolute_colors, spectrum.base_color))

        # Step 2: Reduce to 1D using Locally Linear Embedding
        n_neighbors = min(10, num_points - 1)
        lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=1)
        embedding = lle.fit_transform(relative_colors)

        # Step 3: Sort embedding and colors
        sorted_indices = np.argsort(embedding[:, 0])
        sorted_embedding = embedding[sorted_indices]
        sorted_relative_colors = relative_colors[sorted_indices]

        # Step 3.2: Identify "left" part from origin and remove it
        origin_index = np.where(np.all(sorted_relative_colors == origin, axis=1))[0][0]
        # Assume origin to be close to the extreme left, if exreme right, flip everything
        if origin_index > len(sorted_relative_colors) // 2:
            origin_index = len(sorted_relative_colors) - origin_index - 1
            sorted_embedding = np.flip(sorted_embedding, axis=0)
            sorted_relative_colors = np.flip(sorted_relative_colors, axis=0)
        sorted_embedding = sorted_embedding[origin_index:, :]
        sorted_relative_colors = sorted_relative_colors[origin_index:, :]

        # Initialize segments
        segments = []

        # Find the two representative key colors representing the start and end of the path
        # Hardcode the origin to be the first key color.
        segment_range = range(0, len(sorted_embedding))
        segment_embedding = sorted_embedding[segment_range]
        segment_relative_colors = sorted_relative_colors[segment_range]
        segment_interpolator = LinearRegression().fit(
            [segment_embedding[0], segment_embedding[-1]],
            [segment_relative_colors[0], segment_relative_colors[-1]],
        )
        segment_error = np.sum(
            np.linalg.norm(
                segment_interpolator.predict(segment_embedding)
                - segment_relative_colors,
                axis=1,
                ord=1,
            )
        )
        segment = {
            "range": segment_range,
            "error": segment_error,
        }
        segments.append(segment)

        # Perform a Ramer-Douglas-Peucker-like segmentation based on weighted least squares
        # error reduction
        while len(segments) < num_segments:
            # Eligible segments are those with at least 3 points
            eligible_segments = [seg for seg in segments if len(seg["range"]) > 2]

            # Stop if no segment can be split further
            if len(eligible_segments) == 0:
                warn("Cannot split segments further. Stopping early.")
                break

            # Identify the segment with the highest error, it needs to have at least 3 points
            segment_to_split_index = np.argmax(
                [seg["error"] for seg in eligible_segments]
            )
            segment_to_split = eligible_segments[segment_to_split_index]

            # Find the best split point for the selected segment
            min_error = float("inf")

            for split_point in range(1, len(segment_to_split["range"]) - 1):
                left_segment_range = segment_to_split["range"][:split_point]
                right_segment_range = segment_to_split["range"][split_point:]
                left_segment_embedding = sorted_embedding[left_segment_range]
                right_segment_embedding = sorted_embedding[right_segment_range]
                left_segment_relative_colors = sorted_relative_colors[
                    left_segment_range
                ]
                right_segment_relative_colors = sorted_relative_colors[
                    right_segment_range
                ]

                # Fit models to both sides
                left_segment_interpolator = LinearRegression().fit(
                    [left_segment_embedding[0], left_segment_embedding[-1]],
                    [left_segment_relative_colors[0], left_segment_relative_colors[-1]],
                )
                right_segment_interpolator = LinearRegression().fit(
                    [right_segment_embedding[0], right_segment_embedding[-1]],
                    [
                        right_segment_relative_colors[0],
                        right_segment_relative_colors[-1],
                    ],
                )

                # Compute the error for this split
                left_error = np.sum(
                    np.linalg.norm(
                        left_segment_interpolator.predict(left_segment_embedding)
                        - left_segment_relative_colors,
                        axis=1,
                        ord=1,
                    )
                )
                right_error = np.sum(
                    np.linalg.norm(
                        right_segment_interpolator.predict(right_segment_embedding)
                        - right_segment_relative_colors,
                        axis=1,
                        ord=1,
                    )
                )
                total_error = max(left_error, right_error)

                # Update the best split if this one is better
                if total_error < min_error:
                    min_error = total_error

                    left_segment = {
                        "range": left_segment_range,
                        "error": left_error,
                    }
                    right_segment = {
                        "range": right_segment_range,
                        "error": right_error,
                    }
            # Replace the selected segment with the two new segments
            global_segment_to_split_index = segments.index(segment_to_split)
            segments[global_segment_to_split_index] = left_segment
            segments.insert(global_segment_to_split_index + 1, right_segment)

        # Extract the key relative colors from the segments
        key_relative_colors: list[np.ndarray] = [
            sorted_relative_colors[segment["range"].start] for segment in segments
        ] + [sorted_relative_colors[segments[-1]["range"].stop - 1]]

        max_error = max(seg["error"] for seg in segments)

        if key_relative_colors == []:
            logger.info("No key colors found. Returning default color path.")
            return darsia.ColorPath(
                colors=None,
                relative_colors=num_dofs * [np.zeros(3)],
                base_color=spectrum.base_color,
                values=np.linspace(0.0, 1.0, num_dofs).tolist(),
                mode="rgb",
            )

        # Step 5: Define the corresponding absolute colors
        key_absolute_colors = [spectrum.base_color + c for c in key_relative_colors]

        # Step 6: Verbose output
        if verbose:
            # Print key colors
            for i, (color, rel) in enumerate(
                zip(key_absolute_colors, key_relative_colors)
            ):
                print(f"Key color {i + 1}: RGB = {color}, relative: {rel}")

            # Step 6: Visualize
            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                relative_colors[:, 0],
                relative_colors[:, 1],
                relative_colors[:, 2],
                c=np.clip(absolute_colors, 0, 1),
                s=10,
                alpha=0.5,
            )
            if np.array(key_relative_colors).shape[0] == num_dofs:
                ax.plot(*np.array(key_relative_colors).T, c="black", linewidth=2)
                ax.scatter(
                    *np.array(key_relative_colors).T,
                    c=np.clip(key_absolute_colors, 0, 1),
                    s=100,
                )
            if base_color_spectrum is not None:
                base_x, base_y, base_z = np.where(base_color_spectrum.spectrum)
                base_relative_colors = (
                    np.vstack((base_x, base_y, base_z)).T
                    * 2
                    * self.range
                    / base_color_spectrum.spectrum.shape[0]
                    - self.range
                )
                ax.scatter(
                    base_relative_colors[:, 0],
                    base_relative_colors[:, 1],
                    base_relative_colors[:, 2],
                    c="r",
                    s=10,
                    alpha=0.1,
                )
            plot_title = kwargs.get("plot_title", "Color Path Analysis")
            ax.set_title(plot_title)
            ax.set_xlabel("R")
            ax.set_ylabel("G")
            ax.set_zlabel("B")

            plt.tight_layout()
            if kwargs.get("plot_save", False):
                plot_save_path = kwargs.get("plot_save_path", "color_path_analysis.png")
                dpi = kwargs.get("plot_save_dpi", 300)
                plt.savefig(plot_save_path, dpi=dpi)
            plt.show()

        # Step 7: Construct relative color path
        relative_color_path = darsia.ColorPath(
            colors=None,
            relative_colors=key_relative_colors,
            base_color=spectrum.base_color,
            values=None,
            mode="rgb",
        )
        return relative_color_path

    def find_color_path_clusters(
        self, color_paths: dict[int, darsia.ColorPath]
    ) -> dict:
        """Find clusters of color paths based on their similarity."""
        ...

        # # Extract features from color paths
        # features = []
        # for path in color_paths.values():
        #     features.append(path.relative_colors.flatten())
        # features = np.array(features)

        # # Cluster color paths using DBSCAN
        # clustering = DBSCAN(eps=0.5, min_samples=2).fit(features)
        # labels = clustering.labels_

        # # Organize color paths into clusters
        # clusters = {}
        # for label, path in zip(labels, color_paths.values()):
        #     if label not in clusters:
        #         clusters[label] = []
        #     clusters[label].append(path)

        # return clusters

    def _color_to_index(
        self, color: np.ndarray, resolution: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        """Identify the voxel index of the base color in the [0,1]^3 color space given the resolution"""
        return tuple(
            int(np.clip(c * (r - 1), 0, r - 1)) for c, r in zip(color, resolution)
        )
