"""Provide tools for defining color paths."""

import copy
import logging

import numpy as np
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import darsia

logger = logging.getLogger(__name__)

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


# ! ---- ALGORITHMIC COLOR PATH DEFINITION ----


class LabelColorPathMapRegression:
    """Algorithmic definition of relative color paths through regression for labeled images."""

    def __init__(
        self,
        labels: darsia.Image,
        color_range: darsia.ColorRange,
        resolution: int = 11,
        mask: darsia.Image | None = None,  # TODO extend to allow for None
        ignore_labels: list[int] = [],
    ) -> None:
        self.labels = labels
        """Labeled image to define color paths for."""
        self.color_range = color_range
        """Color range for the regression."""
        self.discrete_color_range = darsia.DiscreteColorRange(
            color_range=color_range, resolution=resolution
        )
        """Discrete color range for regression."""
        self.mask = mask
        """Mask to apply when analyzing colors."""
        self.ignore_labels = ignore_labels
        """Labels to ignore during color path regression."""
        self.color_mode = color_range.color_mode
        """Color mode."""

        # Sanity check
        assert mask is not None
        if self.color_mode != darsia.ColorMode.RELATIVE:
            raise NotImplementedError("Only relative color ranges are supported.")

    @property
    def _shape(self) -> tuple[int, int, int]:
        """Get the shape of the discrete color range."""
        return 3 * (self.discrete_color_range.resolution,)

    @darsia.timing_decorator
    def get_base_colors(
        self, image: darsia.Image, verbose: bool = False
    ) -> darsia.LabelColorMap:
        """Get the base colors for each label in the image.

        Args:
            image (darsia.Image): The image from which to extract base colors.
            verbose (bool): Whether to print additional information.

        Returns:
            LabelColorMap: An object containing the base colors for each label.

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

        return darsia.LabelColorMap(base_colors)

    @darsia.timing_decorator
    def get_mean_base_color(self, image: darsia.Image) -> np.ndarray:
        """Get the mean base color across all labels in the image.

        Args:
            image (darsia.Image): The image from which to extract base colors.

        Returns:
            np.ndarray: The mean base color across all labels.
        """
        base_colors = self.get_base_colors(image)
        return base_colors.mean()

    @darsia.timing_decorator
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

    @darsia.timing_decorator
    def get_color_spectrum(
        self,
        images: list[darsia.Image],
        baseline: darsia.Image | None = None,
        ignore: darsia.LabelColorSpectrumMap | darsia.ColorSpectrum | None = None,
        threshold_zero: float = 0.0,
        threshold_significant: float = 0.0,
        verbose: bool = False,
    ) -> darsia.LabelColorSpectrumMap:
        """Get the color spectrum for each label in the image.

        The color spectrum is calculated by analyzing the distribution of
        colors within each label across the provided images. This involves
        creating a 3D histogram of color occurrences, which is then normalized
        to identify significant colors.

        Args:
            images (list[darsia.Image]): The images to analyze.
            baseline (darsia.Image | None): The baseline image for comparison.
            resolution (tuple[int, int, int]): The resolution of the color histogram.
            ignore (darsia.LabelColorSpectrumMap | darsia.ColorSpectrum | None): Colors to
                ignore in the spectrum.
            threshold_zero (float): The threshold for zeroing out insignificant colors.
            threshold_significant (float): The threshold for significant colors.
            verbose (bool): Whether to print verbose output.

        Returns:
            LabelColorSpectrumMap: The color spectrum for each label.

        """
        # TODO introduce a mask, and remove labels.

        # Get base colors for each label
        if baseline is None:
            base_colors = darsia.LabelColorMap(
                {label: np.zeros(3) for label in np.unique(self.labels.img)}
            )
        else:
            base_colors = self.get_base_colors(baseline)

        # Prepare analysis and result
        color_spectrum_map = darsia.LabelColorSpectrumMap(
            dict(
                (
                    label,
                    darsia.ColorSpectrum(
                        base_color=base_color,
                        histogram=np.zeros(self._shape, dtype=float),
                        spectrum=np.zeros(self._shape, dtype=bool),
                        color_range=self.color_range,
                    ),
                )
                for label, base_color in base_colors.colors.items()
            )
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
                    bins=self._shape,
                    range=self.color_range.range,
                )

                # Ignore colors
                if ignore is not None:
                    ignore_spectrum = (
                        ignore[label].spectrum
                        if isinstance(ignore, darsia.LabelColorSpectrumMap)
                        else ignore.spectrum
                    )
                    assert ignore_spectrum.shape == self._shape
                    assert ignore_spectrum.dtype == bool
                    single_hist[ignore_spectrum] = 0.0

                color_spectrum_map[label].histogram += single_hist

        # Normalize the histogram, and identify the significant colors
        for label in color_spectrum_map.keys():
            color_spectrum_map[label].histogram = color_spectrum_map[
                label
            ].histogram / np.sum(color_spectrum_map[label].histogram)
            color_spectrum_map[label].spectrum = (
                color_spectrum_map[label].histogram > threshold_significant
            )

        # Plot the histogram as a 3D voxel plot
        if verbose:
            # Prepare reusable data
            _, edges = np.histogramdd(
                np.zeros((1, 3)),
                bins=self._shape,
                range=self.color_range.range,
            )
            # Define the respective meshgrid for the histogram
            x_edges = edges[0][:-1] + 0.5 * (edges[0][1] - edges[0][0])
            y_edges = edges[1][:-1] + 0.5 * (edges[1][1] - edges[1][0])
            z_edges = edges[2][:-1] + 0.5 * (edges[2][1] - edges[2][0])
            x_mesh, y_mesh, z_mesh = np.meshgrid(
                x_edges, y_edges, z_edges, indexing="ij"
            )

            for label in np.unique(self.labels.img):
                if not np.any(color_spectrum_map[label].spectrum):
                    logger.info(f"Skip plotting color spectrum for label {label}.")
                    continue
                ax = plt.figure().add_subplot(projection="3d")
                ax.set_title(f"Color Spectrum for Label {label}")

                # Add colors to each mesh point
                c_mesh = np.clip(
                    color_spectrum_map[label].base_color
                    + np.vstack(
                        (x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten())
                    ).T,
                    0.0,
                    1.0,
                ).reshape(self._shape + (3,))

                # Plot the significant boxes
                ax.voxels(color_spectrum_map[label].spectrum, facecolors=c_mesh)
                # Mark the ignored colors
                if ignore is not None:
                    ax.voxels(
                        ignore[label].spectrum
                        if isinstance(ignore, darsia.LabelColorSpectrumMap)
                        else ignore.spectrum,
                        facecolors="red",
                        alpha=0.1,
                    )
                # Highlight the (relative) origin
                origin = np.zeros_like(color_spectrum_map[label].spectrum, dtype=bool)
                origin_index = self.discrete_color_range.color_to_index(
                    np.zeros(3)
                    if self.color_mode == darsia.ColorMode.RELATIVE
                    else color_spectrum_map[label].base_color
                )
                origin[tuple(origin_index)] = True
                ax.voxels(origin, facecolors="red")
                plt.show()

        logger.info("Done. Color spectrum analysis.")
        return color_spectrum_map

    @darsia.timing_decorator
    def expand_color_spectrum(
        self,
        color_spectrum: darsia.ColorSpectrum | darsia.LabelColorSpectrumMap,
        min_points: int = 6,
        verbose: bool = False,
    ) -> darsia.LabelColorSpectrumMap:
        """Expand the color spectrum through linear regression.

        Args:
            color_spectrum_map (LabelColorSpectrumMap): The color spectrum to expand.
            verbose (bool): Whether to print additional information.
            min_points (int): Minimum number of significant points to perform regression.
            # min_weight (float): Minimum weight for the regression.

        Returns:
            LabelColorSpectrumMap: The expanded color spectrum.

        """
        if isinstance(color_spectrum, darsia.ColorSpectrum):
            return self._expand_color_spectrum(
                color_spectrum=color_spectrum,
                min_points=min_points,
                title="Expanded Color Spectrum",
                verbose=verbose,
            )
        elif isinstance(color_spectrum, darsia.LabelColorSpectrumMap):
            expanded_label_color_spectrum_map = copy.deepcopy(color_spectrum)
            for label, color_spectrum in expanded_label_color_spectrum_map.items():
                expanded_label_color_spectrum_map[label] = self._expand_color_spectrum(
                    color_spectrum=color_spectrum,
                    min_points=min_points,
                    title=f"Expanded Color Spectrum for Label {label}",
                    verbose=verbose,
                )
            return expanded_label_color_spectrum_map

    @darsia.timing_decorator
    def _expand_color_spectrum(
        self,
        color_spectrum: darsia.ColorSpectrum,
        min_points: int = 6,
        title="Expanded Color Spectrum",
        verbose: bool = False,
    ) -> darsia.ColorSpectrum:
        """Expand the color spectrum through linear regression.

        Args:
            color_spectrum (LabelColorSpectrum): The color spectrum to expand.
            min_points (int): Minimum number of significant points to perform regression.
            title (str): Title for the verbose plot.
            verbose (bool): Whether to print additional information.

        Returns:
            ColorSpectrum: The expanded color spectrum.

        """
        # Sanity check - compatibility of spectrum with discrete color range
        assert color_spectrum.spectrum.shape == self.discrete_color_range.shape
        assert color_spectrum.color_range == self.discrete_color_range

        # Prepare the result
        expanded_color_spectrum = copy.deepcopy(color_spectrum)

        # Step 0: Add 8 neighbors to significant voxels to densen the spectrum
        spectrum = expanded_color_spectrum.spectrum
        expanded_color_spectrum.spectrum[1:-1, 1:-1, 1:-1] |= (
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

        # Step 1: If too few data points available, return the simply expanded spectrum
        if color_spectrum.relative_colors.shape[0] <= min_points:
            return expanded_color_spectrum

        # Step 2: PCA - base on original input for lower cost and better accuracy
        pca = PCA(n_components=1)
        pca.fit(color_spectrum.relative_colors)
        coef = pca.components_.flatten()

        # Step 3: Expand along the line segment - first continuously.
        # Make sure to use high-resolution and to cover enough length.
        expanded_relative_colors = np.empty((0, 3))
        relative_colors = expanded_color_spectrum.relative_colors
        high_resolution = 10 * self._shape[0]
        rescaled_coef = (
            np.max(expanded_color_spectrum.color_range.range)
            / np.mean(coef)
            / high_resolution
            * coef
        ).flatten()
        for i in range(-2 * high_resolution, 2 * high_resolution + 1):
            shifted_colors = relative_colors + i * rescaled_coef
            expanded_relative_colors = np.vstack(
                (expanded_relative_colors, shifted_colors)
            )

        # Convert to discrete histogram
        expanded_histogram, _ = np.histogramdd(
            expanded_relative_colors,
            bins=self._shape,
            range=self.color_range.range,
        )
        expanded_histogram = expanded_histogram / np.sum(expanded_histogram)
        expanded_spectrum = expanded_histogram > 0.0

        expanded_color_spectrum.histogram = expanded_histogram
        expanded_color_spectrum.spectrum = expanded_spectrum

        # Add same verbose plotting as for get_color_spectrum
        if verbose:
            # Prepare reusable data
            _, edges = np.histogramdd(
                np.zeros((1, 3)),
                bins=self._shape,
                range=self.color_range.range,
            )

            # Define the respective meshgrid for the histogram
            x_edges = edges[0][:-1] + 0.5 * (edges[0][1] - edges[0][0])
            y_edges = edges[1][:-1] + 0.5 * (edges[1][1] - edges[1][0])
            z_edges = edges[2][:-1] + 0.5 * (edges[2][1] - edges[2][0])
            x_mesh, y_mesh, z_mesh = np.meshgrid(
                x_edges, y_edges, z_edges, indexing="ij"
            )

            ax = plt.figure().add_subplot(projection="3d")
            ax.set_title(title)

            # Add colors to each mesh point
            c_mesh = np.clip(
                color_spectrum.base_color
                + np.vstack((x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten())).T,
                0.0,
                1.0,
            ).reshape(self._shape + (3,))

            # Plot the significant boxes
            ax.voxels(spectrum, facecolors=c_mesh, alpha=0.8)
            ax.voxels(expanded_spectrum, facecolors=c_mesh, alpha=0.1)
            # Highlight the base color / relative origin
            origin = np.zeros_like(expanded_spectrum, dtype=bool)
            origin_index = self.discrete_color_range.color_to_index(
                np.zeros(3)
                if self.color_mode == darsia.ColorMode.RELATIVE
                else color_spectrum.base_color
            )
            origin[tuple(origin_index)] = True
            ax.voxels(origin, facecolors="red")
            plt.show()

        logger.info("Done. Expanded color spectrum analysis.")

        return expanded_color_spectrum

    @darsia.timing_decorator
    def _find_color_path(
        self,
        spectrum: darsia.ColorSpectrum,
        ignore: darsia.ColorSpectrum | None = None,
        num_segments: int = 1,
        name: str = "Color Path",
        directory: Path | None = None,
        verbose: bool = False,
    ) -> darsia.ColorPath:
        """Find a relative color path through the significant boxes.

        Args:
            spectrum (ColorSpectrum): The color spectrum to analyze.
            ignore (ColorSpectrum | None): The color spectrum to ignore.
            num_segments (int): The number of segments for the color path.
            name (str): Name of the color path.
            directory (Path | None): Directory to save verbose plots.
            verbose (bool): Whether to print additional information.

        Returns:
            darsia.ColorPath: The relative color path through the significant boxes.

        """
        assert spectrum.color_range.color_mode == darsia.ColorMode.RELATIVE, (
            "Color path regression only implemented for RELATIVE color mode."
        )
        # Step 0: Prepare
        num_dofs = num_segments + 1

        # Step 1: Extract spectrum and convert to relative colors in the color range
        relative_colors = spectrum.relative_colors
        colors = spectrum.colors
        num_points = relative_colors.shape[0]

        # Check for empty color path - need at least two points to define a path
        if num_points <= 1:
            logger.info("""Empty color or path found. Returning default color path.""")
            return darsia.ColorPath(
                colors=None,
                base_color=spectrum.base_color,
                relative_colors=num_dofs * [np.zeros(3)],
                values=np.linspace(0.0, 1.0, num_dofs).tolist(),
                mode="rgb",
                name=name,
            )

        # Step 2: Reduce to 1D using Locally Linear Embedding
        n_neighbors = min(10, num_points - 1)
        lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=1)
        embedding = lle.fit_transform(relative_colors).flatten()

        # Step 3: Sort embedding and colors
        sorted_indices = np.argsort(embedding)
        sorted_embedding = embedding[sorted_indices]
        sorted_relative_colors = relative_colors[sorted_indices]

        # Step 3.2: Identify "left" part from origin and remove it
        origin = np.zeros(3)
        # origin_index = np.where(np.all(sorted_relative_colors == origin, axis=1))[0][0]
        origin_index = np.argmin(
            np.linalg.norm(sorted_relative_colors - origin, axis=1)
        )
        # Assume origin to be close to the extreme left, if exreme right, flip everything
        if origin_index > len(sorted_relative_colors) // 2:
            origin_index = len(sorted_relative_colors) - origin_index - 1
            sorted_embedding = np.flip(sorted_embedding, axis=0)
            sorted_relative_colors = np.flip(sorted_relative_colors, axis=0)
        sorted_embedding = sorted_embedding[origin_index:]
        sorted_relative_colors = sorted_relative_colors[origin_index:, :]

        # Add origin to the beginning
        sorted_embedding = np.hstack(
            (
                sorted_embedding[0]
                + np.sign(sorted_embedding[0] - sorted_embedding[-1]),
                sorted_embedding,
            )
        )
        sorted_relative_colors = np.vstack((origin, sorted_relative_colors))

        # Initialize segments
        segments = []  # Find the two representative key colors representing the start and end of the path

        def segment_error(segment_range):
            segment_embedding = sorted_embedding[segment_range]
            segment_relative_colors = sorted_relative_colors[segment_range]

            # Create proper arrays for LinearRegression
            X = np.array([segment_embedding[0], segment_embedding[-1]]).reshape(-1, 1)
            y = np.array([segment_relative_colors[0], segment_relative_colors[-1]])
            segment_interpolator = LinearRegression().fit(X, y)
            prediction = segment_interpolator.predict(segment_embedding.reshape(-1, 1))
            segment_error = np.sum(
                np.linalg.norm(
                    prediction - segment_relative_colors,
                    axis=1,
                    ord=1,
                )
            )
            return segment_error

        def segment_length(segment_range):
            segment_embedding = sorted_embedding[segment_range]
            length = np.linalg.norm(segment_embedding[-1] - segment_embedding[0])
            return length

        def split_segment(segment_range):
            # Find the best split point for the selected segment.
            # Balance between error reduction and segment length
            min_metric = float("inf")

            total_error = segment_error(segment_range)
            total_length = segment_length(segment_range)

            for split_point in range(1, len(segment_range) - 1):
                left_segment_range = segment_range[:split_point]
                right_segment_range = segment_range[split_point:]
                left_error = segment_error(left_segment_range)
                right_error = segment_error(right_segment_range)
                left_length = segment_length(left_segment_range)
                right_length = segment_length(right_segment_range)
                split_error = max(left_error, right_error)
                split_length = left_length + right_length

                # Update the best split if this one is better
                split_metric = split_error / total_error + split_length / total_length
                if split_metric < min_metric:
                    min_metric = split_metric

                    left_segment = {
                        "range": left_segment_range,
                        "error": left_error,
                        "length": left_length,
                    }
                    right_segment = {
                        "range": right_segment_range,
                        "error": right_error,
                        "length": right_length,
                    }
            return left_segment, right_segment

        segment_range = range(0, len(sorted_embedding))
        _segment_error = segment_error(segment_range)
        _segment_length = segment_length(segment_range)
        segment = {
            "range": segment_range,
            "error": _segment_error,
            "length": _segment_length,
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

            # Split the segment
            left_segment, right_segment = split_segment(segment_to_split["range"])

            # Replace the selected segment with the two new segments
            global_segment_to_split_index = segments.index(segment_to_split)
            segments[global_segment_to_split_index] = left_segment
            segments.insert(global_segment_to_split_index + 1, right_segment)

        # Improve segments by re-evaluating their error
        old_distances = []
        for _ in range(1000):
            # Cache previous segments for convergence check
            previous_segments = copy.deepcopy(segments)

            # Segment by segment, try to improve by combining with neighbor segments
            for i, segment in enumerate(segments):
                if i == len(segments) - 1:
                    continue  # Last segment, nothing to do
                # Combine segment with neighbor segments
                combined_range = range(
                    segment["range"].start, segments[i + 1]["range"].stop
                )
                if len(combined_range) < 3:
                    # Not enough points to split
                    continue
                # Find the best split point again
                left_segment, right_segment = split_segment(combined_range)
                # Replace segments
                segments[i] = left_segment
                segments[i + 1] = right_segment

            # Check for convergence - if no change in segments, stop
            if all(
                segments[i]["range"] == previous_segments[i]["range"]
                for i in range(len(segments))
            ):
                break
            else:
                distance = sum(
                    abs(
                        segments[i]["range"].start - previous_segments[i]["range"].start
                    )
                    + abs(
                        segments[i]["range"].stop - previous_segments[i]["range"].stop
                    )
                    for i in range(len(segments))
                )
                old_distances.append(distance)
                if len(old_distances) > 5 and len(np.unique(old_distances[-5:])) == 1:
                    # No change in the last 5 iterations - oscillation detected
                    break
                print(f"Segment smoothing distance: {distance}")

        # Extract the key relative colors from the segments
        key_relative_colors: list[np.ndarray] = [
            sorted_relative_colors[segment["range"].start] for segment in segments
        ] + [sorted_relative_colors[segments[-1]["range"].stop - 1]]

        # Determine the maximum error for information
        # TODO error quantification
        # max_error = max(seg["error"] for seg in segments)

        if key_relative_colors == []:
            logger.info("No key colors found. Returning default color path.")
            return darsia.ColorPath(
                colors=None,
                base_color=spectrum.base_color,
                relative_colors=num_dofs * [np.zeros(3)],
                values=np.linspace(0.0, 1.0, num_dofs).tolist(),
                mode="rgb",
                name=name,
            )

        # Step 5: Define the corresponding absolute colors
        key_colors = [spectrum.base_color + c for c in key_relative_colors]

        # Step 6: Verbose output

        # Print key colors
        if verbose:
            for i, (color, rel) in enumerate(zip(key_colors, key_relative_colors)):
                print(f"Key color {i + 1}: RGB = {color}, relative: {rel}")

        # Step 6: Visualize
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111, projection="3d")

        # Plot all significant colors
        ax.scatter(
            relative_colors[:, 0],
            relative_colors[:, 1],
            relative_colors[:, 2],
            c=np.clip(colors, 0, 1),
            s=10,
            alpha=0.5,
        )

        # Plot key colors and connecting lines
        if np.array(key_relative_colors).shape[0] == num_dofs:
            ax.plot(*np.array(key_relative_colors).T, c="black", linewidth=2)
            ax.scatter(
                *np.array(key_relative_colors).T,
                c=np.clip(key_colors, 0, 1),
                s=100,
            )

        # Plot connecting lines for all points along the sorted embedding
        ax.plot(*np.array(relative_colors).T, c="gray", linewidth=1, alpha=0.5)

        # Band of ignored colors (in red)
        if ignore is not None:
            base_relative_colors = ignore.relative_colors
            ax.scatter(
                base_relative_colors[:, 0],
                base_relative_colors[:, 1],
                base_relative_colors[:, 2],
                c="r",
                s=10,
                alpha=0.1,
            )

        # Some plot settings
        ax.set_title(name)
        ax.set_xlabel("R")
        ax.set_ylabel("G")
        ax.set_zlabel("B")

        plt.tight_layout()
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
            plot_save_path = directory / f"{name}.png"
            plt.savefig(plot_save_path, dpi=300)
        if verbose:
            plt.show()
        plt.close()

        # Step 7: Construct relative color path
        relative_color_path = darsia.ColorPath(
            colors=None,
            relative_colors=key_relative_colors,
            base_color=spectrum.base_color,
            values=None,
            mode="rgb",
            name=name,
        )
        return relative_color_path

    @darsia.timing_decorator
    def find_color_path(
        self,
        color_spectrum: darsia.LabelColorSpectrumMap,
        ignore: darsia.LabelColorSpectrumMap | None = None,
        num_segments: int = 1,
        directory: Path | None = None,
        verbose: bool = False,
    ) -> darsia.LabelColorPathMap:
        """Find relative color paths for each label in the spectrum map.

        Args:
            label_color_spectrum_map (LabelColorSpectrumMap): The color spectrum map to analyze.
            ignore (LabelColorSpectrumMap | None): The color spectrum map to ignore.
            num_segments (int): The number of segments for the color path.
            verbose (bool): Whether to print additional information.

        Returns:
            LabelColorPathMap: The relative color path map through the significant boxes.

        """
        label_color_path_map = darsia.LabelColorPathMap(
            dict(
                (
                    label,
                    self._find_color_path(
                        spectrum=color_spectrum[label],
                        ignore=ignore[label] if ignore is not None else None,
                        num_segments=num_segments,
                        name=f"Color Path for Label {label}",
                        directory=directory,
                        verbose=verbose,
                    ),
                )
                for label in color_spectrum.keys()
            )
        )
        return label_color_path_map
