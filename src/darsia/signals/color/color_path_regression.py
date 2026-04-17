"""Provide tools for defining color paths."""

import copy
import logging
from pathlib import Path
from typing import Literal
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, TextBox
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.manifold import LocallyLinearEmbedding

import darsia

from .utils import get_mean_color

logger = logging.getLogger(__name__)

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
    def _shape(self) -> tuple[int, ...]:
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
            base_colors[label] = get_mean_color(image, mask=(self.mask.img & mask.img))

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

                # Use histogram values for alpha transparency
                histogram = color_spectrum_map[label].histogram
                # Normalize histogram to [0, 1] range for alpha values
                alpha_mesh = np.zeros(self._shape, dtype=float)
                hist_max = np.max(histogram[color_spectrum_map[label].spectrum])
                if hist_max > 0:
                    alpha_mesh = histogram / hist_max
                else:
                    alpha_mesh[color_spectrum_map[label].spectrum] = 1.0

                # Block alpha into 10 discrete levels.
                num_alpha_levels = 10
                alpha_min = np.min(alpha_mesh)
                alpha_max = np.max(alpha_mesh)
                alpha_levels = np.linspace(alpha_min, alpha_max, num_alpha_levels)
                alpha_blocked = (
                    np.digitize(alpha_mesh, bins=alpha_levels) / num_alpha_levels
                )
                alpha_blocked = np.clip(alpha_blocked, alpha_min, alpha_max)

                # Plot voxels for each alpha level separately
                for alpha_level in alpha_levels:
                    # Create mask for voxels at this alpha level
                    alpha_tolerance = (alpha_max - alpha_min) / (2 * num_alpha_levels)
                    mask_at_level = color_spectrum_map[label].spectrum & (
                        np.abs(alpha_blocked - alpha_level) < alpha_tolerance
                    )
                    # Rescale such that levels span 0..1
                    alpha_level_rescaled = (alpha_level - alpha_min) / (
                        alpha_max - alpha_min
                    )

                    if np.any(mask_at_level):
                        ax.voxels(
                            mask_at_level,
                            facecolors=c_mesh,
                            alpha=alpha_level_rescaled,
                        )

                # Mark the ignored colors
                if ignore is not None:
                    ax.voxels(
                        (
                            ignore[label].spectrum
                            if isinstance(ignore, darsia.LabelColorSpectrumMap)
                            else ignore.spectrum
                        ),
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
        label: int | None = None,
        ignore: darsia.ColorSpectrum | None = None,
        num_segments: int = 1,
        name: str = "Color Path",
        directory: Path | None = None,
        weighting: Literal["threshold", "wls", "wls_sqrt", "wls_log"] = "threshold",
        mode: Literal["auto", "manual"] = "auto",
        preview_image: darsia.Image | None = None,
        preview_images: list[darsia.Image] | None = None,
        preview_baseline: darsia.Image | None = None,
        verbose: bool = False,
    ) -> darsia.ColorPath:
        """Find a relative color path through the significant boxes.

        Args:
            spectrum (ColorSpectrum): The color spectrum to analyze.
            ignore (ColorSpectrum | None): The color spectrum to ignore.
            num_segments (int): The number of segments for the color path.
            name (str): Name of the color path.
            directory (Path | None): Directory to save verbose plots.
            weighting (str): How to use histogram counts when fitting the path.
                ``"threshold"`` (default) uses binary 0/1 weights controlled by
                the spectrum threshold – counts are not used beyond that.
                ``"wls"`` weights each active bin by its normalised probability.
                ``"wls_sqrt"`` weights by the square-root of the probability.
                ``"wls_log"`` weights by ``log(1 + count)`` where count is
                derived from the normalised probability scaled by the total
                number of active bins.
            mode (str): Color-path selection mode.
                ``"auto"`` returns the automated result.
                ``"manual"`` starts from the automated result and allows
                interactive key-color editing before finalizing.
            verbose (bool): Whether to print additional information.

        Returns:
            darsia.ColorPath: The relative color path through the significant boxes.

        """
        assert (
            spectrum.color_range.color_mode == darsia.ColorMode.RELATIVE
        ), "Color path regression only implemented for RELATIVE color mode."
        # Step 0: Prepare
        num_dofs = num_segments + 1

        # Step 1: Extract spectrum and convert to relative colors in the color range
        relative_colors = spectrum.relative_colors
        colors = spectrum.colors
        num_points = relative_colors.shape[0]

        # Build per-bin weights from the histogram.
        # Thresholding is always applied first via spectrum.spectrum (binary 0/1 mask).
        # For "threshold" mode the weights are those binary values (1 for active bins).
        # For WLS modes the full histogram probabilities of the active bins are used.
        active_probs = spectrum.histogram[np.where(spectrum.spectrum)]
        if weighting == "threshold":
            # Binary weights from thresholding: 1 for every active bin, 0 for inactive
            # (inactive bins are not present in relative_colors at all).
            point_weights = np.ones(num_points, dtype=float)
        elif weighting == "wls":
            point_weights = active_probs
        elif weighting == "wls_sqrt":
            point_weights = np.sqrt(active_probs)
        elif weighting == "wls_log":
            # Approximate counts as prob * num_active_bins (avoids storing N separately)
            approx_counts = active_probs * num_points
            point_weights = np.log1p(approx_counts)
        else:
            raise ValueError(
                f"Unknown histogram_weighting value '{weighting}'. "
                "Allowed: 'threshold', 'wls', 'wls_sqrt', 'wls_log'."
            )
        w_sum = point_weights.sum()
        if w_sum > 0.0:
            point_weights = point_weights / w_sum
        else:
            point_weights = np.ones(num_points, dtype=float) / num_points

        # Check for empty color path - need at least two points to define a path
        if num_points <= 1:
            logger.info("""Empty color or path found. Returning default color path.""")
            return darsia.ColorPath(
                colors=None,
                base_color=spectrum.base_color,
                relative_colors=num_dofs * [np.zeros(3)],
                mode="rgb",
                name=name,
            )

        # Step 2: Reduce to 1D using Locally Linear Embedding
        n_neighbors = min(10, num_points - 1)
        lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=1)
        embedding = lle.fit_transform(relative_colors).flatten()

        if verbose:
            # Visualization 1: Original 3D colors vs 1D embedding
            fig = plt.figure(figsize=(14, 5))

            # 3D scatter of original colors
            ax1 = fig.add_subplot(131, projection="3d")
            scatter1 = ax1.scatter(
                relative_colors[:, 0],
                relative_colors[:, 1],
                relative_colors[:, 2],
                c=embedding,  # Color by embedding value
                cmap="viridis",
                s=50,
                alpha=0.7,
            )
            ax1.set_title("Original 3D Colors\n(colored by 1D embedding)")
            ax1.set_xlabel("R")
            ax1.set_ylabel("G")
            ax1.set_zlabel("B")
            plt.colorbar(scatter1, ax=ax1, label="1D Embedding")

            # 1D embedding vs point index
            ax2 = fig.add_subplot(132)
            ax2.scatter(range(num_points), embedding, c=embedding, cmap="viridis", s=50)
            ax2.set_xlabel("Point Index")
            ax2.set_ylabel("1D Embedding Value")
            ax2.set_title("1D Embedding Distribution")
            ax2.grid(True, alpha=0.3)

            # Embedding histogram
            ax3 = fig.add_subplot(133)
            ax3.hist(embedding, bins=20, edgecolor="black", alpha=0.7)
            ax3.set_xlabel("Embedding Value")
            ax3.set_ylabel("Frequency")
            ax3.set_title("1D Embedding Histogram")
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            if directory:
                directory.mkdir(parents=True, exist_ok=True)
                plt.savefig(directory / f"{name}_01_embedding.png", dpi=150)
            plt.show()
            plt.close()

        # Step 3: Sort embedding and colors
        sorted_indices = np.argsort(embedding)
        sorted_embedding = embedding[sorted_indices]
        sorted_relative_colors = relative_colors[sorted_indices]
        sorted_weights = point_weights[sorted_indices]

        logger.info(
            f"Embedding range: [{sorted_embedding[0]:.4f}, {sorted_embedding[-1]:.4f}]"
        )

        # Step 3.2: Identify "left" part from origin and remove it
        origin = np.zeros(3)
        # origin_index = np.where(np.all(sorted_relative_colors == origin, axis=1))[0][0]
        origin_index = np.argmin(
            np.linalg.norm(sorted_relative_colors - origin, axis=1)
        )
        origin_distance = np.linalg.norm(sorted_relative_colors[origin_index] - origin)

        logger.info(
            f"Identified origin index: {origin_index}, "
            f"color: {sorted_relative_colors[origin_index]}, "
            f"distance: {origin_distance:.4f}"
        )

        if verbose:
            # Visualization 2: Origin detection
            fig = plt.figure(figsize=(14, 5))

            # Distance to origin
            distances_to_origin = np.linalg.norm(
                sorted_relative_colors - origin, axis=1
            )
            ax1 = fig.add_subplot(121)
            ax1.plot(
                range(len(sorted_relative_colors)), distances_to_origin, "b-", alpha=0.5
            )
            ax1.scatter(
                origin_index,
                distances_to_origin[origin_index],
                color="red",
                s=100,
                label="Detected Origin",
            )
            ax1.axvline(
                len(sorted_relative_colors) // 2,
                color="gray",
                linestyle="--",
                alpha=0.5,
                label="Midpoint",
            )
            ax1.set_xlabel("Sorted Index")
            ax1.set_ylabel("Distance to Origin")
            ax1.set_title("Distance to Origin (0,0,0)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 3D visualization with origin marked
            ax2 = fig.add_subplot(122, projection="3d")
            ax2.scatter(
                sorted_relative_colors[:, 0],
                sorted_relative_colors[:, 1],
                sorted_relative_colors[:, 2],
                c=range(len(sorted_relative_colors)),
                cmap="viridis",
                s=30,
                alpha=0.7,
            )
            ax2.scatter(
                sorted_relative_colors[origin_index, 0],
                sorted_relative_colors[origin_index, 1],
                sorted_relative_colors[origin_index, 2],
                color="red",
                s=200,
                marker="*",
                label="Detected Origin",
            )
            ax2.scatter(0, 0, 0, color="green", s=100, marker="x", label="True Origin")
            ax2.set_xlabel("R")
            ax2.set_ylabel("G")
            ax2.set_zlabel("B")
            ax2.set_title("Origin Detection in 3D")
            ax2.legend()

            plt.tight_layout()
            if directory:
                plt.savefig(directory / f"{name}_02_origin_detection.png", dpi=150)
            plt.show()
            plt.close()

        # Assume origin to be close to the extreme left, if extreme right, flip everything
        if origin_index > len(sorted_relative_colors) // 2:
            logger.info(
                f"Origin detected at right side (index {origin_index}). "
                f"Flipping direction."
            )
            origin_index = len(sorted_relative_colors) - origin_index - 1
            sorted_embedding = np.flip(sorted_embedding, axis=0)
            sorted_relative_colors = np.flip(sorted_relative_colors, axis=0)
            sorted_weights = np.flip(sorted_weights, axis=0)
            logger.info(f"After flip, origin index: {origin_index}")

        sorted_embedding = sorted_embedding[origin_index:]
        sorted_relative_colors = sorted_relative_colors[origin_index:, :]
        sorted_weights = sorted_weights[origin_index:]

        # Add origin to the beginning with weight of 0 to anchor the path at the
        # relative origin without biasing the weighted fit toward or away from it.
        logger.info(
            f"""After trimming: {len(sorted_embedding)} points, """
            f"""embedding range: [{sorted_embedding[0]:.4f}, {sorted_embedding[-1]:.4f}]"""
        )

        if verbose:
            # Visualization 3: Sorted and trimmed colors
            fig = plt.figure(figsize=(14, 5))

            # Sorted embedding as path
            ax1 = fig.add_subplot(121)
            ax1.plot(
                range(len(sorted_embedding)),
                sorted_embedding,
                "b-",
                marker="o",
                markersize=3,
            )
            ax1.scatter(
                0,
                sorted_embedding[0],
                color="red",
                s=100,
                zorder=5,
                label="Added Origin",
            )
            ax1.set_xlabel("Index")
            ax1.set_ylabel("Embedding Value")
            ax1.set_title("Sorted & Trimmed Embedding")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # 3D path
            ax2 = fig.add_subplot(122, projection="3d")
            ax2.plot(
                sorted_relative_colors[:, 0],
                sorted_relative_colors[:, 1],
                sorted_relative_colors[:, 2],
                "b-",
                alpha=0.5,
                linewidth=2,
            )
            ax2.scatter(
                sorted_relative_colors[:, 0],
                sorted_relative_colors[:, 1],
                sorted_relative_colors[:, 2],
                c=range(len(sorted_relative_colors)),
                cmap="viridis",
                s=50,
            )
            ax2.scatter(0, 0, 0, color="red", s=200, marker="*", label="Origin")
            ax2.set_xlabel("R")
            ax2.set_ylabel("G")
            ax2.set_zlabel("B")
            ax2.set_title("Sorted Path in 3D Color Space")
            ax2.legend()

            plt.tight_layout()
            if directory:
                plt.savefig(directory / f"{name}_03_sorted_path.png", dpi=150)
            plt.show()
            plt.close()

        # Add origin to the beginning
        sorted_embedding = np.hstack(
            (
                sorted_embedding[0]
                + np.sign(sorted_embedding[0] - sorted_embedding[-1]),
                sorted_embedding,
            )
        )
        sorted_relative_colors = np.vstack((origin, sorted_relative_colors))
        sorted_weights = np.hstack((0.0, sorted_weights))
        # Re-normalise after prepending the origin weight
        w_total = sorted_weights.sum()
        if w_total > 0.0:
            sorted_weights = sorted_weights / w_total

        # Initialize segments.
        segments = []

        def segment_error(segment_range):
            """Calculate the quantile-based fitting error for a color segment.

            This function measures how well a linear interpolation fits the color path
            within a given segment. It uses a quantile-based approach to be robust to
            outliers rather than using the raw sum of errors.

            Method:
                1. Extract the embedding values and relative colors for the segment
                2. Create a linear regression model that maps the 1D embedding to the
                3D color space using only the segment's start and end points
                3. Predict the colors for all embedding values within the segment
                4. Calculate the L1 (Manhattan) distance between predicted and actual
                colors at each point
                5. Use the 80th percentile of errors (rather than mean/sum) to exclude
                outliers and get a robust error measure

            Interpretation:
                - Small error: The segment's colors lie close to a straight line in 3D
                color space, indicating the segment is well-approximated by linear
                interpolation between its endpoints
                - Large error: The segment's colors deviate significantly from a linear
                path, suggesting curved behavior that may benefit from splitting into
                multiple segments

            Args:
                segment_range: A range object specifying indices of points in the segment

            Returns:
                float: The 80th percentile of L1 errors between linear regression
                    predictions and actual colors. Values are typically in [0, 3]
                    for normalized RGB color space.

            Note:
                The use of quantile (0.8) instead of mean makes this robust to outlier
                colors that deviate from the main path. This is important for color paths
                that may have occasional noisy measurements or artifacts.

            """
            segment_embedding = sorted_embedding[segment_range]
            segment_relative_colors = sorted_relative_colors[segment_range]

            # Create proper arrays for LinearRegression
            X = np.array([segment_embedding[0], segment_embedding[-1]]).reshape(-1, 1)
            y = np.array([segment_relative_colors[0], segment_relative_colors[-1]])
            segment_interpolator = LinearRegression().fit(X, y)
            prediction = segment_interpolator.predict(segment_embedding.reshape(-1, 1))
            segment_errors = np.linalg.norm(
                prediction - segment_relative_colors,
                axis=1,
                ord=1,
            )
            if weighting == "threshold":
                # Original behaviour: use an 80th-percentile quantile error.
                quantile = 0.8
                return np.quantile(segment_errors, quantile)
            else:
                # Weighted average error using per-bin histogram weights.
                seg_w = sorted_weights[segment_range]
                seg_w_sum = seg_w.sum()
                if seg_w_sum > 0.0:
                    return float(np.dot(seg_w, segment_errors) / seg_w_sum)
                # Fall back to unweighted mean if all weights happen to be zero.
                return float(np.mean(segment_errors))

        def segment_length(segment_range):
            segment_embedding = sorted_embedding[segment_range]
            length = np.linalg.norm(segment_embedding[-1] - segment_embedding[0])
            return length

        def split_segment(segment_range):
            """Split a segment at the point where left and right errors are balanced.

            This function finds the optimal split point by:
            1. Computing error for all possible left/right partitions
            2. Smoothing both error curves to reduce noise
            3. Finding where the smoothed curves intersect
            4. Selecting the intersection closest to the segment midpoint for stability

            The smoothing uses a Savitzky-Golay filter which preserves sharp features
            while removing high-frequency noise from the error curves.

            Args:
                segment_range: A range object specifying indices in the segment

            Returns:
                Tuple of two dicts:
                    - left_segment: {"range": range, "error": float, "length": float}
                    - right_segment: {"range": range, "error": float, "length": float}

            """
            from scipy.signal import savgol_filter

            # total_error = segment_error(segment_range)
            # total_length = segment_length(segment_range)

            pts = []
            left_errors = []
            right_errors = []

            # Compute errors for all possible split points
            for split_point in range(1, len(segment_range) - 1):
                left_segment_range = segment_range[:split_point]
                right_segment_range = segment_range[split_point:]

                left_error = segment_error(left_segment_range)
                right_error = segment_error(right_segment_range)

                pts.append(split_point)
                left_errors.append(left_error)
                right_errors.append(right_error)

            pts = np.array(pts)
            left_errors = np.array(left_errors)
            right_errors = np.array(right_errors)

            # Smooth the error curves to reduce noise
            # Use Savitzky-Golay filter if we have enough points
            if len(pts) >= 5:
                # Window length should be odd and less than data length
                window_length = min(5, len(pts) if len(pts) % 2 == 1 else len(pts) - 1)
                if window_length >= 5:
                    try:
                        left_errors_smooth = savgol_filter(
                            left_errors, window_length=window_length, polyorder=2
                        )
                        right_errors_smooth = savgol_filter(
                            right_errors, window_length=window_length, polyorder=2
                        )
                    except ValueError:
                        # Fallback to simple moving average if Savitzky-Golay fails
                        left_errors_smooth = left_errors
                        right_errors_smooth = right_errors
                else:
                    # Simple moving average for small datasets
                    window = 3
                    left_errors_smooth = np.convolve(
                        left_errors, np.ones(window) / window, mode="same"
                    )
                    right_errors_smooth = np.convolve(
                        right_errors, np.ones(window) / window, mode="same"
                    )
            else:
                # Too few points, use raw errors
                left_errors_smooth = left_errors
                right_errors_smooth = right_errors

            # Find all crossover points where left_error ~= right_error
            error_diff = left_errors_smooth - right_errors_smooth

            # Find sign changes (crossover points)
            sign_changes = np.where(np.diff(np.sign(error_diff)))[0]

            if len(sign_changes) == 0:
                # No crossover found - use the point with minimum absolute difference
                optimal_idx = np.argmin(np.abs(error_diff))
                optimal_split_point = pts[optimal_idx]
                logger.warning(
                    """No error crossover found. Using minimum difference at """
                    f"""index {optimal_split_point}"""
                )
            else:
                # Multiple crossovers possible - choose the one closest to segment center
                segment_center = len(segment_range) / 2
                crossover_indices = sign_changes
                crossover_pts = pts[crossover_indices]

                # Find crossover closest to center
                center_distances = np.abs(crossover_pts - segment_center)
                best_crossover_idx = np.argmin(center_distances)
                optimal_split_point = crossover_pts[best_crossover_idx]

                if verbose:
                    logger.info(
                        f"""Found {len(sign_changes)} crossover point(s). """
                        f"""Selected split at {optimal_split_point} (center: """
                        f"""{segment_center:.1f})"""
                    )

            # Create the left and right segments
            left_segment_range = segment_range[:optimal_split_point]
            right_segment_range = segment_range[optimal_split_point:]

            left_segment = {
                "range": left_segment_range,
                "error": segment_error(left_segment_range),
                "length": segment_length(left_segment_range),
            }
            right_segment = {
                "range": right_segment_range,
                "error": segment_error(right_segment_range),
                "length": segment_length(right_segment_range),
            }

            # Visualization
            if False:
                fig = plt.figure(figsize=(15, 4))

                # Plot 1: Error curves with smoothing
                ax1 = fig.add_subplot(131)
                ax1.plot(pts, left_errors, "b-", alpha=0.3, label="Left Error (raw)")
                ax1.plot(pts, right_errors, "r-", alpha=0.3, label="Right Error (raw)")
                ax1.plot(
                    pts,
                    left_errors_smooth,
                    "b-",
                    linewidth=2,
                    label="Left Error (smooth)",
                )
                ax1.plot(
                    pts,
                    right_errors_smooth,
                    "r-",
                    linewidth=2,
                    label="Right Error (smooth)",
                )
                ax1.axvline(
                    optimal_split_point,
                    color="green",
                    linestyle="--",
                    linewidth=2,
                    label=f"Split @ {optimal_split_point}",
                )
                if len(sign_changes) > 0:
                    # Not available otherwise.
                    ax1.axvline(
                        segment_center,
                        color="gray",
                        linestyle=":",
                        alpha=0.5,
                        label="Center",
                    )
                ax1.set_xlabel("Split Point")
                ax1.set_ylabel("Error")
                ax1.set_title("Error Curves (Raw vs Smoothed)")
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)

                # Plot 2: Error difference
                ax2 = fig.add_subplot(132)
                ax2.plot(pts, error_diff, "k-", linewidth=2, label="Left - Right Error")
                ax2.axhline(0, color="gray", linestyle="-", alpha=0.5)
                ax2.axvline(
                    optimal_split_point,
                    color="green",
                    linestyle="--",
                    linewidth=2,
                    label=f"Split @ {optimal_split_point}",
                )
                if len(sign_changes) > 0:
                    ax2.scatter(
                        pts[sign_changes],
                        error_diff[sign_changes],
                        color="orange",
                        s=100,
                        zorder=5,
                        label="Crossovers",
                    )
                ax2.set_xlabel("Split Point")
                ax2.set_ylabel("Error Difference")
                ax2.set_title("Left Error - Right Error")
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3)

                # Plot 3: Segment info
                ax3 = fig.add_subplot(133)
                ax3.text(
                    0.1,
                    0.9,
                    f"Segment Range: [{segment_range.start}:{segment_range.stop}]",
                    transform=ax3.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    family="monospace",
                )
                ax3.text(
                    0.1,
                    0.8,
                    f"Left Error: {left_segment['error']:.6f}",
                    transform=ax3.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    family="monospace",
                )
                ax3.text(
                    0.1,
                    0.7,
                    f"Right Error: {right_segment['error']:.6f}",
                    transform=ax3.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    family="monospace",
                )
                ax3.text(
                    0.1,
                    0.6,
                    f"Error Balance: "
                    f"{abs(left_segment['error'] - right_segment['error']):.6f}",
                    transform=ax3.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    family="monospace",
                )
                ax3.text(
                    0.1,
                    0.5,
                    f"Left Length: {left_segment['length']:.6f}",
                    transform=ax3.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    family="monospace",
                )
                ax3.text(
                    0.1,
                    0.4,
                    f"Right Length: {right_segment['length']:.6f}",
                    transform=ax3.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    family="monospace",
                )
                ax3.text(
                    0.1,
                    0.3,
                    f"Crossovers Found: {len(sign_changes)}",
                    transform=ax3.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    family="monospace",
                )
                ax3.axis("off")

                plt.tight_layout()
                if directory:
                    plt.savefig(directory / f"{name}_split_analysis.png", dpi=150)
                plt.show()
                plt.close()

            return left_segment, right_segment

        def split_segment_old(segment_range):
            # Find the best split point for the selected segment.
            # Balance between error reduction and segment length
            min_metric = float("inf")

            total_error = segment_error(segment_range)
            total_length = segment_length(segment_range)

            left_segment = None
            right_segment = None

            pts = []
            left_errors = []
            right_errors = []
            split_errors = []
            split_lengths = []
            optimal_split_point = None
            metric_values = []

            for split_point in range(1, len(segment_range) - 1):
                left_segment_range = segment_range[:split_point]
                right_segment_range = segment_range[split_point:]
                left_error = segment_error(left_segment_range)
                right_error = segment_error(right_segment_range)
                left_length = segment_length(left_segment_range)
                right_length = segment_length(right_segment_range)
                split_error = max(left_error, right_error)
                split_length = left_length + right_length

                pts.append(split_point)
                left_errors.append(left_error)
                right_errors.append(right_error)
                split_errors.append(split_error)
                split_lengths.append(split_length)

                # Update the best split if this one is better
                split_metric = (split_error / total_error) + split_length / total_length
                metric_values.append(split_metric)
                if split_metric < min_metric:
                    optimal_split_point = split_point
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

            plt.figure("errors")
            plt.plot(pts, left_errors, label="Left Segment Error")
            plt.plot(pts, right_errors, label="Right Segment Error")
            plt.plot(pts, split_errors, label="Max Segment Error", linestyle="--")
            # Add vertical line at optimal split point
            plt.axvline(
                optimal_split_point, color="red", linestyle=":", label="Optimal Split"
            )
            plt.legend()
            plt.figure("lengths")
            plt.plot(pts, split_lengths, label="Split Segment Length")
            plt.axvline(
                optimal_split_point, color="red", linestyle=":", label="Optimal Split"
            )
            plt.legend()
            plt.figure("metrics")
            plt.plot(pts, metric_values, label="Split Metric (Error Ratio)")
            plt.axvline(
                optimal_split_point, color="red", linestyle=":", label="Optimal Split"
            )
            plt.legend()
            plt.show()
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
        for _ in range(10):
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
            # Check for oscillation
            elif False:
                ...
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
                mode="rgb",
                name=name,
            )

        if mode not in ("auto", "manual"):
            raise ValueError(
                f"Unknown color-path mode '{mode}'. Allowed: 'auto', 'manual'."
            )

        if mode == "manual":
            key_relative_colors = self._manual_postprocess_color_path(
                relative_colors=relative_colors,
                colors=colors,
                key_relative_colors=key_relative_colors,
                base_color=spectrum.base_color,
                ignore=ignore,
                num_dofs=num_dofs,
                name=name,
                label=label,
                preview_image=preview_image,
                preview_images=preview_images,
                preview_baseline=preview_baseline,
            )

        # Step 5: Define the corresponding absolute colors
        key_colors = [spectrum.base_color + c for c in key_relative_colors]

        # Step 6: Construct relative color path
        relative_color_path = darsia.ColorPath(
            colors=None,
            relative_colors=key_relative_colors,
            base_color=spectrum.base_color,
            mode="rgb",
            name=name,
        )

        # Step 7: Verbose output

        # Print key colors
        if verbose:
            for i, (color, rel) in enumerate(zip(key_colors, key_relative_colors)):
                print(f"Key color {i + 1}: RGB = {color}, relative: {rel}")

        # Visualize
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

        return relative_color_path

    @darsia.timing_decorator
    def find_color_path(
        self,
        color_spectrum: darsia.LabelColorSpectrumMap,
        ignore: darsia.LabelColorSpectrumMap | None = None,
        num_segments: int = 1,
        directory: Path | None = None,
        weighting: Literal["threshold", "wls", "wls_sqrt", "wls_log"] = "threshold",
        mode: Literal["auto", "manual"] = "auto",
        target_labels: list[int] | None = None,
        existing_map: darsia.LabelColorPathMap | None = None,
        preview_image: darsia.Image | None = None,
        preview_images: list[darsia.Image] | None = None,
        preview_baseline: darsia.Image | None = None,
        verbose: bool = False,
    ) -> darsia.LabelColorPathMap:
        """Find relative color paths for each label in the spectrum map.

        Args:
            label_color_spectrum_map (LabelColorSpectrumMap): The color spectrum map to
                analyze.
            ignore (LabelColorSpectrumMap | None): The color spectrum map to ignore.
            num_segments (int): The number of segments for the color path.
            weighting (Literal): How to use histogram counts when fitting each path.
                ``"threshold"`` (default) uses binary 0/1 weights from the spectrum
                threshold – counts are not used beyond that.
                ``"wls"``, ``"wls_sqrt"``, and ``"wls_log"`` use count-weighted
                average errors – see :meth:`_find_color_path` for details.
            mode (Literal): Color-path selection mode.
                ``"auto"`` returns the automated result.
                ``"manual"`` enables interactive key-color postprocessing.
            target_labels (list[int] | None): Labels to update. If ``None``, all labels
                from ``color_spectrum`` are calibrated.
            existing_map (LabelColorPathMap | None): Existing map used as basis when
                ``target_labels`` is provided. Non-target labels are copied unchanged.
            preview_image (darsia.Image | None): Calibration image for manual preview.
            preview_images (list[darsia.Image] | None): Calibration images for manual
                preview navigation.
            preview_baseline (darsia.Image | None): Baseline used for relative color preview.
            verbose (bool): Whether to print additional information.

        Returns:
            LabelColorPathMap: The relative color path map through the significant boxes.

        """
        if target_labels is None:
            labels_to_process = list(color_spectrum.keys())
        else:
            labels_to_process = list(target_labels)
            missing_in_spectrum = sorted(
                set(labels_to_process) - set(color_spectrum.keys())
            )
            if missing_in_spectrum:
                raise ValueError(
                    "Requested target_labels are missing in color_spectrum: "
                    f"{missing_in_spectrum}."
                )
            if existing_map is None:
                raise ValueError(
                    "existing_map is required when target_labels is provided."
                )

        updated_map: dict[int, darsia.ColorPath] = (
            {int(label): path for label, path in existing_map.items()}
            if existing_map is not None
            else {}
        )

        for label in labels_to_process:
            updated_map[label] = self._find_color_path(
                spectrum=color_spectrum[label],
                label=label,
                ignore=ignore[label] if ignore is not None else None,
                num_segments=num_segments,
                name=f"Color Path for Label {label}",
                directory=directory,
                weighting=weighting,
                mode=mode,
                preview_image=preview_image,
                preview_images=preview_images,
                preview_baseline=preview_baseline,
                verbose=verbose,
            )

        return darsia.LabelColorPathMap(updated_map)

    def _manual_postprocess_color_path(
        self,
        *,
        relative_colors: np.ndarray,
        colors: np.ndarray,
        key_relative_colors: list[np.ndarray],
        base_color: np.ndarray,
        ignore: darsia.ColorSpectrum | None,
        num_dofs: int,
        name: str,
        label: int | None,
        preview_image: darsia.Image | None,
        preview_images: list[darsia.Image] | None,
        preview_baseline: darsia.Image | None,
    ) -> list[np.ndarray]:
        """Interactively adjust key relative colors and return the updated list."""

        new_key_relative_colors = [
            np.array(c, dtype=float).copy() for c in key_relative_colors
        ]
        if len(new_key_relative_colors) == 0:
            return new_key_relative_colors

        selected_index = 0
        finalized = False

        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(
            nrows=2,
            ncols=2,
            width_ratios=[1, 1],
            height_ratios=[1, 1],
            left=0.05,
            right=0.98,
            bottom=0.24,
            top=0.95,
            wspace=0.1,
            hspace=0.12,
        )
        ax = fig.add_subplot(gs[:, 0], projection="3d")
        ax_preview_image = fig.add_subplot(gs[0, 1])
        ax_preview_interpretation = fig.add_subplot(gs[1, 1])
        ax_preview_image.axis("off")
        ax_preview_interpretation.axis("off")

        preview_candidates = list(preview_images or [])
        if len(preview_candidates) == 0 and preview_image is not None:
            preview_candidates = [preview_image]
        valid_preview_candidates = [
            img
            for img in preview_candidates
            if img.img.shape[:2] == self.labels.img.shape
        ]
        preview_shape_matches = (
            preview_baseline is not None
            and preview_baseline.img.shape[:2] == self.labels.img.shape
            and len(valid_preview_candidates) > 0
        )
        has_preview_data = (
            label is not None and preview_baseline is not None and preview_shape_matches
        )
        min_preview_rows = 120
        mask_interpolation_threshold = 0.5
        preview_downsampling_factor = 4
        if (
            len(preview_candidates) > 0
            and preview_baseline is not None
            and not has_preview_data
        ):
            # Preview inputs were provided, but not all of them matched the label domain.
            logger.warning(
                "Skipping manual color-path preview because image dimensions do not "
                "match the label map."
            )

        preview_signal: np.ndarray | None = None
        preview_signal_is_stale = True
        coarse_preview_images: list[darsia.Image] = []
        coarse_preview_baseline: darsia.Image | None = None
        coarse_labels: darsia.Image | None = None
        coarse_mask: np.ndarray | None = None
        coarse_shape = self.labels.img.shape
        current_preview_idx = 0
        if has_preview_data:
            assert preview_baseline is not None
            preview_rows = max(
                min_preview_rows,
                self.labels.img.shape[0] // preview_downsampling_factor,
            )
            preview_rows = min(preview_rows, self.labels.img.shape[0])
            preview_cols = max(
                1,
                int(
                    round(
                        self.labels.img.shape[1]
                        / self.labels.img.shape[0]
                        * preview_rows
                    )
                ),
            )
            coarse_shape = (preview_rows, preview_cols)
            coarse_preview_images = [
                darsia.resize(img, shape=coarse_shape)
                for img in valid_preview_candidates
            ]
            coarse_preview_baseline = darsia.resize(
                preview_baseline, shape=coarse_shape
            )
            coarse_labels = darsia.resize(
                self.labels, shape=coarse_shape, interpolation="inter_nearest"
            )
            float_mask = darsia.ScalarImage(
                self.mask.img.astype(float), **self.mask.metadata()
            )
            resized_mask = darsia.resize(float_mask, shape=coarse_shape)
            coarse_mask = resized_mask.img > mask_interpolation_threshold

        def selected_label_mask() -> np.ndarray:
            assert label is not None
            assert coarse_labels is not None
            assert coarse_mask is not None
            return np.logical_and(coarse_labels.img == label, coarse_mask)

        def selected_label_mask_full() -> np.ndarray:
            assert label is not None
            return np.logical_and(self.labels.img == label, self.mask.img)

        def current_coarse_preview_image() -> darsia.Image:
            assert len(coarse_preview_images) > 0
            return coarse_preview_images[current_preview_idx]

        def current_original_preview_image() -> darsia.Image:
            assert len(valid_preview_candidates) > 0
            return valid_preview_candidates[current_preview_idx]

        def build_current_color_path() -> darsia.ColorPath:
            return darsia.ColorPath(
                colors=None,
                relative_colors=new_key_relative_colors,
                base_color=base_color,
                mode="rgb",
                name=name,
            )

        def compute_preview_signal() -> np.ndarray | None:
            if not has_preview_data or label is None:
                return None

            assert coarse_preview_baseline is not None
            current_label_mask = selected_label_mask()
            if not np.any(current_label_mask):
                return np.full(coarse_shape, np.nan, dtype=float)

            relative_preview = current_coarse_preview_image().img.astype(
                float
            ) - coarse_preview_baseline.img.astype(float)
            selected_relative_colors = relative_preview[current_label_mask].reshape(
                (-1, 3)
            )
            if selected_relative_colors.shape[0] == 0:
                return np.full(coarse_shape, np.nan, dtype=float)

            color_path = build_current_color_path()
            interpolation = darsia.ColorPathInterpolation(
                color_path=color_path,
                color_mode=darsia.ColorMode.RELATIVE,
                values=color_path.equidistant_distances,
                ignore_spectrum=ignore,
            )
            interpretation_values = interpolation(selected_relative_colors)
            interpretation = np.full(coarse_shape, np.nan, dtype=float)
            interpretation[current_label_mask] = interpretation_values
            return interpretation

        def redraw() -> None:
            ax.cla()
            ax.scatter(
                relative_colors[:, 0],
                relative_colors[:, 1],
                relative_colors[:, 2],
                c=np.clip(colors, 0, 1),
                s=10,
                alpha=0.5,
            )

            key_colors = [base_color + c for c in new_key_relative_colors]
            if np.array(new_key_relative_colors).shape[0] == num_dofs:
                key_arr = np.array(new_key_relative_colors)
                ax.plot(*key_arr.T, c="black", linewidth=2)
                ax.scatter(*key_arr.T, c=np.clip(key_colors, 0, 1), s=100)

            ax.plot(*np.array(relative_colors).T, c="gray", linewidth=1, alpha=0.5)

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

            ax.set_title(f"{name} (manual)")
            ax.set_xlabel("R")
            ax.set_ylabel("G")
            ax.set_zlabel("B")

            ax_preview_image.cla()
            ax_preview_interpretation.cla()
            ax_preview_image.axis("off")
            ax_preview_interpretation.axis("off")

            if has_preview_data and label is not None:
                current_label_mask = selected_label_mask()
                preview_rgb = current_coarse_preview_image().img.copy().astype(float)
                if preview_rgb.ndim == 3:
                    grayscale = np.sum(
                        preview_rgb * np.array([0.299, 0.587, 0.114])[None, None, :],
                        axis=2,
                        keepdims=True,
                    )
                    preview_rgb[~current_label_mask] = (
                        0.7 * grayscale[~current_label_mask]
                        + 0.3 * preview_rgb[~current_label_mask]
                    )
                    ax_preview_image.imshow(preview_rgb)
                else:
                    ax_preview_image.imshow(preview_rgb, cmap="gray")
                ax_preview_image.set_title(
                    f"Calibration image {current_preview_idx + 1} / "
                    f"{len(coarse_preview_images)} (selected label focus)"
                )

                if preview_signal is not None:
                    preview_cmap = build_current_color_path().get_color_map()
                    ax_preview_interpretation.imshow(
                        preview_signal,
                        cmap=preview_cmap,
                        vmin=0,
                        vmax=1,
                    )
                    title = "Native pH indicator interpretation"
                    if preview_signal_is_stale:
                        title += " (stale - click Re-run pH preview)"
                    ax_preview_interpretation.set_title(title)
                else:
                    ax_preview_interpretation.text(
                        0.5,
                        0.5,
                        "No preview available.\nClick Re-run pH preview.",
                        ha="center",
                        va="center",
                        transform=ax_preview_interpretation.transAxes,
                    )
                    ax_preview_interpretation.set_title(
                        "Native pH indicator interpretation"
                    )
            else:
                ax_preview_image.text(
                    0.5,
                    0.5,
                    "Manual preview unavailable.",
                    ha="center",
                    va="center",
                    transform=ax_preview_image.transAxes,
                )
                ax_preview_interpretation.text(
                    0.5,
                    0.5,
                    "Manual preview unavailable.",
                    ha="center",
                    va="center",
                    transform=ax_preview_interpretation.transAxes,
                )
                ax_preview_image.set_title("Calibration image")
                ax_preview_interpretation.set_title(
                    "Native pH indicator interpretation"
                )

            fig.canvas.draw_idle()

        index_ax = plt.axes([0.05, 0.12, 0.08, 0.06])
        r_ax = plt.axes([0.15, 0.12, 0.09, 0.06])
        g_ax = plt.axes([0.26, 0.12, 0.09, 0.06])
        b_ax = plt.axes([0.37, 0.12, 0.09, 0.06])
        update_ax = plt.axes([0.48, 0.12, 0.09, 0.06])
        finalize_ax = plt.axes([0.59, 0.12, 0.09, 0.06])
        pipette_ax = plt.axes([0.48, 0.04, 0.09, 0.06])
        previous_preview_ax = plt.axes([0.70, 0.12, 0.10, 0.06])
        next_preview_ax = plt.axes([0.82, 0.12, 0.10, 0.06])
        rerun_preview_ax = plt.axes([0.70, 0.04, 0.22, 0.06])

        index_box = TextBox(index_ax, "Index", initial=str(selected_index))
        init_color = new_key_relative_colors[selected_index]
        r_box = TextBox(r_ax, "R", initial=f"{init_color[0]:.6f}")
        g_box = TextBox(g_ax, "G", initial=f"{init_color[1]:.6f}")
        b_box = TextBox(b_ax, "B", initial=f"{init_color[2]:.6f}")
        update_button = Button(update_ax, "Update")
        finalize_button = Button(finalize_ax, "Finalize")
        pipette_button = Button(pipette_ax, "Pipette")
        previous_preview_button = Button(previous_preview_ax, "Previous")
        next_preview_button = Button(next_preview_ax, "Next")
        rerun_preview_button = Button(rerun_preview_ax, "Re-run pH preview")

        def refresh_color_fields() -> None:
            color = new_key_relative_colors[selected_index]
            r_box.set_val(f"{color[0]:.6f}")
            g_box.set_val(f"{color[1]:.6f}")
            b_box.set_val(f"{color[2]:.6f}")

        def on_index_submit(text: str) -> None:
            nonlocal selected_index
            try:
                idx = int(float(text))
            except ValueError:
                idx = selected_index
            idx = max(0, min(len(new_key_relative_colors) - 1, idx))
            selected_index = idx
            refresh_color_fields()

        def on_update(_) -> None:
            nonlocal preview_signal_is_stale
            try:
                new_color = np.array(
                    [float(r_box.text), float(g_box.text), float(b_box.text)],
                    dtype=float,
                )
            except ValueError:
                logger.warning(
                    "Manual color-path update skipped due to invalid RGB input."
                )
                return
            new_key_relative_colors[selected_index] = new_color
            preview_signal_is_stale = True
            redraw()

        def on_rerun_preview(_) -> None:
            nonlocal preview_signal
            nonlocal preview_signal_is_stale
            preview_signal = compute_preview_signal()
            preview_signal_is_stale = False
            redraw()

        def on_pipette(_) -> None:
            nonlocal preview_signal_is_stale
            if not has_preview_data or label is None:
                logger.warning("Pipette unavailable without preview calibration data.")
                return

            assert preview_baseline is not None
            current_label_mask = selected_label_mask_full()
            if not np.any(current_label_mask):
                logger.warning(
                    "Pipette skipped because selected label has no active pixels."
                )
                return

            coarse_rows, coarse_cols = current_coarse_preview_image().img.shape[:2]
            if coarse_rows <= 0:
                logger.warning(
                    "Pipette skipped due to invalid preview image dimensions."
                )
                return
            if coarse_cols <= 0:
                logger.warning(
                    "Pipette skipped due to invalid preview image dimensions."
                )
                return
            coarse_col_limits = ax_preview_image.get_xlim()
            coarse_row_limits = ax_preview_image.get_ylim()
            coarse_col_start = int(np.floor(max(0.0, min(coarse_col_limits))))
            coarse_col_stop = int(
                np.ceil(min(float(coarse_cols), max(coarse_col_limits)))
            )
            coarse_row_start = int(np.floor(max(0.0, min(coarse_row_limits))))
            coarse_row_stop = int(
                np.ceil(min(float(coarse_rows), max(coarse_row_limits)))
            )

            if (
                coarse_row_stop <= coarse_row_start
                or coarse_col_stop <= coarse_col_start
            ):
                logger.warning(
                    "Pipette skipped because the current zoom window is empty."
                )
                return

            full_rows, full_cols = current_original_preview_image().img.shape[:2]
            full_row_start = int(np.floor(coarse_row_start / coarse_rows * full_rows))
            full_row_stop = int(np.ceil(coarse_row_stop / coarse_rows * full_rows))
            full_col_start = int(np.floor(coarse_col_start / coarse_cols * full_cols))
            full_col_stop = int(np.ceil(coarse_col_stop / coarse_cols * full_cols))
            full_row_start = max(0, min(full_rows - 1, full_row_start))
            full_row_stop = max(full_row_start + 1, min(full_rows, full_row_stop))
            full_col_start = max(0, min(full_cols - 1, full_col_start))
            full_col_stop = max(full_col_start + 1, min(full_cols, full_col_stop))
            sample = (
                slice(full_row_start, full_row_stop),
                slice(full_col_start, full_col_stop),
            )

            current_original_preview = current_original_preview_image()
            if current_original_preview.img.shape != preview_baseline.img.shape:
                logger.warning(
                    "Pipette skipped because preview and baseline dimensions differ."
                )
                return
            current_relative_preview = np.subtract(
                current_original_preview.img,
                preview_baseline.img,
                dtype=float,
            )
            sampled_relative_preview = current_relative_preview[sample]
            sampled_label_mask = current_label_mask[sample]
            characteristic_colors = darsia.extract_characteristic_data(
                signal=sampled_relative_preview,
                mask=sampled_label_mask,
            )
            if len(characteristic_colors) == 0:
                logger.warning(
                    "Pipette skipped because no characteristic color was found."
                )
                return

            sampled_relative_color = np.asarray(characteristic_colors[0], dtype=float)
            new_key_relative_colors[selected_index] = sampled_relative_color
            refresh_color_fields()
            preview_signal_is_stale = True
            redraw()

        def on_previous_preview(_) -> None:
            nonlocal current_preview_idx
            nonlocal preview_signal
            nonlocal preview_signal_is_stale
            if len(coarse_preview_images) == 0:
                return
            current_preview_idx = (current_preview_idx - 1) % len(coarse_preview_images)
            preview_signal = compute_preview_signal()
            preview_signal_is_stale = False
            redraw()

        def on_next_preview(_) -> None:
            nonlocal current_preview_idx
            nonlocal preview_signal
            nonlocal preview_signal_is_stale
            if len(coarse_preview_images) == 0:
                return
            current_preview_idx = (current_preview_idx + 1) % len(coarse_preview_images)
            preview_signal = compute_preview_signal()
            preview_signal_is_stale = False
            redraw()

        def on_finalize(_) -> None:
            nonlocal finalized
            finalized = True
            plt.close(fig)

        index_box.on_submit(on_index_submit)
        update_button.on_clicked(on_update)
        pipette_button.on_clicked(on_pipette)
        previous_preview_button.on_clicked(on_previous_preview)
        next_preview_button.on_clicked(on_next_preview)
        rerun_preview_button.on_clicked(on_rerun_preview)
        finalize_button.on_clicked(on_finalize)

        preview_signal = compute_preview_signal()
        preview_signal_is_stale = False
        redraw()
        plt.show()
        if not finalized:
            logger.info("Manual color-path editor closed without explicit finalize.")
        return new_key_relative_colors
