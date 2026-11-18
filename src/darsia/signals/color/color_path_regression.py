"""Provide tools for defining color paths."""

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.manifold import LocallyLinearEmbedding

import darsia

from .utils import get_mean_color

logger = logging.getLogger(__name__)


def _voxel_faces(
    filled: np.ndarray, edges: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Visible faces of filled voxels, as quads in the coordinates of the bin edges.

    A face is visible if the neighboring voxel across it is not filled (as in
    ``Axes3D.voxels``). Faces are oriented outwards, for shading.

    Parameters
    ----------
    filled : np.ndarray
        Boolean 3D array of filled voxels.
    edges : list[np.ndarray]
        Bin edges per axis (length = shape + 1).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Quads of shape (n, 4, 3) and the voxel index (n, 3) of each quad.
    """
    voxels = np.argwhere(filled)
    padded = np.pad(filled, 1)
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    quads, owners = [], []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        for side in (0, 1):
            neighbor = voxels + 1  # index into the padded array
            neighbor[:, a] += 1 if side else -1
            visible = voxels[~padded[tuple(neighbor.T)]]
            if len(visible) == 0:
                continue
            order = square if side else square[::-1]
            quad = np.empty((len(visible), 4, 3))
            quad[:, :, a] = edges[a][visible[:, a] + side][:, None]
            quad[:, :, b] = edges[b][visible[:, b][:, None] + order[None, :, 0]]
            quad[:, :, c] = edges[c][visible[:, c][:, None] + order[None, :, 1]]
            quads.append(quad)
            owners.append(visible)
    if not quads:
        return np.empty((0, 4, 3)), np.empty((0, 3), dtype=int)
    return np.concatenate(quads), np.concatenate(owners)


@dataclass
class ColorPathInputs:
    """Data of one label for editing its color path by hand."""

    label: int | None
    """The label."""
    name: str
    """Name of the color path."""
    spectrum: "darsia.ColorSpectrum"
    """Color spectrum of the label (relative colors)."""
    ignore: "darsia.ColorSpectrum | None"
    """Ignored spectrum (baseline fluctuations)."""
    relative_colors: np.ndarray
    """Relative colors of the spectrum."""
    colors: np.ndarray
    """Absolute colors of the spectrum."""
    key_relative_colors: list[np.ndarray]
    """Key colors at the start (previous calibration or automatic fit)."""
    base_color: np.ndarray
    """Base color of the label."""
    auto_fit: Callable[[int], list[np.ndarray]]
    """Automatic fit with a given number of segments."""
    num_segments: int
    """Number of segments of the automatic fit."""


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

        Parameters
        ----------
        image : darsia.Image
            The image from which to extract base colors.
        verbose : bool
            Whether to print additional information.

        Returns
        -------
        LabelColorMap
            An object containing the base colors for each label.
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

        Parameters
        ----------
        image : darsia.Image
            The image from which to extract base colors.

        Returns
        -------
        np.ndarray
            The mean base color across all labels.
        """
        base_colors = self.get_base_colors(image)
        return base_colors.mean()

    @darsia.timing_decorator
    def base_color_image(self, image: darsia.Image) -> darsia.Image:
        """Create an image where each label is colored by its base color.

        Parameters
        ----------
        image : darsia.Image
            The image from which to extract base colors.

        Returns
        -------
        darsia.Image
            An image where each label is colored by its base color.
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
        path: Optional[Path] = None,
        verbose: bool = False,
    ) -> darsia.LabelColorSpectrumMap:
        """Get the color spectrum for each label in the image.

        The color spectrum is calculated by analyzing the distribution of
        colors within each label across the provided images. This involves
        creating a 3D histogram of color occurrences, which is then normalized
        to identify significant colors.

        Parameters
        ----------
        images : list[darsia.Image]
            The images to analyze. An empty list stands for the baseline alone: all
            relative colors are zero, as for ``images=[baseline]``, but without
            copying the baseline or looping over its pixels.
        baseline : darsia.Image | None
            The baseline image for comparison.
        ignore : darsia.LabelColorSpectrumMap | darsia.ColorSpectrum | None
            Colors to
            ignore in the spectrum.
        threshold_zero : float
            The threshold for zeroing out insignificant colors.
        threshold_significant : float
            The threshold for significant colors.
        path : Path
            Folder to store images.
        verbose : bool
            Whether to print verbose output.

        Returns
        -------
        LabelColorSpectrumMap
            The color spectrum for each label.
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

        # Loop over all images; without images (None), each label has the zero relative
        # color only, which gives the same normalized histogram as images=[baseline].
        for image in images if len(images) > 0 else [None]:
            # Get relative image
            if image is not None:
                relative_image = image.copy()
                if baseline is not None:
                    relative_image.img -= baseline.img
                relative_image.img[~self.mask.img] = 0.0

            for label in np.unique(self.labels.img):
                # Fetch relative colors in the label
                if image is None:
                    relative_colors = np.zeros((1, 3))
                else:
                    relative_colors = relative_image.img[
                        self.labels.img == label
                    ].reshape((-1, 3))

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
            for label in np.unique(self.labels.img):
                if not np.any(color_spectrum_map[label].spectrum):
                    logger.info(f"Skip plotting color spectrum for label {label}.")
                    continue
                ax = plt.figure().add_subplot(projection="3d")
                ax.set_title(f"Color Spectrum for Label {label}")
                self._plot_spectrum_voxels(
                    ax,
                    color_spectrum_map[label],
                    ignore=(
                        None
                        if ignore is None
                        else (
                            ignore[label]
                            if isinstance(ignore, darsia.LabelColorSpectrumMap)
                            else ignore
                        )
                    ),
                )
                if path is not None:
                    path.mkdir(parents=True, exist_ok=True)
                    plt.savefig(
                        path / f"spectrum_{label}",
                        dpi=500,
                        bbox_inches="tight",
                        pad_inches=0,
                    )
                    logger.info(f"Spectrum saved for label {label} to {path}")
                plt.show()

        logger.info("Done. Color spectrum analysis.")
        return color_spectrum_map

    def _plot_spectrum_voxels(
        self,
        ax,
        spectrum: darsia.ColorSpectrum,
        ignore: darsia.ColorSpectrum | None = None,
    ) -> None:
        """Draw a color spectrum as voxels at their (relative) RGB coordinates.

        Each voxel has the color it represents; its opacity follows the histogram in
        10 levels. Ignored colors are faint red, the origin (zero relative color) is
        red.

        Parameters
        ----------
        ax : Axes3D
            3D axes to draw into.
        spectrum : darsia.ColorSpectrum
            Spectrum (and histogram) of one label.
        ignore : darsia.ColorSpectrum, optional
            Spectrum of ignored colors of the same label.
        """
        # Voxel edges and centers in color coordinates (same binning as the spectrum)
        _, edges = np.histogramdd(
            np.zeros((1, 3)), bins=self._shape, range=self.color_range.range
        )
        centers = np.meshgrid(*[0.5 * (e[:-1] + e[1:]) for e in edges], indexing="ij")

        # Each voxel in the color it represents; opacity by the histogram, normalized
        # to the most frequent color and blocked into 10 levels
        histogram = spectrum.histogram
        hist_max = (
            np.max(histogram[spectrum.spectrum]) if np.any(spectrum.spectrum) else 0.0
        )
        alpha = (
            np.ceil(10 * np.clip(histogram / hist_max, 0.0, 1.0)) / 10
            if hist_max > 0
            else np.ones(self._shape)
        )
        spectrum_colors = np.concatenate(
            [
                np.clip(spectrum.base_color + np.stack(centers, axis=-1), 0.0, 1.0),
                np.maximum(alpha, 0.1)[..., None],
            ],
            axis=-1,
        )

        # Highlighted (relative) origin
        origin = np.zeros(self._shape, dtype=bool)
        origin_index = self.discrete_color_range.color_to_index(
            np.zeros(3)
            if self.color_mode == darsia.ColorMode.RELATIVE
            else spectrum.base_color
        )
        origin[tuple(origin_index)] = True

        # One shaded collection per layer (built from the visible faces only; much
        # faster than ax.voxels, which loops over the whole grid)
        layers = [(spectrum.spectrum, spectrum_colors)]
        if ignore is not None:
            layers.append((ignore.spectrum, np.array([1.0, 0.0, 0.0, 0.1])))
        layers.append((origin, np.array([1.0, 0.0, 0.0, 1.0])))
        points = []
        for filled, colors in layers:
            quads, owners = _voxel_faces(filled, edges)
            if len(quads) == 0:
                continue
            face_colors = (
                colors[tuple(owners.T)]
                if colors.ndim > 1
                else np.tile(colors, (len(quads), 1))
            )
            ax.add_collection3d(
                Poly3DCollection(quads, facecolors=face_colors, shade=True)
            )
            points.append(quads.reshape(-1, 3))
        if points:
            points = np.concatenate(points)
            ax.auto_scale_xyz(points[:, 0], points[:, 1], points[:, 2])

        ax.set_xlabel("R")
        ax.set_ylabel("G")
        ax.set_zlabel("B")

    @darsia.timing_decorator
    def expand_color_spectrum(
        self,
        color_spectrum: darsia.ColorSpectrum | darsia.LabelColorSpectrumMap,
        min_points: int = 6,
        verbose: bool = False,
    ) -> darsia.LabelColorSpectrumMap:
        """Expand the color spectrum through linear regression.

        Parameters
        ----------
        color_spectrum : ColorSpectrum or LabelColorSpectrumMap
            The color spectrum to expand.
        min_points : int
            Minimum number of significant points to perform regression.
        verbose : bool
            Whether to print additional information.

        Returns
        -------
        LabelColorSpectrumMap
            The expanded color spectrum.
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

        Parameters
        ----------
        color_spectrum : LabelColorSpectrum
            The color spectrum to expand.
        min_points : int
            Minimum number of significant points to perform regression.
        title : str
            Title for the verbose plot.
        verbose : bool
            Whether to print additional information.

        Returns
        -------
        ColorSpectrum
            The expanded color spectrum.
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
    def _prepare_label(
        self,
        spectrum: darsia.ColorSpectrum,
        label: int | None = None,
        ignore: darsia.ColorSpectrum | None = None,
        num_segments: int = 1,
        name: str = "Color Path",
        directory: Path | None = None,
        weighting: Literal["threshold", "wls", "wls_sqrt", "wls_log"] = "threshold",
        initial_key_relative_colors: list[np.ndarray] | None = None,
        verbose: bool = False,
    ) -> "ColorPathInputs | darsia.ColorPath":
        """Fit a relative color path through the significant boxes of one label.

        Parameters
        ----------
        spectrum : ColorSpectrum
            The color spectrum to analyze.
        label : int, optional
            The label (for messages).
        ignore : ColorSpectrum | None
            The color spectrum to ignore.
        num_segments : int
            The number of segments for the color path.
        name : str
            Name of the color path.
        directory : Path | None
            Directory to save verbose plots.
        weighting : str
            How to use histogram counts when fitting the path.
            ``"threshold"`` (default) uses binary 0/1 weights controlled by
            the spectrum threshold - counts are not used beyond that.
            ``"wls"`` weights each active bin by its normalised probability.
            ``"wls_sqrt"`` weights by the square-root of the probability.
            ``"wls_log"`` weights by ``log(1 + count)`` where count is
            derived from the normalised probability scaled by the total
            number of active bins.
        initial_key_relative_colors : list[np.ndarray] | None
            Previously calibrated key colors; replace the automated fit as start
            (manual calibration).
        verbose : bool
            Whether to print additional information.

        Returns
        -------
        ColorPathInputs or darsia.ColorPath
            The key colors (start) with the data to edit them; a fixed default (zero)
            path if no path can be fitted (too few colors, failed embedding, no key
            colors).

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
        #
        # The candidate points are bin centers of a regular color-range grid,
        # so for sparse/degenerate spectra (e.g. near-collinear or clustered
        # colors) the local reconstruction weight matrix can be rank-deficient.
        # sklearn's default eigen_solver="auto" tends to pick "arpack" for
        # this problem size, and arpack's shift-invert factorization at
        # sigma=0 raises "Factor is exactly singular" on such degenerate
        # inputs. "dense" avoids that factorization and is cheap at this
        # scale, so we prefer it and only fall back to arpack's default
        # behavior if dense itself fails for some other reason.
        n_neighbors = min(10, num_points - 1)
        try:
            lle = LocallyLinearEmbedding(
                n_neighbors=n_neighbors, n_components=1, eigen_solver="dense"
            )
            embedding = lle.fit_transform(relative_colors).flatten()
        except (ValueError, np.linalg.LinAlgError) as err:
            logger.warning(
                "LLE embedding failed for label %s (%s active points): %s. "
                "Returning default (zero) color path for this label.",
                label,
                num_points,
                err,
            )
            return darsia.ColorPath(
                colors=None,
                base_color=spectrum.base_color,
                relative_colors=num_dofs * [np.zeros(3)],
                mode="rgb",
                name=name,
            )

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

            Parameters
            ----------
            segment_range : range
                A range object specifying indices of points in the segment.

            Returns
            -------
            float
                The 80th percentile of L1 errors between linear regression
                predictions and actual colors. Values are typically in [0, 3]
                for normalized RGB color space.

            Notes
            -----
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

            Parameters
            ----------
            segment_range : range
                A range object specifying indices in the segment.

            Returns
            -------
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

        def auto_fit(n_segments: int) -> list[np.ndarray]:
            """Run the automated segmentation for a given number of segments."""
            segments: list[dict] = []

            segment_range = range(0, len(sorted_embedding))
            _segment_error = segment_error(segment_range)
            _segment_length = segment_length(segment_range)
            segment = {
                "range": segment_range,
                "error": _segment_error,
                "length": _segment_length,
            }
            segments.append(segment)

            # Perform a Ramer-Douglas-Peucker-like segmentation based on weighted
            # least squares error reduction
            while len(segments) < n_segments:
                # Eligible segments are those with at least 3 points
                eligible_segments = [seg for seg in segments if len(seg["range"]) > 2]

                # Stop if no segment can be split further
                if len(eligible_segments) == 0:
                    warn("Cannot split segments further. Stopping early.")
                    break

                # Identify the segment with the highest error (at least 3 points)
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
                            segments[i]["range"].start
                            - previous_segments[i]["range"].start
                        )
                        + abs(
                            segments[i]["range"].stop
                            - previous_segments[i]["range"].stop
                        )
                        for i in range(len(segments))
                    )
                    old_distances.append(distance)
                    if (
                        len(old_distances) > 5
                        and len(np.unique(old_distances[-5:])) == 1
                    ):
                        # No change in the last 5 iterations - oscillation detected
                        break
                    print(f"Segment smoothing distance: {distance}")

            # Extract the key relative colors from the segments
            return [
                sorted_relative_colors[segment["range"].start] for segment in segments
            ] + [sorted_relative_colors[segments[-1]["range"].stop - 1]]

        if initial_key_relative_colors:
            key_relative_colors = [
                np.array(c, dtype=float) for c in initial_key_relative_colors
            ]
        else:
            key_relative_colors = auto_fit(num_segments)

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

        return ColorPathInputs(
            label=label,
            name=name,
            spectrum=spectrum,
            ignore=ignore,
            relative_colors=relative_colors,
            colors=colors,
            key_relative_colors=key_relative_colors,
            base_color=spectrum.base_color,
            auto_fit=auto_fit,
            num_segments=num_segments,
        )

    def _path_from_keys(
        self,
        inputs: "ColorPathInputs",
        key_relative_colors: list[np.ndarray],
        directory: Path | None = None,
        verbose: bool = False,
    ) -> darsia.ColorPath:
        """Color path through key colors; saves a plot of it (if `directory`).

        Parameters
        ----------
        inputs : ColorPathInputs
            Data of the label (spectrum, base color, name).
        key_relative_colors : list[np.ndarray]
            Key colors relative to the base color.
        directory : Path, optional
            Directory of the plot.
        verbose : bool
            Whether to print the key colors and show the plot.

        Returns
        -------
        darsia.ColorPath
            The relative color path.

        """
        spectrum, ignore, name = inputs.spectrum, inputs.ignore, inputs.name

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

        # Visualize: the spectrum as voxels (as in get_color_spectrum), the path on top.
        # Without computed z-order, artists are drawn in zorder, so 3D depth sorting
        # cannot hide the path behind voxels.
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        self._plot_spectrum_voxels(ax, spectrum, ignore=ignore)

        # Plot key colors and connecting lines
        if len(key_relative_colors) >= 2:
            ax.plot(*np.array(key_relative_colors).T, c="black", linewidth=2, zorder=10)
            ax.scatter(
                *np.array(key_relative_colors).T,
                c=np.clip(key_colors, 0, 1),
                s=100,
                edgecolors="black",
                depthshade=False,
                zorder=11,
            )

        ax.set_title(name)

        plt.tight_layout()
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
            plot_save_path = directory / f"{name}.png"
            plt.savefig(plot_save_path, dpi=300)
        if verbose:
            plt.show()
        plt.close()

        return relative_color_path

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
        verbose: bool = False,
    ) -> darsia.ColorPath:
        """Automatic color path of one label (see `_prepare_label`)."""
        if mode == "manual":
            raise ValueError(
                "Manual color-path mode: use prepare_color_path_editing and "
                "color_path_map with an interactive editor."
            )
        if mode != "auto":
            raise ValueError(
                f"Unknown color-path mode '{mode}'. Allowed: 'auto', 'manual'."
            )
        prepared = self._prepare_label(
            spectrum=spectrum,
            label=label,
            ignore=ignore,
            num_segments=num_segments,
            name=name,
            directory=directory,
            weighting=weighting,
            verbose=verbose,
        )
        if isinstance(prepared, darsia.ColorPath):
            return prepared
        return self._path_from_keys(
            prepared, prepared.key_relative_colors, directory, verbose
        )

    def prepare_color_path_editing(
        self,
        color_spectrum: darsia.LabelColorSpectrumMap,
        ignore: darsia.LabelColorSpectrumMap | None = None,
        num_segments: int = 1,
        weighting: Literal["threshold", "wls", "wls_sqrt", "wls_log"] = "threshold",
        ignore_labels: list[int] | None = None,
        initial_paths: dict[int, list[np.ndarray]] | None = None,
        verbose: bool = False,
    ) -> dict[int, "ColorPathInputs | darsia.ColorPath"]:
        """Start of the manual calibration: the data to edit the paths of all labels.

        Parameters
        ----------
        color_spectrum : LabelColorSpectrumMap
            The color spectrum map to analyze.
        ignore : LabelColorSpectrumMap | None
            The color spectrum map to ignore.
        num_segments : int
            The number of segments of the automatic fit.
        weighting : Literal
            How to use histogram counts when fitting (see `_prepare_label`).
        ignore_labels : list[int] | None
            Labels with a zero path (not editable).
        initial_paths : dict[int, list[np.ndarray]] | None
            Previously calibrated relative key colors per label; start instead of the
            automatic fit.
        verbose : bool
            Whether to print additional information.

        Returns
        -------
        dict
            Per label the inputs of the editor (`ColorPathInputs`), or a fixed path
            (zero path of an ignored label, default path if no path can be fitted).

        """
        ignore_labels = ignore_labels or []
        initial_paths = initial_paths or {}
        prepared = {}
        for label in color_spectrum.keys():
            if label in ignore_labels:
                prepared[label] = darsia.ColorPath(
                    colors=[np.zeros(3), np.zeros(3)],
                    mode="rgb",
                    name=f"Zero Path for Label {label}",
                )
                continue
            prepared[label] = self._prepare_label(
                spectrum=color_spectrum[label],
                label=label,
                ignore=ignore[label] if ignore is not None else None,
                num_segments=num_segments,
                name=f"Color Path for Label {label}",
                weighting=weighting,
                initial_key_relative_colors=initial_paths.get(label),
                verbose=verbose,
            )
        return prepared

    def color_path_map(
        self,
        prepared: dict[int, "ColorPathInputs | darsia.ColorPath"],
        keys: dict[int, list[np.ndarray]],
        directory: Path | None = None,
    ) -> darsia.LabelColorPathMap:
        """Color paths of all labels from the edited key colors.

        Parameters
        ----------
        prepared : dict
            Result of `prepare_color_path_editing`.
        keys : dict[int, list[np.ndarray]]
            Relative key colors of the edited labels.
        directory : Path, optional
            Directory of the plots of the paths.

        Returns
        -------
        LabelColorPathMap
            Edited paths, and the fixed paths of the labels that are not editable.

        """
        return darsia.LabelColorPathMap(
            {
                label: (
                    item
                    if isinstance(item, darsia.ColorPath)
                    else self._path_from_keys(item, keys[label], directory)
                )
                for label, item in prepared.items()
            }
        )

    @darsia.timing_decorator
    def find_color_path(
        self,
        color_spectrum: darsia.LabelColorSpectrumMap,
        ignore: darsia.LabelColorSpectrumMap | None = None,
        num_segments: int = 1,
        directory: Path | None = None,
        weighting: Literal["threshold", "wls", "wls_sqrt", "wls_log"] = "threshold",
        mode: Literal["auto", "manual"] = "auto",
        ignore_labels: list[int] | None = None,
        verbose: bool = False,
    ) -> darsia.LabelColorPathMap:
        """Find relative color paths for each label in the spectrum map (automatic).

        For the manual calibration, see `prepare_color_path_editing` and
        `color_path_map`.

        Parameters
        ----------
        color_spectrum : LabelColorSpectrumMap
            The color spectrum map to
            analyze.
        ignore : LabelColorSpectrumMap | None
            The color spectrum map to ignore.
        num_segments : int
            The number of segments for the color path.
        weighting : Literal
            How to use histogram counts when fitting each path.
            ``"threshold"`` (default) uses binary 0/1 weights from the spectrum
            threshold – counts are not used beyond that.
            ``"wls"``, ``"wls_sqrt"``, and ``"wls_log"`` use count-weighted
            average errors – see :meth:`_find_color_path` for details.
        mode : Literal
            Only ``"auto"``; ``"manual"`` raises (see `prepare_color_path_editing`).
        ignore_labels : list[int] | None
            Label IDs to skip fitting and produce
            zero-paths for instead. Defaults to None (no labels ignored).
        verbose : bool
            Whether to print additional information.

        Returns
        -------
        LabelColorPathMap
            The relative color path map through the significant boxes.
        """
        if ignore_labels is None:
            ignore_labels = []
        if mode == "manual":
            raise ValueError(
                "Manual color-path mode: use prepare_color_path_editing and "
                "color_path_map with an interactive editor."
            )
        if mode != "auto":
            raise ValueError(
                f"Unknown color-path mode '{mode}'. Allowed: 'auto', 'manual'."
            )

        # Helper to create a zero-path (all-zero color path for ignored labels)
        def create_zero_path(label: int) -> darsia.ColorPath:
            """Create a zero-path (constant zero colors) for an ignored label."""
            return darsia.ColorPath(
                colors=[np.zeros(3), np.zeros(3)],
                mode="rgb",
                name=f"Zero Path for Label {label}",
            )

        label_color_path_map = darsia.LabelColorPathMap(
            dict(
                (
                    label,
                    (
                        create_zero_path(label)
                        if label in ignore_labels
                        else self._find_color_path(
                            spectrum=color_spectrum[label],
                            label=label,
                            ignore=ignore[label] if ignore is not None else None,
                            num_segments=num_segments,
                            name=f"Color Path for Label {label}",
                            directory=directory,
                            weighting=weighting,
                            verbose=verbose,
                        )
                    ),
                )
                for label in color_spectrum.keys()
            )
        )
        return label_color_path_map
