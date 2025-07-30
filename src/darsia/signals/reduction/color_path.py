"""Piecewise linear color path in RGB space."""

from typing import Literal

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import darsia


class ColorPath:
    """Piecewise linear color path in RGB space.

    This class allows to define a color path through a list of absolute colors in RGB space.
    In addition, it supports parametrization of given images in terms of the color path.
    Both absolute and relative colors can be used to define the path.

    If using the "rgb" mode, there are no technical restrictions or assumptions on the
    validity of the colors. The class could be used for parametrization of any vector-valued
    image of any dimension. The main purpose though is to provide a convenient way
    to define a color path in RGB space, which can be used for visualization purposes. For
    this, a color map can be created which can be used in matplotlib or other libraries.

    """

    def __init__(
        self,
        colors: np.ndarray | list[np.ndarray],
        values: np.ndarray | list[float] | None = None,
        mode: Literal["rgb", "lab", "hcl"] = "rgb",
    ) -> None:
        """Color path.

        Args:
            colors: Absolute colors in RGB space, defining the color path.
            values: Values from 0 to 1 parametrizing/sampling the colors. If `None`,
                the relative distances between the colors are used as natural values.
            mode: Color space to use for interpolation in between colors.
                Defaults to "rgb".

        """
        if isinstance(colors, np.ndarray):
            colors = [c for c in colors]
        self.colors: list[np.ndarray] = colors
        """Absolute colors in RGB space."""

        self.relative_colors: list[np.ndarray] = [c - colors[0] for c in colors]
        """Relative colors in RGB space wrt. the first color."""

        if values is not None:
            if isinstance(values, np.ndarray):
                values = values.tolist()
            self.values: list = values
            """Relative distances between the colors in the path."""
        else:
            # Utilize the relative distances as natural values
            distances = [
                np.linalg.norm(self.relative_colors[i] - self.relative_colors[i - 1])
                for i in range(1, len(self.relative_colors))
            ]
            self.values = (np.cumsum([0] + distances) / sum(distances)).tolist()

        self.num_segments = len(self.colors) - 1
        """Number of segments in the color path."""

        self.mode = mode
        """Color space to use for interpolation in between colors."""

    def sample_absolute_color_path(self, n_colors: int = 256) -> list[np.ndarray]:
        """Sample the absolute color path through interpolation.

        The interpolation mode depends on the `mode` parameter of the constructor.

        Args:
            n_colors: Number of quantization levels in the colormap.

        Returns:
            list[np.ndarray]: Sampled absolute color path.

        """
        if self.mode == "rgb":
            color_list = []
            for i in range(n_colors):
                ratio = i / (n_colors - 1)
                index = np.searchsorted(self.values, ratio)
                if index == 0:
                    color_list.append(self.colors[0])
                elif index >= len(self.colors):
                    color_list.append(self.colors[-1])
                else:
                    # Interpolate between the two colors
                    ratio_in_segment = (ratio - self.values[index - 1]) / (
                        self.values[index] - self.values[index - 1]
                    )
                    color = (
                        self.colors[index - 1] * (1 - ratio_in_segment)
                        + self.colors[index] * ratio_in_segment
                    )
                    color_list.append(color)

        elif self.mode == "lab":
            raise NotImplementedError(
                "Only RGB mode is currently implemented for sampling the absolute color path."
            )
        elif self.mode == "hcl":
            raise NotImplementedError(
                "Only RGB mode is currently implemented for sampling the absolute color path."
            )

        return color_list

    def get_color_map(
        self, n_colors: int = 256, name="custom_cmap"
    ) -> LinearSegmentedColormap:
        """Create a colormap from the color path, ready for matplotlib.

        Args:
            n_colors: Number of quantization levels in the colormap.
            name: Name of the colormap.

        Returns:
            LinearSegmentedColormap: Colormap created from the color path.

        """
        # Sample the absolute color path
        color_list = self.sample_absolute_color_path(n_colors=n_colors)

        # Ensure the colors are in the range [0, 1]
        clipped_color_list = [np.clip(c, 0, 1) for c in color_list]

        return LinearSegmentedColormap.from_list(
            name,
            clipped_color_list,
            N=n_colors,
        )

    def _parametrize_colors(
        self, image: darsia.Image, supports: list[np.ndarray]
    ) -> darsia.Image:
        """Parametrize the image in terms of the color path.

        Apply brute-force minimization to find the closest color representation
        on the path for each pixel in the image.

        Args:
            image: Input image to be interpreted.
            supports: List of colors defining the color path.

        Returns:
            darsia.Image: Parametrization of the input image in terms of the color path.

        """
        # Sanity check
        assert len(supports) == self.num_segments + 1

        # Find the best-fit scalar interpretation for each segment
        interpretations = []
        for segment in range(self.num_segments):
            scalar_interpretation = self.values[segment] + (
                self.values[segment + 1] - self.values[segment]
            ) * np.tensordot(
                image.img - supports[segment],
                supports[segment + 1] - supports[segment],
                axes=([2], [0]),
            ) / np.dot(
                supports[segment + 1] - supports[segment],
                supports[segment + 1] - supports[segment],
            )
            interpretations.append(scalar_interpretation)

        # Convert each segment to its color interpretation
        shape = (image.img.shape[0], image.img.shape[1], 3)
        color_interpretations = [
            supports[segment]
            + np.outer(
                (interpretations[segment] - self.values[segment])
                / (self.values[segment + 1] - self.values[segment]),
                supports[segment + 1] - supports[segment],
            ).reshape(shape)
            for segment in range(self.num_segments)
        ]

        # Compare the different segments and find the global best-fit segment for each pixel
        distances = np.stack(
            [
                np.linalg.norm(image.img - color_interpretations[segment], axis=-1)
                for segment in range(self.num_segments)
            ]
        )
        closest_segment = np.argmin(distances, axis=0)

        # Finalize minimization by taking the best-fit interpretation
        best_fit_interpretation = darsia.zeros_like(image, mode="voxels")
        for segment in range(self.num_segments):
            mask = closest_segment == segment
            best_fit_interpretation.img[mask] = interpretations[segment][mask]

        return best_fit_interpretation

    def inverse_absolute_color_path(self, image: darsia.Image) -> darsia.Image:
        """Inverse the absolute color path to an image defined by the closest color
        representation on the path.

        Args:
            image: Input image to be interpreted.

        Returns:
            darsia.Image: Parametrization of the input image in terms of the color path.

        """
        return self._parametrize_colors(image, self.absolute_colors)

    def inverse_relative_color_path(self, image: darsia.Image) -> darsia.Image:
        """Inverse the relative color path to an image defined by the closest color
        representation on the path.

        Args:
            image: Input image to be interpreted.

        Returns:
            darsia.Image: Parametrization of the input image in terms of the relative
                color path.

        """
        return self._parametrize_colors(image, self.relative_colors)
