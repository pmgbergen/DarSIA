"""Piecewise linear color path in RGB space."""

import json
import logging
from pathlib import Path
from typing import Literal, overload

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import darsia

logger = logging.getLogger(__name__)


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
        colors: list[np.ndarray] | None = [np.zeros(3), np.ones(3)],
        base_color: np.ndarray | None = None,
        relative_colors: list[np.ndarray] | None = None,
        values: np.ndarray | list[float] | None = None,
        mode: Literal["rgb", "lab", "hcl"] = "rgb",
    ) -> None:
        """Color path.

        Args:
            colors: Absolute colors in RGB space, defining the color path. Defaults
                to a simple grayscale path from black to white.
            base_color: Base color in RGB space, used as the first color in the path.
            relative_colors: Relative colors in RGB space wrt. the base color,
                defining the color path. If provided, `colors` must be `None`.
            values: Values from 0 to 1 parametrizing/sampling the colors. If `None`,
                the relative distances between the colors are used as natural values.
            mode: Color space to use for interpolation in between colors.
                Defaults to "rgb".

        """
        # Sanity checks
        assert colors is not None or relative_colors is not None
        assert not (colors is not None and relative_colors is not None)
        assert not (relative_colors is not None and base_color is None)

        if colors is not None:
            self.colors: list[np.ndarray] = colors
            """Absolute colors in RGB space."""

            if base_color is not None:
                self.base_color = base_color
            else:
                self.base_color = colors[0]

            self.relative_colors: list[np.ndarray] = [
                c - self.base_color for c in colors
            ]
            """Relative colors in RGB space wrt. the first color."""
        elif relative_colors is not None:
            self.relative_colors: list[np.ndarray] = relative_colors
            """Relative colors in RGB space."""
            self.base_color = base_color
            self.colors = [self.base_color + c for c in relative_colors]

        if values is not None:
            if isinstance(values, np.ndarray):
                self.values = values.tolist()
            elif isinstance(values, list):
                self.values = values
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

    def sort(self) -> None:
        """Sort values such that they are in ascending order."""
        sorted_indices = np.argsort(self.values)
        self.colors = [self.colors[i] for i in sorted_indices]
        self.relative_colors = [self.relative_colors[i] for i in sorted_indices]
        self.values = [self.values[i] for i in sorted_indices]

    def update_values(self, values: list[float]) -> None:
        """Update the values of the color path, and resort colors."""
        self.values = values
        self.sort()

    def add_absolute(self, colors, values) -> None:
        """Add absolute colors and their corresponding values to the color path."""
        # Sanity check
        assert len(colors) == len(values)

        # Add colors and values to the color path
        self.colors.extend(colors)
        self.relative_colors.extend([c - self.base_color for c in colors])
        self.values.extend(values)
        self.num_segments = len(self.colors) - 1
        self.sort()

    def add_relative(self, colors: list[np.ndarray], values: list[float]) -> None:
        """Add relative colors and their corresponding values to the color path."""
        # Sanity check
        assert len(colors) == len(values)

        # Add colors and values to the color path
        self.colors.extend([self.base_color + c for c in colors])
        self.relative_colors.extend(colors)
        self.values.extend(values)
        self.num_segments = len(self.colors) - 1
        self.sort()

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
            # Normalize the values to the range [0, 1]
            normalized_values = [
                (v - min(self.values)) / (max(self.values) - min(self.values))
                for v in self.values
            ]
            for i in range(n_colors):
                ratio = i / (n_colors - 1)
                index = np.searchsorted(normalized_values, ratio)
                if index == 0:
                    color_list.append(self.colors[0])
                elif index >= len(self.colors):
                    color_list.append(self.colors[-1])
                else:
                    # Interpolate between the two colors
                    ratio_in_segment = (ratio - normalized_values[index - 1]) / (
                        normalized_values[index] - normalized_values[index - 1]
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

    def show(self) -> None:
        """Visualize the color path as a colormap."""
        cmap = self.get_color_map(n_colors=256)
        plt.imshow([np.arange(256)], aspect="auto", cmap=cmap)
        plt.axis("off")
        plt.show()

    def _parametrize_colors(
        self, array: np.ndarray, supports: list[np.ndarray]
    ) -> np.ndarray:
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
                array - supports[segment],
                supports[segment + 1] - supports[segment],
                axes=([-1], [0]),
            ) / np.dot(
                supports[segment + 1] - supports[segment],
                supports[segment + 1] - supports[segment],
            )
            interpretations.append(scalar_interpretation)

        # Convert each segment to its color interpretation
        shape = array.shape[:-1] + (-1,)
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
                np.linalg.norm(array - color_interpretations[segment], axis=-1)
                for segment in range(self.num_segments)
            ]
        )
        closest_segment = np.argmin(distances, axis=0)

        # Finalize minimization by taking the best-fit interpretation
        best_fit_interpretation = np.zeros(array.shape[:-1])
        for segment in range(self.num_segments):
            mask = closest_segment == segment
            best_fit_interpretation[mask] = interpretations[segment][mask]

        return best_fit_interpretation

    @overload
    def absolute_inverse(self, image: np.ndarray) -> np.ndarray: ...

    @overload
    def absolute_inverse(self, image: darsia.Image) -> darsia.Image: ...

    def absolute_inverse(
        self, image: np.ndarray | darsia.Image
    ) -> np.ndarray | darsia.Image:
        """Inverse the absolute color path to an image defined by the closest color
        representation on the path.

        Args:
            image: Input image to be interpreted.

        Returns:
            darsia.Image: Parametrization of the input image in terms of the color path.

        """
        if isinstance(image, np.ndarray):
            return self._parametrize_colors(image, self.absolute_colors)
        if isinstance(image, darsia.Image):
            return darsia.full_like(
                image,
                fill_value=self._parametrize_colors(image.img, self.absolute_colors),
                mode="voxels",
            )

    @overload
    def relative_inverse(self, image: np.ndarray) -> np.ndarray: ...

    @overload
    def relative_inverse(self, image: darsia.Image) -> darsia.Image: ...

    def relative_inverse(self, image: darsia.Image) -> darsia.Image:
        """Inverse the relative color path to an image defined by the closest color
        representation on the path.

        Args:
            image: Input image to be interpreted.

        Returns:
            darsia.Image: Parametrization of the input image in terms of the relative
                color path.

        """
        if isinstance(image, np.ndarray):
            return self._parametrize_colors(image, self.relative_colors)
        if isinstance(image, darsia.Image):
            return darsia.ScalarImage(
                self._parametrize_colors(image.img, self.relative_colors),
                **image.metadata(),
            )

    def to_dict(self) -> dict:
        """Convert the color path to a dictionary representation.

        Returns:
            dict: Dictionary representation of the color path.

        """
        return {
            "colors": [c.tolist() for c in self.colors],
            "base_color": self.base_color.tolist(),
            "relative_colors": [c.tolist() for c in self.relative_colors],
            "values": self.values,
            "mode": self.mode,
        }

    def save(self, path: Path) -> None:
        """Save the color path to a file.

        Args:
            path (Path): The path to the file where the color path should be saved.

        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    def load(self, path: Path) -> None:
        """Load the color path from a file.

        Args:
            path (Path): The path to the file from which the color path should be loaded.

        """
        with open(path, "r") as f:
            data = json.load(f)
            self.colors = [np.array(c) for c in data["colors"]]
            self.base_color = np.array(data["base_color"])
            self.relative_colors = [np.array(c) for c in data["relative_colors"]]
            self.values = data["values"]
            self.mode = data["mode"]
            self.num_segments = len(self.colors) - 1
            self.sort()
            logger.info(f"Loaded color path from {path}.")
