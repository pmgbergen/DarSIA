"""Piecewise linear color path in RGB space."""

import json
import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import darsia
from .utils import get_mean_color

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
        colors: list[np.ndarray] | None = None,
        base_color: np.ndarray | None = None,
        relative_colors: list[np.ndarray] | None = None,
        mode: Literal["rgb", "lab", "hcl"] = "rgb",
        name: str = "ColorPath",
    ) -> None:
        """Color path.

        Args:
            colors: Absolute colors in RGB space, defining the color path. Defaults
                to a simple grayscale path from black to white.
            base_color: Base color in RGB space, used as the first color in the path.
            relative_colors: Relative colors in RGB space wrt. the base color,
                defining the color path. If provided, `colors` must be `None`.
            mode: Color space to use for interpolation in between colors.
                Defaults to "rgb".
            name: Name of the color path.

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
            assert base_color is not None
            self.base_color = base_color
            self.colors = [self.base_color + c for c in relative_colors]

        # Two useful parametrizations of the color path with values between 0 and 1:
        # - Relative distances between the colors in the path.
        # - Equidistant values between 0, 1/n, 2/n, ..., 1.
        self.relative_distances: list[float] = self._compute_relative_distances()
        """Relative distances between the colors in the path."""

        self.equidistant_distances: list[float] = self._compute_equidistant_distances()
        """Equidistant distance between 0 and 1 for each color in the path."""

        self.num_segments = len(self.colors) - 1
        """Number of segments in the color path."""

        self.mode = mode
        """Color space to use for interpolation in between colors."""

        self.name = name
        """Name of the color path."""

    def __str__(self) -> str:
        """String representation of the color path."""
        return (
            f"ColorPath with {self.num_segments} segments, "
            f"colors {self.colors}, "
            f"base color {self.base_color}, "
            f"relative colors {self.relative_colors}, "
            f"relative distances {self.relative_distances}, "
            f"equidistant distances {self.equidistant_distances}, "
            f"mode {self.mode}, "
            f"name {self.name}."
        )

    def __repr__(self) -> str:
        """String representation of the color path."""
        return (
            f"ColorPath("
            f"colors: {self.colors}, "
            f"base_color: {self.base_color}, "
            f"relative_colors: {self.relative_colors}, "
            f"relative_distances: {self.relative_distances}, "
            f"equidistant_distances: {self.equidistant_distances}, "
            f"mode: {self.mode}, "
            f"name: {self.name}"
            f")"
        )

    def _compute_relative_distances(self) -> list[float]:
        """Compute relative distances between the colors in the path.

        Returns:
            list[float]: Relative distances between the colors in the path.

        """
        distances = [
            np.linalg.norm(self.relative_colors[i] - self.relative_colors[i - 1])
            for i in range(1, len(self.relative_colors))
        ]
        relative_distances = (
            np.cumsum([0.0] + distances) / (sum(distances) if sum(distances) > 0 else 1)
        ).tolist()
        return relative_distances

    def _compute_equidistant_distances(self) -> list[float]:
        """Compute equidistant distances between the colors in the path.

        Returns:
            list[float]: Equidistant distances between the colors in the path.
        """
        return np.linspace(0.0, 1.0, len(self.colors)).tolist()

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
                (v - min(self.relative_distances))
                / (max(self.relative_distances) - min(self.relative_distances))
                for v in self.relative_distances
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

    def show_cmap(self) -> None:
        """Visualize the color path as a colormap."""
        cmap = self.get_color_map(n_colors=256)
        plt.imshow([np.arange(256)], aspect="auto", cmap=cmap)
        plt.axis("off")
        plt.show()

    def show_path(
        self,
        name: str = "",
        directory: Path | None = None,
        delay: bool = False,
        **kwargs,
    ) -> None:
        """Visualize the color path in RGB space."""
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111, projection="3d")

        # Plot all significant colors
        if "relative_colors" in kwargs and "colors" in kwargs:
            relative_colors = kwargs["relative_colors"]
            colors = kwargs["colors"]
            ax.scatter(
                relative_colors[:, 0],
                relative_colors[:, 1],
                relative_colors[:, 2],
                c=np.clip(colors, 0, 1),
                s=10,
                alpha=0.5,
            )

        # Plot key colors and connecting lines
        ax.plot(*np.array(self.relative_colors).T, c="black", linewidth=2)
        ax.scatter(
            *np.array(self.relative_colors).T,
            c=np.clip(self.colors, 0, 1),
            s=100,
        )

        # Plot connecting lines for all points along the sorted embedding
        ax.plot(*np.array(self.relative_colors).T, c="gray", linewidth=1, alpha=0.5)

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
        if not delay:
            plt.show()

    def to_dict(self) -> dict:
        """Convert the color path to a dictionary representation.

        Returns:
            dict: Dictionary representation of the color path.

        """
        return {
            "colors": [c.tolist() for c in self.colors],
            "base_color": self.base_color.tolist(),
            "relative_colors": [c.tolist() for c in self.relative_colors],
            "relative_distances": self.relative_distances,
            "equidistant_distances": self.equidistant_distances,
            "mode": self.mode,
            "name": self.name,
        }

    @classmethod
    def from_dict(self, data: dict) -> "ColorPath":
        """Create a ColorPath instance from a dictionary representation.

        Args:
            data (dict): Dictionary representation of the color path.

        """
        color_path = ColorPath(
            base_color=np.array(data["base_color"]),
            relative_colors=[np.array(c) for c in data["relative_colors"]],
            mode=data["mode"],
            name=data["name"],
        )
        color_path.colors = [np.array(c) for c in data["colors"]]
        return color_path

    def save(self, path: Path) -> None:
        """Save the color path to a file.

        Args:
            path (Path): The path to the file where the color path should be saved.

        """
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(self.to_dict(), f)
        logger.info(f"Saved color path to {path}.")

    @classmethod
    def load(cls, path: Path) -> "ColorPath":
        """Load the color path from a file.

        Args:
            path (Path): The path to the file from which the color path should be loaded.

        """
        with open(path.with_suffix(".json"), "r") as f:
            data = json.load(f)
        color_path = cls.from_dict(data)
        logger.info(f"Loaded color path from {path}.")
        return color_path

        # self.relative_colors = [np.array(c) for c in data["relative_colors"]]
        # self.relative_distances = self._compute_relative_distances()
        # self.equidistant_distances = self._compute_equidistant_distances()
        # self.mode = data["mode"]
        # self.num_segments = len(self.colors) - 1
        # self.name = data["name"]
        # import time

        # tic = time.time()
        ## Create discrete optimization arrays
        # (
        #    self.discrete_equidistant_distances,
        #    self.discrete_equidistant_absolute_colors,
        #    self.discrete_equidistant_relative_colors,
        # ) = self._setup_discrete_optimization(mode="equidistant")
        # (
        #    self.discrete_relative_distances,
        #    self.discrete_relative_absolute_colors,
        #    self.discrete_relative_relative_colors,
        # ) = self._setup_discrete_optimization(mode="relative")
        # print(f"ColorPath setup took {time.time() - tic:.2f} seconds.")

    def refine(
        self,
        num_segments: int,
        distance_to_left: float | None = None,
        distance_to_right: float | None = None,
        mode: Literal["relative", "equidistant"] = "relative",
    ) -> "ColorPath":
        """Redefine the color path with a given number of segments.

        Args:
            num_segments: Number of segments for the refined color path.
            distance_to_left: Value to extend the color path to the left (inter).
            distance_to_right: Value to extend the color path to the right.

        Returns:
            ColorPath: Refined color path with the specified number of segments.

        """

        # Define new relative distances
        num_colors = num_segments + 1
        distances = np.linspace(0, 1, num_colors)
        if distance_to_left:
            distances = np.hstack((distance_to_left, distances))
        if distance_to_right:
            distances = np.hstack((distances, distance_to_right))

        # Interpret corresponding relative colors
        relative_colors = self.interpret(
            distances,
            color_mode=darsia.ColorMode.RELATIVE,
            mode=mode,
        )

        # Create new color path
        return ColorPath(
            base_color=self.base_color,
            relative_colors=relative_colors,
            mode=self.mode,
            name=self.name,
        )

    # ! ---- PARAMETRIZATION OF COLORS ----

    @darsia.timing_decorator
    def fit(
        self,
        colors: np.ndarray,
        color_mode: darsia.ColorMode,
        mode: Literal["equidistant", "relative"] = "relative",
    ) -> np.ndarray:
        """Parametrize colors in terms of the color path.

        Apply brute-force minimization to find the closest color representation
        on the path for each pixel in the image.

        Args:
            colors: Colors to be interpreted.

        Returns:
            np.ndarray: Parametrization of the input image in terms of the color path.

        """
        # Fetch the right supports
        supports = (
            self.colors
            if color_mode == darsia.ColorMode.ABSOLUTE
            else self.relative_colors
        )

        # Fetch the distances along the path
        if mode == "equidistant":
            distances = self.equidistant_distances
        elif mode == "relative":
            distances = self.relative_distances
        else:
            raise ValueError(f"Unknown mode '{mode}' for color path parametrization.")

        # Find the best-fit scalar interpretation for each segment
        interpretations = []
        for segment in range(self.num_segments):
            scalar_interpretation = distances[segment] + (
                distances[segment + 1] - distances[segment]
            ) * np.tensordot(
                colors - supports[segment],
                supports[segment + 1] - supports[segment],
                axes=([-1], [0]),
            ) / np.dot(
                supports[segment + 1] - supports[segment],
                supports[segment + 1] - supports[segment],
            )
            if segment == 0:
                scalar_interpretation = np.clip(
                    scalar_interpretation,
                    None,
                    distances[segment + 1],
                )
            elif segment == self.num_segments - 1:
                scalar_interpretation = np.clip(
                    scalar_interpretation,
                    distances[segment],
                    None,
                )
            else:
                scalar_interpretation = np.clip(
                    scalar_interpretation,
                    distances[segment],
                    distances[segment + 1],
                )
            interpretations.append(scalar_interpretation)

        # Convert each segment to its color interpretation
        shape = colors.shape[:-1] + (3,)
        color_interpretations = [
            supports[segment]
            + np.outer(
                (interpretations[segment] - distances[segment])
                / (distances[segment + 1] - distances[segment]),
                supports[segment + 1] - supports[segment],
            ).reshape(shape)
            for segment in range(self.num_segments)
        ]

        # Optimized method - find closest segment without storing all distances
        closest_segment = np.zeros(colors.shape[:-1], dtype=np.int32)
        min_distances = np.full(colors.shape[:-1], np.inf)

        for segment in range(self.num_segments):
            distances = np.linalg.norm(
                colors - color_interpretations[segment], ord=1, axis=-1
            )
            mask = distances < min_distances
            closest_segment[mask] = segment
            min_distances[mask] = distances[mask]

        # Finalize minimization by taking the best-fit interpretation
        best_fit_interpretation = np.zeros(colors.shape[:-1])
        for segment in range(self.num_segments):
            mask = closest_segment == segment
            best_fit_interpretation[mask] = interpretations[segment][mask]

        # Deal with nan values...
        if np.any(np.isnan(best_fit_interpretation)):
            logger.info(
                f"Some pixels could not be interpreted by the color path "
                f"'{self.name}'. Setting their values to 0."
            )
            best_fit_interpretation[np.isnan(best_fit_interpretation)] = 0.0

        return best_fit_interpretation

    def interpret(
        self,
        parameters: np.ndarray,
        color_mode: darsia.ColorMode,
        mode: Literal["equidistant", "relative"] = "relative",
    ) -> np.ndarray:
        """Interpret parameters in terms of the color path.

        Args:
            parameters: Parameters to be interpreted.
            color_mode: Color mode to use for interpretation.
            mode: Mode to use for interpretation.

        Returns:
            np.ndarray: Interpreted colors.

        """
        # Fetch the right supports
        supports = (
            self.colors
            if color_mode == darsia.ColorMode.ABSOLUTE
            else self.relative_colors
        )

        # Fetch the distances along the path
        if mode == "equidistant":
            distances = self.equidistant_distances
        elif mode == "relative":
            distances = self.relative_distances
        else:
            raise ValueError(f"Unknown mode '{mode}' for color path interpretation.")

        # Find the right segment for each pixel
        shape = parameters.shape + (3,)
        interpretations = np.zeros(shape)
        for segment in range(self.num_segments):
            if segment == 0:
                mask = parameters <= distances[segment + 1]
            elif segment == self.num_segments - 1:
                mask = parameters >= distances[segment]
            else:
                mask = (parameters >= distances[segment]) & (
                    parameters <= distances[segment + 1]
                )
            ratio_in_segment = (parameters[mask] - distances[segment]) / (
                distances[segment + 1] - distances[segment]
            )
            interpretations[mask] = supports[segment] + np.outer(
                ratio_in_segment,
                supports[segment + 1] - supports[segment],
            ).reshape((-1, 3))

        return interpretations


# ! ---- INTERACTIVE COLOR PATH DEFINITION ----


def define_color_path(image: darsia.Image, mask: darsia.Image) -> ColorPath:
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
        mean_color = get_mean_color(image, mask=boxed_mask)
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
