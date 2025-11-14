"""Module defining the ColorSpectrum data structure."""

import numpy as np
from dataclasses import dataclass
import darsia
from pathlib import Path
import json

import logging

logger = logging.getLogger(__name__)


@dataclass
class ColorSpectrum:
    """Data structure to hold a (discrete) color spectrum."""

    base_color: np.ndarray
    """Reference color for the spectrum."""
    spectrum: np.ndarray
    """Distribution of colors in the spectrum."""
    histogram: np.ndarray
    """Histogram of color occurrences."""
    color_range: darsia.ColorRange
    """Color range associated with the spectrum."""

    def __repr__(self) -> str:
        return (
            f"ColorSpectrum(base_color={self.base_color}, "
            f"spectrum_shape={self.spectrum.shape}, "
            f"histogram_shape={self.histogram.shape}, "
            f"active colors={np.sum(self.spectrum)}, "
            f"color_range={self.color_range})"
        )

    @property
    def color_mode(self) -> darsia.ColorMode:
        """Get the color mode of the spectrum."""
        return self.color_range.color_mode

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the shape of the spectrum."""
        assert self.spectrum.shape == self.color_range.shape
        return self.spectrum.shape

    @property
    def colors(self) -> np.ndarray:
        """Compute the absolute color spectrum.

        Returns:
            np.ndarray: Absolute color spectrum.

        """
        return self.base_color + self.relative_colors

    @property
    def relative_colors(self) -> np.ndarray:
        """Compute the effective color spectrum as colors within the color range.

        Returns:
            np.ndarray: Effective color spectrum.

        """
        # Convert discrete spectrum indices to colors
        points = np.where(self.spectrum)
        color_components = []
        for i in range(3):
            color_components.append(
                points[i] / (self.spectrum.shape[i] - 1) * self.color_range.extent[i]
                + self.color_range.min_color[i]
            )
        colors = np.vstack(color_components).T

        # Return relative or absolute colors based on color mode
        if self.color_mode == darsia.ColorMode.RELATIVE:
            return colors
        elif self.color_mode == darsia.ColorMode.ABSOLUTE:
            return colors - self.base_color

    def distance(self, color: np.ndarray) -> np.ndarray | float:
        """Compute the (minimum) distance between a color and the spectrum.

        Args:
            color (np.ndarray): The color to compare against the spectrum.

        Returns:
            The computed distance.

        """
        # Compute the distance as the minimum distance to any color in the spectrum
        if len(color.ravel().shape) == 1:
            return np.min(np.linalg.norm(self.colors - color, axis=1))
        else:
            return np.array(
                [np.min(np.linalg.norm(self.colors - c, axis=1)) for c in color]
            )

    def save(self, file_path: Path) -> None:
        """Save the color spectrum to a file.

        Args:
            file_path (Path): The path to the file where the color spectrum will be saved.

        """
        data = {
            "base_color": self.base_color.tolist(),
            "spectrum": self.spectrum.tolist(),
            "histogram": self.histogram.tolist(),
            "color_range": self.color_range.to_dict(),
        }
        with open(file_path.with_suffix(".json"), "w") as f:
            json.dump(data, f)
        logger.info("Saved color spectrum to %s", file_path.with_suffix(".json"))

    @classmethod
    def load(cls, file_path: Path) -> "ColorSpectrum":
        """Load the color spectrum from a file.

        Args:
            file_path (Path): The path to the file from which the color spectrum will be loaded.

        Returns:
            ColorSpectrum: The loaded color spectrum.

        """
        with open(file_path.with_suffix(".json"), "r") as f:
            data = json.load(f)
        color_spectrum = cls(
            base_color=np.array(data["base_color"]),
            spectrum=np.array(data["spectrum"]),
            histogram=np.array(data["histogram"]),
            color_range=darsia.ColorRange.load_from_dict(data["color_range"]),
        )
        return color_spectrum
