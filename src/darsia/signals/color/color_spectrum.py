"""Module defining the ColorSpectrum data structure."""

import numpy as np
from dataclasses import dataclass
import darsia


@dataclass
class ColorSpectrum:
    """Data structure to hold a color spectrum."""

    base_color: np.ndarray
    """Reference color for the spectrum."""
    spectrum: np.ndarray
    """Distribution of colors in the spectrum."""
    histogram: np.ndarray
    """Histogram of color occurrences."""
    color_range: darsia.ColorRange
    """Color range associated with the spectrum."""

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the shape of the spectrum."""
        return self.spectrum.shape

    @property
    def relative_colors(self) -> np.ndarray:
        """Compute the effective color spectrum as colors within the color range.

        Returns:
            np.ndarray: Effective color spectrum.

        """
        points = np.where(self.spectrum)
        relative_color_components = []
        for i in range(3):
            relative_color_components.append(
                points[i] / (self.spectrum.shape[i] - 1) * self.color_range.extent[i]
                + self.color_range.min_color[i]
            )
        relative_colors = np.vstack(relative_color_components).T
        return relative_colors

    @property
    def absolute_colors(self) -> np.ndarray:
        """Compute the absolute color spectrum.

        Returns:
            np.ndarray: Absolute color spectrum.

        """
        return self.base_color + self.relative_colors
