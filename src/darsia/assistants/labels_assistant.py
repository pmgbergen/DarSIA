"""Module for modifying labels and extracting masks."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

import darsia


class LabelsAssistant(darsia.PointSelectionAssistant):
    def __init__(
        self, img: darsia.Image, background: Optional[darsia.Image] = None, **kwargs
    ) -> None:
        super().__init__(img, **kwargs)

        # Set name for titles in plots
        self.name = "Labels assistant"

        # Cache background image for improved plotting
        self.background = background

    def _plot_2d(self) -> None:
        """Plot in 2d with interactive event handler."""
        if self.background is None:
            self._setup_plot_2d(self.img)
        else:
            self._setup_plot_2d(self.background)
            self._setup_plot_2d(self.img, new_figure=False, alpha=0.3)
        plt.show(block=True)

    def merge(self) -> darsia.Image:
        # Initialize return object
        new_label = self.img.copy()

        # Extract mask corresponding to chosen points
        mask = self.extract_mask()

        # Assign lowest label value to masked area
        labels = np.unique(self.img.img[mask])
        if len(labels) > 0:
            min_label = np.min(labels)
            new_label.img[mask] = min_label

        return new_label

    def extract_mask(self) -> np.ndarray:

        # Identify points characterizing different regions to be merged
        points = self.__call__()

        # Initialize return object
        mask = np.zeros_like(self.img.img, dtype=bool)

        if len(points) > 0:
            # Points provided in col, row format.
            points = np.fliplr(points)

            # Identify corresponding labels
            labels = np.unique([self.img.img[p[0], p[1]] for p in points])

            # Mask detected labels
            for label in labels.tolist():
                mask = np.logical_or(mask, np.isclose(self.img.img, label))

        return mask
