"""Module for modifying labels and extracting masks."""

from typing import Optional

import numpy as np

import darsia


class LabelsMergeAssistant:
    def __init__(
        self, labels: darsia.Image, background: Optional[darsia.Image] = None, **kwargs
    ) -> None:
        # Set name for titles in plots
        self.name = "Labels merge assistant"

        self.point_selection_assistant = darsia.PointSelectionAssistant(
            img=labels, background=background, **kwargs
        )

        # Cache input labels
        self.labels = labels

    def __call__(self):
        # Initialize return object
        new_labels = self.labels.copy()

        # Extract mask corresponding to chosen points
        mask = self.extract_mask()

        # Assign lowest label value to masked area
        labels = np.unique(self.labels.img[mask])
        if len(labels) > 0:
            min_label = np.min(labels)
            new_labels.img[mask] = min_label

        return new_labels

    def extract_mask(self) -> np.ndarray:
        # Identify points characterizing different regions to be merged
        points = self.point_selection_assistant.__call__()

        # Initialize return object
        mask = np.zeros_like(self.labels.img, dtype=bool)

        if len(points) > 0:
            # Points provided in col, row format.
            points = np.fliplr(points)

            # Identify corresponding labels
            labels = np.unique([self.labels.img[p[0], p[1]] for p in points])

            # Mask detected labels
            for label in labels.tolist():
                mask = np.logical_or(mask, np.isclose(self.labels.img, label))

        return mask
