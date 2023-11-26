""""Labels assistant built from modules."""

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

import darsia


class LabelsAssistantMenu(darsia.BaseAssistant):
    """Module for LabelsAssistant to choose what to do with labels.

    Purpose is to forward which module to use to LabelsAssistant. Choices:
    - Merge labels
    - Choose labels
    - Segment labels

    """

    def __init__(
        self, img: darsia.Image, background: Optional[darsia.Image] = None, **kwargs
    ) -> None:
        # TODO decide how to handle kwargs
        """Initialize module.

        Args:
            img (darsia.Image): input image
            background (Optional[darsia.Image]): background image

        """
        self.name = "Labels assistant menu"
        """Name of the module."""

        # Initialize base assistant with reference to current labels
        super().__init__(img=img, background=background, block=True, **kwargs)

        # Print instructions
        self._print_instructions()

    def _print_instructions(self) -> None:
        """Print instructions."""

        print("Welcome to the labels assistant.")
        print("Please choose one of the following options:")
        print("  - 's': segment labels")
        print("  - 'm': merge labels")
        print("  - 'p': pick labels")
        print("  - 'r': refine labels")
        print("  - 'escape': reset labels to input")
        print("  - 'q': quit/abort\n")

    def _on_key_press(self, event) -> None:
        """Finalize selection if 'enter' is pressed, and reset containers if 'escape'
        is pressed.

        Args:
            event: key press event

        """
        super()._on_key_press(event)

        # Additional events
        if event.key == "s":
            self.action = "segment"
            plt.close(self.fig)
        if event.key == "m":
            self.action = "merge"
            plt.close(self.fig)
        if event.key == "p":
            self.action = "pick"
            plt.close(self.fig)
        if event.key == "r":
            self.action = "refine"
            plt.close(self.fig)
        if event.key == "escape":
            self.action = "reset"
            plt.close(self.fig)
        if event.key == "q":
            self.action = "quit"
            plt.close(self.fig)

    def __call__(self) -> darsia.Image:
        """Call the assistant."""

        self.action = "quit"
        super().__call__()
        return self.action


class LabelsSegmentAssistant:
    def __init__(
        self,
        labels: darsia.Image,
        background: darsia.Image,
        mask: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        self.labels = labels  # If labels provided, ask which to keep
        self.background = background
        self.mask = mask
        self.verbosity = kwargs.get("verbosity", False)

        # Safety checks
        if mask is not None:
            assert isinstance(mask, np.ndarray), "Mask must be a numpy array."
            assert mask.dtype == bool, "Mask must be a boolean array."

    def __call__(self) -> darsia.Image:
        point_selection_assistant = darsia.PointSelectionAssistant(
            img=self.labels,
            background=self.background,
            block=True,
            verbosity=self.verbosity,
        )
        points = point_selection_assistant()

        labels = darsia.segment(
            self.background,
            markers_method="supervised",
            edges_method="scharr",
            marker_points=points,
            mask=self.mask,
            region_size=2,
        )

        if self.mask is not None:
            # Use old values
            old_labels = np.unique(self.labels.img[~self.mask])

            # Determine holes in the old labels
            missing_labels = np.setdiff1d(
                np.unique(self.labels.img), np.unique(self.labels.img[~self.mask])
            )
            num_missing_labels = missing_labels.size

            # Fetch new values and map them on consecutive integers into old labels
            unique_new_labels = np.unique(labels.img[self.mask])
            num_new_labels = unique_new_labels.size
            new_labels = np.arange(
                old_labels.max() + 1,
                old_labels.max() + 1 + max(num_new_labels - num_missing_labels, 0),
            )

            # Combine holes and new labels
            unique_mapped_labels = np.concatenate((missing_labels, new_labels))

            # Map new labels onto unique
            for i, new_label in enumerate(unique_new_labels):
                label_mask = np.isclose(labels.img, new_label)
                labels.img[label_mask] = unique_mapped_labels[i]

            # Use old values in non-marked area
            labels.img[~self.mask] = self.labels.img[~self.mask]

        return labels


class LabelsMaskSelectionAssistant:
    def __init__(
        self, labels: darsia.Image, background: Optional[darsia.Image] = None, **kwargs
    ) -> None:
        self.name = "Labels mask selection assistant"
        self.labels = labels
        self.background = background
        self.verbosity = kwargs.get("verbosity", False)

    def __call__(self) -> np.ndarray:
        # Identify points characterizing different regions to be merged
        point_selection_assistant = darsia.PointSelectionAssistant(
            img=self.labels,
            background=self.background,
            block=True,
            verbosity=self.verbosity,
        )
        points = point_selection_assistant()

        # Initialize return object
        mask = np.zeros_like(self.labels.img, dtype=bool)

        if points is not None and len(points) > 0:
            # Points provided in col, row format.
            points = np.fliplr(points)

            # Identify corresponding labels
            labels = np.unique([self.labels.img[p[0], p[1]] for p in points])

            # Mask detected labels
            for label in labels.tolist():
                mask = np.logical_or(mask, np.isclose(self.labels.img, label))

        return mask


class LabelsPickAssistant:
    def __init__(
        self, labels: darsia.Image, background: darsia.Image, **kwargs
    ) -> None:
        self.labels = labels
        self.background = background
        self.verbosity = kwargs.get("verbosity", False)

    def __call__(self) -> darsia.Image:
        # Extract mask corresponding to chosen points
        mask_selection_assistant = darsia.LabelsMaskSelectionAssistant(
            labels=self.labels,
            background=self.background,
            verbosity=self.verbosity,
        )
        mask = mask_selection_assistant()

        # Deactive not selected labels (effective merging)
        new_labels = self.labels.copy()
        new_labels.img[~mask] = 0

        return new_labels


class LabelsMergeAssistant:
    def __init__(
        self, labels: darsia.Image, background: Optional[darsia.Image] = None, **kwargs
    ) -> None:
        self.name = "Labels merge assistant"
        self.labels = labels
        self.background = background
        self.verbosity = kwargs.get("verbosity", False)

    def __call__(self):
        # Extract mask corresponding to chosen points
        mask_selection_assistant = darsia.LabelsMaskSelectionAssistant(
            labels=self.labels,
            background=self.background,
            verbosity=self.verbosity,
        )
        mask = mask_selection_assistant()

        # Assign lowest label value to masked area
        new_labels = self.labels.copy()
        new_labels.img[mask] = np.min(np.unique(self.labels.img[mask]))

        return new_labels


class LabelsAssistant:
    def __init__(
        self,
        labels: Optional[darsia.Image],
        background: Optional[darsia.Image] = None,
        **kwargs,
    ) -> None:
        self.name = "Labels assistant"
        """Name of the assistant."""
        self.labels = labels.copy()
        """Input labels."""
        self.current_labels = labels
        """Reference to current labels."""
        self.background = background
        """Background image."""
        self.next_module = None
        """Next module to be called."""
        self.finalized = False
        """Flag indicating whether the assistant has been finalized."""
        self.verbosity = kwargs.get("verbosity", False)

        # Initialize empty labels - require background then
        if self.labels is None:
            assert (
                self.background is not None
            ), "Background image required to initialize empty labels."
            self.labels = darsia.Image(
                np.zeros_like(self.background.img, dtype=int),
                **self.background.metadata,
            )

    def __call__(self):
        """Call assistant."""

        # Initialize
        if not self.finalized:
            # Call menu
            self._call_menu()

            # Call next module
            self._call_next_module()

            # Repeat
            self.__call__()

        # Return
        return self.current_labels

    def _call_menu(self) -> None:
        """Call menu."""

        # Initialize menu
        self.menu = darsia.LabelsAssistantMenu(
            img=self.current_labels,
            background=self.background,
            verbosity=self.verbosity,
        )

        # Call menu module
        self.next_module = self.menu()
        print("Next module:", self.next_module)

    def _call_next_module(self) -> None:
        """Call next module."""

        if self.next_module == "segment":
            self._call_segment_module()
        elif self.next_module == "pick":
            self._call_pick_module()
        elif self.next_module == "merge":
            self._call_merge_module()
        elif self.next_module == "refine":
            self._call_refine_module()
        elif self.next_module == "reset":
            self.current_labels = self.labels.copy()
        elif self.next_module == "quit":
            self.finalized = True
        else:
            raise ValueError(f"Next module {self.next_module} not recognized.")

    def _call_segment_module(self) -> None:
        """Call segment module."""

        segment_module = darsia.LabelsSegmentAssistant(
            # labels=self.current_labels,
            background=self.background,
            verbosity=self.verbosity,
        )
        self.current_labels = segment_module()

    def _call_refine_module(self) -> None:
        """Call refine module."""

        mask_selection_assistant = darsia.LabelsMaskSelectionAssistant(
            labels=self.current_labels,
            background=self.background,
            verbosity=self.verbosity,
        )
        mask = mask_selection_assistant()

        segment_assistant = darsia.LabelsSegmentAssistant(
            labels=self.current_labels,
            background=self.background,
            mask=mask,
            verbosity=self.verbosity,
        )
        self.current_labels = segment_assistant()

    def _call_pick_module(self) -> None:
        """Call pick module."""

        pick_assistant = darsia.LabelsPickAssistant(
            labels=self.current_labels,
            background=self.background,
            verbosity=self.verbosity,
        )
        self.current_labels = pick_assistant()

    def _call_merge_module(self) -> None:
        """Call merge module."""

        merge_assistant = darsia.LabelsMergeAssistant(
            labels=self.current_labels,
            background=self.background,
        )
        self.current_labels = merge_assistant()
