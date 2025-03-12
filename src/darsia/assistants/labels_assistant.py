"""Labels assistant built from modules."""

from typing import Optional
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np

import darsia


class LabelsAssistantMenu(darsia.BaseAssistant):
    """Module for LabelsAssistant to choose what to do with labels.

    Purpose is to forward which module to use to LabelsAssistant. Detailed instructions
    are printed to screen.

    """

    def __init__(
        self, img: darsia.Image, background: Optional[darsia.Image] = None, **kwargs
    ) -> None:
        """Initialize module.

        Args:
            img (darsia.Image): input image
            background (Optional[darsia.Image]): background image

        """
        self.name = "Labels assistant menu"
        """Name of the module."""

        # Initialize base assistant with reference to current labels
        super().__init__(img=img, background=background, **kwargs)

        # Print instructions
        if kwargs.get("print_instructions", True):
            self._print_instructions()

    def _print_instructions(self) -> None:
        """Print instructions."""

        print("Welcome to the labels assistant.")
        print("Please choose one of the following options:")
        print("  - 'shift+s': segment labels")
        print("  - 'm': merge labels")
        print("  - 'p': pick labels")
        print("  - 'r': refine labels")
        print("  - 'i': print info")
        print("  - 'b': toggle background")
        print("  - 'shift+m': adapt monochromatic background image (default: gray)")
        print("  - 'u': undo")
        print("  - 'escape': reset labels to input")
        print("  - 'q': quit/abort\n")

    def _on_key_press(self, event) -> None:
        """Finalize selection if 'enter' is pressed, and reset containers if 'escape'
        is pressed.

        Args:
            event: key press event

        """
        if self.verbosity:
            print(f"Current key: {event}")

        # Track events
        if event.key == "S":
            self.action = "segment"
            plt.close(self.fig)
        elif event.key == "m":
            self.action = "merge"
            plt.close(self.fig)
        elif event.key == "p":
            self.action = "pick"
            plt.close(self.fig)
        elif event.key == "r":
            self.action = "refine"
            plt.close(self.fig)
        elif event.key == "escape":
            self.action = "reset"
            plt.close(self.fig)
        elif event.key == "b":
            self.action = "toggle_background"
            plt.close(self.fig)
        elif event.key == "M":
            self.action = "monochromatic"
            plt.close(self.fig)
        elif event.key == "u":
            self.action = "undo"
            plt.close(self.fig)
        elif event.key == "i":
            self.action = "info"
            plt.close(self.fig)
        elif event.key == "q":
            self.action = "quit"
            plt.close(self.fig)

    def __call__(self) -> str:
        """Call the assistant.

        Returns:
            str: next action to be executed by LabelsAssistant.

        """
        self.action = None
        super().__call__()
        return self.action


class LabelsSegmentAssistant:
    """Module for LabelsAssistant to segment labels."""

    def __init__(
        self,
        labels: Optional[darsia.Image],
        background: darsia.Image,
        mask: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """Initialize module.

        Args:
            labels (Optional[darsia.Image]): input labels
            background (darsia.Image): background image
            mask (Optional[np.ndarray]): mask to be used for segmentation

        """
        self.labels = labels
        """Input labels."""
        self.background = background
        """Background image."""
        assert self.background is not None, "Background image required."
        self.mask = mask
        """Mask to be used for segmentation."""
        self.verbosity = kwargs.get("verbosity", False)
        """Verbosity flag."""

        # Purely for visualization purposes
        self.roi = None
        """Region of interest."""

        # Safety checks
        if mask is not None:
            assert isinstance(mask, np.ndarray), "Mask must be a numpy array."
            assert mask.dtype == bool, "Mask must be a boolean array."
            self.roi = darsia.Image(
                np.zeros_like(
                    background.img if background.scalar else background.img[:, :, 0],
                    dtype=int,
                ),
                **self.background.metadata(),
            )
            self.roi.img[mask] = 1

    def __call__(self) -> darsia.Image:
        """Call the assistant.

        Returns:
            darsia.Image: new labels through segmentation

        """
        point_selection_assistant = darsia.PointSelectionAssistant(
            name="Pick characteristic points for segmentation.",
            img=self.roi,
            background=self.background,
            verbosity=self.verbosity,
        )
        points = point_selection_assistant()

        new_labels = darsia.segment(
            self.background,
            markers_method="supervised",
            edges_method="scharr",
            marker_points=points,
            mask=self.mask,
            region_size=2,
            clean_up=False,
        )

        if self.mask is not None:
            # Determine a list of possible label ids - more than required.
            # Consider all label ids marked by the mask, and additional ids extending
            # the current label ids. Use a sufficient number of ids.
            num_detected_labels = np.unique(new_labels.img[self.mask]).size
            new_mapped_labels = np.concatenate(
                (
                    np.unique(self.labels.img[self.mask])[:num_detected_labels],
                    self.labels.img.max() + 1 + np.arange(max(0, num_detected_labels)),
                )
            )

            # Assign new label ids
            for i, new_label in enumerate(np.unique(new_labels.img[self.mask])):
                label_mask = np.isclose(new_labels.img, new_label)
                new_labels.img[label_mask] = new_mapped_labels[i]

            # Use old values in non-marked area
            new_labels.img[~self.mask] = self.labels.img[~self.mask]

        return new_labels


class MonochromaticAssistant(darsia.BaseAssistant):
    """Assistant to choose monochromatic image."""

    def __init__(self, img: darsia.Image, **kwargs) -> None:
        """Initialize module.

        Args:
            img (darsia.Image): input image

        """
        self.name = "Monochromatic assistant"
        """Name of the module."""
        self.input_img = img
        """Input image."""
        self.img = img.to_monochromatic("gray")
        """Monochromatic image."""

        # Initialize base assistant with reference to monochromatic image
        super().__init__(img=self.img, **kwargs)

        self.finalized = False
        """Flag indicating whether the assistant has been finalized."""

    def _print_instructions(self) -> None:
        """Print instructions."""
        print("")
        print("Welcome to the monochromatic assistant.")
        print("Please choose one of the following options:")
        print("  - 'g': gray")
        print("  - 'R': red")
        print("  - 'G': green")
        print("  - 'B': blue")
        print("  - 'H': hue")
        print("  - 'S': saturation")
        print("  - 'V': value")
        print("  - 'q/enter/escape': quit\n")

    def _on_key_press(self, event) -> None:
        """Key press event handler.

        Args:
            event: key press event

        """
        if self.verbosity:
            print(f"Current key: {event}")

        # Track events
        if event.key == "g":
            self.img = self.input_img.to_monochromatic("gray")
        elif event.key == "R":
            self.img = self.input_img.to_monochromatic("red")
        elif event.key == "G":
            self.img = self.input_img.to_monochromatic("green")
        elif event.key == "B":
            self.img = self.input_img.to_monochromatic("blue")
        elif event.key == "H":
            self.img = self.input_img.to_monochromatic("hue")
        elif event.key == "S":
            self.img = self.input_img.to_monochromatic("saturation")
        elif event.key == "V":
            self.img = self.input_img.to_monochromatic("value")
        elif event.key in ["q", "enter", "escape"]:
            self.finalized = True
            plt.close(self.fig)

        # Replot
        if event.key in ["g", "R", "G", "B", "H", "S", "V"]:
            self._plot_2d()

    def __call__(self) -> darsia.Image:
        """Call the assistant.

        Returns:
            darsia.Image: monochromatic image

        """
        self._print_instructions()
        plt.ion()
        super().__call__()
        plt.ioff()
        return self.img


class LabelsMaskSelectionAssistant:
    """Module for LabelsAssistant to select a mask from labels."""

    def __init__(
        self, labels: darsia.Image, background: Optional[darsia.Image] = None, **kwargs
    ) -> None:
        """Initialize module.

        Args:
            labels (darsia.Image): input labels
            background (Optional[darsia.Image]): background image

        """
        self.name = "Labels mask selection assistant"
        """Name of the module."""
        self.labels = labels
        """Input labels."""
        self.background = background
        """Background image."""
        self.verbosity = kwargs.get("verbosity", False)
        """Verbosity flag."""

    def __call__(self) -> np.ndarray:
        """Call the assistant.

        Returns:
            np.ndarray: mask

        """
        # Identify points characterizing different regions to be merged
        point_selection_assistant = darsia.PointSelectionAssistant(
            name="Mark labels.",
            img=self.labels,
            background=self.background,
            verbosity=self.verbosity,
        )
        points = point_selection_assistant()

        # Initialize return object
        mask = np.zeros_like(self.labels.img, dtype=bool)

        if points is not None and len(points) > 0:
            # Identify corresponding labels
            labels = np.unique([self.labels.img[p[0], p[1]] for p in points])

            # Mask detected labels
            for label in labels.tolist():
                mask = np.logical_or(mask, np.isclose(self.labels.img, label))

        return mask


class LabelsPickAssistant:
    """Module for LabelsAssistant to pick labels."""

    def __init__(
        self, labels: darsia.Image, background: darsia.Image, **kwargs
    ) -> None:
        """Initialize module.

        Args:
            labels (darsia.Image): input labels
            background (darsia.Image): background image

        """
        self.labels = labels
        """Input labels."""
        self.background = background
        """Background image."""
        self.verbosity = kwargs.get("verbosity", False)
        """Verbosity flag."""

    def __call__(self) -> darsia.Image:
        """Call the assistant.

        Returns:
            darsia.Image: selected labels

        """
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
    """Module for LabelsAssistant to merge labels."""

    def __init__(
        self, labels: darsia.Image, background: Optional[darsia.Image] = None, **kwargs
    ) -> None:
        """Initialize module.

        Args:
            labels (darsia.Image): input labels
            background (Optional[darsia.Image]): background image

        """
        self.name = "Labels merge assistant"
        """Name of the module."""
        self.labels = labels
        """Input labels."""
        self.background = background
        """Background image."""
        self.verbosity = kwargs.get("verbosity", False)
        """Verbosity flag."""

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
    """Assistant to modify labels."""

    def __init__(
        self,
        labels: Optional[darsia.Image] = None,
        background: Optional[darsia.Image] = None,
        **kwargs,
    ) -> None:
        """Initialize module.

        Args:
            labels (Optional[darsia.Image]): input labels
            background (Optional[darsia.Image]): background image

        """
        self.name = "Labels assistant"
        """Name of the assistant."""
        self.background = background
        """Background image."""
        self.monochromatic_background = None
        """Monochromatic background image."""
        if isinstance(self.background, darsia.OpticalImage):
            self.monochromatic_background = self.background.to_monochromatic("gray")
        self.cache_background = None
        """Cache for background image."""
        if labels is None:
            assert self.background is not None, (
                "Background image required to initialize empty labels."
            )
            self.labels = darsia.Image(
                np.zeros_like(self.background.img, dtype=int),
                **self.background.metadata(),
            )
            """Input labels."""
        else:
            self.labels = labels.copy()
        self.previous_labels = self.labels.copy()
        """Reference to previous labels."""
        self.current_labels = self.labels.copy()
        """Reference to current labels."""
        self.next_action = None
        """Next module to be called."""
        self.finalized = False
        """Flag indicating whether the assistant has been finalized."""
        self.verbosity = kwargs.get("verbosity", False)
        """Verbosity flag."""
        self.first_call = True
        """Flag indicating whether the assistant is called for the first time."""

    def __call__(self) -> darsia.Image:
        """Call assistant.

        Always call the menu first. Then call the next module. Repeat until the
        assistant is finalized. The instructions are printed only for the first call.

        Returns:
            darsia.Image: current labels

        """
        if not self.finalized:
            # Call menu
            self._call_menu()

            # Call next module
            self._call_next_action()

            # Repeat
            self.__call__()

        self.first_call = False

        # Return
        return self.current_labels

    def _call_menu(self) -> None:
        """Call menu."""

        # Initialize menu
        self.menu = darsia.LabelsAssistantMenu(
            img=self.current_labels,
            background=self.background,
            verbosity=self.verbosity,
            print_instructions=self.first_call,
        )

        # Call menu module
        self.next_action = self.menu()

    def _call_next_action(self) -> None:
        """Call next module."""

        # Fix previous labels if next action is modifying the labels
        if self.next_action in ["segment", "merge", "refine"]:
            self.previous_labels = self.current_labels.copy()

        # Apply action
        if self.next_action == "segment":
            self._call_segment_module()
        elif self.next_action == "pick":
            self._call_pick_module()
        elif self.next_action == "merge":
            self._call_merge_module()
        elif self.next_action == "refine":
            self._call_refine_module()
        elif self.next_action == "reset":
            self.current_labels = self.labels.copy()
        elif self.next_action == "toggle_background":
            self._toggle_background()
        elif self.next_action == "monochromatic":
            self._call_monochromatic_module()
        elif self.next_action == "undo":
            self.current_labels = self.previous_labels.copy()
        elif self.next_action == "info":
            self._print_info()
        elif self.next_action == "quit":
            self.finalized = True
        elif self.next_action is None:
            pass
        else:
            raise ValueError(f"Next module {self.next_action} not recognized.")

        if self.verbosity:
            print(f"Next module: {self.next_action}")

    def _call_segment_module(self) -> None:
        """Call segment module."""

        if self.background is None:
            assert self.cache_background is not None, "No background image available."
            self.background = self.cache_background.copy()
            self.cache_background = None

        segment_assistant = darsia.LabelsSegmentAssistant(
            labels=None,
            background=self.monochromatic_background,
            verbosity=self.verbosity,
        )
        self.current_labels = segment_assistant()

    def _call_refine_module(self) -> None:
        """Call refine module."""

        mask_selection_assistant = darsia.LabelsMaskSelectionAssistant(
            labels=self.current_labels,
            background=self.background,
            verbosity=self.verbosity,
        )
        mask = mask_selection_assistant()

        if self.background is None:
            assert self.cache_background is not None, "No background image available."
            self.background = self.cache_background.copy()
            self.cache_background = None

        segment_assistant = darsia.LabelsSegmentAssistant(
            labels=self.current_labels,
            background=self.monochromatic_background,
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

    def _print_info(self) -> None:
        """Print out information about the assistant."""

        print("The current labels:")
        print(np.unique(self.current_labels.img).tolist())

        # Plot current labels
        plt.figure("Current labels")
        plt.imshow(self.current_labels.img)
        plt.show()

    def _toggle_background(self) -> None:
        """Toggle background."""

        if self.background is None and self.cache_background is None:
            warn("No background image available.")
        else:
            if self.background is None:
                self.background = self.cache_background.copy()
                self.cache_background = None
            else:
                self.cache_background = self.background.copy()
                self.background = None

    def _call_monochromatic_module(self) -> None:
        """Call monochromatic module."""

        if isinstance(self.background, darsia.OpticalImage):
            monochromatic_assistant = darsia.MonochromaticAssistant(
                img=self.background,
                verbosity=self.verbosity,
            )
            self.monochromatic_background = monochromatic_assistant()
            # For internal processes, values are expected to be between -1 and 1
            if self.monochromatic_background.dtype in [float, np.float32, np.float64]:
                self.monochromatic_background.img = np.clip(
                    self.monochromatic_background.img, -1, 1
                )
