"""Module for point selection."""

from typing import Any, Optional
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np

import darsia


class PointSelectionAssistant(darsia.BaseAssistant):
    def __init__(self, img: darsia.Image, **kwargs) -> None:
        super().__init__(img, use_coordinates=False, **kwargs)

        # Only continue for 2d images
        assert self.img.space_dim == 2, "Only 2d images are supported."

        # Initialize containers
        self._reset()

        # Set name for titles in plots
        self.name = "Point selection assistant"
        """Name of assistant."""

        # Prepare output
        self.pts = None
        """Selected points."""

        # Output mode
        self.array_output = kwargs.get("to_array", True)

    def __call__(self) -> Optional[np.ndarray]:
        """Call the assistant."""

        if not self.finalized:
            # Select points
            self._reset()
            super().__call__()

            # Print selected points
            if self.verbosity:
                self._print_info()

        # Close the figure opened by the base class
        plt.close(self.fig)

        # The segmentation algorithm expects the points as int pixel coordinates
        if self.array_output:
            return np.array(self.pts).astype(int)
        else:
            np.array(self.pts).astype(int).tolist()

    def _print_info(self) -> None:
        """Print out information about the assistant."""

        # Print the determined points to screen so one can hardcode the definition of
        # the subregion if required.
        print("The selected points:")
        print(np.array(self.pts).astype(int).tolist())

    def _reset(self) -> None:
        """Reset list of points."""
        self.pts = []
        self.finalized = False
        """Flag controlling whether the selection has been finalized."""
        if self.verbosity:
            warn("Resetting list of points.")

    def _print_instructions(self) -> None:
        """Print instructions - always print those."""
        print(
            """\n----------------------------------------------------------------"""
            """-----------"""
        )
        print("Welcome to the point selection assistant.")
        print("Select points with 'left mouse click'.")
        print("Press 'd' to remove the latest selected point.")
        print("Press 'escape' to reset the selection.")
        print("Press 'enter' to finalize the selection.")
        print("Press 'q' to quit the assistant (possibly before finalizing).")
        print("NOTE: Do not close the figure yourself.")
        print(
            """------------------------------------------------------------------"""
            """---------\n"""
        )

    def _print_current_selection(self) -> None:
        """Print current selection."""
        if self.verbosity:
            print(f"Current selection of points:")
            print(self.pts)

    def _setup_event_handler(self) -> None:
        """Setup event handler."""
        super()._setup_event_handler()
        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_click)

    def _on_mouse_click(self, event: Any) -> None:
        """Event handler for mouse clicks."""

        # Print event
        self._print_event(event)

        # Only continue if no mode is active
        state = self.fig.canvas.toolbar.mode
        if state == "":
            # Fetch the physical coordinates in 2d plane and interpret 2d point in
            # three dimensions
            if event.button == 1:
                # Add point to subregion (in 2d)
                self.pts.append([event.xdata, event.ydata])

                # Draw a circle around the selected point
                self.ax.plot(
                    event.xdata,
                    event.ydata,
                    "go",
                    markersize=10,
                )
                self.fig.canvas.draw()
                self._print_current_selection()

    def _on_key_press(self, event: Any) -> None:
        """Event for pressing key 'd'."""

        super()._on_key_press(event)

        # Key 'd' is pressed
        if event.key == "d":
            self._remove_last_point(event.xdata, event.ydata)

    def _remove_last_point(self, xdata, ydata) -> None:
        """Remove last point from selection.

        Args:
            xdata: x-coordinate of point to remove
            ydata: y-coordinate of point to remove

        """

        self.pts.pop()
        self._print_current_selection()
        # Draw a red circle around the de-selected point
        self.ax.plot(
            xdata,
            ydata,
            "ro",
            markersize=10,
        )
        self.fig.canvas.draw()

    def _finalize(self) -> None:
        """Finalize selection."""

        self.finalized = True

        # Next round.
        self.__call__()
