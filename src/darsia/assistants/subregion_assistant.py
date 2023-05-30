"""Module for defining subregions interactively."""

from typing import Any, Optional
from warnings import warn

import numpy as np

import darsia


class SubregionAssistant(darsia.BaseAssistant):
    def __init__(self, img: darsia.Image, **kwargs) -> None:
        super().__init__(img, **kwargs)

        # Initialize containers
        self._reset()

        # Set name for titles in plots
        self.name = "Subregion assistant"
        # Prepare output
        self.coordinates = None
        """Coordinates uniquely defining a box."""

    def __call__(self) -> Optional[np.ndarray]:
        """Call the assistant."""

        self._reset()
        super().__call__()

        # Print the determined coordinates to screen so one can hardcode the definition
        # of the subregion if required.
        if self.verbosity:
            print("The determined subregion is defined by the coordinates:")
            print(self.coordinates)

        return self.coordinates

    def _reset(self) -> None:
        """Reset subregion."""
        self.pts = [[] for _ in range(self.img.space_dim)]
        """Selected points distributed in separate lists for each dimension."""
        self.finalized = False
        """Flag controlling whether the selection has been finalized."""
        if self.verbosity:
            warn("Resetting subregion.")

    def _print_instructions(self) -> None:
        """Print instructions - always print those."""
        print(
            """\n----------------------------------------------------------------"""
            """-----------"""
        )
        print("Welcome to the subregion assistant.")
        print(
            "Select points with 'left mouse click' to define coordinates spanning a box."
        )
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
            print(f"Current selection for subregion:")
            print("subregion: {}".format(self.pts))

    def _setup_event_handler(self) -> None:
        """Setup event handler."""
        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    def _on_mouse_click(self, event: Any) -> None:
        """Event handler for mouse clicks."""

        # Print event
        self._print_event(event)

        # Only continue if no mode is active
        state = self.fig.canvas.toolbar.mode
        if state == "":
            # Determine which subplot has been clicked
            for ax_id in range(3):
                if event.inaxes == self.ax[ax_id]:
                    first_axis = "xxy"[ax_id]
                    second_axis = "yzz"[ax_id]
                    break

            first_index = "xyz".find(first_axis)
            second_index = "xyz".find(second_axis)

            # Fetch the physical coordinates in 2d plane and interpret 2d point in
            # three dimensions
            if event.button == 1:
                # Add point to subregion (in 2d)
                self.pts[first_index].append(event.xdata)
                self.pts[second_index].append(event.ydata)

                # Draw a circle around the selected point
                self.ax[ax_id].plot(
                    event.xdata,
                    event.ydata,
                    "go",
                    markersize=10,
                )
                self.fig.canvas.draw()
                self._print_current_selection()

    def _finalize(self) -> None:
        """Finalize selection."""

        # Determine ranges of subregion (i.e., convex hull); if no points have been
        # selected in one dimension, use the full range

        intervals = [None for _ in range(self.img.space_dim)]

        for dim in range(self.img.space_dim):
            if self.pts[dim] is None:
                warn("Selection in dimension {} is empty.".format(dim))
                raise NotImplementedError
            else:
                intervals[dim] = [
                    min(self.pts[dim]),
                    max(self.pts[dim]),
                ]

        # Create by defining two most extreme coordinates in full space
        self.coordinates = np.array(
            [
                [intervals[0][0], intervals[1][0], intervals[2][0]],
                [intervals[0][1], intervals[1][1], intervals[2][1]],
            ]
        )
        self.img = self.img.subregion(coordinates=self.coordinates)
        self.finalized = True

        # Next round.
        self.__call__()
