"""Module for defining subregions interactively."""

from typing import Any
from warnings import warn

import numpy as np
from matplotlib import pyplot as plt

import darsia


class SubregionAssistant(darsia.BaseAssistant):
    def __init__(self, img: darsia.Image, **kwargs) -> None:
        super().__init__(img, **kwargs)

        # Initialize containers
        self._reset()

    def __call__(self):
        """Call the assistant."""

        # Plot 3d image and setup event handler
        if self.img.space_dim == 3:
            self._plot_3d()
        else:
            raise NotImplementedError(
                "Subregion assistant only implemented for 3d images."
            )

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
        print("\nWelcome to the subregion assistant.")
        print("Please select a subregion by clicking on the image.")
        print("Press 'r' to reset the selection.")
        print("Press 'q' to quit the assistant.\n")

    def _print_current_selection(self) -> None:
        """Print current selection."""
        if self.verbosity:
            print(f"Current selection for subregion:")
            print("subregion: {}".format(self.pts))

    def _setup_event_handler(self) -> None:
        """Setup event handler."""
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    def _on_click(self, event: Any) -> None:
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

    def _on_key_press(self, event) -> None:
        """Event handler for key presses."""

        # Print event
        self._print_event(event)

        if event.key == "r":
            # Reset selection
            self._reset()
            self.__call__()

        elif event.key == "q":
            # Quit assistant
            self._finalize()
            plt.close(self.fig)

    def _finalize(self) -> None:
        """Finalize selection."""

        # Determine ranges of subregion (i.e., convex hull); if no points have been
        # selected in one dimension, use the full range

        intervals = [None for _ in range(self.img.space_dim)]

        for dim in range(self.img.space_dim):
            if self.pts[dim] is None:
                warn("Selection in dimension {} is empty.".format(dim))
                # self.pts[dim] = [np.min(self.img.origin[dim], self.img.opposite_corner[dim]), np.max(self.img.origin[dim], self.img.opposite_corner[dim])]
            else:
                intervals[dim] = [
                    min(self.pts[dim]),
                    max(self.pts[dim]),
                ]

        # FIXME
        assert not any([interval is None for interval in intervals])

        # Create by defining two most extreme coordinates in full space
        self.coordinates = np.array(
            [
                [intervals[0][0], intervals[1][0], intervals[2][0]],
                [intervals[0][1], intervals[1][1], intervals[2][1]],
            ]
        )
        print(self.coordinates)
        self.finalized = True
        self.img = self.img.subregion(coordinates=self.coordinates)

        # Allow to continue
        print("Press 'space' to continue.")
        # TODO continue
