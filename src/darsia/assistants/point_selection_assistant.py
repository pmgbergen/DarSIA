"""Module for point selection."""

from typing import Any, Optional, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia


class PointSelectionAssistant(darsia.BaseAssistant):
    def __init__(self, img: darsia.Image, **kwargs) -> None:
        if img is None:
            img = kwargs.get("background")
            assert img is not None, "No image provided."

        # Only continue for 2d images
        assert img.space_dim == 2, "Only 2d images are supported."

        # Set name for titles in plots
        self.name = kwargs.get("name", "Point selection assistant")
        """Name of assistant / short version of instructions."""

        super().__init__(img, use_coordinates=False, **kwargs)

        # Initialize containers
        self.pts: Optional[list[list[float]]] = None
        """Selected points in voxel format with matrix indexing."""
        self._reset()

        # Output mode
        self.return_coordinates = kwargs.get("coordinates", False)
        """Flag controlling whether to return coordinates or voxels."""

        # Activate masking if labels are provided
        self.labels: Optional[darsia.Image] = kwargs.get("labels", None)
        """Labels for the image."""
        if self.labels is not None:
            # Expand image with a (active) mask
            coarse_img = skimage.img_as_float(
                darsia.resize(
                    self.img, fx=0.1, fy=0.1, interpolation="inter_nearest"
                ).img
            )
            mask = np.ones(coarse_img.shape[:2], dtype=np.float32)
            self.coarse_masked_img = np.dstack((coarse_img, mask))
            self.zorder = 0

    def __call__(self) -> Union[darsia.CoordinateArray, darsia.VoxelArray]:
        """Call the assistant."""

        if not self.finalized:
            # Select points
            self._reset()
            super().__call__()

            # Print selected points
            if self.verbosity:
                self._print_info()

        # Convert to right format.
        if hasattr(self, "fig") and self.fig is not None:
            plt.close(self.fig)
        voxels = darsia.VoxelArray(self.pts)
        if self.return_coordinates:
            return self.img.coordinatesystem.coordinate(voxels)
        else:
            return darsia.make_voxel(voxels)

    def _print_info(self) -> None:
        """Print out information about the assistant."""

        # Print the determined points to screen so one can hardcode the definition of
        # the subregion if required.
        print("The selected points in matrix indexing format:")
        print(darsia.VoxelArray(self.pts).tolist())

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
            if event.button == 1 and event.inaxes is not None:
                # Add point to subregion (in 2d) in matrix indexing
                self.pts.append([event.ydata, event.xdata])

                # Draw a circle around the selected point
                dot = self.ax.plot(
                    event.xdata,
                    event.ydata,
                    "go",
                    markersize=10,
                )

                # Auxiliary plot: Mark labeled region
                if self.labels is not None:
                    label = self.labels.img[int(event.ydata), int(event.xdata)]
                    mask = (
                        darsia.resize(
                            skimage.img_as_float(self.labels.img == label),
                            shape=self.coarse_masked_img.shape[:2],
                            interpolation="inter_nearest",
                        )
                        > 0.5
                    )
                    self.coarse_masked_img[mask, -1] = 0.3
                    plt.figure("Auxiliary: Masked image")
                    plt.imshow(self.coarse_masked_img, zorder=self.zorder)
                    self.zorder += 1
                    plt.show()

                self.ax.draw_artist(dot[0])
                self.fig.canvas.blit(self.ax.bbox)

                self._print_current_selection()

    def _on_key_press(self, event: Any) -> None:
        """Event for pressing key 'd'."""

        super()._on_key_press(event)

        # Key 'd' is pressed
        if event.key == "d":
            self._remove_last_point()

    def _remove_last_point(self) -> None:
        """Remove last point from selection."""

        if len(self.pts) == 0:
            return
        rm_pt = self.pts[-1]
        self.pts = type(self.pts)(self.pts[:-1])
        self._print_current_selection()
        # Draw a red circle around the de-selected point
        dot = self.ax.plot(
            rm_pt[1],
            rm_pt[0],
            "ro",
            markersize=10,
        )
        self.ax.draw_artist(dot[0])
        self.fig.canvas.blit(self.ax.bbox)

    def _finalize(self) -> None:
        """Finalize selection."""

        self.finalized = True
        self.pts = darsia.make_voxel(self.pts)

        # Next round - needed to close the figure
        self.__call__()
