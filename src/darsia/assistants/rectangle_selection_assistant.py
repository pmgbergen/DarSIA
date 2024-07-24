"""Module for defining subregions interactively using a Rectangle selector."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import skimage
from matplotlib.widgets import RectangleSelector

import darsia


class RectangleSelectionAssistant(darsia.BaseAssistant):
    """Assistant for selecting a rectangle in an image.

    Similar to the point selection assistant, but for rectangles.

    """

    def __init__(self, img: darsia.Image, **kwargs) -> None:
        """Initialize the assistant.

        Args:
            img (darsia.Image): Image to select a rectangle in.
            **kwargs: Additional arguments.
                - background (darsia.Image): Background image to display.
                - name (str): Name of assistant / short version of instructions.
                - coordinates (bool): Flag controlling whether to return coordinates or voxels.
                - labels (darsia.Image): Labels for the image.

        """
        if img is None:
            img = kwargs.get("background")
            assert img is not None, "No image provided."

        # Only continue for 2d images
        assert img.space_dim == 2, "Only 2d images are supported."

        # Set name for titles in plots
        self.name = kwargs.get("name", "Rectangle selection assistant")
        """Name of assistant / short version of instructions."""

        super().__init__(img, use_coordinates=False, **kwargs)

        # Initialize widget and containers
        self.rs: Optional[RectangleSelector] = None
        """Rectangle selector."""
        self.box: Optional[tuple[slice, ...]] = None
        """Selected box in terms of slices."""
        self.corners: Optional[darsia.VoxelArray] = None
        """Selected corners in voxel format with matrix indexing."""
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

    def __call__(self) -> tuple[slice, ...]:
        """Call the assistant.

        Returns:
            tuple[slice, ...]: Selected box in terms of slices.

        """
        if not self.finalized:
            # Select points
            self._reset()
            super().__call__()

            # Print selected points
            if self.verbosity:
                self._print_info()

        return self.box

    def _reset(self) -> None:
        """Reset assistant."""
        self.corners = None
        self.box = None
        self.finalized = False

    def _onselect(self, eclick, erelease) -> None:
        """Define what to do when a rectangle is selected."""
        self.corners = darsia.VoxelArray(
            np.array([[eclick.ydata, eclick.xdata], [erelease.ydata, erelease.xdata]])
        )
        self.box = darsia.bounding_box(self.corners)
        self._print_info()

    def _print_info(self) -> None:
        """Print info about points so far assigned by the assistant."""
        print(f"Selected box: {self.box}")

    def _finalize(self) -> None:
        """Finalize the assistant."""
        if self.verbosity:
            self._print_info()
        plt.close()

    def _print_instructions(self) -> None:
        """Print instructions."""
        print(
            """\n----------------------------------------------------------------"""
            """-----------"""
        )
        print("Welcome to the rectangle selection assistant.")
        print("Select a rectangle with the mouse.")
        print("Press 'q/enter' to quit the assistant (and finalize the selection).")
        print(
            """----------------------------------------------------------------"""
            """-----------"""
        )

    def _setup_event_handler(self) -> None:
        """Setup event handler and rectangle selector."""
        super()._setup_event_handler()
        self.rs = RectangleSelector(
            self.ax,
            self._onselect,
            useblit=True,
            button=[1, 3],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
