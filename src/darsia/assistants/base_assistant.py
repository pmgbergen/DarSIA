"""Module containing the base assistant class."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia


class BaseAssistant(ABC):
    """Matplotlib based interactive assistant."""

    def __init__(self, img: darsia.Image, **kwargs) -> None:
        # Image(s)
        self.img = img
        """Image to be analyzed."""
        assert not self.img.series, "Image series not supported."
        assert self.img.space_dim in [2, 3], "Only 2d and 3d images supported."
        self.background: Optional[darsia.Image] = kwargs.get("background")
        """Background image for plotting."""

        # Figure options - Generate new figure and axes if not provided
        self.fig = kwargs.get("fig")
        """Figure for analysis."""
        self.ax = kwargs.get("ax")
        """Axes for analysis."""
        assert (self.fig is None) == (
            self.ax is None
        ), "Both fig and ax must be None or not None."
        if self.fig is None and self.ax is None:
            self.fig = plt.figure()  # self.name)
            if self.img.space_dim == 2:
                self.ax = self.fig.subplots(1, 1)
                self.fig.suptitle(self.name)
            elif self.img.space_dim == 3:
                self.ax = self.fig.subplots(1, 3)
                self.fig.suptitle(f"{self.name} -- 2d side views")
        self.block = kwargs.get("block", True)
        """Flag controlling whether figure is blocking."""

        if self.img.space_dim == 2:
            self.plot_grid = kwargs.get("plot_grid", False)
            """Flag controlling whether grid is plotted (in 2d)."""
        elif self.img.space_dim == 3:
            self.threshold = kwargs.get("threshold")
            """Threshold for active voxels (in 3d)."""
            self.relative = kwargs.get("relative", False)
            """Flag controlling whether threshold is relative (in 3d)."""
            self.scaling = kwargs.get("scaling", 1)
            """Scaling of points (in 3d)."""
            self.scatter = kwargs.get("scatter", True)
            """Flag controlling whether scatter plot is used (in 3d)."""
        self.use_coordinates = kwargs.get("use_coordinates", False)
        """Flag controlling whether plots use physical coordinates."""
        self.verbosity = kwargs.get("verbosity", False)
        """Flag controlling verbosity."""

    @abstractmethod
    def _print_instructions() -> None:
        """Print instructions."""
        pass

    def _print_event(self, event) -> None:
        if self.verbosity:
            print(f"{self.name} - event: {event}")

    def _setup_event_handler(self) -> None:
        """Setup event handler."""
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    def _on_key_press(self, event) -> None:
        """Finalize selection if 'enter' is pressed, and reset containers if 'escape'
        is pressed.

        Args:
            event: key press event

        """
        if self.verbosity:
            print(f"Current key: {event.key}")

        if event.key == "escape":
            # Reset and restart
            self.__call__()

        elif event.key == "enter":
            # Process selection
            self._finalize()

        elif event.key == "q":
            # Quit
            plt.close(self.fig)

    def __call__(self) -> Any:
        """Call the assistant."""
        # Setup event handler
        self._setup_event_handler()
        if self.img.space_dim == 2:
            self._plot_2d()
        elif self.img.space_dim == 3:
            self._plot_3d()

    def _plot_2d(self) -> None:
        """Plot in 2d with interactive event handler."""
        if self.img is not None and self.background is None:
            self._setup_plot_2d(self.img)
        elif self.img is None and self.background is not None:
            self._setup_plot_2d(self.background)
        elif self.img is not None and self.background is not None:
            self._setup_plot_2d(self.background, alpha=0.6)
            self._setup_plot_2d(self.img, alpha=0.4)
        else:
            raise ValueError("Either img or background must be provided.")
        plt.show(block=self.block)

    def _setup_plot_2d(self, img: darsia.Image, alpha: float = 1.0) -> None:
        """Plot in 2d with interactive event handler."""

        # Plot the entire 2d image in plain mode. Only works for scalar and optical
        # images.
        assert img.scalar or img.range_num in [1, 3]
        assert not img.series

        # Extract physical coordinates of corners
        if self.use_coordinates:
            origin = img.origin
            opposite_corner = img.opposite_corner

            # Plot
            self.ax.imshow(
                skimage.img_as_float(img.img),
                extent=(origin[0], opposite_corner[0], opposite_corner[1], origin[1]),
                alpha=alpha,
            )
        else:
            self.ax.imshow(skimage.img_as_float(img.img), alpha=alpha)

        # Plot grid
        plot_grid = self.plot_grid
        if plot_grid:
            self.ax.grid()
        self.ax.set_xlabel("x-axis")
        self.ax.set_ylabel("y-axis")
        self.ax.set_aspect("equal")

    def _plot_3d(self) -> None:
        """Side view with interactive event handler."""

        # Print instructions
        self._print_instructions()

        # Fetch bounding box
        corners = np.vstack((self.img.origin, self.img.opposite_corner))
        bbox = np.array([np.min(corners, axis=0), np.max(corners, axis=0)])

        # Extract physical coordinates and flatten
        matrix_indices = np.transpose(np.indices(self.img.shape[:3]).reshape((3, -1)))
        coordinates = self.img.coordinatesystem.coordinate(matrix_indices)

        # Extract values
        array = self.img.img
        flat_array = array.reshape((1, -1))[0]

        # Restrict to active voxels
        threshold = (
            self.threshold if self.threshold is not None else np.min(self.img.img)
        )
        # relative = self.relative

        if self.relative:
            threshold = threshold * np.max(self.img.img)
        active = flat_array > threshold

        # Signal strength
        alpha_min = 0.1
        alpha = np.clip(
            alpha_min
            + (
                (1.0 - alpha_min)
                * (flat_array - np.min(array))
                / (np.max(array) - np.min(array))
            ),
            0,
            1,
        )
        s = self.scaling * alpha

        # xy-plane
        self.ax[0].set_title(self.name + " - x-y plane")
        if self.scatter:
            self.ax[0].scatter(
                coordinates[active, 0],
                coordinates[active, 1],
                s=s[active],
                alpha=alpha[active],
                c=flat_array[active],
                cmap="viridis",
            )
            self.ax[0].set_xlim(bbox[0, 0], bbox[1, 0])
            self.ax[0].set_ylim(bbox[0, 1], bbox[1, 1])
        else:
            reduced_image = darsia.reduce_axis(self.img, axis="z")
            if self.use_coordinates:
                self.ax[0].imshow(
                    skimage.img_as_float(reduced_image.img),
                    cmap="viridis",
                    extent=(
                        reduced_image.origin[0],
                        reduced_image.opposite_corner[0],
                        reduced_image.opposite_corner[1],
                        reduced_image.origin[1],
                    ),
                )
            else:
                self.ax[0].imshow(
                    skimage.img_as_float(reduced_image.img),
                    cmap="viridis",
                )

        self.ax[0].grid()
        self.ax[0].set_xlabel("x-axis")
        self.ax[0].set_ylabel("y-axis")
        self.ax[0].set_aspect("equal")

        # xz-plane
        self.ax[1].set_title(self.name + " - x-z plane")
        if self.scatter:
            self.ax[1].scatter(
                coordinates[active, 0],
                coordinates[active, 2],
                s=s[active],
                alpha=alpha[active],
                c=flat_array[active],
                cmap="viridis",
            )
            self.ax[1].set_xlim(bbox[0, 0], bbox[1, 0])
            self.ax[1].set_ylim(bbox[0, 2], bbox[1, 2])
        else:
            reduced_image = darsia.reduce_axis(self.img, axis="y")
            if self.use_coordinates:
                self.ax[1].imshow(
                    skimage.img_as_float(reduced_image.img),
                    cmap="viridis",
                    extent=(
                        reduced_image.origin[0],
                        reduced_image.opposite_corner[0],
                        reduced_image.opposite_corner[1],
                        reduced_image.origin[1],
                    ),
                )
            else:
                self.ax[1].imshow(
                    skimage.img_as_float(reduced_image.img),
                    cmap="viridis",
                )
        self.ax[1].grid()
        self.ax[1].set_xlabel("x-axis")
        self.ax[1].set_ylabel("z-axis")
        self.ax[1].set_aspect("equal")

        # yz-plane
        self.ax[2].set_title(self.name + " - y-z plane")
        if self.scatter:
            self.ax[2].scatter(
                coordinates[active, 1],
                coordinates[active, 2],
                s=s[active],
                alpha=alpha[active],
                c=flat_array[active],
                cmap="viridis",
            )
            self.ax[2].set_xlim(bbox[0, 1], bbox[1, 1])
            self.ax[2].set_ylim(bbox[0, 2], bbox[1, 2])
        else:
            reduced_image = darsia.reduce_axis(self.img, axis="x")
            if self.use_coordinates:
                self.ax[2].imshow(
                    skimage.img_as_float(reduced_image.img),
                    cmap="viridis",
                    extent=(
                        reduced_image.origin[0],
                        reduced_image.opposite_corner[0],
                        reduced_image.opposite_corner[1],
                        reduced_image.origin[1],
                    ),
                )
            else:
                self.ax[2].imshow(
                    skimage.img_as_float(reduced_image.img),
                    cmap="viridis",
                )
        self.ax[2].grid()
        self.ax[2].set_xlabel("y-axis")
        self.ax[2].set_ylabel("z-axis")
        self.ax[2].set_aspect("equal")

        plt.show(block=self.block)
