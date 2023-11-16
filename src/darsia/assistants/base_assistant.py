"""Module containing the base assistant class."""

from abc import ABC, abstractmethod
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia


class BaseAssistant(ABC):
    """Matplotlib based interactive assistant."""

    def __init__(self, img: darsia.Image, **kwargs) -> None:
        self.img = img
        """Image to be analyzed."""
        assert not self.img.series, "Image series not supported."
        self.kwargs = kwargs
        """Keyword arguments."""
        self.verbosity = kwargs.get("verbosity", False)
        """Flag controlling verbosity."""
        self.fig = None
        """Figure for analysis."""
        self.ax = None
        """Axes for analysis."""
        self.use_coordinates = kwargs.get("use_coordinates", False)
        """Flag controlling whether plots use physical coordinates."""

    @abstractmethod
    def _print_instructions() -> None:
        """Print instructions."""
        pass

    def _print_event(self, event) -> None:
        if self.verbosity:
            print(event)

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

    @abstractmethod
    def __call__(self) -> Any:
        """Call the assistant."""
        if self.fig is not None:
            plt.close(self.fig)
        if self.img.space_dim == 2:
            self._plot_2d()
        elif self.img.space_dim == 3:
            self._plot_3d()
        else:
            raise NotImplementedError

    def _setup_plot_2d(
        self, img: darsia.Image, new_figure: bool = True, alpha: float = 1.0
    ) -> None:
        """Plot in 2d with interactive event handler."""

        # Setup figure
        if new_figure:
            self.fig, self.ax = plt.subplots(1, 1)

        # Setup event handler
        self._setup_event_handler()

        # Print instructions
        self._print_instructions()

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
        plot_grid = self.kwargs.get("plot_grid", False)
        if plot_grid:
            self.ax.grid()
        self.ax.set_xlabel("x-axis")
        self.ax.set_ylabel("y-axis")
        self.ax.set_aspect("equal")

    def _plot_2d(self) -> None:
        """Plot in 2d with interactive event handler."""
        self._setup_plot_2d(self.img)
        plt.show(block=True)

    def _plot_3d(self) -> None:
        """Side view with interactive event handler."""

        # Setup figure
        self.fig, self.ax = plt.subplots(1, 3)
        self.fig.suptitle("2d side views")

        # Setup event handler
        self._setup_event_handler()

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
        threshold = self.kwargs.get("threshold", np.min(self.img.img))
        relative = self.kwargs.get("relative", False)
        if relative:
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
        scaling = self.kwargs.get("scaling", 1)
        s = scaling * alpha

        # Plotting style
        scatter = self.kwargs.get("scatter", True)

        # xy-plane
        self.ax[0].set_title(self.name + " - x-y plane")
        if scatter:
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
        if scatter:
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
        if scatter:
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

        plt.show(block=True)
