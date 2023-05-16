"""Module containing the base assistant class."""

from abc import ABC, abstractmethod
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

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

    @abstractmethod
    def _print_instructions() -> None:
        """Print instructions."""
        pass

    def _print_event(self, event) -> None:
        if self.verbosity:
            print(event)

    @abstractmethod
    def _setup_event_handler() -> None:
        """Setup event handler."""
        pass

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
        if self.img.space_dim == 3:
            self._plot_3d()
        else:
            raise NotImplementedError

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
        alpha = alpha_min + (
            (1.0 - alpha_min)
            * (flat_array - np.min(array))
            / (np.max(array) - np.min(array))
        )
        scaling = self.kwargs.get("scaling", 1)
        s = scaling * alpha

        # xy-plane
        self.ax[0].set_title(self.name + " - x-y plane")
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
        self.ax[0].set_xlabel("x-axis")
        self.ax[0].set_ylabel("y-axis")
        self.ax[0].set_aspect("equal")

        # xz-plane
        self.ax[1].set_title(self.name + " - x-z plane")
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
        self.ax[1].set_xlabel("x-axis")
        self.ax[1].set_ylabel("z-axis")
        self.ax[1].set_aspect("equal")

        # yz-plane
        self.ax[2].set_title(self.name + " - y-z plane")
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
        self.ax[2].set_xlabel("y-axis")
        self.ax[2].set_ylabel("z-axis")
        self.ax[2].set_aspect("equal")

        plt.show()
