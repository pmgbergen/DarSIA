"""Module containing the base assistant class."""

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

import darsia


class BaseAssistant(ABC):
    """Matplotlib based interactive assistant."""

    def __init__(self, img: darsia.Image, **kwargs) -> None:
        self.img = img
        self.kwargs = kwargs

    @abstractmethod
    def _print_instructions() -> None:
        pass

    @abstractmethod
    def _setup_event_handler() -> None:
        pass

    @abstractmethod
    def __call__(self) -> None:
        pass

    @abstractmethod
    def return_result(self) -> None:
        pass

    def _plot_3d(self) -> None:
        # Setup figure
        fig_2d, axs = plt.subplots(1, 3)
        fig_2d.suptitle("2d side views")
        _title = "rotation assistant"

        # Setup event handler
        self.fig = fig_2d
        self.ax = axs
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
        assert not self.img.series
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
        axs[0].set_title(_title + " - x-y plane")
        axs[0].scatter(
            coordinates[active, 0],
            coordinates[active, 1],
            s=s[active],
            alpha=alpha[active],
            c=flat_array[active],
            cmap="viridis",
        )
        axs[0].set_xlim(bbox[0, 0], bbox[1, 0])
        axs[0].set_ylim(bbox[0, 1], bbox[1, 1])
        axs[0].set_xlabel("x-axis")
        axs[0].set_ylabel("y-axis")
        axs[0].set_aspect("equal")

        # xz-plane
        axs[1].set_title(_title + " - x-z plane")
        axs[1].scatter(
            coordinates[active, 0],
            coordinates[active, 2],
            s=s[active],
            alpha=alpha[active],
            c=flat_array[active],
            cmap="viridis",
        )
        axs[1].set_xlim(bbox[0, 0], bbox[1, 0])
        axs[1].set_ylim(bbox[0, 2], bbox[1, 2])
        axs[1].set_xlabel("x-axis")
        axs[1].set_ylabel("z-axis")
        axs[1].set_aspect("equal")

        # yz-plane
        axs[2].set_title(_title + " - y-z plane")
        axs[2].scatter(
            coordinates[active, 1],
            coordinates[active, 2],
            s=s[active],
            alpha=alpha[active],
            c=flat_array[active],
            cmap="viridis",
        )
        axs[2].set_xlim(bbox[0, 1], bbox[1, 1])
        axs[2].set_ylim(bbox[0, 2], bbox[1, 2])
        axs[2].set_xlabel("y-axis")
        axs[2].set_ylabel("z-axis")
        axs[2].set_aspect("equal")

        plt.show()
