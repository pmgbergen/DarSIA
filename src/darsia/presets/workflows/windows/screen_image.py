"""Image in an axes, drawn from a copy of its visible part at screen resolution.

matplotlib processes the whole array of an image at every draw, which is slow for large
images shown in small panels. `ScreenImage` keeps the full array, but gives matplotlib
only the visible part, shrunk (by area averaging) to the size of the axes on screen. The
visible part is cut anew when the view is zoomed, panned or resized, so zoomed views keep
the full detail. Data coordinates stay those of the full array (e.g. for clicks).

"""

from __future__ import annotations

import math

import cv2
import numpy as np


def _shrink(data: np.ndarray, width: int, height: int) -> np.ndarray:
    """Area-averaged copy of an image (NaN ignored in the averages)."""
    if np.issubdtype(data.dtype, np.floating) and np.isnan(data).any():
        valid = ~np.isnan(data)
        total = cv2.resize(
            np.where(valid, data, 0.0), (width, height), interpolation=cv2.INTER_AREA
        )
        weight = cv2.resize(
            valid.astype(data.dtype), (width, height), interpolation=cv2.INTER_AREA
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(weight > 0, total / weight, np.nan)
    return cv2.resize(data, (width, height), interpolation=cv2.INTER_AREA)


class ScreenImage:
    """Image in an axes, drawn at screen resolution.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes of the image.
    data : np.ndarray
        Full image (rows, cols) or (rows, cols, channels).
    **kwargs
        Passed to `ax.imshow` (e.g. `cmap`).

    """

    def __init__(self, ax, data: np.ndarray, **kwargs) -> None:
        self.ax = ax
        self.data = np.ascontiguousarray(data)
        self.image = ax.imshow(self.data, **kwargs)
        """The matplotlib image (shows the screen copy)."""
        ax.set_autoscale_on(False)
        self._updating = False
        ax.callbacks.connect("xlim_changed", self._on_change)
        ax.callbacks.connect("ylim_changed", self._on_change)
        ax.figure.canvas.mpl_connect("resize_event", self._on_change)
        self.update()

    def set_data(self, data: np.ndarray) -> None:
        """Show another image (of any shape).

        Parameters
        ----------
        data : np.ndarray
            Full image.

        """
        shape = self.data.shape[:2]
        self.data = np.ascontiguousarray(data)
        if self.data.shape[:2] != shape:
            rows, cols = self.data.shape[:2]
            self._updating = True
            try:
                self.ax.set_xlim(-0.5, cols - 0.5)
                self.ax.set_ylim(rows - 0.5, -0.5)
            finally:
                self._updating = False
        self.update()

    def get_array(self) -> np.ndarray:
        """Full image."""
        return self.data

    def get_clim(self) -> tuple[float, float]:
        """Color limits (scalar images)."""
        return self.image.get_clim()

    def set_clim(self, vmin: float | None = None, vmax: float | None = None) -> None:
        """Set the color limits (scalar images)."""
        self.image.set_clim(vmin=vmin, vmax=vmax)

    def _on_change(self, *args) -> None:
        if not self._updating:
            self.update()

    def update(self) -> None:
        """Cut the visible part and shrink it to the size of the axes on screen."""
        ax, data = self.ax, self.data
        rows, cols = data.shape[:2]
        x0, x1 = sorted(ax.get_xlim())
        y0, y1 = sorted(ax.get_ylim())
        c0, c1 = max(math.floor(x0 + 0.5), 0), min(math.ceil(x1 + 0.5), cols)
        r0, r1 = max(math.floor(y0 + 0.5), 0), min(math.ceil(y1 + 0.5), rows)
        if c1 <= c0 or r1 <= r0:
            return
        crop = data[r0:r1, c0:c1]
        box = ax.get_window_extent()
        scale = min(box.width / crop.shape[1], box.height / crop.shape[0], 1.0)
        if scale < 1.0:
            width = max(round(crop.shape[1] * scale), 1)
            height = max(round(crop.shape[0] * scale), 1)
            crop = _shrink(np.ascontiguousarray(crop), width, height)
        self._updating = True
        try:
            self.image.set_data(crop)
            self.image.set_extent((c0 - 0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5))
        finally:
            self._updating = False
