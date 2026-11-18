"""Interactive editor of the key colors of a color path (manual color-path calibration).

A color path runs through key colors, relative to the baseline color, in RGB space. For
one label, the editor shows the colors of the calibration images (3D), the key colors
with the path, a calibration image with the label highlighted, and the pH-indicator
interpretation of that image with the current path. The user moves, adds, removes and
aligns key colors, samples one from the image (pipette) or returns to the automatic fit.

`ColorPathEditorState` holds the data and the operations (no matplotlib);
`ColorPathEditorWindow` is the window, built once and updated in place (partial
repaint, previews at screen resolution with linked zoom, preview calculated in the
background). `edit_color_path` is passed as `editor` to
`LabelColorPathMapRegression.find_color_path` in manual mode.

"""

from __future__ import annotations

import logging
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider, TextBox

import darsia
from darsia.presets.workflows.windows.background import BUSY_MESSAGE, BackgroundRunner
from darsia.presets.workflows.windows.blitting import SliderBlitter, StatusText
from darsia.presets.workflows.windows.highlight import LUMINANCE, highlight_region
from darsia.presets.workflows.windows.screen_image import ScreenImage
from darsia.utils.figure_window import maximize_figure_window

logger = logging.getLogger(__name__)

MIN_PREVIEW_ROWS = 120
"""Smallest number of rows of the preview images."""

PREVIEW_DOWNSAMPLING = 4
"""Reduction of the preview images per axis (relative to the labels)."""

MASK_THRESHOLD = 0.5
"""Threshold of the interpolated calibration mask on the preview grid."""


def _parse_index_selection(text: str, size: int) -> list[list[int]]:
    """Parse an index selection like ``"2"``, ``"1-3"`` or ``"1-3, 6"``.

    Parameters
    ----------
    text : str
        Comma-separated indices and inclusive ranges ``a-b``.
    size : int
        Number of available indices; out-of-range entries are dropped.

    Returns
    -------
    list[list[int]]
        Contiguous runs of sorted unique indices; empty if unparsable.

    """
    indices: set[int] = set()
    try:
        for part in text.replace(" ", "").split(","):
            if part == "":
                continue
            start, _, stop = part.partition("-")
            lo, hi = int(start), int(stop) if stop else int(start)
            indices.update(range(min(lo, hi), max(lo, hi) + 1))
    except ValueError:
        logger.warning("Could not parse index selection '%s'.", text)
        return []
    valid = sorted(i for i in indices if 0 <= i < size)
    runs: list[list[int]] = []
    for i in valid:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def _warn(message: str) -> str:
    """Log a warning and return its message (shown in the window)."""
    logger.warning(message)
    return message


class ColorPathEditorState:
    """Key colors of one label's color path and the preview data (no plotting).

    Parameters
    ----------
    key_relative_colors : list[np.ndarray]
        Key colors at the start, relative to the base color.
    base_color : np.ndarray
        Base (baseline) color of the label.
    ignore : darsia.ColorSpectrum, optional
        Spectrum ignored by the interpretation (baseline fluctuations).
    auto_fit : Callable
        Automatic fit with a given number of segments (for Reset).
    default_num_segments : int
        Number of segments of the automatic fit.
    name : str
        Name of the color path.
    label : int, optional
        The label.
    preview_images : list[darsia.Image]
        Calibration images on the grid of `labels`.
    preview_baseline : darsia.Image, optional
        Baseline image on the grid of `labels`.
    labels : darsia.Image
        Labels.
    mask : darsia.Image
        Calibration mask (boolean).

    """

    def __init__(
        self,
        *,
        key_relative_colors: list[np.ndarray],
        base_color: np.ndarray,
        ignore: darsia.ColorSpectrum | None,
        auto_fit: Callable[[int], list[np.ndarray]],
        default_num_segments: int,
        name: str,
        label: int | None,
        preview_images: list[darsia.Image],
        preview_baseline: darsia.Image | None,
        labels: darsia.Image,
        mask: darsia.Image,
    ) -> None:
        self.key_colors = [np.array(c, dtype=float).copy() for c in key_relative_colors]
        """Key colors (relative), edited in place."""
        self.base_color = base_color
        self.ignore = ignore
        self.auto_fit = auto_fit
        self.default_num_segments = default_num_segments
        self.name = name
        self.label = label
        self.labels = labels
        self.mask = mask
        self.selected = 0
        """Index of the selected key color."""
        self.preview_idx = 0
        """Index of the shown calibration image."""
        self.preview_signal: np.ndarray | None = None
        """Interpretation of the shown image with the path when last computed."""
        self.preview_stale = True
        """Whether the key colors changed since the preview was computed."""

        self.images = [
            img for img in preview_images if img.img.shape[:2] == labels.img.shape
        ]
        """Calibration images on the grid of the labels."""
        self.baseline = preview_baseline
        shape_matches = (
            preview_baseline is not None
            and preview_baseline.img.shape[:2] == labels.img.shape
            and len(self.images) > 0
        )
        self.has_preview = (
            label is not None and preview_baseline is not None and (shape_matches)
        )
        """Whether a preview is possible (images and baseline on the label grid)."""
        if (
            len(preview_images) > 0
            and preview_baseline is not None
            and not (self.has_preview)
        ):
            # Preview inputs were provided, but not all of them matched the label domain.
            logger.warning(
                "Skipping manual color-path preview because image dimensions do not "
                "match the label map."
            )

        self.coarse_shape = labels.img.shape
        self.coarse_images: list[darsia.Image] = []
        self.coarse_baseline: darsia.Image | None = None
        self.coarse_labels: darsia.Image | None = None
        self.coarse_mask: np.ndarray | None = None
        if self.has_preview:
            rows = max(MIN_PREVIEW_ROWS, labels.img.shape[0] // PREVIEW_DOWNSAMPLING)
            rows = min(rows, labels.img.shape[0])
            cols = max(1, int(round(labels.img.shape[1] / labels.img.shape[0] * rows)))
            self.coarse_shape = (rows, cols)
            self.coarse_images = [
                darsia.resize(img, shape=self.coarse_shape) for img in self.images
            ]
            self.coarse_baseline = darsia.resize(
                preview_baseline, shape=self.coarse_shape
            )
            self.coarse_labels = darsia.resize(
                labels, shape=self.coarse_shape, interpolation="inter_nearest"
            )
            float_mask = darsia.ScalarImage(mask.img.astype(float), **mask.metadata())
            resized_mask = darsia.resize(float_mask, shape=self.coarse_shape)
            self.coarse_mask = resized_mask.img > MASK_THRESHOLD

    # ! ---- DATA ----

    def label_mask(self) -> np.ndarray:
        """Pixels of the label within the calibration mask (preview grid)."""
        assert self.label is not None
        assert self.coarse_labels is not None and self.coarse_mask is not None
        return np.logical_and(self.coarse_labels.img == self.label, self.coarse_mask)

    def label_mask_full(self) -> np.ndarray:
        """Pixels of the label within the calibration mask (full grid)."""
        assert self.label is not None
        return np.logical_and(self.labels.img == self.label, self.mask.img)

    def coarse_image(self) -> darsia.Image:
        """Shown calibration image (preview grid)."""
        return self.coarse_images[self.preview_idx]

    def full_image(self) -> darsia.Image:
        """Shown calibration image (full grid)."""
        return self.images[self.preview_idx]

    def color_path(self) -> darsia.ColorPath:
        """Color path through the current key colors."""
        return darsia.ColorPath(
            colors=None,
            relative_colors=self.key_colors,
            base_color=self.base_color,
            mode="rgb",
            name=self.name,
        )

    def compute_preview(self) -> np.ndarray | None:
        """pH-indicator interpretation of the shown image with the current path."""
        if not self.has_preview or self.label is None:
            return None
        assert self.coarse_baseline is not None
        label_mask = self.label_mask()
        if not np.any(label_mask):
            return np.full(self.coarse_shape, np.nan, dtype=float)
        relative = self.coarse_image().img.astype(
            float
        ) - self.coarse_baseline.img.astype(float)
        selected = relative[label_mask].reshape((-1, 3))
        if selected.shape[0] == 0:
            return np.full(self.coarse_shape, np.nan, dtype=float)
        color_path = self.color_path()
        interpolation = darsia.ColorPathInterpolation(
            color_path=color_path,
            color_mode=darsia.ColorMode.RELATIVE,
            values=color_path.equidistant_distances,
            ignore_spectrum=self.ignore,
        )
        interpretation = np.full(self.coarse_shape, np.nan, dtype=float)
        interpretation[label_mask] = interpolation(selected)
        return interpretation

    def highlighted_image(self) -> np.ndarray:
        """Shown image with everything outside the label turned (mostly) gray."""
        image = self.coarse_image().img.copy().astype(float)
        if image.ndim != 3:
            return image
        return highlight_region(image, ~self.label_mask(), LUMINANCE)

    # ! ---- OPERATIONS ----

    def select(self, text: str) -> None:
        """Select a key color by its index (as text; invalid text keeps the selection)."""
        try:
            idx = int(float(text))
        except ValueError:
            idx = self.selected
        self.selected = max(0, min(len(self.key_colors) - 1, idx))

    def set_color(self, color: np.ndarray) -> None:
        """Set the selected key color."""
        self.key_colors[self.selected] = color
        self.preview_stale = True

    def align(self, text: str) -> str | None:
        """Place the selected runs of key colors evenly between their neighbours;
        returns a message for runs without anchors on both sides (else None)."""
        n = len(self.key_colors)
        messages = []
        for run in _parse_index_selection(text, n):
            lo, hi = run[0] - 1, run[-1] + 1
            if lo < 0 or hi > n - 1:
                messages.append(
                    _warn(f"Align skipped for {run}: needs anchors on both sides.")
                )
                continue
            c_lo, c_hi = self.key_colors[lo], self.key_colors[hi]
            for i in run:
                self.key_colors[i] = c_lo + (i - lo) / (hi - lo) * (c_hi - c_lo)
        self.preview_stale = True
        return "; ".join(messages) or None

    def add(self) -> None:
        """Insert a key color halfway after the selected one (before the last)."""
        n = len(self.key_colors)
        i = self.selected if self.selected < n - 1 else n - 2
        midpoint = 0.5 * (self.key_colors[i] + self.key_colors[i + 1])
        self.key_colors.insert(i + 1, midpoint)
        self.selected = i + 1
        self.preview_stale = True

    def remove(self) -> str | None:
        """Remove the selected key color; returns a message if only two are left."""
        if len(self.key_colors) <= 2:
            return _warn("Cannot remove: a color path needs at least 2 colors.")
        del self.key_colors[self.selected]
        self.selected = min(self.selected, len(self.key_colors) - 1)
        self.preview_stale = True
        return None

    def reset(self) -> None:
        """Return to the automatic fit."""
        self.key_colors[:] = [
            np.array(c, dtype=float) for c in self.auto_fit(self.default_num_segments)
        ]
        self.selected = 0
        self.preview_stale = True

    def rerun_preview(self) -> None:
        """Compute the preview with the current key colors."""
        self.preview_signal = self.compute_preview()
        self.preview_stale = False

    def step_image(self, step: int) -> None:
        """Show another calibration image (cyclic) and compute its preview."""
        if len(self.coarse_images) == 0:
            return
        self.preview_idx = (self.preview_idx + step) % len(self.coarse_images)
        self.rerun_preview()

    def pipette(self, col_limits, row_limits) -> str | None:
        """Set the selected key color to the characteristic color of the label in a
        region of the shown image.

        Parameters
        ----------
        col_limits, row_limits : tuple[float, float]
            Region in pixels of the preview grid (e.g. the zoomed view).

        Returns
        -------
        str or None
            Message why no color was set; None if a color was set.

        """
        if not self.has_preview or self.label is None:
            return _warn("Pipette unavailable without preview calibration data.")
        assert self.baseline is not None
        label_mask = self.label_mask_full()
        if not np.any(label_mask):
            return _warn("Pipette skipped because selected label has no active pixels.")

        coarse_rows, coarse_cols = self.coarse_image().img.shape[:2]
        if coarse_rows <= 0 or coarse_cols <= 0:
            return _warn("Pipette skipped due to invalid preview image dimensions.")
        col_start = int(np.floor(max(0.0, min(col_limits))))
        col_stop = int(np.ceil(min(float(coarse_cols), max(col_limits))))
        row_start = int(np.floor(max(0.0, min(row_limits))))
        row_stop = int(np.ceil(min(float(coarse_rows), max(row_limits))))
        if row_stop <= row_start or col_stop <= col_start:
            return _warn("Pipette skipped because the current zoom window is empty.")

        full_rows, full_cols = self.full_image().img.shape[:2]
        full_row_start = int(np.floor(row_start / coarse_rows * full_rows))
        full_row_stop = int(np.ceil(row_stop / coarse_rows * full_rows))
        full_col_start = int(np.floor(col_start / coarse_cols * full_cols))
        full_col_stop = int(np.ceil(col_stop / coarse_cols * full_cols))
        full_row_start = max(0, min(full_rows - 1, full_row_start))
        full_row_stop = max(full_row_start + 1, min(full_rows, full_row_stop))
        full_col_start = max(0, min(full_cols - 1, full_col_start))
        full_col_stop = max(full_col_start + 1, min(full_cols, full_col_stop))
        sample = (
            slice(full_row_start, full_row_stop),
            slice(full_col_start, full_col_stop),
        )

        full_image = self.full_image()
        if full_image.img.shape != self.baseline.img.shape:
            return _warn(
                "Pipette skipped because preview and baseline dimensions differ."
            )
        relative = np.subtract(full_image.img, self.baseline.img, dtype=float)
        characteristic_colors = darsia.extract_characteristic_data(
            signal=relative[sample], mask=label_mask[sample]
        )
        if len(characteristic_colors) == 0:
            return _warn("Pipette skipped because no characteristic color was found.")
        self.key_colors[self.selected] = np.asarray(
            characteristic_colors[0], dtype=float
        )
        self.preview_stale = True
        return None


class ColorPathEditorWindow:
    """Window of the color-path editor: built once, updated in place.

    Parameters
    ----------
    state : ColorPathEditorState
        Key colors and preview data.
    relative_colors : np.ndarray
        Relative colors of the calibration images in the label (3D panel).
    colors : np.ndarray
        Their absolute colors (dot colors).

    """

    def __init__(
        self,
        state: ColorPathEditorState,
        relative_colors: np.ndarray,
        colors: np.ndarray,
    ) -> None:
        self.state = state
        self.relative_colors = relative_colors
        self.colors = colors
        self.finalized = False
        """Whether the user finalized the path (otherwise the window was closed)."""
        self.buttons: dict[str, Button] = {}
        """Buttons by their label."""

    # ! ---- BUILDING ----

    def build(self) -> None:
        """Create the figure, panels and controls."""
        fig = plt.figure(figsize=(14, 8))
        self.fig = fig
        maximize_figure_window(fig)
        gs = fig.add_gridspec(
            nrows=2,
            ncols=2,
            width_ratios=[1, 1],
            height_ratios=[1, 1],
            left=0.05,
            right=0.98,
            bottom=0.30,
            top=0.91,
            wspace=0.1,
            hspace=0.12,
        )
        self.ax3d = fig.add_subplot(gs[:, 0], projection="3d")
        self.ax_image = fig.add_subplot(gs[0, 1])
        self.ax_interpretation = fig.add_subplot(gs[1, 1])
        self._build_colors()
        self._build_previews()
        self._build_controls()
        self.blitter.add_panels([self.ax3d, self.ax_image, self.ax_interpretation])
        self.runner = BackgroundRunner(
            fig,
            show_busy=lambda message, now: self.busy_status.set(message, now=now),
            show_refusal=lambda message: self.message.set(message or ""),
            refusal_shown=lambda: self.message.get_text() == BUSY_MESSAGE,
        )
        fig.canvas.mpl_connect("close_event", self._on_close)
        self._show_selected()

    def _build_colors(self) -> None:
        """3D panel: colors of the images, ignored colors, key colors and path."""
        state, ax = self.state, self.ax3d
        relative_colors = self.relative_colors
        ax.scatter(
            relative_colors[:, 0],
            relative_colors[:, 1],
            relative_colors[:, 2],
            c=np.clip(self.colors, 0, 1),
            s=10,
            alpha=0.5,
        )
        ax.plot(*np.array(relative_colors).T, c="gray", linewidth=1, alpha=0.5)
        if state.ignore is not None:
            ignored = state.ignore.relative_colors
            ax.scatter(
                ignored[:, 0], ignored[:, 1], ignored[:, 2], c="r", s=10, alpha=0.1
            )
        [self.path_line] = ax.plot([], [], [], c="black", linewidth=2)
        self.key_points = None
        ax.set_title(f"{state.name} (manual)")
        ax.set_xlabel("R")
        ax.set_ylabel("G")
        ax.set_zlabel("B")
        self._update_path()

    def _update_path(self) -> None:
        """Key colors and the path through them; limits fit all colors."""
        state, ax = self.state, self.ax3d
        keys = np.array(state.key_colors)
        if self.key_points is not None:
            self.key_points.remove()
            self.key_points = None
        if len(keys) >= 2:
            self.path_line.set_data_3d(*keys.T)
            self.key_points = ax.scatter(
                *keys.T, c=np.clip([state.base_color + c for c in keys], 0, 1), s=100
            )
        points = [self.relative_colors, keys.reshape(-1, 3)]
        if state.ignore is not None:
            points.append(state.ignore.relative_colors)
        points = np.concatenate(points)
        ax.auto_scale_xyz(points[:, 0], points[:, 1], points[:, 2], had_data=False)

    def _build_previews(self) -> None:
        """Calibration image (label highlighted) and its interpretation, zoom linked."""
        state = self.state
        ax_image, ax_interpretation = self.ax_image, self.ax_interpretation
        ax_image.axis("off")
        ax_interpretation.axis("off")
        self.image = None
        self.interpretation = None
        if not (state.has_preview and state.label is not None):
            for ax in (ax_image, ax_interpretation):
                ax.text(
                    0.5,
                    0.5,
                    "Manual preview unavailable.",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax_image.set_title("Calibration image")
            ax_interpretation.set_title("Native pH indicator interpretation")
            return
        image = state.highlighted_image()
        self.image = ScreenImage(
            ax_image, image, **({} if image.ndim == 3 else {"cmap": "gray"})
        )
        signal = state.preview_signal
        if signal is None:
            signal = np.full(state.coarse_shape, np.nan)
        self.interpretation = ScreenImage(
            ax_interpretation,
            signal,
            cmap=state.color_path().get_color_map(),
            vmin=0,
            vmax=1,
        )
        ax_interpretation.sharex(ax_image)
        ax_interpretation.sharey(ax_image)
        self._set_preview_titles()

    def _set_preview_titles(self) -> None:
        state = self.state
        self.ax_image.set_title(
            f"Calibration image {state.preview_idx + 1} / "
            f"{len(state.coarse_images)} (selected label focus)"
        )
        self.ax_interpretation.set_title("Native pH indicator interpretation")

    def _build_controls(self) -> None:
        fig, state = self.fig, self.state

        def add_button(name: str, rect: list[float]) -> None:
            self.buttons[name] = Button(fig.add_axes(rect), name)

        add_button("Reset (auto)", [0.01, 0.94, 0.09, 0.05])
        add_button("Finalize", [0.11, 0.94, 0.09, 0.05])

        # Key color: selection and channels (applied with Update)
        add_button("Previous key", [0.02, 0.235, 0.08, 0.045])
        add_button("Next key", [0.18, 0.235, 0.08, 0.045])
        key_text = fig.text(0.14, 0.2575, "", ha="center", va="center", fontsize=9)
        data = np.concatenate([self.relative_colors, np.array(state.key_colors)])
        margin = 0.1
        self.channel_sliders = []
        for row, (channel, low, high) in enumerate(
            zip("RGB", data.min(axis=0) - margin, data.max(axis=0) + margin)
        ):
            slider = Slider(
                fig.add_axes([0.32, 0.255 - row * 0.04, 0.2, 0.025]),
                channel,
                float(low),
                float(high),
                valinit=0.0,
                valstep=0.001,
                valfmt="%.3f",
            )
            self.channel_sliders.append(slider)
        add_button("Update", [0.58, 0.185, 0.08, 0.05])

        add_button("Pipette", [0.02, 0.12, 0.08, 0.05])
        add_button("+", [0.11, 0.12, 0.04, 0.05])
        add_button("-", [0.16, 0.12, 0.04, 0.05])
        self.align_box = TextBox(
            fig.add_axes([0.25, 0.12, 0.12, 0.05]), "Align", initial=""
        )
        # No submit when leaving the text box (Align applies it)
        self.align_box.canvas.mpl_disconnect(self.align_box._cids[-1])
        add_button("Align", [0.38, 0.12, 0.07, 0.05])

        # Preview: image and calculation
        add_button("Previous", [0.70, 0.20, 0.10, 0.06])
        add_button("Next", [0.82, 0.20, 0.10, 0.06])
        add_button("Re-run pH preview", [0.70, 0.12, 0.22, 0.06])

        # Messages (repainted alone)
        self.blitter = SliderBlitter(fig)
        self.blitter.add(self.channel_sliders)
        self.key_status = StatusText(key_text, self.blitter)
        """Selected key color (\"Key k / n\")."""
        self.preview_note = StatusText(
            fig.text(0.70, 0.095, "", fontsize=8, color="darkorange", va="top"),
            self.blitter,
        )
        """Note when the preview does not show the current key colors."""
        self.message = StatusText(
            fig.text(0.02, 0.08, "", fontsize=8, color="red", va="top"), self.blitter
        )
        """Warnings of the last action (e.g. pipette skipped)."""
        self.busy_status = StatusText(
            fig.text(0.98, 0.02, "", fontsize=8, color="darkorange", ha="right"),
            self.blitter,
        )
        """Message shown while the preview is computed."""

        self.buttons["Previous key"].on_clicked(lambda event: self._step_key(-1))
        self.buttons["Next key"].on_clicked(lambda event: self._step_key(1))
        self.buttons["Update"].on_clicked(self._on_update)
        self.buttons["Align"].on_clicked(self._on_align)
        self.buttons["+"].on_clicked(self._on_add)
        self.buttons["-"].on_clicked(self._on_remove)
        self.buttons["Reset (auto)"].on_clicked(self._on_reset)
        self.buttons["Pipette"].on_clicked(self._on_pipette)
        self.buttons["Previous"].on_clicked(lambda event: self._step_image(-1))
        self.buttons["Next"].on_clicked(lambda event: self._step_image(1))
        self.buttons["Re-run pH preview"].on_clicked(self._on_rerun)
        self.buttons["Finalize"].on_clicked(self._on_finalize)

    # ! ---- SHOWING ----

    def _show_selected(self) -> None:
        """Show the selected key color on the sliders (without callbacks)."""
        state = self.state
        self.key_status.set(f"Key {state.selected} / {len(state.key_colors) - 1}")
        for slider, value in zip(
            self.channel_sliders, state.key_colors[state.selected]
        ):
            slider.eventson = False
            try:
                slider.set_val(float(value))
            finally:
                slider.eventson = True

    def _keys_changed(self, message: str | None = None) -> None:
        """Show changed key colors: path, sliders, messages; repaint the 3D panel."""
        self._update_path()
        self._show_selected()
        self.message.set(message or "")
        if self.state.preview_stale:
            self.preview_note.set(
                "Preview does not show the current key colors: click Re-run pH preview."
            )
        self.blitter.redraw([self.ax3d])

    def _show_preview(self) -> None:
        """Show the computed preview (after Re-run, Next or Previous)."""
        state = self.state
        if self.image is not None:
            self.image.set_data(state.highlighted_image())
        if self.interpretation is not None:
            signal = state.preview_signal
            if signal is None:
                signal = np.full(state.coarse_shape, np.nan)
            self.interpretation.set_data(signal)
            self.interpretation.image.set_cmap(state.color_path().get_color_map())
        self._set_preview_titles()
        self.preview_note.set("")
        self.blitter.redraw([self.ax_image, self.ax_interpretation])

    # ! ---- ACTIONS ----

    def _step_key(self, step: int) -> None:
        state = self.state
        state.selected = max(0, min(len(state.key_colors) - 1, state.selected + step))
        self._show_selected()

    def _on_update(self, event) -> None:
        if self.runner.refuse():
            return
        self.state.set_color(
            np.array([slider.val for slider in self.channel_sliders], dtype=float)
        )
        self._keys_changed()

    def _on_align(self, event) -> None:
        if self.runner.refuse():
            return
        self._keys_changed(self.state.align(self.align_box.text))

    def _on_add(self, event) -> None:
        if self.runner.refuse():
            return
        self.state.add()
        self._keys_changed()

    def _on_remove(self, event) -> None:
        if self.runner.refuse():
            return
        message = self.state.remove()
        if message is None:
            self._keys_changed()
        else:
            self.message.set(message)

    def _on_reset(self, event) -> None:
        if self.runner.refuse():
            return
        self.state.reset()
        self._keys_changed()

    def _on_pipette(self, event) -> None:
        if self.runner.refuse():
            return
        message = self.state.pipette(self.ax_image.get_xlim(), self.ax_image.get_ylim())
        if message is None:
            self._keys_changed()
        else:
            self.message.set(message)

    def _on_rerun(self, event) -> None:
        if self.runner.refuse():
            return
        state = self.state
        self.runner.run(
            "Computing preview",
            lambda progress: state.rerun_preview(),
            self._show_preview,
        )

    def _step_image(self, step: int) -> None:
        if self.runner.refuse() or len(self.state.coarse_images) == 0:
            return
        state = self.state
        self.runner.run(
            "Computing preview",
            lambda progress: state.step_image(step),
            self._show_preview,
        )

    def _on_finalize(self, event) -> None:
        self.finalized = True
        plt.close(self.fig)

    def _on_close(self, event) -> None:
        self.runner.wait()

    def run(self) -> bool:
        """Show the window until it is closed.

        Returns
        -------
        bool
            Whether the user finalized the path.

        """
        plt.show()
        self.runner.shutdown()
        return self.finalized


def edit_color_path(
    *,
    relative_colors: np.ndarray,
    colors: np.ndarray,
    key_relative_colors: list[np.ndarray],
    base_color: np.ndarray,
    ignore: darsia.ColorSpectrum | None,
    auto_fit: Callable[[int], list[np.ndarray]],
    default_num_segments: int,
    name: str,
    label: int | None,
    preview_image: darsia.Image | None,
    preview_images: list[darsia.Image] | None,
    preview_baseline: darsia.Image | None,
    labels: darsia.Image,
    mask: darsia.Image,
) -> list[np.ndarray]:
    """Interactively adjust the key relative colors of a label's color path.

    Parameters
    ----------
    relative_colors : np.ndarray
        Relative colors of the calibration images in the label (shown in 3D).
    colors : np.ndarray
        Their absolute colors (dot colors).
    key_relative_colors : list[np.ndarray]
        Key colors at the start.
    base_color : np.ndarray
        Base color of the label.
    ignore : darsia.ColorSpectrum, optional
        Ignored spectrum (shown in 3D, used by the preview).
    auto_fit : Callable
        Automatic fit with a given number of segments (Reset).
    default_num_segments : int
        Number of segments of the automatic fit.
    name : str
        Name of the color path.
    label : int, optional
        The label.
    preview_image : darsia.Image, optional
        Calibration image for the preview (if `preview_images` is empty).
    preview_images : list[darsia.Image], optional
        Calibration images for the preview.
    preview_baseline : darsia.Image, optional
        Baseline image for the relative colors of the preview.
    labels : darsia.Image
        Labels.
    mask : darsia.Image
        Calibration mask.

    Returns
    -------
    list[np.ndarray]
        The key relative colors at the end.

    """
    if len(key_relative_colors) == 0:
        return []
    candidates = list(preview_images or [])
    if len(candidates) == 0 and preview_image is not None:
        candidates = [preview_image]
    state = ColorPathEditorState(
        key_relative_colors=key_relative_colors,
        base_color=base_color,
        ignore=ignore,
        auto_fit=auto_fit,
        default_num_segments=default_num_segments,
        name=name,
        label=label,
        preview_images=candidates,
        preview_baseline=preview_baseline,
        labels=labels,
        mask=mask,
    )
    state.rerun_preview()
    window = ColorPathEditorWindow(state, relative_colors, colors)
    window.build()
    if not window.run():
        logger.info("Manual color-path editor closed without explicit finalize.")
    return state.key_colors
