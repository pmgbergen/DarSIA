"""Interactive editor of the color paths of all labels (manual color-path calibration).

A color path runs through key colors, relative to the baseline color, in RGB space. One
window shows, for the selected label, the colors of the calibration images (3D) with its
key colors and path, a calibration image with the label highlighted, and the
pH-indicator interpretation of the whole domain (each label with its own path). The user
selects a label by a click (or Next Label), moves, adds, removes and aligns key colors,
samples one from the image (pipette) or fits the path automatically (Auto: with the
current number of key colors; Reset: with the number of the config), and saves. The
pH view follows every change (computed in the background; only the edited label). It
shows the signals (Next Signal): all labels in the colormap of the reference label's
path (first, if there is a reference), each label in the colormap of its own path, and
the calibration mask in the colours of the shown image (white outside). The pH signals
cover the whole labels, as the analysis does; the paths are fitted, and the pipette
samples, within the calibration mask only.

The R, G, B sliders show the selected key color (relative to the baseline; range
symmetric around 0 and covering the label's colors); a change applies while dragging.

Like the mass calibration window: changes are tracked, Save stores all paths, closing
with unsaved changes asks whether to save them.

`ColorPathEditorState` holds the data and the operations (no matplotlib);
`ColorPathEditorWindow` is the window; `edit_color_paths` runs it.

"""

from __future__ import annotations

import logging
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import skimage
from matplotlib.widgets import Button, Slider, TextBox

import darsia
from darsia.presets.workflows.windows.background import BUSY_MESSAGE, BackgroundRunner
from darsia.presets.workflows.windows.blitting import SliderBlitter, StatusText
from darsia.presets.workflows.windows.highlight import LUMINANCE, highlight_region
from darsia.presets.workflows.windows.screen_image import ScreenImage
from darsia.signals.color.color_path_regression import ColorPathInputs
from darsia.utils.figure_window import ask_yes_no, maximize_figure_window

logger = logging.getLogger(__name__)

PREVIEW_SIZE = 600
"""Largest size (pixels, long side) of the preview images: about the panel's size on
screen; the previews are for display, colors for the paths come from full resolution."""

MASK_THRESHOLD = 0.5
"""Threshold of the interpolated calibration mask on the preview grid."""

SLIDER_STEP = 0.001
"""Step of the channel sliders (relative color)."""

MIN_SLIDER_RANGE = 0.01
"""Smallest half-width of a channel slider (for a channel with almost no colors)."""

SIGNAL_TITLES = {
    "reference": "colormap of label {reference} (reference)",
    "own": "own colormap of each label",
    "mask": "calibration mask - colours of the image",
}
"""Signals of the pH view (first: reference, if any) and their titles."""

UNSAVED_TEXT = "Changes not saved."
"""Save status while the shown paths differ from the saved ones."""


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


class LabelPath:
    """Key colors of one label's color path and the operations on them.

    Parameters
    ----------
    inputs : ColorPathInputs
        Data of the label (spectrum, start key colors, automatic fit).

    """

    def __init__(self, inputs: ColorPathInputs) -> None:
        self.inputs = inputs
        self.key_colors = [
            np.array(c, dtype=float).copy() for c in inputs.key_relative_colors
        ]
        """Key colors (relative), edited in place."""
        self.selected = 0
        """Index of the selected key color."""
        self.saved = [c.copy() for c in self.key_colors]
        """Key colors when last saved (or at the start)."""
        self.preview: np.ndarray | None = None
        """Interpretation of the shown image with this path (NaN outside the label)."""

    def color_path(
        self, key_colors: list[np.ndarray] | None = None
    ) -> darsia.ColorPath:
        """Color path through the current key colors (or through given ones)."""
        return darsia.ColorPath(
            colors=None,
            relative_colors=self.key_colors if key_colors is None else key_colors,
            base_color=self.inputs.base_color,
            mode="rgb",
            name=self.inputs.name,
        )

    def changed(self) -> bool:
        """Whether the key colors differ from the saved ones."""
        return len(self.key_colors) != len(self.saved) or any(
            not np.array_equal(a, b) for a, b in zip(self.key_colors, self.saved)
        )

    def set_color(self, color: np.ndarray) -> None:
        """Set the selected key color."""
        self.key_colors[self.selected] = color

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
        return "; ".join(messages) or None

    def add(self) -> None:
        """Insert a key color halfway after the selected one (before the last)."""
        n = len(self.key_colors)
        i = self.selected if self.selected < n - 1 else n - 2
        midpoint = 0.5 * (self.key_colors[i] + self.key_colors[i + 1])
        self.key_colors.insert(i + 1, midpoint)
        self.selected = i + 1

    def remove(self) -> str | None:
        """Remove the selected key color; returns a message if only two are left."""
        if len(self.key_colors) <= 2:
            return _warn("Cannot remove: a color path needs at least 2 colors.")
        del self.key_colors[self.selected]
        self.selected = min(self.selected, len(self.key_colors) - 1)
        return None

    def fit(self, segments: int) -> tuple[list[np.ndarray], str | None]:
        """Automatic fit with a number of segments, without changing the path (e.g.
        in the background); returns the key colors and a message if the spectrum
        gives fewer segments."""
        keys = [np.array(c, dtype=float) for c in self.inputs.auto_fit(segments)]
        found = len(keys) - 1
        if found < segments:
            return keys, _warn(
                f"Automatic fit found only {found} of {segments} segments (the "
                "spectrum of the label cannot be split further)."
            )
        return keys, None

    def set_keys(self, keys: list[np.ndarray], selected: int | None = None) -> None:
        """Replace the key colors; select a key (default: keep the selection)."""
        self.key_colors[:] = keys
        self.selected = min(
            self.selected if selected is None else selected, len(keys) - 1
        )

    def reset(self) -> None:
        """Return to the automatic fit (number of segments of the config)."""
        keys, _ = self.fit(self.inputs.num_segments)
        self.set_keys(keys, 0)

    def auto(self) -> str | None:
        """Automatic fit with the current number of key colors; returns a message if
        the spectrum gives fewer segments."""
        keys, message = self.fit(len(self.key_colors) - 1)
        self.set_keys(keys)
        return message


class ColorPathEditorState:
    """Color paths of all labels and the preview data (no plotting).

    Parameters
    ----------
    prepared : dict
        Per label `ColorPathInputs` (editable) or a fixed `darsia.ColorPath`
        (`LabelColorPathMapRegression.prepare_color_path_editing`).
    preview_images : list[darsia.Image]
        Calibration images (on the grid of `labels`); images in an integer type (see
        `native` in `darsia.imread_from_optical`) are converted to floats where needed.
    preview_baseline : darsia.Image, optional
        Baseline image (on the grid of `labels`).
    labels : darsia.Image
        Labels.
    mask : darsia.Image
        Calibration mask (boolean).
    stored : bool
        Whether the paths at the start are the stored ones; otherwise they count as
        unsaved.
    reference_label : int, optional
        Label whose path gives the colormap of the reference signal of the pH view.

    """

    def __init__(
        self,
        prepared: dict,
        preview_images: list[darsia.Image],
        preview_baseline: darsia.Image | None,
        labels: darsia.Image,
        mask: darsia.Image,
        stored: bool = False,
        reference_label: int | None = None,
    ) -> None:
        self.paths = {
            label: LabelPath(item)
            for label, item in sorted(prepared.items())
            if isinstance(item, ColorPathInputs)
        }
        """Editable paths by label."""
        self.fixed_paths = {
            label: item
            for label, item in prepared.items()
            if isinstance(item, darsia.ColorPath)
        }
        """Paths of the labels which are not edited."""
        self.reference_label = reference_label
        self.signals = ["own", "mask"]
        """Signals of the pH view (`SIGNAL_TITLES`)."""
        if reference_label in self.paths or reference_label in self.fixed_paths:
            self.signals = ["reference", "own", "mask"]
        elif reference_label is not None:
            logger.warning(
                "Reference label %s has no color path; the pH view shows each label "
                "in its own colormap only.",
                reference_label,
            )
        self.signal_idx = 0
        """Index of the shown signal."""
        self.label = next(iter(self.paths), None)
        """Selected label."""
        self.labels = labels
        self.mask = mask
        self.stored = stored
        self.preview_idx = 0
        """Index of the shown calibration image."""
        self.picture: np.ndarray | None = None
        """Interpretation of the whole domain as colors (`update_picture`)."""

        self.images = [
            img for img in preview_images if img.img.shape[:2] == labels.img.shape
        ]
        """Calibration images on the grid of the labels."""
        self.baseline = preview_baseline
        self.has_preview = (
            preview_baseline is not None
            and preview_baseline.img.shape[:2] == labels.img.shape
            and len(self.images) > 0
        )
        """Whether a preview is possible (images and baseline on the label grid)."""
        if (
            len(preview_images) > 0
            and preview_baseline is not None
            and not (self.has_preview)
        ):
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
            # Display only: about screen size (the pipette samples full resolution)
            full_rows, full_cols = labels.img.shape[:2]
            scale = min(1.0, PREVIEW_SIZE / max(full_rows, full_cols))
            rows = max(1, int(round(full_rows * scale)))
            cols = max(1, int(round(full_cols * scale)))
            self.coarse_shape = (rows, cols)
            # Area averages: each preview pixel shows the mean color it covers
            self.coarse_images = [
                darsia.resize(
                    img.as_float(), shape=self.coarse_shape, interpolation="inter_area"
                )
                for img in self.images
            ]
            self.coarse_baseline = darsia.resize(
                preview_baseline, shape=self.coarse_shape, interpolation="inter_area"
            )
            self.coarse_labels = darsia.resize(
                labels, shape=self.coarse_shape, interpolation="inter_nearest"
            )
            float_mask = darsia.ScalarImage(mask.img.astype(float), **mask.metadata())
            resized_mask = darsia.resize(float_mask, shape=self.coarse_shape)
            self.coarse_mask = resized_mask.img > MASK_THRESHOLD

    # ! ---- DATA ----

    @property
    def path(self) -> LabelPath:
        """Path of the selected label."""
        return self.paths[self.label]

    def label_region(self, label: int) -> np.ndarray:
        """Pixels of a label (preview grid), with or without calibration mask."""
        assert self.coarse_labels is not None
        return self.coarse_labels.img == label

    def label_mask(self, label: int) -> np.ndarray:
        """Pixels of a label within the calibration mask (preview grid)."""
        assert self.coarse_labels is not None and self.coarse_mask is not None
        return np.logical_and(self.coarse_labels.img == label, self.coarse_mask)

    def label_mask_full(self, label: int) -> np.ndarray:
        """Pixels of a label within the calibration mask (full grid)."""
        return np.logical_and(self.labels.img == label, self.mask.img)

    def coarse_image(self) -> darsia.Image:
        """Shown calibration image (preview grid)."""
        return self.coarse_images[self.preview_idx]

    def full_image(self) -> darsia.Image:
        """Shown calibration image (full grid)."""
        return self.images[self.preview_idx]

    def keys(self) -> dict[int, list[np.ndarray]]:
        """Key colors of all editable labels."""
        return {label: path.key_colors for label, path in self.paths.items()}

    def key_snapshot(self) -> dict[int, list[np.ndarray]]:
        """Copy of the key colors of all editable labels (for a background
        calculation)."""
        return {
            label: [c.copy() for c in path.key_colors]
            for label, path in self.paths.items()
        }

    @property
    def signal(self) -> str:
        """Shown signal of the pH view."""
        return self.signals[self.signal_idx]

    def signal_title(self) -> str:
        """Title of the shown signal."""
        return SIGNAL_TITLES[self.signal].format(reference=self.reference_label)

    def next_signal(self) -> str:
        """Move to the next signal (cyclic)."""
        self.signal_idx = (self.signal_idx + 1) % len(self.signals)
        return self.signal

    def colormap(self, label: int, key_colors: dict | None = None):
        """Colormap of a label in the shown signal: its own path, or the reference
        path (through the key colors of `key_colors` if given)."""
        if self.signal == "reference":
            label = self.reference_label
            if label in self.fixed_paths:
                return self.fixed_paths[label].get_color_map()
        keys = None if key_colors is None else key_colors[label]
        return self.paths[label].color_path(keys).get_color_map()

    def has_unsaved_changes(self) -> bool:
        """Whether the paths differ from the saved ones (or were never saved)."""
        return not self.stored or any(path.changed() for path in self.paths.values())

    def mark_saved(self) -> None:
        """The current paths are the stored ones."""
        for path in self.paths.values():
            path.saved = [c.copy() for c in path.key_colors]
        self.stored = True

    def compute_preview(
        self, label: int, key_colors: list[np.ndarray] | None = None
    ) -> np.ndarray | None:
        """pH-indicator interpretation of the shown image in a label with its path
        (through `key_colors` if given, e.g. a snapshot for a background calculation),
        on the whole label: the calibration mask only limits the fit.
        """
        if not self.has_preview:
            return None
        assert self.coarse_baseline is not None
        path = self.paths[label]
        label_mask = self.label_region(label)
        if not np.any(label_mask):
            return np.full(self.coarse_shape, np.nan, dtype=float)
        relative = self.coarse_image().img.astype(
            float
        ) - self.coarse_baseline.img.astype(float)
        selected = relative[label_mask].reshape((-1, 3))
        if selected.shape[0] == 0:
            return np.full(self.coarse_shape, np.nan, dtype=float)
        color_path = path.color_path(key_colors)
        interpolation = darsia.ColorPathInterpolation(
            color_path=color_path,
            color_mode=darsia.ColorMode.RELATIVE,
            values=color_path.equidistant_distances,
            ignore_spectrum=path.inputs.ignore,
        )
        interpretation = np.full(self.coarse_shape, np.nan, dtype=float)
        interpretation[label_mask] = interpolation(selected)
        return interpretation

    def rerun_preview(
        self, label: int | None = None, key_colors: list[np.ndarray] | None = None
    ) -> None:
        """Compute the preview of a label (default: the selected one)."""
        label = self.label if label is None else label
        self.paths[label].preview = self.compute_preview(label, key_colors)

    def rerun_all_previews(self, key_colors: dict | None = None) -> None:
        """Compute the previews of all labels (e.g. for another image)."""
        for label in self.paths:
            self.rerun_preview(label, None if key_colors is None else key_colors[label])

    def step_image(self, step: int) -> None:
        """Show another calibration image (cyclic) and compute all previews."""
        if len(self.coarse_images) == 0:
            return
        self.preview_idx = (self.preview_idx + step) % len(self.coarse_images)
        self.rerun_all_previews()

    def _paint_label(
        self, picture: np.ndarray, label: int, key_colors: dict | None
    ) -> None:
        path = self.paths[label]
        if path.preview is None:
            return
        valid = ~np.isnan(path.preview)
        colormap = self.colormap(label, key_colors)
        picture[valid] = colormap(np.clip(path.preview[valid], 0, 1))[:, :3]

    def mask_picture(self) -> np.ndarray:
        """The calibration mask in the colours of the shown image; white outside."""
        picture = np.ones((*self.coarse_shape, 3))
        if self.coarse_mask is None or len(self.coarse_images) == 0:
            return picture
        image = self.coarse_image().img
        picture[self.coarse_mask] = np.clip(image[self.coarse_mask], 0, 1)
        return picture

    def domain_picture(self, key_colors: dict | None = None) -> np.ndarray:
        """Interpretation of the whole domain as colors, in the colormaps of the shown
        signal; white where there is no interpretation. The signal "mask" shows the
        calibration mask instead (`mask_picture`). `key_colors`: per label (e.g. a
        snapshot), default the current ones."""
        if self.signal == "mask":
            return self.mask_picture()
        picture = np.ones((*self.coarse_shape, 3))
        for label in self.paths:
            self._paint_label(picture, label, key_colors)
        return picture

    def update_picture(
        self, label: int | None = None, key_colors: dict | None = None
    ) -> None:
        """Update the cached domain picture: one label (its region recolored), or all
        labels (`label` None). `key_colors`: per label, default the current ones. The
        mask picture does not depend on the key colors and stays as it is."""
        if label is None or self.picture is None:
            self.picture = self.domain_picture(key_colors)
            return
        if self.signal == "mask":
            return
        picture = self.picture.copy()
        if self.coarse_labels is not None:
            picture[self.label_region(label)] = 1.0
        self._paint_label(picture, label, key_colors)
        self.picture = picture

    def highlighted_image(self) -> np.ndarray:
        """Shown image with everything outside the selected label turned gray."""
        image = self.coarse_image().img.copy().astype(float)
        if image.ndim != 3:
            return image
        return highlight_region(image, ~self.label_mask(self.label), LUMINANCE)

    # ! ---- SELECTION ----

    def select_label_at(self, row: int, col: int) -> int | None:
        """Select the label at a pixel of the preview grid, if its path is editable."""
        if self.coarse_labels is None:
            return None
        rows, cols = self.coarse_labels.img.shape[:2]
        label = int(
            self.coarse_labels.img[
                min(max(row, 0), rows - 1), min(max(col, 0), cols - 1)
            ]
        )
        if label not in self.paths:
            return None
        self.label = label
        return label

    def next_label(self) -> int:
        """Select the next editable label (cyclic)."""
        labels = list(self.paths)
        self.label = labels[(labels.index(self.label) + 1) % len(labels)]
        return self.label

    # ! ---- PIPETTE ----

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
        if not self.has_preview:
            return _warn("Pipette unavailable without preview calibration data.")
        assert self.baseline is not None
        path = self.path
        label_mask = self.label_mask_full(self.label)
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
        colors = full_image.img[sample]
        if not np.issubdtype(colors.dtype, np.floating):
            colors = skimage.img_as_float(colors)  # as `Image.as_float`
        relative = np.subtract(colors, self.baseline.img[sample], dtype=float)
        characteristic_colors = darsia.extract_characteristic_data(
            signal=relative, mask=label_mask[sample]
        )
        if len(characteristic_colors) == 0:
            return _warn("Pipette skipped because no characteristic color was found.")
        path.key_colors[path.selected] = np.asarray(
            characteristic_colors[0], dtype=float
        )
        return None


class ColorPathEditorWindow:
    """Window of the color-path editor: built once, updated in place.

    Parameters
    ----------
    state : ColorPathEditorState
        Paths and preview data.
    save : Callable, optional
        Stores the paths of all labels (argument: key colors per label).

    """

    def __init__(
        self,
        state: ColorPathEditorState,
        save: Callable[[dict[int, list[np.ndarray]]], None] | None = None,
    ) -> None:
        self.state = state
        self.save = save
        self.buttons: dict[str, Button] = {}
        """Buttons by their label."""
        self._dragging = False
        """Whether a channel slider is dragged (the 3D limits then only grow)."""
        self._queued: set[int] = set()
        """Labels whose preview waits for the running calculation."""

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
        self.key_points = None
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
        fig.canvas.mpl_connect("button_press_event", self._on_click)
        fig.canvas.mpl_connect("button_release_event", self._on_release)
        fig.canvas.mpl_connect("close_event", self._on_close)
        self._set_slider_ranges()
        self._show_selected()
        if not self.state.stored:
            self.save_status.set(UNSAVED_TEXT, "darkorange")

    def _build_colors(self) -> None:
        """3D panel of the selected label: colors of the images, ignored colors, key
        colors and path."""
        state, ax = self.state, self.ax3d
        inputs = state.path.inputs
        ax.cla()
        self.key_points = None
        relative_colors = inputs.relative_colors
        ax.scatter(
            relative_colors[:, 0],
            relative_colors[:, 1],
            relative_colors[:, 2],
            c=np.clip(inputs.colors, 0, 1),
            s=10,
            alpha=0.5,
        )
        ax.plot(*np.array(relative_colors).T, c="gray", linewidth=1, alpha=0.5)
        if inputs.ignore is not None:
            ignored = inputs.ignore.relative_colors
            ax.scatter(
                ignored[:, 0], ignored[:, 1], ignored[:, 2], c="r", s=10, alpha=0.1
            )
        [self.path_line] = ax.plot([], [], [], c="black", linewidth=2)
        ax.set_title(f"{inputs.name} (manual)")
        ax.set_xlabel("R")
        ax.set_ylabel("G")
        ax.set_zlabel("B")
        self._update_path()

    def _update_path(self) -> None:
        """Key colors and the path through them; limits fit all colors (while a slider
        is dragged they only grow, when a key color leaves the view)."""
        state, ax = self.state, self.ax3d
        inputs = state.path.inputs
        keys = np.array(state.path.key_colors)
        if len(keys) >= 2:
            self.path_line.set_data_3d(*keys.T)
        self._draw_keys()
        points = [inputs.relative_colors, keys.reshape(-1, 3)]
        if self._dragging:
            limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
            if np.all((keys >= limits[:, 0]) & (keys <= limits[:, 1])):
                return
            points.append(limits.T)
        if inputs.ignore is not None:
            points.append(inputs.ignore.relative_colors)
        points = np.concatenate(points)
        ax.auto_scale_xyz(points[:, 0], points[:, 1], points[:, 2], had_data=False)

    def _draw_keys(self) -> None:
        """Key colors in their colors; the selected one larger with a black ring."""
        path = self.state.path
        if self.key_points is not None:
            self.key_points.remove()
            self.key_points = None
        keys = np.array(path.key_colors)
        if len(keys) < 2:
            return
        selected = np.arange(len(keys)) == path.selected
        self.key_points = self.ax3d.scatter(
            *keys.T,
            c=np.clip([path.inputs.base_color + c for c in keys], 0, 1),
            s=np.where(selected, 220, 100),
            edgecolors=["black" if s else "none" for s in selected],
            linewidths=np.where(selected, 2.0, 0.0),
        )

    def _build_previews(self) -> None:
        """Calibration image (label highlighted) and the interpretation of the whole
        domain; zoom linked, a click selects the label."""
        state = self.state
        ax_image, ax_interpretation = self.ax_image, self.ax_interpretation
        ax_image.axis("off")
        ax_interpretation.axis("off")
        self.image = None
        self.interpretation = None
        if not state.has_preview:
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
        self.interpretation = ScreenImage(ax_interpretation, state.picture)
        ax_interpretation.sharex(ax_image)
        ax_interpretation.sharey(ax_image)
        self._set_preview_titles()

    def _set_preview_titles(self) -> None:
        state = self.state
        self.ax_image.set_title(
            f"Calibration image {state.preview_idx + 1} / "
            f"{len(state.coarse_images)} - label {state.label} (click selects a label)"
        )
        self.ax_interpretation.set_title(
            f"pH indicator interpretation - {self.state.signal_title()}"
        )

    def _build_controls(self) -> None:
        fig = self.fig

        def add_button(name: str, rect: list[float]) -> None:
            self.buttons[name] = Button(fig.add_axes(rect), name)

        # Top: Save with its status (left), navigation (right) as in the mass
        # calibration window
        add_button("Save", [0.02, 0.95, 0.08, 0.04])
        add_button("Next Signal", [0.72, 0.95, 0.08, 0.04])
        add_button("Next Label", [0.81, 0.95, 0.08, 0.04])
        add_button("Next Image", [0.90, 0.95, 0.08, 0.04])

        # Below the panels, two groups: the key color under the 3D panel (left), the
        # operations on the path under the image and pH view (right)
        left, right = 0.05, 0.54
        title_y = 0.255
        rows = [0.195, 0.145, 0.095]
        height = 0.04
        for x, title in ((left, "Key color"), (right, "Path")):
            fig.text(x, title_y, title, fontsize=9, fontweight="bold", va="center")

        # Key color: which key, selection, pipette (sets the selected key from the
        # zoomed region of the image), and its channels (applied while dragging)
        key_text = fig.text(left, rows[0] + 0.5 * height, "", ha="left", va="center")
        add_button("Previous key", [left + 0.08, rows[0], 0.09, height])
        add_button("Next key", [left + 0.18, rows[0], 0.09, height])
        add_button("Pipette (zoom)", [left + 0.28, rows[0], 0.11, height])
        self.channel_sliders = [
            Slider(
                fig.add_axes([left + 0.03, y, 0.33, 0.025]),
                channel,
                -MIN_SLIDER_RANGE,
                MIN_SLIDER_RANGE,
                valinit=0.0,
                valstep=SLIDER_STEP,
                valfmt="%+.3f",
            )
            for y, channel in zip([0.15, 0.11, 0.07], "RGB")
        ]

        # Path: automatic fit, keys, alignment
        add_button("Auto", [right, rows[0], 0.1, height])
        add_button("Reset (auto)", [right + 0.11, rows[0], 0.1, height])
        add_button("Add key", [right, rows[1], 0.1, height])
        add_button("Remove key", [right + 0.11, rows[1], 0.1, height])
        self.align_box = TextBox(
            fig.add_axes([right + 0.04, rows[2], 0.1, height]), "Keys", initial=""
        )
        # No submit when leaving the text box (Align applies it)
        self.align_box.canvas.mpl_disconnect(self.align_box._cids[-1])
        add_button("Align", [right + 0.15, rows[2], 0.06, height])

        # Messages (repainted alone)
        self.blitter = SliderBlitter(fig)
        self.blitter.add(self.channel_sliders)
        self.key_status = StatusText(key_text, self.blitter)
        """Selected key color ("Key k / n")."""
        self.save_status = StatusText(
            fig.text(0.11, 0.97, "", fontsize=9, color="green", va="center"),
            self.blitter,
        )
        """Result of saving, or that there are unsaved changes."""
        self.message = StatusText(
            fig.text(left, 0.045, "", fontsize=8, color="red", va="top"), self.blitter
        )
        """Warnings of the last action (e.g. pipette skipped)."""
        self.busy_status = StatusText(
            fig.text(0.98, 0.02, "", fontsize=8, color="darkorange", ha="right"),
            self.blitter,
        )
        """Message shown while the preview is computed."""

        for slider in self.channel_sliders:
            slider.on_changed(self._on_slider)
        self.buttons["Save"].on_clicked(self._on_save)
        self.buttons["Next Signal"].on_clicked(self._on_next_signal)
        self.buttons["Next Label"].on_clicked(self._on_next_label)
        self.buttons["Next Image"].on_clicked(self._on_next_image)
        self.buttons["Previous key"].on_clicked(lambda event: self._step_key(-1))
        self.buttons["Next key"].on_clicked(lambda event: self._step_key(1))
        self.buttons["Align"].on_clicked(self._on_align)
        self.buttons["Add key"].on_clicked(self._on_add)
        self.buttons["Remove key"].on_clicked(self._on_remove)
        self.buttons["Reset (auto)"].on_clicked(self._on_reset)
        self.buttons["Auto"].on_clicked(self._on_auto)
        self.buttons["Pipette (zoom)"].on_clicked(self._on_pipette)

    # ! ---- SHOWING ----

    def _set_slider_ranges(self) -> None:
        """Ranges of the channel sliders: [-a, a] per channel with a the largest
        absolute value of the selected label's colors and key colors (at most 1), on
        the slider step."""
        path = self.state.path
        colors = np.concatenate(
            [
                np.asarray(path.inputs.relative_colors).reshape(-1, 3),
                np.array(path.key_colors).reshape(-1, 3),
            ]
        )
        extent = np.max(np.abs(colors), axis=0) if len(colors) > 0 else np.zeros(3)
        for slider, value in zip(self.channel_sliders, extent):
            self._set_range(slider, value)

    @staticmethod
    def _set_range(slider: Slider, value: float) -> None:
        """Slider range [-a, a] containing a value (a on the slider step, in
        [MIN_SLIDER_RANGE, 1])."""
        steps = np.ceil(np.clip(abs(value), MIN_SLIDER_RANGE, 1.0) / SLIDER_STEP)
        half = float(steps * SLIDER_STEP)
        slider.valmin, slider.valmax = -half, half
        slider.ax.set_xlim(-half, half)
        slider.poly.set_x(-half)
        slider.poly.set_width(slider.val + half)

    def _show_selected(self) -> None:
        """Show which key color is selected and its value on the sliders (without
        callbacks); a slider range grows to a value outside it."""
        path = self.state.path
        self.key_status.set(f"Key {path.selected} / {len(path.key_colors) - 1}")
        resized = []
        for slider, value in zip(self.channel_sliders, path.key_colors[path.selected]):
            value = float(value)
            if not slider.valmin <= value <= slider.valmax:
                self._set_range(slider, value)
                resized.append(slider.ax)
            if value == slider.val and slider.ax not in resized:
                continue
            slider.eventson = False
            try:
                slider.set_val(value)
            finally:
                slider.eventson = True
            # Without callbacks the slider does not repaint itself
            if slider.ax not in resized:
                self.blitter.blit(slider)
        if resized:
            self.blitter.redraw(resized)

    def _mark_unsaved(self) -> None:
        if self.save_status.get_text() != UNSAVED_TEXT:
            self.save_status.set(UNSAVED_TEXT, "darkorange")

    def _keys_changed(self, message: str | None = None) -> None:
        """Show changed key colors: path, sliders, messages; repaint the 3D panel and
        compute the preview."""
        self._update_path()
        self._show_selected()
        self.message.set(message or "")
        self._mark_unsaved()
        self.blitter.redraw([self.ax3d])
        self._request_preview(self.state.label)

    def _request_preview(self, label: int) -> None:
        """Compute the preview of a label in the background with its current key
        colors; while another calculation runs, it is queued (latest colors win)."""
        if self.interpretation is None:
            return
        if self.runner.busy:
            self._queued.add(label)
            return
        state = self.state
        key_colors = state.key_snapshot()
        # In the reference signal the reference path colors all labels
        recolor_all = state.signal == "reference" and label == state.reference_label

        def work(progress) -> None:
            state.rerun_preview(label, key_colors[label])
            state.update_picture(None if recolor_all else label, key_colors)

        self.runner.run("Computing preview", work, self._show_interpretation)

    def _show_interpretation(self) -> None:
        """Show the domain picture after a preview of one label; run a queued one."""
        self.interpretation.set_data(self.state.picture)
        self.blitter.redraw([self.ax_interpretation])
        self._run_queued()

    def _run_queued(self) -> None:
        if self._queued:
            self._request_preview(self._queued.pop())

    def _show_previews(self) -> None:
        """Show the calibration image and the interpretation of the whole domain."""
        state = self.state
        if self.image is not None:
            self.image.set_data(state.highlighted_image())
        if self.interpretation is not None:
            self.interpretation.set_data(state.picture)
        self._set_preview_titles()
        self.blitter.redraw([self.ax_image, self.ax_interpretation])
        self._run_queued()

    def _switch_label(self) -> None:
        """Show the label selected in the state."""
        self.message.set("")
        self._set_slider_ranges()
        self._build_colors()
        self._show_selected()
        if self.image is not None:
            self.image.set_data(self.state.highlighted_image())
        self._set_preview_titles()
        self.blitter.redraw(
            [self.ax3d, self.ax_image] + [slider.ax for slider in self.channel_sliders]
        )

    # ! ---- ACTIONS ----

    def _on_slider(self, *args) -> None:
        """Selected key color = slider values, within [-1, 1] (relative to the
        baseline of each pixel)."""
        self._dragging = True
        values = np.array([slider.val for slider in self.channel_sliders], dtype=float)
        self.state.path.set_color(np.clip(values, -1.0, 1.0))
        self._keys_changed()

    def _on_release(self, event) -> None:
        """End of a slider drag: the 3D view fits all colors."""
        if not self._dragging:
            return
        self._dragging = False
        self._update_path()
        self.blitter.redraw([self.ax3d])

    def _step_key(self, step: int) -> None:
        path = self.state.path
        selected = max(0, min(len(path.key_colors) - 1, path.selected + step))
        if selected == path.selected:
            return
        path.selected = selected
        self._show_selected()
        self._draw_keys()
        self.blitter.redraw([self.ax3d])

    def _on_click(self, event) -> None:
        """A click in an image selects the label at that pixel."""
        if event.inaxes not in (self.ax_image, self.ax_interpretation):
            return
        if event.button != 1 or self.image is None:
            return
        toolbar = self.fig.canvas.toolbar
        if toolbar is not None and toolbar.mode:
            return  # zooming or panning
        if self.runner.refuse():
            return
        row, col = int(round(event.ydata)), int(round(event.xdata))
        previous = self.state.label
        if self.state.select_label_at(row, col) is None:
            self.message.set("This label has no editable color path.")
            return
        if self.state.label != previous:
            self._switch_label()

    def _on_next_label(self, event) -> None:
        if self.runner.refuse():
            return
        self.state.next_label()
        self._switch_label()

    def _on_next_image(self, event) -> None:
        if self.runner.refuse() or len(self.state.coarse_images) == 0:
            return
        state = self.state
        # The image index changes here (the pipette uses it); the previews of all
        # labels are computed in the background with the key colors of now
        state.preview_idx = (state.preview_idx + 1) % len(state.coarse_images)
        key_colors = state.key_snapshot()

        def work(progress) -> None:
            state.rerun_all_previews(key_colors)
            state.update_picture(None, key_colors)

        self.runner.run("Computing preview", work, self._show_previews)

    def _on_next_signal(self, event) -> None:
        if self.runner.refuse() or self.interpretation is None:
            return
        state = self.state
        state.next_signal()
        key_colors = state.key_snapshot()

        def work(progress) -> None:
            state.update_picture(None, key_colors)

        self.runner.run("Coloring", work, self._show_signal)

    def _show_signal(self) -> None:
        """Show the domain picture of another signal (and its title)."""
        self._set_preview_titles()
        self._show_interpretation()

    def _on_align(self, event) -> None:
        self._keys_changed(self.state.path.align(self.align_box.text))

    def _on_add(self, event) -> None:
        self.state.path.add()
        self._keys_changed()

    def _on_remove(self, event) -> None:
        message = self.state.path.remove()
        if message is None:
            self._keys_changed()
        else:
            self.message.set(message)

    def _on_reset(self, event) -> None:
        path = self.state.path
        self._run_fit(path.inputs.num_segments, selected=0)

    def _on_auto(self, event) -> None:
        self._run_fit(len(self.state.path.key_colors) - 1)

    def _run_fit(self, segments: int, selected: int | None = None) -> None:
        """Automatic fit of the selected label in the background; the key colors are
        replaced when it is done (label switches wait for it)."""
        if self.runner.refuse():
            return
        path = self.state.path
        result = {}

        def work(progress) -> None:
            result["fit"] = path.fit(segments)

        def done() -> None:
            keys, message = result["fit"]
            path.set_keys(keys, selected)
            self._keys_changed(message)

        self.runner.run("Fitting the path", work, done)

    def _on_pipette(self, event) -> None:
        message = self.state.pipette(self.ax_image.get_xlim(), self.ax_image.get_ylim())
        if message is None:
            self._keys_changed()
        else:
            self.message.set(message)

    # ! ---- SAVE AND CLOSE ----

    def _save(self) -> bool:
        """Store the paths of all labels; return if it was successful."""
        if self.save is None:
            logger.warning("No way to save the color paths provided.")
            return False
        try:
            self.save(self.state.keys())
        except Exception:
            logger.exception("Saving the color paths failed.")
            return False
        self.state.mark_saved()
        logger.info("Saved the color paths.")
        return True

    def _on_save(self, event) -> None:
        """Save; the window stays open."""
        if self.runner.refuse():
            return
        saved = self._save()
        self.save_status.set(
            "Saved." if saved else "Saving failed, see log.",
            "green" if saved else "red",
        )

    def _on_close(self, event) -> None:
        """Closing ends the editing; unsaved changes are saved or discarded, as the
        user decides."""
        self.runner.wait()
        try:
            if self.state.has_unsaved_changes():
                keep = ask_yes_no(
                    "Unsaved color paths",
                    "The color paths have unsaved changes.",
                    yes="Save",
                    no="Discard",
                )
                # Without a dialog (e.g. a script), keep the changes
                if keep is None or keep:
                    self._save()
                else:
                    logger.info("Discarded the unsaved changes of the color paths.")
        except Exception:
            logger.exception("Closing the color-path editor failed.")

    def run(self) -> bool:
        """Show the window until it is closed.

        Returns
        -------
        bool
            Whether the stored paths equal the paths at the end.

        """
        plt.show()
        self.runner.shutdown()
        return not self.state.has_unsaved_changes()


def edit_color_paths(
    prepared: dict,
    *,
    preview_images: list[darsia.Image] | None,
    preview_baseline: darsia.Image | None,
    labels: darsia.Image,
    mask: darsia.Image,
    save: Callable[[dict[int, list[np.ndarray]]], None] | None = None,
    stored: bool = False,
    reference_label: int | None = None,
) -> bool:
    """Interactively edit the color paths of all labels in one window.

    Parameters
    ----------
    prepared : dict
        Per label `ColorPathInputs` (editable) or a fixed `darsia.ColorPath`
        (`LabelColorPathMapRegression.prepare_color_path_editing`).
    preview_images : list[darsia.Image], optional
        Calibration images for the preview.
    preview_baseline : darsia.Image, optional
        Baseline image for the relative colors of the preview.
    labels : darsia.Image
        Labels.
    mask : darsia.Image
        Calibration mask.
    save : Callable, optional
        Stores the paths (argument: key colors per editable label); called by Save and
        when the window is closed with unsaved changes which the user keeps.
    stored : bool
        Whether the paths at the start are the stored ones.
    reference_label : int, optional
        Label whose path colors all labels in the first signal of the pH view.

    Returns
    -------
    bool
        Whether the stored paths equal the paths at the end (False if changes were
        discarded or saving failed).

    """
    with darsia.timed("editor start"):
        state = ColorPathEditorState(
            prepared,
            list(preview_images or []),
            preview_baseline,
            labels,
            mask,
            stored=stored,
            reference_label=reference_label,
        )
        if len(state.paths) == 0:
            logger.info("No label with an editable color path.")
            if not stored and save is not None:
                save({})
            return True
        state.rerun_all_previews()
        state.update_picture()
        window = ColorPathEditorWindow(state, save)
        window.build()
    return window.run()
