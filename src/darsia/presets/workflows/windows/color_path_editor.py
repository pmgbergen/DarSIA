"""Interactive editor of the key colors of a color path (manual color-path calibration).

A color path runs through key colors, relative to the baseline color, in RGB space. For
one label, the editor shows the colors of the calibration images (3D), the key colors
with the path, a calibration image with the label highlighted, and the pH-indicator
interpretation of that image with the current path. The user moves, adds, removes and
aligns key colors, samples one from the image (pipette) or returns to the automatic fit.

`ColorPathEditorState` holds the data and the operations (no matplotlib);
`edit_color_path` is the window. It is passed as `editor` to
`LabelColorPathMapRegression.find_color_path` in manual mode.

"""

from __future__ import annotations

import logging
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, TextBox

import darsia
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
        label_mask = self.label_mask()
        gray = np.sum(
            image * np.array([0.299, 0.587, 0.114])[None, None, :],
            axis=2,
            keepdims=True,
        )
        image[~label_mask] = 0.7 * gray[~label_mask] + 0.3 * image[~label_mask]
        return image

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

    def align(self, text: str) -> None:
        """Place the selected runs of key colors evenly between their neighbours."""
        n = len(self.key_colors)
        for run in _parse_index_selection(text, n):
            lo, hi = run[0] - 1, run[-1] + 1
            if lo < 0 or hi > n - 1:
                logger.warning(
                    "Align skipped for %s: needs anchors on both sides.", run
                )
                continue
            c_lo, c_hi = self.key_colors[lo], self.key_colors[hi]
            for i in run:
                self.key_colors[i] = c_lo + (i - lo) / (hi - lo) * (c_hi - c_lo)
        self.preview_stale = True

    def add(self) -> None:
        """Insert a key color halfway after the selected one (before the last)."""
        n = len(self.key_colors)
        i = self.selected if self.selected < n - 1 else n - 2
        midpoint = 0.5 * (self.key_colors[i] + self.key_colors[i + 1])
        self.key_colors.insert(i + 1, midpoint)
        self.selected = i + 1
        self.preview_stale = True

    def remove(self) -> bool:
        """Remove the selected key color; False if only two are left."""
        if len(self.key_colors) <= 2:
            logger.warning("Cannot remove: a color path needs at least 2 colors.")
            return False
        del self.key_colors[self.selected]
        self.selected = min(self.selected, len(self.key_colors) - 1)
        self.preview_stale = True
        return True

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

    def pipette(self, col_limits, row_limits) -> bool:
        """Set the selected key color to the characteristic color of the label in a
        region of the shown image.

        Parameters
        ----------
        col_limits, row_limits : tuple[float, float]
            Region in pixels of the preview grid (e.g. the zoomed view).

        Returns
        -------
        bool
            Whether a color was found (otherwise a warning is logged).

        """
        if not self.has_preview or self.label is None:
            logger.warning("Pipette unavailable without preview calibration data.")
            return False
        assert self.baseline is not None
        label_mask = self.label_mask_full()
        if not np.any(label_mask):
            logger.warning(
                "Pipette skipped because selected label has no active pixels."
            )
            return False

        coarse_rows, coarse_cols = self.coarse_image().img.shape[:2]
        if coarse_rows <= 0 or coarse_cols <= 0:
            logger.warning("Pipette skipped due to invalid preview image dimensions.")
            return False
        col_start = int(np.floor(max(0.0, min(col_limits))))
        col_stop = int(np.ceil(min(float(coarse_cols), max(col_limits))))
        row_start = int(np.floor(max(0.0, min(row_limits))))
        row_stop = int(np.ceil(min(float(coarse_rows), max(row_limits))))
        if row_stop <= row_start or col_stop <= col_start:
            logger.warning("Pipette skipped because the current zoom window is empty.")
            return False

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
            logger.warning(
                "Pipette skipped because preview and baseline dimensions differ."
            )
            return False
        relative = np.subtract(full_image.img, self.baseline.img, dtype=float)
        characteristic_colors = darsia.extract_characteristic_data(
            signal=relative[sample], mask=label_mask[sample]
        )
        if len(characteristic_colors) == 0:
            logger.warning("Pipette skipped because no characteristic color was found.")
            return False
        self.key_colors[self.selected] = np.asarray(
            characteristic_colors[0], dtype=float
        )
        self.preview_stale = True
        return True


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
    finalized = False

    fig = plt.figure(figsize=(14, 8))
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
    ax = fig.add_subplot(gs[:, 0], projection="3d")
    ax_preview_image = fig.add_subplot(gs[0, 1])
    ax_preview_interpretation = fig.add_subplot(gs[1, 1])
    ax_preview_image.axis("off")
    ax_preview_interpretation.axis("off")

    def redraw() -> None:
        key_colors = state.key_colors
        ax.cla()
        ax.scatter(
            relative_colors[:, 0],
            relative_colors[:, 1],
            relative_colors[:, 2],
            c=np.clip(colors, 0, 1),
            s=10,
            alpha=0.5,
        )
        absolute_key_colors = [base_color + c for c in key_colors]
        if len(key_colors) >= 2:
            key_arr = np.array(key_colors)
            ax.plot(*key_arr.T, c="black", linewidth=2)
            ax.scatter(*key_arr.T, c=np.clip(absolute_key_colors, 0, 1), s=100)
        ax.plot(*np.array(relative_colors).T, c="gray", linewidth=1, alpha=0.5)
        if ignore is not None:
            base_relative_colors = ignore.relative_colors
            ax.scatter(
                base_relative_colors[:, 0],
                base_relative_colors[:, 1],
                base_relative_colors[:, 2],
                c="r",
                s=10,
                alpha=0.1,
            )
        ax.set_title(f"{name} (manual)")
        ax.set_xlabel("R")
        ax.set_ylabel("G")
        ax.set_zlabel("B")

        ax_preview_image.cla()
        ax_preview_interpretation.cla()
        ax_preview_image.axis("off")
        ax_preview_interpretation.axis("off")

        if state.has_preview and label is not None:
            image = state.highlighted_image()
            if image.ndim == 3:
                ax_preview_image.imshow(image)
            else:
                ax_preview_image.imshow(image, cmap="gray")
            ax_preview_image.set_title(
                f"Calibration image {state.preview_idx + 1} / "
                f"{len(state.coarse_images)} (selected label focus)"
            )
            if state.preview_signal is not None:
                ax_preview_interpretation.imshow(
                    state.preview_signal,
                    cmap=state.color_path().get_color_map(),
                    vmin=0,
                    vmax=1,
                )
                title = "Native pH indicator interpretation"
                if state.preview_stale:
                    title += " (stale - click Re-run pH preview)"
                ax_preview_interpretation.set_title(title)
            else:
                ax_preview_interpretation.text(
                    0.5,
                    0.5,
                    "No preview available.\nClick Re-run pH preview.",
                    ha="center",
                    va="center",
                    transform=ax_preview_interpretation.transAxes,
                )
                ax_preview_interpretation.set_title(
                    "Native pH indicator interpretation"
                )
        else:
            for preview_ax in (ax_preview_image, ax_preview_interpretation):
                preview_ax.text(
                    0.5,
                    0.5,
                    "Manual preview unavailable.",
                    ha="center",
                    va="center",
                    transform=preview_ax.transAxes,
                )
            ax_preview_image.set_title("Calibration image")
            ax_preview_interpretation.set_title("Native pH indicator interpretation")
        fig.canvas.draw_idle()

    index_ax = plt.axes([0.05, 0.20, 0.08, 0.06])
    r_ax = plt.axes([0.15, 0.20, 0.09, 0.06])
    g_ax = plt.axes([0.26, 0.20, 0.09, 0.06])
    b_ax = plt.axes([0.37, 0.20, 0.09, 0.06])
    update_ax = plt.axes([0.48, 0.20, 0.09, 0.06])
    finalize_ax = plt.axes([0.11, 0.94, 0.09, 0.05])
    pipette_ax = plt.axes([0.05, 0.12, 0.09, 0.06])
    previous_preview_ax = plt.axes([0.70, 0.20, 0.10, 0.06])
    next_preview_ax = plt.axes([0.82, 0.20, 0.10, 0.06])
    rerun_preview_ax = plt.axes([0.70, 0.12, 0.22, 0.06])
    align_ax = plt.axes([0.05, 0.04, 0.12, 0.06])
    align_button_ax = plt.axes([0.19, 0.04, 0.09, 0.06])
    add_ax = plt.axes([0.15, 0.12, 0.04, 0.06])
    remove_ax = plt.axes([0.20, 0.12, 0.04, 0.06])
    reset_ax = plt.axes([0.01, 0.94, 0.09, 0.05])

    index_box = TextBox(index_ax, "Index", initial=str(state.selected))
    init_color = state.key_colors[state.selected]
    r_box = TextBox(r_ax, "R", initial=f"{init_color[0]:.6f}")
    g_box = TextBox(g_ax, "G", initial=f"{init_color[1]:.6f}")
    b_box = TextBox(b_ax, "B", initial=f"{init_color[2]:.6f}")
    # No submit on leaving the text boxes (values are applied with Update)
    for box in (index_box, r_box, g_box, b_box):
        box.canvas.mpl_disconnect(box._cids[-1])
    align_box = TextBox(align_ax, "Align", initial="")
    align_box.canvas.mpl_disconnect(align_box._cids[-1])
    align_button = Button(align_button_ax, "Align")
    add_button = Button(add_ax, "+")
    remove_button = Button(remove_ax, "-")
    reset_button = Button(reset_ax, "Reset (auto)")
    update_button = Button(update_ax, "Update")
    finalize_button = Button(finalize_ax, "Finalize")
    pipette_button = Button(pipette_ax, "Pipette")
    previous_preview_button = Button(previous_preview_ax, "Previous")
    next_preview_button = Button(next_preview_ax, "Next")
    rerun_preview_button = Button(rerun_preview_ax, "Re-run pH preview")

    def refresh_color_fields() -> None:
        color = state.key_colors[state.selected]
        r_box.set_val(f"{color[0]:.6f}")
        g_box.set_val(f"{color[1]:.6f}")
        b_box.set_val(f"{color[2]:.6f}")

    def on_index_submit(text: str) -> None:
        state.select(text)
        refresh_color_fields()

    def on_update(_) -> None:
        try:
            new_color = np.array(
                [float(r_box.text), float(g_box.text), float(b_box.text)], dtype=float
            )
        except ValueError:
            logger.warning("Manual color-path update skipped due to invalid RGB input.")
            return
        state.set_color(new_color)
        redraw()

    def on_align(_) -> None:
        state.align(align_box.text)
        refresh_color_fields()
        redraw()

    def on_add(_) -> None:
        state.add()
        refresh_color_fields()
        redraw()

    def on_remove(_) -> None:
        if state.remove():
            refresh_color_fields()
            redraw()

    def on_reset(_) -> None:
        state.reset()
        refresh_color_fields()
        redraw()

    def on_rerun_preview(_) -> None:
        state.rerun_preview()
        redraw()

    def on_pipette(_) -> None:
        if not state.has_preview or label is None:
            logger.warning("Pipette unavailable without preview calibration data.")
            return
        if state.pipette(ax_preview_image.get_xlim(), ax_preview_image.get_ylim()):
            refresh_color_fields()
            redraw()

    def on_previous_preview(_) -> None:
        if len(state.coarse_images) == 0:
            return
        state.step_image(-1)
        redraw()

    def on_next_preview(_) -> None:
        if len(state.coarse_images) == 0:
            return
        state.step_image(1)
        redraw()

    def on_finalize(_) -> None:
        nonlocal finalized
        finalized = True
        plt.close(fig)

    index_box.on_submit(on_index_submit)
    update_button.on_clicked(on_update)
    align_button.on_clicked(on_align)
    add_button.on_clicked(on_add)
    remove_button.on_clicked(on_remove)
    reset_button.on_clicked(on_reset)
    pipette_button.on_clicked(on_pipette)
    previous_preview_button.on_clicked(on_previous_preview)
    next_preview_button.on_clicked(on_next_preview)
    rerun_preview_button.on_clicked(on_rerun_preview)
    finalize_button.on_clicked(on_finalize)

    state.rerun_preview()
    redraw()
    plt.show()
    if not finalized:
        logger.info("Manual color-path editor closed without explicit finalize.")
    return state.key_colors
