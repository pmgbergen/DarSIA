"""Color helper workflow."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb_to_hsv
from matplotlib.widgets import Button

import darsia
from darsia.presets.workflows.analysis.analysis_context import prepare_analysis_context
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.utils.images import load_images_with_cache

logger = logging.getLogger(__name__)
# Padding fraction for histogram ranges when min and max values are equal.
HISTOGRAM_RANGE_PADDING = 0.1


def _to_rgb_array(image: darsia.Image) -> np.ndarray:
    arr = np.asarray(image.img).astype(float, copy=False)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Color helper requires 3-channel images.")
    return arr


def _normalize_unit(arr: np.ndarray) -> np.ndarray:
    min_val = float(np.nanmin(arr))
    max_val = float(np.nanmax(arr))
    if np.isclose(min_val, max_val):
        return np.zeros_like(arr)
    scaled = (arr - min_val) / (max_val - min_val)
    return np.clip(scaled, 0.0, 1.0)


def _scale_for_display(arr: np.ndarray) -> np.ndarray:
    if np.nanmin(arr) < 0 or np.nanmax(arr) > 1:
        return _normalize_unit(arr)
    return arr


def _clamp_range(limits: tuple[float, float], max_value: int) -> tuple[int, int]:
    low, high = sorted(limits)
    start = int(np.floor(max(0.0, low)))
    stop = int(np.ceil(min(float(max_value), high)))
    return start, stop


def _view_slices(image_ax, shape: tuple[int, int]) -> tuple[slice, slice]:
    rows, cols = shape
    col_start, col_stop = _clamp_range(image_ax.get_xlim(), cols)
    row_start, row_stop = _clamp_range(image_ax.get_ylim(), rows)
    if col_stop <= col_start or row_stop <= row_start:
        return slice(0, rows), slice(0, cols)
    return slice(row_start, row_stop), slice(col_start, col_stop)


def launch_color_helper(
    images: list[darsia.Image], *, baseline: darsia.Image, title_prefix: str
) -> None:
    if len(images) == 0:
        raise ValueError("Color helper received no images.")

    baseline_arr = _to_rgb_array(baseline)
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.18, wspace=0.25)

    state = {"idx": 0, "space": "rgb", "relative": False}

    def _current_raw() -> np.ndarray:
        img_arr = _to_rgb_array(images[state["idx"]])
        if state["relative"]:
            return img_arr - baseline_arr
        return img_arr

    def _current_display(raw: np.ndarray) -> np.ndarray:
        return _scale_for_display(raw)

    def _hist_data(raw: np.ndarray) -> np.ndarray:
        """Return channel data; HSV normalization is applied via _normalize_unit."""
        if state["space"] == "hsv":
            normalized = _normalize_unit(raw)
            return rgb_to_hsv(normalized)
        return raw

    def _update_hist() -> None:
        raw = _current_raw()
        data = _hist_data(raw)
        view = data[_view_slices(ax_img, data.shape[:2])]
        ax_hist.cla()

        if state["space"] == "rgb":
            labels = ["R", "G", "B"]
            colors = ["red", "green", "blue"]
            title = "Histogram (RGB)"
        else:
            labels = ["H", "S", "V"]
            colors = ["tab:orange", "tab:green", "tab:blue"]
            title = "Histogram (HSV)"

        for idx, (label, color) in enumerate(zip(labels, colors)):
            channel = view[..., idx].ravel()
            channel = channel[np.isfinite(channel)]
            if channel.size == 0:
                continue
            min_val = float(np.nanmin(channel))
            max_val = float(np.nanmax(channel))
            if np.isclose(min_val, max_val):
                delta = 1.0 if min_val == 0 else abs(min_val) * HISTOGRAM_RANGE_PADDING
                min_val -= delta
                max_val += delta
            ax_hist.hist(
                channel,
                bins=256,
                range=(min_val, max_val),
                alpha=0.5,
                color=color,
                label=label,
            )

        mode = "Relative" if state["relative"] else "Absolute"
        ax_hist.set_title(f"{title} - {mode}")
        ax_hist.legend(loc="upper right")
        fig.canvas.draw_idle()

    def _render() -> None:
        ax_img.cla()
        raw = _current_raw()
        display = _current_display(raw)
        ax_img.imshow(display)
        name = images[state["idx"]].name
        mode = "Relative" if state["relative"] else "Absolute"
        ax_img.set_title(
            f"{title_prefix}: [{state['idx'] + 1}/{len(images)}] - {name} ({mode})"
        )
        ax_img.set_axis_off()
        _update_hist()

    def _on_prev(_: object) -> None:
        state["idx"] = (state["idx"] - 1) % len(images)
        _render()

    def _on_next(_: object) -> None:
        state["idx"] = (state["idx"] + 1) % len(images)
        _render()

    def _on_rgb(_: object) -> None:
        state["space"] = "rgb"
        _update_hist()

    def _on_hsv(_: object) -> None:
        state["space"] = "hsv"
        _update_hist()

    def _on_relative(_: object) -> None:
        state["relative"] = not state["relative"]
        _render()

    def _on_view_changed(_: object) -> None:
        _update_hist()

    ax_img.callbacks.connect("xlim_changed", _on_view_changed)
    ax_img.callbacks.connect("ylim_changed", _on_view_changed)

    prev_ax = fig.add_axes([0.58, 0.05, 0.10, 0.08])
    next_ax = fig.add_axes([0.70, 0.05, 0.10, 0.08])
    rgb_ax = fig.add_axes([0.06, 0.05, 0.10, 0.08])
    hsv_ax = fig.add_axes([0.18, 0.05, 0.10, 0.08])
    rel_ax = fig.add_axes([0.30, 0.05, 0.22, 0.08])

    prev_btn = Button(prev_ax, "Prev")
    next_btn = Button(next_ax, "Next")
    rgb_btn = Button(rgb_ax, "RGB")
    hsv_btn = Button(hsv_ax, "HSV")
    rel_btn = Button(rel_ax, "Absolute/Relative")

    prev_btn.on_clicked(_on_prev)
    next_btn.on_clicked(_on_next)
    rgb_btn.on_clicked(_on_rgb)
    hsv_btn.on_clicked(_on_hsv)
    rel_btn.on_clicked(_on_relative)

    _render()
    plt.show()


def helper_color(
    rig_cls: type[Rig], path: Path | list[Path], show: bool = False
) -> None:
    if show:
        logger.info(
            "helper_color received show=True. Interactive color helper always opens."
        )

    config = FluidFlowerConfig(path, require_data=True, require_results=False)
    config.check("rig", "data", "protocol")
    if config.helper is None or config.helper.color is None:
        raise ValueError("Color helper requires a [helper.color] configuration.")

    ctx = prepare_analysis_context(
        cls=rig_cls,
        path=path,
        all=False,
        require_color_to_mass=False,
        section="helper",
        require_results=False,
        require_data=True,
        sub_config=config.helper.color,
    )

    image_paths = ctx.image_paths
    if len(image_paths) == 0:
        raise ValueError("Color helper received no images.")
    source_images = load_images_with_cache(
        rig=ctx.fluidflower,
        paths=image_paths,
        use_cache=ctx.config.data.use_cache if ctx.config.data else False,
        cache_dir=ctx.config.data.cache if ctx.config.data else None,
    )

    launch_color_helper(
        source_images,
        baseline=ctx.fluidflower.baseline,
        title_prefix="Color helper",
    )
