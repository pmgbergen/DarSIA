"""Interactive ROI viewer workflow."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Dropdown

import darsia
from darsia.presets.workflows.analysis.analysis_context import select_image_paths
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.config.roi import RoiConfig
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.utils.images import load_images_with_cache
from darsia.presets.workflows.utils.roi_visualization import (
    build_active_mask_from_rois,
    draw_active_region,
)

logger = logging.getLogger(__name__)


def _compute_coarse_shape(
    shape: tuple[int, int], *, min_rows: int = 120, downsampling_factor: int = 4
) -> tuple[int, int]:
    rows, cols = shape
    coarse_rows = max(min_rows, rows // downsampling_factor)
    coarse_rows = min(coarse_rows, rows)
    coarse_cols = max(1, int(round(cols / rows * coarse_rows)))
    return (coarse_rows, coarse_cols)


def _preload_coarse_images(
    images: list[darsia.Image], *, min_rows: int = 120, downsampling_factor: int = 4
) -> list[darsia.Image]:
    if len(images) == 0:
        raise ValueError("ROI Viewer received no images.")
    coarse_shape = _compute_coarse_shape(
        images[0].img.shape[:2],
        min_rows=min_rows,
        downsampling_factor=downsampling_factor,
    )
    return [darsia.resize(image, shape=coarse_shape) for image in images]


def _build_roi_selection_masks(
    image: darsia.Image,
    roi_entries: dict[str, RoiConfig],
) -> dict[str, np.ndarray | None]:
    selection_masks: dict[str, np.ndarray | None] = {"none": None}
    all_masks: list[np.ndarray] = []
    for key, roi_config in roi_entries.items():
        mask = build_active_mask_from_rois(image, roi_config.roi)
        selection_masks[key] = mask
        if mask is not None:
            all_masks.append(mask)
    if len(all_masks) == 0:
        selection_masks["all"] = None
    else:
        selection_masks["all"] = np.logical_or.reduce(all_masks)
    return selection_masks


def launch_roi_viewer(
    images: list[darsia.Image], *, roi_entries: dict[str, RoiConfig], title_prefix: str
) -> None:
    if len(images) == 0:
        raise ValueError("ROI Viewer received no images.")
    selection_masks = _build_roi_selection_masks(images[0], roi_entries)
    selection_order = ["all", "none", *list(roi_entries.keys())]

    fig, ax = plt.subplots(figsize=(11, 8))
    plt.subplots_adjust(bottom=0.16)
    state = {"idx": 0, "selection": "all"}

    def _render() -> None:
        image = images[state["idx"]]
        selection = state["selection"]
        ax.cla()
        draw_active_region(
            ax=ax,
            image=image,
            active_mask=selection_masks.get(selection),
            title=(
                f"{title_prefix}: [{state['idx'] + 1}/{len(images)}] - "
                f"{image.name} - ROI: {selection}"
            ),
        )
        fig.canvas.draw_idle()

    def _on_previous(_: object) -> None:
        state["idx"] = (state["idx"] - 1) % len(images)
        _render()

    def _on_next(_: object) -> None:
        state["idx"] = (state["idx"] + 1) % len(images)
        _render()

    def _on_selection(selection: str) -> None:
        state["selection"] = selection
        _render()

    prev_ax = fig.add_axes([0.60, 0.04, 0.12, 0.07])
    next_ax = fig.add_axes([0.74, 0.04, 0.12, 0.07])
    selector_ax = fig.add_axes([0.08, 0.04, 0.48, 0.07])

    prev_btn = Button(prev_ax, "Previous")
    next_btn = Button(next_ax, "Next")
    prev_btn.on_clicked(_on_previous)
    next_btn.on_clicked(_on_next)

    roi_selector = Dropdown(
        selector_ax,
        "ROI",
        selection_order,
        value=state["selection"],
    )
    roi_selector.on_select(_on_selection)

    _render()
    plt.show()


def helper_roi_viewer(
    cls: type[Rig], path: Path | list[Path], show: bool = False
) -> None:
    if show:
        logger.info(
            "helper_roi_viewer received show=True. Interactive ROI viewer always opens its viewer."
        )
    config = FluidFlowerConfig(path, require_data=True, require_results=False)
    config.check("rig", "data", "protocol")

    assert config.data is not None
    assert config.rig is not None
    assert config.helper is not None
    assert config.helper.roi_viewer is not None

    if config.roi_registry is None:
        raise ValueError(
            "ROI Viewer requires a top-level ROI registry. Define entries in [roi.*]."
        )

    roi_entries = config.roi_registry.resolve_rois(config.roi_registry.keys())
    if len(roi_entries) == 0:
        raise ValueError(
            "ROI Viewer requires plain ROI entries in the ROI registry."
        )

    helper_config = config.helper.roi_viewer
    experiment = darsia.ProtocolledExperiment.init_from_config(config)
    fluidflower = cls.load(config.rig.path, config.corrections)
    fluidflower.load_experiment(experiment)

    image_paths = select_image_paths(
        config,
        experiment,
        all=False,
        sub_config=helper_config,
        data_registry=config.data.registry,
    )
    source_images = load_images_with_cache(
        rig=fluidflower,
        paths=image_paths,
        use_cache=config.data.use_cache,
        cache_dir=config.data.cache,
    )
    coarse_images = _preload_coarse_images(source_images)
    launch_roi_viewer(
        coarse_images,
        roi_entries=roi_entries,
        title_prefix="ROI Viewer",
    )
