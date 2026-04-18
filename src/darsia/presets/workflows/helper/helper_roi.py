"""ROI helper workflow."""

# TODO: Add option to use coarser images, but keep physical coordinates for the output.
# TODO: This can be done by utilizing the coordinatesystem of the darsia.Image.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RectangleSelector

import darsia
from darsia.presets.workflows.analysis.analysis_context import select_image_paths
from darsia.presets.workflows.analysis.scalar_products import (
    analysis_scalar_products,
    requires_rescaled_modes,
)
from darsia.presets.workflows.basis import select_labels_for_basis
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.heterogeneous_color_to_mass_analysis import (
    HeterogeneousColorToMassAnalysis,
)
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.utils.images import load_images_with_cache

logger = logging.getLogger(__name__)


def _normalize_mode(mode: str) -> str:
    return "mass_total" if mode == "mass" else mode


def _to_scalar_display_array(image: darsia.Image) -> np.ndarray:
    """Return image data as a display-ready 2D scalar array."""
    array = np.asarray(image.img)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]
    return array


def _build_color_to_mass_analysis(
    config: FluidFlowerConfig,
    experiment: darsia.ProtocolledExperiment,
    fluidflower: Rig,
) -> HeterogeneousColorToMassAnalysis:
    assert config.color_to_mass is not None
    _, analysis_labels = select_labels_for_basis(
        fluidflower, config.color_to_mass.basis
    )

    experiment_start = experiment.experiment_start
    state = experiment.pressure_temperature_protocol.get_state(experiment_start)
    gradient = experiment.pressure_temperature_protocol.get_gradient(experiment_start)
    co2_mass_analysis = darsia.CO2MassAnalysis(
        baseline=fluidflower.baseline,
        atmospheric_pressure=state.pressure,
        atmospheric_temperature=state.temperature,
        atmospheric_pressure_gradient=gradient.pressure,
        atmospheric_temperature_gradient=gradient.temperature,
    )
    return HeterogeneousColorToMassAnalysis.load(
        folder=config.color_to_mass.calibration_folder,
        baseline=fluidflower.baseline,
        labels=analysis_labels,
        co2_mass_analysis=co2_mass_analysis,
        geometry=fluidflower.geometry,
        restoration=None,
        basis=config.color_to_mass.basis,
    )


def _scalar_image_for_mode(
    *,
    mode: str,
    result: darsia.SimpleMassAnalysisResults,
    fluidflower: Rig,
    experiment: darsia.ProtocolledExperiment,
    color_to_mass_analysis: HeterogeneousColorToMassAnalysis,
    img_date: Any,
) -> darsia.Image:
    normalized_mode = _normalize_mode(mode)
    requested_modes = {normalized_mode}
    scalar_kwargs = {}
    if requires_rescaled_modes(requested_modes):
        scalar_kwargs = {
            "geometry": fluidflower.geometry,
            "injection_protocol": experiment.injection_protocol,
            "co2_mass_analysis": color_to_mass_analysis.co2_mass_analysis,
            "date": img_date,
        }
    mode_images, _ = analysis_scalar_products(
        mass_analysis_result=result,
        requested_modes=requested_modes,
        **scalar_kwargs,
    )
    return mode_images[normalized_mode]


def format_roi_template(corner_1: np.ndarray, corner_2: np.ndarray) -> str:
    return (
        "[roi.roi_name]\n"
        'name = "roi_name"\n'
        f"corner_1 = [{float(corner_1[0]):.8g}, {float(corner_1[1]):.8g}]\n"
        f"corner_2 = [{float(corner_2[0]):.8g}, {float(corner_2[1]):.8g}]\n"
    )


def _copy_roi_to_clipboard(template: str) -> None:
    try:
        import tkinter as tk
    except ModuleNotFoundError:
        logger.info("ROI template:\n%s", template)
        return

    try:
        root = tk.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(template)
        logger.info("ROI template copied to clipboard.")
    except Exception as exc:  # pragma: no cover - platform specific
        logger.warning(f"Copy to clipboard failed:\n{exc}")
    finally:
        if "root" in locals():
            root.destroy()


def _box_from_zoom_limits(
    *, xlim: tuple[float, float], ylim: tuple[float, float], shape: tuple[int, int]
) -> tuple[slice, slice] | None:
    cols = shape[1]
    rows = shape[0]
    col_start = int(np.floor(max(0.0, min(xlim))))
    col_stop = int(np.ceil(min(float(cols), max(xlim))))
    row_start = int(np.floor(max(0.0, min(ylim))))
    row_stop = int(np.ceil(min(float(rows), max(ylim))))
    if col_stop <= col_start or row_stop <= row_start:
        return None
    return (slice(row_start, row_stop), slice(col_start, col_stop))


def _corners_from_box(
    image: darsia.Image, box: tuple[slice, slice]
) -> tuple[np.ndarray, np.ndarray]:
    row_slice, col_slice = box
    row0 = int(row_slice.start)
    row1 = int(max(row0, row_slice.stop - 1))
    col0 = int(col_slice.start)
    col1 = int(max(col0, col_slice.stop - 1))
    pixels = darsia.VoxelArray([[row0, col0], [row1, col1]])
    coords = image.coordinatesystem.coordinate(pixels)
    return np.asarray(coords[0]), np.asarray(coords[1])


def _box_from_rectangle_selection(
    *,
    x_press: float | None,
    y_press: float | None,
    x_release: float | None,
    y_release: float | None,
    shape: tuple[int, int],
) -> tuple[slice, slice] | None:
    if x_press is None or y_press is None or x_release is None or y_release is None:
        return None
    rows, cols = shape
    row_start = int(np.floor(max(0.0, min(y_press, y_release))))
    row_stop = int(np.ceil(min(float(rows), max(y_press, y_release))))
    col_start = int(np.floor(max(0.0, min(x_press, x_release))))
    col_stop = int(np.ceil(min(float(cols), max(x_press, x_release))))
    if row_stop <= row_start or col_stop <= col_start:
        return None
    return (slice(row_start, row_stop), slice(col_start, col_stop))


def launch_roi_helper_viewer(
    images: list[darsia.Image], *, mode: str, title_prefix: str = "ROI helper"
) -> None:
    if len(images) == 0:
        raise ValueError("ROI helper received no images.")

    fig, ax = plt.subplots(figsize=(11, 8))
    plt.subplots_adjust(bottom=0.16)

    state = {"idx": 0, "box": None, "selector": None, "artist": None}

    def _current_image() -> darsia.Image:
        return images[state["idx"]]

    def _render() -> None:
        ax.cla()
        img = _current_image()
        array = np.asarray(img.img)
        if array.ndim == 2:
            ax.imshow(array, cmap="viridis")
        elif array.ndim == 3 and array.shape[2] == 1:
            ax.imshow(array[..., 0], cmap="viridis")
        else:
            ax.imshow(array)
        ax.set_title(
            f"{title_prefix}: {mode} [{state['idx'] + 1}/{len(images)}] - {img.name}"
        )
        ax.set_axis_off()
        state["box"] = None
        state["artist"] = None
        fig.canvas.draw_idle()

    def _on_select(eclick: Any, erelease: Any) -> None:
        state["box"] = _box_from_rectangle_selection(
            x_press=eclick.xdata,
            y_press=eclick.ydata,
            x_release=erelease.xdata,
            y_release=erelease.ydata,
            shape=_current_image().img.shape[:2],
        )

    def _on_previous(_: Any) -> None:
        state["idx"] = (state["idx"] - 1) % len(images)
        _render()

    def _on_next(_: Any) -> None:
        state["idx"] = (state["idx"] + 1) % len(images)
        _render()

    def _on_roi(_: Any) -> None:
        box = state["box"]
        if box is None:
            box = _box_from_zoom_limits(
                xlim=ax.get_xlim(),
                ylim=ax.get_ylim(),
                shape=_current_image().img.shape[:2],
            )
        if box is None:
            logger.warning("No ROI box selected and no valid zoom box available.")
            return
        corner_1, corner_2 = _corners_from_box(_current_image(), box)
        template = format_roi_template(corner_1, corner_2)

        logger.info("ROI template:\n%s", template)
        _copy_roi_to_clipboard(template)

    prev_ax = fig.add_axes([0.58, 0.04, 0.12, 0.07])
    next_ax = fig.add_axes([0.72, 0.04, 0.12, 0.07])
    roi_ax = fig.add_axes([0.86, 0.04, 0.10, 0.07])
    prev_btn = Button(prev_ax, "Previous")
    next_btn = Button(next_ax, "Next")
    roi_btn = Button(roi_ax, "ROI")
    prev_btn.on_clicked(_on_previous)
    next_btn.on_clicked(_on_next)
    roi_btn.on_clicked(_on_roi)

    state["selector"] = RectangleSelector(
        ax,
        _on_select,
        useblit=True,
        button=[1],
        minspanx=3,
        minspany=3,
        spancoords="pixels",
        interactive=True,
    )
    _render()
    plt.show()


def helper_roi(cls: type[Rig], path: Path | list[Path], show: bool = False) -> None:
    if show:
        logger.info(
            "helper_roi received show=True. Interactive ROI helper always opens its viewer."
        )
    config = FluidFlowerConfig(path, require_data=True, require_results=False)
    config.check("rig", "data", "protocol")

    assert config.data is not None
    assert config.rig is not None
    assert config.helper is not None
    assert config.helper.roi is not None

    helper_config = config.helper.roi
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

    mode = helper_config.mode
    if mode == "none":
        display_images = source_images
    else:
        if config.color_to_mass is None:
            raise ValueError(
                "helper.roi.mode requires [color_to_mass] calibration when mode != 'none'."
            )
        color_to_mass_analysis = _build_color_to_mass_analysis(
            config, experiment, fluidflower
        )
        display_images = []
        for img in source_images:
            result = color_to_mass_analysis(img)
            scalar_image = _scalar_image_for_mode(
                mode=mode,
                result=result,
                fluidflower=fluidflower,
                experiment=experiment,
                color_to_mass_analysis=color_to_mass_analysis,
                img_date=img.date,
            )
            display_images.append(
                darsia.ScalarImage(
                    _to_scalar_display_array(scalar_image),
                    **img.metadata(),
                )
            )
    launch_roi_helper_viewer(display_images, mode=mode)
