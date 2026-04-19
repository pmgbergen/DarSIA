"""Result reader helper workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

import darsia
from darsia.presets.workflows.analysis.analysis_context import select_image_paths
from darsia.presets.workflows.analysis.image_export_formats import ImageExportFormats
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.config.format_registry import ImageExportFormat
from darsia.presets.workflows.rig import Rig

logger = logging.getLogger(__name__)


@dataclass
class ResultFrame:
    image: darsia.Image
    source_name: str
    result_path: Path
    minimum: float
    maximum: float
    value_sum: float
    integral: float


def _to_scalar_array(image: darsia.Image) -> np.ndarray:
    arr = np.asarray(image.img)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError("ResultReader supports 2D scalar images only.")
    return arr


def _resolve_result_format(
    config: FluidFlowerConfig, format_key: str
) -> ImageExportFormat:
    if (
        config.format_registry is not None
        and format_key in config.format_registry.keys()
    ):
        specs = config.format_registry.resolve(format_key)
        if len(specs) != 1:
            raise ValueError(
                f"helper.results.format '{format_key}' must resolve to exactly one format."
            )
        spec = specs[0]
    else:
        key = format_key.strip().lower()
        spec = ImageExportFormat(type=key, identifier=key)
    if spec.type not in {"npz", "csv"}:
        raise ValueError("ResultReader supports only npz and csv formats.")
    return spec


def _collect_result_files(
    source_paths: list[Path], result_folder: Path, suffix: str
) -> list[Path]:
    available = sorted(result_folder.glob(f"*.{suffix}"))
    if len(available) == 0:
        raise FileNotFoundError(
            f"No '*.{suffix}' files found in result folder {result_folder}."
        )

    by_stem = {path.stem: path for path in available}
    selected: list[Path] = []
    for i, source in enumerate(source_paths):
        match = by_stem.get(source.stem)
        if match is None and i < len(available):
            match = available[i]
        if match is None:
            raise FileNotFoundError(
                f"Could not match result file for source image '{source.name}' "
                f"in folder {result_folder}."
            )
        selected.append(match)
    return selected


def _compute_statistics(
    image: darsia.Image,
    *,
    geometry: darsia.Geometry,
) -> tuple[float, float, float, float]:
    arr = _to_scalar_array(image).astype(float, copy=False)
    minimum = float(np.nanmin(arr))
    maximum = float(np.nanmax(arr))
    value_sum = float(np.nansum(arr))
    integral = float(geometry.integrate(image))
    return minimum, maximum, value_sum, integral


def _resolve_cmap(config: FluidFlowerConfig, cmap: str | None):
    if cmap is None:
        return "viridis"
    exporter = ImageExportFormats(config, [])
    resolved = exporter._resolve_cmap(cmap)
    return resolved if resolved is not None else "viridis"


def launch_result_reader(frames: list[ResultFrame], *, mode: str, cmap) -> None:
    if len(frames) == 0:
        raise ValueError("ResultReader received no result frames.")

    fig, ax = plt.subplots(figsize=(11, 8))
    plt.subplots_adjust(bottom=0.16)
    state = {"idx": 0}

    def _current() -> ResultFrame:
        return frames[state["idx"]]

    def _render() -> None:
        ax.cla()
        frame = _current()
        arr = _to_scalar_array(frame.image)
        ax.imshow(arr, cmap=cmap)
        ax.set_axis_off()
        ax.set_title(
            f"ResultReader: {mode} [{state['idx'] + 1}/{len(frames)}] - "
            f"{frame.source_name}"
        )
        stats = (
            f"min: {frame.minimum:.6g}\n"
            f"max: {frame.maximum:.6g}\n"
            f"sum: {frame.value_sum:.6g}\n"
            f"integral: {frame.integral:.6g}"
        )
        ax.text(
            0.02,
            0.98,
            stats,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.6, "edgecolor": "none"},
        )
        fig.canvas.draw_idle()

    def _on_previous(_: object) -> None:
        state["idx"] = (state["idx"] - 1) % len(frames)
        _render()

    def _on_next(_: object) -> None:
        state["idx"] = (state["idx"] + 1) % len(frames)
        _render()

    def _on_click(event) -> None:
        if event.inaxes is not ax:
            return
        if event.button == 1:
            _on_next(event)
        elif event.button == 3:
            _on_previous(event)

    prev_ax = fig.add_axes([0.64, 0.04, 0.12, 0.07])
    next_ax = fig.add_axes([0.78, 0.04, 0.12, 0.07])
    prev_btn = Button(prev_ax, "Previous")
    next_btn = Button(next_ax, "Next")
    prev_btn.on_clicked(_on_previous)
    next_btn.on_clicked(_on_next)
    fig.canvas.mpl_connect("button_press_event", _on_click)

    _render()
    plt.show()


def helper_result_reader(
    rig_cls: type[Rig], path: Path | list[Path], show: bool = False
):
    if show:
        logger.info(
            "helper_result_reader received show=True. "
            "Interactive ResultReader always opens its viewer."
        )

    config = FluidFlowerConfig(path, require_data=True, require_results=True)
    config.check("rig", "data", "protocol")
    assert config.data is not None
    assert config.rig is not None
    assert config.helper is not None
    assert config.helper.results is not None

    helper_config = config.helper.results
    experiment = darsia.ProtocolledExperiment.init_from_config(config)
    fluidflower = rig_cls.load(config.rig.path, config.corrections)
    fluidflower.load_experiment(experiment)

    image_paths = select_image_paths(
        config,
        experiment,
        all=False,
        sub_config=helper_config,
        data_registry=config.data.registry,
    )
    format_spec = _resolve_result_format(config, helper_config.format)
    result_folder = config.data.results / helper_config.mode / format_spec.folder_name
    result_paths = _collect_result_files(image_paths, result_folder, format_spec.type)

    roi = None
    geometry: darsia.Geometry = fluidflower.geometry
    if helper_config.roi is not None and len(helper_config.roi) > 0:
        assert config.roi_registry is not None
        roi_key = helper_config.roi[0]
        if len(helper_config.roi) > 1:
            logger.warning(
                "ResultReader currently supports one active ROI. "
                f"Using first ROI key '{roi_key}'."
            )
        roi = config.roi_registry.resolve_rois([roi_key])[roi_key].roi
        geometry = fluidflower.geometry.subregion(roi)

    frames: list[ResultFrame] = []
    for source_path, result_path in zip(image_paths, result_paths):
        date = experiment.get_datetime(source_path)
        if format_spec.type == "npz":
            image = darsia.imread(result_path)
        else:
            image = fluidflower.import_from_csv(
                result_path,
                delimiter=format_spec.delimiter,
                date=date,
                reference_date=getattr(fluidflower, "reference_date", None),
                name=source_path.name,
            )
        if roi is not None:
            image = image.subregion(roi)
        minimum, maximum, value_sum, integral = _compute_statistics(
            image,
            geometry=geometry,
        )
        frames.append(
            ResultFrame(
                image=image,
                source_name=source_path.name,
                result_path=result_path,
                minimum=minimum,
                maximum=maximum,
                value_sum=value_sum,
                integral=integral,
            )
        )

    launch_result_reader(
        frames,
        mode=helper_config.mode,
        cmap=_resolve_cmap(config, helper_config.cmap),
    )
