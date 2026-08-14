"""Template for color signal analysis."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from darsia.presets.workflows.analysis.analysis_context import (
    AnalysisContext,
    prepare_analysis_context,
)
from darsia.presets.workflows.analysis.image_export_formats import ImageExportFormats
from darsia.presets.workflows.analysis.progress import (
    AnalysisProgressEvent,
    publish_image_progress,
)
from darsia.presets.workflows.analysis.streaming import publish_stream_images
from darsia.presets.workflows.config.analysis import AnalysisVolumeConfig
from darsia.presets.workflows.rig import Rig

logger = logging.getLogger(__name__)


def analysis_volume_from_context(
    ctx: AnalysisContext,
    show: bool = False,
    stream_callback: Callable[[dict[str, bytes] | None], None] | None = None,
    progress_callback: Callable[[AnalysisProgressEvent], None] | None = None,
) -> None:
    """Volume analysis using pre-prepared context.

    Args:
        ctx: Pre-prepared analysis context with color_to_mass_analysis initialized.
        show: Whether to show the images.

    """
    assert ctx.config.data is not None
    assert ctx.config.analysis is not None
    assert ctx.color_to_mass_analysis is not None

    config = ctx.config
    fluidflower = ctx.fluidflower
    image_paths = ctx.image_paths
    color_to_mass_analysis = ctx.color_to_mass_analysis

    # ! ---- ENSURE VOLUME CONFIGURATION ----
    if config.analysis.volume is None:
        config.analysis.volume = AnalysisVolumeConfig().load(
            sec={
                "volume": {
                    "roi": {
                        "full": {
                            "name": "full",
                            "corner_1": fluidflower.baseline.origin,
                            "corner_2": fluidflower.baseline.opposite_corner,
                        }
                    }
                }
            },
            results=config.data.results,
        )

    # ! ---- GEOMETRY FOR INTEGRATION ----
    geometry = {}
    geometry.update(
        {
            roi_config.name: fluidflower.geometry.subregion(roi_config.roi)
            for roi_config in config.analysis.volume.roi.values()
        }
    )
    geometry.update(
        {
            roi_and_label_config.name: fluidflower.geometry.subregion(
                roi_and_label_config.roi
            )
            for roi_and_label_config in config.analysis.volume.roi_and_label.values()
        }
    )

    # Initialize DataFrame for storing integrated masses
    detected_cols = [
        f"{roi_config.name}_detected_volume"
        for roi_config in config.analysis.volume.roi.values()
    ] + [
        f"{roi_and_label_config.name}_detected_volume"
        for roi_and_label_config in config.analysis.volume.roi_and_label.values()
    ]
    detected_cols_g = [
        f"{roi_config.name}_detected_volume_g"
        for roi_config in config.analysis.volume.roi.values()
    ] + [
        f"{roi_and_label_config.name}_detected_volume_g"
        for roi_and_label_config in config.analysis.volume.roi_and_label.values()
    ]
    detected_cols_aq = [
        f"{roi_config.name}_detected_volume_aq"
        for roi_config in config.analysis.volume.roi.values()
    ] + [
        f"{roi_and_label_config.name}_detected_volume_aq"
        for roi_and_label_config in config.analysis.volume.roi_and_label.values()
    ]
    columns = (
        ["time", "datetime", "stem"]
        + detected_cols
        + detected_cols_g
        + detected_cols_aq
    )
    volume_df = pd.DataFrame(columns=columns)
    csv_path = config.data.results / "sparse_data" / "integrated_volume.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Storing and plotting
    folder_saturation_g = config.data.results / "saturation_g"
    folder_concentration_aq = config.data.results / "concentration_aq"
    folder_saturation_g.mkdir(parents=True, exist_ok=True)
    folder_concentration_aq.mkdir(parents=True, exist_ok=True)
    exporter = ImageExportFormats.from_analysis_config(config, fallback_formats=["npz"])

    # ! ---- ANALYSIS ----

    # Loop over images and analyze
    step_started_at = time.monotonic()
    image_total = len(image_paths)

    # Random shuffle for faster preview of analysis.
    if ctx.config.analysis.random_traverse:
        random.shuffle(image_paths)

    for image_index, path in enumerate(image_paths, start=1):
        image_started_at = time.monotonic()
        try:
            img = fluidflower.read_image(path)
        except Exception as e:
            logger.error(f"Failed to read image '{path}': {e}")
            continue

        # Extract color signal and assign mass
        mass_analysis_result = color_to_mass_analysis(img)

        # Log time
        image_time = mass_analysis_result.time

        # Fetch results
        saturation_g = mass_analysis_result.saturation_g
        concentration_aq = mass_analysis_result.concentration_aq
        saturation_aq = concentration_aq.copy()
        saturation_aq.img *= 1 - saturation_g.img

        # Store coarse data to disk
        exporter.export_image(
            saturation_g,
            folder_saturation_g,
            path.stem,
            supported_types={"jpg", "png", "npz", "npy", "csv"},
        )
        exporter.export_image(
            concentration_aq,
            folder_concentration_aq,
            path.stem,
            supported_types={"jpg", "png", "npz", "npy", "csv"},
        )

        # Prepare row data for DataFrame
        row_data = {"time": image_time, "datetime": img.date, "stem": path.stem}

        # Compute exact mass in ROIs and add to row data
        for roi_config in config.analysis.volume.roi.values():
            key = roi_config.name
            roi = roi_config.roi

            # Integrate over chosen roi
            volume_g_roi = geometry[key].integrate(saturation_g.subregion(roi))
            volume_aq_roi = geometry[key].integrate(saturation_aq.subregion(roi))
            volume_roi = volume_g_roi + volume_aq_roi

            # Store
            row_data[f"{key}_detected_volume"] = volume_roi
            row_data[f"{key}_detected_volume_g"] = volume_g_roi
            row_data[f"{key}_detected_volume_aq"] = volume_aq_roi

        for roi_and_label_config in config.analysis.volume.roi_and_label.values():
            key = roi_and_label_config.name
            label = roi_and_label_config.label
            roi = roi_and_label_config.roi

            # Restrict mass arrays to labeled area.
            _saturation_g = saturation_g.copy()
            _saturation_g.img[ctx.analysis_labels.img != label] = 0.0
            _saturation_aq = saturation_aq.copy()
            _saturation_aq.img[ctx.analysis_labels.img != label] = 0.0

            # Integrate over chosen roi
            volume_g_roi = geometry[key].integrate(_saturation_g.subregion(roi))
            volume_aq_roi = geometry[key].integrate(_saturation_aq.subregion(roi))
            volume_roi = volume_g_roi + volume_aq_roi

            # Store
            row_data[f"{key}_detected_volume"] = volume_roi
            row_data[f"{key}_detected_volume_g"] = volume_g_roi
            row_data[f"{key}_detected_volume_aq"] = volume_aq_roi

        # Add row to DataFrame using pd.concat for better performance
        new_row = pd.DataFrame([row_data])
        volume_df = pd.concat([volume_df, new_row], ignore_index=True)

        # Save DataFrame to CSV after each image analysis
        volume_df.to_csv(csv_path, index=False)
        logger.info(f"Processed {path.stem} at time {image_time}")

        # Log the current analysis results
        for roi_config in config.analysis.volume.roi.values():
            key = roi_config.name
            detected = row_data[f"{key}_detected_volume"]
            detected_g = row_data[f"{key}_detected_volume_g"]
            detected_aq = row_data[f"{key}_detected_volume_aq"]
            logger.info(
                f"""  {key}: detected = {detected:.6f}, """
                f"""detected_g={detected_g:.6f}, """
                f"""detected_aq={detected_aq:.6f}"""
            )

        if show:
            import matplotlib.pyplot as plt

            img.show(title=f"Image at {path.stem}", delay=True)
            saturation_g.show(title=f"Saturation_g at {path.stem}", delay=True)
            concentration_aq.show(title=f"Concentration_aq at {path.stem}", delay=True)
            plt.show()

        publish_stream_images(
            stream_callback=stream_callback,
            image_payload={
                "volume_source_image": img,
                "saturation_g": saturation_g,
                "concentration_aq": concentration_aq,
                "saturation_aq": saturation_aq,
            },
            logger=logger,
            error_message=f"Failed to stream volume previews for image '{path}'.",
        )
        publish_image_progress(
            progress_callback,
            step="volume",
            image_path=str(path.resolve()),
            image_index=image_index,
            image_total=image_total,
            image_duration_s=time.monotonic() - image_started_at,
            step_elapsed_s=time.monotonic() - step_started_at,
        )


def analysis_volume(
    cls: type[Rig],
    path: Path,
    all: bool = False,
    show: bool = False,
    stream_callback: Callable[[dict[str, bytes] | None], None] | None = None,
):
    """Volume analysis (standalone entry point).

    Args:
        cls: Rig class.
        path: Path to config file.
        all: Whether to use all images.
        show: Whether to show the images.

    """
    ctx = prepare_analysis_context(
        cls=cls,
        path=path,
        all=all,
        require_color_to_mass=True,
    )
    analysis_volume_from_context(ctx, show=show, stream_callback=stream_callback)
