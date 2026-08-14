"""Template for mass analysis."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

import darsia
from darsia.presets.workflows.analysis.analysis_context import (
    AnalysisContext,
    prepare_analysis_context,
)
from darsia.presets.workflows.analysis.image_export_formats import ImageExportFormats
from darsia.presets.workflows.analysis.progress import (
    AnalysisProgressEvent,
    publish_image_progress,
)
from darsia.presets.workflows.analysis.scalar_products import analysis_scalar_products
from darsia.presets.workflows.analysis.streaming import publish_stream_images
from darsia.presets.workflows.config.analysis import (
    SUPPORTED_ANALYSIS_MASS_EXPORT_MODES,
)
from darsia.presets.workflows.rig import Rig

logger = logging.getLogger(__name__)

_DEFAULT_MASS_EXPORT_MODES = ["mass"]


def _safe_finite_max(name: str, values: np.ndarray) -> float:
    """Return max finite value and fail with clear error for all-NaN arrays."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError(
            f"Cannot determine scalar export vmax: '{name}' contains no finite values."
        )
    return float(np.max(finite))


def _save_scalar_image_artifacts(
    image: darsia.Image,
    folder: Path,
    stem: str,
    exporter: ImageExportFormats,
    quality: int = 50,
    scalar_write_kwargs: dict[str, float] | None = None,
) -> None:
    """Store scalar image in configured analysis export formats."""
    exporter.export_image(
        image,
        folder,
        stem,
        supported_types={"jpg", "png", "npz", "npy", "csv"},
        jpg_quality=quality,
        scalar_write_kwargs=scalar_write_kwargs,
    )


def analysis_mass_from_context(
    ctx: AnalysisContext,
    show: bool = False,
    stream_callback: Callable[[dict[str, bytes] | None], None] | None = None,
    progress_callback: Callable[[AnalysisProgressEvent], None] | None = None,
) -> None:
    """Mass analysis using pre-prepared context.

    Args:
        ctx: Pre-prepared analysis context with color_to_mass_analysis initialized.
        show: Whether to show the images.

    """
    assert ctx.config.data is not None
    assert ctx.config.analysis is not None
    assert ctx.color_to_mass_analysis is not None

    config = ctx.config
    experiment = ctx.experiment
    fluidflower = ctx.fluidflower
    image_paths = ctx.image_paths
    color_to_mass_analysis = ctx.color_to_mass_analysis
    if not hasattr(color_to_mass_analysis, "co2_mass_analysis"):
        raise AttributeError(
            "Mass rescaling requires 'co2_mass_analysis' on color_to_mass_analysis."
        )
    co2_mass_analysis = color_to_mass_analysis.co2_mass_analysis

    # ! ---- ENSURE MASS CONFIGURATION ----
    if config.analysis.mass is None:
        raise ValueError("Mass analysis requires an explicit [analysis.mass] section.")

    # ! ---- GEOMETRY FOR INTEGRATION ----
    geometry = {}
    geometry.update(
        {
            roi_config.name: fluidflower.geometry.subregion(roi_config.roi)
            for roi_config in config.analysis.mass.roi.values()
        }
    )
    geometry.update(
        {
            roi_and_label_config.name: fluidflower.geometry.subregion(
                roi_and_label_config.roi
            )
            for roi_and_label_config in config.analysis.mass.roi_and_label.values()
        }
    )

    # ! ---- ANALYSIS ----

    # Plotting
    raw_export_modes = getattr(config.analysis.mass, "export", None)
    export_modes = (
        _DEFAULT_MASS_EXPORT_MODES
        if raw_export_modes is None
        else [
            mode
            for mode in raw_export_modes
            if mode in SUPPORTED_ANALYSIS_MASS_EXPORT_MODES
        ]
    )
    output_folders = {
        "mass": config.data.results / "mass",
        "rescaled_mass": config.data.results / "rescaled_mass",
        "extensive_mass": config.data.results / "extensive_mass",
        "extensive_rescaled_mass": config.data.results / "extensive_rescaled_mass",
        "saturation_g": config.data.results / "saturation_g",
        "rescaled_saturation_g": config.data.results / "rescaled_saturation_g",
        "concentration_aq": config.data.results / "concentration_aq",
        "rescaled_concentration_aq": config.data.results / "rescaled_concentration_aq",
    }
    for mode in export_modes:
        output_folders[mode].mkdir(parents=True, exist_ok=True)
    exporter = ImageExportFormats.from_analysis_config(
        config, fallback_formats=["npz", "jpg"]
    )

    vmax_mass = max(
        _safe_finite_max("solubility_co2", co2_mass_analysis.solubility_co2),
        _safe_finite_max("density_gaseous_co2", co2_mass_analysis.density_gaseous_co2),
    )
    geometry_voxel_size = float(np.prod(np.asarray(fluidflower.geometry.voxel_size)))
    depth = getattr(fluidflower.geometry, "depth", 1.0)
    if hasattr(depth, "img"):
        max_depth = _safe_finite_max("geometry.depth", depth.img)
    else:
        max_depth = _safe_finite_max("geometry.depth", depth)
    vmax_extensive_mass = geometry_voxel_size * max_depth * vmax_mass
    scalar_write_kwargs_by_mode: dict[str, dict[str, float]] = {
        "mass": {"vmin": 0.0, "vmax": vmax_mass},
        "rescaled_mass": {"vmin": 0.0, "vmax": vmax_mass},
        "extensive_mass": {"vmin": 0.0, "vmax": vmax_extensive_mass},
        "extensive_rescaled_mass": {"vmin": 0.0, "vmax": vmax_extensive_mass},
        "saturation_g": {"vmin": 0.0, "vmax": 1.0},
        "rescaled_saturation_g": {"vmin": 0.0, "vmax": 1.0},
        "concentration_aq": {"vmin": 0.0, "vmax": 1.0},
        "rescaled_concentration_aq": {"vmin": 0.0, "vmax": 1.0},
    }

    # Initialize DataFrame for storing integrated masses
    detected_cols = [
        f"{roi_config.name}_detected_mass"
        for roi_config in config.analysis.mass.roi.values()
    ] + [
        f"{roi_and_label_config.name}_detected_mass"
        for roi_and_label_config in config.analysis.mass.roi_and_label.values()
    ]
    detected_cols_g = [
        f"{roi_config.name}_detected_mass_g"
        for roi_config in config.analysis.mass.roi.values()
    ] + [
        f"{roi_and_label_config.name}_detected_mass_g"
        for roi_and_label_config in config.analysis.mass.roi_and_label.values()
    ]
    detected_cols_aq = [
        f"{roi_config.name}_detected_mass_aq"
        for roi_config in config.analysis.mass.roi.values()
    ] + [
        f"{roi_and_label_config.name}_detected_mass_aq"
        for roi_and_label_config in config.analysis.mass.roi_and_label.values()
    ]
    exact_cols = [
        f"{roi_config.name}_exact_mass"
        for roi_config in config.analysis.mass.roi.values()
    ]
    columns = (
        [
            "time",
            "datetime",
            "image_stem",
            "detected_mass_total",
            "exact_mass_total",
            "detected_mass_total_rescaled",
            "mass_scaling_factor",
        ]
        + exact_cols
        + detected_cols
        + detected_cols_g
        + detected_cols_aq
    )
    mass_df = pd.DataFrame(columns=columns)
    # TODO control this path through toml/config.
    csv_path = config.data.results / "sparse_data" / "integrated_mass.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Loop over images and analyze
    step_started_at = time.monotonic()
    image_total = len(image_paths)

    # Random shuffle for faster preview of analysis.
    if config.analysis.random_traverse:
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

        products, rescaled = analysis_scalar_products(
            mass_analysis_result=mass_analysis_result,
            requested_modes={
                "rescaled_mass",
                "rescaled_saturation_g",
                "rescaled_concentration_aq",
            },
            geometry=fluidflower.geometry,
            injection_protocol=experiment.injection_protocol,
            co2_mass_analysis=co2_mass_analysis,
            date=img.date,
            expert_knowledge_adapter=ctx.expert_knowledge_adapter,
        )
        mass = products["mass_total"]
        mass_g = products["mass_g"]
        mass_aq = products["mass_aq"]
        saturation_g = products["saturation_g"]
        concentration_aq = products["concentration_aq"]
        assert rescaled is not None
        rescaled_mass = products["rescaled_mass"]
        rescaled_saturation_g = products["rescaled_saturation_g"]
        rescaled_concentration_aq = products["rescaled_concentration_aq"]

        # Store data to disk
        export_images: dict[str, darsia.Image] = {
            "mass": mass,
            "rescaled_mass": rescaled_mass,
            "saturation_g": saturation_g,
            "rescaled_saturation_g": rescaled_saturation_g,
            "concentration_aq": concentration_aq,
            "rescaled_concentration_aq": rescaled_concentration_aq,
        }
        if "extensive_mass" in export_modes:
            export_images["extensive_mass"] = fluidflower.geometry.make_extensive(mass)
        if "extensive_rescaled_mass" in export_modes:
            export_images["extensive_rescaled_mass"] = (
                fluidflower.geometry.make_extensive(rescaled_mass)
            )
        for mode in export_modes:
            _save_scalar_image_artifacts(
                export_images[mode],
                output_folders[mode],
                path.stem,
                exporter=exporter,
                scalar_write_kwargs=scalar_write_kwargs_by_mode.get(mode),
            )

        # Prepare row data for DataFrame
        row_data = {"time": image_time, "datetime": img.date, "image_stem": path.stem}
        row_data["detected_mass_total"] = rescaled.detected_mass_total
        row_data["exact_mass_total"] = rescaled.exact_mass_total
        row_data["detected_mass_total_rescaled"] = fluidflower.geometry.integrate(
            rescaled_mass
        )
        row_data["mass_scaling_factor"] = rescaled.mass_scaling_factor

        # Compute exact mass in ROIs and add to row data
        for roi_config in config.analysis.mass.roi.values():
            key = roi_config.name
            roi = roi_config.roi
            # Fetch exact mass from injection protocol
            exact_mass_roi = experiment.injection_protocol.injected_mass(
                date=img.date, roi=roi_config.roi
            )

            # Integrate over chosen roi
            mass_roi = geometry[key].integrate(mass.subregion(roi))
            mass_g_roi = geometry[key].integrate(mass_g.subregion(roi))
            mass_aq_roi = geometry[key].integrate(mass_aq.subregion(roi))

            # Store
            row_data[f"{key}_exact_mass"] = exact_mass_roi
            row_data[f"{key}_detected_mass"] = mass_roi
            row_data[f"{key}_detected_mass_g"] = mass_g_roi
            row_data[f"{key}_detected_mass_aq"] = mass_aq_roi

        # Compute integrated mass, mass_g, mass_aq in sub-ROIs and add to row data
        for roi_and_label_config in config.analysis.mass.roi_and_label.values():
            key = roi_and_label_config.name
            label = roi_and_label_config.label
            roi = roi_and_label_config.roi

            # Restrict mass arrays to labeled area.
            _mass = mass.copy()
            _mass.img[ctx.analysis_labels.img != label] = 0.0
            _mass_g = mass_g.copy()
            _mass_g.img[ctx.analysis_labels.img != label] = 0.0
            _mass_aq = mass_aq.copy()
            _mass_aq.img[ctx.analysis_labels.img != label] = 0.0

            # Integrate over chosen roi
            mass_roi = geometry[key].integrate(_mass.subregion(roi))
            mass_g_roi = geometry[key].integrate(_mass_g.subregion(roi))
            mass_aq_roi = geometry[key].integrate(_mass_aq.subregion(roi))

            # Store
            row_data[f"{key}_detected_mass"] = mass_roi
            row_data[f"{key}_detected_mass_g"] = mass_g_roi
            row_data[f"{key}_detected_mass_aq"] = mass_aq_roi

        # Add row to DataFrame using pd.concat for better performance
        new_row = pd.DataFrame([row_data])
        mass_df = pd.concat([mass_df, new_row], ignore_index=True)

        # Sort by time to ensure chronological order
        mass_df.sort_values(by="time", inplace=True)

        # Save DataFrame to CSV after each image analysis
        mass_df.to_csv(csv_path, index=False)  # Log the current analysis results
        logger.info(f"Processed {path.stem} at time {image_time}")

        for roi_config in config.analysis.mass.roi.values():
            key = roi_config.name
            exact = row_data[f"{key}_exact_mass"]
            detected = row_data[f"{key}_detected_mass"]
            error = (detected - exact) / max(exact, 1e-8) if exact else 0
            logger.info(
                f"  {key}: detected={detected:.6f}, exact={exact:.6f}, "
                f"error={error:.2%}"
            )

        if show:
            import matplotlib.pyplot as plt

            img.show(title=f"Image at {path.stem}", delay=True)
            mass.show(title=f"Mass at {path.stem}", delay=True)
            mass_g.show(title=f"Mass G at {path.stem}", delay=True)
            mass_aq.show(title=f"Mass AQ at {path.stem}", delay=True)
            plt.show()

        publish_stream_images(
            stream_callback=stream_callback,
            image_payload={
                "mass_source_image": img,
                "mass_total": mass,
                "mass_g": mass_g,
                "mass_aq": mass_aq,
                "rescaled_mass": rescaled_mass,
                "rescaled_saturation_g": rescaled_saturation_g,
                "rescaled_concentration_aq": rescaled_concentration_aq,
            },
            logger=logger,
            error_message=f"Failed to stream mass previews for image '{path}'.",
        )
        publish_image_progress(
            progress_callback,
            step="mass",
            image_path=str(path.resolve()),
            image_index=image_index,
            image_total=image_total,
            image_duration_s=time.monotonic() - image_started_at,
            step_elapsed_s=time.monotonic() - step_started_at,
        )


def analysis_mass(
    cls: type[Rig],
    path: Path | list[Path],
    show: bool = False,
    all: bool = False,
    stream_callback: Callable[[dict[str, bytes] | None], None] | None = None,
):
    """Mass analysis (standalone entry point).

    Args:
        cls: Rig class.
        path: Path or list of paths to config files.
        show: Whether to show the images.
        all: Whether to use all images.

    """
    ctx = prepare_analysis_context(
        cls=cls,
        path=path,
        all=all,
        require_color_to_mass=True,
    )
    analysis_mass_from_context(ctx, show=show, stream_callback=stream_callback)
