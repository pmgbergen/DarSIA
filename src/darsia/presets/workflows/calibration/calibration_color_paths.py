import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from matplotlib import pyplot as plt

import darsia
from darsia.presets.workflows.analysis.analysis_context import select_image_paths
from darsia.presets.workflows.basis import label_ids_from_image
from darsia.presets.workflows.calibration.metadata import write_calibration_metadata
from darsia.signals.color import ColorPathEmbedding
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.utils.images import load_images_with_cache
from darsia.presets.workflows.utils.roi_visualization import draw_active_region
from darsia.utils.standard_images import roi_to_mask

logger = logging.getLogger(__name__)


def calibration_color_paths(cls: type[Rig], path: Path, show: bool = False) -> None:
    """Calibration of color paths for a given fluidflower class and configuration.

    Args:
        cls: Rig class.
        path: The path to the configuration file.
        show: Whether to display plots during processing.
    """

    config = FluidFlowerConfig(path, require_data=True, require_results=False)
    config.check("rig", "data", "protocol", "color", "calibration.color")

    # Mypy type checking
    assert config.color is not None
    assert config.calibration is not None
    assert config.calibration.color is not None
    assert config.rig is not None
    assert config.data is not None
    assert config.protocol is not None
    assert config.protocol.imaging is not None
    assert config.protocol.injection is not None
    assert config.protocol.pressure_temperature is not None

    # ! ---- LOAD EXPERIMENT ----
    experiment = darsia.ProtocolledExperiment.init_from_config(config)

    # ! ---- LOAD RIG ----
    fluidflower = cls.load(config.rig.path)
    fluidflower.load_experiment(experiment)

    embedding = config.calibration.color.color
    assert embedding is not None
    if not isinstance(embedding, ColorPathEmbedding):
        raise NotImplementedError(
            "calibration.color currently supports only color path embeddings."
        )
    selected_basis = embedding.basis
    selected_labels = embedding.get_labels(fluidflower)

    # ! ---- LOAD IMAGES ----

    calibration_image_paths = select_image_paths(
        config,
        experiment,
        all=False,
        sub_config=embedding,
    )

    # Cache baseline images for performance
    baseline_sub_config = SimpleNamespace(data=embedding.baseline_data)
    baseline_image_paths = select_image_paths(
        config,
        experiment,
        all=False,
        sub_config=baseline_sub_config,
    )
    baseline_images: list[darsia.Image] = load_images_with_cache(
        rig=fluidflower,
        paths=baseline_image_paths,
        use_cache=config.data.use_cache,
        cache_dir=config.data.cache,
    )

    # Cache calibration images for performance
    calibration_images: list[darsia.Image] = load_images_with_cache(
        rig=fluidflower,
        paths=calibration_image_paths,
        use_cache=config.data.use_cache,
        cache_dir=config.data.cache,
    )

    # ! ---- BUILD CALIBRATION MASK ----

    # Porosity mask restricted to the union of ROIs listed in config.color_paths.rois.
    calibration_mask = fluidflower.boolean_porosity.copy()
    if embedding.rois and config.roi_registry is not None:
        roi_entries = config.roi_registry.resolve_rois(embedding.rois)
        rois = [roi_cfg.roi for roi_cfg in roi_entries.values()]
        union_mask = roi_to_mask(rois, calibration_mask, mode="voxels")
        calibration_mask.img &= union_mask.img
        if not np.any(calibration_mask.img):
            logger.warning(
                "The union of the provided ROIs does not overlap with the "
                "porosity mask. Falling back to the full porosity mask for "
                "colour-path calibration."
            )
            calibration_mask = fluidflower.boolean_porosity.copy()

    if show:
        # Plot the calibration mask for sanity check with contours of the ROIs if provided.
        _, ax = plt.subplots(num="calibration mask")
        draw_active_region(
            ax=ax,
            image=fluidflower.baseline,
            active_mask=calibration_mask,
            title="Calibration Mask for Color Path Calibration",
        )
        plt.show()

    # ! ---- IDENTIFY AND STORE (RELATIVE) COLOR RANGE ----

    tracer_color_range = darsia.ColorRange.from_images(
        images=calibration_images,
        baseline=fluidflower.baseline,
        mask=calibration_mask,
    )
    tracer_color_range.save(embedding.color_range_file)

    # ! ---- COLOR PATH TOOL ----

    color_path_regression = darsia.LabelColorPathMapRegression(
        labels=selected_labels,
        color_range=tracer_color_range,
        mask=calibration_mask,
        resolution=embedding.resolution,
        ignore_labels=embedding.ignore_labels,
    )

    # ! ---- ANALYZE FLUCTUATIONS IN BASELINE IMAGES ----

    ignore_mode = embedding.ignore_baseline_spectrum
    ignore_spectrum: darsia.LabelColorSpectrumMap | None = None

    if ignore_mode in ("baseline", "expanded"):
        baseline_color_spectrum: darsia.LabelColorSpectrumMap = (
            color_path_regression.get_color_spectrum(
                images=baseline_images,
                baseline=fluidflower.baseline,
                threshold_significant=embedding.threshold_baseline,
                verbose=show,
            )
        )
        baseline_color_spectrum.save(embedding.baseline_color_spectrum_folder)

        if ignore_mode == "expanded":
            # Expand the baseline color spectrum through linear regression
            expanded_baseline_color_spectrum: darsia.LabelColorSpectrumMap = (
                color_path_regression.expand_color_spectrum(
                    color_spectrum=baseline_color_spectrum,
                    verbose=False,
                )
            )
            expanded_baseline_color_spectrum.save(
                embedding.baseline_color_spectrum_folder
            )
            ignore_spectrum = expanded_baseline_color_spectrum
        else:
            ignore_spectrum = baseline_color_spectrum

    # Free memory for performance
    del baseline_images

    # ! ---- EXTRACT COLOR SPECTRUM OF TRACERS ---- ! #

    tracer_color_spectrum = color_path_regression.get_color_spectrum(
        images=calibration_images,
        baseline=fluidflower.baseline,
        ignore=ignore_spectrum,
        threshold_significant=embedding.threshold_calibration,
        verbose=show,
    )
    preview_calibration_image = calibration_images[0] if calibration_images else None

    # Find a relative color path through the significant boxes
    label_color_path_map: darsia.LabelColorPathMap = (
        color_path_regression.find_color_path(
            color_spectrum=tracer_color_spectrum,
            ignore=ignore_spectrum,
            num_segments=embedding.num_segments,
            directory=embedding.color_paths_folder,
            weighting=embedding.histogram_weighting,
            mode=embedding.calibration_mode,
            preview_image=preview_calibration_image,
            preview_images=calibration_images,
            preview_baseline=fluidflower.baseline,
            verbose=show,
        )
    )

    # Store the color paths to file
    label_color_path_map.save(embedding.color_paths_folder)
    write_calibration_metadata(
        embedding.color_paths_folder / "metadata.json",
        basis=selected_basis,
        label_ids=label_ids_from_image(selected_labels),
    )

    # Display the color paths
    # if show:
    # print(label_color_path_map)
    # label_color_path_map.show_cmaps()
    # label_color_path_map.show_paths()
    # TODO: Provide advanced plotting - mostly for paper publication.

    logger.info("Calibration of color paths completed.")


def collect_existing_calibration_paths_to_delete(path: Path | list[Path]) -> list[Path]:
    """Collect existing calibration paths that would be deleted.

    Args:
        path: Path(s) to the configuration file(s).

    Returns:
        List of unique existing paths in deletion order.
    """

    config = FluidFlowerConfig(path, require_data=False, require_results=False)

    paths_to_delete: list[Path] = []
    if config.color is not None:
        for embedding in config.color.embeddings.values():
            paths_to_delete.append(embedding.calibration_root)
    if config.data is not None and config.data.cache is not None:
        paths_to_delete.append(config.data.cache)

    existing: list[Path] = []
    seen: set[Path] = set()
    for current_path in paths_to_delete:
        if current_path.exists() and current_path not in seen:
            seen.add(current_path)
            existing.append(current_path)
    return existing


def delete_calibration(
    path: Path | list[Path], *, require_confirmation: bool = True
) -> None:
    """Delete existing calibration files and cached images.

    Removes the color paths calibration file, baseline color spectrum folder,
    color range file, and all cached images in the results/cache folder.

    Args:
        path: Path(s) to the configuration file(s).
        require_confirmation: If True, ask for command-line confirmation before
            deleting. Set to False for already-confirmed non-interactive flows
            (e.g. GUI confirmation dialogs).

    """
    logger.warning(
        """\033[91mDeleting existing calibration data. Use with caution as this """
        """will delete existing results.\033[0m"""
    )

    existing = collect_existing_calibration_paths_to_delete(path)
    if not existing:
        logger.info("No existing calibration data found to delete.")
        return

    logger.info("The following will be deleted:")
    for p in existing:
        logger.info(f"  {p}")

    if require_confirmation:
        user_input = input(
            "\033[91mAre you sure you want to delete the existing calibration data? "
            "This action cannot be undone. (y/n): \033[0m"
        )
        if user_input.lower() != "y":
            logger.info("Calibration data deletion aborted.")
            return

    for p in existing:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
    logger.info("Calibration data deleted.")
