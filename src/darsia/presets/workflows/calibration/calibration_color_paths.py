import logging
import shutil
from pathlib import Path

import numpy as np
import skimage.measure
from matplotlib import pyplot as plt

import darsia
from darsia.presets.workflows.analysis.analysis_context import select_image_paths
from darsia.presets.workflows.basis import label_ids_from_image, select_labels_for_basis
from darsia.presets.workflows.calibration.metadata import write_calibration_metadata
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.utils.images import load_images_with_cache
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
    config.check("rig", "data", "protocol", "color_paths")

    # Mypy type checking
    assert config.color_paths is not None
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

    requested_basis = config.color_paths.basis
    selected_basis, selected_labels = select_labels_for_basis(
        fluidflower, requested_basis
    )

    # ! ---- LOAD IMAGES ----

    calibration_image_paths = select_image_paths(
        config, experiment, all=False, sub_config=config.color_paths
    )

    # Cache baseline images for performance
    baseline_images: list[darsia.Image] = load_images_with_cache(
        rig=fluidflower,
        paths=config.color_paths.baseline_image_paths,
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
    if config.color_paths.rois and config.roi_registry is not None:
        roi_entries = config.roi_registry.resolve_rois(config.color_paths.rois)
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
        full_image = fluidflower.baseline.copy()
        gray_full_image = full_image.to_monochromatic("gray")
        full_image.img[~calibration_mask.img] = gray_full_image.img[
            ~calibration_mask.img
        ][:, None]
        contours = skimage.measure.find_contours(
            calibration_mask.img.astype(float), level=0.5
        )
        plt.figure("calibration mask")
        plt.imshow(full_image.img)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], color="white", linewidth=2)
        plt.title("Calibration Mask for Color Path Calibration")
        plt.axis("off")
        plt.show()

    # ! ---- IDENTIFY AND STORE (RELATIVE) COLOR RANGE ----

    tracer_color_range = darsia.ColorRange.from_images(
        images=calibration_images,
        baseline=fluidflower.baseline,
        mask=calibration_mask,
    )
    tracer_color_range.save(config.color_paths.color_range_file)

    # ! ---- COLOR PATH TOOL ----

    color_path_regression = darsia.LabelColorPathMapRegression(
        labels=selected_labels,
        color_range=tracer_color_range,
        mask=calibration_mask,
        resolution=config.color_paths.resolution,
        ignore_labels=config.color_paths.ignore_labels,
    )

    # ! ---- ANALYZE FLUCTUATIONS IN BASELINE IMAGES ----

    ignore_mode = config.color_paths.ignore_baseline_spectrum
    ignore_spectrum: darsia.LabelColorSpectrumMap | None = None

    if ignore_mode in ("baseline", "expanded"):
        baseline_color_spectrum: darsia.LabelColorSpectrumMap = (
            color_path_regression.get_color_spectrum(
                images=baseline_images,
                baseline=fluidflower.baseline,
                threshold_significant=config.color_paths.threshold_baseline,
                verbose=show,
            )
        )
        baseline_color_spectrum.save(config.color_paths.baseline_color_spectrum_folder)

        if ignore_mode == "expanded":
            # Expand the baseline color spectrum through linear regression
            expanded_baseline_color_spectrum: darsia.LabelColorSpectrumMap = (
                color_path_regression.expand_color_spectrum(
                    color_spectrum=baseline_color_spectrum,
                    verbose=False,
                )
            )
            expanded_baseline_color_spectrum.save(
                config.color_paths.baseline_color_spectrum_folder
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
        threshold_significant=config.color_paths.threshold_calibration,
        verbose=show,
    )
    # Free memory for performance
    del calibration_images

    # Find a relative color path through the significant boxes
    label_color_path_map: darsia.LabelColorPathMap = (
        color_path_regression.find_color_path(
            color_spectrum=tracer_color_spectrum,
            ignore=ignore_spectrum,
            num_segments=config.color_paths.num_segments,
            directory=config.color_paths.calibration_file,
            weighting=config.color_paths.histogram_weighting,
            mode=config.color_paths.mode,
            verbose=show,
        )
    )

    # Store the color paths to file
    label_color_path_map.save(config.color_paths.calibration_file)
    write_calibration_metadata(
        config.color_paths.calibration_file / "metadata.json",
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
    if config.color_paths is not None:
        paths_to_delete.append(config.color_paths.calibration_file)
        paths_to_delete.append(config.color_paths.baseline_color_spectrum_folder)
        paths_to_delete.append(config.color_paths.color_range_file)
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
