import logging
import shutil
from pathlib import Path

import numpy as np
import skimage.measure
from matplotlib import pyplot as plt

import darsia
from darsia.presets.workflows.analysis.analysis_context import select_image_paths
from darsia.presets.workflows.basis import label_ids_from_image, select_labels_for_basis
from darsia.presets.workflows.calibration.metadata import (
    read_calibration_metadata,
    validate_basis_metadata,
    write_calibration_metadata,
)
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.utils.images import load_images_with_cache
from darsia.utils.standard_images import roi_to_mask

logger = logging.getLogger(__name__)


def _resolve_target_labels(
    selected_labels: darsia.Image, target_labels: list[int], ignore_labels: list[int]
) -> list[int]:
    valid_labels = set(label_ids_from_image(selected_labels))
    if not target_labels:
        raise ValueError(
            "Single-label calibration requires non-empty [color_paths].target_labels."
        )
    targets = [int(label) for label in target_labels]
    invalid = sorted(set(targets).difference(valid_labels))
    if invalid:
        raise ValueError(
            "Requested target_labels are not present in the selected basis labels: "
            f"{invalid}."
        )
    ignored = sorted(set(targets).intersection(set(ignore_labels)))
    if ignored:
        raise ValueError(
            "Requested target_labels include ignored labels from "
            "[color_paths].ignore_labels: "
            f"{ignored}."
        )
    return targets


def _load_or_compute_tracer_color_spectrum(
    *,
    color_path_regression: darsia.LabelColorPathMapRegression,
    calibration_images: list[darsia.Image],
    baseline: darsia.Image,
    ignore_spectrum: darsia.LabelColorSpectrumMap | None,
    threshold_calibration: float,
    tracer_color_spectrum_folder: Path,
    strict_stored_artifacts: bool,
    verbose: bool,
) -> darsia.LabelColorSpectrumMap:
    tracer_files = list(tracer_color_spectrum_folder.glob("color_spectrum_*.json"))
    if tracer_files:
        logger.info(
            "Loading stored tracer spectrum from %s", tracer_color_spectrum_folder
        )
        return darsia.LabelColorSpectrumMap.load(tracer_color_spectrum_folder)

    if strict_stored_artifacts:
        raise FileNotFoundError(
            "Missing stored tracer color spectrum in strict mode: "
            f"{tracer_color_spectrum_folder}"
        )

    logger.warning(
        "Stored tracer spectrum not found at %s. Falling back to recomputation.",
        tracer_color_spectrum_folder,
    )
    return color_path_regression.get_color_spectrum(
        images=calibration_images,
        baseline=baseline,
        ignore=ignore_spectrum,
        threshold_significant=threshold_calibration,
        verbose=verbose,
    )


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

    if config.color_paths.calibration_scope == "single_label":
        if not config.color_paths.color_range_file.with_suffix(".json").exists():
            raise FileNotFoundError(
                "Single-label calibration requires stored color_range_file: "
                f"{config.color_paths.color_range_file.with_suffix('.json')}"
            )
        tracer_color_range = darsia.ColorRange.load(config.color_paths.color_range_file)
    else:
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

    if ignore_mode in ("baseline", "expanded") and (
        config.color_paths.calibration_scope == "single_label"
    ):
        if config.color_paths.baseline_color_spectrum_folder.exists():
            ignore_spectrum = darsia.LabelColorSpectrumMap.load(
                config.color_paths.baseline_color_spectrum_folder
            )
        elif config.color_paths.strict_stored_artifacts:
            raise FileNotFoundError(
                "Single-label calibration in strict mode requires stored "
                "baseline_color_spectrum_folder: "
                f"{config.color_paths.baseline_color_spectrum_folder}"
            )
        else:
            logger.warning(
                "Stored baseline spectrum not found at %s. Falling back to "
                "recomputation.",
                config.color_paths.baseline_color_spectrum_folder,
            )
            baseline_color_spectrum = color_path_regression.get_color_spectrum(
                images=baseline_images,
                baseline=fluidflower.baseline,
                threshold_significant=config.color_paths.threshold_baseline,
                verbose=show,
            )
            if ignore_mode == "expanded":
                ignore_spectrum = color_path_regression.expand_color_spectrum(
                    color_spectrum=baseline_color_spectrum,
                    verbose=False,
                )
            else:
                ignore_spectrum = baseline_color_spectrum
    elif ignore_mode in ("baseline", "expanded"):
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

    tracer_color_spectrum = _load_or_compute_tracer_color_spectrum(
        color_path_regression=color_path_regression,
        calibration_images=calibration_images,
        baseline=fluidflower.baseline,
        ignore_spectrum=ignore_spectrum,
        threshold_calibration=config.color_paths.threshold_calibration,
        tracer_color_spectrum_folder=config.color_paths.tracer_color_spectrum_folder,
        strict_stored_artifacts=(
            config.color_paths.calibration_scope == "single_label"
            and config.color_paths.strict_stored_artifacts
        ),
        verbose=show,
    )
    tracer_color_spectrum.save(config.color_paths.tracer_color_spectrum_folder)
    write_calibration_metadata(
        config.color_paths.tracer_color_spectrum_folder / "metadata.json",
        basis=selected_basis,
        label_ids=label_ids_from_image(selected_labels),
    )
    # Free memory for performance
    del calibration_images

    # Find a relative color path through the significant boxes
    if config.color_paths.calibration_scope == "single_label":
        existing_map = darsia.LabelColorPathMap.load(
            config.color_paths.calibration_file
        )
        validate_basis_metadata(
            metadata=read_calibration_metadata(
                config.color_paths.calibration_file / "metadata.json"
            ),
            expected_basis=selected_basis,
            expected_label_ids=label_ids_from_image(selected_labels),
            artifact="color_paths",
            strict=True,
        )
        validate_basis_metadata(
            metadata=read_calibration_metadata(
                config.color_paths.tracer_color_spectrum_folder / "metadata.json"
            ),
            expected_basis=selected_basis,
            expected_label_ids=label_ids_from_image(selected_labels),
            artifact="tracer_color_spectrum",
            strict=True,
        )
        target_labels = _resolve_target_labels(
            selected_labels=selected_labels,
            target_labels=config.color_paths.target_labels,
            ignore_labels=config.color_paths.ignore_labels,
        )
    else:
        existing_map = None
        target_labels = None

    label_color_path_map: darsia.LabelColorPathMap = (
        color_path_regression.find_color_path(
            color_spectrum=tracer_color_spectrum,
            ignore=ignore_spectrum,
            num_segments=config.color_paths.num_segments,
            directory=config.color_paths.calibration_file,
            weighting=config.color_paths.histogram_weighting,
            mode=config.color_paths.mode,
            target_labels=target_labels,
            existing_map=existing_map,
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


def delete_calibration(path: Path | list[Path]) -> None:
    """Delete existing calibration files and cached images.

    Removes the color paths calibration file, baseline color spectrum folder,
    color range file, and all cached images in the results/cache folder.

    Args:
        path: Path(s) to the configuration file(s).

    """
    logger.warning(
        """\033[91mDeleting existing calibration data. Use with caution as this """
        """will delete existing results.\033[0m"""
    )

    config = FluidFlowerConfig(path, require_data=False, require_results=False)

    # Collect paths to delete
    paths_to_delete: list[Path] = []
    if config.color_paths is not None:
        paths_to_delete.append(config.color_paths.calibration_file)
        paths_to_delete.append(config.color_paths.baseline_color_spectrum_folder)
        paths_to_delete.append(config.color_paths.tracer_color_spectrum_folder)
        paths_to_delete.append(config.color_paths.color_range_file)
    if config.data is not None and config.data.cache is not None:
        paths_to_delete.append(config.data.cache)

    existing = [p for p in paths_to_delete if p.exists()]
    if not existing:
        logger.info("No existing calibration data found to delete.")
        return

    logger.info("The following will be deleted:")
    for p in existing:
        logger.info(f"  {p}")

    user_input = input(
        "\033[91mAre you sure you want to delete the existing calibration data? "
        "This action cannot be undone. (y/n): \033[0m"
    )
    if user_input.lower() == "y":
        for p in existing:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
        logger.info("Calibration data deleted.")
    else:
        logger.info("Calibration data deletion aborted.")
