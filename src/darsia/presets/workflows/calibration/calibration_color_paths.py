import copy
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from matplotlib import pyplot as plt

import darsia
from darsia.presets.workflows.analysis.analysis_context import (
    AnalysisContext,
    select_image_paths,
)
from darsia.presets.workflows.basis import label_ids_from_image
from darsia.presets.workflows.calibration.metadata import (
    read_calibration_metadata,
    validate_basis_metadata,
    write_calibration_metadata,
)
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.config.sections import (
    list_required_sections,
    required_sections,
)
from darsia.presets.workflows.processing.image_porosity import load_image_porosity
from darsia.presets.workflows.utils.images import load_images_with_cache
from darsia.presets.workflows.utils.roi_visualization import draw_active_region
from darsia.presets.workflows.windows.color_path_editor import edit_color_paths
from darsia.signals.color import ColorPathEmbedding
from darsia.utils.standard_images import roi_to_mask

logger = logging.getLogger(__name__)


def _with_data(image: darsia.Image, array: np.ndarray) -> darsia.Image:
    """`image` with the data `array`: a shallow copy, so no data is copied."""
    result = copy.copy(image)
    result.img = array
    return result


def _load_initial_paths(
    embedding: ColorPathEmbedding, label_ids: list[int]
) -> dict[int, list[np.ndarray]]:
    """Load relative key colors of a previous calibration, if compatible.

    Returns an empty dict if no calibration exists or it does not match the
    current basis and labels.

    """
    csv_file = Path(embedding.color_paths_csv_file)
    if not csv_file.exists():
        return {}
    try:
        validate_basis_metadata(
            metadata=read_calibration_metadata(Path(embedding.metadata_file)),
            expected_basis=embedding.basis,
            expected_label_ids=label_ids,
            artifact="color path",
        )
    except ValueError as error:
        logger.warning("Not using existing color paths as initialization: %s", error)
        return {}
    color_path_map = darsia.LabelColorPathMap.load_csv(csv_file)
    initial_paths = {
        label: [c - path.base_color for c in path.colors]
        for label, path in color_path_map.items()
    }
    logger.info(
        "Initializing from existing color paths for labels %s", list(initial_paths)
    )
    return initial_paths


def save_color_paths(
    label_color_path_map: darsia.LabelColorPathMap,
    *,
    embedding: ColorPathEmbedding,
    basis,
    label_ids: list[int],
    ignore_labels: list[int],
) -> None:
    """Store color paths: metadata, CSV and the colormap of each path.

    Parameters
    ----------
    label_color_path_map : darsia.LabelColorPathMap
        Color paths of all labels.
    embedding : ColorPathEmbedding
        The embedding (files and folders of the calibration).
    basis : optional
        Basis of the labels (metadata).
    label_ids : list[int]
        Labels (metadata).
    ignore_labels : list[int]
        Labels with a zero path (metadata).

    """
    write_calibration_metadata(
        embedding.metadata_file,
        basis=basis,
        label_ids=label_ids,
        extra={"ignore_labels": ignore_labels},
    )
    label_color_path_map.save_csv(embedding.color_paths_csv_file)
    # Colormap of each color path (format registry: color_path.<id>.<label>)
    label_color_path_map.save_cmaps(
        embedding.figures_folder, name=f"color_path.{embedding.embedding_id}"
    )


@required_sections("rig", "data", "protocols", "color", "calibration.color")
def calibration_color_paths_from_context(
    ctx: AnalysisContext, show: bool = False
) -> None:
    """Calibration of color paths for a given fluidflower class and configuration.

    Parameters
    ----------
    ctx : AnalysisContext
        Analysis context, providing the config, experiment, and rig.
    show : bool
        Whether to display plots during processing.
    """
    # ! ---- LOAD FROM CONTEXT ----

    config = ctx.config
    experiment = ctx.experiment
    fluidflower = ctx.fluidflower

    # Mypy type checking
    config.check(*list_required_sections(calibration_color_paths_from_context))
    assert config.color is not None
    assert config.calibration is not None
    assert config.calibration.color is not None
    assert config.rig is not None
    assert config.data is not None
    assert config.protocols is not None
    assert config.protocols.imaging is not None
    assert config.protocols.injection is not None
    assert config.protocols.pressure_temperature is not None

    # ! ---- SELECT EMBEDDING AND LABELS ----

    embedding = config.calibration.color.embedding
    assert embedding is not None
    if not isinstance(embedding, ColorPathEmbedding):
        logger.error(
            "calibration.color.embedding '%s' is a %s; only color-path calibration is "
            "implemented.",
            embedding.embedding_id,
            type(embedding).__name__,
        )
        raise NotImplementedError(
            f"Calibration of {type(embedding).__name__} is not yet implemented; "
            "only ColorPathEmbedding is supported."
        )
    selected_basis = embedding.basis
    selected_labels = embedding.get_labels(fluidflower)

    # ! ---- LOAD IMAGES ----

    # Validate that calibration data is configured
    color_cfg = config.calibration.color
    if color_cfg.data_selection is None:
        raise ValueError(
            "calibration.color.data_selection is not configured. "
            "Cannot proceed with calibration."
        )

    # Resolve calibration image paths from the calibration config's data_selection field
    calibration_image_paths = select_image_paths(
        config,
        experiment,
        all=False,
        sub_config=color_cfg,
    )

    # Baseline images (no tracer) are only needed to ignore their color fluctuations.
    # Without any, the rig baseline stands in: an empty list, for which all relative
    # colors are zero (see LabelColorPathMapRegression.get_color_spectrum).
    baseline_images: list[darsia.Image] = []
    uses_baseline_spectrum = color_cfg.color_path is not None and (
        color_cfg.color_path.ignore_baseline_spectrum in ("baseline", "expanded")
    )
    if uses_baseline_spectrum and color_cfg.baseline:
        baseline_sub_config = SimpleNamespace(data_selection=color_cfg.baseline)
        baseline_image_paths = select_image_paths(
            config,
            experiment,
            all=False,
            sub_config=baseline_sub_config,
        )
        with darsia.timed("baseline images"):
            baseline_images = load_images_with_cache(
                rig=fluidflower,
                paths=baseline_image_paths,
                use_cache=config.data.use_cache,
                cache_dir=config.data.cache,
                native=True,
            )
    elif uses_baseline_spectrum:
        logger.info(
            "No baseline images configured (calibration.color.baseline): the rig "
            "baseline is used, i.e. only its own colors (zero relative colors) are "
            "ignored."
        )

    # Cache calibration images for performance
    with darsia.timed("calibration images"):
        calibration_images: list[darsia.Image] = load_images_with_cache(
            rig=fluidflower,
            paths=calibration_image_paths,
            use_cache=config.data.use_cache,
            cache_dir=config.data.cache,
            native=True,
        )

    # ! ---- BUILD CALIBRATION MASK ----

    # Porosity mask ([calibration.color] mask, "none": all pixels) restricted
    # to the union of ROIs listed in the calibration config.
    if color_cfg.mask == "boolean_porosity":
        boolean_porosity = load_image_porosity(
            config.image_porosity, config.data.results
        ).boolean_on(fluidflower.baseline)
    else:
        boolean_porosity = darsia.ones_like(
            fluidflower.baseline, mode="voxels", dtype=bool
        )
    # Masks are combined into new arrays (no copies, nothing changed in place).
    calibration_mask = boolean_porosity
    if color_cfg.rois and config.roi_registry is not None:
        roi_entries = config.roi_registry.resolve_rois(color_cfg.rois)
        rois = [roi_cfg.roi for roi_cfg in roi_entries.values()]
        union_mask = roi_to_mask(rois, boolean_porosity, mode="voxels")
        restricted = boolean_porosity.img & union_mask.img
        if np.any(restricted):
            calibration_mask = _with_data(boolean_porosity, restricted)
        else:
            logger.warning(
                "The union of the provided ROIs does not overlap with the "
                "porosity mask. Falling back to the full porosity mask for "
                "colour-path calibration."
            )

    # Always exclude the union of no_rois, regardless of the positive selection
    # above - e.g. a known-contaminated sub-region that happens to share a label
    # with porous rock, which would otherwise pollute that label's fitted color
    # path even when it is included in 'rois'.
    if color_cfg.no_rois and config.roi_registry is not None:
        no_roi_entries = config.roi_registry.resolve_rois(color_cfg.no_rois)
        no_rois = [roi_cfg.roi for roi_cfg in no_roi_entries.values()]
        exclude_mask = roi_to_mask(no_rois, calibration_mask, mode="voxels")
        calibration_mask = _with_data(
            calibration_mask, calibration_mask.img & ~exclude_mask.img
        )
        if not np.any(calibration_mask.img):
            logger.warning(
                "The union of 'no_rois' excludes the entire calibration mask "
                "(porosity intersected with 'rois', if any). Colour-path "
                "calibration will proceed with an empty mask - check for "
                "overlapping 'rois'/'no_rois' configuration."
            )

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

    with darsia.timed("colour range"):
        tracer_color_range = darsia.ColorRange.from_images(
            images=calibration_images,
            baseline=fluidflower.baseline,
            mask=calibration_mask,
        )

    # ! ---- COLOR PATH TOOL ----

    calibration_cfg = config.calibration.color.color_path
    assert calibration_cfg is not None

    color_path_regression = darsia.LabelColorPathMapRegression(
        labels=selected_labels,
        color_range=tracer_color_range,
        mask=calibration_mask,
        resolution=calibration_cfg.resolution,
        ignore_labels=calibration_cfg.ignore_labels,
    )

    # ! ---- ANALYZE FLUCTUATIONS IN BASELINE IMAGES ----

    ignore_mode = calibration_cfg.ignore_baseline_spectrum
    ignore_spectrum: darsia.LabelColorSpectrumMap | None = None

    if ignore_mode in ("baseline", "expanded"):
        with darsia.timed("baseline spectrum"):
            baseline_color_spectrum: darsia.LabelColorSpectrumMap = (
                color_path_regression.get_color_spectrum(
                    images=baseline_images,
                    baseline=fluidflower.baseline,
                    threshold_significant=calibration_cfg.threshold_baseline,
                    verbose=show,
                )
            )
        baseline_color_spectrum.save(embedding.baseline_color_spectrum_folder)

        if ignore_mode == "expanded":
            # Expand the baseline color spectrum through linear regression
            with darsia.timed("expanded spectrum"):
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

    with darsia.timed("tracer spectrum"):
        tracer_color_spectrum = color_path_regression.get_color_spectrum(
            images=calibration_images,
            baseline=fluidflower.baseline,
            ignore=ignore_spectrum,
            threshold_significant=calibration_cfg.threshold_calibration,
            path=embedding.figures_folder,
            verbose=show,
        )

    # Initialize manual calibration from a previous calibration, if available
    initial_paths = (
        _load_initial_paths(embedding, label_ids_from_image(selected_labels))
        if calibration_cfg.calibration_mode == "manual"
        else {}
    )

    label_ids = label_ids_from_image(selected_labels)

    def save(label_color_path_map: darsia.LabelColorPathMap) -> None:
        save_color_paths(
            label_color_path_map,
            embedding=embedding,
            basis=selected_basis,
            label_ids=label_ids,
            ignore_labels=calibration_cfg.ignore_labels,
        )

    if calibration_cfg.calibration_mode == "manual":
        # One window for all labels; it saves (Save button, or on closing if the
        # user wants to keep unsaved changes)
        with darsia.timed("path preparation"):
            prepared = color_path_regression.prepare_color_path_editing(
                color_spectrum=tracer_color_spectrum,
                ignore=ignore_spectrum,
                num_segments=calibration_cfg.num_segments,
                weighting=calibration_cfg.histogram_weighting,
                ignore_labels=calibration_cfg.ignore_labels,
                initial_paths=initial_paths,
                verbose=show,
            )
        editable = [
            label
            for label, item in prepared.items()
            if not isinstance(item, darsia.ColorPath)
        ]
        stored = len(editable) > 0 and all(label in initial_paths for label in editable)
        saved = edit_color_paths(
            prepared,
            preview_images=calibration_images,
            preview_baseline=fluidflower.baseline,
            labels=color_path_regression.labels,
            mask=color_path_regression.mask,
            save=lambda keys: save(
                color_path_regression.color_path_map(
                    prepared, keys, directory=embedding.figures_folder
                )
            ),
            stored=stored,
            reference_label=embedding.reference_label,
        )
        if not saved:
            logger.info(
                "The color paths in the window were not saved; the stored color "
                "paths are unchanged."
            )
            return
    else:
        # Find a relative color path through the significant boxes
        save(
            color_path_regression.find_color_path(
                color_spectrum=tracer_color_spectrum,
                ignore=ignore_spectrum,
                num_segments=calibration_cfg.num_segments,
                directory=embedding.figures_folder,
                weighting=calibration_cfg.histogram_weighting,
                ignore_labels=calibration_cfg.ignore_labels,
                verbose=show,
            )
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

    Parameters
    ----------
    path : Path | list[Path]
        Path(s) to the configuration file(s).

    Returns
    -------
        List of unique existing paths in deletion order.
    """

    config = FluidFlowerConfig(path, require_data=False, require_results=False)

    paths_to_delete: list[Path] = []
    if config.color is not None:
        for embedding in config.color.resolve_all().values():
            paths_to_delete.append(embedding.root)
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

    Parameters
    ----------
    path : Path | list[Path]
        Path(s) to the configuration file(s).
    require_confirmation : bool
        If True, ask for command-line confirmation before
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
