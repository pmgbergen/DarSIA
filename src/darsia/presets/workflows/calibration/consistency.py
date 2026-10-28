"""Consistency checks between calibration steps."""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

import darsia

logger = logging.getLogger(__name__)


def color_to_mass_inconsistencies(
    color_paths: darsia.LabelColorPathMap,
    color_path_interpretation: dict[int, darsia.ColorPathInterpolation],
) -> dict[int, str]:
    """Compare the color paths of a mass calibration with the current color paths.

    A mass calibration becomes outdated when the color paths are recalibrated
    afterwards, e.g. when the number of key colors of a label changes.

    Parameters
    ----------
    color_paths : LabelColorPathMap
        Current color paths (result of the color path calibration).
    color_path_interpretation : dict[int, ColorPathInterpolation]
        Color path interpretations stored with the mass calibration.

    Returns
    -------
    dict[int, str]
        Description of the mismatch per label; empty if everything is consistent.

    """
    inconsistencies: dict[int, str] = {}
    for label, interpretation in color_path_interpretation.items():
        if label not in color_paths:
            inconsistencies[label] = "no color path in the current color paths"
            continue
        stored = interpretation.color_path.colors
        current = color_paths[label].colors
        if len(stored) != len(current):
            inconsistencies[label] = (
                f"{len(stored)} key colors in the mass calibration, "
                f"but {len(current)} in the current color paths"
            )
        elif not np.allclose(stored, current, atol=1e-6):
            inconsistencies[label] = "key colors differ from the current color paths"
    return inconsistencies


def mass_calibration_info(embedding, mode: darsia.EvaluationMode | None = None) -> dict:
    """Describe the stored mass calibration of a color path embedding.

    Parameters
    ----------
    embedding : ColorPathEmbedding
        Embedding the mass calibration belongs to.
    mode : EvaluationMode, optional
        Evaluation mode; loaded from the calibration folder if not provided.

    Returns
    -------
    dict
        Embedding, folder, evaluation mode (data and description), whether it is
        the default for calibrations without stored mode, and the time of saving.

    """
    folder = Path(embedding.color_to_mass_folder)
    legacy = not (folder / darsia.EvaluationMode.FILE_NAME).exists()
    if mode is None:
        mode = darsia.EvaluationMode.load(folder)
    reference = folder / "metadata.json"
    if not reference.exists():
        reference = folder
    saved_at = (
        datetime.fromtimestamp(reference.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        if reference.exists()
        else None
    )
    return {
        "embedding": embedding.embedding_id,
        "folder": str(folder),
        "evaluation_mode": mode.to_dict(),
        "description": mode.describe(),
        "legacy_without_evaluation_file": legacy,
        "saved_at": saved_at,
    }


def mass_calibration_status_line(info: dict) -> str:
    """One-line summary of :func:`mass_calibration_info` for logs and the GUI."""
    legacy = ", no evaluation.json" if info["legacy_without_evaluation_file"] else ""
    saved = "" if info["saved_at"] is None else f", saved {info['saved_at']}"
    return (
        f"Mass calibration of embedding '{info['embedding']}': "
        f"{info['description']}{legacy}{saved}"
    )


def configured_evaluation_mode(config) -> darsia.EvaluationMode | None:
    """Evaluation mode requested by ``[calibration.mass]`` of a config, if any."""
    if config.calibration is None or config.calibration.mass is None:
        return None
    mass_cfg = config.calibration.mass
    return darsia.EvaluationMode.from_name(
        mass_cfg.evaluation_mode, mass_cfg.evaluation_parameters()
    )


def mass_calibration_status(config_path: Path) -> dict | None:
    """Stored mass calibration of a config and whether the config differs from it.

    Lightweight (no images are loaded). Returns None if there is nothing to describe
    (no mass analysis, no color path embedding, no stored mass calibration, unreadable
    files). Otherwise the dictionary of :func:`mass_calibration_info`, extended by
    ``config_description`` and ``config_differs`` (the evaluation mode which
    ``[calibration.mass]`` requests, and whether it differs from the stored one).

    """
    from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
    from darsia.signals.color import ColorPathEmbedding

    try:
        config = FluidFlowerConfig(
            config_path, require_data=False, require_results=False
        )
        if (
            config.color is None
            or config.analysis is None
            or config.analysis.mass is None
        ):
            return None
        embedding = config.color.resolve(config.analysis.mass.color)
        if not isinstance(embedding, ColorPathEmbedding):
            return None
        if not Path(embedding.color_to_mass_folder).exists():
            return None
        info = mass_calibration_info(embedding)
        stored = darsia.EvaluationMode.from_dict(info["evaluation_mode"])
        requested = configured_evaluation_mode(config)
        info["config_description"] = None if requested is None else requested.describe()
        info["config_differs"] = requested is not None and not requested.same_as(stored)
        return info
    except Exception as error:  # a check must never block the workflow
        logger.debug("Could not describe the mass calibration: %s", error)
        return None


def outdated_mass_calibration(config_path: Path) -> dict[int, str]:
    """Check the stored mass calibration of a config against its color paths.

    Lightweight (no images are loaded), e.g. for a check before a GUI run. Anything
    that cannot be compared (no mass analysis, no color paths, no mass calibration,
    unreadable files) results in an empty dictionary.

    Parameters
    ----------
    config_path : Path
        Config file of the experiment.

    Returns
    -------
    dict[int, str]
        Description of the mismatch per label; empty if consistent or not comparable.

    """
    from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
    from darsia.signals.color import ColorPathEmbedding

    try:
        config = FluidFlowerConfig(
            config_path, require_data=False, require_results=False
        )
        if (
            config.color is None
            or config.analysis is None
            or config.analysis.mass is None
        ):
            return {}
        embedding = config.color.resolve(config.analysis.mass.color)
        if not isinstance(embedding, ColorPathEmbedding):
            return {}
        if not Path(embedding.color_paths_csv_file).exists():
            return {}
        folder = embedding.color_to_mass_folder / "color_path_interpretation"
        color_paths = darsia.LabelColorPathMap.load_csv(embedding.color_paths_csv_file)
        interpretation = {
            label: darsia.ColorPathInterpolation.load(
                folder / f"color_path_interpretation_{label}"
            )
            for label in color_paths
            if (folder / f"color_path_interpretation_{label}.json").exists()
        }
        return color_to_mass_inconsistencies(color_paths, interpretation)
    except Exception as error:  # a check must never block the workflow
        logger.debug("Could not check the mass calibration: %s", error)
        return {}
