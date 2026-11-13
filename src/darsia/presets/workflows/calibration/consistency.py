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


def mass_calibration_info(
    embedding, mode: darsia.EvaluationMode | None = None, quantity: str = "mass"
) -> dict:
    """Describe the stored mass calibration of a color path embedding.

    Parameters
    ----------
    embedding : ColorPathEmbedding
        Embedding the mass calibration belongs to.
    mode : EvaluationMode, optional
        Evaluation mode; loaded from the calibration folder if not provided.
    quantity : str
        Quantity of the calibration (`[physics]` interpretation), selecting the folder.

    Returns
    -------
    dict
        Embedding, folder, evaluation mode (data and description), whether it is
        the default for calibrations without stored mode, and the time of saving.

    """
    folder = Path(embedding.calibration_folder(quantity))
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
        quantity = config.physics.interpretation.quantity
        if not Path(embedding.calibration_folder(quantity)).exists():
            return None
        info = mass_calibration_info(embedding, quantity=quantity)
        stored = darsia.EvaluationMode.from_dict(info["evaluation_mode"])
        requested = configured_evaluation_mode(config)
        info["config_description"] = None if requested is None else requested.describe()
        info["config_differs"] = requested is not None and not requested.same_as(stored)
        return info
    except Exception as error:  # a check must never block the workflow
        logger.debug("Could not describe the mass calibration: %s", error)
        return None


def calibration_embedding(config, prefer_analysis: bool = False):
    """Color path embedding of the interpretation calibration of a loaded config.

    By default `[calibration.mass].embedding`, falling back to the embedding of
    `[analysis.mass]`; with `prefer_analysis` the other way round (the embedding an
    analysis actually uses). None if neither resolves to a color path embedding.
    """
    from darsia.signals.color import ColorPathEmbedding

    def from_calibration():
        if config.calibration is not None and config.calibration.mass is not None:
            return config.calibration.mass.embedding
        return None

    def from_analysis():
        if (
            config.color is not None
            and config.analysis is not None
            and config.analysis.mass is not None
        ):
            return config.color.resolve(config.analysis.mass.color)
        return None

    first, second = (
        (from_analysis, from_calibration)
        if prefer_analysis
        else (from_calibration, from_analysis)
    )
    embedding = first()
    if embedding is None:
        embedding = second()
    return embedding if isinstance(embedding, ColorPathEmbedding) else None


def _stored_interpretation_inconsistencies(embedding, folder: Path) -> dict[int, str]:
    """Compare the color paths stored with a calibration with the current ones."""
    if not Path(embedding.color_paths_csv_file).exists():
        return {}
    color_paths = darsia.LabelColorPathMap.load_csv(embedding.color_paths_csv_file)
    interpretation_folder = Path(folder) / "color_path_interpretation"
    interpretation = {
        label: darsia.ColorPathInterpolation.load(
            interpretation_folder / f"color_path_interpretation_{label}"
        )
        for label in color_paths
        if (interpretation_folder / f"color_path_interpretation_{label}.json").exists()
    }
    return color_to_mass_inconsistencies(color_paths, interpretation)


def interpretation_calibration_folder(config, prefer_analysis: bool = False):
    """Calibration folder of the configured interpretation, or None (no color path)."""
    embedding = calibration_embedding(config, prefer_analysis=prefer_analysis)
    if embedding is None:
        return None
    return Path(embedding.calibration_folder(config.physics.interpretation.quantity))


def calibration_outdated_reason(config, prefer_analysis: bool = False) -> str | None:
    """Why the interpretation calibration of a loaded config is outdated, or None.

    Uses the color path fingerprint stored in the calibration metadata (content hash of
    `color_paths.csv`); calibrations without it fall back to comparing the stored key
    colors with the current color paths. None also if there is nothing to compare.
    """
    from darsia.presets.workflows.calibration.metadata import (
        COLOR_PATHS_HASH_KEY,
        file_sha256,
        read_calibration_metadata,
    )

    embedding = calibration_embedding(config, prefer_analysis=prefer_analysis)
    if embedding is None:
        return None
    folder = Path(embedding.calibration_folder(config.physics.interpretation.quantity))
    if not folder.exists():
        return None
    metadata = read_calibration_metadata(folder / "metadata.json") or {}
    stored = (metadata.get("inputs") or {}).get(COLOR_PATHS_HASH_KEY)
    if stored is not None:
        current = file_sha256(embedding.color_paths_csv_file)
        if current is None:
            return "the color paths of this calibration are missing"
        if current != stored:
            return "the color paths were recalibrated after this calibration"
        return None
    inconsistencies = _stored_interpretation_inconsistencies(embedding, folder)
    if not inconsistencies:
        return None
    details = "; ".join(
        f"label {label}: {message}"
        for label, message in sorted(inconsistencies.items())
    )
    return f"the color paths changed after this calibration ({details})"


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

    try:
        config = FluidFlowerConfig(
            config_path, require_data=False, require_results=False
        )
        embedding = calibration_embedding(config, prefer_analysis=True)
        if embedding is None:
            return {}
        folder = embedding.calibration_folder(config.physics.interpretation.quantity)
        return _stored_interpretation_inconsistencies(embedding, folder)
    except Exception as error:  # a check must never block the workflow
        logger.debug("Could not check the mass calibration: %s", error)
        return {}
