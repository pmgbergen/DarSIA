"""Consistency checks between calibration steps."""

import logging
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
