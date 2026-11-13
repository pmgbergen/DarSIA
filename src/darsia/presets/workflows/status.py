"""Status of workflow steps: missing, valid or outdated.

A step is *missing* without output, *valid* with output that still matches its inputs, and
*outdated* if an input changed after the step ran. Covered dependencies (colour chain):

- color path -> interpretation calibration: content hash of `color_paths.csv` stored in
  the calibration metadata (legacy calibrations: comparison of the stored key colors);
- interpretation calibration -> analyses that use it (field maps, volume, and
  segmentation/fingers/thresholding with a calibrated mode): outdated if the calibration
  is outdated or newer than the results (modification times).

Best effort: a status never raises; anything that cannot be checked is reported by its
output only (missing or valid).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from darsia.presets.workflows.results_folder import (
    FIELD_MAPS_CSV,
    has_workflow_output,
    suggested_workflow_results_folder,
)

logger = logging.getLogger(__name__)

MISSING = "missing"
VALID = "valid"
OUTDATED = "outdated"

CALIBRATION_STEPS = ("mass", "default mass")
"""Calibration actions writing the interpretation calibration."""

ANALYSIS_STEPS_USING_CALIBRATION = (
    "mass",
    "volume",
    "segmentation",
    "fingers",
    "thresholding",
)
"""Analysis actions that may use the interpretation calibration."""


@dataclass(frozen=True)
class StepStatus:
    """Status of one workflow step."""

    state: str
    """One of MISSING, VALID, OUTDATED."""
    reason: str | None = None
    """Why the step is outdated (None otherwise)."""


def _newest_mtime(folder: Path, depth: int = 2) -> float | None:
    """Newest modification time of the files in a folder, up to `depth` levels down.

    Files, not folders: overwriting a file does not update the folder's time, so a rerun
    that overwrites its results would otherwise still look older than its inputs.
    """
    newest = None
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return None
    for entry in entries:
        try:
            if entry.is_file():
                mtime = entry.stat().st_mtime
            elif entry.is_dir() and depth > 0:
                mtime = _newest_mtime(Path(entry.path), depth - 1)
            else:
                continue
        except OSError:
            continue
        if mtime is not None and (newest is None or mtime > newest):
            newest = mtime
    return newest


def _load_config(config_path: Path):
    from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig

    return FluidFlowerConfig(config_path, require_data=False, require_results=False)


def _calibration_status(config) -> StepStatus:
    from darsia.presets.workflows.calibration.consistency import (
        calibration_outdated_reason,
    )

    reason = calibration_outdated_reason(config)
    return StepStatus(VALID) if reason is None else StepStatus(OUTDATED, reason)


def _results_time(config_path: Path, action: str) -> float | None:
    """Time of the newest output of an analysis step."""
    folder = suggested_workflow_results_folder("analysis", config_path, [action])
    times = [] if folder is None else [_newest_mtime(folder)]
    if action == "mass" and folder is not None:
        csv = folder.parent / FIELD_MAPS_CSV
        if csv.exists():
            times.append(csv.stat().st_mtime)
    times = [t for t in times if t is not None]
    return max(times) if times else None


def _analysis_status(config, config_path: Path, action: str) -> StepStatus:
    from darsia.presets.workflows.analysis.analysis_context import (
        require_color_to_mass,
    )
    from darsia.presets.workflows.calibration.consistency import (
        calibration_outdated_reason,
        interpretation_calibration_folder,
    )

    if not require_color_to_mass(config, **{f"include_{action}": True}):
        return StepStatus(VALID)
    folder = interpretation_calibration_folder(config, prefer_analysis=True)
    if folder is None or not (folder / "metadata.json").exists():
        return StepStatus(VALID)
    reason = calibration_outdated_reason(config, prefer_analysis=True)
    if reason is not None:
        return StepStatus(OUTDATED, f"its calibration is outdated: {reason}")
    results_time = _results_time(config_path, action)
    if results_time is not None and (
        (folder / "metadata.json").stat().st_mtime > results_time
    ):
        return StepStatus(
            OUTDATED, "the interpretation was recalibrated after this analysis"
        )
    return StepStatus(VALID)


def step_statuses(
    config_path: Path, steps: list[tuple[str, str]]
) -> dict[tuple[str, str], StepStatus]:
    """Status of several workflow steps of one config.

    Parameters
    ----------
    config_path : Path
        Config file of the experiment.
    steps : list[tuple[str, str]]
        (workflow, action) pairs, e.g. ("calibration", "mass"), ("analysis", "mass").

    Returns
    -------
    dict[tuple[str, str], StepStatus]
        Status per step. The config is loaded at most once, and only if a step with
        output depends on the interpretation calibration.
    """
    config_path = Path(config_path)
    statuses: dict[tuple[str, str], StepStatus] = {}
    config = None
    config_failed = False
    calibration_status = None

    for workflow, action in steps:
        if not has_workflow_output(workflow, config_path, [action]):
            statuses[(workflow, action)] = StepStatus(MISSING)
            continue
        depends = (workflow == "calibration" and action in CALIBRATION_STEPS) or (
            workflow == "analysis" and action in ANALYSIS_STEPS_USING_CALIBRATION
        )
        if not depends:
            statuses[(workflow, action)] = StepStatus(VALID)
            continue
        try:
            if config is None and not config_failed:
                config = _load_config(config_path)
            if config is None:
                raise RuntimeError("config could not be loaded")
            if workflow == "calibration":
                if calibration_status is None:
                    calibration_status = _calibration_status(config)
                statuses[(workflow, action)] = calibration_status
            else:
                statuses[(workflow, action)] = _analysis_status(
                    config, config_path, action
                )
        except Exception as error:  # a status must never block the GUI
            config_failed = config_failed or config is None
            logger.debug("Could not check %s/%s: %s", workflow, action, error)
            statuses[(workflow, action)] = StepStatus(VALID)
    return statuses
