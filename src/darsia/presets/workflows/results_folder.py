"""Results-folder inference and OS file-explorer helpers for workflow GUIs.

Extracted from ``user_interface_gui.py`` (the tkinter GUI) so that GUI
frontends other than tkinter (e.g. the Qt GUI under ``darsia.gui``) can reuse
this pure, framework-independent logic without depending on a module slated
for removal once the tkinter GUI is deprecated.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

_ANALYSIS_MODE_ACTIONS = {
    "cropping",
    "segmentation",
    "mass",
    "volume",
    "fingers",
    "thresholding",
}
_ANALYSIS_MODE_DEFAULT_SUBFOLDER = {
    "segmentation": "segmentation",
    "mass": "mass",
    "volume": "volume",
    "fingers": "fingers",
    "thresholding": "thresholding",
}


def _deep_merge_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge update dictionary into base dictionary."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def _load_merged_workflow_config(config_paths: list[Path]) -> dict[str, Any]:
    """Load and deeply merge workflow TOML config files."""
    merged: dict[str, Any] = {}
    for path in config_paths:
        _deep_merge_dict(merged, tomllib.loads(path.read_text()))
    return merged


def _results_folder_from_merged_config(merged: dict[str, Any]) -> Path | None:
    """Extract configured [data].results folder from merged config."""
    data = merged.get("data")
    if not isinstance(data, dict):
        return None
    results_raw = data.get("results")
    if not isinstance(results_raw, str) or not results_raw.strip():
        return None
    return Path(results_raw).expanduser()


def suggested_analysis_results_folder(
    config_paths: list[Path], actions: list[str]
) -> Path | None:
    """Return suggested analysis results folder for completed runs."""
    merged = _load_merged_workflow_config(config_paths)
    results = _results_folder_from_merged_config(merged)
    if results is None:
        return None

    mode_actions = [action for action in actions if action in _ANALYSIS_MODE_ACTIONS]
    if len(mode_actions) != 1:
        return results

    mode = mode_actions[0]
    if mode == "cropping":
        return results / "cropping"

    analysis = merged.get("analysis")
    if isinstance(analysis, dict):
        mode_section = analysis.get(mode)
        if isinstance(mode_section, dict):
            folder = mode_section.get("folder")
            if isinstance(folder, str) and folder.strip():
                return Path(folder).expanduser()

    return results / _ANALYSIS_MODE_DEFAULT_SUBFOLDER[mode]


def suggested_workflow_results_folder(
    workflow: str, config_paths: list[Path], actions: list[str]
) -> Path | None:
    """Return suggested output folder for successful GUI workflow runs."""
    merged = _load_merged_workflow_config(config_paths)
    results = _results_folder_from_merged_config(merged)
    if results is None:
        return None

    if workflow == "analysis":
        return suggested_analysis_results_folder(config_paths, actions)

    selected_actions = {action.strip().lower() for action in actions}

    if workflow == "setup":
        setup_candidates: list[Path] = []
        if "depth" in selected_actions:
            setup_candidates.append(results / "setup" / "depth")
        if "segmentation" in selected_actions:
            setup_candidates.append(results / "setup" / "labels")
        if "facies" in selected_actions:
            setup_candidates.append(results / "setup" / "facies")
        if "rig" in selected_actions:
            setup_candidates.append(results / "setup" / "rig")
        if "protocols" in selected_actions:
            setup_candidates.append(results / "setup")
        if "all" in selected_actions:
            setup_candidates.append(results / "setup")
        if len(setup_candidates) == 0:
            return None
        all_setup_same = all(path == setup_candidates[0] for path in setup_candidates)
        return setup_candidates[0] if all_setup_same else results / "setup"

    if workflow == "calibration":
        if (
            "color embedding" in selected_actions
            or "mass" in selected_actions
            or "default mass" in selected_actions
        ):
            return results / "calibration"
        return None

    if workflow == "comparison":
        has_events = "events" in selected_actions
        has_wasserstein = (
            "wasserstein compute" in selected_actions
            or "wasserstein assemble" in selected_actions
        )
        if has_events and has_wasserstein:
            return results
        if has_events:
            events = merged.get("events")
            if isinstance(events, dict):
                events_path_raw = events.get("path")
                if isinstance(events_path_raw, str) and events_path_raw.strip():
                    return Path(events_path_raw).expanduser().parent
            return results / "events"
        if has_wasserstein:
            wasserstein = merged.get("wasserstein")
            if isinstance(wasserstein, dict):
                wasserstein_results_raw = wasserstein.get("results")
                if (
                    isinstance(wasserstein_results_raw, str)
                    and wasserstein_results_raw.strip()
                ):
                    return Path(wasserstein_results_raw).expanduser()
            return results / "wasserstein"
        return None

    if workflow == "utils":
        utils_candidates: list[Path] = []
        if "media" in selected_actions:
            utils_candidates.append(results / "videos")
        if "export calibration" in selected_actions:
            utils_candidates.append(results / "calibration")
        if "import calibration" in selected_actions:
            utils_candidates.append(results / "calibration")
        if "download" in selected_actions:
            download = merged.get("download")
            if isinstance(download, dict):
                folder_raw = download.get("folder")
                if isinstance(folder_raw, str) and folder_raw.strip():
                    utils_candidates.append(Path(folder_raw).expanduser())
                else:
                    utils_candidates.append(results / "raw_data")
            else:
                utils_candidates.append(results / "raw_data")
        if len(utils_candidates) == 0:
            return None
        all_utils_same = all(path == utils_candidates[0] for path in utils_candidates)
        return utils_candidates[0] if all_utils_same else results

    return None


def open_in_file_explorer(path: Path) -> None:
    """Open path in the OS file explorer."""
    target = path.expanduser().resolve()
    if not target.exists():
        for parent in target.parents:
            if parent.exists():
                target = parent
                break
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")
    if target.is_file():
        target = target.parent

    if os.name == "nt":
        os.startfile(str(target))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        try:
            subprocess.run(["open", str(target)], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise RuntimeError("Failed to open folder with 'open'.") from e
    else:
        try:
            subprocess.run(["xdg-open", str(target)], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise RuntimeError("Failed to open folder with 'xdg-open'.") from e
