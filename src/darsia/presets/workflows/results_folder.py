"""Results-folder inference and OS file-explorer helpers for workflow GUIs.

Pure, framework-independent logic shared by GUI frontends (e.g. the Qt GUI
under ``darsia.gui``) for inferring a workflow run's results folder and
opening it in the OS file explorer.
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


def _load_workflow_config(config_path: Path) -> dict[str, Any]:
    """Load a workflow TOML config file."""
    return tomllib.loads(config_path.read_text())


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
    config_path: Path, actions: list[str]
) -> Path | None:
    """Return suggested analysis results folder for completed runs."""
    merged = _load_workflow_config(config_path)
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


def _color_path_embedding_root(
    merged: dict[str, Any], results: Path, embedding_id: str
) -> Path:
    """Resolve a [[color_path]] embedding's root folder from merged config.

    Mirrors parse_color_path_embedding's own default
    (<results>/color/color_path/<embedding_id>), honoring an explicit
    per-entry `root =` override the same way that function does.
    """
    color_path_entries = merged.get("color_path")
    if isinstance(color_path_entries, list):
        for entry in color_path_entries:
            if isinstance(entry, dict) and entry.get("name") == embedding_id:
                root_raw = entry.get("root")
                if isinstance(root_raw, str) and root_raw.strip():
                    return Path(root_raw).expanduser()
                break
    return results / "color" / "color_path" / embedding_id


def suggested_workflow_results_folder(
    workflow: str, config_path: Path, actions: list[str]
) -> Path | None:
    """Return suggested output folder for successful GUI workflow runs."""
    merged = _load_workflow_config(config_path)
    results = _results_folder_from_merged_config(merged)
    if results is None:
        return None

    if workflow == "analysis":
        return suggested_analysis_results_folder(config_path, actions)

    selected_actions = {action.strip().lower().replace("_", " ") for action in actions}

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
        if "color embedding" in selected_actions:
            calibration = merged.get("calibration")
            color_section = (
                calibration.get("color") if isinstance(calibration, dict) else None
            )
            embedding_id = (
                color_section.get("embedding")
                if isinstance(color_section, dict)
                else None
            )
            if isinstance(embedding_id, str) and embedding_id.strip():
                return _color_path_embedding_root(merged, results, embedding_id.strip())
            return None
        if "mass" in selected_actions or "default mass" in selected_actions:
            calibration = merged.get("calibration")
            mass_section = (
                calibration.get("mass") if isinstance(calibration, dict) else None
            )
            embedding_id = (
                mass_section.get("embedding")
                if isinstance(mass_section, dict)
                else None
            )
            if isinstance(embedding_id, str) and embedding_id.strip():
                return (
                    _color_path_embedding_root(merged, results, embedding_id.strip())
                    / "interpolation"
                    / "mass"
                )
            return None
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


def has_workflow_output(workflow: str, config_path: Path, actions: list[str]) -> bool:
    """Return whether a workflow step already has non-empty output on disk.

    Drives the GUI sidebar's per-step completion dot. Best-effort: any failure
    to resolve or read the config is treated as "not started" rather than
    raised, since this powers a passive visual hint, not a gate.
    """
    try:
        folder = suggested_workflow_results_folder(workflow, config_path, actions)
    except (OSError, ValueError, tomllib.TOMLDecodeError):
        return False
    if folder is None:
        return False
    try:
        return folder.exists() and any(folder.iterdir())
    except OSError:
        return False


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _resolve_output_images(
    workflow: str, config_path: Path, actions: list[str]
) -> list[Path]:
    """Return every image file under the resolved results folder for this
    workflow step, in arbitrary order, or [] if no folder resolves, the
    folder does not exist yet, or it contains no image file.

    Recursively globs for an image rather than encoding each step's own
    subfolder convention: those conventions vary a lot across steps (flat,
    mode/fmt/stem, fmt/layer/stem, nested plot-kind trees) and shift
    whenever a workflow's export logic changes, so this stays
    low-maintenance at the cost of not knowing which image variant it
    picked among several a step might produce.
    """
    try:
        folder = suggested_workflow_results_folder(workflow, config_path, actions)
    except (OSError, ValueError, tomllib.TOMLDecodeError):
        return []
    if folder is None:
        return []
    try:
        if not folder.exists():
            return []
        return [
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        ]
    except OSError:
        return []


def list_workflow_output_images(
    workflow: str, config_path: Path, actions: list[str]
) -> list[Path]:
    """Return every image file for this workflow step, sorted by filename.
    Drives the GUI's View panel's Results-mode browsing.
    """
    images = _resolve_output_images(workflow, config_path, actions)
    return sorted(images, key=lambda path: path.name)


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
