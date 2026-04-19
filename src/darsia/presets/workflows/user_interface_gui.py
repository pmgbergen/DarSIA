"""GUI interface for preset workflows.

This GUI is additive and does not replace the existing command-line
``user_interface_*`` modules.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import time
import tomllib
import traceback
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from queue import Empty, Full
from typing import Any, Callable, Protocol, TypedDict

from darsia.presets.workflows.analysis.progress import (
    AnalysisProgressEvent,
    normalize_progress_event,
)
from darsia.presets.workflows.rig import Rig

logger = logging.getLogger(__name__)
SESSION_CACHE_FILE_NAME = "workflows_gui_session.json"
SESSION_CACHE_VERSION = 1
PREVIEW_LIST_LIMIT = 8
WORKFLOW_ERROR_DETAILS_PREFIX = "__DARSIA_WORKFLOW_ERROR_DETAILS__:"
UTILS_CONFLICT_PREVIEW_LIMIT = 8


class SupportsLogQueue(Protocol):
    """Protocol for queue-like objects used for log forwarding."""

    def put(self, obj: str) -> Any:
        """Put one log message in the queue."""


class SupportsQueue(Protocol):
    """Protocol for queue-like objects used for generic payload forwarding."""

    def get_nowait(self) -> Any:
        """Get one queue element without blocking."""

    def put_nowait(self, obj: Any) -> Any:
        """Put one queue element without blocking."""


class UtilsWorkflowOptions(TypedDict):
    media: bool
    download: bool
    export_calibration: bool
    import_calibration: bool
    export_bundle: str
    import_bundle: str
    import_conflict_action: str


def _require_tkinter() -> tuple[Any, Any, Any, Any]:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "tkinter is required for the DarSIA GUI. Install Python Tk support "
            "(e.g. apt install python3-tk on Debian/Ubuntu) and retry."
        ) from e
    return tk, filedialog, messagebox, ttk


def resolve_rig_class(spec: str) -> type[Rig]:
    """Resolve a rig class from ``module:ClassName`` notation."""
    if spec.strip() == "":
        return Rig
    if ":" not in spec:
        raise ValueError("Rig class must be formatted as 'module.path:ClassName'.")

    module_name, class_name = spec.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ValueError(f"Class '{class_name}' not found in module '{module_name}'.")
    if not isinstance(cls, type) or not issubclass(cls, Rig):
        raise ValueError(f"'{spec}' is not a subclass of Rig.")
    return cls


def normalize_paths(paths: list[str]) -> list[Path]:
    """Normalize path strings to unique absolute paths preserving order."""
    unique: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        stripped = raw.strip()
        if not stripped:
            continue
        path = Path(stripped).expanduser().resolve()
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def deduplicate_paths(paths: list[Path]) -> list[Path]:
    """Deduplicate Path objects preserving order."""
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def default_session_cache_file() -> Path:
    """Return default path for GUI session cache file."""
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    base = (
        Path(xdg_cache_home).expanduser() if xdg_cache_home else Path.home() / ".cache"
    )
    return base / "darsia" / SESSION_CACHE_FILE_NAME


def read_session_cache(path: Path) -> tuple[list[Path], str]:
    """Read cached GUI session (config paths + rig spec)."""
    if not path.exists():
        return [], ""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to read session cache {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"Invalid session cache format in {path}.")
    version = data.get("version", SESSION_CACHE_VERSION)
    if not isinstance(version, int) or version != SESSION_CACHE_VERSION:
        raise ValueError(
            f"Unsupported session cache version in {path}: {version}. "
            f"Expected {SESSION_CACHE_VERSION}."
        )

    config_paths_raw = data.get("config_paths", [])
    if not isinstance(config_paths_raw, list):
        raise ValueError(f"Invalid session cache format in {path}: config_paths.")
    config_paths = normalize_paths(
        [item for item in config_paths_raw if isinstance(item, str)]
    )

    rig_spec = data.get("rig_spec", "")
    if not isinstance(rig_spec, str):
        rig_spec = ""
    return config_paths, rig_spec


def write_session_cache(path: Path, config_paths: list[Path], rig_spec: str) -> None:
    """Write cached GUI session (config paths + rig spec)."""
    payload = {
        "version": SESSION_CACHE_VERSION,
        "config_paths": [str(p) for p in config_paths],
        "rig_spec": rig_spec,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _find_template_file() -> Path:
    candidates: list[Path] = []
    try:
        packaged = resources.files("darsia.presets.workflows.templates").joinpath(
            "config.toml"
        )
        candidates.append(Path(str(packaged)))
    except (ModuleNotFoundError, AttributeError, FileNotFoundError):
        pass
    candidates.append(Path(__file__).resolve().parent / "templates" / "config.toml")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _configure_queue_logging(log_queue: SupportsLogQueue) -> None:
    """Attach queue logging handler to root logger."""
    handler = QueueLogHandler(log_queue)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)


def _worker_entry(
    log_queue: SupportsLogQueue, fn: Callable[..., None], args: tuple[Any, ...]
) -> None:
    """Worker process entry point with queue-forwarded logging."""
    _configure_queue_logging(log_queue)
    try:
        fn(*args)
    except Exception:
        log_queue.put(encode_workflow_error_details(traceback.format_exc()))
        raise


def clear_queue(queue: SupportsQueue) -> None:
    """Drain all currently queued items."""
    try:
        while True:
            queue.get_nowait()
    except Empty:
        pass


def publish_latest_queue_item(queue: SupportsQueue, payload: Any) -> None:
    """Keep only the latest payload in queue."""
    clear_queue(queue)
    try:
        queue.put_nowait(payload)
    except Full:
        pass


def encode_workflow_error_details(details: str) -> str:
    """Encode workflow error details for transfer over the log queue."""
    return f"{WORKFLOW_ERROR_DETAILS_PREFIX}{details}"


def decode_workflow_error_details(message: str) -> str | None:
    """Decode workflow error details from a log-queue message."""
    if message.startswith(WORKFLOW_ERROR_DETAILS_PREFIX):
        return message[len(WORKFLOW_ERROR_DETAILS_PREFIX) :]
    return None


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
        if "protocol" in selected_actions:
            setup_candidates.append(results / "setup")
        if "all" in selected_actions:
            setup_candidates.append(results / "setup")
        if len(setup_candidates) == 0:
            return None
        all_setup_same = all(path == setup_candidates[0] for path in setup_candidates)
        return setup_candidates[0] if all_setup_same else results / "setup"

    if workflow == "calibration":
        if (
            "color paths" in selected_actions
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

_BATCH_MONITOR_IDLE_MESSAGE = "No active analysis batch."
_BATCH_MONITOR_WAITING_MESSAGE = "Waiting for analysis progress..."


def enabled_option_labels(
    options: dict[str, bool], *, exclude: set[str] | None = None
) -> list[str]:
    """Return enabled option labels suitable for display."""
    excluded = exclude or set()
    return [
        key.replace("_", " ")
        for key, value in options.items()
        if value and key not in excluded
    ]


def format_duration_seconds(seconds: float | None) -> str:
    """Format duration in seconds as H:MM:SS or M:SS."""
    if seconds is None or not isinstance(seconds, (float, int)):
        return "n/a"
    seconds_float = float(seconds)
    if seconds_float < 0 or seconds_float != seconds_float:
        return "n/a"
    seconds_int = int(round(seconds_float))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def rolling_average_runtime(
    runtimes: list[float], *, max_samples: int = 5
) -> float | None:
    """Return rolling average runtime based on last valid samples."""
    if max_samples <= 0:
        return None
    valid = [
        runtime
        for runtime in runtimes
        if isinstance(runtime, (float, int)) and runtime > 0 and runtime == runtime
    ]
    if len(valid) == 0:
        return None
    tail = valid[-max_samples:]
    return float(sum(tail) / len(tail))


def remaining_image_count(processed: int, total: int) -> int:
    """Return remaining image count."""
    return max(0, max(0, total) - max(0, processed))


def estimate_remaining_time_seconds(
    avg_runtime_seconds: float | None,
    processed_images: int,
    total_images: int,
) -> float | None:
    """Estimate remaining time from average runtime and remaining image count."""
    if avg_runtime_seconds is None:
        return None
    if processed_images < 2:
        return None
    if avg_runtime_seconds <= 0:
        return None
    remaining = remaining_image_count(processed_images, total_images)
    if remaining <= 0:
        return 0.0
    return avg_runtime_seconds * remaining


def progress_percent(processed: int, total: int) -> float:
    """Compute progress percentage in [0, 100]."""
    if total <= 0:
        return 0.0
    return min(100.0, max(0.0, 100.0 * max(0, processed) / total))


def format_batch_monitor_text(
    *,
    step: str,
    image_path: str,
    processed: int,
    total: int,
    last_image_seconds: float | None,
    step_elapsed_seconds: float | None,
    overall_elapsed_seconds: float | None,
    eta_seconds: float | None,
) -> str:
    """Format human-readable batch monitor status text."""
    percent = progress_percent(processed, total)
    step_name = step if step else "n/a"
    current_image = image_path if image_path else "n/a"
    return (
        f"Current analysis step: {step_name}\n"
        f"Current image path: {current_image}\n"
        f"Image count: {processed}/{total} ({percent:.1f}%)\n"
        f"Last image elapsed: {format_duration_seconds(last_image_seconds)}\n"
        f"Current step elapsed: {format_duration_seconds(step_elapsed_seconds)}\n"
        f"Overall elapsed: {format_duration_seconds(overall_elapsed_seconds)}\n"
        f"Estimated remaining: {format_duration_seconds(eta_seconds)}"
    )


def resolve_utils_bundle_defaults(config_paths: list[str]) -> tuple[str, str]:
    """Resolve configured default bundle paths for utils export/import."""
    from darsia.presets.workflows.config.workflow_utils import WorkflowUtilsConfig

    paths = normalize_paths(config_paths)
    if not paths:
        return "", ""
    try:
        config = WorkflowUtilsConfig().load(paths)
    except KeyError:
        return "", ""
    export_bundle = (
        ""
        if config.export_calibration_bundle is None
        else str(config.export_calibration_bundle)
    )
    import_bundle = (
        ""
        if config.import_calibration_bundle is None
        else str(config.import_calibration_bundle)
    )
    return export_bundle, import_bundle


def map_conflict_dialog_choice_to_policy(choice: bool | None) -> str | None:
    """Map askyesnocancel result to import conflict policy."""
    if choice is True:
        return "overwrite_all"
    if choice is False:
        return "skip_all"
    return None


def format_workflow_start_message(
    workflow: str, actions: list[str], config_paths: list[Path], rig_spec: str
) -> str:
    """Format a detailed run-start message."""
    action_str = ", ".join(actions) if actions else "none"
    config_str = ", ".join(str(path) for path in config_paths)
    rig_str = rig_spec.strip() or "darsia.presets.workflows.rig:Rig"
    return (
        f"Starting {workflow} workflow. Actions: {action_str}. "
        f"Configs: {config_str}. Rig: {rig_str}."
    )


def format_workflow_done_message(
    workflow: str, actions: list[str], config_count: int, duration_seconds: float
) -> str:
    """Format a detailed completion message."""
    action_str = ", ".join(actions) if actions else "none"
    return (
        f"{workflow.capitalize()} completed. Actions: {action_str}. "
        f"Configs: {config_count}. Duration: {duration_seconds:.1f}s."
    )


def format_workflow_error_message(
    workflow: str, actions: list[str], exit_code: int | None
) -> str:
    """Format a detailed failure message."""
    return (
        f"ERROR: {workflow} workflow failed with exit code {exit_code}. "
        f"Actions: {', '.join(actions) or 'none'}."
    )


def completion_dialog_spec(
    workflow: str, exit_code: int | None, abort_requested: bool
) -> tuple[str, str, str] | None:
    """Return modal dialog information for terminal workflow states."""
    if abort_requested:
        return None
    if exit_code == 0:
        return ("info", "Done", f"{workflow.capitalize()} workflow completed.")
    return (
        "error",
        "Error",
        f"{workflow.capitalize()} workflow failed with exit code {exit_code}.",
    )


def format_error_details_text(details: str) -> str:
    """Normalize detailed traceback text for error-detail display."""
    stripped_details = details.strip()
    if not stripped_details:
        return "No workflow error details available."
    return stripped_details


def abort_process(process: mp.Process | None) -> bool:
    """Abort a running process.

    Returns:
        True if a running process was aborted, otherwise False.
    """
    if process is None or not process.is_alive():
        return False
    process.terminate()
    process.join(timeout=1.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)
    return True


def _run_setup_workflow(
    config_paths: list[str], rig_spec: str, options: dict[str, bool]
) -> None:
    """Run setup workflow in a worker process."""
    from darsia.presets.workflows.setup.setup_depth import setup_depth_map
    from darsia.presets.workflows.setup.setup_facies import setup_facies
    from darsia.presets.workflows.setup.setup_labeling import segment_colored_image
    from darsia.presets.workflows.setup.setup_protocols import setup_imaging_protocol
    from darsia.presets.workflows.setup.setup_rig import delete_rig, setup_rig

    paths = normalize_paths(config_paths)
    rig_cls = resolve_rig_class(rig_spec)
    show = options["show"]
    if options["all"] or options["depth"]:
        setup_depth_map(paths, key="depth", show=show)
    if options["all"] or options["segmentation"]:
        segment_colored_image(paths, show=show)
    if options["all"] or options["facies"]:
        setup_facies(rig_cls, paths, show=show)
    if options["all"] or options["rig"]:
        setup_rig(rig_cls, paths, show=show)
    if options["protocol"]:
        setup_imaging_protocol(paths, force=options["force"], show=show)
    if options["delete_rig"]:
        delete_rig(rig_cls, paths, show=show)


def _run_calibration_workflow(
    config_paths: list[str], rig_spec: str, options: dict[str, bool]
) -> None:
    """Run calibration workflow in a worker process."""
    from darsia.presets.workflows.calibration import (
        calibration_color_to_mass_analysis as c2m_analysis_module,
    )
    from darsia.presets.workflows.calibration.calibration_color_paths import (
        calibration_color_paths,
        delete_calibration,
    )

    paths = normalize_paths(config_paths)
    rig_cls = resolve_rig_class(rig_spec)
    if options["delete"]:
        delete_calibration(
            paths,
            require_confirmation=not options.get("skip_delete_confirmation", False),
        )
        return
    if options["color_paths"]:
        calibration_color_paths(rig_cls, paths, options["show"])
    if options["mass"] or options["default_mass"]:
        c2m_analysis_module.calibration_color_to_mass_analysis(
            rig_cls,
            paths,
            reset=options["reset"],
            show=options["show"],
            default=options["default_mass"],
        )


def _run_analysis_workflow(
    config_paths: list[str],
    rig_spec: str,
    options: dict[str, bool],
    stream_queue: SupportsQueue | None = None,
    progress_queue: SupportsQueue | None = None,
) -> None:
    """Run analysis workflow in a worker process."""
    from darsia.presets.workflows.user_interface_analysis import run_analysis

    paths = normalize_paths(config_paths)
    rig_cls = resolve_rig_class(rig_spec)
    stream_callback: Callable[[dict[str, bytes] | None], None] | None = None
    progress_callback: Callable[[AnalysisProgressEvent], None] | None = None
    if options.get("streaming", False) and stream_queue is not None:

        def _stream_callback(payload: dict[str, bytes] | None) -> None:
            try:
                publish_latest_queue_item(stream_queue, payload)
            except Exception:
                logger.exception("Failed to publish stream payload to GUI queue.")
                try:
                    publish_latest_queue_item(
                        stream_queue,
                        {"__error__": b"Failed to publish stream data to queue."},
                    )
                except Exception:
                    pass

        stream_callback = _stream_callback
    if progress_queue is not None:

        def _progress_callback(payload: AnalysisProgressEvent) -> None:
            try:
                publish_latest_queue_item(progress_queue, payload)
            except Exception:
                logger.exception("Failed to publish progress payload to GUI queue.")

        progress_callback = _progress_callback
    args = argparse.Namespace(
        config=paths,
        all=options["all"],
        cropping=options["cropping"],
        segmentation=options["segmentation"],
        fingers=options["fingers"],
        mass=options["mass"],
        volume=options["volume"],
        thresholding=options.get("thresholding", False),
        show=options["show"],
        info=False,
    )
    run_analysis(
        rig_cls,
        args,
        stream_callback=stream_callback,
        progress_callback=progress_callback,
    )


def _run_helper_workflow(
    config_paths: list[str],
    rig_spec: str,
    options: dict[str, bool],
) -> None:
    """Run helper workflow in a worker process."""
    from darsia.presets.workflows.user_interface_helper import run_helper

    paths = normalize_paths(config_paths)
    rig_cls = resolve_rig_class(rig_spec)
    args = argparse.Namespace(
        config=paths,
        roi=options["roi"],
        roi_viewer=options.get("roi_viewer", False),
        results=options.get("results", False),
        show=options["show"],
        info=False,
    )
    run_helper(rig_cls, args)


def _run_comparison_workflow(
    config_path: str, rig_spec: str, options: dict[str, bool]
) -> None:
    """Run comparison workflow in a worker process."""
    from darsia.presets.workflows.user_interface_comparison import run_comparison

    path = Path(config_path)
    rig_cls = resolve_rig_class(rig_spec)
    args = argparse.Namespace(
        config=path,
        events=options["events"],
        wasserstein_compute=options["wasserstein_compute"],
        wasserstein_assemble=options["wasserstein_assemble"],
        info=False,
        show=False,
    )
    run_comparison(rig_cls, args)


def _run_utils_workflow(config_paths: list[str], options: UtilsWorkflowOptions) -> None:
    """Run utility workflow in a worker process."""
    from darsia.presets.workflows.utils.calibration_bundle import (
        export_calibration_bundle,
        import_calibration_bundle,
    )
    from darsia.presets.workflows.utils.utils_download import download_data
    from darsia.presets.workflows.utils.utils_media import build_media

    paths = normalize_paths(config_paths)
    if options["download"]:
        download_data(paths)
    if options["export_calibration"]:
        bundle = Path(options["export_bundle"]) if options["export_bundle"] else None
        export_calibration_bundle(paths, bundle=bundle)
    if options["import_calibration"]:
        bundle_raw = options["import_bundle"]
        if not bundle_raw:
            raise ValueError("Import calibration requires a bundle path.")
        import_calibration_bundle(
            paths,
            bundle=Path(bundle_raw),
            conflict_action=options["import_conflict_action"],
        )
    if options["media"]:
        build_media(paths)


class QueueLogHandler(logging.Handler):
    """Log handler writing to a queue for GUI consumption."""

    def __init__(self, queue: SupportsLogQueue):
        super().__init__()
        self._queue = queue

    def emit(self, record: logging.LogRecord) -> None:
        self._queue.put(self.format(record))


@dataclass
class RunContext:
    config_paths: list[Path]
    rig_cls: type[Rig]


class WorkflowGUI:
    """Tkinter-based GUI for preset workflow execution."""

    def __init__(self, root: Any):
        self.tk, self.filedialog, self.messagebox, self.ttk = _require_tkinter()
        self.root = root
        self.root.title("DarSIA Workflows GUI")
        self.root.geometry("1200x800")

        self.current_config_file: Path | None = None
        self._mp_context = mp.get_context("spawn")
        self.log_queue: SupportsLogQueue = self._mp_context.Queue()
        # maxsize=1 keeps only the newest preview frame and bounds memory usage.
        self.stream_queue: SupportsQueue = self._mp_context.Queue(maxsize=1)
        self.progress_queue: SupportsQueue = self._mp_context.Queue(maxsize=1)
        self._worker_process: mp.Process | None = None
        self._abort_requested = False
        self._active_workflow = ""
        self._active_actions: list[str] = []
        self._active_config_paths: list[Path] = []
        self._active_rig_spec = ""
        self._worker_started_at: float | None = None
        self._last_error_details = ""
        self._latest_stream_payload: dict[str, bytes | None] | None = None
        self._stream_photo: Any | None = None
        self._batch_active = False
        self._batch_overall_total = 0
        self._batch_overall_processed = 0
        self._batch_step_total = 0
        self._batch_step_processed = 0
        self._batch_last_step_index = 0
        self._batch_current_step = ""
        self._batch_current_image_path = ""
        self._batch_last_image_runtime: float | None = None
        self._batch_step_elapsed: float | None = None
        self._batch_overall_started_at: float | None = None
        self._batch_elapsed_override: float | None = None
        self._batch_recent_image_runtimes: list[float] = []
        self._sv_ttk: Any = None
        self._native_theme_name: str | None = None
        self._session_cache_file: Path | None = None
        try:
            self._session_cache_file = default_session_cache_file()
        except (OSError, RuntimeError) as e:
            logger.warning("Failed to initialize GUI session cache path: %s", e)

        self._setup_icon()
        self._setup_logging()
        self._build_layout()
        self._setup_themes()
        self._poll_logs()
        self._poll_stream()
        self._poll_batch_progress()
        self._update_dashboard()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_icon(self) -> None:
        logo_path = (
            Path(__file__).resolve().parent
            / "interface"
            / "DarSIA_Horisontal_Positiv_part.png"
        )
        icon = self.tk.PhotoImage(file=logo_path)
        self.root.iconphoto(True, icon)

    def _setup_logging(self) -> None:
        _configure_queue_logging(self.log_queue)

    def _build_layout(self) -> None:
        top = self.ttk.Frame(self.root)
        top.pack(fill=self.tk.X, padx=8, pady=(8, 0))
        self.ttk.Label(top, text="Theme:").pack(side=self.tk.LEFT)
        self.theme_var = self.tk.StringVar(value="System")
        self.theme_select = self.ttk.Combobox(
            top,
            values=["System", "Light", "Dark"],
            textvariable=self.theme_var,
            state="readonly",
            width=10,
        )
        self.theme_select.pack(side=self.tk.LEFT, padx=4)
        self.theme_select.bind("<<ComboboxSelected>>", self._on_theme_selected)

        main = self.ttk.Panedwindow(self.root, orient=self.tk.HORIZONTAL)
        main.pack(fill=self.tk.BOTH, expand=True)

        left = self.ttk.Frame(main)
        right = self.ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=2)

        self._build_config_manager(left)
        self._build_workflow_controls(left)
        self._build_mode_views(right)
        self._build_log_view(right)

    def _build_config_manager(self, parent) -> None:
        frame = self.ttk.LabelFrame(
            parent, text="Config paths (merged in listed order)"
        )
        frame.pack(fill=self.tk.X, padx=8, pady=8)

        list_frame = self.ttk.Frame(frame)
        list_frame.pack(fill=self.tk.X, padx=5, pady=5)

        self.config_list = self.tk.Listbox(list_frame, height=6)
        self.config_list.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)

        buttons = self.ttk.Frame(list_frame)
        buttons.pack(side=self.tk.RIGHT, fill=self.tk.Y, padx=5)
        self.ttk.Button(buttons, text="Add", command=self._add_config_path).pack(
            fill=self.tk.X
        )
        self.ttk.Button(buttons, text="Remove", command=self._remove_config_path).pack(
            fill=self.tk.X, pady=4
        )
        self.ttk.Button(
            buttons, text="Open in editor", command=self._open_selected_config_in_editor
        ).pack(fill=self.tk.X)
        self.ttk.Button(buttons, text="Up", command=lambda: self._move_config(-1)).pack(
            fill=self.tk.X, pady=4
        )
        self.ttk.Button(
            buttons, text="Down", command=lambda: self._move_config(1)
        ).pack(fill=self.tk.X)
        self.ttk.Button(
            buttons, text="Load previous session", command=self._load_previous_session
        ).pack(fill=self.tk.X, pady=(8, 0))
        self.config_list.bind("<Double-Button-1>", self._open_selected_config_in_editor)

        rig_frame = self.ttk.Frame(frame)
        rig_frame.pack(fill=self.tk.X, padx=5, pady=5)
        self.ttk.Label(rig_frame, text="Custom Rig class (optional):").pack(
            anchor=self.tk.W
        )
        self.rig_spec = self.tk.StringVar(value="")
        self.ttk.Entry(rig_frame, textvariable=self.rig_spec).pack(fill=self.tk.X)
        self.ttk.Label(
            rig_frame,
            text="Format: package.module:ClassName (must inherit Rig)",
        ).pack(anchor=self.tk.W)

    def _build_workflow_controls(self, parent) -> None:
        notebook = self.ttk.Notebook(parent)
        notebook.pack(fill=self.tk.BOTH, expand=True, padx=8, pady=8)

        self.setup_frame = self.ttk.Frame(notebook)
        self.calibration_frame = self.ttk.Frame(notebook)
        self.analysis_frame = self.ttk.Frame(notebook)
        self.helper_frame = self.ttk.Frame(notebook)
        self.comparison_frame = self.ttk.Frame(notebook)
        self.utils_frame = self.ttk.Frame(notebook)
        notebook.add(self.setup_frame, text="Setup")
        notebook.add(self.calibration_frame, text="Calibration")
        notebook.add(self.analysis_frame, text="Analysis")
        notebook.add(self.helper_frame, text="Helper")
        notebook.add(self.comparison_frame, text="Comparison")
        notebook.add(self.utils_frame, text="Utils")

        self._build_setup_tab()
        self._build_calibration_tab()
        self._build_analysis_tab()
        self._build_helper_tab()
        self._build_comparison_tab()
        self._build_utils_tab()

    def _build_setup_tab(self) -> None:
        self.setup_all = self.tk.BooleanVar(value=False)
        self.setup_depth = self.tk.BooleanVar(value=False)
        self.setup_seg = self.tk.BooleanVar(value=False)
        self.setup_facies = self.tk.BooleanVar(value=False)
        self.setup_protocol = self.tk.BooleanVar(value=False)
        self.setup_rig = self.tk.BooleanVar(value=False)
        self.setup_delete = self.tk.BooleanVar(value=False)
        self.setup_show = self.tk.BooleanVar(value=False)
        for label, var in [
            ("All", self.setup_all),
            ("Depth", self.setup_depth),
            ("Segmentation", self.setup_seg),
            ("Facies", self.setup_facies),
            ("Protocol", self.setup_protocol),
            ("Rig", self.setup_rig),
            ("Delete rig", self.setup_delete),
            ("Show plots", self.setup_show),
        ]:
            self.ttk.Checkbutton(self.setup_frame, text=label, variable=var).pack(
                anchor=self.tk.W
            )
        self.ttk.Button(
            self.setup_frame, text="Run setup", command=self._run_setup_clicked
        ).pack(fill=self.tk.X, pady=6)

    def _build_calibration_tab(self) -> None:
        self.cal_color_paths = self.tk.BooleanVar(value=False)
        self.cal_mass = self.tk.BooleanVar(value=False)
        self.cal_default_mass = self.tk.BooleanVar(value=False)
        self.cal_reset = self.tk.BooleanVar(value=False)
        self.cal_delete = self.tk.BooleanVar(value=False)
        self.cal_show = self.tk.BooleanVar(value=False)
        for label, var in [
            ("Color paths", self.cal_color_paths),
            ("Mass", self.cal_mass),
            ("Default mass", self.cal_default_mass),
            ("Reset", self.cal_reset),
            ("Delete calibration", self.cal_delete),
            ("Show plots", self.cal_show),
        ]:
            self.ttk.Checkbutton(self.calibration_frame, text=label, variable=var).pack(
                anchor=self.tk.W
            )
        self.ttk.Button(
            self.calibration_frame,
            text="Run calibration",
            command=self._run_calibration_clicked,
        ).pack(fill=self.tk.X, pady=6)

    def _build_analysis_tab(self) -> None:
        self.an_all = self.tk.BooleanVar(value=False)
        self.an_crop = self.tk.BooleanVar(value=False)
        self.an_seg = self.tk.BooleanVar(value=False)
        self.an_fingers = self.tk.BooleanVar(value=False)
        self.an_mass = self.tk.BooleanVar(value=False)
        self.an_volume = self.tk.BooleanVar(value=False)
        self.an_thresholding = self.tk.BooleanVar(value=False)
        self.an_show = self.tk.BooleanVar(value=False)
        self.an_stream = self.tk.BooleanVar(value=False)
        for label, var in [
            ("All images", self.an_all),
            ("Cropping", self.an_crop),
            ("Segmentation", self.an_seg),
            ("Fingers", self.an_fingers),
            ("Mass", self.an_mass),
            ("Volume", self.an_volume),
            ("Thresholding", self.an_thresholding),
            ("Show plots", self.an_show),
            ("Enable streaming", self.an_stream),
        ]:
            self.ttk.Checkbutton(self.analysis_frame, text=label, variable=var).pack(
                anchor=self.tk.W
            )
        self.ttk.Button(
            self.analysis_frame, text="Run analysis", command=self._run_analysis_clicked
        ).pack(fill=self.tk.X, pady=6)

    def _build_comparison_tab(self) -> None:
        self.comp_events = self.tk.BooleanVar(value=False)
        self.comp_w_compute = self.tk.BooleanVar(value=False)
        self.comp_w_assemble = self.tk.BooleanVar(value=False)
        for label, var in [
            ("Events", self.comp_events),
            ("Wasserstein compute", self.comp_w_compute),
            ("Wasserstein assemble", self.comp_w_assemble),
        ]:
            self.ttk.Checkbutton(self.comparison_frame, text=label, variable=var).pack(
                anchor=self.tk.W
            )
        self.ttk.Button(
            self.comparison_frame,
            text="Run comparison",
            command=self._run_comparison_clicked,
        ).pack(fill=self.tk.X, pady=6)

    def _build_helper_tab(self) -> None:
        self.helper_roi = self.tk.BooleanVar(value=False)
        self.helper_roi_viewer = self.tk.BooleanVar(value=False)
        self.helper_results = self.tk.BooleanVar(value=False)
        self.helper_show = self.tk.BooleanVar(value=False)
        for label, var in [
            ("ROI", self.helper_roi),
            ("ROI Viewer", self.helper_roi_viewer),
            ("ResultReader", self.helper_results),
            ("Show plots", self.helper_show),
        ]:
            self.ttk.Checkbutton(self.helper_frame, text=label, variable=var).pack(
                anchor=self.tk.W
            )
        self.ttk.Button(
            self.helper_frame, text="Run helper", command=self._run_helper_clicked
        ).pack(fill=self.tk.X, pady=6)

    def _build_utils_tab(self) -> None:
        self.utils_download = self.tk.BooleanVar(value=False)
        self.utils_export_calibration = self.tk.BooleanVar(value=False)
        self.utils_import_calibration = self.tk.BooleanVar(value=False)
        self.utils_export_bundle = self.tk.StringVar(value="")
        self.utils_import_bundle = self.tk.StringVar(value="")
        self.utils_media = self.tk.BooleanVar(value=False)
        self.ttk.Checkbutton(
            self.utils_frame,
            text="Build protocol-time media (MP4/GIF)",
            variable=self.utils_media,
        ).pack(anchor=self.tk.W)
        self.ttk.Checkbutton(
            self.utils_frame,
            text="Download/cache data",
            variable=self.utils_download,
        ).pack(anchor=self.tk.W)
        self.ttk.Checkbutton(
            self.utils_frame,
            text="Export calibration",
            variable=self.utils_export_calibration,
        ).pack(anchor=self.tk.W, pady=(6, 0))
        self.ttk.Label(self.utils_frame, text="Export zip destination").pack(
            anchor=self.tk.W
        )
        export_row = self.ttk.Frame(self.utils_frame)
        export_row.pack(fill=self.tk.X)
        self.ttk.Entry(export_row, textvariable=self.utils_export_bundle).pack(
            side=self.tk.LEFT, fill=self.tk.X, expand=True
        )
        self.ttk.Button(
            export_row, text="Browse", command=self._browse_utils_export_bundle
        ).pack(side=self.tk.LEFT, padx=4)

        self.ttk.Checkbutton(
            self.utils_frame,
            text="Import calibration",
            variable=self.utils_import_calibration,
        ).pack(anchor=self.tk.W, pady=(6, 0))
        self.ttk.Label(self.utils_frame, text="Import zip source").pack(
            anchor=self.tk.W
        )
        import_row = self.ttk.Frame(self.utils_frame)
        import_row.pack(fill=self.tk.X)
        self.ttk.Entry(import_row, textvariable=self.utils_import_bundle).pack(
            side=self.tk.LEFT, fill=self.tk.X, expand=True
        )
        self.ttk.Button(
            import_row, text="Browse", command=self._browse_utils_import_bundle
        ).pack(side=self.tk.LEFT, padx=4)
        self.ttk.Button(
            self.utils_frame,
            text="Load utils defaults from config",
            command=lambda: self._load_utils_defaults_from_config(force=True),
        ).pack(fill=self.tk.X, pady=(6, 0))
        self.ttk.Button(
            self.utils_frame, text="Run utils", command=self._run_utils_clicked
        ).pack(fill=self.tk.X, pady=6)

    def _build_mode_views(self, parent) -> None:
        self.mode_notebook = self.ttk.Notebook(parent)
        self.mode_notebook.pack(fill=self.tk.BOTH, expand=True, padx=8, pady=8)
        self.editor_mode_frame = self.ttk.Frame(self.mode_notebook)
        self.dashboard_mode_frame = self.ttk.Frame(self.mode_notebook)
        self.batch_mode_frame = self.ttk.Frame(self.mode_notebook)
        self.streaming_mode_frame = self.ttk.Frame(self.mode_notebook)
        self.mode_notebook.add(self.editor_mode_frame, text="Config editor")
        self.mode_notebook.add(self.dashboard_mode_frame, text="Dashboard")
        self.mode_notebook.add(self.batch_mode_frame, text="Batch monitor")
        self.mode_notebook.add(self.streaming_mode_frame, text="Streaming")
        self._build_editor(self.editor_mode_frame)
        self._build_dashboard(self.dashboard_mode_frame)
        self._build_batch_monitor(self.batch_mode_frame)
        self._build_streaming_mode(self.streaming_mode_frame)

    def _build_editor(self, parent) -> None:
        frame = self.ttk.LabelFrame(parent, text="Config editor")
        frame.pack(fill=self.tk.BOTH, expand=True, padx=8, pady=8)

        toolbar = self.ttk.Frame(frame)
        toolbar.pack(fill=self.tk.X, padx=5, pady=5)
        self.ttk.Button(
            toolbar, text="New from template", command=self._new_from_template
        ).pack(side=self.tk.LEFT)
        self.ttk.Button(toolbar, text="Open", command=self._open_config).pack(
            side=self.tk.LEFT, padx=4
        )
        self.ttk.Button(toolbar, text="Save", command=self._save_config).pack(
            side=self.tk.LEFT
        )
        self.ttk.Button(toolbar, text="Save as", command=self._save_config_as).pack(
            side=self.tk.LEFT, padx=4
        )

        self.editor_path = self.tk.StringVar(value="")
        self.ttk.Label(frame, textvariable=self.editor_path).pack(
            anchor=self.tk.W, padx=5
        )

        self.editor = self.tk.Text(frame, wrap=self.tk.NONE)
        self.editor.pack(fill=self.tk.BOTH, expand=True, padx=5, pady=5)

    def _build_dashboard(self, parent) -> None:
        frame = self.ttk.LabelFrame(parent, text="System dashboard")
        frame.pack(fill=self.tk.BOTH, expand=True, padx=8, pady=8)
        self.dashboard_cpu = self.tk.StringVar(value="CPU: n/a")
        self.dashboard_memory = self.tk.StringVar(value="Memory: n/a")
        self.dashboard_worker = self.tk.StringVar(value="Workflow process: idle")
        for variable in [
            self.dashboard_cpu,
            self.dashboard_memory,
            self.dashboard_worker,
        ]:
            self.ttk.Label(frame, textvariable=variable).pack(
                anchor=self.tk.W, padx=5, pady=2
            )

    def _build_batch_monitor(self, parent) -> None:
        frame = self.ttk.LabelFrame(parent, text="Batch monitor")
        frame.pack(fill=self.tk.BOTH, expand=True, padx=8, pady=8)
        self.batch_monitor_text = self.tk.StringVar(value=_BATCH_MONITOR_IDLE_MESSAGE)
        self.batch_monitor_progress = self.tk.DoubleVar(value=0.0)
        self.ttk.Label(frame, textvariable=self.batch_monitor_text).pack(
            anchor=self.tk.W, padx=5, pady=5
        )
        self.ttk.Progressbar(
            frame,
            orient=self.tk.HORIZONTAL,
            mode="determinate",
            maximum=100.0,
            variable=self.batch_monitor_progress,
        ).pack(fill=self.tk.X, padx=5, pady=(0, 5))

    def _build_streaming_mode(self, parent) -> None:
        frame = self.ttk.LabelFrame(parent, text="Streaming monitor")
        frame.pack(fill=self.tk.BOTH, expand=True, padx=8, pady=8)

        self.stream_status = self.tk.StringVar(value="No stream yet.")
        self.ttk.Label(frame, textvariable=self.stream_status).pack(
            anchor=self.tk.W, padx=5, pady=5
        )

        selector = self.ttk.Frame(frame)
        selector.pack(fill=self.tk.X, padx=5, pady=5)
        self.ttk.Label(selector, text="Stream key:").pack(side=self.tk.LEFT)
        self.stream_key_var = self.tk.StringVar(value="")
        self.stream_key_selector = self.ttk.Combobox(
            selector,
            textvariable=self.stream_key_var,
            values=[],
            state="readonly",
            width=30,
        )
        self.stream_key_selector.pack(side=self.tk.LEFT, padx=4)
        self.stream_key_selector.bind(
            "<<ComboboxSelected>>", self._on_stream_key_changed
        )

        self.stream_image_label = self.ttk.Label(
            frame,
            text="No streamed image.",
            anchor=self.tk.CENTER,
        )
        self.stream_image_label.pack(fill=self.tk.BOTH, expand=True, padx=5, pady=5)

    def _build_log_view(self, parent) -> None:
        frame = self.ttk.LabelFrame(parent, text="Execution log")
        frame.pack(fill=self.tk.BOTH, expand=True, padx=8, pady=8)

        controls = self.ttk.Frame(frame)
        controls.pack(fill=self.tk.X, padx=5, pady=(5, 0))
        self.abort_button = self.ttk.Button(
            controls,
            text="Abort running workflow",
            command=self._abort_worker_clicked,
            state=self.tk.DISABLED,
        )
        self.abort_button.pack(side=self.tk.RIGHT)

        self.log = self.tk.Text(frame, height=15, state=self.tk.DISABLED)
        self.log.pack(fill=self.tk.BOTH, expand=True, padx=5, pady=5)

    def _append_log(self, msg: str) -> None:
        self.log.config(state=self.tk.NORMAL)
        self.log.insert(self.tk.END, msg + "\n")
        self.log.see(self.tk.END)
        self.log.config(state=self.tk.DISABLED)

    def _consume_log_queue_messages(self) -> None:
        """Consume queued log messages and keep latest encoded error details."""
        while True:
            msg = self.log_queue.get_nowait()
            details = decode_workflow_error_details(msg)
            if details is not None:
                self._last_error_details = details
                continue
            self._append_log(msg)

    def _poll_logs(self) -> None:
        try:
            self._consume_log_queue_messages()
        except Empty:
            pass
        self.root.after(100, self._poll_logs)

    def _center_dialog_on_screen(self, dialog: Any) -> None:
        """Center a toplevel dialog on the current screen."""
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        x_pos = max(0, (screen_width - width) // 2)
        y_pos = max(0, (screen_height - height) // 2)
        dialog.geometry(f"+{x_pos}+{y_pos}")

    def _show_done_dialog_with_open_folder(
        self, message: str, suggested_folder: Path | None
    ) -> None:
        if suggested_folder is None:
            self.messagebox.showinfo("Done", message)
            return

        dialog = self.tk.Toplevel(self.root)
        dialog.title("Done")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        frame = self.ttk.Frame(dialog, padding=12)
        frame.pack(fill=self.tk.BOTH, expand=True)
        self.ttk.Label(frame, text=message, wraplength=480, justify=self.tk.LEFT).pack(
            anchor=self.tk.W, pady=(0, 10)
        )

        buttons = self.ttk.Frame(frame)
        buttons.pack(anchor=self.tk.E)

        open_requested = {"value": False}

        def _close_ok() -> None:
            dialog.destroy()

        def _open_folder() -> None:
            open_requested["value"] = True
            dialog.destroy()

        self.ttk.Button(buttons, text="OK", command=_close_ok).pack(
            side=self.tk.RIGHT, padx=(8, 0)
        )
        self.ttk.Button(buttons, text="Open in folder", command=_open_folder).pack(
            side=self.tk.RIGHT
        )
        dialog.protocol("WM_DELETE_WINDOW", _close_ok)
        self._center_dialog_on_screen(dialog)
        self.root.wait_window(dialog)

        if open_requested["value"]:
            try:
                open_in_file_explorer(suggested_folder)
            except Exception as e:
                self.messagebox.showerror(
                    "Open failed",
                    f"Failed to open {suggested_folder}:\n{e}",
                )

    def _show_error_dialog_with_details(
        self, title: str, message: str, details: str
    ) -> None:
        """Show modal error dialog with on-demand, scrollable details."""
        dialog = self.tk.Toplevel(self.root)
        dialog.title(title)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(True, True)

        frame = self.ttk.Frame(dialog, padding=12)
        frame.pack(fill=self.tk.BOTH, expand=True)
        self.ttk.Label(frame, text=message, wraplength=540, justify=self.tk.LEFT).pack(
            anchor=self.tk.W
        )

        details_container = self.ttk.Frame(frame)
        details_container.pack(fill=self.tk.BOTH, expand=True, pady=(10, 0))
        details_container.pack_forget()

        details_scrollbar = self.ttk.Scrollbar(
            details_container, orient=self.tk.VERTICAL
        )
        details_scrollbar.pack(side=self.tk.RIGHT, fill=self.tk.Y)
        details_text = self.tk.Text(
            details_container,
            height=12,
            width=80,
            wrap=self.tk.WORD,
            yscrollcommand=details_scrollbar.set,
        )
        details_text.insert(self.tk.END, format_error_details_text(details))
        details_text.config(state=self.tk.DISABLED)
        details_text.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)
        details_scrollbar.config(command=details_text.yview)

        buttons = self.ttk.Frame(frame)
        buttons.pack(anchor=self.tk.E, pady=(10, 0))

        def _close_ok() -> None:
            dialog.destroy()

        def _show_details() -> None:
            details_button.config(state=self.tk.DISABLED)
            details_container.pack(fill=self.tk.BOTH, expand=True, pady=(10, 0))
            self._center_dialog_on_screen(dialog)

        self.ttk.Button(buttons, text="OK", command=_close_ok).pack(
            side=self.tk.RIGHT, padx=(8, 0)
        )
        details_button = self.ttk.Button(
            buttons, text="Show details", command=_show_details
        )
        details_button.pack(side=self.tk.RIGHT)
        dialog.protocol("WM_DELETE_WINDOW", _close_ok)
        self._center_dialog_on_screen(dialog)
        self.root.wait_window(dialog)

    def _poll_stream(self) -> None:
        try:
            while True:
                payload = self.stream_queue.get_nowait()
                self._consume_stream_payload(payload)
        except Empty:
            pass
        self.root.after(100, self._poll_stream)

    def _reset_batch_monitor_state(
        self, message: str = _BATCH_MONITOR_IDLE_MESSAGE
    ) -> None:
        """Reset batch monitor counters and UI state."""
        self._batch_active = False
        self._batch_overall_total = 0
        self._batch_overall_processed = 0
        self._batch_step_total = 0
        self._batch_step_processed = 0
        self._batch_last_step_index = 0
        self._batch_current_step = ""
        self._batch_current_image_path = ""
        self._batch_last_image_runtime = None
        self._batch_step_elapsed = None
        self._batch_overall_started_at = None
        self._batch_elapsed_override = None
        self._batch_recent_image_runtimes = []
        self.batch_monitor_text.set(message)
        self.batch_monitor_progress.set(0.0)

    def _poll_batch_progress(self) -> None:
        try:
            while True:
                payload = self.progress_queue.get_nowait()
                self._consume_batch_progress_payload(payload)
        except Empty:
            pass
        self.root.after(100, self._poll_batch_progress)

    def _consume_batch_progress_payload(self, payload: Any) -> None:
        event = normalize_progress_event(payload)
        if event is None:
            return

        event_name = event.get("event")
        step = event.get("step", "")
        image_total = event.get("image_total", 0)

        if event_name == "step_start":
            if self._batch_overall_started_at is None:
                self._batch_overall_started_at = time.monotonic()
            self._batch_active = True
            self._batch_current_step = step
            self._batch_step_total = image_total
            self._batch_step_processed = 0
            self._batch_last_step_index = 0
            self._batch_step_elapsed = 0.0
            self._batch_overall_total += image_total
            self._update_batch_monitor_display()
            return

        if event_name == "image_progress":
            image_index = event.get("image_index", 0)
            delta = max(0, image_index - self._batch_last_step_index)
            self._batch_overall_processed += delta
            self._batch_last_step_index = max(self._batch_last_step_index, image_index)
            self._batch_step_processed = max(0, image_index)
            if image_total > 0:
                self._batch_step_total = image_total
            self._batch_current_step = step
            self._batch_current_image_path = event.get("image_path", "")
            self._batch_last_image_runtime = event.get("image_duration_s")
            step_elapsed = event.get("step_elapsed_s")
            if step_elapsed is not None:
                self._batch_step_elapsed = step_elapsed
            if self._batch_last_image_runtime is not None:
                self._batch_recent_image_runtimes.append(self._batch_last_image_runtime)
            self._update_batch_monitor_display()
            return

        if event_name == "step_complete":
            if image_total > 0:
                missing = max(0, image_total - self._batch_last_step_index)
                self._batch_overall_processed += missing
                self._batch_last_step_index = image_total
                self._batch_step_total = image_total
                self._batch_step_processed = image_total
            self._batch_current_step = step
            step_elapsed = event.get("step_elapsed_s")
            if step_elapsed is not None:
                self._batch_step_elapsed = step_elapsed
            self._update_batch_monitor_display()

    def _overall_elapsed_seconds(self) -> float | None:
        """Return current overall elapsed seconds."""
        if self._batch_elapsed_override is not None:
            return self._batch_elapsed_override
        if self._batch_overall_started_at is None:
            return None
        return max(0.0, time.monotonic() - self._batch_overall_started_at)

    def _update_batch_monitor_display(self, prefix: str | None = None) -> None:
        """Refresh batch monitor text and progressbar."""
        overall_elapsed = self._overall_elapsed_seconds()
        avg_runtime = rolling_average_runtime(self._batch_recent_image_runtimes)
        eta_seconds = estimate_remaining_time_seconds(
            avg_runtime,
            self._batch_overall_processed,
            self._batch_overall_total,
        )
        text = format_batch_monitor_text(
            step=self._batch_current_step,
            image_path=self._batch_current_image_path,
            processed=self._batch_overall_processed,
            total=self._batch_overall_total,
            last_image_seconds=self._batch_last_image_runtime,
            step_elapsed_seconds=self._batch_step_elapsed,
            overall_elapsed_seconds=overall_elapsed,
            eta_seconds=eta_seconds,
        )
        if prefix:
            text = f"{prefix}\n{text}"
        self.batch_monitor_text.set(text)
        self.batch_monitor_progress.set(
            progress_percent(self._batch_overall_processed, self._batch_overall_total)
        )

    def _finalize_batch_monitor_state(self, status: str) -> None:
        """Finalize batch monitor after analysis workflow termination."""
        self._batch_active = False
        elapsed = self._overall_elapsed_seconds()
        self._batch_elapsed_override = elapsed
        if self._batch_overall_total <= 0:
            self.batch_monitor_text.set(f"{status}\n{_BATCH_MONITOR_IDLE_MESSAGE}")
            self.batch_monitor_progress.set(0.0)
            return
        self._update_batch_monitor_display(prefix=status)

    def _consume_stream_payload(self, payload: Any) -> None:
        if payload is None:
            self._latest_stream_payload = None
            self._show_stream_message("Nothing is streamed.")
            return
        if not isinstance(payload, dict):
            self._show_stream_message(
                f"Stream error: expected dict payload, received "
                f"{type(payload).__name__}."
            )
            return

        if "__error__" in payload:
            self._show_stream_message("Stream error.")
            return

        self._latest_stream_payload = payload
        image_keys = [key for key, value in payload.items() if value is not None]
        if len(image_keys) == 0:
            self._show_stream_message("Nothing is streamed.")
            return

        current_key = self.stream_key_var.get()
        selected_key = current_key if current_key in image_keys else image_keys[0]
        self.stream_key_selector.config(values=image_keys)
        self.stream_key_var.set(selected_key)
        self._render_selected_stream_image()

    def _on_stream_key_changed(self, event: Any | None = None) -> None:
        del event
        self._render_selected_stream_image()

    def _render_selected_stream_image(self) -> None:
        payload = self._latest_stream_payload
        if payload is None:
            self._show_stream_message("Nothing is streamed.")
            return

        selected_key = self.stream_key_var.get()
        selected_image = payload.get(selected_key)
        if selected_image is None:
            self._show_stream_message("Nothing is streamed.")
            return

        try:
            encoded = base64.b64encode(selected_image).decode("ascii")
            self._stream_photo = self.tk.PhotoImage(data=encoded)
        except Exception:
            self._stream_photo = None
            self._show_stream_message("Stream error.")
            return

        self.stream_image_label.config(image=self._stream_photo, text="")
        self.stream_status.set(f"Showing stream key: {selected_key}")

    def _show_stream_message(self, message: str) -> None:
        self._stream_photo = None
        self.stream_image_label.config(image="", text=message)
        self.stream_status.set(message)

    def _selected_paths(self) -> list[Path]:
        values = [self.config_list.get(i) for i in range(self.config_list.size())]
        return normalize_paths(values)

    def _set_selected_paths(self, paths: list[Path]) -> None:
        self.config_list.delete(0, self.tk.END)
        for path in paths:
            self.config_list.insert(self.tk.END, str(path))

    def _persist_session_cache(self) -> None:
        if self._session_cache_file is None:
            return
        try:
            write_session_cache(
                self._session_cache_file, self._selected_paths(), self.rig_spec.get()
            )
        except OSError as e:
            logger.warning("Failed to persist GUI session cache: %s", e)

    def _context(self) -> RunContext:
        paths = self._selected_paths()
        if not paths:
            raise ValueError("Please add at least one config file.")
        rig_cls = resolve_rig_class(self.rig_spec.get())
        return RunContext(config_paths=paths, rig_cls=rig_cls)

    def _set_worker_state(self, running: bool) -> None:
        state = self.tk.NORMAL if running else self.tk.DISABLED
        self.abort_button.config(state=state)

    def _abort_worker_clicked(self) -> None:
        if abort_process(self._worker_process):
            self._abort_requested = True
            workflow = self._active_workflow or "workflow"
            self.log_queue.put(f"Abort requested for {workflow}.")
            return
        self.log_queue.put("No active workflow to abort.")

    def _run_async(
        self,
        workflow: str,
        actions: list[str],
        config_paths: list[Path],
        rig_spec: str,
        fn: Callable[..., None],
        *args: Any,
    ) -> None:
        """Run a workflow function in a child process and monitor it."""
        if self._worker_process is not None and self._worker_process.is_alive():
            self.messagebox.showwarning("Busy", "Another workflow is still running.")
            return
        if workflow != "analysis":
            self._reset_batch_monitor_state(_BATCH_MONITOR_IDLE_MESSAGE)
        self._active_workflow = workflow
        self._active_actions = actions
        self._active_config_paths = config_paths
        self._active_rig_spec = rig_spec
        self._last_error_details = ""
        self.log_queue.put(
            format_workflow_start_message(workflow, actions, config_paths, rig_spec)
        )
        self._worker_process = self._mp_context.Process(
            target=_worker_entry,
            args=(self.log_queue, fn, args),
            daemon=True,
        )
        self._worker_process.start()
        self._worker_started_at = time.monotonic()
        self._abort_requested = False
        self._set_worker_state(True)
        self.root.after(200, self._poll_worker_completion)

    def _poll_worker_completion(self) -> None:
        """Poll process completion and translate exit status to GUI log messages."""
        process = self._worker_process
        if process is None:
            self._set_worker_state(False)
            return
        if process.is_alive():
            self.root.after(200, self._poll_worker_completion)
            return

        exit_code = process.exitcode
        duration = (
            0.0
            if self._worker_started_at is None
            else max(0.0, time.monotonic() - self._worker_started_at)
        )
        workflow = self._active_workflow
        actions = self._active_actions
        config_paths = self._active_config_paths
        config_count = len(self._active_config_paths)
        try:
            self._consume_log_queue_messages()
        except Empty:
            pass
        dialog_spec = completion_dialog_spec(workflow, exit_code, self._abort_requested)
        if self._abort_requested:
            self.log_queue.put(f"{workflow.capitalize()} workflow aborted.")
        elif exit_code == 0:
            self.log_queue.put(
                format_workflow_done_message(
                    workflow,
                    actions,
                    config_count,
                    duration,
                )
            )
        else:
            self.log_queue.put(
                format_workflow_error_message(workflow, actions, exit_code)
            )
        if workflow == "analysis":
            if self._abort_requested:
                self._finalize_batch_monitor_state("Analysis aborted.")
            elif exit_code == 0:
                self._finalize_batch_monitor_state("Analysis completed.")
            else:
                self._finalize_batch_monitor_state(
                    f"Analysis failed (exit code {exit_code})."
                )
        self._worker_process = None
        self._worker_started_at = None
        self._active_workflow = ""
        self._active_actions = []
        self._active_config_paths = []
        self._active_rig_spec = ""
        self._set_worker_state(False)
        if dialog_spec is not None:
            kind, title, message = dialog_spec
            if kind == "info":
                suggested_folder = suggested_workflow_results_folder(
                    workflow, config_paths, actions
                )
                self._show_done_dialog_with_open_folder(message, suggested_folder)
            else:
                self._show_error_dialog_with_details(
                    title, message, self._last_error_details
                )

    def _run_setup_clicked(self) -> None:
        try:
            ctx = self._context()
        except Exception as e:
            self.messagebox.showerror("Invalid configuration", str(e))
            return
        options = {
            "all": self.setup_all.get(),
            "depth": self.setup_depth.get(),
            "segmentation": self.setup_seg.get(),
            "facies": self.setup_facies.get(),
            "protocol": self.setup_protocol.get(),
            "rig": self.setup_rig.get(),
            "delete_rig": self.setup_delete.get(),
            "show": self.setup_show.get(),
            "force": False,
        }
        protocol_enabled = options["protocol"]
        if protocol_enabled:
            from darsia.presets.workflows.setup.setup_protocols import (
                preview_protocol_setup_conflicts,
            )

            conflicts = preview_protocol_setup_conflicts(ctx.config_paths)
            if conflicts:
                max_preview = UTILS_CONFLICT_PREVIEW_LIMIT
                preview = "\n".join(str(path) for path in conflicts[:max_preview])
                remaining = max(0, len(conflicts) - max_preview)
                suffix = "" if remaining <= 0 else f"\n... and {remaining} more."
                choice = self.messagebox.askyesnocancel(
                    "Setup protocol conflicts",
                    "Some protocol files already exist and would be overwritten.\n\n"
                    f"{preview}{suffix}\n\n"
                    "Yes: overwrite existing files\nNo/Cancel: abort",
                )
                if choice is not True:
                    logger.info("Setup protocol generation aborted by user.")
                    return
                options["force"] = True
        actions = enabled_option_labels(options, exclude={"force"})
        self._run_async(
            "setup",
            actions,
            ctx.config_paths,
            self.rig_spec.get(),
            _run_setup_workflow,
            [str(path) for path in ctx.config_paths],
            self.rig_spec.get(),
            options,
        )

    def _run_calibration_clicked(self) -> None:
        try:
            ctx = self._context()
        except Exception as e:
            self.messagebox.showerror("Invalid configuration", str(e))
            return
        options = {
            "delete": self.cal_delete.get(),
            "color_paths": self.cal_color_paths.get(),
            "mass": self.cal_mass.get(),
            "default_mass": self.cal_default_mass.get(),
            "reset": self.cal_reset.get(),
            "show": self.cal_show.get(),
        }
        if options["delete"]:
            from darsia.presets.workflows.calibration.calibration_color_paths import (
                collect_existing_calibration_paths_to_delete,
            )

            existing = collect_existing_calibration_paths_to_delete(ctx.config_paths)
            if not existing:
                self.messagebox.showinfo(
                    "Delete calibration",
                    "No existing calibration data found to delete.",
                )
                return
            max_preview = PREVIEW_LIST_LIMIT
            preview = "\n".join(str(path) for path in existing[:max_preview])
            remaining = max(0, len(existing) - max_preview)
            suffix = "" if remaining <= 0 else f"\n... and {remaining} more."
            confirmed = self.messagebox.askyesno(
                "Confirm calibration deletion",
                "The following calibration files/folders will be deleted:\n\n"
                f"{preview}{suffix}\n\n"
                "This action cannot be undone.\n\nProceed?",
            )
            if not confirmed:
                logger.info("Calibration data deletion aborted by user.")
                return
        actions = enabled_option_labels(options)
        run_options = options.copy()
        if options["delete"]:
            run_options["skip_delete_confirmation"] = True
        self._run_async(
            "calibration",
            actions,
            ctx.config_paths,
            self.rig_spec.get(),
            _run_calibration_workflow,
            [str(path) for path in ctx.config_paths],
            self.rig_spec.get(),
            run_options,
        )

    def _run_analysis_clicked(self) -> None:
        try:
            ctx = self._context()
        except Exception as e:
            self.messagebox.showerror("Invalid configuration", str(e))
            return
        options = {
            "all": self.an_all.get(),
            "cropping": self.an_crop.get(),
            "segmentation": self.an_seg.get(),
            "fingers": self.an_fingers.get(),
            "mass": self.an_mass.get(),
            "volume": self.an_volume.get(),
            "thresholding": self.an_thresholding.get(),
            "show": self.an_show.get(),
            "streaming": self.an_stream.get(),
        }
        if options["streaming"]:
            clear_queue(self.stream_queue)
            self._show_stream_message("Streaming enabled. Waiting for data...")
        else:
            self._show_stream_message("Streaming disabled.")
        clear_queue(self.progress_queue)
        self._reset_batch_monitor_state(_BATCH_MONITOR_WAITING_MESSAGE)
        actions = enabled_option_labels(options)
        self._run_async(
            "analysis",
            actions,
            ctx.config_paths,
            self.rig_spec.get(),
            _run_analysis_workflow,
            [str(path) for path in ctx.config_paths],
            self.rig_spec.get(),
            options,
            self.stream_queue,
            self.progress_queue,
        )

    def _run_comparison_clicked(self) -> None:
        try:
            ctx = self._context()
        except Exception as e:
            self.messagebox.showerror("Invalid configuration", str(e))
            return
        if len(ctx.config_paths) != 1:
            self.messagebox.showerror(
                "Invalid configuration",
                "Comparison currently supports exactly one config path.",
            )
            return

        options = {
            "events": self.comp_events.get(),
            "wasserstein_compute": self.comp_w_compute.get(),
            "wasserstein_assemble": self.comp_w_assemble.get(),
        }
        actions = enabled_option_labels(options)
        self._run_async(
            "comparison",
            actions,
            ctx.config_paths,
            self.rig_spec.get(),
            _run_comparison_workflow,
            str(ctx.config_paths[0]),
            self.rig_spec.get(),
            options,
        )

    def _run_helper_clicked(self) -> None:
        try:
            ctx = self._context()
        except Exception as e:
            self.messagebox.showerror("Invalid configuration", str(e))
            return
        options = {
            "roi": self.helper_roi.get(),
            "roi_viewer": self.helper_roi_viewer.get(),
            "results": self.helper_results.get(),
            "show": self.helper_show.get(),
        }
        actions = enabled_option_labels(options)
        self._run_async(
            "helper",
            actions,
            ctx.config_paths,
            self.rig_spec.get(),
            _run_helper_workflow,
            [str(path) for path in ctx.config_paths],
            self.rig_spec.get(),
            options,
        )

    def _run_utils_clicked(self) -> None:
        try:
            ctx = self._context()
        except Exception as e:
            self.messagebox.showerror("Invalid configuration", str(e))
            return

        self._load_utils_defaults_from_config(force=False)

        action_flags = {
            "download": self.utils_download.get(),
            "export_calibration": self.utils_export_calibration.get(),
            "import_calibration": self.utils_import_calibration.get(),
            "media": self.utils_media.get(),
        }
        actions = enabled_option_labels(action_flags)
        if not any(action_flags.values()):
            logger.info("No utility option selected.")
            return

        import_bundle = self.utils_import_bundle.get().strip()
        if action_flags["import_calibration"] and not import_bundle:
            self.messagebox.showerror(
                "Invalid configuration",
                "Import calibration requires an input bundle zip path.",
            )
            return

        import_conflict_action = "error"
        if action_flags["import_calibration"]:
            from darsia.presets.workflows.utils.calibration_bundle import (
                preview_calibration_bundle_import_conflicts,
            )

            conflicts = preview_calibration_bundle_import_conflicts(
                ctx.config_paths,
                Path(import_bundle),
            )
            if conflicts:
                max_preview = PREVIEW_LIST_LIMIT
                preview = "\n".join(str(path) for path in conflicts[:max_preview])
                remaining = max(0, len(conflicts) - max_preview)
                suffix = "" if remaining <= 0 else f"\n... and {remaining} more."
                choice = self.messagebox.askyesnocancel(
                    "Calibration import conflicts",
                    "Some calibration files already exist and would be overwritten.\n\n"
                    f"{preview}{suffix}\n\n"
                    "Yes: overwrite all\nNo: skip all existing\nCancel: abort",
                )
                policy = map_conflict_dialog_choice_to_policy(choice)
                if policy is None:
                    logger.info("Calibration import aborted by user.")
                    return
                import_conflict_action = policy

        options = {
            "download": action_flags["download"],
            "export_calibration": action_flags["export_calibration"],
            "import_calibration": action_flags["import_calibration"],
            "export_bundle": self.utils_export_bundle.get().strip(),
            "import_bundle": import_bundle,
            "import_conflict_action": import_conflict_action,
            "media": action_flags["media"],
        }
        self._run_async(
            "utils",
            actions,
            ctx.config_paths,
            self.rig_spec.get(),
            _run_utils_workflow,
            [str(path) for path in ctx.config_paths],
            options,
        )

    def _load_utils_defaults_from_config(self, *, force: bool) -> None:
        selected = self._selected_paths()
        if not selected:
            return
        export_default, import_default = resolve_utils_bundle_defaults(
            [str(path) for path in selected]
        )
        if export_default and (force or not self.utils_export_bundle.get().strip()):
            self.utils_export_bundle.set(export_default)
        if import_default and (force or not self.utils_import_bundle.get().strip()):
            self.utils_import_bundle.set(import_default)

    def _browse_utils_export_bundle(self) -> None:
        path = self.filedialog.asksaveasfilename(
            title="Select export calibration bundle path",
            defaultextension=".zip",
            filetypes=[("Zip files", "*.zip"), ("All files", "*.*")],
        )
        if path:
            self.utils_export_bundle.set(path)

    def _browse_utils_import_bundle(self) -> None:
        path = self.filedialog.askopenfilename(
            title="Select calibration bundle zip",
            filetypes=[("Zip files", "*.zip"), ("All files", "*.*")],
        )
        if path:
            self.utils_import_bundle.set(path)

    def _add_config_path(self) -> None:
        path = self.filedialog.askopenfilename(
            title="Select config file",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if path:
            self.config_list.insert(self.tk.END, str(Path(path).resolve()))
            self._persist_session_cache()
            self._load_utils_defaults_from_config(force=False)

    def _remove_config_path(self) -> None:
        selection = list(self.config_list.curselection())
        for idx in reversed(selection):
            self.config_list.delete(idx)
        if selection:
            self._persist_session_cache()
            self._load_utils_defaults_from_config(force=False)

    def _move_config(self, direction: int) -> None:
        selected = self.config_list.curselection()
        if len(selected) != 1:
            return
        i = selected[0]
        j = i + direction
        if j < 0 or j >= self.config_list.size():
            return
        value = self.config_list.get(i)
        self.config_list.delete(i)
        self.config_list.insert(j, value)
        self.config_list.selection_set(j)
        self._persist_session_cache()
        self._load_utils_defaults_from_config(force=False)

    def _load_previous_session(self) -> None:
        if self._session_cache_file is None:
            self.messagebox.showwarning(
                "Session cache warning",
                "Session cache is unavailable in this environment.",
            )
            return
        try:
            cached_paths, cached_rig_spec = read_session_cache(self._session_cache_file)
        except ValueError as e:
            self.messagebox.showwarning("Session cache warning", str(e))
            return
        if not cached_paths and not cached_rig_spec.strip():
            self.messagebox.showwarning(
                "No previous session",
                f"No cached GUI session found at:\n{self._session_cache_file}",
            )
            return

        available_paths: list[Path] = []
        unavailable_paths: list[Path] = []
        for path in cached_paths:
            if path.exists():
                available_paths.append(path)
            else:
                unavailable_paths.append(path)

        current_paths = self._selected_paths()
        merged = deduplicate_paths(current_paths + available_paths)
        self._set_selected_paths(merged)

        if cached_rig_spec.strip():
            self.rig_spec.set(cached_rig_spec)

        if unavailable_paths:
            max_display = 10
            shown = unavailable_paths[:max_display]
            unavailable_list = "\n".join(str(path) for path in shown)
            remaining = len(unavailable_paths) - len(shown)
            suffix = "" if remaining <= 0 else f"\n... and {remaining} more."
            self.messagebox.showwarning(
                "Missing config files",
                "Some files from previous session are not available:\n"
                f"{unavailable_list}{suffix}",
            )

        self._persist_session_cache()

    def _new_from_template(self) -> None:
        template = _find_template_file()
        if not template.exists():
            self.messagebox.showerror(
                "Missing template", f"Template not found: {template}"
            )
            return
        self.current_config_file = None
        self.editor_path.set("New file (from template)")
        self.editor.delete("1.0", self.tk.END)
        self.editor.insert(self.tk.END, template.read_text())

    def _open_config(self) -> None:
        path = self.filedialog.askopenfilename(
            title="Open config file",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if not path:
            return
        self._load_config_into_editor(Path(path).resolve(), add_to_context=True)

    def _open_selected_config_in_editor(self, event: Any | None = None) -> None:
        del event
        selection = self.config_list.curselection()
        if len(selection) != 1:
            self.messagebox.showwarning(
                "No config selected",
                "Select exactly one config in the context list to open in editor.",
            )
            return
        path = Path(self.config_list.get(selection[0])).resolve()
        self._load_config_into_editor(path, add_to_context=False)

    def _load_config_into_editor(self, path: Path, *, add_to_context: bool) -> None:
        try:
            text = path.read_text()
        except OSError as e:
            self.messagebox.showerror("Open failed", f"Failed to open {path}:\n{e}")
            return
        self.current_config_file = path
        self.editor_path.set(str(path))
        self.editor.delete("1.0", self.tk.END)
        self.editor.insert(self.tk.END, text)
        if add_to_context and str(path) not in self.config_list.get(0, self.tk.END):
            self.config_list.insert(self.tk.END, str(path))
            self._persist_session_cache()

    def _save_config(self) -> None:
        if self.current_config_file is None:
            self._save_config_as()
            return
        try:
            self.current_config_file.write_text(self.editor.get("1.0", "end-1c"))
        except OSError as e:
            self.messagebox.showerror(
                "Save failed", f"Failed to save {self.current_config_file}:\n{e}"
            )
            return
        self.editor_path.set(str(self.current_config_file))
        if str(self.current_config_file) not in self.config_list.get(0, self.tk.END):
            self.config_list.insert(self.tk.END, str(self.current_config_file))
            self._persist_session_cache()

    def _save_config_as(self) -> None:
        path = self.filedialog.asksaveasfilename(
            title="Save config file",
            defaultextension=".toml",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if not path:
            return
        self.current_config_file = Path(path).resolve()
        self._save_config()

    def _setup_themes(self) -> None:
        style = self.ttk.Style(self.root)
        self._native_theme_name = style.theme_use()
        try:
            import sv_ttk

            self._sv_ttk = sv_ttk
        except ModuleNotFoundError:
            self._sv_ttk = None
        self._apply_theme("System", announce=False)

    def _on_theme_selected(self, event: Any | None = None) -> None:
        del event
        self._apply_theme(self.theme_var.get(), announce=True)

    def _apply_theme(self, theme: str, *, announce: bool) -> None:
        style = self.ttk.Style(self.root)
        selected = theme.lower()
        if self._sv_ttk is not None:
            if selected == "light":
                self._sv_ttk.set_theme("light")
            elif selected == "dark":
                self._sv_ttk.set_theme("dark")
            else:
                if self._native_theme_name is not None:
                    style.theme_use(self._native_theme_name)
            if announce:
                self.log_queue.put(f"Theme changed to {theme}.")
            return

        if selected == "system":
            if self._native_theme_name is not None:
                style.theme_use(self._native_theme_name)
        elif selected == "light":
            preferred = self._pick_preferred_theme(
                style, ["vista", "aqua", "default", "clam", "alt"]
            )
            if preferred is not None:
                style.theme_use(preferred)
        else:
            preferred = self._pick_preferred_theme(style, ["clam", "alt", "default"])
            if preferred is not None:
                style.theme_use(preferred)
            if announce:
                self.log_queue.put(
                    "Dark theme fallback in use. Install optional 'sv_ttk' "
                    "for richer dark mode."
                )
        if announce:
            self.log_queue.put(f"Theme changed to {theme}.")

    def _pick_preferred_theme(self, style: Any, candidates: list[str]) -> str | None:
        names = set(style.theme_names())
        for candidate in candidates:
            if candidate in names:
                return candidate
        return None

    def _update_dashboard(self) -> None:
        cpu_text = "CPU: n/a"
        memory_text = "Memory: n/a"
        try:
            import psutil

            cpu_text = f"CPU: {psutil.cpu_percent(interval=None):.1f}%"
            memory_text = (
                f"Memory: {psutil.virtual_memory().percent:.1f}% system, "
                f"{psutil.Process(os.getpid()).memory_info().rss / (1024**2):.1f} MB GUI"
            )
        except ModuleNotFoundError:
            if hasattr(os, "getloadavg"):
                load1, _, _ = os.getloadavg()
                cpu_text = f"CPU load avg (1m): {load1:.2f}"
        except Exception:
            pass

        process = self._worker_process
        if process is not None and process.is_alive():
            worker_text = (
                "Workflow process: running "
                f"(pid={process.pid}, {self._active_workflow})"
            )
        else:
            worker_text = "Workflow process: idle"

        self.dashboard_cpu.set(cpu_text)
        self.dashboard_memory.set(memory_text)
        self.dashboard_worker.set(worker_text)
        self.root.after(1000, self._update_dashboard)

    def _on_close(self) -> None:
        try:
            self._persist_session_cache()
        except Exception:
            logger.exception("Failed while persisting GUI session cache during close.")
        finally:
            abort_process(self._worker_process)
            self.root.destroy()


def launch_workflows_gui() -> None:
    tk, _, _, _ = _require_tkinter()
    root = tk.Tk()
    WorkflowGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_workflows_gui()
