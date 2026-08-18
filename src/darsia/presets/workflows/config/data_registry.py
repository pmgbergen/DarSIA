"""Global data registry for shared time/path data definitions in FluidFlower workflows.

The registry reads from a top-level ``[data]`` section that contains three optional
sub-sections, one per ``TimeData`` loading mode:

* ``[data.interval.*]``  – time intervals (was ``image_time_interval``)
* ``[data.time.*]``      – explicit image times (was ``image_times``)
* ``[data.path.*]``      – direct file paths (was ``image_paths``)

Each named entry (e.g. ``calibration1``, ``phase_1``) is loaded into a
:class:`TimeData` object and stored in a flat lookup dictionary. All key names
must be unique across the three sub-registries; a :class:`ValueError` is raised
at load time if any duplicate is detected.

Example TOML structure::

    [data.interval.calibration1]
    start = "01:00:00"
    end   = "23:00:00"
    num   = 5
    tol   = "00:10:00"

    [data.interval.phase_1]
    start = "00:00:00"
    end   = "01:00:00"
    num   = 13
    tol   = "00:01:00"

    [data.time.manual_snap]
    times = ["00:30:00", "01:00:00"]
    tol   = "00:05:00"

    [data.path.baseline_images]
    paths = ["baseline/DSC00155.JPG", "DSC00160.JPG"]

Tasks reference entries by key::

    [color.path.calibration]
    data = ["calibration1"]

    [analysis.mass]
    data = "phase_1"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .time_data import (
    ImageTimeData,
    ImageTimeIntervalData,
    PathData,
    TimeData,
    TimeInterval,
    TimeWindow,
)

logger = logging.getLogger(__name__)


@dataclass
class DataRegistry:
    """Registry of named time/path entries loaded from a ``[data]`` section.

    Stores three separate sub-registries (interval, time, path), each containing
    homogeneous entry types, enabling GUI editing of named selections.

    Attributes:
        interval_registry: Mapping from name to TimeInterval.
        window_registry: Mapping from name to TimeWindow.
        time_registry: Mapping from name to ImageTimeData.
        path_registry: Mapping from name to PathData.
    """

    interval_registry: dict[str, TimeInterval] = field(
        default_factory=dict,
        metadata={
            "name": "Interval Entries",
            "help": "Named time intervals (e.g., 'injection', 'analysis').",
            "group": "Time intervals",
            "widget": "time_interval_map",
        },
    )
    """Named interval entries from [data.interval.*]."""
    window_registry: dict[str, TimeWindow] = field(
        default_factory=dict,
        metadata={
            "name": "Window Entries",
            "help": "Named time windows (e.g., 'injection', 'analysis').",
            "group": "Time windows",
            "widget": "time_window_map",
        },
    )
    """Named window entries from [data.interval.*]."""
    time_registry: dict[str, ImageTimeData] = field(
        default_factory=dict,
        metadata={
            "name": "Time Entries",
            "help": "Named explicit time lists (e.g., 'calibration').",
            "group": "Time points",
            "widget": "time_data_map",
        },
    )
    """Named time entries from [data.time.*]."""
    path_registry: dict[str, PathData] = field(
        default_factory=dict,
        metadata={
            "name": "Path Entries",
            "help": "Named file path lists.",
            "group": "Paths",
            "widget": "path_data_map",
        },
    )
    """Named path entries from [data.path.*]."""

    def load(
        self, sec: dict, data_folder: Path | list[Path] | None = None
    ) -> "DataRegistry":
        """Populate the registry from a ``[data]`` config section dict.

        Args:
            sec: The dictionary corresponding to the ``[data]`` TOML section.
            data_folder: Base folder used to resolve relative paths for
                ``[data.path.*]`` entries.

        Returns:
            self

        Raises:
            ValueError: If any key appears in more than one of the three
                sub-registries (``interval``, ``time``, ``path``).
        """
        interval_keys: set[str] = set()
        window_keys: set[str] = set()
        time_keys: set[str] = set()
        path_keys: set[str] = set()

        # --- interval sub-registry ---
        interval_sec = sec.get("interval_registry", {})
        if isinstance(interval_sec, dict):
            for key, entry in interval_sec.items():
                interval_keys.add(key)
                interval_data = ImageTimeIntervalData().load({"interval": {key: entry}})
                # Store the raw interval object
                # (load() populates either intervals or windows dict, not both)
                if interval_data.intervals:
                    # Get the single interval value from the dict
                    self.interval_registry[key] = next(
                        iter(interval_data.intervals.values())
                    )

        # --- window sub-registry ---
        window_sec = sec.get("window_registry", {})
        if isinstance(window_sec, dict):
            for key, entry in window_sec.items():
                window_keys.add(key)
                window_data = ImageTimeIntervalData().load({"interval": {key: entry}})
                # Store the raw window object
                if window_data.windows:
                    # Get the single window value from the dict
                    self.window_registry[key] = next(iter(window_data.windows.values()))

        # --- time sub-registry ---
        time_sec = sec.get("time_registry", {})
        if isinstance(time_sec, dict):
            for key, entry in time_sec.items():
                time_keys.add(key)
                time_data = ImageTimeData().load({"time": {key: entry}})
                self.time_registry[key] = time_data

        # --- path sub-registry ---
        path_sec = sec.get("path_registry", {})
        if isinstance(path_sec, dict):
            for key, entry in path_sec.items():
                path_keys.add(key)
                path_data = PathData().load({"path": {key: entry}}, data_folder)
                self.path_registry[key] = path_data

        # --- sanity check: duplicate keys across sub-registries ---
        duplicates = (
            (interval_keys & window_keys)
            | (interval_keys & time_keys)
            | (interval_keys & path_keys)
            | (window_keys & time_keys)
            | (window_keys & path_keys)
            | (time_keys & path_keys)
        )
        if duplicates:
            raise ValueError(
                f"DataRegistry: duplicate key(s) found across 'interval', 'window', 'time', "
                f"and 'path' sub-registries: {sorted(duplicates)}. "
                f"Each key must be unique across all four sub-registries."
            )

        total_entries = (
            len(self.interval_registry)
            + len(self.time_registry)
            + len(self.path_registry)
        )
        logger.debug(
            f"DataRegistry loaded {total_entries} entries: "
            f"intervals={sorted(self.interval_registry.keys())}, "
            f"times={sorted(self.time_registry.keys())}, "
            f"paths={sorted(self.path_registry.keys())}"
        )
        return self

    def resolve(self, keys: str | list[str] | None) -> TimeData:
        """Resolve one or more registry keys into a merged :class:`TimeData` object.

        Args:
            keys: A single key string, a list of key strings, or None (returns empty TimeData).

        Returns:
            A :class:`TimeData` object whose ``image_paths``, ``image_times``,
            and sub-data attributes are the union of all matched entries, deduplicated
            and sorted.

        Raises:
            KeyError: If any requested key is not found in the registry.
        """
        if keys is None:
            return TimeData()

        if isinstance(keys, str):
            keys = [keys]

        merged = TimeData()

        for key in keys:
            # Determine which registry this key is in
            if key in self.interval_registry:
                entry_obj = self.interval_registry[key]
                # Wrap TimeInterval in ImageTimeIntervalData
                interval_data = ImageTimeIntervalData()
                if isinstance(entry_obj, TimeInterval):
                    interval_data.intervals[key] = entry_obj
                merged.image_interval_data.intervals.update(interval_data.intervals)
            elif key in self.window_registry:
                entry_obj = self.window_registry[key]
                # Wrap TimeWindow in ImageTimeIntervalData
                window_data = ImageTimeIntervalData()
                if isinstance(entry_obj, TimeWindow):
                    window_data.windows[key] = entry_obj
                merged.image_interval_data.windows.update(window_data.windows)
            elif key in self.time_registry:
                entry_obj = self.time_registry[key]
                # entry_obj is ImageTimeData
                merged.image_time_data.times.extend(entry_obj.times)
                merged.image_time_data.times_with_tolerance.extend(
                    entry_obj.times_with_tolerance
                )
            elif key in self.path_registry:
                entry_obj = self.path_registry[key]
                # entry_obj is PathData
                merged.image_path_data.paths.extend(entry_obj.paths)
            else:
                available = self.keys()
                raise KeyError(
                    f"DataRegistry: key '{key}' not found. Available keys: {available}"
                )

        # Deduplicate and sort paths
        merged.image_path_data.paths = sorted(set(merged.image_path_data.paths))

        # Deduplicate and sort times
        merged.image_time_data.times = sorted(set(merged.image_time_data.times))
        merged.image_time_data.times_with_tolerance = sorted(
            set(merged.image_time_data.times_with_tolerance), key=lambda x: x[0]
        )

        # Deduplicate windows and intervals (by key)
        merged.image_interval_data.intervals = {
            key: merged.image_interval_data.intervals[key]
            for key in sorted(merged.image_interval_data.intervals)
        }
        merged.image_interval_data.windows = {
            key: merged.image_interval_data.windows[key]
            for key in sorted(merged.image_interval_data.windows)
        }

        # Rebuild combined fields
        merged._combine_data()

        return merged

    def keys(self) -> list[str]:
        """Return all registered entry names across all four sub-registries.

        Returns:
            Sorted list of key strings in the registry.
        """
        all_keys = (
            set(self.interval_registry.keys())
            | set(self.window_registry.keys())
            | set(self.time_registry.keys())
            | set(self.path_registry.keys())
        )
        return sorted(all_keys)

    def to_toml_dict(self) -> dict:
        """Serialize the registry back into TOML-compatible dict structure.

        Returns:
            A dict with keys 'interval', 'window', 'time', 'path', each containing
            the serialized entries (as dicts matching the TOML structure).
            Missing sub-registries are omitted from the output.
        """
        result = {}

        # Serialize interval registry (TimeInterval only)
        if self.interval_registry:
            interval_dict = {}
            for name, entry in self.interval_registry.items():
                if isinstance(entry, TimeInterval):
                    interval_dict[name] = {
                        "start": _format_hours(entry.start),
                        "end": _format_hours(entry.end),
                        "num": entry.num,
                        "tol": _format_hours(entry.tol),
                    }
            if interval_dict:
                result["interval"] = interval_dict

        # Serialize window registry (TimeWindow only)
        if self.window_registry:
            window_dict = {}
            for name, entry in self.window_registry.items():
                if isinstance(entry, TimeWindow):
                    window_dict[name] = {
                        "start": _format_hours(entry.start),
                        "end": _format_hours(entry.end),
                    }
            if window_dict:
                result["window"] = window_dict

        # Serialize time registry
        if self.time_registry:
            time_dict = {}
            for name, entry in self.time_registry.items():
                time_dict[name] = {
                    "times": [_format_hours(t) for t in entry.times],
                }
                if entry.times_with_tolerance:
                    # Store tolerance as the diff from first occurrence
                    # (time_data.py stores as (time, tolerance) pairs)
                    # For now, use the tolerance from the first occurrence if available
                    if entry.times_with_tolerance:
                        tol_val = entry.times_with_tolerance[0][1]
                        time_dict[name]["tol"] = _format_hours(tol_val)
            if time_dict:
                result["time"] = time_dict

        # Serialize path registry
        if self.path_registry:
            path_dict = {}
            for name, entry in self.path_registry.items():
                path_dict[name] = {
                    "paths": [str(p) for p in entry.paths],
                }
            if path_dict:
                result["path"] = path_dict

        return result


def _format_hours(hours: float) -> str:
    """Convert floating-point hours to HH:MM:SS string format.

    Args:
        hours: Hours as a float (can be fractional).

    Returns:
        Formatted time string "HH:MM:SS".
    """
    total_seconds = int(hours * 3600)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
