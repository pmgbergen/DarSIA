"""Global data registry for shared time/path data definitions in FluidFlower workflows.

The registry reads from four optional top-level array-of-tables sections in TOML:

* ``[[data_interval]]``  – time intervals (start/end/num/tol)
* ``[[data_window]]``    – time windows (start/end)
* ``[[data_time]]``      – explicit image times (times, tol)
* ``[[data_path]]``      – direct file paths (paths)

Each array entry MUST have a ``name`` field (the registry key). All key names
must be unique across the four sub-registries; a :class:`ValueError` is raised
at load time if any duplicate is detected.

Example TOML structure::

    [[data_interval]]
    name = "calibration"
    start = "01:00:00"
    end   = "23:00:00"
    num   = 5
    tol   = "00:10:00"

    [[data_interval]]
    name = "analysis"
    start = "00:00:00"
    end   = "01:00:00"
    num   = 13
    tol   = "00:01:00"

    [[data_time]]
    name = "manual_snap"
    times = ["00:30:00", "01:00:00"]
    tol   = "00:05:00"

    [[data_path]]
    name = "baseline_images"
    paths = ["baseline/DSC00155.JPG", "DSC00160.JPG"]

Tasks reference entries by key::

    [color.path.calibration]
    data = ["calibration"]

    [analysis.mass]
    data = "analysis"
"""

from __future__ import annotations

import logging
import tomllib
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
from .utils import _format_hours

logger = logging.getLogger(__name__)


@dataclass
class DataRegistry:
    """Registry of named time/path entries loaded from a ``[data]`` section.

    Stores four separate sub-registries (interval, window, time, path), each containing
    homogeneous entry types. Fields carry metadata for both runtime use and GUI editing
    of named selections.

    Attributes:
        interval_registry: Mapping from name to TimeInterval.
        window_registry: Mapping from name to TimeWindow.
        time_registry: Mapping from name to ImageTimeData.
        path_registry: Mapping from name to PathData.
    """

    interval_registry: dict[str, TimeInterval] = field(
        default_factory=dict,
        metadata={
            "name": "Intervals",
            "help": "Named time intervals (start/end/num/tol).",
            "widget": "dataclass_group_map",
            "array_key": "data_interval",
        },
    )
    """Named interval entries from [[data_interval]]."""
    window_registry: dict[str, TimeWindow] = field(
        default_factory=dict,
        metadata={
            "name": "Windows",
            "help": "Named time windows (start/end).",
            "widget": "dataclass_group_map",
            "array_key": "data_window",
        },
    )
    """Named window entries from [[data_window]]."""
    time_registry: dict[str, ImageTimeData] = field(
        default_factory=dict,
        metadata={
            "name": "Times",
            "help": "Named explicit time lists (times, tol).",
            "widget": "dataclass_group_map",
            "array_key": "data_time",
        },
    )
    """Named time entries from [[data_time]]."""
    path_registry: dict[str, PathData] = field(
        default_factory=dict,
        metadata={
            "name": "Paths",
            "help": "Named file path lists.",
            "widget": "dataclass_group_map",
            "array_key": "data_path",
        },
    )
    """Named path entries from [[data_path]]."""

    def load(
        self,
        paths: Path | list[Path] | dict,
        data_folder: Path | list[Path] | None = None,
    ) -> "DataRegistry":
        """Populate the registry from TOML files or dicts with array-of-tables format.

        Reads four separate top-level arrays: [[data_interval]], [[data_window]],
        [[data_time]], [[data_path]]. Each entry must have a 'name' field (the registry key).

        Args:
            paths: One or more TOML file paths, a single TOML file path, or a dict
                containing the TOML data directly (for testing/programmatic use).
            data_folder: Base folder used to resolve relative paths for
                [[data_path]] entries.

        Returns:
            self

        Raises:
            ValueError: If any key appears in more than one of the four
                sub-registries, or if array format is incorrect.
        """
        # Handle dict input (for tests and programmatic use)
        if isinstance(paths, dict):
            toml_data_list = [paths]
        else:
            # Handle Path/list[Path] input
            if isinstance(paths, Path):
                paths = [paths]
            toml_data_list = []
            for path in paths:
                with open(path, "rb") as f:
                    toml_data_list.append(tomllib.load(f))

        interval_keys: set[str] = set()
        window_keys: set[str] = set()
        time_keys: set[str] = set()
        path_keys: set[str] = set()

        for toml_data in toml_data_list:
            # --- data_interval array ---
            interval_entries = toml_data.get("data_interval", [])
            if not isinstance(interval_entries, list):
                raise ValueError(
                    "The [[data_interval]] section must be an array-of-tables format "
                    "(use [[data_interval]]), not nested tables."
                )
            for idx, entry in enumerate(interval_entries):
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"[[data_interval]] entry {idx} must be a table/dict."
                    )
                entry_name = entry.get("name")
                if entry_name is None:
                    raise ValueError(
                        f"[[data_interval]] entry {idx} is missing required 'name' field."
                    )
                entry_name = str(entry_name).strip()
                if entry_name in interval_keys:
                    raise ValueError(
                        f"DataRegistry: duplicate key '{entry_name}' in [[data_interval]]. "
                        "Names must be globally unique."
                    )
                interval_keys.add(entry_name)
                interval_data = ImageTimeIntervalData().load(
                    {"interval": {entry_name: entry}}
                )
                if interval_data.intervals:
                    self.interval_registry[entry_name] = next(
                        iter(interval_data.intervals.values())
                    )

            # --- data_window array ---
            window_entries = toml_data.get("data_window", [])
            if not isinstance(window_entries, list):
                raise ValueError(
                    "The [[data_window]] section must be an array-of-tables format "
                    "(use [[data_window]]), not nested tables."
                )
            for idx, entry in enumerate(window_entries):
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"[[data_window]] entry {idx} must be a table/dict."
                    )
                entry_name = entry.get("name")
                if entry_name is None:
                    raise ValueError(
                        f"[[data_window]] entry {idx} is missing required 'name' field."
                    )
                entry_name = str(entry_name).strip()
                if entry_name in window_keys:
                    raise ValueError(
                        f"DataRegistry: duplicate key '{entry_name}' in [[data_window]]. "
                        "Names must be globally unique."
                    )
                window_keys.add(entry_name)
                window_data = ImageTimeIntervalData().load(
                    {"interval": {entry_name: entry}}
                )
                if window_data.windows:
                    self.window_registry[entry_name] = next(
                        iter(window_data.windows.values())
                    )

            # --- data_time array ---
            time_entries = toml_data.get("data_time", [])
            if not isinstance(time_entries, list):
                raise ValueError(
                    "The [[data_time]] section must be an array-of-tables format "
                    "(use [[data_time]]), not nested tables."
                )
            for idx, entry in enumerate(time_entries):
                if not isinstance(entry, dict):
                    raise ValueError(f"[[data_time]] entry {idx} must be a table/dict.")
                entry_name = entry.get("name")
                if entry_name is None:
                    raise ValueError(
                        f"[[data_time]] entry {idx} is missing required 'name' field."
                    )
                entry_name = str(entry_name).strip()
                if entry_name in time_keys:
                    raise ValueError(
                        f"DataRegistry: duplicate key '{entry_name}' in [[data_time]]. "
                        "Names must be globally unique."
                    )
                time_keys.add(entry_name)
                time_data = ImageTimeData().load({"time": {entry_name: entry}})
                self.time_registry[entry_name] = time_data

            # --- data_path array ---
            path_entries = toml_data.get("data_path", [])
            if not isinstance(path_entries, list):
                raise ValueError(
                    "The [[data_path]] section must be an array-of-tables format "
                    "(use [[data_path]]), not nested tables."
                )
            for idx, entry in enumerate(path_entries):
                if not isinstance(entry, dict):
                    raise ValueError(f"[[data_path]] entry {idx} must be a table/dict.")
                entry_name = entry.get("name")
                if entry_name is None:
                    raise ValueError(
                        f"[[data_path]] entry {idx} is missing required 'name' field."
                    )
                entry_name = str(entry_name).strip()
                if entry_name in path_keys:
                    raise ValueError(
                        f"DataRegistry: duplicate key '{entry_name}' in [[data_path]]. "
                        "Names must be globally unique."
                    )
                path_keys.add(entry_name)
                path_data = PathData().load({"path": {entry_name: entry}}, data_folder)
                self.path_registry[entry_name] = path_data

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
            + len(self.window_registry)
            + len(self.time_registry)
            + len(self.path_registry)
        )
        logger.debug(
            f"DataRegistry loaded {total_entries} entries: "
            f"intervals={sorted(self.interval_registry.keys())}, "
            f"windows={sorted(self.window_registry.keys())}, "
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
            A dict with keys 'data_interval', 'data_window', 'data_time', 'data_path', each
            containing a list of dicts matching the array-of-tables TOML structure.
            Missing sub-registries are omitted from the output.
        """
        result = {}

        # Serialize interval registry as [[data_interval]]
        if self.interval_registry:
            interval_list = []
            for name, entry in sorted(self.interval_registry.items()):
                if isinstance(entry, TimeInterval):
                    interval_dict = {
                        "name": name,
                        "start": entry.start,
                        "end": entry.end,
                    }
                    if entry.num is not None:
                        interval_dict["num"] = entry.num
                    if entry.step:
                        interval_dict["step"] = entry.step
                    if entry.tol:
                        interval_dict["tol"] = entry.tol
                    interval_list.append(interval_dict)
            if interval_list:
                result["data_interval"] = interval_list

        # Serialize window registry as [[data_window]]
        if self.window_registry:
            window_list = []
            for name, entry in sorted(self.window_registry.items()):
                if isinstance(entry, TimeWindow):
                    window_dict = {
                        "name": name,
                        "start": entry.start,
                        "end": entry.end,
                    }
                    if entry.step:
                        window_dict["step"] = entry.step
                    window_list.append(window_dict)
            if window_list:
                result["data_window"] = window_list

        # Serialize time registry as [[data_time]]
        if self.time_registry:
            time_list = []
            for name, entry in sorted(self.time_registry.items()):
                time_entry = {
                    "name": name,
                    "times": [_format_hours(t) for t in entry.times],
                }
                if entry.times_with_tolerance:
                    tol_val = entry.times_with_tolerance[0][1]
                    time_entry["tol"] = _format_hours(tol_val)
                time_list.append(time_entry)
            if time_list:
                result["data_time"] = time_list

        # Serialize path registry as [[data_path]]
        if self.path_registry:
            path_list = []
            for name, entry in sorted(self.path_registry.items()):
                path_list.append(
                    {
                        "name": name,
                        "paths": [str(p) for p in entry.paths],
                    }
                )
            if path_list:
                result["data_path"] = path_list

        return result
