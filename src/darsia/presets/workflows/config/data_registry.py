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

    [color_paths]
    data = ["calibration1"]

    [analysis.mass]
    data = "phase_1"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from warnings import warn

from .time_data import ImagePathData, ImageTimeData, ImageTimeIntervalData, TimeData

logger = logging.getLogger(__name__)


@dataclass
class DataRegistry:
    """Registry of named :class:`TimeData` entries loaded from a ``[data]`` section.

    Attributes:
        _registry: Flat mapping from entry key to :class:`TimeData` object.
    """

    _registry: dict[str, TimeData] = field(default_factory=dict)

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
        time_keys: set[str] = set()
        path_keys: set[str] = set()

        # --- interval sub-registry ---
        interval_sec = sec.get("interval", {})
        if isinstance(interval_sec, dict):
            for key, entry in interval_sec.items():
                interval_keys.add(key)
                td = TimeData()
                td.image_interval_data = ImageTimeIntervalData().load(
                    {"interval": {key: entry}}
                )
                td._combine_data()
                self._registry[key] = td

        # --- time sub-registry ---
        time_sec = sec.get("time", {})
        if isinstance(time_sec, dict):
            for key, entry in time_sec.items():
                time_keys.add(key)
                td = TimeData()
                td.image_time_data = ImageTimeData().load({"time": {key: entry}})
                td._combine_data()
                self._registry[key] = td

        # --- path sub-registry ---
        path_sec = sec.get("path", {})
        if isinstance(path_sec, dict):
            for key, entry in path_sec.items():
                path_keys.add(key)
                td = TimeData()
                td.image_path_data = ImagePathData().load(
                    {"path": {key: entry}}, data_folder
                )
                td._combine_data()
                self._registry[key] = td

        # --- sanity check: duplicate keys across sub-registries ---
        duplicates = (
            (interval_keys & time_keys)
            | (interval_keys & path_keys)
            | (time_keys & path_keys)
        )
        if duplicates:
            raise ValueError(
                f"DataRegistry: duplicate key(s) found across 'interval', 'time', "
                f"and 'path' sub-registries: {sorted(duplicates)}. "
                f"Each key must be unique across all three sub-registries."
            )

        logger.debug(
            f"DataRegistry loaded {len(self._registry)} entries: "
            f"{sorted(self._registry)}"
        )
        return self

    def resolve(self, keys: str | list[str]) -> TimeData:
        """Resolve one or more registry keys into a merged :class:`TimeData` object.

        Args:
            keys: A single key string or a list of key strings to look up and merge.

        Returns:
            A :class:`TimeData` object whose ``image_paths``, ``image_times``,
            and sub-data attributes are the union of all matched entries, deduplicated
            and sorted.

        Raises:
            KeyError: If any requested key is not found in the registry.
        """
        if isinstance(keys, str):
            keys = [keys]

        merged = TimeData()

        for key in keys:
            if key not in self._registry:
                available = sorted(self._registry.keys())
                raise KeyError(
                    f"DataRegistry: key '{key}' not found. "
                    f"Available keys: {available}"
                )
            entry = self._registry[key]

            # Merge image_path_data
            merged.image_path_data.paths.extend(entry.image_path_data.paths)

            # Merge image_time_data
            merged.image_time_data.times.extend(entry.image_time_data.times)
            merged.image_time_data.times_with_tolerance.extend(
                entry.image_time_data.times_with_tolerance
            )

            # Merge image_interval_data
            for interval_key, interval in entry.image_interval_data.intervals.items():
                merged.image_interval_data.intervals[interval_key] = interval

        # Deduplicate and sort paths
        merged.image_path_data.paths = sorted(set(merged.image_path_data.paths))

        # Deduplicate and sort times
        merged.image_time_data.times = sorted(set(merged.image_time_data.times))
        merged.image_time_data.times_with_tolerance = sorted(
            set(merged.image_time_data.times_with_tolerance), key=lambda x: x[0]
        )

        # Rebuild combined fields
        merged._combine_data()

        return merged

    def keys(self) -> list[str]:
        """Return all registered entry names.

        Returns:
            Sorted list of key strings in the registry.
        """
        return sorted(self._registry.keys())


# ---------------------------------------------------------------------------
# Selector-resolution helpers (formerly in data_selection.py)
# ---------------------------------------------------------------------------

_DEFAULT_WARNING_STACKLEVEL = 3


def _deprecation_message(section: str, key: str) -> str:
    return (
        f"Inline selector definitions under [{section}.{key}] are deprecated. "
        "Define selectors centrally under [data.interval.*], [data.time.*], or "
        "[data.path.*], and reference selector key(s) instead."
    )


def resolve_time_data_selector(
    sec: dict,
    key: str,
    *,
    section: str,
    data: Path | None,
    data_registry: DataRegistry | None,
    required: bool = True,
    warning_stacklevel: int = _DEFAULT_WARNING_STACKLEVEL,
) -> TimeData | None:
    """Resolve a workflow selector to :class:`TimeData`.

    Supports registry-key references and legacy inline selector tables.
    Legacy inline selector tables are still parsed but emit ``DeprecationWarning``.
    """
    if key not in sec:
        if required:
            raise KeyError(key)
        return None

    selector = sec[key]
    if isinstance(selector, (str, list)):
        if data_registry is None:
            raise ValueError(
                f"{section}.{key} references selector key(s), but no DataRegistry "
                "is available. Define top-level [data] selectors."
            )
        return data_registry.resolve(selector)

    if isinstance(selector, dict):
        warn(
            _deprecation_message(section, key),
            DeprecationWarning,
            stacklevel=warning_stacklevel,
        )
        return TimeData().load(selector, data)

    raise ValueError(
        f"{section}.{key} must be a selector key, list of selector keys, or table."
    )


def resolve_path_selector(
    sec: dict,
    key: str,
    *,
    section: str,
    data: Path | None,
    data_registry: DataRegistry | None,
) -> list[Path]:
    """Resolve a selector and validate that it provides explicit image paths."""
    resolved = resolve_time_data_selector(
        sec,
        key,
        section=section,
        data=data,
        data_registry=data_registry,
        required=True,
    )
    if resolved is None:
        raise ValueError(f"{section}.{key} is required.")
    if len(resolved.image_paths) == 0:
        raise ValueError(
            f"{section}.{key} must resolve to explicit image paths "
            "(use [data.path.*] selectors)."
        )
    return resolved.image_paths
