"""Utility functions for TOML parsing"""

from darsia.fluidflower import preset_wasserstein_inputs as _preset_io

import os
import sys
import tomllib
from copy import deepcopy
from datetime import timedelta
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


def _get_section(data: dict, section: str) -> dict:
    """Utility to get a section from a toml-loaded dictionary."""
    try:
        return data[section]
    except KeyError:
        raise KeyError(f"Section {section} not found.")


def _deep_merge(base: dict, update: dict) -> dict:
    """Recursively merge update dict into base dict.

    When both dicts have the same key with dict values, merges them recursively
    instead of overwriting.

    Parameters
    ----------
    base : dict
        Base dictionary to merge into.
    update : dict
        Dictionary with updates to merge.

    Returns
    -------
        Merged dictionary (base is modified in-place).

    Examples
    --------
        >>> base = {"a": {"x": 1, "y": 2}, "b": 3}
        >>> update = {"a": {"y": 20, "z": 30}, "c": 4}
        >>> _deep_merge(base, update)
        {"a": {"x": 1, "y": 20, "z": 30}, "b": 3, "c": 4}
    """
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            _deep_merge(base[key], value)
        else:
            # Overwrite or add new key
            base[key] = value
    return base


def _merge_named_registry_sources(documents, sections, *, label):
    """Combine consistent inherited entries without accepting ambiguous names.

    A merged run config may repeat a common entry, or retain only its name.
    Such repeats across files denote one definition. Repeats inside a single
    file, and conflicting values across files, remain configuration errors.
    """
    def same(left, right):
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(same(left[key], right[key]) for key in left)
        if isinstance(left, (list, tuple)):
            return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
        return left == right

    merged = {section: {} for section in sections}
    for document in documents:
        for section in sections:
            rows = document.get(section, [])
            if not isinstance(rows, list):
                raise ValueError(f"The [[{section}]] section must be an array-of-tables")
            seen = set()
            for row in rows:
                if not isinstance(row, dict) or row.get("name") is None:
                    raise ValueError(f"Each [[{section}]] entry must be a table with a name")
                name = str(row["name"]).strip()
                if name in seen:
                    raise ValueError(f"{label}: duplicate key '{name}' in [[{section}]] in one configuration source")
                seen.add(name)
                previous = merged[section].get(name)
                if previous is None:
                    merged[section][name] = deepcopy(row)
                    continue
                for key, value in row.items():
                    if key == "name":
                        continue
                    if key in previous and not same(previous[key], value):
                        raise ValueError(f"{label}: duplicate key '{name}' in [[{section}]] has conflicting '{key}' values")
                    previous[key] = deepcopy(value)
    return {section: list(rows.values()) for section, rows in merged.items()}


def _in_alias_job_worker() -> bool:
    """Return True while an alias-job worker executes a task.

    Such a worker maps every input through its machine's ``stores.toml`` (the
    active per-run I/O binding of ``darsia.fluidflower.pipeline.run_io``), so
    the per-user path conventions of this module must not apply there.
    """
    run_io = sys.modules.get("darsia.fluidflower.pipeline.run_io")
    return run_io is not None and run_io.current_binding() is not None


def _map_strings(value: Any, remap) -> Any:
    """Return ``value`` with ``remap`` applied to every string and table key.

    Lists and tables are rebuilt only when an entry changed; otherwise the
    original object is returned, so an identity ``remap`` changes nothing.
    """
    if isinstance(value, str):
        result = remap(value)
        return value if result == value else result
    if isinstance(value, list):
        items = [_map_strings(item, remap) for item in value]
        changed = any(new is not old for new, old in zip(items, value))
        return items if changed else value
    if isinstance(value, dict):
        table: dict = {}
        changed = False
        for key, item in value.items():
            new_key = _map_strings(key, remap)
            new_item = _map_strings(item, remap)
            if new_key in table:
                raise ValueError(
                    f"Storage path aliases map two keys to the same path: {new_key}"
                )
            table[new_key] = new_item
            changed = changed or new_key is not key or new_item is not item
        return table if changed else value
    return value


def _with_storage_aliases(section: dict) -> dict:
    """Read alias-prefixed absolute paths of a section at this machine's location.

    Every string value and table key that starts with a source prefix of the
    per-user registry's ``[storage.path_aliases]`` is passed through
    ``darsia.fluidflower.storage_paths.resolve_storage_path``. Without an alias
    the section is returned unchanged. Only the values read change, never the
    file bytes, so content (SHA) bindings of configuration files stay valid.
    Alias-job workers already map their inputs and are left untouched.
    """
    if _in_alias_job_worker() and _preset_io.current_scope() is None:
        return section
    from darsia.fluidflower.storage_paths import storage_path_remapper

    return _map_strings(section, storage_path_remapper())


def _series_root(config: Path | list[Path]) -> Path | None:
    """Return the series project root holding configuration file(s), or None.

    The root is the nearest folder above a file that contains ``series.toml``.
    Files without such a folder are ignored; files of two different series
    give None, as do files outside any series.
    """
    paths = [config] if isinstance(config, (str, os.PathLike)) else list(config)
    roots = set()
    for path in paths:
        folder = Path(path).absolute().parent
        for candidate in (folder, *folder.parents):
            try:
                found = _preset_io.exists(candidate / "series.toml", 'is_file')
            except OSError:  # An unreadable parent holds no usable series.
                found = False
            if found:
                roots.add(candidate)
                break
    return roots.pop() if len(roots) == 1 else None


def _is_relative_path(value: Any) -> bool:
    """Return True for a non-empty path without a drive or root in any spelling."""
    text = str(value)
    return (
        bool(text.strip())
        and not PurePosixPath(text).anchor
        and not PureWindowsPath(text).anchor
    )


def _config_path(value: Any, config: Path | list[Path] | None) -> Path:
    """Return one path value of a native configuration as a Path.

    A relative value of a configuration inside a series project folder (one
    holding ``series.toml``) is read against that series root, so a cloned
    project finds its own files wherever it is checked out. Outside a series,
    without ``config``, and in alias-job workers a relative value keeps
    DarSIA's working-directory meaning. Other values are returned as Paths.
    Values are read as this computer reads stored paths
    (``darsia.fluidflower.storage_paths.machine_local_path``), so a Windows
    spelling is understood elsewhere; on Windows without a machine path map
    that is ``Path(value)``.
    """
    scope = _preset_io.current_scope()
    if scope is not None:
        return scope.config_path(value, config)
    from darsia.fluidflower.storage_paths import machine_local_path

    path = machine_local_path(value)
    run_io = sys.modules.get("darsia.fluidflower.pipeline.run_io")
    binding = None if run_io is None else run_io.current_binding()
    original_setup = getattr(binding, "series_setup_original_reader", False) is True
    if (config is None or not _is_relative_path(value)
            or _in_alias_job_worker() and not original_setup):
        return path
    root = _series_root(config)
    return path if root is None else root / path


def _get_section_from_toml(path: Path | list[Path], section: str) -> dict:
    if isinstance(path, Path):
        data = tomllib.loads(_preset_io.read_text(path))
    elif isinstance(path, list):
        data = {}
        for p in path:
            part = tomllib.loads(_preset_io.read_text(p))
            _deep_merge(data, part)
    else:
        raise TypeError(f"Path must be a Path or list of Paths. It is {type(path)}.")
    sec = _get_section(data, section)
    return _with_storage_aliases(sec)


def _get_key(section: dict, key: str, default=None, required=True, type_=None) -> Any:
    """Utility to get a key from a section with type conversion and default value.

    Handles blank strings ("") gracefully for int/float types: a blank value with
    required=False returns default (matching the behavior when the key is absent),
    while required=True raises a clear ValueError naming the key/type.
    """
    if required and key not in section:
        raise KeyError(f"Missing key '{key}' in section {section}.")

    if key in section:
        value = section[key]
        # Handle blank strings for int/float: treat as "not provided" for optional
        # fields, or raise a clear error for required fields (not the cryptic
        # "invalid literal for int() with base 10: ''" traceback).
        if type_ in (int, float) and isinstance(value, str) and value.strip() == "":
            if required:
                raise ValueError(
                    f"Key '{key}' in section {section} is blank; "
                    f"expected a {type_.__name__}."
                )
            return default
        return type_(value) if type_ else value
    else:
        return default


def _convert_to_hours(time_value: float | str) -> float:
    """Convert time value to hours.

    Parameters
    ----------
    time_value : float | str
        Time as float (hours) or string in "DD:HH:MM:SS", "HH:MM:SS",
        "HH:MM", "HH", or numeric format.

    Returns
    -------
        Time in hours as float.
    """
    if isinstance(time_value, (int, float)):
        return float(time_value)

    if isinstance(time_value, str):
        # If no colon, treat as numeric hours (e.g., "5", "0.0")
        if ":" not in time_value:
            return float(time_value)

        # Handle "DD:HH:MM:SS", "HH:MM:SS", "HH:MM", or "HH" format
        parts = time_value.split(":")
        if len(parts) == 4:
            # DD:HH:MM:SS format
            days = int(parts[0])
            hours = int(parts[1])
            minutes = int(parts[2])
            seconds = int(parts[3])
        elif len(parts) == 3:
            # HH:MM:SS format
            days = 0
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
        elif len(parts) == 2:
            # HH:MM format
            days = seconds = 0
            hours = int(parts[0])
            minutes = int(parts[1])
        elif len(parts) == 1:
            # HH format
            days = minutes = seconds = 0
            hours = int(parts[0])
            total_hours = hours
        else:
            raise ValueError(
                f"Invalid time format: {time_value}. Use DD:HH:MM:SS, HH:MM:SS, "
                "HH:MM, HH, or numeric format"
            )
        total_hours = (
            timedelta(
                days=days, hours=hours, minutes=minutes, seconds=seconds
            ).total_seconds()
            / 3600
        )
        return total_hours

    raise ValueError(f"Invalid time value: {time_value}. Must be float or time format")


def _convert_none(v):
    return None if ((isinstance(v, str) and v.lower() == "none") or v is None) else v


def _normalize_mode(value: str, canonical: set[str], *, key: str) -> str:
    """Case-insensitively match `value` against `canonical`, returning the
    canonically-cased option.

    Some option values are plain lowercase identifiers ("exif", "constant"), but
    others carry a specific display casing meant to read naturally in the GUI
    dropdown (e.g. "Load from CSV") — case-folding those with a blanket
    ``.lower()`` on load would silently break round-tripping with the dropdown's
    own (exact-case) option list. This tolerates a hand-typed TOML value in any
    case while always storing/returning the one canonical spelling.

    Raises
    ------
    ValueError
        If `value` doesn't case-insensitively match any entry in `canonical`.
    """
    text = str(value)
    for option in canonical:
        if option.lower() == text.lower():
            return option
    raise ValueError(f"{key} must be one of {sorted(canonical)}, got {value!r}.")


def _validate_choice(value: str, *, allowed: set[str], context: str, key: str) -> str:
    if value not in allowed:
        raise ValueError(
            f"Invalid {context}.{key} '{value}'. Allowed values are: {sorted(allowed)}."
        )
    return value


def _format_hours(hours: float) -> str:
    """Convert floating-point hours to HH:MM:SS string format.

    Parameters
    ----------
    hours : float
        Hours as a float (can be fractional).

    Returns
    -------
        Formatted string in HH:MM:SS format (no day component).
    """
    if hours is None:
        return None
    total_seconds = int(hours * 3600)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _normalize_time_string(value: float | str) -> str:
    """Normalize time value to canonical HH:MM:SS format.

    Accepts float hours or any string format supported by _convert_to_hours,
    converts to float hours, then formats to canonical HH:MM:SS.

    Parameters
    ----------
    value : float | str
        Time as float (hours) or string in any recognized format.

    Returns
    -------
        Canonical HH:MM:SS format string.

    Raises
    ------
    ValueError
        If the input cannot be parsed as a time value.
    """
    return _format_hours(_convert_to_hours(value))
