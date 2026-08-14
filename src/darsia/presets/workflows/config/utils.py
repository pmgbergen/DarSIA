"""Utility functions for TOML parsing"""

import tomllib
from datetime import timedelta
from pathlib import Path
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

    Args:
        base: Base dictionary to merge into.
        update: Dictionary with updates to merge.

    Returns:
        Merged dictionary (base is modified in-place).

    Example:
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


def _get_section_from_toml(path: Path | list[Path], section: str) -> dict:
    if isinstance(path, Path):
        data = tomllib.loads(path.read_text())
    elif isinstance(path, list):
        data = {}
        for p in path:
            part = tomllib.loads(p.read_text())
            _deep_merge(data, part)
    else:
        raise TypeError(f"Path must be a Path or list of Paths. It is {type(path)}.")
    sec = _get_section(data, section)
    return sec


def _get_key(section: dict, key: str, default=None, required=True, type_=None) -> Any:
    """Utility to get a key from a section with type conversion and default value."""
    if required and key not in section:
        raise KeyError(f"Missing key '{key}' in section {section}.")

    if key in section:
        value = section[key]
        return type_(value) if type_ else value
    else:
        return default


def _convert_to_hours(time_value: float | str) -> float:
    """Convert time value to hours.

    Args:
        time_value: Time as float (hours) or string in "DD:HH:MM:SS" format

    Returns:
        Time in hours as float

    """
    if isinstance(time_value, (int, float)):
        return float(time_value)

    if isinstance(time_value, str):
        # Handle "DD:HH:MM:SS", "HH:MM:SS", "HH:MM", or "HH" format
        assert ":" in time_value
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
                f"Invalid time format: {time_value}. Use DD:HH:MM:SS, HH:MM:SS, HH:MM, or HH"
            )
        total_hours = (
            timedelta(
                days=days, hours=hours, minutes=minutes, seconds=seconds
            ).total_seconds()
            / 3600
        )
        return total_hours

    raise ValueError(
        f"Invalid time value: {time_value}. Must be float or DD:HH:MM:SS format"
    )


def _convert_none(v):
    return None if ((isinstance(v, str) and v.lower() == "none") or v is None) else v


def _validate_choice(value: str, *, allowed: set[str], context: str, key: str) -> str:
    if value not in allowed:
        raise ValueError(
            f"Invalid {context}.{key} '{value}'. Allowed values are: {sorted(allowed)}."
        )
    return value
