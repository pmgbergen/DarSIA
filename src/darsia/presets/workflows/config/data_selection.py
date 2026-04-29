"""Centralized helpers for resolving workflow data selectors."""

# TODO: Get rid of this file, and use directly data_registry instead.
# The only reason this exists is to support legacy inline selector tables,
# but these should be phased out in favor of central registry definitions and references.

from __future__ import annotations

from pathlib import Path
from warnings import warn

from .data_registry import DataRegistry
from .time_data import TimeData

# user code -> config.load() -> resolve_time_data_selector() -> warn()
DEFAULT_WARNING_STACKLEVEL = 3


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
    warning_stacklevel: int = DEFAULT_WARNING_STACKLEVEL,
) -> TimeData | None:
    """Resolve a workflow selector to ``TimeData``.

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

# TODO Make this part of DataRegistry?
def _resolve_selector(
    cfg: dict,
    key: str,
    *,
    section: str,
    data: Path | None,
    data_registry: DataRegistry | None,
    required: bool = True,
) -> TimeData | None:
    if key not in cfg:
        if required:
            raise KeyError(f"{section}.{key}")
        return None
    selector = cfg[key]
    if isinstance(selector, list) and not all(
        isinstance(token, str) for token in selector
    ):
        raise ValueError(f"{section}.{key} selector lists must contain only strings.")
    if not isinstance(selector, (str, list, dict)):
        raise ValueError(
            f"{section}.{key} must be a selector key (str), list of selector keys, or "
            "inline table (dict)."
        )
    return resolve_time_data_selector(
        cfg,
        key,
        section=section,
        data=data,
        data_registry=data_registry,
        required=required,
    )