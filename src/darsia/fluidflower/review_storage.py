"""Canonical NAS locations for byte-preserved legacy review sessions.

Alias registries are series data, never code. Frozen documents keep their old
paths and hashes; readers and resumed writers resolve those paths at the edge.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .series_paths import series_path
from .storage_paths import _is_absolute, _normalized, _windows_spelled, resolve_storage_path

FORMAT = "fluidflower-review-storage-aliases-v1"
_cached_key = None
_cached_stamp = None
_cached_aliases: tuple[dict, ...] = ()
_cached_lookup: tuple[tuple[str, str, str, Path, Path, int], ...] = ()


def series_review_root(series: str) -> Path:
    """Return the selected series' authoritative location for new reviews."""
    return series_path(series, "reviews")


def _key(path) -> str:
    return str(_absolute(path)).replace("\\", "/").rstrip("/").casefold()


def _absolute(path) -> Path:
    # Path.resolve/ntpath.realpath opens each UNC path; alias lookup must be
    # lexical and independent of NAS availability, including for future writes.
    text = os.fspath(path)
    if os.name != "nt" and _windows_spelled(text):
        # A Windows-written locator read elsewhere keeps its drive or UNC root
        # until the storage aliases and the machine path map translate it.
        return Path(_normalized(text))
    return Path(os.path.abspath(os.path.expanduser(text)))


def _inside(path_key: str, root_key: str) -> bool:
    return path_key == root_key or path_key.startswith(root_key + "/")


def _configuration_stamp(configured: str | None) -> tuple:
    """Cheap local change detector; avoid opening series roots for every path read."""
    registry = os.environ.get("DARSIA_SERIES_REGISTRY")
    if registry:
        registry_path = Path(os.path.expanduser(registry))
    else:
        config_root = os.environ.get("APPDATA") if os.name == "nt" else os.environ.get("XDG_CONFIG_HOME")
        root = Path(config_root) if config_root else Path.home() / (
            "AppData/Roaming" if os.name == "nt" else ".config")
        registry_path = root / "DarSIA" / "series_registry.toml"
    try:
        stat = os.stat(registry_path)
        revision = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        revision = None
    project_roots = tuple(sorted((name, value) for name, value in os.environ.items()
        if name.startswith("DARSIA_") and name.endswith("_PROJECT_ROOT")))
    return (configured, str(registry_path), revision, project_roots,
            os.environ.get("DARSIA_FLUIDFLOWER_DATA_ROOT"))


def review_aliases(*, refresh=False) -> tuple[dict, ...]:
    """Read registries once per configuration; explicit refresh sees new copies."""
    global _cached_key, _cached_stamp, _cached_aliases, _cached_lookup
    from .series_paths import configured_series

    configured = os.environ.get("DARSIA_REVIEW_ALIAS_FILES")
    stamp = _configuration_stamp(configured)
    if not refresh and stamp == _cached_stamp:
        return _cached_aliases
    series = configured_series()
    key = (configured, tuple((name, str(root)) for name, _label, root in series))
    if not refresh and key == _cached_key:
        _cached_stamp = stamp
        return _cached_aliases
    if configured:
        paths = tuple(Path(p) for p in configured.split(os.pathsep) if p)
    else:
        available = []
        for name, _label, _root in series:
            try:
                available.append(series_review_root(name) / "storage_aliases.json")
            except (OSError, ValueError):
                continue
        paths = tuple(available)
    records = {}
    for path in paths:
        if not path.is_file():
            continue
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        if document.get("format") != FORMAT:
            raise ValueError(f"Unsupported review storage aliases: {path}")
        for raw in document.get("aliases", []):
            row = dict(raw)
            for field in ("source", "destination", "original_save_root", "save_root"):
                if not _is_absolute(row[field]):
                    raise ValueError(f"Review storage alias requires an absolute {field}: {path}")
            source_key = _key(row["source"])
            if source_key == _key(row["destination"]):
                raise ValueError(f"Review storage alias points at itself: {path}")
            previous = records.get(source_key)
            if previous is not None and previous != row:
                raise ValueError(f"Conflicting review storage aliases: {row['source']}")
            records[source_key] = row
    _cached_aliases = tuple(records[key] for key in sorted(records, key=len, reverse=True))
    # Alias files are stable for this configuration. Re-parsing 342 source and
    # destination paths on each native-image read dominates discovery time.
    _cached_lookup = tuple((
        _key(row["source"]), _key(row["destination"]), _key(row["original_save_root"]),
        _absolute(row["save_root"]), _absolute(row["destination"]),
        len(_absolute(row["source"]).parts),
    ) for row in _cached_aliases)
    _cached_key = key
    _cached_stamp = stamp
    return _cached_aliases


def resolve_review_path(path, *, context=None) -> Path:
    """Resolve a legacy review path for both reading and subsequent writes.

    ``context`` identifies the session when resolving its old shared save root.
    A missing NAS destination remains authoritative: never fall back to stale
    local evidence after migration.
    """
    result = _absolute(path)
    review_aliases()
    lookup = _cached_lookup
    if context is not None:
        context_key = _key(context)
        for source_key, destination_key, original_save_key, save_root, _destination, _parts in lookup:
            if _inside(context_key, source_key) or _inside(context_key, destination_key):
                if _key(result) == original_save_key:
                    return _absolute(resolve_storage_path(save_root))
                break
    seen = set()
    while True:
        current_key = _key(result)
        if current_key in seen:
            raise ValueError(f"Cyclic review storage aliases for {path}")
        seen.add(current_key)
        for source_key, _destination_key, _original_save_key, _save_root, destination, source_parts in lookup:
            if _inside(current_key, source_key):
                # relative_to is case-insensitive on the Windows runtime; use
                # path components so a differently cased legacy root also works.
                suffix = result.parts[source_parts:]
                result = _absolute(destination.joinpath(*suffix))
                break
        else:
            return _absolute(resolve_storage_path(result))
