"""User-configured tool locations and lexical aliases for moved storage.

Optional registry tables: ``[tool_paths.<series>]`` and
``[storage.path_aliases]``. Relative tool paths use the series project root;
relative alias destinations use the registry directory. No drive is assumed.
A series manifest may also keep its aliases in pinned table files, named by
``[storage].path_alias_tables`` (see :func:`series_path_aliases`).

A machine path map (:func:`set_machine_path_map`) names this computer's folder
for a shared root, for example a share mounted at ``/Volumes/<share>``. Only a
GUI installs one; without it every function here behaves as before.
"""
from __future__ import annotations

import ntpath
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tomllib

from .series_paths import series_key, series_registry_path

_settings_stamp = None
_settings_data: dict = {}
_tool_cache: dict = {}
_shared_alias_cache: dict = {}
_shared_alias_roots: dict[str, set[str]] = {}
_shared_alias_generation = 0
# The installed machine path map: rows as given, canonical root -> local
# folder (normalized, case-folded keys) and local folder -> stored spelling.
_machine_map: tuple[tuple[str, str], ...] = ()
_machine_sources: dict[str, str] = {}
_machine_locals: dict[str, tuple[str, bool]] = {}

_DEFAULTS = {
    "calibration_logs": ("calibration", ""),
    "analysis_v2": ("results", "analysis"),
    "combined_reviews": ("reviews", "combined"),
    "review_campaign": ("reviews", ""),
    "physics_layer_exports": ("results", "physics_layers"),
    "raw_inputs": ("data", "raw"),
    "article_inputs": ("data", "article"),
    "article_raw_inputs": ("data", "article/raw"),
    "runtime_inputs": ("data", "runtime"),
    "previous_results": ("results", ""),
    "dataset_root": ("root", ""),
}


def _settings() -> tuple[Path, dict]:
    global _settings_stamp, _settings_data
    # The registry is local configuration. Never resolve a candidate image here.
    registry = series_registry_path()
    try:
        stat = registry.stat()
        revision = (stat.st_mtime_ns, stat.st_size)
    except FileNotFoundError:
        revision = None
    stamp = (str(registry), revision)
    if stamp != _settings_stamp:
        if revision is None:
            data = {}
        else:
            with registry.open("rb") as stream:
                data = tomllib.load(stream)
            if data.get("schema_version") != 1:
                raise ValueError(f"Unsupported DarSIA series registry: {registry}")
        _settings_data = data
        _settings_stamp = stamp
        _tool_cache.clear()
    return registry, _settings_data


def _normalized(value) -> str:
    value = os.path.expanduser(os.fspath(value)).replace("\\", "/")
    # ntpath supports Windows spellings even in tests on a CPU-only Unix host.
    result = ntpath.normpath(value).replace("\\", "/").rstrip("/")
    if len(result) == 2 and result[1] == ":":
        result += "/"
    return result or "/"


def _is_absolute(value: str) -> bool:
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


def refresh_storage_paths(series_root=None) -> None:
    """Refresh shared location aliases after an explicit configuration save."""
    global _shared_alias_generation
    _shared_alias_generation += 1
    if series_root is not None:
        _shared_alias_roots.setdefault(str(series_registry_path()), set()).add(str(Path(series_root).absolute()))
    _shared_alias_cache.clear()
    _tool_cache.clear()
    from .series_paths import _cached_manifest_paths
    _cached_manifest_paths.cache_clear()


def _windows_spelled(text: str) -> bool:
    """Return True for a path with a Windows drive or UNC root, in either slash form."""
    return bool(PureWindowsPath(text).drive)


def validate_machine_path_map(rows) -> tuple[tuple[str, str], ...]:
    """Return a machine path map's (canonical root, local folder) rows, checked.

    The canonical root is absolute in any spelling, normally the UNC root of a
    network share (``\\\\host\\share``) or a drive (``X:``), or a POSIX folder.
    The local folder is absolute on this computer and not inside a canonical
    root. A canonical root appears once; surrounding blanks are removed.
    Raises ValueError with a plain explanation otherwise.
    """
    cleaned = []
    for row in rows:
        canonical, local = (str(value).strip() for value in row)
        if not canonical or not local:
            raise ValueError("Each row needs a shared root and a folder on this computer.")
        spelled = _normalized(canonical)
        if not (_is_absolute(spelled) or PurePosixPath(spelled).is_absolute()) or not spelled.rstrip("/"):
            raise ValueError(f"The shared root must be a full path, such as \\\\server\\share: {canonical}")
        if not Path(os.path.expanduser(local)).is_absolute():
            raise ValueError(f"The folder on this computer must be a full path: {local}")
        if not _normalized(local).rstrip("/"):
            raise ValueError("The folder on this computer cannot be the top of the disk.")
        cleaned.append((canonical, local))
    sources = {}
    for canonical, local in cleaned:
        key = _normalized(canonical).rstrip("/").casefold()
        if key in sources:
            raise ValueError(f"The shared root is listed twice: {canonical}")
        sources[key] = local
    for canonical, local in cleaned:
        if _longest_source(_normalized(local).casefold(), sources) is not None:
            raise ValueError(f"The folder on this computer lies inside a shared root: {local}")
    return tuple(cleaned)


def set_machine_path_map(rows) -> None:
    """Install this computer's folder for each canonical storage root.

    ``rows`` are (canonical root, local folder) pairs, checked by
    :func:`validate_machine_path_map`. Readers then find a stored path below a
    canonical root below the local folder, after the storage aliases, and
    :func:`machine_stored_text` stores local paths in the canonical spelling.
    For one local folder the first row gives that spelling. An empty sequence
    removes the map. Cached locations are refreshed.
    """
    global _machine_map, _machine_sources, _machine_locals
    checked = validate_machine_path_map(rows)
    sources, locals_ = {}, {}
    for canonical, local in checked:
        stored = _normalized(canonical)
        sources[stored.rstrip("/").casefold()] = _normalized(local).rstrip("/")
        locals_.setdefault(_normalized(local).rstrip("/").casefold(), (stored, _windows_spelled(stored)))
    _machine_map, _machine_sources, _machine_locals = checked, sources, locals_
    refresh_storage_paths()


def machine_path_map() -> tuple[tuple[str, str], ...]:
    """Return the installed (canonical root, local folder) rows; empty by default."""
    return _machine_map


def _machine_applies() -> bool:
    """Return True when a machine path map is installed and no job input scope is open.

    A job reads its inputs as recorded, so the map never applies inside one.
    """
    if not _machine_sources:
        return False
    from .preset_wasserstein_inputs import current_scope
    return current_scope() is None


def _on_this_machine(path) -> Path | None:
    """Return ``path`` below this computer's folder for its canonical root, else None."""
    if not _machine_applies():
        return None
    text = _normalized(path)
    source = _longest_source(text.casefold(), _machine_sources)
    return None if source is None else Path(_machine_sources[source] + text[len(source):])


def machine_local_root(path) -> str | None:
    """Return this computer's folder of the machine path map that holds ``path``.

    The folder is normalized and case-folded, so two paths below the same
    folder give the same value. None when no installed folder holds the path.
    """
    if not _machine_locals:
        return None
    return _longest_source(_normalized(path).casefold(), _machine_locals)


def _through_aliases(path, rows, table=None) -> Path:
    """Resolve ``path`` through already read alias rows, as this computer reads it.

    Aliases name shared roots, so with a machine path map a path below this
    computer's folder is looked up in its shared spelling, and the result is
    read below this computer's folder again. Without a map this is
    :func:`_resolve_with`.
    """
    if not _machine_applies():
        return _resolve_with(path, rows, table)
    resolved = _resolve_with(machine_stored_text(path), rows, table)
    mapped = _on_this_machine(resolved)
    return resolved if mapped is None else mapped


def machine_local_path(value) -> Path:
    """Return a stored path as this computer reads it, without storage aliases.

    A path below a canonical root of the machine path map is read below this
    computer's folder. On a computer other than Windows, a remaining Windows
    spelling (a drive, a UNC root or backslash separators) is read with forward
    slashes. On Windows without a map the result is ``Path(value)``.
    """
    mapped = _on_this_machine(value)
    if mapped is not None:
        return mapped
    text = os.fspath(value)
    if os.name != "nt" and (_windows_spelled(text) or "\\" in text):
        return Path(PureWindowsPath(text).as_posix())
    return Path(value)


def machine_local_text(text: str) -> str:
    """Return stored path text as this computer reads it, without storage aliases.

    Text below a canonical root of the machine path map becomes the path below
    this computer's folder; any other text, and all text without a map, is
    returned unchanged (the same string).
    """
    mapped = _on_this_machine(text)
    return text if mapped is None else str(mapped)


def machine_stored_text(path) -> str:
    """Return the text a shared record stores for ``path``.

    A path below this computer's folder of the machine path map is written in
    its canonical spelling (with Windows separators for a Windows root), so
    other computers read it. Any other path, and every path when no map is
    installed, is ``str(path)``.
    """
    text = str(path)
    local = machine_local_root(text)
    if local is None:
        return text
    canonical, windows = _machine_locals[local]
    parts = [part for part in _normalized(text)[len(local):].split("/") if part]
    return str((PureWindowsPath if windows else PurePosixPath)(canonical, *parts))


def _aliases(table, base):
    rows = {}
    for source, destination in table.items():
        source, destination = _normalized(source), _normalized(destination)
        if not _is_absolute(source):
            raise ValueError(f"Storage alias source must be absolute: {source}")
        if not _is_absolute(destination):
            destination = _normalized(base / destination)
        rows[source.rstrip("/").casefold()] = destination.rstrip("/")
    return rows


ALIAS_TABLE_LIMIT = 16 * 1024 * 1024


def only_added_storage_aliases(before, after, folder) -> bool:
    """True when two series manifests differ only by added storage alias rows.

    Compares the complete alias tables (inline rows and pinned table files,
    read against the manifest folder ``folder``): every earlier row must stay
    unchanged and at least one row must be added. All other settings must be
    equal. A relocation adds such rows; it cannot change another saved choice.
    """
    import json
    from copy import deepcopy
    old, new = deepcopy(before), deepcopy(after)
    old_aliases = series_path_aliases(old, folder)
    new_aliases = series_path_aliases(new, folder)
    if (len(new_aliases) <= len(old_aliases)
            or any(key not in new_aliases or new_aliases[key] != value
                   for key, value in old_aliases.items())):
        return False
    for document in (old, new):
        if isinstance(document.get("storage"), dict):
            document["storage"].pop("path_aliases", None)
            document["storage"].pop("path_alias_tables", None)
            if not document["storage"]:
                document.pop("storage")
    return json.dumps(old, sort_keys=True, allow_nan=False) == json.dumps(
        new, sort_keys=True, allow_nan=False)


def series_path_aliases(document, folder) -> dict:
    """Return a series manifest's complete storage alias table, as written.

    Rows come from ``[storage.path_aliases]`` and from the table files that
    ``[storage].path_alias_tables`` names. Each table entry gives the file's
    path (relative to the manifest folder ``folder``), SHA-256 and size; the
    file holds ``[path_aliases]`` with the same row syntax, and its relative
    destinations are read against the manifest folder too, so rows can move
    between the manifest and a table unchanged. A missing or changed table
    file is refused, and so is one source given two destinations.
    """
    import hashlib
    storage = document.get("storage") or {}
    rows = dict(storage.get("path_aliases") or {})
    seen = {_normalized(source).casefold(): destination for source, destination in rows.items()}
    for entry in storage.get("path_alias_tables") or []:
        if (not isinstance(entry, dict) or not isinstance(entry.get("path"), str)
                or not isinstance(entry.get("sha256"), str) or type(entry.get("size_bytes")) is not int):
            raise ValueError("A storage alias table needs its path, SHA-256 and size")
        path = Path(entry["path"])
        path = path if _is_absolute(entry["path"]) else Path(folder) / path
        with path.open("rb") as stream:
            content = stream.read(ALIAS_TABLE_LIMIT + 1)
        if (len(content) != entry["size_bytes"] or len(content) > ALIAS_TABLE_LIMIT
                or hashlib.sha256(content).hexdigest() != entry["sha256"]):
            raise ValueError("A storage alias table differs from its pinned bytes: " + str(path))
        table = tomllib.loads(content.decode("utf-8-sig"))
        if table.get("schema_version") != 1 or not isinstance(table.get("path_aliases"), dict):
            raise ValueError("Unsupported storage alias table: " + str(path))
        for source, destination in table["path_aliases"].items():
            key = _normalized(source).casefold()
            if key in seen and seen[key] != destination:
                raise ValueError("Storage alias tables give two destinations for: " + source)
            seen[key] = destination
            rows.setdefault(source, destination)
    return rows


def _shared_alias_rows(registry, settings, local, series_roots):
    """Read each shared configuration once, without recursive series discovery."""
    raw_roots = [row["root"] for row in settings.get("series", [])]
    raw_roots += [value for key, value in sorted(os.environ.items())
                  if re.fullmatch(r"DARSIA_[A-Z][A-Z0-9_]*_PROJECT_ROOT", key) and value]
    raw_roots += [*_shared_alias_roots.get(str(registry), ()), *series_roots]
    local_rows = sorted(local.items(), key=lambda row: len(row[0]), reverse=True)
    roots = []
    for raw in raw_roots:
        path = Path(raw).expanduser()
        if not _is_absolute(str(path)):
            path = registry.parent / path
        roots.append(str(_through_aliases(path, local_rows)))
    key = (_settings_stamp, _shared_alias_generation, tuple(sorted(set(roots))))
    if key in _shared_alias_cache:
        return _shared_alias_cache[key]
    rows = {}
    for raw in key[2]:
        root = Path(raw)
        for manifest in (root / "series.toml", root / "DarSIA" / "series.toml"):
            try:
                with manifest.open("rb") as stream:
                    content = stream.read(1024 * 1024 + 1)
            except OSError:
                continue
            if len(content) > 1024 * 1024:
                raise ValueError("Shared series settings exceed 1 MiB: " + str(manifest))
            document = tomllib.loads(content.decode("utf-8-sig"))
            aliases = _aliases(series_path_aliases(document, manifest.parent), manifest.parent)
            for source, destination in aliases.items():
                if source in rows and rows[source] != destination and source not in local:
                    raise ValueError("Shared series define conflicting storage destinations: " + source)
                rows[source] = destination
            break
    _shared_alias_cache[key] = rows
    return rows


def _alias_rows(*, series_roots=()) -> list[tuple[str, str]]:
    """Return shared and local normalized (source, destination) prefix aliases.

    Sources are case-folded and sorted longest first; relative destinations are
    read against their configuration folder. Explicit local aliases take
    precedence for the same source. An empty list means no alias is set.
    """
    from .preset_wasserstein_inputs import current_scope
    scope = current_scope()
    if scope is not None:
        return [(row["source"], row["destination"]) for row in scope.path_aliases]
    registry, settings = _settings()
    local = _aliases(settings.get("storage", {}).get("path_aliases", {}), registry.parent)
    shared = _shared_alias_rows(registry, settings, local, series_roots)
    rows = list(dict(shared, **local).items())
    rows.sort(key=lambda row: len(row[0]), reverse=True)
    return rows


def storage_path_remapper(*, series_roots=()):
    """Return a function that reads path strings through the storage aliases.

    The registry is read once, for callers that map many paths at a time. The
    function returns a string that an alias source or a canonical root of the
    machine path map prefixes as :func:`resolve_storage_path` would; any other
    string is returned unchanged, character for character, so a machine
    without aliases reads exactly the authored text.
    """
    rows = _alias_rows(series_roots=series_roots) if series_roots else _alias_rows()
    table = dict(rows)

    def remap(value) -> str:
        text = os.fspath(value)
        lookup = machine_stored_text(text) if _machine_applies() else text
        if rows and _longest_source(_normalized(lookup).casefold(), table) is not None:
            return str(_through_aliases(text, rows, table))
        mapped = _on_this_machine(text)
        return text if mapped is None else str(mapped)

    return remap


def resolve_storage_path(path, *, series_roots=()) -> Path:
    """Apply explicitly configured prefix aliases without accessing old storage.

    Missing destinations do not fall back to an obsolete source. This changes
    only the path used by a reader, never the bytes of its frozen document.
    With a machine path map, aliases see the shared spelling and the result is
    read below this computer's folder.
    """
    return _through_aliases(path, _alias_rows(series_roots=series_roots) if series_roots else _alias_rows())


def same_storage_path(first, second) -> bool:
    """Compare explicit locators through configured aliases without file reads."""
    return _normalized(resolve_storage_path(first)).casefold() == _normalized(resolve_storage_path(second)).casefold()


def same_storage_descriptor(first, second) -> bool:
    """Keep every descriptor field/type exact, allowing only aliased path spelling."""
    def located(value):
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            return value
        return dict(value, path=_normalized(resolve_storage_path(value["path"])).casefold())
    return json.dumps(located(first), sort_keys=True, allow_nan=False) == json.dumps(
        located(second), sort_keys=True, allow_nan=False)


def _longest_source(key, table):
    """Return the longest alias source that is ``key`` or a folder above it, else None.

    ``key`` is a normalized, case-folded path; ``table`` maps sources in the
    same form. Walking up the path's own folders finds the same source as
    trying every source longest first, at the cost of the path's depth.
    """
    probe = key
    while probe:
        if probe in table:
            return probe
        cut = probe.rfind("/")
        if cut <= 0:
            return None
        probe = probe[:cut]
    return None


def _resolve_with(path, rows, table=None) -> Path:
    """Resolve ``path`` through already read alias rows (see _alias_rows)."""
    if not rows:
        return Path(path).expanduser()
    table = dict(rows) if table is None else table
    result = _normalized(path)
    seen = set()
    applied = set()
    while True:
        key = result.casefold()
        if key in seen:
            raise ValueError(f"Cyclic storage path aliases for {path}")
        seen.add(key)
        source = _longest_source(key, table)
        if source is None:
            return Path(result)
        if source in applied:
            raise ValueError(f"Cyclic storage path aliases for {path}")
        applied.add(source)
        result = table[source] + result[len(source):]
        if len(result) == 2 and result[1] == ":":
            result += "/"


def configured_tool_path(dataset: str, key: str) -> Path:
    """Resolve a tool default without embedding a researcher's storage layout.

    User overrides precede explicit manifest paths and legacy input locations.
    Output defaults stay inside the registered project. Unconfigured projects
    use a portable local project location; they never select someone else's
    scientific inputs. Explicit CLI paths continue to override tool defaults.
    """
    if key not in _DEFAULTS:
        raise KeyError(f"Unknown tool path: {key}")
    registry, settings = _settings()
    dataset = series_key(dataset)
    explicit_root = os.environ.get(f"DARSIA_{dataset.upper()}_PROJECT_ROOT")
    cache_key = (dataset, key, explicit_root)
    if cache_key in _tool_cache:
        return _tool_cache[cache_key]
    raw_root = explicit_root or next((row["root"] for row in settings.get("series", [])
                                     if series_key(row["id"]) == dataset), None)
    if raw_root:
        root = resolve_storage_path(raw_root)
        if not _is_absolute(str(root)):
            root = registry.parent / root
        # Accept experiment-root registration as well as project registration.
        if not (root / "series.toml").is_file() and (root / "DarSIA/series.toml").is_file():
            root = root / "DarSIA"
    else:
        from .project import default_project_root
        root = default_project_root() / dataset
    manifest_path = root / "series.toml"
    manifest = {}
    if manifest_path.is_file():
        with manifest_path.open("rb") as stream:
            manifest = tomllib.load(stream)
        if manifest.get("schema_version") != 1 or series_key(manifest["series_id"]) != dataset:
            raise ValueError(f"Wrong series manifest: {manifest_path}")
    overrides = settings.get("tool_paths", {}).get(dataset, {})
    raw = overrides.get(key, manifest.get("paths", {}).get(key,
                        manifest.get("legacy", {}).get(key)))
    if raw is None and key == "article_raw_inputs":
        article = overrides.get("article_inputs", manifest.get("paths", {}).get(
            "article_inputs", manifest.get("legacy", {}).get("article_inputs")))
        if article is not None:
            raw = str(Path(article) / "raw")
    if raw is None:
        base, suffix = _DEFAULTS[key]
        if key == "calibration_logs":
            from .series_paths import series_folder_path
            raw = str(series_folder_path(root, "calibration"))
        else:
            raw = str(Path(manifest.get("paths", {}).get(base, "." if base == "root" else base)) / suffix)
    result = resolve_storage_path(raw)
    if not _is_absolute(str(result)):
        result = root / result
    result = Path(os.path.abspath(result))
    _tool_cache[cache_key] = result
    return result
