"""Shared location choices of a series; changing them never moves saved products."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import tomllib
import uuid

from .series_creation import _relative_folder
from .series_paths import series_key
from .storage_paths import machine_stored_text, resolve_storage_path

LIMIT = 1024 * 1024


def _read(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(LIMIT + 1)
    if len(raw) > LIMIT:
        raise ValueError("Series configuration exceeds 1 MiB: " + str(path))
    return raw, tomllib.loads(raw.decode("utf-8-sig"))


def _path(root, value):
    path = resolve_storage_path(value)
    return Path(os.path.abspath(path if path.is_absolute() else root / path))


def read_series_locations(root):
    """Read shared locations and the original manifest bytes, without scanning data."""
    root = Path(root)
    raw, document = _read(root / "series.toml")
    if document.get("schema_version") != 1:
        raise ValueError("A schema-1 series manifest is required")
    series_key(document["series_id"])
    paths = document.get("paths") or {}
    return dict(series_root=root, series_id=document["series_id"], manifest_bytes=raw,
                project_folder=_path(root, paths.get("project", "..")),
                results_folder=_path(root, paths.get("results", "results")),
                jobs_folder=_path(root, paths.get("jobs", "jobs")))


def _preserve_inputs(config, merged, old_results):
    """Keep native input defaults bound when only future result output changes."""
    for section, key, suffix in (
        ("rig", "path", "setup/rig"),
        ("depth", "depth_map", "setup/depth/depth_map.npz"),
        ("labeling", "labels", "setup/labels/labels.npz"),
        ("facies", "path", "setup/facies/facies.npz"),
    ):
        if section in merged and not (merged[section] or {}).get(key):
            config.setdefault(section, {})[key] = machine_stored_text(old_results / suffix)
    for section, kind in (("color_path", "color_path"), ("color_range", "color_range"),
                          ("color_channel", "color_channel")):
        if section in config:
            entries = deepcopy(config[section])
            for entry in entries:
                if not entry.get("root"):
                    entry["root"] = machine_stored_text(old_results / "color" / kind / str(entry["name"]))
            config[section] = entries
    colour = merged.get("color_to_mass")
    if isinstance(colour, dict) and not colour.get("calibration_folder"):
        config.setdefault("color_to_mass", {})["calibration_folder"] = machine_stored_text(
            old_results / "calibration" / "color_to_mass" / ("from_" + str(colour.get("basis", "labels"))))


def save_series_locations(root, project_folder, results_folder, *, jobs_folder=None, expected_bytes=None,
                          preserve_inputs=True, replace_custom_results=False, path_aliases=None):
    """Save shared defaults and native run outputs using the existing TOML save path.

    Explicit per-run outputs are retained unless ``replace_custom_results`` is
    selected. Existing input references and all saved products stay where they are.
    """
    from .series_geometry import _merge
    from .series_method import _write_manifest, _manifest_bytes, ManifestChangedError

    root = Path(root)
    current = read_series_locations(root)
    original = current["manifest_bytes"]
    if expected_bytes is not None and original != expected_bytes:
        raise ManifestChangedError("Series locations changed since Load; load them again")
    document = tomllib.loads(original.decode("utf-8-sig"))
    updated = deepcopy(document)
    project, results = _path(root, project_folder), _path(root, results_folder)
    paths = updated.setdefault("paths", {})
    paths.update(project=_relative_folder(project, root), results=_relative_folder(results, root))
    if path_aliases:
        updated.setdefault("storage", {}).setdefault("path_aliases", {}).update(
            {machine_stored_text(source): machine_stored_text(destination)
             for source, destination in path_aliases.items()})
    roots = updated.setdefault("roots", {"series": "."})
    roots["results"] = paths["results"]
    if jobs_folder is not None:
        paths["jobs"] = _relative_folder(_path(root, jobs_folder), root)
        roots["jobs"] = paths["jobs"]
    if project not in (root.absolute(), results):
        roots["project"] = paths["project"]
    else:
        roots.pop("project", None)
    common_path = _path(root, paths.get("common_config", "config/common.toml"))
    common_before, common = _read(common_path) if common_path.is_file() else (None, {})
    runs = document.get("runs") or {}
    records = []
    for run in runs.get("ids") or []:
        config_path = _path(root, (runs.get("config_files") or {}).get(run,
            str(Path(paths.get("run_configs", "config/runs")) / (run + ".toml"))))
        before, config = _read(config_path)
        merged = _merge(common, config)
        raw_results = (merged.get("data") or {}).get("results")
        old_results = _path(root, raw_results) if raw_results else current["results_folder"] / run
        replace = replace_custom_results or old_results == current["results_folder"] / run
        records.append((run, config_path, before, config, merged, old_results, replace))
    split_sections = []
    if preserve_inputs and any(replace and old != results / run for run, _, _, _, _, old, replace in records):
        split_sections = [section for section in ("color_path", "color_range", "color_channel")
                          if any(not entry.get("root") for entry in common.get(section, []))]
    changes, retained = [], []
    for run, config_path, before, config, merged, old_results, replace in records:
        after = deepcopy(config)
        for section in split_sections:
            after[section] = deepcopy(common[section]) + deepcopy(config.get(section, []))
        if not replace:
            retained.append(run)
        if preserve_inputs and (split_sections or replace and old_results != results / run):
            _preserve_inputs(after, merged, old_results)
        if replace:
            after.setdefault("data", {})["results"] = machine_stored_text(results / run)
        if after != config:
            changes.append((config_path, before, after))
    if split_sections:
        after_common = deepcopy(common)
        for section in split_sections:
            del after_common[section]
        changes.append((common_path, common_before, after_common))
    # Compare every affected document before the first write, then reuse each
    # original writer's compare-and-swap. The shared defaults are saved last.
    for path, before, _after in changes:
        if path.read_bytes() != before:
            raise ManifestChangedError("Run configuration changed while saving: " + str(path))
    written = []
    try:
        for path, before, after in [*changes, (root / "series.toml", original, updated)]:
            saved = _manifest_bytes(before, after)[0]
            _write_manifest(path, before, after)
            written.append((path, before, saved))
    except Exception as error:
        for path, before, saved in reversed(written):
            temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".rollback")
            try:
                if path.read_bytes() != saved:
                    error.add_note("Preserved a concurrent edit while rolling back: " + str(path))
                    continue
                with temporary.open("xb") as stream:
                    stream.write(before)
                if path.read_bytes() == saved:
                    os.replace(temporary, path)
                else:
                    error.add_note("Preserved a concurrent edit while rolling back: " + str(path))
            except OSError as rollback_error:
                error.add_note("Could not restore " + str(path) + ": " + str(rollback_error))
            finally:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    error.add_note("Could not remove own rollback temporary: " + str(cleanup_error))
        from .storage_paths import refresh_storage_paths
        refresh_storage_paths(root)
        raise
    from .storage_paths import refresh_storage_paths
    refresh_storage_paths(root)
    return dict(**read_series_locations(root), updated_runs=[str(path) for path, _, _ in changes if path != common_path],
                retained_runs=retained)


def save_series_jobs_folder(root, folder, *, expected_bytes=None):
    """Save one shared queue choice only; no job, worker or product is moved."""
    from .series_method import _write_manifest, ManifestChangedError
    root = Path(root)
    original, document = _read(root / "series.toml")
    if expected_bytes is not None and original != expected_bytes:
        raise ManifestChangedError("Series locations changed; load them again")
    if document.get("schema_version") != 1:
        raise ValueError("A schema-1 series manifest is required")
    series_key(document["series_id"])
    chosen = Path(folder).expanduser()
    if not chosen.is_absolute() or ".." in chosen.parts:
        raise ValueError("Choose an absolute shared Jobs folder")
    updated = deepcopy(document)
    relative = _relative_folder(chosen, root)
    updated.setdefault("paths", {})["jobs"] = relative
    updated.setdefault("roots", {"series": "."})["jobs"] = relative
    _write_manifest(root / "series.toml", original, updated)
    from .storage_paths import refresh_storage_paths
    refresh_storage_paths(root)
    return read_series_locations(root)


def machine_jobs_folder(root, shared_folder, stores_path):
    """Resolve this computer's explicit jobs mount; the series path stays shared."""
    from .store_map import read_store_map
    root, shared = Path(root), Path(shared_folder)
    declared = read_series_locations(root)
    from .storage_paths import same_storage_path
    if not same_storage_path(shared, declared["jobs_folder"]) or stores_path is None or not Path(stores_path).is_file():
        return shared
    stores = read_store_map(stores_path, forbidden_roots=[root])
    return Path(stores.roots(declared["series_id"]).get("jobs", str(shared)))


def store_map_with_series_locations(raw, root):
    """Prepare editor text using shared locations; machine caches remain unchanged."""
    import toml
    from .store_map import parse_store_map

    current = read_series_locations(root)
    document = tomllib.loads(raw.decode("utf-8-sig"))
    stores = document.setdefault("stores", {})
    key = next((name for name in stores if series_key(name) == series_key(current["series_id"])),
               current["series_id"])
    selected = stores.setdefault(key, {})
    shared = tomllib.loads(current["manifest_bytes"].decode("utf-8-sig"))
    if shared.get("roots"):
        from .store_map import manifest_roots
        for alias, path in manifest_roots(Path(root).absolute(), shared).items():
            selected.setdefault(alias, str(path))
    selected.update(series=str(Path(root).absolute()), results=str(current["results_folder"]))
    updated = toml.dumps(document)
    parse_store_map(updated.encode("utf-8"))
    return updated
