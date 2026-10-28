"""Portable references to phase-model versions; scientific assets stay in place.

The library is an index, not a second copy of a model. Its references never
grant review or mass eligibility: existing saved-model admission still does so.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tomllib
from typing import Iterable
import uuid

from .project import sha256_file
from .review_storage import resolve_review_path

FORMAT = "phase-model-library-reference-v1"
_HEADS = ("co2_presence_signal.json", "gas_within_co2_signal.json")


class NotPhaseModelSource(ValueError):
    """A summary belongs to a report/provenance copy, not a model version."""


def library_root(series_root: Path | str) -> Path:
    from .series_paths import series_folder_path
    return series_folder_path(series_root, "phase_model_library")


def _library_owner(location: Path) -> Path | None:
    """The phase-model series folder whose library or model folder is ``location``.

    Only series that set [paths] phase_model are recognised here; the fixed
    historical folder names are matched by ``records_for_location``.
    Known series declarations also cover libraries outside their series root.
    """
    from .series_paths import configured_series, phase_model_root, series_folder_path
    key = os.path.normcase(os.path.abspath(location))
    candidates = [*Path(location).parents, *(root for _key, _label, root in configured_series())]
    for candidate in dict.fromkeys(candidates):
        if not (candidate / "series.toml").is_file():
            continue
        phase_root = phase_model_root(candidate)
        if phase_root is None:
            continue
        folders = (phase_root, library_root(candidate),
                   series_folder_path(candidate, "phase_models"), series_folder_path(candidate, "models"))
        if key in {os.path.normcase(os.path.abspath(folder)) for folder in folders}:
            return candidate
    return None


def configured_library_records() -> list[dict]:
    from .series_paths import configured_series
    return [record for _series, _label, root in configured_series() for record in library_records(root)]


def records_for_location(location: Path | str) -> list[dict]:
    """Accept a selected series, models folder or its portable phase library."""
    location = resolve_review_path(location)
    if (location / "series.toml").is_file():
        return library_records(location)
    owner = _library_owner(location)
    if owner is not None:
        return library_records(owner)
    if location.name.casefold() == "phase" and location.parent.name.casefold() == "models":
        return library_records(location.parent.parent)
    if location.name.casefold() in {"models", "phase_models"}:
        return library_records(location.parent)
    if library_root(location).is_dir():
        return library_records(location)
    return []


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


@lru_cache(maxsize=4096)
def _verified_metadata_sha(path: str, size: int, mtime_ns: int) -> str:
    # Metadata-only session cache; a changed file must be rehashed.
    return sha256_file(Path(path))


def _metadata_sha(path: Path) -> str:
    info = path.stat()
    return _verified_metadata_sha(str(path), info.st_size, info.st_mtime_ns)


def source_identity(root: Path | str) -> dict:
    """Bind the model version and runtime, never images, labels or sessions.

    Summary IDs distinguish inference/post-processing revisions with unchanged
    learned heads. Runtime profiles prevent copies with different settings from
    being collapsed merely because their optical heads match.
    """
    root = resolve_review_path(root)
    summary = _read(root / "summary.json") if (root / "summary.json").is_file() else {}
    request = _read(root / "training_request.json") if not summary and (root / "training_request.json").is_file() else {}
    if not summary and not request:
        raise NotPhaseModelSource(f"No saved phase-model identity: {root}")
    declared_id = summary.get("model_id") or summary.get("campaign_id") or request.get("model_id")
    model_id = str(declared_id or root.name)
    files = [root / name for name in ("summary.json", "training_request.json", "regional_model.json")
             if (root / name).is_file()]
    files.extend(sorted(root.glob("phase_*.json")))
    if (root / "models").is_dir():
        files.extend(sorted((root / "models").glob("*/*_signal.json")))
    if not (root.name.startswith("phase-model-")
            or declared_id and root.name == model_id
            or any(path.name.endswith("_signal.json") for path in files)):
        raise NotPhaseModelSource(f"Summary is report/provenance metadata, not a phase-model version: {root}")
    hashes = {path.relative_to(root).as_posix(): _metadata_sha(path) for path in files}
    adapter_heads = {}
    for name in hashes:
        parts = PurePosixPath(name).parts
        if len(parts) == 3 and parts[0] == "models" and parts[-1].endswith("_signal.json"):
            adapter_heads.setdefault(parts[1], set()).add(parts[-1])
    complete = bool(adapter_heads) and all(set(_HEADS) <= heads for heads in adapter_heads.values())
    # A relocation can rewrite paths in the provenance request without changing
    # the saved model/runtime. Each reference still binds its exact request,
    # while copy equivalence uses the actual completed model parameters.
    version_hashes = {name: sha for name, sha in hashes.items()
                      if not complete or name != "training_request.json"}
    digest = hashlib.sha256(json.dumps(version_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"model_id": model_id, "version_sha256": digest,
            "metadata_sha256": hashes, "complete_heads": complete}


def deduplicate_versions(roots: Iterable[Path | str]) -> list[Path]:
    """Keep the first source of an identical, complete saved model version.

    An interrupted attempt without both heads does not prove copy equivalence,
    so distinct incomplete folders remain visible with their existing warnings.
    """
    from .phase_model_history import deduplicate_model_roots
    selected, identities = [], set()
    for root in deduplicate_model_roots(roots):
        try:
            identity = source_identity(root)
        except (OSError, ValueError, TypeError):
            identity = {}
        key = (identity.get("model_id"), identity.get("version_sha256"))
        if identity.get("complete_heads"):
            if key in identities:
                continue
            identities.add(key)
        selected.append(root)
    return selected


def _portable_target(series_root: Path, root: Path) -> dict:
    try:
        relative = root.resolve().relative_to(series_root.resolve())
    except ValueError:
        return {"path": str(root), "relative_to": "absolute"}
    return {"path": relative.as_posix(), "relative_to": "series_root"}


def _target(entry: dict, series_root: Path) -> Path:
    target = entry.get("target") or {}
    value = target.get("path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Library reference has no target path")
    if target.get("relative_to") == "series_root":
        relative = PurePosixPath(value.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or ":" in value:
            raise ValueError("Library target must stay inside its series root")
        return resolve_review_path(series_root / Path(*relative.parts))
    if target.get("relative_to") == "absolute" and Path(value).is_absolute():
        return resolve_review_path(value)
    raise ValueError("Unsupported library target path policy")


def resolve_reference(path: Path | str, *, series_root: Path | str | None = None) -> Path:
    """Resolve only the exact indexed source, failing closed on missing/stale data."""
    path = resolve_review_path(path)
    if series_root is None:
        series_root = _library_owner(path.parent) or path.parent.parent.parent
    series_root = Path(series_root)
    entry = _read(path)
    if entry.get("format") != FORMAT or entry.get("kind") != "phase":
        raise ValueError(f"Not a phase-model library reference: {path}")
    root = _target(entry, series_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Indexed model source is unavailable: {root}")
    if source_identity(root) != entry.get("source_identity"):
        raise ValueError(f"Indexed model source identity changed: {root}")
    return root


def _metadata(root: Path, *, series: str) -> dict:
    summary = _read(root / "summary.json") if (root / "summary.json").is_file() else {}
    request = _read(root / "training_request.json") if (root / "training_request.json").is_file() else {}
    created = str(summary.get("created_utc") or request.get("created_utc") or "")
    timestamp = 0.0
    try:
        timestamp = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
    except ValueError:
        match = re.search(r"(\d{8}T\d{6}Z)", root.name)
        if match:
            stamp = datetime.strptime(match[1], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            created, timestamp = stamp.isoformat(), stamp.timestamp()
        else:
            # This is the original metadata timestamp, never the new index mtime.
            timestamp = max((p.stat().st_mtime for p in (root / "summary.json", root / "training_request.json")
                             if p.is_file()), default=0.0)
    inventory = summary.get("inventory") or {}
    return {"display_name": str(summary.get("display_name") or ""), "created_utc": created,
            "created_timestamp": timestamp, "series": series,
            "datasets": list(inventory.get("datasets") or ([series] if series else []))}


def _series_id(series_root: Path) -> str:
    manifest = series_root / "series.toml"
    if not manifest.is_file():
        return ""
    with manifest.open("rb") as stream:
        value = tomllib.load(stream)
    series = value.get("series") or {}
    return str(series.get("id") or value.get("dataset") or value.get("series_id") or "")


def index_model(series_root: Path | str, model_root: Path | str) -> Path:
    """Create/update just the portable index pointer, never its source artifacts."""
    series_root = resolve_review_path(series_root).resolve()
    root = resolve_review_path(model_root).resolve()
    identity = source_identity(root)
    target = _portable_target(series_root, root)
    folder = library_root(series_root)
    folder.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", identity["model_id"])
    path = folder / f"{name}.json"
    previous_targets = []
    if path.exists():
        old = _read(path)
        previous_targets = list(old.get("previous_targets") or [])
        # Different sources under the same ID remain separately discoverable
        # when their runtime differs, or copy equivalence is not proven.
        same_version = ((old.get("source_identity") or {}).get("model_id") == identity["model_id"]
                        and (old.get("source_identity") or {}).get("version_sha256") == identity["version_sha256"])
        if (not same_version or not identity["complete_heads"]):
            old_target = _target(old, series_root)
            if old_target != root:
                suffix = hashlib.sha256(json.dumps(target, sort_keys=True).encode()).hexdigest()[:12]
                path = folder / f"{name}--{identity['version_sha256'][:12]}-{suffix}.json"
        elif same_version and _target(old, series_root) != root:
            # Existing canonical target wins. Do not redirect review sessions to
            # a different copy solely because the migration scanned it later.
            try:
                resolve_reference(path, series_root=series_root)
                return path
            except (OSError, ValueError, TypeError):
                # Explicit indexing may repair a broken pointer from another
                # verified copy. Ordinary resolve_reference never does this.
                if old.get("target") not in previous_targets:
                    previous_targets.append(old["target"])
    value = {"format": FORMAT, "kind": "phase", "source_identity": identity,
             "target": target, "metadata": _metadata(root, series=_series_id(series_root))}
    if previous_targets:
        value["previous_targets"] = previous_targets
    if path.exists() and _read(path) == value:
        return path
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def library_records(series_root: Path | str) -> list[dict]:
    """Keep bad references visible as unavailable instead of loading a substitute."""
    series_root = resolve_review_path(series_root)
    records = []
    try:
        paths = sorted(library_root(series_root).glob("*.json"))
    except OSError:
        return records
    for path in paths:
        entry, target = {}, None
        try:
            entry = _read(path)
            target = _target(entry, series_root)
            root = resolve_reference(path, series_root=series_root)
            records.append({"reference": str(path), "root": str(root), "available": True,
                            "source_identity": entry["source_identity"], "metadata": entry.get("metadata") or {}})
        except (OSError, ValueError, TypeError, KeyError) as error:
            metadata = entry.get("metadata") or {}
            records.append({"reference": str(path), "root": str(path), "target_root": str(target or ""),
                            "available": False, "source_identity": entry.get("source_identity") or {},
                            "metadata": metadata, "error": str(error)})
    return records


def unavailable_history_entry(record: dict) -> dict:
    metadata = record.get("metadata") or {}
    error = str(record.get("error") or "Unavailable library reference")
    return {"schema": 1, "format": "darsia-fluidflower-phase-model-history-v1",
            "root": record["root"], "model_id": (record.get("source_identity") or {}).get("model_id") or Path(record["root"]).stem,
            "display_name": metadata.get("display_name") or "", "created_utc": metadata.get("created_utc") or "",
            "created_timestamp": metadata.get("created_timestamp") or 0., "datasets": metadata.get("datasets") or [],
            "series_scope": metadata.get("series") or "", "state": "unavailable_library_reference",
            "state_label": "Unavailable library reference", "errors": [error],
            "library_available": False, "library_reference": record["reference"],
            "validation_collection_error": error, "validation_binding": None}


def attach_library_metadata(entries: list[dict], records: list[dict]) -> list[dict]:
    """Record canonical routing and owning series without changing review state."""
    by_root = {os.path.normcase(os.path.normpath(record["root"])): record
               for record in records if record["available"]}
    for entry in entries:
        record = by_root.get(os.path.normcase(os.path.normpath(str(entry["root"]))))
        if not record:
            continue
        entry["library_reference"] = record["reference"]
        entry["library_available"] = True
        entry["library_source_identity"] = record["source_identity"]
        if not entry.get("datasets") and not entry.get("series_scope"):
            # Interrupted fits may not have an inventory yet. Their enclosing
            # series is still documented by its portable library/series.toml.
            entry["series_scope"] = record.get("metadata", {}).get("series") or ""
    return entries


def discover_series_sources(series_root: Path | str) -> list[Path]:
    """Bounded transitional scan; no image/cache/model-asset tree is traversed."""
    from .phase_model_history import _model_directories
    series_root = resolve_review_path(series_root)
    records = library_records(series_root)
    roots = [Path(record["root"]) for record in records if record["available"]]
    roots.extend(registration_sources(series_root / "results"))
    locations = [series_root / "models", series_root / "phase_models", series_root / "results"]
    manifest = series_root / "series.toml"
    if manifest.is_file():
        with manifest.open("rb") as stream:
            settings = tomllib.load(stream)
        for key in ("models", "results"):
            value = (settings.get("paths") or {}).get(key)
            if value:
                path = Path(value)
                locations.append(path if path.is_absolute() else series_root / path)
    locations.extend(_phase_model_locations(series_root, locations))
    for location in locations:
        roots.extend(_model_directories(location, max_depth=4))
    # A stale indexed target cannot bypass its identity refusal via legacy scan.
    valid_targets = {os.path.normcase(os.path.normpath(record["root"])) for record in records if record["available"]}
    blocked = {os.path.normcase(os.path.normpath(record["target_root"])) for record in records
               if not record["available"] and record.get("target_root")} - valid_targets
    return deduplicate_versions(root for root in roots
                                if os.path.normcase(os.path.normpath(str(root))) not in blocked)


def _phase_model_locations(series_root: Path, scanned: Iterable[Path]) -> list[Path]:
    """The [paths] phase_models folder of a series, unless already scanned."""
    from .series_paths import phase_model_root, series_folder_path
    if phase_model_root(series_root) is None:
        return []
    folder = series_folder_path(series_root, "phase_models")
    known = {os.path.normcase(os.path.abspath(path)) for path in scanned}
    return [] if os.path.normcase(os.path.abspath(folder)) in known else [folder]


def registration_sources(results: Path) -> list[Path]:
    results = resolve_review_path(results)
    roots = []
    for path in sorted((results / "phase_model_review_registrations").glob("*.json")):
        try:
            value = _read(path)
            if value.get("model_root"):
                roots.append(resolve_review_path(value["model_root"]))
        except (OSError, ValueError, TypeError):
            continue
    return roots


def index_series(series_root: Path | str) -> dict:
    """Index every discoverable version, retaining QA/incomplete states as-is."""
    series_root = resolve_review_path(series_root)
    # Deliberately inspect legacy sources independently of stale references: an
    # explicit migration can refresh an index after an incomplete fit finishes.
    from .phase_model_history import _model_directories
    roots = registration_sources(series_root / "results")
    roots.extend(discover_series_sources(series_root))
    locations = [series_root / "models", series_root / "phase_models", series_root / "results"]
    for location in (*locations, *_phase_model_locations(series_root, locations)):
        roots.extend(_model_directories(location, max_depth=4))
    indexed, errors, ignored = [], [], []
    for root in deduplicate_versions(roots):
        try:
            indexed.append(str(index_model(series_root, root)))
        except NotPhaseModelSource as error:
            ignored.append({"source": str(root), "reason": str(error)})
        except (OSError, ValueError, TypeError) as error:
            errors.append({"source": str(root), "error": str(error)})
    consolidated = []
    seen = {}
    # Consolidate index pointers only. Originals and the independent legacy
    # discovery routes remain untouched. Failed/unavailable references are kept.
    references = sorted(library_root(series_root).glob("*.json"), key=lambda path: ("--" in path.stem, path.name))
    for path in references:
        try:
            entry = _read(path)
            if entry.get("format") != FORMAT:
                continue
            target = _target(entry, series_root)
            try:
                identity = source_identity(target)
            except NotPhaseModelSource:
                # A report's summary.json is not a saved phase model. Removing
                # its accidentally indexed pointer does not remove that report.
                _remove_index_pointer(path, library_root(series_root))
                indexed = [value for value in indexed if value != str(path)]
                consolidated.append({"reference": str(path), "reason": "not_a_phase_model", "source_preserved": str(target)})
                continue
            old_identity = entry.get("source_identity") or {}
            if (old_identity != identity and old_identity.get("model_id") == identity["model_id"]
                    and old_identity.get("metadata_sha256") == identity["metadata_sha256"]):
                # The bound source bytes are unchanged. Update only a previous
                # index digest policy, not a changed/unavailable source version.
                entry["source_identity"] = identity
                temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
                try:
                    with temporary.open("x", encoding="utf-8") as stream:
                        json.dump(entry, stream, ensure_ascii=False, indent=2)
                        stream.write("\n")
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            if not identity["complete_heads"] or identity != entry.get("source_identity"):
                continue
            key = identity["model_id"], identity["version_sha256"]
            if key in seen:
                _remove_index_pointer(path, library_root(series_root))
                indexed = [value for value in indexed if value != str(path)]
                consolidated.append({"reference": str(path), "canonical_reference": str(seen[key]), "source_preserved": str(target)})
            else:
                seen[key] = path
        except (OSError, ValueError, TypeError):
            continue
    legacy_marker = None
    legacy = series_root / "phase_models"
    if legacy.is_dir():
        marker = legacy / "phase_library_location.json"
        if not marker.exists():
            try:
                target, relative_to = os.path.relpath(library_root(series_root), legacy).replace("\\", "/"), "this_file"
            except ValueError:  # A configured library on another drive.
                target, relative_to = str(library_root(series_root)), "absolute"
            with marker.open("x", encoding="utf-8") as stream:
                json.dump({"format": "phase-model-library-location-v1",
                           "target": target, "relative_to": relative_to,
                           "legacy_sources_preserved": True}, stream, indent=2)
                stream.write("\n")
        legacy_marker = str(marker)
    final_records = library_records(series_root)
    final_indexed = sorted(record["reference"] for record in final_records if record["available"])
    return {"library": str(library_root(series_root)), "indexed": final_indexed, "errors": errors,
            "counts": {"saved_references": len(final_records), "verified_sources": len(final_indexed),
                       "complete_head_sets": sum(record.get("source_identity", {}).get("complete_heads", False)
                                                 for record in final_records if record["available"]),
                       "unavailable_references": sum(not record["available"] for record in final_records)},
            "ignored_non_models": ignored, "consolidated_references": consolidated,
            "legacy_location_marker": legacy_marker,
            "source_assets_moved": False, "default_changed": False}


def _remove_index_pointer(path: Path, folder: Path) -> None:
    """Delete only a validated library-index JSON within the exact new folder."""
    resolved_folder, resolved_path = folder.resolve(), path.resolve()
    if resolved_path.parent != resolved_folder or resolved_path.suffix != ".json":
        raise ValueError(f"Refusing index cleanup outside phase library: {resolved_path}")
    if _read(path).get("format") != FORMAT:
        raise ValueError(f"Refusing to remove a non-index artifact: {path}")
    path.unlink()


def index_registered_model(model_root: Path | str, *, results: Path | str | None = None) -> Path | None:
    """Storage-only registration hook, requiring a documented series root."""
    root = resolve_review_path(model_root)
    candidates = [Path(results).parent] if results is not None else []
    candidates.extend(root.parents)
    for series_root in candidates:
        if (series_root / "series.toml").is_file():
            return index_model(series_root, root)
    return None
