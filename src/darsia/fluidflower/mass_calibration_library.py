"""Portable references to real saved mass calibrations, never settings profiles.

Indexing writes small new catalog entries only. Original fitted artifacts,
historical results, defaults and concentration.toml remain untouched.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re

FORMAT = "mass-calibration-library-reference-v1"
KINDS = {
    "native-darsia-fixed-phase-mass-v1": "native",
    "native-darsia-preflash-signal-calibration-v1": "native_signal",
    "joint-chromatic-od-mass-calibration-v1": "joint_optical",
    "joint-chromatic-od-history-experiment-v1": "joint_history",
    "joint-chromatic-local-dilution-experiment-v1": "joint_local",
    "series-mass-measured-gas-v1": "series_mass",
}
_FILES = {"calibration.json", "fits.json"}
_PRUNE = {"preparation", "prepared", "native_frames", "run_packages", "reviews", "signal_cache", "field_cache", "raw_cache", "cache", "work"}
_VERIFIED = {}
_JSON_CACHE = {}
_AUTO_INDEXED = set()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def library_root(series_root):
    from .series_paths import series_folder_path
    return series_folder_path(series_root, "mass_calibration_library")


def series_root_for_results(results_root):
    from .series_paths import configured_series, series_folder_path
    path = Path(results_root).resolve()
    for _key, _name, root in configured_series():
        if path == (series_folder_path(root, "results") / "concentration_mass").resolve():
            return root
    if path.name == "concentration_mass" and path.parent.name.casefold() in {"results", "calibration"}:
        candidate = path.parent.parent / "DarSIA"
        if (candidate / "series.toml").is_file() and path == (series_folder_path(candidate, "results") / path.name).resolve():
            return candidate
        return path.parent.parent
    return None


def portable_target(path, series_root, *, pointer=""):
    path, root = Path(path).resolve(), Path(series_root).resolve()
    try:
        target = {"path": path.relative_to(root).as_posix(), "relative_to": "series_root"}
    except ValueError:
        target = {"path": str(path), "relative_to": "absolute"}
    if pointer:
        target["json_pointer"] = pointer
    return target


def target_path(target, series_root):
    root = Path(series_root).resolve()
    if target.get("relative_to") == "series_root":
        value = Path(target["path"])
        if value.is_absolute() or ".." in value.parts or value.drive:
            raise ValueError("A portable mass-library target must stay inside the series")
        path = (root / value).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Mass-library target escapes its series")
        return path
    if target.get("relative_to") == "absolute" and Path(target["path"]).is_absolute():
        return Path(target["path"]).resolve()
    raise ValueError("Invalid mass-library target base")


def _pointer(value, pointer):
    if pointer and not pointer.startswith("/"):
        raise ValueError("Invalid calibration JSON pointer")
    for token in pointer.split("/")[1:] if pointer else ():
        value = value[token.replace("~1", "/").replace("~0", "~")]
    return value


def _finite_parameters(value):
    if isinstance(value, dict):
        return bool(value) and all(_finite_parameters(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_finite_parameters(v) for v in value)
    return value is None or (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value))


def calibration_kind(value):
    """Require saved fit evidence or an actual bound native frozen snapshot."""
    if not isinstance(value, dict) or value.get("format") not in KINDS:
        return None
    name = KINDS[value["format"]]
    if name in {"native", "native_signal"}:
        provenance = value.get("native_calibrations")
        if not isinstance(value.get("parameters"), dict) or not _finite_parameters(value["parameters"]) or not isinstance(provenance, dict) or not provenance:
            return None
        if not all(isinstance(row, dict) and row.get("calibration_files") for row in provenance.values()):
            return None
        fitted = value.get("fit_performed") is True and bool(value.get("training_indices"))
        frozen = value.get("fit_performed") is False and bool(value.get("physical_bindings"))
        return {"method": name, "record_type": "fitted" if fitted else "frozen_native_snapshot"} if fitted or frozen else None
    if name.startswith("joint"):
        if (not isinstance(value.get("parameters_by_run"), dict) or not _finite_parameters(value["parameters_by_run"]) or not value.get("training_indices_by_run")
                or not value.get("predictions") or not isinstance(value.get("diagnostics"), dict)
                or not math.isfinite(float(value["diagnostics"].get("objective", float("nan"))))):
            return None
        return {"method": name, "record_type": "fitted"}
    if name == "series_mass":
        if (not isinstance(value.get("parameters"), dict) or not _finite_parameters(value["parameters"])
                or not value.get("runs") or not value.get("by_run")
                or not isinstance(value.get("optimizer_success"), bool) or not value.get("parameter_fit")):
            return None
        return {"method": name, "record_type": "fitted"}
    return None


def _calibrations(value, pointer=""):
    kind = calibration_kind(value)
    if kind:
        yield pointer, value, kind
        return
    # fits.json stores authoritative fit definitions. Do not mine arbitrary
    # settings, requests, predictions or per-frame result JSON for parameters.
    if isinstance(value, dict):
        for key in ("individual", "shared"):
            if isinstance(value.get(key), dict):
                escaped = key.replace("~", "~0").replace("/", "~1")
                child = value[key]
                if calibration_kind(child):
                    yield pointer + "/" + escaped, child, calibration_kind(child)
                elif key == "individual":
                    for run, fit in sorted(child.items()):
                        if calibration_kind(fit):
                            token = run.replace("~", "~0").replace("/", "~1")
                            yield pointer + "/" + escaped + "/" + token, fit, calibration_kind(fit)


def _runs(fit):
    values = fit.get("parameters_by_run") or fit.get("native_calibrations") or fit.get("runs") or {}
    return sorted(map(str, values))


def _saved_context(path, root):
    created, delivery = "", "unknown"
    for folder in (path.parent, *path.parents):
        if not folder.is_relative_to(root):
            break
        for filename in ("delivery.json", "request.json"):
            candidate = folder / filename
            if candidate.is_file():
                try:
                    value = read_json(candidate)
                except (OSError, ValueError):
                    continue
                created = created or str(value.get("created_utc") or value.get("requested_utc") or "")
                if filename == "delivery.json" and delivery == "unknown":
                    delivery = "complete" if value.get("complete") is True else "incomplete"
        if folder == root:
            break
    match = re.search(r"(?<!\d)(20\d{6})(?:T(\d{4,6})Z)?", str(path.relative_to(root)))
    hint = (f"{match[1][:4]}-{match[1][4:6]}-{match[1][6:8]}" if match else "date not recorded")
    return created, hint, delivery


def index_file(path, series_root):
    """Add references to canonical definitions in one existing immutable file."""
    path, root = Path(path).resolve(), Path(series_root).resolve()
    before = sha_file(path)
    value = read_json(path)
    if sha_file(path) != before:
        raise ValueError("Saved calibration changed while it was read")
    created, date_hint, delivery = _saved_context(path, root) if path.is_relative_to(root) else ("", "date not recorded", "unknown")
    indexed = []
    for pointer, fit, kind in _calibrations(value):
        digest = payload_sha(fit)
        identifier = "mass-calibration-" + kind["method"].replace("_", "-") + "-" + digest[:20]
        runs = _runs(fit)
        scope = str(fit.get("scope") or "scope not recorded")
        cohort = ", ".join(run.upper() for run in runs) if len(runs) <= 2 else f"{len(runs)} experiments"
        metadata = {
            "display_name": f"{root.parent.name} · {kind['method'].replace('_', ' ')} · {scope} · {cohort} · {date_hint}",
            "created_utc": str(fit.get("created_utc") or created), "source_date_hint": date_hint,
            "series": str(fit.get("dataset") or root.parent.name), "method": kind["method"],
            "record_type": kind["record_type"], "scope": scope, "runs": runs,
            "saved_delivery_status": delivery, "scientific_status": str(fit.get("validation_scope") or "experimental calibration; no scientific approval inferred"),
            "warnings": list(fit.get("warnings") or []),
            "requires_current_source_geometry_radiometry_preflight": True,
            # Only this existing worker accepts a standalone native fit file.
            "application_adapter": "native_frozen" if kind["method"] == "native" and not pointer else None,
            "mass_eligible": bool(fit.get("mass_eligible", False)),
            "approval_not_inferred_by_indexing": True,
        }
        entry = {"format": FORMAT, "kind": "mass_calibration", "source_identity": {
            "calibration_id": identifier, "version_sha256": digest, "artifact_sha256": before},
            "target": portable_target(path, root, pointer=pointer), "metadata": metadata}
        destination = library_root(root) / (identifier + ".json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open("x", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, allow_nan=False)
        except FileExistsError:
            previous = read_json(destination)
            if previous.get("format") != FORMAT or previous.get("source_identity", {}).get("version_sha256") != digest:
                raise ValueError("Mass-library version identity collision")
            # Equal payloads already have one reference. Preserve its first
            # immutable origin instead of replacing history with a delivery copy.
        indexed.append(str(destination))
    if sha_file(path) != before:
        raise ValueError("Saved calibration changed while it was indexed")
    return indexed


def resolve_reference(entry_path, series_root):
    entry = read_json(entry_path)
    if entry.get("format") != FORMAT or entry.get("kind") != "mass_calibration":
        raise ValueError("Not a mass-calibration library reference")
    path = target_path(entry["target"], series_root)
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    actual = _VERIFIED.get(key)
    if actual is None:
        actual = sha_file(path)
        _VERIFIED[key] = actual
    identity = entry["source_identity"]
    if actual != identity["artifact_sha256"]:
        raise ValueError("Saved mass-calibration artifact SHA differs from its library reference")
    document = _JSON_CACHE.get(key)
    if document is None:
        document = read_json(path)
        _JSON_CACHE[key] = document
    fit = _pointer(document, entry["target"].get("json_pointer", ""))
    if not calibration_kind(fit) or payload_sha(fit) != identity["version_sha256"]:
        raise ValueError("Saved mass-calibration version identity changed")
    return path, fit, entry


def discover_mass_calibrations(series_root):
    """Metadata inventory with explicit unavailable entries; no archive scan."""
    rows = []
    from .series_paths import calibration_read_folders
    seen = set()
    paths = [path for folder in calibration_read_folders(series_root, "mass_calibration_library")
             for path in sorted(folder.glob("mass-calibration-*.json"))]
    for path in paths:
        if path.name.casefold() in seen:
            continue
        seen.add(path.name.casefold())
        entry = {}
        try:
            entry = read_json(path)
            source, fit, entry = resolve_reference(path, series_root)
            rows.append({**entry["metadata"], "entry_path": str(path), "calibration_path": str(source),
                "json_pointer": entry["target"].get("json_pointer", ""), "available": True, "unavailable_reason": ""})
        except (OSError, ValueError, KeyError, TypeError) as error:
            metadata = entry.get("metadata", {})
            rows.append({**metadata, "entry_path": str(path), "available": False,
                "application_adapter": None, "unavailable_reason": str(error)})
    return sorted(rows, key=lambda r: (bool(r.get("created_utc") or r.get("source_date_hint") not in (None, "date not recorded")),
        r.get("created_utc") or r.get("source_date_hint", ""), r.get("display_name", "")), reverse=True)


def migrate_library(series_root, *, roots=None):
    root = Path(series_root).resolve()
    canonical = root / "concentration.toml"
    canonical_before = sha_file(canonical) if canonical.is_file() else None
    from .series_paths import calibration_read_folders, series_folder_path
    defaults = [*(folder / "concentration_mass" for folder in calibration_read_folders(root, "calibration")),
                series_folder_path(root, "results") / "concentration_mass",
                root / "results" / "concentration_mass"]
    scan_roots = list(dict.fromkeys(Path(p).resolve() for p in (defaults if roots is None else roots)))
    files, errors, suppressed_copies = [], [], 0
    for scan in scan_roots:
        if not scan.is_dir():
            continue
        for directory, folders, names in os.walk(scan):
            folders[:] = [name for name in folders if name.casefold() not in _PRUNE and not name.startswith("frame_")]
            folder = Path(directory)
            for name in sorted(set(names) & _FILES):
                path = folder / name
                if name == "calibration.json" and any((ancestor / "fits.json").is_file() for ancestor in path.parents if ancestor.is_relative_to(scan)):
                    suppressed_copies += 1
                    continue
                files.append(path)
    indexed, ignored = [], []
    for path in sorted(set(files)):
        try:
            entries = index_file(path, root)
            indexed.extend(entries)
            if not entries:
                ignored.append(str(path))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append({"path": str(path), "reason": str(error)})
    inventory = discover_mass_calibrations(root)
    canonical_after = sha_file(canonical) if canonical.is_file() else None
    if canonical_after != canonical_before:
        errors.append({"path": str(canonical), "reason": "Canonical concentration settings changed during library indexing; no settings are written by this operation"})
    return {"format": "mass-calibration-library-migration-v1", "series_root": str(root),
        "library_root": str(library_root(root)), "scanned_calibration_files": len(files),
        "indexed_versions": len(inventory), "referenced_entries": len(set(indexed)),
        "suppressed_per_run_delivery_copies": suppressed_copies, "ignored_non_fit_files": ignored, "errors": errors,
        "methods": dict(Counter(row.get("method", "unknown") for row in inventory)),
        "availability": dict(Counter("available" if row["available"] else "unavailable" for row in inventory)),
        "record_types": dict(Counter(row.get("record_type", "unknown") for row in inventory)),
        "application_adapters": dict(Counter(row.get("application_adapter") or "unsupported" for row in inventory)),
        "saved_delivery_status": dict(Counter(row.get("saved_delivery_status", "unknown") for row in inventory)),
        "scientific_approval_inferred": False,
        "canonical_concentration_audit": {"path": str(canonical), "present": canonical_before is not None,
            "sha256_before": canonical_before, "sha256_after": canonical_after,
            "unchanged": canonical_before == canonical_after},
        "profiles_not_models": [p.name for p in (root / "config" / "concentration_profiles").glob("*.toml")],
        "empty": not inventory, "original_artifacts_modified": False, "defaults_modified": False,
        "default_selection_operation": "none", "fit_performed": False, "mass_analysis_performed": False}


def index_saved_result(result_path, results_root):
    """Storage-only save hook; scan only the completed result's saved fit owner."""
    root = series_root_for_results(results_root)
    if root is None:
        return []
    path = Path(result_path).resolve()
    result = read_json(path)
    if result.get("complete") is not True:
        return []
    fits_file = next((folder / "fits.json" for folder in path.parents
        if folder.is_relative_to(Path(results_root).resolve()) and (folder / "fits.json").is_file()), None)
    explicit = result.get("series_calibration")
    calibration_file = Path(explicit) if explicit else path.with_name("calibration.json")
    if not calibration_file.is_absolute():
        calibration_file = path.parent / calibration_file
    target = fits_file or calibration_file
    if not target.is_file():
        return []
    stat = target.stat()
    key = (str(target.resolve()), stat.st_size, stat.st_mtime_ns)
    if key in _AUTO_INDEXED:
        return []
    entries = index_file(target, root)
    _AUTO_INDEXED.add(key)
    return entries
