"""Native model-list visibility only; model and validation files stay in place.

The original library remains authoritative. This optional browser overlay never
changes admission, training, references, scientific artifacts or saved choices.
Callers supply roots in active review/training and retain the original list if a
read fails. There is deliberately no delete or move API.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

from .phase_model_library import library_root, resolve_reference, source_identity
from .review_storage import resolve_review_path
from .series_method import read_series_method

FORMAT = "native-phase-model-browser-archive-v1"
_LIMIT = 4 * 1024 * 1024


def archive_path(series_root):
    # Not a top-level *.json library pointer: discovery must never ingest this.
    return library_root(series_root) / "ui" / "browser_archive.json"


def _root(value):
    return resolve_review_path(value).resolve()


def _same_root(left, right):
    left, right = _root(left), _root(right)
    if os.path.normcase(str(left)) == os.path.normcase(str(right)):
        return True
    try:
        return left.samefile(right)
    except OSError:
        return False


def _path_key(value):
    """Lexical comparison only: no resolve, stat, alias discovery or existence."""
    return os.path.normcase(os.path.normpath(os.fspath(value)))


def _validate_overlay(value, *, lexical=False):
    if (not isinstance(value, dict) or value.get("format") != FORMAT
            or not isinstance(value.get("series_root"), str)
            or not Path(value["series_root"]).is_absolute()
            or not isinstance(value.get("models"), list)):
        raise ValueError("The model archive is unreadable or has another format")
    roots = []
    for row in value["models"]:
        if (not isinstance(row, dict) or not isinstance(row.get("root"), str)
                or not Path(row["root"]).is_absolute()
                or not isinstance(row.get("source_identity"), dict)
                or type(row.get("archived")) is not bool
                or not isinstance(row.get("history"), list)):
            raise ValueError("The model archive contains a malformed entry")
        if any((_path_key(row["root"]) == _path_key(old) if lexical
                else _same_root(row["root"], old)) for old in roots):
            raise ValueError("The model archive contains duplicate entries")
        roots.append(row["root"])
    return value


def _load(path, *, lexical=False):
    if not path.exists():
        return None, dict(format=FORMAT, models=[])
    if path.stat().st_size > _LIMIT:
        raise ValueError("The model archive exceeds the metadata limit")
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8-sig"))
    return raw, _validate_overlay(value, lexical=lexical)


def read_archive_snapshot(series_root):
    """Read the small overlay on explicit bind/refresh, never inspect models.

    The original library resolver supplies its configured location. Validation
    compares recorded paths lexically; it does not resolve or stat model roots,
    read the saved method choice, or rehash sources. The caller separately
    supplies its cached original ``read_series_method`` result to filtering.
    """
    series_root = Path(series_root)
    if not series_root.is_absolute():
        raise ValueError("An absolute series root is required for the archive snapshot")
    _raw, document = _load(archive_path(series_root), lexical=True)
    owner = document.get("series_root")
    if owner is not None and _path_key(owner) != _path_key(series_root):
        raise ValueError("The model archive belongs to another series")
    snapshot = deepcopy(document)
    snapshot["series_root"] = str(series_root)
    return snapshot


def _snapshot_protected(snapshot, saved, entry, root, protected_keys):
    if root in protected_keys:
        return True
    reference = saved.model_version.strip()
    if not reference:
        return False
    if reference == entry.get("model_id"):
        return True
    candidate = Path(reference)
    if candidate.is_absolute() or any(char in reference for char in ("/", "\\")) or candidate.suffix == ".json":
        candidate = candidate if candidate.is_absolute() else Path(snapshot["series_root"]) / candidate
        key = _path_key(candidate)
        if key == root:
            return True
        pointer = entry.get("library_reference")
        if pointer and key == _path_key(pointer):
            return True
    return False


def filter_snapshot(snapshot, entries, saved_selection, protected_roots=()):
    """Pure UI filtering of cached metadata; never access the filesystem.

    Only an exact recorded root and matching cached source identity can hide a
    model. Missing/stale identity or malformed metadata raises, so callers keep
    the original list visible and show the error. Protected models remain
    visible even when their catalogue identity needs refreshing. Filesystem
    aliases are deliberately not inferred from cached strings.
    """
    _validate_overlay(snapshot, lexical=True)
    if saved_selection is None or getattr(saved_selection, "error", ""):
        raise ValueError("The saved series choice cannot be checked from cached metadata")
    reference = getattr(saved_selection, "model_version", None)
    if not isinstance(reference, str):
        raise ValueError("The cached series choice has no readable model reference")
    if getattr(saved_selection, "active", "") == "phase_model" and not reference.strip():
        raise ValueError("The saved phase-model choice has no model reference")
    protected_keys = {_path_key(root) for root in protected_roots}
    archived = {_path_key(row["root"]): row for row in snapshot["models"] if row["archived"]}
    visible = []
    for entry in entries:
        root = entry.get("root")
        key = _path_key(root) if root else None
        row = archived.get(key)
        if row is None or _snapshot_protected(snapshot, saved_selection, entry, key, protected_keys):
            visible.append(entry)
            continue
        identity = entry.get("library_source_identity") or entry.get("source_identity")
        if (entry.get("library_available") is False or not isinstance(identity, dict)
                or identity != row["source_identity"]
                or not identity.get("model_id") or entry.get("model_id") != identity.get("model_id")):
            raise ValueError("An archived model identity is unknown or changed; refresh before hiding any models")
    return visible


def _series_state(series_root):
    manifest = Path(series_root) / "series.toml"
    original = manifest.read_bytes()
    saved = read_series_method(series_root)
    if saved.error:
        raise ValueError("The saved series choice cannot be checked: " + saved.error)
    if saved.active == "phase_model" and not saved.model_version.strip():
        raise ValueError("The saved phase-model choice has no model reference")
    if manifest.read_bytes() != original:
        raise RuntimeError("The series choice changed while it was being read")
    return original, saved


def _checked_entry(entry, *, require_binding=True):
    if not isinstance(entry, dict) or not entry.get("root"):
        raise ValueError("Select a saved model with an exact source identity")
    if entry.get("library_available") is False:
        raise ValueError("An unavailable model reference cannot be archived")
    root = _root(entry["root"])
    expected = entry.get("library_source_identity") or entry.get("source_identity")
    if require_binding and not isinstance(expected, dict):
        raise ValueError("Import or reindex this model before changing its archive visibility")
    actual = source_identity(root)
    if (expected is not None and expected != actual
            or entry.get("model_id") != actual.get("model_id")):
        raise ValueError("The selected model identity changed; refresh the model list")
    pointer = entry.get("library_reference")
    if pointer and not _same_root(resolve_reference(pointer), root):
        raise ValueError("The selected library reference changed its model target")
    return root, actual


def _protected(series_root, saved, entry, root, protected_roots):
    if any(_same_root(root, active) for active in protected_roots):
        return "The model is in an active review or training session"
    reference = saved.model_version.strip()
    if not reference:
        return ""
    if reference == entry.get("model_id"):
        return "The model is saved for this series"
    # Saved references are original IDs, model folders or library pointers.
    candidate = Path(reference)
    if candidate.is_absolute() or any(char in reference for char in ("/", "\\")) or candidate.suffix == ".json":
        candidate = _root(candidate if candidate.is_absolute() else Path(series_root) / candidate)
        if candidate.suffix == ".json":
            candidate = resolve_reference(candidate, series_root=series_root)
        if _same_root(candidate, root):
            return "The model is saved for this series"
    return ""


def _matching_row(document, root):
    return next((row for row in document["models"] if _same_root(row["root"], root)), None)


def _require_series(document, series_root):
    owner = document.get("series_root")
    if owner is not None and not _same_root(owner, series_root):
        raise ValueError("The model archive belongs to another series")


def _change(series_root, entry, actor, reason, protected_roots, *, archived):
    actor, reason = str(actor or "").strip(), str(reason or "").strip()
    if not actor or not reason:
        raise ValueError("Record the person and reason for this visibility change")
    series_root = _root(series_root)
    protected_roots = tuple(protected_roots)
    original_manifest, saved = _series_state(series_root)
    root, identity = _checked_entry(entry)
    if archived and (blocker := _protected(series_root, saved, entry, root, protected_roots)):
        raise ValueError(blocker)
    path = archive_path(series_root)
    original, document = _load(path)
    _require_series(document, series_root)
    updated = deepcopy(document)
    updated["series_root"] = str(series_root)
    row = _matching_row(updated, root)
    if row is not None and row["source_identity"] != identity:
        raise ValueError("The archived model identity changed; its history was preserved")
    if row is None:
        if not archived:
            return path
        row = dict(root=str(root), source_identity=identity, archived=False, history=[])
        updated["models"].append(row)
    if row["archived"] is archived:
        return path
    row["archived"] = archived
    row["history"].append(dict(
        action="archived from model lists" if archived else "restored to model lists",
        person=actor, reason=reason, recorded_utc=datetime.now(timezone.utc).isoformat()))
    raw = (json.dumps(updated, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if len(raw) > _LIMIT:
        raise ValueError("The model archive exceeds the metadata limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
        if (Path(series_root) / "series.toml").read_bytes() != original_manifest:
            raise RuntimeError("The saved series choice changed; no visibility change was saved")
        if _checked_entry(entry)[1] != identity:
            raise ValueError("The selected model identity changed while saving")
        current = path.read_bytes() if path.exists() else None
        if current != original:
            raise RuntimeError("The model archive changed while saving; the external edit was preserved")
        os.replace(temporary, path)
    finally:
        # Only this action's own unfinished temporary overlay, never user data.
        temporary.unlink(missing_ok=True)
    return path


def archive_model(series_root, entry, actor, reason, protected_roots=()):
    """Hide one verified unused model; retain all files and original pointers."""
    return _change(series_root, entry, actor, reason, protected_roots, archived=True)


def restore_model(series_root, entry, actor, reason, protected_roots=()):
    """Show the same exact model again, retaining the complete action history."""
    return _change(series_root, entry, actor, reason, protected_roots, archived=False)


def filter_entries(series_root, entries, protected_roots=()):
    """Hide exact archived versions; saved and active models always stay visible.

    On malformed/stale metadata this raises: the caller must retain its original
    entries and show the error. Neither passive filtering nor restoring rewrites
    a saved series choice or grants scientific admission.
    """
    series_root = _root(series_root)
    protected_roots = tuple(protected_roots)
    _manifest, saved = _series_state(series_root)
    _raw, document = _load(archive_path(series_root))
    _require_series(document, series_root)
    visible = []
    for entry in entries:
        root = entry.get("root")
        row = _matching_row(document, root) if root else None
        if row is None or not row["archived"]:
            visible.append(entry)
            continue
        checked_root, identity = _checked_entry(entry, require_binding=False)
        if row["source_identity"] != identity:
            raise ValueError("An archived model identity changed; refresh before hiding any models")
        if _protected(series_root, saved, entry, checked_root, protected_roots):
            visible.append(entry)
    if (series_root / "series.toml").read_bytes() != _manifest:
        raise RuntimeError("The saved series choice changed; refresh before hiding any models")
    return visible
