"""Native UI names in existing phase-library pointers, never model artifacts.

Read helpers are passive. Only explicit import/reindex and a real completion
callback call the pointer writer. The retained index_model is not modified.
"""

from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from .phase_model_library import FORMAT, index_model, library_root, source_identity

NAME_FORMAT = "phase-model-ui-name-v1"
INDEX_NAME_FORMAT = "phase-model-index-ui-name-v1"


def _read(path, limit=262144):
    path = Path(path)
    if path.stat().st_size > limit:
        raise ValueError("Display-name metadata is too large: " + str(path))
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Display-name metadata must be an object")
    return raw, value


def _request_sha(identity):
    hashes = identity.get("metadata_sha256")
    return hashes.get("training_request.json") if isinstance(hashes, dict) else None


def _valid_name(value, identity, *, index=False):
    return (
        isinstance(value, dict)
        and value.get("format") == (INDEX_NAME_FORMAT if index else NAME_FORMAT)
        and isinstance(identity.get("model_id"), str)
        and bool(identity["model_id"].strip())
        and value.get("model_id") == identity.get("model_id")
        and isinstance(value.get("training_request_sha256"), str)
        and re.fullmatch(r"[a-f0-9]{64}", value["training_request_sha256"]) is not None
        and value["training_request_sha256"] == _request_sha(identity)
        and isinstance(value.get("display_name"), str)
        and bool(value["display_name"].strip())
    )


def saved_ui_name(root, model_id=None):
    """Read the existing immutable request-bound sidecar, without any writes."""
    root = Path(root)
    try:
        _raw, value = _read(root / "model_ui_name.json", 65536)
        request_raw, request = _read(root / "training_request.json", 2 * 1024 * 1024)
        identity = {
            "model_id": model_id or request.get("model_id"),
            "metadata_sha256": {
                "training_request.json": hashlib.sha256(request_raw).hexdigest()
            },
        }
        if request.get("model_id") != identity["model_id"] or not _valid_name(
            value, identity
        ):
            return None
        return value
    except (OSError, ValueError, TypeError):
        return None


def _pointer_binding(document, expected_identity=None):
    if document.get("format") != FORMAT or document.get("kind") != "phase":
        raise ValueError("Expected an existing phase-model library pointer")
    identity = document.get("source_identity") or {}
    if not isinstance(identity, dict):
        raise ValueError("Model pointer source identity must be an object")
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("Model pointer identity changed")
    metadata = document.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("Model pointer metadata must be an object")
    value = metadata.get("ui_display_name")
    if (
        _valid_name(value, identity, index=True)
        and metadata.get("display_name") == value["display_name"]
    ):
        return value
    return None


def write_index_ui_name(pointer, value, *, expected_bytes=None):
    """CAS-update display metadata only; preserve every unrelated pointer field."""
    pointer = Path(pointer)
    original, document = _read(pointer)
    if expected_bytes is not None and original != expected_bytes:
        raise RuntimeError(
            "Model pointer changed since it was loaded; display name was not saved"
        )
    _pointer_binding(document)
    identity = document.get("source_identity") or {}
    if not _valid_name(value, identity):
        raise ValueError(
            "Display name does not match this model ID and training-request SHA"
        )
    existing = (document.get("metadata") or {}).get("ui_display_name")
    bound = dict(value, format=INDEX_NAME_FORMAT)
    if existing is not None and existing != bound:
        raise ValueError(
            "Another bound display name is already present; it was preserved"
        )
    updated = deepcopy(document)
    metadata = updated.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Model pointer metadata must be an object")
    metadata["display_name"] = value["display_name"]
    metadata["ui_display_name"] = bound
    if updated == document:
        return pointer
    temporary = pointer.with_name(
        "." + pointer.name + ".ui-name-" + uuid.uuid4().hex + ".tmp"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(
                (
                    json.dumps(updated, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n"
                ).encode("utf-8")
            )
        if pointer.read_bytes() != original:
            raise RuntimeError(
                "Model pointer changed while saving; external edit was preserved"
            )
        os.replace(temporary, pointer)
    finally:
        temporary.unlink(missing_ok=True)
    return pointer


def index_model_with_ui_name(
    series_root, model_root, *, indexer=None, expected_model_id=None
):
    """Explicit native reindex/import: retain the producer, then reapply UI name.

    This does not protect arbitrary external calls to the original indexer.
    A valid original sidecar remains a read-only display fallback after one.
    """
    root = Path(model_root)
    identity = source_identity(root)
    if expected_model_id is not None and identity["model_id"] != expected_model_id:
        raise ValueError("Completed model ID differs; display metadata was not indexed")
    value = saved_ui_name(root, identity.get("model_id"))
    canonical = library_root(series_root) / (
        re.sub(r"[^A-Za-z0-9_.-]", "_", identity["model_id"]) + ".json"
    )
    if value is None and canonical.is_file():
        _old_raw, old = _read(canonical)
        # Reapply only the same request-bound version, never a colliding ID.
        previous = _pointer_binding(old)
        if previous and previous.get("training_request_sha256") == _request_sha(
            identity
        ):
            value = dict(previous, format=NAME_FORMAT)
    path = (indexer or index_model)(series_root, root)
    if value is not None:
        raw, _doc = _read(path)
        write_index_ui_name(path, value, expected_bytes=raw)
    return path


def model_ui_display_name(entry):
    """Prefer this entry's bound index name; otherwise its original sidecar."""
    identity = entry.get("library_source_identity")
    reference = entry.get("library_reference")
    if reference and isinstance(identity, dict):
        try:
            _raw, document = _read(reference)
            bound = _pointer_binding(document, identity)
            if bound is not None and bound["model_id"] == entry.get("model_id"):
                return bound["display_name"]
        except (OSError, ValueError, TypeError):
            pass
    root = entry.get("root")
    value = saved_ui_name(root, entry.get("model_id")) if root else None
    return value["display_name"] if value else ""


def model_reference_label(series_root, reference):
    """Read only an explicit pointer, ID, or model folder; never scan history."""
    fallback = str(reference or "not referenced")
    if not series_root or not reference:
        return fallback
    reference = str(reference)
    candidate = Path(reference)
    library = library_root(series_root).resolve()
    pointer = None
    if candidate.suffix == ".json" and candidate.is_absolute():
        if candidate.resolve().parent == library:
            pointer = candidate
    elif candidate.name == reference and not any(char in reference for char in "/\\"):
        pointer = library / (re.sub(r"[^A-Za-z0-9_.-]", "_", reference) + ".json")
    try:
        if pointer is not None and pointer.is_file():
            _raw, document = _read(pointer)
            bound = _pointer_binding(document)
            if bound and (candidate.is_absolute() or bound["model_id"] == reference):
                return bound["display_name"] + " \u00b7 " + bound["model_id"]
        if candidate.is_absolute() and candidate.is_dir():
            value = saved_ui_name(candidate)
            if value:
                return value["display_name"] + " \u00b7 " + value["model_id"]
    except (OSError, ValueError, TypeError):
        pass
    return fallback
