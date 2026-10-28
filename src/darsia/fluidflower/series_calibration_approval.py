"""Explicit, file-bound series choices; no fitting or execution authority.

Only the caller's Use/Unlock action writes the existing series manifest. A
legacy choice without this optional record remains a legacy choice. Approval
does not replace the pipeline's original model or calibration admission.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import getpass
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re
import tomllib

FORMAT = "darsia-series-calibration-approval-v1"
MAX_METADATA_BYTES = 16 * 1024 * 1024


def _json(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        raw = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("Calibration metadata exceeds the supported size.")
    document = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(document, dict):
        raise ValueError("The selected calibration metadata must be an object.")
    return document, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                          size_bytes=len(raw))


def _path(root, reference):
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("Select an existing model and mass calibration first.")
    path = Path(reference)
    joined = path if path.is_absolute() or PureWindowsPath(str(path)).is_absolute() else Path(root) / path
    from .storage_paths import resolve_storage_path
    return resolve_storage_path(joined, series_roots=(Path(root).absolute(),)).resolve(strict=True)


def _binding_at_current_locations(root, binding):
    """Compare saved identities at their explicitly relocated physical locations."""
    from .storage_paths import resolve_storage_path
    current = copy.deepcopy(binding)
    phase = current.get("phase_model") or {}
    for section, key in ((current.get("calibration") or {}, "path"), (phase, "root"),
                         (phase.get("pointer") or {}, "path")):
        if key in section:
            section[key] = str(resolve_storage_path(section[key],
                series_roots=(Path(root).absolute(),)).resolve(strict=True))
    return current


def same_selection_binding(root, current, recorded):
    """Keep exact source identities while accepting explicit storage relocation."""
    return _binding_at_current_locations(root, current) == _binding_at_current_locations(root, recorded)


def selection_binding(root, selection):
    """Bind only the explicit saved phase model and supported frozen chemistry.

    Original library resolution checks a pointer's actual model identity. A
    direct model folder uses the same original identity reader. No history or
    latest-result discovery, source reinterpretation, or calibration rewrite.
    """
    from .phase_model_library import library_root, resolve_reference, source_identity
    from .pipeline.local_overflow_refit import FROZEN_FORMAT as OVERFLOW_FORMAT
    from .pipeline.refit_runner import FREEZE_FORMAT

    if selection.active != "phase_model":
        raise ValueError("This approval is for the phase-map mass calibration.")
    calibration, descriptor = _json(_path(root, selection.mass_calibration))
    if (calibration.get("format") not in (FREEZE_FORMAT, OVERFLOW_FORMAT)
            or calibration.get("frozen") is not True):
        raise ValueError("Choose a completed frozen mass calibration.")
    if calibration["format"] == OVERFLOW_FORMAT:
        # Every field phase two reads must be present in an overflow
        # (variant F) calibration too.
        from .production_selection import production_calibration_fields
        production_calibration_fields(calibration)
    if calibration.get("chemistry_model") != "titration":
        raise ValueError(
            "The calibration must explicitly record Titration. Other chemistry "
            "models are not available yet; legacy files are not relabelled."
        )
    parameters = calibration.get("parameters")
    if not isinstance(parameters, dict) or any(
        type(parameters.get(key)) not in (int, float)
        or not math.isfinite(parameters[key]) for key in ("gain", "beta")
    ):
        raise ValueError("The frozen calibration has no finite gain and beta.")
    reference = selection.model_version
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("Select an existing phase-model version.")
    candidate = Path(reference)
    if candidate.name == reference and not any(char in reference for char in "/\\"):
        candidate = library_root(root) / (re.sub(r"[^A-Za-z0-9_.-]", "_", reference) + ".json")
    else:
        candidate = _path(root, reference)
    pointer = None
    if candidate.is_file():
        _document, pointer = _json(candidate)
        model_root = resolve_reference(candidate, series_root=root)
    else:
        model_root = candidate.resolve(strict=True)
    identity = source_identity(model_root)
    if identity.get("complete_heads") is not True:
        raise ValueError("The phase model has no complete saved model identity.")
    result = dict(
        active=selection.active, model_version=reference,
        mass_calibration=selection.mass_calibration,
        chemistry_model="titration", calibration=descriptor,
        phase_model=dict(root=str(model_root), identity=identity),
    )
    if pointer is not None:
        result["phase_model"]["pointer"] = pointer
    return result


def _record(document):
    method = document.get("method", {})
    if not isinstance(method, dict):
        raise ValueError("The method section must be a TOML table.")
    record = method.get("approval")
    if record is None:
        return None
    if (not isinstance(record, dict) or record.get("format") != FORMAT
            or type(record.get("locked")) is not bool
            or not isinstance(record.get("binding"), dict)
            or not isinstance(record.get("events"), list)
            or not record["events"]):
        raise ValueError("The saved calibration approval is malformed.")
    for event in record["events"]:
        if (not isinstance(event, dict) or event.get("action") not in {"approve", "unlock"}
                or not isinstance(event.get("actor"), str) or not event["actor"].strip()
                or not isinstance(event.get("utc"), str) or not event["utc"].strip()
                or not isinstance(event.get("reason"), str) or not event["reason"].strip()):
            raise ValueError("The saved approval history is malformed.")
    approvals = [event for event in record["events"] if event["action"] == "approve"]
    if not approvals or approvals[-1].get("binding") != record["binding"]:
        raise ValueError("The saved approval does not match its recorded files.")
    binding = record["binding"]
    phase = binding.get("phase_model", {})
    identity = phase.get("identity", {}) if isinstance(phase, dict) else {}
    if (binding.get("active") != "phase_model" or binding.get("chemistry_model") != "titration"
            or not isinstance(identity, dict) or not isinstance(identity.get("model_id"), str)
            or not identity["model_id"].strip()
            or not isinstance(binding.get("mass_calibration"), str)):
        raise ValueError("The saved calibration/model binding is malformed.")
    if (record["events"][-1]["action"] == "approve") != record["locked"]:
        raise ValueError("The saved approval lock and history disagree.")
    return record


def require_unlocked(document):
    """An ordinary reference save cannot bypass an explicit locked choice."""
    record = _record(document)
    if record is not None and record["locked"]:
        raise ValueError("This choice is locked. Unlock it with a reason before changing it.")


def read_approval(root, *, verify=True):
    """Read the optional lock and, while locked, recheck its actual file binding."""
    from .series_method import _selection

    document = tomllib.loads((Path(root) / "series.toml").read_text(encoding="utf-8-sig"))
    record = _record(document)
    if record is not None and record["locked"] and verify:
        if not same_selection_binding(root, selection_binding(root, _selection(document)), record["binding"]):
            raise ValueError("The approved calibration or model changed. Unlock and review the choice.")
    return copy.deepcopy(record)


def _loaded(root, expected_bytes):
    from .series_method import ManifestChangedError

    path = Path(root) / "series.toml"
    original = path.read_bytes()
    if expected_bytes is not None and original != expected_bytes:
        raise ManifestChangedError("The series changed since this choice was shown.")
    document = tomllib.loads(original.decode("utf-8-sig"))
    return path, original, document


def _event(action, reason):
    return dict(action=action, actor=getpass.getuser(),
                utc=datetime.now(timezone.utc).isoformat(), reason=reason)


def approve_series_method(root, selection, *, expected_bytes=None, expected_binding=None):
    """One explicit Use action binds the chosen files and locks this section."""
    from .series_method import _selection, _write_manifest, missing_method_references

    path, original, document = _loaded(root, expected_bytes)
    require_unlocked(document)
    if missing_method_references(selection):
        raise ValueError("Choose the phase model and mass calibration first.")
    binding = selection_binding(root, selection)
    if expected_binding is not None and binding != expected_binding:
        raise ValueError("The selected files changed since they were shown. Check the choice again.")
    updated = copy.deepcopy(document)
    method = updated.setdefault("method", {})
    previous = _record(document)
    method["active"] = "phase_model"
    references = method.setdefault("phase_model", {})
    if not isinstance(references, dict):
        raise ValueError("method.phase_model must be a TOML table.")
    references.update(model_version=selection.model_version,
                      mass_calibration=selection.mass_calibration)
    events = copy.deepcopy(previous["events"]) if previous else []
    event = _event("approve", "Use this mass calibration and Titration model for this series.")
    event["binding"] = copy.deepcopy(binding)
    events.append(event)
    method["approval"] = dict(format=FORMAT, locked=True, binding=binding, events=events)
    # Recheck immediately before the original manifest CAS, including pointer,
    # actual model metadata, calibration bytes and the explicit chemistry field.
    if selection_binding(root, selection) != binding:
        raise ValueError("The selected calibration or model changed during approval.")
    parsed = _write_manifest(path, original, updated)
    return _selection(parsed)


def unlock_series_method(root, reason, *, expected_bytes=None):
    """Unlock without replacing the last approved files or the saved references."""
    from .series_method import _selection, _write_manifest

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Give a reason before unlocking this choice.")
    path, original, document = _loaded(root, expected_bytes)
    record = _record(document)
    if record is None or not record["locked"]:
        raise ValueError("There is no locked calibration choice to unlock.")
    updated = copy.deepcopy(document)
    approved = updated["method"]["approval"]
    approved["locked"] = False
    approved["events"].append(_event("unlock", reason.strip()))
    parsed = _write_manifest(path, original, updated)
    return _selection(parsed)
