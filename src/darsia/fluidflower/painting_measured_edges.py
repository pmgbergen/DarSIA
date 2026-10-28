"""Read saved measured wall and bottom lines for a passive painting overlay.

Only the exact camera-U photograph is supported. This reader never opens an
image, constructs a top edge, fits geometry, or changes an adoption decision.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re


MAX_JSON_BYTES = 16 * 1024 * 1024
GEOMETRY_FORMAT = "darsia-measured-run-geometry-v1"
EDGE_NAMES = ("left", "bottom", "right")


@dataclass(frozen=True)
class SavedMeasuredEdges:
    """Saved ``a*x + b*y + c = 0`` lines, in unchanged camera-U pixels."""

    run_id: str
    shape_hw: tuple[int, int]
    lines: tuple[tuple[str, tuple[float, float, float]], ...]
    provenance: dict


def _object(value, label):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _run(value):
    return _text(value, "Run id").strip().casefold()


def _run_fields(row, expected, label, *, required=False):
    fields = [row[key] for key in ("run", "run_id") if key in row]
    if required and not fields:
        raise ValueError(f"{label} has no run id")
    if any(_run(value) != expected for value in fields):
        raise ValueError(f"{label} names another run")


def _shape(value, label):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(v) is not int or v <= 0 for v in value)):
        raise ValueError(f"{label} must contain two positive integers")
    return tuple(value)


def _descriptor(value, label):
    value = _object(value, label)
    path = _text(value.get("path"), f"{label} path")
    digest = value.get("sha256")
    size = value.get("size_bytes")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{label} needs an exact lowercase SHA-256")
    if type(size) is not int or size <= 0:
        raise ValueError(f"{label} needs a positive byte size")
    return dict(path=path, sha256=digest, size_bytes=size)


def _path_key(value):
    # Photograph locators are compared passively: no stat/resolve on RAW paths.
    text = str(value)
    if "\\" in text or PureWindowsPath(text).is_absolute():
        return str(PureWindowsPath(text)).casefold()
    return str(Path(text))


def _same_descriptor(left, right, label):
    left, right = _descriptor(left, label), _descriptor(right, label)
    if (left["sha256"] != right["sha256"]
            or left["size_bytes"] != right["size_bytes"]
            or _path_key(left["path"]) != _path_key(right["path"])):
        raise ValueError(f"{label} does not match the exact saved binding")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON contains a duplicate object key")
        result[key] = value
    return result


def _read_document(value, *, base, label, cache, unpinned=False):
    if unpinned and isinstance(value, (str, Path)):
        expected = None
        path = Path(value)
    else:
        expected = _descriptor(value, label)
        path = Path(expected["path"])
    path = path if path.is_absolute() else base / path
    key = (str(path.absolute()), None if expected is None else
           (expected["sha256"], expected["size_bytes"]))
    if key in cache:
        return cache[key]
    if expected is not None and expected["size_bytes"] > MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds the bounded JSON size")
    with path.open("rb") as stream:
        content = stream.read(MAX_JSON_BYTES + 1)
    if len(content) > MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds the bounded JSON size")
    descriptor = dict(path=str(path.absolute()), sha256=hashlib.sha256(content).hexdigest(),
                      size_bytes=len(content))
    if expected is not None and any(descriptor[key] != expected[key]
                                    for key in ("sha256", "size_bytes")):
        raise ValueError(f"{label} changed: SHA-256 or byte size differs")
    try:
        document = json.loads(content.decode("utf-8-sig"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    document = _object(document, label)
    cache[key] = document, descriptor, path.parent
    return cache[key]


def _selected_run(rows, run, label, *, mapping=False):
    selected, seen = None, set()
    if mapping and isinstance(rows, Mapping):
        entries = rows.items()
    elif isinstance(rows, list):
        entries = ((None, row) for row in rows)
    else:
        raise ValueError(f"{label} has no run entries")
    for key, row in entries:
        row = _object(row, f"{label} run entry")
        identity = _run(key if key is not None else row.get("run_id", row.get("run")))
        _run_fields(row, identity, f"{label} entry", required=key is None)
        if identity in seen:
            raise ValueError(f"{label} has a duplicate run")
        seen.add(identity)
        if identity == run:
            selected = row
    if selected is None:
        raise ValueError(f"{label} has no matching run")
    return selected


def _pointer(document, pointer):
    pointer = _text(pointer, "Source pointer")
    if not pointer.startswith("/"):
        raise ValueError("Source pointer must be an exact JSON pointer")
    value = document
    for part in pointer[1:].split("/"):
        if re.search(r"~(?![01])", part):
            raise ValueError("Source pointer has an invalid escape")
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, Mapping):
            if part not in value:
                raise ValueError("Source pointer does not exist")
            value = value[part]
        elif isinstance(value, list) and re.fullmatch(r"0|[1-9][0-9]*", part):
            index = int(part)
            if index >= len(value):
                raise ValueError("Source pointer does not exist")
            value = value[index]
        else:
            raise ValueError("Source pointer does not exist")
    return value


def _line(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(type(v) not in (int, float) for v in value)):
        raise ValueError("A saved camera line must have three numeric coefficients")
    try:
        coefficients = tuple(float(v) for v in value)
    except OverflowError as error:
        raise ValueError("A saved camera line must be finite") from error
    if not all(math.isfinite(v) for v in coefficients) or coefficients[:2] == (0., 0.):
        raise ValueError("A saved camera line must be finite and nondegenerate")
    return coefficients


def read_saved_measured_edges(index, run, *, photograph_binding, camera_u_binding,
                             shape_hw, expected_confirmation):
    """Read exactly confirmed saved L/B/R lines; never infer a crop or top.

    ``index`` is a JSON path or an exact path/SHA-256/size descriptor. All
    subsequent JSON inputs must be pinned descriptors. ``photograph_binding``
    is the caller's verified photo descriptor; its image is never opened here.
    ``camera_u_binding`` additionally needs ``profile_id`` and ``camera_wh``.
    ``expected_confirmation`` is the caller's explicit external schema value,
    compared exactly, rather than a display setting or inferred approval.
    Only an identical model photograph is supported; registrations to a
    different photograph and corrected-baseline grids are refused.
    """
    run = _run(run)
    shape = _shape(shape_hw, "Painting shape")
    confirmation = _text(expected_confirmation, "Expected confirmation")
    photograph = _descriptor(photograph_binding, "Painting photograph")
    camera = _object(camera_u_binding, "Camera-U binding")
    _descriptor(camera, "Camera-U binding")
    profile_id = _text(camera.get("profile_id"), "Camera-U profile id")
    if _shape(camera.get("camera_wh"), "Camera-U size") != shape[::-1]:
        raise ValueError("Camera-U binding and painting have different grids")
    cache = {}
    document, index_ref, index_folder = _read_document(
        index, base=Path.cwd(), label="Geometry index", cache=cache, unpinned=True)
    if document.get("format") != GEOMETRY_FORMAT or document.get("grid") != "camera_u":
        raise ValueError("Geometry index must describe the camera-U grid")
    entry = _selected_run(document.get("runs"), run, "Geometry index", mapping=True)
    model, model_ref, model_folder = _read_document(entry.get("reference_model"),
        base=index_folder, label="Reference model", cache=cache)
    if model.get("format") != GEOMETRY_FORMAT or model.get("grid") != "camera_u":
        raise ValueError("Reference model must describe the camera-U grid")
    _run_fields(model, run, "Reference model", required=True)
    if _shape(model.get("shape_hw"), "Reference model shape") != shape:
        raise ValueError("Reference model and painting have different grids")
    _same_descriptor(model.get("reference_image"), photograph, "Reference photograph")
    _same_descriptor(model.get("model_photograph"), photograph, "Model photograph")
    saved_camera = _object(model.get("camera_profile"), "Saved camera-U binding")
    _same_descriptor(saved_camera, camera, "Camera-U profile")
    if (saved_camera.get("profile_id") != profile_id
            or _shape(saved_camera.get("camera_wh"), "Saved camera-U size") != shape[::-1]):
        raise ValueError("Saved camera-U profile does not match the painting grid")
    profile, profile_ref, _ = _read_document(saved_camera, base=model_folder,
        label="Camera-U profile", cache=cache)
    registration = _object(profile.get("registration", profile), "Camera registration")
    common = _object(registration.get("common"), "Camera registration settings")
    sensor = _object(common.get("sensor"), "Camera sensor settings")
    declared = _object(common.get("profiles"), "Declared camera profiles")
    if (profile_id not in declared
            or _shape(sensor.get("camera_wh"), "Declared camera-U size") != shape[::-1]):
        raise ValueError("Bound camera profile does not declare the painting grid")
    source = _object(profile.get("source"), "Camera profile source")
    assignments = _object(source.get("profiles"), "Run camera profiles")
    normalized = {}
    for key, value in assignments.items():
        identity = _run(key)
        if identity in normalized:
            raise ValueError("Camera profile has a duplicate run assignment")
        normalized[identity] = value
    if normalized.get(run) != profile_id:
        raise ValueError("Bound camera profile names another profile for this run")
    ledger, ledger_ref, ledger_folder = _read_document(model.get("current_edges"),
        base=model_folder, label="Measured edge ledger", cache=cache)
    row = _selected_run(ledger.get("rows"), run, "Measured edge ledger")
    if row.get("all_three_B_confirmed") is not True:
        raise ValueError("The saved measured edges are not all confirmed")
    edges = _object(row.get("edges"), "Measured edges")
    lines, records, source_refs = [], {}, {}
    for name in EDGE_NAMES:
        edge = _object(edges.get(name), f"Measured {name} edge")
        if edge.get("support") != "full" or edge.get("physical_B_confirmation") != confirmation:
            raise ValueError(f"The measured {name} edge is not explicitly confirmed with full support")
        candidate = _text(edge.get("candidate_id"), "Measured edge candidate")
        saved_line = _line(edge.get("camera_line_abc"))
        source, source_ref, _ = _read_document(edge.get("source"), base=ledger_folder,
            label=f"Measured {name} source", cache=cache)
        _run_fields(source, run, f"Measured {name} source")
        if _line(_pointer(source, edge.get("pointer"))) != saved_line:
            raise ValueError(f"The measured {name} source line differs from the ledger")
        source_refs[name] = {"line": source_ref}
        if "candidate_ID_pointer" in edge or "candidate_ID_source" in edge:
            candidate_source, candidate_ref, _ = _read_document(edge.get("candidate_ID_source"),
                base=ledger_folder, label=f"Measured {name} candidate source", cache=cache)
            _run_fields(candidate_source, run, f"Measured {name} candidate source")
            selected = _pointer(candidate_source, edge.get("candidate_ID_pointer"))
            if not isinstance(selected, Mapping) or selected.get(name) != candidate:
                raise ValueError(f"The measured {name} candidate differs from its source pointer")
            source_refs[name]["candidate"] = candidate_ref
        lines.append((name, saved_line))
        records[name] = deepcopy(dict(edge))
    provenance = dict(index=index_ref, reference_model=model_ref, camera_profile=profile_ref,
                      photograph=deepcopy(photograph), measured_edge_ledger=ledger_ref,
                      sources=source_refs, edge_records=records, grid="camera_u",
                      read_only=True, adoption_changed=False)
    return SavedMeasuredEdges(run, shape, tuple(lines), provenance)
