"""Closed current-Build DATA bindings, without IO, imports of darsia or release.

These descriptors are prospective inputs to original post-startup verifiers.
Shape validation is not an authenticated Build or scientific eligibility.
The enclosing configuration/task/plan must never be pinned inside this value.
"""
from __future__ import annotations

from copy import deepcopy
import json
import ntpath
from pathlib import PurePosixPath, PureWindowsPath
import re


FORMAT = "darsia-current-build-binding-v1"
PREFIX = "current_build."
_DESCRIPTOR = {"path", "sha256", "size_bytes"}
_FRAME = {"case_id", "raw_acquisition", "bridge_frame", "geometry",
          "area_provenance", "static_provenance"}


def _keys(value, expected, where):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(where + " requires its exact closed DATA fields")


def _absolute(value):
    if (not isinstance(value, str) or not value or "\x00" in value
            or not (PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute())
            or ".." in value.replace("\\", "/").split("/")):
        raise ValueError("Current Build requires an explicit nonescaping absolute path")
    return value


def _path_key(value):
    return ntpath.normcase(ntpath.normpath(value))


def _descriptor(value):
    _keys(value, _DESCRIPTOR, "Current Build descriptor")
    _absolute(value["path"])
    if (not isinstance(value["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
            or type(value["size_bytes"]) is not int or value["size_bytes"] < 0):
        raise ValueError("Current Build descriptor requires literal SHA-256 and nonnegative integer size")
    return value


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False)


def checked_current_build_binding(value, *, config=None):
    """Check literal associations only; no supplied claim can grant eligibility."""
    fields = {"format", "run", "publication", "operation_result", "bridge", "frames"}
    if isinstance(value, dict) and "independent_eligibility" in value:
        fields.add("independent_eligibility")
    if isinstance(value, dict) and "local_build_reuse" in value:
        fields.add("local_build_reuse")
    if isinstance(value, dict) and "metric_model" in value:
        fields.add("metric_model")
    _keys(value, fields, "Current Build binding")
    # Reject nonfinite values before any comparison can hide NaN or bool/int changes.
    json.dumps(value, sort_keys=True, allow_nan=False)
    if (value["format"] != FORMAT or not isinstance(value["run"], str)
            or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value["run"]) is None):
        raise ValueError("An explicit typed current Build run is required")
    if value.get("independent_eligibility") is not None:
        raise ValueError("Independent current Build eligibility is unimplemented; no supplied claim is accepted")
    _descriptor(value["operation_result"])
    _descriptor(value["bridge"])
    if "metric_model" in value:
        _descriptor(value["metric_model"])
    if "local_build_reuse" in value:
        _descriptor(value["local_build_reuse"])
    publication = value["publication"]
    _keys(publication, {"root", "job", "task", "done", "operation_result", "readset"},
          "Current Build publication")
    root = _absolute(publication["root"])
    for key in ("job", "task", "done", "operation_result"):
        _descriptor(publication[key])
    if any(publication["operation_result"][key] != value["operation_result"][key]
           for key in ("sha256", "size_bytes")):
        raise ValueError("Shared and original Build operation receipts must have identical bytes")
    rows = publication["readset"]
    if not isinstance(rows, list) or not rows or len(rows) > 8192:
        raise ValueError("A bounded explicit original publication readset is required")
    paths, descriptors = [], {}
    for row in rows:
        _keys(row, {"relative_path", "descriptor"}, "Publication readset row")
        relative = row["relative_path"]
        if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
                or relative.startswith("/") or any(part in {"", ".", ".."} for part in relative.split("/"))):
            raise ValueError("Publication readset paths must be literal relative file paths")
        bound = _descriptor(row["descriptor"])
        expected = root.rstrip("/\\") + "/" + relative
        key = _path_key(bound["path"])
        if key != _path_key(expected) or key in descriptors:
            raise ValueError("Publication readset escapes its original root or duplicates a file")
        paths.append(relative)
        descriptors[key] = bound
    if paths != sorted(paths):
        raise ValueError("Publication readset must retain canonical relative-path order")
    for key in ("job", "task", "done", "operation_result"):
        bound = publication[key]
        if not _same(descriptors.get(_path_key(bound["path"])), bound):
            raise ValueError("Publication readset omits its exact named " + key)
    if _path_key(publication["job"]["path"]) != _path_key(root.rstrip("/\\") + "/job.json"):
        raise ValueError("The Build job must belong to its original publication root")
    frames = value["frames"]
    if not isinstance(frames, list) or not frames or len(frames) > 8192:
        raise ValueError("Explicit ordered current Build frame bindings are required")
    cases, sources = set(), set()
    for frame in frames:
        _keys(frame, _FRAME, "Current Build frame")
        case = frame["case_id"]
        if (not isinstance(case, str) or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", case) is None
                or case in cases):
            raise ValueError("Current Build case ids must be explicit and unique")
        cases.add(case)
        for key in _FRAME - {"case_id"}:
            _descriptor(frame[key])
        source = frame["raw_acquisition"]
        identity = (_path_key(source["path"]), source["sha256"])
        if identity in sources:
            raise ValueError("Each current Build frame must name its own unique RAW acquisition")
        sources.add(identity)
    if config is not None:
        if (not isinstance(config, dict) or config.get("format") not in {
                    "darsia-native-phase-n1-run-v1", "darsia-native-phase-statistics-run-v1"}
                or config.get("run") != value["run"] or not isinstance(config.get("frames"), list)
                or any(not isinstance(row, dict) for row in config["frames"])
                or [row.get("case_id") for row in config["frames"]] != [row["case_id"] for row in frames]):
            raise ValueError("Current Build binding differs from the actual ordered native N1/P1 run")
        if config["format"] == "darsia-native-phase-statistics-run-v1":
            template = config.get("statistics")
        else:
            mass = config.get("mass")
            template = mass.get("template") if isinstance(mass, dict) else None
        if not isinstance(template, dict) or "pixel_domain" in template:
            raise ValueError("Current Build requires the original N1/P1 physical consumer; pixel_domain is unsupported")
        for bound, spec in zip(frames, config["frames"]):
            source = spec.get("source") or {}
            raw = (source.get("source_binding") or {}).get("raw_acquisition")
            if (not _same(raw, bound["raw_acquisition"])
                    or source.get("source_sha256") != bound["raw_acquisition"]["sha256"]
                    or not _same(spec.get("geometry"), bound["geometry"])
                    or (spec.get("native_task") or {}).get("kind") != "raw_rig"):
                raise ValueError("Current Build frame differs from the queued RAW source or geometry")
    return deepcopy(value)


def current_build_binding_inputs(value):
    """Yield the closed direct named pins; transitive original roles are collected separately."""
    value = checked_current_build_binding(value)
    yield PREFIX + "operation_result", value["operation_result"], "configuration"
    yield PREFIX + "bridge", value["bridge"], "configuration"
    if "metric_model" in value:
        yield PREFIX + "metric_model", value["metric_model"], "configuration"
    if "local_build_reuse" in value:
        yield PREFIX + "local_reuse.record", value["local_build_reuse"], "configuration"
    for key in ("job", "task", "done", "operation_result"):
        yield PREFIX + "publication." + key, value["publication"][key], "configuration"
    for index, row in enumerate(value["publication"]["readset"]):
        yield PREFIX + "publication.readset." + str(index), row["descriptor"], "configuration"
    for frame in value["frames"]:
        for key in sorted(_FRAME - {"case_id"}):
            role = "image" if key == "raw_acquisition" else "rig_model" if key == "geometry" else "configuration"
            yield PREFIX + "frame." + frame["case_id"] + "." + key, frame[key], role
