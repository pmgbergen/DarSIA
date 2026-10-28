"""Explicit local reuse of accepted Build bytes with a new frame selection.

This is a metadata decision, not a Build completion or scientific qualification.
Only the old live series-manifest read is replaced. Every original numerical
Build input, TOML, protocol and physical output remains the loader's obligation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import tomllib


FORMAT = "darsia-local-build-reuse-v1"
SCOPE = "local-p1-selection-only"
_FIELDS = {"format", "scope", "operation_result", "current_series_manifest",
           "frame_configuration", "accepted_by", "reason"}
_FRAME_FORMAT = "darsia-build-raw-phase-inputs-v1"


def verify_local_build_frame_configuration(configuration, record_data):
    """Join queued frame roles to the separate RAW-phase DATA bound by reuse.

    ``record_data`` is the decoded ``frame_configuration`` document, not the
    reuse decision itself. It cannot be the enclosing staged configuration:
    that would create a configuration/decision SHA dependency cycle.
    """
    from .mass.native_optical_source import _same
    if (not isinstance(configuration, dict) or not isinstance(record_data, dict)
            or record_data.get("format") != _FRAME_FORMAT
            or configuration.get("run") != record_data.get("run")
            or not _same(configuration.get("native"), record_data.get("native"))
            or not isinstance(configuration.get("frames"), list)
            or not isinstance(record_data.get("frames"), list) or not record_data["frames"]
            or len(configuration["frames"]) != len(record_data["frames"])):
        raise ValueError("Local Build reuse requires separate matching RAW-phase frame DATA")
    keys = ("case_id", "frame_index", "elapsed_h", "source", "geometry", "phase_roles")
    for actual, expected in zip(configuration["frames"], record_data["frames"]):
        if (not isinstance(actual, dict) or not isinstance(expected, dict)
                or any(key not in actual or key not in expected for key in keys)
                or not _same({key: actual[key] for key in keys},
                             {key: expected[key] for key in keys})):
            raise ValueError("Local Build reuse frame DATA differs from the actual queued frames")


def verify_local_build_reuse(binding, *, operation_result, request, run, reader,
                            metadata_bytes=None):
    """Bind a local decision to current metadata; leave frame numerics to P1.

    ``binding`` is an exact descriptor of a separately saved JSON record. Its
    frame configuration is new explicit DATA, never an edited old receipt.
    The caller must retain and recheck every descriptor returned in ``inputs``.
    """
    from .mass.native_optical_source import _bound, _same
    binding = deepcopy(_bound(binding))
    record = reader.json(binding)
    # The record is pinned by its descriptor; who accepted it is read from it.
    if (not isinstance(record, dict) or set(record) != _FIELDS
            or record["format"] != FORMAT or record["scope"] != SCOPE
            or not isinstance(record["accepted_by"], str) or not record["accepted_by"].strip()
            or not isinstance(record["reason"], str) or not record["reason"].strip()):
        raise ValueError("Local Build reuse requires its explicit recorded decision: who accepted it and a plain reason")
    if not _same(_bound(record["operation_result"]), _bound(operation_result)):
        raise ValueError("Local Build reuse names another original completed Build")
    manifest = deepcopy(_bound(record["current_series_manifest"]))
    frames = deepcopy(_bound(record["frame_configuration"]))
    original_manifest = _bound(request["series_context"]["manifest"])
    if Path(manifest["path"]).resolve() != Path(original_manifest["path"]).resolve():
        raise ValueError("Local Build reuse names another original series location")
    if manifest["size_bytes"] > 16 * 1024 * 1024:
        raise ValueError("Current local series metadata exceeds the bounded reader")
    path = reader.path(manifest)
    if metadata_bytes is None:
        with path.open("rb") as stream:
            raw = stream.read(manifest["size_bytes"] + 1)
    else:
        raw = metadata_bytes(path)
    if (len(raw) != manifest["size_bytes"]
            or hashlib.sha256(raw).hexdigest() != manifest["sha256"]):
        raise ValueError("Current local series metadata changed during reading")
    series = tomllib.loads(raw.decode("utf-8-sig"))
    runs = (series.get("runs") or {}).get("ids")
    if (not isinstance(series.get("series_id"), str) or not series["series_id"]
            or not isinstance(runs, list) or not all(isinstance(item, str) for item in runs)
            or len(set(runs)) != len(runs) or run not in runs):
        raise ValueError("Current local series does not uniquely register the Build run")
    native = series.get("native") or {}
    native_path = native.get("configuration")
    if not isinstance(native_path, str) or not native_path:
        raise ValueError("Current local series lacks its original native profile")
    native_path = Path(native_path)
    if not native_path.is_absolute():
        native_path = Path(manifest["path"]).parent / native_path
    if (native.get("sha256") != request["profile"]["sha256"]
            or native_path.resolve() != Path(request["profile"]["path"]).resolve()):
        raise ValueError("Current local series changed the numerical Build native profile")
    configuration = reader.json(frames)
    if (not isinstance(configuration, dict) or configuration.get("format") != _FRAME_FORMAT
            or configuration.get("run") != run
            or not _same((configuration.get("native") or {}).get("configuration"), request["profile"])
            or not isinstance(configuration.get("frames"), list) or not configuration["frames"]):
        raise ValueError("Local Build reuse requires explicit current frames for the same run and native profile")
    inputs = [binding, manifest, frames]
    for item in inputs:
        reader.path(item)
    return dict(inputs=inputs, provenance=dict(
        record=binding, operation_result=deepcopy(operation_result),
        current_series_manifest=manifest, frame_configuration=frames,
        accepted_by=record["accepted_by"], reason=record["reason"], scope=SCOPE,
        numerical_build_inputs_unchanged=True, scientific_qualification_claimed=False))
