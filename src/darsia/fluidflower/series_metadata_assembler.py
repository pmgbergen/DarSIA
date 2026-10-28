"""In-memory DATA mapping of already supplied native/phase/N1 role records.

No role is estimated here. In particular, missing geometry, homographies,
photographic support, atmosphere or N1 domains fail instead of falling back to
another frame. The existing run validator owns the chronological DATA contract.
This module neither reads/writes files nor adopts a series or creates a job.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import re
from typing import Any

from darsia.fluidflower.pipeline.series import validate_run


_STAGES = ("native_task", "geometry", "phase_roles", "mass_spec")


def _identity(source):
    if not isinstance(source, Mapping):
        raise ValueError("Explicit acquisition source DATA must be a mapping.")
    path, sha = source.get("source_path"), source.get("source_sha256")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Explicit role records require a nonblank literal source path.")
    if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None:
        raise ValueError("Explicit source SHA must be canonical64 lowercase hexadecimal.")
    return path, sha


def _exact(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":"))


def _required_roles(spec):
    if not isinstance(spec, Mapping):
        raise ValueError("Supplied frame DATA must be a mapping.")
    for field in ("case_id", "elapsed_h", "source", *_STAGES):
        if field not in spec:
            raise ValueError("Missing supplied frame DATA: " + field)
    for field in ("source", *_STAGES):
        if not isinstance(spec[field], Mapping) or not spec[field]:
            raise ValueError("Supplied frame DATA needs a nonempty mapping: " + field)
    _identity(spec["source"])
    if not isinstance(spec["source"].get("source_binding"), Mapping) or not spec["source"]["source_binding"]:
        raise ValueError("Missing supplied native/phase acquisition source binding.")
    native = spec["native_task"]
    for field in ("kind", "geometry", "processor"):
        if field not in native:
            raise ValueError("Missing supplied native role: " + field)
    if native["kind"] == "raw_rig":
        # The existing RAW ABI has no JPEG-parent fields. The worker and the
        # original run validator require precisely these three task keys.
        if set(native) != {"kind", "geometry", "processor"} or set(native["processor"]) != {"raw_binding"}:
            raise ValueError("RAW native roles require only geometry and the explicit RAW binding")
    else:
        for field in ("parent_frame_index", "parent_reference"):
            if field not in native:
                raise ValueError("Missing supplied native role: " + field)
    phase = spec["phase_roles"]
    for field in ("area", "current_h", "reference_to_current_h", "static",
                  "retained_source_cache", "encoding_contract", "light_epoch_id"):
        if field not in phase:
            raise ValueError("Missing supplied phase role: " + field)
    branch = ("probability_receipt",) if phase.get("native_row") is not None else (
        "photo_sand_reference", "photo_proof")
    for field in branch:
        if field not in phase:
            raise ValueError("Missing supplied phase role: " + field)
    for field in ("atmosphere", "inner_polygon_native", "original_domain", "photo_top",
                  "source_receipt", "nominal_injected_mass_g", "selected_target_h"):
        if field not in spec["mass_spec"]:
            raise ValueError("Missing supplied N1 role: " + field)


def assemble_run_metadata(
    previous: Mapping[str, Any],
    selected_sources: list[Mapping[str, Any]],
    supplied_role_records: list[Mapping[str, Any]],
    *,
    storage_data: Mapping[str, Any] | None = None,
    precision: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy supplied records in selected order, retaining old role values exactly.

    A role record is the complete frame-stage DATA structure, with explicit
    scientific source bindings. The acquisition identity/time must agree with
    the SHA-finalized camera selection. Operational work indices and predecessor
    links alone are rebuilt for the newly supplied chronological selection.
    Original case IDs, stage data and all other source-role fields stay exact.

    Optional storage DATA is validated explicitly before a caller serializes a
    NEW run configuration. No manifest or loader inheritance occurs here.
    Supplying storage requires the future job precision; it is not selected.
    """
    if not isinstance(previous, Mapping):
        raise ValueError("Original run DATA must be a mapping.")
    run = previous.get("run")
    if not isinstance(run, str) or not run.strip() or run != run.strip():
        raise ValueError("Original run must have nonempty actual literal run text.")
    validate_run(deepcopy(dict(previous)))
    if not isinstance(selected_sources, list) or not selected_sources:
        raise ValueError("A fixed, nonempty camera selection is required.")
    if not isinstance(supplied_role_records, list):
        raise ValueError("Explicit per-acquisition role records are required.")
    index_field = previous["selection_index_field"]
    if index_field in {"frame_index", "source_index", "input_index", "selection_index", "selected_index"}:
        raise ValueError("A distinct explicit work-index field is required; acquisition indices remain unchanged.")
    records = {}
    for record in supplied_role_records:
        _required_roles(record)
        identity = _identity(record["source"])
        if identity in records:
            raise ValueError("Duplicate supplied source-role identity.")
        records[identity] = record
    chosen = [_identity(source) for source in selected_sources]
    if len(set(chosen)) != len(chosen) or set(chosen) != set(records):
        raise ValueError("Selection and supplied roles must have exact one-to-one coverage.")

    old_frames = previous["frames"]
    old_ids = [_identity(frame["source"]) for frame in old_frames]
    old_by_id = dict(zip(old_ids, old_frames))
    if [identity for identity in chosen if identity in old_by_id] != old_ids:
        raise ValueError("Original source membership or relative order changed.")
    predecessor = previous["phase"]["initial_previous"]["source"]["source_sha256"]
    frames = []
    for work_index, (source, identity) in enumerate(zip(selected_sources, chosen), 1):
        record = records[identity]
        for field in ("source_path", "source_sha256", "elapsed_h", "camera_datetime"):
            if field not in source or _exact(record["source"].get(field)) != _exact(source[field]):
                raise ValueError("Supplied roles disagree with the selected acquisition: " + field)
        if record["source"].get("run") != run or record["source"].get("selected") is not True:
            raise ValueError("Supplied role acquisition is not selected in this run.")
        if _exact(record["elapsed_h"]) != _exact(source["elapsed_h"]):
            raise ValueError("Supplied top-level and acquisition time differ.")
        old = old_by_id.get(identity)
        if old is not None:
            if old["case_id"] != record["case_id"]:
                raise ValueError("Original case ID changed.")
            for stage in _STAGES:
                if _exact(old[stage]) != _exact(record[stage]):
                    raise ValueError("Original scientific role changed: " + stage)
            remove = {index_field, "previous_source_sha256"}
            if _exact({k: v for k, v in old["source"].items() if k not in remove}) != _exact(
                {k: v for k, v in record["source"].items() if k not in remove}):
                raise ValueError("Original acquisition source-role metadata changed.")
            old_whole, record_whole = deepcopy(dict(old)), deepcopy(dict(record))
            for whole in (old_whole, record_whole):
                whole.pop("frame_index", None)
                whole["source"].pop(index_field, None)
                whole["source"].pop("previous_source_sha256", None)
            if _exact(old_whole) != _exact(record_whole):
                raise ValueError("Original whole frame DATA changed outside documented operational fields.")
        frame = deepcopy(dict(record))
        frame["frame_index"] = work_index
        frame["source"][index_field] = work_index
        frame["source"]["previous_source_sha256"] = predecessor
        frames.append(frame)
        predecessor = source["source_sha256"]
    candidate = deepcopy(dict(previous))
    candidate["frames"] = frames
    candidate["ordered_case_ids"] = [frame["case_id"] for frame in frames]
    if storage_data is not None:
        if not isinstance(storage_data, Mapping):
            raise ValueError("Explicit storage DATA must be a mapping.")
        from darsia.fluidflower.storage_outputs import parse_request

        storage = deepcopy(dict(storage_data))
        parse_request({"storage": storage}, precision)
        candidate["storage"] = storage
    validate_run(candidate)
    return candidate, dict(run=run, old_frame_count=len(old_frames), frame_count=len(frames),
                          new_frame_count=len(frames) - len(old_frames),
                          old_source_paths_SHAs_roles_relative_order_exact=True,
                          work_index_and_predecessor_rebuilt_only=True,
                          roles_invented=False, payloads_read=False,
                          adopted_for_execution=False, job_created=False)
