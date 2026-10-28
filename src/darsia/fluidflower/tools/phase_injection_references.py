"""Injection-end reference selection, independent of the Studio widgets.

Catalog loading reads metadata only. Missing photographs are not replaced by a
sparse reviewed successor: an explicit background materializer must produce the
selected acquisition using the review package's existing correction contract.
"""
from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence
from darsia.fluidflower.review_storage import resolve_review_path

POLICY = "first-available-acquisition-at-or-after-injection-end-v1"
RIG_SAVED_POLICY = "first-saved-camera-timeline-frame-at-or-after-injection-end-v1"
FULL_MOVIE_INVENTORIES = local_cache_root() / 'phase_diagnostics/full_movie_inventory_20260912'


def _read(path: Path) -> dict:
    return json.loads(resolve_review_path(path).read_text(encoding="utf-8-sig"))


def _sha(path: Path) -> str:
    return hashlib.sha256(resolve_review_path(path).read_bytes()).hexdigest()


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _identity(frame: Mapping) -> tuple[str, float | None]:
    # Native caches can carry either drive-letter or UNC aliases of a source.
    name = str(frame.get("source_name") or str(frame.get("source_path", "")).replace("\\", "/").rsplit("/", 1)[-1])
    elapsed = _finite(frame.get("elapsed_h"))
    return name.casefold(), None if elapsed is None else round(elapsed, 9)


def injection_end_events(manifest: Mapping) -> list[dict]:
    """Return one end per continuous injection window, in chronological order.

    Contiguous rate segments belong to one injection; a real pause separates
    windows. Simultaneous ends for different ports remain separate references.
    """
    stage_sources = [
        ("injection_anchor_contract.current_optical_injection_stages",
         (manifest.get("injection_anchor_contract") or {}).get("current_optical_injection_stages")),
        ("phase_topology.injection_stages", (manifest.get("phase_topology") or {}).get("injection_stages")),
        ("injection_stages", manifest.get("injection_stages")),
    ]
    timing_source, stages = next(((key, value) for key, value in stage_sources if value), (None, None))
    groups = []
    if stages:
        for index, stage in enumerate(stages):
            intervals = [(r.get("start_h"), r.get("end_h")) for r in stage.get("rate_segments", [])
                         if _finite(r.get("rate_kg_s", 1)) is not None and float(r.get("rate_kg_s", 1)) > 0]
            groups.append((str(stage.get("id") or f"injection-{index+1}"),
                           intervals or [(stage.get("start_h"), stage.get("end_h"))], timing_source))
    else:
        for index, point in enumerate(manifest.get("injection_points") or []):
            groups.append((str(point.get("id") or f"injection-{index+1}"),
                           point.get("active_windows_h") or [], "injection_points.active_windows_h"))
    events = []
    for source_id, intervals, timing_source in groups:
        valid = sorted((a,b) for row in intervals if len(row) == 2
                       if (a := _finite(row[0])) is not None and (b := _finite(row[1])) is not None and b > a)
        merged = []
        for start,end in valid:
            if merged and start <= merged[-1][1] + 1e-12:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start,end])
        for window,(start,end) in enumerate(merged):
            events.append({"id":f"{source_id}-end-{window+1}", "start_h":start,
                "end_h":end, "injector_id":source_id, "timing_source":timing_source})
    events.sort(key=lambda event:(event["end_h"],event["start_h"],event["id"]))
    return [{**event,"label":f"Injection {index+1} end"} for index,event in enumerate(events)]


def first_at_or_after(frames: Sequence[Mapping], end_h: float) -> tuple[dict | None, list[dict]]:
    """Select without nearest-neighbor rounding or a silent before-end fallback."""
    end = _finite(end_h)
    if end is None:
        raise ValueError("Injection end must be finite")
    ordered = sorted((dict(row) for row in frames if _finite(row.get("elapsed_h")) is not None
                      and float(row["elapsed_h"]) >= end), key=lambda row:(float(row["elapsed_h"]),_identity(row)[0]))
    skipped = []
    for row in ordered:
        if row.get("blacklisted") or row.get("available") is False:
            skipped.append({"source_name":_identity(row)[0],"elapsed_h":float(row["elapsed_h"]),
                "reason":"blacklisted" if row.get("blacklisted") else "unavailable"})
            continue
        return row, skipped
    return None, skipped


def _manifest_path(review_root: Path, run: str) -> Path:
    return review_root / "native_frames" / run / "manifest.json"


def reference_catalog_path(review_root: Path, run: str) -> Path:
    return Path(review_root) / "native_references" / run.casefold() / "injection_ends.json"


def rig_reference_frame_index(manifest: Mapping, reference: Mapping | None = None) -> int | None:
    """Find the saved source by SHA and camera time, never by proximity.

    Rig references stay in their own native geometry; historical translation-only
    registration is not required when their own saved map is displayed.
    """
    from darsia.fluidflower.phase_ffac_rig_camera_metadata import SOURCE_KEY, SOURCE_CONTRACT
    marker = manifest.get(SOURCE_KEY) or {}
    selected=manifest.get("ffac_selected_policy_a_sources") or {}
    if selected:
        if selected.get("contract") != "ffac-selected-policy-a-full-rig-native-review-v1" or selected.get("inference_authorized") is not False:
            return None
    elif marker.get("contract") != SOURCE_CONTRACT or marker.get("inference_authorized") is not False:
        return None
    if reference is None:
        reference = manifest.get("reference") or {}
        if _finite(reference.get("elapsed_h")) is None or float(reference["elapsed_h"]) >= 0:
            return None
    matches = [index for index, frame in enumerate(manifest.get("frame_sources") or [])
               if _identity(frame) == _identity(reference)
               and len(str(reference.get("source_sha256") or "")) == 64
               and frame.get("source_sha256") == reference.get("source_sha256")]
    return matches[0] if len(matches) == 1 else None


def _rig_saved_references(manifest, root, run, dataset, path, digest):
    """Expose saved end controls explicitly, without unsaved acquisition claims."""
    from darsia.fluidflower.phase_ffac_rig_camera_metadata import SOURCE_KEY, SOURCE_CONTRACT
    marker = manifest.get(SOURCE_KEY)
    selected=manifest.get("ffac_selected_policy_a_sources") or {}
    if selected:
        if selected.get("contract") != "ffac-selected-policy-a-full-rig-native-review-v1" or selected.get("inference_authorized") is not False:
            raise ValueError("Invalid selected camera reference contract")
    elif not isinstance(marker, Mapping):
        return None
    elif marker.get("contract") != SOURCE_CONTRACT or marker.get("inference_authorized") is not False:
        raise ValueError("Invalid rig camera reference contract")
    rows = []
    for event in injection_end_events(manifest):
        chosen, skipped = first_at_or_after(manifest.get("frame_sources") or [], event["end_h"])
        index = rig_reference_frame_index(manifest, chosen) if chosen is not None else None
        rows.append({**event, "dataset":dataset, "run":run, "review_root":str(root),
            "native_manifest":str(path), "native_manifest_sha256":digest,
            "selection_policy":RIG_SAVED_POLICY, "selection_scope":"Saved camera-clock review timeline only",
            "description":"First saved reviewed image at/after the end (not the first camera acquisition)",
            "available":index is not None, "needs_materialization":False,
            "frame_index":index, "path":None, "reference_offset":{"x":0., "y":0.},
            "source_name":(chosen.get("source_name") or str(chosen.get("source_path", "")).replace("\\", "/").rsplit("/", 1)[-1]) if chosen else None,
            "source_path":chosen.get("source_path") if chosen else None,
            "source_sha256":chosen.get("source_sha256") if chosen else None,
            "elapsed_h":chosen.get("elapsed_h") if chosen else None,
            "gap_seconds":(float(chosen["elapsed_h"])-event["end_h"])*3600 if chosen else None,
            "skipped_sources":skipped,
            "reason":"Existing source-bound photograph and map; no reference inference" if index is not None
                     else "No source-bound saved frame at/after this injection end"})
    return rows


def _offset(row: Mapping | None) -> dict | None:
    if row is None:
        return None
    return {"x":float(row.get("x", row.get("dx_canvas_fraction", 0.0))),
            "y":float(row.get("y", row.get("dy_canvas_fraction", 0.0)))}


def _can_materialize(manifest: Mapping, chosen: Mapping, dataset: str) -> bool:
    """Metadata-only capability check; the resolver verifies files before use."""
    contract = manifest.get("correction_contract") or {}
    if not chosen.get("source_path") or chosen.get("available") is False or not manifest.get("canvas_wh"):
        return False
    if dataset == "ffbm":
        return bool(contract.get("radiometry_profile") and contract.get("registration_config_sha256")
                    and contract.get("curvature_grid_sha256") and (manifest.get("domain") or {}).get("source"))
    return dataset in {"ffac", "ffum"} and bool(contract)


def _inventory_paths(pair: Mapping, manifest: Mapping, review_root: Path, run: str):
    for source in (pair, manifest):
        hint = source.get("acquisition_inventory")
        hint = hint.get("path") if isinstance(hint, Mapping) else hint
        if hint:
            path = Path(str(hint))
            yield path if path.is_absolute() else review_root/path
    yield review_root/"native_references"/run/"acquisition_inventory.json"
    # Existing complete acquisition metadata; no movie pixels are used.
    yield FULL_MOVIE_INVENTORIES/f"{run}.json"


def discover_acquisition_inventory(pair: Mapping, manifest: Mapping, review_root: Path) -> tuple[list[dict] | None, dict]:
    """Read a matching complete inventory; do not scan raw folders on the GUI."""
    run = str(pair.get("run") or manifest.get("run") or "").casefold()
    dataset = str(pair.get("dataset") or manifest.get("dataset") or "").casefold()
    # Some native packages give a retained negative control a synthetic -1e-6 h
    # timeline position while retaining a real member's source filename. It is
    # not an acquisition timestamp and must not invalidate the source clock.
    reviewed = {_identity(row) for row in manifest.get("frame_sources") or [] if row.get("source_path")
                and not (row.get("selection_status") == "pre_injection_negative_control"
                         and _finite(row.get("elapsed_h")) is not None and float(row["elapsed_h"]) < 0)}
    rejected = []
    for path in dict.fromkeys(_inventory_paths(pair, manifest, Path(review_root), run)):
        if not path.is_file(): continue
        try:
            inventory = _read(path)
            if str(inventory.get("run", run)).casefold() != run or str(inventory.get("dataset", dataset)).casefold() != dataset:
                raise ValueError("Inventory belongs to another experiment")
            if inventory.get("complete") is False:
                raise ValueError("Acquisition inventory is incomplete")
            frames = [dict(row) for row in inventory["frames"]]
            identities = {_identity(row) for row in frames}
            # Camera numbering can restart in another acquisition folder. Keep
            # every timestamp for a filename; collapsing a set into one value
            # made valid reset sequences depend on Python's hash iteration.
            names = {name for name,_ in identities}
            if any(name in names and (name,elapsed) not in identities for name,elapsed in reviewed):
                raise ValueError("Inventory and native review timestamps disagree")
            if reviewed and not (reviewed & identities):
                raise ValueError("Inventory shares no source identities with this review")
            return frames, {"path":str(path),"sha256":_sha(path),"complete":True,
                **{key:inventory[key] for key in ("selection_scope","source_format") if key in inventory}}
        except (OSError,ValueError,KeyError,TypeError) as error:
            rejected.append({"path":str(path),"reason":str(error)})
    return None, {"complete":False,"rejected":rejected}


def _two_arm_reference_rows(manifest: Mapping, review_root: Path, run: str,
                            dataset: str, native_manifest_path: Path,
                            native_manifest_sha256: str) -> list[dict] | None:
    """Expose two-arm injection-end controls only as unavailable references.

    A copied acquisition inventory or correction contract must never cause an
    injection-end photo to be prepared with the former photometry arm.
    """
    if "two_arm_photometry" not in manifest:
        return None
    photometry = manifest["two_arm_photometry"]
    reason = "Arm-specific injection-end references have not been materialized"
    availability_path = review_root / "native_references" / run / "reference_availability.json"
    baseline_path = review_root / "native_references" / run / "pre_injection.jpg"
    try:
        if not isinstance(photometry, Mapping):
            raise ValueError("two-arm photometry marker is malformed")
        availability = _read(availability_path)
        baseline = availability["baseline"]
        if (availability.get("format") != "ffac-two-arm-reference-availability-v1"
                or availability.get("arm") != photometry.get("arm")
                or availability.get("run") != run
                or availability.get("materialized_manifest_sha256") != native_manifest_sha256
                or availability.get("old_reference_assets_reused") is not False
                or baseline.get("available") is not True
                or Path(baseline["materialized_native_path"]).resolve() != baseline_path.resolve()
                or _sha(baseline_path) != baseline["materialized_native_sha256"]
                or availability["injection_ends"].get("available") is not False
                or availability["clearwater"].get("available") is not False):
            raise ValueError("two-arm reference provenance disagrees with the current native image")
    except (OSError, ValueError, KeyError, TypeError) as error:
        reason = f"Arm-specific references are not verified: {error}"
    return [{**event, "dataset":dataset, "run":run, "review_root":str(review_root),
             "native_manifest":str(native_manifest_path),
             "native_manifest_sha256":native_manifest_sha256,
             "selection_policy":POLICY, "available":False,
             "needs_materialization":False, "path":None, "frame_index":None,
             "source_name":None, "source_path":None, "elapsed_h":None,
             "reason":reason, "reference_offset":None}
            for event in injection_end_events(manifest)]


def injection_end_references(pair: Mapping, review_root: Path) -> list[dict]:
    """Return Studio rows; unavailable exact selections remain visibly disabled."""
    root = resolve_review_path(review_root)
    run = str(pair.get("run") or "").casefold()
    path = _manifest_path(root, run)
    try:
        manifest = _read(path)
    except (OSError,ValueError):
        return []
    digest = _sha(path)
    dataset = str(pair.get("dataset") or manifest.get("dataset") or "").casefold()
    rig_rows = _rig_saved_references(manifest, root, run, dataset, path, digest)
    if rig_rows is not None:
        return rig_rows
    two_arm_rows = _two_arm_reference_rows(manifest, root, run, dataset, path, digest)
    if two_arm_rows is not None:
        return two_arm_rows
    events = injection_end_events(manifest)
    frames = [dict(row) for row in manifest.get("frame_sources") or []]
    by_identity = {_identity(row):index for index,row in enumerate(frames)}
    inventory, inventory_record = discover_acquisition_inventory(pair,manifest,root)
    prepared = {}
    try:
        saved = _read(reference_catalog_path(root,run))
        if saved.get("native_manifest_sha256") == digest and saved.get("selection_policy") == POLICY:
            prepared = {str(row["id"]):row for row in saved.get("references",[])}
    except (OSError,ValueError,KeyError,TypeError):
        pass
    result = []
    for event in events:
        chosen, skipped = first_at_or_after(inventory if inventory is not None else frames,event["end_h"])
        known_first = inventory is not None or (chosen is not None and float(chosen["elapsed_h"]) == event["end_h"])
        item = {**event,"dataset":dataset,"run":run,"review_root":str(root),
            "native_manifest":str(path),"native_manifest_sha256":digest,"selection_policy":POLICY,
            "correction_contract":manifest.get("correction_contract",{}),
            "canvas_wh":manifest.get("canvas_wh"),"content_box_xywh":manifest.get("content_box_xywh"),
            "frame_index":None,"path":None,"available":False,"needs_materialization":False,
            "source_name":None,"source_path":None,"elapsed_h":None,"gap_seconds":None,
            "inventory":inventory_record,"skipped_sources":skipped,"reference_offset":None}
        for key in ("selection_scope","source_format"):
            if key in inventory_record:
                item[key] = inventory_record[key]
        if item.get("source_format") == "published-tiff":
            item["description"] = "First available published TIFF at or after injection end; same input format and correction as Studio"
        if chosen is None:
            item["reason"] = "No available acquisition at or after this injection end"
        elif not known_first:
            item["reason"] = "Full acquisition inventory is needed; sparse reviewed successor is not an end reference"
            item["needs_inventory"] = True
        else:
            index = by_identity.get(_identity(chosen))
            item.update(elapsed_h=float(chosen["elapsed_h"]),source_name=chosen.get("source_name"),
                source_path=chosen.get("source_path"),source_sha256=chosen.get("source_sha256"),
                gap_seconds=(float(chosen["elapsed_h"])-event["end_h"])*3600,
                frame_index=index,reason="",needs_materialization=index is None)
            if index is not None:
                # The caller's existing _frame_asset resolves native JPEG/PNG.
                item["available"] = True
                offsets = manifest.get("reference_registration") or []
                item["reference_offset"] = _offset(next((row for row in offsets if row.get("frame_index") == index),None))
            else:
                item["reason"] = "First acquisition selected; matching corrected reference must be prepared"
                if _can_materialize(manifest,chosen,dataset):
                    item.update(available=True,reason="Exact source will be prepared on first use")
                cached = prepared.get(event["id"],{})
                if _identity(cached) == _identity(chosen) and cached.get("path") and cached.get("sha256"):
                    cached_path = Path(str(cached["path"]))
                    cached_path = cached_path if cached_path.is_absolute() else root/cached_path
                    if cached_path.is_file():
                        item.update(path=str(cached_path),available=True,needs_materialization=False,
                            reference_offset=_offset(cached.get("reference_offset")),sha256=cached["sha256"],reason="")
        result.append(item)
    return result


def resolve_reference_asset(reference: Mapping, review_root: Path, run: str) -> Path:
    """Resolve in the existing deferred-image worker, never on the GUI thread.

    Prepared pixels are accepted only after the current catalog binding and the
    asset SHA256 agree. Preparation owns its per-run lock and atomic writes.
    """
    root = Path(review_root)
    pair = {"dataset":reference.get("dataset"),"run":run}
    reference_id = str(reference["id"])
    def current():
        row = next((row for row in injection_end_references(pair,root) if row["id"] == reference_id),None)
        if row is None:
            raise ValueError(f"Injection reference is no longer in this review: {reference_id}")
        if reference.get("native_manifest_sha256") != row.get("native_manifest_sha256"):
            raise ValueError("Native review manifest changed since this reference was selected")
        if reference.get("source_name") and _identity(reference) != _identity(row):
            raise ValueError("Injection reference source changed since it was selected")
        return row
    row = current()
    if row.get("selection_policy") == RIG_SAVED_POLICY:
        if not row.get("available"):
            raise FileNotFoundError(row.get("reason"))
        from darsia.fluidflower.phase_photographic_assets import resolve_display_mode
        return resolve_display_mode(root, run, row["frame_index"], "natural")
    if row.get("frame_index") is not None:
        folder = root/"native_frames"/str(row["run"])
        stem = f"frame_{int(row['frame_index']):03d}"
        for suffix in (".png", ".jpg"):
            path = folder/(stem+suffix)
            if path.is_file():
                return path
        raise FileNotFoundError(f"Reviewed injection reference frame is missing: {folder/stem}")
    if row.get("needs_materialization"):
        raise FileNotFoundError("This injection reference has no saved inspection photograph; historical preparation is unavailable")
    if not row.get("path") or not row.get("sha256"):
        raise FileNotFoundError(row.get("reason") or "No verified injection reference asset is available")
    path = resolve_review_path(row["path"])
    if _sha(path) != row["sha256"]:
        raise ValueError(f"Injection reference asset checksum mismatch: {path}")
    return path


def inventory_from_experiment(experiment: Any, *, dataset: str, run: str) -> dict:
    """Background API for AC/UM protocol metadata; does not load a rig or image."""
    frames = []
    for path in experiment.data:
        path = Path(path)
        try:
            date = experiment.get_datetime(path)
            frames.append({"source_path":str(path),"source_name":path.name,
                "acquired":date.isoformat(),"elapsed_h":float(experiment.time_since_start(date)),
                "blacklisted":bool(experiment.is_blacklisted(path)),"available":True})
        except (ValueError,KeyError,OSError):
            # Unknown time cannot be ordered; do not claim a complete inventory.
            raise ValueError(f"Cannot resolve acquisition timestamp: {path}")
    return {"format":"injection-reference-inventory-v1","dataset":dataset,"run":run,"complete":True,
        "t0":experiment.experiment_start.isoformat(),"frames":sorted(frames,key=lambda row:row["elapsed_h"])}


def inventory_from_bm_filenames(manifest: Mapping) -> dict:
    """Background API: BM published TIFF filenames encode acquisition timestamps."""
    known = [row for row in manifest.get("frame_sources") or [] if row.get("source_path")]
    origins = []
    for row in known:
        stamp = datetime.strptime(Path(row["source_path"]).name[:17],"%y%m%d_time%H%M%S")
        origins.append(stamp-timedelta(hours=float(row["elapsed_h"])))
    if not origins or any(abs((date-origins[0]).total_seconds())>1e-5 for date in origins):
        raise ValueError("BM reviewed timestamps do not establish one consistent time origin")
    frames = []
    for folder in dict.fromkeys(Path(row["source_path"]).parent for row in known):
        for path in folder.glob("*.TIF"):
            stamp = datetime.strptime(path.name[:17],"%y%m%d_time%H%M%S")
            frames.append({"source_path":str(path),"source_name":path.name,"acquired":stamp.isoformat(),
                "elapsed_h":(stamp-origins[0]).total_seconds()/3600,"available":True,"blacklisted":False})
    if not frames:
        raise ValueError("No BM source TIFFs are available")
    return {"format":"injection-reference-inventory-v1","dataset":str(manifest.get("dataset") or "ffbm"),"run":manifest["run"],
        "source_format":"published-tiff","selection_scope":"all published TIFF acquisitions in the original Studio input series; original camera JPEGs have different image processing",
        "complete":True,"t0":origins[0].isoformat(),"frames":sorted(frames,key=lambda row:row["elapsed_h"])}
