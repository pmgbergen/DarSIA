"""Read-only discovery and timeline alignment of historical phase reviews.

This module never saves decisions or modifies frozen training inputs.  Its return
value is a separate display layer: consumers must not merge it into a review store.
Unverifiable items remain in the result with ``displayable=False``.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from darsia.fluidflower.review_storage import resolve_review_path


FORMAT = "fluidflower-review-history-v1"
_GEOMETRY = ("canvas_wh", "content_box_xywh", "analysis_image_shape_hw")
_SPATIAL_CONTRACTS = (
    "geometry_sha256", "geometry_fingerprint", "spatial_transform_sha256",
    "registration_sha256", "crop_sha256", "geometry_contract", "spatial_contract",
)
_VERDICT_FIELDS = (
    "schema", "run", "comparison_mode", "reviewer", "reviewer_id", "choice",
    "winner_candidate_id", "flags", "notes", "updated_utc", "candidate_a",
    "candidate_b", "frame_choice_schema",
)


def _key(path: Path) -> str:
    return str(resolve_review_path(path).absolute()).replace("\\", "/").rstrip("/").casefold()


def _path(value: Any, base: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return resolve_review_path(path if path.is_absolute() else base / path, context=base)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _finite(value: Any) -> float | None:
    from darsia.fluidflower.mass_inputs import _number as _canonical_finite_number
    return _canonical_finite_number(value)


def _same_time(a: Any, b: Any) -> bool:
    left, right = _finite(a), _finite(b)
    return left is not None and right is not None and abs(left - right) <= 1e-6


def _photo(row: Mapping[str, Any]) -> str:
    name = row.get("source_name") or row.get("source_path") or ""
    return str(name).replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _default_review_roots() -> Sequence[Path]:
    # This helper only returns local paths; it does not inspect model/image data.
    from darsia.fluidflower.phase_workflow import default_phase_review_roots

    return default_phase_review_roots()


class _Reader:
    def __init__(self) -> None:
        self.cache: dict[str, dict[str, Any] | None] = {}
        self.digests: dict[str, str] = {}
        self.warnings: list[str] = []
        self.selected_model_root: Path | None = None

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def read(self, path: Path | None, *, required: bool = False,
             expected_sha256: str = "") -> dict[str, Any] | None:
        if path is None:
            return None
        path = resolve_review_path(path)
        identity = _key(path)
        if identity not in self.cache:
            try:
                content = path.read_bytes()
                self.digests[identity] = hashlib.sha256(content).hexdigest()
                payload = json.loads(content.decode("utf-8-sig"))
                if not isinstance(payload, dict):
                    raise ValueError("expected a JSON object")
                self.cache[identity] = payload
            except (OSError, ValueError, TypeError) as exc:
                self.cache[identity] = None
                if required:
                    self.warn(f"Cannot read history source {path}: {exc}")
        elif required and self.cache[identity] is None:
            self.warn(f"History source unavailable: {path}")
        if expected_sha256 and self.digests.get(identity) != expected_sha256:
            self.warn(f"Historical source SHA-256 mismatch: {path}; source excluded")
            return None
        return self.cache[identity]


def _snapshot_path(record: Any, model_root: Path, suffix: str) -> Path | None:
    if isinstance(record, str):
        path = _path(record, model_root)
    elif isinstance(record, Mapping):
        path = _path(record.get("snapshot_path") or record.get("source_path"), model_root)
    else:
        return None
    # Models have been migrated between cache/NAS. Prefer the immutable bundle
    # physically beside this request over its obsolete absolute snapshot path.
    if path is not None:
        local = model_root / "input_snapshot" / path.name
        if local.is_file():
            return local
    return path


def _model_sources(campaign_root: Path, reader: _Reader) -> list[dict[str, Any]]:
    receipt = reader.read(campaign_root / "collection_receipt.json") or {}
    manifest = reader.read(campaign_root / "postflight_manifest.json") or {}
    if not receipt:
        raw_receipt = manifest.get("collection_receipt")
        if isinstance(raw_receipt, Mapping):
            receipt = dict(raw_receipt)
        elif isinstance(raw_receipt, str):
            receipt = reader.read(_path(raw_receipt, campaign_root)) or {}
    initial = _path(receipt.get("source_model_root") or manifest.get("source_model_root"),
                    campaign_root)
    reader.selected_model_root = initial
    if initial is None:
        reader.warn(f"No source model recorded for review collection {campaign_root}")
        return []
    queue: deque[Path] = deque([initial])
    seen: set[str] = set()
    families: set[str] = set()
    sources: list[dict[str, Any]] = []
    # Enumerate immediate model siblings only, never recursively search the NAS.
    def family(root: Path) -> None:
        identity = _key(root)
        if identity in families:
            return
        families.add(identity)
        if root.is_dir():
            try:
                queue.extend(sorted(p for p in root.iterdir() if p.is_dir()
                                    and (p / "input_snapshot").is_dir()))
            except OSError as exc:
                reader.warn(f"Cannot enumerate model history family {root}: {exc}")

    family(initial.parent)
    # Earlier Studio models use this documented sibling layout.
    for ancestor in initial.parents:
        if ancestor.name.casefold() == "phase_models":
            family(ancestor / "studio_retraining" / initial.parent.name)
            break
    while queue and len(seen) < 128:
        model_root = queue.popleft()
        identity = _key(model_root)
        if identity in seen:
            continue
        seen.add(identity)
        request = reader.read(model_root / "training_request.json") or {}
        summary = reader.read(model_root / "summary.json") or {}
        if not request and not summary and not (model_root / "input_snapshot").is_dir():
            reader.warn(f"Historical model metadata unavailable: {model_root}")
            continue
        family(model_root.parent)
        settings = request.get("settings") or {}
        if isinstance(settings, Mapping):
            for field in ("parent_model", "evidence_parent_model", "physics_parent_model",
                          "comparison_baseline_parent_model"):
                parent = _path(settings.get(field), model_root)
                if parent:
                    queue.append(parent)
        evaluation_parent = summary.get("evaluation_parent") or {}
        if isinstance(evaluation_parent, Mapping):
            parent = _path(evaluation_parent.get("model_root"), model_root)
            if parent:
                queue.append(parent)
        inherited_record=request.get("inherited_training_request")
        if (request.get("format")=="ffac-selected-policy-a-frozen-phase-review-request-v1"
                and isinstance(inherited_record,Mapping)):
            inherited=_path(inherited_record.get("path"),model_root)
            if inherited and reader.read(inherited,required=True,expected_sha256=str(inherited_record.get("sha256") or "")) is not None:
                if not inherited_record.get("sha256"):
                    reader.warn("Selected inherited training request has no source SHA; ancestry excluded")
                else:
                    queue.append(inherited.parent)
        else:
            inherited = _path(inherited_record, model_root)
            if inherited:
                queue.append(inherited.parent)
        bundles = request.get("bundles") or summary.get("source_snapshot") or []
        if not isinstance(bundles, list):
            reader.warn(f"Unknown bundle structure in {model_root}")
            bundles = []
        discovered: set[str] = set()
        for bundle in bundles:
            if not isinstance(bundle, Mapping):
                reader.warn(f"Unknown bundle entry in {model_root}")
                continue
            decisions = _snapshot_path(bundle.get("decisions"), model_root, "decisions")
            manifest_path = _snapshot_path(bundle.get("combined_manifest"), model_root,
                                           "combined_manifest")
            if decisions is None:
                reader.warn(f"Bundle without decisions in {model_root}")
                continue
            discovered.add(_key(decisions))
            sources.append({"kind": "model_snapshot", "decisions_path": str(decisions),
                            "manifest_path": str(manifest_path) if manifest_path else "",
                            "decisions_sha256": str((bundle.get("decisions") or {}).get("sha256") or "")
                            if isinstance(bundle.get("decisions"), Mapping)
                            else str(bundle.get("decisions_sha256") or ""),
                            "manifest_sha256": str((bundle.get("combined_manifest") or {}).get("sha256") or "")
                            if isinstance(bundle.get("combined_manifest"), Mapping)
                            else str(bundle.get("combined_manifest_sha256") or ""),
                            "model_root": str(model_root),
                            "model_id": str(request.get("model_id") or summary.get("model_id")
                                            or model_root.name),
                            "updated_utc": str(request.get("created_utc") or "")})
        for decisions in sorted((model_root / "input_snapshot").glob("*_decisions.json")):
            if _key(decisions) in discovered:
                continue
            manifest_path = decisions.with_name(decisions.name.replace("_decisions.json",
                                                                       "_combined_manifest.json"))
            sources.append({"kind": "model_snapshot", "decisions_path": str(decisions),
                            "manifest_path": str(manifest_path), "model_root": str(model_root),
                            "model_id": str(request.get("model_id") or model_root.name)})
    if queue:
        reader.warn("Model history discovery reached its 128-model safety limit; some sources omitted")
    return sources


def _session_sources(review_roots: Sequence[Path], active_path: Path | None,
                     reader: _Reader) -> list[dict[str, Any]]:
    roots = [resolve_review_path(p) for p in review_roots]
    if active_path:
        roots.append(active_path.parent)
    try:
        roots.extend(_default_review_roots())
    except (ImportError, OSError) as exc:
        reader.warn(f"Default local review roots unavailable: {exc}")
    # The caller normally provides just the active manual_pairwise_review dir.
    # Expand the session's explicit local root to include sibling sessions.
    for root in list(roots):
        session = reader.read(root / "review_session.json") or {}
        parent = _path(session.get("local_review_root"), root)
        if parent:
            roots.append(parent)
    unique: dict[str, Path] = {_key(p): p for p in roots}
    compact = [p for p in unique.values()
               if not any(p != q and q in p.parents for q in unique.values())]
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in compact:
        # Review roots are local session stores. A broad network traversal is
        # deliberately refused; frozen NAS inputs are reached by explicit paths.
        if not root.is_dir():
            continue
        try:
            network = str(root).startswith(("\\\\", "//"))
            if os.name == "nt" and root.drive and not network:
                import ctypes
                network = ctypes.windll.kernel32.GetDriveTypeW(str(root.anchor)) == 4
            if network:
                # Explicit session dirs on the NAS are supported. Only named
                # review-session layouts are inspected, never a network rglob.
                session_paths = [root / "review_session.json"]
                for pattern in ("sessions/*/manual_pairwise_review/review_session.json",
                                "*/sessions/*/manual_pairwise_review/review_session.json",
                                "*/*/*/sessions/*/manual_pairwise_review/review_session.json"):
                    session_paths.extend(root.glob(pattern))
                session_paths = sorted(p for p in session_paths if p.is_file())
            else:
                session_paths = sorted(root.rglob("review_session.json"))
        except OSError as exc:
            reader.warn(f"Cannot enumerate local review sessions {root}: {exc}")
            continue
        for session_path in session_paths:
            if (session_path.parent.name == "original"
                    and (session_path.parent.parent / "reveal.json").is_file()):
                # Frozen session metadata is trusted only through its receipt.
                continue
            if _key(session_path) in seen:
                continue
            seen.add(_key(session_path))
            session = reader.read(session_path, required=True) or {}
            directory = _path(session.get("decision_dir"), session_path.parent) or session_path.parent
            decisions = directory / "pairwise_review_decisions.json"
            combined = _path(session.get("combined_root"), session_path.parent)
            source = {"kind": "local_session", "decisions_path": str(decisions),
                      "manifest_path": str(combined / "postflight_manifest.json") if combined else "",
                      "session_path": str(session_path),
                      "session_id": str(session.get("review_session_id") or ""),
                      "reviewer": str(session.get("reviewer") or ""),
                      "updated_utc": str(session.get("updated_utc") or ""),
                      "model_id": str(session.get("target_model_id") or "")}
            if decisions.is_file():
                sources.append(source)
            frozen = session.get("frozen_blind_snapshot")
            if isinstance(frozen, Mapping):
                frozen = frozen.get("path") or frozen.get("snapshot_path")
            frozen_path = _path(frozen, session_path.parent)
            if frozen_path:
                frozen_payload = reader.read(frozen_path, required=True, expected_sha256=str(
                    session.get("frozen_blind_snapshot_sha256") or ""))
                if frozen_payload is None:
                    continue
                if frozen_payload.get("format") == "phase-blind-reveal-snapshot-v1":
                    decisions_path = _path(frozen_payload.get("frozen_decisions"), frozen_path.parent)
                    hashes = frozen_payload.get("snapshot_sha256") or {}
                    if decisions_path is None or not isinstance(hashes, Mapping):
                        reader.warn(f"Invalid frozen blind receipt: {frozen_path}")
                        continue
                    expected = hashes.get(str(decisions_path)) or frozen_payload.get("source_decisions_sha256") or ""
                    original_session_path = decisions_path.parent / "review_session.json"
                    original_session = reader.read(original_session_path, required=True,
                        expected_sha256=str(hashes.get(str(original_session_path)) or "")) or {}
                    original_root = _path(original_session.get("combined_root"), original_session_path.parent)
                    sources.append({**source, "kind": "frozen_blind", "decisions_path": str(decisions_path),
                                    "decisions_sha256": str(expected), "frozen_receipt": str(frozen_path),
                                    "manifest_path": str(original_root / "postflight_manifest.json")
                                    if original_root else "",
                                    "session_id": str(original_session.get("review_session_id")
                                                      or frozen_payload.get("blind_session_id") or ""),
                                    "model_id": str(frozen_payload.get("evaluation_model_id") or "")})
                elif isinstance(frozen_payload.get("decisions"), Mapping):
                    sources.append({**source, "kind": "frozen_blind", "decisions_path": str(frozen_path)})
                else:
                    reader.warn(f"Unknown frozen blind snapshot structure: {frozen_path}")
        # Include legacy saved reviews without session metadata. No raw images
        # are opened, and this traversal is restricted to the local review root.
        legacy = ([root / "pairwise_review_decisions.json"] if network
                  else sorted(root.rglob("pairwise_review_decisions.json")))
        for decisions in legacy:
            if not decisions.is_file():
                continue
            if decisions.parent != root and decisions.parent.name != "manual_pairwise_review":
                # Frozen originals are admitted only through their checked
                # receipt. A legacy scan must not bypass receipt hash failures.
                continue
            if any(_key(Path(s["decisions_path"])) == _key(decisions) for s in sources):
                continue
            sources.append({"kind": "local_legacy", "decisions_path": str(decisions),
                            "manifest_path": ""})
    return sources


def _native_options(source: Mapping[str, Any], run: str, payload: Mapping[str, Any],
                    decision: Mapping[str, Any], reader: _Reader) -> list[Path]:
    manifest_path = _path(source.get("manifest_path"), Path(source["decisions_path"]).parent)
    manifest = reader.read(manifest_path, required=bool(manifest_path),
                           expected_sha256=str(source.get("manifest_sha256") or "")) or {}
    jobs = manifest.get("jobs") or []
    if not isinstance(jobs, list):
        reader.warn(f"Unknown jobs structure in {manifest_path}")
        return []
    jobs = [job for job in jobs if isinstance(job, Mapping)
            and str(job.get("run") or "").casefold() == run]
    candidate_id = payload.get("candidate_id")
    candidate = str(payload.get("candidate") or "").lower()
    if not candidate_id and candidate in ("a", "b"):
        candidate_id = (decision.get(f"candidate_{candidate}") or {}).get("candidate_id")
    if candidate_id:
        matched = [job for job in jobs if job.get("candidate_id") == candidate_id]
        if matched:
            jobs = matched
    paths: dict[str, Path] = {}
    for job in jobs:
        root = _path(job.get("analysis_source_review_root") or job.get("review_output_dir"),
                     manifest_path.parent if manifest_path else Path(source["decisions_path"]).parent)
        if root:
            path = root / "native_frames" / run / "manifest.json"
            paths[_key(path)] = path
    return list(paths.values())


def _frame_rows(native: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    return [dict(row) for row in (native or {}).get("frame_sources", [])
            if isinstance(row, Mapping)]


def _normalize_geometry(payload: dict[str, Any], native: Mapping[str, Any]) -> str | None:
    """Convert recorded analysis pixels only after native geometry is verified."""
    canvas_w, canvas_h = native["canvas_wh"]
    offset_x, offset_y, _, _ = native["content_box_xywh"]
    if not (_finite(canvas_w) and _finite(canvas_h)) or min(canvas_w, canvas_h) <= 0:
        return "Invalid native canvas geometry; overlay disabled"
    points = payload.get("points")
    if not points and payload.get("analysis_polygon_px"):
        try:
            points = [[(float(p[0]) + offset_x) / canvas_w,
                       (float(p[1]) + offset_y) / canvas_h]
                      for p in payload["analysis_polygon_px"]]
            payload["points"] = points
            payload["shape"] = "freehand"
        except (TypeError, ValueError, IndexError):
            return "Invalid historical analysis polygon; overlay disabled"
    if points:
        try:
            xy = [(float(p.get("x")), float(p.get("y"))) if isinstance(p, Mapping)
                  else (float(p[0]), float(p[1])) for p in points]
        except (TypeError, ValueError, IndexError):
            return "Invalid historical normalized polygon; overlay disabled"
        if len(xy) < 3 or any(not math.isfinite(v) or not 0 <= v <= 1 for p in xy for v in p):
            return "Historical polygon is outside the normalized canvas; overlay disabled"
        x0, y0 = min(x for x, _ in xy), min(y for _, y in xy)
        x1, y1 = max(x for x, _ in xy), max(y for _, y in xy)
        payload.update(x=x0, y=y0, width=x1 - x0, height=y1 - y0)
        payload["points"] = [[x, y] for x, y in xy]
    elif not all(key in payload for key in ("x", "y", "width", "height")):
        bbox = payload.get("analysis_bbox_xywh")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return "Historical region has no usable recorded geometry; overlay disabled"
        try:
            x, y, width, height = map(float, bbox)
            payload.update(x=(x + offset_x) / canvas_w, y=(y + offset_y) / canvas_h,
                           width=width / canvas_w, height=height / canvas_h, shape="rectangle")
        except (TypeError, ValueError):
            return "Invalid historical analysis bounding box; overlay disabled"
    values = [_finite(payload.get(key)) for key in ("x", "y", "width", "height")]
    if any(value is None for value in values):
        return "Historical region has non-finite geometry; overlay disabled"
    x, y, width, height = values
    if min(x, y) < 0 or min(width, height) <= 0 or x + width > 1 + 1e-9 or y + height > 1 + 1e-9:
        return "Historical region is outside the normalized canvas; overlay disabled"
    if payload.get("human_choice") and not payload.get("category"):
        from darsia.fluidflower.phase_labels import PHASE_CHOICE_TO_MANUAL_CATEGORY
        payload["category"] = PHASE_CHOICE_TO_MANUAL_CATEGORY.get(str(payload["human_choice"]), "other")
    return None


def _align(payload: Mapping[str, Any], original: Mapping[str, Any] | None,
           current: Mapping[str, Any] | None, *, geometry: bool) -> dict[str, Any]:
    if "ffac_selected_policy_a_sources" in (current or {}):
        from darsia.fluidflower.phase_annotation_binding import align_selected_annotation
        checked=deepcopy(dict(payload))
        if geometry and original:
            error=_normalize_geometry(checked,original)
            if error:
                return dict(payload=checked,displayable=False,mapped_by="camera_source_sha256",mapping_error=error,mapping_warnings=[])
        return align_selected_annotation(checked,original,current,geometry=geometry)
    result: dict[str, Any] = {"mapped_by": "", "displayable": False,
                              "mapping_error": None, "mapping_warnings": []}
    mapped = deepcopy(dict(payload))
    result["payload"] = mapped
    target_rows = _frame_rows(current)
    if not target_rows:
        result["mapping_error"] = "Current native timeline unavailable"
        return result
    for field in ("run", "dataset"):
        if original and (not original.get(field) or not (current or {}).get(field)):
            result["mapping_error"] = f"Original/current native {field} identity is unknown"
            return result
        if original and original.get(field) and current and current.get(field):
            if str(original[field]).casefold() != str(current[field]).casefold():
                result["mapping_error"] = f"Source and current native {field} differ"
                return result
    elapsed = _finite(payload.get("elapsed_h"))
    old_rows = _frame_rows(original)
    old = next((row for row in old_rows
                if str(row.get("frame_index")) == str(payload.get("frame_index"))), None)
    if old and elapsed is not None and not _same_time(old.get("elapsed_h"), elapsed):
        # Session bindings can change after the evidence was created. Its old
        # integer is not evidence of the frame in the new session's timeline.
        result["mapping_warnings"].append("Stored index disagrees with original timeline; matched explicit elapsed time")
        timed = [row for row in old_rows if _same_time(row.get("elapsed_h"), elapsed)]
        old = timed[0] if len(timed) == 1 else None
    photo = _photo(payload) or _photo(old or {})
    source_hash = str(payload.get("source_sha256") or (old or {}).get("source_sha256") or "")
    if elapsed is None and old:
        elapsed = _finite(old.get("elapsed_h"))
    if source_hash:
        choices = [row for row in target_rows if row.get("source_sha256") == source_hash]
        # Older/current manifests may omit source hashes; then the verified
        # source name is the available identity, subject to uniqueness/time.
        if not choices and any(row.get("source_sha256") for row in target_rows):
            result["mapping_error"] = "Source photo SHA-256 is absent from current timeline"
            return result
        method = "source_sha256"
    else:
        choices = []
        method = "source_name"
    if not choices and photo:
        choices = [row for row in target_rows if _photo(row) == photo]
        method = "source_name"
        if not choices:
            result["mapping_error"] = "Source photo is absent from current timeline"
            return result
    if not choices and not photo and elapsed is not None:
        choices = [row for row in target_rows if _same_time(row.get("elapsed_h"), elapsed)]
        method = "elapsed_h"
        result["mapping_warnings"].append("Source photo identity unavailable; unique elapsed-time match only")
    if elapsed is not None and len(choices) > 1:
        choices = [row for row in choices if _same_time(row.get("elapsed_h"), elapsed)]
    if len(choices) != 1:
        result["mapping_error"] = ("Ambiguous source photo/time in current timeline" if choices
                                   else "No uniquely verifiable source photo/time; old frame index was not reused")
        return result
    target = choices[0]
    try:
        target_index = int(target.get("frame_index"))
        if target_index < 0 or target_index != float(target.get("frame_index")):
            raise ValueError
    except (TypeError, ValueError):
        result["mapping_error"] = "Current native timeline has an invalid frame index"
        return result
    if elapsed is not None and not _same_time(target.get("elapsed_h"), elapsed):
        result["mapping_warnings"].append("Same source photo has a different elapsed time in current timeline")
    mapped["frame_index"] = target_index
    mapped["elapsed_h"] = target.get("elapsed_h")
    result["mapped_by"] = method
    if method == "elapsed_h":
        result["mapping_error"] = "Elapsed time matched but source photo identity is unavailable; display disabled"
        return result
    if geometry:
        if not original:
            result["mapping_error"] = "Original native geometry unavailable; overlay disabled"
            return result
        for field in _GEOMETRY:
            if not original.get(field) or not (current or {}).get(field):
                result["mapping_error"] = f"Native geometry unknown ({field}); overlay disabled"
                return result
            if original[field] != current[field]:
                result["mapping_error"] = f"Native geometry differs ({field}); overlay disabled"
                return result
        for field in _SPATIAL_CONTRACTS:
            if field in original or field in (current or {}):
                if original.get(field) != (current or {}).get(field):
                    result["mapping_error"] = f"Spatial contract differs or is missing ({field}); overlay disabled"
                    return result
        # Rig identity is spatial evidence; colour-processing revisions are not.
        old_contract = original.get("correction_contract") or {}
        new_contract = (current or {}).get("correction_contract") or {}
        for field in ("legacy_rig", "rig_sha256", "geometry_sha256"):
            if old_contract.get(field) and new_contract.get(field):
                if old_contract[field] != new_contract[field]:
                    result["mapping_error"] = f"Correction geometry differs ({field}); overlay disabled"
                    return result
        geometry_error = _normalize_geometry(mapped, original)
        if geometry_error:
            result["mapping_error"] = geometry_error
            return result
    result["displayable"] = True
    return result


def _record_id(run: str, kind: str, payload: Mapping[str, Any]) -> str:
    identifier = (payload.get("id") or payload.get("proposal_id"))
    if not identifier:
        identifier = (f"{payload.get('frame_index')}:{payload.get('candidate_id', '')}"
                      if kind == "frame" else _digest(payload)[:20])
    return f"{run}:{kind}:{identifier}"


def _verdict(decision: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(decision[key]) for key in _VERDICT_FIELDS if key in decision}


def _evidence_digest(payload: Mapping[str, Any], kind: str, decision: Mapping[str, Any]) -> str:
    if kind == "frame" or (kind in {"manual", "quiz"} and not payload.get("candidate_id")):
        return _digest({"payload": payload, "candidate_a": decision.get("candidate_a"),
                        "candidate_b": decision.get("candidate_b")})
    return _digest(payload)


def _items(decision: Mapping[str, Any], field: str, reader: _Reader,
           path: Path, run: str) -> list[dict[str, Any]]:
    raw = decision.get(field) or []
    if field == "frame_choices" and isinstance(raw, Mapping):
        raw = [{"frame_index": index, "choice": choice} for index, choice in raw.items()]
    if not isinstance(raw, list):
        reader.warn(f"Unknown {field} structure for {run} in {path}")
        return []
    if any(not isinstance(item, Mapping) for item in raw):
        reader.warn(f"Invalid {field} entries for {run} in {path}")
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def load_review_history(*, campaign_root: Path, current_native_paths: Mapping[str, Path],
                        review_roots: Sequence[Path] = (),
                        active_decision_path: Path | None = None,
                        retraction_projects: Mapping[str, Path] | None = None,
                        include_withdrawn: bool = False) -> dict[str, Any]:
    """Load independent historical display rows, preserving every revision/origin.

    ``annotations`` and ``frame_decisions`` contain already mapped payloads under
    ``annotation`` and ``decision``. Only ``displayable`` rows may be overlaid or
    navigated. ``run_decisions`` always describe the original candidate context.
    Exact active payloads are excluded; neither caller inputs nor files are mutated.
    """
    reader = _Reader()
    native_paths = {str(run).casefold(): Path(path) for run, path in current_native_paths.items()}
    current = {run: reader.read(path, required=True) for run, path in native_paths.items()}
    by_run = {run: {"annotations": [], "frame_decisions": [], "run_decisions": []}
              for run in native_paths}
    active_path = Path(active_decision_path) if active_decision_path else None
    active = reader.read(active_path, required=bool(active_path)) or {}
    active_decisions = active.get("decisions") or {}
    active_hashes: set[tuple[str, str, str]] = set()
    if isinstance(active_decisions, Mapping):
        for run, decision in active_decisions.items():
            if not isinstance(decision, Mapping):
                continue
            run = str(run).casefold()
            for field, kind in (("annotations", "manual"), ("phase_quiz_answers", "quiz"),
                                ("frame_choices", "frame")):
                for item in _items(decision, field, reader, active_path or Path("."), run):
                    active_hashes.add((run, kind, _evidence_digest(item, kind, decision)))
            active_hashes.add((run, "run", _digest(_verdict(decision))))
    sources = _model_sources(Path(campaign_root), reader)
    sources.extend(_session_sources(review_roots, active_path, reader))
    # Exact same file may be referenced by multiple model revisions: retain each
    # origin but read it once. Deduplicate identical evidence only, not revisions.
    rows_by_identity: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for source in sources:
        path = Path(source["decisions_path"])
        if active_path and _key(path) == _key(active_path):
            continue
        document = reader.read(path, required=True,
                               expected_sha256=str(source.get("decisions_sha256") or ""))
        if document is None:
            continue
        decisions = document.get("decisions")
        if not isinstance(decisions, Mapping):
            reader.warn(f"Unknown historical decisions structure in {path}")
            continue
        for run, decision in decisions.items():
            run = str(run).casefold()
            if run not in by_run:
                continue
            if not isinstance(decision, Mapping):
                reader.warn(f"Unknown decision structure for {run} in {path}")
                continue
            origin = {**source, "reviewer": str(decision.get("reviewer") or source.get("reviewer") or ""),
                      "updated_utc": str(decision.get("updated_utc") or document.get("updated_utc")
                                         or source.get("updated_utc") or ""),
                      "candidate_a": deepcopy(decision.get("candidate_a")),
                      "candidate_b": deepcopy(decision.get("candidate_b")),
                      "comparison_mode": decision.get("comparison_mode")}
            for field, kind, destination in (
                ("annotations", "manual", "annotations"),
                ("phase_quiz_answers", "quiz", "annotations"),
                ("frame_choices", "frame", "frame_decisions"),
                (None, "run", "run_decisions"),
            ):
                items = [_verdict(decision)] if field is None else _items(decision, field, reader, path, run)
                for payload in items:
                    if (run, kind, _evidence_digest(payload, kind, decision)) in active_hashes:
                        continue
                    if kind == "run" and not any(payload.get(key) for key in ("choice", "notes", "flags")):
                        continue
                    source_id = _record_id(run, kind, payload)
                    raw_digest = _digest(payload)
                    identity = (run, kind, source_id, _evidence_digest(payload, kind, decision))
                    existing = rows_by_identity.get(identity)
                    row: dict[str, Any] = {"source_id": source_id, "origins": [deepcopy(origin)],
                                           "payload_sha256": raw_digest, "historical": True,
                                           "kind": kind}
                    if kind == "run":
                        row["decision"] = deepcopy(payload)
                        row["displayable"] = True
                    else:
                        options = _native_options(source, run, payload, decision, reader)
                        alignments: list[dict[str, Any]] = []
                        for native_path in options or [None]:
                            native = reader.read(native_path, required=bool(native_path))
                            if "ffac_selected_policy_a_sources" in (current.get(run) or {}):
                                from darsia.fluidflower.phase_annotation_binding import resolve_camera_lineage
                                try:
                                    native=resolve_camera_lineage(native,reader.read)
                                except (KeyError,ValueError,TypeError,OSError) as exc:
                                    alignment=dict(payload=deepcopy(payload),displayable=False,mapped_by="camera_source_sha256",mapping_error=str(exc),mapping_warnings=[])
                                else:
                                    alignment=_align(payload,native,current.get(run),geometry=kind != "frame")
                            else:
                                alignment = _align(payload, native, current.get(run), geometry=kind != "frame")
                            alignment["original_native_path"] = str(native_path) if native_path else ""
                            alignments.append(alignment)
                        valid = [item for item in alignments if item["displayable"]]
                        identities = {_digest(item["payload"]) for item in valid}
                        if len(identities) > 1:
                            alignment = alignments[0]
                            alignment["displayable"] = False
                            alignment["mapping_error"] = "Ambiguous original native timelines for historical candidate"
                        else:
                            alignment = valid[0] if valid else alignments[0]
                        key = "decision" if kind == "frame" else "annotation"
                        row[key] = alignment.pop("payload")
                        row[f"original_{key}"] = deepcopy(payload)
                        row.update(alignment)
                        row["origin_mappings"] = [{"decisions_path": str(path),
                                                   "original_native_path": row["original_native_path"],
                                                   "displayable": row["displayable"],
                                                   "mapping_error": row["mapping_error"],
                                                   "frame_index": row[key].get("frame_index")}]
                        if row.get("mapping_error"):
                            reader.warn(f"{source_id}: {row['mapping_error']} ({path})")
                    if existing:
                        origins = existing["origins"]
                        if origin not in origins:
                            origins.append(deepcopy(origin))
                        mappings = existing.get("origin_mappings", []) + row.get("origin_mappings", [])
                        if kind != "run":
                            key = "decision" if kind == "frame" else "annotation"
                            valid_frames = {item["frame_index"] for item in mappings if item["displayable"]}
                            if len(valid_frames) > 1:
                                existing["displayable"] = False
                                existing["mapping_error"] = "Historical origins disagree on the source photo; display disabled"
                                reader.warn(f"{source_id}: {existing['mapping_error']}")
                            elif not existing["displayable"] and row["displayable"]:
                                existing.update(row)
                            existing["origin_mappings"] = mappings
                        existing["origins"] = origins
                        continue
                    rows_by_identity[identity] = row
                    by_run[run][destination].append(row)
    result = {"format": FORMAT, "by_run": by_run, "sources": deepcopy(sources),
              "warnings": reader.warnings, "current_native": deepcopy(current)}
    _apply_retraction_status(result, current, reader.selected_model_root,
                             retraction_projects=retraction_projects,
                             include_withdrawn=include_withdrawn)
    return result


def _apply_retraction_status(result: dict[str, Any], current: Mapping[str, Any],
                             selected_model_root: Path | None, *,
                             retraction_projects: Mapping[str, Path] | None,
                             include_withdrawn: bool) -> None:
    """Use the same logical targets as training, including every historic copy.

    The selected model's applied snapshot governs hiding. A newer live withdrawal
    remains visible as pending while inspecting an older model. Ledger errors
    disable new withdrawals and are surfaced; originals are never rewritten.
    """
    import darsia.fluidflower.phase_review_retractions as retractions

    result["retraction_context"] = {}
    result["withdrawn_by_run"] = {}
    contexts: dict[str, dict[str, Any]] = {}
    for run, rows in result["by_run"].items():
        series = str((current.get(run) or {}).get("dataset") or "").casefold().replace("_", "")
        if not series:
            result["warnings"].append(f"{run}: review series is unknown; historical withdrawal unavailable")
            continue
        if retraction_projects is not None and series not in retraction_projects:
            continue
        if series not in contexts:
            context: dict[str, Any] = {"series": series, "project_root": "", "error": "",
                                       "current_snapshot": None, "applied_snapshot": None}
            if selected_model_root is not None:
                try:
                    context["applied_snapshot"] = retractions.model_retractions(selected_model_root, series=series)
                except (OSError, ValueError, RuntimeError) as exc:
                    raise RuntimeError(f"Cannot verify the selected model's withdrawal snapshot: {exc}") from exc
            try:
                project = retractions.resolve_project_root(series,
                    None if retraction_projects is None else retraction_projects[series])
                context["project_root"] = str(project)
                context["current_snapshot"] = retractions.load_retractions(project, series=series)
            except (OSError, ValueError, RuntimeError) as exc:
                context["error"] = f"{type(exc).__name__}: {exc}"
                result["warnings"].append(f"{series}: withdrawal status could not be verified: {exc}")
            contexts[series] = context
        context = contexts[series]
        result["retraction_context"][run] = context
        _include_replacement_history(result, run, context, current.get(run))
        withdrawn = {name: [] for name in ("annotations", "frame_decisions", "run_decisions")}
        for name in withdrawn:
            visible = []
            for entry in rows[name]:
                status = retractions.history_retraction_status(entry, series=series, run=run,
                    current_snapshot=context["current_snapshot"], applied_snapshot=context["applied_snapshot"])
                if status:
                    entry["retraction_status"] = status
                if status == "applied":
                    withdrawn[name].append(entry)
                if status != "applied" or include_withdrawn:
                    visible.append(entry)
            rows[name] = visible
        result["withdrawn_by_run"][run] = withdrawn



def _include_replacement_history(result: dict[str, Any], run: str,
                                 context: Mapping[str, Any], current: Mapping[str, Any] | None) -> None:
    """Expose atomic replacement assessments without touching active annotations."""
    applied = {event["event_id"]: event for event in
               (context.get("applied_snapshot") or {}).get("events", [])}
    events = {**applied, **{event["event_id"]: event for event in
                          (context.get("current_snapshot") or {}).get("events", [])}}
    rows = result["by_run"][run]["annotations"]
    for event_id, event in events.items():
        replacement = event.get("replacement") or {}
        if event.get("action") != "reclassify" or replacement.get("run") != run:
            continue
        annotation = replacement.get("annotation") or {}
        if not annotation.get("id"):
            continue
        native = replacement.get("native_manifest") or {}
        combined = replacement.get("combined_manifest") or {}
        job = replacement.get("job") or {}
        from darsia.fluidflower.phase_review_retractions import ledger_folder
        event_path = (str(ledger_folder(Path(context["project_root"])) / (event_id + ".json"))
                      if context.get("project_root") else "")
        origin = {"kind": "evidence_action", "event_id": event_id,
                  "event_path": event_path, "decisions_path": event_path,
                  "manifest_path": combined.get("path", ""),
                  "manifest_sha256": combined.get("sha256", ""),
                  "reviewer": event.get("actor", ""), "updated_utc": event.get("created_utc", ""),
                  "model_id": job.get("candidate_id", ""), "model_root": job.get("phase_model_root", "")}
        status = "applied" if event_id in applied else "pending"
        matching = [row for row in rows
                    if (row.get("original_annotation") or row.get("annotation") or {}).get("id") == annotation["id"]]
        if matching:
            for row in matching:
                row.update(replacement_event_id=event_id, replacement_status=status,
                           replaces_target=deepcopy(event.get("target")))
                if origin not in row["origins"]:
                    row["origins"].append(deepcopy(origin))
            continue
        alignment = _align(annotation, replacement.get("native_manifest_data"), current, geometry=True)
        mapped = alignment.pop("payload")
        row = {"kind": "manual", "annotation": mapped, "original_annotation": deepcopy(annotation),
               "source_id": _record_id(run, "manual", annotation), "payload_sha256": _digest(annotation),
               "original_native_path": native.get("path", ""), "origins": [origin],
               "replacement_event_id": event_id, "replacement_status": status,
               "replaces_target": deepcopy(event.get("target")), **alignment}
        rows.append(row)
        if row["mapping_error"]:
            result.setdefault("warnings", []).append(f"{row['source_id']}: {row['mapping_error']}")


def apply_saved_retraction(payload: Mapping[str, Any], event: Mapping[str, Any]) -> dict[str, Any]:
    """Mark every copy immediately from an acknowledged append, without NAS IO."""
    import darsia.fluidflower.phase_review_retractions as retractions

    result = deepcopy(dict(payload))
    target = event["target"]
    series, run = target["series"], target["run"]
    context = (result.get("retraction_context") or {}).get(run)
    events = list(((context or {}).get("current_snapshot") or {}).get("events") or [])
    if not any(row.get("event_id") == event.get("event_id") for row in events):
        events.append(dict(event))
    snapshot = retractions.snapshot_from_events(series, events)
    if context is not None:
        context["current_snapshot"] = snapshot
        _include_replacement_history(result, run, context,
            (result.get("current_native") or {}).get(run))
    for entries in (result.get("by_run") or {}).get(run, {}).values():
        for entry in entries:
            if retractions.entry_matches_retractions(entry, series=series, run=run, snapshot=snapshot):
                if entry.get("retraction_status") != "applied":
                    entry["retraction_status"] = "pending"
    return result


__all__ = ["FORMAT", "load_review_history", "apply_saved_retraction"]
