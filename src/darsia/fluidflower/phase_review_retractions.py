"""Append-only review withdrawals and immutable training-time selection policies.

Original review documents and old model bundles are never edited here. A logical
manual/quiz ID is withdrawn across all its copies and revisions within one run
and series. The selected payload and its origins are retained as audit evidence.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

EVENT_FORMAT = "fluidflower-review-retraction-event-v1"
FORMAT = "fluidflower-review-retractions-v1"
DOCUMENT_KEY = "review_retractions_snapshot"
MODEL_FILE = "review_retractions.json"


class FreshCalibrationRequired(ValueError):
    """Fresh optical fitting cannot safely inherit this parent's learned gates."""


def _series(value):
    return "".join(c for c in str(value).casefold() if c.isalnum())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_project_root(series, project_root=None):
    from .series_paths import series_root

    root = Path(project_root).expanduser().resolve() if project_root else series_root(series)
    if not root.is_dir():
        raise FileNotFoundError(f"Review-evidence project unavailable: {root}")
    return root


def _snapshot(series, events):
    events = sorted(deepcopy(list(events)), key=lambda row: row["event_id"])
    targets = {}
    for event in events:
        if event.get("format") != EVENT_FORMAT or event.get("action") not in {"retract", "reclassify"}:
            raise ValueError("Unsupported review-evidence event")
        if not event.get("actor") or not event.get("created_utc"):
            raise ValueError("Review withdrawal requires an actor and timestamp")
        target = event["target"]
        if target.get("kind") not in {"manual", "quiz", "frame", "run"}:
            raise ValueError("Unknown review-evidence kind")
        if not all(target.get(key) for key in ("series", "run", "identifier")):
            raise ValueError("Incomplete review-evidence identity")
        if series != "all" and target["series"] != _series(series):
            raise ValueError("Review withdrawal belongs to another experimental series")
        targets[_digest(target)] = target
        if event["action"] == "reclassify":
            replacement = event.get("replacement") or {}
            annotation = replacement.get("annotation") or {}
            new_target = _target(target["series"], target["run"], "manual", annotation)
            if new_target == target or replacement.get("series") != target["series"] or replacement.get("run") != target["run"]:
                raise ValueError("Replacement must identify a new region in the same experiment")
            if not replacement.get("native_manifest_data") or not replacement.get("combined_manifest_data"):
                raise ValueError("Replacement lacks its frozen image/experiment provenance")
    children, replacement_ids = {}, {}
    for event in events:
        key = _digest(event["target"])
        if key in children and children[key] != event["event_id"]:
            raise ValueError("Concurrent evidence actions target the same region; resolve the ledger conflict")
        children[key] = event["event_id"]
        if event["action"] == "reclassify":
            new = _target(event["target"]["series"], event["target"]["run"], "manual", event["replacement"]["annotation"])
            new_key = _digest(new)
            if new_key in replacement_ids:
                raise ValueError("Two replacement events reuse an annotation ID")
            replacement_ids[new_key] = key
    for key in replacement_ids:
        visited = set()
        while key in replacement_ids:
            if key in visited:
                raise ValueError("Cyclic evidence replacement chain")
            visited.add(key)
            key = replacement_ids[key]
    result = {"format": FORMAT, "series": _series(series), "events": events,
              "targets": [targets[k] for k in sorted(targets)]}
    result["digest"] = _digest(result)
    return result


def empty_snapshot(series):
    return _snapshot(_series(series), [])


def snapshot_from_events(series, events):
    """Pure constructor, also usable after a successful write and failed refresh."""
    return _snapshot(_series(series), events)


def validate_snapshot(snapshot):
    if not isinstance(snapshot, Mapping) or snapshot.get("format") != FORMAT:
        raise ValueError("Invalid frozen review-withdrawal snapshot")
    expected = _snapshot(snapshot["series"], snapshot.get("events") or [])
    if dict(snapshot) != expected:
        raise ValueError("Frozen review-withdrawal snapshot digest/content mismatch")
    return expected


def ledger_folder(root, *, before_phase_model=False):
    """The withdraw/reclassify event folder of a series folder.

    It is ``events`` inside [paths] evidence_actions. ``before_phase_model``
    gives the folder the series used before it set [paths] phase_model.
    """
    from .series_paths import series_folder_path
    return series_folder_path(root, "evidence_actions", before_phase_model=before_phase_model) / "events"


def load_retractions(project_root=None, *, series):
    root = resolve_project_root(series, project_root)
    folder = ledger_folder(root)
    earlier = ledger_folder(root, before_phase_model=True)
    if earlier != folder and earlier.is_dir() and any(earlier.glob("*.json")):
        # Moving the ledger by configuration must not drop applied withdrawals.
        raise ValueError(f"Review-evidence events remain in {earlier}, but [paths] place the "
                         f"ledger in {folder}; move the events or set [paths] evidence_actions")
    # No recursive discovery and no local fallback: an unavailable registered
    # project cannot silently look like an empty training-withdrawal policy.
    events = []
    if folder.exists():
        for path in sorted(folder.glob("*.json")):
            event = _read(path)
            if path.stem != event.get("event_id"):
                raise ValueError(f"Review event filename/identity mismatch: {path}")
            events.append(event)
    return _snapshot(_series(series), events)


def _context(payload, decision=None):
    context = {}
    for key in ("candidate_id", "candidate_a", "candidate_b"):
        value = payload.get(key) or (decision or {}).get(key)
        if isinstance(value, Mapping):
            value = {k: value[k] for k in ("candidate_id", "model_id", "params_sha256") if value.get(k)}
        if value:
            context[key] = value
    return context


@lru_cache(maxsize=128)
def _native_timeline(path, modified, size):
    return _read(path)


def _frame_payload(payload, native_path):
    """Bind a verdict to its original photo, not its old/remapped frame number."""
    result = deepcopy(dict(payload))
    if payload.get("source_sha256"):
        result["_retraction_source_photo"] = {"sha256": str(payload["source_sha256"]).casefold()}
        return result
    if not native_path:
        return result
    path = Path(native_path)
    try:
        stat = path.stat()
        native = _native_timeline(str(path), stat.st_mtime_ns, stat.st_size)
    except (OSError, ValueError):
        return result
    rows = native.get("frame_sources") or []
    elapsed = payload.get("elapsed_h")
    try:
        if elapsed is not None and math.isfinite(float(elapsed)):
            matches = [row for row in rows if row.get("elapsed_h") is not None
                       and abs(float(row["elapsed_h"]) - float(elapsed)) <= 1e-6]
        else:
            matches = [row for row in rows if str(row.get("frame_index")) == str(payload.get("frame_index"))]
    except (TypeError, ValueError):
        return result
    if len(matches) == 1:
        row = matches[0]
        if row.get("source_sha256"):
            result["_retraction_source_photo"] = {"sha256": str(row["source_sha256"]).casefold()}
        elif row.get("source_name"):
            result["_retraction_source_photo"] = {"name": str(row["source_name"]).casefold()}
    return result


def _target(series, run, kind, payload, decision=None):
    if kind in {"manual", "quiz"}:
        identifier = payload.get("id") or payload.get("proposal_id")
        if not identifier:
            raise ValueError("Historical region has no stable ID; its revisions cannot safely be withdrawn")
        identifier = str(identifier)
        # Retired quiz pools are migrated by reconcile_phase_quiz_answers into
        # deterministic manual IDs. Both representations are the same evidence.
        if kind == "quiz" and payload.get("proposal_id"):
            candidate = str(payload.get("candidate") or "A").upper()
            identifier = "legacy-quiz-" + hashlib.sha256(
                f"{candidate}:{payload['proposal_id']}".encode("utf-8")).hexdigest()[:20]
        elif kind == "manual" and identifier.startswith("legacy-quiz-"):
            kind = "quiz"
    else:
        context = _context(payload, decision)
        if not context:
            raise ValueError("Historical verdict has no original candidate identity")
        if kind == "frame":
            photo = payload.get("_retraction_source_photo")
            if not photo:
                raise ValueError("Historical frame verdict has no verifiable source photo identity")
            context["source_photo"] = photo
        identifier = "context:" + _digest(context)
    return {"series": _series(series), "run": str(run).casefold(),
            "kind": kind, "identifier": identifier}


def _entry_target(entry, *, series, run):
    kind = entry.get("kind")
    if kind not in {"manual", "quiz", "frame", "run"}:
        raise ValueError("Select a historical annotation or verdict")
    payload = (entry.get("original_annotation") or entry.get("original_decision")
               or entry.get("decision") or entry.get("annotation") or {})
    origins = entry.get("origins") or []
    origin = origins[0] if origins else {}
    target_payload = (_frame_payload(payload, entry.get("original_native_path"))
                      if kind == "frame" else payload)
    return _target(series, run, kind, target_payload, origin), payload


def entry_matches_retractions(entry, *, series, run, snapshot):
    if not snapshot:
        return False
    checked = validate_snapshot(snapshot)
    try:
        target, _ = _entry_target(entry, series=series, run=run)
    except ValueError:
        return False
    return target in checked["targets"]


def _verified_entry(entry, *, series, run, root):
    target, payload = _entry_target(entry, series=series, run=run)
    origins = deepcopy(entry.get("origins") or [])
    if not origins:
        raise ValueError("Historical evidence has no verifiable source provenance")
    # Validate the selected record against at least one original source document.
    verified = []
    source_kind = entry["kind"]
    field = {"manual": "annotations", "quiz": "phase_quiz_answers", "frame": "frame_choices"}.get(source_kind)
    for origin in origins:
        path = Path(str(origin.get("decisions_path") or ""))
        if not path.is_file():
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        expected = origin.get("decisions_sha256")
        if expected and digest != expected:
            continue
        document = json.loads(data.decode("utf-8-sig"))
        if origin.get("kind") == "evidence_action":
            expected_path = ledger_folder(root) / (str(origin.get("event_id")) + ".json")
            if path.resolve() != expected_path.resolve() or document.get("event_id") != origin.get("event_id"):
                continue
            replacement = document.get("replacement") or {}
            if (document.get("action") == "reclassify" and replacement.get("annotation") == payload
                    and replacement.get("series") == _series(series) and replacement.get("run") == str(run).casefold()):
                _snapshot(_series(series), [document])
                verified.append({"path": str(path), "sha256": digest})
            continue
        decision = (document.get("decisions") or {}).get(str(run).casefold(), {})
        records = decision.get(field) or [] if field else [decision]
        if isinstance(records, Mapping):
            records = [{"frame_index": key, "choice": val} for key, val in records.items()]
        if source_kind == "run":
            found = all(decision.get(key) == value for key, value in payload.items())
        else:
            found = any(dict(item) == dict(payload) for item in records if isinstance(item, Mapping))
        if found:
            verified.append({"path": str(path), "sha256": digest})
    if not verified:
        raise ValueError("Selected historical record changed or cannot be verified; refresh history")
    return target, payload, origins, verified


def _new_event(entry, *, series, run, actor, root, reason, action):
    if not str(actor).strip():
        raise ValueError("A reviewer is required to change saved evidence")
    target, payload, origins, verified = _verified_entry(entry, series=series, run=run, root=root)
    event = {"format": EVENT_FORMAT, "event_id": str(uuid.uuid4()), "action": action,
             "created_utc": datetime.now(timezone.utc).isoformat(), "actor": actor,
             "reason": str(reason), "target": target, "scope": "all_copies_and_revisions",
             "selected_payload": deepcopy(payload), "selected_payload_sha256": _digest(payload),
             "selected_source_kind": entry["kind"],
             "origins": origins, "verified_sources": verified,
             "original_context": _context(payload, origins[0]),
             "meaning": "Exclude from future training; hide ordinary history once a new model applies this event"}
    if entry.get("replacement_event_id"):
        event["replaces_event_id"] = str(entry["replacement_event_id"])
    return event


@contextmanager
def _ledger_write(root):
    folder = ledger_folder(root)
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / ".write.lock"
    try:
        stream = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ValueError("Another evidence change is being saved; refresh and retry") from exc
    try:
        with stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink()


def _append_event(root, event):
    folder = ledger_folder(root)
    folder.mkdir(parents=True, exist_ok=True)
    target_path = folder / (event["event_id"] + ".json")
    temporary = folder / (event["event_id"] + ".pending")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(event, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # UUID destination plus exclusive temporary file; rename publishes one whole
    # event and never rewrites another reviewer's event or the source document.
    if target_path.exists():
        raise FileExistsError(target_path)
    temporary.rename(target_path)
    return event


def retract_history_entry(entry, *, series, run, actor, project_root=None, reason=""):
    root = resolve_project_root(series, project_root)
    event = _new_event(entry, series=series, run=run, actor=actor, root=root, reason=reason, action="retract")
    with _ledger_write(root):
        current = load_retractions(root, series=series)
        prior = next((e for e in current["events"] if e["target"] == event["target"]), None)
        if prior:
            if prior["action"] != "retract":
                raise ValueError("This region already has a replacement; select the current replacement to withdraw it")
            return prior
        _snapshot(_series(series), [*current["events"], event])
        return _append_event(root, event)


def _photo_row(payload, native):
    rows = native.get("frame_sources") or []
    matches = [row for row in rows if str(row.get("frame_index")) == str(payload.get("frame_index"))]
    elapsed = payload.get("elapsed_h")
    if elapsed is not None and (len(matches) != 1 or abs(float(matches[0].get("elapsed_h", math.inf)) - float(elapsed)) > 1e-6):
        matches = [row for row in rows if abs(float(row.get("elapsed_h", math.inf)) - float(elapsed)) <= 1e-6]
    if len(matches) != 1:
        raise ValueError("Region does not identify exactly one source photograph")
    return matches[0]


def _geometry(payload, native):
    """Canonical canvas geometry, preserving every polygon vertex."""
    width, height = map(float, native["canvas_wh"])
    ox, oy, _, _ = native["content_box_xywh"]
    points = payload.get("points") or []
    if not points and payload.get("analysis_polygon_px"):
        points = [[(float(x) + ox) / width, (float(y) + oy) / height]
                  for x, y in payload["analysis_polygon_px"]]
    if points:
        xy = [[float(p["x"]), float(p["y"])] if isinstance(p, Mapping)
              else [float(p[0]), float(p[1])] for p in points]
        if len(xy) < 3 or any(not math.isfinite(v) or not 0 <= v <= 1 for p in xy for v in p):
            raise ValueError("Invalid normalized region polygon")
        return {"shape": "freehand", "points": xy}
    if all(k in payload for k in ("x", "y", "width", "height")):
        box = [float(payload[k]) for k in ("x", "y", "width", "height")]
    else:
        x, y, w, h = map(float, payload["analysis_bbox_xywh"])
        box = [(x + ox) / width, (y + oy) / height, w / width, h / height]
    x, y, w, h = box
    if any(not math.isfinite(v) for v in box) or min(x, y) < 0 or min(w, h) <= 0 or x + w > 1 + 1e-9 or y + h > 1 + 1e-9:
        raise ValueError("Invalid normalized region rectangle")
    return {"shape": "rectangle", "box": box}


def _same_geometry(left, right):
    if left["shape"] != right["shape"]:
        return False
    a, b = left.get("points") or [left["box"]], right.get("points") or [right["box"]]
    return len(a) == len(b) and all(len(x) == len(y) and all(abs(v - w) <= 1e-10 for v, w in zip(x, y)) for x, y in zip(a, b))


def _same_photo(left, right):
    for key in ("source_sha256", "corrected_array_sha256"):
        if left.get(key) and right.get(key):
            return left[key] == right[key]
    def name(row):
        return str(row.get("source_name") or str(row.get("source_path") or "").replace("\\", "/").split("/")[-1]).casefold()
    return bool(name(left)) and name(left) == name(right)


def _verify_replacement_copy(payload, event, native_path):
    replacement = event["replacement"]
    original = replacement["annotation"]
    # Timeline alignment is permitted, but copied provenance cannot authorize a
    # new classification, reviewer, confidence, scope, or annotation geometry.
    mutable = {"frame_index", "elapsed_h", "original_frame_index", "review_evidence_action", "training_sampling"}
    if {k: v for k, v in payload.items() if k not in mutable} != {k: v for k, v in original.items() if k not in mutable}:
        raise ValueError("A copied replacement differs from its immutable human classification; create a new evidence action")
    if native_path is None:
        raise ValueError("Replacement copy has no verifiable native image source")
    native = _read(native_path)
    source_native = replacement["native_manifest_data"]
    for key in ("run", "dataset", "canvas_wh", "content_box_xywh", "analysis_image_shape_hw",
                "geometry_sha256", "geometry_fingerprint", "spatial_transform_sha256", "registration_sha256",
                "crop_sha256", "geometry_contract", "spatial_contract"):
        if native.get(key) != source_native.get(key):
            raise ValueError("Replacement copy changed native geometry without a supported evidence-action mapping")
    frame = payload.get("frame_index")
    elapsed = payload.get("elapsed_h")
    if (type(frame) is not int or frame < 0 or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)):
        raise ValueError("Replacement copy has an invalid frame index or elapsed time")
    resolved = _photo_row(payload, native)
    if resolved.get("frame_index") != frame or abs(float(resolved.get("elapsed_h", math.inf)) - elapsed) > 1e-6:
        raise ValueError("Replacement copy frame index and source-photo time disagree")
    if not _same_photo(resolved, replacement["source_photo"]):
        raise ValueError("Replacement copy moved to another source photograph")


def reclassify_history_entry(entry, *, series, run, actor, replacement_annotation,
                             current_manifest_path, current_review_job,
                             project_root=None, reason=""):
    """Atomically supersede one historical region with a new human classification.

    The one event is both the withdrawal and the new immutable training evidence.
    Nothing is appended to the current editable review session.
    """
    root = resolve_project_root(series, project_root)
    event = _new_event(entry, series=series, run=run, actor=actor, root=root, reason=reason, action="reclassify")
    if entry.get("kind") not in {"manual", "quiz"}:
        raise ValueError("Only a historical region can be reclassified")
    manifest_path = Path(current_manifest_path).resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    name, dataset = str(run).casefold(), _series(series)
    jobs = [row for row in manifest.get("jobs") or [] if str(row.get("run")).casefold() == name
            and _series(row.get("dataset")) == dataset
            and row.get("candidate_id") == current_review_job.get("candidate_id")]
    if len(jobs) != 1:
        raise ValueError("Current review job is not uniquely registered in its manifest")
    job = deepcopy(jobs[0])
    review_root = Path(job.get("analysis_source_review_root") or job.get("review_output_dir") or "")
    if not review_root.is_absolute():
        review_root = manifest_path.parent / review_root
    native_path = review_root / "native_frames" / name / "manifest.json"
    native_bytes = native_path.read_bytes()
    native = json.loads(native_bytes.decode("utf-8-sig"))
    if str(native.get("run")).casefold() != name or _series(native.get("dataset")) != dataset:
        raise ValueError("Current native image belongs to another experiment")
    original_path = Path(entry.get("original_native_path") or "")
    original = None
    for source in event["verified_sources"]:
        document = _read(source["path"])
        prior = document.get("replacement") or {}
        if document.get("action") == "reclassify" and prior.get("annotation") == event["selected_payload"]:
            original = prior.get("native_manifest_data")
            break
    if original is None:
        original = _read(original_path)
        # The geometry source must be registered by a verified historical origin.
        bound = False
        for origin in event["origins"]:
            path = Path(origin.get("manifest_path") or "")
            if not path.is_file() or (origin.get("manifest_sha256") and _file_hash(path) != origin["manifest_sha256"]):
                continue
            for old_job in _read(path).get("jobs") or []:
                if str(old_job.get("run")).casefold() != name or _series(old_job.get("dataset")) != dataset:
                    continue
                old_root = Path(old_job.get("analysis_source_review_root") or old_job.get("review_output_dir") or "")
                if not old_root.is_absolute():
                    old_root = path.parent / old_root
                if (old_root / "native_frames" / name / "manifest.json").resolve() == original_path.resolve():
                    bound = True
        if not bound:
            raise ValueError("Historical native geometry is not bound to its original review manifest")
    for key in ("run", "dataset", "canvas_wh", "content_box_xywh", "analysis_image_shape_hw",
                "geometry_sha256", "geometry_fingerprint", "spatial_transform_sha256", "registration_sha256",
                "crop_sha256", "geometry_contract", "spatial_contract"):
        if key in original or key in native:
            if original.get(key) != native.get(key):
                raise ValueError("Historical and current native geometry differ: " + key)
        elif key in {"run", "dataset", "canvas_wh", "content_box_xywh", "analysis_image_shape_hw"}:
            raise ValueError("Historical native geometry is incomplete: " + key)
    annotation = deepcopy(dict(replacement_annotation))
    for key in ("evidence_audit", "training_sampling", "review_evidence_action"):
        annotation.pop(key, None)
    if not annotation.get("id") or str(annotation["id"]).startswith("legacy-quiz-"):
        raise ValueError("Replacement requires a new independent annotation ID")
    if not annotation.get("category"):
        raise ValueError("Replacement classification category is required")
    old_photo = _photo_row(event["selected_payload"], original)
    new_photo = _photo_row(annotation, native)
    if type(annotation.get("frame_index")) is not int or new_photo["frame_index"] != annotation["frame_index"]:
        raise ValueError("Replacement uses a stale frame index")
    if old_photo.get("source_sha256") and new_photo.get("source_sha256"):
        same_photo = old_photo["source_sha256"] == new_photo["source_sha256"]
    elif old_photo.get("corrected_array_sha256") and new_photo.get("corrected_array_sha256"):
        same_photo = old_photo["corrected_array_sha256"] == new_photo["corrected_array_sha256"]
    else:
        def photo(row):
            return str(row.get("source_name") or str(row.get("source_path") or "").replace("\\", "/").split("/")[-1]).casefold()
        same_photo = bool(photo(old_photo)) and photo(old_photo) == photo(new_photo)
        if sum(photo(row) == photo(new_photo) for row in native.get("frame_sources") or []) != 1:
            same_photo = False
    if not same_photo:
        raise ValueError("Replacement is not on the original source photograph")
    if not _same_geometry(_geometry(event["selected_payload"], original), _geometry(annotation, native)):
        raise ValueError("Reclassification may change the class, not the historical region geometry")
    if annotation.get("candidate_id") and job.get("candidate_id") and annotation["candidate_id"] != job["candidate_id"]:
        raise ValueError("Replacement classification identifies another current candidate")
    annotation.update(reviewer=str(actor).strip(), updated_utc=event["created_utc"])
    job.update(analysis_source_review_root=str(review_root.resolve()))
    combined = {key: deepcopy(value) for key, value in manifest.items() if key != "jobs"}
    combined["jobs"] = [deepcopy(job)]
    event["replacement"] = dict(annotation=annotation, series=dataset, run=name, job=job,
        combined_manifest=dict(path=str(manifest_path), sha256=hashlib.sha256(manifest_bytes).hexdigest()),
        native_manifest=dict(path=str(native_path.resolve()), sha256=hashlib.sha256(native_bytes).hexdigest()),
        native_manifest_data=native, combined_manifest_data=combined, source_photo=deepcopy(new_photo),
        selection_kind="explicit_historical_reclassification", independent_holdout=False,
        inclusion_scope="Only this new human-classified region; evaluation and other labels remain unchanged")
    with _ledger_write(root):
        current = load_retractions(root, series=series)
        prior = next((e for e in current["events"] if e["target"] == event["target"]), None)
        if prior:
            if prior["action"] == "reclassify":
                # An acknowledged event may reach a retry after an interrupted UI.
                # Return the existing result, never append a competing sibling.
                earlier = prior["replacement"]
                semantics = ("category", "human_choice", "training_phase_class", "annotation_scope", "candidate_id")
                if (str(prior["actor"]).strip() == str(actor).strip()
                        and all(earlier["annotation"].get(k) == annotation.get(k) for k in semantics)
                        and _same_geometry(_geometry(earlier["annotation"], earlier["native_manifest_data"]), _geometry(annotation, native))
                        and earlier["native_manifest"]["sha256"] == event["replacement"]["native_manifest"]["sha256"]
                        and earlier["annotation"].get("frame_index") == annotation.get("frame_index")):
                    return prior
                raise ValueError("This region already has another replacement; refresh and edit the current replacement")
            raise ValueError("This historical region is already withdrawn; refresh history")
        new_target = _target(dataset, name, "manual", annotation)
        if new_target in current["targets"] or any(new_target == _target(dataset, name, "manual", e["replacement"]["annotation"])
                for e in current["events"] if e.get("action") == "reclassify"):
            raise ValueError("Replacement annotation ID has already been used")
        _snapshot(dataset, [*current["events"], event])
        return _append_event(root, event)


def active_replacements(snapshot):
    checked = validate_snapshot(snapshot)
    return [deepcopy(event) for event in checked["events"] if event["action"] == "reclassify"
            and _target(event["target"]["series"], event["target"]["run"], "manual",
                        event["replacement"]["annotation"]) not in checked["targets"]]


def replacement_documents(snapshot):
    """One authoritative input per active replacement; no live ledger reads."""
    rows = []
    for event in active_replacements(snapshot):
        replacement = event["replacement"]
        record = replacement["native_manifest"]
        if _file_hash(record["path"]) != record["sha256"]:
            raise ValueError("Replacement native image manifest changed after human classification")
        annotation = deepcopy(replacement["annotation"])
        annotation["review_evidence_action"] = {"event_id": event["event_id"], "annotation_sha256": _digest(replacement["annotation"])}
        manifest = deepcopy(replacement["combined_manifest_data"])
        manifest.update(include_in_training=True, requires_explicit_training_selection=False,
            evidence_role="explicit_historical_reclassification", independent_holdout=False,
            evidence_action_event_id=event["event_id"])
        for job in manifest["jobs"]:
            job.update(include_in_training=True, requires_explicit_training_selection=False,
                evidence_role="explicit_historical_reclassification", independent_holdout=False)
        decisions = {"decisions": {replacement["run"]: {"reviewer": event["actor"], "annotations": [annotation]}},
                     DOCUMENT_KEY: deepcopy(snapshot), "evidence_action_event_id": event["event_id"]}
        rows.append((event, decisions, manifest))
    return rows


def materialize_replacements(snapshot, output_root):
    """Write derived inputs only inside the caller's NEW training transaction."""
    result = []
    for event, decisions, manifest in replacement_documents(snapshot):
        folder = Path(output_root) / event["event_id"]
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, payload in (("decisions.json", decisions), ("manifest.json", manifest), ("event.json", event)):
            path = folder / name
            if path.exists():
                if _read(path) != payload:
                    raise ValueError("Frozen replacement training input changed")
            else:
                path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            paths.append(path)
        result.append((event, paths[0], paths[1], paths[2]))
    return result


def model_retractions(model_root, *, series=None):
    root = Path(model_root)
    path = root / MODEL_FILE
    request_path = root / "training_request.json"
    request = _read(request_path) if request_path.is_file() else {}
    record = request.get("review_retractions") or {}
    if not path.is_file():
        if record:
            raise FileNotFoundError(f"Model withdrawal snapshot missing: {path}")
        return None
    if record and _file_hash(path) != record.get("sha256"):
        raise ValueError("Model withdrawal snapshot file digest mismatch")
    snapshot = validate_snapshot(_read(path))
    if series and snapshot["series"] not in {_series(series), "all"}:
        raise ValueError("Model withdrawal snapshot belongs to another series")
    return snapshot


def history_retraction_status(entry, *, series, run, current_snapshot, applied_snapshot):
    if entry_matches_retractions(entry, series=series, run=run, snapshot=applied_snapshot):
        return "applied"
    if entry_matches_retractions(entry, series=series, run=run, snapshot=current_snapshot):
        return "pending"
    return ""


def snapshot_for_bundles(bundles, settings=None):
    settings = settings or {}
    # Settings are commonly copied from a parent. Their old snapshot must never
    # override the current ledger when preparing a NEW job. Workers read a frozen
    # model snapshot; callers freezing a transaction pass it explicitly instead.
    series = sorted({_series(job.get("dataset")) for _, path in bundles
                     for job in (_read(path).get("jobs") or []) if job.get("dataset")})
    if not series:
        if any(_read(path).get("decisions") for path, _ in bundles):
            raise ValueError("Training evidence does not identify its experimental series")
        return empty_snapshot("all")
    roots = settings.get("review_project_roots") or {}
    snapshots = [load_retractions(roots.get(name) or settings.get("project_root"), series=name) for name in series]
    fresh = snapshots[0] if len(snapshots) == 1 else _snapshot("all", [e for s in snapshots for e in s["events"]])
    prior = [settings.get(DOCUMENT_KEY)] + [_read(path).get(DOCUMENT_KEY) for path, _ in bundles]
    for snapshot in prior:
        if snapshot:
            _assert_policy_extends(fresh, snapshot)
    return fresh


def _assert_policy_extends(current, previous):
    fresh, prior = validate_snapshot(current), validate_snapshot(previous)
    known = {row["event_id"]: row for row in fresh["events"]}
    series = fresh["series"]
    if any(known.get(event["event_id"]) != event for event in prior["events"]
           if series == "all" or event["target"]["series"] == series):
        raise ValueError("The project ledger is missing or changed evidence actions already applied by the parent; restore the correct ledger before training")


def filter_document(document, manifest, snapshot):
    """Return an audited derived view. Never mutate the caller or a review file."""
    snapshot = validate_snapshot(snapshot)
    result = deepcopy(document)
    if not snapshot["targets"]:
        return result, []
    by_run = {}
    native_by_run = {}
    for job in manifest.get("jobs") or []:
        run, series = str(job.get("run") or "").casefold(), _series(job.get("dataset"))
        if run in by_run and by_run[run] != series:
            raise ValueError("Run is ambiguous across experimental series")
        by_run[run] = series
        review_root = job.get("analysis_source_review_root") or job.get("review_output_dir")
        if review_root:
            native_by_run[run] = Path(review_root) / "native_frames" / run / "manifest.json"
    removed = []
    replacements = {_digest(_target(e["target"]["series"], e["target"]["run"], "manual", e["replacement"]["annotation"])): e
                    for e in active_replacements(snapshot)}
    for run, decision in (result.get("decisions") or {}).items():
        series = by_run.get(str(run).casefold())
        if not series:
            continue
        try:
            run_target = _target(series, run, "run", decision, decision)
        except ValueError:
            run_target = None
        if run_target in snapshot["targets"]:
            removed.append({"target": run_target, "payload_sha256": _digest(decision)})
            for key in ("choice", "winner_candidate_id", "legacy_run_choice", "notes", "flags"):
                decision.pop(key, None)
        for field, kind in (("annotations", "manual"), ("phase_quiz_answers", "quiz"), ("frame_choices", "frame")):
            records = decision.get(field) or []
            if isinstance(records, Mapping):
                records = [{"frame_index": key, "choice": val} for key, val in records.items()]
            kept = []
            for payload in records:
                try:
                    target_payload = (_frame_payload(payload, native_by_run.get(str(run).casefold()))
                                      if kind == "frame" else payload)
                    target = _target(series, run, kind, target_payload, decision)
                except ValueError:
                    kept.append(payload)
                    continue
                if target in snapshot["targets"]:
                    removed.append({"target": target, "payload_sha256": _digest(payload)})
                else:
                    replacement = replacements.get(_digest(target))
                    provenance = payload.get("review_evidence_action") or {}
                    if replacement and (provenance.get("event_id") != replacement["event_id"]
                            or provenance.get("annotation_sha256") != _digest(replacement["replacement"]["annotation"])):
                        removed.append({"target": target, "payload_sha256": _digest(payload),
                                        "reason": "Replaced by the authoritative evidence-action input"})
                        continue
                    if replacement:
                        _verify_replacement_copy(payload, replacement, native_by_run.get(str(run).casefold()))
                    # Parent-prediction-guided extra sampling is itself fitted
                    # influence. Fresh withdrawal fits use uniform region sampling.
                    retained = dict(payload)
                    retained.pop("training_sampling", None)
                    kept.append(retained)
            if field in decision:
                decision[field] = kept
    if snapshot["targets"]:
        result[DOCUMENT_KEY] = deepcopy(snapshot)
    return result, removed


def write_model_snapshot(output_root, snapshot):
    snapshot = validate_snapshot(snapshot)
    path = Path(output_root) / MODEL_FILE
    if path.exists():
        if validate_snapshot(_read(path)) != snapshot:
            raise ValueError("Cannot replace a frozen model withdrawal policy")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(snapshot, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    return {"path": str(path), "sha256": _file_hash(path), "digest": snapshot["digest"]}


def require_fresh_fit(snapshot, settings):
    if not validate_snapshot(snapshot)["targets"]:
        return
    reused = [key for key in ("resume_completed_heads", "completed_fit_source", "imported_completed_heads", "inherited_heads") if settings.get(key)]
    if reused:
        raise ValueError("Withdrawn evidence requires fresh fitting; inherited/resumed fits are prohibited: " + ", ".join(reused))


def assert_reuse_policy_current(model_root):
    """Reject reuse of frozen samples/weights created before a withdrawal."""
    root = Path(model_root)
    request = _read(root / "training_request.json")
    settings = request.get("settings") or {}
    series = settings.get("series") or []
    if isinstance(series, str):
        series = [series]
    if not series and (root / "summary.json").is_file():
        series = (_read(root / "summary.json").get("inventory") or {}).get("datasets") or []
    if not series:
        raise ValueError("Cannot verify the withdrawal policy of reused samples without a series")
    applied = model_retractions(root)
    prior_targets = applied["targets"] if applied else []
    roots = settings.get("review_project_roots") or {}
    snapshots = []
    for name in series:
        current = load_retractions(roots.get(_series(name)) or settings.get("project_root"), series=name)
        snapshots.append(current)
        if applied:
            _assert_policy_extends(current, applied)
        if any(target not in prior_targets for target in current["targets"]):
            raise ValueError("Frozen samples or model weights predate review withdrawals; "
                             "prepare a fresh full fit and calibration from the current evidence selection")
    # Return the exact policy checked at transaction entry. In particular, a
    # legacy clone must not read a newer ledger after copying old samples.
    return applied if applied is not None else (snapshots[0] if len(snapshots) == 1
        else _snapshot("all", [e for snapshot in snapshots for e in snapshot["events"]]))


def assert_calibration_can_be_inherited(parent, snapshot):
    """Do not promise forgetting while silently copying an opaque fitted gate."""
    snapshot = validate_snapshot(snapshot)
    parent = Path(parent)
    applied = model_retractions(parent)
    if applied:
        _assert_policy_extends(snapshot, applied)
    if not snapshot["targets"]:
        return
    prior = applied["targets"] if applied else []
    pending = [target for target in snapshot["targets"] if target not in prior and target["kind"] in {"manual", "quiz"}]
    if not pending:
        return
    # Reference, regional and boundary fits may have inherited human evidence via
    # multiple ancestors without complete per-region receipts. Require an audited
    # fresh calibration instead of treating missing provenance as non-exposure.
    fitted_markers = ("phase_native_reference.json", "phase_facies_reference.json", "phase_reviewed_reference.json",
                      "phase_reference_support.json", "reference_calibration", "boundary_code_snapshot",
                      "aqueous_fringe_v3_code_snapshot")
    if any((parent / name).exists() for name in fitted_markers):
        raise FreshCalibrationRequired("Review withdrawals require fresh reference/physical calibration before continuing this parent. "
                         "Its inherited fitted calibration may contain withdrawn human evidence; annotation filtering alone is insufficient.")
