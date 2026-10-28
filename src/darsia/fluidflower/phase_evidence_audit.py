"""Immutable queues for human review of model/evidence disagreements, without Qt."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .project import sha256_file, utc_now
from .review_storage import resolve_review_path

FORMAT = "darsia-phase-evidence-audit-v1"


def read(path):
    return json.loads(resolve_review_path(path).read_text(encoding="utf-8-sig"))


def payload_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False, default=str).encode()).hexdigest()


def disagreement_reasons(row, threshold=.75):
    """Rank conflicts with region constraints, not label-error probabilities."""
    if not 0 <= threshold <= 1:
        raise ValueError("Disagreement threshold must be between zero and one")
    c, old, new = (float(row[k]) for k in ("co2_fraction", "old_gas_fraction", "new_gas_fraction"))
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (c, old, new)):
        raise ValueError("Invalid phase-area fraction")
    if old > c or new > c:
        raise ValueError("Gas area cannot exceed total CO2 area")
    label = row["class"]
    reasons = []
    if label == "gas" and old > 0 and new == 0:
        reasons.append(("temporal_rule_conflict", 1., "Experimental rule removes all detected gas from a gas-labeled region"))
    aqueous = float(row.get("aqueous_fraction", c-old))
    if not math.isfinite(aqueous) or not 0 <= aqueous <= c:
        raise ValueError("Invalid aqueous phase-area fraction")
    required = {
        "gas_present_region": {"gas": old},
        "aqueous_present_region": {"aqueous": aqueous},
        "aqueous_present_gas_absent_region": {"aqueous": aqueous},
        "co2_present_region": {"co2": c},
        "mixed_gas_aqueous": {"gas": old, "aqueous": aqueous},
        "mixed_aqueous_water": {"aqueous": aqueous, "water": 1-c},
        "mixed_gas_aqueous_water": {"gas": old, "aqueous": aqueous, "water": 1-c},
    }
    for phase, area in required.get(label, {}).items():
        if area == 0:
            reasons.append((f"missing_required_{phase}", 1.,
                f"No {phase} pixels are detected although the regional label requires their presence"))
    score = None
    if label == "gas":
        score = 1-old
    elif label == "aqueous":
        score = 1-aqueous
    elif label in {"gas_absent_region", "aqueous_present_gas_absent_region", "mixed_aqueous_water"}:
        score = old
    elif label == "aqueous_absent_region":
        score = aqueous
    elif label in {"water", "co2_absent_region", "current_experiment_co2_artifact"}:
        score = c
    elif label == "mixed_gas_aqueous":
        score = 1-c  # Closed two-phase mixture excludes untouched water.
    # A review prompt, not a constraint violation: a presence annotation may
    # legitimately contain mostly water, so never turn it into a full-box label.
    if label in {"aqueous_present_region", "aqueous_present_gas_absent_region"} and 0 < aqueous < .5:
        priority = 1-aqueous
        if priority >= threshold:
            reasons.append(("low_aqueous_coverage", priority,
                f"Only {aqueous:.1%} of this aqueous-presence region is detected. "
                "Inspect weak colour changes; this is a review prompt, not proof the label or model is wrong"))
    # Presence-only labels allow any nonzero area; mixtures have no target fraction.
    if score is not None and score > 0 and score >= threshold:
        reasons.append(("model_label_disagreement", score, "Evaluated phase-area fraction conflicts with the regional label"))
    dominance = row.get("dominant_phase")
    if label == "mixed_gas_aqueous" and dominance in {"gas", "aqueous"}:
        area = old if dominance == "gas" else aqueous
        if area <= .5 and 1-area >= threshold:
            reasons.append(("dominance_violation", 1-area,
                f"The regional label requires a strict {dominance} majority, but its visible fraction is {area:.3f}"))
    return reasons


def verified_review_binding(parent, source=None, *, saved_comparison=False):
    """Validate a completed native campaign or an exact composite collection.

    Viewing saved maps may use an older recorded runtime, but only while the
    model's public profile exactly matches the profile bound to those maps.
    Derived private runtime caches are not part of the serialized profile.
    Packages without a profile snapshot require the exact current runtime hash,
    which binds the full profile as well as its implementation and libraries.
    Correction and inference retain the current-runtime check by default.
    """
    from .phase_validation import _model_identity
    from .phase_runtime import load_physics_profile, physics_runtime_identity
    parent = resolve_review_path(parent).resolve()
    source = resolve_review_path(source if source is not None else parent / "validation").resolve()
    manifest_path = source / "postflight_manifest.json"
    manifest = read(manifest_path)
    if manifest.get("format") == "fluidflower-phase-review-collection-v1":
        from darsia.fluidflower.tools.phase_review_collections import validated_collection
        return validated_collection(
            manifest_path, parent, full=True, saved_comparison=saved_comparison
        )["binding"]
    model_id, identity = _model_identity(parent)
    binding = read(source / "validation_binding.json")
    if not binding.get("completed_utc"):
        raise ValueError("Native model predictions are not complete")
    if (binding.get("model_id"), binding.get("model_sha256")) != (model_id, identity):
        raise ValueError("Native predictions belong to a different model")
    profile = load_physics_profile(parent)
    if saved_comparison:
        if "physics_profile" in binding:
            def public_profile(value):
                return ({key: item for key, item in value.items() if not key.startswith("_")}
                        if isinstance(value, dict) else value)
            profile_matches = public_profile(binding["physics_profile"]) == public_profile(profile)
        else:
            profile_matches = binding.get("physics_runtime") == physics_runtime_identity(profile)
        if not profile_matches:
            raise ValueError("Saved native maps do not match this model's physical profile")
    elif binding.get("physics_runtime") != (physics_runtime_identity(profile) if profile is not None else None):
        raise ValueError("Native predictions belong to different physical rules")
    if (manifest.get("model_id"), manifest.get("model_sha256"), manifest.get("physics_runtime")) != (
            model_id, identity, binding.get("physics_runtime")):
        raise ValueError("Native prediction manifest differs from its binding")
    if saved_comparison and manifest.get("physics_profile") != binding.get("physics_profile"):
        raise ValueError("Saved native maps have a different recorded physical profile")
    def same_directory(left, right):
        # Windows UNC hostnames and IP aliases need not have equal spellings.
        # File identity still requires both paths to exist on the same volume.
        try:
            return resolve_review_path(left).samefile(resolve_review_path(right))
        except OSError:
            return False

    if not same_directory(binding["combined_root"], source):
        prepared = resolve_review_path(binding["combined_root"]).resolve()
        # The ordinary discovery root may point at an immutable prepared set.
        # Follow only this model's direct publication pointer, with identical
        # bound metadata; arbitrary campaign redirects remain invalid.
        if (not same_directory(source, parent / 'validation')
                or not same_directory(prepared.parent, parent / 'validation/prepared_sets')
                or read(prepared / 'validation_binding.json') != binding
                or read(prepared / 'postflight_manifest.json') != manifest):
            raise ValueError("Native review binding points to another campaign")
        from darsia.fluidflower.tools.phase_reference_readiness import require_reference_readiness
        require_reference_readiness(source, allow_pending=True)
        require_reference_readiness(prepared, allow_pending=True)
        return verified_review_binding(parent, prepared, saved_comparison=saved_comparison)
    return binding


def captured_annotation_origins(report):
    """Read only a hash-bound detached capture's pre-alignment source records."""
    reference = report.get("source_capture_audit")
    if not reference:
        return {}, None
    path = Path(reference["path"])
    if sha256_file(path) != reference["sha256"]:
        raise ValueError("Frozen disagreement capture audit changed")
    origin_reference = read(path).get("annotation_origins")
    if not origin_reference:  # Existing canonical queues retain their original contract.
        return {}, None
    origin_path = Path(origin_reference["path"])
    if sha256_file(origin_path) != origin_reference["sha256"]:
        raise ValueError("Captured annotation origins changed")
    origins = read(origin_path)
    if origins.get("format") != "phase-annotation-prealignment-origins-v1":
        raise ValueError("Unsupported captured annotation origins")
    return origins["regions"], origin_reference


def align_native_region(region, original, selected):
    """A timeline remap must never silently move a box to different geometry."""
    from .phase_training import trainer
    for key in ("canvas_wh", "content_box_xywh", "geometry_revision_id"):
        if original.get(key) != selected.get(key):
            raise ValueError(f"{region.region_id}: different native geometry requires an explicit mapping")
    return trainer._align_region_timeline(region, original, selected)


def build_queue(evaluation_path, output, *, threshold=.75, evidence_pairs=None, verified_binding=None):
    """Bind reported regions to frozen annotations and the exact image timeline."""
    from .phase_validation import _model_identity, _source_jobs
    from .phase_training import trainer

    evaluation_path, output = Path(evaluation_path), Path(output)
    report = read(evaluation_path)
    region_audit = bool(report.get("model_root") and report.get("new_manifest_sha256"))
    if not report.get("completed_utc") and not region_audit:
        raise ValueError("A completed evaluation is required")
    parent = Path(report["model_root"] if region_audit else report["parent"])
    model_id, model_identity = _model_identity(parent)
    if not region_audit and report["parent_identity"] != model_identity:
        raise ValueError("The evaluation belongs to different model weights")
    automatic = report.get("format") == "saved-phase-annotation-disagreement-v1"
    origins, origin_reference = captured_annotation_origins(report) if automatic else ({}, None)
    binding = (verified_binding if verified_binding is not None else
               verified_review_binding(parent, report.get("review_root")) if automatic else
               read(parent / "validation/validation_binding.json"))
    if verified_binding is not None and binding != verified_review_binding(parent, binding["combined_root"]):
        raise ValueError("Supplied review binding is not the verified source binding")
    source = Path(binding["combined_root"])
    source_manifest = source / "postflight_manifest.json"
    if region_audit and (sha256_file(source_manifest) != report["new_manifest_sha256"]
                         or (not automatic and read(source_manifest)["model_sha256"] != model_identity)):
        raise ValueError("Region audit belongs to a different published model campaign")
    jobs = {j["run"]: j for j in _source_jobs(source)}
    if evidence_pairs is None and automatic:
        evidence_pairs = []
        for pair in report["frozen_evidence"]:
            for entry in pair:
                if sha256_file(Path(entry["path"])) != entry["sha256"]:
                    raise ValueError("Frozen disagreement evidence changed")
            evidence_pairs.append(tuple(Path(entry["path"]) for entry in pair))
    if evidence_pairs is None:
        request = read(parent / "training_request.json")
        evidence_pairs = [(Path(b["decisions"]["snapshot_path"]), Path(b["combined_manifest"]["snapshot_path"]))
                          for b in request["bundles"]]
    from .phase_review_basis import REGIONAL_VERSION, SAVED_NATIVE_VERSION
    regional = automatic and report.get("phase_review_basis") in {REGIONAL_VERSION, SAVED_NATIVE_VERSION}
    options = {"preserve_native_revisions": True} if regional else {}
    sources, regions, _ = trainer._load_sources_and_regions([trainer.ReviewBundle(*pair) for pair in evidence_pairs], **options)
    originals = {s.run: read(s.asset_manifest_path) for s in sources}
    region_originals = {}
    if regional:
        from darsia.fluidflower.phase_saved_runtime_views import native_source
        manifests, original_natives = {}, {}
        for region in regions:
            document = origins[region.region_id]["document_index"]
            if document not in manifests:
                manifests[document] = read(evidence_pairs[document][1])
            key = (document, region.run)
            if key not in original_natives:
                jobs_for_run = [job for job in manifests[document]["jobs"] if job["run"].lower() == region.run]
                _, path, native = native_source(jobs_for_run[0])
                if report["prediction_sources"].get(str(path)) != sha256_file(path):
                    raise ValueError("Original annotation image revision changed after evaluation")
                original_natives[key] = native
            region_originals[region.region_id] = original_natives[key]
    indexed = {r.region_id: r for r in regions}
    natives, entries = {}, []
    for original_row in report["regions"]:
        row = original_row
        region = indexed[row["region_id"]]
        if region_audit:
            if (not automatic and row["frame"] != region.frame_index) or row["class"] != region.phase_class:
                raise ValueError("Region audit differs from its frozen training evidence")
            row = {**row, "old_gas_fraction": row["new"]["gas_fraction"],
                   "new_gas_fraction": row["new"]["gas_fraction"],
                   "co2_fraction": row["new"]["co2_fraction"],
                   **({"aqueous_fraction": row["new"]["aqueous_fraction"]} if "aqueous_fraction" in row["new"] else {})}
        reasons = disagreement_reasons(row, threshold)
        annotation_conflicts = row.get("annotation_conflicts", []) if automatic else []
        if annotation_conflicts:
            reasons.append(("annotation_conflict", 1., "; ".join(sorted({item["reason"] for item in annotation_conflicts}))))
        if not reasons:
            continue
        job = jobs[region.run]
        if automatic and report.get("phase_review_basis") == SAVED_NATIVE_VERSION:
            # Exact original RAW/V5 native reader, without inventing a display
            # map path or replacing the source timeline with an optical one.
            from .phase_saved_runtime_views import native_source
            _, native_path, native_value = native_source(job)
        else:
            native_path = Path(job["analysis_source_review_root"]) / "native_frames" / region.run / "manifest.json"
            native_value = None
        if region.run not in natives:
            natives[region.run] = {"path": str(native_path), "sha256": sha256_file(native_path),
                                   "manifest": native_value if native_value is not None else read(native_path)}
        native = natives[region.run]["manifest"]
        original = region_originals[region.region_id] if regional else originals[region.run]
        aligned = (align_native_region(region, original, native) if regional else
                   trainer._align_region_timeline(region, original, native))
        if (not region_audit or automatic) and (aligned.frame_index != row["frame"] or abs(aligned.elapsed_h-row["elapsed_h"]) > 1e-6):
            raise ValueError(f"Evaluation image mismatch: {region.region_id}")
        if automatic and report.get("phase_review_basis") == SAVED_NATIVE_VERSION:
            from .raw_photographic_review import timeline_rows
            frame = timeline_rows(native, region.run)[aligned.frame_index]
        else:
            frame = native["frame_sources"][aligned.frame_index]
        if abs(float(frame["elapsed_h"])-aligned.elapsed_h) > 1e-6:
            raise ValueError("Annotation time does not match its source frame")
        original_annotation = region.payload
        origin = origins.get(region.region_id)
        if origin_reference is not None and origin is None:
            raise ValueError("Disagreement region has no captured annotation origin")
        if origin is not None:
            if origin["aligned_annotation_sha256"] != payload_hash(region.payload):
                raise ValueError("Captured annotation origin differs from aligned training evidence")
            original_annotation = origin["original_annotation"]
            if payload_hash(original_annotation) != origin["original_annotation_sha256"]:
                raise ValueError("Captured pre-alignment annotation fingerprint changed")
            if origin["original_region_id"] != region.region_id:
                raise ValueError("Anonymous annotation identity changed during alignment; an explicit identity is required for re-review")
            if trainer.canonical_phase_class(original_annotation, region.source) != region.phase_class:
                raise ValueError("Captured annotation origin changed the original phase meaning")
        peers = {peer["region_id"]: dict(peer, annotation_sha256=payload_hash(peer["annotation"]))
                 for item in annotation_conflicts for peer in item["conflict_peers"]
                 if peer["region_id"] != region.region_id}
        entries.append({"id": region.region_id, "run": region.run, "dataset": region.dataset,
            "frame_index": aligned.frame_index, "elapsed_h": aligned.elapsed_h,
            "source_name": frame.get("source_name") or Path((frame.get("acquisition") or {}).get("path", "")).name,
            "original_class": region.phase_class,
            "original_reviewer": region.reviewer, "original_annotation": original_annotation,
            "original_annotation_sha256": payload_hash(original_annotation),
            "annotation": aligned.payload, "reasons": [r[0] for r in reasons],
            "reason_text": "; ".join(r[2] for r in reasons), "score": max(r[1] for r in reasons),
            "old_gas_fraction": row["old_gas_fraction"], "new_gas_fraction": row["new_gas_fraction"],
            "co2_fraction": row["co2_fraction"],
            **({"aqueous_fraction": row["aqueous_fraction"]} if "aqueous_fraction" in row else {}),
            **({"dominant_phase": row["dominant_phase"]} if row.get("dominant_phase") else {}),
            **({"conflict_peers": list(peers.values()), "annotation_conflicts": annotation_conflicts} if annotation_conflicts else {})})
    entries.sort(key=lambda r: ("annotation_conflict" not in r["reasons"], "temporal_rule_conflict" not in r["reasons"], -r["score"], r["run"], r["frame_index"], r["id"]))
    queue = {"format": FORMAT, "created_utc": utc_now(), "model_root": str(parent),
        "model_id": model_id, "model_identity": model_identity,
        "evaluation_path": str(evaluation_path), "evaluation_sha256": sha256_file(evaluation_path),
        "experimental_rule": report.get("new_front_settings", {}).get("format"),
        "disagreement_threshold": threshold, "binding": binding,
        "source_manifest_sha256": sha256_file(source / "postflight_manifest.json"),
        "natives": {k: {x:v[x] for x in ("path", "sha256")} for k,v in natives.items()},
        "entries": entries}
    if automatic:
        queue.update(automatic_saved_map_review=True, automatic_review=True, coverage=report["coverage"],
            phase_review_basis=report.get("phase_review_basis", "archived-supported"),
            frozen_evidence=report["frozen_evidence"], prediction_sources=report["prediction_sources"],
            source_capture_audit=report.get("source_capture_audit"))
        if origin_reference is not None:
            queue["annotation_origins"] = origin_reference
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(queue, stream, indent=2, allow_nan=False)
    return queue


def validate_pool(queue, pool):
    """Keep supplemental evidence bound to the original model and evaluation."""
    for key in ("model_identity", "model_id", "evaluation_sha256", "source_manifest_sha256"):
        if pool.get(key) != queue.get(key):
            raise ValueError(f"Disagreement pool differs from the original queue: {key}")
    original = {r["id"]: r for r in queue["entries"]}
    for row in pool["entries"]:
        if row["id"] in original and row["original_annotation_sha256"] != original[row["id"]]["original_annotation_sha256"]:
            raise ValueError("Disagreement pool changed an original annotation")


def select_additional_regions(queue, pool, selected_ids, count, *, run=""):
    """Return the next ranked, unique regions without changing either source."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("The number of additional regions must be a positive integer")
    validate_pool(queue, pool)
    excluded = {r["id"] for r in queue["entries"]} | set(selected_ids)
    available = [r for r in pool["entries"] if r["id"] not in excluded and (not run or r["run"] == run)]
    return sorted(available, key=lambda r: (-r["score"], r["run"], r["elapsed_h"], r["id"]))[:count]


def restore_selected_regions(queue, pool, state, *, queue_sha256, pool_sha256):
    validate_pool(queue, pool)
    if state.get("queue_sha256") != queue_sha256 or state.get("pool_sha256") != pool_sha256:
        raise ValueError("Saved selection belongs to different evidence")
    ids = state.get("selected_ids")
    if not isinstance(ids, list) or len(ids) != len(set(ids)):
        raise ValueError("Invalid or duplicate selected regions")
    indexed = {r["id"]: r for r in pool["entries"]}
    originals = {r["id"] for r in queue["entries"]}
    if any(identifier not in indexed or identifier in originals for identifier in ids):
        raise ValueError("Saved selection references unknown or already included evidence")
    return [indexed[identifier] for identifier in ids]


def build_additional_pool(queue_path, output):
    queue = load_queue(queue_path, verify_sources=True)
    evaluation = Path(queue["evaluation_path"])
    if sha256_file(evaluation) != queue["evaluation_sha256"]:
        raise ValueError("The frozen evaluation changed")
    pool = build_queue(evaluation, output, threshold=0.)
    validate_pool(queue, pool)
    return pool


def load_queue(path, *, verify_sources=False):
    queue = read(path)
    if queue.get("format") != FORMAT or not isinstance(queue.get("entries"), list):
        raise ValueError("Not a phase-evidence audit queue")
    ids = [row["id"] for row in queue["entries"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate audit regions")
    for row in queue["entries"]:
        if payload_hash(row["original_annotation"]) != row["original_annotation_sha256"]:
            raise ValueError("Original annotation fingerprint changed")
        for peer in row.get("conflict_peers", []):
            if "annotation_sha256" in peer and payload_hash(peer["annotation"]) != peer["annotation_sha256"]:
                raise ValueError("Conflicting annotation fingerprint changed")
    if verify_sources:
        from .phase_validation import _model_identity
        if _model_identity(Path(queue["model_root"]))[1] != queue["model_identity"]:
            raise ValueError("Audit model changed")
        source = Path(queue["binding"]["combined_root"]) / "postflight_manifest.json"
        if sha256_file(source) != queue["source_manifest_sha256"]:
            raise ValueError("Audit source campaign changed")
        for native in queue["natives"].values():
            if sha256_file(Path(native["path"])) != native["sha256"]:
                raise ValueError("Audit image timeline changed")
        if queue.get("automatic_saved_map_review"):
            if sha256_file(Path(queue["evaluation_path"])) != queue["evaluation_sha256"]:
                raise ValueError("Frozen automatic disagreement evaluation changed")
            for pair in queue["frozen_evidence"]:
                for entry in pair:
                    if sha256_file(Path(entry["path"])) != entry["sha256"]:
                        raise ValueError("Frozen disagreement evidence changed")
            for path, digest in queue["prediction_sources"].items():
                if sha256_file(Path(path)) != digest:
                    raise ValueError("Saved disagreement prediction changed")
            for reference in (queue.get("source_capture_audit"), queue.get("annotation_origins")):
                if reference and sha256_file(Path(reference["path"])) != reference["sha256"]:
                    raise ValueError("Frozen disagreement source-origin receipt changed")
    return queue


def queue_directory(model):
    """Keep study work products alongside the series on NAS."""
    from .phase_review_retractions import resolve_project_root
    from .series_paths import series_folder_path
    series = read(Path(model) / "summary.json")["inventory"]["datasets"]
    if len(series) != 1:
        raise ValueError("Select one experimental series for disagreements")
    return series_folder_path(resolve_project_root(series[0]), "evidence_audits")


def discover_queues():
    from .phase_workflow import default_phase_review_roots
    rows = []
    from .phase_review_retractions import resolve_project_root
    from .series_paths import configured_series, series_folder_path
    directories = [root / "evidence_audits" for root in default_phase_review_roots()]
    for series, _label, _root in configured_series():
        try:
            directories.append(series_folder_path(resolve_project_root(series), "evidence_audits"))
        except (OSError, ValueError):
            continue
    for directory in dict.fromkeys(directories):
        for path in directory.glob("*.json"):
            try:
                queue = load_queue(path)
                rows.append({**queue, "path": str(path)})
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return sorted(rows, key=lambda r: r["created_utc"], reverse=True)


def _verified_pre_geometry_hash(payload):
    """Recover an exact reviewed fingerprint from a lossless geometry receipt.

    Native-revision training retains the transported original and the live
    human correction on their respective image grids. Their payload hashes
    differ even though the correction still targets the same original review.
    Require both the unchanged transported geometry and the complete original
    payload digest; a matching ID or similar coordinates is never sufficient.
    """
    geometry = payload.get("geometry_migration") or {}
    if (geometry.get("format") != "phase-training-geometry-revision-v1"
            or not geometry.get("frame_receipt_sha256")
            or not geometry.get("geometry_revision_id")
            or geometry.get("stale_sampling_removed") is not False):
        return None
    keys = {"coordinate_space", "shape", "points", "x", "y", "width", "height",
            "frame_index", "elapsed_h"}
    source, target = geometry.get("source_geometry"), geometry.get("target_geometry")
    if (not isinstance(source, dict) or not isinstance(target, dict)
            or not {"x", "y", "width", "height", "frame_index"} <= source.keys()
            or not source.keys() <= keys or not target.keys() <= keys
            or target != {key: payload[key] for key in keys if key in payload}):
        return None
    original = {key: value for key, value in payload.items()
                if key not in keys and key != "geometry_migration"}
    original.update(source)
    bridge = geometry.get("supersession_rebinding")
    if bridge:
        if (bridge.get("human_verdict_and_target_id_unchanged") is not True
                or not isinstance(bridge.get("original_evidence_audit"), dict)):
            return None
        original["evidence_audit"] = bridge["original_evidence_audit"]
    # Geometry receipts use the original (spaced JSON) digest contract.
    digest = hashlib.sha256(json.dumps(original, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if digest != geometry.get("source_annotation_sha256"):
        return None
    return payload_hash(original)


# The fields a RAW camera-basis transfer rewrites: geometry and photograph
# identity. Every other field, the human decision included, stays as reviewed.
_BASIS_TRANSFER_FIELDS = {"shape", "points", "x", "y", "width", "height", "frame_index", "elapsed_h",
                          "coordinate_space", "photo_identity", "source_label_identity",
                          "source_binding_contract", "source_sha256", "basis_transfer"}


def _basis_transfer_source_hashes(payload):
    """Fingerprints of the reviewed evidence a RAW camera-basis transfer started from.

    The transfer keeps the exact reviewed annotation as
    ``basis_transfer.source_annotation`` and rewrites only its geometry and
    photograph identity, so a re-review of the original still targets it. The
    source must match its recorded digest and every other field must be
    unchanged; otherwise nothing is returned. Fingerprints of earlier
    geometry migrations or transfers of that source are included.
    """
    from .phase_training.annotation_basis_transfer import FORMAT, _digest
    transfer = payload.get("basis_transfer")
    if not isinstance(transfer, dict) or transfer.get("format") != FORMAT:
        return set()
    source = transfer.get("source_annotation")
    if not isinstance(source, dict) or transfer.get("source_annotation_sha256") != _digest(source):
        return set()
    kept = lambda value: {key: item for key, item in value.items() if key not in _BASIS_TRANSFER_FIELDS}
    if kept(payload) != kept(source):
        return set()
    hashes = {payload_hash(source)} | _basis_transfer_source_hashes(source)
    earlier = _verified_pre_geometry_hash(source)
    if earlier is not None:
        hashes.add(earlier)
    return hashes


def superseded_evidence_ids(records):
    """Apply explicit human revisions only to the in-memory training selection."""
    from collections import defaultdict
    originals, revisions = defaultdict(set), defaultdict(dict)
    for identifier, payload in records:
        originals[identifier].add(payload_hash(payload))
        original_hash = _verified_pre_geometry_hash(payload)
        if original_hash is not None:
            originals[identifier].add(original_hash)
        originals[identifier].update(_basis_transfer_source_hashes(payload))
        audit = payload.get("evidence_audit")
        if not audit:
            continue
        if (audit.get("verdict") not in {"confirmed", "reclassified"}
                or not payload.get("updated_utc") or not payload.get("reviewer")):
            raise ValueError("Evidence revision requires a recorded human decision")
        target = str(audit.get("original_region_id") or "")
        if not target or target == identifier or target.split(":")[0] != identifier.split(":")[0]:
            raise ValueError("Invalid evidence revision target")
        existing = revisions[target].get(identifier)
        if existing is None or payload["updated_utc"] > existing["updated_utc"]:
            revisions[target][identifier] = payload
    blocked = set()
    for target, candidates in revisions.items():
        for payload in candidates.values():
            digest = payload["evidence_audit"].get("original_annotation_sha256")
            if target in originals and digest not in originals[target]:
                raise ValueError(f"Original evidence changed before re-review: {target}")
        ordered = sorted(candidates, key=lambda key: (candidates[key]["updated_utc"], key))
        winner = ordered[-1]
        if len(ordered) > 1:
            previous = ordered[-2]
            if (candidates[winner]["updated_utc"] == candidates[previous]["updated_utc"]
                    and candidates[winner].get("category") != candidates[previous].get("category")):
                raise ValueError(f"Conflicting simultaneous re-reviews: {target}")
        blocked.add(target)
        blocked.update(key for key in candidates if key != winner)
    return blocked


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--evaluation", type=Path)
    source.add_argument("--expand-queue", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=.75)
    args = parser.parse_args()
    result = (build_additional_pool(args.expand_queue, args.output) if args.expand_queue
              else build_queue(args.evaluation, args.output, threshold=args.threshold))
    print(f"Registered {len(result['entries'])} review regions in {args.output}")
