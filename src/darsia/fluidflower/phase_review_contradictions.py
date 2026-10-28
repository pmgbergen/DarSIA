"""Fresh checks of saved human evidence against the original saved phase maps.

This finite worker route performs no inference, fitting, queue scheduling or
scientific acceptance. It reuses the original evidence loader, timeline join,
region rasterizer, optical-map selection, phase colours and disagreement rules.
Only a new output directory is written. Call this off the GUI thread after the
ordinary worker admission; it never starts another process itself.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import uuid

from .phase_evidence_audit import payload_hash, read
from .project import sha256_file, utc_now

REPORT_FORMAT = "saved-phase-annotation-disagreement-v1"
CAPTURE_FORMAT = "phase-contradiction-source-capture-v1"
ROLE_FIELDS = (
    "evidence_role", "include_in_training", "requires_explicit_training_selection",
    "independent_holdout", "training_seen", "training_role", "training_scope",
    "evaluation_scope", "blindness_status", "post_reveal_edits_are_blind",
    "source_model_id", "source_model_sha256", "evaluation_model_sha256",
)


class UnsupportedReviewSource(ValueError):
    """A saved source needs a dedicated evaluator; never substitute other maps."""


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return dict(path=str(path.resolve()), sha256=sha256_file(path))


def _pin(path, pins):
    path = Path(path).resolve()
    digest = sha256_file(path)
    if str(path) in pins and pins[str(path)] != digest:
        raise ValueError("A contradiction source changed during evaluation: " + str(path))
    pins[str(path)] = digest
    return dict(path=str(path), sha256=digest)


def _verify_pins(pins):
    for path, digest in pins.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("A contradiction source changed during evaluation: " + path)


def evidence_role(*, annotation, decision, document, manifest, session=None, source_jobs=()):
    """Retain every declared role at its original level; never enroll evidence."""
    scopes = {name: {key: deepcopy(value[key]) for key in ROLE_FIELDS if key in value}
              for name, value in (("annotation", annotation), ("decision", decision),
                                  ("document", document), ("manifest", manifest),
                                  ("session", session or {}))}
    scopes.update({f"source_job_{index}": {key: deepcopy(job[key]) for key in ROLE_FIELDS if key in job}
                   for index, job in enumerate(source_jobs)})
    values = list(scopes.values())
    heldout = any(value.get("include_in_training") is False
                  or value.get("independent_holdout") is True
                  or value.get("requires_explicit_training_selection") is True
                  or value.get("evidence_role") in {"blind_transfer_validation",
                      "independent_blind_gas_validation", "model_transfer_evaluation"}
                  or value.get("training_role") in {"evaluation-only", "evaluation_only", "heldout"}
                  for value in values)
    return dict(source_scopes=scopes,
                training_selection="excluded_or_requires_explicit_selection" if heldout else "unchanged",
                source_payload_sha256=payload_hash(annotation))


def _source_records(decisions, manifest, session):
    """Exact raw payloads, including quiz answers; stable IDs are mandatory."""
    result = {}
    for run, decision in (decisions.get("decisions") or {}).items():
        for field, kind in (("annotations", "manual"), ("phase_quiz_answers", "quiz")):
            for annotation in decision.get(field) or []:
                identifier = annotation.get("id") or annotation.get("proposal_id")
                if not identifier:
                    raise ValueError("Fresh contradiction review requires stable annotation/proposal IDs")
                key = f"{str(run).casefold()}:{kind}:{identifier}"
                if key in result:
                    raise ValueError("Duplicate source evidence ID: " + key)
                result[key] = dict(annotation=deepcopy(annotation),
                    role=evidence_role(annotation=annotation, decision=decision,
                        document=decisions, manifest=manifest, session=session,
                        source_jobs=[job for job in manifest.get("jobs") or []
                                     if str(job.get("run")).casefold() == str(run).casefold()]))
    return result


def _freeze_evidence(inputs, output):
    from .phase_training.gas_negative_sampling import freeze_document_sources
    from .phase_training.trainer import ReviewBundle
    bundles, frozen, originals, capture = [], [], [], []
    for index, item in enumerate(inputs):
        # Deliberately explicit paths; no discovery, latest-session or history guess.
        decisions_path, manifest_path = Path(item["decisions"]), Path(item["manifest"])
        original_pins = {}
        decision_ref, manifest_ref = _pin(decisions_path, original_pins), _pin(manifest_path, original_pins)
        decisions, manifest = read(decisions_path), read(manifest_path)
        session_path = item.get("session")
        if session_path is None and (decisions_path.parent / "review_session.json").is_file():
            session_path = decisions_path.parent / "review_session.json"
        session_ref = _pin(session_path, original_pins) if session_path else None
        session = read(session_path) if session_path else {}
        for job in manifest.get("jobs") or []:
            source = job.get("analysis_source_review_root") or job.get("review_output_dir")
            if not source or not Path(source).is_absolute():
                raise UnsupportedReviewSource("Evidence manifests must bind absolute native review roots")
        originals.append(_source_records(decisions, manifest, session))
        directory = output / "inputs" / str(index)
        frozen_decisions = freeze_document_sources(decisions, directory, source_root=decisions_path.parent)
        pair = [_write_new(directory / "decisions.json", frozen_decisions),
                _write_new(directory / "manifest.json", manifest)]
        if session_ref:
            _write_new(directory / "review_session.json", session)
        _verify_pins(original_pins)
        bundles.append(ReviewBundle(Path(pair[0]["path"]), Path(pair[1]["path"])))
        frozen.append(pair)
        capture.append(dict(document_index=index, original_decisions=decision_ref,
                            original_manifest=manifest_ref, original_session=session_ref))
    return bundles, frozen, originals, capture


def _check_frame(region, original, native):
    from math import isfinite
    from .phase_evidence_audit import align_native_region
    from .raw_photographic_review import RAW_REVIEW_FORMAT, timeline_rows
    aligned = align_native_region(region, original, native)
    old = timeline_rows(original, region.run)[region.frame_index]
    current = timeline_rows(native, region.run)[aligned.frame_index]
    old_raw, new_raw = original.get("format") == RAW_REVIEW_FORMAT, native.get("format") == RAW_REVIEW_FORMAT
    if old_raw or new_raw:
        if not old_raw or not new_raw:
            raise ValueError("RAW acquisition evidence cannot be rebound to a different source representation")
        for key in ("native_origin_xy", "native_shape_hw", "phase_shape_hw"):
            if original.get(key) != native.get(key):
                raise ValueError("RAW annotation and prediction use different native geometry")
        for key in ("native_geometry", "rig_geometry"):
            old_digest, new_digest = (old.get(key) or {}).get("sha256"), (current.get(key) or {}).get("sha256")
            if not old_digest or old_digest != new_digest:
                raise ValueError("RAW annotation and prediction use different recorded geometry")
    old_sha = ((old.get("acquisition") or {}).get("sha256") if old_raw else
               (old.get("source_label_identity") or {}).get("camera_source_sha256") or old.get("source_sha256"))
    new_sha = ((current.get("acquisition") or {}).get("sha256") if new_raw else
               (current.get("source_label_identity") or {}).get("camera_source_sha256") or current.get("source_sha256"))
    if old_raw:
        if not old_sha or not new_sha:
            raise ValueError("RAW annotation camera identity requires an acquisition SHA on both timelines")
        old_time = float(old["elapsed_h"])
        if (int(old["frame_index"]) != region.frame_index or not isfinite(old_time)
                or not isfinite(region.elapsed_h) or abs(old_time - region.elapsed_h) > 1e-6):
            raise ValueError("Annotation time differs from the original RAW frame")
    if old_sha or new_sha:
        if not old_sha or old_sha != new_sha:
            raise ValueError("Annotation and saved prediction refer to different camera acquisitions")
    elif not old.get("source_name") or old.get("source_name") != current.get("source_name"):
        raise ValueError("Annotation camera identity is not bound on both timelines")
    current_time = float(current["elapsed_h"])
    if (int(current["frame_index"]) != aligned.frame_index
            or not isfinite(current_time) or not isfinite(aligned.elapsed_h)
            or abs(current_time - aligned.elapsed_h) > 1e-6):
        raise ValueError("Annotation time differs from the saved prediction frame")
    return aligned


def evaluate_region(region, native, rgba, domain):
    """Original rasterization/phase display semantics, with no resampling."""
    import numpy as np
    from .phase_training.review_regions import annotation_mask
    from .phase_rendering import two_phase_rgba, AQUEOUS_RGBA, GAS_RGBA
    width, height = (int(value) for value in native["canvas_wh"])
    x, y, cw, ch = (int(value) for value in native["content_box_xywh"])
    if (rgba.dtype != np.uint8 or rgba.shape != (height, width, 4)
            or domain.dtype != np.bool_ or domain.shape != (ch, cw)
            or x < 0 or y < 0 or min(cw, ch) <= 0 or x + cw > width or y + ch > height):
        raise ValueError("Saved map, registered domain and annotation geometry differ")
    # Rasterization clips to the source crop; refuse wholly outside boxes instead
    # of counting the clipped border pixel as reviewed content.
    p = region.payload
    bounds = [float(p[key]) for key in ("x", "y", "width", "height")]
    if not all(np.isfinite(bounds)) or min(bounds[2:]) <= 0:
        raise ValueError("Annotation has no finite positive bounding box")
    if (bounds[0] * width >= x + cw or (bounds[0] + bounds[2]) * width <= x
            or bounds[1] * height >= y + ch or (bounds[1] + bounds[3]) * height <= y):
        raise ValueError("Annotation is outside the registered image content")
    mask = annotation_mask(p, canvas_wh=native["canvas_wh"], content_box=native["content_box_xywh"])
    selected = mask & domain
    count = int(np.count_nonzero(selected))
    if count == 0:
        raise UnsupportedReviewSource("annotation_has_no_registered_domain_pixels")
    image = two_phase_rgba(rgba)[y:y + ch, x:x + cw]
    active = image[..., 3] > 0
    aqueous = active & np.all(image[..., :3] == AQUEOUS_RGBA[:3], axis=-1)
    gas = active & np.all(image[..., :3] == GAS_RGBA[:3], axis=-1)
    if np.any(selected & active & ~(aqueous | gas)):
        raise UnsupportedReviewSource("Saved map contains unresolved or unsupported colours in the annotation")
    gas_fraction = float(np.count_nonzero(selected & gas) / count)
    aqueous_fraction = float(np.count_nonzero(selected & aqueous) / count)
    return _fraction_row(region, gas_fraction, aqueous_fraction, count, int(np.count_nonzero(mask)))


def _fraction_row(region, gas_fraction, aqueous_fraction, count, annotation_pixels):
    from .phase_evidence_audit import disagreement_reasons
    p = region.payload
    row = dict(region_id=region.region_id, run=region.run, dataset=region.dataset,
               frame=region.frame_index, elapsed_h=region.elapsed_h, **{"class": region.phase_class},
               old_gas_fraction=gas_fraction, new_gas_fraction=gas_fraction,
               co2_fraction=gas_fraction + aqueous_fraction, aqueous_fraction=aqueous_fraction,
               evaluated_pixels=count, annotation_pixels=annotation_pixels,
               area_measure="native pixel fraction inside the registered domain")
    dominance = p.get("phase_dominance") or p.get("dominant_phase")
    if dominance in {"gas", "aqueous"}:
        row["dominant_phase"] = dominance
    # Validate all fraction/class rules through the retained evaluator. The queue
    # applies the caller's ranking threshold to these same actual measurements.
    disagreement_reasons(row, 0.)
    row["new"] = dict(gas_fraction=gas_fraction, aqueous_fraction=aqueous_fraction,
                      co2_fraction=row["co2_fraction"])
    return row


def evaluate_saved_labels(region, native, labels, domain, support):
    """Measure exact original categorical labels within verified scientific support."""
    import numpy as np
    from .phase_training.review_regions import annotation_mask
    width, height = (int(value) for value in native["canvas_wh"])
    x, y, cw, ch = (int(value) for value in native["content_box_xywh"])
    if (labels.dtype != np.uint8 or labels.shape != (ch, cw)
            or domain.dtype != np.bool_ or domain.shape != labels.shape
            or support.dtype != np.bool_ or support.shape != labels.shape
            or x < 0 or y < 0 or min(cw, ch) <= 0 or x + cw > width or y + ch > height):
        raise ValueError("Saved labels, scientific support and native annotation geometry differ")
    bounds = [float(region.payload[key]) for key in ("x", "y", "width", "height")]
    if not all(np.isfinite(bounds)) or min(bounds[2:]) <= 0:
        raise ValueError("Annotation has no finite positive bounding box")
    if (bounds[0] * width >= x + cw or (bounds[0] + bounds[2]) * width <= x
            or bounds[1] * height >= y + ch or (bounds[1] + bounds[3]) * height <= y):
        raise ValueError("Annotation is outside the registered image content")
    mask = annotation_mask(region.payload, canvas_wh=native["canvas_wh"], content_box=native["content_box_xywh"])
    selected = mask & domain
    if np.any(selected & ~support):
        raise UnsupportedReviewSource("Annotation crosses pixels without verified saved scientific support")
    count = int(np.count_nonzero(selected))
    if count == 0:
        raise UnsupportedReviewSource("annotation_has_no_registered_domain_pixels")
    if np.any(selected & ~np.isin(labels, (0, 1, 2))):
        raise UnsupportedReviewSource("Saved labels contain unresolved or unsupported classes in the annotation")
    row = _fraction_row(region, float(np.count_nonzero(selected & (labels == 2)) / count),
        float(np.count_nonzero(selected & (labels == 1)) / count), count, int(np.count_nonzero(mask)))
    row["scientific_support_verified"] = True
    row["phase_encoding"] = "0=water,1=aqueous,2=free-gas"
    return row


def _domain(model_root, profile, native_path, native, job, pins):
    import numpy as np
    from .phase_runtime import load_run_context
    # These cohorts have frame-varying observation support. A static mask or
    # old supported map is not a substitute for their dedicated saved reader.
    if (native.get("v5_frame_geometry") or native.get("source_kind") == "raw"
            or native.get("format") == "darsia-raw-photographic-review-v1"):
        raise UnsupportedReviewSource("Per-frame RAW/V5 support requires its dedicated saved-map evaluator")
    _, _, width, height = native["content_box_xywh"]
    if profile is None:
        return np.ones((height, width), dtype=bool)
    if profile.get("format") in {"ffac-selected-policy-a-review-physics-v1", "ffac-rig-camera-physics-v1"}:
        raise UnsupportedReviewSource("Selected-camera and rig revisions require their dedicated saved-map evaluator")
    context, record = load_run_context(model_root, profile, job["dataset"], job["run"],
                                     native_manifest_sha256=sha256_file(native_path))
    reference = _pin(Path(model_root) / record["path"], pins)
    if reference["sha256"] != record["sha256"]:
        raise ValueError("Registered phase domain changed during evaluation")
    return context.domain


def build_contradiction_queue(model_root, review_root, evidence_inputs, output_root, *, threshold=.75):
    """Freshly evaluate explicit evidence and publish an original audit queue.

    Each evidence input supplies ``decisions``, ``manifest`` and, for live saved
    sessions, ``session``. Output must be a new directory. Missing or unsupported
    sources are counted in coverage, never treated as agreement. Corrupt changed
    sources fail the operation and leave its partial artifacts for inspection.
    """
    from PIL import Image
    import numpy as np
    from .phase_evidence_audit import build_queue, verified_review_binding, load_queue
    from .phase_validation import _model_identity, _source_jobs
    from .phase_training import trainer
    from .phase_runtime import load_physics_profile
    from .phase_saved_runtime_views import native_source
    from .phase_review_basis import optical_path, uses_optical, is_regional, VERSION, REGIONAL_VERSION, SAVED_NATIVE_VERSION
    from .phase_review_raw_contradictions import load_saved_phase_frame, UnsupportedSavedPhaseFrame
    from .phase_blind_review_policy import require_predictions_visible
    if not 0 <= threshold <= 1 or not evidence_inputs:
        raise ValueError("Choose evidence and a disagreement threshold from zero to one")
    model_root, review_root, output = Path(model_root).resolve(), Path(review_root).resolve(), Path(output_root).resolve()
    if output == model_root or output.is_relative_to(model_root) or output == review_root or output.is_relative_to(review_root):
        raise ValueError("Contradiction outputs must stay outside the immutable model and review sources")
    binding = verified_review_binding(model_root, review_root)
    review_root = Path(binding["combined_root"]).resolve()
    if output == review_root or output.is_relative_to(review_root):
        raise ValueError("Contradiction outputs must stay outside the bound review source")
    model_id, identity = _model_identity(model_root)
    profile, profile_loaded = None, False
    jobs = {job["run"]: job for job in _source_jobs(review_root)}
    output.mkdir(parents=True, exist_ok=False)
    bundles, frozen, originals, capture = _freeze_evidence(evidence_inputs, output)
    origins_by_id, sources = {}, {}
    audit = {}
    _, regions, conflicts = trainer._load_sources_and_regions(bundles, preserve_native_revisions=True,
                                                            region_sources=sources, label_policy_audit=audit)
    if not regions:
        raise ValueError("The given evidence contains no scorable visible-phase labels")
    indexes = {str(bundle.decisions_path): index for index, bundle in enumerate(bundles)}
    pins = {}
    manifest_ref = _pin(review_root / "postflight_manifest.json", pins)
    rows, skipped, role_records, bases = [], [], {}, set()
    saved_frame_key, saved_frame = None, None
    for region in regions:
        source = sources[region.region_id]
        index = indexes.get(str(source.decisions_path))
        if index is None or region.region_id not in originals[index]:
            raise UnsupportedReviewSource("Effective replacement evidence must be explicitly materialized among the inputs")
        original = originals[index][region.region_id]
        role_records[region.region_id] = original["role"]
        origins_by_id[region.region_id] = dict(document_index=index, original_region_id=region.region_id,
            original_annotation=original["annotation"], original_annotation_sha256=payload_hash(original["annotation"]),
            aligned_annotation_sha256=payload_hash(region.payload))
        original_native_ref = _pin(source.asset_manifest_path, pins)
        original_native = read(original_native_ref["path"])
        job = jobs.get(region.run)
        if job is None:
            skipped.append(dict(region_id=region.region_id, run=region.run, reason="no_saved_predictions_for_run"))
            continue
        try:
            if str(job.get("dataset")).casefold() != region.dataset.casefold():
                raise ValueError("Saved prediction and annotation belong to different series")
            require_predictions_visible(model_root, region.run)
            _native_root, native_path, native = native_source(job)
            native_ref = _pin(native_path, pins)
            aligned = _check_frame(region, original_native, native)
            saved_native = (native.get("source_kind") == "raw"
                or native.get("format") == "darsia-raw-photographic-review-v1"
                or "v5_frame_geometry" in native or "v5_photometry" in native
                or (job.get("quality_flags") or {}).get("raw_production_review") is True)
            if saved_native:
                key = region.run, aligned.frame_index, native_ref["sha256"]
                if key != saved_frame_key:
                    saved_frame = None  # Retain at most one frame of scientific arrays.
                    saved_frame = load_saved_phase_frame(model_root, review_root, job, aligned.frame_index,
                        expected_native_sha256=native_ref["sha256"])
                    saved_frame_key = key
                if saved_frame.native != native:
                    raise ValueError("Saved categorical labels belong to another native annotation source")
                for source_path, digest in saved_frame.pins.items():
                    if source_path in pins and pins[source_path] != digest:
                        raise ValueError("Saved categorical source identity changed during evaluation")
                    pins[source_path] = digest
                row = evaluate_saved_labels(aligned, native, saved_frame.labels,
                                            saved_frame.domain, saved_frame.support)
                row.update(prediction=deepcopy(saved_frame.provenance["prediction"]),
                           saved_phase_source=deepcopy(saved_frame.provenance))
                bases.add(SAVED_NATIVE_VERSION)
            else:
                if not uses_optical(job):
                    raise UnsupportedReviewSource("No original optical prediction is bound; supported maps are not substituted")
                if not profile_loaded:
                    profile = load_physics_profile(model_root)
                    profile_loaded = True
                domain = _domain(model_root, profile, native_path, native, job, pins)
                path = optical_path(job, aligned.frame_index)
                if not path.is_absolute():
                    raise UnsupportedReviewSource("The saved optical-map root must be absolute")
                if not path.is_file():
                    skipped.append(dict(region_id=region.region_id, run=region.run, reason="saved_phase_map_missing"))
                    continue
                map_ref = _pin(path, pins)
                with Image.open(io.BytesIO(path.read_bytes())) as image:
                    if image.mode != "RGBA":
                        raise ValueError("The saved categorical map must retain its original RGBA encoding")
                    rgba = np.array(image)
                row = evaluate_region(aligned, native, rgba, domain)
                row.update(prediction=map_ref)
                bases.add(REGIONAL_VERSION if is_regional(job) else VERSION)
            row.update(original_native=original_native_ref, prediction_native=native_ref)
            rows.append(row)
        except (UnsupportedReviewSource, UnsupportedSavedPhaseFrame, PermissionError) as error:
            skipped.append(dict(region_id=region.region_id, run=region.run, reason=str(error)))
    if len(bases) > 1:
        raise UnsupportedReviewSource("A queue must use one original saved-prediction review basis")
    basis = next(iter(bases), REGIONAL_VERSION if all(is_regional(job) for job in jobs.values()) else VERSION)
    # The original nonregional queue builder retains one native revision per run.
    # Refuse multiple revisions there instead of quietly moving older boxes.
    if basis == VERSION:
        revisions = {}
        for source in sources.values():
            revisions.setdefault(source.run, set()).add(str(source.asset_manifest_path))
        if any(len(value) > 1 for value in revisions.values()):
            raise UnsupportedReviewSource("Multiple nonregional native revisions need separate contradiction queues")
        _, queue_regions, _ = trainer._load_sources_and_regions(bundles)
        queue_ids = {region.region_id for region in queue_regions}
        omitted = [row for row in rows if row["region_id"] not in queue_ids]
        rows = [row for row in rows if row["region_id"] in queue_ids]
        skipped.extend(dict(region_id=row["region_id"], run=row["run"],
                            reason="original_nonregional_loader_deduplicated_geometry") for row in omitted)
    origin_ref = _write_new(output / "annotation_origins.json", dict(
        format="phase-annotation-prealignment-origins-v1", regions=origins_by_id))
    capture_ref = _write_new(output / "source_capture.json", dict(format=CAPTURE_FORMAT,
        inputs=capture, annotation_origins=origin_ref, region_evidence_roles=role_records,
        original_evidence_changed=False, training_selection_changed=False))
    counts = Counter(row["reason"] for row in skipped)
    coverage = dict(complete=not skipped and not conflicts and not audit.get("excluded_candidate_issue_count"),
        total_effective_regions=len(regions), scored_regions=len(rows), skipped_regions=len(skipped),
        missing_maps=counts["saved_phase_map_missing"],
        unmapped_runs=sorted({row["run"] for row in skipped if row["reason"] == "no_saved_predictions_for_run"}),
        skipped_reason_counts=dict(counts), inherited_geometry_conflicts=conflicts,
        excluded_candidate_specific_issues=audit.get("excluded_candidate_issue_count", 0))
    report = dict(format=REPORT_FORMAT, model_root=str(model_root), model_id=model_id,
        parent_identity=identity, review_root=str(review_root), new_manifest_sha256=manifest_ref["sha256"],
        completed_utc=utc_now(), phase_review_basis=basis, regions=rows, skipped=skipped,
        coverage=coverage, frozen_evidence=frozen, prediction_sources=pins,
        source_capture_audit=capture_ref, region_evidence_roles=role_records,
        scientific_acceptance=False, training_performed=False, inference_performed=False)
    _verify_pins(pins)
    if _model_identity(model_root) != (model_id, identity):
        raise ValueError("Selected model changed during contradiction evaluation")
    report_path = output / "evaluation.json"
    _write_new(report_path, report)
    queue_path = output / "queue.json"
    queue = build_queue(report_path, queue_path, threshold=threshold, verified_binding=binding)
    load_queue(queue_path, verify_sources=True)
    return dict(queue_path=str(queue_path), evaluation_path=str(report_path),
                entries=len(queue["entries"]), coverage=coverage)


def correction_role(queue, entry):
    """Read the original role for a dock correction, including expanded pools."""
    report_path = Path(queue["evaluation_path"])
    if sha256_file(report_path) != queue["evaluation_sha256"]:
        raise ValueError("The contradiction evaluation changed before correction")
    report = read(report_path)
    role = (report.get("region_evidence_roles") or {}).get(entry["id"])
    if role is None:
        raise ValueError("This contradiction has no preserved source evidence role")
    if role["source_payload_sha256"] != entry["original_annotation_sha256"]:
        raise ValueError("The correction role belongs to another original annotation")
    return deepcopy(role)


def preserve_correction_role(annotation, role):
    """Dock hook before save; never promote a held-out re-review into training.

    The caller must apply the same exclusion to its review session metadata;
    item fields alone do not replace the original session-selection guard.
    After prediction disclosure a correction is not a new blind observation.
    """
    result = deepcopy(annotation)
    # The retained ReviewStore intentionally strips unknown annotation fields;
    # evidence_audit is its supported provenance extension point.
    result.setdefault("evidence_audit", {})["source_evidence_role"] = deepcopy(role)
    result["evidence_audit"]["post_reveal_edits_are_blind"] = False
    if role["training_selection"] == "excluded_or_requires_explicit_selection":
        result["evidence_audit"].update(include_in_training=False,
            requires_explicit_training_selection=True)
    return result


def require_correction_session_role(role, session):
    """Item provenance cannot bypass the original session enrollment gate."""
    if role["training_selection"] == "excluded_or_requires_explicit_selection":
        if session.get("include_in_training") is not False:
            raise ValueError("Heldout evidence requires an explicitly training-excluded correction session")
    if session.get("editing_locked"):
        raise ValueError("The correction session is locked")


def require_disclosure(session_root, model_root, runs):
    """Honor the existing sealed blind snapshot before revealing disagreement."""
    from .phase_blind_review_policy import require_predictions_visible
    marker = read_reveal_marker(session_root)
    for run in sorted(set(runs)):
        require_predictions_visible(Path(model_root), run)
    return marker


def read_reveal_marker(session_root):
    """Validate the existing marker without relaxing model publication holds."""
    session_root = Path(session_root).resolve()
    marker = read(session_root / "blind_reveal.json")
    if (marker.get("format") != "blind-gas-reveal-v1"
            or marker.get("blindness_status") != "revealed"
            or marker.get("post_reveal_edits_are_blind") is not False
            or not marker.get("first_prediction_reveal_utc")):
        raise ValueError("Finish and seal the blind review before viewing contradictions")
    snapshot = Path(marker["frozen_blind_snapshot"])
    if (not snapshot.is_absolute() or not snapshot.resolve().is_relative_to(session_root / "blind_snapshots")
            or sha256_file(snapshot) != marker["frozen_blind_snapshot_sha256"]):
        raise ValueError("The sealed blind evidence snapshot is missing or changed")
    captured = read(snapshot)
    captured_session = captured.get("session_metadata")
    current_session = read(session_root / "review_session.json")
    if (captured.get("blind_snapshot_format") != "fluidflower-shared-phase-review-before-disclosure-v1"
            or not isinstance(captured_session, dict)
            or current_session.get("format") != "fluidflower-local-review-session-v1"
            or marker.get("review_session_root") != str(session_root)
            or any(not current_session.get(key) or captured_session.get(key) != current_session[key]
                   or marker.get(marker_key) != current_session[key]
                   for key, marker_key in (("target_model_id", "model_id"), ("reviewer", "reviewer"),
                                           ("review_session_id", "review_session_id")))):
        raise ValueError("The sealed snapshot belongs to another model, review session or owner")
    return deepcopy(marker)


def seal_blind_evidence(decisions_path, session_path, *, model_id, reviewer):
    """Seal durable original documents after the UI's save barrier has drained.

    The caller owns the editor lease and disables further edits while this runs.
    This records disclosure, never independent validation or training eligibility.
    Existing model publication holds are not modified. Repeat calls verify and
    return the first sealed snapshot without overwriting its contents.
    """
    decisions_path, session_path = Path(decisions_path), Path(session_path)
    session_root = session_path.parent
    if decisions_path.parent.resolve() != session_root.resolve():
        raise ValueError("The blind snapshot must belong to the canonical session store")
    marker_path = session_root / "blind_reveal.json"
    pins = {}
    decisions_ref, session_ref = _pin(decisions_path, pins), _pin(session_path, pins)
    document, session = read(decisions_path), read(session_path)
    if (session.get("format") != "fluidflower-local-review-session-v1"
            or session.get("target_model_id") != model_id
            or session.get("reviewer") != reviewer or not reviewer.strip()
            or not session.get("review_session_id")
            or session.get("editing_locked") or not isinstance(document.get("decisions"), dict)):
        raise ValueError("The durable blind documents do not belong to this editable model/reviewer session")
    if marker_path.exists():
        return read_reveal_marker(session_root)
    stamp = utc_now()
    snapshot_root = session_root / "blind_snapshots" / uuid.uuid4().hex
    snapshot = deepcopy(document)
    snapshot.update(sealed_utc=stamp, reviewer=reviewer,
        source_document_format=document.get("format"),
        blind_snapshot_format="fluidflower-shared-phase-review-before-disclosure-v1",
        source_decisions=decisions_ref, source_session=session_ref,
        session_metadata=deepcopy(session), post_reveal_edits_are_blind=False)
    reference = _write_new(snapshot_root / "blind_labels.json", snapshot)
    _write_new(snapshot_root / "review_session.json", session)
    with (snapshot_root / "blind_labels.sha256").open("x", encoding="ascii") as stream:
        stream.write(reference["sha256"] + "  blind_labels.json\n")
        stream.flush()
        os.fsync(stream.fileno())
    _verify_pins(pins)
    marker = dict(format="blind-gas-reveal-v1", blindness_status="revealed",
        model_id=model_id, review_session_id=session["review_session_id"], reviewer=reviewer,
        review_session_root=str(session_root.resolve()),
        first_prediction_reveal_utc=stamp, frozen_blind_snapshot=reference["path"],
        frozen_blind_snapshot_sha256=reference["sha256"], post_reveal_edits_are_blind=False)
    _write_new(marker_path, marker)
    return read_reveal_marker(session_root)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="JSON list of explicit decisions/manifest/session paths")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--threshold", type=float, default=.75)
    args = parser.parse_args(argv)
    from .series_native_context import saved_reader_ui_context
    with saved_reader_ui_context():
        result = build_contradiction_queue(args.model, args.review, read(args.inputs), args.output, threshold=args.threshold)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
