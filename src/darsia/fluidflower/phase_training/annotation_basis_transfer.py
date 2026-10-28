"""Project saved review annotations through their authenticated camera geometry.

Only metadata is read here. The original image producer remains responsible
for corrected pixels, and the normal trainer rasterizes the resulting polygons.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from ..phase_saved_assets import VerifiedAssetReader
from ..raw_photographic_review import RAW_REVIEW_FORMAT, raw_sources, validate_raw_review_manifest
from ..review_storage import resolve_review_path

FORMAT = "darsia-annotation-camera-basis-transfer-v1"
REGION_FIELDS = ("annotations", "phase_quiz_answers")
FRAME_FIELDS = ("frame_choices", "frame_verdicts")


class UnboundAnnotation(ValueError):
    """Saved evidence has no unique authenticated source frame."""


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _bound(value):
    if (not isinstance(value, Mapping) or not isinstance(value.get("path"), str)
            or not isinstance(value.get("sha256"), str) or len(value["sha256"]) != 64):
        raise ValueError("Annotation transfer needs an exact metadata path and SHA-256")
    return {key: value[key] for key in ("path", "sha256", "size_bytes") if key in value}


def _matrix(value):
    h = np.asarray(value, np.float64)
    if (h.shape != (3, 3) or not np.isfinite(h).all()
            or abs(np.linalg.det(h)) < 1e-12 or abs(h[2, 2]) < 1e-12):
        raise ValueError("Annotation geometry needs a finite invertible homography")
    return h / h[2, 2]


def _layout(manifest):
    canvas = tuple(manifest.get("canvas_wh") or ())
    content = tuple(manifest.get("content_box_xywh") or ())
    if (len(canvas) != 2 or len(content) != 4
            or any(type(v) is not int for v in (*canvas, *content))
            or min(canvas) <= 0 or min(content[2:]) <= 0
            or min(content[:2]) < 0
            or content[0] + content[2] > canvas[0]
            or content[1] + content[3] > canvas[1]):
        raise ValueError("Annotation review has no exact native canvas/content geometry")
    return canvas, content


def _rows(manifest):
    rows = manifest.get("frames") if manifest.get("format") == RAW_REVIEW_FORMAT else manifest.get("frame_sources")
    if (not isinstance(rows, list) or not rows
            or [row.get("frame_index") for row in rows] != list(range(len(rows)))
            or manifest.get("frame_count") != len(rows)):
        raise ValueError("Annotation review has an incomplete or unordered frame timeline")
    return rows


def _receipt(frame, *, baseline=False):
    prefix = "v5_baseline_receipt" if baseline else "v5_frame_receipt"
    return _bound(dict(path=frame.get(prefix + "_path"), sha256=frame.get(prefix + "_sha256")))


def _legacy_acquisition(frame, reader, *, baseline=False):
    bound = _receipt(frame, baseline=baseline)
    receipt = reader.json(bound)
    source = receipt.get("source") or {}
    if not source.get("camera_arw") or not source.get("camera_arw_sha256"):
        raise UnboundAnnotation("Source frame has no unique RAW acquisition (for example a historical composite)")
    raw = _bound(dict(path=source.get("camera_arw"), sha256=source.get("camera_arw_sha256")))
    if not baseline and (raw["sha256"] != frame.get("v5_raw_sha256")
                         or receipt.get("frame_index") != frame.get("frame_index")
                         or receipt.get("elapsed_h") != frame.get("elapsed_h")):
        raise ValueError("Old annotation receipt belongs to another RAW acquisition/timeline")
    return raw, bound, receipt


def source_manifest_binding(job, *, reader=None):
    """The original Studio job's SHA-bound native manifest, through storage aliases."""
    root = resolve_review_path(job.get("analysis_source_review_root") or job["review_output_dir"])
    run = str(job.get("run") or "").casefold()
    digest = job.get("native_manifest_sha256") or (job.get("v5_raw_photometry") or {}).get("materialized_manifest_sha256")
    bound = _bound(dict(path=str(root / "native_frames" / run / "manifest.json"), sha256=digest))
    (reader or VerifiedAssetReader()).json(bound)
    return bound


def required_annotation_sources(decisions_bound, combined_bound, *, reader=None, on_unbound="error",
                                source_native_configuration=None):
    """List exact annotated RAWs, their original immediate predecessors and B.

    This never hashes or opens acquisitions. Their identities are taken only
    from authenticated producer metadata; the normal producer verifies pixels.
    """
    reader = reader or VerifiedAssetReader()
    if on_unbound not in ("error", "exclude"):
        raise ValueError("Choose error or explicit exclusion for unbound old annotations")
    decisions_bound, combined_bound = _bound(decisions_bound), _bound(combined_bound)
    decisions, combined = reader.json(decisions_bound), reader.json(combined_bound)
    jobs = {}
    for job in combined.get("jobs", ()):
        run = str(job.get("run") or "").casefold()
        if run in jobs:
            previous = source_manifest_binding(jobs[run], reader=reader)
            current = source_manifest_binding(job, reader=reader)
            if previous["sha256"] != current["sha256"]:
                raise ValueError("A review bundle has conflicting source jobs for " + run)
        jobs[run] = job
    result, excluded = {}, []
    for original_run, decision in decisions.get("decisions", {}).items():
        annotations = [(field, item) for field in (*REGION_FIELDS, *FRAME_FIELDS)
                       for item in decision.get(field) or []]
        if not annotations:
            continue
        run = str(original_run).casefold()
        if run not in jobs:
            raise ValueError("Annotated run has no bound source review: " + run)
        bound = source_manifest_binding(jobs[run], reader=reader)
        marker = jobs[run].get("v5_raw_photometry") or {}
        processor_provenance = (_bound(dict(path=marker["batch_receipt_path"],
            sha256=marker.get("batch_receipt_sha256"))) if marker.get("batch_receipt_path") else None)
        manifest = reader.json(bound)
        if str(manifest.get("run") or "").casefold() != run:
            raise ValueError("Annotation source manifest belongs to another run")
        rows = _rows(manifest)
        wanted = {}
        valid = []
        for field, annotation in annotations:
            try:
                if not isinstance(annotation, Mapping):
                    raise UnboundAnnotation("Historical frame verdict map lacks item photograph identities")
                index = annotation.get("frame_index")
                if type(index) is not int or not 0 <= index < len(rows):
                    raise UnboundAnnotation("Annotation has no exact original timeline frame")
                if annotation.get("elapsed_h") != rows[index].get("elapsed_h"):
                    raise UnboundAnnotation("Annotation time differs from its original frame")
                # Admit the region and its history atomically. An old composite
                # is not silently replaced with one of its contributing RAWs.
                if manifest.get("format") != RAW_REVIEW_FORMAT:
                    _legacy_acquisition(rows[index], reader)
                    if index:
                        _legacy_acquisition(rows[index - 1], reader)
                if source_native_configuration is not None:
                    _frame_basis(manifest, rows[index], reader, source_native_configuration, processor_provenance)
            except UnboundAnnotation as error:
                if on_unbound == "error":
                    raise
                excluded.append(dict(run=run, field=field, source_annotation=deepcopy(annotation),
                    source_annotation_sha256=_digest(annotation), reason=str(error), source_manifest=bound))
                continue
            valid.append(annotation)
            wanted.setdefault(index, set()).add("annotation" if field in REGION_FIELDS else "frame_event")
            if index:
                wanted.setdefault(index - 1, set()).add("history")
        if not wanted:
            continue
        acquisitions = []
        for index, roles in sorted(wanted.items()):
            frame = rows[index]
            raw = (_bound(frame["acquisition"]) if manifest.get("format") == RAW_REVIEW_FORMAT
                   else _legacy_acquisition(frame, reader)[0])
            acquisitions.append(dict(frame_index=index, elapsed_h=frame["elapsed_h"],
                                     acquisition=raw, roles=sorted(roles)))
        reference = manifest.get("reference") or {}
        baseline = (_bound(reference["acquisition"]) if manifest.get("format") == RAW_REVIEW_FORMAT
                    else _legacy_acquisition(reference, reader, baseline=True)[0])
        matches = [row for row in acquisitions if row["acquisition"]["sha256"] == baseline["sha256"]]
        if matches:
            matches[0]["roles"].append("baseline")
        else:
            acquisitions.insert(0, dict(frame_index=None, elapsed_h=None, acquisition=baseline, roles=["baseline"]))
        result[run] = dict(source_manifest=bound, acquisitions=acquisitions,
                           annotation_ids=[a.get("id") or a.get("proposal_id") for a in valid],
                           source_processor_provenance=processor_provenance)
    return dict(runs=result, excluded_annotations=excluded,
                provenance=dict(decisions=decisions_bound, combined_manifest=combined_bound))


def _configuration_geometry(bound, profile, reader):
    document = reader.json(_bound(bound))
    registration = document.get("registration", document)
    try:
        common = registration["common"]
        sensor = common["sensor"]
        # Sampling geometry only: photometry, crop and fitting policy may change.
        geometry = dict(camera_wh=sensor["camera_wh"], raw_wh=sensor["raw_wh"],
                        lens=common["lens"], profile=common["profiles"][profile])
    except (KeyError, TypeError) as error:
        raise ValueError("The bound configuration lacks its camera-U lens basis") from error
    return geometry, document


def _check_certificate_geometry(certificate, geometry):
    if (certificate.get("camera_jpeg_wh") != geometry["camera_wh"]
            or certificate.get("raw_visible_wh") != geometry["raw_wh"]
            or certificate.get("sony_distortion_table") != geometry["profile"]["distortion"]
            or certificate.get("sony_ca_red_table") != geometry["lens"]["ca_red"]
            or certificate.get("sony_ca_blue_table") != geometry["lens"]["ca_blue"]):
        raise ValueError("Renderer certificate and declared camera-U lens basis disagree")


def _original_renderer_declaration(receipt, raw, processor_provenance, original, reader):
    """Authenticate a historical source-specific wrapper's unchanged frames.

    A wrapper may declare one exceptional acquisition. It does not make that
    acquisition compatible with the original lens parameters; those labels
    remain unbound until an exact compatible source basis is supplied.
    """
    if processor_provenance is None:
        raise UnboundAnnotation("Old renderer parameters lack their original source-code identity")
    batch = reader.json(_bound(processor_provenance))
    identities = list((batch.get("processor_identities") or {}).values())
    processor = receipt.get("processor") or {}
    matches = [value for value in identities if value == processor]
    if len(matches) != 1:
        raise UnboundAnnotation("Old renderer wrapper has no exact materialization declaration")
    declaration = matches[0]
    exception = declaration.get("source_exact_alt_lens_exception") or {}
    base = [value for value in identities if value.get("processor_module_sha256") ==
            declaration.get("base_processor_module_sha256")]
    original_hashes = {row.get("git_blob_sha256") for row in original.get("source_modules", ())}
    if (len(base) != 1 or base[0].get("renderer_module_sha256") not in original_hashes
            or exception.get("all_other_frames_use_original_renderer") is not True
            or not isinstance(exception.get("arw_sha256"), str) or len(exception["arw_sha256"]) != 64):
        raise UnboundAnnotation("Old renderer wrapper does not prove its original lens parameters")
    if raw["sha256"] == exception["arw_sha256"]:
        raise UnboundAnnotation("Source-specific alternative-lens acquisition has no compatible original camera-U basis")
    return dict(materialization=_bound(processor_provenance), declaration=deepcopy(declaration),
                original_renderer_sha256=base[0]["renderer_module_sha256"])


def _frame_basis(manifest, frame, reader, source_configuration=None, processor_provenance=None):
    canvas, content = _layout(manifest)
    if manifest.get("format") == RAW_REVIEW_FORMAT:
        validate_raw_review_manifest(manifest, manifest["run"])
        native = reader.json(_bound(frame["native_record"]))
        sources = raw_sources(native["full"], _bound(frame["audit"]), reader=reader)
        audit = reader.json(_bound(frame["audit"]))
        certificate = audit["renderer"]
        if (sources["acquisition"]["sha256"] != frame["acquisition"]["sha256"]
                or (native.get("audit") or {}).get("sha256") != frame["audit"]["sha256"]
                or (audit.get("processor", {}).get("source_binding") or {}).get("sha256")
                    != manifest["raw_binding"]["sha256"]
                or tuple(sources["content_box_xywh"]) != content
                or tuple(sources["phase_shape_hw"]) != (canvas[1], canvas[0])):
            raise ValueError("RAW review frame and native producer geometry disagree")
        config = _bound(manifest["native_configuration"])
        geometry, _ = _configuration_geometry(config, sources["profile_id"], reader)
        _check_certificate_geometry(certificate, geometry)
        return dict(raw=sources["acquisition"], h=_matrix(sources["native_to_camera_u_h"]),
                    geometry=geometry, config=config, audit=_bound(frame["audit"]), canvas=canvas, content=content)
    raw, pin, receipt = _legacy_acquisition(frame, reader)
    certificate = (receipt.get("certificate") or {}).get("renderer_certificate") or {}
    if source_configuration is None:
        raise ValueError("Old V5 annotation transfer requires its authenticated original camera configuration")
    config = _bound(source_configuration)
    geometry, document = _configuration_geometry(config, certificate.get("profile"), reader)
    original_bound = _bound(document.get("native_parameter_source"))
    original_geometry, original = _configuration_geometry(original_bound, certificate.get("profile"), reader)
    renderer_hash = (receipt.get("processor") or {}).get("renderer_module_sha256")
    if geometry != original_geometry:
        raise ValueError("The declared original camera parameters changed from their bound source")
    if not renderer_hash or renderer_hash not in {
            row.get("git_blob_sha256") for row in original.get("source_modules", ())}:
        wrapper = _original_renderer_declaration(receipt, raw, processor_provenance, original, reader)
    else:
        wrapper = None
    _check_certificate_geometry(certificate, geometry)
    expected_transform = sha256(json.dumps(certificate, sort_keys=True, default=str).encode()).hexdigest()
    if ((receipt.get("certificate") or {}).get("transform_sha256") != expected_transform
            or certificate.get("sources", {}).get("arw_sha256") != raw["sha256"]
            or tuple(certificate.get("review_canvas_wh") or ()) != canvas
            or tuple(certificate.get("content_box_xywh") or ()) != content
            or receipt.get("run") != manifest.get("run")):
        raise ValueError("Old renderer transform is not bound to this original annotation canvas/RAW")
    return dict(raw=raw, h=_matrix(certificate.get("review_to_camera_h")), geometry=geometry,
                config=config, audit=pin, original_parameters=original_bound, canvas=canvas, content=content,
                source_renderer_declaration=wrapper)


def _clip(polygon, width, height):
    """Clip a polygon to the target native pixel lattice, not vertex-wise clamp."""
    points = [np.asarray(p, np.float64) for p in polygon]
    for axis, bound, lower in ((0, 0., True), (0, width - 1., False),
                               (1, 0., True), (1, height - 1., False)):
        output = []
        if not points:
            break
        previous = points[-1]
        inside_previous = previous[axis] >= bound if lower else previous[axis] <= bound
        for point in points:
            inside = point[axis] >= bound if lower else point[axis] <= bound
            if inside != inside_previous:
                fraction = (bound - previous[axis]) / (point[axis] - previous[axis])
                output.append(previous + fraction * (point - previous))
            if inside:
                output.append(point)
            previous, inside_previous = point, inside
        points = output
    return np.asarray(points, np.float64).reshape(-1, 2)


def transfer_raw_annotation(annotation, source_manifest_bound, target_manifest_bound, *,
                            source_native_configuration=None, source_processor_provenance=None, reader=None,
                            left_out=None):
    """Return a normal review polygon and exact provenance on a new RAW basis.

    Rectangle vertices use the original trainer's clip/round convention before
    projection. Missing identities and changes to the camera-U lens basis are
    rejected; wholly cropped annotations are reported explicitly. ``left_out``
    maps the RAW SHA-256 of frames left out of the target to the reason; an
    annotation on such a frame is excluded with that reason.
    """
    reader = reader or VerifiedAssetReader()
    source_pin, target_pin = _bound(source_manifest_bound), _bound(target_manifest_bound)
    source, target = reader.json(source_pin), reader.json(target_pin)
    if (source.get("run") != target.get("run") or target.get("format") != RAW_REVIEW_FORMAT):
        raise ValueError("Annotation transfer needs the same run and a current RAW target")
    rows = _rows(source)
    index = annotation.get("frame_index")
    if (type(index) is not int or not 0 <= index < len(rows)
            or annotation.get("elapsed_h") != rows[index].get("elapsed_h")):
        raise ValueError("Annotation lacks an exact original frame/time binding")
    before = _frame_basis(source, rows[index], reader, source_native_configuration, source_processor_provenance)
    identity = annotation.get("photo_identity") or {}
    labelled = annotation.get("source_label_identity") or {}
    declared = [value for value in (identity.get("source_sha256"),
                                   labelled.get("camera_source_sha256"), annotation.get("source_sha256")) if value]
    allowed = {before["raw"]["sha256"], rows[index].get("source_sha256")}
    if (any(value not in allowed for value in declared)
            or identity.get("run", source["run"]) != source["run"]
            or labelled.get("run", source["run"]) != source["run"]):
        raise ValueError("Annotation photograph identity differs from its source frame")
    matches = [row for row in _rows(target) if row["acquisition"]["sha256"] == before["raw"]["sha256"]]
    if not matches and before["raw"]["sha256"] in (left_out or {}):
        return dict(annotation=None, provenance=dict(source_manifest=source_pin, target_manifest=target_pin,
            acquisition=deepcopy(before["raw"])),
            exclusion_reason="frame-left-out: " + left_out[before["raw"]["sha256"]])
    if len(matches) != 1:
        raise ValueError("Annotation RAW acquisition needs one exact current target frame")
    after = _frame_basis(target, matches[0], reader)
    if before["geometry"] != after["geometry"]:
        raise ValueError("Annotation source and target use different camera-U lens geometry")
    points = annotation.get("points") or []
    if not points:
        if annotation.get("shape") not in (None, "rectangle"):
            raise ValueError("Only saved review rectangles and polygons can be transferred")
        x, y, w, h = (float(annotation[k]) for k in ("x", "y", "width", "height"))
        if w <= 0 or h <= 0:
            raise ValueError("Annotation rectangle must have positive dimensions")
        points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    points = np.asarray(points, np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Annotation has no finite review polygon")
    native = points * before["canvas"] - before["content"][:2]
    native = np.rint(np.clip(native, [0, 0], np.array(before["content"][2:]) - 1))
    transform = _matrix(np.linalg.inv(after["h"]) @ before["h"])
    homogeneous = np.column_stack((native, np.ones(len(native)))) @ transform.T
    denominator = homogeneous[:, 2]
    if np.any(np.abs(denominator) < 1e-12) or (denominator.min() < 0 < denominator.max()):
        raise ValueError("Annotation projection crosses a projective horizon")
    projected = homogeneous[:, :2] / denominator[:, None]
    clipped = _clip(projected, *after["content"][2:])
    provenance = dict(format=FORMAT, source_annotation=deepcopy(dict(annotation)),
        source_annotation_sha256=_digest(annotation), source_manifest=source_pin, target_manifest=target_pin,
        acquisition=deepcopy(before["raw"]), source_audit=before["audit"], target_audit=after["audit"],
        source_configuration=before["config"], target_configuration=after["config"],
        camera_geometry_sha256=_digest(before["geometry"]),
        H_source_native_to_camera_u=before["h"].tolist(), H_target_native_to_camera_u=after["h"].tolist(),
        H_source_native_to_target_native=transform.tolist(), source_canvas_wh=list(before["canvas"]),
        source_content_box_xywh=list(before["content"]), target_canvas_wh=list(after["canvas"]),
        target_content_box_xywh=list(after["content"]), source_vertex_rounding="original-review-region-rint-clip",
        target_clipped=not (len(projected) == len(clipped) and np.allclose(projected, clipped)),
        source_frame_index=index, target_frame_index=matches[0]["frame_index"])
    if "original_parameters" in before:
        provenance["source_original_parameters"] = before["original_parameters"]
    if before.get("source_renderer_declaration") is not None:
        provenance["source_renderer_declaration"] = before["source_renderer_declaration"]
    area = 0. if len(clipped) < 3 else abs(float(np.dot(clipped[:, 0], np.roll(clipped[:, 1], 1)) -
                                             np.dot(clipped[:, 1], np.roll(clipped[:, 0], 1)))) / 2
    if area < .5:
        return dict(annotation=None, provenance=provenance, exclusion_reason="outside-current-native-crop")
    normalized = (clipped + after["content"][:2]) / after["canvas"]
    result = deepcopy(dict(annotation))
    for key in ("source_label_identity", "source_binding_contract", "source_sha256"):
        result.pop(key, None)
    from ..phase_annotation_binding import stamp_camera_identity
    from ..raw_photographic_review import studio_rows
    dataset = source.get("dataset") or identity.get("dataset") or ""
    result = stamp_camera_identity(result, studio_rows(target)[matches[0]["frame_index"]],
                                   target["run"], dataset=dataset)
    result.update(shape="polygon", points=normalized.tolist(), x=float(normalized[:, 0].min()),
                  y=float(normalized[:, 1].min()), width=float(np.ptp(normalized[:, 0])),
                  height=float(np.ptp(normalized[:, 1])), frame_index=matches[0]["frame_index"],
                  elapsed_h=matches[0]["elapsed_h"], coordinate_space="normalized_native_review_canvas_16x9",
                  basis_transfer=provenance)
    return dict(annotation=result, provenance=provenance, exclusion_reason=None)


def transfer_bundle(decisions_bound, combined_bound, target_manifests, *,
                    source_native_configuration=None, reader=None, on_unbound="error", left_out=None):
    """Build new ordinary Studio documents; the caller owns their output folder.

    ``left_out`` maps a run to {RAW SHA-256: reason} for frames left out of its
    target review; annotations and frame events on them are excluded with the
    reason instead of stopping the transfer.
    """
    left_out = {str(run).casefold(): dict(value) for run, value in (left_out or {}).items()}
    reader = reader or VerifiedAssetReader()
    plan = required_annotation_sources(decisions_bound, combined_bound, reader=reader, on_unbound=on_unbound,
                                       source_native_configuration=source_native_configuration)
    decisions = deepcopy(reader.json(_bound(decisions_bound)))
    combined = deepcopy(reader.json(_bound(combined_bound)))
    excluded = deepcopy(plan["excluded_annotations"])
    excluded_keys = {(row["run"], row["field"], row["source_annotation_sha256"]) for row in excluded}
    for original_run, decision in decisions.get("decisions", {}).items():
        run = str(original_run).casefold()
        for field in (*REGION_FIELDS, *FRAME_FIELDS):
            if field in decision:
                decision[field] = [item for item in decision[field]
                    if (run, field, _digest(item)) not in excluded_keys]
        if run not in plan["runs"]:
            continue
        for field in REGION_FIELDS:
            if field not in decision:
                continue
            projected = []
            for annotation in decision[field]:
                result = transfer_raw_annotation(annotation, plan["runs"][run]["source_manifest"],
                    target_manifests[run], source_native_configuration=source_native_configuration,
                    source_processor_provenance=plan["runs"][run].get("source_processor_provenance"), reader=reader,
                    left_out=left_out.get(run))
                if result["annotation"] is None:
                    excluded.append(dict(run=run, field=field, **result))
                else:
                    projected.append(result["annotation"])
            decision[field] = projected
        for field in FRAME_FIELDS:
            kept_events = []
            for event in decision.get(field) or ():
                source = reader.json(plan["runs"][run]["source_manifest"])
                frame = _rows(source)[event["frame_index"]]
                raw = (_bound(frame["acquisition"]) if source.get("format") == RAW_REVIEW_FORMAT
                       else _legacy_acquisition(frame, reader)[0])
                target = reader.json(_bound(target_manifests[run]))
                matches = [row for row in _rows(target) if row["acquisition"]["sha256"] == raw["sha256"]]
                if not matches and raw["sha256"] in left_out.get(run, {}):
                    excluded.append(dict(run=run, field=field, source_event=deepcopy(event), acquisition=raw,
                        exclusion_reason="frame-left-out: " + left_out[run][raw["sha256"]]))
                    continue
                if len(matches) != 1:
                    raise ValueError("Frame review event has no unique current RAW frame")
                kept_events.append(event)
                event.update(frame_index=matches[0]["frame_index"], elapsed_h=matches[0]["elapsed_h"],
                    basis_transfer=dict(format=FORMAT, source_event=deepcopy(event),
                        source_manifest=plan["runs"][run]["source_manifest"], target_manifest=_bound(target_manifests[run]),
                        acquisition=raw))
                for key in ("source_label_identity", "source_binding_contract", "source_sha256", "photo_identity"):
                    event.pop(key, None)
                from ..phase_annotation_binding import stamp_camera_identity
                from ..raw_photographic_review import studio_rows
                stamped = stamp_camera_identity(event, studio_rows(target)[matches[0]["frame_index"]],
                    run, dataset=source.get("dataset") or "")
                event.update(stamped)
            if field in decision:
                decision[field] = kept_events
    for job in combined.get("jobs", ()):
        run = str(job.get("run") or "").casefold()
        if run not in plan["runs"]:
            continue
        target = _bound(target_manifests[run])
        path = reader.path(target)
        root = str(path.parent.parent.parent)
        original_job = deepcopy(job)
        descriptive = {key: deepcopy(job[key]) for key in (
            "job_id", "run", "dataset", "role", "family", "variant", "grid_order",
            "candidate_id", "params_sha256", "quality_flags") if key in job}
        job.clear()
        job.update(descriptive)
        job.update(review_output_dir=root, analysis_source_review_root=root,
                   native_manifest_sha256=target["sha256"])
        # Historical output/model paths are provenance, not current inputs.
        job["annotation_transfer"] = dict(source_manifest=plan["runs"][run]["source_manifest"],
                                          target_manifest=target, original_job=original_job)
    provenance = dict(format=FORMAT, **plan["provenance"], excluded_annotations=excluded)
    decisions["annotation_basis_transfer"] = deepcopy(provenance)
    combined["annotation_basis_transfer"] = deepcopy(provenance)
    return dict(decisions=decisions, combined_manifest=combined, provenance=provenance)
