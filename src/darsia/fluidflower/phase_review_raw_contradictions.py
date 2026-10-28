"""Read qualified saved RAW phase predictions, without recreating science.

The supported prediction route is the original registered RAW production
review. Its labels and final modelled/filled classes come from the same saved
production frame. The original RAW validity reader supplies the current
scientific support. Photographic support, inherited categorical caches and
newly reconstructed feature masks are never substitutes.

V5 scientific validity has a separate read-only qualification API. It does not
claim to be V5 feature support: the original V5 reader also warps baseline and
history validity and erodes feature support. Its receipt records a mask digest,
not a loadable saved mask. That prediction route remains explicitly unavailable
until an original saved final domain/support product can be bound to the map.

All arrays returned to the evaluator are exact native-content rectangles. The
original manifest and full phase-grid geometry remain in the result. No files,
display copies, models, queues or publication policies are changed here.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from .phase_review_basis import SAVED_NATIVE_VERSION as SAVED_NATIVE_BASIS

CLASS_FIELDS = ("rig_domain", "modelled", "filled_observed", "filled_unobserved")
V5_SUPPORT_UNAVAILABLE = (
    "V5 saved final domain/feature-support arrays are unavailable: original "
    "V5FrameGeometryCohort.feature_support records a digest, while rebuilding "
    "its warped/eroded support is outside saved-prediction review"
)


class UnsupportedSavedPhaseFrame(ValueError):
    """Required saved support is absent; this is unscored, never agreement."""


@dataclass(frozen=True)
class SavedPhaseFrame:
    """Native-content uint8 labels and bool domain/support, with original pins.

    Label codes retain the original pipeline encoding: water=0, aqueous=1,
    gas=2, and outside/unresolved=255. A consumer must reject a region touching
    ``domain & ~support``; it must not count excluded/filled pixels as water.
    ``native`` is the unchanged full original manifest, not a cropped manifest.
    """

    labels: Any
    native: dict
    domain: Any
    support: Any
    pins: dict[str, str]
    provenance: dict


@dataclass(frozen=True)
class SavedV5Validity:
    """Original source validity only; not a prediction domain or feature mask."""

    valid: Any
    native: dict
    pins: dict[str, str]
    provenance: dict


class _Pins:
    """Record every original reader input and verify the bytes again at return."""

    def __init__(self):
        from .phase_saved_assets import VerifiedAssetReader
        self.reader = VerifiedAssetReader()
        self.pins = {}

    def path(self, descriptor):
        from .phase_saved_review_metadata import bound_path
        if (not isinstance(descriptor, Mapping)
                or not isinstance(descriptor.get("path"), str)
                or re.fullmatch(r"[0-9a-f]{64}", str(descriptor.get("sha256"))) is None):
            raise ValueError("Saved contradiction inputs require an exact path and SHA-256")
        # The retained resolver knows the configured original path aliases.
        path = bound_path(descriptor).resolve()
        normalized = dict(descriptor, path=str(path))
        self.reader.path(normalized)
        if "size_bytes" in descriptor and (
                type(descriptor["size_bytes"]) is not int
                or path.stat().st_size != descriptor["size_bytes"]):
            raise ValueError("Saved contradiction input size differs")
        key, digest = str(path), descriptor["sha256"]
        if key in self.pins and self.pins[key] != digest:
            raise ValueError("One saved contradiction input has conflicting identities")
        self.pins[key] = digest
        return path

    def pin(self, path):
        from .phase_saved_assets import asset
        reference = asset(Path(path).resolve())
        self.path(reference)
        return reference

    def json(self, descriptor):
        path = self.path(descriptor)
        result = self.reader.json(dict(descriptor, path=str(path)))
        self.path(descriptor)
        return result

    def array(self, descriptor):
        path = self.path(descriptor)
        result = self.reader.array(dict(descriptor, path=str(path)))
        self.path(descriptor)
        return result

    def verify(self):
        from .project import sha256_file
        for path, digest in self.pins.items():
            if sha256_file(Path(path)) != digest:
                raise ValueError("Saved contradiction input changed during reading: " + path)


def _geometry(native):
    """Validate the original integer canvas/content placement; never resize."""
    canvas, box = native.get("canvas_wh"), native.get("content_box_xywh")
    if (not isinstance(canvas, (list, tuple)) or len(canvas) != 2
            or not isinstance(box, (list, tuple)) or len(box) != 4
            or any(type(value) is not int for value in (*canvas, *box))):
        raise ValueError("Saved frame needs its exact integer native geometry")
    width, height = canvas
    x, y, cw, ch = box
    if min(width, height, cw, ch) <= 0 or min(x, y) < 0 or x + cw > width or y + ch > height:
        raise ValueError("Saved native-content rectangle leaves its recorded canvas")
    return (height, width), (ch, cw), tuple(box)


def _classes(reader, descriptor, shape):
    """Validate the original pipeline's saved final class partition."""
    import numpy as np
    path = reader.path(descriptor)
    with np.load(path, allow_pickle=False) as archive:
        if not set(CLASS_FIELDS).issubset(archive.files):
            raise UnsupportedSavedPhaseFrame("RAW saved final classes lack modelled/filled/domain arrays")
        fields = {name: archive[name].copy() for name in CLASS_FIELDS}
    reader.path(descriptor)
    if any(value.dtype != np.bool_ or value.shape != shape for value in fields.values()):
        raise ValueError("RAW saved classes have another dtype or phase grid")
    # Same disjoint complete modelled/filled contract as load_rig_classes.
    total = sum(fields[name].astype(np.uint8) for name in CLASS_FIELDS[1:])
    if np.any(total > 1) or not np.array_equal(total == 1, fields["rig_domain"]):
        raise ValueError("RAW saved modelled/filled classes do not partition their original domain")
    return fields


def _categorical(reader, descriptor, shape):
    import numpy as np
    from PIL import Image
    path = reader.path(descriptor)
    with Image.open(path) as image:
        # No convert(), palette interpretation, colour quantization or thresholds.
        if image.mode != "L":
            raise ValueError("RAW final labels must retain their original uint8 categorical PNG")
        labels = np.array(image)
    reader.path(descriptor)
    if labels.dtype != np.uint8 or labels.shape != shape or not np.isin(labels, (0, 1, 2, 255)).all():
        raise ValueError("RAW final labels have another native grid or categorical encoding")
    return labels


def _read_raw_production(model_root, job, native_path, native, frame_index, reader):
    import numpy as np
    from .raw_production_review import validated_job
    from .raw_photographic_review import validate_raw_review_manifest, raw_review_record, raw_sources
    from .phase_training.raw_frame_context import RawFrameGeometryCohort
    from .phase.domain import crop_native_canvas

    record, run, recorded_native = validated_job(job)
    if Path(record["source_model_root"]).resolve() != model_root:
        raise ValueError("RAW saved predictions belong to another selected model root")
    if reader.path(run["native"]) != native_path or recorded_native != native:
        raise ValueError("RAW prediction and annotation native manifests differ")
    reader.path(job["production_review"])
    for descriptor in record["pins"]:
        reader.path(descriptor)
    reader.path(run["timeline"])
    manifest = validate_raw_review_manifest(native, job["run"])
    frame = raw_review_record(manifest, frame_index)
    full_shape, content_shape, box = _geometry(manifest)
    if (tuple(manifest["phase_shape_hw"]) != full_shape or tuple(run["shape"]) != full_shape
            or tuple(manifest["native_shape_hw"]) != content_shape
            or tuple(manifest["native_origin_xy"]) != (-box[0], -box[1])
            or len(run["labels"]) != manifest["frame_count"]
            or len(run["times"]) != manifest["frame_count"]
            or run["times"][frame_index] != frame["elapsed_h"]):
        raise ValueError("RAW saved phase grid, source timeline or native origin differs")
    classes_ref = frame.get("rig_classes")
    if not classes_ref:
        raise UnsupportedSavedPhaseFrame(
            "RAW prediction has no saved final rig_classes: photo_support cannot replace modelled/filled support")
    current = reader.json(frame["native_record"])
    if current.get("complete") is not True or current.get("processor_kind") != "raw_rig":
        raise ValueError("RAW scientific source is not its original complete native record")
    sources = raw_sources(current["full"], current["audit"], reader=reader)
    audit = reader.json(sources["audit"])
    if (sources["acquisition"] != frame["acquisition"] or sources["audit"] != frame["audit"]
            or current.get("source_sha256") != frame["acquisition"]["sha256"]
            or (audit.get("processor", {}).get("source_binding") or {}).get("sha256")
                != manifest["raw_binding"]["sha256"]
            or any(tuple(sources[key]) != tuple(manifest[key]) for key in
                   ("phase_shape_hw", "native_shape_hw", "native_origin_xy", "content_box_xywh"))):
        raise ValueError("RAW prediction source, scientific record and review geometry differ")
    for descriptor in (manifest["raw_binding"], manifest["native_configuration"],
                       frame["native_geometry"], frame["rig_geometry"], frame["acquisition"],
                       sources["log_linear_rgb"]):
        reader.path(descriptor)
    cohort = RawFrameGeometryCohort.from_manifest(native_path)
    if cohort.native_manifest_sha256 != reader.pins[str(native_path)]:
        raise ValueError("RAW source changed before the original validity reader was bound")
    cohort._reader = reader
    valid, valid_sha = cohort._valid(frame_index)
    if valid_sha != sources["valid_mask"]["sha256"]:
        raise ValueError("RAW validity belongs to another scientific source")
    fields = _classes(reader, classes_ref, full_shape)
    label_ref = run["labels"][frame_index]
    labels = _categorical(reader, label_ref, full_shape)
    # Original producer invariant: domain.fill_rig_labels initializes outside
    # the rig to 255 and resolves its complete modelled/filled partition.
    # infer.apply_locked_postprocessing preserves that outside category;
    # rules._floor_core additionally sets the excluded floor pixels to 255.
    # infer_labels clips all final classes to that same corrected domain, and
    # pipeline.runner._phase_outputs writes those labels/classes unchanged.
    # No alternate zero-background production contract is admitted here.
    if not np.array_equal(fields["rig_domain"], labels != 255):
        raise ValueError("RAW final categorical labels disagree with their original saved rig domain")
    local_labels = crop_native_canvas(labels, box).copy()
    local_domain = crop_native_canvas(fields["rig_domain"], box).copy()
    modelled = crop_native_canvas(fields["modelled"], box)
    support = modelled & valid
    provenance = dict(
        route="original-registered-raw-production", phase_review_basis=SAVED_NATIVE_BASIS,
        prediction=deepcopy(label_ref), prediction_native=dict(path=str(native_path),
            sha256=reader.pins[str(native_path)]), acquisition=deepcopy(frame["acquisition"]),
        native_record=deepcopy(frame["native_record"]), scientific_validity=deepcopy(sources["valid_mask"]),
        final_classes=deepcopy(classes_ref), run=job["run"], frame_index=frame_index,
        case_id=frame["case_id"], elapsed_h=frame["elapsed_h"],
        source_model_identity=[record["source_model_id"], record["source_model_sha256"]],
        grid="native_content", full_phase_shape_hw=list(full_shape),
        content_shape_hw=list(content_shape), content_box_xywh=list(box),
        native_origin_xy=deepcopy(manifest["native_origin_xy"]),
        label_encoding={"water": 0, "aqueous": 1, "gas": 2, "outside_or_unresolved": 255},
        support_policy="saved_final_modelled_intersect_original_raw_native_validity",
        filled_observed_pixels=int(np.count_nonzero(crop_native_canvas(fields["filled_observed"], box))),
        filled_unobserved_pixels=int(np.count_nonzero(crop_native_canvas(fields["filled_unobserved"], box))),
        unsupported_domain_pixels=int(np.count_nonzero(local_domain & ~support)),
        excluded_full_canvas_border=True, scientific_acceptance=False,
        inference_performed=False, support_recomputed=False)
    for array in (local_labels, local_domain, support):
        array.flags.writeable = False
    return SavedPhaseFrame(local_labels, deepcopy(native), local_domain, support,
                           dict(reader.pins), provenance)


def read_v5_source_validity(native_manifest_path, frame_index, *, expected_native_sha256):
    """Qualify saved V5 scientific validity through the original reader only.

    This intentionally cannot be passed as ``SavedPhaseFrame.support``. The
    original per-frame feature-support predicate also depends on baseline and
    previous-frame coverage. Neither that predicate nor a phase domain is
    reconstructed here. The caller supplies the exact previously bound native
    manifest SHA; timestamps or frame numbers are not an identity substitute.
    """
    from .phase_training.log_linear_input import RunV5LinearInput, EXACT_VALID_MASK_POLICY
    reader = _Pins()
    path = reader.path(dict(path=str(native_manifest_path), sha256=expected_native_sha256))
    native = reader.json(dict(path=str(path), sha256=expected_native_sha256))
    _, shape, _ = _geometry(native)
    rows = native.get("frame_sources") or []
    if type(frame_index) is not int or not 0 <= frame_index < len(rows):
        raise ValueError("V5 source frame is outside its recorded timeline")
    if native.get("v5_float_support_policy") != EXACT_VALID_MASK_POLICY:
        raise UnsupportedSavedPhaseFrame("V5 source has no exact saved per-frame scientific-valid mask")
    frame_id = f"frame_{frame_index:03d}"
    photo = path.parent / (frame_id + ".jpg")
    original = RunV5LinearInput(native_manifest_path=path, run_id=native["run"], content_shape_hw=shape)
    if original.manifest_sha256 != expected_native_sha256:
        raise ValueError("V5 manifest changed before the original validity reader was bound")
    row = rows[frame_index]
    for descriptor in (
            dict(path=str(photo), sha256=row["native_jpeg_sha256"]),
            dict(path=row["v5_log_linear_rgb_path"], sha256=row["v5_log_linear_rgb_sha256"]),
            dict(path=row["v5_log_linear_valid_mask_path"], sha256=row["v5_log_linear_valid_mask_sha256"])):
        reader.path(descriptor)
    valid, origin = original.load_validity(photo, frame_id, extended=False)
    if tuple(origin) != (0, 0) or valid.shape != shape:
        raise ValueError("V5 source validity has another native-content grid")
    reader.verify()
    valid = valid.copy()
    valid.flags.writeable = False
    return SavedV5Validity(valid, deepcopy(native), dict(reader.pins), dict(
        native_manifest=dict(path=str(path), sha256=expected_native_sha256),
        run=native["run"], frame_index=frame_index, source_frame=deepcopy(row),
        grid="native_content", native_origin_xy=[0, 0],
        scope="source_validity_only", prediction_support_qualified=False,
        feature_support_recomputed=False, limitation=V5_SUPPORT_UNAVAILABLE))


def load_saved_phase_frame(model_root, review_root, job, frame_index, *, expected_native_sha256=None):
    """Read an exact selected saved frame or explain its missing support route.

    The original campaign and selected job are revalidated here, independently
    of UI state. Every original prediction-publication gate is checked before
    labels are opened and again before return. This function never releases a
    hold. Call it in the admitted saved-reader worker, never on the GUI thread.
    """
    from .phase_evidence_audit import verified_review_binding
    from .phase_validation import _model_identity, _source_jobs
    from .phase_saved_runtime_views import native_source
    from .phase_blind_review_policy import require_predictions_visible
    from .raw_photographic_review import RAW_REVIEW_FORMAT
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError("An explicit nonnegative frame index is required")
    model_root, review_root = Path(model_root).resolve(), Path(review_root).resolve()
    identity = _model_identity(model_root)
    binding = verified_review_binding(model_root, review_root, saved_comparison=True)
    review_root = Path(binding["combined_root"]).resolve()
    selected = [row for row in _source_jobs(review_root) if row["run"] == job.get("run")]
    if len(selected) != 1 or selected[0] != job:
        raise ValueError("The requested frame is not the original selected campaign job")
    roots = [model_root]
    for name in ("candidate_folder", "phase_map_output_dir", "raw_phase_map_output_dir"):
        value = job.get(name)
        if value:
            path = Path(value)
            roots.append(path.resolve() if path.is_absolute() else (review_root / path).resolve())
    for root in roots:
        require_predictions_visible(root, job["run"])
    reader = _Pins()
    for path in (review_root / "postflight_manifest.json", review_root / "validation_binding.json",
                 model_root / "summary.json", model_root / "models/generic/co2_presence_signal.json",
                 model_root / "models/generic/gas_within_co2_signal.json"):
        reader.pin(path)
    _, native_path, native = native_source(job)
    native_path = Path(native_path).resolve()
    native_ref = reader.pin(native_path)
    if expected_native_sha256 is not None and native_ref["sha256"] != expected_native_sha256:
        raise ValueError("Saved frame native SHA differs from the evaluator's selected source")
    if reader.json(native_ref) != native:
        raise ValueError("Saved frame native manifest changed while its source was selected")
    if (job.get("quality_flags") or {}).get("raw_production_review") is True:
        if native.get("format") != RAW_REVIEW_FORMAT:
            raise ValueError("Registered RAW predictions lack their original RAW review source")
        result = _read_raw_production(model_root, job, native_path, native, frame_index, reader)
    elif native.get("format") == RAW_REVIEW_FORMAT:
        raise UnsupportedSavedPhaseFrame(
            "RAW optical maps lack a registered same-frame final labels/classes binding; "
            "photographic or predecessor-model support is not substituted")
    elif native.get("v5_frame_geometry") or native.get("v5_photometry"):
        raise UnsupportedSavedPhaseFrame(V5_SUPPORT_UNAVAILABLE)
    elif native.get("ffac_rig_camera_sources") or (job.get("quality_flags") or {}).get("selected_camera_review"):
        raise UnsupportedSavedPhaseFrame(
            "Dedicated saved rig/selected-camera prediction support has no qualified same-map receipt adapter; "
            "ancestral retained labels and photographic rig_domain are not substituted")
    else:
        raise UnsupportedSavedPhaseFrame("No original RAW/V5 saved prediction/support contract applies")
    if _model_identity(model_root) != identity or tuple(result.provenance["source_model_identity"]) != identity:
        raise ValueError("Selected model identity changed during saved-frame reading")
    if verified_review_binding(model_root, review_root, saved_comparison=True) != binding:
        raise ValueError("Saved campaign/profile binding changed during frame reading")
    if [row for row in _source_jobs(review_root) if row["run"] == job["run"]] != [job]:
        raise ValueError("Selected saved campaign job changed during frame reading")
    reader.verify()
    for root in roots:
        require_predictions_visible(root, job["run"])
    return result


__all__ = ["SavedPhaseFrame", "SavedV5Validity", "UnsupportedSavedPhaseFrame",
           "SAVED_NATIVE_BASIS", "load_saved_phase_frame", "read_v5_source_validity"]
