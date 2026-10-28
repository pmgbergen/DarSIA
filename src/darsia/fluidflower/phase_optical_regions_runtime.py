"""Ordinary Studio dispatch for a separately trained optical region model.

The shared post-optical stage supplies registered geometry and rendering. No
legacy context completion, plume envelope, or historical phase mask is run.
"""
import hashlib
import json
from copy import deepcopy
from pathlib import Path
import re

import cv2
import numpy as np

from . import phase_optical_regions as regional
from .project import sha256_file

VERSION = regional.VERSION
PROFILE_FILE = "phase_optical_regions.json"
INCREMENTAL_CONTRACT = "ffac-track1-incremental-inference-v1"
RUN_BINDING_CONTRACT = "ffac-track1-incremental-run-binding-v1"


def _read_bound(path, digest, label):
    path = Path(path).resolve()
    if (not isinstance(digest, str) or len(digest) != 64 or not path.is_file()
            or sha256_file(path) != digest):
        raise ValueError(f"Incremental {label} changed or is absent")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _bundle_model_identity(root):
    """The same frozen optical identity used by phase_validation, without a cycle."""
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8-sig"))
    digest = hashlib.sha256()
    for path in (root / "summary.json",
                 root / "models/generic/co2_presence_signal.json",
                 root / "models/generic/gas_within_co2_signal.json"):
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return [str(summary.get("model_id") or root.name).strip(), digest.hexdigest()]


def _unchanged_context_entry(original, entry, binding):
    """Only source identity and the declared display/support policy may be rebound."""
    before, after = deepcopy(original), deepcopy(entry)
    old_provenance = before.pop("provenance", {})
    new_provenance = after.pop("provenance", {})
    before.pop("native_manifest_sha256", None)
    after.pop("native_manifest_sha256", None)
    if before != after:
        raise ValueError("Incremental nominal context differs from frozen context")
    old_photo = old_provenance.pop("v5_track1_photo_geometry", {})
    new_photo = new_provenance.pop("v5_track1_photo_geometry", {})
    old_raw = old_provenance.pop("v5_raw_photometry", {})
    new_raw = new_provenance.pop("v5_raw_photometry", {})
    expected_photo = dict(old_photo,
        native_manifest_sha256=binding["materialized_manifest_sha256"],
        enclosed_hole_policy="fill_enclosed_v1", vertical_gap_max_length_px=64,
        vertical_gap_rule="same_column_supported_above_and_below_v1")
    if old_provenance != new_provenance or new_photo != expected_photo:
        raise ValueError("Incremental frame geometry or context provenance changed")
    allowed_raw = {"receipt_path", "receipt_sha256", "materialized_manifest_sha256",
                   "source_manifest_sha256", "physical_grid_inherited"}
    if ({key: value for key, value in old_raw.items() if key not in allowed_raw}
            != {key: value for key, value in new_raw.items() if key not in allowed_raw}):
        raise ValueError("Incremental fitted photometry provenance changed")
    if (new_raw.get("receipt_path") != binding["source_receipt_path"]
            or new_raw.get("receipt_sha256") != binding["source_receipt_sha256"]
            or new_raw.get("materialized_manifest_sha256") != binding["materialized_manifest_sha256"]
            or new_raw.get("source_manifest_sha256") != old_photo.get("original_source_manifest_sha256")
            or new_raw.get("physical_grid_inherited") is not False
            or entry.get("native_manifest_sha256") != binding["materialized_manifest_sha256"]):
        raise ValueError("Incremental context does not bind its exact source receipt")


def _incremental_overlay(root, profile):
    """Resolve append-only, independently sealed run sources; never modify the profile.

    The static profile and full acquisition plan identify one inference pipeline.
    Each run package separately binds its immutable native source SHA. Pending
    acquisitions retain their old context entry and cannot be mapped by a shard.
    """
    marker = profile.get("v5_track1_incremental_inference")
    if marker is None:
        return profile
    if (not isinstance(marker, dict) or set(marker) != {
            "plan_path", "plan_sha256", "binding_directory"}
            or marker["plan_path"] != "inference_incremental_plan.json"
            or marker["binding_directory"] != "inference_run_bindings"):
        raise ValueError("Invalid incremental inference profile marker")
    plan = _read_bound(root / marker["plan_path"], marker["plan_sha256"], "source plan")
    identity = _bundle_model_identity(root)
    expected_rows = plan.get("expected_sources") or []
    if (plan.get("contract") != INCREMENTAL_CONTRACT
            or (profile.get("v5_track1_review_scope") or {}).get("mass_analysis_allowed") is not False
            or plan.get("model_identity") != identity
            or plan.get("run_count") != 43 or plan.get("frame_count") != 864
            or len(expected_rows) != 43
            or sum(row.get("frame_count", 0) for row in expected_rows) != 864
            or not isinstance(plan.get("source_job_path"), str)):
        raise ValueError("Incremental plan has another model or acquisition cohort")
    expected = {row.get("run"): row for row in expected_rows}
    if (len(expected) != 43 or any(not isinstance(run, str) or not re.fullmatch(r"ac\d+", run)
                                 for run in expected)
            or set(profile["runs"]) != {f"ffac/{run}" for run in expected}):
        raise ValueError("Incremental plan run set differs from frozen profile")
    source_job = _read_bound(plan["source_job_path"], plan.get("source_job_sha256"), "source job")
    versions = source_job.get("source_versions") or []
    if (source_job.get("inference_only") is not True or source_job.get("retraining") is not False
            or source_job.get("labels_changed") is not False
            or {row.get("run"): row.get("sha256") for row in versions}
            != {run: row.get("source_manifest_sha256") for run, row in expected.items()}
            or len(versions) != 43):
        raise ValueError("Incremental source job differs from the acquisition plan")
    static = deepcopy(profile)
    result = deepcopy(profile)
    sealed = []
    directory = root / marker["binding_directory"]
    for path in sorted(directory.glob("*.json")):
        binding = json.loads(path.read_text(encoding="utf-8-sig"))
        run = binding.get("run")
        if (run not in expected or path.name != f"{run}.json"
                or binding.get("contract") != RUN_BINDING_CONTRACT
                or binding.get("plan_sha256") != marker["plan_sha256"]
                or binding.get("model_identity") != identity):
            raise ValueError("Incremental run binding identifies another plan/model/run")
        event = binding.get("worker_completed_event") or {}
        cache = event.get("source_verification_cache") or {}
        if (event.get("event") != "run_complete" or event.get("run") != run
                or event.get("source_manifest_sha256") != expected[run]["source_manifest_sha256"]
                or event.get("frame_count") != expected[run]["frame_count"]
                or cache.get("sealed_actual_sha_rechecks") is not True
                or any(event.get(event_key) != binding.get(binding_key)
                       for event_key, binding_key in (
                           ("materialized_manifest_path", "materialized_manifest_path"),
                           ("materialized_manifest_sha256", "materialized_manifest_sha256"),
                           ("receipt_path", "source_receipt_path"),
                           ("receipt_sha256", "source_receipt_sha256")))):
            raise ValueError("Incremental source worker has not sealed its exact acquisition")
        native = _read_bound(binding["materialized_manifest_path"],
                             binding["materialized_manifest_sha256"], f"{run} native source")
        receipt = _read_bound(binding["source_receipt_path"], binding["source_receipt_sha256"],
                              f"{run} materialization receipt")
        if (receipt.get("run") != run
                or receipt.get("source_manifest_sha256") != expected[run]["source_manifest_sha256"]
                or receipt.get("materialized_manifest_sha256") != binding["materialized_manifest_sha256"]
                or receipt.get("materialized_manifest_path") != binding["materialized_manifest_path"]
                or receipt.get("frame_count") != expected[run]["frame_count"]):
            raise ValueError("Incremental materialization receipt identifies another acquisition")
        original_sources = plan.get("original_sources_root")
        original_path = (Path(original_sources) / run / "original_job.json" if original_sources
                         else Path(plan["old_candidate"]) / "review_source" / run / "original_job.json")
        original_job = _read_bound(original_path,
            (plan.get("original_source_job_sha256_by_run") or {}).get(run), f"{run} original source job")
        from .phase_saved_runtime_views import native_source
        _, old_native_path, old_native = native_source(original_job)
        key = f"ffac/{run}"
        original_entry = profile["runs"][key]
        if sha256_file(old_native_path) != original_entry.get("native_manifest_sha256"):
            raise ValueError("Incremental original native source changed")
        old_photo = old_native.get("v5_track1_photo_geometry") or {}
        photo = native.get("v5_track1_photo_geometry") or {}
        photometry = native.get("v5_photometry") or {}
        processor = photometry.get("processor") or {}
        old_processor = (old_native.get("v5_photometry") or {}).get("processor") or {}
        if (native.get("run") != run or native.get("frame_count") != expected[run]["frame_count"]
                or photo.get("original_source_manifest_sha256") != expected[run]["source_manifest_sha256"]
                or photo.get("original_source_manifest_sha256") != old_photo.get("original_source_manifest_sha256")
                or photo.get("frames") != old_photo.get("frames")
                or photo.get("enclosed_hole_policy") != "fill_enclosed_v1"
                or photo.get("vertical_gap_max_length_px") != 64
                or photo.get("vertical_gap_rule") != "same_column_supported_above_and_below_v1"
                or processor.get("enclosed_hole_policy") != "fill_enclosed_v1"
                or processor.get("enclosed_hole_max_area_px") != 60000
                or processor.get("enclosed_hole_vertical_gap_max_px") != 64
                or processor.get("display_mode") != plan.get("display_mode", "photographic_v1")
                or (plan.get("display_mode") == "natural_modes_v1" and
                    processor.get("isolated_invalid_policy") != plan.get("isolated_invalid_policy"))
                or photometry.get("display_jpeg_is_training_input") is not False
                or photometry.get("unit_denominator") != 65535):
            raise ValueError("Incremental photo geometry or scientific input policy changed")
        for name in ("analysis_image_shape_hw", "canvas_wh", "content_box_xywh"):
            if native.get(name) != old_native.get(name):
                raise ValueError("Incremental native grid changed")
        frames, old_frames = native.get("frame_sources") or [], old_native.get("frame_sources") or []
        if len(frames) != len(old_frames) or len(frames) != expected[run]["frame_count"]:
            raise ValueError("Incremental source timeline is incomplete")
        for before, after in zip(old_frames, frames):
            if any(before.get(name) != after.get(name) for name in (
                    "frame_index", "elapsed_h", "source_sha256", "source_raw_sha256",
                    "v5_raw_sha256", "v5_raw_members")):
                raise ValueError("Incremental original acquisition or timestamp changed")
        for name in ("calibration_manifest_sha256", "photo_source_aggregate_sha256"):
            if old_processor.get(name) is not None and processor.get(name) != old_processor[name]:
                raise ValueError("Incremental fitted normalization changed")
        entry = binding.get("context_entry")
        if not isinstance(entry, dict):
            raise ValueError("Incremental binding lacks its frozen context")
        _unchanged_context_entry(original_entry, entry, binding)
        context_path = (root / entry["path"]).resolve()
        if not context_path.is_relative_to(root) or sha256_file(context_path) != entry["sha256"]:
            raise ValueError("Incremental nominal context array changed or escaped bundle")
        result["runs"][key] = deepcopy(entry)
        sealed.append(run)
    result["_incremental_static_public"] = static
    result["_incremental_sealed_runs"] = sealed
    return result


def load_profile(root):
    root = Path(root).resolve()
    path = root / PROFILE_FILE
    if not path.is_file():
        return None
    profile = json.loads(path.read_text(encoding="utf-8-sig"))
    if profile.get("format") != VERSION or not profile.get("runs"):
        raise ValueError("Invalid optical region profile")
    model_path = (root / profile["model"]["path"]).resolve()
    if not model_path.is_relative_to(root) or sha256_file(model_path) != profile["model"]["sha256"]:
        raise ValueError("Trained optical region artifact changed or escaped its model bundle")
    model = json.loads(model_path.read_text(encoding="utf-8-sig"))
    if model.get("format") != VERSION or tuple(model["feature_names"]) != regional.FEATURE_NAMES:
        raise ValueError("Trained optical region feature contract differs")
    regional.Settings(**model["settings"])
    return dict(_incremental_overlay(root, profile), _learned_model=model)


def runtime_identity(profile):
    public = profile.get("_incremental_static_public")
    if public is None:
        public = {key:value for key,value in profile.items() if not key.startswith("_")}
    digest = hashlib.sha256(json.dumps(public,sort_keys=True).encode())
    for path in (__file__,regional.__file__):
        digest.update(sha256_file(Path(path)).encode())
    digest.update(f"{np.__version__}:{cv2.__version__}".encode())
    return digest.hexdigest()


class OpticalRegionSequence:
    def __init__(self, context, profile):
        self.context=context
        self.model=profile["_learned_model"]
        self.settings=regional.Settings(**self.model["settings"])
        self.baseline=None
        self.last_diagnostics={}

    def bind_reference(self, baseline):
        if self.baseline is not None:
            raise ValueError("The optical region reference is immutable")
        self.baseline=np.array(baseline,copy=True)

    def bind_native_frame_reference(self, baseline):
        """Use a registered reference on this native frame without moving its photo.

        The original pre-injection array remains immutable.  V5's exact
        per-frame sand transform supplies this aligned feature reference; it
        must be applied in both regional training and final-map inference.
        """
        if self.baseline is None:
            raise ValueError("Bind the original optical reference first")
        source=getattr(self, "source_baseline", self.baseline)
        value=np.asarray(baseline)
        if (value.shape != source.shape
                or value.dtype != source.dtype
                or not np.isfinite(value).all()):
            raise ValueError("Native-frame optical reference differs from its source grid")
        if not hasattr(self, "source_baseline"):
            self.source_baseline=source
        self.baseline=value.view()
        self.baseline.setflags(write=False)

    def apply_with_reference(self, predictions, elapsed_h, current):
        if self.baseline is None:
            raise ValueError("Bind the registered pre-injection reference before prediction")
        p=predictions["co2_presence_probability"]
        g=predictions["gas_probability_conditional_on_co2"]
        raw=np.where(p>=.5,np.where(g>=.6,2,1),0).astype(np.uint8)
        spacing=(abs(float(self.context.y_m[1]-self.context.y_m[0])),
                 abs(float(self.context.x_m[1]-self.context.x_m[0])))
        prepared=regional.prepare(current,self.baseline,p,g,self.context.domain,spacing,self.settings)
        labels,report=regional.predict(prepared,self.model,raw,self.context.domain)
        co2=np.where(labels>0,np.maximum(p,.5),np.minimum(p,.49)).astype(np.float32)
        gas=np.where(labels==1,np.minimum(g,.4),g).astype(np.float32)
        # Returned values are classification scores, not concentration or
        # calibrated posterior probabilities. The exact masks own the display.
        result=dict(co2_presence_probability=co2,gas_probability_conditional_on_co2=gas,
            untouched_water_probability=1-co2,aqueous_co2_probability=co2*(1-gas),
            free_co2_gas_probability=co2*gas)
        self.last_diagnostics={"optical_region_added":(labels==1)&(raw==0),
                               "optical_region_removed":(labels==0)&(raw==1)}
        return result,dict(report,format=VERSION,elapsed_h=elapsed_h,
            reference_basis="registered pre-injection RGB",context_completion_used=False,
            mass_constraints_applied=False,classification_scores_not_concentration=True)
