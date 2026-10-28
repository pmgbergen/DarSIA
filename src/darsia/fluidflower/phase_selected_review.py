"""Read-only Studio adapter for the completed selected-camera FF_AC timeline.

The timeline index is a presentation index, never a historical label index.
Every numerical and photographic asset remains bound to its camera-source SHA.
No old rig-camera scientific contract is imitated by this review adapter.
"""
from __future__ import annotations

from copy import deepcopy
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from contextvars import ContextVar

from . import phase_saved_assets as assets
from .phase_saved_review_metadata import configured_parameters

_PARAMETER_SECTION = ContextVar("selected_review_parameter_section", default="selected_review")

def _parameters():
    return configured_parameters(_PARAMETER_SECTION.get())

def _export(name):
    item=_parameters()["exports"][name]
    return tuple(item["value"]) if item["python_type"]=="tuple" else item["value"]

_METADATA_EXPORT_NAMES = ('PROFILE_FILE', 'COMPLETION_CONTRACT', 'PROFILE_FORMAT', 'SOURCE_KEY', 'SOURCE_CONTRACT', 'INPUT_CONTRACT', 'RUNTIME_CONTRACT', 'DISPLAY_CONTRACT', 'MODES')

def __getattr__(name):
    if name not in _METADATA_EXPORT_NAMES: raise AttributeError(name)
    return _export(name)

_READER = assets.VerifiedAssetReader()
_DOCUMENTS = OrderedDict()
_NATIVE_DESCRIPTORS = OrderedDict()


def _document(descriptor,reader):
    path = reader.path(descriptor)
    stamp = path.stat()
    key = (id(reader),str(path.resolve()),descriptor["sha256"],stamp.st_size,stamp.st_mtime_ns)
    if key not in _DOCUMENTS:
        _DOCUMENTS[key] = reader.json(descriptor)
        while len(_DOCUMENTS)>160:
            _DOCUMENTS.popitem(last=False)
    _DOCUMENTS.move_to_end(key)
    return _DOCUMENTS[key]


def _native_document(path,reader):
    path = Path(path)
    stamp = path.stat()
    key = (str(path.resolve()),stamp.st_size,stamp.st_mtime_ns)
    if key not in _NATIVE_DESCRIPTORS:
        _NATIVE_DESCRIPTORS[key] = assets.asset(path)
        while len(_NATIVE_DESCRIPTORS)>64:
            _NATIVE_DESCRIPTORS.popitem(last=False)
    return _document(_NATIVE_DESCRIPTORS[key],reader)


def public_profile(profile):
    return {key:deepcopy(value) for key,value in profile.items() if not key.startswith("_")}


def runtime_identity(profile):
    p=_parameters()
    current=assets.asset(__file__)
    expected=p["current_code_source"]
    if current["sha256"]!=expected["sha256"] or current["size_bytes"]!=expected["size_bytes"]:
        raise ValueError("Current saved review reader differs from its declared source")
    producer=p["original_reader_source"]
    source=_READER.path(producer)
    if source.stat().st_size!=producer["size_bytes"]:
        raise ValueError("Original saved review producer size changed")
    payload = dict(profile=public_profile(profile),reader_sha256=producer["sha256"])
    return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()


def load_profile(root, *, reader=None):
    reader = reader or _READER
    path = Path(root)/_export("PROFILE_FILE")
    profile = _native_document(path,reader)
    if (profile.get("format") != _export("PROFILE_FORMAT")
            or profile.get("publication_scope") != "studio_validation_only"
            or any(profile.get(key) is not False for key in
                ("mass_eligible","scientific_acceptance","deployment_allowed","automatic_activation","training","labels_changed","round3_allowed","mass_admission"))):
        raise ValueError("Selected-camera profile is not a dedicated saved review")
    gate = _document(profile["accepted_phase_gate"],reader)
    runtime = _document(profile["runtime_manifest"],reader)
    if (gate.get("contract") != _export("COMPLETION_CONTRACT") or gate.get("complete") is not True
            or gate.get("all_runs_complete") is not True or gate.get("coarse_area_gate_passed") is not True
            or gate.get("scorer_gate_pending") is not True or gate.get("round3_allowed") is not False
            or gate.get("mass_admission") is not False or gate.get("legacy_strict_gate_claimed") is not False
            or runtime.get("contract") != _export("RUNTIME_CONTRACT") or runtime.get("complete") is not True
            or runtime.get("pipeline_sha256") != profile.get("pipeline_sha256")
            or runtime.get("accepted_phase_gate") != profile["accepted_phase_gate"]
            or runtime.get("selected_frame_count") != gate.get("selected_frames")
            or runtime.get("baseline_frame_count") != gate.get("baseline_frames")
            or runtime.get("run_count") != len(gate.get("runs",()))
            or set(runtime.get("runs",{})) != set(gate["runs"])):
        raise ValueError("Selected-camera profile has no exact completed source inventory")
    return dict(profile,_selected_camera_runtime=dict(root=str(Path(root)),manifest=runtime))


def load_native_source(review_root, run, manifest=None, *, reader=None, verify_geometry=True):
    """Verify small source receipts only; never open scientific RAW sidecars."""
    reader = reader or _READER
    run = str(run).casefold()
    path = Path(review_root)/"native_frames"/run/"manifest.json"
    native = _native_document(path,reader) if manifest is None else manifest
    marker = native.get(_export("SOURCE_KEY")) or {}
    if (marker.get("contract") != _export("SOURCE_CONTRACT") or marker.get("inference_authorized") is not False
            or native.get("run") != run or native.get("kind") != "selected_camera_full_rig_review_frames"
            or any(key in native for key in ("ffac_rig_camera_sources","v5_native_source_alias","v5_frame_geometry"))):
        raise ValueError("Selected-camera sources require their own explicit review contract")
    inputs = _document(marker["input_manifest"],reader)
    gate = _document(marker["accepted_phase_gate"],reader)
    rows = native.get("frame_sources") or []
    if (inputs.get("contract") != _export("INPUT_CONTRACT") or inputs.get("complete") is not True
            or inputs.get("run") != run or inputs.get("frames") != rows
            or inputs.get("reference") != native.get("reference")
            or inputs.get("accepted_phase_gate") != marker["accepted_phase_gate"]
            or gate.get("contract") != _export("COMPLETION_CONTRACT")
            or gate.get("all_runs_complete") is not True or gate.get("coarse_area_gate_passed") is not True
            or gate.get("scorer_gate_pending") is not True or gate.get("round3_allowed") is not False
            or gate.get("mass_admission") is not False or gate.get("legacy_strict_gate_claimed") is not False
            or inputs.get("run_manifest") != gate.get("run_manifests",{}).get(run)
            or len(rows) != native.get("frame_count") or not rows
            or rows[0].get("baseline") is not True
            or any(row.get("frame_index") != index for index,row in enumerate(rows))):
        raise ValueError("Selected Studio timeline differs from its completed source bindings")
    for row in [native["reference"],*rows]:
        if (set(row.get("photographs",{})) != set(_export("MODES"))
                or row.get("phase_shape_hw") != native["phase_shape_hw"]
                or row.get("source_binding",{}).get("source_sha256") != row.get("source_sha256")
                or row.get("source_binding",{}).get("elapsed_h") != row.get("elapsed_h")
                or row.get("source_work_index") != row["source_binding"].get("work_index")):
            raise ValueError("Selected timeline index changed an acquisition, time or raster")
        if verify_geometry:
            display = _document(row["display_receipt"],reader)
            if (display.get("contract") != _export("DISPLAY_CONTRACT") or display.get("complete") is not True
                    or display.get("binding",{}).get("phase_receipt") != row["phase_receipt"]
                    or display["binding"].get("scientific_receipt") != row["scientific_receipt"]
                    or display.get("source_sha256") != row["source_sha256"]
                    or display.get("frame_index") != row["source_work_index"]
                    or display.get("photographs") != row["photographs"]
                    or display.get("phase_shape_hw") != native["phase_shape_hw"]
                    or display.get("native_origin_xy") != native["native_origin_xy"]
                    or display["binding"].get("baseline_reference_scientific") != row["baseline_extended_scientific"]
                    or display["binding"].get("baseline_reference_role") != "separately prepared historical pre-injection reference"
                    or display["binding"].get("current_frame0_is_not_substituted") is not True
                    or not display["binding"].get("baseline_role_proof")
                    or not display["binding"].get("baseline_role_display_code")):
                raise ValueError("Selected photographic assets belong to another source or grid")
    return dict(root=Path(review_root),native_path=path,native=native,inputs=inputs,
        input_path=Path(marker["input_manifest"]["path"]))


def native_frame_assets(review_root,run,frame_index=None,*,reference=False,manifest=None,reader=None):
    state = load_native_source(review_root,run,manifest,reader=reader,verify_geometry=False)
    if reference:
        record = state["native"]["reference"]
    else:
        rows = state["native"]["frame_sources"]
        if type(frame_index) is not int or not 0 <= frame_index < len(rows):
            raise ValueError("Selected Studio frame is outside its camera timeline")
        record = rows[frame_index]
    return state,record,record


def resolve_photograph(review_root,run,frame_index=None,*,mode="natural",reference=False,verify=True,reader=None):
    if mode not in _export("MODES"):
        raise ValueError("Unknown selected photographic mode")
    reader = reader or _READER
    _state,record,_binding = native_frame_assets(review_root,run,frame_index,reference=reference,reader=reader)
    # The session reader verifies each path+size+mtime+SHA once. Nothing large
    # is re-hashed after choosing another mode or returning to the same frame.
    return reader.path(record["photographs"][mode]) if verify else Path(record["photographs"][mode]["path"])


def resolve_phase_map(review_root,run,frame_index,*,phase="all",reader=None):
    reader = reader or _READER
    _state,record,_binding = native_frame_assets(review_root,run,frame_index,reader=reader)
    name = {"all":"combined","aqueous":"aqueous","gas":"gas","mixed":"mixed"}.get(phase,phase)
    if name not in record["maps"]:
        raise ValueError("Unknown selected phase-map view")
    return reader.path(record["maps"][name])


def resolve_photo_sand_domain(review_root,run,frame_index=None,*,reference=False,canvas=True,reader=None):
    reader = reader or _READER
    state,record,_binding = native_frame_assets(review_root,run,frame_index,reference=reference,reader=reader)
    import numpy as np
    with np.load(reader.path(record["rig_domain_revision"]["classes"]),allow_pickle=False) as classes:
        domain = np.asarray(classes["rig_domain"],bool).copy()
    if list(domain.shape) != state["native"]["phase_shape_hw"]:
        raise ValueError("Selected photo-sand support belongs to another raster")
    if not canvas:
        x,y,width,height = state["native"]["content_box_xywh"]
        domain = domain[y:y+height,x:x+width]
    return domain


def require_dedicated_inference():
    raise ValueError("Selected saved Studio maps cannot authorize generic inference or model training")


__all__ = ["load_profile","runtime_identity","public_profile","load_native_source","native_frame_assets",
    "resolve_photograph","resolve_phase_map","resolve_photo_sand_domain","require_dedicated_inference"]


class SelectedReviewBindings:
    """Bind one explicit saved profile role to the shared read-only implementation.

    Context-local role selection keeps simultaneous catalog/photo requests
    independent; all source/model guards use the unchanged function bodies.
    """
    _functions = (
        '_document', '_native_document', 'public_profile', 'runtime_identity',
        'load_profile', 'load_native_source', 'native_frame_assets',
        'resolve_photograph', 'resolve_phase_map', 'resolve_photo_sand_domain',
        'require_dedicated_inference',
    )

    def __init__(self, section):
        self.section = section

    def __getattr__(self, name):
        if name == '_READER':
            return _READER
        if name in _METADATA_EXPORT_NAMES:
            token = _PARAMETER_SECTION.set(self.section)
            try:
                return _export(name)
            finally:
                _PARAMETER_SECTION.reset(token)
        if name not in self._functions:
            raise AttributeError(name)
        function = globals()[name]
        def bound(*args, **kwargs):
            token = _PARAMETER_SECTION.set(self.section)
            try:
                return function(*args, **kwargs)
            finally:
                _PARAMETER_SECTION.reset(token)
        return bound
