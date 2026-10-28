"""Bounded read-only attachment of original N1/native-phase saved outputs.

An explicit product index and machine/collection bindings are the only entry
point. Opening admits small metadata; selecting a frame invokes the original
protocol's saved reader. Nothing discovers jobs, collects files, exports, or
evaluates models. Native records never become the Studio delivery schema.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import re

from . import storage_n1_saved_products as n1
from . import storage_saved_products as saved
from .pipeline import products_publication as publication
from .storage_indexed_products import indexed_asset_resolver


def completion_pipeline(completion):
    """Keep the original mass-only absence distinct from native-phase's literal."""
    if "pipeline" not in completion:
        return "n1"
    if completion["pipeline"] == "native-phase-n1":
        return "native-phase-n1"
    raise ValueError("Unsupported original native saved completion protocol")


def _sha256(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(label + " requires its explicit lowercase SHA-256")
    return value


def _identity(bound):
    saved._descriptor(bound)
    return {key: bound[key] for key in ("path", "sha256", "size_bytes")}


class NativeSavedReview:
    """One explicitly selected run and calibration from an immutable index.

    Resumed aggregates are deliberately unsupported: their frames may belong
    to different original manifests, and this adapter does not reproduce the
    job layer's producer-lineage validator. No arrays are retained internally.
    """

    def __init__(self, index, *, current_machine_id, expected_run,
                 expected_calibration_sha256, collections=(), source_paths=None,
                 geometry_producer_sha256=None, expected_pipeline=None, photographs=None):
        if not isinstance(expected_run, str) or not expected_run:
            raise ValueError("An explicit expected native run is required")
        _sha256(expected_calibration_sha256, "Expected calibration")
        if geometry_producer_sha256 is not None:
            _sha256(geometry_producer_sha256, "Geometry producer")
        if expected_pipeline is not None and expected_pipeline not in {"n1", "native-phase-n1"}:
            raise ValueError("Unsupported expected native saved protocol")
        if photographs is not None and not isinstance(photographs, Mapping):
            raise ValueError("Original photographs require an explicit case-to-receipt mapping")
        self._index = deepcopy(index)
        self._current_machine_id = deepcopy(current_machine_id)
        self._expected_run = expected_run
        self._expected_calibration_sha256 = expected_calibration_sha256
        self._collections = deepcopy(tuple(collections))
        self._source_paths = deepcopy(source_paths)
        self._geometry_producer_sha256 = geometry_producer_sha256
        self._photographs = deepcopy(dict(photographs or {}))
        self._resolve = indexed_asset_resolver(
            self._index, current_machine_id=self._current_machine_id,
            collections=self._collections, source_paths=self._source_paths)
        _root, index_document, _products = publication._read_index(self._index)
        completion = saved._json(publication.read_published_metadata(self._index))
        self._pipeline = completion_pipeline(completion)
        if expected_pipeline is not None and self._pipeline != expected_pipeline:
            raise ValueError("Native completion protocol differs from the authenticated job/task")
        if index_document["run"] != expected_run or completion.get("run") != expected_run:
            raise ValueError("Native saved result differs from the selected run")
        if (completion.get("calibration", {}).get("sha256") != expected_calibration_sha256
                or ("calibration_sha256" in completion
                    and completion["calibration_sha256"] != expected_calibration_sha256)):
            raise ValueError("Native saved result differs from the selected calibration")
        if (completion.get("resume_checkpoint") is not None
                or completion.get("resumed_frame_count", 0) != 0):
            raise ValueError("Resumed native aggregates are not supported by this review attachment")
        if self._pipeline == "native-phase-n1" and (
                completion.get("reused_case_ids") != []
                or type(completion.get("newly_computed_frames")) is not int
                or completion["newly_computed_frames"] != completion.get("frames")):
            raise ValueError("Resumed native-phase aggregates are not supported by this review attachment")

        # The original stream reader validates run/manifest metadata now. Its
        # frame generator is never iterated while constructing the selector.
        read_run = n1.read_n1_stored_run if self._pipeline == "n1" else saved.read_stored_run
        admitted = read_run(completion, resolve_asset=self._resolve, load_before=False, stream=True)
        admitted["frames"].close()
        self._completion = admitted["original_completion"]
        self._manifest_asset = deepcopy(self._completion["manifest"])
        self._manifest = saved._json(saved._read_bytes(
            self._manifest_asset, self._resolve, []))
        task_identity = self._manifest.get("task_identity") or {}
        if (task_identity.get("run") != expected_run
                or task_identity.get("calibration_sha256") != expected_calibration_sha256
                or task_identity.get("pipeline", "n1") != self._pipeline):
            raise ValueError("Original native manifest differs from the selected run/calibration")
        if self._pipeline == "n1" and self._manifest.get("resume_checkpoint") is not None:
            raise ValueError("Resumed native aggregates are not supported by this review attachment")
        if self._pipeline == "native-phase-n1" and self._manifest.get("resume_checkpoint") is not None:
            # The native-phase runner ALWAYS writes this descriptor, including
            # a fresh attempt with an empty prefix. Its original publication
            # does not index the checkpoint. Retain it as passive provenance,
            # just as read_stored_run does; do not probe the producer path or
            # claim to qualify checkpoint contents. The original Jobs terminal
            # remains authoritative. Empty reused_case_ids, all-new count and
            # the exact current manifest for EVERY producer/frame below admit
            # only frames from this attempt, without reproducing lineage proof.
            saved._descriptor(self._manifest["resume_checkpoint"])
        self._case_ids = tuple(admitted["ordered_case_ids"])
        if (not self._photographs.keys() <= set(self._case_ids)
                or self._pipeline == "n1" and self._photographs):
            raise ValueError("Explicit photographs must name only this native-phase run's exact cases")
        for bound in self._photographs.values():
            saved._descriptor(bound)
        self._records = tuple(deepcopy(self._completion["records"]))
        producers = self._completion.get("frame_producers")
        if (not isinstance(producers, list)
                or [row.get("case_id") for row in producers] != list(self._case_ids)):
            raise ValueError("Native saved result lacks its exact original frame producers")
        manifest_identity = _identity(self._manifest_asset)
        for record, producer in zip(self._records, producers):
            if (_identity(producer["manifest"]) != manifest_identity
                    or record.get("manifest_sha256") != manifest_identity["sha256"]):
                raise ValueError("Resumed or mixed-manifest native aggregates are not supported")

    @property
    def index(self):
        return deepcopy(self._index)

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def photographs(self):
        return deepcopy(self._photographs)

    @property
    def current_machine_id(self):
        return self._current_machine_id

    @property
    def expected_run(self):
        return self._expected_run

    @property
    def expected_calibration_sha256(self):
        return self._expected_calibration_sha256

    @property
    def collections(self):
        return deepcopy(self._collections)

    @property
    def source_paths(self):
        return deepcopy(self._source_paths)

    @property
    def geometry_producer_sha256(self):
        return self._geometry_producer_sha256

    @property
    def case_ids(self):
        return self._case_ids

    def _position(self, index):
        if type(index) is not int or not 0 <= index < len(self._case_ids):
            raise IndexError("Choose an existing native saved-frame index")
        return index

    def frame_metadata(self, index):
        """Copy the original manifest spec, including its literal elapsed time."""
        return deepcopy(self._manifest["frames"][self._case_ids[self._position(index)]])

    def read_frame(self, index):
        """Read only the selected frame through its unchanged original reader."""
        record = self._records[self._position(index)]
        if self._pipeline == "n1":
            stored = n1.read_n1_stored_frame(record, manifest=self._manifest,
                manifest_asset=self._manifest_asset, resolve_asset=self._resolve,
                load_before=False)
        else:
            # These are the exact expected identities supplied by the original
            # read_stored_run generator, without visiting any preceding frame.
            stored = saved.read_stored_frame(record, resolve_asset=self._resolve,
                load_before=False, expected=self._native_expected(record["frame_id"]))
        if stored["storage_document"]["calibration_sha256"] != self._expected_calibration_sha256:
            raise ValueError("Native frame differs from the selected calibration")
        return stored

    def _native_expected(self, case):
        return dict(job_id=self._manifest["job_id"], run=self._expected_run, case_id=case,
            precision=self._completion["precision"], code_commit=self._manifest["code"]["commit"],
            configuration=deepcopy(self._completion["configuration"]),
            calibration_sha256=self._expected_calibration_sha256,
            source=deepcopy(self._manifest["frames"][case]["source"]))

    def read_photo(self, stored):
        """Read the bound original photograph, never a source or corrected fallback."""
        case = stored.get("case_id")
        if (stored.get("run") != self._expected_run or case not in self._case_ids
                or stored.get("storage_document", {}).get("calibration_sha256")
                != self._expected_calibration_sha256):
            raise ValueError("Photograph frame differs from this native review attachment")
        record = self._records[self._case_ids.index(case)]
        from .calibration_review_native_bindings import photo_geometry
        if self._pipeline == "n1":
            if stored.get("storage_receipt") != saved._record_asset(record, "storage_completed.json"):
                raise ValueError("Photograph frame differs from the original indexed storage receipt")
            geometry = photo_geometry(stored, expected=self._geometry_producer_sha256)
            return n1.read_n1_original_photo(stored, resolve_asset=self._resolve,
                geometry_producer_sha256=geometry)
        # Native-phase photographs are a separate explicit original export
        # binding; neither its completion nor its storage document declares one.
        if case not in self._photographs:
            raise FileNotFoundError("Native-phase original photograph has no explicit indexed receipt binding")
        result = saved._json(saved._read_bytes(saved._record_asset(record, "result.json"), self._resolve, []))
        entry = saved.output.validate_publication_binding(result["prediction"]["pipeline"])
        receipt = entry["receipt"]
        document = saved._json(saved._read_bytes(receipt, self._resolve, []))
        if (stored.get("storage_receipt") != receipt or stored.get("storage_document") != document
                or any(document.get(key) != value for key, value in self._native_expected(case).items())):
            raise ValueError("Photograph frame differs from the original native-phase storage receipt")
        geometry = photo_geometry(stored, expected=self._geometry_producer_sha256, pipeline=self._pipeline)
        from .storage_saved_export import _photograph
        return _photograph(self._photographs[case], stored, resolve_asset=self._resolve,
                           geometry_producer_sha256=geometry)
