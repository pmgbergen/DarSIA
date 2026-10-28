"""Collect genuine Build-bound RAW phase inputs without fitting optics or mass.

Input format ``darsia-build-raw-phase-inputs-v1`` contains run, frames, native,
phase, history_policy, selection_index_field, reference_spec,
initial_previous_spec and temporal_pad_px; data_aliases and provenance are
optional. Phase initially contains only its explicit configuration descriptor.
Every specification supplies case_id, frame_index, elapsed_h, source, geometry,
phase_roles and raw_acquisition. Native tasks and phase products are produced
here. No fitted mass calibration or expected current labels are inputs.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from ..mass.build_native_bridge import BuildNativeBridge, _fresh
from ..mass.native_optical_source import _bound, _same
from ..mass.n1_pipeline import DataReader, asset, _write_json
from ..native import source as native_source
from ..phase.infer import infer_labels
from .job_queue import name
from .raw_acquisition import FORMAT as ACQUISITION_FORMAT, RawAcquisitionInputs
from .runner import _phase_outputs
from .series import SeriesInputs, _no_executable_configuration


FORMAT = "darsia-build-raw-phase-inputs-v1"
_CONFIG = {"format", "run", "frames", "native", "phase", "history_policy",
           "selection_index_field", "reference_spec", "initial_previous_spec", "temporal_pad_px"}
_SPEC = {"case_id", "frame_index", "elapsed_h", "source", "geometry", "phase_roles", "raw_acquisition"}
_ROLES = {"area", "current_h", "reference_to_current_h", "static", "retained_source_cache",
          "encoding_contract", "light_epoch_id", "inherited_spacing_m", "probability_spacing_m",
          "placement_metadata", "photo_sand_reference", "photo_proof"}


def _keys(value, required, optional=()):
    if not isinstance(value, dict) or not required <= set(value) or set(value) - required - set(optional):
        raise ValueError("Complete explicit Build RAW phase DATA is required")


def _specification(spec, run, index_field, *, selected):
    """Retain the original RAW source, area and acquisition-identity guards."""
    _keys(spec, _SPEC, {"provenance", "context_acceptance"})
    name(spec["case_id"])
    acquisition = _bound(spec["raw_acquisition"])
    source = spec["source"]
    if (type(spec["frame_index"]) is not int or spec["frame_index"] < 0
            or type(spec["elapsed_h"]) not in {int, float} or not math.isfinite(spec["elapsed_h"])
            or not isinstance(source, dict) or source.get("run") != run
            or source.get("source_sha256") != acquisition["sha256"]
            or not isinstance(source.get("source_binding"), dict)
            or source["source_binding"].get("source_sha256") != acquisition["sha256"]
            or source.get(index_field) != spec["frame_index"]
            or type(source.get(index_field)) is not int
            or type(source.get("elapsed_h")) is not type(spec["elapsed_h"])
            or source["elapsed_h"] != spec["elapsed_h"]
            or selected and source.get("selected") is not True):
        raise ValueError("RAW phase source/run/index/time binding differs")
    _bound(spec["geometry"])
    if "context_acceptance" in spec:
        _bound(spec["context_acceptance"])
    roles = spec["phase_roles"]
    _keys(roles, _ROLES, {"native_row", "probability_receipt", "retained_frame_index"})
    if any(roles.get(key) is not None for key in (
            "retained_source_cache", "native_row", "probability_receipt", "retained_frame_index", "encoding_contract")):
        raise ValueError("Fresh RAW phase inputs cannot borrow JPEG-positioned phase products")
    photo = _bound(roles["photo_sand_reference"])
    proof = roles["photo_proof"]
    if (not isinstance(proof, dict) or proof.get("sha256") != photo["sha256"]
            or proof.get("phase_labels_used") is not False):
        raise ValueError("RAW phase photo support needs its exact independent mask proof")
    if source.get("studio_anchor") and type((source.get("parent_source") or {}).get("source_changed")) is not bool:
        raise ValueError("RAW Studio anchor needs its explicit source-geometry flag")
    area = roles["area"]
    if (not isinstance(area, dict)
            or area.get("role") != ("measured_parent" if source.get("studio_anchor") else "selected")
            or type(area.get("factor")) not in {int, float}
            or not math.isfinite(area["factor"]) or area["factor"] <= 0):
        raise ValueError("Original RAW phase area route differs")
    if area["role"] == "measured_parent":
        if not {"historical", "measured"} <= set(area):
            raise ValueError("A parent needs its separate original and measured areas")
    elif ({"historical", "measured"} & set(area)
          or not isinstance(area.get("revision"), dict) or not area["revision"]):
        raise ValueError("Selected area needs its explicit revision, without borrowed parent areas")


class BuildRawPhaseCollector:
    """Original RAW/phase producers with one explicit collection acknowledgement.

    Bridge preparation measures every selected acquisition before SeriesInputs
    is constructed, because its original RAW-mode admission needs actual tasks.
    The original native producer then corrects those acquisitions again when
    iterated. Both sets of source records are retained; no equivalence is
    invented by substituting buffers into the original producer.
    """

    def __init__(self, bridge, configuration, output, *, prepared_frames=None):
        if not isinstance(bridge, BuildNativeBridge):
            raise TypeError("The original completed-Build native bridge is required")
        data = deepcopy(configuration)
        reuse = bridge.context.provenance.get("local_build_reuse")
        if reuse is not None:
            current = bridge.reader.json(_bound(reuse["frame_configuration"]))
            if not _same(data, current):
                raise ValueError("P1 DATA differs from the explicit current frame configuration for local Build reuse")
        _no_executable_configuration(data)
        _keys(data, _CONFIG, {"data_aliases", "provenance"})
        if data["format"] != FORMAT or data["run"] != bridge.context.run:
            raise ValueError("Explicit phase-only DATA must name the completed Build run")
        _keys(data["phase"], {"configuration"})
        _bound(data["phase"]["configuration"])
        if (not isinstance(data["native"], dict) or "configuration" not in data["native"]
                or not _same(data["native"]["configuration"], bridge.context.provenance["native_configuration"])):
            raise ValueError("RAW phase native configuration differs from the completed Build")
        if (type(data["temporal_pad_px"]) is not int or data["temporal_pad_px"] < 0
                or not isinstance(data["history_policy"], str) or not data["history_policy"]
                or not isinstance(data["selection_index_field"], str) or not data["selection_index_field"]
                or not isinstance(data["frames"], list) or not data["frames"]):
            raise ValueError("Explicit ordered RAW phase/history inputs are required")
        run, index_field = data["run"], data["selection_index_field"]
        for spec in (data["reference_spec"], data["initial_previous_spec"]):
            _specification(spec, run, index_field, selected=False)
        for spec in data["frames"]:
            _specification(spec, run, index_field, selected=True)
        if not _same(data["reference_spec"]["raw_acquisition"], bridge.context.provenance["raw_acquisition"]):
            raise ValueError("The phase reference must be the actual Build baseline RAW acquisition")
        previous = data["initial_previous_spec"]["source"]["source_sha256"]
        last = (data["initial_previous_spec"]["elapsed_h"], data["initial_previous_spec"]["frame_index"])
        ids, digests = set(), {previous}
        for spec in data["frames"]:
            source, position = spec["source"], (spec["elapsed_h"], spec["frame_index"])
            if (spec["case_id"] in ids or source["source_sha256"] in digests
                    or position[0] <= last[0] or position[1] <= last[1]
                    or source.get("previous_source_sha256") != previous):
                raise ValueError("RAW phase selection changed its unique chronological predecessor chain")
            ids.add(spec["case_id"])
            digests.add(source["source_sha256"])
            previous, last = source["source_sha256"], position
        same_reference = _same(data["reference_spec"], data["initial_previous_spec"])
        if not same_reference and data["reference_spec"]["case_id"] == data["initial_previous_spec"]["case_id"]:
            raise ValueError("Distinct reference/history specifications need distinct case identifiers")
        self.bridge, self.owner = bridge, None
        self._prepared_frames = prepared_frames
        if prepared_frames is not None:
            from ..mass.build_native_bridge import BuildNativeFrame
            required = {spec["case_id"] for spec in (
                data["reference_spec"], data["initial_previous_spec"], *data["frames"])}
            if (not isinstance(prepared_frames, dict) or set(prepared_frames) != required
                    or any(not isinstance(value, BuildNativeFrame) for value in prepared_frames.values())):
                raise ValueError("Prepared RAW frames must cover this complete exact selection")
        self.reader = DataReader(data.get("data_aliases", ()), unc_only=False)
        self._retained_pins = {}
        self._pin_tree(data["phase"]["configuration"])
        self._pin_tree(data["native"]["configuration"])
        phase_document = self.reader.json(data["phase"]["configuration"])
        self._geometry_fields = deepcopy(phase_document["geometry_fields"])
        self._geometry_parameters = deepcopy(phase_document["parameters"]["geometry"])
        self._pin_phase_sources(phase_document)
        self._input_fields = deepcopy(phase_document["parameters"]["inputs"]["fields"])
        for spec in (data["reference_spec"], data["initial_previous_spec"], *data["frames"]):
            self._pin_tree(spec["geometry"])
            geometry = self.reader.json(spec["geometry"])
            if (not isinstance(geometry, dict) or not isinstance(geometry.get("raw_source"), dict)
                    or geometry["raw_source"].get("sha256") != spec["raw_acquisition"]["sha256"]
                    or "run" in geometry and geometry["run"] != run):
                raise ValueError("RAW phase geometry differs from its exact RAW acquisition or run")
            self._pin_tree(geometry["frame_photo_top_keep"])
            roles = spec["phase_roles"]
            self._pin_tree(roles["photo_sand_reference"])
            self._pin_tree(roles["static"]["components"])
            if roles["area"]["role"] == "measured_parent":
                self._pin_tree(roles["area"]["historical"])
                self._pin_tree(roles["area"]["measured"])
            if "context_acceptance" in spec:
                self._pin_tree(spec["context_acceptance"])
        self.revalidate()
        self.output = _fresh(output, [*bridge.forbidden, str(bridge.output)])
        for folder in ("bridge", "acquisitions", "phase"):
            (self.output / folder).mkdir()
        (self.output / "bridge" / "selected").mkdir()
        for spec in (data["reference_spec"], data["initial_previous_spec"], *data["frames"]):
            self._same_geometry(spec, self.reader.json(spec["geometry"]))
        self.pending, self._index, self._closed, self._failed = None, 0, False, False
        reference, self.reference_bridge_frame = self._bridge_spec(
            data["reference_spec"], self.output / "bridge" / "reference")
        if same_reference:
            initial, self.initial_previous_bridge_frame = deepcopy(reference), self.reference_bridge_frame
        else:
            initial, self.initial_previous_bridge_frame = self._bridge_spec(
                data["initial_previous_spec"], self.output / "bridge" / "initial_previous")
        selected = [self._bridge_spec(spec, self.output / "bridge" / "selected" / spec["case_id"])
                    for spec in data["frames"]]
        preparation = dict(format=ACQUISITION_FORMAT, run=run, native=deepcopy(data["native"]),
            phase=deepcopy(data["phase"]), temporal_pad_px=data["temporal_pad_px"],
            acquisitions=[reference] if same_reference else [reference, initial],
            data_aliases=deepcopy(data.get("data_aliases", [])))
        acquisitions = RawAcquisitionInputs(preparation, unc_only=False)
        try:
            self.reference_record = acquisitions.prepare(reference, self.output / "acquisitions" / "reference")
            self._same_render(self.reference_record, self.reference_bridge_frame)
            initial_record = (deepcopy(self.reference_record) if same_reference else
                acquisitions.prepare(initial, self.output / "acquisitions" / "initial_previous"))
            self._same_render(initial_record, self.initial_previous_bridge_frame)
            self._pin_acquisition(self.reference_record)
            self._pin_acquisition(initial_record)
        finally:
            acquisitions.processors.clear()
        self.configuration = {key: deepcopy(value) for key, value in data.items()
                              if key not in {"reference_spec", "initial_previous_spec", "temporal_pad_px"}}
        self.configuration["frames"] = [spec for spec, _frame in selected]
        self.configuration["phase"].update(
            reference=dict(full=deepcopy(self.reference_record["full"]),
                native_row=deepcopy(self.reference_record["native_row"]),
                source_sha256=reference["source"]["source_sha256"]),
            initial_previous={key: deepcopy(initial_record[key]) for key in ("full", "native_row", "source", "current_h")})
        self.configuration_binding = _write_json(self.output / "configuration.json", self.configuration)
        self.owner = SeriesInputs(deepcopy(self.configuration), self.output / "phase", unc_only=False)
        self._selected = selected
        self.revalidate()

    def _pin_tree(self, value):
        """Retain explicit descriptors, without discovering files or reading arrays."""
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                self.reader.path(value)
                self._retained_pins[(value["path"], value["sha256"])] = deepcopy(value)
            for child in value.values():
                self._pin_tree(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                self._pin_tree(child)

    def _pin_phase_sources(self, document):
        """Pin the roles the original model loader consumes, without ancestry walks."""
        models = document["model_bundles"]
        for role in ("native", "edge"):
            item = models[role]
            for key in item["requirements"]["expected_sha256"]:
                self._pin_tree(item["documents"][key])
        placement = models["placement"]
        self._pin_tree(placement["pixel_model"])
        self._pin_tree(placement["regional_model"])
        if placement.get("projected_models") is not None:
            for key in ("pixel", "regional"):
                self._pin_tree(placement["projected_models"][key])
        chain = document["chain_assets"]
        self._pin_tree(chain["envelopes"])
        envelope = self.reader.json(chain["envelopes"])
        for key in ("envelope1", "envelope2"):
            self._pin_tree(envelope[key])
        self._pin_tree(chain["floor"])
        self._pin_tree(chain["zones"])

    def _pin_scientific(self, record):
        """Retain the actual scientific centre, support, ring and audit descriptors."""
        full = record["full"]
        self._pin_tree(full["native_centre"])
        self._pin_tree(full["ring_windows"])
        self._pin_tree(full["binding"]["producer_audit"])
        if "audit" in record:
            self._pin_tree(record["audit"])

    def _pin_acquisition(self, record):
        """Retain the flat path/SHA fields used by original masked history readers."""
        self._pin_scientific(record)
        self._pin_tree(record["native_row"])
        row = self.reader.json(record["native_row"])
        native, temporal = self._input_fields["native"], self._input_fields["temporal"]
        if len(native) != 4 or len(temporal) != 4:
            raise ValueError("Unsupported explicit native/temporal history field schema")
        for path_index, sha_index in ((0, 1), (2, 3)):
            self._pin_tree(dict(path=row[native[path_index]], sha256=row[native[sha_index]]))
        if row.get(temporal[0]) is not None or row.get(temporal[1]) is not None:
            self._pin_tree(dict(path=row[temporal[0]], sha256=row[temporal[1]]))
        self._pin_tree(row["raw_acquisition"])
        self._pin_tree(row["raw_producer_audit"])

    def revalidate(self):
        """Check consumed source stamps before publication or history advancement."""
        self.bridge._revalidate()
        for bound in tuple(self._retained_pins.values()):
            self.reader.path(bound)
        if hasattr(self, "configuration_binding"):
            if not _same(self.reader.json(self.configuration_binding), self.configuration):
                raise ValueError("The prepared RAW phase configuration changed in memory")

    def _bridge_spec(self, supplied, output):
        """Use only an actual bridge task and its measured phase-role transforms."""
        spec = deepcopy(supplied)
        acquisition = spec["raw_acquisition"]
        produced = (self.bridge.frame(acquisition, output) if self._prepared_frames is None
                    else self._prepared_frames[spec["case_id"]])
        receipt = self.bridge.reader.json(produced.receipt)
        if (not _same(receipt["bridge"], self.bridge.proof)
                or not _same(receipt["native_frame"], produced.native_task["geometry"])):
            raise ValueError("Prepared RAW frame belongs to another measured Build bridge")
        self._pin_tree(produced.receipt)
        self._pin_tree(produced.products)
        spec["native_task"] = deepcopy(produced.native_task)
        geometry = self.bridge.reader.json(produced.native_task["geometry"])
        if (produced.native_task.get("kind") != "raw_rig"
                or not _same(geometry["acquisition"], acquisition)
                or geometry["run"] != self.bridge.context.run):
            raise ValueError("The produced native task differs from the selected RAW acquisition")
        for field, actual in (("current_h", "baseline_to_native_h"),
                              ("reference_to_current_h", "reference_to_native_h")):
            if not native_source._same_h(self.bridge._matrix(spec["phase_roles"][field]),
                    self.bridge._matrix(geometry[actual]), parameters=self.bridge.parameters):
                raise ValueError("Supplied phase transform differs from measured RAW bridge: " + field)
        return spec, produced

    def _same_geometry(self, spec, row):
        """Join the consumed rig/area grid to the same frozen-reference transform."""
        field = self._geometry_fields["reference_to_frame"]
        expected = self.bridge._matrix(spec["phase_roles"]["reference_to_current_h"])
        for actual in (row[field], row["area_parameters"][field]):
            if not native_source._same_h(self.bridge._matrix(actual), expected,
                                         parameters=self.bridge.parameters):
                raise ValueError("Supplied rig/area transform conflicts with the phase grid required by the measured RAW bridge")
        gp, extension = self._geometry_parameters, self.bridge.parameters["extension"]
        if (not _same(row["native_origin_xy"], gp["rig_origin_xy"])
                or not _same(row["native_origin_xy"], extension["native_origin_xy"])
                or not _same(row["source_native_shape_hw"], gp["native_shape_hw"])
                or not _same(row["source_native_shape_hw"], extension["native_shape_hw"])
                or not _same(row["source_native_shape_hw"], self.bridge.context.grid["num_voxels"])
                or not _same(row["source_content_box_xywh"], gp["source_box_xywh"])):
            raise ValueError("Supplied rig origin or native shape differs from the measured RAW phase grid")

    def _same_render(self, record, bridge_frame):
        """Reject a different second registration before phase or history use."""
        full = record["full"]
        actual = self.reader.json(full["binding"]["producer_audit"])
        expected = self.bridge.reader.json(bridge_frame.receipt)["correction"]
        current, measured = actual["renderer"], expected["renderer"]
        if (list(full["native_centre"]["native_shape_hw"]) != list(self.bridge.context.grid["num_voxels"])
                or current["camera_u_wh"] != measured["camera_u_wh"]
                or not native_source._same_h(self.bridge._matrix(current["native_to_camera_u_h"]),
                    self.bridge._matrix(measured["native_to_camera_u_h"]), parameters=self.bridge.parameters)
                or actual["sources"]["camera_arw"]["sha256"] != expected["sources"]["camera_arw"]["sha256"]):
            raise ValueError("The original RAW producer changed the bridge registration or native grid")

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed or self._failed:
            raise RuntimeError("The RAW phase collector is closed or failed; original outputs remain retained")
        if self.pending is not None:
            raise RuntimeError("Acknowledge the collected frame before requesting its chronological successor")
        if self._index == len(self._selected):
            raise StopIteration
        spec, bridge_frame = self._selected[self._index]
        try:
            self.revalidate()
            stage = self.output / "phase" / spec["case_id"]
            native = self.owner.native(spec, stage)
            self._same_render(native["record"], bridge_frame)
            self._pin_scientific(native["record"])
            self._pin_tree(native["record_asset"])
            self.revalidate()
            frame = self.owner.frame(spec, native)
            gap, gap_provenance = None, None
            if "approved_analysis_gap" in self.bridge.context.provenance:
                from ..painted_analysis_exclusions import transport_build_exclusion, restrict_phase_domain
                bridge_receipt = self.bridge.reader.json(bridge_frame.receipt)
                gap, gap_provenance = transport_build_exclusion(self.bridge.context, self.reader,
                    bridge_receipt["build_to_frame_native_h"], frame["context"].origin_xy,
                    frame["context"].rig_domain.shape, self._geometry_parameters)
                frame = restrict_phase_domain(frame, gap, gap_provenance)
            inferred = infer_labels(frame, self.owner.previous, self.owner.model_bundle, self.owner.parameters)
            if gap is not None:
                if np.any(inferred["labels"][gap] != 255):
                    raise ValueError("Phase inference filled approved unquantified analysis gaps")
                inferred["classes"]["approved_analysis_gap"] = gap
                inferred["classes"]["physical_rig_domain"] = frame["physical_rig_domain"]
                inferred["qa"]["approved_analysis_gap"] = gap_provenance
            products = _phase_outputs(stage, spec, frame, inferred)
            self._pin_tree(products)
            self._pin_tree(native["record_asset"])
            self.revalidate()
            receipt = _write_json(stage / "build_raw_phase.json", dict(
                format="darsia-build-raw-phase-frame-v1", complete=True, source=deepcopy(spec["source"]),
                bridge=self.bridge.proof, bridge_frame=bridge_frame.receipt,
                configuration=self.configuration_binding,
                phase_configuration=self.configuration["phase"]["configuration"],
                native_configuration=self.configuration["native"]["configuration"],
                native=native["record_asset"], products=products, producer=asset(Path(__file__))))
            self._pin_tree(receipt)
            self.pending = dict(spec=deepcopy(spec), native=native, frame=frame, inferred=inferred,
                                phase_products=products, bridge_frame=bridge_frame, phase_receipt=receipt)
            return self.pending
        except BaseException:
            self._failed = True
            raise

    def commit(self, row):
        """Acknowledge successful caller collection, then advance original history."""
        if self._closed or self._failed or row is not self.pending or self.pending is None:
            raise ValueError("Only this collector's pending frame may advance its history")
        spec, _bridge_frame = self._selected[self._index]
        if not _same(row["spec"], spec):
            raise ValueError("Pending source metadata changed before collection acknowledgement")
        self.revalidate()
        self.owner.advance(row["frame"])
        self.pending = None
        self._index += 1

    def close(self):
        """Release original array owners without deleting or advancing any output."""
        if not self._closed:
            self._closed = True
            if self.owner is not None:
                self.owner.close()
            self.pending = None
