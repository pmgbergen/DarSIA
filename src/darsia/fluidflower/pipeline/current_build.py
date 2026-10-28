"""Post-start joins to original owned Build evidence; no eligibility authority.

This opt-in reader neither produces a Build nor decodes a frame a second time.
The original publication/physical/coordinate readers remain authoritative for
their limited scopes. The original N1 physical consumer cannot yet represent
the current Build's full-domain roles. Both physical entry and response
preparation therefore remain closed, including the phase-one statistics route.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

PHYSICAL_INPUTS_UNAVAILABLE = (
    "Current Build physical input adapter is unavailable: saved Build fields, "
    "metric coordinates and full-domain roles have no original typed N1 join; "
    "Freewater facies 0 cannot be filled as missing sand, cropped or relabelled. "
    "Phase-one statistics and mass response preparation remain disabled"
)
_ACQUIRED = object()


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False)


def _identity(value):
    return value["sha256"], value.get("size_bytes", value.get("bytes"))


def _physical_context_equal(recorded, current):
    """Keep original numerical provenance exact when local selection is added later."""
    if _same(recorded, current):
        return True
    if (not isinstance(recorded, dict) or not isinstance(current, dict)
            or "local_build_reuse" in recorded or "local_build_reuse" not in current):
        return False
    return _same(recorded, {key: value for key, value in current.items()
                            if key != "local_build_reuse"})


def _require_supported_physical_revision(adopted):
    """Keep the original configuration mapping; direct leaves use the bound scope."""
    configuration = adopted.get("configuration", {})
    if not isinstance(configuration, dict):
        raise ValueError("Current Build adopted configuration must retain its original mapping")


class _DirectReads:
    """Bind original painted readers to the exact replayed named file closure.

    The existing reader scope supplies only its context lifetime. This adapter
    has no discovery, fallback, directory enumeration or new admission path.
    """

    def __init__(self, reader):
        self.reader = reader

    def descriptor(self, path, expected=None):
        from .run_io import _key
        row = self.reader.binding.locations.get(_key(path))
        if row is None or row.reference not in self.reader.allowed:
            raise ValueError("Current Build direct source lacks its named catalogue consumer: " + str(path))
        bound = dict(path=str(path), sha256=row.reference["sha256"], size_bytes=row.reference["bytes"])
        if expected is not None and any(expected.get(key, bound[key]) != bound[key]
                                        for key in ("sha256", "size_bytes")):
            raise ValueError("Current Build direct source descriptor changed")
        self.reader.path(bound, original=True)
        return bound

    def read(self, path, *, kind="bytes", sheet=None):
        if kind != "bytes" or sheet is not None:
            raise ValueError("Current Build direct source has no table consumer")
        bound = self.descriptor(path)
        with self.reader.path(bound, original=True).open("rb") as stream:
            raw = stream.read(bound["size_bytes"] + 1)
        if len(raw) != bound["size_bytes"] or hashlib.sha256(raw).hexdigest() != bound["sha256"]:
            raise ValueError("Current Build direct source changed at its actual read")
        self.descriptor(path, bound)
        return raw

    def stat(self, path):
        bound = self.descriptor(path)
        result = self.reader.path(bound, original=True).stat()
        self.descriptor(path, bound)
        return result

    def exists(self, path, kind):
        if kind not in {"exists", "is_file"}:
            raise ValueError("Current Build direct reader has no declared directory query")
        self.descriptor(path)
        return True

    def inventory(self, path, pattern):
        raise ValueError("Current Build direct reader has no declared directory inventory")

    @contextmanager
    def enter(self):
        from ..preset_wasserstein_inputs import _Scope
        with _Scope.enter(self):
            yield self


class _Reader:
    """Original decoder restricted to the explicitly replayed Build consumers."""

    def __init__(self, binding, task):
        from ..mass.n1_pipeline import DataReader
        self.binding = binding
        self.original = DataReader()
        self.allowed = [row["reference"] for row in task["input_catalog"]["inputs"]
                        if any(use.startswith("current_build.") for use in row["uses"])]
        self.consumed = {}

    def path(self, bound, *, original=False):
        from .run_io import _key
        row = self.binding.locations.get(_key(bound["path"]))
        if row is None or row.reference not in self.allowed:
            raise ValueError("Current Build source lacks its named catalogue consumer: " + bound["path"])
        physical = self.binding.physical_descriptor(bound)
        if original and Path(physical["path"]).resolve() != Path(bound["path"]).resolve():
            raise ValueError("Original Build ownership/publication paths cannot be relocated by this reader")
        path = self.original.path(bound)
        self.consumed[_key(bound["path"])] = deepcopy(bound)
        return path

    def json(self, bound):
        self.path(bound)
        result = self.original.json(bound)
        self.path(bound)
        return result

    def recheck(self):
        for bound in tuple(self.consumed.values()):
            self.path(bound)

    @contextmanager
    def direct(self):
        self.recheck()
        with _DirectReads(self).enter():
            yield
        self.recheck()


class _Publication:
    """Pre-admit every direct original terminal read, preserving original paths."""

    def __init__(self, publication, reader):
        from .run_io import _key
        self.root = Path(publication["root"])
        self.reader = reader
        self.entries = {_key(row["descriptor"]["path"]): row["descriptor"]
                        for row in publication["readset"]}
        self.used = set()
        for bound in self.entries.values():
            reader.path(bound, original=True)

    def read(self, path, actual=None):
        from .run_io import _key
        key = _key(path)
        if key not in self.entries:
            raise ValueError("Original Build publication read is outside its exact readset: " + str(path))
        bound = self.entries[key]
        if actual is not None and _identity(actual) != _identity(bound):
            raise ValueError("Original Build publication descriptor changed")
        self.used.add(key)
        return self.reader.json(bound)

    def reference(self, root, reference):
        from .products_publication import _relative
        relative = _relative(reference["relative_path"])
        path = Path(root).joinpath(*relative.split("/"))
        return path, self.read(path, reference)

    def verify(self, publication):
        from .series_setup_results import setup_terminal
        job = self.read(publication["job"]["path"])
        task = self.read(publication["task"]["path"])
        done = self.read(publication["done"]["path"])
        # setup_terminal also has direct immutable-index/copy readers. Admit
        # those exact paths before entering it; no document path is rewritten.
        self.reference(self.root, done["completed"])
        index_path, index = self.reference(self.root, done["products_index"])
        products = index["products"]
        if {row["product_id"] for row in products} != {"completed.json", "operation_result.json"}:
            raise ValueError("Current Build requires exactly the original Setup publication products")
        for row in products:
            self.reference(index_path.parent, row["metadata_copy"])
        self.read(index_path.parent.parent / "owner.json")
        completed, _frames, _lineage = setup_terminal(self.root, task, job, done, self.read)
        shared_receipt = self.read(publication["operation_result"]["path"])
        original = completed["files"]["operation_result.json"]
        if (_identity(original) != _identity(publication["operation_result"])
                or any(completed.get(key) != value for key, value in shared_receipt.items())):
            raise ValueError("Current Build shared receipt differs from its original terminal")
        if self.used != set(self.entries):
            raise ValueError("Current Build publication readset contains unconsumed metadata")
        self.reader.recheck()
        return completed, shared_receipt


def _source_equal(actual, expected, reader):
    """Allow only the existing admitted locator for a named original source."""
    from .run_io import _key
    path = reader.path(expected)
    return (actual.get("sha256") == expected["sha256"]
            and _key(actual.get("path", "")) in {_key(expected["path"]), _key(path)}
            and ("size_bytes" not in actual or actual["size_bytes"] == expected.get("size_bytes")))


class CheckedCurrentBuild:
    """Factory-acquired original evidence; never a physical eligibility token."""

    def __init__(self, token, *, config, task, binding, reader, physical, bridge):
        if token is not _ACQUIRED:
            raise ValueError("Current Build must be acquired through the original readers")
        self._config, self._task = deepcopy(config), deepcopy(task)
        self._binding, self._reader = binding, reader
        self._physical, self._bridge = physical, bridge
        self._declaration = deepcopy(config["current_build_binding"])
        self._phase = {}
        self._local_inputs = None

    def physical_inputs(self):
        """Join an explicit measured model to the one decoded actual Build."""
        if "metric_model" not in self._declaration:
            return None
        self._active()
        if self._local_inputs is None:
            from .local_build import LocalBuildInputs
            self._local_inputs = LocalBuildInputs(self._config,
                operation_result=self._declaration["operation_result"],
                metric_model=self._declaration["metric_model"],
                frame_receipts={row["case_id"]: row["bridge_frame"]
                                for row in self._declaration["frames"]},
                reader=self._reader.original, _checked_build=self)
        return self._local_inputs

    def _active(self):
        from .run_io import for_task
        if for_task(self._task) is not self._binding:
            raise ValueError("Current Build lost its exact active run binding")
        self._reader.recheck()

    def validate_mass_config(self, mass_config, *, statistics_only):
        if self._config.get("format") not in {
                "darsia-native-phase-statistics-run-v1", "darsia-native-phase-n1-run-v1"}:
            raise ValueError("Current Build has no physical consumer for this configuration format")
        phase1 = self._config["format"] == "darsia-native-phase-statistics-run-v1"
        if type(statistics_only) is not bool or statistics_only != phase1:
            raise ValueError("Current Build consumer stage differs from its exact configuration")
        template = self._config["statistics"] if phase1 else self._config["mass"]["template"]
        expected = dict(template, run=self._config["run"],
                        frames=self._config["frames"], data_aliases=self._config.get("data_aliases", ()))
        if not _same(mass_config, expected) or "pixel_domain" in mass_config:
            raise ValueError("Current Build requires its unchanged original physical consumer configuration")

    def require_physical_inputs(self):
        """Stop before legacy physical-reference decoding or any physical math.

        A saved archive/receipt cannot supply the missing original role-aware
        consumer. In particular, the existing sand fill gives ID zero a meaning
        that differs from the actual Build's Freewater role.
        """
        self._active()
        raise ValueError(PHYSICAL_INPUTS_UNAVAILABLE)

    def check_phase(self, spec, native, frame):
        """Join already prepared phase inputs to original RAW/coordinate receipts."""
        # Only one pending frame may retain dense buffers. A new frame also
        # releases a failed prior pending frame; no run-sized buffer archive.
        self._phase.clear()
        self._active()
        from .build_area_data import _read_measured_area_context
        from .build_static_components import _read_old_static
        from ..native import source, raw_rig_source
        import numpy as np

        selected = [row for row in self._config["frames"] if row["case_id"] == spec["case_id"]]
        declared = [row for row in self._declaration["frames"] if row["case_id"] == spec["case_id"]]
        if len(selected) != 1 or len(declared) != 1 or not _same(selected[0], spec):
            raise ValueError("Current Build phase input differs from its exact selected case")
        declared = declared[0]
        receipt = self._reader.json(declared["bridge_frame"])
        area = self._reader.json(declared["area_provenance"])
        static = self._reader.json(declared["static_provenance"])
        roles = spec["phase_roles"]
        if (not _same(receipt["bridge"], self._declaration["bridge"])
                or not _physical_context_equal(receipt["physical_context"], self._physical.provenance)
                or not _same(receipt["native_frame"], spec["native_task"]["geometry"])
                or not _same(receipt["raw_source_binding"], spec["native_task"]["processor"]["raw_binding"])
                or not _same(area["area"], roles["area"])
                or roles["area"].get("role") != "measured_parent"):
            raise ValueError("Current Build RAW/area receipt differs from the actual phase roles")
        inputs = area["inputs"]
        historical = self._reader.json(inputs["historical_frame_receipt"])
        measured = self._reader.json(inputs["measured_frame_receipt"])
        original_area = dict(area["area"], historical=historical["rig_domain_revision"]["area_map"],
                             measured=measured["rig_domain_revision"]["area_map"])
        context = _read_measured_area_context(geometry=declared["geometry"], area=original_area,
            reader=self._reader, **{key: inputs[key] for key in (
                "measurement", "physical_scale_plan", "physical_scale_receipt",
                "measured_frame_receipt", "historical_frame_receipt")})
        if (not _same(context["inputs"], inputs)
                or not _same(context["inputs"]["new_bridge_frame"], declared["bridge_frame"])
                or not _same(context["inputs"]["native_configuration"], self._config["native"]["configuration"])
                or not _same(context["inputs"]["phase_configuration"], self._config["phase"]["configuration"])
                or not _same(context["geometry"], native["geometry_row"])
                or not _same(context["native"], native["native_geometry_row"])
                or not _same(context["native"]["acquisition"], declared["raw_acquisition"])):
            raise ValueError("Current Build original geometry/source-map inputs differ")
        parameters = context["current"]
        raw_binding = self._reader.json(receipt["raw_source_binding"])
        raw_rig_source.validate_binding(raw_binding, parameters)
        raw_rig_source.validate_frame(native["native_geometry_row"], raw_binding, parameters)
        audit, prior = native["audit"], receipt["correction"]
        for key in ("contract", "jpeg_inputs", "source_identity"):
            if not _same(audit[key], prior[key]):
                raise ValueError("Actual RAW audit differs from its Build acquisition")
        if not _source_equal(audit["sources"]["camera_arw"], declared["raw_acquisition"], self._reader):
            raise ValueError("Actual RAW decoder consumed another acquisition")
        for key in ("models", "execution_identity", "profile_id", "native_shape_hw"):
            if not _same(audit["processor"][key], prior["processor"][key]):
                raise ValueError("Actual RAW processor differs from the Build bridge")
        if not _source_equal(audit["processor"]["source_binding"], receipt["raw_source_binding"], self._reader):
            raise ValueError("Actual RAW processor consumed another source binding")
        for left, right in ((audit["renderer"]["native_to_camera_u_h"], prior["renderer"]["native_to_camera_u_h"]),
                            (audit[context["reference_field"]], context["native"]["reference_to_native_h"]),
                            (audit["H_v5_B_to_frame_native"], context["native"]["baseline_to_native_h"]),
                            (frame["current_h"], roles["current_h"]),
                            (frame["reference_to_current_h"], roles["reference_to_current_h"])):
            if not source._same_h(source.source_homography(left, parameters=parameters),
                                  source.source_homography(right, parameters=parameters), parameters=parameters):
                raise ValueError("Actual RAW/phase homography differs from the original coordinate join")
        old_static, old_inputs = _read_old_static(static["inputs"]["old_configuration"], spec["case_id"],
            run=context["native"]["run"], acquisition=context["native"]["acquisition"],
            old_geometry=inputs["old_geometry"], old_native_configuration=inputs["old_native_configuration"],
            old_raw_binding=inputs["old_raw_binding"], old_audit=context["old"],
            reference_field=context["reference_field"], parameters=context["previous"],
            old_shape=context["old"]["renderer"]["native_shape_hw"], reader=self._reader)
        geometry_inputs = {key: value for key, value in inputs.items() if key not in (
            "measurement", "physical_scale_plan", "physical_scale_receipt", "measured_frame_receipt",
            "historical_frame_receipt", "parent_seal", "parent_registration")}
        expected_static_inputs = dict(geometry_inputs, **old_inputs, producer=static["inputs"]["producer"])
        if (static.get("format") != "darsia-static-component-coordinate-data-v1"
                or static.get("physical_qualification") is not False
                or not _same(static["inputs"], expected_static_inputs)
                or not _same(static["components"], roles["static"]["components"])
                or roles["static"].get("unchanged_source") is not True
                or not _same(static["source_component_ids"], old_static["source_ids"])
                or not _same(static["source_component_ids"], roles["static"]["source_ids"])
                or not _same(frame["source_component_ids"], static["source_component_ids"])
                or not _same(static["acquisition"], declared["raw_acquisition"])
                or static["case_id"] != spec["case_id"] or static["run"] != self._config["run"]):
            raise ValueError("Current Build static-coordinate lineage differs")
        for bound in [area["producer"], *static["inputs"].values()]:
            self._reader.path(bound)
        if (frame["context"] is not native["context"] or frame["full_log"] is not native["full_log"]
                or not _same(frame["source"], spec["source"])
                or not _same(frame["physical_scale_revision"], area["numerical_scale_qa"])
                or frame["original_area"].dtype != np.float32
                or frame["pixel_area_mm2"].dtype != np.float32
                or list(frame["pixel_area_mm2"].shape) != context["shape"]):
            raise ValueError("Actual original phase buffers/provenance differ from the checked Build roles")
        self._reader.recheck()
        gap, gap_provenance = None, None
        if "approved_analysis_gap" in self._physical.provenance:
            from ..painted_analysis_exclusions import transport_build_exclusion, restrict_phase_domain
            with self._reader.direct():
                gap, gap_provenance = transport_build_exclusion(self._physical, self._reader,
                    receipt["build_to_frame_native_h"], frame["context"].origin_xy,
                    frame["context"].rig_domain.shape, context["phase"]["geometry_parameters"])
            frame = restrict_phase_domain(frame, gap, gap_provenance)
        self._phase[spec["case_id"]] = dict(source=deepcopy(spec["source"]["source_binding"]),
            current=deepcopy(native["record"]["full"]), shape=context["shape"],
            frame=frame, receipt=receipt, gap=gap, gap_provenance=gap_provenance,
            geometry_parameters=context["phase"]["geometry_parameters"])
        return frame

    def check_inferred(self, spec, frame, inferred):
        """Retain the original approved-gap check and phase output provenance."""
        import numpy as np
        self._active()
        row = self._phase.get(spec.get("case_id"))
        if row is None or frame is not row["frame"]:
            raise ValueError("Current Build inference lacks its exact checked phase frame")
        if row["gap"] is not None:
            gap = row["gap"]
            if np.any(inferred["labels"][gap] != 255):
                raise ValueError("Phase inference filled approved unquantified analysis gaps")
            inferred["classes"]["approved_analysis_gap"] = gap
            inferred["classes"]["physical_rig_domain"] = frame["physical_rig_domain"]
            inferred["qa"]["approved_analysis_gap"] = row["gap_provenance"]
        row["labels"] = inferred["labels"]

    def check_mass(self, spec, frame, x, y, photo, report, *, mass_config=None):
        """Inspect the one original loaded tuple, retaining the physical STOP."""
        # Drop the owned dense reference before any validation can raise.
        row = self._phase.pop(spec.get("case_id"), None)
        self._phase.clear()
        self._active()
        self.validate_mass_config(mass_config, statistics_only=(
            self._config["format"] == "darsia-native-phase-statistics-run-v1"))
        from ..mass.n1_models import FrameData
        import numpy as np
        if row is None or "labels" not in row or not isinstance(frame, FrameData):
            raise ValueError("Current Build mass input lacks its actual checked phase/FrameData")
        if (frame.run_id != self._config["run"] or frame.frame_index != spec["frame_index"]
                or frame.elapsed_h != spec["elapsed_h"]
                or list(frame.phase_labels.shape) != row["shape"]
                or not _same(spec["source_binding"], row["source"])
                or not _same(spec["current"], row["current"])
                or not _same(frame.provenance.get("source_binding"), row["source"])):
            raise ValueError("Loaded N1 frame differs from its original current Build phase source")
        shape = tuple(row["shape"])
        if (any(not isinstance(value, np.ndarray) or value.shape != shape
                or value.dtype != np.float32 for value in (x, y))
                or not np.isfinite(x).all() or not np.isfinite(y).all()
                or not isinstance(photo, np.ndarray) or photo.shape != shape or photo.dtype != bool
                or not np.array_equal(frame.phase_labels, row["labels"])
                or not isinstance(report, dict)
                or not _same(report, frame.provenance.get("round4_sandmap"))
                or report.get("contract") != "darsia-n1-corrected-geometry-v1"
                or report.get("run") != frame.run_id or report.get("frame_index") != frame.frame_index
                or report.get("elapsed_h") != frame.elapsed_h):
            raise ValueError("Original loaded coordinates/photo/report differ from the checked phase tuple")
        # These are identity/structure checks, never a replacement for an actual
        # original physical-field/coordinate/role consumer. No tuple can enable
        # this route while that original join is absent.
        self._reader.recheck()
        raise ValueError(PHYSICAL_INPUTS_UNAVAILABLE)


def acquire_current_build(config, task, io_binding):
    """Acquire an explicit original Build after startup under exact active I/O."""
    from darsia_cache.current_build_binding import checked_current_build_binding
    from .run_io import RunIOBinding, for_task
    if (not isinstance(io_binding, RunIOBinding) or for_task(task) is not io_binding
            or not _same({key: value for key, value in config.items() if key != "_configuration_sha256"},
                         io_binding.configuration)):
        raise ValueError("Current Build requires its actual replayed configuration and active RunIOBinding")
    declaration = checked_current_build_binding(config["current_build_binding"], config=config)
    reader = _Reader(io_binding, task)
    completed, shared_receipt = _Publication(declaration["publication"], reader).verify(declaration["publication"])
    if (completed["operation"] != "build" or completed["run"] != config["run"]
            or not _same(completed["files"]["operation_result.json"], declaration["operation_result"])):
        raise ValueError("Current Build publication belongs to another original Build")
    receipt = reader.json(declaration["operation_result"])
    if not _same(receipt, shared_receipt):
        raise ValueError("Original and shared Build operation receipts differ")
    certificate = reader.json(receipt["baseline_renderer"])
    plan = reader.json(receipt["setup_plan"])
    _require_supported_physical_revision(reader.json(plan["adopted_revision"]))
    # The original physical reader verifies live owned-directory identities.
    # Preserve that contract rather than pretending relocated archives qualify.
    for bound in [receipt["baseline_renderer"], *certificate["outputs"]]:
        reader.path(bound, original=True)
    from ..mass.native_optical_source import load_build_physical_context
    with reader.direct():
        physical = load_build_physical_context(declaration["operation_result"], reader=reader,
                                               local_reuse=declaration.get("local_build_reuse"))
    if "local_build_reuse" in declaration:
        from ..local_build_reuse import verify_local_build_frame_configuration
        reuse = physical.provenance["local_build_reuse"]
        verify_local_build_frame_configuration(config, reader.json(reuse["frame_configuration"]))
    bridge = reader.json(declaration["bridge"])
    if (bridge.get("format") != "darsia-build-native-bridge-v1"
            or not _same(bridge["operation_result"], declaration["operation_result"])
            or not _same(bridge["native_configuration"], config["native"]["configuration"])
            or not _same(bridge["saved_build_to_camera_u_h"], certificate["H_rig_corrected_native_to_baseline_camera_u"])):
        raise ValueError("Current Build bridge differs from its actual physical producer")
    reader.recheck()
    return CheckedCurrentBuild(_ACQUIRED, config=config, task=task, binding=io_binding,
                               reader=reader, physical=physical, bridge=bridge)
