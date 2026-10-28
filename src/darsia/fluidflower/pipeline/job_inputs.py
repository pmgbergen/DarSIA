"""Input catalogues from named native/phase/N1 consumers; metadata/stat only."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import ntpath
from pathlib import Path
import re
import stat
from ..store_map import input_reference, validate_input_reference
from .io_ops import retry_io

CATALOG_FORMAT = "darsia-role-input-catalog-v1"
MAX_METADATA_BYTES = 128 * 1024 * 1024


def _key(path):
    return ntpath.normcase(ntpath.normpath(str(path)))


def _json_bytes(raw):
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ValueError("Duplicate metadata key: " + key)
            value[key] = item
        return value
    def invalid(value):
        raise ValueError("Nonfinite metadata constant: " + value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("Input metadata must be a JSON object")
    return value


class InputCatalog:
    """Read-only role preparation, never an execution or SHA-check bypass.

    Original metadata bytes are SHA-verified. Payload SHA is the original
    descriptor's value; actual full payload verification remains J3 pending.
    """
    def __init__(self, roots, aliases=(), *, metadata_bytes=None, file_stat=None):
        self.roots, self.aliases = dict(roots), {}
        for row in aliases:
            logical, physical = row["logical"], row["physical"]
            if logical["sha256"] != physical["sha256"]:
                raise ValueError("Input alias changes source SHA")
            if "size_bytes" in logical and logical["size_bytes"] != physical.get("size_bytes"):
                raise ValueError("Input alias changes source size")
            key = _key(logical["path"])
            if key in self.aliases and self.aliases[key] != physical:
                raise ValueError("Conflicting source input aliases")
            self.aliases[key] = dict(physical)
        self.metadata_bytes = metadata_bytes or self._read_metadata
        self.file_stat = file_stat or (lambda path: Path(path).stat())
        self.entries, self.documents, self.sizes = {}, {}, {}
        self.consumers = {}
        self.source_setups = set()
        self.native_exact_inputs = {}

    def add_native(self, descriptor, role, use):
        """Admit one consumed native file without granting its parent tree."""
        from ..store_map import StoreMapError
        binding = getattr(self, "binding", None)
        if binding is not None:
            reference = self.add(descriptor, role, use)
            if reference["root_alias"] == "native_exact":
                matches = [row for row in binding.task["input_catalog"].get("native_exact_inputs", [])
                           if row["reference"] == reference]
                if len(matches) != 1:
                    raise ValueError("Native exact input lost its original immutable file binding")
                self.native_exact_inputs[reference["relative_path"]] = deepcopy(matches[0])
            return reference
        try:
            return self.add(descriptor, role, use)
        except StoreMapError as error:
            if not str(error).startswith("No declared root contains input path:"):
                raise
        actual = self.physical(descriptor)
        if (role != "rig_model" or not use.startswith("setup.raw.native.")
                or type(descriptor.get("size_bytes")) is not int):
            raise StoreMapError("No declared root contains input path: " + str(actual["path"]))
        from ..store_map import _path
        _path(actual["path"])
        info = self.file_stat(actual["path"])
        if not stat.S_ISREG(info.st_mode) or info.st_size != descriptor["size_bytes"]:
            raise ValueError("Consumed native exact input size/type changed")
        relative = descriptor["sha256"] + "-" + hashlib.sha256(_key(actual["path"]).encode()).hexdigest()[:16] + ".input"
        reference = dict(root_alias="native_exact", relative_path=relative,
            sha256=descriptor["sha256"], bytes=info.st_size, role=role)
        validate_input_reference(reference)
        record = dict(reference=reference, descriptor=dict(path=actual["path"],
            sha256=descriptor["sha256"], size_bytes=info.st_size))
        previous = self.native_exact_inputs.get(relative)
        if previous is not None and previous != record:
            raise ValueError("Conflicting native exact input binding")
        self.native_exact_inputs[relative] = record
        if use in self.consumers and self.consumers[use] != reference:
            raise ValueError("Ambiguous native exact input consumer")
        self.consumers[use] = deepcopy(reference)
        key = (reference["root_alias"], relative, reference["sha256"], role)
        row = self.entries.setdefault(key, dict(reference=reference, uses=[]))
        if use not in row["uses"]:
            row["uses"].append(use)
        return deepcopy(reference)

    @staticmethod
    @retry_io
    def _read_metadata(path):
        with Path(path).open("rb") as stream:
            raw = stream.read(MAX_METADATA_BYTES + 1)
        if len(raw) > MAX_METADATA_BYTES:
            raise ValueError("Explicit input metadata exceeds 128 MiB: " + str(path))
        return raw

    def physical(self, descriptor):
        if (not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str)
                or re.fullmatch(r"[0-9a-f]{64}", descriptor.get("sha256", "")) is None):
            raise ValueError("Consumed input needs its original path and SHA")
        result = self.aliases.get(_key(descriptor["path"]), descriptor)
        if result["sha256"] != descriptor["sha256"]:
            raise ValueError("Input locator and source identity differ")
        return result

    @retry_io
    def add(self, descriptor, role, use):
        actual = self.physical(descriptor)
        # Root admission precedes any stat/open. Passive provenance is not visited.
        reference = input_reference(actual["path"], self.roots,
            sha256=descriptor["sha256"], size_bytes=0, role=role)
        identity = (_key(actual["path"]), descriptor["sha256"])
        if identity not in self.sizes:
            info = self.file_stat(actual["path"])
            if not stat.S_ISREG(info.st_mode) or type(info.st_size) is not int or info.st_size < 0:
                raise ValueError("Consumed input is not a regular file: " + str(actual["path"]))
            self.sizes[identity] = info.st_size
        size = self.sizes[identity]
        declared = descriptor.get("size_bytes", actual.get("size_bytes"))
        if declared is not None and (type(declared) is not int or declared != size):
            raise ValueError("Consumed input size changed: " + str(actual["path"]))
        reference["bytes"] = size  # Actual stat, never guessed from provenance.
        key = (reference["root_alias"], reference["relative_path"], reference["sha256"], role)
        if not isinstance(use, str) or not use:
            raise ValueError("Consumed input needs a named original consumer")
        previous = self.consumers.get(use)
        if previous is not None and previous != reference:
            raise ValueError("Ambiguous input consumer maps to different references: " + use)
        self.consumers[use] = deepcopy(reference)
        row = self.entries.setdefault(key, {"reference": reference, "uses": []})
        if use not in row["uses"]:
            row["uses"].append(use)
        return deepcopy(reference)

    @retry_io
    def doc(self, descriptor, role, use):
        reference = self.add(descriptor, role, use)
        actual = self.physical(descriptor)
        key = (_key(actual["path"]), descriptor["sha256"])
        if key not in self.documents:
            raw = self.metadata_bytes(actual["path"])
            if (not isinstance(raw, bytes) or len(raw) > MAX_METADATA_BYTES
                    or len(raw) != reference["bytes"]
                    or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]):
                raise ValueError("Consumed metadata SHA/size changed: " + str(actual["path"]))
            self.documents[key] = _json_bytes(raw)
        return deepcopy(self.documents[key])

    def statistics_document(self, descriptor, role, use):
        """Decode only a named, SHA-verified statistics document, including gzip."""
        import gzip
        import io
        reference = self.add(descriptor, role, use)
        actual = self.physical(descriptor)
        raw = self.metadata_bytes(actual["path"])
        if (not isinstance(raw, bytes) or len(raw) != reference["bytes"]
                or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]):
            raise ValueError("Consumed statistics SHA/size changed")
        if raw[:2] == b"\x1f\x8b":
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                decoded = stream.read(MAX_METADATA_BYTES + 1)
        else:
            decoded = raw
        if len(decoded) > MAX_METADATA_BYTES:
            raise ValueError("Decoded statistics exceed the metadata byte limit")
        return _json_bytes(decoded)

    def pair(self, value, path_key, sha_key, role, use):
        return self.add({"path": value[path_key], "sha256": value[sha_key]}, role, use)

    def pair_doc(self, value, path_key, sha_key, role, use):
        return self.doc({"path": value[path_key], "sha256": value[sha_key]}, role, use)

    def full(self, value, use):
        centre = value["native_centre"]
        self.add(centre["log_linear_rgb"], "image", use + ".centre.log")
        if "valid_mask" in centre:
            self.add(centre["valid_mask"], "rig_model", use + ".centre.valid")
        for index, window in enumerate(value["ring_windows"]):
            self.add(window["log_linear_rgb"], "image", use + ".ring." + str(index))

    def original_native_row(self, row, parameters, use):
        keys = parameters["fields"]["native"]
        if len(keys) != 4:
            raise ValueError("Unsupported native descriptor-field schema")
        self.pair(row, keys[0], keys[1], "image", use + ".masked_raw")
        self.pair(row, keys[2], keys[3], "rig_model", use + ".masked_valid")
        temporal = parameters["fields"]["temporal"]
        if len(temporal) < 2:
            raise ValueError("Unsupported temporal descriptor-field schema")
        if all(row.get(key) is not None for key in temporal):
            self.pair(row, temporal[0], temporal[1], "image", use + ".temporal_ring")

    def original_acquisition(self, value, parameters, use):
        row = self.doc(value["native_row"], "configuration", use + ".native_row")
        self.full(value["full"], use + ".full")
        self.original_native_row(row, parameters, use)

    def mass(self, config, frames, *, standalone=False):
        from ..accepted_pressure_policy import verify_accepted_pressure
        verify_accepted_pressure(config.get("atmosphere_series", {}), catalog=self)
        calibration = self.doc(config["calibration"], "calibration", "mass.calibration")
        self.doc(config["physical_correction"], "rig_model", "mass.physical_correction")
        shared = config["shared"]
        for key in ("physical_reference", "floor_curve"):
            self.add(shared[key], "rig_model", "mass." + key)
        if isinstance(shared["depth_points"], dict):
            self.add(shared["depth_points"], "rig_model", "mass.depth_points")
        if ("reference_matrix" not in shared and isinstance(shared["physical_coordinate_provenance"], dict)
                and "path" in shared["physical_coordinate_provenance"]):
            self.doc(shared["physical_coordinate_provenance"], "rig_model", "mass.coordinate_provenance")
        for key in ("rgb", "valid"):
            self.add(shared["clearwater"][key], "image" if key == "rgb" else "rig_model", "mass.clearwater." + key)
        optics = shared["optics"]
        self.doc(optics["metadata"], "model", "mass.optics.metadata")
        for group in ("color_paths", "signals"):
            for label, bound in optics[group].items():
                self.add(bound, "model", "mass.optics." + group + "." + str(label))
        self.add(optics["flash"], "model", "mass.optics.flash")
        if "path" in config["boxes"]:
            self.add(config["boxes"], "rig_model", "mass.boxes")
        for frame in frames:
            use = "frame." + frame["case_id"] + ".mass"
            self.add(frame["photo_top"], "rig_model", use + ".photo_top")
            self.add(frame["original_domain"]["parent_labels_map"], "rig_model",
                     use + ".original_domain.parent_labels_map")
            if standalone:
                self.full(frame["current"], use + ".current")
                self.full(frame["baseline"], use + ".baseline")
                for key in ("labels", "classes", "area"):
                    self.add(frame[key], "model" if key in {"labels", "classes"} else "rig_model", use + "." + key)
        if standalone and "phase1" in config:
            self.phase2_evidence(config, frames)
        if "pixel_domain" in config:
            self.n16_inputs(config, frames, calibration)

    def phase2_evidence(self, config, frames):
        """Catalogue only the saved phase-one evidence the ordinary N1 writer reads."""
        from . import staged_data as staged
        from .series import bind_mass_frame
        staged.validate_phase2_run(config)
        binding = config["phase1"]
        phase1 = staged.validate_phase1_run(self.doc(binding["configuration"], "configuration", "mass.phase1.configuration"))
        core = self.doc(binding["completion"], "configuration", "mass.phase1.completion")
        order = [spec["case_id"] for spec in frames]
        if (core.get("format") != staged.COMPLETION_FORMAT or core.get("complete") is not True
                or core.get("history_complete") is not True or core.get("run") != config["run"]
                or core.get("configuration") != binding["configuration"] or core.get("precision") != config["precision"]
                or core.get("code_commit") != binding["code_commit"] or core.get("ordered_case_ids") != order
                or phase1["ordered_case_ids"] != order or phase1["run"] != config["run"]
                or phase1["precision"] != config["precision"]
                or not isinstance(core.get("frames"), list) or len(core["frames"]) != len(order)):
            raise ValueError("Phase-two catalogue requires its actual complete phase-one source/order")
        for key, value in phase1["statistics"].items():
            actual = config.get(key)
            if key == "provenance" and isinstance(actual, dict) and "common_frozen_refit" in actual:
                from .common_calibration import JOIN_FORMAT
                if not isinstance(actual["common_frozen_refit"], dict) or actual["common_frozen_refit"].get("format") != JOIN_FORMAT:
                    raise ValueError("Only the recognized verified common C provenance extension may be added")
                actual = {field: content for field, content in actual.items() if field != "common_frozen_refit"}
            if key != "format" and not staged._same_json(actual, value):
                raise ValueError("Phase-two scientific DATA differs from its phase-one template: " + key)
        for spec, original, committed in zip(frames, phase1["frames"], core["frames"]):
            refs = binding["frames"][spec["case_id"]]
            use = "frame." + spec["case_id"] + ".mass.phase1"
            if not staged._same_json(committed, refs["commit"]):
                raise ValueError("Phase-two case commit differs from its actual phase-one completion")
            record = staged._frame_record(self.doc(refs["commit"], "configuration", use + ".commit"),
                original, binding["configuration"], config["precision"], refs["commit"])
            expected = bind_mass_frame(original, record["native"], record["phase"], baseline=phase1["phase"]["reference"]["full"])
            compared = {k: v for k, v in spec.items() if k not in {"pixel_domain_geometry", "frozen_response_basis"}}
            if not staged._same_json(expected, compared):
                raise ValueError("Phase-two original N1 frame differs from its sealed phase-one source")
            for field in ("statistics", "before_rules", "photograph"):
                if not staged._same_json(refs.get(field), record.get(field)):
                    raise ValueError("Phase-two saved evidence differs from its original commit: " + field)
            before = record["before_rules"]
            self.doc(before["receipt"], "configuration", use + ".before_receipt")
            if before["state"] == "complete":
                self.add(before["png"], "model", use + ".before_png")
            if "photograph" in record:
                photo = record["photograph"]
                self.doc(photo["receipt"], "configuration", use + ".photo_receipt")
                if photo["state"] == "complete":
                    self.add(photo["png"], "image", use + ".photo_png")
        if "common_frozen_refit" in config.get("provenance", {}):
            from .common_calibration import catalogue_refit
            catalogue_refit(self, config)

    def n16_inputs(self, config, frames, calibration):
        """Positive named N16 consumers; passive historical paths stay unvisited."""
        from .staged_data import validate_phase2_run, validate_n16_frame_binding
        validate_phase2_run(config)
        policy = config["pixel_domain"]
        self.doc(policy["authorized_decisions"], "configuration", "mass.pixel_domain.authorized_decisions")
        authority = self.doc(policy["n16"]["authority"], "configuration", "mass.pixel_domain.n16.authority")
        for spec in frames:
            use = "frame." + spec["case_id"] + ".mass.n16"
            declared, basis = spec["pixel_domain_geometry"], spec["frozen_response_basis"]
            source = self.doc(declared["source_receipt"], "rig_model", use + ".source_geometry")
            original = self.doc(basis["result"], "configuration", use + ".frozen_response_result")
            validate_n16_frame_binding(config, spec, authority=authority, source=source,
                original=original, calibration=calibration)
            self.add(declared["target_depth"], "rig_model", use + ".target_depth")
            self.add(basis["geometry"], "rig_model", use + ".frozen_response_geometry")
            for key, bound in declared["n16_masks"].items():
                if bound is not None:
                    self.add(bound, "rig_model", use + "." + key)

    def statistics(self, config, frames, *, standalone=False):
        """Original summary-source consumers, omitting only fitted mass calibration."""
        from ..accepted_pressure_policy import verify_accepted_pressure
        verify_accepted_pressure(config.get("atmosphere_series", {}), catalog=self)
        self.doc(config["physical_correction"], "rig_model", "mass.physical_correction")
        shared = config["shared"]
        for key in ("physical_reference", "floor_curve"):
            self.add(shared[key], "rig_model", "mass." + key)
        if isinstance(shared["depth_points"], dict):
            self.add(shared["depth_points"], "rig_model", "mass.depth_points")
        if ("reference_matrix" not in shared and isinstance(shared["physical_coordinate_provenance"], dict)
                and "path" in shared["physical_coordinate_provenance"]):
            self.doc(shared["physical_coordinate_provenance"], "rig_model", "mass.coordinate_provenance")
        for key in ("rgb", "valid"):
            self.add(shared["clearwater"][key], "image" if key == "rgb" else "rig_model", "mass.clearwater." + key)
        optics = shared["optics"]
        self.doc(optics["metadata"], "model", "mass.optics.metadata")
        for group in ("color_paths", "signals"):
            for label, bound in optics[group].items():
                self.add(bound, "model", "mass.optics." + group + "." + str(label))
        self.add(optics["flash"], "model", "mass.optics.flash")
        if "path" in config["boxes"]:
            self.add(config["boxes"], "rig_model", "mass.boxes")
        for frame in frames:
            use = "frame." + frame["case_id"] + ".mass"
            self.add(frame["photo_top"], "rig_model", use + ".photo_top")
            self.add(frame["original_domain"]["parent_labels_map"], "rig_model",
                     use + ".original_domain.parent_labels_map")
            if standalone:
                self.full(frame["current"], use + ".current")
                self.full(frame["baseline"], use + ".baseline")
                for key in ("labels", "classes", "area"):
                    self.add(frame[key], "model" if key in {"labels", "classes"} else "rig_model", use + "." + key)

    def phase(self, config):
        phase = self.doc(config["phase"]["configuration"], "configuration", "phase.configuration")
        p = phase["parameters"]
        native = self.doc(config["native"]["configuration"], "configuration", "native.configuration")
        for key in ("reference", "initial_previous"):
            self.original_acquisition(config["phase"][key], p["inputs"], "phase." + key)
            if any(frame["native_task"]["kind"] == "raw_rig" for frame in config["frames"]):
                from .phase_inputs import verify_raw_acquisition
                record = config["phase"][key]
                row = self.doc(record["native_row"], "configuration", "phase." + key + ".native_row")
                audit = self.doc(row["raw_producer_audit"], "configuration", "phase." + key + ".raw_audit")
                verify_raw_acquisition(record, row, audit, parameters=p["inputs"])
                self.add(row["raw_acquisition"], "image", "phase." + key + ".raw_acquisition")
        bundles = phase["model_bundles"]
        for role in ("native", "edge"):
            item = bundles[role]
            for key in item["requirements"]["expected_sha256"]:
                self.doc(item["documents"][key], "model", "phase.model." + role + "." + key)
        for key in ("pixel_model", "regional_model"):
            self.doc(bundles["placement"][key], "model", "phase.placement." + key)
        if bundles["placement"].get("projected_models") is not None:
            for key in ("pixel", "regional"):
                self.doc(bundles["placement"]["projected_models"][key], "model", "phase.placement.projected." + key)
        chain = phase["chain_assets"]
        envelopes = self.doc(chain["envelopes"], "model", "phase.envelopes")
        for key in ("envelope1", "envelope2"):
            self.add(envelopes[key], "rig_model", "phase." + key)
        for key in ("floor", "zones"):
            self.add(chain[key], "rig_model", "phase." + key)
        return phase, native

    def retained(self, binding, parameters, use):
        parent = self.doc(binding["parent_native_manifest"], "configuration", use + ".parent_native_manifest")
        receipt = self.doc(binding["parent_frame_receipt"], "model", use + ".parent_frame_receipt")
        exported = self.doc(binding["parent_probability_receipt"], "model", use + ".parent_probability_receipt")
        self.doc({"path": exported["metadata"]["native_manifest_path"], "sha256": exported["binding"]["native_manifest_sha256"]},
                 "configuration", use + ".original_probability_native_manifest")
        frame = parent["frame_sources"][binding["frame_index"]]
        for name in parameters["parent_array_names"]:
            self.pair(frame, name + "_path", name + "_sha256", "image", use + ".parent." + name)
        self.add(exported["files"][parameters["mask_file"]], "rig_model", use + ".masks")
        self.add(receipt["labels_map"], "model", use + ".labels_proof")

    def _cohort(self, source_row, parameters, use):
        source = self.pair_doc(source_row, "source_manifest_path", "source_manifest_sha256", "configuration", use + ".source_manifest")
        symbols = parameters["source"]["symbols"]
        for index, row in enumerate(source_row["frames"]):
            tag = use + ".cohort." + str(index)
            receipt = self.pair_doc(row, "receipt_path", "receipt_sha256", "configuration", tag + ".receipt")
            for key in ("native_review_jpeg", "camera_jpeg", "v5_B_native_jpeg", "v5_B_arw", "v5_B_camera_jpeg"):
                self.pair(receipt, key + "_path", key + "_sha256", "image", tag + "." + key)
            self.pair(receipt, "v5_B_registration_certificate_path", "v5_B_registration_certificate_sha256", "rig_model", tag + ".B_registration")
            temporal = self.pair_doc(receipt, "temporal_geometry_receipt_path", "temporal_geometry_receipt_sha256", "rig_model", tag + ".temporal")
            if temporal["frames"][index].get("temporal_registration_pass") is not True:
                self.pair_doc(receipt, symbols["temporal_exception_path_key"], symbols["temporal_exception_sha_key"], "rig_model", tag + ".temporal_exception")
            self.pair_doc(receipt, "photo_sand_reference_receipt_path", "photo_sand_reference_receipt_sha256", "rig_model", tag + ".photo_acceptance")
            self.pair(receipt, "photo_sand_reference_mask_path", "photo_sand_reference_mask_sha256", "rig_model", tag + ".photo_mask")
            if source["run"] == symbols["prepared_source_id"] and index == 0:
                original = source["frame_sources"][0]
                self.add({"path": original["source_path"], "sha256": original.get("source_sha256") or original["source_raw_sha256"]}, "image", tag + ".prepared_source")

    def _source_setup(self, source_row, calibration_bound, aggregate_bound, parameters, use):
        identity = (source_row["run"], source_row["source_manifest_sha256"], calibration_bound["sha256"], aggregate_bound["sha256"])
        if identity in self.source_setups:
            return
        calibration = self.doc(calibration_bound, "calibration", use + ".calibration")
        self.doc(aggregate_bound, "configuration", use + ".photo_aggregate")
        self._cohort(source_row, parameters, use)
        calibration_path = Path(self.physical(calibration_bound)["path"])
        for source_id in (source_row["run"], "virtual_reference"):
            rows = [row for row in calibration["results"] if row.get("run") == source_id]
            if len(rows) != 1:
                raise ValueError("Exactly one original baseline binding required: " + source_id)
            row = rows[0]
            stem = calibration_path.parent / (source_id + "_baseline")
            self.add({"path": str(stem.with_suffix(".npz")), "sha256": row["baseline_sha256"]}, "calibration", use + ".baseline." + source_id)
            self.doc({"path": str(stem.with_suffix(".json")), "sha256": row["sidecar_sha256"]}, "calibration", use + ".baseline_sidecar." + source_id)
        for row in calibration.get("old_B_local_residual_fields", []):
            if row.get("run") == source_row["run"]:
                path = self.aliases.get(_key(row["field_path"]), {}).get("path", row["field_path"])
                if not Path(path).is_absolute():
                    path = str(calibration_path.parents[3] / path)
                self.add({"path": path, "sha256": row["field_sha256"]}, "calibration", use + ".old_B_residual")
                break
        self.source_setups.add(identity)

    def native_task(self, task, geometry, parameters, use):
        processor = task["processor"]
        if task["kind"] == "raw_rig":
            from ..native.raw_rig_source import validate_binding, validate_frame
            from ..image_formats import image_family
            binding = self.doc(processor["raw_binding"], "configuration", use + ".raw_binding")
            validate_binding(binding, parameters)
            if parameters["source"].get("raw_execution_identity") != binding["execution_identity"]:
                raise ValueError("Queued RAW source needs its explicit native execution binding")
            validate_frame(geometry, binding, parameters)
            if "normalisation" in binding:
                self.add(binding["normalisation"]["record"], "configuration", use + ".normalisation.record")
            for name, descriptor in (("baseline_raw", binding["baseline_raw"]),
                                     ("acquisition", geometry["acquisition"])):
                if image_family(descriptor["path"]) != "raw":
                    raise ValueError("RAW-only native task has a non-RAW acquisition")
                self.add(descriptor, "image", use + "." + name)
            for name in ("fit", "reference"):
                model = binding["models"][name]
                self.add(model["data"], "calibration", use + "." + name + ".data")
                self.doc(model["sidecar"], "calibration", use + "." + name + ".sidecar")
            if binding["models"]["old_field"] is not None:
                self.add(binding["models"]["old_field"], "calibration", use + ".old_field")
            for index, descriptor in enumerate(binding["shape_assets"]):
                self.add(descriptor, "rig_model", use + ".shape." + str(index))
        elif task["kind"] == "rig":
            manifest = self.doc(processor["native_manifest"], "configuration", use + ".native_manifest")
            binding = manifest["v5_photometry"]
            self.pair_doc(binding, "source_manifest", "source_manifest_sha256", "configuration", use + ".source_manifest")
            source = binding["processor"]
            calibration = {"path": source["calibration_manifest_path"], "sha256": source["calibration_manifest_sha256"]}
            aggregate = {"path": source["photo_source_aggregate_path"], "sha256": source["photo_source_aggregate_sha256"]}
            photos = self.doc(aggregate, "configuration", use + ".photo_aggregate")
            matches = [row for row in photos["rows"] if row["run"] == manifest["run"] and row["source_manifest_sha256"] == binding["source_manifest_sha256"]]
            if len(matches) != 1:
                raise ValueError("Exactly one original photo cohort is required")
            self._source_setup(matches[0], calibration, aggregate, parameters, use)
            parent = task.get("parent_frame_index")
            original = manifest["reference"] if task.get("parent_reference") else (manifest["frame_sources"][parent] if parent is not None else None)
            if original is not None:
                for key in ("v5_log_linear_rgb", "v5_log_linear_valid_mask"):
                    self.pair(original, key + "_path", key + "_sha256", "image", use + ".parent." + key)
                if task.get("parent_reference"):
                    self.pair(original, "v5_baseline_receipt_path", "v5_baseline_receipt_sha256", "configuration", use + ".parent.baseline_receipt")
            for key in ("camera_jpeg", "camera_arw", "native_review_jpeg"):
                self.pair(geometry, key + "_path", key + "_sha256", "image", use + "." + key)
            exact_b = (geometry["camera_arw_sha256"] == geometry["v5_B_arw_sha256"]
                       and geometry["camera_jpeg_sha256"] == geometry["v5_B_camera_jpeg_sha256"])
            if not exact_b:
                self.pair(geometry, "v5_B_registration_certificate_path", "v5_B_registration_certificate_sha256", "rig_model", use + ".B_registration")
                self.pair(geometry, "v5_B_camera_jpeg_path", "v5_B_camera_jpeg_sha256", "image", use + ".B_camera")
        elif task["kind"] == "source":
            source_row = self.doc(processor["source_row"], "configuration", use + ".source_row")
            self._source_setup(source_row, processor["calibration_manifest"], processor["photo_aggregate"], parameters, use)
            for key in ("camera_jpeg", "camera_arw", "review_jpeg"):
                self.add(task["arguments"][key], "image", use + "." + key)
        else:
            raise ValueError("Unsupported native processor kind: " + str(task["kind"]))

    def local_build_reuse(self, binding, *, operation_result, request, config):
        """Admit only the explicit reuse record and its current metadata roles."""
        from ..local_build_reuse import (
            verify_local_build_frame_configuration, verify_local_build_reuse,
        )
        catalogue = self
        names = {}

        def use(bound):
            identity = (_key(bound["path"]), bound["sha256"], bound.get("size_bytes"))
            if identity not in names:
                names[identity] = "current_build.local_reuse.input." + str(len(names))
            return names[identity]

        class Reader:
            def json(self, bound):
                return catalogue.doc(bound, "configuration", use(bound))

            def path(self, bound):
                catalogue.add(bound, "configuration", use(bound))
                return Path(catalogue.physical(bound)["path"])

        reader = Reader()
        checked = verify_local_build_reuse(binding, operation_result=operation_result,
            request=request, run=config["run"], reader=reader,
            metadata_bytes=self.metadata_bytes)
        current = reader.json(checked["provenance"]["frame_configuration"])
        verify_local_build_frame_configuration(config, current)
        return checked

    def current_build(self, config):
        """Catalogue original current-Build readers, without claiming their acceptance.

        Only named original receipt fields are expanded. Payloads are admitted
        by root/stat here; their bytes and original scientific relationships
        remain the worker's existing verification responsibility.
        """
        from darsia_cache.current_build_binding import (
            checked_current_build_binding, current_build_binding_inputs,
        )
        binding = checked_current_build_binding(config["current_build_binding"], config=config)
        same = lambda left, right: json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
            right, sort_keys=True, allow_nan=False)

        def source(bound, use, role="configuration", *, metadata=True, original=False):
            if original and _key(self.physical(bound)["path"]) != _key(bound["path"]):
                raise ValueError("Current Build publication and owned outputs require original physical locators")
            return (self.doc if metadata else self.add)(bound, role, "current_build." + use)

        for use, bound, role in current_build_binding_inputs(binding):
            source(bound, use.removeprefix("current_build."), role,
                   metadata=role != "image", original=use.startswith("current_build.publication.")
                   or use == "current_build.operation_result")
        publication = binding["publication"]
        job = source(publication["job"], "publication.job")
        task = source(publication["task"], "publication.task")
        done = source(publication["done"], "publication.done")
        result = source(binding["operation_result"], "operation_result")
        copied = source(publication["operation_result"], "publication.operation_result")
        if (not same(result, copied) or result.get("format") != "darsia-series-setup-result-v1"
                or result.get("pipeline") != "series-setup" or result.get("operation") != "build"
                or result.get("complete") is not True or result.get("worker_references_closed") is not True
                or result.get("run") != config["run"] or "files" in result
                or job.get("pipeline") != "series-setup" or task.get("pipeline") != "series-setup"
                or task.get("operation") != "build" or task.get("run") != result["run"]
                or done.get("format") != "darsia-alias-done-v1" or done.get("run") != result["run"]):
            raise ValueError("Current Build requires its original operation receipt and declared Setup publication")
        # These checks join declared metadata. setup_terminal still authenticates
        # the complete original owner/index/generation chain after startup.
        shared = {row["relative_path"]: row["descriptor"] for row in publication["readset"]}

        def published(reference, use):
            bound = shared.get(reference.get("relative_path")) if isinstance(reference, dict) else None
            if (bound is None or bound["sha256"] != reference.get("sha256")
                    or bound["size_bytes"] != reference.get("bytes", reference.get("size_bytes"))):
                raise ValueError("Current Build publication readset omits its original " + use)
            return source(bound, "publication." + use)

        completed = published(done["completed"], "completed")
        index = published(done["products_index"], "products_index")
        if (index.get("format") != "darsia-products-index-v1"
                or index.get("run") != result["run"] or index.get("attempt_id") != done.get("attempt_id")
                or completed.get("attempt_id") != done.get("attempt_id")
                or any(not same(completed.get(key), value) for key, value in result.items())
                or not same(completed.get("files", {}).get("operation_result.json"), binding["operation_result"])):
            raise ValueError("Current Build original receipt differs from its shared completion/index")
        products = index.get("products")
        if (not isinstance(products, list) or len(products) != 2
                or {row.get("product_id") for row in products} != {"completed.json", "operation_result.json"}):
            raise ValueError("Current Build must retain the original two-product Setup publication")
        index_relative = done["products_index"]["relative_path"]
        parent = index_relative.rsplit("/", 1)[0]
        for product in products:
            copy = product.get("metadata_copy") or {}
            relative = parent + "/" + str(copy.get("relative_path", ""))
            bound = shared.get(relative)
            if (bound is None or bound["sha256"] != product.get("sha256")
                    or bound["size_bytes"] != product.get("bytes")
                    or copy.get("sha256") != bound["sha256"] or copy.get("bytes") != bound["size_bytes"]):
                raise ValueError("Current Build readset omits an original indexed metadata copy")
            if product["product_id"] == "operation_result.json" and not same(bound, publication["operation_result"]):
                raise ValueError("Current Build shared operation receipt is not the original indexed copy")

        plan = source(result["setup_plan"], "setup_plan")
        builds = [row for row in plan.get("build_plans", []) if row.get("run_id") == config["run"]]
        if plan.get("format") != "darsia-series-setup-plan-v1" or len(builds) != 1:
            raise ValueError("Current Build has no unique original Setup plan run")
        build = builds[0]
        request = source(result["raw_build_request"], "raw_build_request")
        if (request.get("format") != "darsia-raw-rig-build-request-v1" or request.get("run_id") != config["run"]
                or not same({key: value for key, value in request.items() if key != "configuration_files"},
                            build["request_template"])
                or not same(request["configuration_files"], result["configuration_files"])
                or [row["path"] for row in request["configuration_files"]] != build["configuration_paths"]
                or not same(request["profile"], build["native_parameters"])
                or not same(build["native_parameters"], config["native"]["configuration"])
                or not same(request["series_context"]["manifest"], plan["series_manifest"])):
            raise ValueError("Current Build request/configuration differs from its actual Setup plan")
        adopted = source(plan["adopted_revision"], "adopted_revision")
        if "metric_model" in binding:
            metric = source(binding["metric_model"], "metric_model")
            for key in ("source_model", "current_edges", "baseline_configuration", "model_photograph"):
                source(metric[key], "metric." + key, "rig_model", metadata=False)
            if metric.get("baseline_to_model_registration") is not None:
                source(metric["baseline_to_model_registration"], "metric.baseline_to_model_registration",
                       "rig_model", metadata=False)
            table = adopted["artifacts"]["source_mode_files"]["facies_properties"]
            source(table, "facies_properties", "rig_model", metadata=False)
            preparation = (adopted["configuration"].get("painted_adoption_table") or {}).get("preparation")
            if preparation is not None:
                prepared = source(preparation, "chart.preparation")
                chart = prepared.get("chart_transfers", {}).get(config["run"])
                if chart is not None:
                    artifact = chart["artifact"]
                    relative = Path(artifact["path"])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("Chart artifact escapes its original preparation")
                    bound = dict(artifact, path=str(Path(preparation["path"]).parent / relative))
                    source(bound, "chart.arrays", "rig_model", metadata=False)
        source(build["native_parameters"], "native_configuration")
        source(request["baseline"], "baseline_raw", "image", metadata=False)
        if "local_build_reuse" in binding:
            self.local_build_reuse(binding["local_build_reuse"],
                operation_result=binding["operation_result"], request=request, config=config)
        else:
            source(request["series_context"]["manifest"], "series_manifest", metadata=False)
        source(request["shape_basis_certificate"], "shape_basis_certificate", "rig_model", metadata=False)
        for index_, bound in enumerate(request["configuration_files"]):
            source(bound, "configuration_files." + str(index_), metadata=False)
        from ..protocol_content import current_source
        for index_, row in enumerate(request["protocol_inputs"]):
            # A corrected injection position keeps the Build (protocol_content).
            source(current_source(row), "protocol." + str(index_), "protocol", metadata=False)
        certificate = source(result["baseline_renderer"], "baseline_renderer", "rig_model", original=True)
        if (not same(certificate.get("outputs"), result["produced_files"])
                or not same(certificate.get("inputs", {}).get("operation_request"), result["raw_build_request"])):
            raise ValueError("Current Build certificate differs from its original request/outputs")
        outputs = certificate["outputs"]
        if not isinstance(outputs, list) or not outputs:
            raise ValueError("Original owned Build outputs are required")
        for index_, bound in enumerate(outputs):
            source(bound, "outputs." + str(index_), "rig_model", metadata=False, original=True)
        ownership = certificate["output_ownership"]
        for key in ("owner", "setup_started", "save_started", "save_completed"):
            if ownership[key] not in outputs:
                raise ValueError("Current Build ownership record is outside its declared original outputs")
            source(ownership[key], "ownership." + key, "rig_model", original=True)
        # Reuse the original selected native dependency catalogue. Its callback
        # admits a named role BEFORE each original metadata read; binary leaves
        # are not decoded or SHA-swept during preparation.
        native_contract = request.get("native_product", request.get("analysis_exclusions"))
        exclusion_config = adopted.get("configuration", {})
        if not isinstance(exclusion_config, dict):
            raise ValueError("Current Build adopted configuration must retain its original mapping")
        if "analysis_exclusions" in exclusion_config:
            self.current_build_exclusions(exclusion_config, native_contract, source)
        elif native_contract is not None:
            from ..rig_raw_build import _native_request_dependencies
            declared = {}

            class NativeDependencies:
                def path(_self, bound):
                    identity = (_key(bound["path"]), bound["sha256"], bound["size_bytes"])
                    if identity not in declared:
                        use = "native_dependency." + str(len(declared))
                        source(bound, use, "rig_model", metadata=False, original=True)
                        declared[identity] = use
                    return Path(bound["path"])

            _native_request_dependencies(request, reader=NativeDependencies())

        bridge = source(binding["bridge"], "bridge")
        if (bridge.get("format") != "darsia-build-native-bridge-v1"
                or not same(bridge.get("operation_result"), binding["operation_result"])
                or not same(bridge.get("native_configuration"), config["native"]["configuration"])):
            raise ValueError("Current Build bridge belongs to another original Build or native configuration")
        for key in ("photometry_source", "raw_source_binding"):
            raw = source(bridge[key], "bridge." + key)
            source(raw["baseline_raw"], "bridge." + key + ".baseline_raw", "image", metadata=False)
            for model in ("fit", "reference"):
                for field in ("data", "sidecar"):
                    source(raw["models"][model][field], "bridge." + key + "." + model + "." + field,
                           "calibration", metadata=field == "sidecar")
            if raw["models"]["old_field"] is not None:
                source(raw["models"]["old_field"], "bridge." + key + ".old_field", "calibration", metadata=False)
            for index_, bound in enumerate(raw["shape_assets"]):
                source(bound, "bridge." + key + ".shape." + str(index_), "rig_model", metadata=False)
        source(bridge["producer"]["module"], "bridge.producer", metadata=False)
        geometry_inputs = {"old_geometry", "old_raw_audit", "old_native_configuration", "native_configuration",
            "phase_configuration", "new_bridge_frame", "bridge", "native_frame", "old_raw_binding",
            "raw_source_binding", "old_acquisition_collection", "old_acquisition_configuration",
            "hardware_pose", "top_photo_evidence", "top_curve"}
        context_inputs = geometry_inputs - {"top_curve"} | {"geometry", "geometry_producer"}
        for row, spec in zip(binding["frames"], config["frames"]):
            use = "frame." + row["case_id"]
            receipt = source(row["bridge_frame"], use + ".bridge_frame")
            if (receipt.get("format") != "darsia-build-native-frame-v1"
                    or not same(receipt.get("bridge"), binding["bridge"])
                    or not same(receipt.get("native_frame"), spec["native_task"]["geometry"])
                    or not same(receipt.get("raw_source_binding"), bridge["raw_source_binding"])
                    or not same(receipt["raw_source_binding"], spec["native_task"]["processor"]["raw_binding"])):
                raise ValueError("Current Build bridge frame differs from the queued native task")
            native = source(receipt["native_frame"], use + ".native_frame")
            if native.get("run") != config["run"] or not same(native.get("acquisition"), row["raw_acquisition"]):
                raise ValueError("Current Build frame belongs to another RAW acquisition")
            source(receipt["products"], use + ".native_products", "rig_model", metadata=False)
            geometry = source(row["geometry"], use + ".geometry", "rig_model")
            transport = geometry["coordinate_transport"]
            declared = transport["inputs"]
            if (not isinstance(declared, dict) or set(declared) != geometry_inputs
                    or not same(declared["new_bridge_frame"], row["bridge_frame"])
                    or not same(declared["native_configuration"], config["native"]["configuration"])
                    or not same(declared["phase_configuration"], config["phase"]["configuration"])):
                raise ValueError("Current Build geometry requires its complete original named coordinate graph")
            for key in sorted(declared):
                source(declared[key], use + ".geometry_inputs." + key, "rig_model", metadata=key != "top_curve")
            source(transport["producer"], use + ".geometry_producer", metadata=False)
            source(geometry["frame_photo_top_keep"], use + ".photo_top", "rig_model", metadata=False)
            expected_context = {key: value for key, value in declared.items() if key != "top_curve"}
            expected_context.update(geometry=row["geometry"], geometry_producer=transport["producer"])
            for stage, extra in (("area", {"measurement", "physical_scale_plan", "physical_scale_receipt",
                    "measured_frame_receipt", "historical_frame_receipt", "parent_seal", "parent_registration"}),
                    ("static", {"old_configuration", "old_phase_configuration", "old_components", "old_native_frame", "producer"})):
                proof = source(row[stage + "_provenance"], use + "." + stage + "_provenance")
                inputs = proof["inputs"]
                if (not isinstance(inputs, dict) or set(inputs) != context_inputs | extra
                        or any(not same(inputs.get(key), value) for key, value in expected_context.items())):
                    raise ValueError("Current Build " + stage + " lost its original named source graph")
                for key in sorted(inputs):
                    source(inputs[key], use + "." + stage + "_inputs." + key, "rig_model",
                           metadata=key not in {"geometry_producer", "producer", "old_components"})
                if stage == "area":
                    if not same(proof["area"], spec["phase_roles"]["area"]):
                        raise ValueError("Current Build area proof differs from the actual queued area roles")
                    source(proof["producer"], use + ".area_producer", metadata=False)
                elif (proof.get("format") != "darsia-static-component-coordinate-data-v1"
                        or proof.get("run") != config["run"] or proof.get("case_id") != row["case_id"]
                        or not same(proof.get("acquisition"), row["raw_acquisition"])
                        or not same(proof.get("components"), spec["phase_roles"]["static"]["components"])
                        or proof.get("physical_qualification") is not False):
                    raise ValueError("Current Build static proof differs from the original queued component DATA")
        return binding

    def current_build_exclusions(self, configuration, native_contract, source):
        """Expand the original optional readers under metadata-only admission.

        This scope supplies declared binary identities after root/stat admission,
        never a binary verification result. Only explicitly consumed JSON and
        its review display's exact series manifest may be opened. Original
        worker readers still hash/decode every scientific
        input and replay the approvals; no declaration grants their acceptance.
        """
        from ..preset_wasserstein_inputs import _Scope
        from ..painted_analysis_exclusions import exclusion_sources
        from ..painted_target_integration import target_sources
        from ..rig_raw_build import _native_request_dependencies
        same = lambda left, right: json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
            right, sort_keys=True, allow_nan=False)
        preparation_bound = (configuration.get("painted_adoption_table") or {}).get("preparation")
        if (not isinstance(preparation_bound, dict) or not isinstance(native_contract, dict)
                or not same(native_contract.get("preparation"), preparation_bound)):
            raise ValueError("Current Build analysis exclusions require their original preparation and native request closure")
        preparation = source(preparation_bound, "exclusions.preparation", original=True)
        if (not isinstance(configuration.get("analysis_exclusions"), dict)
                or not same(configuration["analysis_exclusions"], preparation.get("analysis_exclusions"))):
            raise ValueError("Current Build analysis exclusions differ from their original preparation")
        catalog = self

        class Dependencies(_Scope):
            def __init__(_self):
                _self.declared = {}
                _self.display_series = {}

            def descriptor(_self, path, expected=None):
                key = _key(path)
                if expected is not None:
                    if (not isinstance(expected, dict)
                            or not {"path", "sha256", "size_bytes"} <= set(expected)
                            or _key(expected["path"]) != key):
                        raise ValueError("Current Build exclusion source needs its exact original descriptor")
                    bound = {name: expected[name] for name in ("path", "sha256", "size_bytes")}
                    previous = _self.declared.get(key)
                    if previous is not None and any(previous[name] != bound[name] for name in ("sha256", "size_bytes")):
                        raise ValueError("Current Build exclusion source changed between original consumers")
                    if previous is None:
                        if len(_self.declared) >= 8192:
                            raise ValueError("Current Build exclusion sources exceed the original bounded file count")
                        source(bound, "exclusions.source." + str(len(_self.declared)),
                               "rig_model", metadata=False, original=True)
                        _self.declared[key] = bound
                if key not in _self.declared:
                    raise ValueError("Current Build exclusion reader requested an undeclared original source: " + str(path))
                return deepcopy(_self.declared[key])

            def path(_self, bound):
                _self.descriptor(bound["path"], bound)
                return Path(bound["path"])

            def read(_self, path, *, kind="bytes", sheet=None):
                if kind != "bytes" or sheet is not None:
                    raise ValueError("Current Build exclusion preparation cannot read tables or scientific payloads")
                bound = _self.descriptor(path)
                if _key(path) not in _self.display_series:
                    if Path(path).suffix.casefold() != ".json":
                        raise ValueError("Current Build exclusion preparation may open only named JSON or its exact display series metadata")
                    value = source(bound, "exclusions.metadata." + str(list(_self.declared).index(_key(path))),
                                   "rig_model", original=True)
                    # The original review reader stats its path before _checked.
                    # Admit that exact named review first; never invent its hash.
                    review = value.get("review") or (value.get("basis") or {}).get("review")
                    if isinstance(review, dict) and {"path", "sha256", "size_bytes"} <= set(review):
                        _self.descriptor(review["path"], review)
                    display = value.get("display_context")
                    if (value.get("format") == "darsia-painted-gap-review-v1"
                            and isinstance(display, dict)
                            and display.get("format") == "darsia-painted-review-display-context-v1"):
                        manifest = display.get("series_manifest")
                        if (not isinstance(manifest, dict) or not isinstance(manifest.get("path"), str)
                                or Path(manifest["path"]).name != "series.toml"):
                            raise ValueError("Current Build review display requires its exact original series.toml")
                        admitted = _self.descriptor(manifest["path"], manifest)
                        key = _key(admitted["path"])
                        if key not in _self.display_series:
                            source(admitted, "exclusions.display_series." + str(len(_self.display_series)),
                                   "rig_model", metadata=False, original=True)
                            _self.display_series[key] = admitted
                if bound["size_bytes"] > MAX_METADATA_BYTES:
                    raise ValueError("Current Build exclusion metadata exceeds its bounded read limit")
                raw = catalog.metadata_bytes(bound["path"])
                if (not isinstance(raw, bytes) or len(raw) != bound["size_bytes"]
                        or hashlib.sha256(raw).hexdigest() != bound["sha256"]):
                    raise ValueError("Current Build exclusion metadata changed during its original read")
                return raw

            def stat(_self, path):
                bound = _self.descriptor(path)
                return catalog.file_stat(bound["path"])

            def exists(_self, path, kind="exists"):
                raise ValueError("Current Build exclusion preparation cannot probe undeclared filesystem state")

            def inventory(_self, path, pattern):
                raise ValueError("Current Build exclusion preparation cannot enumerate filesystem inputs")

            def role(_self, role):
                raise ValueError("Current Build exclusion preparation accepts only its named JSON/source roles")

        scope = Dependencies()
        with scope.enter():
            scope.path(preparation_bound)
            # Selected native closure includes source_map, base/native products,
            # source/target models, renderer archives and saved source maps.
            _native_request_dependencies({"native_product": native_contract}, reader=scope)
            for name, reader in (("approval", exclusion_sources), ("target", target_sources)):
                for index, bound in enumerate(reader(preparation)):
                    scope.path(bound)
                    source(bound, "exclusions." + name + "." + str(index),
                           "rig_model", metadata=False, original=True)

    def collect(self, config_descriptor, config):
        self.add(config_descriptor, "configuration", "run.configuration")
        if config.get("format") == "darsia-series-setup-operation-v1":
            from .series_setup_inputs import catalogue_setup_inputs
            self.setup_controls = catalogue_setup_inputs(self, config)
        elif config.get("format") == "darsia-preset-wasserstein-operation-v1":
            from ..preset_wasserstein_inputs import catalogue_preset_wasserstein_inputs
            catalogue_preset_wasserstein_inputs(self, config)
        elif config.get("format") == "darsia-n1-run-v1":
            self.mass(config, config["frames"], standalone=True)
        elif config.get("format") == "darsia-native-phase-n1-run-v1":
            if "current_build_binding" in config:
                self.current_build(config)
            phase, native = self.phase(config)
            specs = [dict(frame["mass_spec"], case_id=frame["case_id"]) for frame in config["frames"]]
            self.mass(config["mass"]["template"], specs)
            for frame in config["frames"]:
                use = "frame." + frame["case_id"]
                task = frame["native_task"]
                geometry = self.doc(task["geometry"], "rig_model", use + ".native_geometry")
                rig = self.doc(frame["geometry"], "rig_model", use + ".rig_geometry")
                self.add(rig["frame_photo_top_keep"], "rig_model", use + ".photo_top")
                self.native_task(task, geometry, native, use + ".native")
                roles = frame["phase_roles"]
                if roles.get("native_row") is not None:
                    row = self.doc(roles["native_row"], "configuration", use + ".phase.native_row")
                    self.original_native_row(row, phase["parameters"]["inputs"], use + ".phase")
                    probability = self.doc(roles["probability_receipt"], "model", use + ".phase.probability_receipt")
                    self.add(probability["files"][phase["parameters"]["inputs"]["mask_file"]], "rig_model", use + ".phase.photo_mask")
                else:
                    self.add(roles["photo_sand_reference"], "rig_model", use + ".phase.photo_sand_reference")
                if roles["area"]["role"] == "measured_parent":
                    for key in ("historical", "measured"):
                        self.add(roles["area"][key], "rig_model", use + ".area." + key)
                self.add(roles["static"]["components"], "rig_model", use + ".static_components")
                if roles.get("retained_source_cache") is not None:
                    self.retained(roles["retained_source_cache"], phase["parameters"]["retained"], use + ".retained")
        elif config.get("format") == "darsia-native-phase-statistics-run-v1":
            from .staged_data import validate_phase1_run
            validate_phase1_run(config)
            if "current_build_binding" in config:
                self.current_build(config)
            phase, native = self.phase(config)
            specs = [dict(frame["mass_spec"], case_id=frame["case_id"]) for frame in config["frames"]]
            self.statistics(config["statistics"], specs)
            for frame in config["frames"]:
                use = "frame." + frame["case_id"]
                task = frame["native_task"]
                geometry = self.doc(task["geometry"], "rig_model", use + ".native_geometry")
                rig = self.doc(frame["geometry"], "rig_model", use + ".rig_geometry")
                self.add(rig["frame_photo_top_keep"], "rig_model", use + ".photo_top")
                self.native_task(task, geometry, native, use + ".native")
                roles = frame["phase_roles"]
                if roles.get("native_row") is not None:
                    row = self.doc(roles["native_row"], "configuration", use + ".phase.native_row")
                    self.original_native_row(row, phase["parameters"]["inputs"], use + ".phase")
                    probability = self.doc(roles["probability_receipt"], "model", use + ".phase.probability_receipt")
                    self.add(probability["files"][phase["parameters"]["inputs"]["mask_file"]], "rig_model", use + ".phase.photo_mask")
                else:
                    self.add(roles["photo_sand_reference"], "rig_model", use + ".phase.photo_sand_reference")
                if roles["area"]["role"] == "measured_parent":
                    for key in ("historical", "measured"):
                        self.add(roles["area"][key], "rig_model", use + ".area." + key)
                self.add(roles["static"]["components"], "rig_model", use + ".static_components")
                if roles.get("retained_source_cache") is not None:
                    self.retained(roles["retained_source_cache"], phase["parameters"]["retained"], use + ".retained")
            for case, request in config.get("photographs", {}).items():
                for field in ("native_geometry", "geometry_certificate", "current_geometry"):
                    self.doc(request[field], "rig_model", "frame." + case + ".photo." + field)
                self.add(request["saved_native_review"], "image", "frame." + case + ".photo.saved_native_review")
        elif config.get("format") == "darsia-n1-refit-plan-v1":
            # A refit consumes the baseline calibration and every fit-run statistics file.
            from .refit_runner import validate_plan
            validate_plan(config)
            self.doc(config["baseline"]["calibration"], "calibration", "refit.baseline_calibration")
            for run in sorted(config["statistics"]):
                for frame in config["statistics"][run]:
                    self.add(frame["statistics"], "calibration",
                             "refit." + run + ".frame." + str(frame["frame_index"]) + ".statistics")
            if "common_phase1" in config:
                from .common_calibration import catalogue_common_phase1
                catalogue_common_phase1(self, config["common_phase1"], use="refit.common_phase1")
        else:
            raise ValueError("Unsupported role catalogue run format: " + str(config.get("format")))
        result = {"format": CATALOG_FORMAT, "complete": True, "run": config["run"],
                "configuration": self.add(config_descriptor, "configuration", "run.configuration"),
                "inputs": [self.entries[key] for key in sorted(self.entries)],
                "payload_hashes_verified": False, "numerical_calls_started": False,
                "runtime_support": "J3 pending; no alias task may be claimed yet"}
        if self.native_exact_inputs:
            result["native_exact_inputs"] = [self.native_exact_inputs[key] for key in sorted(self.native_exact_inputs)]
        return result


def validate_catalog(value):
    if value.get("format") != CATALOG_FORMAT or value.get("complete") is not True:
        raise ValueError("Complete explicit role input catalogue required")
    validate_input_reference(value["configuration"])
    if not isinstance(value.get("inputs"), list) or not value["inputs"]:
        raise ValueError("Role input catalogue is empty")
    seen, consumers = set(), {}
    for row in value["inputs"]:
        validate_input_reference(row["reference"])
        identity = tuple((key,row["reference"][key]) for key in sorted(row["reference"]))
        if identity in seen:
            raise ValueError("Duplicate portable input catalogue reference")
        seen.add(identity)
        if not isinstance(row.get("uses"), list) or not row["uses"] or any(not isinstance(use, str) or not use for use in row["uses"]):
            raise ValueError("Input catalogue needs named original consumers")
        if len(row["uses"]) != len(set(row["uses"])):
            raise ValueError("Duplicate input catalogue consumer")
        for use in row["uses"]:
            if use in consumers and consumers[use] != row["reference"]:
                raise ValueError("Ambiguous input consumer maps to different references: " + use)
            consumers[use] = row["reference"]
    if value["configuration"] not in [row["reference"] for row in value["inputs"]]:
        raise ValueError("Catalogue omits its bound configuration input")
    order = [catalog_order_key(row["reference"]) for row in value["inputs"]]
    if order != sorted(order):
        raise ValueError("Role input catalogue is not in its canonical order")
    exact = value.get("native_exact_inputs", [])
    if not isinstance(exact, list):
        raise ValueError("Native exact inputs require explicit file bindings")
    declared = {}
    from ..store_map import _path
    for row in exact:
        if not isinstance(row, dict) or set(row) != {"reference", "descriptor"}:
            raise ValueError("Native exact input binding has unsupported fields")
        reference, pin = row["reference"], row["descriptor"]
        validate_input_reference(reference)
        if (not isinstance(pin, dict) or set(pin) != {"path", "sha256", "size_bytes"}
                or reference["root_alias"] != "native_exact" or reference["role"] != "rig_model"
                or pin["sha256"] != reference["sha256"] or type(pin["size_bytes"]) is not int
                or pin["size_bytes"] != reference["bytes"]):
            raise ValueError("Native exact input differs from its file SHA/size/role")
        _path(pin["path"])
        relative = pin["sha256"] + "-" + hashlib.sha256(_key(pin["path"]).encode()).hexdigest()[:16] + ".input"
        if reference["relative_path"] != relative or relative in declared:
            raise ValueError("Native exact input path binding is ambiguous")
        declared[relative] = reference
    used = set()
    for row in value["inputs"]:
        reference = row["reference"]
        if reference["root_alias"] == "native_exact":
            if (declared.get(reference["relative_path"]) != reference
                    or any(not use.startswith("setup.raw.native.") for use in row["uses"])):
                raise ValueError("Native exact input is outside its consumed native descriptor role")
            used.add(reference["relative_path"])
    if used != set(declared):
        raise ValueError("Native exact input binding has an unused file")
    return value


def catalog_order_key(reference):
    """Return the key InputCatalog sorts its inputs by: alias, path, SHA-256, role.

    A replay rebuilds the catalogue in this order and compares whole lists, so
    a catalogue in any other order can never run. Refuse it when it is made.
    """
    return (reference["root_alias"], reference["relative_path"], reference["sha256"], reference["role"])
