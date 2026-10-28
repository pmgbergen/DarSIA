"""Explicit two-phase DATA validation and immutable configuration sealing.

PRIVATE candidate: no native processor, classifier, fit, mass evaluator, queue
or stage-completion writer is called here. Public worker/catalogue admission
for the new phase-one format remains a separate required integration.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import ntpath
from pathlib import Path, PureWindowsPath
import re

from .job_queue import descriptor, name, write_immutable

PHASE1_FORMAT = "darsia-native-phase-statistics-run-v1"
STATISTICS_FORMAT = "darsia-native-n1-statistics-template-v1"
COMPLETION_FORMAT = "darsia-phase-statistics-completion-v1"
FRAME_FORMAT = "darsia-phase-statistics-frame-v1"
BEFORE_FORMAT = "darsia-phase-before-rules-v1"


def _keys(value, required, optional=(), *, where):
    if not isinstance(value, dict):
        raise ValueError(where + " must be explicit JSON DATA")
    missing, unknown = set(required) - value.keys(), value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(where + " fields differ: missing=" + repr(sorted(missing))
                         + "; unknown=" + repr(sorted(unknown)))
    return value


def _finite(value, *, where, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (positive and value <= 0)):
        raise ValueError(where + " must be an explicit finite number")
    return value


def _sha(value, *, where):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError(where + " needs a full lowercase SHA-256")


def _bound(value, *, where, array=False):
    _keys(value, {"path", "sha256", "size_bytes"}, {"array_key"} if array else (), where=where)
    if not isinstance(value["path"], str) or not value["path"]:
        raise ValueError(where + " needs its explicit locator")
    _sha(value["sha256"], where=where)
    if type(value["size_bytes"]) is not int or value["size_bytes"] < 0:
        raise ValueError(where + " needs its exact byte count")
    if "array_key" in value and (not isinstance(value["array_key"], str) or not value["array_key"]):
        raise ValueError(where + " needs an explicit array key")
    return value


def _data(value):
    from .series import _no_executable_configuration
    _no_executable_configuration(value)
    def exact_json(item):
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise ValueError("JSON DATA keys must be literal strings")
            for child in item.values(): exact_json(child)
        elif isinstance(item, list):
            for child in item: exact_json(child)
        elif item is not None and type(item) not in {str, int, float, bool}:
            raise ValueError("DATA must retain exact JSON types without coercion")
    exact_json(value)
    # Reject leaked arrays, unsupported scalar objects and NaN/Infinity before IO.
    json.dumps(value, ensure_ascii=False, allow_nan=False)


def _same_json(first, second):
    """Identity of original JSON DATA, including bool/int/float distinctions."""
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            _same_json(first[key], second[key]) for key in first)
    if isinstance(first, list):
        return len(first) == len(second) and all(
            _same_json(left, right) for left, right in zip(first, second))
    return first == second


def _statistics_template(value, *, require_optics=True):
    required = {"format", "shared", "physical_correction", "recipe", "dimensions_m",
                "summary", "bindings", "geometry_fields"}
    _keys(value, required, {"boxes", "atmosphere_series", "dose", "contracts", "provenance", "opencl_acceleration"},
          where="statistics template")
    if "opencl_acceleration" in value:
        from ..mass.mass_opencl import configuration
        configuration(value["opencl_acceleration"])
    if value["format"] != STATISTICS_FORMAT:
        raise ValueError("Explicit native statistics template format required")
    from ..mass.n1_pipeline import SHARED_REQUIRED, _geometry_fields
    shared = value["shared"]
    _keys(shared, set(SHARED_REQUIRED) - {"physical_reference"},
          {"reference_matrix", "legacy_dimensions_m", "facies_properties",
           "physical_coordinate_document", "physical_coordinate_provenance", "physical_reference"}, where="statistics shared roles")
    if "physical_reference" in shared and "reference_matrix" not in shared and "physical_coordinate_provenance" not in shared:
        raise ValueError("Statistics need the original explicit physical coordinate binding")
    for key in ("physical_reference", "floor_curve"):
        if key in shared:
            _bound(shared[key], where="statistics shared " + key, array=True)
    if isinstance(shared["depth_points"], dict):
        _bound(shared["depth_points"], where="statistics depth points")
    _bound(value["physical_correction"], where="statistics physical correction")
    _geometry_fields(value)
    summary = value["summary"]
    _keys(summary, {"baseline_gain", "intervals", "max_column_distance_m",
                    "max_depth_m", "max_porosity"}, {"contract", "anchor_target_h"}, where="historical statistics summary")
    for field in ("baseline_gain", "anchor_target_h", "max_column_distance_m", "max_depth_m", "max_porosity"):
        if field == "anchor_target_h" and field not in summary:
            continue
        _finite(summary[field], where="summary " + field, positive=True)
    if summary["intervals"] != 32768 or type(summary["intervals"]) is not int:
        raise ValueError("Original 32768-interval summary definition required")
    if summary["max_column_distance_m"] != .020:
        raise ValueError("Original 20 mm summary column definition required")
    if summary["max_porosity"] > 1:
        raise ValueError("Summary porosity bound exceeds one")
    bindings = value["bindings"]
    if not isinstance(bindings, dict):
        raise ValueError("Statistics method bindings must be explicit")
    for field in ("round4_spec_sha256", "amendment_sha256"):
        _sha(bindings.get(field), where="statistics method binding " + field)
    _keys(shared["clearwater"], {"rgb", "valid", "H_reference_to_grid"}, where="clear-water roles")
    matrix = shared["clearwater"]["H_reference_to_grid"]
    if (not isinstance(matrix, list) or len(matrix) != 3
            or any(not isinstance(row, list) or len(row) != 3 for row in matrix)):
        raise ValueError("Original clear-water H_reference_to_grid must be an explicit 3x3 matrix")
    for row in matrix:
        for matrix_value in row:
            _finite(matrix_value, where="clear-water H_reference_to_grid")
    for field in ("rgb", "valid"):
        _bound(shared["clearwater"][field], where="clear-water " + field, array=True)
    if shared["optics"] is None and not require_optics:
        return  # Only the internal fresh-optics phase pass may defer this role.
    optics = _keys(shared["optics"], {"metadata", "color_paths", "signals", "flash"}, where="saved optical roles")
    for field in ("metadata", "flash"):
        _bound(optics[field], where="saved optics " + field)
    for group in ("color_paths", "signals"):
        if not isinstance(optics[group], dict) or not optics[group]:
            raise ValueError("Explicit saved optical role group required: " + group)
        for label, bound in optics[group].items():
            if not isinstance(label, str) or not label.isdecimal():
                raise ValueError("Explicit optical label key required")
            _bound(bound, where="saved optics " + group + "." + label)
    if "boxes" not in value:
        return
    if isinstance(value["boxes"], dict) and "path" in value["boxes"]:
        _bound(value["boxes"], where="original box definitions")
    else:
        _keys(value["boxes"], {"box1", "box2"}, where="original box roles")


def validate_phase1_run(config, *, require_optics=True):
    """Validate new explicit DATA only; read no referenced payload or fitted K."""
    _data(config)
    _keys(config, {"format", "run", "precision", "frames", "native", "phase", "statistics",
                   "history_policy", "selection_index_field", "ordered_case_ids", "storage"},
          {"data_aliases", "provenance", "photographs", "gas_probability_storage", "current_build_binding"},
          where="phase-one configuration")
    if "gas_probability_storage" in config:
        from ..phase.probability_storage import parse_request
        parse_request(config)
    if config["format"] != PHASE1_FORMAT or config["precision"] not in {"float32", "float64"}:
        raise ValueError("Explicit phase-one format and precision required")
    name(config["run"])
    _keys(config["native"], {"configuration"}, {"source_certificate_fields"}, where="native roles")
    if "source_certificate_fields" in config["native"]:
        certificate = _keys(config["native"]["source_certificate_fields"], {"renderer", "photometry"}, where="source certificate roles")
        if certificate != {"renderer": "renderer_certificate", "photometry": "v4r5_native_info"}:
            raise ValueError("Original SourceProcessor certificate field binding differs")
    _bound(config["native"]["configuration"], where="native configuration")
    _keys(config["phase"], {"configuration", "reference", "initial_previous"}, where="phase roles")
    _bound(config["phase"]["configuration"], where="phase configuration")
    reference = _keys(config["phase"]["reference"], {"full", "native_row", "source_sha256"}, where="reference acquisition roles")
    _sha(reference["source_sha256"], where="phase reference acquisition")
    _bound(reference["native_row"], where="phase reference row")
    initial = _keys(config["phase"]["initial_previous"], {"current_h", "full", "native_row", "source"}, where="initial acquisition roles")
    _bound(initial["native_row"], where="phase initial row")
    _statistics_template(config["statistics"], require_optics=require_optics)
    from ..storage_outputs import parse_request
    _keys(config["storage"], {"format", "resolution_mm", "gas_score_encoding"}, where="phase-one native evidence request")
    request = parse_request(config, config["precision"])
    if request["resolution_mm"] != "native":
        raise ValueError("Phase one explicitly retains native BEFORE evidence; final mass resolution is chosen later")
    if not isinstance(config["history_policy"], str) or not config["history_policy"]:
        raise ValueError("Explicit acquisition-history policy required")
    index_field = config["selection_index_field"]
    if not isinstance(index_field, str) or not index_field:
        raise ValueError("Explicit selected-frame index field required")
    frames = config["frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError("An explicit nonempty chronological selection is required")
    raw_mode = any(frame.get("native_task", {}).get("kind") == "raw_rig" for frame in frames)
    if raw_mode and any(frame.get("native_task", {}).get("kind") != "raw_rig" for frame in frames):
        raise ValueError("A RAW-only run cannot mix JPEG-positioned native processors")
    ids, digests, last = set(), set(), None
    predecessor = config["phase"]["initial_previous"]["source"]["source_sha256"]
    _sha(predecessor, where="initial predecessor")
    for spec in frames:
        _keys(spec, {"case_id", "frame_index", "elapsed_h", "source", "native_task",
                     "geometry", "phase_roles", "mass_spec"}, {"provenance"}, where="selected frame")
        identifier = name(spec["case_id"])
        index, elapsed, source = spec["frame_index"], spec["elapsed_h"], spec["source"]
        if not isinstance(source, dict) or not isinstance(source.get("source_binding"), dict) or not source["source_binding"]:
            raise ValueError("Original source metadata and binding must be explicit")
        _sha(source.get("source_sha256"), where="selected source")
        if identifier in ids or source["source_sha256"] in digests:
            raise ValueError("Duplicate selected case or source")
        if type(index) is not int or index < 0:
            raise ValueError("Exact selected frame index required")
        _finite(elapsed, where="selected elapsed_h")
        if (source.get("run") != config["run"] or source.get("selected") is not True
                or type(source.get(index_field)) is not int or source.get(index_field) != index
                or type(source.get("elapsed_h")) is not type(elapsed) or source.get("elapsed_h") != elapsed
                or source.get("previous_source_sha256") != predecessor):
            raise ValueError("Selected source/run/index/time/predecessor binding differs")
        if last is not None and not (index > last[0] and elapsed > last[1]):
            raise ValueError("Selection must retain strict chronological order")
        task = _keys(spec["native_task"], {"kind", "geometry", "processor"},
                     {"parent_frame_index", "parent_reference", "arguments", "reference_to_frame_h"}, where="native task roles")
        if task["kind"] not in {"rig", "source", "raw_rig"}:
            raise ValueError("Only the original fixed native processors are admitted")
        if task["kind"] == "raw_rig":
            _keys(task, {"kind", "geometry", "processor"}, where="RAW-only native task roles")
            _keys(task["processor"], {"raw_binding"}, where="RAW-only native processor roles")
            _bound(task["processor"]["raw_binding"], where="RAW-only source binding")
        elif task["kind"] == "rig":
            _keys(task, {"kind", "geometry", "processor", "parent_frame_index", "parent_reference"}, where="rig native task roles")
            _keys(task["processor"], {"native_manifest"}, where="rig native processor roles")
            _bound(task["processor"]["native_manifest"], where="rig native manifest")
            if "arguments" in task:
                raise ValueError("Rig native task has unexpected source-processor arguments")
        else:
            _keys(task["processor"], {"source_row", "calibration_manifest", "photo_aggregate",
                                      "enclosed_hole_policy", "display_mode", "isolated_invalid_policy"},
                  where="source native processor roles")
            for field in ("enclosed_hole_policy", "display_mode", "isolated_invalid_policy"):
                if not isinstance(task["processor"][field], str) or not task["processor"][field]:
                    raise ValueError("The original source processor policy must be explicit: " + field)
            for field in ("source_row", "calibration_manifest", "photo_aggregate"):
                _bound(task["processor"][field], where="source native " + field)
            if "source_certificate_fields" not in config["native"]:
                raise ValueError("The original source certificate renderer/photometry fields are required")
            arguments = _keys(task.get("arguments"), {"run", "frame_id", "source_manifest_sha256", "camera_jpeg",
                              "camera_arw", "review_jpeg", "content_shape_hw", "content_box_xywh"}, where="source native arguments")
            for field in ("camera_jpeg", "camera_arw", "review_jpeg"):
                _bound(arguments[field], where="source native " + field)
            if arguments["run"] != config["run"] or not isinstance(arguments["frame_id"], str):
                raise ValueError("Original source run/frame identity must be explicit")
            frame_id = arguments["frame_id"]
            if frame_id != "pre_injection_reference" and (not frame_id.startswith("frame_") or not frame_id[6:].isdecimal()):
                raise ValueError("Original SourceProcessor frame_id differs; no selection-index inference is allowed")
            _sha(arguments["source_manifest_sha256"], where="original source manifest")
            shape, box = arguments["content_shape_hw"], arguments["content_box_xywh"]
            if (not isinstance(shape, list) or len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape)
                    or not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                    or min(box[:2]) < 0 or min(box[2:]) < 1 or shape != [box[3], box[2]]):
                raise ValueError("Explicit original source content shape/box differs")
            matrix = task.get("reference_to_frame_h")
            if (not isinstance(matrix, list) or len(matrix) != 3
                    or any(not isinstance(row, list) or len(row) != 3 for row in matrix)):
                raise ValueError("Original source reference_to_frame_h must be explicit 3x3 DATA")
            for row in matrix:
                for matrix_value in row:
                    _finite(matrix_value, where="original source reference_to_frame_h")
        _bound(task["geometry"], where="native geometry")
        _bound(spec["geometry"], where="phase geometry")
        roles = _keys(spec["phase_roles"], {"area", "current_h", "reference_to_current_h", "static",
                      "retained_source_cache", "encoding_contract", "light_epoch_id", "inherited_spacing_m",
                      "probability_spacing_m", "placement_metadata"},
                      {"native_row", "probability_receipt", "photo_sand_reference", "photo_proof", "retained_frame_index"},
                      where="frame phase roles")
        if task["kind"] == "raw_rig" and (roles.get("retained_source_cache") is not None
                                          or roles.get("native_row") is not None):
            raise ValueError("RAW-only positioning requires fresh inference without JPEG-positioned parent roles")
        if task["kind"] == "raw_rig":
            if roles["encoding_contract"] is not None:
                raise ValueError("Fresh RAW inference cannot carry a historical probability encoding receipt")
            if source.get("studio_anchor") and type((source.get("parent_source") or {}).get("source_changed")) is not bool:
                raise ValueError("RAW Studio anchor requires its unchanged source-geometry flag")
        if roles["retained_source_cache"] is not None and (
                type(roles.get("retained_frame_index")) is not int or roles["retained_frame_index"] < 0):
            raise ValueError("Original retained cache frame index must be explicit; no selection-index inference")
        if roles.get("native_row") is not None:
            _bound(roles["native_row"], where="phase native row")
            _bound(roles.get("probability_receipt"), where="phase probability proof")
        else:
            _bound(roles.get("photo_sand_reference"), where="phase photo-sand reference", array=True)
            if not isinstance(roles.get("photo_proof"), dict):
                raise ValueError("Explicit phase photo-sand proof required")
        area = _keys(roles["area"], {"role", "factor"}, {"historical", "measured", "revision"}, where="physical-area roles")
        wanted = "measured_parent" if source.get("studio_anchor") else "selected"
        if area["role"] != wanted:
            raise ValueError("Historical and selected area routes differ")
        _finite(area["factor"], where="physical area factor", positive=True)
        if wanted == "measured_parent":
            for field in ("historical", "measured"):
                _bound(area.get(field), where="parent physical area " + field, array=True)
        else:
            if {"historical", "measured"} & area.keys():
                raise ValueError("Selected area cannot contain undeclared parent roles")
            if not isinstance(area.get("revision"), dict) or not area["revision"]:
                raise ValueError("Original selected-area revision provenance must be explicit")
        _keys(roles["static"], {"components", "source_ids", "unchanged_source"}, where="static phase roles")
        _bound(roles["static"]["components"], where="static phase components")
        _keys(spec["mass_spec"], {"atmosphere", "inner_polygon_native", "original_domain", "photo_top",
                                 "source_receipt", "nominal_injected_mass_g", "selected_target_h", "light_epoch"},
              {"no_outflow_evidence", "provenance"}, where="statistics frame roles")
        ids.add(identifier)
        digests.add(source["source_sha256"])
        predecessor, last = source["source_sha256"], (index, elapsed)
    if config["ordered_case_ids"] != [spec["case_id"] for spec in frames]:
        raise ValueError("Explicit ordered_case_ids differs from the selection")
    if "photographs" in config:
        photos = config["photographs"]
        if not isinstance(photos, dict) or not photos.keys() <= ids:
            raise ValueError("Optional photographs must name only the selected case IDs")
        for request in photos.values():
            _keys(request, {"native_geometry", "geometry_certificate", "current_geometry",
                            "saved_native_review", "geometry_producer_sha256"}, where="saved original photograph roles")
            for field in ("native_geometry", "geometry_certificate", "current_geometry", "saved_native_review"):
                _bound(request[field], where="original photograph " + field)
            _sha(request["geometry_producer_sha256"], where="original photograph geometry producer")
    if "data_aliases" in config:
        if not isinstance(config["data_aliases"], list):
            raise ValueError("Explicit DATA aliases must be a list")
        seen, physical = set(), {}
        for alias in config["data_aliases"]:
            _keys(alias, {"logical", "physical"}, where="DATA alias roles")
            for field in ("logical", "physical"):
                _bound(alias[field], where="DATA alias " + field)
            logical, actual = alias["logical"], alias["physical"]
            key = ntpath.normcase(ntpath.normpath(logical["path"]))
            target = ntpath.normcase(ntpath.normpath(actual["path"]))
            identity = (actual["sha256"], actual["size_bytes"])
            if (key in seen or logical["sha256"] != actual["sha256"]
                    or logical["size_bytes"] != actual["size_bytes"]
                    or target in physical and physical[target] != identity):
                raise ValueError("Duplicate or contradictory explicit DATA alias")
            seen.add(key)
            physical[target] = identity
    if "current_build_binding" in config:
        from darsia_cache.current_build_binding import checked_current_build_binding
        checked_current_build_binding(config["current_build_binding"], config=config)
    return config


def seal_phase1_configuration(destination, config):
    """Publish one new validated file with the existing exclusive writer."""
    validate_phase1_run(config)
    write_immutable(Path(destination), deepcopy(config))
    return descriptor(Path(destination))


def _same_bound(first, second):
    return all(first.get(key) == second.get(key) for key in ("path", "sha256", "size_bytes"))


def _owned(bound, inventory, *, where, array=False):
    _bound(bound, where=where, array=array)
    basename = PureWindowsPath(bound["path"]).name
    if basename not in inventory or not _same_bound(bound, inventory[basename]):
        raise ValueError(where + " is not the actual inventoried stage file")


def _before_metadata(evidence, target_shape):
    """Require the original encoder's source definition; decode no pixels."""
    if not isinstance(evidence, dict) or evidence.get("source_array") != "NativeCentreResult.g3_pixel":
        raise ValueError("BEFORE is not the original NativeCentreResult.g3_pixel")
    if evidence.get("state") == "pending" and evidence.get("reason") == "native_centre_not_computed":
        _keys(evidence, {"state", "reason", "source_array"}, where="missing native-centre evidence")
        return
    required = {"state", "source_array", "validity_array", "source_box_xywh", "source_shape_hw", "target_shape_hw",
                "finite_valid_pixels", "finite_min", "finite_max", "source_identity", "validity_identity",
                "existing_probability_encoding", "rescaled", "clipped"}
    if evidence.get("state") == "complete":
        required |= {"encoding", "quantization", "missing", "coarse_reduction"}
    elif evidence.get("state") == "pending" and evidence.get("reason") == "pre_rule_score_outside_probability_contract":
        required.add("reason")
    else:
        raise ValueError("BEFORE pending evidence lacks its original explicit reason")
    _keys(evidence, required, where="original BEFORE metadata")
    if (evidence["validity_array"] != "NativeCentreResult.flatfield_valid"
            or evidence["rescaled"] is not False or evidence["clipped"] is not False):
        raise ValueError("BEFORE support or no-rescaling contract differs")
    if (not isinstance(target_shape, list) or len(target_shape) != 2
            or any(type(v) is not int or v < 1 for v in target_shape)
            or evidence["target_shape_hw"] != target_shape):
        raise ValueError("BEFORE target grid differs from its actual native canvas")
    box = evidence["source_box_xywh"]
    if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
            or min(box[:2]) < 0 or min(box[2:]) < 1 or box[0] + box[2] > target_shape[1]
            or box[1] + box[3] > target_shape[0] or evidence["source_shape_hw"] != [box[3], box[2]]):
        raise ValueError("BEFORE source box/grid differs")
    count = evidence["finite_valid_pixels"]
    if type(count) is not int or not 0 <= count <= box[2] * box[3]:
        raise ValueError("BEFORE finite support count differs")
    for field, dtypes in (("source_identity", {"<f4", ">f4"}), ("validity_identity", {"|b1"})):
        bound = _keys(evidence[field], {"sha256", "dtype", "shape"}, where="BEFORE " + field)
        _sha(bound["sha256"], where="BEFORE " + field)
        if bound["dtype"] not in dtypes or bound["shape"] != evidence["source_shape_hw"]:
            raise ValueError("BEFORE source/support array identity differs")
    if count == 0:
        if evidence["finite_min"] is not None or evidence["finite_max"] is not None:
            raise ValueError("Empty BEFORE support cannot have finite extrema")
    else:
        _finite(evidence["finite_min"], where="BEFORE finite minimum")
        _finite(evidence["finite_max"], where="BEFORE finite maximum")
        if evidence["finite_min"] > evidence["finite_max"]:
            raise ValueError("BEFORE finite extrema reversed")
    if evidence["state"] == "complete":
        from ..storage_outputs import LA_ENCODING
        encoding_contract = dict(
            quantization="L=round-to-nearest-even(255*p); decode L/255 only where A=255",
            missing="A=0; L=0 is padding, not evidence",
            coarse_reduction="arithmetic mean L/255 over A=255 pixels; valid-pixel fraction of all cell pixels; no-valid mean NaN")
        if (evidence["encoding"] != LA_ENCODING
                or any(evidence[field] != literal for field, literal in encoding_contract.items())
                or count and not 0 <= evidence["finite_min"] <= evidence["finite_max"] <= 1):
            raise ValueError("BEFORE receipt does not retain the original Q28 probability encoding")


def _frame_record(record, spec, configuration, precision, commit_bound):
    _keys(record, {"format", "complete", "run", "case_id", "frame_index", "elapsed_h", "source",
                   "configuration", "precision", "native", "native_receipt", "phase", "statistics",
                   "before_rules", "files"}, {"photograph", "opencv_ipp", "probabilities"}, where="phase-one frame receipt")
    from darsia._opencv_runtime import validate_observation
    if "opencv_ipp" in record:
        validate_observation(record["opencv_ipp"])
    if (record["format"] != FRAME_FORMAT or record["complete"] is not True
            or record["run"] != spec["source"]["run"] or record["case_id"] != spec["case_id"]
            or not _same_json(record["frame_index"], spec["frame_index"])
            or not _same_json(record["elapsed_h"], spec["elapsed_h"])
            or not _same_json(record["source"], spec["source"])
            or not _same_json(record["configuration"], configuration)
            or record["precision"] != precision):
        raise ValueError("Phase-one frame identity differs from the sealed selection")
    inventory = record["files"]
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("Actual phase-one stage inventory required")
    parent = PureWindowsPath(commit_bound["path"]).parent
    if parent.name != spec["case_id"] or PureWindowsPath(commit_bound["path"]).name != "phase1_commit.json":
        raise ValueError("Phase-one commit is not in its own case namespace")
    locators, basenames = set(), set()
    for basename, bound in inventory.items():
        _bound(bound, where="phase-one inventory")
        key = ntpath.normcase(ntpath.normpath(bound["path"]))
        if (not isinstance(basename, str) or PureWindowsPath(basename).name != basename
                or basename != PureWindowsPath(bound["path"]).name or key in locators
                or basename.casefold() in basenames or PureWindowsPath(bound["path"]).parent != parent):
            raise ValueError("Repeated or escaping phase-one inventory locator")
        locators.add(key)
        basenames.add(basename.casefold())
    native = _keys(record["native"], {"complete", "source_sha256", "full", "audit", "processor_kind"}, {"opencv_ipp"}, where="saved native roles")
    phase = _keys(record["phase"], {"complete", "source_sha256", "labels", "classes", "area", "qa",
                                  "operator_masks", "history_input"}, {"opencv_ipp"}, where="saved phase roles")
    for saved in (native, phase):
        if "opencv_ipp" in saved:
            validate_observation(saved["opencv_ipp"])
    if (native["complete"] is not True or phase["complete"] is not True
            or native["source_sha256"] != spec["source"]["source_sha256"]
            or phase["source_sha256"] != native["source_sha256"]
            or native["processor_kind"] != spec["native_task"]["kind"]):
        raise ValueError("Saved native/phase acquisition differs")
    for field in ("labels", "classes", "area", "qa", "operator_masks", "history_input"):
        _owned(phase[field], inventory, where="saved phase " + field, array=field == "area")
    if phase["area"].get("array_key") != "pixel_area_mm2":
        raise ValueError("Saved phase area does not bind the original pixel_area_mm2 array")
    for field in ("audit",):
        _owned(native[field], inventory, where="saved native " + field)
    _owned(record["native_receipt"], inventory, where="saved native receipt")
    _owned(record["statistics"], inventory, where="native statistics")
    full = native["full"]
    if (not isinstance(full, dict) or full.get("complete") is not True
            or full.get("source_sha256") != native["source_sha256"]):
        raise ValueError("Saved full native acquisition differs")
    for field in ("log_linear_rgb", "valid_mask"):
        _owned(full["native_centre"][field], inventory, where="native centre " + field)
    for window in full["ring_windows"]:
        _owned(window["log_linear_rgb"], inventory, where="native ring")
    before = _keys(record["before_rules"], {"state", "receipt"}, {"png"}, where="BEFORE evidence roles")
    if before["state"] not in {"complete", "pending"} or ((before["state"] == "complete") != ("png" in before)):
        raise ValueError("BEFORE evidence completion/pending binding differs")
    _owned(before["receipt"], inventory, where="BEFORE receipt")
    if "png" in before:
        _owned(before["png"], inventory, where="BEFORE PNG")
    if "photograph" in record:
        photo = _keys(record["photograph"], {"state", "receipt"}, {"png"}, where="saved photograph evidence")
        if photo["state"] not in {"complete", "pending"} or ((photo["state"] == "complete") != ("png" in photo)):
            raise ValueError("Saved photograph completion/pending binding differs")
        _owned(photo["receipt"], inventory, where="saved photograph receipt")
        if "png" in photo:
            _owned(photo["png"], inventory, where="saved photograph PNG")
    if "probabilities" in record:
        roles = _keys(record["probabilities"], {"archive", "receipt"}, where="decision probability roles")
        for field in ("archive", "receipt"):
            _owned(roles[field], inventory, where="decision probability " + field)
        if (PureWindowsPath(roles["archive"]["path"]).name != "phase_probabilities.npz"
                or PureWindowsPath(roles["receipt"]["path"]).name != "phase_probabilities_receipt.json"):
            raise ValueError("Decision probabilities require their exact owned output leaves")
    return record


def verify_phase1_completion(completion_bound, configuration_bound, *, expected_commit, reader):
    """Verify metadata and all declared file bytes, never decode scientific arrays.

    `reader` must be the original positive DATA reader in production. A caller
    cannot substitute metadata-only success for this verification at sealing.
    """
    _bound(completion_bound, where="phase-one completion")
    _bound(configuration_bound, where="phase-one configuration")
    from ..mass.n1_pipeline import DataReader
    if type(reader) is not DataReader:
        raise TypeError("The original content-verifying DataReader is required")
    if not isinstance(expected_commit, str) or re.fullmatch("[0-9a-f]{40}", expected_commit) is None:
        raise ValueError("Actual expected phase-one code commit required")
    config = validate_phase1_run(reader.json(configuration_bound))
    completion = reader.json(completion_bound)
    _data(completion)
    _keys(completion, {"format", "complete", "run", "configuration", "precision", "code_commit",
                      "ordered_case_ids", "frames", "history_complete"}, {"opencv_ipp"}, where="phase-one completion")
    if "opencv_ipp" in completion:
        from darsia._opencv_runtime import validate_observation
        validate_observation(completion["opencv_ipp"])
    if (completion["format"] != COMPLETION_FORMAT or completion["complete"] is not True
            or completion["history_complete"] is not True or completion["run"] != config["run"]
            or completion["configuration"] != configuration_bound or completion["precision"] != config["precision"]
            or completion["code_commit"] != expected_commit or completion["ordered_case_ids"] != config["ordered_case_ids"]
            or not isinstance(completion["frames"], list) or len(completion["frames"]) != len(config["frames"])):
        raise ValueError("Phase-one run completion differs from its actual producer")
    records = []
    for spec, bound in zip(config["frames"], completion["frames"]):
        _bound(bound, where="phase-one frame commit")
        record = _frame_record(reader.json(bound), spec, configuration_bound, config["precision"], bound)
        for artifact in record["files"].values():
            reader.path(artifact)
        reader.json(record["native"]["audit"])
        if reader.json(record["native_receipt"]) != record["native"]:
            raise ValueError("Native receipt differs from the recorded actual native producer")
        history = reader.json(record["phase"]["history_input"])
        from .series import restore_history
        restore_history(history, spec)
        if (history.get("complete") is not True or history.get("full") != record["native"]["full"]
                or not _same_json(history.get("source"), spec["source"])):
            raise ValueError("Phase-one history is not its actual native acquisition")
        for field in ("temporal_log", "temporal_valid"):
            _owned(history[field], record["files"], where="saved history " + field)
            reader.path(history[field])
        before = reader.json(record["before_rules"]["receipt"])
        _keys(before, {"format", "complete", "source", "before_rules"}, {"png"}, where="BEFORE evidence receipt")
        if not isinstance(before["before_rules"], dict):
            raise ValueError("BEFORE evidence metadata must be explicit")
        if (before["format"] != BEFORE_FORMAT or before["complete"] is not True
                or not _same_json(before["source"], spec["source"])
                or before["before_rules"].get("state") != record["before_rules"]["state"]
                or before.get("png") != record["before_rules"].get("png")):
            raise ValueError("BEFORE receipt/source/state differs")
        _before_metadata(before["before_rules"], record["native"]["full"]["phase_shape_hw"])
        if "photograph" in record:
            photo = record["photograph"]
            evidence = reader.json(photo["receipt"])
            if photo["state"] == "complete":
                request = config.get("photographs", {}).get(spec["case_id"])
                if request is None:
                    raise ValueError("Complete photograph lacks its explicit original producer request")
                if (evidence.get("geometry_producer_sha256") != request["geometry_producer_sha256"]
                        or evidence.get("camera_source", {}).get("sha256") != spec["source"]["source_sha256"]
                        or evidence.get("current_phase_products_changed") is not False
                        or evidence.get("display_dtype") != "uint8"):
                    raise ValueError("Saved photograph source/producer/output contract differs")
                if (evidence.get("format") != "darsia-stored-registered-original-photo-v1"
                        or evidence.get("complete") is not True
                        or evidence.get("run") != config["run"] or evidence.get("case_id") != spec["case_id"]
                        or evidence.get("precision") != config["precision"]
                        or not _same_json(evidence.get("source"), spec["source"])
                        or evidence.get("photograph") != photo["png"]
                        or evidence.get("colour_light_or_v5_changes") is not False
                        or evidence.get("camera_RAW_or_correction_chain_executed") is not False):
                    raise ValueError("Saved photograph differs from its original pre-light producer")
            elif (evidence.get("format") != "darsia-phase-original-photograph-pending-v1"
                    or evidence.get("state") != "pending" or evidence.get("reason") != "explicit_saved_pre_light_photograph_not_bound"
                    or not _same_json(evidence.get("source"), spec["source"])):
                raise ValueError("Saved photograph pending evidence differs")
        if ("probabilities" in record) != ("gas_probability_storage" in config):
            raise ValueError("Phase-one probability request/actual product differs")
        if "probabilities" in record:
            from ..phase.probability_storage import validate_receipt, _head_bindings, same_metadata
            roles = record["probabilities"]
            saved = validate_receipt(reader.json(roles["receipt"]), roles, source=spec["source"],
                                     case_id=spec["case_id"], shape=record["native"]["full"]["phase_shape_hw"])
            models = reader.json(config["phase"]["configuration"])["model_bundles"]
            if (not same_metadata(saved["phase_configuration"], config["phase"]["configuration"])
                    or not same_metadata(saved["head_bindings"], _head_bindings(models))
                    or not same_metadata(saved["labels"], record["phase"]["labels"])):
                raise ValueError("Probability head/configuration provenance differs from the actual producer")
        records.append(dict(commit=deepcopy(bound), record=record))
    return dict(configuration=config, completion=completion, records=records,
                configuration_descriptor=deepcopy(configuration_bound), completion_descriptor=deepcopy(completion_bound),
                payload_bytes_verified=True, scientific_arrays_decoded=False,
                verified_scope="phase-one configuration/completion/declared frame outputs",
                queue_ownership_or_retirement_verified=False)


def validate_phase2_run(config):
    """Admit an explicit sealed N1 stage; never infer a stage or domain policy."""
    from ..mass.n1_pipeline import CONFIG_FORMAT, CONFIG_REQUIRED, FRAME_REQUIRED
    _keys(config, {"format", *CONFIG_REQUIRED, "precision", "phase1", "geometry_fields"},
          {"atmosphere_series", "dose", "contracts", "provenance", "data_aliases", "storage", "pixel_domain",
           "storage_profile", "opencl_acceleration", "overflow_phase_split"},
          where="phase-two configuration")
    if "opencl_acceleration" in config:
        from ..mass.mass_opencl import configuration
        configuration(config["opencl_acceleration"])
    if config["format"] != CONFIG_FORMAT or config["precision"] not in {"float32", "float64"}:
        raise ValueError("Explicit precision-bound ordinary N1 stage required")
    if "storage_profile" in config:
        from ..storage_profile import resolve
        if resolve(config)["colour_intermediates"] != "keep":
            raise ValueError("Phase two writes no colour intermediates; their retention is not set here")
    _bound(config["calibration"], where="phase-two fitted calibration")
    if not isinstance(config["frames"], list) or not config["frames"]:
        raise ValueError("Explicit nonempty phase-two case order required")
    ids, previous = [], None
    for spec in config["frames"]:
        if not isinstance(spec, dict) or not set(FRAME_REQUIRED).issubset(spec):
            raise ValueError("Phase-two frame omits the original N1 ABI")
        case = spec["case_id"]
        name(case)
        if case in ids:
            raise ValueError("Distinct explicit phase-two case IDs required")
        if type(spec["frame_index"]) is not int or spec["frame_index"] < 0:
            raise ValueError("Actual phase-two acquisition index required")
        _finite(spec["elapsed_h"], where="phase-two frame time")
        position = (float(spec["elapsed_h"]), spec["frame_index"])
        if previous is not None and not (position[0] > previous[0] and position[1] > previous[1]):
            raise ValueError("Phase-two acquisition chronology differs")
        previous = position
        ids.append(case)
    binding = config["phase1"]
    _keys(binding, {"configuration", "completion", "code_commit", "frames"}, {"origin"}, where="phase-two evidence")
    if "origin" in binding:
        from .reproduction_lineage import validate_phase1_origin
        validate_phase1_origin(binding["origin"])
        if binding["origin"]["run"] != config["run"]:
            raise ValueError("Phase-two original producer belongs to another run")
    for field in ("configuration", "completion"):
        _bound(binding[field], where="phase-two " + field)
    if not isinstance(binding["code_commit"], str) or re.fullmatch(r"[0-9a-f]{40}", binding["code_commit"]) is None:
        raise ValueError("Actual phase-one code commit required")
    if not isinstance(binding["frames"], dict) or set(binding["frames"]) != set(ids):
        raise ValueError("Phase-two evidence omits or adds a selected case")
    for refs in binding["frames"].values():
        _keys(refs, {"commit", "statistics", "before_rules"}, {"photograph"}, where="phase-two case evidence")
        for field in ("commit", "statistics"):
            _bound(refs[field], where="phase-two " + field)
    if "pixel_domain" in config:
        policy = config["pixel_domain"]
        if not isinstance(policy, dict) or "n16" not in policy:
            raise ValueError("The sealed domain extension requires explicit genuine N16 inputs")
        _keys(policy, {"contract", "adopted_for_execution", "fixture_scope_complete", "fixture_names",
                       "injector_hardware_excluded", "authorized_decisions", "max_column_distance_m",
                       "raster_scale", "raster_shift", "geometry_chunk_pixels", "below_floor_phase_unknown_policy",
                       "include_estimated_mass", "table_fields", "baseline_coverage_tolerance", "heldout_fit_run", "n16"},
              where="explicit N15/N16 policy")
        _keys(policy["n16"], {"contract", "authority", "capacity_selector"}, where="N16 policy")
        from ..mass.pixel_pipeline import validate_policy
        validate_policy(policy)
        if (policy["n16"]["contract"] != "darsia-n16-q85-a-v1"
                or policy["n16"]["capacity_selector"] != "literal-R3-hidden255-solubility"
                or policy.get("contract") != "darsia-pixel-domain-v1"
                or policy.get("adopted_for_execution") is not True
                or policy.get("fixture_scope_complete") is not True
                or policy.get("max_column_distance_m") != .020
                or policy.get("include_estimated_mass") is not False):
            raise ValueError("The existing explicit N15/N16 Q85-A policy is required")
        _bound(policy["authorized_decisions"], where="N15 domain authority")
        _bound(policy["n16"]["authority"], where="N16 Q85 authority")
        for spec in config["frames"]:
            _keys(spec["pixel_domain_geometry"],
                  {"source_receipt", "json_pointer", "source_frame_index", "inner_polygon_native",
                   "fixture_polygons_native", "native_origin_xy", "phase_shape_hw", "target_depth", "n16_masks"},
                  where="N16 source geometry")
            declared = spec["pixel_domain_geometry"]
            _bound(declared["source_receipt"], where="N16 source geometry receipt")
            _bound(declared["target_depth"], where="N16 measured depth", array=True)
            _keys(declared["n16_masks"], {"below_photo_strip", "not_sand"}, where="N16 source masks")
            for bound in declared["n16_masks"].values():
                if bound is not None:
                    _bound(bound, where="N16 explicit mask")
            basis = spec["frozen_response_basis"]
            _keys(basis, {"result", "geometry", "arrays", "per_frame_geometry", "source_sha256",
                          "phase_sha256", "calibration_sha256"}, where="N16 original response basis")
            for field in ("result", "geometry"):
                _bound(basis[field], where="N16 original response " + field)
    if "overflow_phase_split" in config:
        _overflow_phase_split(config)
    _data(config)
    return config


def _overflow_phase_split(config):
    """Check a sealed per-image overflow phase split against the case order.

    Each image has the target of the run's own measurement or a
    pressure-matched estimate, or none with its reason, and the stored
    probabilities and acquisition that the worker reads for it.
    """
    from ..mass.overflow_phase_split import POLICY, TARGET_FORMAT
    block = config["overflow_phase_split"]
    _keys(block, {"policy", "frames"}, where="overflow phase split")
    frames = block["frames"]
    if block["policy"] != POLICY or not isinstance(frames, dict) or list(frames) != [
            spec["case_id"] for spec in config["frames"]]:
        raise ValueError("The overflow phase split must cover the exact phase-two case order")
    for spec in config["frames"]:
        image = frames[spec["case_id"]]
        _keys(image, {"target", "probability_roles", "phase_source"}, {"reason"}, where="overflow image")
        target = image["target"]
        if target is None:
            if not isinstance(image.get("reason"), str) or not image["reason"]:
                raise ValueError("An image without an overflow reference needs its reason")
            continue
        if (not isinstance(target, dict) or target.get("format") != TARGET_FORMAT
                or target.get("policy") != POLICY
                or target.get("kind") not in {"own_measurement", "pressure_matched_estimate"}
                or target.get("run") != config["run"] or target.get("elapsed_h") != spec["elapsed_h"]):
            raise ValueError("The overflow target must belong to this exact run and image time")
        _finite(target.get("target_volume_m3"), where="overflow gas-volume target")
        if target["target_volume_m3"] < 0:
            raise ValueError("The overflow gas-volume target must be nonnegative")
        if not isinstance(image["probability_roles"], dict) or not isinstance(image["phase_source"], dict):
            raise ValueError("The overflow image needs its stored probabilities and acquisition")


def validate_n16_frame_binding(config, spec, *, authority, source, original, calibration):
    """Check actual existing producer metadata, without constructing mass or geometry."""
    policy = config["pixel_domain"]
    # Callers read the record through its SHA-pinned configuration binding.
    from ..series_decisions import recorded_decision
    recorded_decision(authority, "N16 hidden-mass decision")
    declared = spec["pixel_domain_geometry"]
    pointer = declared["json_pointer"]
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("Explicit N16 source geometry JSON pointer required")
    for key in pointer[1:].split("/"):
        source = source[key.replace("~1", "/").replace("~0", "~")]
    if (source.get("run") != config["run"] or source.get("frame_index") != spec["frame_index"]
            or declared["source_frame_index"] != spec["frame_index"]
            or source.get("source_sha256") != spec["source_binding"]["source_sha256"]):
        raise ValueError("N16 geometry belongs to another actual source frame")
    for field, source_field in (("inner_polygon_native", "frame_inner_polygon_native"),
                                ("fixture_polygons_native", "frame_fixture_polygons_native"),
                                ("native_origin_xy", "native_origin_xy"), ("phase_shape_hw", "phase_shape_hw"),
                                ("n16_masks", "n16_masks")):
        if not _same_json(declared[field], source[source_field]):
            raise ValueError("N16 declared geometry differs from its actual receipt: " + field)
    if (declared["inner_polygon_native"] != spec["inner_polygon_native"]
            or declared["native_origin_xy"] != spec["current"]["native_origin_xy"]
            or declared["phase_shape_hw"] != spec["current"]["phase_shape_hw"]
            or set(declared["fixture_polygons_native"]) != set(policy["fixture_names"])
            or len(declared["fixture_polygons_native"]) != 3):
        raise ValueError("N16 source polygons/grid differ from the original N1 frame")
    run = str(config["run"]).lower().removeprefix("ac")
    if run.isdigit() and 17 <= int(run) <= 38 and declared["n16_masks"]["below_photo_strip"] is None:
        raise ValueError("AC17-38 requires its actual source-bound below-photo strip")
    basis = spec["frozen_response_basis"]
    expected = dict(source_sha256=spec["source_binding"]["source_sha256"],
                    phase_sha256=spec["labels"]["sha256"], calibration_sha256=config["calibration"]["sha256"])
    if (original.get("complete") is not True or original.get("run") != config["run"]
            or original.get("frame_index") != spec["frame_index"] or original.get("elapsed_h") != spec["elapsed_h"]
            or any(original["bindings"].get(k) != v or basis.get(k) != v for k, v in expected.items())
            or original["scalar_result"]["parameters"] != calibration["parameters"]
            or original["scalar_result"]["precision"] != config["precision"]
            or original["anchor"] != config["anchor"]):
        raise ValueError("N16 actual response basis differs from this calibration, anchor or frame")
    static = original["static"]
    if "storage" in static or static.get("asset") is None:
        raise ValueError("N16 measured depth needs a full-profile reference frame")
    if (any(basis["geometry"][key] != static["asset"][key] for key in ("sha256", "size_bytes"))
            or basis["arrays"] != static["arrays"] or basis["per_frame_geometry"] != static["per_frame_geometry"]
            or declared["target_depth"].get("array_key") != "depth_m"
            or any(declared["target_depth"][key] != basis["geometry"][key] for key in ("sha256", "size_bytes"))):
        raise ValueError("N16 measured depth/basis differs from its actual producer")


def seal_phase2_configuration(destination, completion_bound, configuration_bound, *,
                              expected_phase1_commit, fitted_calibration, fitted_anchor,
                              storage_data=None, pixel_domain_data=None, reader,
                              storage_profile=None, frozen_refit=None, phase1_origin=None,
                              overflow_phase_split=None):
    """Seal ordinary N1 DATA from actual phase-one products and explicit new fit.

    No native/inference or mass evaluation occurs. The returned DATA still needs
    positive public catalogue/worker/reader integration; it is not a job.
    ``overflow_phase_split`` ({policy, targets by case}) seals each image's
    overflow target with its stored gas probabilities; None seals none.
    """
    verified = verify_phase1_completion(completion_bound, configuration_bound,
                                       expected_commit=expected_phase1_commit, reader=reader)
    if phase1_origin is not None:
        from .reproduction_lineage import verify_phase1_origin
        phase1_origin = verify_phase1_origin(phase1_origin,
            configuration=configuration_bound, completion=completion_bound,
            expected_commit=expected_phase1_commit,
            ordered_case_ids=verified["configuration"]["ordered_case_ids"])["origin"]
    _bound(fitted_calibration, where="explicit fitted mass calibration")
    calibration = reader.json(fitted_calibration)
    if not isinstance(calibration, dict) or not isinstance(calibration.get("parameters"), dict):
        raise ValueError("The explicit fitted calibration lacks original N1 parameters")
    from ..mass.n1_arithmetic import validate_parameters
    validate_parameters(calibration["parameters"])
    _keys(fitted_anchor, {"elapsed_h", "affected_pore_volume_m3"}, where="explicit fitted mass anchor")
    _finite(fitted_anchor["elapsed_h"], where="fitted anchor time")
    _finite(fitted_anchor["affected_pore_volume_m3"], where="fitted anchor volume", positive=True)
    refit_join = None
    if frozen_refit is not None:
        from .common_calibration import verify_frozen_refit
        refit_join = verify_frozen_refit(frozen_refit, verified_phase1=verified,
            fitted_calibration=fitted_calibration, fitted_anchor=fitted_anchor,
            expected_commit=expected_phase1_commit, reader=reader)
    from .series import bind_mass_frame
    from ..mass.n1_pipeline import CONFIG_FORMAT, CONFIG_REQUIRED, FRAME_REQUIRED
    phase1 = verified["configuration"]
    candidate = deepcopy(phase1["statistics"])
    candidate.update(format=CONFIG_FORMAT, run=phase1["run"], precision=phase1["precision"],
                     calibration=deepcopy(fitted_calibration), anchor=deepcopy(fitted_anchor))
    if refit_join is not None:
        provenance = deepcopy(candidate.get("provenance", {}))
        if not isinstance(provenance, dict) or "common_frozen_refit" in provenance:
            raise ValueError("The explicit common refit join cannot replace existing provenance")
        candidate["provenance"] = dict(provenance, common_frozen_refit=refit_join)
    candidate["frames"] = [bind_mass_frame(spec, item["record"]["native"], item["record"]["phase"],
                            baseline=phase1["phase"]["reference"]["full"])
                           for spec, item in zip(phase1["frames"], verified["records"])]
    if "data_aliases" in phase1:
        candidate["data_aliases"] = deepcopy(phase1["data_aliases"])
    candidate["phase1"] = dict(configuration=deepcopy(configuration_bound), completion=deepcopy(completion_bound),
        code_commit=expected_phase1_commit,
        frames={spec["case_id"]: dict(commit=deepcopy(item["commit"]), statistics=deepcopy(item["record"]["statistics"]),
                    before_rules=deepcopy(item["record"]["before_rules"]))
                for spec, item in zip(phase1["frames"], verified["records"])})
    if phase1_origin is not None:
        candidate["phase1"]["origin"] = deepcopy(phase1_origin)
    for spec, item in zip(phase1["frames"], verified["records"]):
        if "photograph" in item["record"]:
            candidate["phase1"]["frames"][spec["case_id"]]["photograph"] = deepcopy(item["record"]["photograph"])
    if storage_data is not None:
        candidate["storage"] = deepcopy(storage_data)
        from ..storage_outputs import parse_request
        parse_request(candidate, candidate["precision"])
    if storage_profile is not None:
        candidate["storage_profile"] = deepcopy(storage_profile)
    if overflow_phase_split is not None:
        _keys(overflow_phase_split, {"policy", "targets"}, where="overflow phase-split inputs")
        targets, frames = overflow_phase_split["targets"], {}
        if not isinstance(targets, dict) or set(targets) != {spec["case_id"] for spec in phase1["frames"]}:
            raise ValueError("Overflow targets must cover the complete original case order")
        for spec, item in zip(phase1["frames"], verified["records"]):
            given = targets[spec["case_id"]]
            if "probabilities" not in item["record"]:
                raise ValueError("The overflow phase split needs each image's stored gas probabilities")
            image = dict(target=None if given.get("kind") == "none" else deepcopy(given),
                         probability_roles=deepcopy(item["record"]["probabilities"]),
                         phase_source=deepcopy(spec["source"]))
            if given.get("kind") == "none":
                image["reason"] = given.get("reason")
            frames[spec["case_id"]] = image
        candidate["overflow_phase_split"] = dict(policy=overflow_phase_split["policy"], frames=frames)
    if pixel_domain_data is not None:
        _keys(pixel_domain_data, {"pixel_domain", "frames"}, where="explicit N16 phase-two inputs")
        additions = pixel_domain_data["frames"]
        if not isinstance(additions, dict) or set(additions) != {f["case_id"] for f in candidate["frames"]}:
            raise ValueError("N16 supplied inputs must match the complete original case order")
        candidate["pixel_domain"] = deepcopy(pixel_domain_data["pixel_domain"])
        for spec in candidate["frames"]:
            extra = additions[spec["case_id"]]
            _keys(extra, {"pixel_domain_geometry", "frozen_response_basis"}, where="explicit N16 case inputs")
            spec.update(deepcopy(extra))
        validate_phase2_run(candidate)
        reader.json(candidate["pixel_domain"]["authorized_decisions"])
        authority = reader.json(candidate["pixel_domain"]["n16"]["authority"])
        for spec in candidate["frames"]:
            declared, basis = spec["pixel_domain_geometry"], spec["frozen_response_basis"]
            validate_n16_frame_binding(candidate, spec, authority=authority,
                source=reader.json(declared["source_receipt"]), original=reader.json(basis["result"]),
                calibration=calibration)
            for bound in (declared["target_depth"], basis["geometry"], *declared["n16_masks"].values()):
                if bound is not None:
                    reader.path(bound)
    validate_phase2_run(candidate)
    for frame in candidate["frames"]:
        if not set(FRAME_REQUIRED).issubset(frame):
            raise ValueError("Actual phase-one products do not complete the unchanged N1 frame ABI")
    _data(candidate)
    write_immutable(Path(destination), candidate)
    return dict(configuration=descriptor(Path(destination)), format=CONFIG_FORMAT,
                phase1_source=deepcopy(completion_bound), payload_bytes_verified=True,
                verified_scope="phase-one configuration/completion/declared frame outputs and fitted calibration",
                queue_ownership_or_retirement_verified=False,
                native_or_phase_calls=0, mass_evaluation_calls=0, job_created=False,
                execution_ready=False,
                execution_blocker="Phase-one catalogue/worker and genuine N1 saved-storage reader admission require their separate gates")
