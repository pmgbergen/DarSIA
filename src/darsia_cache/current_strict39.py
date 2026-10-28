"""Prospective strict39 DATA contracts; never an admission or authority provider.

Only stdlib metadata is used here. Original publication, physical independence,
Build-to-frame, native retirement and independently issued authority verification
are still required. No body, callback, report or caller flag lifts admission STOP.
The original comparator and its sixteen retained scientific helpers are unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import PureWindowsPath
import re


ROLES = ("accepted_reference", "predeclared_contract", "postproducer_release", "strict_acceptance")
FORMATS = {role: "darsia-current-strict39-" + role.replace("_", "-") + "-data-v1" for role in ROLES}
STATE = "SOURCE_DATA_ONLY"
REPORT_FORMAT = "prospective-current-plan-Main-RAW39-original-Add8-strict-N56-v1"
CASE_IDS = tuple("ac52_f" + str(i) for i in (
    1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21,
    23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43))
FLOATS = ("concentration_aqueous", "saturation_free_gas", "mass_total", "mass_aqueous", "mass_free_gas")
GEOMETRY = ("pixel_area_m2", "depth_m", "porosity", "facies", "transport_weight_l")
ORIGINAL_SPEC_HASHES = {
    "base_spec_sha256": "36be522e1323c6bf376f0868e3ab52a21090185d6908322d7c48ed450df2ed96",
    "spec_sha256": "36aa8e5f5dd4a3e5e3537c5d330aa1f6d6d4b6454d43cd69def9fac8b904404d",
    "precision_policy_sha256": "4f73dc13f28dd6eb4ae0c24a614c854838ddcb8ad1528906cede74304041ca60",
}
DOMAIN_SCALARS = tuple("domains." + domain + "." + quantity
    for domain in ("full_rig", "box1", "box2")
    for quantity in ("total_mass_g", "aqueous_mass_g", "gas_mass_g"))
ORIGINAL_HELPER_BYTES = {
    "numeric_checker": ("9949a184fef7e3b55c75336544fa04f52a2062f1b37cfdad1018f2243280b5ce", 17239),
    "record_extractors": ("95dc41bf655c5d22a91dd5bf7cf55430f0f5a511d8d7cdabe8f8e68378b6aa13", 21587),
    "read_locator": ("ba09417f90f79b0d6ce4a96ac93d9a9f8969bc2b1daa37dbd219f3ed718d758e", 8067),
    "typed_ipp_validator": ("a7f2550125fb102d1e297c7d72d2de01594fe91640d78a3e72869314bcdedb15", 3901),
}
# Names identify original documents/receipts, not schemas invented for their bodies.
PRODUCER_ROLES = ("task", "job", "done", "products_index", "completion", "request",
    "provider", "observation", "native_completion", "retirement", "service_receipt",
    "seal", "eligibility", "exit_inventory", "closed_readset", "all_stage_reference", "once_marker")
DIAGNOSTICS = ("all_five_maps_bit_identical", "phase_labels_changed_pixels_zero",
    "native_geometry_array_identities_exact", "original_scalar_result_JSON_exact",
    "original_threshold_flags_exact", "all_extracted_scientific_quantities_exact")
PRODUCT_FLAGS = ("all_scientific_product_arrays_bit_exact", "products_code_sha256_per_host_valid",
    "scalar_result_exact", "native_statistics_JSON_content_exact",
    "all_retained_stage_NPY_NPZ_PNG_bytes_exact", "no_scientific_content_projected",
    "code_identity_values_not_normalized")
MISSING_ORIGINAL_VERIFIERS = (
    "CURRENT_TYPED_REFERENCE_AND_COMPARISON_AUTHORITY_UNAVAILABLE",
    "AUTHENTICATED_BUILD_TO_QUEUED_FRAME_JOIN_UNAVAILABLE",
    "BOTH_ORIGINAL_PUBLICATIONS_PHYSICAL_INDEPENDENCE_AND_NATIVE_RETIREMENTS_UNBOUND",
    "ORIGINAL_COMPARATOR_RETIREMENT_AND_SOURCE_PAIR_VERIFICATION_UNAVAILABLE",
)


class ContractError(ValueError):
    """Malformed or inconsistent prospective DATA, never a scientific verdict."""


class CurrentAdmissionUnavailable(ContractError):
    """The original authority/verifier chain has not been implemented."""


def require_original_current_admission(*_args, **_kwargs):
    """Unconditional STOP. Neither validated DATA nor injected callbacks grant GO."""
    raise CurrentAdmissionUnavailable("; ".join(MISSING_ORIGINAL_VERIFIERS))


def absent_role_bindings():
    return {role: None for role in ROLES}


def _need(condition, message):
    if not condition:
        raise ContractError(message)


def _exact(value, keys, role):
    _need(type(value) is dict and set(value) == set(keys), "Closed DATA fields required: " + role)
    return value


def _hex(value, length, role):
    _need(type(value) is str and re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", value) is not None,
          "Exact hexadecimal string required: " + role)


def _integer(value, role, minimum=0):
    _need(type(value) is int and value >= minimum, "Exact integer required: " + role)


def _number(value, role, minimum=0):
    _need(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
          "Finite number required: " + role)


def _json(value):
    """Reject non-JSON objects/nonfinite numbers even for direct Python callers."""
    if type(value) is dict:
        _need(all(type(k) is str for k in value), "JSON object keys must be strings")
        for v in value.values():
            _json(v)
    elif type(value) is list:
        for v in value:
            _json(v)
    elif type(value) is float:
        _need(math.isfinite(value), "Nonfinite DATA value")
    else:
        _need(value is None or type(value) in (str, int, bool), "JSON DATA only")


def canonical_digest(value):
    _json(value)
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def descriptor(value):
    _exact(value, {"path", "sha256", "size_bytes"}, "descriptor")
    path = value["path"]
    _need(type(path) is str and PureWindowsPath(path).is_absolute()
          and ".." not in PureWindowsPath(path).parts, "Absolute original descriptor required")
    _hex(value["sha256"], 64, "descriptor SHA")
    _integer(value["size_bytes"], "descriptor length")
    return deepcopy(value)


def _path(value):
    return str(PureWindowsPath(value["path"])).casefold()


def _pins(value, fields, role):
    _exact(value, fields, role)
    for pin in value.values():
        descriptor(pin)


def _scope(value):
    expected = dict(pipeline="native-phase-n1", backend="cpu", precision="float32",
                    run="ac52", ordered_case_ids=list(CASE_IDS), threads=1)
    _exact(value, expected, "strict39 scope")
    _integer(value["threads"], "threads", 1)
    _need(value == expected, "Original CPU1 float32 exact ordered39 scope required")


def _base(value, role, fields):
    _json(value)
    _exact(value, {"format", "state", "scope"} | set(fields), role)
    _need(value["format"] == FORMATS[role] and value["state"] == STATE,
          "Prospective SOURCE_DATA_ONLY format required")
    _scope(value["scope"])


def _source_pair(value):
    _exact(value, {"relationship", "Reference", "Candidate"}, "source pair")
    for host in ("Reference", "Candidate"):
        _exact(value[host], {"commit", "tree"}, host + " source")
        for field in ("commit", "tree"):
            _hex(value[host][field], 40, host + " " + field)
    same = value["Reference"]["commit"] == value["Candidate"]["commit"]
    _need(value["relationship"] in {"same_source_independent_reexecution", "reviewed_source_pair"},
          "Explicit source relationship required")
    _need((value["relationship"] == "same_source_independent_reexecution") == same
          and (not same or value["Reference"]["tree"] == value["Candidate"]["tree"]),
          "Source relationship differs from commit/tree DATA")


def checked_accepted_reference_data(value):
    fields = {"current_plan", "configuration", "input_catalog_sha256", "current_v9_build",
              "producer", "authority_record"}
    _base(value, "accepted_reference", fields)
    for role in ("current_plan", "current_v9_build", "authority_record"):
        descriptor(value[role])
    _hex(value["input_catalog_sha256"], 64, "catalogue digest")
    ref = _exact(value["configuration"], {"root_alias", "relative_path", "sha256", "role", "bytes"},
                 "original configuration reference")
    _need(type(ref["root_alias"]) is str and bool(ref["root_alias"])
          and type(ref["relative_path"]) is str and bool(ref["relative_path"]), "Named configuration required")
    relative = PureWindowsPath(ref["relative_path"])
    _need(not relative.is_absolute() and not relative.drive and not relative.root
          and ".." not in relative.parts and ref["role"] == "configuration", "Rooted configuration role required")
    _hex(ref["sha256"], 64, "configuration SHA")
    _integer(ref["bytes"], "configuration bytes")
    _pins(value["producer"], PRODUCER_ROLES, "original reference producer descriptors")
    return deepcopy(value)


def checked_predeclared_contract_data(value):
    fields = {"accepted_reference", "current_plan", "candidate_task", "source_pair", "source_pair_review",
              "sources", "public_command", "report_format"}
    _base(value, "predeclared_contract", fields)
    for role in ("accepted_reference", "current_plan", "candidate_task", "source_pair_review"):
        descriptor(value[role])
    _source_pair(value["source_pair"])
    _pins(value["sources"], set(ORIGINAL_HELPER_BYTES) | {"adapter"}, "original comparison sources")
    for role, identity in ORIGINAL_HELPER_BYTES.items():
        pin = value["sources"][role]
        _need((pin["sha256"], pin["size_bytes"]) == identity, "Original helper bytes changed: " + role)
    command = value["public_command"]
    _need(type(command) is list and bool(command) and all(type(arg) is str and arg for arg in command)
          and value["sources"]["adapter"]["path"] in command, "Exact adapter argv DATA required")
    _need(value["report_format"] == REPORT_FORMAT, "Original prospective strict39 report required")
    return deepcopy(value)


def checked_postproducer_release_data(value):
    fields = {"predeclared_contract", "accepted_reference", "producers", "request_payload_sha256", "authority_record"}
    _base(value, "postproducer_release", fields)
    for role in ("predeclared_contract", "accepted_reference", "authority_record"):
        descriptor(value[role])
    _hex(value["request_payload_sha256"], 64, "request payload digest")
    _exact(value["producers"], {"Reference", "Candidate"}, "two original producers")
    for producer in value["producers"].values():
        _pins(producer, PRODUCER_ROLES, "original producer descriptors")
    for role in ("task", "job", "done", "completion", "native_completion", "retirement"):
        _need(_path(value["producers"]["Reference"][role]) != _path(value["producers"]["Candidate"][role]),
              "Obvious self-comparison descriptor refused: " + role)
    # Different names are only a rejection filter; physical independence is unverified.
    return deepcopy(value)


def checked_strict_acceptance_data(value):
    fields = {"postproducer_release", "predeclared_contract", "report", "comparator_retirement",
              "source_runtime_before_after", "actual_command", "actual_exit_code", "authority_record"}
    _base(value, "strict_acceptance", fields)
    for role in fields - {"actual_command", "actual_exit_code"}:
        descriptor(value[role])
    _need(type(value["actual_exit_code"]) is int and value["actual_exit_code"] == 0,
          "Exact original comparator OS-return DATA required")
    command = value["actual_command"]
    _need(type(command) is list and command and all(type(arg) is str and arg for arg in command), "Original argv DATA required")
    return deepcopy(value)


def _strict_products(value, *, source_pair, expected_stage_files):
    _exact(value, set(PRODUCT_FLAGS) | {"all_products_bit_exact", "comparisons"}, "N56 products")
    _need(all(value[k] is True for k in PRODUCT_FLAGS), "All original N56 scientific checks required")
    rows = value["comparisons"]
    _need(type(rows) is list and len(rows) >= 2, "Actual original N56 array and stage rows required")
    _need([row.get("role") for row in rows[:2]] == ["products.npz", "native_geometry.npz"], "Original archive order required")
    all_bits = []
    for archive in rows[:2]:
        _exact(archive, {"role", "arrays", "bit_exact", "all_scientific_arrays_bit_exact", "per_host_code_identity_validated"}, "N56 archive")
        product = archive["role"] == "products.npz"
        _need(archive["all_scientific_arrays_bit_exact"] is True
              and archive["per_host_code_identity_validated"] is product, "Original archive checks required")
        arrays = archive["arrays"]
        _need(type(arrays) is list and arrays and all(type(row) is dict for row in arrays), "Nonempty original arrays required")
        keys = [row.get("key") for row in arrays]
        _need(all(type(key) is str and key for key in keys) and len(keys) == len(set(keys)), "Unique original array names required")
        required = (set(FLOATS) | {"phase_labels", "main_mass_eligible_mask", "unquantified_facies_mask",
            "E_lo_g", "E_hi_g", "U_rest_g", "U_rest_defined", "U_rest_amendment_sha256",
            "source_sha256", "phase_sha256", "calibration_sha256", "code_sha256"}) if product else set(GEOMETRY)
        _need(required <= set(keys), "Required original scientific arrays are missing")
        if not product:
            _need(keys == list(GEOMETRY), "Original native geometry array order required")
        bits = []
        image_shape = None
        for row in arrays:
            code = product and row["key"] == "code_sha256"
            fields = {"key", "shape", "dtype", "bit_exact"}
            if code:
                fields |= {"per_host_actual_commit_identity_valid", "per_host_identities"}
            _exact(row, fields, "N56 array")
            _need(type(row["shape"]) is list and all(type(v) is int and v >= 0 for v in row["shape"])
                  and type(row["dtype"]) is str and re.fullmatch(r"[<>=|][biufcSUV][0-9]+", row["dtype"]) is not None,
                  "Original non-object dtype and shape required")
            _need(type(row["bit_exact"]) is bool, "Exact array verdict required")
            if row["key"] in (set(FLOATS) | {"phase_labels"} if product else set(GEOMETRY)):
                _need(len(row["shape"]) == 2 and all(v > 0 for v in row["shape"]), "Original image shape required")
                if image_shape is None:
                    image_shape = row["shape"]
                _need(row["shape"] == image_shape, "Original same-grid array shape required")
            if product and row["key"] in FLOATS:
                _need(row["dtype"] == "<f4", "Original float32 product dtype required")
            if product and row["key"] == "phase_labels":
                _need(row["dtype"] == "|u1", "Original uint8 labels required")
            if product and row["key"] in {"main_mass_eligible_mask", "unquantified_facies_mask", "U_rest_defined"}:
                _need(row["dtype"] == "|b1", "Original boolean product dtype required")
            if product and row["key"] in {"E_lo_g", "E_hi_g", "U_rest_g"}:
                _need(row["dtype"] == "<f8" and row["shape"] == [], "Original float64 mass scalar required")
            if product and row["key"] in {"source_sha256", "phase_sha256", "calibration_sha256"}:
                _need(row["dtype"] == "<U64" and row["shape"] == [], "Original scalar provenance dtype required")
            if code:
                _need(row["shape"] == [] and row["dtype"] == "<U64"
                      and row["per_host_actual_commit_identity_valid"] is True, "Only original scalar code identity exception permitted")
                identities = row["per_host_identities"]
                _need(type(identities) is list and len(identities) == 2, "Both actual code identities required")
                for host, identity in zip(("Reference", "Candidate"), identities):
                    _exact(identity, {"host", "actual_commit", "stored_code_sha256", "expected_code_sha256", "shape", "dtype", "raw_array_sha256"}, "code identity")
                    commit = source_pair[host]["commit"]
                    expected = hashlib.sha256(commit.encode("ascii")).hexdigest()
                    _need(identity["host"] == host and identity["actual_commit"] == commit
                          and identity["stored_code_sha256"] == identity["expected_code_sha256"] == expected
                          and identity["shape"] == [] and identity["dtype"] == "<U64", "Per-host code identity differs")
                    _hex(identity["raw_array_sha256"], 64, "original code array SHA")
                    _need(identity["raw_array_sha256"] == hashlib.sha256(expected.encode("utf-32-le")).hexdigest(),
                          "Original scalar <U64 code bytes differ")
                exact_code = identities[0]["raw_array_sha256"] == identities[1]["raw_array_sha256"]
                _need(row["bit_exact"] is exact_code
                      and exact_code is (source_pair["Reference"]["commit"] == source_pair["Candidate"]["commit"]), "Code inequality must remain reported")
            else:
                _need(row["bit_exact"] is True, "A scientific array difference cannot be accepted")
            bits.append(row["bit_exact"])
        _need(archive["bit_exact"] is all(bits), "Archive raw-equality summary differs")
        all_bits.append(archive["bit_exact"])
    expected_names = list(expected_stage_files["Reference"])
    _need(set(expected_names) == set(expected_stage_files["Candidate"]), "Actual retained stage roles differ")
    _need([row.get("role") for row in rows[2:]] == expected_names, "Original retained stage order/coverage differs")
    for row in rows[2:]:
        _exact(row, {"role", "bit_exact", "reference", "candidate"}, "retained stage bytes")
        for host, role in (("Reference", "reference"), ("Candidate", "candidate")):
            descriptor(row[role])
            _need(row[role] == expected_stage_files[host][row["role"]], "Stage row differs from original result descriptor")
        _need(row["bit_exact"] is True and (row["reference"]["sha256"], row["reference"]["size_bytes"])
              == (row["candidate"]["sha256"], row["candidate"]["size_bytes"]), "Retained stage bytes differ")
        all_bits.append(True)
    _need(value["all_products_bit_exact"] is all(all_bits), "N56 raw equality summary differs")


def checked_strict39_case(value, *, case_id, source_pair, results):
    """Check original report DATA and links, without reading scientific payloads."""
    _json(value)
    _source_pair(source_pair)
    _exact(results, {"Reference", "Candidate"}, "both original results")
    required = {"contract", "complete", "limits_passed", "spec_sha256", "base_spec_sha256", "precision_policy_sha256",
        "frame_pixels", "frame_finite_pixels_above_any_map_limit", "frame_allowed_exceeding_pixels", "max_exceeding_pixel_fraction",
        "comparison_kind", "reference_settings", "candidate_settings", "run", "frame_index", "labels", "maps", "scalars", "flags",
        "first_failure", "chunk_pixels", "qualification_complete", "whole_run_history_and_wasserstein_still_required",
        "production_permission_granted", "strict_FFAC_bit_identical", "strict_exact_diagnostics", "strict_N56_exact_products",
        "case_id", "reference_result", "candidate_result", "actual_host_provenance", "actual_separately_derived_scientific_bindings",
        "per_host_phase_output_sha256", "geometry_array_identities", "geometry_identical_required"}
    _exact(value, required, "original strict39 case")
    _need(case_id in CASE_IDS and value["case_id"] == case_id and value["run"] == "ac52"
          and value["contract"] == "darsia-production-addendum8v3-frame-comparison-v1"
          and value["comparison_kind"] == "same_precision" and value["first_failure"] is None, "Original case identity/verdict differs")
    for key in ("complete", "limits_passed", "strict_FFAC_bit_identical", "geometry_identical_required", "whole_run_history_and_wasserstein_still_required"):
        _need(value[key] is True, "Required original case check missing: " + key)
    for key in ("qualification_complete", "production_permission_granted"):
        _need(value[key] is False, "Frame report cannot grant qualification or production")
    _exact(value["strict_exact_diagnostics"], DIAGNOSTICS, "strict diagnostics")
    _need(all(v is True for v in value["strict_exact_diagnostics"].values()), "All exact diagnostics required")
    for key in ("spec_sha256", "base_spec_sha256", "precision_policy_sha256"):
        _hex(value[key], 64, key)
        _need(value[key] == ORIGINAL_SPEC_HASHES[key], "Original numerical policy bytes differ")
    for settings in (value["reference_settings"], value["candidate_settings"]):
        _exact(settings, {"host", "backend", "precision"}, "original comparison settings")
        _need(settings == dict(host="Main", backend="cpu", precision="float32"), "Original CPU float32 comparison settings required")
    _integer(value["frame_pixels"], "frame pixels", 1)
    _integer(value["frame_index"], "frame index")
    _integer(value["chunk_pixels"], "chunk pixels", 1)
    _integer(value["frame_finite_pixels_above_any_map_limit"], "frame exceedance")
    _need(value["frame_finite_pixels_above_any_map_limit"] == 0, "Strict maps cannot exceed limits")
    _number(value["max_exceeding_pixel_fraction"], "original allowed fraction")
    _need(value["max_exceeding_pixel_fraction"] == 1e-5, "Original Add8v3 allowance must remain unchanged")
    _integer(value["frame_allowed_exceeding_pixels"], "allowed pixels")
    _need(value["frame_allowed_exceeding_pixels"] == math.floor(value["frame_pixels"] * value["max_exceeding_pixel_fraction"]), "Original frame allowance differs")
    _exact(value["maps"], FLOATS, "five original maps")
    for row in value["maps"].values():
        _exact(row, {"reference_maximum", "absolute_limit", "max_difference", "failing_pixels", "first_failing_flat_index",
            "nonfinite_support_mismatch_pixels", "bit_identical", "finite_pixels_above_limit", "allowed_exceeding_pixels", "passed"}, "map metrics")
        for key in ("reference_maximum", "absolute_limit", "max_difference"):
            _number(row[key], key)
        for key in ("failing_pixels", "nonfinite_support_mismatch_pixels", "finite_pixels_above_limit", "allowed_exceeding_pixels"):
            _integer(row[key], key)
        _need(row["bit_identical"] is True and row["passed"] is True and row["first_failing_flat_index"] is None
              and row["max_difference"] == row["failing_pixels"] == row["nonfinite_support_mismatch_pixels"] == row["finite_pixels_above_limit"] == 0
              and row["allowed_exceeding_pixels"] == value["frame_allowed_exceeding_pixels"], "Original strict map metrics differ")
    labels = value["labels"]
    areas = {"reference_co2", "reference_gas", "changed_co2_related", "changed_co2_support", "changed_gas", "changed_other", "co2_limit_cm2", "gas_limit_cm2"}
    _exact(labels, areas | {"changed_pixels", "reference_class_pixels", "candidate_class_pixels", "passed"}, "label metrics")
    for key in areas:
        _number(labels[key], key)
    _integer(labels["changed_pixels"], "changed labels")
    _need(labels["passed"] is True and labels["changed_pixels"] == 0
          and all(labels[key] == 0 for key in areas if key.startswith("changed_"))
          and labels["reference_class_pixels"] == labels["candidate_class_pixels"], "Original exact label metrics differ")
    counts = labels["reference_class_pixels"]
    _need(type(counts) is dict and counts and all(type(k) is str and k.isdecimal() for k in counts), "Original class counts required")
    for count in counts.values():
        _integer(count, "class count")
    _need(sum(counts.values()) == value["frame_pixels"], "Class counts do not cover original image")
    _need(type(value["scalars"]) is dict and set(DOMAIN_SCALARS) <= set(value["scalars"]), "Original full-rig/Box1/Box2 mass scalars required")
    for row in value["scalars"].values():
        _need(type(row) is dict, "Original scalar row required")
        nullable = row.get("reference") is None or row.get("candidate") is None
        keys = {"reference", "candidate", "difference", "limit", "passed"} | ({"numeric_checked"} if nullable else set())
        _exact(row, keys, "scalar metrics")
        _need(row["passed"] is True and type(row["reference"]) is type(row["candidate"])
              and row["reference"] == row["candidate"], "Original exact scalar mismatch")
        if nullable:
            _need(row["reference"] is row["candidate"] is row["difference"] is row["limit"] is None
                  and row["numeric_checked"] is False, "Original null scalar semantics differ")
        else:
            _need(type(row["reference"]) in (int, float) and type(row["candidate"]) in (int, float), "Actual scalar numbers required")
            _number(row["difference"], "scalar difference")
            _number(row["limit"], "scalar limit")
            _need(row["difference"] == 0, "Strict scalar difference must be zero")
    _need(type(value["flags"]) is dict, "Original flags object required")
    for row in value["flags"].values():
        _exact(row, {"reference", "candidate", "different", "reported_only"}, "threshold flag")
        _need(type(row["reference"]) is type(row["candidate"]) and row["reference"] == row["candidate"]
              and row["different"] is False and row["reported_only"] is True, "Original threshold flags differ")
    identities = value["geometry_array_identities"]
    _need(type(identities) is list and len(identities) == 2 and identities[0] == identities[1]
          and type(identities[0]) is dict and set(identities[0]) == set(GEOMETRY), "Both exact original geometry identities required")
    for key, identity in identities[0].items():
        _exact(identity, {"dtype", "shape", "sha256"}, "original geometry array identity")
        _hex(identity["sha256"], 64, "geometry SHA")
        _need(type(identity["dtype"]) is str and bool(identity["dtype"])
              and type(identity["shape"]) is list and len(identity["shape"]) == 2
              and all(type(n) is int and n > 0 for n in identity["shape"])
              and math.prod(identity["shape"]) == value["frame_pixels"], "Actual original geometry shape/dtype required")
    stages = {}
    for host, original in results.items():
        _need(original.get("case_id") == case_id and original.get("complete") is True
              and original.get("frame_index") == value["frame_index"]
              and original.get("backend") == "cpu" and original.get("precision") == "float32", "Original result case/scope differs")
        files = original.get("prediction", {}).get("pipeline", {}).get("files")
        _need(type(files) is dict and files, "Original retained stage inventory required")
        stages[host] = {name: pin for name, pin in files.items() if PureWindowsPath(name).suffix in {".npy", ".npz", ".png"}}
    _need(stages["Reference"] and stages["Candidate"], "Original retained stage checks cannot be empty")
    _strict_products(value["strict_N56_exact_products"], source_pair=source_pair, expected_stage_files=stages)
    for row in value["strict_N56_exact_products"]["comparisons"][1]["arrays"]:
        identity = identities[0][row["key"]]
        _need(row["shape"] == identity["shape"] and row["dtype"] == identity["dtype"], "N56 geometry metadata differs from original identity")
    return deepcopy(value)


def checked_strict39_report(report, *, contract, release, read):
    """Validate original emitted DATA, not payloads, OS retirement or authority."""
    contract = checked_predeclared_contract_data(contract)
    release = checked_postproducer_release_data(release)
    _json(report)
    fields = {"complete", "utc", "format", "comparison_label", "profile_or_worker_qualified", "actual_host_commits",
        "scientific_source_equivalence", "case_count", "ordered_case_ids", "comparisons", "frame_add8_limits_passed",
        "all39_strict_FFAC_bit_identical", "strict_acceptance", "numeric_checker", "comparison_kind", "reference_completed",
        "candidate_completed", "environment_identity_proof", "candidate_closed_readset", "source_relationship",
        "independent_scientific_qualification", "Wasserstein_performed", "Wasserstein_qualification",
        "production_permission_granted", "eighty_GO", "new_scientific_arrays_computed", "native_read_closure",
        "copy_locator_closure", "accepted_prior_comparisons", "new_comparisons_count"}
    _exact(report, fields, "original strict39 report")
    _need(report["format"] == contract["report_format"] and report["comparison_kind"] == "same_precision"
          and report["ordered_case_ids"] == list(CASE_IDS) and report["accepted_prior_comparisons"] == []
          and type(report["case_count"]) is int and report["case_count"] == 39
          and type(report["new_comparisons_count"]) is int and report["new_comparisons_count"] == 39, "Fresh original ordered39 report required")
    for field in ("complete", "frame_add8_limits_passed", "all39_strict_FFAC_bit_identical"):
        _need(report[field] is True, "Original strict39 verdict missing")
    for field in ("profile_or_worker_qualified", "independent_scientific_qualification", "Wasserstein_performed",
                  "production_permission_granted", "eighty_GO", "new_scientific_arrays_computed"):
        _need(report[field] is False, "Comparator cannot grant authority or broader qualification")
    _need(report["numeric_checker"] == contract["sources"]["numeric_checker"]
          and report["source_relationship"] == contract["source_pair"]["relationship"]
          and report["actual_host_commits"] == {h: contract["source_pair"][h]["commit"] for h in ("Reference", "Candidate")}, "Actual original source pair differs")
    completed = {}
    for host, role in (("Reference", "reference_completed"), ("Candidate", "candidate_completed")):
        pin = release["producers"][host]["completion"]
        _need(report[role] == pin, "Report names another original completion")
        completed[host] = read(descriptor(pin))
        doc = completed[host]
        _need(type(doc) is dict and doc.get("complete") is True and doc.get("whole_run") is True
              and doc.get("worker_references_closed") is True
              and all(type(doc.get(key)) is int and doc[key] == 39 for key in ("newly_computed_frames", "frames", "case_count"))
              and doc.get("reused_case_ids") == [] and doc.get("ordered_case_ids") == list(CASE_IDS)
              and [r.get("frame_id") for r in doc.get("records", [])] == list(CASE_IDS)
              and [r.get("case_id") for r in doc.get("frame_producers", [])] == list(CASE_IDS), "Original fresh39 completion DATA differs")
        _need(doc.get("commit") == contract["source_pair"][host]["commit"]
              and doc.get("pipeline") == "native-phase-n1" and doc.get("backend") == "cpu"
              and doc.get("precision") == "float32", "Original completion source/scope DATA differs")
    rows = report["comparisons"]
    _need(type(rows) is list and len(rows) == 39, "Exactly39 original comparison receipts required")
    for index, (case, row) in enumerate(zip(CASE_IDS, rows)):
        _exact(row, {"case_id", "passed", "first_failure", "receipt"}, "case receipt index")
        _need(row["case_id"] == case and row["passed"] is True and row["first_failure"] is None, "Original comparison order/verdict differs")
        actual = read(descriptor(row["receipt"]))
        results = {}
        for host, role in (("Reference", "reference_result"), ("Candidate", "candidate_result")):
            pin = completed[host]["frame_producers"][index]["result"]
            _need(actual.get(role) == descriptor(pin), "Case receipt names another original result")
            results[host] = read(pin)
        checked_strict39_case(actual, case_id=case, source_pair=contract["source_pair"], results=results)
    return deepcopy(report)


def checked_role_bindings(roles, *, read):
    """Consumed pre-STOP metadata path. None means absent, never accepted.

    ``read`` must be the caller's existing bounded exact descriptor reader. No
    authority, retirement or Build document body is interpreted by this module.
    The caller must retain its unconditional original-admission STOP afterwards.
    """
    if roles is None:
        roles = absent_role_bindings()
    _exact(roles, ROLES, "prospective roles")
    validators = (checked_accepted_reference_data, checked_predeclared_contract_data,
                  checked_postproducer_release_data, checked_strict_acceptance_data)
    documents = {}
    for index, (role, check) in enumerate(zip(ROLES, validators)):
        pin = roles[role]
        if pin is None:
            _need(all(roles[later] is None for later in ROLES[index + 1:]), "Later DATA requires all predecessor roles")
            continue
        documents[role] = check(read(descriptor(pin)))
    if "predeclared_contract" in documents:
        reference, contract = documents["accepted_reference"], documents["predeclared_contract"]
        _need(contract["accepted_reference"] == roles["accepted_reference"] and contract["current_plan"] == reference["current_plan"]
              and contract["scope"] == reference["scope"], "Predeclared DATA differs from reference role")
    if "postproducer_release" in documents:
        release = documents["postproducer_release"]
        _need(release["predeclared_contract"] == roles["predeclared_contract"] and release["accepted_reference"] == roles["accepted_reference"]
              and release["producers"]["Reference"] == reference["producer"]
              and release["producers"]["Candidate"]["task"] == contract["candidate_task"]
              and release["scope"] == contract["scope"], "Postproducer DATA differs from predeclared roles")
    if "strict_acceptance" in documents:
        acceptance = documents["strict_acceptance"]
        _need(acceptance["postproducer_release"] == roles["postproducer_release"] and acceptance["predeclared_contract"] == roles["predeclared_contract"]
              and acceptance["actual_command"] == contract["public_command"] and acceptance["scope"] == contract["scope"], "Strict acceptance DATA names another released comparison")
        checked_strict39_report(read(acceptance["report"]), contract=contract, release=release, read=read)
    return dict(state=STATE, present_roles=list(documents), absent_roles=[role for role in ROLES if role not in documents],
                metadata_only=True, authority_granted=False, original_admission_available=False,
                missing_original_verifiers=list(MISSING_ORIGINAL_VERIFIERS))
