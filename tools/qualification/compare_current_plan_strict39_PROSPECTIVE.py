"""Prospective current-plan RAW39 comparator; SOURCE PREPARATION ONLY.

No released current-V9 accepted-reference schema or complete Build-to-task
verifier exists in Q432 9486ebfe. Every public/direct comparison entry stops
before loading helper code, numerical imports, product reads or output writes.
The retained original strict scientific checks are unreachable pending a new
independently reviewed original-verifier implementation and actual Root release.
No request document, environment flag or caller-supplied object can lift STOP.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import socket
import sys
import traceback
import types
import copy
import re
import subprocess
import tomllib

# Bound only after future original admission; no historical identity is retargeted.
COMMIT = None
HOST_COMMITS = {"Reference": None, "Candidate": None}
SOURCE_EQUIVALENCE_SHA256 = None
SERIES_ROOT = None
HISTORICAL_COMPARATOR_SHA256 = "06ac1a5cdef18f71de623b9fa83f2b89645b56d934189a8c8a28e247c0089aa2"
DELIVERED_SOURCE = "dc0affc84262acf5fc7bcbf03e29e7b54865f384"
Q432_SOURCE = "9486ebfe3a3ddfa123613d225709b9925ce2008e"
REQUEST_FORMAT = "darsia-prospective-current-plan-strict39-request-v1"
PLAN_FORMAT = "darsia-current-v9-reference-plan-v1"
# These are immutable original scientific readers, not new acceptance providers.
ORIGINAL_HELPER_BYTES = {
    "numeric_checker": ("9949a184fef7e3b55c75336544fa04f52a2062f1b37cfdad1018f2243280b5ce", 17239),
    "record_extractors": ("95dc41bf655c5d22a91dd5bf7cf55430f0f5a511d8d7cdabe8f8e68378b6aa13", 21587),
    "read_locator": ("ba09417f90f79b0d6ce4a96ac93d9a9f8969bc2b1daa37dbd219f3ed718d758e", 8067),
    "typed_ipp_validator": ("a7f2550125fb102d1e297c7d72d2de01594fe91640d78a3e72869314bcdedb15", 3901),
}
CASES = ["ac52_f" + str(i) for i in
         (1,2,3,4,5,6,7,9,10,11,12,13,14,16,17,18,19,20,21,23,25,26,27,
          28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43)]
FLOATS = ("concentration_aqueous", "saturation_free_gas", "mass_total",
          "mass_aqueous", "mass_free_gas")
GEOMETRY = ("pixel_area_m2", "depth_m", "porosity", "facies", "transport_weight_l")
GIB = 1024 ** 3


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def key(path):
    return str(PureWindowsPath(path)).casefold()


def version(path):
    s = path.stat(follow_symlinks=False)
    if getattr(s, "st_file_attributes", 0) & 0x400:
        raise ValueError("Reparse input refused: " + str(path))
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns,
            getattr(s, "st_file_attributes", 0)]


def metadata(bound, limit=32 * 1024 ** 2):
    expected = bound.get("size_bytes", bound.get("bytes"))
    p = Path(bound["path"])
    before = version(p)
    if type(expected) is not int or not 0 <= expected <= limit or before[2] != expected:
        raise ValueError("Exact finite control length differs: " + str(p))
    with p.open("rb") as f:
        b = f.read(expected + 1)
    if (len(b) != expected or hashlib.sha256(b).hexdigest() != bound["sha256"]
            or before != version(p)):
        raise ValueError("Exact control SHA/version differs: " + str(p))
    return b


def document(bound):
    return _strict_json(metadata(bound))


def load(bound, name):
    b = metadata(bound)
    m = types.ModuleType(name)
    m.__file__ = bound["path"]
    sys.modules[name] = m
    exec(compile(b, bound["path"], "exec"), m.__dict__)
    return m


def write(path, value):
    b = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    with path.open("xb") as f:
        f.write(b)
        f.flush()
        os.fsync(f.fileno())
    return dict(path=str(path), sha256=hashlib.sha256(b).hexdigest(), size_bytes=len(b))


class CurrentReferenceBindingStop(ValueError):
    """Missing original authority/evidence verifier; never a comparison verdict."""


MISSING_ORIGINAL_VERIFIERS = (
    "CURRENT_V9_TYPED_ACCEPTED_REFERENCE_AND_COMPARISON_RELEASE_UNAVAILABLE",
    "AUTHENTICATED_BUILD_PUBLICATION_TO_QUEUED_FRAME_GEOMETRY_JOIN_UNIMPLEMENTED",
    "BOTH_ORIGINAL_PUBLICATIONS_NATIVE_FAMILY_RETIREMENTS_AND_INDEPENDENCE_UNBOUND",
    "CURRENT_SOURCE_RUNTIME_PAIR_COMPATIBILITY_AND_REGISTERED_LOCATORS_UNBOUND",
)


def _exact(value, fields, role):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("Closed prospective schema required: " + role)
    return value


def _pin(value, role):
    _exact(value, {"path", "sha256", "size_bytes"}, role)
    path = value["path"]
    if (not isinstance(path, str) or not PureWindowsPath(path).is_absolute()
            or ".." in PureWindowsPath(path).parts
            or not isinstance(value["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
            or type(value["size_bytes"]) is not int or value["size_bytes"] < 0):
        raise ValueError("Exact absolute descriptor required: " + role)
    return value


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Duplicate JSON control key: " + name)
        result[name] = value
    return result


def _strict_json(raw):
    def invalid(value):
        raise ValueError("Nonfinite JSON control value: " + value)
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            invalid(value)
        return result
    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=invalid,
                      parse_float=finite_float)


def _plan_digest(value):
    """Exact Q432 _canonical_json/_digest encoding; no acceptance claim."""
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def checked_prospective_request(value):
    """Check closed DATA shape only. This function never accepts a reference.

    No invented acceptance/completion format replaces the original publication
    readers. The opaque descriptor roles remain unavailable evidence until the
    original missing verifiers are implemented and independently reviewed.
    """
    pins = {"current_plan", "registered_series_manifest", "reference_acceptance",
        "Root_release", "reference_completed", "candidate_completed",
        "reference_retirement", "candidate_retirement", "reference_native_completion",
        "candidate_native_completion", "reference_source_runtime", "candidate_source_runtime",
        "environment_identity_proof", "scientific_source_equivalence",
        "reference_closed_readset", "candidate_closed_readset", "frozen_calibration"}
    optional = {"strict39_roles"} if isinstance(value, dict) and "strict39_roles" in value else set()
    _exact(value, pins | {"format", "source_pair", "host_tasks", "host_execution_series",
        "source_pins", "candidate_attempt_directory", "candidate_copy_root",
        "product_readset", "output", "accepted_prior_comparisons"} | optional, "request")
    if value["format"] != REQUEST_FORMAT or value["accepted_prior_comparisons"] != []:
        raise ValueError("Prospective full39 request required; no prior comparisons")
    for role in pins:
        _pin(value[role], role)
    roles = value.get("strict39_roles")
    if roles is not None:
        _exact(roles, {"accepted_reference", "predeclared_contract",
                       "postproducer_release", "strict_acceptance"}, "strict39 DATA roles")
        for role, bound in roles.items():
            if bound is not None:
                _pin(bound, "strict39." + role)
    pair = _exact(value["source_pair"], {"relationship", "Reference", "Candidate"}, "source pair")
    for host in ("Reference", "Candidate"):
        source = _exact(pair[host], {"commit", "tree"}, host + " source")
        if any(not isinstance(source[k], str)
               or re.fullmatch(r"[0-9a-f]{40}", source[k]) is None for k in source):
            raise ValueError("Both actual full source identities are required")
    same = pair["Reference"]["commit"] == pair["Candidate"]["commit"]
    if pair["relationship"] not in {"same_source_independent_reexecution", "reviewed_source_pair"}:
        raise ValueError("Explicit source relationship required")
    if ((pair["relationship"] == "same_source_independent_reexecution") != same
            or (same and pair["Reference"]["tree"] != pair["Candidate"]["tree"])):
        raise ValueError("Source relationship differs from actual commit/tree identities")
    # Same-source reruns can demonstrate repeatability, not an independent
    # scientific method. Different commit strings also do not prove independence.
    for role in ("host_tasks", "host_execution_series"):
        _exact(value[role], {"Reference", "Candidate"}, role)
        for host, bound in value[role].items():
            _pin(bound, role + "." + host)
    _exact(value["source_pins"], ORIGINAL_HELPER_BYTES, "original helper pins")
    for role, bound in value["source_pins"].items():
        _pin(bound, role)
        if (bound["sha256"], bound["size_bytes"]) != ORIGINAL_HELPER_BYTES[role]:
            raise ValueError("Original checker/reader bytes changed: " + role)
    for a, b in (("reference_completed", "candidate_completed"),
                 ("reference_retirement", "candidate_retirement"),
                 ("reference_native_completion", "candidate_native_completion")):
        if key(value[a]["path"]) == key(value[b]["path"]):
            raise ValueError("Self-comparison evidence path refused: " + a)
    if not isinstance(value["product_readset"], list) or not value["product_readset"]:
        raise ValueError("Explicit complete original product allowlist required")
    seen = set()
    for bound in value["product_readset"]:
        _pin(bound, "product")
        if key(bound["path"]) in seen:
            raise ValueError("Duplicate product allowlist path")
        seen.add(key(bound["path"]))
    for role in ("output", "candidate_attempt_directory", "candidate_copy_root"):
        path = value[role]
        if (not isinstance(path, str) or not PureWindowsPath(path).is_absolute()
                or ".." in PureWindowsPath(path).parts):
            raise ValueError("Absolute original output/read locator required: " + role)
    # The original C-in-place locator remains unchanged. A new locator route
    # needs separate source review; path declarations cannot bless a relocation.
    if key(value["candidate_attempt_directory"]) != key(value["candidate_copy_root"]):
        raise ValueError("Original in-place read locator required")
    return copy.deepcopy(value)


def checked_current_plan_metadata(q, *, read=document):
    """Additional Q432-shaped identity diagnostics, never original admission.

    Keep the exact prospective plan shape from 9486ebfe; do not mistake these
    metadata joins for checked_current_reference_plan, whose unconditional
    Build execution STOP remains required. In particular no supplied Build,
    acceptance, release or retirement document is interpreted as authority.
    """
    plan = read(q["current_plan"])
    _exact(plan, {"format", "authority_ids", "scope", "configuration",
        "configuration_source", "input_catalog_sha256", "current_v9_build",
        "native_parameters", "python_executable", "capacity_bound_plan",
        "requested_native_layout", "output_writers"}, "Q432 current plan")
    scope = dict(pipeline="native-phase-n1", backend="cpu", precision="float32",
                 run="ac52", ordered_case_ids=CASES)
    if plan["format"] != PLAN_FORMAT or plan["scope"] != scope:
        raise ValueError("Exact current-plan CPU/float32/ordered39 scope required")
    ids = plan["authority_ids"]
    if (not isinstance(ids, list) or not ids
            or any(not isinstance(item, str) or not item for item in ids)
            or len(set(ids)) != len(ids) or "Q432" not in ids):
        raise ValueError("Distinct prospective current-plan authority identifiers required")
    for role in ("configuration_source", "current_v9_build", "native_parameters",
                 "python_executable", "capacity_bound_plan"):
        _pin(plan[role], "plan." + role)
    if (not isinstance(plan["input_catalog_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", plan["input_catalog_sha256"]) is None):
        raise ValueError("Exact current-plan catalogue digest required")
    tasks = {host: read(q["host_tasks"][host]) for host in ("Reference", "Candidate")}
    for host, task in tasks.items():
        if (not isinstance(task, dict) or any(task.get(k) != v for k, v in scope.items())
                or type(task.get("threads")) is not int or task["threads"] != 1
                or task.get("commit") != q["source_pair"][host]["commit"]
                or task.get("configuration") != plan["configuration"]
                or not isinstance(task.get("job_id"), str) or not task["job_id"]
                or not isinstance(task.get("attempt_id"), str) or not task["attempt_id"]):
            raise ValueError("Prospective actual task scope/source/configuration differs")
    catalog = tasks["Reference"].get("input_catalog")
    if (not isinstance(catalog, dict) or catalog.get("complete") is not True
            or catalog.get("format") != "darsia-role-input-catalog-v1"
            or catalog.get("configuration") != plan["configuration"]
            or catalog.get("run") != "ac52"
            or not isinstance(catalog.get("inputs"), list) or not catalog["inputs"]
            or _plan_digest(catalog) != plan["input_catalog_sha256"]):
        raise ValueError("Prospective reference task differs from the exact plan catalogue")
    if (tasks["Reference"]["job_id"] == tasks["Candidate"]["job_id"]
            or tasks["Reference"]["attempt_id"] == tasks["Candidate"]["attempt_id"]):
        raise ValueError("Separate genuine producer jobs and attempts are required")
    # These identifiers only reject obvious self-comparison; original publication,
    # physical output identity and native-family proofs must still authenticate it.
    return copy.deepcopy(plan)


def _require_original_current_verifiers(*_unused):
    """Unconditional SOURCE STOP, including direct calls; no truthy escape hatch.

    Minimal original extensions still required:
    * closed independently issued reference acceptance and exact comparison
      release, bound to plan/task/Build/config/source/runtime/completion;
    * Q432 authenticated Build publication, physical/source-map provenance and
      every queued frame join, preserving its existing unconditional STOP;
    * original published_jobs_snapshot._terminal + native lifecycle/OS0 proofs
      for BOTH separate attempts, output physical identities and closed readsets;
    * actual source/runtime before-after closure, registered alias/locator
      validation, source-pair review and unchanged original writer/layout checks.
    After comparison, original OS0/retirement/source closure and an independent
    typed strict39 acceptance must precede any separate profile publication.
    The old nonempty-dictionary acceptance checks do not implement these seams.
    """
    raise CurrentReferenceBindingStop("; ".join(MISSING_ORIGINAL_VERIFIERS))


class Reader:
    def __init__(self, locations, allowlist):
        self.locations, self.allowed, self.seen = locations, {}, {}
        self.hash_read_bytes = 0
        for row in allowlist:
            k = key(row["path"])
            old = self.allowed.setdefault(k, row)
            if old != row:
                raise ValueError("Conflicting admitted product descriptors")

    def verified(self, bound):
        expected = bound.get("size_bytes", bound.get("bytes"))
        canonical = dict(path=bound["path"], sha256=bound["sha256"], size_bytes=expected)
        k = key(canonical["path"])
        if self.allowed.get(k) != canonical:
            raise ValueError("Unlisted or changed product descriptor: " + str(bound))
        p = self.locations.path(canonical["path"])
        before = version(p)
        if k in self.seen:
            if self.seen[k]["version"] != before:
                raise ValueError("Previously read product changed")
            return p
        if type(expected) is not int or expected < 0 or before[2] != expected:
            raise ValueError("Product length differs")
        digest = hashlib.sha256()
        read = 0
        with p.open("rb") as f:
            while b := f.read(8 * 1024 ** 2):
                read += len(b)
                if read > expected:
                    raise ValueError("Product exceeds its admitted length")
                digest.update(b)
        if read != expected or digest.hexdigest() != canonical["sha256"] or before != version(p):
            raise ValueError("Original product SHA/version differs")
        self.hash_read_bytes += read
        self.seen[k] = dict(canonical, version=before)
        return p

    def doc(self, bound):
        p = self.verified(bound)
        if p.stat().st_size > 32 * 1024 ** 2:
            raise ValueError("Finite JSON input exceeds32MiB")
        return json.loads(p.read_bytes())

    def finish(self):
        for row in self.seen.values():
            if version(self.locations.path(row["path"])) != row["version"]:
                raise ValueError("Consumed product changed at closure")
        return dict(files=list(self.seen.values()), hash_read_bytes=self.hash_read_bytes,
                    physical_versions_unchanged=True)


def member(publication, name):
    row = publication["files"][name]
    return dict(path=str(PureWindowsPath(publication["nas_directory"]) / name),
                sha256=row["sha256"], size_bytes=row["size_bytes"])



def _same_json(a, b):
    return (json.dumps(a,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(",",":"))
            == json.dumps(b,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(",",":")))


def _current_task_basis(q):
    """Retain canonical catalogues and exactly nine operational differences."""
    _require_original_current_verifiers()
    plan = document(q["current_plan"])
    tasks, series = {}, {}
    for host in ("Reference","Candidate"):
        task = document(q["host_tasks"][host])
        if (task["commit"] != HOST_COMMITS[host] or task["pipeline"] != "native-phase-n1"
                or task["backend"] != "cpu" or task["threads"] != 1
                or task["precision"] != "float32" or task["ordered_case_ids"] != CASES
                or task["input_catalog"].get("format") != "darsia-role-input-catalog-v1"
                or task["input_catalog"].get("complete") is not True
                or task["configuration"] != plan["configuration"]
                or (host == "Reference" and _plan_digest(task["input_catalog"])
                    != plan["input_catalog_sha256"])):
            raise ValueError("Actual current task/source/input39 scope differs")
        inputs = task["input_catalog"]["inputs"]
        identities = [tuple(row["reference"][field] for field in
                            ("root_alias","relative_path","sha256","role")) for row in inputs]
        if identities != sorted(identities) or len(set(identities)) != len(identities):
            raise ValueError("Actual input catalogue is not unique canonical InputCatalog order")
        roles = [row for row in inputs if "run.execution_series" in row["uses"]]
        if len(roles) != 1:
            raise ValueError("Exactly one execution-series catalogue role required")
        role = roles[0]
        if role["uses"] != ["run.execution_series"] or role["reference"] != task["series"]:
            raise ValueError("Exact execution-series catalogue role differs")
        bound = dict(path=str(PureWindowsPath(SERIES_ROOT)/task["series"]["relative_path"]),
                     sha256=task["series"]["sha256"],size_bytes=task["series"]["bytes"])
        if bound != q["host_execution_series"][host]:
            raise ValueError("Actual host series descriptor differs")
        series[host] = tomllib.loads(metadata(bound).decode("utf-8"))
        if series[host]["job_id"] != task["job_id"] or series[host]["job"]["commit"] != HOST_COMMITS[host]:
            raise ValueError("Actual host series job/source differs")
        tasks[host] = task
    projected_series = {h:copy.deepcopy(v) for h,v in series.items()}
    for value in projected_series.values():
        del value["job_id"]
        del value["job"]["commit"]
    if not _same_json(*projected_series.values()):
        raise ValueError("Execution series differ beyond exact host job_id/validated commit")
    projected = {h:copy.deepcopy(v) for h,v in tasks.items()}
    execution_roles = {}
    for host,value in projected.items():
        del value["commit"]
        del value["job_id"]
        del value["attempt_id"]
        inputs = value["input_catalog"]["inputs"]
        indices = [i for i,row in enumerate(inputs) if row["uses"] == ["run.execution_series"]]
        if len(indices) != 1:
            raise ValueError("Validated execution-series role was not preserved")
        role = inputs.pop(indices[0])
        for row in (value["series"],role["reference"]):
            for field in ("relative_path","sha256","bytes"):
                del row[field]
        execution_roles[host] = role
    # Removing only this unique operational row preserves every other input's
    # original canonical position, full descriptor, use list and list order.
    if (not _same_json(*execution_roles.values())
            or not _same_json(*projected.values())):
        raise ValueError("Scientific task differs beyond exact9 operational fields and source-byte proof")
    return tasks


def _pair(value, ipp, *, mass=False):
    required = {"policy","before","after","saved_inputs"}
    if mass:
        required.add("mass_evaluation")
    if not isinstance(value,dict) or set(value) != required:
        raise ValueError("Exact current before/after IPP schema required")
    ipp.validate_observation(value)
    if value["saved_inputs"] != "original provenance; no retroactive IPP observation":
        raise ValueError("Current IPP provenance marker differs")
    if mass:
        _pair(value["mass_evaluation"],ipp)


def _current_completion(c, host, task, reader, ipp):
    if (c.get("pipeline") != "native-phase-n1" or c.get("whole_run") is not True
            or c.get("newly_computed_frames") != 39 or c.get("reused_case_ids") != []
            or c.get("task") != task or c.get("attempt_id") != task["attempt_id"]
            or c.get("configuration") != task["configuration"]
            or c.get("calibration_sha256") != task["calibration_sha256"]
            or c["calibration"]["sha256"] != task["calibration_sha256"]
            or [p["case_id"] for p in c.get("frame_producers",[])] != CASES):
        raise ValueError("Actual current whole39 task/producers required; no prior replay")
    if c["actual_worker"]["machine"] != "main-cpu":
        raise ValueError("Actual public managed worker required on both sides")
    ipp.validate_observation(c["opencv_ipp"])
    worker = c["opencv_ipp_worker"]
    if set(worker) != {"start","completed"}:
        raise ValueError("Current worker IPP boundaries required")
    for value in worker.values():
        ipp.validate_observation(value)
    manifest = reader.doc(c["manifest"])
    if (manifest["code"]["commit"] != HOST_COMMITS[host]
            or manifest["code_sha256"] != hashlib.sha256(HOST_COMMITS[host].encode("ascii")).hexdigest()
            or manifest["settings"]["host"].casefold() != {"Reference":"reference-host","Candidate":"reference-host"}[host]
            or manifest["settings"]["backend"] != "cpu"
            or manifest["settings"]["precision"] != "float32"
            or manifest["configuration"] != task["configuration"]):
        raise ValueError("Actual per-host manifest/source/runtime binding differs")
    return manifest


def _current_frame(c, pub, record, commit, index, reader, ipp):
    """Verify raw publication/receipts before any archive decode."""
    producer = c["frame_producers"][index]
    if (producer["manifest"] != c["manifest"]
            or producer["result"] != member(pub,"result.json")
            or producer["commit"] != member(pub,"commit.json")):
        raise ValueError("Actual frame producer descriptors differ")
    registered = reader.doc(producer["frame_registration"])
    if (reader.doc(producer["publication"]) != pub
            or registered["frame_id"] != CASES[index]
            or registered["manifest_sha256"] != pub["manifest_sha256"]
            or registered["commit"] != producer["commit"]
            or registered["files"] != commit["files"]):
        raise ValueError("Actual immutable publication/registration differs")
    stage = record["prediction"]["pipeline"]
    spec = reader.doc(c["manifest"])["frames"][CASES[index]]
    if (stage["source"] != spec["source"]
            or record["source_binding"] != spec["source"]["source_binding"]
            or record["frame_index"] != spec["frame_index"]
            or record["elapsed_h"] != spec["elapsed_h"]
            or record["bindings"]["source_sha256"] != spec["source"]["source_sha256"]
            or stage["native"]["source_sha256"] != spec["source"]["source_sha256"]
            or stage["phase"]["source_sha256"] != spec["source"]["source_sha256"]):
        raise ValueError("Actual per-host frame/native/phase acquisition differs from manifest")
    if (stage["format"] != "darsia-series-frame-v1" or stage["complete"] is not True
            or stage["case_id"] != CASES[index]
            or stage["configuration_sha256"] != c["configuration"]["sha256"]
            or stage["numerical_phase_calls"] != 1 or "probabilities" in stage
            or record["opencv_ipp"] != stage["opencv_ipp"]
            or record["opencv_ipp"] != commit["opencv_ipp"]):
        raise ValueError("Actual native-phase-n1 stage/IPP binding differs")
    _pair(record["opencv_ipp"],ipp,mass=True)
    _pair(stage["native"]["opencv_ipp"],ipp)
    native_audit = reader.doc(stage["native"]["audit"])
    _pair(native_audit["opencv_ipp"],ipp)
    phase_qa = reader.doc(stage["phase"]["qa"])
    _pair(phase_qa["opencv_ipp"],ipp)
    for bound in stage["files"].values():
        reader.verified(bound)
    if record["bindings"]["phase_sha256"] != stage["phase"]["labels"]["sha256"]:
        raise ValueError("Actual frame consumes another phase-label output")
    return dict(producer=producer,opencv_ipp=record["opencv_ipp"],
                native_opencv_ipp=stage["native"]["opencv_ipp"],
                phase_opencv_ipp=phase_qa["opencv_ipp"],raw_records_not_projected=True)


def _n56_array_sha(value, np):
    """Existing N56 array_sha primitive, same body; numpy is supplied explicitly."""
    h=hashlib.sha256()
    for part in np.nditer(value,flags=["external_loop","buffered","zerosize_ok"],op_flags=["readonly"],order="C",buffersize=262144):
        h.update(memoryview(np.ascontiguousarray(part)).cast("B"))
    return h.hexdigest()


def _strict_diagnostics(row, geometry_identities, scalar_exact, flags_exact, quantities_exact):
    """Additional exact acceptance; original Add8 values/limits remain untouched."""
    maps = row["maps"]
    return dict(
        all_five_maps_bit_identical=(set(maps) == set(FLOATS) and all(maps[k]["bit_identical"] is True for k in FLOATS)),
        phase_labels_changed_pixels_zero=row["labels"]["changed_pixels"] == 0,
        native_geometry_array_identities_exact=geometry_identities[0] == geometry_identities[1],
        original_scalar_result_JSON_exact=scalar_exact,
        original_threshold_flags_exact=flags_exact,
        all_extracted_scientific_quantities_exact=quantities_exact)


def _strict_n56_products(np, reader, publications, records_pair):
    """N56 scientific products stay exact; code provenance binds each actual commit.

    Every NPZ name/order, scientific dtype/shape/raw array SHA, scalar result,
    native-statistics JSON content and retained stage byte check stays exact.
    The sole operational field products.npz/code_sha256 is validated per host.
    Its original values and cross-side inequality remain visible in the report.
    """
    import gzip
    comparisons = []
    for name in ("products.npz", "native_geometry.npz"):
        paths = [reader.verified(member(p, name)) for p in publications]
        rows = []
        with np.load(paths[0], allow_pickle=False) as a, np.load(paths[1], allow_pickle=False) as b:
            if a.files != b.files:
                raise ValueError("N56 exact product array names/order differ: " + name)
            if name == "products.npz" and "code_sha256" not in a.files:
                raise ValueError("Required per-host products.npz/code_sha256 is missing")
            for key in a.files:
                x, y = a[key], b[key]
                exact = x.dtype == y.dtype and x.shape == y.shape and _n56_array_sha(x, np) == _n56_array_sha(y, np)
                row = dict(key=key, shape=list(x.shape), dtype=x.dtype.str, bit_exact=exact)
                if name == "products.npz" and key == "code_sha256":
                    identities = []
                    for host, value, record in zip(("Reference", "Candidate"), (x, y), records_pair):
                        actual_commit = HOST_COMMITS[host]
                        expected = hashlib.sha256(actual_commit.encode("ascii")).hexdigest()
                        if (value.shape != () or value.dtype != np.dtype("<U64")
                                or value.item() != expected
                                or record["bindings"]["code_sha256"] != expected):
                            raise ValueError("N56 products.npz/code_sha256 differs from its actual " + host + " commit")
                        identities.append(dict(host=host, actual_commit=actual_commit,
                            stored_code_sha256=value.item(), expected_code_sha256=expected,
                            shape=list(value.shape), dtype=value.dtype.str,
                            raw_array_sha256=_n56_array_sha(value, np)))
                    row.update(per_host_actual_commit_identity_valid=True, per_host_identities=identities)
                elif not exact:
                    raise ValueError("N56 bit-exact product failure: " + name + "/" + key)
                rows.append(row)
        comparisons.append(dict(role=name, arrays=rows,
            bit_exact=all(row["bit_exact"] for row in rows),
            all_scientific_arrays_bit_exact=True,
            per_host_code_identity_validated=(name == "products.npz")))
    if not _same_json(records_pair[0]["scalar_result"], records_pair[1]["scalar_result"]):
        raise ValueError("N56 original scalar_result JSON is not exact")
    paths = [reader.verified(member(p, "native_statistics.json.gz")) for p in publications]
    with gzip.open(paths[0], "rt", encoding="utf-8") as a, gzip.open(paths[1], "rt", encoding="utf-8") as b:
        if not _same_json(json.load(a), json.load(b)):
            raise ValueError("N56 native_statistics JSON content is not exact")
    stages = [r["prediction"]["pipeline"]["files"] for r in records_pair]
    if set(stages[0]) != set(stages[1]):
        raise ValueError("N56 retained stage product roles differ")
    for name, bound in stages[0].items():
        other = stages[1][name]
        if PureWindowsPath(name).suffix not in {".npy", ".npz", ".png"}:
            continue
        for value in (bound, other):
            reader.verified(value)
        exact = (bound["sha256"], bound.get("size_bytes",bound.get("bytes"))) == (other["sha256"], other.get("size_bytes",other.get("bytes")))
        comparisons.append(dict(role=name, bit_exact=exact, reference=bound, candidate=other))
        if not exact:
            raise ValueError("N56 original stage product bytes differ: " + name)
    return dict(all_products_bit_exact=all(row["bit_exact"] for row in comparisons),
        all_scientific_product_arrays_bit_exact=True, products_code_sha256_per_host_valid=True,
        scalar_result_exact=True, native_statistics_JSON_content_exact=True,
        all_retained_stage_NPY_NPZ_PNG_bytes_exact=True, comparisons=comparisons,
        no_scientific_content_projected=True, code_identity_values_not_normalized=True)


def check_current_source_pair(_proof):
    """No historical provider/count/Part0 boolean can admit current source bytes."""
    _require_original_current_verifiers()


def check_candidate_retirement(q, candidate):
    """Require the actual managed scientific process/family's external exit proof."""
    retired = document(q["candidate_retirement"])
    process = retired["process"]
    actual = candidate["actual_worker"]
    if (retired.get("complete") is not True or retired.get("all_absent") is not True
            or type(process.get("exit_code")) is not int or process["exit_code"] != 0
            or process["host"].casefold() != "reference-host"
            or type(process.get("pid")) is not int
            or process["pid"] != candidate["opencv_ipp_worker"]["start"]["pid"]
            or process["pid"] != candidate["opencv_ipp_worker"]["completed"]["pid"]
            or process["pid"] != actual["pid"]
            or float(process["create_time"]) <= 0
            or float(process["create_time"]) != float(actual["process_create_time"])):
        raise ValueError("Actual candidate OS0 and complete same-epoch native family retirement required")


def run(request_path):
    """Only CLI admission path: bounded stdlib metadata, then unconditional STOP."""
    path = Path(request_path)
    before = version(path)
    if before[2] > 32 * 1024 ** 2:
        raise ValueError("Prospective request exceeds32MiB")
    with path.open("rb") as stream:
        raw = stream.read(32 * 1024 ** 2 + 1)
    if len(raw) != before[2] or before != version(path):
        raise ValueError("Prospective request changed during its bounded read")
    q = checked_prospective_request(_strict_json(raw))
    checked_current_plan_metadata(q)
    if "strict39_roles" in q:
        from darsia_cache.current_strict39 import checked_role_bindings
        checked_role_bindings(q["strict39_roles"], read=document)
    _require_original_current_verifiers(q)
    # No current release or accepted-reference object is fabricated here.
    return _compare_after_original_admission(q, request_path)


def _compare_after_original_admission(q, request_path):
    """Retained original scientific comparison, inaccessible before real admission."""
    _require_original_current_verifiers(q)
    # A future reviewed original admission must bind these actual identities.
    # The source constants remain None in this source-only increment.
    source_equivalence = document(q["scientific_source_equivalence"])
    for bound in q["source_pins"].values():
        metadata(bound)
    for name in ("reference_completed","candidate_completed","candidate_closed_readset",
                 "candidate_attempt_directory","candidate_copy_root","product_readset"):
        if q.get(name) is None:
            raise ValueError("Actual RAW39 completion/closed-readset REQUIRED_PENDING: "+name)
    if q.get("accepted_prior_comparisons") != []:
        raise ValueError("All39 new comparisons required; no prior replay")
    tasks = _current_task_basis(q)
    reference = document(q["reference_completed"])
    candidate = document(q["candidate_completed"])
    check_candidate_retirement(q, candidate)
    for c, host in [(reference,"Reference"),(candidate,"Candidate")]:
        if (c.get("complete") is not True or c.get("worker_references_closed") is not True
                or c["commit"] != HOST_COMMITS[host] or c["backend"] != "cpu" or c["precision"] != "float32"
                or c["ordered_case_ids"] != CASES or c["case_count"] != 39
                or [r["frame_id"] for r in c["records"]] != CASES
                or c["actual_worker"]["machine"] != "main-cpu"):
            raise ValueError("Actual RAW39 complete39/library-or-reference/CPU32 scope differs")
    for name in ["configuration", "calibration", "physical_correction", "anchor"]:
        if reference[name] != candidate[name]:
            raise ValueError("Original same-method binding differs: " + name)
    transfer = document(q["candidate_closed_readset"])
    files = transfer["files"]
    if (transfer.get("complete") is not True
            or transfer["unique_files"] != len(files)
            or len({key(row["original_literal_path"]) for row in files}) != len(files)
            or transfer["unique_bytes"] != sum(row["size_bytes"] for row in files)
            or key(transfer["original_attempt_directory"]) != key(q["candidate_attempt_directory"])
            or key(transfer["physical_attempt_root"]) != key(q["candidate_copy_root"])
            or transfer["original_completed"] != q["candidate_completed"]
            or {key(p["attempt_directory"]) for p in candidate["frame_producers"]}
               != {key(q["candidate_attempt_directory"])}):
        raise ValueError("Actual closed in-place whole39 readset differs")
    if (key(q["candidate_attempt_directory"]) != key(q["candidate_copy_root"])
            or any(key(row["original_literal_path"]) != key(row["physical_read_locator"]) for row in files)):
        raise ValueError("Closed in-place readset must be literal = physical")
    locmod = load(q["source_pins"]["read_locator"], "_n43_existing_locator")
    locations = locmod.ExactReadLocations(literal_root=q["candidate_attempt_directory"],
        physical_root=transfer["physical_attempt_root"],files=files,
        directories=transfer["directories"])
    numeric = load(q["source_pins"]["numeric_checker"], "_n43_unchanged_add8v3")
    records = load(q["source_pins"]["record_extractors"], "_n43_original_record_extractors")
    reader = Reader(locations, q["product_readset"])
    ipp = load(q["source_pins"]["typed_ipp_validator"], "_n43_actual_typed_ipp")
    manifests = [_current_completion(c,h,tasks[h],reader,ipp)
                 for c,h in ((reference,"Reference"),(candidate,"Candidate"))]
    freeze = document(q["frozen_calibration"])
    output = Path(q["output"])
    if output.parent != Path(request_path).parent or output.exists():
        raise ValueError("Fresh sibling output required")
    output.mkdir()
    import numpy as np
    import psutil
    results = []
    prior = []
    try:
        for index, (pa,pb) in enumerate(zip(reference["records"],candidate["records"])):
            _require_original_current_verifiers(q)
            if psutil.virtual_memory().available < 40*GIB or psutil.Process().memory_info().rss > 20*GIB:
                raise ValueError("20GiB comparison/40GiB available reserve bound exceeded")
            case_id = CASES[index]
            ba,bb = member(pa,"result.json"),member(pb,"result.json")
            a,b = reader.doc(ba),reader.doc(bb)
            ca,cb = reader.doc(member(pa,"commit.json")),reader.doc(member(pb,"commit.json"))
            for record,pub,commit in [(a,pa,ca),(b,pb,cb)]:
                if (record.get("complete") is not True or record["case_id"] != case_id
                        or record["precision"] != "float32" or record["backend"] != "cpu"
                        or commit.get("complete") is not True or commit["frame_id"] != case_id
                        or commit["manifest_sha256"] != pub["manifest_sha256"]):
                    raise ValueError("Actual frame/commit provenance differs")
                for name in ["result.json","products.npz","native_geometry.npz","native_statistics.json.gz"]:
                    if commit["files"][name] != pub["files"][name]:
                        raise ValueError("Committed member descriptor differs")
            for name in ["frame_index","elapsed_h","selected_target_h","source_binding","anchor","physical_correction","units"]:
                if a[name] != b[name]:
                    raise ValueError("Same-acquisition scientific binding differs: " + name)
            # The original producer records the generated labels artifact SHA here.
            # It is an output identity, not a cross-host acquisition equality gate.
            for record in (a, b):
                if (record["bindings"]["phase_sha256"]
                        != record["prediction"]["pipeline"]["phase"]["labels"]["sha256"]):
                    raise ValueError("Host result consumes a different saved phase output")
            for host, record in (("Reference", a), ("Candidate", b)):
                if record["bindings"]["code_sha256"] != hashlib.sha256(HOST_COMMITS[host].encode("ascii")).hexdigest():
                    raise ValueError("Actual per-host result code hash differs")
            if ({k:v for k,v in a["bindings"].items() if k not in ("phase_sha256", "code_sha256")}
                    != {k:v for k,v in b["bindings"].items() if k not in ("phase_sha256", "code_sha256")}):
                raise ValueError("Original frame source/calibration/code bindings differ")
            host_provenance = [_current_frame(c,pub,record,commit,index,reader,ipp)
                for c,pub,record,commit in ((reference,pa,a,ca),(candidate,pb,b,cb))]
            products, geometries, identities = [], [], []
            for pub,record in [(pa,a),(pb,b)]:
                product = member(pub,"products.npz")
                if (product["sha256"],product["size_bytes"]) != (record["products"]["sha256"],record["products"]["size_bytes"]):
                    raise ValueError("Actual product/result binding differs")
                with np.load(reader.verified(product),allow_pickle=False) as archive:
                    products.append({k:archive[k] for k in (*FLOATS,"phase_labels")})
                    if products[-1]["phase_labels"].dtype != np.uint8:
                        raise ValueError("Original uint8 phase labels required")
                with np.load(reader.verified(member(pub,"native_geometry.npz")),allow_pickle=False) as archive:
                    geo = {k:archive[k] for k in GEOMETRY}
                ids = {k:records.array_identity(v) for k,v in geo.items()}
                if ids != {k:record["static"]["arrays"][k] for k in GEOMETRY}:
                    raise ValueError("Actual geometry differs from saved array identities")
                geometries.append(geo)
                identities.append(ids)
            if a["scalar_result"]["parameters"] != freeze["parameters"] or b["scalar_result"]["parameters"] != freeze["parameters"]:
                raise ValueError("Frozen parameters differ")
            binding_a = dict(input_bindings={case_id:a["source_binding"]},
                calibration={"selected":reference["calibration"]},
                # 'commit' retains the admitted reference's actual method/source identity.
                # The actual candidate commit is verified by task, manifest and record bindings.
                frozen_method={"commit":COMMIT,"scientific_source_equivalence_sha256":SOURCE_EQUIVALENCE_SHA256,"physical_correction":reference["physical_correction"]})
            binding_b = dict(input_bindings={case_id:b["source_binding"]},
                calibration={"selected":candidate["calibration"]},
                # 'commit' retains the admitted reference's actual method/source identity.
                # The actual candidate commit is verified by task, manifest and record bindings.
                frozen_method={"commit":COMMIT,"scientific_source_equivalence_sha256":SOURCE_EQUIVALENCE_SHA256,"physical_correction":candidate["physical_correction"]})
            if not _same_json(binding_a,binding_b):
                raise ValueError("Separately derived scientific numeric bindings differ")
            row = numeric.compare_retained(a["scalar_result"],b["scalar_result"],*products,
                reference_settings=dict(host="Main",backend=a["scalar_result"]["pixel_backend"],precision="float32"),
                candidate_settings=dict(host="Main",backend=b["scalar_result"]["pixel_backend"],precision="float32"),
                reference_bindings=binding_a,candidate_bindings=binding_b,
                pixel_area_cm2=geometries[0]["pixel_area_m2"]*1e4,kind="same_precision",
                reference_quantities=records.quantities(a),candidate_quantities=records.quantities(b),
                reference_flags=records.threshold_flags(a),candidate_flags=records.threshold_flags(b),
                normalize_locator=records.locator)
            exact_diagnostics = _strict_diagnostics(row, identities,
                _same_json(a["scalar_result"], b["scalar_result"]),
                _same_json(records.threshold_flags(a), records.threshold_flags(b)),
                _same_json(records.quantities(a), records.quantities(b)))
            if not all(exact_diagnostics.values()):
                raise ValueError("FF_AC strict scientific exactness failed: " + repr(exact_diagnostics))
            exact_products = _strict_n56_products(np, reader, (pa,pb), (a,b))
            row.update(strict_FFAC_bit_identical=True, strict_exact_diagnostics=exact_diagnostics,
                        strict_N56_exact_products=exact_products,
                        case_id=case_id,reference_result=ba,candidate_result=bb,
                        actual_host_provenance=host_provenance,
                        actual_separately_derived_scientific_bindings=[binding_a,binding_b],
                       per_host_phase_output_sha256=[r["bindings"]["phase_sha256"] for r in (a,b)],
                       geometry_array_identities=identities,geometry_identical_required=identities[0]==identities[1])
            bound = write(output/(case_id+".json"),row)
            results.append(dict(case_id=case_id,passed=row["limits_passed"] and row["strict_FFAC_bit_identical"],first_failure=row["first_failure"],receipt=bound))
            print(json.dumps(dict(utc=utc(),case_id=case_id,passed=row["limits_passed"],first_failure=row["first_failure"])),flush=True)
            del products,geometries,geo,ids
        closure = reader.finish()
        result = dict(complete=True,utc=utc(),format="prospective-current-plan-Main-RAW39-original-Add8-strict-N56-v1",
            comparison_label="Current-plan separate RAW39 producer comparison",profile_or_worker_qualified=False,
            actual_host_commits=HOST_COMMITS,scientific_source_equivalence=q["scientific_source_equivalence"],
            case_count=len(results),ordered_case_ids=CASES,comparisons=results,
            frame_add8_limits_passed=all(r["passed"] for r in results),
            all39_strict_FFAC_bit_identical=all(r["passed"] for r in results),
            strict_acceptance="Original Add8 PASS AND all N56 scientific product checks exact. Only products.npz/code_sha256 is validated as a 0-d <U64 SHA256 of each side's actual commit, also bound by result.json. Every other key, scalar, native-statistics JSON and retained stage byte check stays exact. Any other difference causes STOP/nonzero; no tolerance, data normalization or scientific projection. Raw products.npz equality is reported honestly.",
            numeric_checker=q["source_pins"]["numeric_checker"],comparison_kind="same_precision",
            reference_completed=q["reference_completed"],candidate_completed=q["candidate_completed"],
            environment_identity_proof=q["environment_identity_proof"],candidate_closed_readset=q["candidate_closed_readset"],
            source_relationship=q["source_pair"]["relationship"],
             independent_scientific_qualification=False,Wasserstein_performed=False,
            Wasserstein_qualification="NOT_AVAILABLE_FROM_THESE_SAVED_FRAME_RESULTS; NO_NEW_DISTANCE_COMPUTED",
            production_permission_granted=False,eighty_GO=False,new_scientific_arrays_computed=False,
            native_read_closure=closure,copy_locator_closure=locations.finish())
        result["accepted_prior_comparisons"] = []
        result["new_comparisons_count"] = len(results)
        write(output/"completed.json",result)
        for bound in q["source_pins"].values():
            metadata(bound)
        return 0 if result["frame_add8_limits_passed"] else 1
    except BaseException as e:
        write(output/"STOP.json",dict(utc=utc(),literal=type(e).__name__+": "+str(e),
            traceback=traceback.format_exc(),completed_case_rows=results,partials_preserved=True,no_retry=True))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request",required=True)
    raise SystemExit(run(p.parse_args().request))
