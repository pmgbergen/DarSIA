"""One explicitly released P1 -> C -> P2 cache qualification.

This is a separate typed DAG. It never converts the historical full/all-stage
provider into a staged or lean profile. Original claims, reservations, native
supervision, retained-file accounting and byte publication remain the owners
of execution. Incomplete chains supply no ordinary import or job eligibility.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re

from .policy import CachePolicyError, PHYSICAL_RESERVE_BYTES, _descriptor
from .transient import retry_transient
from .compute_control import optional_n1_configuration
from .first_qualification import (
    FirstQualification, _digest, _exact_keys, _pins,
    _slot_read as _original_slot_read, _slot_write as _original_slot_write,
    administrative_entitlement, checked_required_capacity_bound,
)

PROVIDER = "darsia-common-phase-qualified-provider-v1"
REQUEST = "darsia-common-phase-qualification-request-v1"
ROOT_RELEASE = "darsia-common-phase-root-release-v1"
OUTPUT = "darsia-common-phase-output-entitlement-v1"
OUTPUT_SCOPE = "darsia-common-phase-output-scope-v1"
STAGE_PROOF = "darsia-common-phase-closed-stage-v1"
BUNDLE = "darsia-common-phase-completed-bundle-v1"
JOB_PROFILE = "darsia-common-phase-job-profile-v1"
IMPORT_PROFILE = "darsia-common-phase-import-profile-v1"
STAGES = ("phase1", "calibration", "phase2")
PIPELINES = dict(phase1="native-phase-statistics", calibration="n1-calibration-refit", phase2="n1")
PROFILES = dict(phase1="full", calibration="full", phase2="lean")
COMMON_SCOPE_FIELDS = {"commit", "backend", "precision", "run", "ordered_case_ids"}
STAGE_SLOTS = {"observation", "writer_evidence", "measurement", "native_completion",
               "worker_scientific_evidence", "scientific_evidence", "completion"}
DEVICE_ENVIRONMENT = ("DARSIA_OPENCL_DEVICE", "DARSIA_CUDA_DEVICE", "OPENCV_OPENCL_DEVICE", "OPENCV_OPENCL_RUNTIME")
REQUIRED_SOURCE_SUFFIXES = {
    "darsia_cache/common_qualification.py", "darsia_cache/first_qualification.py",
    "darsia_cache/transient.py", "darsia_cache/compute_control.py", "darsia_cache/publication_paths.py",
    "darsia_cache/bootstrap.py", "darsia_cache/control.py", "darsia_cache/worker_adapter.py",
    "darsia_cache/owned_process.py", "darsia_cache/service_context.py", "darsia_cache/fresh_scope.py",
    "darsia/fluidflower/pipeline/worker.py", "darsia/fluidflower/pipeline/worker_io.py",
    "darsia/fluidflower/pipeline/capacity.py", "darsia/fluidflower/pipeline/io_ops.py",
    "darsia/fluidflower/pipeline/phase_statistics.py", "darsia/fluidflower/pipeline/phase1_protocol.py",
    "darsia/fluidflower/pipeline/refit_runner.py", "darsia/fluidflower/pipeline/refit_jobs.py",
    "darsia/fluidflower/pipeline/common_calibration.py", "darsia/fluidflower/pipeline/staged_data.py",
    "darsia/fluidflower/pipeline/common_worker_closure.py",
    "darsia/fluidflower/pipeline/job_inputs.py", "darsia/fluidflower/pipeline/publication_runtime.py",
    "darsia/fluidflower/pipeline/lean_capacity.py", "darsia/fluidflower/published_jobs_snapshot.py",
    "darsia/fluidflower/pipeline/worker_io.py", "darsia/fluidflower/pipeline/run_io.py",
    "darsia/fluidflower/pipeline/store_runtime.py", "darsia/fluidflower/store_map.py",
    "darsia/fluidflower/mass/mass_opencl.py", "darsia/fluidflower/mass/production_precision.py",
    "darsia/fluidflower/storage_profile.py", "darsia/fluidflower/mass/n1_pipeline.py",
    "darsia/fluidflower/mass/n1_refit.py", "darsia/fluidflower/mass/n1_statistics.py",
    "numba/core/caching.py", "matplotlib/font_manager.py", "tempfile.py",
    "importlib/_bootstrap_external.py",
}


CLOSE_PUBLICATION_ATTEMPTS = 4


def _close_slot_names(provider, role):
    """Return the finite released common-close candidate names; legacy slots stay exact."""
    leaf = provider["slots"][role]
    if not isinstance(leaf, str):
        raise CachePolicyError("Explicit common metadata slot name required")
    if role not in {"scientific_evidence", "completion"} or "pipeline" not in provider["scope"]:
        return [leaf]
    return [leaf] + [leaf[:-5] + ".close" + str(i) + ".json"
                     for i in range(2, CLOSE_PUBLICATION_ATTEMPTS + 1)]


def _close_candidate_valid(value, role):
    """Only a complete typed common close can select a retained immutable candidate."""
    expected = ("darsia-common-phase-scientific-closure-v1" if role == "scientific_evidence" else STAGE_PROOF)
    return isinstance(value, dict) and value.get("format") == expected and value.get("complete") is True


def _close_candidates(provider, role):
    """Read bounded candidates without deleting or trusting partial JSON objects."""
    from .windows import WindowsNamespace
    from .capacity import _json_bytes
    target = provider["output_namespace"]
    namespace = WindowsNamespace(target["path"], target["identity"])
    rows = []
    try:
        present = {name.casefold() for name in namespace.names()}
        for name in _close_slot_names(provider, role):
            if name.casefold() not in present:
                rows.append((name, None, None, None)); continue
            raw, _identity = retry_transient(lambda: namespace.read_once(name, maximum=16 * 1024**2))
            try:
                value = _json_bytes(raw)
            except (ValueError, CachePolicyError, UnicodeError):
                value = None
            pin = dict(path=str(Path(target["path"]) / name),
                sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
            rows.append((name, raw, value, pin))
    finally:
        namespace.close()
    return rows


def _slot_read(provider, role):
    """Resolve complete common candidates; unchanged roles use the original reader."""
    if len(_close_slot_names(provider, role)) == 1:
        return retry_transient(lambda: _original_slot_read(provider, role))
    valid = [(value, pin, raw) for _name, raw, value, pin in _close_candidates(provider, role)
             if _close_candidate_valid(value, role)]
    if not valid:
        raise FileNotFoundError("No complete common metadata-close candidate: " + role)
    if any(raw != valid[0][2] for _value, _pin, raw in valid[1:]):
        raise CachePolicyError("Retained complete common-close candidates disagree")
    return deepcopy(valid[0][:2])


def _slot_write(provider, role, value):
    """Publish one bounded close candidate, or reuse the same complete retained bytes.

    A failed exclusive write retains its original candidate. Retrying uses the
    next released name. Reading an identical complete candidate proves bytes;
    it does not assert that a previously failed flush was performed again.
    """
    if len(_close_slot_names(provider, role)) == 1:
        return _original_slot_write(provider, role, value)
    from .first_qualification import _slot_encoded
    from .windows import WindowsNamespace
    if not _close_candidate_valid(value, role):
        raise CachePolicyError("Only complete common metadata can select a close candidate")
    raw = _slot_encoded(value)
    rows = _close_candidates(provider, role)
    complete = [(old, pin) for _name, old, document, pin in rows if _close_candidate_valid(document, role)]
    if any(old != raw for old, _pin in complete):
        raise CachePolicyError("An immutable complete common close differs from the retry")
    if complete:
        return complete[0][1]
    for name, old, _document, _pin in rows:
        if old is not None:
            continue
        target = provider["output_namespace"]
        namespace = WindowsNamespace(target["path"], target["identity"])
        try:
            namespace.write_once(name, raw, maximum=16 * 1024**2)
        finally:
            namespace.close()
        # Reopen bytes after acknowledgement; never infer success from presence.
        return _slot_read(provider, role)[1]
    raise CachePolicyError("Common metadata-close candidates exhausted; retained files need a new independent release")


def _integer(value, where, *, positive=False):
    """Validate an explicit nonnegative byte or count field."""
    if type(value) is not int or value < (1 if positive else 0):
        raise CachePolicyError(where + " must be an explicit bounded integer")
    return value


def _cases(value):
    """Require finite distinct selected cases in their original order."""
    if (not isinstance(value, list) or not value or len(value) > 100000
            or any(not isinstance(case, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case) is None
                   for case in value) or len(set(value)) != len(value)):
        raise CachePolicyError("Exact nonempty distinct selected case order required")
    return value


def _device(root, read, *, current=False):
    """Verify the actual declared runtime and unchanged device environment."""
    value = read(_descriptor(root["device_identity"]))
    _exact_keys(value, {"format", "backend", "mass_device_id", "python_executable",
        "namespace_admission", "module_closure_sha256", "driver_sources", "device_environment",
        "opencl_compute_qualified", "opencl_drift_policy"}, "Exact declared CPU/runtime/device binding required")
    if (value["format"] != "darsia-common-cpu-runtime-binding-v1" or value["backend"] != "cpu"
            or value["mass_device_id"] != "cpu" or value["python_executable"] != root["python_executable"]
            or value["namespace_admission"] != root["namespace_admission"]
            or value["module_closure_sha256"] != _digest(root["module_closure"])
            or value["opencl_compute_qualified"] is not False
            or value["opencl_drift_policy"] != "existing unchanged drift route; no forced-off setting"):
        raise CachePolicyError("Common CPU proof cannot claim another device/runtime or OpenCL compute qualification")
    drivers = _pins(value["driver_sources"])
    if any(pin not in root["runtime_source_closure"] for pin in drivers):
        raise CachePolicyError("Actual declared device/runtime binary sources are outside the exact startup closure")
    _exact_keys(value["device_environment"], DEVICE_ENVIRONMENT, "Exact unchanged device-selection environment required")
    if any(setting is not None and not isinstance(setting, str) for setting in value["device_environment"].values()):
        raise CachePolicyError("Device-selection environment cannot have implicit or numeric settings")
    if current and value["device_environment"] != {key: os.environ.get(key) for key in DEVICE_ENVIRONMENT}:
        raise CachePolicyError("Actual device-selection environment changed; no backend fallback or forced-off repair")
    return deepcopy(value)


def checked_provider(value):
    """Immutable root pins static scopes/slots, never hashes of future outputs."""
    fields = {"format", "qualification_id", "scope", "namespace_admission", "output_namespace",
              "slots", "stages", "python_executable", "startup", "module_closure",
              "producer_source_closure", "runtime_source_closure", "device_identity"}
    _exact_keys(value, fields, "Exact common P1/C/P2 provider required")
    if value["format"] != PROVIDER or re.fullmatch(r"[0-9a-f]{32}", str(value["qualification_id"])) is None:
        raise CachePolicyError("Explicit finite common provider identity required")
    scope = _exact_keys(value["scope"], COMMON_SCOPE_FIELDS, "Exact common source/run scope required")
    cases = _cases(scope["ordered_case_ids"])
    if (not isinstance(scope["run"], str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", scope["run"]) is None
            or scope["backend"] != "cpu" or scope["precision"] not in {"float32", "float64"}
            or re.fullmatch(r"[0-9a-f]{40}", str(scope["commit"])) is None):
        raise CachePolicyError("Common qualification needs its declared finite run, CPU precision and pinned source")
    for field in ("namespace_admission", "python_executable", "startup", "device_identity"):
        _descriptor(value[field])
    output = _exact_keys(value["output_namespace"], {"path", "identity"}, "Accepted common receipt root required")
    if not PureWindowsPath(output["path"]).is_absolute() or not isinstance(output["identity"], dict):
        raise CachePolicyError("Common receipts require their actual accepted namespace identity")
    _exact_keys(value["slots"], {"completion", "job_profile", "import_profile"}, "Fixed common final slots required")
    _exact_keys(value["stages"], STAGES, "All three actual stages must be declared")
    names = list(value["slots"].values())
    for stage in STAGES:
        row = _exact_keys(value["stages"][stage],
            {"pipeline", "run", "ordered_case_ids", "storage_profile", "colour_intermediates", "slots"},
            "Exact common stage declaration required")
        _cases(row["ordered_case_ids"])
        if (row["pipeline"] != PIPELINES[stage] or row["storage_profile"] != PROFILES[stage]
                or row["colour_intermediates"] != "keep"
                or row["run"] != ("refit" if stage == "calibration" else scope["run"])
                or (stage != "calibration" and row["ordered_case_ids"] != cases)):
            raise CachePolicyError("Actual stage, selected order, Full/Lean or KEEP differs")
        _exact_keys(row["slots"], STAGE_SLOTS, "Fixed per-stage append-once slots required")
        names.extend(name for role in row["slots"] for name in _close_slot_names(stage_provider(value, stage), role))
    if (any(not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,180}\.json", name) is None for name in names)
            or len(names) != len({name.casefold() for name in names})):
        raise CachePolicyError("Common slots must be distinct finite flat JSON names")
    from .windows import _component
    for name in names:
        _component(name)
    union = _pins(value["producer_source_closure"]) + _pins(value["runtime_source_closure"])
    closure = _pins(value["module_closure"])
    if len(union) != len(closure) or {_digest(pin) for pin in union} != {_digest(pin) for pin in closure}:
        raise CachePolicyError("Exact producer/runtime union must equal the startup module closure")
    actual = {str(pin["path"]).replace("\\", "/").casefold() for pin in union}
    if any(not any(path.endswith("/" + suffix) for path in actual) for suffix in REQUIRED_SOURCE_SUFFIXES):
        raise CachePolicyError("Common closure omits an actual stage/provider/resource/output boundary")
    return deepcopy(value)


def stage_provider(provider, stage):
    """An explicit view of declared root slots; no new file or future descriptor."""
    if stage not in STAGES:
        raise CachePolicyError("Explicit common stage required")
    row = provider["stages"][stage]
    result = deepcopy(provider)
    result["slots"] = deepcopy(row["slots"])
    result["scope"] = dict(deepcopy(provider["scope"]), pipeline=row["pipeline"],
        run=row["run"], ordered_case_ids=deepcopy(row["ordered_case_ids"]))
    return result


def checked_output(value, *, task, request_binding):
    """Bind whole-stage output and service allowances to the exact released task."""
    from .first_qualification import checked_bound_output_scope
    _exact_keys(value, {"format", "stage", "task", "request_scope", "maximum_output_bytes",
        "maximum_service_bytes", "watched_output_scope"}, "Exact common stage resource entitlement required")
    if (value["format"] != OUTPUT or value["stage"] not in STAGES
            or value["task"] != task or value["request_scope"] != request_binding):
        raise CachePolicyError("Common resource plan differs from its actual independently released task/scope")
    _integer(value["maximum_output_bytes"], "Whole common stage output entitlement", positive=True)
    _integer(value["maximum_service_bytes"], "Whole common stage service entitlement", positive=True)
    checked_bound_output_scope(value["watched_output_scope"], common=True)
    return deepcopy(value)


class CommonQualification(FirstQualification):
    """Use the original single-stage native controller for one released chain step.

    Each continuation has a real task and hashes only already-closed predecessor
    evidence. The root provider is the same across all steps; no stores repin or
    temporary passing profile breaks the DAG. Claims still use the inherited
    original atomic claim-consumption transaction.
    """
    common_mode = True

    def __init__(self, binding, *, provider_binding, stores_source, namespace, read,
                 read_authorities=None, historical_authorities=None):
        """Read the exact independently released scope without writing or starting a worker."""
        if historical_authorities is not None:
            raise CachePolicyError("Historical first-qualification authorities cannot authorize a common chain")
        self.binding, self.provider_binding = _descriptor(binding), _descriptor(provider_binding)
        self.request = read(self.binding)
        fields = {"format", "stage", "execution_authorized", "root_release", "provider",
            "stores_source", "namespace_admission", "public_command", "task", "job", "job_root",
            "configuration", "capacity_bound_plan", "output_entitlement_plan", "retained_cache_inventory",
            "cache_stat_period_seconds", "predecessors"}
        _exact_keys(self.request, fields, "Exact independently released common-stage request required")
        self.stage = self.request["stage"]
        self.root_provider = checked_provider(read(self.provider_binding))
        self.provider = stage_provider(self.root_provider, self.stage)
        self.device = _device(self.root_provider, read, current=True)
        self.stores_source = deepcopy(stores_source)
        if (self.request["format"] != REQUEST or self.request["execution_authorized"] is not True
                or self.request["provider"] != self.provider_binding
                or self.request["stores_source"] != stores_source
                or self.request["namespace_admission"] != namespace
                or namespace != self.provider["namespace_admission"]
                or not isinstance(self.request["public_command"], list) or not self.request["public_command"]
                or any(not isinstance(arg, str) for arg in self.request["public_command"])
                or type(self.request["cache_stat_period_seconds"]) not in {int, float}
                or not 0 < self.request["cache_stat_period_seconds"] <= 30):
            raise CachePolicyError("Common stage/source/stores/namespace/public-command admission differs")
        payload = {key: value for key, value in self.request.items() if key != "root_release"}
        release = read(_descriptor(self.request["root_release"]))
        if (release.get("format") != ROOT_RELEASE or release.get("execution_authorized") is not True
                or release.get("provider") != self.provider_binding
                or release.get("stores_source") != stores_source
                or release.get("scope") != self.root_provider["scope"]
                or release.get("request_payload_sha256") != _digest(payload)):
            raise CachePolicyError("Exact independent release of this common stage is required")
        _checked_source_review(read(_descriptor(release.get("source_review"))), self.root_provider)
        self.task = read(_descriptor(self.request["task"]))
        self.job = read(_descriptor(self.request["job"]))
        self.require_identity(self.task)
        command = self.request["public_command"]
        positions = [i for i, argument in enumerate(command) if argument == "--job"]
        job_root = PureWindowsPath(self.request["job_root"])
        if (self.job.get("commit") != self.task["commit"]
                or self.job.get("precision") != self.task["precision"]
                or self.job.get("expected_runs") != [self.task["run"]]
                or not job_root.is_absolute() or ".." in job_root.parts
                or PureWindowsPath(self.request["job"]["path"]) != job_root / "job.json"
                or self.task.get("job_id") != job_root.name or len(positions) != 1
                or positions[0] + 1 >= len(command)
                or PureWindowsPath(command[positions[0] + 1]) != job_root):
            raise CachePolicyError("One exact existing queued stage job is required")
        _descriptor(self.request["configuration"])
        if (self.request["configuration"]["sha256"] != self.task["configuration"]["sha256"]
                or self.request["configuration"]["size_bytes"] != self.task["configuration"].get("bytes",
                    self.task["configuration"].get("size_bytes"))):
            raise CachePolicyError("Original configuration and rooted task bytes differ")
        self.required_capacity_bound = checked_required_capacity_bound(
            read(_descriptor(self.request["capacity_bound_plan"])))
        if (self.required_capacity_bound["enforcement_state"] != "PREDECLARED_Q106_VOLUME_BOUND"
                or self.required_capacity_bound["volume_cap_observation"].get("virtual_size_bytes")
                    != self.required_capacity_bound["volume_cap_observation"].get("binding", {}).get("maximum_size_bytes")):
            raise CachePolicyError("A held prelaunch plan cannot authorize an actual common-stage claim")
        self.output_entitlement = checked_output(read(_descriptor(self.request["output_entitlement_plan"])),
            task=self.request["task"], request_binding=self.provider_binding)
        if self.output_entitlement["stage"] != self.stage:
            raise CachePolicyError("The output entitlement belongs to another common stage")
        self._read = read
        self._read_authorities = read_authorities
        self._check_predecessors()
        self.bound_mode, self.reference_build = True, False
        self.reference_administration = None
        self.reference_roots_validated = False
        self.reference_transport_observation = None
        self.reference_retained_inventory = None
        self.q123_cold_scope = None
        self.started, self.claim = False, None

    def require_identity(self, task):
        """Require the original source, stage, storage profile and selected task order."""
        scope, row = self.provider["scope"], self.root_provider["stages"][self.stage]
        keys = ("commit", "pipeline", "backend", "precision", "run", "ordered_case_ids")
        if (not isinstance(task, dict) or any(task.get(key) != scope[key] for key in keys)
                or re.fullmatch(r"[0-9a-f]{32}", str(task.get("attempt_id"))) is None
                or task.get("storage_profile", "full") != row["storage_profile"]
                or task.get("format_version") != (2 if self.stage == "phase2" else 1)
                or task.get("resume_from") is not None):
            raise CachePolicyError("Common bootstrap requires its exact stage/profile/source/task; no prefix replay is invented")
        return True

    def _check_predecessors(self):
        """Admit only exact already-closed predecessor stages."""
        expected = STAGES[:STAGES.index(self.stage)]
        _exact_keys(self.request["predecessors"], expected, "Only actual closed predecessor stages may continue")
        for stage in expected:
            actual, pin = _slot_read(stage_provider(self.root_provider, stage), "completion")
            if self.request["predecessors"][stage] != pin:
                raise CachePolicyError("Common continuation does not name its fixed actual predecessor")
            checked_stage(actual, root=self.root_provider, provider_binding=self.provider_binding,
                          stores_source=self.stores_source, read=self._read,
                          read_authorities=self._read_authorities)

    def _once_key(self):
        """Identify this original provider stage without changing its claim ledger."""
        return _digest(dict(common=self.provider_binding, stage=self.stage,
                            scope=self.provider["scope"], modules=self.provider["module_closure"]))

    def initial_cache_bytes(self, lease):
        """Verify actual retained cache files before the next finite stage."""
        from .retained_qualification import require_prior_reservations_closed
        from .service_context import verify_retained_cache_inventory
        pin = self.request["retained_cache_inventory"]
        if pin is None:
            if self.stage != "phase1":
                raise CachePolicyError("A continuation requires its actual retained-file inventory")
            return 0
        inventory = self._read(_descriptor(pin))
        require_prior_reservations_closed(lease.ledger.control.names())
        return verify_retained_cache_inventory(inventory, lease.policy.root)["logical_bytes"]

    def consume(self, lease):
        """Admit one original capped stage claim under the existing ledger lock."""
        from .control import _encoded
        from .windows import WindowsNamespace
        target = self.provider["output_namespace"]
        namespace = WindowsNamespace(target["path"], target["identity"])
        try:
            present = {name.casefold() for name in namespace.names()}
            declared = {name.casefold() for role in self.provider["slots"]
                        for name in _close_slot_names(self.provider, role)}
            if present & declared:
                raise CachePolicyError("Common stage output already exists; recover only its retired metadata close")
        finally:
            namespace.close()
        self._namespace(lease)
        with lease.ledger.locked():
            report = lease.ledger.project(0)
            cap = lease.policy.volume_cap
            expected = self.required_capacity_bound["volume_cap_observation"]
            actual = report.get("volume_cap_observation", {})
            if (report.get("hard_filesystem_quota_verified") is not True
                    or report.get("current_cache_logical_bytes") != self.initial_cache_bytes(lease)
                    or report.get("active_cache_remaining_bytes") != 0
                    or cap is None or cap.maximum_size_bytes != lease.policy.quota_bytes
                    or expected.get("binding") != cap.document()
                    or any(actual.get(key) != expected.get(key) for key in (
                        "binding", "virtual_size_bytes", "volume_total_bytes", "filesystem", "is_loaded", "verified"))
                    or report.get("reserve_bytes") != PHYSICAL_RESERVE_BYTES
                    or self.output_entitlement["maximum_service_bytes"] > administrative_entitlement(report)):
                raise CachePolicyError("Actual exclusive capped common-stage cache/resources differ from their release")
            names, key = lease.ledger.control.names(), self._once_key()
            if "cache-first-" + key + ".json" in names or "cache-first-" + key + "-claim.json" in names:
                raise CachePolicyError("This actual common-stage claim was already consumed")
            # Same original marker and inherited atomic consume writer; no new
            # queue, claim writer or executor is introduced by the new type.
            lease.ledger.control.write_once(self._launch_marker_name(), _encoded(dict(
                format="darsia-cache-first-qualification-launch-admitted-v1", request=self.binding,
                provider=self.provider_binding, stores_source=self.stores_source,
                owner=lease.owner, scope=self.provider["scope"], no_retry=True)))
        self.started = True

    def service_entitlement(self, lease):
        """Reserve the independently bounded stage service using fresh actual headroom."""
        self._require_launch_admission(lease)
        report = lease.ledger.project(0)
        maximum = self.output_entitlement["maximum_service_bytes"]
        if maximum > administrative_entitlement(report):
            raise CachePolicyError("Fresh original cache admission cannot cover the released common stage")
        self.administrative = dict(maximum_bytes=maximum, measured_profile=False,
            semantics="predeclared common stage service entitlement; original physical bound/reserves",
            request=self.binding, admission=report)
        return maximum

    def require_task(self, lease, task):
        """Require the exact released task and its original cache lease."""
        self.require_identity(task)
        if (getattr(lease, "first_qualification", None) is not self or task != self.task
                or lease.source != self.stores_source or self.claim is not None and self.claim != task):
            raise CachePolicyError("Only this exact independently released common-stage task may claim")
        return True

    def remember_claim_locked(self, lease, claim):
        """Preserve original atomic claim consumption inside the released output scope."""
        output = PureWindowsPath(str(claim.output))
        watched = PureWindowsPath(self.output_entitlement["watched_output_scope"]["path"])
        if output != watched and watched not in output.parents:
            raise CachePolicyError("The actual original claim output leaves its released watched stage root")
        return super().remember_claim_locked(lease, claim)

    def loaded_modules(self):
        """Record actual compute context and the original worker scientific-close pin."""
        value = super().loaded_modules()
        _device(self.root_provider, self._read, current=True)
        value["common_device_identity"] = deepcopy(self.root_provider["device_identity"])
        value["actual_device_environment"] = {key: os.environ.get(key) for key in DEVICE_ENVIRONMENT}
        value["common_worker_scientific_evidence"] = deepcopy(getattr(self, "worker_scientific_evidence", None))
        return value


@dataclass(frozen=True)
class CommonStageEntitlement:
    """Keep the whole-stage bound reserved until the original attempt closes."""
    run: str
    case_count: int
    maximum_output_bytes: int
    request: dict
    output_plan: dict
    scope: dict
    model: str = "predeclared common whole-stage entitlement; not a measured profile"
    products_per_frame_estimated_bytes: int = 0
    products_estimated_bytes: int = 0

    @property
    def transient_estimated_bytes(self):
        """Keep the whole stage output bound charged until retirement."""
        return self.maximum_output_bytes

    def needed(self, *, other_remaining_bytes=0, reserve_bytes=PHYSICAL_RESERVE_BYTES):
        """Include other output claims and the unchanged operational reserve."""
        return (self.maximum_output_bytes + _integer(other_remaining_bytes, "Other output reservations")
                + _integer(reserve_bytes, "Original physical reserve"))


def require_phase2_predecessor(first, config):
    """Refuse a different P1/C join before any phase-two scientific computation."""
    if first.stage != "phase2":
        return
    p1 = _slot_read(stage_provider(first.root_provider, "phase1"), "scientific_evidence")[0]["phase1"]
    calibration = _slot_read(stage_provider(first.root_provider, "calibration"), "scientific_evidence")[0]
    recorded = config.get("provenance", {}).get("common_frozen_refit")
    if (not isinstance(recorded, dict) or recorded.get("actual_frozen_refit_verified") is not True
            or _digest(recorded) != _digest(calibration["calibration_join"])
            or config.get("phase1", {}).get("configuration") != p1["configuration"]
            or config.get("phase1", {}).get("completion") != p1["completion"]
            or config.get("phase1", {}).get("code_commit") != first.task["commit"]
            or recorded.get("calibration") != config.get("calibration")
            or recorded.get("anchor") != config.get("anchor")):
        raise CachePolicyError("Phase-two admission differs from its actual closed predecessor P1/C join")


def common_output_entitlement(runtime, task, config, rows):
    """Positive ClaimCapacity hook; ordinary/default estimate branches stay intact."""
    first = runtime.cache_adapter.lease.first_qualification
    if not isinstance(first, CommonQualification):
        raise CachePolicyError("An actual common bootstrap lease is required")
    first.require_task(runtime.cache_adapter.lease, task)
    if config.get("run") != task["run"]:
        raise CachePolicyError("Common resource configuration belongs to another run")
    if first.stage != "calibration":
        if [frame.get("case_id") for frame in config.get("frames", ())] != task["ordered_case_ids"]:
            raise CachePolicyError("Common resource estimate differs from the selected frame order")
    if first.stage == "phase2":
        from darsia.fluidflower.storage_profile import resolve
        if resolve(config) != dict(profile="lean", colour_intermediates="keep"):
            raise CachePolicyError("Common phase two must retain KEEP")
        require_phase2_predecessor(first, config)
        from darsia.fluidflower.pipeline.capacity import _phase2_lean_estimate
        estimate = _phase2_lean_estimate(runtime.stores, runtime.job["series_id"], task, config,
            repository=runtime.repository, metadata_rows=rows)
        if first.output_entitlement["maximum_output_bytes"] < estimate.needed(reserve_bytes=0):
            raise CachePolicyError("Released common output entitlement does not cover exact lean bounds")
    return CommonStageEntitlement(task["run"], len(task["ordered_case_ids"]),
        first.output_entitlement["maximum_output_bytes"], deepcopy(first.binding),
        deepcopy(first.request["output_entitlement_plan"]), deepcopy(first.provider["scope"]))


def checked_stage(value, *, root, provider_binding, stores_source, read, read_authorities):
    """Validate a closed stage and every original observation/native descriptor."""
    _exact_keys(value, {"format", "provider", "stores_source", "stage", "request", "task",
        "job", "completion", "configuration", "observation", "native_completion", "measurement",
        "writer_evidence", "scientific_evidence", "device_identity", "once_marker", "complete", "passed"},
        "Exact original closed common stage required")
    stage = value["stage"]
    if (stage not in STAGES or value["format"] != STAGE_PROOF or value["provider"] != provider_binding
            or value["stores_source"] != stores_source or value["complete"] is not True
            or value["passed"] is not True or value["device_identity"] != root["device_identity"]):
        raise CachePolicyError("Common stage is incomplete or outside the original source/device/stores")
    projected = stage_provider(root, stage)
    for field in ("request", "task", "job", "completion", "device_identity"):
        read(_descriptor(value[field]))
    _descriptor(value["configuration"])
    request, task = read(value["request"]), read(value["task"])
    first = CommonQualification(value["request"], provider_binding=provider_binding,
        stores_source=stores_source, namespace=root["namespace_admission"], read=read,
        read_authorities=read_authorities)
    if first.stage != stage or first.task != task:
        raise CachePolicyError("The actual independently released stage proof differs")
    expected = projected["scope"]
    if (request.get("format") != REQUEST or request.get("stage") != stage
            or request.get("provider") != provider_binding or request.get("stores_source") != stores_source
            or request.get("task") != value["task"]
            or any(task.get(key) != expected[key] for key in (
                "commit", "pipeline", "backend", "precision", "run", "ordered_case_ids"))):
        raise CachePolicyError("The common stage differs from its actual released task")
    observation, op = _slot_read(projected, "observation")
    native, np = _slot_read(projected, "native_completion")
    measured, mp = _slot_read(projected, "measurement")
    evidence, ep = _slot_read(projected, "writer_evidence")
    if any(value[field] != pin for field, pin in (
            ("observation", op), ("native_completion", np), ("measurement", mp), ("writer_evidence", ep))):
        raise CachePolicyError("A common-stage observation is outside its fixed immutable slot")
    if (observation.get("actual_claim") != task or observation.get("source_runtime_before_after_exact") is not True
            or observation.get("actual_public_worker_exit_code") != 0
            or observation.get("operational_reserve_crossed") is not False
            or native.get("format") != "darsia-native-worker-first-qualification-completion-v1"
            or native.get("complete") is not True or native.get("passed") is not True
            or native.get("observer_native_handles_closed") is not True
            or native.get("observation") != op or native.get("retirement") != observation.get("retirement")
            or native.get("module_closure") != root["module_closure"]
            or measured.get("format") != "darsia-common-phase-closed-output-diagnostic-v1"
            or measured.get("complete") is not True or measured.get("first_stop") is not None
            or measured.get("closed_output_stat_observation") is not True
            or measured.get("original_retirement_accepted") is not True
            or measured.get("cache_operational_reserve_crossed") is not False
            or evidence.get("observation_sha256") != _digest(observation)):
        raise CachePolicyError("Common stage lacks exact native retirement/source/cache/output evidence")
    owners = observation.get("actual_registered_owner_exit_inventories")
    if (not isinstance(owners, list) or not owners
            or any(row.get("all_file_backed_modules_predeclared_and_exact") is not True
                   or row.get("common_device_identity") != root["device_identity"]
                   or row.get("actual_device_environment") != first.device["device_environment"] for row in owners)):
        raise CachePolicyError("Common registered owners lack their exact source/device/runtime exit bindings")
    from .first_qualification import checked_bounded_evidence
    checked_bounded_evidence(observation, evidence, first=first)
    for document in (observation, native, measured, evidence):
        if (document.get("request") != value["request"] or document.get("provider") != provider_binding
                or document.get("stores_source") != stores_source or document.get("scope") != expected):
            raise CachePolicyError("Common original stage evidence belongs to another request or scope")
    _original_stage_native(first, observation, evidence, native, measured,
                           once_marker=value["once_marker"], read=read)
    _checked_cache_retention(first, evidence, measured, read)
    science, sp = _slot_read(projected, "scientific_evidence")
    if value["scientific_evidence"] != sp:
        raise CachePolicyError("Common scientific evidence is outside its declared immutable slot")
    _checked_science(science, stage=stage, task=task, value=value, read=read)
    worker, wp = _slot_read(projected, "worker_scientific_evidence")
    if science["worker_scientific_evidence"] != wp:
        raise CachePolicyError("Scientific closure replaces its original worker-finalized evidence")
    _checked_worker_evidence(worker, first, observation, wp, read)
    for field in ("completion", "configuration", "phase1", "calibration_join", "output_files"):
        if science[field] != worker[field]:
            raise CachePolicyError("Final scientific metadata replaces an actual worker result: " + field)
    if ({key: item for key, item in science["result_space"].items() if key != "lean_run_bytes"}
            != worker["result_space"]):
        raise CachePolicyError("Final resource metadata replaces its admitted-worker calculation")
    expected_compute = dict(deepcopy(worker["compute_evidence"]),
        kernel_lifetime_memory=deepcopy(observation.get("kernel_lifetime_memory")),
        native_launch_to_retirement_seconds=observation.get("native_launch_to_retirement_seconds"))
    if science["compute_evidence"] != expected_compute:
        raise CachePolicyError("Final compute metadata replaces its actual worker observations")
    if science["result_space"]["maximum_output_bytes"] != first.output_entitlement["maximum_output_bytes"]:
        raise CachePolicyError("The completed common stage changes its released output allowance")
    if value["configuration"] != request["configuration"] or value["job"] != request["job"]:
        raise CachePolicyError("Common closure replaces the released original job or configuration")
    if (native.get("startup") != root["startup"] or native.get("cache_profile_qualified") is not False
            or observation.get("output_measurement") != mp
            or measured.get("output_plan") != request["output_entitlement_plan"]
            or measured.get("output_scope") != first.output_entitlement["watched_output_scope"]
            or measured.get("first_observation_error") is not None
            or measured.get("whole_output_entitlement_observed_exceeded") is not False):
        raise CachePolicyError("Common stage output/native binding differs from its released original observations")
    return deepcopy(value)


def _original_stage_native(first, observation, evidence, native, measured, *, once_marker, read):
    """The original bound-DAG retirement/service checks, without old full fit claims."""
    if (native.get("maximum_family_processes") != observation.get("kernel_total_processes")
            or type(native.get("maximum_family_processes")) is not int
            or native["maximum_family_processes"] <= 0):
        raise CachePolicyError("Actual native kernel lifetime family bound is required")
    retirement = read(_descriptor(observation["retirement"]))
    if (retirement.get("format") != "darsia-native-owned-worker-retirement-v1"
            or retirement.get("native_launch") != observation.get("native_launch")
            or retirement.get("service_receipt") != observation.get("service_receipt")
            or retirement.get("whole_family_kernel_observation") is not True
            or retirement.get("terminal") != observation.get("original_kernel_terminal")
            or retirement.get("terminal", {}).get("wrapper_exit_code") != 0
            or retirement.get("terminal", {}).get("accounting", {}).get("active_processes") != 0):
        raise CachePolicyError("Actual original whole-family kernel retirement is required")
    primary = read(_descriptor(retirement["actual_primary_binding"]))
    if (primary.get("actual_service_receipt") != observation["service_receipt"]
            or primary.get("original_native_launch") != observation["native_launch"]
            or primary.get("actual_primary") != retirement.get("primary")
            or primary.get("held_process_handle_observed_while_live") is not True
            or primary.get("same_original_job_membership_observed") is not True):
        raise CachePolicyError("Actual common primary lacks the original retained native handle binding")
    eligibility = native.get("eligibility", {})
    if (eligibility != observation.get("original_publication_and_retirement_eligibility")
            or eligibility.get("original_retirement_verifier_accepted") is not True
            or read(_descriptor(eligibility["seal"])).get("value", {}).get("outcome") != "success"):
        raise CachePolicyError("Original whole-run publication/seal and retirement verifier are required")
    marker_pin = _descriptor(once_marker)
    marker = read(marker_pin)
    if (PureWindowsPath(marker_pin["path"]).name != "cache-first-" + first._once_key() + ".json"
            or marker.get("format") != "darsia-cache-first-qualification-consumed-v1"
            or any(marker.get(key) != observation.get(key) for key in ("request", "provider", "stores_source", "scope"))
            or marker.get("no_retry") is not True):
        raise CachePolicyError("The actual original atomic common-stage consume marker differs")
    service = read(_descriptor(observation["service_receipt"]))
    if (service.get("format") != "darsia-cache-import-service-receipt-v1"
            or service.get("policy") != first.stores_source or service.get("profile") != first.provider_binding
            or service.get("manifest", {}).get("purpose") != "jobs" or service.get("entry_kind") != "worker"
            or not isinstance(service.get("role_identifier"), str)
            or service.get("namespace") != first.provider["namespace_admission"]
            or service.get("manifest", {}).get("root") != evidence["service_namespace"]
            or service.get("manifest", {}).get("reservation") != evidence["actual_service_reservation"]
            or service.get("manifest", {}).get("owner") != evidence["actual_service_reservation_document"]["owner"]
            or service.get("manifest", {}).get("maximum_bytes") != evidence["administrative_service_entitlement"]["maximum_bytes"]
            or service.get("manifest", {}).get("native_launch_context") != observation["native_launch"]
            or service.get("first_qualification", {}).get("request") != first.binding
            or service.get("first_qualification", {}).get("provider") != first.provider_binding
            or service.get("first_qualification", {}).get("administrative_entitlement") != evidence["administrative_service_entitlement"]):
        raise CachePolicyError("Actual original common-stage service/stores/namespace/admin differs")
    from .environment import FIRST_WRITER_SUBDIRECTORIES, FIRST_WRITER_ENVIRONMENT_KEYS
    fresh = service["manifest"].get("fresh_scope_provenance", {})
    if (fresh.get("format") != "darsia-fresh-exclusive-empty-service-scope-v1"
            or fresh.get("owner") != service["manifest"]["owner"]
            or fresh.get("token") != evidence["actual_service_reservation"]["token"]
            or fresh.get("purpose") != "jobs"
            or fresh.get("root_identity") != service["manifest"].get("root_identity")
            or fresh.get("creation_disposition") != "NtCreateFile.FILE_CREATE"
            or fresh.get("exclusive_root_handle") is not True or fresh.get("initial_names") != []
            or fresh.get("observation_precedes_writer_subdirectories_and_imports") is not True
            or fresh.get("writer_subdirectories") != list(FIRST_WRITER_SUBDIRECTORIES)
            or fresh.get("writer_environment_keys") != list(FIRST_WRITER_ENVIRONMENT_KEYS)):
        raise CachePolicyError("Each common stage needs its actual fresh exclusive original service")
    from .control import _time
    _time(fresh["initial_empty_observed_utc"])
    lifecycle = dict(reservation=service["manifest"]["reservation"], policy=first.stores_source,
        profile=first.provider_binding, namespace=first.provider["namespace_admission"], manifest=service["manifest"])
    for row in evidence["registered_owner_environments"]:
        registered = read(_descriptor(row["registration"]))
        if (registered.get("format") != "darsia-cache-service-lifecycle-v1"
                or registered.get("kind") != "owner" or registered.get("binding") != lifecycle
                or registered.get("actor") != row["owner"] or registered.get("value", {}).get("owner") != row["owner"]
                or registered.get("value", {}).get("native_job_membership") != observation["native_launch"]):
            raise CachePolicyError("Original common-stage registration is outside the actual native family")
        exited = read(_descriptor(row["inventory"]))
        modules = exited.get("first_qualification_modules", {})
        if (exited.get("owner") != row["owner"] or modules.get("request") != first.binding
                or modules.get("provider") != first.provider_binding
                or modules.get("all_file_backed_modules_predeclared_and_exact") is not True
                or modules.get("undeclared") != []
                or modules.get("predeclared_module_closure_sha256") != _digest(first.provider["module_closure"])
                or modules.get("actual_writer_environment") != row["actual_writer_environment"]
                or modules.get("writer_environment_is_actual_IO_or_lifetime_proof") is not False
                or modules.get("common_device_identity") != first.root_provider["device_identity"]
                or modules.get("actual_device_environment") != first.device["device_environment"]):
            raise CachePolicyError("Original common-stage owner exit/source/device inventory differs")
    if (measured.get("actual_retirement") != observation["retirement"]
            or measured.get("actual_terminal") != observation["original_kernel_terminal"]
            or measured.get("entitlement_bytes") != first.output_entitlement["maximum_output_bytes"]
            or measured.get("sampled_highwater_is_lifetime_peak") is not False
            or measured.get("output_hard_quota") is not False):
        raise CachePolicyError("Actual common output observation is not its original retirement/whole-stage bound")
    samples, admission = measured.get("output_samples"), measured.get("admission")
    if (measured.get("target_sample_period_seconds") != 1 or not isinstance(samples, list) or not samples
            or not isinstance(admission, dict) or admission.get("logical_bytes") != 0
            or admission.get("observed_scan_anomalies") != []
            or admission.get("volume", {}).get("free_bytes", 0) - PHYSICAL_RESERVE_BYTES < measured["entitlement_bytes"]):
        raise CachePolicyError("Original empty output admission and actual output samples are required")
    import math
    previous = None
    for sample in samples:
        instant, elapsed = sample.get("started_monotonic_seconds"), sample.get("elapsed_seconds")
        if (type(instant) not in {int, float} or not math.isfinite(instant)
                or type(elapsed) not in {int, float} or not math.isfinite(elapsed) or elapsed < 0
                or (previous is None and sample.get("actual_gap_since_previous_start_seconds") is not None)
                or (previous is not None and (instant < previous
                    or sample.get("actual_gap_since_previous_start_seconds") != instant - previous))
                or sample.get("sampled") is not True or sample.get("complete_lifetime_peak") is not False
                or sample.get("free_bytes", 0) <= PHYSICAL_RESERVE_BYTES
                or type(sample.get("logical_bytes")) is not int
                or not 0 <= sample["logical_bytes"] <= measured["entitlement_bytes"]):
            raise CachePolicyError("STOP, nonfinite output timing or inconsistent actual samples cannot qualify")
        previous = instant
    if (samples[-1].get("forced_terminal") is not True
            or measured.get("sampled_highwater_bytes") != max(row["logical_bytes"] for row in samples)
            or measured.get("final_snapshot", {}).get("logical_bytes") != samples[-1]["logical_bytes"]
            or measured.get("closed_output_logical_bytes") != samples[-1]["logical_bytes"]):
        raise CachePolicyError("The actual forced-terminal output observation is missing or inconsistent")


def _checked_cache_retention(first, evidence, measured, read):
    """Bind whole-root retained files to the original redirected exiting writers."""
    from .environment import first_qualification_environment
    namespace = read(_descriptor(first.root_provider["namespace_admission"]))
    if not isinstance(namespace.get("cache_identity"), dict) or not namespace["cache_identity"]:
        raise CachePolicyError("Retained-cache snapshots require the original nonempty cache root identity")
    service = PureWindowsPath(evidence["service_namespace"])
    root = service.parent.parent
    for row in evidence["registered_owner_environments"]:
        actual = row["actual_writer_environment"]
        if first_qualification_environment(actual, scope_root=str(service)) != actual:
            raise CachePolicyError("Actual exited common writer changed its redirected driver-cache locations")
    if (measured.get("retained_cache_files_deleted_by_this_code") is not False
            or measured.get("retained_cache_scope_is_exhaustive_OS_write_proof") is not False):
        raise CachePolicyError("Retained cache accounting cannot assert cleanup or exhaustive OS writes")
    for key in ("retained_cache_before", "retained_cache_after"):
        snapshot = measured.get(key)
        if (not isinstance(snapshot, dict) or PureWindowsPath(snapshot.get("root", "")) != root
                or snapshot.get("root_identity") != namespace.get("cache_identity")
                or snapshot.get("fresh") is not True or snapshot.get("no_reparse_traversal") is not True
                or snapshot.get("payload_hashes_included") is not True
                or snapshot.get("observed_scan_anomalies") != []
                or not isinstance(snapshot.get("files"), list)):
            raise CachePolicyError("Before/terminal whole-root retained-cache hashes are required")
        elapsed = snapshot.get("inventory_elapsed_seconds")
        import math
        if type(elapsed) not in {int, float} or not math.isfinite(elapsed) or elapsed < 0:
            raise CachePolicyError("Actual retained-cache inventory time is required")
        total, paths = 0, set()
        for row in snapshot["files"]:
            relative = PureWindowsPath(row["relative_path"])
            if (relative.is_absolute() or not relative.parts or ".." in relative.parts
                    or str(relative).casefold() in paths
                    or re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))) is None):
                raise CachePolicyError("Retained-cache inventory escaped, duplicated or omitted a file hash")
            paths.add(str(relative).casefold())
            total += _integer(row.get("size_bytes"), "Retained cache bytes")
            for field in ("dev", "ino", "mtime_ns"):
                _integer(row.get(field), "Retained cache file identity: " + field)
        if total != snapshot.get("logical_bytes"):
            raise CachePolicyError("Whole-root retained cache total omits actual counted files")
    return deepcopy(measured["retained_cache_after"])


def bounded_evidence(first, context, observation, *, service_receipt):
    """Retain the original physical-bound *unqualified* observation schema.

    This is actual service/volume evidence, not an old full qualification or a
    new profile. Its original checker explicitly prohibits ordinary-use claims.
    """
    from .first_qualification import bounded_evidence as original_evidence
    return original_evidence(first, context, observation, service_receipt=service_receipt)


def _checked_science(science, *, stage, task, value, read):
    """Check original stage identities, input closure and admitted output bounds."""
    fields = {"format", "stage", "task", "completion", "configuration", "pipeline", "run",
        "ordered_case_ids", "storage_profile", "colour_intermediates", "phase1", "calibration_join",
        "output_files", "resume_policy", "source_bytes_reverified", "source_verification",
        "compute_evidence", "result_space", "lean_frame_bytes", "worker_scientific_evidence", "complete"}
    _exact_keys(science, fields, "Exact actual common scientific closure required")
    if (science["format"] != "darsia-common-phase-scientific-closure-v1"
            or science["stage"] != stage or science["task"] != task
            or science["completion"] != value["completion"]
            or science["configuration"] != value["configuration"]
            or science["pipeline"] != PIPELINES[stage] or science["run"] != task["run"]
            or science["ordered_case_ids"] != task["ordered_case_ids"]
            or science["storage_profile"] != PROFILES[stage] or science["colour_intermediates"] != "keep"
            or science["source_bytes_reverified"] is not True or science["complete"] is not True
            or science["resume_policy"] != "original-no-prefix-requested; phase-one-prefix-refused"
            or not isinstance(science["output_files"], list) or not science["output_files"]):
        raise CachePolicyError("Scientific closure does not match the original actual stage")
    for row in science["output_files"]:
        _descriptor(row)
    _exact_keys(science["phase1"], {"configuration", "completion"}, "Exact original P1 configuration/completion closure required")
    if stage == "phase1":
        if science["calibration_join"] is not None or not isinstance(science["phase1"], dict):
            raise CachePolicyError("P1 cannot claim fitted calibration/application data")
    elif not isinstance(science["calibration_join"], dict):
        raise CachePolicyError("A genuine global C/P1/application join is required")
    if stage != "phase1":
        if (science["calibration_join"].get("format") != "darsia-common-phase-calibration-join-v1"
                or science["calibration_join"].get("actual_frozen_refit_verified") is not True):
            raise CachePolicyError("An unverified calibration hash cannot close the common DAG")
    for row in science["phase1"].values():
        # These are scientific NAS inputs, verified by the actual close. The
        # pre-import cache resolver validates their pins without opening them.
        _descriptor(row)
    _checked_source_verification(science["source_verification"], task)
    _descriptor(science["worker_scientific_evidence"])
    _checked_compute_evidence(science["compute_evidence"], stage=stage, cases=task["ordered_case_ids"])
    if any(row.get("source") not in science["output_files"]
           for row in science["compute_evidence"]["observations"]):
        raise CachePolicyError("Compute facts must come from this stage's actual indexed outputs")
    resource = science["result_space"]
    if (not isinstance(resource, dict) or resource.get("configuration") != value["configuration"]
            or resource.get("case_count") != len(task["ordered_case_ids"])):
        raise CachePolicyError("Common output bound differs from its exact stage configuration")
    _integer(resource.get("maximum_output_bytes"), "Released common output bound", positive=True)
    if stage == "phase2":
        rows = science["lean_frame_bytes"]
        if (not isinstance(rows, list) or [row.get("case_id") for row in rows] != task["ordered_case_ids"]
                or any(type(row.get("actual_bytes")) is not int or row["actual_bytes"] < 0
                    or type(row.get("admitted_bytes")) is not int
                    or row["actual_bytes"] > row["admitted_bytes"] for row in rows)):
            raise CachePolicyError("Every actual lean frame must fit its admitted byte bound")
    elif science["lean_frame_bytes"] != []:
        raise CachePolicyError("Only the actual lean stage can claim lean-frame measurements")
    return deepcopy(science)


def _checked_source_verification(value, task):
    """Require exact post-retirement rehash coverage of the original catalogue."""
    expected = task["input_catalog"]["inputs"]
    if (not isinstance(value, dict) or value.get("format") != "darsia-common-retired-input-bytes-v1"
            or value.get("catalogue_sha256") != _digest(task["input_catalog"])
            or value.get("inputs") != expected or value.get("input_count") != len(expected)
            or value.get("post_retirement") is not True or value.get("all_sha_and_sizes_verified") is not True):
        raise CachePolicyError("Actual post-retirement verification of every consumed input is required")
    _integer(value.get("distinct_physical_files"), "Verified physical input count", positive=True)
    return deepcopy(value)


def _hash_bound_file(binding, *, expected_version=None):
    """Stream only original bytes after retirement; no scientific package import."""
    import stat
    pin = _descriptor(binding)
    path = Path(pin["path"])
    before = retry_transient(path.lstat)
    version = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    if (not stat.S_ISREG(before.st_mode) or getattr(before, "st_reparse_tag", 0)
            or getattr(before, "st_file_attributes", 0) & 0x400
            or expected_version is not None and tuple(expected_version) != version(before)):
        raise CachePolicyError("An original resolved input/output identity changed before hash close")
    digest = hashlib.sha256()
    total = 0
    with retry_transient(lambda: path.open("rb")) as stream:
        if version(os.fstat(stream.fileno())) != version(before):
            raise CachePolicyError("An original file changed while opening its hash close")
        while block := retry_transient(lambda: stream.read(1024 * 1024)):
            total += len(block)
            digest.update(block)
        opened_after = os.fstat(stream.fileno())
    if (version(retry_transient(path.lstat)) != version(before) or version(opened_after) != version(before)
            or total != pin["size_bytes"] or digest.hexdigest() != pin["sha256"]):
        raise CachePolicyError("Actual original input/output SHA or bytes changed after retirement")
    return version(before)


def _checked_resolved_inputs(resolved, task):
    """Cross-link original worker-resolved paths/versions to every catalogue input."""
    expected = task["input_catalog"]["inputs"]
    if (not isinstance(resolved, list)
            or [dict(reference=row.get("reference"), uses=row.get("uses")) for row in resolved] != expected):
        raise CachePolicyError("The actual worker resolution omits or changes an original catalogue input")
    for row in resolved:
        version = row.get("version")
        if (not isinstance(row.get("path"), str) or not row["path"]
                or not isinstance(version, list) or len(version) != 4
                or any(type(value) is not int or value < 0 for value in version)):
            raise CachePolicyError("An actual worker-resolved input has no exact file path/version")
    return resolved


def _reverify_source_inputs(first, lease, resolved):
    """Rehash every exact original worker-resolved catalogue input after retirement."""
    _hash_bound_file(first.stores_source)
    expected = first.task["input_catalog"]["inputs"]
    _checked_resolved_inputs(resolved, first.task)
    versions = {}
    for row in resolved:
        version = row.get("version")
        if (not isinstance(version, list) or len(version) != 4
                or any(type(value) is not int or value < 0 for value in version)):
            raise CachePolicyError("An actual worker-resolved input has no exact file version")
        pin = dict(path=row["path"], sha256=row["reference"]["sha256"],
                   size_bytes=row["reference"]["bytes"])
        current = _hash_bound_file(pin, expected_version=version)
        if pin["path"] in versions and versions[pin["path"]] != current:
            raise CachePolicyError("One original input has inconsistent physical resolutions")
        versions[pin["path"]] = current
    proof = dict(format="darsia-common-retired-input-bytes-v1",
        catalogue_sha256=_digest(first.task["input_catalog"]), inputs=deepcopy(expected),
        input_count=len(resolved), distinct_physical_files=len(versions),
        post_retirement=True, all_sha_and_sizes_verified=True)
    _checked_source_verification(proof, first.task)
    return proof


def _checked_worker_evidence(worker, first, observation, pin, read):
    """Bind actual scientific work to the original owner's recorded exit inventory."""
    _exact_keys(worker, {"format", "request", "provider", "stores_source", "stage", "task", "owner",
        "native_launch", "service_receipt", "configuration", "completion", "publication", "phase1",
        "calibration_join", "output_files", "frame_records", "resolved_inputs", "compute_evidence",
        "result_space", "lean_bound", "complete", "process_retirement_claimed"},
        "Exact admitted-worker scientific closure required")
    if (worker["format"] != "darsia-common-worker-scientific-closure-v1"
            or worker["request"] != first.binding or worker["provider"] != first.provider_binding
            or worker["stores_source"] != first.stores_source or worker["stage"] != first.stage
            or worker["task"] != first.task or worker["configuration"] != first.request["configuration"]
            or worker["native_launch"] != observation.get("native_launch")
            or worker["service_receipt"] != observation.get("service_receipt")
            or worker["complete"] is not True or worker["process_retirement_claimed"] is not False):
        raise CachePolicyError("Scientific work is outside its actual original admitted worker")
    owners = observation.get("actual_registered_owner_exit_inventories", ())
    matches = [row for row in owners if row.get("common_worker_scientific_evidence") == pin]
    if len(matches) != 1 or matches[0].get("owner") != worker["owner"]:
        raise CachePolicyError("Scientific closure lacks exactly one original owner exit inventory")
    exited = read(_descriptor(matches[0]["inventory"]))
    modules = exited.get("first_qualification_modules", {})
    if (exited.get("owner") != worker["owner"] or modules.get("actual_task") != first.task
            or modules.get("common_worker_scientific_evidence") != pin
            or modules.get("predeclared_module_closure_sha256") != _digest(first.provider["module_closure"])
            or modules.get("all_file_backed_modules_predeclared_and_exact") is not True):
        raise CachePolicyError("Scientific closure was not finalized before its exact source/module exit inventory")
    _checked_resolved_inputs(worker["resolved_inputs"], first.task)
    _descriptor(worker["completion"])
    _exact_keys(worker["publication"], {"job", "done", "products_index"}, "Original worker publication pins required")
    if worker["publication"]["job"] != first.request["job"]:
        raise CachePolicyError("Scientific closure replaces the original immutable job")
    for bound in [*worker["publication"].values(), *worker["output_files"]]:
        _descriptor(bound)
    _checked_compute_evidence(worker["compute_evidence"], stage=first.stage, cases=first.task["ordered_case_ids"])
    if first.stage == "phase2":
        if (not isinstance(worker["lean_bound"], dict)
                or [row.get("frame_id") for row in worker["frame_records"]] != first.task["ordered_case_ids"]):
            raise CachePolicyError("Actual admitted-worker lean calculation omits its exact frames")
        _integer(worker["lean_bound"].get("admitted_frame_bytes"), "Original lean frame bound", positive=True)
        _integer(worker["lean_bound"].get("admitted_run_bytes"), "Original lean run bound", positive=True)
    elif worker["lean_bound"] is not None:
        raise CachePolicyError("A non-lean worker cannot claim a lean calculation")
    return worker


def _compute_control(config, stage):
    """Normalize only the SHA-bound selection fields; safe before numerical imports."""
    source = config.get("statistics", {}) if stage == "phase1" else config
    value = source.get("opencl_acceleration")
    if stage == "calibration":
        return None
    if "opencl_acceleration" not in source or value is None:
        return None
    try:
        return optional_n1_configuration(value)
    except ValueError as error:
        raise CachePolicyError(str(error)) from error



def _checked_compute_evidence(value, *, stage, cases):
    """Check declared controls and actual per-frame candidate and CPU comparisons."""
    if (not isinstance(value, dict) or value.get("format") != "darsia-common-observed-compute-v1"
            or value.get("cpu_fallback_qualified") is not True
            or not isinstance(value.get("qualified_devices"), list)
            or not isinstance(value.get("observations"), list)
            or value.get("declared_control") is not (stage != "calibration")):
        raise CachePolicyError("Actual optional-compute observations and CPU fallback are required")
    if stage == "calibration":
        if value["observations"] or value["qualified_devices"] or value.get("control") is not None:
            raise CachePolicyError("The refit cannot claim selected-field OpenCL evidence")
        return value
    if [row.get("case_id") for row in value["observations"]] != cases:
        raise CachePolicyError("Optional-compute evidence omits or reorders actual frames")
    if not isinstance(value.get("control"), dict) or value["control"].get("mode") not in {"cpu", "auto"}:
        raise CachePolicyError("Common science needs its explicitly declared CPU or auto control")
    devices = []
    for row in value["observations"]:
        _descriptor(row.get("source"))
        facts = row.get("facts")
        if (not isinstance(facts, dict) or facts.get("control") != value.get("control")
                or type(facts.get("used")) is not bool):
            raise CachePolicyError("An actual frame changes its optional-compute control")
        passed = []
        for candidate in facts.get("candidates", ()):
            if candidate.get("device") is not None:
                _checked_device_identity(candidate["device"])
            if candidate.get("identical") is True:
                device = candidate.get("device")
                _checked_device_identity(device)
                passed.append(device)
                if device not in devices:
                    devices.append(deepcopy(device))
        selected = facts.get("selected_device")
        if selected is not None:
            _checked_device_identity(selected)
            if selected not in passed:
                raise CachePolicyError("The selected device lacks actual bit-identity replays")
        if facts["used"] and selected is None:
            raise CachePolicyError("Actual acceleration has no selected device/driver identity")
        if facts["used"] and (facts.get("verification_required") is not True
                or type(facts.get("accelerated_calls")) is not int
                or facts["accelerated_calls"] < 1
                or facts.get("cpu_comparison_calls") != facts["accelerated_calls"]
                or facts.get("comparison_failures") != 0):
            raise CachePolicyError("Every accelerated proof call must match its original CPU fields")
    if devices != value["qualified_devices"]:
        raise CachePolicyError("Approved optional devices differ from actual passing candidates")
    return value


def _checked_device_identity(value):
    """Require all actual selected device and driver identity fields."""
    keys = {"device_id", "platform", "platform_version", "name", "driver", "global_memory_bytes"}
    _exact_keys(value, keys, "An exact observed device/driver identity is required")
    _integer(value["device_id"], "Observed device index")
    _integer(value["global_memory_bytes"], "Observed device memory", positive=True)
    if any(not isinstance(value[key], str) or not value[key].strip()
           for key in ("platform", "platform_version", "name", "driver")):
        raise CachePolicyError("Observed device/runtime identity is incomplete")


def _compute_evidence(config, stage, cases, files, reader):
    """Collect compute facts only from named original indexed stage outputs."""
    control, observations, devices = _compute_control(config, stage), [], []
    if stage != "calibration":
        leaf = "native_statistics_runtime.json" if stage == "phase1" else "result.json"
        found = {PureWindowsPath(row["path"]).parent.name: row for row in files
                 if PureWindowsPath(row["path"]).name == leaf}
        if set(found) != set(cases):
            raise CachePolicyError("Every actual frame must retain its compute runtime evidence")
        for case in cases:
            record = reader.json(found[case])
            facts = deepcopy(record.get("pixel_backend_facts", {}).get("opencl_n1"))
            if facts is None:
                raise CachePolicyError("An actual frame lacks optional-compute facts")
            observations.append(dict(case_id=case, facts=facts, source=deepcopy(found[case])))
            for candidate in facts.get("candidates", ()):
                if candidate.get("identical") is True and candidate.get("device") not in devices:
                    devices.append(deepcopy(candidate["device"]))
    result = dict(format="darsia-common-observed-compute-v1", control=control,
        declared_control=stage != "calibration" and control is not None,
        observations=observations, qualified_devices=devices, cpu_fallback_qualified=True)
    _checked_compute_evidence(result, stage=stage, cases=cases)
    return result


def close_stage(first, lease):
    """Compose immutable metadata after retirement; never import scientific code."""
    worker, wp = _slot_read(first.provider, "worker_scientific_evidence")
    observation = _slot_read(first.provider, "observation")[0]
    _checked_worker_evidence(worker, first, observation, wp, first._read)
    for pin in [worker["completion"], *worker["publication"].values(), *worker["output_files"]]:
        _hash_bound_file(pin)
    source_verification = _reverify_source_inputs(first, lease, worker["resolved_inputs"])
    stage, task = first.stage, first.task
    compute = deepcopy(worker["compute_evidence"])
    compute["kernel_lifetime_memory"] = deepcopy(observation.get("kernel_lifetime_memory"))
    compute["native_launch_to_retirement_seconds"] = observation.get("native_launch_to_retirement_seconds")
    result_space = deepcopy(worker["result_space"])
    lean_bytes = []
    if stage == "phase2":
        measured = _slot_read(first.provider, "measurement")[0]
        lean_bytes, run_bytes = _lean_frame_measurements(first,
            dict(records=worker["frame_records"]), worker["output_files"],
            measured, worker["lean_bound"])
        result_space["lean_run_bytes"] = run_bytes
    science = dict(format="darsia-common-phase-scientific-closure-v1", stage=stage,
        task=deepcopy(task), completion=deepcopy(worker["completion"]),
        configuration=deepcopy(worker["configuration"]),
        pipeline=PIPELINES[stage], run=task["run"], ordered_case_ids=deepcopy(task["ordered_case_ids"]),
        storage_profile=PROFILES[stage], colour_intermediates="keep", phase1=deepcopy(worker["phase1"]),
        calibration_join=deepcopy(worker["calibration_join"]), output_files=deepcopy(worker["output_files"]),
        resume_policy="original-no-prefix-requested; phase-one-prefix-refused",
        source_bytes_reverified=True, source_verification=source_verification,
        worker_scientific_evidence=wp,
        compute_evidence=compute, result_space=result_space, lean_frame_bytes=lean_bytes, complete=True)
    sp = _slot_write(first.provider, "scientific_evidence", science)
    marker_name = "cache-first-" + first._once_key() + ".json"
    with lease.ledger.locked(create_binding=False):
        raw, _identity = lease.ledger.control.read_once(marker_name)
        once_marker = dict(path=str(Path(lease.ledger.bound["path"]) / marker_name),
            sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    pins = {role: _slot_read(first.provider, role)[1] for role in (
        "observation", "native_completion", "measurement", "writer_evidence")}
    proof = dict(format=STAGE_PROOF, provider=deepcopy(first.provider_binding),
        stores_source=deepcopy(first.stores_source), stage=stage, request=deepcopy(first.binding),
        task=deepcopy(first.request["task"]), job=deepcopy(first.request["job"]), completion=deepcopy(worker["completion"]),
        configuration=deepcopy(worker["configuration"]), scientific_evidence=sp,
        device_identity=deepcopy(first.root_provider["device_identity"]), once_marker=once_marker,
        complete=True, passed=True, **pins)
    checked_stage(proof, root=first.root_provider, provider_binding=first.provider_binding,
        stores_source=first.stores_source, read=first._read, read_authorities=first._read_authorities)
    return _slot_write(first.provider, "completion", proof)


def _lean_frame_measurements(first, completed, files, measured, bound):
    """Count terminal file lengths, including retained serialization partials."""
    from .publication_paths import retained_member
    watched = PureWindowsPath(first.output_entitlement["watched_output_scope"]["path"])
    snapshot = measured.get("final_snapshot", {})
    stats = snapshot.get("file_stats")
    if not isinstance(stats, list) or snapshot.get("observed_scan_anomalies"):
        raise CachePolicyError("Complete terminal output file lengths are required for lean validation")
    lengths = {}
    for row in stats:
        relative = PureWindowsPath(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise CachePolicyError("A terminal output length leaves its watched root")
        path = watched / relative
        if path in lengths:
            raise CachePolicyError("A terminal output file is counted twice")
        lengths[path] = _integer(row["size_bytes"], "Actual terminal file length")
    rows, attempt = [], None
    for record in completed["records"]:
        case = record["frame_id"]
        directory = PureWindowsPath(record["nas_directory"])
        if not directory.is_relative_to(watched) or directory.name != case:
            raise CachePolicyError("The actual lean frame leaves its released watched output")
        producer = record.get("producer")
        if not isinstance(producer, dict) or producer.get("case_id") != case:
            raise CachePolicyError("The lean frame needs its original checkpoint producer descriptors")
        own_attempt = PureWindowsPath(producer["attempt_directory"])
        if not own_attempt.is_relative_to(watched) or not directory.is_relative_to(own_attempt):
            raise CachePolicyError("The original lean producer leaves its released output attempt")
        if attempt is not None and own_attempt != attempt:
            raise CachePolicyError("A common lean run mixes actual attempt directories")
        attempt = own_attempt
        original = [_descriptor(producer[field]) for field in ("commit", "frame_registration", "publication")]
        if any(pin not in files for pin in original):
            raise CachePolicyError("An original checkpoint receipt is absent from the actual indexed outputs")
        receipts = [PureWindowsPath(pin["path"]) for pin in original[1:]]
        if any(not path.is_relative_to(own_attempt) or path not in lengths for path in receipts):
            raise CachePolicyError("Both actual lean operation receipts must be measured")
        actual = sum(size for path, size in lengths.items()
            if any(retained_member(path, target) for target in [directory, *receipts]))
        admitted = bound["admitted_frame_bytes"]
        if actual > admitted:
            raise CachePolicyError("Actual lean frame bytes exceed the admitted bound: " + case)
        rows.append(dict(case_id=case, actual_bytes=actual, admitted_bytes=admitted,
                         retained_partials_included=True, operation_receipts_included=True))
    if [row["case_id"] for row in rows] != first.task["ordered_case_ids"]:
        raise CachePolicyError("Lean byte measurements omit or reorder an actual frame")
    names = bound.get("run_record_files")
    if (not isinstance(names, list) or len(names) != len(set(names)) or not names
            or any(not isinstance(name, str) or PureWindowsPath(name).name != name for name in names)):
        raise CachePolicyError("The original admitted writer must declare its exact run-record file names")
    run_bytes = sum(size for path, size in lengths.items()
        if any(retained_member(path, attempt / name) for name in names))
    if run_bytes > bound["admitted_run_bytes"]:
        raise CachePolicyError("Actual lean run-store/record bytes exceed the admitted bound")
    return rows, dict(actual_bytes=run_bytes, admitted_bytes=bound["admitted_run_bytes"], retained_partials_included=True)


def _closed_stages(binding, root, stores_source, read, read_authorities):
    """Resolve and cross-check all three genuine immutable predecessor joins."""
    stages, proofs = {}, {}
    for stage in STAGES:
        proof, pin = _slot_read(stage_provider(root, stage), "completion")
        checked_stage(proof, root=root, provider_binding=binding, stores_source=stores_source,
                      read=read, read_authorities=read_authorities)
        stages[stage], proofs[stage] = pin, proof
    scientific = {stage: _slot_read(stage_provider(root, stage), "scientific_evidence")[0] for stage in STAGES}
    p1 = scientific["phase1"]["phase1"]
    c = scientific["calibration"]["calibration_join"]
    if (scientific["calibration"]["phase1"] != p1 or scientific["phase2"]["phase1"] != p1
            or scientific["phase2"]["calibration_join"] != c
            or c["phase1_configuration"] != p1["configuration"]
            or c["phase1_completion"] != p1["completion"]
            or c["binding"] != dict(completion=proofs["calibration"]["completion"],
                plan=proofs["calibration"]["configuration"])):
        raise CachePolicyError("Completed common stages do not share the genuine exact P1/C/P2 join")
    return stages, proofs


def _checked_source_review(value, root):
    """Check the released source closure and exact validation/release record text."""
    _exact_keys(value, {"format", "verdict", "base_commit", "tree", "module_closure_sha256",
        "producer_source_closure", "answer"}, "Content-bound full source review required")
    answer = _exact_keys(value["answer"], {"id", "sha256", "text"}, "Exact review answer identity required")
    if (value["format"] != "darsia-common-source-review-v1" or value["verdict"] != "PASS"
            or value["base_commit"] != root["scope"]["commit"]
            or re.fullmatch(r"[0-9a-f]{40}", str(value["tree"])) is None
            or value["module_closure_sha256"] != _digest(root["module_closure"])
            or value["producer_source_closure"] != root["producer_source_closure"]
            or not isinstance(answer["id"], str) or not answer["id"].strip()
            or not isinstance(answer["text"], str) or not answer["text"].strip()
            or hashlib.sha256(answer["text"].encode("utf-8")).hexdigest() != answer["sha256"]):
        raise CachePolicyError("The full review does not cover the actual released source and answer bytes")
    return deepcopy(value)


def _checked_validation_receipt(value, *, kind, root):
    """Require a passing, nonempty test result on the released source closure."""
    _exact_keys(value, {"format", "kind", "verdict", "scope", "module_closure_sha256",
        "tests_passed", "tests_failed"}, "Exact completed validation receipt required")
    if (value["format"] != "darsia-common-validation-receipt-v1" or value["kind"] != kind
            or value["verdict"] != "PASS" or value["scope"] != root["scope"]
            or value["module_closure_sha256"] != _digest(root["module_closure"])
            or type(value["tests_failed"]) is not int or value["tests_failed"] != 0):
        raise CachePolicyError("A readable validation file is not a passing test result for this source")
    _integer(value["tests_passed"], "Actually passed tests", positive=True)
    return deepcopy(value)


def _checked_comparison(value, *, binding, root, stores_source, stages, comparator, read):
    """Bind one auto chain's outputs, in-run CPU verification and intact report.

    The comparator is independently accepted by the final release. This reader
    uses the existing per-call verification predicates. CPU timings cover the
    matched N1 calls inside this chain; no separate CPU end-to-end run is
    implied. Report values are linked to their original observations without
    imposing performance or memory acceptance thresholds.
    """
    _exact_keys(value, {"format", "verdict", "comparator", "scope", "stores_source", "provider", "stages",
        "stage_results", "selected_fields", "measurement"}, "Accepted in-run comparison content required")
    if (value["format"] != "darsia-common-inrun-comparison-v1" or value["verdict"] != "PASS"
            or value["scope"] != root["scope"] or value["stores_source"] != stores_source
            or value["comparator"] != comparator or value["provider"] != binding or value["stages"] != stages):
        raise CachePolicyError("The accepted comparator does not bind this proof scope and stores")
    _hash_bound_file(_descriptor(comparator))
    sciences = {}
    for stage in STAGES:
        science, science_pin = _slot_read(stage_provider(root, stage), "scientific_evidence")
        compute = _checked_compute_evidence(science["compute_evidence"], stage=stage,
            cases=root["stages"][stage]["ordered_case_ids"])
        if stage != "calibration" and compute["control"]["mode"] != "auto":
            raise CachePolicyError("The original proof chain changes its declared auto control")
        sciences[stage] = science, science_pin
    _exact_keys(value["stage_results"], STAGES, "P1, global calibration and P2 comparisons required")
    for stage in STAGES:
        result = _exact_keys(value["stage_results"][stage],
            {"scientific_evidence", "outputs", "verdict"},
            "Exact complete stage output acceptance required")
        if result["verdict"] != "PASS":
            raise CachePolicyError("Every actual stage output must pass the accepted comparator")
        science, pin = sciences[stage]
        if result["scientific_evidence"] != pin or result["outputs"] != science["output_files"]:
            raise CachePolicyError("Output acceptance omits or replaces an actual indexed stage output")
    fields = ("aqueous", "gas", "concentration", "mass_total", "mass_aqueous", "mass_gas")
    _exact_keys(value["selected_fields"], {"phase1", "phase2"}, "Both phases need six-field acceptance")
    for stage in ("phase1", "phase2"):
        result = value["selected_fields"][stage]
        observations = sciences[stage][0]["compute_evidence"]["observations"]
        if not isinstance(result, list) or len(result) != len(observations):
            raise CachePolicyError("Each selected frame needs all six original pixel fields accepted")
        for row, observed in zip(result, observations):
            _exact_keys(row, {"case_id", "source", "facts_sha256", "fields"},
                "Original per-frame CPU verification source required")
            if (row["case_id"] != observed["case_id"] or row["source"] != observed["source"]
                    or row["facts_sha256"] != _digest(observed["facts"])
                    or row["fields"] != {field: "PASS" for field in fields}):
                raise CachePolicyError("Six-field acceptance replaces its original in-run CPU verification facts")
    measurement = read(_descriptor(value["measurement"]))
    _exact_keys(measurement, {"format", "provider", "stages", "scope", "end_to_end_scope", "stage_measurements"},
        "Intact original throughput and memory report required")
    if (measurement["format"] != "darsia-common-inrun-measurement-v1" or measurement["scope"] != root["scope"]
            or measurement["provider"] != binding or measurement["stages"] != stages
            or measurement["end_to_end_scope"] != "original native launch through accepted retirement"):
        raise CachePolicyError("Measurements must report this original chain and actual wall-time scope")
    rows = _exact_keys(measurement["stage_measurements"], STAGES, "Every original stage needs its measurement report")
    for stage in STAGES:
        row = _exact_keys(rows[stage], {"stage", "end_to_end_wall_seconds", "kernel_lifetime_memory",
            "host_binding", "application_device_buffer_bytes", "compute_observations"},
            "Original stage wall, memory, CPU/device timings and verification overhead required")
        science, _pin = sciences[stage]
        compute = science["compute_evidence"]
        if (row["stage"] != stages[stage] or row["kernel_lifetime_memory"] != compute["kernel_lifetime_memory"]
                or row["end_to_end_wall_seconds"] != compute.get("native_launch_to_retirement_seconds")
                or row["host_binding"] != root["namespace_admission"]
                or row["compute_observations"] != compute["observations"]):
            raise CachePolicyError("The report replaces original native, host or in-run CPU/device observations")
        observed_peak = max((fact["facts"].get("peak_application_device_buffer_bytes",
            fact["facts"].get("application_device_buffer_bytes", 0))
            for fact in compute["observations"]), default=0)
        if row["application_device_buffer_bytes"] != observed_peak:
            raise CachePolicyError("Reported application memory differs from the original device buffer facts")
        read(_descriptor(row["host_binding"]))
    return deepcopy(value)


def _ordinary_release(value, *, binding, root, stores_source, stages, read):
    """Require independent review, executed tests and accepted comparison contents."""
    _exact_keys(value, {"format", "provider", "stores_source", "scope", "stages",
        "ordinary_use_authorized", "source_review", "importer_focus", "full_suite",
        "accepted_comparator", "accepted_comparison"},
        "Explicit final release, completed tests and accepted in-run comparison required")
    if (value["format"] != "darsia-common-phase-ordinary-release-v1"
            or value["provider"] != binding or value["stores_source"] != stores_source
            or value["scope"] != root["scope"] or value["stages"] != stages
            or value["ordinary_use_authorized"] is not True):
        raise CachePolicyError("A completed common chain is not its own ordinary-use authorization")
    _checked_source_review(read(_descriptor(value["source_review"])), root)
    for role in ("importer_focus", "full_suite"):
        _checked_validation_receipt(read(_descriptor(value[role])), kind=role, root=root)
    _checked_comparison(read(_descriptor(value["accepted_comparison"])), binding=binding, root=root,
        stores_source=stores_source, stages=stages, comparator=_descriptor(value["accepted_comparator"]), read=read)
    return deepcopy(value)


def _profile_documents(binding, root, stores_source, stages, proofs, read):
    """Derive both ordinary profiles from actual closed stage evidence."""
    evidence = {stage: _slot_read(stage_provider(root, stage), "writer_evidence")[0] for stage in STAGES}
    native = {stage: _slot_read(stage_provider(root, stage), "native_completion")[0] for stage in STAGES}
    bounds = [row["capacity_bound"] for row in evidence.values()]
    if any(bound["volume_cap_observation"]["binding"] != bounds[0]["volume_cap_observation"]["binding"]
           for bound in bounds):
        raise CachePolicyError("Common stages used different actual physical-volume bounds")
    maximum = max(_integer(row["administrative_service_entitlement"]["maximum_bytes"],
                           "Actual common service reservation", positive=True) for row in evidence.values())
    family = max(_integer(row["maximum_family_processes"], "Actual native family count", positive=True)
                 for row in native.values())
    observations = [_slot_read(stage_provider(root, stage), "observation")[0] for stage in STAGES]
    roles = [read(_descriptor(row["service_receipt"]))["role_identifier"] for row in observations]
    if len(set(roles)) != 1:
        raise CachePolicyError("Common stages did not use the same exact original worker entry role")
    sampled = max(_integer(row["sampled_cache_logical_high_water_bytes"], "Actual retained cache sample")
                  for row in observations)
    sciences = {stage: _slot_read(stage_provider(root, stage), "scientific_evidence")[0]
                for stage in STAGES}
    shared = dict(provider=deepcopy(binding), stores_source=deepcopy(stores_source),
        entry_kind="worker", role_identifier=roles[0],
        scope=deepcopy(root["scope"]), stages=deepcopy(stages),
        stage_routes={stage: {key: deepcopy(row[key]) for key in (
            "pipeline", "storage_profile", "colour_intermediates")}
                      for stage, row in root["stages"].items()},
        compute_profiles={stage: deepcopy(row["compute_evidence"]) for stage, row in sciences.items()},
        result_spaces={stage: deepcopy(row["result_space"]) for stage, row in sciences.items()},
        namespace_admission=deepcopy(root["namespace_admission"]), device_identity=deepcopy(root["device_identity"]),
        python_executable=deepcopy(root["python_executable"]), module_closure=deepcopy(root["module_closure"]),
        producer_source_closure=deepcopy(root["producer_source_closure"]),
        runtime_source_closure=deepcopy(root["runtime_source_closure"]),
        capacity_bound=deepcopy(bounds[-1]), maximum_cache_bytes=maximum,
        sampled_cache_highwater_bytes=sampled,
        operational_service_entitlement_bytes=maximum, additional_claim_cache_bytes=0,
        allocation_model="shared-service-only-no-claim-files-v1", result_estimate_excludes_cache=True,
        actual_writer_IO_proven=False, sampled_highwater_is_lifetime_peak=False,
        writer_contract="physical-volume-bound-shared-service-v1",
        completion=deepcopy(stages["phase2"]),
        measurement=deepcopy(proofs["phase2"]["measurement"]),
        native_supervision=dict(format="darsia-native-worker-supervision-profile-v1",
            completion=deepcopy(proofs["phase2"]["native_completion"]), startup=deepcopy(root["startup"]),
            maximum_family_processes=family, publication_contract="alias-done-small-metadata-v1"))
    return dict(import_profile=dict(format=IMPORT_PROFILE, purpose="jobs", **deepcopy(shared)),
                job_profile=dict(format=JOB_PROFILE, purpose="jobs", **deepcopy(shared)))


def publish_profiles(binding, *, stores_source, ordinary_release, read, read_authorities):
    """Explicit explicit composition from real closed slots; no science or claim.

    The independent final release names only already-existing stage proofs and
    reviews, so profiles and their bundle have no circular/future hashes. This
    function adopts no map/model/calibration and changes no store settings.
    """
    binding = _descriptor(binding)
    root = checked_provider(read(binding))
    stages, proofs = _closed_stages(binding, root, stores_source, read, read_authorities)
    _ordinary_release(read(_descriptor(ordinary_release)), binding=binding, root=root,
        stores_source=stores_source, stages=stages, read=read)
    profiles = _profile_documents(binding, root, stores_source, stages, proofs, read)
    pins = {role: _slot_write(root, role, value) for role, value in profiles.items()}
    bundle = dict(format=BUNDLE, provider=deepcopy(binding), stores_source=deepcopy(stores_source),
        scope=deepcopy(root["scope"]), stages=stages, profiles=pins,
        ordinary_use_release=deepcopy(ordinary_release), complete=True, passed=True)
    return _slot_write(root, "completion", bundle)


def resolve_profile(binding, value, kind, read, *, stores_source=None, read_authorities=None):
    """Passive closed-DAG resolution; an incomplete first stage never qualifies."""
    root = checked_provider(value)
    _device(root, read, current=True)
    if kind not in {"import", "jobs"} or stores_source is None:
        raise CachePolicyError("Common resolution needs its actual stores and typed role")
    bundle, _ = _slot_read(root, "completion")
    _exact_keys(bundle, {"format", "provider", "stores_source", "scope", "stages", "profiles",
        "ordinary_use_release", "complete", "passed"}, "Exact completed common provider DAG required")
    if (bundle["format"] != BUNDLE or bundle["provider"] != binding
            or bundle["stores_source"] != stores_source or bundle["scope"] != root["scope"]
            or bundle["complete"] is not True or bundle["passed"] is not True):
        raise CachePolicyError("Actual common provider/source/stores are not completed")
    stages, proofs = _closed_stages(binding, root, stores_source, read, read_authorities)
    if bundle["stages"] != stages:
        raise CachePolicyError("The common bundle replaces or omits an actual closed stage")
    _ordinary_release(read(_descriptor(bundle["ordinary_use_release"])), binding=binding, root=root,
        stores_source=stores_source, stages=stages, read=read)
    profiles = _profile_documents(binding, root, stores_source, stages, proofs, read)
    _exact_keys(bundle["profiles"], profiles.keys(), "Both exact common import/job roles required")
    selected = None
    for role, expected in profiles.items():
        actual, pin = _slot_read(root, role)
        if bundle["profiles"][role] != pin or _digest(actual) != _digest(expected):
            raise CachePolicyError("A common profile differs from its exact original provider DAG")
        if role == ("import_profile" if kind == "import" else "job_profile"):
            selected = actual
    return selected


def require_applicable(profile, task, *, config=None):
    """Same source/device/profile rule for both production phases; no fallback."""
    if profile.get("format") != JOB_PROFILE or profile.get("purpose") != "jobs":
        raise CachePolicyError("Actually resolved common job profile required")
    scope = profile["scope"]
    if any(task.get(key) != scope[key] for key in ("commit", "backend", "precision")):
        raise CachePolicyError("Common profile does not cover another source/backend/precision")
    matches = [stage for stage, row in profile["stage_routes"].items()
               if row["pipeline"] == task.get("pipeline")]
    if len(matches) != 1:
        raise CachePolicyError("Common profile does not cover this actual pipeline")
    stage = matches[0]
    _cases(task.get("ordered_case_ids"))
    if (task.get("storage_profile", "full") != PROFILES[stage]
            or task.get("format_version") != (2 if stage == "phase2" else 1)):
        raise CachePolicyError("Common stage Full/Lean identity differs; no full-profile fallback")
    if config is not None:
        if config.get("run") != task.get("run"):
            raise CachePolicyError("Common configuration belongs to another run")
        if stage != "calibration" and [row.get("case_id") for row in config.get("frames", ())] != task["ordered_case_ids"]:
            raise CachePolicyError("Common configuration changes the exact selected case order")
        if stage == "phase2":
            from darsia.fluidflower.storage_profile import resolve
            setting = resolve(config)
            if setting != dict(profile="lean", colour_intermediates="keep"):
                raise CachePolicyError("Common phase two requires its actual Lean/KEEP configuration")
        compute = profile.get("compute_profiles", {}).get(stage)
        if not isinstance(compute, dict) or compute.get("control") != _compute_control(config, stage):
            raise CachePolicyError("The optional-compute control differs from its actual common proof")
    return stage


def runtime_compute_policy(profile, task, config):
    """Explicit per-run policy from the resolved profile, never an environment override."""
    stage = require_applicable(dict(profile, format=JOB_PROFILE), task, config=config)
    compute = profile["compute_profiles"][stage]
    if compute.get("cpu_fallback_qualified") is not True:
        raise CachePolicyError("The common proof does not cover the original CPU fallback")
    return dict(qualified_devices=deepcopy(compute["qualified_devices"]), cpu_fallback_qualified=True)




def checked_import_context(profile, *, purpose, role_identifier, entry_kind, policy, namespace):
    """Common native/startup eligibility stays worker/jobs only and cap exact."""
    if (profile.get("format") != IMPORT_PROFILE or purpose != "jobs" or entry_kind != "worker"
            or profile.get("purpose") != "jobs" or profile.get("entry_kind") != entry_kind
            or profile.get("role_identifier") != role_identifier or profile.get("namespace_admission") != namespace
            or policy.profiles.get("jobs") != profile.get("provider")
            or policy.volume_cap is None or policy.volume_cap.maximum_size_bytes != policy.quota_bytes
            or profile["capacity_bound"]["volume_cap_observation"].get("virtual_size_bytes") != policy.volume_cap.maximum_size_bytes
            or profile["capacity_bound"]["volume_cap_observation"].get("binding") != policy.volume_cap.document()
            or profile.get("additional_claim_cache_bytes") != 0
            or profile.get("actual_writer_IO_proven") is not False
            or profile.get("sampled_highwater_is_lifetime_peak") is not False):
        raise CachePolicyError("Common import context is outside the actual worker/source/namespace/physical cap")
    return _integer(profile["operational_service_entitlement_bytes"], "Actual qualified common service entitlement", positive=True)
