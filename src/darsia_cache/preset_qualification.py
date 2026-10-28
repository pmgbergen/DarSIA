"""One explicitly released first original Comparison execution.

This pre-import contract supplies administration, never a fabricated measured
profile. It reuses the original owned process, physical cache cap, one-shot
claim markers and retirement verifier. No scientific package is imported here.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import PureWindowsPath
import re

from .first_qualification import (
    FirstQualification, _digest, _exact_keys, _pins, _slot_read, _slot_write,
    administrative_entitlement, checked_required_capacity_bound,
    checked_bound_output_scope,
)
from .policy import CachePolicyError, PHYSICAL_RESERVE_BYTES, _descriptor

PROVIDER = "darsia-preset-wasserstein-first-provider-v1"
REQUEST = "darsia-preset-wasserstein-first-request-v1"
RELEASE = "darsia-preset-wasserstein-first-release-v1"
OUTPUT = "darsia-preset-wasserstein-first-output-v1"
CLOSED = "darsia-preset-wasserstein-first-closed-v1"
BOUNDED_PROFILE = "darsia-preset-wasserstein-bounded-profile-v1"
BOUNDED_COMPLETION = "darsia-preset-wasserstein-bounded-completion-v1"
SCOPE_FIELDS = {"commit", "pipeline", "backend", "precision", "target_machine", "physical_host",
    "run", "operation_sha256", "preset_inputs_sha256", "requested_time_count",
    "comparison_time", "expected_comparisons"}


def _positive(value, where):
    if type(value) is not int or value <= 0:
        raise CachePolicyError(where + " must be an explicit positive operator-supplied integer")
    return value


def _path(value):
    path = PureWindowsPath(value) if isinstance(value, str) else PureWindowsPath()
    if not path.is_absolute() or ".." in path.parts:
        raise CachePolicyError("An exact absolute preset qualification location is required")
    return path


def _metadata(binding):
    from .adapters import _worker_metadata
    pin = _descriptor(binding)
    return _worker_metadata(pin["path"], expected=pin)[0]


def checked_provider(value):
    _exact_keys(value, {"format", "qualification_id", "scope", "namespace_admission",
        "output_namespace", "slots", "python_executable", "startup", "module_closure",
        "producer_source_closure", "runtime_source_closure"}, "Exact preset first provider required")
    scope = _exact_keys(value["scope"], SCOPE_FIELDS, "Exact original preset first scope required")
    if (value["format"] != PROVIDER or re.fullmatch(r"[0-9a-f]{32}", str(value["qualification_id"])) is None
            or scope["pipeline"] != "preset-wasserstein" or scope["backend"] != "cpu"
            or scope["precision"] != "original" or re.fullmatch(r"[0-9a-f]{40}", str(scope["commit"])) is None
            or any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", str(scope[key])) is None
                   for key in ("target_machine", "physical_host"))
            or any(re.fullmatch(r"[0-9a-f]{64}", str(scope[key])) is None
                   for key in ("operation_sha256", "preset_inputs_sha256"))):
        raise CachePolicyError("Only the exact declared CPU/original preset route is admitted")
    for key in ("requested_time_count", "expected_comparisons"):
        _positive(scope[key], key)
    for key in ("namespace_admission", "python_executable", "startup"):
        _descriptor(value[key])
    target = _exact_keys(value["output_namespace"], {"path", "identity"}, "Actual accepted receipt namespace required")
    _path(target["path"])
    if not isinstance(target["identity"], dict) or not target["identity"]:
        raise CachePolicyError("Actual receipt namespace identity required")
    _exact_keys(value["slots"], {"observation", "measurement", "native_completion", "completion",
                               "bounded_profile", "bounded_completion"},
                "Only exact original first-execution evidence slots are supported")
    names = list(value["slots"].values())
    if (len(set(name.casefold() for name in names if isinstance(name, str))) != len(names)
            or any(not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,180}\.json", name) is None
                   for name in names)):
        raise CachePolicyError("Preset qualification slots must be distinct flat immutable names")
    union = _pins(value["producer_source_closure"]) + _pins(value["runtime_source_closure"])
    closure = _pins(value["module_closure"])
    if len(union) != len(closure) or {_digest(pin) for pin in union} != {_digest(pin) for pin in closure}:
        raise CachePolicyError("Exact producer/runtime union must equal the startup module closure")
    paths = {str(pin["path"]).replace("\\", "/").casefold() for pin in union}
    required = ("darsia_cache/preset_qualification.py", "darsia_cache/first_qualification.py",
        "darsia_cache/bootstrap.py", "darsia_cache/service_context.py", "darsia_cache/owned_process.py",
        "darsia_cache/service_lifecycle.py", "darsia_cache/worker_adapter.py", "darsia_cache/fresh_scope.py",
        "darsia/fluidflower/preset_wasserstein_capacity.py", "darsia/fluidflower/preset_wasserstein_inputs.py",
        "darsia/fluidflower/preset_wasserstein_runner.py", "darsia/fluidflower/preset_wasserstein_results.py",
        "darsia/fluidflower/preset_wasserstein_protocol.py", "darsia/fluidflower/pipeline/run_io.py",
        "darsia/fluidflower/pipeline/worker.py", "darsia/fluidflower/pipeline/worker_io.py",
        "darsia/presets/workflows/comparison/comparison_wasserstein.py", "darsia/presets/workflows/utils/mass.py",
        "darsia/presets/workflows/rig.py", "darsia/measure/wasserstein.py",
        "darsia/utils/plotting.py", "tempfile.py", "importlib/_bootstrap_external.py")
    if any(not any(path.endswith("/" + suffix) for path in paths) for suffix in required):
        raise CachePolicyError("Preset closure omits an actual startup/reader/science/output/lifecycle boundary")
    return deepcopy(value)


class PresetQualification(FirstQualification):
    """Exactly one immutable original task, one machine and one native family."""
    preset_mode = True

    def __init__(self, binding, *, provider_binding, stores_source, namespace, read,
                 read_authorities=None, historical_authorities=None, read_metadata=None):
        if historical_authorities is not None:
            raise CachePolicyError("Historical native qualifications cannot authorize a preset execution")
        self.binding, self.provider_binding = _descriptor(binding), _descriptor(provider_binding)
        self.request, self.provider = read(self.binding), checked_provider(read(self.provider_binding))
        self.stores_source, self._read = deepcopy(stores_source), read
        self._read_metadata = read_metadata or _metadata
        _exact_keys(self.request, {"format", "execution_authorized", "root_release", "provider", "stores_source",
            "namespace_admission", "public_command", "task", "job", "job_root", "configuration",
            "capacity_bound_plan", "maximum_service_bytes", "cache_stat_period_seconds", "output_entitlement_plan"},
            "Exact independently released preset first request required")
        if (self.request["format"] != REQUEST or self.request["execution_authorized"] is not True
                or self.request["provider"] != self.provider_binding or self.request["stores_source"] != stores_source
                or self.request["namespace_admission"] != namespace or namespace != self.provider["namespace_admission"]
                or type(self.request["cache_stat_period_seconds"]) not in (int, float)
                or not 0 < self.request["cache_stat_period_seconds"] <= 30):
            raise CachePolicyError("Preset request/provider/stores/namespace differs")
        admission = read(_descriptor(namespace))
        if str(admission.get("host", "")).casefold() != self.provider["scope"]["physical_host"].casefold():
            raise CachePolicyError("Preset physical host differs from its actual namespace admission")
        release = read(_descriptor(self.request["root_release"]))
        if (release.get("format") != RELEASE or release.get("execution_authorized") is not True
                or release.get("provider") != self.provider_binding or release.get("stores_source") != stores_source
                or release.get("scope") != self.provider["scope"]
                or release.get("request_payload_sha256") != _digest({k: v for k, v in self.request.items() if k != "root_release"})):
            raise CachePolicyError("Independent exact operator release of this preset request required")
        self.task = self._read_metadata(_descriptor(self.request["task"]))
        self.job = self._read_metadata(_descriptor(self.request["job"]))
        self.configuration = self._read_metadata(_descriptor(self.request["configuration"]))
        self._control_identity()
        # This shared parser checks only the genuine public finite worker argv;
        # it performs no Setup execution or evidence compatibility conversion.
        from .setup_qualification import _command
        _command(self.request["public_command"], self.request, self.provider)
        self.maximum_service_bytes = _positive(self.request["maximum_service_bytes"], "Preset service entitlement")
        self.required_capacity_bound = checked_required_capacity_bound(read(_descriptor(self.request["capacity_bound_plan"])))
        if self.required_capacity_bound["enforcement_state"] != "PREDECLARED_Q106_VOLUME_BOUND":
            raise CachePolicyError("An actual predeclared physical cache cap is required")
        output = read(_descriptor(self.request["output_entitlement_plan"]))
        _exact_keys(output, {"format", "task", "scope", "maximum_output_bytes", "maximum_transient_bytes", "watched_output_scope"},
                    "Explicit original whole-time output/transient administration required")
        if output["format"] != OUTPUT or output["task"] != self.request["task"] or output["scope"] != self.provider["scope"]:
            raise CachePolicyError("Preset output entitlement is for another actual task/scope")
        _positive(output["maximum_output_bytes"], "Whole-time output entitlement")
        if type(output["maximum_transient_bytes"]) is not int or output["maximum_transient_bytes"] < 0:
            raise CachePolicyError("Explicit nonnegative whole-time transient entitlement required")
        checked_bound_output_scope(output["watched_output_scope"], preset=True)
        self.output_entitlement = deepcopy(output)
        self.bound_mode = self.reference_build = False
        self.reference_administration = self.reference_retained_inventory = self.q123_cold_scope = None
        self.reference_roots_validated, self.reference_transport_observation = False, None
        self.started, self.claim = False, None

    def _control_identity(self):
        task, job, op, scope = self.task, self.job, self.configuration, self.provider["scope"]
        if (task.get("format") != "darsia-alias-task-v1" or job.get("format") != "darsia-alias-job-v1"
                or task.get("ordered_case_ids") != [scope["run"]] or task.get("threads") != 1
                or task.get("resume_from") is not None or task.get("storage_profile") != "full"
                or task.get("legacy_paths") is not False or task.get("execution_ready") is not True
                or re.fullmatch(r"[0-9a-f]{32}", str(task.get("attempt_id"))) is None
                or any(task.get(k) != scope[k] for k in ("commit", "pipeline", "backend", "precision",
                    "run", "preset_inputs_sha256", "comparison_time", "expected_comparisons"))
                or any(job.get(k) != scope[k] for k in ("commit", "pipeline", "backend", "precision", "preset_inputs_sha256"))
                or op.get("format") != "darsia-preset-wasserstein-operation-v1" or op.get("run") != scope["run"]
                or op.get("comparison_time") != scope["comparison_time"]):
            raise CachePolicyError("Original preset task/job/operation source and time identity differs")
        times = job.get("preset_comparison_times")
        time = scope["comparison_time"]
        if (not isinstance(times, list) or len(times) != scope["requested_time_count"]
                or type(time.get("order")) is not int or not 0 <= time["order"] < len(times)
                or times[time["order"]] != time or op.get("preset_inputs", {}).get("requested_times") != times
                or job.get("comparisons_per_time") != scope["expected_comparisons"]):
            raise CachePolicyError("Exact original requested times and pair/ROI count required")
        # Match the producer's canonical input digest without importing DarSIA.
        import hashlib, json
        digest = hashlib.sha256(json.dumps(op["preset_inputs"], sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode()).hexdigest()
        root = _path(self.request["job_root"])
        if (digest != scope["preset_inputs_sha256"] or task.get("job_id") != job.get("job_id")
                or _path(self.request["job"]["path"]) != root / "job.json"
                or _path(self.request["task"]["path"]) != root / "results" / task["run"] / "task.json"
                or self.request["configuration"]["sha256"] != scope["operation_sha256"]
                or task.get("configuration", {}).get("sha256") != scope["operation_sha256"]
                or self.request["configuration"]["size_bytes"] != task.get("configuration", {}).get("bytes")):
            raise CachePolicyError("Exact original preset configuration/task/job descriptors differ")

    def _once_key(self):
        return _digest(dict(preset=self.provider_binding, task=self.request["task"]))

    def consume(self, lease):
        # Identical existing namespace/cap/one-shot marker operations. Only the
        # source contract above and this class's exact task are admitted.
        from .setup_qualification import PositiveSetupServiceQualification
        if lease.context["host"].casefold() != self.provider["scope"]["physical_host"].casefold():
            raise CachePolicyError("Preset first request belongs to another physical machine")
        if lease.document.get("machine", {}).get("id") != self.provider["scope"]["target_machine"]:
            raise CachePolicyError("Preset first request belongs to another logical store machine")
        return PositiveSetupServiceQualification.consume(self, lease)

    def service_entitlement(self, lease):
        self._require_launch_admission(lease)
        report = lease.ledger.project(0)
        if self.maximum_service_bytes > administrative_entitlement(report):
            raise CachePolicyError("Fresh original cache headroom cannot cover the explicit preset service entitlement")
        self.administrative = dict(maximum_bytes=self.maximum_service_bytes, measured_profile=False,
            semantics="explicit original preset first-service entitlement within actual physical cap and reserves",
            request=self.binding, admission=report)
        return self.maximum_service_bytes

    def require_task(self, lease, task):
        if (getattr(lease, "first_qualification", None) is not self or task != self.task
                or lease.source != self.stores_source or self.claim is not None and self.claim != task
                or lease.context["host"].casefold() != self.provider["scope"]["physical_host"].casefold()
                or lease.document.get("machine", {}).get("id") != self.provider["scope"]["target_machine"]):
            raise CachePolicyError("Only the exact released preset task on its actual original machine may claim")
        return True

    def require_output(self, output):
        actual, watched = _path(str(output)), _path(self.output_entitlement["watched_output_scope"]["path"])
        if actual != watched and watched not in actual.parents:
            raise CachePolicyError("Original preset output leaves its independently released watched namespace")

    def remember_claim_locked(self, lease, claim):
        self.require_output(claim.output)
        return super().remember_claim_locked(lease, claim)


def _operation_proof(first, native_pin, *, read):
    """Retain actual execution/retirement evidence; do not mint a measured PASS."""
    native = read(_descriptor(native_pin))
    fixed_native, fixed_native_pin = _slot_read(first.provider, "native_completion")
    observation, observation_pin = _slot_read(first.provider, "observation")
    measured, measured_pin = _slot_read(first.provider, "measurement")
    eligibility = native.get("eligibility", {})
    if (native_pin != fixed_native_pin or native != fixed_native
            or native.get("format") != "darsia-native-worker-first-qualification-completion-v1"
            or native.get("complete") is not True or native.get("passed") is not True
            or native.get("observer_native_handles_closed") is not True
            or native.get("request") != first.binding or native.get("provider") != first.provider_binding
            or native.get("stores_source") != first.stores_source or native.get("scope") != first.provider["scope"]
            or native.get("startup") != first.provider["startup"]
            or native.get("module_closure") != first.provider["module_closure"]
            or native.get("observation") != observation_pin or observation.get("actual_claim") != first.task
            or observation.get("format") != "darsia-cache-first-qualification-observation-v1"
            or observation.get("request") != first.binding or observation.get("provider") != first.provider_binding
            or observation.get("stores_source") != first.stores_source or observation.get("scope") != first.provider["scope"]
            or observation.get("source_runtime_before_after_exact") is not True
            or observation.get("actual_public_worker_exit_code") != 0
            or observation.get("operational_reserve_crossed") is not False
            or eligibility.get("original_retirement_verifier_accepted") is not True
            or eligibility != observation.get("original_publication_and_retirement_eligibility")):
        raise CachePolicyError("Original preset native family, source closure and output/reserve observations did not close")
    retirement_pin = _descriptor(native.get("retirement"))
    retirement = read(retirement_pin)
    terminal = observation.get("original_kernel_terminal", {})
    if (retirement_pin != observation.get("retirement")
            or retirement.get("format") != "darsia-native-owned-worker-retirement-v1"
            or retirement.get("native_launch") != observation.get("native_launch")
            or retirement.get("service_receipt") != observation.get("service_receipt")
            or retirement.get("whole_family_kernel_observation") is not True
            or retirement.get("terminal") != terminal or terminal.get("wrapper_exit_code") != 0
            or terminal.get("accounting", {}).get("active_processes") != 0
            or terminal.get("accounting", {}).get("total_processes") != native.get("maximum_family_processes")):
        raise CachePolicyError("Original preset whole-family kernel retirement is required")
    primary = read(_descriptor(retirement.get("actual_primary_binding")))
    if (primary.get("actual_service_receipt") != observation["service_receipt"]
            or primary.get("original_native_launch") != observation["native_launch"]
            or primary.get("actual_primary") != retirement.get("primary")
            or primary.get("held_process_handle_observed_while_live") is not True
            or primary.get("same_original_job_membership_observed") is not True):
        raise CachePolicyError("The original preset native primary handle and Job membership are required")
    service = read(_descriptor(observation.get("service_receipt")))
    manifest = service.get("manifest", {})
    if (service.get("format") != "darsia-cache-import-service-receipt-v1"
            or service.get("policy") != first.stores_source or service.get("profile") != first.provider_binding
            or service.get("entry_kind") != "worker" or manifest.get("purpose") != "jobs"
            or service.get("namespace") != first.provider["namespace_admission"]
            or manifest.get("native_launch_context") != observation["native_launch"]
            or manifest.get("maximum_bytes") != first.maximum_service_bytes
            or str(manifest.get("owner", {}).get("machine", "")).casefold() != first.provider["scope"]["physical_host"].casefold()
            or service.get("first_qualification", {}).get("request") != first.binding
            or service.get("first_qualification", {}).get("provider") != first.provider_binding):
        raise CachePolicyError("Actual original preset service/stores/namespace/reservation differs")
    lifecycle = dict(reservation=manifest.get("reservation"), policy=first.stores_source,
        profile=first.provider_binding, namespace=first.provider["namespace_admission"], manifest=manifest)
    inventories = observation.get("actual_registered_owner_exit_inventories")
    if (not isinstance(inventories, list) or not inventories
            or any(row.get("all_file_backed_modules_predeclared_and_exact") is not True or row.get("undeclared")
                   for row in inventories)):
        raise CachePolicyError("All actual preset owners need their original exact source exit records")
    actual_tasks = []
    for row in inventories:
        registered = read(_descriptor(row.get("registration")))
        if (registered.get("format") != "darsia-cache-service-lifecycle-v1"
                or registered.get("kind") != "owner" or registered.get("binding") != lifecycle
                or registered.get("actor") != row.get("owner")
                or registered.get("value", {}).get("owner") != row.get("owner")
                or registered.get("value", {}).get("native_job_membership") != observation["native_launch"]):
            raise CachePolicyError("Original owner registration is outside the actual preset native family")
        exited = read(_descriptor(row.get("inventory")))
        modules = exited.get("first_qualification_modules", {})
        if (exited.get("owner") != row.get("owner") or modules.get("request") != first.binding
                or modules.get("provider") != first.provider_binding
                or modules.get("predeclared_module_closure_sha256") != _digest(first.provider["module_closure"])
                or modules.get("all_file_backed_modules_predeclared_and_exact") is not True or modules.get("undeclared")):
            raise CachePolicyError("Original preset owner exit source evidence differs")
        from .environment import FIRST_WRITER_ENVIRONMENT_KEYS, first_qualification_environment
        actual = modules.get("actual_writer_environment")
        if (not isinstance(actual, dict) or set(actual) != set(FIRST_WRITER_ENVIRONMENT_KEYS)
                or actual.get("NUMBA_CACHE_LOCATOR_CLASSES") != "UserProvidedCacheLocator"
                or modules.get("writer_environment_is_actual_IO_or_lifetime_proof") is not False):
            raise CachePolicyError("Every original preset owner needs all actual writer environment locations")
        try:
            expected = first_qualification_environment(actual, scope_root=manifest["root"])
        except (ValueError, TypeError, KeyError) as error:
            raise CachePolicyError("An actual preset writer environment escapes its original service") from error
        if expected != actual:
            raise CachePolicyError("Actual preset writer environment differs from its original service mapping")
        if modules.get("actual_task") is not None:
            actual_tasks.append(modules["actual_task"])
    if actual_tasks != [first.task]:
        raise CachePolicyError("Exactly one actual original preset worker claim must have completed")
    family = _positive(native.get("maximum_family_processes"), "Actual native family bound")
    if family != observation.get("conservative_family_process_bound_from_kernel_total"):
        raise CachePolicyError("The preset service family bound must retain its actual native lifetime counter")
    if (observation.get("output_measurement") != measured_pin
            or measured.get("format") != "darsia-preset-wasserstein-closed-output-diagnostic-v1"
            or measured.get("request") != first.binding or measured.get("provider") != first.provider_binding
            or measured.get("stores_source") != first.stores_source or measured.get("scope") != first.provider["scope"]
            or measured.get("output_plan") != first.request["output_entitlement_plan"]
            or measured.get("output_scope") != first.output_entitlement["watched_output_scope"]
            or measured.get("entitlement_bytes") != first.output_entitlement["maximum_output_bytes"] + first.output_entitlement["maximum_transient_bytes"]
            or measured.get("actual_retirement") != retirement_pin or measured.get("actual_terminal") != terminal
            or measured.get("original_retirement_accepted") is not True
            or measured.get("closed_output_stat_observation") is not True
            or measured.get("complete") is not True or measured.get("first_stop") is not None
            or measured.get("first_observation_error") is not None
            or measured.get("whole_output_entitlement_observed_exceeded") is not False
            or measured.get("cache_operational_reserve_crossed") is not False
            or measured.get("sampled_highwater_is_lifetime_peak") is not False
            or measured.get("output_hard_quota") is not False):
        raise CachePolicyError("Preset output diagnostic does not belong to this actual retired execution and explicit entitlement")
    seal = read(_descriptor(eligibility["seal"]))
    value = seal.get("value", {})
    if (seal.get("format") != "darsia-cache-service-lifecycle-v1" or seal.get("kind") != "seal"
            or seal.get("binding") != lifecycle or seal.get("actor") != manifest.get("owner")
            or value.get("outcome") != "success"):
        raise CachePolicyError("Original preset service did not seal successfully")
    publication = first._read_metadata(_descriptor(value["publication"]))
    rows = publication.get("completed", [])
    if (publication.get("format") != "darsia-native-worker-service-publication-v1"
            or publication.get("original_done_and_product_index_metadata_verified") is not True
            or publication.get("job") != first.request["job"] or publication.get("terminal_failures") != []
            or len(rows) != 1 or rows[0].get("run") != first.task["run"]
            or rows[0].get("attempt_id") != first.task["attempt_id"]):
        raise CachePolicyError("Exactly one actual original preset publication is required")
    result_pin = _descriptor(rows[0]["completed"])
    result = first._read_metadata(result_pin)
    if (result.get("format") != "darsia-preset-wasserstein-result-v1"
            or result.get("pipeline") != "preset-wasserstein" or result.get("complete") is not True
            or result.get("worker_references_closed") is not True or result.get("task") != first.task
            or result.get("attempt_id") != first.task["attempt_id"]
            or any(result.get(key) != first.task[key] for key in
                   ("commit", "run", "job_id", "backend", "precision", "preset_inputs_sha256"))
            or result.get("comparison_time") != first.task["comparison_time"]
            or result.get("comparison_count") != first.task["expected_comparisons"]):
        raise CachePolicyError("Closed original Comparison result receipt required")
    receipt = first._read_metadata(_descriptor(result["execution_receipt"]))
    receipt_fields = {"pipeline", "complete", "run", "job_id", "backend", "precision", "comparison_time",
        "preset_inputs_sha256", "expected_count", "comparison_count", "comparisons", "files", "output_count",
        "vtk", "worker_references_closed", "nonfinite_distances", "nonfinite_normalized_distances",
        "result_statuses", "validation"}
    if (set(receipt) != receipt_fields | {"format"}
            or receipt.get("format") != "darsia-preset-wasserstein-execution-receipt-v1"
            or any(receipt[key] != result.get(key) for key in receipt_fields)):
        raise CachePolicyError("Original finite execution receipt differs from its completed preset publication")
    return dict(format=CLOSED, request=first.binding,
        provider=first.provider_binding, scope=first.provider["scope"], native_completion=native_pin,
        observation=observation_pin, output_measurement=measured_pin, result=result_pin,
        execution_receipt=result["execution_receipt"], retirement=native["retirement"],
        complete=True, execution_closed=True, numerical_acceptance_claimed=False,
        sampled_highwater_is_lifetime_peak=False, actual_writer_IO_proven=False,
        no_ordinary_profile_created=True, cache_reservation_closed=False)


def close_operation(first, native_pin, *, read):
    return _slot_write(first.provider, "completion", _operation_proof(first, native_pin, read=read))


def _bounded_document(binding, root, stores_source, read, *, read_metadata=None):
    """Replay actual first retirement/publication; retain explicit administration."""
    closed, closed_pin = _slot_read(root, "completion")
    first = PresetQualification(_descriptor(closed.get("request")), provider_binding=binding,
        stores_source=stores_source, namespace=root["namespace_admission"], read=read,
        read_metadata=read_metadata)
    if closed != _operation_proof(first, closed["native_completion"], read=read):
        raise CachePolicyError("Original preset CLOSED evidence differs from its actual retired execution")
    native = read(closed["native_completion"])
    observation, observation_pin = _slot_read(root, "observation")
    measured, measured_pin = _slot_read(root, "measurement")
    service = read(_descriptor(observation["service_receipt"]))
    if (service.get("policy") != stores_source or service.get("profile") != binding
            or service.get("namespace") != root["namespace_admission"]
            or service.get("first_qualification", {}).get("request") != first.binding
            or service.get("manifest", {}).get("owner", {}).get("machine", "").casefold()
                != root["scope"]["physical_host"].casefold()
            or service.get("manifest", {}).get("maximum_bytes") != first.maximum_service_bytes):
        raise CachePolicyError("Retired preset service host/namespace/source/entitlement differs")
    return dict(format=BOUNDED_PROFILE, purpose="jobs", entry_kind="worker",
        role_identifier=service["role_identifier"], provider=deepcopy(binding), stores_source=deepcopy(stores_source),
        scope=deepcopy(root["scope"]), namespace_admission=deepcopy(root["namespace_admission"]),
        python_executable=deepcopy(root["python_executable"]), module_closure=deepcopy(root["module_closure"]),
        producer_source_closure=deepcopy(root["producer_source_closure"]),
        runtime_source_closure=deepcopy(root["runtime_source_closure"]),
        capacity_bound=deepcopy(first.required_capacity_bound), maximum_cache_bytes=first.maximum_service_bytes,
        maximum_output_bytes=first.output_entitlement["maximum_output_bytes"],
        maximum_transient_bytes=first.output_entitlement["maximum_transient_bytes"],
        additional_claim_cache_bytes=0, allocation_model="explicit-shared-service-administration-no-claim-scope-v1",
        supported_workers=1, actual_writer_IO_proven=False, sampled_highwater_is_lifetime_peak=False,
        not_a_measured_profile=True, numerical_acceptance_claimed=False,
        writer_contract="physical-volume-bound-original-preset-v1", first_closed=closed_pin,
        measurement=measured_pin, observation=observation_pin, first_request=first.binding,
        output_administration=first.request["output_entitlement_plan"],
        closed_output_logical_bytes=measured["closed_output_logical_bytes"],
        sampled_output_highwater_bytes=measured["sampled_highwater_bytes"],
        sampled_cache_highwater_bytes=observation["sampled_cache_logical_high_water_bytes"],
        qualified_watchdog="--watchdog" in first.request["public_command"],
        native_supervision=dict(format="darsia-native-worker-supervision-profile-v1",
            startup=deepcopy(root["startup"]), completion=deepcopy(closed["native_completion"]),
            maximum_family_processes=native["maximum_family_processes"],
            publication_contract="alias-done-small-metadata-v1"))


def _ordinary_release(release, binding, root, stores_source, profile):
    if (release.get("format") != "darsia-preset-wasserstein-bounded-release-v1"
            or release.get("execution_authorized") is not True or release.get("provider") != binding
            or release.get("stores_source") != stores_source or release.get("scope") != root["scope"]
            or release.get("first_closed") != profile["first_closed"]
            or release.get("maximum_service_bytes") != profile["maximum_cache_bytes"]
            or release.get("maximum_output_bytes") != profile["maximum_output_bytes"]
            or release.get("maximum_transient_bytes") != profile["maximum_transient_bytes"]):
        raise CachePolicyError("Independent exact ordinary bounded preset administration release required")


def publish_bounded_profile(binding, *, stores_source, ordinary_release, read, read_metadata=None):
    """Publish a distinct bounded administrative route after actual first close.

    This accepts no replacement observations, caps or success flags. The
    independent ordinary release names the actual CLOSED record and the same
    explicit caps. The fixed-slot profile is never a lifetime/IO measurement.
    """
    binding = _descriptor(binding)
    root = checked_provider(read(binding))
    profile = _bounded_document(binding, root, stores_source, read, read_metadata=read_metadata)
    _ordinary_release(read(_descriptor(ordinary_release)), binding, root, stores_source, profile)
    pin = _slot_write(root, "bounded_profile", profile)
    return _slot_write(root, "bounded_completion", dict(format=BOUNDED_COMPLETION,
        provider=binding, stores_source=deepcopy(stores_source), scope=deepcopy(root["scope"]),
        first_closed=profile["first_closed"], bounded_profile=pin, ordinary_release=ordinary_release,
        complete=True, passed=True, semantics="administrative bounded admission only; no numerical acceptance",
        not_a_measured_profile=True, actual_writer_IO_proven=False, sampled_highwater_is_lifetime_peak=False))


def resolve_bounded_profile(binding, root, kind, read, *, stores_source, read_metadata=None):
    if kind not in {"import", "jobs"} or stores_source is None:
        raise CachePolicyError("Exact original preset worker/jobs service context required")
    root = checked_provider(root)
    bundle, completion_pin = _slot_read(root, "bounded_completion")
    profile = _bounded_document(binding, root, stores_source, read, read_metadata=read_metadata)
    _ordinary_release(read(_descriptor(bundle.get("ordinary_release"))), binding, root, stores_source, profile)
    actual, pin = _slot_read(root, "bounded_profile")
    expected = dict(format=BOUNDED_COMPLETION, provider=binding, stores_source=deepcopy(stores_source),
        scope=deepcopy(root["scope"]), first_closed=profile["first_closed"], bounded_profile=pin,
        ordinary_release=bundle["ordinary_release"], complete=True, passed=True,
        semantics="administrative bounded admission only; no numerical acceptance", not_a_measured_profile=True,
        actual_writer_IO_proven=False, sampled_highwater_is_lifetime_peak=False)
    if actual != profile or bundle != expected:
        raise CachePolicyError("Preset bounded profile is not the exact actual retired first-execution DAG")
    return dict(actual, completion=completion_pin)


def checked_import_context(profile, *, purpose, role_identifier, entry_kind, policy, namespace, machine_id):
    import socket
    if (profile.get("format") != BOUNDED_PROFILE or purpose != "jobs" or entry_kind != "worker"
            or profile.get("purpose") != purpose or profile.get("entry_kind") != entry_kind
            or profile.get("role_identifier") != role_identifier or profile.get("namespace_admission") != namespace
            or profile.get("scope", {}).get("pipeline") != "preset-wasserstein"
            or profile.get("scope", {}).get("backend") != "cpu" or profile.get("scope", {}).get("precision") != "original"
            or profile.get("scope", {}).get("target_machine") != machine_id
            or str(profile.get("scope", {}).get("physical_host", "")).casefold() != socket.gethostname().casefold()
            or profile.get("supported_workers") != 1 or policy.profiles.get("jobs") != profile.get("provider")
            or policy.volume_cap is None or policy.volume_cap.maximum_size_bytes != policy.quota_bytes
            or profile["capacity_bound"]["volume_cap_observation"]["binding"] != policy.volume_cap.document()
            or profile["capacity_bound"]["volume_cap_observation"].get("virtual_size_bytes") != policy.volume_cap.maximum_size_bytes
            or profile.get("actual_writer_IO_proven") is not False or profile.get("sampled_highwater_is_lifetime_peak") is not False
            or profile.get("not_a_measured_profile") is not True or profile.get("numerical_acceptance_claimed") is not False
            or profile.get("additional_claim_cache_bytes") != 0
            or profile.get("allocation_model") != "explicit-shared-service-administration-no-claim-scope-v1"
            or profile.get("writer_contract") != "physical-volume-bound-original-preset-v1"):
        raise CachePolicyError("Preset bounded import/service context differs from its original exact physical scope")
    return _positive(profile["maximum_cache_bytes"], "Explicit qualified preset service entitlement")


def require_public_command(profile, command):
    # This source-scope one-worker admission preserves the public executable,
    # stores, pipeline and machine checks; ordinary work may choose a new job.
    from .setup_qualification import _command
    if (not isinstance(command, list) or any(not isinstance(arg, str) for arg in command)
            or ("--watchdog" in command) is not profile.get("qualified_watchdog")):
        raise CachePolicyError("Ordinary preset command changes its qualified native supervision shape")
    jobs = [i for i, item in enumerate(command) if item == "--job"]
    stores = [i for i, item in enumerate(command) if item == "--stores"]
    if len(jobs) != 1 or jobs[0] + 1 >= len(command) or len(stores) > 1:
        raise CachePolicyError("One exact public preset job and at most one stores argument required")
    # The remote launcher intentionally uses its already admitted default map.
    # Validate that actual lease source; do not add a local path to remote argv.
    validated = list(command)
    if not stores:
        validated.extend(["--stores", profile["stores_source"]["path"]])
    request = dict(job_root=command[jobs[0] + 1], stores_source=profile["stores_source"])
    _command(validated, request, profile)
