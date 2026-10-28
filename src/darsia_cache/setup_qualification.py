"""Positive first-service admission for the original Apply and Build queue.

Only existing Setup controls and actual native service completion are consumed.
An import profile describes the bounded service, never a measured analysis job.
No application or numerical package is imported before the original reservation.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import PureWindowsPath
import re

from .first_qualification import (
    FirstQualification, _digest, _exact_keys, _pins, _slot_read, _slot_write,
    administrative_entitlement, checked_required_capacity_bound,
)
from .policy import (CachePolicyError, PHYSICAL_RESERVE_BYTES, _descriptor,
                     setup_control_descriptor, SETUP_CONTROL_MAX_BYTES)

PROVIDER = "darsia-setup-service-provider-v1"
REQUEST = "darsia-setup-service-request-v1"
ROOT_RELEASE = "darsia-setup-service-release-v1"
CLOSED = "darsia-setup-service-closed-operation-v1"
BUNDLE = "darsia-setup-service-completion-v1"
IMPORT_PROFILE = "darsia-setup-service-import-profile-v1"
OPERATIONS = ("apply", "build")
SUPPORTED_OPERATIONS = (*OPERATIONS, "image-preparation")
OPERATION_SLOTS = {"observation", "native_completion", "completion"}
SCOPE_FIELDS = {"commit", "pipeline", "backend", "precision", "target_machine", "job"}
REQUIRED_SOURCE_SUFFIXES = {
    "darsia_cache/setup_qualification.py", "darsia_cache/bootstrap.py",
    "darsia_cache/owned_process.py", "darsia_cache/worker_adapter.py",
    "darsia_cache/service_context.py", "darsia_cache/service_lifecycle.py",
    "darsia/fluidflower/pipeline/worker.py",
    "darsia/fluidflower/pipeline/series_setup_worker_admission.py",
    "darsia/fluidflower/pipeline/series_setup_operations.py",
    "darsia/fluidflower/pipeline/series_setup_results.py",
    "darsia/fluidflower/pipeline/series_setup_sink.py",
}


def _positive(value, where):
    if type(value) is not int or value <= 0:
        raise CachePolicyError(where + " must be a literal positive integer")
    return value


def _path(value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CachePolicyError("An explicit absolute Setup locator is required")
    path = PureWindowsPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise CachePolicyError("Setup locators cannot be relative or traverse parents")
    return path


def _original_metadata(binding):
    """Use the original bounded worker reader for local or shared job metadata."""
    from .adapters import _worker_metadata
    binding = _descriptor(binding)
    return _worker_metadata(binding["path"], expected=binding)[0]


def setup_control_metadata(binding):
    """Keep the original exact reader, with the existing Setup catalogue ceiling."""
    from .adapters import _worker_metadata
    binding = setup_control_descriptor(binding)
    return _worker_metadata(binding["path"], expected=binding, maximum_bytes=SETUP_CONTROL_MAX_BYTES)[0]


def checked_provider(value):
    """Keep Setup separate from every existing analysis provider and its scope."""
    _exact_keys(value, {"format", "qualification_id", "scope", "namespace_admission",
        "output_namespace", "slots", "operations", "python_executable", "startup",
        "module_closure", "producer_source_closure", "runtime_source_closure"},
        "Exact Setup service provider required")
    scope = _exact_keys(value["scope"], SCOPE_FIELDS, "Exact Setup service source scope required")
    if (value["format"] != PROVIDER
            or re.fullmatch(r"[0-9a-f]{32}", str(value["qualification_id"])) is None
            or re.fullmatch(r"[0-9a-f]{40}", str(scope["commit"])) is None
            or scope["pipeline"] != "series-setup" or scope["backend"] != "cpu"
            or scope["precision"] not in {"float32", "float64"}
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", str(scope["target_machine"])) is None):
        raise CachePolicyError("Only the exact declared CPU Setup service is admitted")
    for field in ("namespace_admission", "python_executable", "startup"):
        _descriptor(value[field])
    _descriptor(scope["job"])
    output = _exact_keys(value["output_namespace"], {"path", "identity"},
                         "Original accepted Setup receipt namespace required")
    _path(output["path"])
    if not isinstance(output["identity"], dict) or not output["identity"]:
        raise CachePolicyError("The receipt namespace needs its actual accepted identity")
    _exact_keys(value["slots"], {"completion", "import_profile"}, "Only Setup import completion slots are supported")
    actual_operations = tuple(value["operations"])
    if set(actual_operations) not in (set(OPERATIONS), set(SUPPORTED_OPERATIONS)):
        raise CachePolicyError("Original Setup operation routes and optional image preparation are required")
    names = list(value["slots"].values())
    for operation in actual_operations:
        _exact_keys(value["operations"][operation], OPERATION_SLOTS, "Fixed Setup operation slots required")
        names.extend(value["operations"][operation].values())
    if (any(not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,180}\.json", name) is None for name in names)
            or len(names) != len({name.casefold() for name in names})):
        raise CachePolicyError("Setup slots must be distinct flat immutable JSON names")
    union = _pins(value["producer_source_closure"]) + _pins(value["runtime_source_closure"])
    closure = _pins(value["module_closure"])
    if len(union) != len(closure) or {_digest(pin) for pin in union} != {_digest(pin) for pin in closure}:
        raise CachePolicyError("The exact producer/runtime union must equal the original startup closure")
    paths = {str(pin["path"]).replace("\\", "/").casefold() for pin in union}
    if any(not any(path.endswith("/" + suffix) for path in paths) for suffix in REQUIRED_SOURCE_SUFFIXES):
        raise CachePolicyError("Setup closure omits an original worker, admission or publication boundary")
    return deepcopy(value)


def operation_provider(root, operation):
    """Select existing fixed operation slots without future tasks or file hashes."""
    if operation not in SUPPORTED_OPERATIONS:
        raise CachePolicyError("One literal Apply or Build operation is required")
    result = deepcopy(root)
    result["slots"] = deepcopy(root["operations"][operation])
    result["scope"] = dict(deepcopy(root["scope"]), operation=operation)
    return result


def _control_identity(task, job, config, scope, operation):
    """Check preimport metadata; the original worker still validates its catalogue."""
    if (not isinstance(task, dict) or not isinstance(job, dict) or not isinstance(config, dict)
            or task.get("format") != "darsia-alias-task-v1"
            or job.get("format") != "darsia-alias-job-v1"
            or task.get("format_version") != 1 or task.get("legacy_paths") is not False
            or task.get("operation") != operation or config.get("operation") != operation
            or config.get("format") != "darsia-series-setup-operation-v1"
            or task.get("ordered_case_ids") != [operation]
            or task.get("resume_from") is not None
            or re.fullmatch(r"[0-9a-f]{32}", str(task.get("attempt_id"))) is None):
        raise CachePolicyError("One original queued Setup operation is required; no analysis prefix is admitted")
    for field in ("commit", "pipeline", "backend", "precision"):
        if task.get(field) != scope[field] or job.get(field) != scope[field]:
            raise CachePolicyError("Setup job/task source or compute identity differs")
    runs, operations = job.get("expected_runs"), job.get("setup_operations")
    if (not isinstance(runs, list) or len(runs) < 2 or len({str(run).casefold() for run in runs}) != len(runs)
            or not isinstance(operations, list)
            or operations != [dict(run=run, operation="apply" if index == 0 else
                              "image-preparation" if run == "setup-image-preparation" and index == len(runs)-1 else "build")
                              for index, run in enumerate(runs)]
            or task.get("run") not in runs or config.get("run") != task["run"]
            or next(row["operation"] for row in operations if row["run"] == task["run"]) != operation
            or task.get("target_machine") != scope["target_machine"]
            or config.get("target_machine") != scope["target_machine"]
            or task.get("setup_plan") != job.get("setup_plan")):
        raise CachePolicyError("Setup must use its original ordered job, target and immutable plan")
    catalog = task.get("input_catalog", {})
    if (not isinstance(catalog, dict) or catalog.get("format") != "darsia-role-input-catalog-v1"
            or catalog.get("complete") is not True or catalog.get("run") != task["run"]
            or catalog.get("configuration") != task.get("configuration")
            or not isinstance(catalog.get("inputs"), list)
            or sum(row.get("reference") == task["setup_plan"] for row in catalog["inputs"]
                   if isinstance(row, dict)) != 1):
        raise CachePolicyError("The original Setup catalogue must bind its actual controls")
    plan = setup_control_descriptor(config.get("setup_plan"))
    if plan["sha256"] != task["setup_plan"].get("sha256") or plan["size_bytes"] != task["setup_plan"].get("bytes"):
        raise CachePolicyError("Setup configuration names another plan")
    caps = _exact_keys(config.get("metadata_output"), {"maximum_output_bytes", "maximum_file_bytes"},
                       "Original whole and per-file Setup caps required")
    if _positive(caps["maximum_file_bytes"], "Metadata file cap") > _positive(caps["maximum_output_bytes"], "Whole metadata cap"):
        raise CachePolicyError("The metadata file cap exceeds its whole operation cap")
    outputs = _exact_keys(config.get("output_scope"), {"metadata_roots", "scratch_roots"},
                          "Original Setup output scopes required")
    for role, roots in outputs.items():
        if not isinstance(roots, list) or role == "metadata_roots" and not roots:
            raise CachePolicyError("Original explicit metadata and scratch roots required")
        if len({_path(root) for root in roots}) != len(roots):
            raise CachePolicyError("Setup output roots cannot be duplicated")
    if operation == "build":
        for field in ("raw_build_request", "apply_completion", "coordinator_completion"):
            setup_control_descriptor(config.get(field))
    return True


def _command(command, request, provider):
    """Keep the genuine public one-slot, one-claim worker command."""
    if (not isinstance(command, list) or not command or any(not isinstance(arg, str) for arg in command)
            or _path(command[0]) != _path(provider["python_executable"]["path"])
            or command.count("-m") != 1 or "-B" not in command):
        raise CachePolicyError("The original selected interpreter and public worker with -B are required")
    index = command.index("-m")
    if command[index + 1:index + 2] != ["darsia.fluidflower.worker"]:
        raise CachePolicyError("Setup cannot substitute another executor")
    flags = command[1:index]
    while flags:
        flag, flags = flags[0], flags[1:]
        if flag == "-X" and flags[:1] == ["utf8"]:
            flags = flags[1:]
        elif flag not in {"-B", "-u", "-Xutf8"}:
            raise CachePolicyError("Only the original bounded public interpreter flags are supported")
    args, switches = command[index + 2:], {}
    while args:
        option, args = args[0], args[1:]
        if option in switches or option not in {"--job", "--backend", "--workers", "--machine", "--stores", "--once", "--watchdog"}:
            raise CachePolicyError("Only the exact finite public Setup worker arguments are supported")
        if option in {"--once", "--watchdog"}:
            switches[option] = True
        else:
            if not args:
                raise CachePolicyError("A public Setup worker argument has no value")
            switches[option], args = args[0], args[1:]
    if (set(switches) - {"--watchdog"} != {"--job", "--backend", "--workers", "--machine", "--stores", "--once"}
            or switches["--backend"] != "cpu" or switches["--workers"] != "1"
            or switches["--machine"] != provider["scope"]["target_machine"]
            or _path(switches["--job"]) != _path(request["job_root"])
            or _path(switches["--stores"]) != _path(request["stores_source"]["path"])):
        raise CachePolicyError("Setup qualification requires its exact job/stores/target and one --once worker")


class PositiveSetupServiceQualification(FirstQualification):
    """Admit one real Apply or Build through the existing native queue owner.

    The task descriptor binds original results/<run>/task.json. Pending queue
    locators move on claim and cannot be used as stable qualification inputs.
    Shared job controls use the original worker metadata reader, independently
    from the local authority reader used by preimport provider/release records.
    """
    setup_mode = True

    def __init__(self, binding, *, provider_binding, stores_source, namespace, read,
                 read_authorities=None, historical_authorities=None, read_metadata=None):
        if historical_authorities is not None:
            raise CachePolicyError("Historical analysis authorities cannot authorize a Setup service")
        self.binding, self.provider_binding = _descriptor(binding), _descriptor(provider_binding)
        self.request = read(self.binding)
        _exact_keys(self.request, {"format", "operation", "execution_authorized", "root_release", "provider",
            "stores_source", "namespace_admission", "public_command", "task", "job", "job_root",
            "configuration", "capacity_bound_plan", "maximum_service_bytes",
            "cache_stat_period_seconds", "predecessor"}, "Exact released Setup service request required")
        self.operation = self.request["operation"]
        self.root_provider = checked_provider(read(self.provider_binding))
        self.provider = operation_provider(self.root_provider, self.operation)
        self.stores_source, self._read = deepcopy(stores_source), read
        self._read_metadata = read_metadata or _original_metadata
        if (self.request["format"] != REQUEST or self.request["execution_authorized"] is not True
                or self.request["provider"] != self.provider_binding or self.request["stores_source"] != stores_source
                or self.request["namespace_admission"] != namespace or namespace != self.provider["namespace_admission"]
                or self.request["job"] != self.root_provider["scope"]["job"]
                or type(self.request["cache_stat_period_seconds"]) not in {int, float}
                or not 0 < self.request["cache_stat_period_seconds"] <= 30):
            raise CachePolicyError("Setup service release/source/stores/namespace differs")
        self.maximum_service_bytes = _positive(self.request["maximum_service_bytes"], "Setup service entitlement")
        release = read(_descriptor(self.request["root_release"]))
        if (release.get("format") != ROOT_RELEASE or release.get("execution_authorized") is not True
                or release.get("provider") != self.provider_binding or release.get("stores_source") != stores_source
                or release.get("scope") != self.root_provider["scope"]
                or release.get("request_payload_sha256") != _digest({key: value for key, value in self.request.items() if key != "root_release"})):
            raise CachePolicyError("An independent exact Setup request release is required")
        self.task = self._read_metadata(_descriptor(self.request["task"]))
        self.job = self._read_metadata(_descriptor(self.request["job"]))
        self.configuration = self._read_metadata(_descriptor(self.request["configuration"]))
        _control_identity(self.task, self.job, self.configuration, self.root_provider["scope"], self.operation)
        root = _path(self.request["job_root"])
        config_ref = self.task["configuration"]
        if (_path(self.request["job"]["path"]) != root / "job.json" or self.task.get("job_id") != root.name
                or _path(self.request["task"]["path"]) != root / "results" / self.task["run"] / "task.json"
                or self.request["configuration"]["sha256"] != config_ref.get("sha256")
                or self.request["configuration"]["size_bytes"] != config_ref.get("bytes")):
            raise CachePolicyError("Setup task and exact original job/configuration files differ")
        _command(self.request["public_command"], self.request, self.root_provider)
        self.required_capacity_bound = checked_required_capacity_bound(read(_descriptor(self.request["capacity_bound_plan"])))
        if self.required_capacity_bound["enforcement_state"] != "PREDECLARED_Q106_VOLUME_BOUND":
            raise CachePolicyError("A held capacity plan cannot authorize a Setup service")
        self.output_entitlement = dict(maximum_service_bytes=self.maximum_service_bytes,
            metadata_output=deepcopy(self.configuration["metadata_output"]),
            output_scope=deepcopy(self.configuration["output_scope"]))
        self.bound_mode, self.reference_build = False, False
        self.reference_administration = self.reference_retained_inventory = self.q123_cold_scope = None
        self.reference_roots_validated, self.reference_transport_observation = False, None
        self.started, self.claim = False, None
        if self.operation == "apply":
            if self.request["predecessor"] is not None:
                raise CachePolicyError("Apply cannot claim a future or unrelated predecessor")
        else:
            previous, pin = _slot_read(operation_provider(self.root_provider, "apply"), "completion")
            if self.request["predecessor"] != pin:
                raise CachePolicyError("Build must name the actual closed Apply operation")
            checked_operation(previous, provider_binding=self.provider_binding, root=self.root_provider,
                              stores_source=self.stores_source, operation="apply", read=read,
                              read_metadata=self._read_metadata)
            applied = self._read_metadata(previous["result"])
            children = [row for row in applied.get("children", ()) if row.get("run_id") == self.task["run"]]
            if (len(children) != 1 or children[0].get("configuration") != self.request["configuration"]
                    or children[0].get("raw_build_request") != self.configuration["raw_build_request"]
                    or applied.get("coordinator_completion") != self.configuration["coordinator_completion"]):
                raise CachePolicyError("Build is not the actual post-Apply child and captured RAW request")

    def _once_key(self):
        return _digest(dict(setup=self.provider_binding, operation=self.operation))

    def consume(self, lease):
        """Reuse original locked launch admission and exact physical-cap accounting."""
        from .control import _encoded
        from .windows import WindowsNamespace
        target = self.provider["output_namespace"]
        namespace = WindowsNamespace(target["path"], target["identity"])
        try:
            if {name.casefold() for name in namespace.names()} & {name.casefold() for name in self.provider["slots"].values()}:
                raise CachePolicyError("Setup evidence already exists; no automatic retry is admitted")
        finally:
            namespace.close()
        self._namespace(lease)
        with lease.ledger.locked():
            report = lease.ledger.project(0)
            cap = lease.policy.volume_cap
            expected = self.required_capacity_bound["volume_cap_observation"]
            actual = report.get("volume_cap_observation", {})
            if (report.get("hard_filesystem_quota_verified") is not True
                    or cap is None or cap.maximum_size_bytes != lease.policy.quota_bytes
                    or expected.get("binding") != cap.document()
                    or any(actual.get(key) != expected.get(key) for key in (
                        "binding", "virtual_size_bytes", "volume_total_bytes", "filesystem", "is_loaded", "verified"))
                    or report.get("reserve_bytes") != PHYSICAL_RESERVE_BYTES
                    or self.maximum_service_bytes > administrative_entitlement(report)):
                raise CachePolicyError("Original Setup service cap, namespace or fresh physical reserve differs")
            names, key = lease.ledger.control.names(), self._once_key()
            if "cache-first-" + key + ".json" in names or "cache-first-" + key + "-claim.json" in names:
                raise CachePolicyError("This original Setup operation was already consumed")
            lease.ledger.control.write_once(self._launch_marker_name(), _encoded(dict(
                format="darsia-cache-first-qualification-launch-admitted-v1", request=self.binding,
                provider=self.provider_binding, stores_source=self.stores_source,
                owner=lease.owner, scope=self.provider["scope"], no_retry=True)))
        self.started = True

    def service_entitlement(self, lease):
        self._require_launch_admission(lease)
        report = lease.ledger.project(0)
        if self.maximum_service_bytes > administrative_entitlement(report):
            raise CachePolicyError("Fresh service headroom cannot cover the released Setup entitlement")
        self.administrative = dict(maximum_bytes=self.maximum_service_bytes, measured_profile=False,
            semantics="released Setup service bytes within the actual physical cap and original reserves",
            request=self.binding, admission=report)
        return self.maximum_service_bytes

    def require_task(self, lease, task):
        if (getattr(lease, "first_qualification", None) is not self or task != self.task
                or lease.source != self.stores_source or self.claim is not None and self.claim != task):
            raise CachePolicyError("Only this exact released Setup task and original service may claim")
        return True


def _completed_result(first, observation, native, read, read_metadata):
    """Read the original sealed publication, never manufacture a Setup success."""
    eligibility = native.get("eligibility", {})
    if (eligibility.get("original_retirement_verifier_accepted") is not True
            or eligibility != observation.get("original_publication_and_retirement_eligibility")
            or native.get("retirement") != observation.get("retirement")):
        raise CachePolicyError("Original native family retirement and publication verification are required")
    seal = read(_descriptor(eligibility.get("seal")))
    value = seal.get("value", {})
    if value.get("outcome") != "success":
        raise CachePolicyError("The actual original service did not seal successfully")
    publication = read_metadata(_descriptor(value.get("publication")))
    completed = publication.get("completed", ())
    if (publication.get("format") != "darsia-native-worker-service-publication-v1"
            or publication.get("job") != first.request["job"] or publication.get("terminal_failures") != []
            or publication.get("original_done_and_product_index_metadata_verified") is not True
            or len(completed) != 1 or completed[0].get("run") != first.task["run"]
            or completed[0].get("attempt_id") != first.task["attempt_id"]):
        raise CachePolicyError("One actual original Setup publication is required")
    result_pin = _descriptor(completed[0].get("completed"))
    result = read_metadata(result_pin)
    if (result.get("format") != "darsia-series-setup-result-v1" or result.get("pipeline") != "series-setup"
            or result.get("complete") is not True or result.get("worker_references_closed") is not True
            or result.get("task") != first.task or result.get("attempt_id") != first.task["attempt_id"]
            or result.get("commit") != first.task["commit"]
            or any(result.get(key) != first.task[key] for key in ("run", "operation", "backend", "precision"))
            or result.get("setup_plan") != first.configuration["setup_plan"]
            or set(result.get("files", {})) != {"operation_result.json"}):
        raise CachePolicyError("Actual closed Setup result differs from its original task")
    receipt = read_metadata(_descriptor(result["files"]["operation_result.json"]))
    if "files" in receipt or any(result.get(key) != value for key, value in receipt.items()) or not receipt:
        raise CachePolicyError("Original Setup operation receipt differs from its completed publication")
    return result_pin


def _operation_proof(first, native_pin, read, read_metadata=None):
    read_metadata = read_metadata or first._read_metadata
    native = read(_descriptor(native_pin))
    observation, observation_pin = _slot_read(first.provider, "observation")
    if (native.get("format") != "darsia-native-worker-first-qualification-completion-v1"
            or native.get("complete") is not True or native.get("passed") is not True
            or native.get("observer_native_handles_closed") is not True
            or native.get("request") != first.binding or native.get("provider") != first.provider_binding
            or native.get("stores_source") != first.stores_source or native.get("scope") != first.provider["scope"]
            or native.get("startup") != first.provider["startup"]
            or native.get("module_closure") != first.provider["module_closure"]
            or native.get("observation") != observation_pin
            or observation.get("actual_claim") != first.task
            or observation.get("source_runtime_before_after_exact") is not True
            or observation.get("actual_public_worker_exit_code") != 0
            or observation.get("operational_reserve_crossed") is not False):
        raise CachePolicyError("Actual source-bound Setup native completion is required")
    retirement = read(_descriptor(native.get("retirement")))
    terminal = observation.get("original_kernel_terminal", {})
    if (retirement.get("format") != "darsia-native-owned-worker-retirement-v1"
            or retirement.get("native_launch") != observation.get("native_launch")
            or retirement.get("service_receipt") != observation.get("service_receipt")
            or retirement.get("whole_family_kernel_observation") is not True
            or retirement.get("terminal") != terminal or terminal.get("wrapper_exit_code") != 0
            or terminal.get("accounting", {}).get("active_processes") != 0
            or terminal.get("accounting", {}).get("total_processes") != native.get("maximum_family_processes")):
        raise CachePolicyError("Original whole-family kernel retirement is required")
    primary = read(_descriptor(retirement.get("actual_primary_binding")))
    if (primary.get("actual_service_receipt") != observation["service_receipt"]
            or primary.get("original_native_launch") != observation["native_launch"]
            or primary.get("actual_primary") != retirement.get("primary")
            or primary.get("held_process_handle_observed_while_live") is not True
            or primary.get("same_original_job_membership_observed") is not True):
        raise CachePolicyError("The original native primary handle and Job membership are required")
    service = read(_descriptor(observation.get("service_receipt")))
    manifest = service.get("manifest", {})
    if (service.get("format") != "darsia-cache-import-service-receipt-v1"
            or service.get("policy") != first.stores_source or service.get("profile") != first.provider_binding
            or service.get("entry_kind") != "worker" or manifest.get("purpose") != "jobs"
            or service.get("namespace") != first.provider["namespace_admission"]
            or manifest.get("native_launch_context") != observation["native_launch"]
            or manifest.get("maximum_bytes") != first.maximum_service_bytes
            or service.get("first_qualification", {}).get("request") != first.binding
            or service.get("first_qualification", {}).get("provider") != first.provider_binding):
        raise CachePolicyError("Actual original Setup service/stores/namespace/reservation differs")
    lifecycle = dict(reservation=manifest.get("reservation"), policy=first.stores_source,
        profile=first.provider_binding, namespace=first.provider["namespace_admission"], manifest=manifest)
    inventories = observation.get("actual_registered_owner_exit_inventories")
    if (not isinstance(inventories, list) or not inventories
            or any(row.get("all_file_backed_modules_predeclared_and_exact") is not True or row.get("undeclared")
                   for row in inventories)):
        raise CachePolicyError("All actual registered owners need their original exact source exit records")
    actual_tasks = []
    for row in inventories:
        registered = read(_descriptor(row.get("registration")))
        if (registered.get("format") != "darsia-cache-service-lifecycle-v1"
                or registered.get("kind") != "owner" or registered.get("binding") != lifecycle
                or registered.get("actor") != row.get("owner")
                or registered.get("value", {}).get("owner") != row.get("owner")
                or registered.get("value", {}).get("native_job_membership") != observation["native_launch"]):
            raise CachePolicyError("Original owner registration is outside the actual Setup native family")
        exited = read(_descriptor(row.get("inventory")))
        modules = exited.get("first_qualification_modules", {})
        if (exited.get("owner") != row.get("owner") or modules.get("request") != first.binding
                or modules.get("provider") != first.provider_binding
                or modules.get("predeclared_module_closure_sha256") != _digest(first.provider["module_closure"])
                or modules.get("all_file_backed_modules_predeclared_and_exact") is not True or modules.get("undeclared")):
            raise CachePolicyError("Original registered-owner exit source evidence differs")
        from .environment import FIRST_WRITER_ENVIRONMENT_KEYS, first_qualification_environment
        actual = modules.get("actual_writer_environment")
        if (not isinstance(actual, dict) or set(actual) != set(FIRST_WRITER_ENVIRONMENT_KEYS)
                or actual.get("NUMBA_CACHE_LOCATOR_CLASSES") != "UserProvidedCacheLocator"
                or modules.get("writer_environment_is_actual_IO_or_lifetime_proof") is not False):
            raise CachePolicyError("Every original Setup owner needs all actual writer environment locations")
        try:
            expected = first_qualification_environment(actual, scope_root=manifest["root"])
        except (ValueError, TypeError, KeyError) as error:
            raise CachePolicyError("An actual Setup writer environment escapes its original service") from error
        if expected != actual:
            raise CachePolicyError("Actual Setup writer environment differs from the original first-service mapping")
        if modules.get("actual_task") is not None:
            actual_tasks.append(modules["actual_task"])
    if actual_tasks != [first.task]:
        raise CachePolicyError("Exactly one actual original Setup worker claim must have completed")
    family = _positive(native.get("maximum_family_processes"), "Actual native family bound")
    if family != observation.get("conservative_family_process_bound_from_kernel_total"):
        raise CachePolicyError("The service family bound must retain its actual native lifetime counter")
    result = _completed_result(first, observation, native, read, read_metadata)
    return dict(format=CLOSED, provider=deepcopy(first.provider_binding), stores_source=deepcopy(first.stores_source),
        scope=deepcopy(first.root_provider["scope"]), operation=first.operation, request=deepcopy(first.binding),
        task=deepcopy(first.request["task"]), result=result, observation=observation_pin,
        native_completion=deepcopy(native_pin), capacity_bound=deepcopy(first.required_capacity_bound),
        maximum_service_bytes=first.maximum_service_bytes, maximum_family_processes=family,
        complete=True, passed=True, sampled_highwater_is_lifetime_peak=False, actual_writer_IO_proven=False)


def close_operation(first, native_completion_binding, *, read, read_metadata=None):
    """Close only fixed genuine native evidence after the original supervisor retires."""
    if not isinstance(first, PositiveSetupServiceQualification):
        raise CachePolicyError("The actual positive Setup qualification is required")
    native, pin = _slot_read(first.provider, "native_completion")
    if pin != native_completion_binding or native != read(_descriptor(native_completion_binding)):
        raise CachePolicyError("Native completion must occupy its original fixed Setup slot")
    proof = _operation_proof(first, pin, read, read_metadata)
    return _slot_write(first.provider, "completion", proof)


def checked_operation(value, *, provider_binding, root, stores_source, operation, read, read_metadata=None):
    """Replay the actual closed evidence; a readable PASS flag is insufficient."""
    if not isinstance(value, dict) or value.get("format") != CLOSED:
        raise CachePolicyError("The actual closed Setup operation proof is required")
    first = PositiveSetupServiceQualification(value["request"], provider_binding=provider_binding,
        stores_source=stores_source, namespace=root["namespace_admission"], read=read,
        read_metadata=read_metadata)
    if first.operation != operation or value != _operation_proof(first, value["native_completion"], read):
        raise CachePolicyError("Closed Setup evidence is not its exact original native/publication projection")
    return value


def _profile_document(binding, root, stores_source, read, read_metadata=None):
    operations, proofs = {}, {}
    for operation in OPERATIONS:
        value, pin = _slot_read(operation_provider(root, operation), "completion")
        checked_operation(value, provider_binding=binding, root=root, stores_source=stores_source,
                          operation=operation, read=read, read_metadata=read_metadata)
        operations[operation], proofs[operation] = pin, value
    bounds = [proof["capacity_bound"] for proof in proofs.values()]
    if bounds[0]["volume_cap_observation"]["binding"] != bounds[1]["volume_cap_observation"]["binding"]:
        raise CachePolicyError("Apply and Build used different physical cache caps")
    receipts = [read(_descriptor(_slot_read(operation_provider(root, operation), "observation")[0]["service_receipt"]))
                for operation in OPERATIONS]
    roles = {row.get("role_identifier") for row in receipts}
    if len(roles) != 1 or not next(iter(roles)):
        raise CachePolicyError("Apply and Build must use the same original worker entry role")
    maximum = max(proof["maximum_service_bytes"] for proof in proofs.values())
    profile = dict(format=IMPORT_PROFILE, purpose="jobs", entry_kind="worker", role_identifier=roles.pop(),
        provider=deepcopy(binding), stores_source=deepcopy(stores_source), scope=deepcopy(root["scope"]),
        operations=operations, namespace_admission=deepcopy(root["namespace_admission"]),
        python_executable=deepcopy(root["python_executable"]), module_closure=deepcopy(root["module_closure"]),
        capacity_bound=deepcopy(bounds[1]), maximum_cache_bytes=maximum, operational_service_entitlement_bytes=maximum,
        supported_workers=1,
        actual_writer_IO_proven=False, sampled_highwater_is_lifetime_peak=False,
        writer_contract="physical-volume-bound-setup-service-v1", completion=operations["build"],
        native_supervision=dict(format="darsia-native-worker-supervision-profile-v1",
            startup=deepcopy(root["startup"]), completion=deepcopy(proofs["build"]["native_completion"]),
            maximum_family_processes=max(proof["maximum_family_processes"] for proof in proofs.values()),
            publication_contract="alias-done-small-metadata-v1"))
    return profile


def _ordinary_release(value, binding, root, stores_source, profile):
    if (value.get("format") != "darsia-setup-service-ordinary-release-v1"
            or value.get("execution_authorized") is not True or value.get("provider") != binding
            or value.get("stores_source") != stores_source or value.get("scope") != root["scope"]
            or value.get("operations") != profile["operations"]):
        raise CachePolicyError("Independent ordinary use release must name both actual completed Setup operations")


def publish_import_profile(binding, *, stores_source, ordinary_release, read, read_metadata=None):
    """Publish a service-only profile from actual closed Apply and Build evidence."""
    binding = _descriptor(binding)
    root = checked_provider(read(binding))
    profile = _profile_document(binding, root, stores_source, read, read_metadata)
    _ordinary_release(read(_descriptor(ordinary_release)), binding, root, stores_source, profile)
    pin = _slot_write(root, "import_profile", profile)
    return _slot_write(root, "completion", dict(format=BUNDLE, provider=binding, stores_source=deepcopy(stores_source),
        scope=deepcopy(root["scope"]), operations=deepcopy(profile["operations"]), import_profile=pin,
        ordinary_use_release=deepcopy(ordinary_release), complete=True, passed=True))


def resolve_profile(binding, value, kind, read, *, stores_source=None, read_authorities=None, read_metadata=None):
    """Resolve only the complete original service DAG; never return a job profile."""
    root = checked_provider(value)
    if kind != "import" or stores_source is None:
        raise CachePolicyError("Setup qualifies its import/service role only, never a measured analysis job")
    bundle, _ = _slot_read(root, "completion")
    profile = _profile_document(binding, root, stores_source, read, read_metadata)
    _ordinary_release(read(_descriptor(bundle.get("ordinary_use_release"))), binding, root, stores_source, profile)
    actual, pin = _slot_read(root, "import_profile")
    if (bundle.get("format") != BUNDLE or bundle.get("complete") is not True or bundle.get("passed") is not True
            or bundle.get("provider") != binding or bundle.get("stores_source") != stores_source
            or bundle.get("scope") != root["scope"] or bundle.get("operations") != profile["operations"]
            or bundle.get("import_profile") != pin or actual != profile):
        raise CachePolicyError("The Setup service profile differs from its actual completed original DAG")
    return actual


def checked_import_context(profile, *, purpose, role_identifier, entry_kind, policy, namespace):
    """Keep ordinary use inside the same Setup worker/source/namespace/cap."""
    if (profile.get("format") != IMPORT_PROFILE or purpose != "jobs" or entry_kind != "worker"
            or profile.get("purpose") != purpose or profile.get("entry_kind") != entry_kind
            or profile.get("role_identifier") != role_identifier or profile.get("namespace_admission") != namespace
            or profile.get("scope", {}).get("pipeline") != "series-setup"
            or type(profile.get("supported_workers")) is not int or profile["supported_workers"] != 1
            or policy.profiles.get("jobs") != profile.get("provider")
            or policy.volume_cap is None or policy.volume_cap.maximum_size_bytes != policy.quota_bytes
            or profile["capacity_bound"]["volume_cap_observation"]["binding"] != policy.volume_cap.document()
            or profile["capacity_bound"]["volume_cap_observation"].get("virtual_size_bytes") != policy.volume_cap.maximum_size_bytes
            or profile.get("actual_writer_IO_proven") is not False
            or profile.get("sampled_highwater_is_lifetime_peak") is not False
            or profile.get("maximum_cache_bytes") != profile.get("operational_service_entitlement_bytes")
            or profile.get("writer_contract") != "physical-volume-bound-setup-service-v1"):
        raise CachePolicyError("Ordinary Setup service context differs from its genuine qualification")
    return _positive(profile["maximum_cache_bytes"], "Qualified Setup service entitlement")


def require_public_command(profile, command):
    """Check the original controller command; inherited spawn children have no CLI."""
    if (profile.get("format") != IMPORT_PROFILE or type(profile.get("supported_workers")) is not int
            or profile["supported_workers"] != 1):
        raise CachePolicyError("An actually resolved one-worker Setup service profile is required")
    if not isinstance(command, list) or any(not isinstance(arg, str) for arg in command):
        raise CachePolicyError("The actual original public worker command is required")
    if command.count("-m") != 1:
        raise CachePolicyError("Ordinary Setup service requires the original public worker command")
    index = command.index("-m")
    args = command[index + 2:]
    if (command[index + 1:index + 2] != ["darsia.fluidflower.worker"]
            or args.count("--workers") != 1 or args[args.index("--workers") + 1:args.index("--workers") + 2] != ["1"]):
        raise CachePolicyError("Only the actual one-worker Setup service concurrency is qualified")
    return True


def require_applicable(profile, task, *, config=None):
    """Apply service evidence only to genuine current Setup tasks, without a job estimate."""
    scope = profile.get("scope", {})
    if (profile.get("format") != IMPORT_PROFILE or profile.get("purpose") != "jobs"
            or scope.get("pipeline") != "series-setup" or type(profile.get("supported_workers")) is not int
            or profile["supported_workers"] != 1
            or any(task.get(key) != scope.get(key) for key in ("commit", "pipeline", "backend", "precision", "target_machine"))
            or task.get("operation") not in SUPPORTED_OPERATIONS or task.get("ordered_case_ids") != [task.get("operation")]
            or task.get("format") != "darsia-alias-task-v1" or task.get("format_version") != 1
            or task.get("legacy_paths") is not False or task.get("resume_from") is not None):
        raise CachePolicyError("Setup service qualification cannot cover another source/target/pipeline")
    if config is not None:
        if (config.get("format") != "darsia-series-setup-operation-v1"
                or any(config.get(key) != task.get(key) for key in ("run", "operation", "target_machine"))):
            raise CachePolicyError("Ordinary Setup configuration differs from its actual queued operation")
    return True
