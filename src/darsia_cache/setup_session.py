"""A job-bound operational cache budget for the original Setup queue.

Setup already enforces whole-operation RAW, metadata and scratch ceilings. Its
service can therefore use the existing fixed-volume first-service admission
without borrowing a measured analysis profile. This contract is never published
as a measurement and cannot admit P1, calibration or another Setup job.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import PureWindowsPath
import re

from .first_qualification import (FirstQualification, _digest, _exact_keys, _pins,
    administrative_entitlement, checked_required_capacity_bound)
from .policy import CachePolicyError, _descriptor
from .setup_qualification import (PositiveSetupServiceQualification,
    REQUIRED_SOURCE_SUFFIXES, SCOPE_FIELDS, _control_identity, setup_control_metadata)

PROVIDER = "darsia-setup-job-service-provider-v1"
REQUEST = "darsia-setup-job-service-request-v1"
RELEASE = "darsia-setup-job-service-release-v1"


def checked_provider(value):
    _exact_keys(value, {"format", "qualification_id", "scope", "namespace_admission",
        "output_namespace", "slots", "python_executable", "startup", "module_closure",
        "producer_source_closure", "runtime_source_closure", "workers"},
        "Exact job-bound Setup service contract required")
    scope = _exact_keys(value["scope"], SCOPE_FIELDS, "Exact Setup source scope required")
    if (value["format"] != PROVIDER or scope["pipeline"] != "series-setup"
            or scope["backend"] != "cpu" or scope["precision"] not in {"float32", "float64"}
            or not re.fullmatch(r"[0-9a-f]{40}", str(scope["commit"]))
            or not re.fullmatch(r"[0-9a-f]{32}", str(value["qualification_id"]))
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", str(scope["target_machine"]))
            or type(value["workers"]) is not int or value["workers"] < 1):
        raise CachePolicyError("Only the selected CPU Setup job and worker count are admitted")
    for pin in (scope["job"], value["namespace_admission"],
                value["python_executable"], value["startup"]):
        _descriptor(pin)
    output = _exact_keys(value["output_namespace"], {"path", "identity"},
        "Exact existing Setup receipt namespace required")
    if not PureWindowsPath(output["path"]).is_absolute() or not output["identity"]:
        raise CachePolicyError("Existing receipt directory identity required")
    slots = _exact_keys(value["slots"], {"observation", "native_completion", "completion"},
        "Only this service's actual observation and retirement slots are supported")
    if (len(set(slots.values())) != 3 or any(not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,180}\.json", str(name)) for name in slots.values())):
        raise CachePolicyError("Distinct flat immutable receipt names required")
    union = _pins(value["producer_source_closure"]) + _pins(value["runtime_source_closure"])
    closure = _pins(value["module_closure"])
    if len(union) != len(closure) or {_digest(pin) for pin in union} != {_digest(pin) for pin in closure}:
        raise CachePolicyError("Original producer/runtime union differs from startup closure")
    paths = {pin["path"].replace("\\", "/").casefold() for pin in union}
    for suffix in REQUIRED_SOURCE_SUFFIXES | {"darsia_cache/setup_session.py"}:
        if not any(path.endswith("/" + suffix) for path in paths):
            raise CachePolicyError("Setup source closure omits its original writer/admission boundary")
    return deepcopy(value)


def require_command(command, request, provider):
    if not isinstance(command, list) or any(not isinstance(arg, str) for arg in command):
        raise CachePolicyError("Original public worker command required")
    expected = [provider["python_executable"]["path"], "-B", "-m", "darsia.fluidflower.worker",
        "--job", request["job_root"], "--backend", "cpu", "--workers", str(provider["workers"]),
        "--machine", provider["scope"]["target_machine"], "--watchdog",
        "--exit-when-idle", "--stores", request["stores_source"]["path"]]
    if command != expected:
        raise CachePolicyError("Setup service command differs from its exact job, interpreter, stores or concurrency")


class SetupJobService(PositiveSetupServiceQualification):
    """Reuse native ownership, fixed cap, reserves and original per-claim sinks."""
    setup_job_mode = True

    def __init__(self, binding, *, provider_binding, stores_source, namespace, read,
                 read_authorities=None, historical_authorities=None, read_metadata=None):
        if historical_authorities is not None:
            raise CachePolicyError("Historical analysis authority cannot admit a Setup job")
        self.binding, self.provider_binding = _descriptor(binding), _descriptor(provider_binding)
        self.request = read(self.binding)
        _exact_keys(self.request, {"format", "execution_authorized", "root_release", "provider",
            "stores_source", "namespace_admission", "public_command", "task", "job", "job_root",
            "configuration", "capacity_bound_plan", "budget", "cache_stat_period_seconds"},
            "Exact released Setup job service request required")
        self.provider = self.root_provider = checked_provider(read(self.provider_binding))
        self.stores_source, self._read = deepcopy(stores_source), read
        self._read_metadata = read_metadata or setup_control_metadata
        if (self.request["format"] != REQUEST or self.request["execution_authorized"] is not True
                or self.request["provider"] != self.provider_binding
                or self.request["stores_source"] != stores_source
                or self.request["namespace_admission"] != namespace
                or namespace != self.provider["namespace_admission"]
                or self.request["job"] != self.provider["scope"]["job"]
                or type(self.request["cache_stat_period_seconds"]) not in {int, float}
                or not 0 < self.request["cache_stat_period_seconds"] <= 30):
            raise CachePolicyError("Setup job service differs from its selected stores/source/namespace")
        if self.request["budget"] != "fresh-fixed-volume-headroom-with-original-reserves":
            raise CachePolicyError("Existing locked physical-volume administrative budget required")
        release = read(_descriptor(self.request["root_release"]))
        if (release.get("format") != RELEASE or release.get("execution_authorized") is not True
                or release.get("provider") != self.provider_binding or release.get("stores_source") != stores_source
                or release.get("scope") != self.provider["scope"]
                or release.get("request_payload_sha256") != _digest({key: value for key, value in self.request.items()
                                                                      if key != "root_release"})):
            raise CachePolicyError("Exact Start-authorized Setup service release required")
        self.task = self._read_metadata(self.request["task"])
        self.job = self._read_metadata(self.request["job"])
        self.configuration = self._read_metadata(self.request["configuration"])
        _control_identity(self.task, self.job, self.configuration, self.provider["scope"], self.task["operation"])
        root = PureWindowsPath(self.request["job_root"])
        if (not root.is_absolute() or ".." in root.parts
                or PureWindowsPath(self.request["job"]["path"]) != root / "job.json"
                or self.task.get("job_id") != root.name
                or PureWindowsPath(self.request["task"]["path"]) != root / "results" / self.task["run"] / "task.json"
                or self.request["configuration"]["sha256"] != self.task["configuration"].get("sha256")
                or self.request["configuration"]["size_bytes"] != self.task["configuration"].get("bytes")):
            raise CachePolicyError("Setup service must bind genuine immutable job/task/control bytes")
        require_command(self.request["public_command"], self.request, self.provider)
        self.required_capacity_bound = checked_required_capacity_bound(read(self.request["capacity_bound_plan"]))
        if self.required_capacity_bound["enforcement_state"] != "PREDECLARED_Q106_VOLUME_BOUND":
            raise CachePolicyError("Actual accepted physical capacity plan required")
        self.operation = "job"
        self.bound_mode = self.reference_build = False
        self.reference_administration = self.reference_retained_inventory = self.q123_cold_scope = None
        self.reference_roots_validated, self.reference_transport_observation = False, None
        self.started, self.claim = False, None

    def consume(self, lease):
        from .control import _encoded
        from .windows import WindowsNamespace
        target = self.provider["output_namespace"]
        with_namespace = WindowsNamespace(target["path"], target["identity"])
        try:
            if set(with_namespace.names()) & set(self.provider["slots"].values()):
                raise CachePolicyError("Setup launch evidence already exists; no automatic retry")
        finally:
            with_namespace.close()
        self._namespace(lease)
        with lease.ledger.locked():
            report = lease.ledger.project(0)
            actual = report.get("volume_cap_observation", {})
            expected = self.required_capacity_bound["volume_cap_observation"]
            if (any(actual.get(key) != expected.get(key) for key in (
                    "binding", "virtual_size_bytes", "volume_total_bytes", "filesystem", "is_loaded", "verified"))
                    or lease.policy.volume_cap is None
                    or expected.get("binding") != lease.policy.volume_cap.document()):
                raise CachePolicyError("Setup service differs from its actual fixed-volume binding")
            administrative_entitlement(report)  # Original hard cap, reserve and control margin.
            if self._launch_marker_name() in lease.ledger.control.names():
                raise CachePolicyError("This exact Setup service launch was already consumed")
            lease.ledger.control.write_once(self._launch_marker_name(), _encoded(dict(
                format="darsia-cache-first-qualification-launch-admitted-v1", request=self.binding,
                provider=self.provider_binding, stores_source=self.stores_source,
                owner=lease.owner, scope=self.provider["scope"], no_retry=True)))
        self.started = True

    service_entitlement = FirstQualification.service_entitlement

    def require_task(self, lease, task):
        if (getattr(lease, "first_qualification", None) is not self or lease.source != self.stores_source
                or task.get("job_id") != PureWindowsPath(self.request["job_root"]).name
                or task.get("setup_plan") != self.job["setup_plan"]
                or any(task.get(key) != self.provider["scope"][key]
                       for key in ("commit", "pipeline", "backend", "precision", "target_machine"))
                or task.get("run") not in self.job["expected_runs"]
                or task.get("format") != "darsia-alias-task-v1" or task.get("format_version") != 1
                or task.get("legacy_paths") is not False or task.get("resume_from") is not None
                or task.get("ordered_case_ids") != [task.get("operation")]
                or not re.fullmatch(r"[0-9a-f]{32}", str(task.get("attempt_id")))):
            raise CachePolicyError("Bounded Setup service cannot admit another job, source or pipeline")
        expected = next(row["operation"] for row in self.job["setup_operations"] if row["run"] == task["run"])
        if task.get("operation") != expected:
            raise CachePolicyError("Setup task operation differs from its original immutable job")
        return True

    def _remember_claim_locked(self, lease, task):
        # The original queue/RunIO validated this actual task before its atomic
        # claim. A separate immutable marker is needed for each real attempt.
        from .control import _encoded
        from .capacity import _json_bytes
        lease.ledger._require_lock()
        key = _digest(dict(service=self.provider_binding, attempt=task["attempt_id"]))
        name, once = "cache-first-" + key + "-claim.json", "cache-first-" + key + ".json"
        value = dict(format="darsia-cache-first-single-claim-v1", request=self.binding,
            provider=self.provider_binding, task=task, owner=lease.owner)
        consumed = dict(format="darsia-cache-first-qualification-consumed-v1", request=self.binding,
            provider=self.provider_binding, stores_source=self.stores_source,
            owner=lease.owner, scope=self.provider["scope"], no_retry=True)
        names = lease.ledger.control.names()
        if name in names:
            raw, _ = lease.ledger.control.read_once(name)
            if _json_bytes(raw) != value or once not in names:
                raise CachePolicyError("Another owner already consumed this Setup attempt")
            raw, _ = lease.ledger.control.read_once(once)
            if _json_bytes(raw) != consumed:
                raise CachePolicyError("Setup claim consumption bytes differ")
        else:
            lease.ledger.control.write_once(once, _encoded(consumed))
            lease.ledger.control.write_once(name, _encoded(value))
        self.claim = deepcopy(task)

    def whole_job_published(self, publication):
        """Retirement of a partial service is not completion of this Setup job."""
        return (publication.get("job") == self.request["job"]
            and publication.get("terminal_failures") == []
            and publication.get("original_done_and_product_index_metadata_verified") is True
            and len(publication.get("completed", [])) == len(self.job["expected_runs"])
            and {row.get("run") for row in publication["completed"]} == set(self.job["expected_runs"]))
