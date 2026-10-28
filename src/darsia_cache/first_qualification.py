"""Explicit once-only cold qualification, never an ordinary measured profile.

The immutable provider precedes stores; the released request binds both. Only
the append-once completed bundle names final profile hashes. This is a DAG,
not a placeholder hash or an exception to actual stores/source equality.

Sampling records observations, not lifetime byte peaks. External NumPy/Numba/
Matplotlib/tempfile writers currently have no exact allocation journal here.
Their absence is a publication refusal, not an invented passing profile.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import sys
import time
import uuid

from .policy import CachePolicyError, PHYSICAL_RESERVE_BYTES, _descriptor, _module_source_descriptor, nonnegative
from .control import _encoded

REQUEST_ENV = "DARSIA_CACHE_FIRST_QUALIFICATION"
PROVIDER = "darsia-cache-qualified-profile-provider-v1"
REQUEST = "darsia-cache-first-qualification-request-v1"
BOUND_REQUEST = "darsia-cache-first-qualification-bound-request-v1"
REFERENCE_REQUEST = "darsia-n43-reference-build-request-v1"
REFERENCE_ADMINISTRATION = "darsia-n43-reference-build-administration-v1"
REFERENCE_ENTITLEMENT_BYTES = 65728995330
REFERENCE_CASE_IDS = ["ac52_f" + str(i) for i in range(1, 44) if i not in {8, 15, 22, 24}]
# Reference builds the project owner released, each pinned exactly here and reviewed as code:
# authority ids -> the answer records with their deciders, and the accepted plan.
# Every one keeps the aggregate entitlement REFERENCE_ENTITLEMENT_BYTES.
REFERENCE_RELEASES = {
    ("N43", "Q109", "Q106"): dict(
        records={
            "N43-OWNER-GO-execute-plan-9dc0979d-tonight-Main-reference-and-Second-qualification": "owner",
            "Q109-answer-option-A-whole-output-entitlement-65728995330B-all-stages-count": "coordinator",
            "Q106-answer-sampled-operational-reserve-hard-bound-is-Q87-volume-no-new-seam": "coordinator",
        },
        plan_sha256="9dc0979d5cd48ce18df6391253ed3f178f181c0c321626cef179c25a2e59e988",
    ),
    # R3, the RAW39 reference: N134 is the project owner's release, N136 names the plan and
    # entitlement, and N135 is his choice of this exact entry. N139 lets R3
    # start with the retained Q123 font file counted instead of an empty cache.
    # R3's first attempt (service 95c37c1c) stopped before its claim under this
    # entry. The entry stays so that attempt's request can still be rebuilt, for
    # its close; its one-file count no longer matches the cache, so it cannot
    # start again.
    ("N134", "N136", "N139"): dict(
        records={
            "N134-OWNER-r3-inputs-into-a-new-ff-ac-folder-reference-build-once-fresh-cache-released-after-the-code-change": "owner",
            "N136-COORDINATOR-r3-plan-3ea1caa8-entitlement-65728995330B-pinned-for-the-exact-code-entry": "coordinator",
            "N139-OWNER-r3-starts-with-the-verified-117590-byte-retained-font-file-counted-file-and-old-approval-untouched": "owner",
        },
        plan_sha256="3ea1caa8ff432f5c17d2904f7f43080e23aeac039a447d8667a1586749e0426e",
        retained_cache=dict(
            inventory_sha256="be5c8bf204e6a5021f79bd10125542bcfd87adaf80c2401e66cb7ce9463852a2",
            logical_bytes=117590,
        ),
    ),
    # N142 adds the font file of that stopped attempt. R3's second attempt
    # (service 52294fac) stopped at its first frame under this entry, which
    # stays for that attempt's close; with three files on disk it cannot start.
    ("N134", "N136", "N139", "N142"): dict(
        records={
            "N134-OWNER-r3-inputs-into-a-new-ff-ac-folder-reference-build-once-fresh-cache-released-after-the-code-change": "owner",
            "N136-COORDINATOR-r3-plan-3ea1caa8-entitlement-65728995330B-pinned-for-the-exact-code-entry": "coordinator",
            "N139-OWNER-r3-starts-with-the-verified-117590-byte-retained-font-file-counted-file-and-old-approval-untouched": "owner",
            "N142-OWNER-r3-close-failed-service-95c-count-both-font-files-235180-bytes-one-new-start": "owner",
        },
        plan_sha256="3ea1caa8ff432f5c17d2904f7f43080e23aeac039a447d8667a1586749e0426e",
        retained_cache=dict(
            inventory_sha256="668f556a8d20c7d82484a9449521e46522b4d22b1bf2a0290e9f77d323c90a35",
            logical_bytes=235180,
        ),
    ),
    # N145 adds the font file of the second attempt. R3's third attempt
    # (service 7fdd0fc9) stopped before its claim under this entry, which stays
    # for that attempt's close; with four files on disk it cannot start.
    ("N134", "N136", "N139", "N142", "N145"): dict(
        records={
            "N134-OWNER-r3-inputs-into-a-new-ff-ac-folder-reference-build-once-fresh-cache-released-after-the-code-change": "owner",
            "N136-COORDINATOR-r3-plan-3ea1caa8-entitlement-65728995330B-pinned-for-the-exact-code-entry": "coordinator",
            "N139-OWNER-r3-starts-with-the-verified-117590-byte-retained-font-file-counted-file-and-old-approval-untouched": "owner",
            "N142-OWNER-r3-close-failed-service-95c-count-both-font-files-235180-bytes-one-new-start": "owner",
            "N145-OWNER-yes-q157-paired-model-staging-close-5229-count-three-font-files-one-new-start": "owner",
        },
        plan_sha256="3ea1caa8ff432f5c17d2904f7f43080e23aeac039a447d8667a1586749e0426e",
        retained_cache=dict(
            inventory_sha256="96986702430169ac39fe43ea62195db66d36229cd287ddac6fe57f5e46890fce",
            logical_bytes=352770,
        ),
    ),
    # R3 on the drift-fixed RAW positioning (Q159, N152). The integration agent re-minted the
    # RAW binding, configurations and plan for it, and N154 names that plan.
    # N152 adds the font file of the third attempt: R3 starts counting exactly
    # this four-file inventory, verified on disk, and nothing else.
    ("N134", "N139", "N142", "N145", "N152", "N154"): dict(
        records={
            "N134-OWNER-r3-inputs-into-a-new-ff-ac-folder-reference-build-once-fresh-cache-released-after-the-code-change": "owner",
            "N139-OWNER-r3-starts-with-the-verified-117590-byte-retained-font-file-counted-file-and-old-approval-untouched": "owner",
            "N142-OWNER-r3-close-failed-service-95c-count-both-font-files-235180-bytes-one-new-start": "owner",
            "N145-OWNER-yes-q157-paired-model-staging-close-5229-count-three-font-files-one-new-start": "owner",
            "N152-OWNER-yes-raw-drift-fix-first-close-7fdd-count-four-fonts-r3-on-fixed-code-standing-yes": "owner",
            "N154-COORDINATOR-r3-plan-a88d0950-drift-fix-repins-entitlement-65728995330B-four-fonts-pinned": "coordinator",
        },
        plan_sha256="a88d0950617b445a6ec93fae826d2b359a930857cc97bfe24d58b5bc746e2301",
        retained_cache=dict(
            inventory_sha256="5ab92bb37ce086192baeba8ffe0b86ef0a0777d3af22d1bf9f1bda89a6c74b51",
            logical_bytes=470360,
        ),
    ),
}
# Source preparation only. No current V9 plan or human execution GO exists.
# Root must bind the actual new plan and answer records in reviewed source;
# supplied JSON, a preparation answer, or an old registry entry cannot fill it.
CURRENT_V9_REFERENCE_RELEASE = None
CURRENT_V9_REFERENCE_RELEASE_FORMAT = "darsia-current-v9-reference-release-v1"
CURRENT_V9_REFERENCE_PLAN = "darsia-current-v9-reference-plan-v1"
CURRENT_V9_PREPARATION_RECORD = (
    "Q_ROOT_Q430_CURRENT_V9_REFERENCE_AUTHORITY_20261023T0306Z-answer-no-existing-"
    "V9-reference-known-source-prepare-the-minimal-addition-now-execution-waits-for-owner")
RETAINED_INVENTORY = "darsia-cache-retained-file-inventory-v1"


def released_reference(authority_ids):
    """The REFERENCE_RELEASES entry these authority ids name exactly, or None."""

    if not isinstance(authority_ids, list) or not all(isinstance(item, str) for item in authority_ids):
        return None
    historical = REFERENCE_RELEASES.get(tuple(authority_ids))
    if historical is not None:
        return historical
    if CURRENT_V9_REFERENCE_RELEASE is None:
        return None
    try:
        current = checked_current_reference_release(CURRENT_V9_REFERENCE_RELEASE)
    except CachePolicyError:
        return None
    return current if current["authority_ids"] == authority_ids else None
CAPACITY_BOUND = "darsia-cache-required-capacity-bound-v1"
BOUNDED_EVIDENCE = "darsia-cache-physical-bound-sampled-evidence-v1"
Q104_OUTPUT_ENTITLEMENT_BYTES = 27374032352  # Historical metadata parser only; Q119 governs the active bound route.
FIRST_OUTPUT_BASE_ENTITLEMENT_BYTES = 65728995330
BOOTSTRAP_CONTROL_MARGIN_BYTES = 1024**3  # Q133: retained outside the service reservation.
BOUND_OUTPUT_ADMINISTRATION = "darsia-first-qualification-whole-output-administration-v1"
BOUND_BUNDLE = "darsia-cache-volume-bound-completed-bundle-v1"
BOUND_IMPORT_PROFILE = "darsia-cache-volume-bound-import-profile-v1"
BOUND_JOB_PROFILE = "darsia-cache-volume-bound-job-profile-v1"
BOUND_WRITER_CONTRACT = "physical-volume-bound-shared-service-v1"
OBSERVATION = "darsia-cache-first-qualification-observation-v1"
BUNDLE = "darsia-cache-first-qualification-completed-bundle-v1"
SLOTS = {"observation", "writer_evidence", "measurement", "completion",
         "import_profile", "job_profile", "native_completion"}
WRITERS = ["bytecode", "matplotlib", "numba", "tempfile"]


def _same_path(a, b):
    return str(PureWindowsPath(a)).casefold() == str(PureWindowsPath(b)).casefold()


def _digest(value):
    # Hash-only identity is not a generic control record or an output write.
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_json(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":")) + "\n").encode("utf-8")


def _slot_encoded(value):
    if not isinstance(value, dict):
        raise CachePolicyError("First-qualification slot must be a JSON object")
    raw = _canonical_json(value)
    if not 0 < len(raw) <= 16 * 1024**2:
        raise CachePolicyError("First-qualification slot exceeds existing sixteen MiB bound; full samples not recorded")
    return raw


def _exact_keys(value, keys, message):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise CachePolicyError(message)
    return value


def _pins(rows):
    if not isinstance(rows, list) or not rows:
        raise CachePolicyError("Nonempty predeclared exact source/runtime closure required")
    pins = [_module_source_descriptor(row) for row in rows]
    if len({str(PureWindowsPath(pin["path"])).casefold() for pin in pins}) != len(pins):
        raise CachePolicyError("Duplicate first-qualification source identity")
    return pins


def checked_provider(value):
    _exact_keys(value, {"format", "qualification_id", "scope", "namespace_admission",
        "output_namespace", "slots", "python_executable", "module_closure", "startup",
        "producer_source_closure", "runtime_source_closure"}, "Exact immutable provider required")
    if value["format"] != PROVIDER or re.fullmatch(r"[0-9a-f]{32}", value["qualification_id"]) is None:
        raise CachePolicyError("Explicit finite qualification provider required")
    scope = _exact_keys(value["scope"], {"commit", "pipeline", "backend", "precision",
        "run", "ordered_case_ids"}, "Exact case/source/stage scope required")
    if (re.fullmatch(r"[0-9a-f]{40}", scope["commit"]) is None
            or scope["pipeline"] != "native-phase-n1" or scope["backend"] != "cpu"
            or scope["precision"] != "float32" or scope["run"] != "ac52"
            or not isinstance(scope["ordered_case_ids"], list)
            or len(scope["ordered_case_ids"]) != 39
            or len(set(scope["ordered_case_ids"])) != 39
            or any(not isinstance(case, str) or not case for case in scope["ordered_case_ids"])):
        raise CachePolicyError("First qualification is only the predeclared AC52 exact39 CPU float32 route")
    _descriptor(value["namespace_admission"]); _descriptor(value["python_executable"])
    _descriptor(value["startup"])
    target = _exact_keys(value["output_namespace"], {"path", "identity"}, "Owned output namespace required")
    if not isinstance(target["identity"], dict) or not PureWindowsPath(target["path"]).is_absolute():
        raise CachePolicyError("Actual preaccepted output namespace identity required")
    _exact_keys(value["slots"], SLOTS, "All finite append-once output slots required")
    names = list(value["slots"].values())
    if (len(set(names)) != len(names) or any(not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,180}\.json", name) is None for name in names)):
        raise CachePolicyError("Provider slots must be distinct flat immutable JSON names")
    union = _pins(value["producer_source_closure"]) + _pins(value["runtime_source_closure"])
    actual = _pins(value["module_closure"])
    if len(union) != len(actual) or {_digest(pin) for pin in union} != {_digest(pin) for pin in actual}:
        raise CachePolicyError("Producer/runtime union must exactly equal pre-import module closure")
    return deepcopy(value)


def _slot(provider, role):
    return str(Path(provider["output_namespace"]["path"]) / provider["slots"][role])


def _slot_read(provider, role):
    """Read through the preaccepted directory handle; never resolve a new root."""
    from .windows import WindowsNamespace
    target = provider["output_namespace"]
    namespace = WindowsNamespace(target["path"], target["identity"])
    try:
        raw, _ = namespace.read_once(provider["slots"][role], maximum=16*1024**2)
    finally:
        namespace.close()
    from .capacity import _json_bytes
    return _json_bytes(raw), dict(path=_slot(provider, role),
        sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _slot_write(provider, role, value):
    from .windows import WindowsNamespace
    target = provider["output_namespace"]; raw = _slot_encoded(value)
    namespace = WindowsNamespace(target["path"], target["identity"])
    try:
        namespace.write_once(provider["slots"][role], raw, maximum=16*1024**2)
    finally:
        namespace.close()
    return dict(path=_slot(provider, role), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _outputs_absent(provider):
    from .windows import WindowsNamespace
    bound = provider["output_namespace"]
    namespace = WindowsNamespace(bound["path"], bound["identity"])
    try:
        if set(provider["slots"].values()) & set(namespace.names()):
            raise CachePolicyError("A first-qualification output already exists; no rerun, retry or prior-profile bypass")
    finally:
        namespace.close()


def validate_startup(lease):
    """Same actual pre-import closure checks; no invented prior native profile."""
    from .bootstrap import _exact
    from .owned_process import _exact_source
    provider = lease.first_qualification.provider
    startup = _exact(provider["startup"], lease.package_file)
    if (startup.get("format") != "darsia-qualified-worker-startup-source-v1"
            or startup.get("python_executable") != provider["python_executable"]
            or startup.get("module_closure") != provider["module_closure"]
            or startup.get("include_system_site_packages") is not False
            or not startup.get("source_files") or not sys.flags.dont_write_bytecode
            or [str(Path(value or os.getcwd()).resolve()) for value in sys.path] != startup.get("sys_path")):
        raise CachePolicyError("Exact predeclared first startup/interpreter/-B/sys.path required")
    for pin in startup["source_files"]:
        _exact_source(pin)
    for name, module in tuple(sys.modules.items()):
        if getattr(module, "__file__", None) and not name.startswith("darsia."):
            path = str(Path(module.__file__).resolve())
            if startup.get("allowed_loaded_modules", {}).get(name) != path:
                raise CachePolicyError("Undeclared first pre-import module: " + name)
    if any(name.split(".")[0] in {"numba", "matplotlib", "cupy", "cv2", "numpy", "PIL"}
           for name in sys.modules):
        raise CachePolicyError("Numerical/cache producer imported before finite first admission")
    return dict(startup=provider["startup"], publication_contract="alias-done-small-metadata-v1",
        first_qualification_only=True, maximum_family_processes=None)


def administrative_entitlement(report):
    """Keep the original reserve plus Q133's explicit bootstrap/control margin."""
    if report.get("hard_filesystem_quota_verified") is not True:
        raise CachePolicyError("Actual enforced fixed-volume cap required before first allocation")
    if report.get("reserve_bytes") != PHYSICAL_RESERVE_BYTES:
        raise CachePolicyError("Original 40 GiB reserve cannot change in first mode")
    headroom = report["quota_headroom_bytes"]
    available = (report["free_physical_bytes"] - report["active_cache_remaining_bytes"]
        - report["required_noncache_bytes"] - PHYSICAL_RESERVE_BYTES
        - BOOTSTRAP_CONTROL_MARGIN_BYTES)
    if type(headroom) is not int or type(available) is not int or min(headroom, available) <= 0:
        raise CachePolicyError("No actual positive first-mode administrative entitlement")
    return min(headroom, available)


class FirstQualification:
    """Read exact metadata on init; no cache/output writes or environment changes."""
    def __init__(self, binding, *, provider_binding, stores_source, namespace, read,
                 read_authorities=None, historical_authorities=None):
        self.binding = _descriptor(binding)
        self.request = read(self.binding)
        self.reference_build = self.request.get("format") == REFERENCE_REQUEST
        self.bound_mode = self.request.get("format") == BOUND_REQUEST
        fields = {"format", "execution_authorized", "authority_ids", "root_release",
            "provider", "stores_source", "namespace_admission", "scope", "public_command", "task",
            "cache_stat_period_seconds"}
        if self.bound_mode:
            fields |= {"capacity_bound_plan", "output_entitlement_plan"}
        if self.reference_build:
            fields.add("reference_administration")
        if "q123_cold_scope" in self.request:
            if not self.bound_mode:
                raise CachePolicyError("Q123 retained admission requires BOUND_REQUEST")
            fields.add("q123_cold_scope")
        _exact_keys(self.request, fields, "Exact Root-released finite first request required")
        expected_format = (REFERENCE_REQUEST if self.reference_build else
            BOUND_REQUEST if self.bound_mode else REQUEST)
        # A reference build names one exactly released authority set (REFERENCE_RELEASES).
        authorities = (None if self.reference_build else
            ["Q97", "Q99", "Q102", "Q104", "Q106", "Q109", "Q117", "Q119"] if self.bound_mode else ["Q97", "Q99"])
        released = released_reference(self.request.get("authority_ids")) is not None
        if (self.request["format"] != expected_format or self.request["execution_authorized"] is not True
                or (not released if authorities is None else self.request["authority_ids"] != authorities)
                or self.request["provider"] != provider_binding
                or self.request["stores_source"] != stores_source
                or self.request["namespace_admission"] != namespace):
            raise CachePolicyError("Finite release/stores/provider/authority binding differs")
        period=self.request["cache_stat_period_seconds"]
        if type(period) not in {int,float} or not 0<period<=30:
            raise CachePolicyError("Explicit sampled cache-stat period in (0,30] seconds required")
        release = read(_descriptor(self.request["root_release"]))
        payload={key:value for key,value in self.request.items() if key!="root_release"}
        if (release.get("format") != "darsia-cache-first-qualification-root-release-v1"
                or release.get("execution_authorized") is not True
                or release.get("provider") != provider_binding
                or release.get("stores_source") != stores_source
                or release.get("scope") != self.request["scope"]
                or release.get("request_payload_sha256") != _digest(payload)):
            raise CachePolicyError("Actual Root-released first scope required")
        if "historical_authority_resolution" in release:
            if not self.bound_mode or "q123_cold_scope" not in self.request:
                raise CachePolicyError("Q134 map belongs only to a new bound retained-scope request")
            from .bootstrap import _HistoricalAuthorityReader
            if historical_authorities is not None:
                if not isinstance(historical_authorities, _HistoricalAuthorityReader):
                    raise CachePolicyError("Exact Root-bound historical authority reader required")
                historical_authorities.for_original(self.binding)
            historical_authorities = _HistoricalAuthorityReader(
                release["historical_authority_resolution"], __file__)
            read_authorities = historical_authorities.current_authorities
        elif historical_authorities is not None:
            from .bootstrap import _HistoricalAuthorityReader
            if not isinstance(historical_authorities, _HistoricalAuthorityReader):
                raise CachePolicyError("Exact Root-bound historical authority reader required")
            historical_authorities.for_original(self.binding)
        self.historical_authorities = historical_authorities
        self.provider_binding = _descriptor(provider_binding)
        self.provider = checked_provider(read(self.provider_binding))
        if (self.request["scope"] != self.provider["scope"]
                or namespace != self.provider["namespace_admission"]
                or not isinstance(self.request["public_command"], list)
                or not self.request["public_command"]
                or any(not isinstance(arg, str) for arg in self.request["public_command"])):
            raise CachePolicyError("First request is not its immutable case/source provider scope")
        self.stores_source = deepcopy(stores_source)
        self.task=read(_descriptor(self.request["task"]))
        if (not isinstance(self.task,dict)
                or {key:self.task.get(key) for key in ("commit","pipeline","precision","run","ordered_case_ids")}
                   != {key:self.provider["scope"][key] for key in ("commit","pipeline","precision","run","ordered_case_ids")}
                or re.fullmatch(r"[0-9a-f]{32}",str(self.task.get("attempt_id"))) is None):
            raise CachePolicyError("Exact original predeclared task/input/calibration binding required")
        self.required_capacity_bound = None
        self.output_entitlement = None
        if self.bound_mode:
            self.required_capacity_bound = checked_required_capacity_bound(
                read(_descriptor(self.request["capacity_bound_plan"])))
            self.output_entitlement = checked_output_entitlement(
                self.task, read(_descriptor(self.request["output_entitlement_plan"])),
                task_binding=self.request["task"], read=read, read_authorities=read_authorities)
        self.reference_administration = None
        self.reference_roots_validated = False
        self.reference_transport_observation = None
        self.reference_retained_inventory = None
        if self.reference_build:
            self.reference_administration = checked_reference_administration(
                read(_descriptor(self.request["reference_administration"])), self, read=read,
                read_authorities=read_authorities)
            if self.request["authority_ids"] != self.reference_administration["authority_ids"]:
                raise CachePolicyError("Finite release/stores/provider/authority binding differs")
            if "retained_cache_inventory" in self.reference_administration:
                # Already checked against its release pin; kept for admission.
                self.reference_retained_inventory = read(
                    _descriptor(self.reference_administration["retained_cache_inventory"]))
                self._reference_retained_seal = _digest(self.reference_retained_inventory)
        self.q123_cold_scope = None
        if "q123_cold_scope" in self.request:
            from .retained_qualification import RetainedColdScope
            self.q123_cold_scope = RetainedColdScope(self, self.request["q123_cold_scope"],
                read=read, read_authorities=read_authorities,
                historical_authorities=historical_authorities)
        self.started = False
        self.claim = None

    def consume(self, lease):
        """Admit one native launch; only a successful claim consumes the scope."""
        if self.bound_mode:
            if (not isinstance(self.output_entitlement, dict)
                    or self.output_entitlement.get("format") != BOUND_OUTPUT_ADMINISTRATION
                    or not isinstance(self.required_capacity_bound, dict)
                    or self.required_capacity_bound.get("enforcement_state") != "PREDECLARED_Q106_VOLUME_BOUND"):
                raise CachePolicyError("Bound prelaunch requires actual Q117/Q119 reference, output and physical-bound data")
        _outputs_absent(self.provider)
        self._namespace(lease)
        key = self._once_key()
        with lease.ledger.locked():
            report=lease.ledger.project(0)
            if getattr(self, "q123_cold_scope", None) is not None:
                self.q123_cold_scope.admit(self, lease, report)
            elif (report["current_cache_logical_bytes"]!=self.initial_cache_bytes(lease)
                    or report["active_cache_remaining_bytes"]!=0
                    or (not self.reference_build
                        and report.get("hard_filesystem_quota_verified") is not True)):
                raise CachePolicyError("First cold qualification requires its actual empty exclusive capped cache")
            if self.reference_build:
                self.reference_admission(lease, report=report, initial=True)
            elif self.bound_mode:
                cap = getattr(lease.policy, "volume_cap", None)
                actual = report.get("volume_cap_observation", {})
                planned = self.required_capacity_bound["volume_cap_observation"]
                if (cap is None or lease.policy.quota_bytes != 500*1024**3
                        or cap.maximum_size_bytes != 500*1024**3
                        or planned.get("binding") != cap.document()
                        or any(actual.get(key) != planned.get(key) for key in (
                            "binding", "virtual_size_bytes", "volume_total_bytes", "filesystem", "is_loaded", "verified"))
                        or actual.get("virtual_size_bytes") != 500*1024**3
                        or report.get("reserve_bytes") != PHYSICAL_RESERVE_BYTES
                        or report.get("free_physical_bytes", 0) <= PHYSICAL_RESERVE_BYTES):
                    raise CachePolicyError("Fresh original Q87 hard cap, same namespace and operational reserve required")
            # A prior actual claim, including legacy consumed markers, remains
            # final. A failed preclaim launch retains separate immutable proof;
            # only a new Root-bound request may have another launch marker.
            names = lease.ledger.control.names()
            if ("cache-first-" + key + ".json" in names
                    or "cache-first-" + key + "-claim.json" in names):
                raise CachePolicyError("An actual first claim already consumed this qualification")
            lease.ledger.control.write_once(self._launch_marker_name(), _encoded(dict(
                format="darsia-cache-first-qualification-launch-admitted-v1", request=self.binding,
                provider=self.provider_binding, stores_source=self.stores_source,
                owner=lease.owner, scope=self.provider["scope"], no_retry=True)))
        self.started = True

    def initial_cache_bytes(self, lease):
        """Cache bytes a cold start may find: 0, or its release's retained files.

        Only a REFERENCE_RELEASES entry with retained_cache admits files, and
        then exactly its inventory, verified on disk, with every earlier
        reservation closed. Call it with the host ledger lock held.
        """
        inventory = getattr(self, "reference_retained_inventory", None)
        if inventory is None:
            return 0
        from .retained_qualification import require_prior_reservations_closed
        from .service_context import verify_retained_cache_inventory
        if _digest(inventory) != self._reference_retained_seal:
            raise CachePolicyError("Released retained inventory changed in memory")
        require_prior_reservations_closed(lease.ledger.control.names())
        return verify_retained_cache_inventory(inventory, lease.policy.root)["logical_bytes"]

    def _launch_marker_name(self):
        return "cache-first-launch-" + _digest(dict(
            qualification=self._once_key(), request=self.binding)) + ".json"

    def _once_key(self):
        return _digest(dict(scope=self.provider["scope"], python=self.provider["python_executable"],
            sources=self.provider["module_closure"], namespace=self.provider["namespace_admission"]))

    def _namespace(self, lease):
        if self.reference_build:
            return reference_namespace(self, lease)
        if (self.provider["output_namespace"] != lease.admission["receipts"]
                or self.provider["namespace_admission"] != lease.document["cache_bootstrap"]["namespace_admission"]
                or lease.policy.profiles.get("jobs")!=self.provider_binding
                or lease.policy.volume_cap is None
                or type(lease.policy.quota_bytes) is not int or lease.policy.quota_bytes <= 0
                or getattr(lease.policy.volume_cap, "maximum_size_bytes", None) != lease.policy.quota_bytes):
            raise CachePolicyError("First output/cache namespace is not the actual accepted one-host cap")

    def service_entitlement(self, lease):
        self._require_launch_admission(lease)
        report = lease.ledger.project(0)
        if getattr(self, "q123_cold_scope", None) is not None:
            self.q123_cold_scope.admit(self, lease, report)
        if self.reference_build:
            admission = self.reference_admission(lease, report=report, initial=True)
            self.administrative = dict(maximum_bytes=REFERENCE_ENTITLEMENT_BYTES,
                measured_profile=False, allocation_model="N43-predeclared-all-stage-aggregate-once",
                request=self.binding, admission=admission)
            return REFERENCE_ENTITLEMENT_BYTES
        maximum = administrative_entitlement(report)
        self.administrative = dict(maximum_bytes=maximum, measured_profile=False,
            bootstrap_control_margin_bytes=BOOTSTRAP_CONTROL_MARGIN_BYTES,
            semantics="exclusive administrative entitlement from actual quota/free less original reserves and explicit bootstrap/control margin",
            admission=report)
        return maximum

    def _require_launch_admission(self, lease):
        self._namespace(lease)
        # Called inside the original service reservation transaction. A direct
        # worker cannot bypass the exact native-controller launch admission.
        if getattr(lease,"native_launch_context",None) is None:
            raise CachePolicyError("First service must be inside the original controlled native Job")
        raw,_=lease.ledger.control.read_once(self._launch_marker_name())
        from .capacity import _json_bytes
        marker=_json_bytes(raw)
        if (marker.get("format")!="darsia-cache-first-qualification-launch-admitted-v1"
                or marker.get("request")!=self.binding or marker.get("provider")!=self.provider_binding
                or marker.get("stores_source")!=self.stores_source
                or marker.get("scope")!=self.provider["scope"] or marker.get("no_retry") is not True):
            raise CachePolicyError("Original one-shot first launch marker differs")

    def require_task(self, lease, task):
        scope = self.provider["scope"]
        if (getattr(lease, "first_qualification", None) is not self
                or {key: task.get(key) for key in ("commit", "pipeline", "precision", "run", "ordered_case_ids")}
                   != {key: scope[key] for key in ("commit", "pipeline", "precision", "run", "ordered_case_ids")}
                or lease.source != self.stores_source or task!=self.task):
            raise CachePolicyError("First claim differs from exact released source/stage/cases/stores")
        if self.claim is not None and self.claim != task:
            raise CachePolicyError("Only one actual exact39 claim may use this first service")
        return True

    def remember_claim_locked(self, lease, claim):
        """The original atomic queue claim succeeded; retain consumption first."""
        lease.ledger._require_lock()
        if claim.token != claim.task.get("attempt_id"):
            raise CachePolicyError("Actual first claim token differs from its released task")
        self.require_task(lease, claim.task)
        self._require_launch_admission(lease)
        if self.reference_build:
            reference_writer_coverage(self, lease, writer_environment=True)
        self._remember_claim_locked(lease, claim.task)

    def _remember_claim_locked(self, lease, task):
        # The caller holds the original result-volume -> cache transaction.
        # Consume before any later result/cache reservation or receipt can fail.
        lease.ledger._require_lock()
        name = "cache-first-" + self._once_key() + "-claim.json"
        once = "cache-first-" + self._once_key() + ".json"
        value = dict(format="darsia-cache-first-single-claim-v1", request=self.binding,
            provider=self.provider_binding, task=task, owner=lease.owner)
        consumed = dict(format="darsia-cache-first-qualification-consumed-v1",
            request=self.binding, provider=self.provider_binding, stores_source=self.stores_source,
            owner=lease.owner, scope=self.provider["scope"], no_retry=True)
        names = lease.ledger.control.names()
        if name in names:
            from .capacity import _json_bytes
            raw, _ = lease.ledger.control.read_once(name)
            if _json_bytes(raw) != value or once not in names:
                raise CachePolicyError("A different first claim/owner already consumed this entitlement")
            raw, _ = lease.ledger.control.read_once(once)
            if _json_bytes(raw) != consumed:
                raise CachePolicyError("Actual consumed first-claim marker differs")
        else:
            lease.ledger.control.write_once(once, _encoded(consumed))
            lease.ledger.control.write_once(name, _encoded(value))
        self.claim = deepcopy(task)

    def reference_admission(self, lease, *, report=None, initial=False, other_remaining=0):
        """One explicit aggregate, independently admitted on each affected volume."""
        if not self.reference_build:
            raise CachePolicyError("Reference administration was not explicitly released")
        self._namespace(lease)
        if report is None:
            report = lease.ledger.project(0)
        if report["reserve_bytes"] != PHYSICAL_RESERVE_BYTES or other_remaining != 0:
            raise CachePolicyError("N43 requires the original reserve and an exclusive result allocation")
        hard=report.get("hard_filesystem_quota_verified") is True
        if hard != (self.reference_administration["host_policy"]=="main-q87-fixed500"):
            raise CachePolicyError("N43 actual hard-cap observation differs from the declared host exception")
        if initial and (report["current_cache_logical_bytes"] != self.initial_cache_bytes(lease)
                or report["active_cache_remaining_bytes"] != 0):
            raise CachePolicyError("N43 requires its fresh exclusive cache and no prior service allocation")
        snapshot = reference_scope_snapshot(self, lease)
        if snapshot["observed_scan_anomalies"]:
            raise CachePolicyError("N43 admission requires an actual complete scan without observed identity/rename errors")
        if initial and any(row["logical_bytes"] for row in snapshot["roots"] if row["role"]=="results"):
            raise CachePolicyError("N43 new reference requires its actual empty result namespace; no borrowed/resumed products")
        if snapshot["logical_bytes"] > REFERENCE_ENTITLEMENT_BYTES:
            latch_reference_stop(self, lease, "AGGREGATE_ENTITLEMENT_EXCEEDED", snapshot)
        require_reference_not_stopped(self, lease)
        if any(v["free_bytes"] - PHYSICAL_RESERVE_BYTES < REFERENCE_ENTITLEMENT_BYTES
               for v in snapshot["volumes"]):
            raise CachePolicyError("J6 free minus original40GiB does not cover the whole N43 entitlement")
        return dict(format="darsia-n43-predeclared-admission-v1", request=self.binding,
            entitlement_bytes=REFERENCE_ENTITLEMENT_BYTES, aggregate_counted_once=True,
            measured_profile=False, snapshot=snapshot, sufficient=True,
            reserve_bytes=PHYSICAL_RESERVE_BYTES,
            hard_cap_exists=self.reference_administration["host_policy"]=="main-q87-fixed500",
            no_ordinary_profile_adoption=True)

    def loaded_modules(self):
        """Actual exiting child inventory; unknown files are not relabelled known."""
        from .owned_process import _exact_source
        declared={str(Path(pin["path"]).resolve()).casefold():(index,pin)
            for index,pin in enumerate(self.provider["module_closure"])}
        actual=[];unknown=[];checked=set()
        for name,module in sorted(tuple(sys.modules.items())):
            filename=getattr(module,"__file__",None)
            if filename is None:
                actual.append(dict(name=name,file=None,builtin_or_no_file=True))
                continue
            path=str(Path(filename).resolve());bound=declared.get(path.casefold())
            if bound is None:
                unknown.append(dict(name=name,path=path))
                actual.append(dict(name=name,path=path,source_index=None))
            elif path.casefold() not in checked:
                _exact_source(bound[1]);checked.add(path.casefold())
            if bound is not None:
                actual.append(dict(name=name,source_index=bound[0]))
        result = dict(format="darsia-cache-first-actual-loaded-modules-v1",request=self.binding,
            provider=self.provider_binding,actual=actual,undeclared=unknown,
            all_file_backed_modules_predeclared_and_exact=not unknown,
            predeclared_module_closure_sha256=_digest(self.provider["module_closure"]),
            module_inventory_scope="actual Python sys.modules file bindings; not an OS DLL enumeration",
            actual_task=deepcopy(self.claim),this_is_not_process_retirement=True)
        if self.bound_mode or getattr(self, "setup_mode", False) or getattr(self, "preset_mode", False):
            from .environment import FIRST_WRITER_ENVIRONMENT_KEYS
            result["actual_writer_environment"] = {
                key: os.environ.get(key) for key in FIRST_WRITER_ENVIRONMENT_KEYS}
            result["writer_environment_is_actual_IO_or_lifetime_proof"] = False
        return result


def request_existing_frame_boundary_stop(first, lease):
    """Existing master-only append command; never kill, spoof or rewrite a job."""
    argv = first.request["public_command"]
    positions = [i for i, argument in enumerate(argv) if argument == "--job"]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        return dict(requested=False, state="UNSUPPORTED_PUBLIC_JOB_ARGUMENT")
    root = argv[positions[0] + 1]
    if not PureWindowsPath(root).is_absolute():
        return dict(requested=False, state="UNSUPPORTED_NONABSOLUTE_JOB_ROOT")
    try:
        from darsia.fluidflower import job_control
        job = job_control.read_json(Path(root) / "job.json")
        # Alias job.json has no job_id field: the original queue binds it to
        # the job directory name. Never invent a new JSON schema here.
        identity_differs = (job != first.job or job.get("job_id") != first.task.get("job_id")
                            if getattr(first, "preset_mode", False)
                            else PureWindowsPath(root).name != first.task.get("job_id"))
        if (identity_differs
                or job.get("commit") != first.provider["scope"]["commit"]):
            raise CachePolicyError("Actual public job differs from the released first task")
        current = job_control.read_control(root)
        if current["action"] in {"pause", "stop"}:
            return dict(requested=False, state="EXISTING_BOUNDARY_EXIT_COMMAND",
                actual_control=current)
        command = job_control.publish_control(root, "stop", machine=lease.owner["machine"],
            expected_previous=current.get("command"))
        return dict(requested=True, state="EXISTING_MASTER_STOP_COMMAND_PUBLISHED",
            command=command, semantics="cooperative exit at original frame boundary; not immediate termination")
    except Exception as error:
        return dict(requested=False, state="STOP_COMMAND_NOT_PUBLISHED", error_type=type(error).__name__,
            message=str(error), semantics="reserve crossing is STOP even when the command is unavailable")


class FiniteObservations:
    """Truthful samples plus independently kernel-maintained lifetime counters."""
    def __init__(self, first, *, clock=None):
        self.first = first
        self.clock=time.monotonic if clock is None else clock
        self.cache_stat_period_seconds=first.request["cache_stat_period_seconds"]
        self.samples = 0
        self.native_poll_count=0
        self.cache_samples=[]
        self.last_cache_start=None
        self.sampled_cache_logical_high_water_bytes = 0
        self.sampled_active_processes_high_water = 0
        self.last = None
        self.reserve_samples = []
        self.last_reserve_start = None
        self.reserve_crossed = False
        self.stop_response = None
        self.aggregate_samples = []
        self.last_aggregate = None

    def observe_reserve(self, owned, started, *, force=False):
        """Q106 sampled operational reserve; a sample never proves a free-floor."""
        if not (self.first.bound_mode or self.first.reference_build
                or getattr(self.first, "setup_mode", False) or getattr(self.first, "preset_mode", False)):
            return
        period = (1 if self.first.reference_build else
            self.first.required_capacity_bound["reserve_watchdog_period_seconds"])
        if (not force and self.last_reserve_start is not None
                and started - self.last_reserve_start < period):
            return
        from .bootstrap import _utc
        from .capacity import volume_info
        aggregate = None
        if self.first.reference_build:
            try:
                aggregate = reference_scope_snapshot(self.first,owned.lease)
            except BaseException as error:
                latch_reference_stop(self.first,owned.lease,"AGGREGATE_OBSERVATION_UNAVAILABLE",
                    dict(error_type=type(error).__name__,message=str(error)))
                raise
        row = volume_info(owned.lease.policy.root)
        crossed = (any(v["free_bytes"] <= PHYSICAL_RESERVE_BYTES for v in aggregate["volumes"])
            if aggregate is not None else row["free_bytes"] <= PHYSICAL_RESERVE_BYTES)
        if aggregate is not None:
            aggregate["started_monotonic_seconds"] = started
            aggregate["actual_gap_since_previous_start_seconds"] = (None if self.last_reserve_start is None
                else started-self.last_reserve_start)
            self.last_aggregate = aggregate
            self.aggregate_samples.append(dict(utc=aggregate["utc"],
                started_monotonic_seconds=started,
                actual_gap_since_previous_start_seconds=aggregate["actual_gap_since_previous_start_seconds"],
                logical_bytes=aggregate["logical_bytes"],entries=aggregate["entries"],
                roots_logical_bytes={row["role"]:row["logical_bytes"] for row in aggregate["roots"]},
                volumes_free_bytes={row["volume_id"]:row["free_bytes"] for row in aggregate["volumes"]},
                forced_terminal=force,sampled=True,complete_lifetime_peak=False,
                observed_scan_anomalies=aggregate["observed_scan_anomalies"]))
            if aggregate["logical_bytes"] > REFERENCE_ENTITLEMENT_BYTES:
                latch_reference_stop(self.first, owned.lease, "AGGREGATE_ENTITLEMENT_EXCEEDED", aggregate)
        self.reserve_samples.append(dict(utc=_utc(), monotonic_seconds=started,
            actual_gap_since_previous_start_seconds=(None if self.last_reserve_start is None
                else started - self.last_reserve_start), forced_terminal=force,
            operational_reserve_bytes=PHYSICAL_RESERVE_BYTES,
            free_volume_observation=row, at_or_below_operational_reserve=crossed))
        self.last_reserve_start = started
        if crossed and not self.reserve_crossed:
            self.reserve_crossed = True  # Permanent STOP, even if free bytes later recover.
            if self.first.reference_build:
                latch_reference_stop(self.first, owned.lease, "OPERATIONAL_RESERVE_CROSSED", aggregate)
            self.stop_response = request_existing_frame_boundary_stop(self.first, owned.lease)

    def observe(self, owned, accounting, *, force_cache=False):
        if not force_cache:self.native_poll_count+=1
        self.sampled_active_processes_high_water = max(self.sampled_active_processes_high_water,
            accounting["active_processes"])
        started=self.clock()
        self.observe_reserve(owned, started, force=force_cache)
        if (not force_cache and self.last_cache_start is not None
                and started-self.last_cache_start<self.cache_stat_period_seconds):
            return  # Native retirement/accounting polling remains cheap.
        from .bootstrap import _utc
        utc_started=_utc()
        lease = owned.lease
        if self.first.reference_build:
            # Diagnostic view of the already observed aggregate, NOT a fresh
            # cache-ledger/per-token measurement. No second handle-based walk.
            aggregate=self.last_aggregate
            if aggregate is None:
                raise CachePolicyError("N43 diagnostic cache view has no actual aggregate observation")
            cache=[value for value in aggregate["roots"] if value["role"]=="cache"]
            if len(cache)!=1 or not _same_path(cache[0]["path"],lease.policy.root):
                raise CachePolicyError("N43 diagnostic cache row differs from the actual cache root")
            row=dict(format="darsia-n43-aggregate-cache-diagnostic-view-v1",root=cache[0]["path"],
                logical_bytes=cache[0]["logical_bytes"],source_aggregate_utc=aggregate["utc"],
                source_aggregate_monotonic_seconds=aggregate["started_monotonic_seconds"],
                actual_sample_age_seconds=started-aggregate["started_monotonic_seconds"],
                observed_scan_anomalies=aggregate["observed_scan_anomalies"],
                non_atomic_sample=True,per_token_accounting_provided=False,
                ordinary_cache_ledger_measurement=False,complete_lifetime_peak=False)
        else:
            with lease.ledger.locked(create_binding=False):
                row = lease.measure(lease.policy.root)
        ended=self.clock()
        self.cache_samples.append(dict(started_utc=utc_started,finished_utc=_utc(),
            started_monotonic_seconds=started,elapsed_seconds=ended-started,
            actual_gap_since_previous_start_seconds=(None if self.last_cache_start is None
                else started-self.last_cache_start),forced_terminal=force_cache,observation=row))
        self.last_cache_start=started
        self.samples += 1
        self.sampled_cache_logical_high_water_bytes = max(self.sampled_cache_logical_high_water_bytes,
            row["logical_bytes"])
        self.last = row

    def document(self, owned, terminal, retirement, eligibility):
        peaks = owned.api.lifetime_memory(owned.job)
        total = nonnegative(terminal["accounting"]["total_processes"], "Actual kernel total process count")
        if not total or terminal["accounting"]["active_processes"] != 0:
            raise CachePolicyError("Kernel whole-family retirement required for observations")
        result = dict(format=OBSERVATION, request=self.first.binding,
            provider=self.first.provider_binding, stores_source=self.first.stores_source,
            scope=self.first.provider["scope"], service_receipt=owned.service_binding,
            native_launch=owned.observed_binding, retirement=retirement,
            original_publication_and_retirement_eligibility=eligibility,
            actual_public_worker_exit_code=terminal["wrapper_exit_code"],
            original_kernel_terminal=deepcopy(terminal),
            sample_count=self.samples, final_sample=self.last,
            native_accounting_poll_count=self.native_poll_count,
            cache_stat_period_seconds=self.cache_stat_period_seconds,
            cache_samples=self.cache_samples,
            cache_sampling_semantics="initial, declared bounded periodic and forced terminal lstat samples; actual times/gaps retained; no peak interpolation",
            sampled_cache_logical_high_water_bytes=self.sampled_cache_logical_high_water_bytes,
            sampled_cache_logical_high_water_is_lifetime_peak=False,
            sampled_active_processes_high_water=self.sampled_active_processes_high_water,
            kernel_total_processes=total, kernel_total_processes_is_concurrent_peak=False,
            conservative_family_process_bound_from_kernel_total=total,
            kernel_lifetime_memory=peaks,
            reserve_watchdog_period_seconds=(1 if (self.first.bound_mode or self.first.reference_build
                or getattr(self.first, "setup_mode", False) or getattr(self.first, "preset_mode", False)) else None),
            reserve_samples=self.reserve_samples, operational_reserve_crossed=self.reserve_crossed,
            operational_stop_response=self.stop_response,
            reserve_semantics="operational admission plus sampled watchdog; not a hard free-volume floor",
            hard_bound_semantics=("NO_HARD_CAP: explicit one-run SecondD N43 human exception"
                if self.first.reference_build and self.first.reference_administration["host_policy"]=="second-n43-one-run-uncapped-d"
                else "actual fixed-volume capacity only; never capacity minus40GiB"),
            reserve_samples_prove_no_crossing_between_samples=False,
            reserve_sampling_scope="existing native await_family observer plus forced terminal sample; initial startup/blocked scan gaps are not continuous coverage",
            writer_lifetime_cache_bytes_proof=None,
            profile_publication_state="REFUSED_UNSUPPORTED_LIFETIME_WRITER_COVERAGE",
            complete=True, passed=False, no_ordinary_profile_created=True)
        if self.first.reference_build:
            result.update(reference_aggregate_samples=self.aggregate_samples,
                final_reference_aggregate=self.last_aggregate,aggregate_samples_are_lifetime_peak=False)
        return result


def checked_bound_output_scope(value, *, common=False, preset=False):
    """One current output root, not the old reference tree or the cache cap."""
    _exact_keys(value, {"format", "role", "path", "identity", "stat_identity", "volume_id",
        "entry_limit", "sample_period_seconds"}, "Exact current Q104 output scope required")
    path = PureWindowsPath(value["path"])
    if common and preset:
        raise CachePolicyError("One exact output route is required")
    expected_format = ("darsia-preset-wasserstein-output-scope-v1" if preset else "darsia-common-phase-output-scope-v1" if common
                       else "darsia-first-q104-output-scope-v1")
    if (value["format"] != expected_format or value["role"] != "results"
            or not path.is_absolute() or ".." in path.parts
            or not isinstance(value["identity"], dict) or not value["identity"]
            or not isinstance(value["volume_id"], str) or not value["volume_id"]
            or type(value["entry_limit"]) is not int or not 0 < value["entry_limit"] <= 100000
            or type(value["sample_period_seconds"]) not in {int, float}
            or value["sample_period_seconds"] != 1):
        raise CachePolicyError("One finite identity-bound output root and original one-second target required")
    _exact_keys(value["stat_identity"], {"st_dev", "st_ino", "creation_ns"}, "Actual output stat identity required")
    if any(type(v) is not int or v <= 0 for v in value["stat_identity"].values()):
        raise CachePolicyError("Actual output stat identity cannot be UNKNOWN")
    return deepcopy(value)


def bound_output_snapshot(scope, *, entitlement_bytes, include_file_stats=False, common=False, preset=False):
    """N43's finite no-follow length route, confined to ONE output-only root.

    No payload is read. Rename anomalies remain explicit lower-bound samples;
    no output quota, cache-bound or complete lifetime peak is asserted.
    """
    import stat
    from .capacity import volume_info
    from .bootstrap import _utc
    scope = checked_bound_output_scope(scope, common=common, preset=preset)
    minimum = 1 if common or preset else FIRST_OUTPUT_BASE_ENTITLEMENT_BYTES
    if type(entitlement_bytes) is not int or entitlement_bytes < minimum:
        raise CachePolicyError("Actual predeclared Q119 entitlement required before sampling")
    root = Path(scope["path"])
    before = _reference_stat_identity(_reference_directory_stat(root))
    if before != scope["stat_identity"]:
        raise CachePolicyError("Q104 current output root identity changed")
    total = entries = 0; identities = set(); anomalies = []; files = []
    def walk(directory):
        nonlocal total, entries
        try:
            start = _reference_stat_identity(_reference_directory_stat(directory))
            with os.scandir(directory) as stream:
                for entry in stream:
                    entries += 1
                    if entries > scope["entry_limit"]:
                        raise CachePolicyError("Q104 finite output accounting entry limit exceeded")
                    try:
                        info = Path(entry.path).lstat()
                    except FileNotFoundError:
                        anomalies.append(dict(path=entry.path, state="VANISHED_DURING_SAMPLE")); continue
                    if getattr(info, "st_reparse_tag", 0) or getattr(info, "st_file_attributes", 0) & 0x400:
                        raise CachePolicyError("Q104 output reparse entry refused")
                    if stat.S_ISDIR(info.st_mode):
                        walk(Path(entry.path))
                    else:
                        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                            raise CachePolicyError("Q104 output file type/hardlink refused")
                        identity = (info.st_dev, info.st_ino)
                        if identity in identities:
                            anomalies.append(dict(path=entry.path, state="IDENTITY_SEEN_TWICE_DURING_SAMPLE")); continue
                        identities.add(identity); total += info.st_size
                        if include_file_stats:
                            files.append(dict(relative_path=str(Path(entry.path).relative_to(root)),
                                size_bytes=info.st_size, st_dev=info.st_dev, st_ino=info.st_ino,
                                mtime_ns=info.st_mtime_ns,
                                creation_ns=getattr(info, "st_birthtime_ns", info.st_ctime_ns)))
            if _reference_stat_identity(_reference_directory_stat(directory)) != start:
                anomalies.append(dict(path=str(directory), state="DIRECTORY_IDENTITY_CHANGED_DURING_SAMPLE"))
        except FileNotFoundError:
            anomalies.append(dict(path=str(directory), state="DIRECTORY_MOVED_DURING_SAMPLE"))
    walk(root)
    after = _reference_stat_identity(_reference_directory_stat(root))
    if before != after:
        raise CachePolicyError("Q104 output root changed during length sample")
    volume = volume_info(scope["path"])
    if volume["volume_id"] != scope["volume_id"]:
        raise CachePolicyError("Q104 actual output volume identity changed")
    return dict(utc=_utc(), root=scope["path"], identity=scope["identity"],
        stat_identity_before=before, stat_identity_after=after, logical_bytes=total,
        entries=entries, volume=volume, entitlement_bytes=entitlement_bytes,
        counting_scope="whole current owned result root: all preparation, in-flight, final and ops files once",
        sampled=True, non_atomic=True, complete_lifetime_peak=False, output_hard_quota=False,
        observed_scan_anomalies=anomalies, observed_bytes_are_lower_bound_if_scan_anomalies=bool(anomalies),
        file_stats=(files if include_file_stats else None), payload_SHA_or_arrays_read=False)


def latch_bound_output_stop(first, lease, reason, snapshot):
    """Same immutable first-stop record and existing cooperative command route."""
    name = "first-q104-stop-" + first._once_key() + ".json"
    # The full stat inventory belongs in the existing sixteen-MiB measurement
    # slot. Keep the existing one-MiB host-ledger control record bounded.
    summary = {key: value for key, value in snapshot.items() if key != "file_stats"}
    with _reference_lock(lease):
        if name not in lease.ledger.control.names():
            lease.ledger.control.write_once(name, _encoded(dict(
                format="darsia-first-q104-output-stop-v1", request=first.binding,
                provider=first.provider_binding, scope=first.provider["scope"], reason=reason,
                observation=summary, no_retry=True, no_ordinary_profile=True, partial_files_retained=True)))
    return request_existing_frame_boundary_stop(first, lease)


class BoundOutputObservations(FiniteObservations):
    """Only the existing BOUND branch adds a separate sampled output watch."""
    def __init__(self, first, *, clock=None):
        super().__init__(first, clock=clock)
        self.common = getattr(first, "common_mode", False)
        self.preset = getattr(first, "preset_mode", False)
        if self.preset:
            from .preset_qualification import PresetQualification, OUTPUT
            if (not isinstance(first, PresetQualification) or self.common or first.bound_mode
                    or first.output_entitlement.get("format") != OUTPUT):
                raise CachePolicyError("Preset output watch requires its exact separate first entitlement")
            self.entitlement_bytes = (first.output_entitlement["maximum_output_bytes"]
                                      + first.output_entitlement["maximum_transient_bytes"])
            self.output_scope = checked_bound_output_scope(first.output_entitlement["watched_output_scope"], preset=True)
            self.output_samples = []; self.last_output_start = None; self.final_output_snapshot = None
            self.output_stop = None; self.output_error = None
            return
        if self.common:
            from .common_qualification import OUTPUT
            if not first.bound_mode or first.output_entitlement.get("format") != OUTPUT:
                raise CachePolicyError("Common output watch requires its actual stage entitlement")
            self.entitlement_bytes = first.output_entitlement["maximum_output_bytes"]
            if type(self.entitlement_bytes) is not int or self.entitlement_bytes <= 0:
                raise CachePolicyError("Positive released common-stage output bound required")
            self.output_scope = checked_bound_output_scope(
                first.output_entitlement["watched_output_scope"], common=True)
            self.output_samples = []; self.last_output_start = None; self.final_output_snapshot = None
            self.output_stop = None; self.output_error = None
            self.retained_cache_before = None; self.retained_cache_after = None
            return
        if (not first.bound_mode or not isinstance(first.output_entitlement, dict)
                or first.output_entitlement.get("format") != BOUND_OUTPUT_ADMINISTRATION):
            raise CachePolicyError("Bound output watch requires its actual current output-scope plan")
        self.entitlement_bytes = first.output_entitlement["administrative_output_entitlement_bytes"]
        if type(self.entitlement_bytes) is not int or self.entitlement_bytes < FIRST_OUTPUT_BASE_ENTITLEMENT_BYTES:
            raise CachePolicyError("Bound output watch requires the actual predeclared Q119 entitlement")
        self.output_scope = checked_bound_output_scope(first.output_entitlement["watched_output_scope"])
        self.output_samples = []; self.last_output_start = None; self.final_output_snapshot = None
        self.output_stop = None; self.output_error = None

    def admit_output(self, lease):
        output, cache = PureWindowsPath(self.output_scope["path"]), PureWindowsPath(str(lease.policy.root))
        if output == cache or output in cache.parents or cache in output.parents:
            raise CachePolicyError("Q104 output scope must be disjoint from cache accounting")
        started = self.clock()
        options = {"preset": True} if self.preset else {"common": True} if self.common else {}
        current = bound_output_snapshot(self.output_scope, entitlement_bytes=self.entitlement_bytes, **options)
        if (current["logical_bytes"] != 0 or current["observed_scan_anomalies"]
                or current["volume"]["free_bytes"] - PHYSICAL_RESERVE_BYTES < self.entitlement_bytes):
            raise CachePolicyError("Fresh complete output scope and free minus original40GiB covering wholeQ104 required")
        current["started_monotonic_seconds"] = started
        current["elapsed_seconds"] = self.clock() - started
        self.output_admission = current
        if self.common:
            with lease.ledger.locked(create_binding=False):
                self.retained_cache_before = lease.measure(lease.policy.root,
                    include_file_stats=True, hash_files=True)

    def observe_output(self, owned, *, force=False):
        started = self.clock()
        if not force and self.last_output_start is not None and started - self.last_output_start < 1:
            return
        try:
            options = {"preset": True} if self.preset else {"common": True} if self.common else {}
            current = bound_output_snapshot(self.output_scope, entitlement_bytes=self.entitlement_bytes,
                                            include_file_stats=force, **options)
            if self.common and force:
                with owned.lease.ledger.locked(create_binding=False):
                    self.retained_cache_after = owned.lease.measure(owned.lease.policy.root,
                        include_file_stats=True, hash_files=True)
        except BaseException as error:
            import traceback
            self.output_error = dict(error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
            if self.output_stop is None:
                self.output_stop = dict(reason="OUTPUT_OBSERVATION_UNAVAILABLE", diagnostic=self.output_error)
                try:
                    self.output_stop["response"] = latch_bound_output_stop(self.first, owned.lease,
                        self.output_stop["reason"], self.output_error)
                except BaseException as stop_error:
                    self.output_stop["stop_record_error"] = dict(error_type=type(stop_error).__name__, message=str(stop_error))
                    error.add_note("Output STOP record/command failed: " + repr(stop_error))
            raise
        elapsed = self.clock() - started
        self.output_samples.append(dict(utc=current["utc"], started_monotonic_seconds=started,
            elapsed_seconds=elapsed, actual_gap_since_previous_start_seconds=(None if self.last_output_start is None
                else started - self.last_output_start), forced_terminal=force,
            actual_gap_since_admission_start_seconds=(started - self.output_admission["started_monotonic_seconds"]
                if hasattr(self, "output_admission") else None),
            logical_bytes=current["logical_bytes"], entries=current["entries"], free_bytes=current["volume"]["free_bytes"],
            observed_scan_anomalies=current["observed_scan_anomalies"], sampled=True, complete_lifetime_peak=False))
        self.last_output_start = started
        if force: self.final_output_snapshot = current
        reason = ("OUTPUT_ENTITLEMENT_EXCEEDED" if current["logical_bytes"] > self.entitlement_bytes else
            "OUTPUT_OPERATIONAL_RESERVE_CROSSED" if current["volume"]["free_bytes"] <= PHYSICAL_RESERVE_BYTES else None)
        if reason is not None and self.output_stop is None:
            self.output_stop = dict(reason=reason, sample_index=len(self.output_samples) - 1)
            self.output_stop["response"] = latch_bound_output_stop(self.first, owned.lease, reason, current)

    def observe(self, owned, accounting, *, force_cache=False):
        # Old Q106 cache/reserve observations retain their original code.
        try:
            super().observe(owned, accounting, force_cache=force_cache)
        except BaseException as error:
            # Loss of a real cache/reserve observation is not permission to run
            # without the watchdog. Use the same cooperative boundary command.
            if self.output_stop is None:
                self.output_stop = dict(reason="CACHE_OR_RESERVE_OBSERVATION_UNAVAILABLE",
                    error_type=type(error).__name__, message=str(error))
                try:
                    self.output_stop["response"] = latch_bound_output_stop(self.first, owned.lease,
                        self.output_stop["reason"], self.output_stop.copy())
                except BaseException as stop_error:
                    error.add_note("Observation STOP record/command failed: " + repr(stop_error))
            raise
        self.observe_output(owned, force=force_cache)

    def output_document(self, terminal, retirement, eligibility):
        final = self.final_output_snapshot
        kernel_retired = (isinstance(terminal, dict) and terminal.get("accounting", {}).get("active_processes") == 0)
        accepted_retirement = isinstance(eligibility, dict) and eligibility.get("original_retirement_verifier_accepted") is True
        closed = (kernel_retired and retirement is not None and accepted_retirement
            and final is not None and not final["observed_scan_anomalies"] and self.output_error is None)
        if self.common:
            closed = closed and self.retained_cache_before is not None and self.retained_cache_after is not None
        complete = (closed and self.output_stop is None and not self.reserve_crossed
            and terminal.get("wrapper_exit_code") == 0)
        result = dict(format=("darsia-preset-wasserstein-closed-output-diagnostic-v1" if self.preset else "darsia-common-phase-closed-output-diagnostic-v1" if self.common
                            else "darsia-first-q104-closed-output-diagnostic-v1"), request=self.first.binding,
            provider=self.first.provider_binding, stores_source=self.first.stores_source,
            scope=self.first.provider["scope"],
            output_plan=self.first.request["output_entitlement_plan"], output_scope=self.output_scope,
            entitlement_bytes=self.entitlement_bytes, admission=getattr(self, "output_admission", None),
            output_samples=self.output_samples, target_sample_period_seconds=1,
            sampling_scope="initial admission, native await-family observer and forced terminal stat; actual startup/blocked-scan gaps retained, not continuous coverage",
            sampled_highwater_bytes=max((row["logical_bytes"] for row in self.output_samples), default=None),
            sampled_highwater_is_lifetime_peak=False, final_snapshot=final, first_stop=self.output_stop,
            first_observation_error=self.output_error, actual_terminal=terminal, actual_retirement=retirement,
            original_retirement_accepted=accepted_retirement, closed_output_stat_observation=closed,
            closed_output_logical_bytes=(final["logical_bytes"] if closed else None),
            whole_output_entitlement_observed_exceeded=any(row["logical_bytes"] > self.entitlement_bytes for row in self.output_samples),
            cache_operational_reserve_crossed=self.reserve_crossed,
            output_hard_quota=False, file_payload_SHA_or_arrays_read=False, original_comparison=None,
            state=("CLOSED_DIAGNOSTIC_ONLY" if complete else "STOP_OR_INCOMPLETE_DIAGNOSTIC"),
            complete=complete, passed=False, no_result_space_profile_created=True, no_ordinary_profile_created=True)
        if self.common:
            result.update(retained_cache_before=self.retained_cache_before,
                retained_cache_after=self.retained_cache_after,
                retained_cache_payload_hashes_read=True,
                file_payload_SHA_or_arrays_read=True, result_payload_SHA_or_arrays_read=False,
                scientific_arrays_decoded=False,
                retained_cache_files_deleted_by_this_code=False,
                retained_cache_scope_is_exhaustive_OS_write_proof=False)
        return result


class BoundServiceReserveObservations:
    """Ordinary typed-bound service: original native observer/reserve/STOP seam.

    Only metadata is sampled. Original measured/direct services do not enter
    this branch; output entitlement is a first-qualification-only concern.
    """
    def __init__(self, lease, command, *, clock=None):
        from types import SimpleNamespace
        if lease.profile.get("format") not in {BOUND_IMPORT_PROFILE, "darsia-common-phase-import-profile-v1",
                                              "darsia-setup-service-import-profile-v1",
                                              "darsia-preset-wasserstein-bounded-profile-v1"}:
            raise CachePolicyError("The ordinary bound watchdog needs an actually resolved bound import profile")
        positions = [i for i, value in enumerate(command) if value == "--job"]
        if len(positions) != 1 or positions[0]+1 >= len(command):
            raise CachePolicyError("Actual original public job argv required for bound watchdog")
        job = PureWindowsPath(command[positions[0]+1])
        if not job.is_absolute() or ".." in job.parts:
            raise CachePolicyError("Actual ordinary job path required before native start")
        # This is the actual ordinary command's metadata view for the unchanged
        # reserve/command routine, not a first request or scientific executor.
        actual = SimpleNamespace(bound_mode=True, reference_build=False,
            required_capacity_bound=lease.profile["capacity_bound"],
            request=dict(cache_stat_period_seconds=30, public_command=list(command)),
            task=dict(job_id=job.name), provider=dict(scope=lease.profile["scope"]))
        if lease.profile.get("format") == "darsia-preset-wasserstein-bounded-profile-v1":
            from .adapters import _worker_metadata
            actual.job = _worker_metadata(str(job / "job.json"))[0]
            actual.preset_mode = True
            actual.task["job_id"] = actual.job["job_id"]
        self.collector = FiniteObservations(actual, clock=clock)
        self.lease, self.command = lease, list(command)
        self.token = uuid.uuid4().hex
        self.chunks = []; self.flushed = 0; self.error = None; self.closed = None

    def admit(self):
        from .capacity import volume_info
        if volume_info(self.lease.policy.root)["free_bytes"] <= PHYSICAL_RESERVE_BYTES:
            raise CachePolicyError("Actual ordinary bound service has no original40GiB reserve before native start")

    def _flush(self, owned):
        rows = self.collector.reserve_samples[self.flushed:]
        if not rows: return
        # Finite metadata chunks stay below the original control-document size.
        # No sampling gap is dropped, interpolated or called lifetime coverage.
        pin = self.lease._record("native-bound-reserve-" + self.token + "-" + str(len(self.chunks)) + ".json",
            dict(format="darsia-native-bound-reserve-samples-v1", profile=self.lease.profile_binding,
                actual_command=self.command, native_launch=owned.observed_binding,
                first_sample_index=self.flushed, samples=deepcopy(rows),
                target_period_seconds=1, samples_prove_no_crossing_between_samples=False))
        self.chunks.append(pin); self.flushed += len(rows)

    def observe(self, owned, accounting, *, force=False):
        try:
            self.collector.observe_reserve(owned, self.collector.clock(), force=force)
            if len(self.collector.reserve_samples)-self.flushed >= 128:
                self._flush(owned)
        except BaseException as error:
            import traceback
            if self.error is None:
                self.error = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
                try:
                    self.collector.stop_response = request_existing_frame_boundary_stop(self.collector.first, owned.lease)
                except BaseException as stop_error:
                    error.add_note("Ordinary bound watchdog STOP failed: " + repr(stop_error))
            raise

    def finish(self, owned, terminal, retirement):
        self._flush(owned)
        self.closed = self.lease._record("native-bound-reserve-" + self.token + "-closed.json",
            dict(format="darsia-native-bound-operational-reserve-closure-v1", profile=self.lease.profile_binding,
                actual_command=self.command, native_launch=owned.observed_binding,
                actual_terminal=terminal, actual_retirement=retirement, chunks=self.chunks,
                sample_count=len(self.collector.reserve_samples), target_period_seconds=1,
                operational_reserve_bytes=PHYSICAL_RESERVE_BYTES,
                operational_reserve_crossed=self.collector.reserve_crossed,
                stop_response=self.collector.stop_response, first_observation_error=self.error,
                samples_prove_no_crossing_between_samples=False, lifetime_peak_proven=False,
                complete=terminal is not None and retirement is not None and self.error is None))
        return self.closed

    def require_no_stop(self):
        if self.collector.reserve_crossed or self.error is not None:
            raise CachePolicyError("Ordinary bound operational reserve STOP; no recovered PASS or automatic finalization")


def _close_common_stage_after_retirement(first, lease, native_completion):
    """The existing controller's final common seam accepts metadata only after close."""
    if (not getattr(first, "common_mode", False) or native_completion.get("passed") is not True
            or native_completion.get("observer_native_handles_closed") is not True):
        raise CachePolicyError("Common closure requires original accepted retirement and closed native handles")
    from .common_qualification import close_stage
    try:
        return close_stage(first, lease)
    except BaseException as error:
        held = dict(request=first.binding, provider=first.provider_binding,
            error_type=type(error).__name__, message=str(error), no_retry=False,
            retry_action="metadata-close-only", scientific_worker_rerun_allowed=False,
            native_completion_retained=True, reservation_closed=False)
        try:
            lease._record("first-held-" + uuid.uuid4().hex + ".json", held)
        except BaseException as diagnostic:
            error.add_note("Retryable metadata-close diagnostic was not recorded: " + repr(diagnostic))
        raise


def retry_common_stage_close(first, lease):
    """Retry a retired common stage's metadata only; never consume or launch again."""
    if not getattr(first, "common_mode", False):
        raise CachePolicyError("Metadata-close recovery requires the exact released common request")
    from .common_qualification import _slot_read as common_slot_read
    native, _pin = common_slot_read(first.provider, "native_completion")
    return _close_common_stage_after_retirement(first, lease, native)


def _common_native_completion_present(first):
    """Test the bound native slot name before opening it; absence is not a read error."""
    from .windows import WindowsNamespace
    from .transient import retry_transient
    target = first.provider["output_namespace"]
    namespace = WindowsNamespace(target["path"], target["identity"])
    try:
        present = {name.casefold() for name in retry_transient(namespace.names)}
        return first.provider["slots"]["native_completion"].casefold() in present
    finally:
        namespace.close()


def supervise_first_worker(lease, *, api=None):
    """Same exact native launch and natural retirement; never a numerical path."""
    from .owned_process import OwnedWorker, _public_command
    first = lease.first_qualification
    validate_startup(lease)
    command = _public_command(lease)
    if command != first.request["public_command"]:
        raise CachePolicyError("Actual normal public command differs from the finite released argv")
    if getattr(first, "common_mode", False) and _common_native_completion_present(first):
        retry_common_stage_close(first, lease)
        return 0
    first.consume(lease)
    owned = OwnedWorker(lease, api=api)
    preset = getattr(first, "preset_mode", False)
    watches_output = first.bound_mode or preset
    observations = (BoundOutputObservations(first) if watches_output else FiniteObservations(first))
    native_completion=None
    setup_failure=None
    terminal=retirement=observation_pin=output_measurement_pin=None
    try:
        if watches_output:
            observations.admit_output(lease)
        launch_started = time.monotonic()
        owned.start(command, dict(os.environ), Path.cwd())
        terminal = owned.await_family(observer=observations.observe)
        observations.observe(owned,terminal["accounting"],force_cache=True)
        retirement = owned.record_closed_family(terminal)
        from .service_context import resolve_original_service_context
        from .service_lifecycle import ServiceLifecycle
        from .adapters import verify_worker_publication
        from .bootstrap import _utc
        context=resolve_original_service_context(owned.service_binding,package_file=lease.package_file,
            first_request=first.binding)
        setup_retirement = {}
        def verify_original_retirement(snapshot, binding):
            accepted = owned.verify_retirement(snapshot, binding)
            if accepted and getattr(first, "setup_job_mode", False):
                setup_retirement.update(snapshot=snapshot, binding=binding)
            return accepted
        eligibility=ServiceLifecycle(context).retention_eligibility(now_utc=_utc(),
            retirement=retirement,verify_retirement=verify_original_retirement,
            verify_publication=verify_worker_publication)
        launch_to_retirement_seconds = time.monotonic() - launch_started
        from .owned_process import _exact_source
        from .bootstrap import _exact
        for pin in first.provider["module_closure"]:
            _exact_source(pin)
        startup=_exact(first.provider["startup"],lease.package_file)
        for pin in startup["source_files"]:
            _exact_source(pin)
        observation = observations.document(owned, terminal, retirement, eligibility)
        from .owned_process import _control
        lifecycle=ServiceLifecycle(context)
        with context.ledger.locked(create_binding=False):
            records=lifecycle._records()
        inventories=[];tasks=[];modules_exact=True
        for (kind,key),registered in sorted(records.items()):
            if kind!="owner":continue
            owner=registered["document"]["value"]["owner"]
            exit_name=(context.reservation["token"]+".exit-intent-"+str(owner["pid"])
                +"-"+owner["creation_filetime_100ns"]+".json")
            exited,exit_pin=_control(lease,exit_name)
            modules=exited.get("first_qualification_modules",{})
            if (exited.get("owner")!=owner or modules.get("request")!=first.binding
                    or modules.get("provider")!=first.provider_binding
                    or modules.get("predeclared_module_closure_sha256")!=_digest(first.provider["module_closure"])):
                raise CachePolicyError("Actual registered owner's exit inventory differs from first request")
            modules_exact=modules_exact and modules.get("all_file_backed_modules_predeclared_and_exact") is True
            if getattr(first, "common_mode", False):
                if (modules.get("common_device_identity") != first.root_provider["device_identity"]
                        or modules.get("actual_device_environment") != first.device["device_environment"]):
                    raise CachePolicyError("Actual common owner device/runtime environment differs from its predeclared CPU scope")
            if modules.get("actual_task") is not None:tasks.append(modules["actual_task"])
            inventories.append(dict(owner=owner,registration=registered["binding"],inventory=exit_pin,
                all_file_backed_modules_predeclared_and_exact=modules.get("all_file_backed_modules_predeclared_and_exact"),
                undeclared=modules.get("undeclared")))
            if getattr(first, "setup_job_mode", False):
                inventories[-1]["actual_task"] = modules.get("actual_task")
            if watches_output:
                inventories[-1]["actual_writer_environment"] = modules.get("actual_writer_environment")
                inventories[-1]["writer_environment_is_actual_IO_or_lifetime_proof"] = modules.get(
                    "writer_environment_is_actual_IO_or_lifetime_proof")
                if getattr(first, "common_mode", False):
                    inventories[-1]["common_device_identity"] = modules["common_device_identity"]
                    inventories[-1]["actual_device_environment"] = modules["actual_device_environment"]
                    inventories[-1]["common_worker_scientific_evidence"] = modules.get("common_worker_scientific_evidence")
        if getattr(first, "setup_job_mode", False):
            if not tasks:
                raise CachePolicyError("At least one actual Setup worker claim is required")
            for task in tasks:
                first.require_task(lease, task)
            observation["actual_claims"] = tasks
        else:
            if len(tasks)!=1:
                raise CachePolicyError("Exactly one actual worker-slot first claim is required")
            first.require_task(lease,tasks[0])
            observation["actual_claim"]=tasks[0]
        if getattr(first, "common_mode", False):
            observation["native_launch_to_retirement_seconds"] = launch_to_retirement_seconds
        observation["actual_registered_owner_exit_inventories"]=inventories
        observation["source_runtime_before_after_exact"]=True
        if watches_output:
            observation["profile_publication_state"] = (
                "PRESET_EXECUTION_OBSERVATION_NOT_A_MEASURED_PROFILE" if preset else "COMMON_STAGE_OBSERVATION_AWAITING_COMPLETE_P1_C_P2_DAG"
                if getattr(first, "common_mode", False) else "BOUND_OBSERVATION_AWAITING_ORIGINAL_COMPARISON")
            output_measurement_pin = _slot_write(first.provider, "measurement",
                observations.output_document(terminal, retirement, eligibility))
            observation["output_measurement"] = output_measurement_pin
        observation_pin=_slot_write(first.provider, "observation", observation)
        if first.bound_mode:
            # A separate record, never translated into old measured-v1 coverage.
            # It remains unqualified until independent output/comparison evidence
            # and the checked completed DAG exist. Q117 permits that distinct DAG.
            if getattr(first, "common_mode", False):
                from .common_qualification import bounded_evidence as common_evidence
                writer_evidence = common_evidence(first, context, observation,
                    service_receipt=_exact(context.receipt, lease.package_file))
            else:
                writer_evidence = bounded_evidence(first, context, observation,
                    service_receipt=_exact(context.receipt, lease.package_file))
            _slot_write(first.provider, "writer_evidence", writer_evidence)
        seal=_exact(eligibility["seal"],lease.package_file)
        setup_job_complete = True
        if getattr(first, "setup_job_mode", False):
            setup_job_complete = False
            if seal["value"]["outcome"] == "success":
                published = _exact(seal["value"]["publication"], lease.package_file)
                setup_job_complete = first.whole_job_published(published)
            elif seal["value"]["outcome"] == "failed":
                setup_failure = dict(seal=eligibility["seal"], traceback=seal["value"]["traceback"])
        native_completion=dict(format="darsia-native-worker-first-qualification-completion-v1",
            request=first.binding,provider=first.provider_binding,stores_source=lease.source,
            scope=first.provider["scope"],startup=first.provider["startup"],
            module_closure=first.provider["module_closure"],observation=observation_pin,
            retirement=retirement,eligibility=eligibility,
            maximum_family_processes=observation["conservative_family_process_bound_from_kernel_total"],
            family_bound_semantics="actual kernel lifetime TotalProcesses; conservative concurrency bound, not sampled peak",
            sampled_concurrent_high_water=observation["sampled_active_processes_high_water"],
            complete=True,passed=(terminal["wrapper_exit_code"]==0
                and eligibility.get("original_retirement_verifier_accepted") is True
                and seal.get("value",{}).get("outcome")=="success"
                 and modules_exact and bool(inventories) and setup_job_complete and not observations.reserve_crossed
                  and not (watches_output and observations.output_stop is not None)
                 and not (first.reference_build and reference_stopped(first, lease))),
            cache_profile_qualified=False)
        if (getattr(first, "setup_job_mode", False)
                and eligibility.get("original_retirement_verifier_accepted") is True):
            if setup_failure is not None:
                native_completion["service_failure"] = setup_failure
            # Whole-family handles are still held. Release only the unwritten
            # reservation; every retained cache file remains counted. No cleanup.
            with lease.ledger.locked():
                lease.ledger.close(context.reservation, owner=context.manifest["owner"],
                    verify_no_further_writes=lambda row: (
                        row["owner"] == context.manifest["owner"]
                        and bool(setup_retirement)
                        and owned.verify_retirement(setup_retirement["snapshot"], setup_retirement["binding"])))
            native_completion["cache_reservation_closed"] = True
        # Deliberately no guessed measured profile, no automatic scope adoption,
        # reservation close or cleanup. Original actual cache remains charged.
    except BaseException as error:
        if setup_failure is not None:
            error.add_note("Original sealed Setup service failure:\n" + setup_failure["traceback"])
        if first.reference_build and observation_pin is None:
            # Preserve genuine samples even if cooperative STOP prevents the
            # ordinary whole-run publication/eligibility or a final scan fails.
            try:
                observation_pin=_slot_write(first.provider,"observation",dict(format=OBSERVATION,
                    request=first.binding,provider=first.provider_binding,scope=first.provider["scope"],
                    state="STOP_PARTIAL_DIAGNOSTIC",complete=False,passed=False,
                    error_type=type(error).__name__,message=str(error),actual_terminal=terminal,
                    actual_retirement=retirement,reference_aggregate_samples=observations.aggregate_samples,
                    final_reference_aggregate=observations.last_aggregate,
                    reserve_samples=observations.reserve_samples,operational_reserve_crossed=observations.reserve_crossed,
                    sampled_observed_highwater_is_complete=False,partial_files_retained=True,
                    no_ordinary_profile_created=True,no_reference_acceptance=True))
            except BaseException as diagnostic_error:
                error.add_note("Partial observation was not recorded; full samples are not durable: "
                    + type(diagnostic_error).__name__ + ": " + str(diagnostic_error))
        elif watches_output and observation_pin is None:
            # Ordinary eligibility/seal can fail after a real reserve/output
            # STOP. Preserve actual samples rather than requiring success to log
            # the failure. This is never partial acceptance or a reusable profile.
            try:
                import traceback
                original_traceback = traceback.format_exc()
                if output_measurement_pin is None:
                    try:
                        output_measurement_pin = _slot_write(first.provider, "measurement",
                            observations.output_document(terminal, retirement, None))
                    except BaseException as diagnostic_error:
                        error.add_note("Output diagnostic slot was not completed: " + repr(diagnostic_error))
                observation_pin = _slot_write(first.provider, "observation", dict(format=OBSERVATION,
                    request=first.binding, provider=first.provider_binding, scope=first.provider["scope"],
                    state="STOP_PARTIAL_DIAGNOSTIC", complete=False, passed=False,
                    error_type=type(error).__name__, message=str(error), traceback=original_traceback,
                    output_measurement=output_measurement_pin,
                    output_diagnostic_if_measurement_unwritten=(observations.output_document(terminal, retirement, None)
                        if output_measurement_pin is None else None),
                    actual_terminal=terminal, actual_retirement=retirement,
                    cache_samples=observations.cache_samples, reserve_samples=observations.reserve_samples,
                    sampled_cache_logical_high_water_bytes=observations.sampled_cache_logical_high_water_bytes,
                    sampled_observed_highwater_is_complete=False,
                    operational_reserve_crossed=observations.reserve_crossed,
                    operational_stop_response=observations.stop_response,
                    partial_files_retained=True, no_ordinary_profile_created=True,
                    no_reference_acceptance=True))
            except BaseException as diagnostic_error:
                error.add_note("Bound partial observation was not recorded; full samples are not durable: "
                    + type(diagnostic_error).__name__ + ": " + str(diagnostic_error))
        held=dict(request=first.binding,provider=first.provider_binding,error_type=type(error).__name__,
            message=str(error),no_retry=True,reservation_closed=False)
        if first.reference_build or watches_output:
            held["actual_partial_observation"]=observation_pin
        try:
            lease._record("first-held-"+uuid.uuid4().hex+".json",held)
        except BaseException as diagnostic_error:
            error.add_note("First-held diagnostic was not recorded: "
                + type(diagnostic_error).__name__ + ": " + str(diagnostic_error))
        if owned.resume_attempted and owned.info is not None:
            owned.await_family()
        raise
    finally:
        owned.close()
    native_completion["observer_native_handles_closed"]=True
    if first.reference_build:
        native_completion["producer_complete_unqualified"] = native_completion["passed"]
        native_completion["passed"] = False
        native_completion["state"] = "UNQUALIFIED_REFERENCE_PRODUCER_COMPLETE"
        native_completion["root_reference_acceptance_required"] = True
    native_pin = _slot_write(first.provider,"native_completion",native_completion)
    if getattr(first, "setup_job_mode", False):
        if not native_completion["passed"]:
            if "service_failure" in native_completion:
                raise CachePolicyError("Setup service failed; actual partial evidence is retained\n"
                    + native_completion["service_failure"]["traceback"])
            raise CachePolicyError("Setup service stopped; actual partial evidence is retained")
        _slot_write(first.provider, "completion", dict(
            format="darsia-setup-job-service-completion-v1", request=first.binding,
            provider=first.provider_binding, native_completion=native_pin,
            complete=True, passed=True, measured_profile=False, cache_profile_qualified=False,
            cache_files_retained=True, cache_reservation_closed=True))
        return 0
    if preset:
        if not native_completion["passed"]:
            raise CachePolicyError("Preset first execution stopped; original evidence retained")
        from .preset_qualification import close_operation
        from .bootstrap import _exact
        close_operation(first, native_pin, read=lambda pin: _exact(pin, lease.package_file))
        return 0
    if getattr(first, "setup_mode", False):
        if not native_completion["passed"]:
            raise CachePolicyError("Setup service qualification stopped; original evidence retained")
        from .setup_qualification import close_operation
        from .bootstrap import _exact
        close_operation(first, native_pin, read=lambda pin: _exact(pin, lease.package_file))
        return 0
    if getattr(first, "common_mode", False):
        if not native_completion["passed"]:
            raise CachePolicyError("Common stage stopped; partial evidence retained, no ordinary profile")
        _close_common_stage_after_retirement(first, lease, native_completion)
        return 0  # Original controller retires this real stage, not the whole common DAG.
    if first.reference_build:
        # Durable observations/retirement are genuine; this is deliberately not
        # a passing profile, self-comparison, accepted reference or adoption.
        if not native_completion["producer_complete_unqualified"]:
            raise CachePolicyError("N43 reference producer STOP; partial products retained, no reference promotion")
        return 0  # Original external controller/SystemExit contract; not a profile PASS.
    if first.bound_mode:
        if not native_completion["passed"]:
            raise CachePolicyError("Bound qualification producer STOP; diagnostics retained, no profiles")
        return 0  # Real producer OS0 only; independent original comparison still required.
    # A successful scientific job is not an accepted cache profile. Make the
    # missing qualification explicit after durable observations and close.
    raise CachePolicyError("First observations complete; cache profile publication refused: no qualified lifetime writer coverage")


def checked_bounded_evidence(observation, value, *, first=None):
    """Read the Q106 physical bound honestly; this does not admit a profile."""
    q123 = getattr(first, "q123_cold_scope", None)
    q123_fields = {"q123_cold_scope"} if q123 is not None else set()
    _exact_keys(value, {"format", "request", "provider", "stores_source", "scope",
        "namespace_admission", "capacity_bound_plan", "capacity_bound", "service_namespace",
        "administrative_service_entitlement", "actual_service_reservation", "actual_service_reservation_document",
        "actual_volume_observation",
        "registered_owner_environments", "observation_sha256",
        "source_runtime_before_after_exact", "retirement", "actual_writer_IO_proven",
        "writer_lifetime_cache_bytes_proof", "sampled_highwater_is_lifetime_peak",
        "output_evidence", "original_comparison", "ordinary_use_authorization",
        "profile_publication_state", "no_ordinary_profile_created"} | q123_fields,
        "Exact bounded evidence required; old measured coverage is not interchangeable")
    external_mpl = None
    if q123 is not None:
        if value["q123_cold_scope"] != first.request["q123_cold_scope"] or value["q123_cold_scope"] != q123.binding:
            raise CachePolicyError("Q123 external writer role differs from the released request")
        external_mpl = q123.matplotlib_root(first)
    if (value["format"] != BOUNDED_EVIDENCE
            or observation.get("format") != OBSERVATION or observation.get("complete") is not True
            or any(value[key] != observation.get(key)
                for key in ("request", "provider", "stores_source", "scope", "retirement"))
            or value["observation_sha256"] != _digest(observation)
            or value["source_runtime_before_after_exact"] is not True
            or observation.get("source_runtime_before_after_exact") is not True
            or observation.get("actual_public_worker_exit_code") != 0
            or observation.get("operational_reserve_crossed") is not False
            or observation.get("sampled_cache_logical_high_water_is_lifetime_peak") is not False
            or observation.get("reserve_samples_prove_no_crossing_between_samples") is not False
            or value["actual_writer_IO_proven"] is not False
            or value["writer_lifetime_cache_bytes_proof"] is not None
            or value["sampled_highwater_is_lifetime_peak"] is not False
            or value["output_evidence"] is not None or value["original_comparison"] is not None
            or value["ordinary_use_authorization"] is not None
            or value["profile_publication_state"] != "UNQUALIFIED_BOUND_OBSERVATION"
            or value["no_ordinary_profile_created"] is not True):
        raise CachePolicyError("Bounded evidence cannot assert writer lifetime, IO, missing comparison or ordinary PASS")
    _descriptor(value["namespace_admission"])
    _descriptor(value["capacity_bound_plan"])
    checked_required_capacity_bound(value["capacity_bound"])
    if getattr(first, "common_mode", False) is True:
        required = first.required_capacity_bound
        expected = required["volume_cap_observation"]
        maximum = expected.get("binding", {}).get("maximum_size_bytes")
        if (value["capacity_bound"] != required or value["actual_volume_observation"] != expected
                or type(maximum) is not int or maximum <= 0
                or expected.get("virtual_size_bytes") != maximum):
            raise CachePolicyError("Fresh actual common volume observation differs from its pinned bound")
    elif (value["actual_volume_observation"] != value["capacity_bound"]["volume_cap_observation"]
            or value["actual_volume_observation"].get("virtual_size_bytes") != 500 * 1024**3):
        raise CachePolicyError("Fresh actual fixed500GiB volume observation differs from its pinned bound")
    root = PureWindowsPath(value["service_namespace"])
    if not root.is_absolute() or not root.drive or root.drive.startswith("\\\\") or ".." in root.parts:
        raise CachePolicyError("Actual owned local service namespace required")
    administrative = value["administrative_service_entitlement"]
    if (not isinstance(administrative, dict) or administrative.get("measured_profile") is not False
            or type(administrative.get("maximum_bytes")) is not int
            or administrative["maximum_bytes"] <= 0
            or administrative["maximum_bytes"] > value["capacity_bound"]["operational_admission_headroom_bytes"]):
        raise CachePolicyError("Actual operational service reservation is separate from the physical volume bound")
    reservation = _exact_keys(value["actual_service_reservation"], {"token", "pool", "sha256"},
        "Original actual administrative reservation required")
    reserved = value["actual_service_reservation_document"]
    if (not isinstance(reserved, dict) or reserved.get("format") != "darsia-cache-budget-reservation-v2"
            or reserved.get("maximum_bytes") != administrative["maximum_bytes"]
            or reserved.get("token") != reservation["token"] or reserved.get("pool") != reservation["pool"]
            or _digest(reserved) != reservation["sha256"]):
        raise CachePolicyError("Actual reservation document/hash/administrative maximum differs")
    from .environment import FIRST_WRITER_ENVIRONMENT_KEYS
    rows = value["registered_owner_environments"]
    actual = observation.get("actual_registered_owner_exit_inventories")
    if not isinstance(rows, list) or not rows or rows != actual:
        raise CachePolicyError("Every actual registered owner exit inventory is required")
    owners = set()
    for row in rows:
        key = _digest(row["owner"])
        if (key in owners or row.get("all_file_backed_modules_predeclared_and_exact") is not True
                or row.get("undeclared") != []
                or row.get("writer_environment_is_actual_IO_or_lifetime_proof") is not False):
            raise CachePolicyError("Missing, duplicate or unknown actual owner/module evidence")
        owners.add(key)
        _descriptor(row["registration"]); _descriptor(row["inventory"])
        environment = _exact_keys(row.get("actual_writer_environment"), FIRST_WRITER_ENVIRONMENT_KEYS,
            "Every actual exiting owner's writer environment is required")
        for name, path in environment.items():
            if name == "NUMBA_CACHE_LOCATOR_CLASSES":
                if path != "UserProvidedCacheLocator":
                    raise CachePolicyError("Actual Numba locator differs")
            elif name == "MPLCONFIGDIR" and external_mpl is not None:
                if path != external_mpl:
                    raise CachePolicyError("Q123 Matplotlib path differs from its exact counted namespace")
            elif (not isinstance(path, str) or not PureWindowsPath(path).is_absolute()
                    or ".." in PureWindowsPath(path).parts
                    or (PureWindowsPath(path) != root and root not in PureWindowsPath(path).parents)):
                raise CachePolicyError("Actual owner writer environment escapes the owned service: " + name)
    if _digest(reserved.get("owner")) not in owners:
        raise CachePolicyError("Administrative reservation owner is absent from actual exit inventories")
    samples = observation.get("reserve_samples")
    if (observation.get("reserve_watchdog_period_seconds") != 1
            or not isinstance(samples, list) or not samples
            or observation.get("sample_count") != len(observation.get("cache_samples", []))):
        raise CachePolicyError("Actual reserve/cache samples and gaps required")
    previous = None
    for sample in samples:
        instant = sample.get("monotonic_seconds")
        if (type(instant) not in {int, float} or not math.isfinite(instant)
                or not isinstance(sample.get("utc"), str) or not sample["utc"]
                or sample.get("operational_reserve_bytes") != PHYSICAL_RESERVE_BYTES
                or sample.get("at_or_below_operational_reserve") is not False
                or nonnegative(sample["free_volume_observation"]["free_bytes"], "Actual sampled free bytes") <= PHYSICAL_RESERVE_BYTES
                or (previous is None and sample.get("actual_gap_since_previous_start_seconds") is not None)
                or (previous is not None and (instant < previous
                    or sample.get("actual_gap_since_previous_start_seconds") != instant - previous))):
            raise CachePolicyError("Reserve crossing or inconsistent actual sample timeline")
        previous = instant
    previous = None; highwater = 0
    for sample in observation["cache_samples"]:
        instant, elapsed = sample.get("started_monotonic_seconds"), sample.get("elapsed_seconds")
        if (type(instant) not in {int, float} or not math.isfinite(instant)
                or type(elapsed) not in {int, float} or not math.isfinite(elapsed) or elapsed < 0
                or (previous is None and sample.get("actual_gap_since_previous_start_seconds") is not None)
                or (previous is not None and (instant < previous
                    or sample.get("actual_gap_since_previous_start_seconds") != instant - previous))):
            raise CachePolicyError("Cache samples must retain actual elapsed times and gaps")
        highwater = max(highwater, nonnegative(sample["observation"]["logical_bytes"], "Actual sampled cache bytes"))
        previous = instant
    if (not observation["cache_samples"] or observation.get("final_sample") != observation["cache_samples"][-1]["observation"]
            or highwater != observation.get("sampled_cache_logical_high_water_bytes")):
        raise CachePolicyError("Sampled highwater must equal its retained observations, never a lifetime estimate")
    return deepcopy(value)


def bounded_evidence(first, context, observation, *, service_receipt):
    """Produce observations only from the existing actual native/exit records."""
    if not first.bound_mode:
        raise CachePolicyError("Bounded evidence belongs only to the existing bound-request branch")
    with context.ledger.locked(create_binding=False):
        # This is an existing metadata-only observer ledger. project() would
        # intentionally invoke its refusing no_cache_measurement callback.
        actual = deepcopy(context.ledger.volume_cap_observation)
        reserved, _ = context.ledger.require_reservation(context.reservation, context.manifest["owner"], context.maximum)
    value = dict(format=BOUNDED_EVIDENCE, request=first.binding, provider=first.provider_binding,
        stores_source=first.stores_source, scope=first.provider["scope"],
        namespace_admission=first.provider["namespace_admission"],
        capacity_bound_plan=first.request["capacity_bound_plan"],
        capacity_bound=first.required_capacity_bound, service_namespace=str(context.root),
        administrative_service_entitlement=service_receipt["first_qualification"]["administrative_entitlement"],
        actual_service_reservation=context.reservation,
        actual_service_reservation_document=reserved,
        actual_volume_observation=actual,
        registered_owner_environments=observation["actual_registered_owner_exit_inventories"],
        observation_sha256=_digest(observation), source_runtime_before_after_exact=True,
        retirement=observation["retirement"], actual_writer_IO_proven=False,
        writer_lifetime_cache_bytes_proof=None, sampled_highwater_is_lifetime_peak=False,
        output_evidence=None, original_comparison=None, ordinary_use_authorization=None,
        profile_publication_state="UNQUALIFIED_BOUND_OBSERVATION", no_ordinary_profile_created=True)
    if getattr(first, "q123_cold_scope", None) is not None:
        first.q123_cold_scope.require_service(first, context)
        value["q123_cold_scope"] = first.request["q123_cold_scope"]
    return checked_bounded_evidence(observation, value, first=first)


def require_publishable(observation, writer_evidence):
    """No polling/redirect/source-audit assertion can mint a lifetime maximum."""
    if (observation.get("format") != OBSERVATION or observation.get("complete") is not True
            or writer_evidence is None):
        raise CachePolicyError("Actual completed observation and all-writer lifetime evidence required")
    if writer_evidence.get("format") == BOUNDED_EVIDENCE:
        checked_bounded_evidence(observation, writer_evidence)
        raise CachePolicyError("Bounded observation alone is not a profile; the checked native/output/original-comparison fixed-slot DAG is required")
    # The existing four external runtime writers do not expose an exact
    # lifetime-allocation proof seam. No caller-supplied boolean/number is a
    # qualified provider. Implement that audited source seam before accepting
    # any new evidence format or publishing first measured profiles.
    raise CachePolicyError("No qualified all-writer lifetime allocation evidence provider is implemented")


def resolve_profile(binding, kind, read, *, stores_source=None, read_authorities=None):
    """Old direct profiles unchanged; incomplete new providers always refuse.

    The Q117 bound DAG is distinct from old lifetime-writer metadata. All actual
    native/output/original-comparison roles are checked before applicability.
    """
    value = read(binding)
    if value.get("format") == "darsia-preset-wasserstein-first-provider-v1":
        from .preset_qualification import resolve_bounded_profile
        return resolve_bounded_profile(binding, value, kind, read, stores_source=stores_source)
    if value.get("format") == "darsia-preset-wasserstein-bounded-profile-v1":
        raise CachePolicyError("Preset bounded admission must resolve through its actual retired provider DAG")
    if value.get("format") == "darsia-setup-service-provider-v1":
        from .setup_qualification import resolve_profile as setup_profile
        return setup_profile(binding, value, kind, read, stores_source=stores_source,
                             read_authorities=read_authorities)
    if value.get("format") == "darsia-setup-service-import-profile-v1":
        raise CachePolicyError("Setup import profiles must resolve through their complete original provider")
    if value.get("format") == "darsia-common-phase-qualified-provider-v1":
        from .common_qualification import resolve_profile as common_profile
        return common_profile(binding, value, kind, read, stores_source=stores_source,
                              read_authorities=read_authorities)
    if value.get("format") in {"darsia-common-phase-import-profile-v1", "darsia-common-phase-job-profile-v1"}:
        raise CachePolicyError("Common profiles must resolve through their complete original provider DAG")
    if value.get("format") in {BOUND_IMPORT_PROFILE, BOUND_JOB_PROFILE}:
        raise CachePolicyError("Typed bound profiles must resolve through the original checked provider DAG")
    if value.get("format") != PROVIDER:
        return value
    provider = checked_provider(value)
    if kind not in {"import","jobs"}:
        raise CachePolicyError("Unknown typed provider profile role")
    bundle, bundle_pin = _slot_read(provider, "completion")
    if bundle.get("format") == BOUND_BUNDLE:
        return _resolve_bound_profile(binding, provider, bundle, kind, read,
            stores_source=stores_source, read_authorities=read_authorities)
    _exact_keys(bundle, {"format", "request", "provider", "stores_source", "scope",
        "observation", "writer_evidence", "profiles", "retirement"}, "Exact completed provider bundle required")
    if (bundle["format"] != BUNDLE or bundle["provider"] != binding
            or bundle["scope"] != provider["scope"]
            or (stores_source is not None and bundle["stores_source"] != stores_source)):
        raise CachePolicyError("Completed provider differs from actual stores/source/cases")
    first = FirstQualification(bundle["request"], provider_binding=binding,
        stores_source=bundle["stores_source"], namespace=provider["namespace_admission"], read=read,
        read_authorities=read_authorities)
    observation, pin = _slot_read(provider, "observation")
    evidence, evidence_pin = _slot_read(provider, "writer_evidence")
    if (bundle["observation"] != pin or bundle["writer_evidence"] != evidence_pin
            or observation.get("request") != first.binding
            or observation.get("provider") != binding
            or observation.get("stores_source") != bundle["stores_source"]
            or observation.get("scope") != provider["scope"]
            or observation.get("retirement") != bundle["retirement"]
            or observation.get("source_runtime_before_after_exact") is not True):
        raise CachePolicyError("Provider actual producer/observation/retirement binding differs")
    _exact_keys(bundle["profiles"],{"import","jobs"},"Typed import/jobs profile roles required")
    selected,selected_pin=_slot_read(provider,"import_profile" if kind=="import" else "job_profile")
    if selected_pin!=bundle["profiles"][kind]:
        raise CachePolicyError("Completed provider profile differs from its fixed immutable slot")
    native,native_pin=_slot_read(provider,"native_completion")
    if (native.get("request")!=first.binding or native.get("provider")!=binding
            or native.get("stores_source")!=bundle["stores_source"]
            or native.get("scope")!=provider["scope"] or native.get("startup")!=provider["startup"]
            or native.get("module_closure")!=provider["module_closure"]
            or native.get("observation")!=pin or native.get("retirement")!=bundle["retirement"]
            or native.get("complete") is not True or native.get("passed") is not True
            or native.get("observer_native_handles_closed") is not True):
        raise CachePolicyError("Actual completed native/module/retirement evidence differs")
    if kind=="import":
        block=selected.get("native_supervision",{})
        if (selected.get("format")!="darsia-cache-import-profile-v1"
                or selected.get("python_executable")!=provider["python_executable"]
                or selected.get("module_closure")!=provider["module_closure"]
                or block.get("startup")!=provider["startup"] or block.get("completion")!=native_pin
                or block.get("maximum_family_processes")!=native.get("maximum_family_processes")):
            raise CachePolicyError("Import/native profile differs from actual producer evidence")
    elif (selected.get("format")!="darsia-cache-measured-profile-v1"
            or selected.get("commit")!=provider["scope"]["commit"]
            or selected.get("pipeline")!=provider["scope"]["pipeline"]
            or selected.get("precision")!=provider["scope"]["precision"]
            or selected.get("case_count")!=len(provider["scope"]["ordered_case_ids"])):
        raise CachePolicyError("Job profile differs from original case/stage/source scope")
    require_publishable(observation, evidence)
    return selected  # Today unreachable: unproved writer evidence always refuses.


def _checked_bound_comparison_contract(task, reference, contract, *, read):
    """The exact reviewed adapter/source pair is bound BEFORE the single Run."""
    _exact_keys(contract, {"format", "scope", "reference_commit", "candidate_commit",
        "reference_completed", "sources", "source_pair_compatibility", "adapter_source_review",
        "public_command", "report_format", "comparison_kind"}, "Exact reviewed current/current comparison contract required")
    if (contract["format"] != "darsia-first-volume-bound-current-comparison-contract-v1"
            or contract["scope"] != {key:task.get(key) for key in (
                "commit", "pipeline", "backend", "precision", "run", "ordered_case_ids")}
            or contract["reference_commit"] != reference["reference_commit"]
            or contract["candidate_commit"] != task["commit"]
            or contract["reference_completed"] != reference["reference_completion"]
            or contract["source_pair_compatibility"] != reference["reference_source_compatibility"]
            or contract["comparison_kind"] != "same_precision"
            or not isinstance(contract["report_format"], str) or not contract["report_format"]
            or not isinstance(contract["public_command"], list) or not contract["public_command"]):
        raise CachePolicyError("Actual comparison is not the released exact candidate/reference source pair")
    sources = _exact_keys(contract["sources"], {"numeric_checker", "record_extractors", "adapter",
        "read_locator", "typed_ipp_validator"}, "All original numerical/current data adapter sources required")
    originals = {"numeric_checker": ("9949a184fef7e3b55c75336544fa04f52a2062f1b37cfdad1018f2243280b5ce", 17239),
        "record_extractors": ("95dc41bf655c5d22a91dd5bf7cf55430f0f5a511d8d7cdabe8f8e68378b6aa13", 21587)}
    for role, source in sources.items():
        _module_source_descriptor(source)
        if role in originals and (source["sha256"], source["size_bytes"]) != originals[role]:
            raise CachePolicyError("Original comparator/extractor bytes must not change")
    from .owned_process import _exact_source
    for source in sources.values():
        _exact_source(source)
    if sources["adapter"]["path"] not in contract["public_command"]:
        raise CachePolicyError("Actual comparator argv omits the reviewed data adapter")
    for role in ("adapter_source_review", "source_pair_compatibility"):
        if not isinstance(read(_descriptor(contract[role])), dict):
            raise CachePolicyError("Actual source review/compatibility evidence required")
    return deepcopy(contract)


def _checked_bound_comparison(first, binding, *, candidate, read):
    """Read actual original-comparator evidence, never execute or synthesize it."""
    proof = read(_descriptor(binding))
    _exact_keys(proof, {"format", "contract", "report", "actual_command", "actual_exit_code",
        "retirement", "source_runtime_before_after", "Root_acceptance"},
        "Actual whole39 comparison, OS return and independent closure required")
    if (proof["format"] != "darsia-first-volume-bound-original-comparison-v1"
            or type(proof["actual_exit_code"]) is not int or proof["actual_exit_code"] != 0
            or proof["contract"] != first.output_entitlement["comparison_contract"]):
        raise CachePolicyError("Original comparator did not genuinely return OS0 under the predeclared contract")
    reference = read(_descriptor(first.output_entitlement["all_stage_reference"]))
    contract = _checked_bound_comparison_contract(first.task, reference,
        read(_descriptor(proof["contract"])), read=read)
    sources = contract["sources"]
    if proof["actual_command"] != contract["public_command"]:
        raise CachePolicyError("Actual comparator argv differs from the predeclared original route")
    report = read(_descriptor(proof["report"]))
    if (report.get("format") != contract["report_format"] or report.get("complete") is not True
            or report.get("comparison_kind") != "same_precision"
            or report.get("numeric_checker") != sources["numeric_checker"]
            or report.get("reference_completed") != contract["reference_completed"]
            or report.get("candidate_completed") != candidate["reference_completion"]
            or report.get("case_count") != 39 or report.get("new_comparisons_count") != 39
            or report.get("ordered_case_ids") != first.task["ordered_case_ids"]
            or report.get("accepted_prior_comparisons") != []
            or report.get("frame_add8_limits_passed") is not True
            or report.get("new_scientific_arrays_computed") is not False):
        raise CachePolicyError("A genuine fresh original whole39 comparison is required, not producer completion")
    comparisons = report.get("comparisons")
    if (not isinstance(comparisons, list)
            or [row.get("case_id") for row in comparisons] != first.task["ordered_case_ids"]):
        raise CachePolicyError("Original comparison exact39 order differs")
    old = read(contract["reference_completed"])
    new = read(candidate["reference_completion"])
    for index, row in enumerate(comparisons):
        if row.get("passed") is not True or row.get("first_failure") is not None:
            raise CachePolicyError("Original whole39 comparison contains a failed or missing case")
        actual = read(_descriptor(row["receipt"]))
        if (actual.get("case_id") != row["case_id"] or actual.get("limits_passed") is not True
                or actual.get("first_failure") is not None
                or actual.get("reference_result") != old["frame_producers"][index]["result"]
                or actual.get("candidate_result") != new["frame_producers"][index]["result"]):
            raise CachePolicyError("Actual original per-case comparison/result bindings differ")
    retired = read(_descriptor(proof["retirement"]))
    if (retired.get("format") != "darsia-first-original-comparator-retirement-v1"
            or type(retired.get("actual_exit_code")) is not int or retired["actual_exit_code"] != 0
            or retired.get("actual_command") != proof["actual_command"]
            or retired.get("all_owned_retired") is not True
            or not isinstance(retired.get("known_epochs"), list) or not retired["known_epochs"]
            or any(type(row.get("pid")) is not int or row["pid"] <= 0
                or type(row.get("process_create_time")) not in {int, float}
                or not math.isfinite(row["process_create_time"]) or row["process_create_time"] <= 0
                or row.get("exact_incarnation_absent") is not True for row in retired["known_epochs"])):
        raise CachePolicyError("Actual comparator OS return and all exact owned epochs must be closed")
    closure = read(_descriptor(proof["source_runtime_before_after"]))
    if (closure.get("module_closure") != first.provider["module_closure"]
            or closure.get("source_runtime_before_after_exact") is not True
            or closure.get("comparison_sources") != sources):
        raise CachePolicyError("Actual comparator source/runtime before/after evidence differs")
    accepted = read(_descriptor(proof["Root_acceptance"]))
    if not isinstance(accepted, dict) or not accepted:
        raise CachePolicyError("Independent actual comparison acceptance role is missing")
    return deepcopy(proof)


def _checked_bound_writer_compatibility(reference, candidate):
    """Five literal existing writer blobs, without importing DarSIA pre-admission.

    Before scientific allocation, first_output_entitlement uses the original Git writer checker on
    BOTH actual commits. Here the exact predeclared common writer SHA is checked
    against startup-bound current bytes. No extra subprocess is spawned during
    ordinary pre-import resolution; LF/CRLF retains the original EOL semantics.
    """
    root = Path(__file__).resolve().parents[2]
    required = {"src/darsia/fluidflower/mass/n1_pipeline.py", "src/darsia/fluidflower/mass/n1_statistics.py",
        "src/darsia/fluidflower/pipeline/series.py", "src/darsia/fluidflower/native/sidecar.py",
        "src/darsia/fluidflower/pipeline/runner.py"}
    writers = candidate["output_writers"]
    if (writers != reference["output_writers"] or not isinstance(writers, list) or len(writers) != len(required)
            or {row.get("relative_path") for row in writers} != required
            or candidate["requested_native_layout"] != reference["requested_native_layout"]):
        raise CachePolicyError("Qualified full native layout and all five whole output writers must remain exact")
    for row in writers:
        _exact_keys(row, {"relative_path", "sha256"}, "Exact original whole-writer identity required")
        if re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"])) is None:
            raise CachePolicyError("Canonical whole-writer SHA required")
        path = (root / row["relative_path"]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise CachePolicyError("Actual qualified writer escaped its source root")
        before = path.stat(); raw = path.read_bytes(); after = path.stat()
        identity = lambda value:(value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
        if (identity(before) != identity(after)
                or row["sha256"] not in {hashlib.sha256(raw).hexdigest(),
                    hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}):
            raise CachePolicyError("Actual reference/candidate writer blobs differ; no writer exception is inferred")


def _bound_profile_documents(first, bindings, *, read):
    """Pure checked fixed-slot DAG composition; no Run, comparison or adoption."""
    if (not first.bound_mode or not isinstance(first.output_entitlement, dict)
            or first.output_entitlement.get("format") != BOUND_OUTPUT_ADMINISTRATION
            or not isinstance(first.required_capacity_bound, dict)
            or first.required_capacity_bound.get("enforcement_state") != "PREDECLARED_Q106_VOLUME_BOUND"):
        raise CachePolicyError("Only the actual Q117/Q119 predeclared bound route may complete this DAG")
    _exact_keys(bindings, {"candidate_all_stage", "original_comparison", "once_marker"},
        "Actual closed-output, original comparison and original consume marker roles required")
    observation, op = _slot_read(first.provider, "observation")
    evidence, ep = _slot_read(first.provider, "writer_evidence")
    output, mp = _slot_read(first.provider, "measurement")
    native, np = _slot_read(first.provider, "native_completion")
    checked_bounded_evidence(observation, evidence, first=first)
    if (observation.get("actual_claim") != first.task or observation.get("output_measurement") != mp
            or native.get("format") != "darsia-native-worker-first-qualification-completion-v1"
            or any(native.get(key) != observation.get(key) for key in ("request", "provider", "stores_source", "scope", "retirement"))
            or native.get("observation") != op or native.get("startup") != first.provider["startup"]
            or native.get("module_closure") != first.provider["module_closure"]
            or native.get("complete") is not True or native.get("passed") is not True
            or native.get("observer_native_handles_closed") is not True
            or native.get("cache_profile_qualified") is not False
            or native.get("maximum_family_processes") != observation.get("kernel_total_processes")
            or type(native.get("maximum_family_processes")) is not int or native["maximum_family_processes"] <= 0):
        raise CachePolicyError("Actual native/source/startup/kernel completion is missing or differs")
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
        raise CachePolicyError("Actual primary was not bound by the original retained native handle")
    eligibility = native.get("eligibility", {})
    if (eligibility != observation.get("original_publication_and_retirement_eligibility")
            or eligibility.get("original_retirement_verifier_accepted") is not True
            or read(_descriptor(eligibility["seal"])).get("value", {}).get("outcome") != "success"):
        raise CachePolicyError("Original whole-run publication/seal and retirement verifier are required")
    marker_pin = _descriptor(bindings["once_marker"]); marker = read(marker_pin)
    if (PureWindowsPath(marker_pin["path"]).name != "cache-first-" + first._once_key() + ".json"
            or marker.get("format") != "darsia-cache-first-qualification-consumed-v1"
            or any(marker.get(key) != observation.get(key) for key in ("request", "provider", "stores_source", "scope"))
            or marker.get("no_retry") is not True):
        raise CachePolicyError("Actual original immutable once marker differs")
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
        raise CachePolicyError("Actual original worker service role/stores/namespace/admin differs")
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
        raise CachePolicyError("Actual cold service must predate every qualified writer with its exact empty FILE_CREATE provenance")
    from .control import _time
    _time(fresh["initial_empty_observed_utc"])
    lifecycle_binding = dict(reservation=service["manifest"]["reservation"], policy=first.stores_source,
        profile=first.provider_binding, namespace=first.provider["namespace_admission"], manifest=service["manifest"])
    for row in evidence["registered_owner_environments"]:
        registered = read(_descriptor(row["registration"]))
        if (registered.get("format") != "darsia-cache-service-lifecycle-v1"
                or registered.get("kind") != "owner" or registered.get("binding") != lifecycle_binding
                or registered.get("actor") != row["owner"]
                or registered.get("value", {}).get("owner") != row["owner"]
                or registered.get("value", {}).get("native_job_membership") != observation["native_launch"]):
            raise CachePolicyError("Original namespace/owner registration is outside the actual native family")
        exited = read(_descriptor(row["inventory"]))
        modules = exited.get("first_qualification_modules", {})
        if (exited.get("owner") != row["owner"]
                or modules.get("request") != first.binding or modules.get("provider") != first.provider_binding
                or modules.get("all_file_backed_modules_predeclared_and_exact") is not True
                or modules.get("undeclared") != []
                or modules.get("predeclared_module_closure_sha256") != _digest(first.provider["module_closure"])
                or modules.get("actual_writer_environment") != row["actual_writer_environment"]
                or modules.get("writer_environment_is_actual_IO_or_lifetime_proof") is not False):
            raise CachePolicyError("Original owner's actual exit inventory and writer environment differ")
    candidate = read(_descriptor(bindings["candidate_all_stage"]))
    if (candidate.get("reference_commit") != first.task["commit"]
            or candidate.get("candidate_commit") != first.task["commit"]):
        raise CachePolicyError("Closed output must be the actual candidate, never a relabelled Main reference")
    checked_all_stage_reference(first.task, candidate, read=read,
        maximum_output_bytes=first.output_entitlement["administrative_output_entitlement_bytes"])
    reference = read(_descriptor(first.output_entitlement["all_stage_reference"]))
    _checked_bound_writer_compatibility(reference, candidate)
    closed = read(_descriptor(candidate["measurement"]))
    if (output.get("format") != "darsia-first-q104-closed-output-diagnostic-v1"
            or output.get("request") != first.binding or output.get("provider") != first.provider_binding
            or output.get("stores_source") != first.stores_source
            or output.get("scope") != first.provider["scope"]
            or output.get("output_plan") != first.request["output_entitlement_plan"]
            or output.get("output_scope") != first.output_entitlement["watched_output_scope"]
            or output.get("complete") is not True or output.get("first_stop") is not None
            or output.get("first_observation_error") is not None
            or output.get("cache_operational_reserve_crossed") is not False
            or output.get("whole_output_entitlement_observed_exceeded") is not False
            or output.get("actual_retirement") != observation["retirement"]
            or output.get("actual_terminal") != observation["original_kernel_terminal"]
            or output.get("entitlement_bytes") != first.output_entitlement["administrative_output_entitlement_bytes"]
            or output.get("closed_output_logical_bytes") != closed["total_logical_bytes"]
            or output.get("sampled_highwater_is_lifetime_peak") is not False
            or output.get("output_hard_quota") is not False
            or not _same_path(output["output_scope"]["path"], closed["root"])):
        raise CachePolicyError("Actual closed whole-output inventory differs from the original in-run watch")
    samples = output.get("output_samples")
    admission = output.get("admission")
    if (output.get("target_sample_period_seconds") != 1 or not isinstance(samples, list) or not samples
            or not isinstance(admission, dict) or admission.get("logical_bytes") != 0
            or admission.get("observed_scan_anomalies") != []
            or admission.get("volume", {}).get("free_bytes", 0) - PHYSICAL_RESERVE_BYTES < output["entitlement_bytes"]):
        raise CachePolicyError("Original empty output admission and genuine one-second-target samples are required")
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
                or nonnegative(sample["logical_bytes"], "Actual output sample bytes") > output["entitlement_bytes"]):
            raise CachePolicyError("Output STOP or inconsistent actual sample/gap timeline cannot be qualified")
        previous = instant
    if (samples[-1].get("forced_terminal") is not True
            or output.get("sampled_highwater_bytes") != max(row["logical_bytes"] for row in samples)
            or output.get("final_snapshot", {}).get("logical_bytes") != samples[-1]["logical_bytes"]):
        raise CachePolicyError("Actual forced terminal and sampled output highwater must remain reproducible")
    sampled = output.get("final_snapshot", {}).get("file_stats")
    if (not isinstance(sampled, list)
            or len({str(PureWindowsPath(row["relative_path"])).casefold() for row in sampled}) != len(sampled)
            or {str(PureWindowsPath(row["relative_path"])).casefold():row["size_bytes"] for row in sampled} != {
                str(PureWindowsPath(row["relative_path"])).casefold():row["bytes"] for row in closed["inventory"]}
            or closed["root_identity"] != output["output_scope"]["identity"]):
        raise CachePolicyError("Closed output SHA inventory must cover every originally observed retained file")
    _checked_bound_comparison(first, bindings["original_comparison"], candidate=candidate, read=read)
    common = dict(request=first.binding, provider=first.provider_binding, stores_source=first.stores_source,
        scope=first.provider["scope"], configuration=first.task["configuration"],
        qualification=deepcopy(bindings), observation=op,
        writer_evidence=ep, measurement=mp, completion=np,
        namespace_admission=first.provider["namespace_admission"],
        producer_source_closure=first.provider["producer_source_closure"],
        runtime_source_closure=first.provider["runtime_source_closure"],
        capacity_bound=evidence["capacity_bound"],
        operational_service_entitlement_bytes=evidence["administrative_service_entitlement"]["maximum_bytes"],
        sampled_cache_highwater_bytes=observation["sampled_cache_logical_high_water_bytes"],
        sampled_highwater_is_lifetime_peak=False, actual_writer_IO_proven=False,
        writer_contract=BOUND_WRITER_CONTRACT)
    imported = dict(common, format=BOUND_IMPORT_PROFILE, purpose="jobs", role_identifier=service["role_identifier"], entry_kind="worker",
        maximum_cache_bytes=common["operational_service_entitlement_bytes"],
        python_executable=first.provider["python_executable"], module_closure=first.provider["module_closure"],
        native_supervision=dict(format="darsia-native-worker-supervision-profile-v1",
            startup=first.provider["startup"], completion=np,
            maximum_family_processes=native["maximum_family_processes"],
            publication_contract="alias-done-small-metadata-v1"))
    # The original result-estimate model consumes genuine closed inventory; this
    # is separate from the physical cache bound and from sampled cache peaks.
    result_profile = dict(format="darsia-expected-space-profile-v1", pipeline=first.task["pipeline"],
        precision=first.task["precision"], measured_commit=first.task["commit"],
        measured_case_ids=first.task["ordered_case_ids"], measured_native_layout=candidate["requested_native_layout"],
        writers=candidate["output_writers"], measurement=deepcopy(closed),
        completion=dict(complete=True, passed=True, actual_commit=first.task["commit"], actual_case_count=39,
            precision=first.task["precision"], original_comparison=bindings["original_comparison"],
            actual_native_completion=np), reference_compatibility=reference["reference_source_compatibility"])
    job = dict(common, format=BOUND_JOB_PROFILE, purpose="jobs", additional_claim_cache_bytes=0,
        allocation_model="shared-service-only-no-claim-files-v1", result_estimate_excludes_cache=True,
        result_profile=result_profile)
    return imported, job, dict(observation=op, writer_evidence=ep, measurement=mp,
        native_completion=np, retirement=observation["retirement"])


def _publish_bound_profiles(first, bindings, *, read):
    """Existing fixed slots only; called by reviewed post-comparison metadata composition.

    No slot is overwritten and the final completion is written last. Neither
    profile embeds the final completion descriptor, so the DAG has no hash cycle.
    """
    if not first.bound_mode:
        raise CachePolicyError("Only the existing bound request may publish this distinct typed evidence")
    imported, jobs, actual = _bound_profile_documents(first, bindings, read=read)
    ip = _slot_write(first.provider, "import_profile", imported)
    jp = _slot_write(first.provider, "job_profile", jobs)
    return _slot_write(first.provider, "completion", dict(format=BOUND_BUNDLE,
        request=first.binding, provider=first.provider_binding, stores_source=first.stores_source,
        scope=first.provider["scope"], qualification=deepcopy(bindings),
        profiles=dict(import_profile=ip, job_profile=jp), **actual))


def _resolve_bound_profile(binding, provider, bundle, kind, read, *, stores_source, read_authorities):
    _exact_keys(bundle, {"format", "request", "provider", "stores_source", "scope", "qualification",
        "observation", "writer_evidence", "measurement", "native_completion", "retirement", "profiles"},
        "Exact completed volume-bound provider DAG required")
    if (kind not in {"import", "jobs"} or bundle["format"] != BOUND_BUNDLE or bundle["provider"] != binding
            or bundle["scope"] != provider["scope"] or stores_source != bundle["stores_source"]):
        raise CachePolicyError("Typed bound profile/stores/source role differs")
    first = FirstQualification(bundle["request"], provider_binding=binding,
        stores_source=bundle["stores_source"], namespace=provider["namespace_admission"],
        read=read, read_authorities=read_authorities)
    if not first.bound_mode:
        raise CachePolicyError("Typed bound bundle cannot disguise an old direct/reference provider")
    from .owned_process import _exact_source
    for source in provider["module_closure"]:
        _exact_source(source)
    imported, jobs, actual = _bound_profile_documents(first, bundle["qualification"], read=read)
    if any(bundle[key] != value for key, value in actual.items()):
        raise CachePolicyError("Final bound completion does not bind all actual producer slots")
    _exact_keys(bundle["profiles"], {"import_profile", "job_profile"}, "Both fixed typed profile slots required")
    for role, expected in (("import_profile", imported), ("job_profile", jobs)):
        document, pin = _slot_read(provider, role)
        if pin != bundle["profiles"][role] or document != expected:
            raise CachePolicyError("Actual typed profile is not the exact checked fixed-slot projection")
    return imported if kind == "import" else jobs


def checked_bound_import_context(profile, *, purpose, role_identifier, entry_kind, policy, namespace):
    """Applicability of an already fully resolved distinct bounded profile.

    Actual startup's original ledger still rechecks the physical volume, free
    space, namespace and reservation. This never turns samples into a maximum.
    """
    if (profile.get("format") != BOUND_IMPORT_PROFILE or purpose != "jobs" or entry_kind != "worker"
            or profile.get("purpose") != purpose or profile.get("entry_kind") != entry_kind
            or profile.get("role_identifier") != role_identifier
            or profile.get("namespace_admission") != namespace
            or profile.get("writer_contract") != BOUND_WRITER_CONTRACT
            or profile.get("sampled_highwater_is_lifetime_peak") is not False
            or profile.get("actual_writer_IO_proven") is not False):
        raise CachePolicyError("The actual ordinary entry/namespace is outside this qualified worker-only bound")
    cap = policy.volume_cap
    bound = checked_required_capacity_bound(profile["capacity_bound"])
    maximum = nonnegative(profile["operational_service_entitlement_bytes"], "Qualified operational service entitlement")
    if (cap is None or cap.maximum_size_bytes != 500*1024**3 or policy.quota_bytes != 500*1024**3
            or bound["enforcement_state"] != "PREDECLARED_Q106_VOLUME_BOUND"
            or bound["volume_cap_observation"].get("binding") != cap.document()
            or bound["volume_cap_observation"].get("virtual_size_bytes") != 500*1024**3
            or not 0 < maximum <= bound["operational_admission_headroom_bytes"]
            or profile["maximum_cache_bytes"] != maximum):
        raise CachePolicyError("Actual Q87 cap and separate operational entitlement differ")
    return maximum


def checked_required_capacity_bound(value):
    """Q106 hard volume bound and separate sampled operational reserve; no GO."""
    _exact_keys(value, {"format", "authority_id", "volume_cap_observation",
        "volume_capacity_bytes", "free_volume_reserve_bytes", "required_capacity_bound_bytes",
        "operational_admission_headroom_bytes", "reserve_watchdog_period_seconds",
        "hard_bound_semantics", "reserve_semantics", "exclusive_volume_writer_required",
        "sampled_observed_highwater_is_complete", "enforcement_state"},
        "Exact typed Q106 volume-bound and operational-reserve plan required")
    observed = value["volume_cap_observation"]
    if (value["format"] != CAPACITY_BOUND
            or value["authority_id"] != "Q106-answer-sampled-operational-reserve-hard-bound-is-Q87-volume-no-new-seam"
            or not isinstance(observed, dict) or observed.get("verified") is not True
            or observed.get("format") != "darsia-readonly-fixed-vhdx-cap-observation-v1"
            or observed.get("filesystem") != "NTFS"
            or observed.get("is_loaded") is not True
            or value["sampled_observed_highwater_is_complete"] is not False
            or value["hard_bound_semantics"] != "actual-fixed-volume-only-not-capacity-minus-reserve"
            or value["reserve_semantics"] != "operational-admission-plus-sampled-watchdog-not-hard-floor"
            or value["exclusive_volume_writer_required"] is not True
            or type(value["reserve_watchdog_period_seconds"]) not in {int,float}
            or value["reserve_watchdog_period_seconds"] != 1
            or value["enforcement_state"] not in {"PRELAUNCH_HOLD_N43_REFERENCE_OUTPUT_BUDGET_AND_PROFILE_PRODUCER",
                "PREDECLARED_Q106_VOLUME_BOUND"}):
        raise CachePolicyError("Actual volume-only bound and separately sampled operational reserve required")
    capacity = nonnegative(value["volume_capacity_bytes"], "Actual usable volume capacity")
    if (capacity != observed.get("volume_total_bytes")
            or capacity <= PHYSICAL_RESERVE_BYTES
            or value["free_volume_reserve_bytes"] != PHYSICAL_RESERVE_BYTES
            or type(value["required_capacity_bound_bytes"]) is not int
            or value["required_capacity_bound_bytes"] != capacity
            or type(value["operational_admission_headroom_bytes"]) is not int
            or value["operational_admission_headroom_bytes"] != capacity - PHYSICAL_RESERVE_BYTES):
        raise CachePolicyError("Hard bound is actual volume capacity; capacity minus40GiB is only operational headroom")
    return deepcopy(value)


def checked_output_entitlement(task, value, *, task_binding, read, read_authorities=None):
    """Q104 source-bound administration, never an expected-space profile."""
    if isinstance(value, dict) and value.get("format") == BOUND_OUTPUT_ADMINISTRATION:
        return checked_whole_output_administration(task, value, task_binding=task_binding,
            read=read, read_authorities=read_authorities)
    if isinstance(value, dict) and value.get("format") == "darsia-first-qualification-output-entitlement-v3":
        _exact_keys(value, {"format", "authority_id", "task", "scope", "R4_footprint",
            "administrative_output_entitlement_bytes", "measured_profile", "enforcement_state",
            "all_stage_reference", "watched_output_scope"}, "Exact current Q104 output-watch plan required")
        original = {key: deepcopy(row) for key, row in value.items() if key != "watched_output_scope"}
        original["format"] = "darsia-first-qualification-output-entitlement-v2"
        checked_output_entitlement(task, original, task_binding=task_binding, read=read)
        checked_bound_output_scope(value["watched_output_scope"])
        return deepcopy(value)
    if isinstance(value, dict) and value.get("format") == "darsia-first-qualification-output-entitlement-v2":
        _exact_keys(value, {"format", "authority_id", "task", "scope", "R4_footprint",
            "administrative_output_entitlement_bytes", "measured_profile", "enforcement_state",
            "all_stage_reference"}, "Exact first-only Q104 all-stage reference required")
        original = {key: deepcopy(row) for key, row in value.items() if key != "all_stage_reference"}
        original["format"] = "darsia-first-qualification-output-entitlement-v1"
        checked_output_entitlement(task, original, task_binding=task_binding, read=read)
        checked_all_stage_reference(task, read(_descriptor(value["all_stage_reference"])), read=read)
        return deepcopy(value)
    _exact_keys(value, {"format", "authority_id", "task", "scope", "R4_footprint",
        "administrative_output_entitlement_bytes", "measured_profile", "enforcement_state"},
        "Exact predeclared Q104 output entitlement required")
    scope = {key: task.get(key) for key in ("commit", "pipeline", "precision", "run", "ordered_case_ids")}
    if (value["format"] != "darsia-first-qualification-output-entitlement-v1"
            or value["authority_id"] != "Q104-answer-first-qualification-produces-result-space-profile-under-predeclared-entitlement"
            or value["task"] != _descriptor(task_binding)
            or value["scope"] != scope or scope["pipeline"] != "native-phase-n1"
            or scope["run"] != "ac52" or scope["precision"] != "float32"
            or not isinstance(scope["ordered_case_ids"], list)
            or len(scope["ordered_case_ids"]) != 39
            or len(set(scope["ordered_case_ids"])) != 39
            or value["measured_profile"] is not False
            or value["enforcement_state"] != "PRELAUNCH_HOLD_N43_REFERENCE_OUTPUT_BUDGET_AND_PROFILE_PRODUCER"):
        raise CachePolicyError("Q104 requires actual current exact39 task and truthful N43 administrative HOLD")
    footprint = read(_descriptor(value["R4_footprint"]))
    if (footprint.get("format") != "first-qualification-predeclared-output-entitlement-source-data-not-profile-v1"
            or footprint.get("authority_id") != value["authority_id"]
            or footprint.get("run") != scope["run"]
            or footprint.get("case_count") != 39
            or footprint.get("ordered_case_ids") != scope["ordered_case_ids"]
            or footprint.get("factor") != 2
            or footprint.get("not_current_measurement_or_profile") is not True
            or footprint.get("no_first_qualification_or_adoption") is not True):
        raise CachePolicyError("Actual pinned original R4 exact39 footprint required")
    rows = footprint.get("files")
    if not isinstance(rows, list) or not rows:
        raise CachePolicyError("Nonempty actual original output file-size ledger required")
    total = 0
    names = set()
    coverage = {case: set() for case in scope["ordered_case_ids"]}
    for row in rows:
        name = str(PureWindowsPath(row["path"])).casefold()
        size = nonnegative(row["size_bytes"], "Original actual result file length")
        if (name in names or not PureWindowsPath(row["path"]).is_absolute()
                or row.get("actual_stat", {}).get("size_bytes") != size
                or re.fullmatch(r"[0-9a-f]{64}", str(row.get("declared_original_sha256"))) is None):
            raise CachePolicyError("Unique source-bound actual result file lengths required")
        roles = row.get("roles")
        case = row.get("case_id")
        if (not isinstance(roles, list) or not roles or any(not isinstance(role, str) for role in roles)
                or (case is not None and case not in coverage)):
            raise CachePolicyError("Exact original per-case output roles required")
        if case is not None:
            coverage[case].update(roles)
        names.add(name)
        total += size
    required = {"products.npz", "result.json", "preview.png", "native_statistics.json.gz", "geometry"}
    if (footprint.get("unique_output_file_count") != len(rows)
            or any(not required <= roles for roles in coverage.values())):
        raise CachePolicyError("R4 footprint must include every original exact39 output/geometry role")
    if (not total or footprint.get("actual_result_footprint_bytes") != total
            or footprint.get("administrative_output_entitlement_bytes") != 2 * total
            or type(value["administrative_output_entitlement_bytes"]) is not int
            or value["administrative_output_entitlement_bytes"] != 2 * total):
        raise CachePolicyError("Q104 entitlement is exactly twice the bound actual R4 footprint")
    return deepcopy(value)


def checked_whole_output_administration(task, value, *, task_binding, read, read_authorities=None):
    """Q119 supersedes product-only Q104, without a guessed footprint or quota."""
    _exact_keys(value, {"format", "authority_ids", "answers_source", "task", "scope",
        "all_stage_reference", "administrative_output_entitlement_bytes", "measured_profile",
        "watched_output_scope", "producer_state", "comparison_contract"}, "Exact Q117/Q119 first-only output administration required")
    scope = {key: task.get(key) for key in ("commit", "pipeline", "backend", "precision", "run", "ordered_case_ids")}
    if (value["format"] != BOUND_OUTPUT_ADMINISTRATION
            or value["authority_ids"] != ["Q104", "Q109", "Q117", "Q119"]
            or value["task"] != _descriptor(task_binding) or value["scope"] != scope
            or scope["pipeline"] != "native-phase-n1" or scope["backend"] != "cpu"
            or scope["precision"] != "float32" or scope["run"] != "ac52"
            or not isinstance(scope["ordered_case_ids"], list) or len(scope["ordered_case_ids"]) != 39
            or len(set(scope["ordered_case_ids"])) != 39
            or value["measured_profile"] is not False or value["producer_state"] != "PREDECLARED_NOT_QUALIFIED"):
        raise CachePolicyError("Actual exact39 first task and truthful Q119 administrative state required")
    authorities = (read if read_authorities is None else read_authorities)(_descriptor(value["answers_source"]))
    by_id = {row.get("id"): row for row in authorities} if isinstance(authorities, list) else {}
    required = ["Q109-answer-option-A-whole-output-entitlement-65728995330B-all-stages-count",
        "Q117-answer-yes-this-is-the-Q102-Q104-Q106-evidence-contract",
        "Q119-answer-Q109-entitlement-already-covers-first-qualification-or-twice-measured-allstage"]
    if any(by_id.get(name, {}).get("decided_by") != "coordinator" for name in required):
        raise CachePolicyError("Actual Q109/Q117/Q119 decisions required; no product-only fallback")
    reference = read(_descriptor(value["all_stage_reference"]))
    measurement = read(_descriptor(reference["measurement"]))
    observed = nonnegative(measurement["total_logical_bytes"], "Actual complete Main all-stage footprint")
    maximum = max(FIRST_OUTPUT_BASE_ENTITLEMENT_BYTES, 2 * observed)
    if (not observed or type(value["administrative_output_entitlement_bytes"]) is not int
            or value["administrative_output_entitlement_bytes"] != maximum):
        raise CachePolicyError("Predeclare max65,728,995,330B or twice actual all-stage bytes before Run")
    checked_all_stage_reference(task, reference, read=read, maximum_output_bytes=maximum)
    _checked_bound_comparison_contract(task, reference,
        read(_descriptor(value["comparison_contract"])), read=read)
    checked_bound_output_scope(value["watched_output_scope"])
    return deepcopy(value)


def checked_all_stage_reference(task, value, *, read, maximum_output_bytes=Q104_OUTPUT_ENTITLEMENT_BYTES):
    """Q104 prelaunch data, separate from cache qualification and compression estimates.

    The immutable Root-released plan must bind an actual complete current result
    inventory. Input-layout getters or the old R4 five-product footprint alone
    are insufficient. Reference and candidate commits deliberately stay distinct.
    """
    _exact_keys(value, {"format", "candidate_commit", "reference_commit", "configuration",
        "ordered_case_ids", "reference_completion", "reference_Root_acceptance", "reference_native_host_evidence",
        "reference_source_compatibility", "requested_native_layout", "output_writers",
        "measurement", "not_a_cache_profile"}, "Exact current all-stage reference metadata required")
    if (value["format"] != "darsia-first-qualification-all-stage-reference-v1"
            or value["candidate_commit"] != task["commit"]
            or re.fullmatch(r"[0-9a-f]{40}", str(value["reference_commit"])) is None
            or value["configuration"] != task.get("configuration")
            or value["ordered_case_ids"] != task["ordered_case_ids"]
            or value["not_a_cache_profile"] is not True):
        raise CachePolicyError("Actual reference commit, unchanged inputs and candidate commit must be explicit")
    completed = read(_descriptor(value["reference_completion"]))
    # Root acceptance is an explicit required immutable role in the released
    # plan. It is not inferred from producer complete or an OS return code.
    acceptance = read(_descriptor(value["reference_Root_acceptance"]))
    # The unchanged CLI's main-cpu label is not physical-host evidence. Require
    # its independent actual Root/native role too; do not invent its future schema.
    native_host_evidence = read(_descriptor(value["reference_native_host_evidence"]))
    compatibility = read(_descriptor(value["reference_source_compatibility"]))
    if (not isinstance(acceptance, dict) or not acceptance
            or not isinstance(native_host_evidence, dict) or not native_host_evidence
            or not isinstance(compatibility, dict) or not compatibility
            or completed.get("complete") is not True or completed.get("whole_run") is not True
            or completed.get("worker_references_closed") is not True
            or completed.get("pipeline") != task["pipeline"]
            or completed.get("precision") != task["precision"]
            or completed.get("backend") != task.get("backend") or task.get("backend") != "cpu"
            or not isinstance(completed.get("actual_worker"), dict)
            or completed.get("actual_worker", {}).get("machine") != "main-cpu"
            or completed.get("commit") != value["reference_commit"]
            or completed.get("ordered_case_ids") != task["ordered_case_ids"]
            or completed.get("case_count") != 39 or completed.get("frames") != 39
            or completed.get("newly_computed_frames") != 39
            or completed.get("configuration") != task["configuration"]
            or completed.get("reused_case_ids") != []
            or not isinstance(completed.get("records"), list)
            or [row.get("frame_id") for row in completed["records"]] != task["ordered_case_ids"]
            or not isinstance(completed.get("frame_producers"), list)
            or [row.get("case_id") for row in completed["frame_producers"]] != task["ordered_case_ids"]):
        raise CachePolicyError("Accepted genuine fresh current whole39 reference and source-compatibility roles required")
    measure = read(_descriptor(value["measurement"]))
    _exact_keys(measure, {"format", "reference_completion", "root", "root_identity", "inventory",
        "total_logical_bytes", "inventory_scope", "closed_writer_evidence", "complete"},
        "Actual all-stage output measurement required")
    if (measure["format"] != "darsia-first-qualification-all-stage-output-inventory-v1"
            or measure["reference_completion"] != value["reference_completion"]
            or measure["inventory_scope"] != "whole-owned-result-attempt-including-preparation-and-ops"
            or measure["complete"] is not True or not isinstance(measure["root_identity"], dict)
            or not measure["root_identity"]):
        raise CachePolicyError("Final-products-only or incomplete output inventory is not Q104 all-stage evidence")
    read(_descriptor(measure["closed_writer_evidence"]))
    root = PureWindowsPath(measure["root"])
    if not root.is_absolute() or ".." in root.parts:
        raise CachePolicyError("One actual owned result-attempt root required")
    inventory = measure["inventory"]
    if not isinstance(inventory, list) or not inventory:
        raise CachePolicyError("Actual output file inventory cannot be UNKNOWN")
    seen = {}; total = 0
    for row in inventory:
        _exact_keys(row, {"relative_path", "bytes", "sha256"}, "Exact named output file size/SHA required")
        relative = PureWindowsPath(row["relative_path"])
        key = str(relative).casefold()
        if (not row["relative_path"] or relative.is_absolute() or relative.drive or relative.root
                or ".." in relative.parts or key in seen
                or re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"])) is None):
            raise CachePolicyError("Output inventory contains an alias, duplicate or unbound file")
        seen[key] = row; total += nonnegative(row["bytes"], "Actual all-stage output bytes")

    def inventoried(binding):
        # Payload identities can exceed the JSON control-size limit. Check only
        # the named metadata descriptor; read() below is used solely for JSON.
        _exact_keys(binding, {"path", "sha256", "size_bytes"}, "Exact actual publication file binding required")
        path = PureWindowsPath(binding["path"])
        if (not path.is_absolute() or ".." in path.parts or root not in path.parents
                or re.fullmatch(r"[0-9a-f]{64}", str(binding["sha256"])) is None):
            raise CachePolicyError("Reference publication file escapes the bound result-attempt root")
        relative = str(path.relative_to(root)).casefold()
        row = seen.get(relative)
        if (row is None or row["bytes"] != nonnegative(binding["size_bytes"], "Published file bytes")
                or row["sha256"] != binding["sha256"]):
            raise CachePolicyError("Actual publication/preparation file size/SHA is absent or differs in whole-output inventory")
        return path

    inventoried(completed["manifest"])
    inventoried(completed["table"])
    # At minimum every case has its retained native/temporal buffers and final
    # publication. Extra ops/metadata are counted too, never silently excluded.
    for case in task["ordered_case_ids"]:
        required = {str(PureWindowsPath("preparation", case, name)).casefold()
            for name in ("native_log.npy", "native_valid.npy", "temporal_log.npy", "temporal_valid.npy", "phase_area.npz")}
        final = [row for row in inventory if PureWindowsPath(row["relative_path"]).name == "commit.json"
            and case in PureWindowsPath(row["relative_path"]).parts]
        if not required <= set(seen) or len(final) != 1:
            raise CachePolicyError("Retained preparation/final case outputs are missing from the whole-result inventory")
    for case, record, producer in zip(task["ordered_case_ids"], completed["records"], completed["frame_producers"]):
        _exact_keys(producer, {"case_id", "attempt_directory", "manifest", "commit",
            "frame_registration", "publication", "result"}, "Original exact frame-producer roles required")
        directory = PureWindowsPath(record["nas_directory"])
        files = record.get("files")
        if (not _same_path(producer["attempt_directory"], root) or producer["manifest"] != completed["manifest"]
                or directory != root / "frames" / case or not isinstance(files, dict)
                or not {"products.npz", "native_geometry.npz", "native_statistics.json.gz", "result.json", "commit.json"} <= set(files)):
            raise CachePolicyError("Actual reference frame publication/attempt directory or final files differ")
        for filename, identity in files.items():
            if PureWindowsPath(filename).name != filename or filename in {".", ".."}:
                raise CachePolicyError("Original published final file name required")
            _exact_keys(identity, {"sha256", "size_bytes"}, "Original classic final file identity required")
            inventoried(dict(path=str(directory / filename), **identity))
        for role in ("commit", "result"):
            expected = dict(path=str(directory / (role + ".json")), **files[role + ".json"])
            if producer[role] != expected:
                raise CachePolicyError("Actual producer JSON binding differs from its published frame: " + role)
        commit = read(_descriptor(producer["commit"]))
        result = read(_descriptor(producer["result"]))
        inventoried(producer["frame_registration"]); inventoried(producer["publication"])
        registration = read(_descriptor(producer["frame_registration"]))
        publication = read(_descriptor(producer["publication"]))
        expected_files = {name: row for name, row in files.items() if name != "commit.json"}
        if (commit.get("complete") is not True or commit.get("frame_id") != case
                or commit.get("manifest_sha256") != completed["manifest"]["sha256"]
                or commit.get("files") != expected_files or publication != record
                or registration.get("frame_id") != case or registration.get("commit") != producer["commit"]
                or registration.get("files") != expected_files
                or registration.get("manifest_sha256") != completed["manifest"]["sha256"]
                or result.get("complete") is not True or result.get("case_id") != case):
            raise CachePolicyError("Actual final commit/result and original ops publication/registration differ")
        if (PureWindowsPath(producer["frame_registration"]["path"]) != root / "ops" / ("frame_commit_" + commit["commit_id"] + ".json")
                or PureWindowsPath(producer["publication"]["path"]) != root / "ops" / ("nas_copy_" + commit["commit_id"] + ".json")):
            raise CachePolicyError("Original ops paths differ from the committed frame identity")
        stage = result.get("prediction", {}).get("pipeline", {})
        stage_files = stage.get("files")
        if (stage.get("case_id") != case or stage.get("complete") is not True
                or stage.get("configuration_sha256") != task["configuration"]["sha256"]
                or not isinstance(stage_files, dict) or not stage_files):
            raise CachePolicyError("Actual result preparation inventory/configuration is missing")
        for filename, binding in stage_files.items():
            path = inventoried(binding)
            if path != root / "preparation" / case / filename or PureWindowsPath(filename).name != filename:
                raise CachePolicyError("Actual preparation file escaped its original case-stage scope")
    if (measure["total_logical_bytes"] != total or not total
            or total > maximum_output_bytes):
        raise CachePolicyError("Actual all-stage output exceeds fixed Q104 entitlement or has inconsistent totals")
    return deepcopy(value)


def checked_current_reference_release(value):
    """Validate a future source-registered entry, without registering a release.

    The production slot stays None until actual human GO and plan bindings
    have their full source review. No data file or caller boolean can register
    an entry, reuse a historical GO, or retarget a historical accepted plan.
    """
    _exact_keys(value, {"format", "authority_ids", "records", "human_go_record_id",
        "plan_sha256"}, "Complete source-bound current V9 reference release required")
    ids, records, human = value["authority_ids"], value["records"], value["human_go_record_id"]
    old_ids = {item for key in REFERENCE_RELEASES for item in key}
    old_records = {item for release in REFERENCE_RELEASES.values() for item in release["records"]}
    if (value["format"] != CURRENT_V9_REFERENCE_RELEASE_FORMAT
            or not isinstance(ids, list) or len(ids) < 2
            or any(not isinstance(item, str) or not item for item in ids)
            or len(set(ids)) != len(ids) or "Q432" not in ids or set(ids) & old_ids
            or not isinstance(human, str) or not human or human in old_records
            or human == CURRENT_V9_PREPARATION_RECORD
            or not isinstance(records, dict)
            or records != {CURRENT_V9_PREPARATION_RECORD: "coordinator", human: "owner"}
            or re.fullmatch(r"[0-9a-f]{64}", str(value["plan_sha256"])) is None
            or value["plan_sha256"] in {release["plan_sha256"] for release in REFERENCE_RELEASES.values()}):
        raise CachePolicyError("New exact-plan authority and a separate actual owner GO are required")
    return deepcopy(value)


def checked_current_build_plan_metadata(config, operation_result, catalog):
    """Join prospective named DATA without importing post-startup Build readers.

    This association is not original publication, physical or eligibility
    verification. It cannot release the current reference execution STOP.
    """
    from .current_build_binding import checked_current_build_binding, current_build_binding_inputs
    try:
        if not isinstance(config, dict) or not isinstance(catalog, dict):
            raise ValueError("Actual queued configuration and named catalogue DATA are required")
        binding = checked_current_build_binding(config.get("current_build_binding"), config=config)
        if _descriptor(operation_result) != binding["operation_result"]:
            raise ValueError("The current plan differs from the embedded original Build operation receipt")
        inputs = catalog.get("inputs")
        if not isinstance(inputs, list):
            raise ValueError("The current Build requires its original named input catalogue")
        for use, bound, role in current_build_binding_inputs(binding):
            rows = [row.get("reference") for row in inputs if isinstance(row, dict)
                    and isinstance(row.get("uses"), list) and use in row["uses"]]
            if (len(rows) != 1 or not isinstance(rows[0], dict) or rows[0].get("role") != role
                    or rows[0].get("sha256") != bound["sha256"]
                    or type(rows[0].get("bytes")) is not int or rows[0]["bytes"] != bound["size_bytes"]):
                raise ValueError("The current Build lost its exact named catalogue role: " + use)
    except (KeyError, TypeError, ValueError) as exc:
        raise CachePolicyError("Current Build metadata binding refused: " + str(exc)) from exc
    return binding


def checked_current_reference_plan(value, release, administration, first, *, read, authorities):
    """Join prospective queued Build DATA, then STOP before execution authority.

    The named catalogue and post-startup readers can bind original Build and
    coordinate evidence. No supplied record can lift this execution STOP:
    complete source closure, independent physical/N1 eligibility and original
    reference acceptance remain unavailable. Schema and byte equality alone
    grant none of those authorities.
    Original source/runtime, RunIO, volume, writer, once-only, publication and
    retirement verification remain mandatory and unchanged.
    """
    release = checked_current_reference_release(release)
    _exact_keys(value, {"format", "authority_ids", "scope", "configuration",
        "configuration_source", "input_catalog_sha256", "current_v9_build", "native_parameters",
        "python_executable", "capacity_bound_plan", "requested_native_layout",
        "output_writers"}, "Exact new current V9 reference plan bindings required")
    scope = {key: first.provider["scope"][key] for key in
        ("pipeline", "backend", "precision", "run", "ordered_case_ids")}
    # Cache bootstrap must not import darsia before original startup admission.
    # These are additional identity checks, not a replacement catalogue reader.
    catalog = first.task.get("input_catalog")
    if (not isinstance(catalog, dict) or catalog.get("complete") is not True
            or catalog.get("format") != "darsia-role-input-catalog-v1"
            or not isinstance(catalog.get("inputs"), list) or not catalog["inputs"]
            or any(not isinstance(row, dict) or not isinstance(row.get("reference"), dict)
                or not isinstance(row.get("uses"), list) or not row["uses"]
                or any(not isinstance(use, str) or not use for use in row["uses"])
                for row in catalog["inputs"])):
        raise CachePolicyError("Actual original complete named input catalogue required")
    if (value["format"] != CURRENT_V9_REFERENCE_PLAN
            or value["authority_ids"] != release["authority_ids"]
            or value["authority_ids"] != administration["authority_ids"]
            or value["scope"] != scope or scope != dict(pipeline="native-phase-n1",
                backend="cpu", precision="float32", run="ac52", ordered_case_ids=REFERENCE_CASE_IDS)
            or value["configuration"] != first.task.get("configuration")
            or value["configuration"] != administration["output_configuration"]
            or catalog.get("configuration") != first.task["configuration"]
            or catalog.get("run") != first.task.get("run")
            or value["input_catalog_sha256"] != _digest(catalog)
            or _descriptor(value["python_executable"]) != first.provider["python_executable"]
            or value["requested_native_layout"] != administration["requested_native_layout"]
            or value["output_writers"] != administration["output_writers"]
            or administration["host_policy"] != "main-q87-fixed500"):
        raise CachePolicyError("Current exact39 Build/config/input/runtime/layout and Main capacity scope differ")
    # A duplicate authority id must not silently replace a human decision.
    if not isinstance(authorities, list) or any(
            len([row for row in authorities if isinstance(row, dict) and row.get("id") == record]) != 1
            for record in release["records"]):
        raise CachePolicyError("Unique actual preparation and human GO records are required")
    for record, decider in release["records"].items():
        if next(row for row in authorities if isinstance(row, dict) and row.get("id") == record).get("decided_by") != decider:
            raise CachePolicyError("Actual source-bound current V9 authority decider differs")

    def role(use, descriptor):
        descriptor = _descriptor(descriptor)
        rows = [row["reference"] for row in catalog["inputs"] if use in row["uses"]]
        if (len(rows) != 1 or rows[0].get("role") != "configuration"
                or rows[0].get("sha256") != descriptor["sha256"]
                or type(rows[0].get("bytes")) is not int
                or rows[0]["bytes"] != descriptor["size_bytes"]):
            raise CachePolicyError("Exact named queued configuration SHA/bytes required: " + use)
        return rows[0]

    configuration_source = _descriptor(value["configuration_source"])
    if role("run.configuration", configuration_source) != first.task["configuration"]:
        raise CachePolicyError("The actual queued configuration differs from its named source")
    config = read(configuration_source)
    if (not isinstance(config, dict) or config.get("format") != "darsia-native-phase-n1-run-v1"
            or config.get("run") != scope["run"]):
        raise CachePolicyError("The actual queued current native run configuration is required")
    native_pin = _descriptor(value["native_parameters"])
    if (not isinstance(config.get("native"), dict)
            or native_pin != _descriptor(config["native"].get("configuration"))
            or native_pin["sha256"] != value["requested_native_layout"].get("native_configuration_sha256")):
        raise CachePolicyError("Current native parameters differ from queued config or requested layout")
    role("native.configuration", native_pin)
    native = read(native_pin)
    if not isinstance(native, dict) or not native:
        raise CachePolicyError("Actual queued native parameter metadata is missing")
    checked_required_capacity_bound(read(_descriptor(value["capacity_bound_plan"])))
    # Exact bounded bytes alone do not prove an original completed Build. Do
    # not import a numerical package for a partial schema check before startup.
    read(_descriptor(value["current_v9_build"]))
    checked_current_build_plan_metadata(config, value["current_v9_build"], catalog)
    # setup_terminal authenticates original publication/owner/lease. The full
    # load_build_physical_context verifier also decodes saved physical arrays;
    # it cannot run before admission and explicitly leaves per-frame geometry,
    # phase configuration and source registration to the later collector.
    # This joins DATA only. Original proof replay belongs after startup, and
    # the current source/eligibility/reference authority is still incomplete.
    raise CachePolicyError("Current V9 execution binding STOP: queued Build DATA is bound, but released "
        "original source/physical/N1 eligibility and reference acceptance are not implemented")


def checked_reference_administration(value, first, *, read, read_authorities=None):
    """A released reference build is an explicit producer entitlement, never a measured profile.

    Its authority ids must name one entry of REFERENCE_RELEASES exactly; that
    entry's answer records and plan SHA-256 must match (N43/Q109 is the first).
    """
    release = released_reference(value.get("authority_ids")) if isinstance(value, dict) else None
    retained = (release or {}).get("retained_cache")
    _exact_keys(value, {"format", "authority_ids", "answers_source", "accepted_plan",
        "scope", "task", "host", "host_policy", "owned_roots", "entry_limit",
        "aggregate_entitlement_bytes", "counting_semantics", "watchdog_period_seconds",
        "operational_reserve_bytes", "minimum_family_memory_limit_bytes",
        "memory_reserve_bytes", "all_external_writers_redirected", "exclusive_producer",
        "output_configuration", "requested_native_layout", "output_writers", "producer_state"}
        | ({"retained_cache_inventory"} if retained else set()), "Exact N43 reference administration required")
    if (value["format"] != REFERENCE_ADMINISTRATION
            or release is None
            or value["scope"] != first.provider["scope"]
            or value["scope"]["ordered_case_ids"] != REFERENCE_CASE_IDS
            or value["task"] != first.request["task"]
            or value["aggregate_entitlement_bytes"] != REFERENCE_ENTITLEMENT_BYTES
            or value["counting_semantics"] != "all-stages-products-intermediates-receipts-previews-caches-once"
            or value["watchdog_period_seconds"] != 1
            or value["operational_reserve_bytes"] != PHYSICAL_RESERVE_BYTES
            or value["minimum_family_memory_limit_bytes"] != 20*1024**3
            or value["memory_reserve_bytes"] != PHYSICAL_RESERVE_BYTES
            or value["all_external_writers_redirected"] is not True
            or value["exclusive_producer"] is not True
            or value["producer_state"] != "PREDECLARED_UNQUALIFIED_REFERENCE_BUILD"):
        raise CachePolicyError("N43 exact human scope, aggregate entitlement, reserves and one producer required")
    nonnegative(value["entry_limit"], "N43 finite accounting entry limit")
    if not value["entry_limit"]:
        raise CachePolicyError("Positive N43 finite accounting entry limit required")
    authorities=(read if read_authorities is None else read_authorities)(
        _descriptor(value["answers_source"]))
    ids={row.get("id"):row for row in authorities} if isinstance(authorities,list) else {}
    if any(ids.get(key,{}).get("decided_by")!=by for key,by in release["records"].items()):
        raise CachePolicyError("Actual released humanGO and authority source records required")
    argv=first.request["public_command"]
    def option(name, expected):
        positions=[i for i,arg in enumerate(argv) if arg==name]
        return len(positions)==1 and positions[0]+1<len(argv) and argv[positions[0]+1]==expected
    if (not option("--workers","1") or not option("--backend","cpu")
            or argv.count("--once")!=1 or "--watchdog" in argv):
        raise CachePolicyError("N43 preserves one original whole-run CPU worker/once and forbids automatic retry")
    _descriptor(value["accepted_plan"])
    if value["accepted_plan"]["sha256"]!=release["plan_sha256"]:
        raise CachePolicyError("The released humanGO names the exact retained plan "+release["plan_sha256"][:8])
    accepted_plan = read(value["accepted_plan"])
    if not isinstance(value["requested_native_layout"],dict) or not value["output_writers"]:
        raise CachePolicyError("Predeclared actual native layout and closed output-writer pins required")
    if value["output_configuration"]!=first.task.get("configuration"):
        raise CachePolicyError("Exact original queued run configuration required")
    host=value["host"].casefold() if isinstance(value["host"],str) else ""
    if (host,value["host_policy"]) not in {("reference-host","main-q87-fixed500"),
            ("second-host","second-n43-one-run-uncapped-d")}:
        raise CachePolicyError("Only explicit MainQ87 or human one-run SecondD policy is supported")
    roots=value["owned_roots"]
    roles=[row.get("role") for row in roots] if isinstance(roots,list) else []
    if (not {"cache","results","job"}<=set(roles) or len(set(roles))!=len(roles)
            or not set(roles)<={"cache","results","job","receipts","capacity_state","host_ledger"}):
        raise CachePolicyError("Explicit disjoint data and optional receipt/state/host-ledger roots required")
    for row in roots:
        _exact_keys(row,{"role","path","identity","volume_id","stat_identity"},"Actual preaccepted N43 root identity required")
        _exact_keys(row["stat_identity"],{"st_dev","st_ino","creation_ns"},"Actual no-follow root stat identity required")
        if any(type(v) is not int or v<=0 for v in row["stat_identity"].values()):
            raise CachePolicyError("Positive actual root device/inode/creation identity required")
        path=PureWindowsPath(row["path"])
        if not path.is_absolute() or not isinstance(row["identity"],dict) or not row["identity"] or not row["volume_id"]:
            raise CachePolicyError("Actual absolute owned namespace and volume binding required")
        if row["role"]=="job":
            if (not path.drive.startswith("\\") or row["identity"].get("format")!="original-j1-shared-folder-admission-v1"
                    or row["identity"].get("protocol") not in {"smb","nfs"}):
                raise CachePolicyError("N43 job retains the original positively admitted UNC RunIO queue route")
        if row["role"] in {"cache","results"} and host=="second-host" and path.drive.casefold()!="d:":
            raise CachePolicyError("Second-host one-run exception is confined to its explicitly assigned D namespace")
    by_role={row["role"]:row for row in roots}
    for option_name, expected in [("--job",by_role["job"]["path"]),("--stores",first.stores_source["path"])]:
        positions=[i for i,arg in enumerate(argv) if arg==option_name]
        if (len(positions)!=1 or positions[0]+1>=len(argv)
                or not _same_path(argv[positions[0]+1],expected)):
            raise CachePolicyError("N43 original public job/stores argv differs from the released namespaces")
    if not _same_path(argv[0],first.provider["python_executable"]["path"]):
        raise CachePolicyError("N43 exact selected public interpreter required")
    for i,row in enumerate(roots):
        for other in roots[i+1:]:
            left,right=PureWindowsPath(row["path"]),PureWindowsPath(other["path"])
            if left==right or left in right.parents or right in left.parents:
                raise CachePolicyError("N43 roots overlap; all-stage files must count exactly once")
    if retained:
        # The release pins the exact retained file set; admission verifies it on disk.
        pin=_descriptor(value["retained_cache_inventory"])
        if pin["sha256"]!=retained["inventory_sha256"]:
            raise CachePolicyError("The released retained cache is exactly inventory "+retained["inventory_sha256"][:8])
        inventory=read(pin)
        if (not isinstance(inventory,dict) or inventory.get("format")!=RETAINED_INVENTORY
                or inventory.get("no_cleanup") is not True
                or type(inventory.get("logical_bytes")) is not int
                or inventory["logical_bytes"]!=retained["logical_bytes"]
                or not _same_path(inventory.get("root",""),by_role["cache"]["path"])):
            raise CachePolicyError("Released retained inventory must count exactly its files in this cache root")
    if release.get("format") == CURRENT_V9_REFERENCE_RELEASE_FORMAT:
        checked_current_reference_plan(accepted_plan, release, value, first,
            read=read, authorities=authorities)
    return deepcopy(value)


def reference_namespace(first, lease):
    value=first.reference_administration
    roots={row["role"]:row for row in value["owned_roots"]}
    host=str(lease.context["host"]).casefold()
    if (value["host"].casefold()!=host
            or first.provider["output_namespace"]!=lease.admission["receipts"]
            or first.provider["namespace_admission"]!=lease.document["cache_bootstrap"]["namespace_admission"]
            or lease.policy.profiles.get("jobs")!=first.provider_binding
            or not _same_path(roots["cache"]["path"],lease.policy.root)
            or roots["cache"]["identity"]!=lease.admission["cache_identity"]):
        raise CachePolicyError("N43 actual host/cache/receipts/provider namespace differs")
    if value["host_policy"]=="main-q87-fixed500":
        if (lease.policy.volume_cap is None or lease.policy.quota_bytes!=500*1024**3
                or lease.policy.volume_cap.maximum_size_bytes!=500*1024**3
                or "49079372-0000-0000-0000-100000000000" not in lease.policy.volume_cap.volume_guid.casefold()
                or PureWindowsPath(lease.policy.root).drive.casefold()!="q:"):
            raise CachePolicyError("Main N43 requires the actual accepted fixed500GiB Q87 volume")
    elif lease.policy.volume_cap is not None:
        raise CachePolicyError("One-run second-host uncapped exception must report the actual absence of a hard cap")
    reference_writer_coverage(first,lease)
    validate_reference_roots(first)
    return True


def reference_root_for_path(first, path):
    target=PureWindowsPath(str(path))
    if not target.is_absolute() or ".." in target.parts:
        raise CachePolicyError("N43 writer path must be explicit absolute and normalized")
    matches=[row for row in first.reference_administration["owned_roots"]
        if target==PureWindowsPath(row["path"]) or PureWindowsPath(row["path"]) in target.parents]
    if len(matches)!=1:
        raise CachePolicyError("N43 actual writer is omitted or counted by more than one aggregate root: "+str(path))
    return matches[0]


def reference_writer_coverage(first, lease, *, writer_environment=False):
    """Actual receipt/control/state paths, not only a predeclared boolean."""
    paths={"receipts":lease.admission["receipts"]["path"],
        "host_ledger":lease.admission["host_ledger"]["path"],
        "capacity_state":lease.document["capacity"]["state_directory"]}
    explicit={row["role"]:row for row in first.reference_administration["owned_roots"]}
    coverage=[]
    for role,path in paths.items():
        row=reference_root_for_path(first,path)
        if role in explicit:
            if not _same_path(explicit[role]["path"],path):
                raise CachePolicyError("N43 extra control root differs from the actual "+role)
            if role in {"receipts","host_ledger"} and explicit[role]["identity"]!=lease.admission[role]["identity"]:
                raise CachePolicyError("N43 extra control root kernel identity differs")
        coverage.append(dict(writer_role=role,path=str(path),aggregate_root_role=row["role"]))
    if writer_environment:
        from .environment import FIRST_WRITER_ENVIRONMENT_KEYS
        for key in FIRST_WRITER_ENVIRONMENT_KEYS:
            path=os.environ.get(key)
            if key=="NUMBA_CACHE_LOCATOR_CLASSES":
                if path!="UserProvidedCacheLocator":
                    raise CachePolicyError("N43 requires the original explicit Numba user-provided locator")
                coverage.append(dict(writer_role=key,value=path,semantics="original locator selector, not a filesystem path"))
                continue
            if path is None:
                raise CachePolicyError("N43 actual writer environment not redirected: "+key)
            actual=PureWindowsPath(path);service=PureWindowsPath(str(lease.root))
            if actual!=service and service not in actual.parents:
                raise CachePolicyError("N43 actual writer environment escapes its existing service scope: "+key)
            row=reference_root_for_path(first,path)
            coverage.append(dict(writer_role=key,path=path,aggregate_root_role=row["role"]))
    return coverage


def _reference_directory_stat(path):
    """Existing no-follow lstat route; no directory handle held over live I/O."""
    import stat
    path=Path(path)
    for part in (path,*path.parents):
        info=part.lstat()
        if (not stat.S_ISDIR(info.st_mode) or getattr(info,"st_reparse_tag",0)
                or getattr(info,"st_file_attributes",0)&0x400):
            raise CachePolicyError("N43 observed non-directory/reparse ancestor refused")
    return path.lstat()


def _reference_stat_identity(info):
    return dict(st_dev=info.st_dev,st_ino=info.st_ino,
        creation_ns=getattr(info,"st_birthtime_ns",info.st_ctime_ns))


def validate_reference_roots(first):
    """Original immutable local admission, or original J1/RunIO SMB contract."""
    if first.reference_roots_validated:
        return
    from .windows import WindowsNamespace
    from dataclasses import asdict
    for row in first.reference_administration["owned_roots"]:
        if _reference_stat_identity(_reference_directory_stat(row["path"]))!=row["stat_identity"]:
            raise CachePolicyError("N43 actual root stat identity differs from its Root-bound namespace")
        if row["role"]=="job":
            from darsia.fluidflower.job_filesystem import inspect_shared_folder
            actual=inspect_shared_folder(row["path"])
            if not actual.allowed or actual.protocol!=row["identity"]["protocol"]:
                raise CachePolicyError("N43 original RunIO shared-folder admission differs")
            first.reference_transport_observation=asdict(actual)
        else:
            # Admit immutable root identity, then CLOSE before scanning live
            # descendants. Never hold shareREAD child handles over a rename.
            namespace=WindowsNamespace(row["path"],row["identity"])
            namespace.close()
    first.reference_roots_validated=True


def reference_scope_snapshot(first, lease):
    """No-follow length samples, identity before/after; not atomic/lifetime proof."""
    import stat
    from .capacity import volume_info
    from .bootstrap import _utc
    value=first.reference_administration
    total=entries=0;roots=[];volumes={};identities=set();terminal_metadata=[];anomalies=[]
    for row in value["owned_roots"]:
        root=Path(row["path"]);subtotal=0
        before=_reference_stat_identity(_reference_directory_stat(root))
        if before!=row["stat_identity"]:
            raise CachePolicyError("N43 actual aggregate root identity changed")
        def walk(directory):
            nonlocal entries,total,subtotal
            try:
                start=_reference_stat_identity(_reference_directory_stat(directory))
                pending=[]
                with os.scandir(directory) as stream:
                    for entry in stream:
                        entries+=1
                        if entries>value["entry_limit"]:
                            raise CachePolicyError("N43 finite aggregate accounting entry limit exceeded")
                        try:
                            info=Path(entry.path).lstat()
                        except FileNotFoundError:
                            anomalies.append(dict(path=entry.path,state="VANISHED_DURING_SAMPLE"));continue
                        if getattr(info,"st_reparse_tag",0) or getattr(info,"st_file_attributes",0)&0x400:
                            raise CachePolicyError("N43 aggregate reparse entry refused")
                        if not stat.S_ISDIR(info.st_mode) and (not stat.S_ISREG(info.st_mode) or info.st_nlink!=1):
                            raise CachePolicyError("N43 aggregate file type/hardlink refused")
                        pending.append((entry.name,entry.path,info))
                # Close this directory's iterator before inspecting children or
                # descending; an ancestor monitor handle must not span recursion.
                for name,path,info in pending:
                    if stat.S_ISDIR(info.st_mode):
                        walk(Path(path))
                    else:
                        identity=(info.st_dev,info.st_ino)
                        if identity in identities:
                            # A live directory rename can expose the same
                            # one-link file twice. Count once and mark the
                            # non-atomic sample, never call this a peak.
                            anomalies.append(dict(path=path,state="IDENTITY_SEEN_TWICE_DURING_SAMPLE"));continue
                        identities.add(identity);total+=info.st_size;subtotal+=info.st_size
                        relative=Path(path).relative_to(root)
                        if name in {"candidate_completed.json","completed.json"} or (
                                row["role"]=="job" and relative==Path("done")/"ac52.json"):
                            terminal_metadata.append(dict(path=path,size_bytes=info.st_size,
                                role="original terminal/candidate metadata; not Root reference acceptance"))
                if _reference_stat_identity(_reference_directory_stat(directory))!=start:
                    anomalies.append(dict(path=str(directory),state="DIRECTORY_IDENTITY_CHANGED_DURING_SAMPLE"))
            except FileNotFoundError:
                anomalies.append(dict(path=str(directory),state="DIRECTORY_MOVED_DURING_SAMPLE"))
        walk(root)
        after=_reference_stat_identity(_reference_directory_stat(root))
        if before!=after:
            raise CachePolicyError("N43 aggregate root changed during length sample")
        volume=volume_info(row["path"])
        if volume["volume_id"]!=row["volume_id"]:
            raise CachePolicyError("N43 actual owned-root volume identity changed")
        volumes[volume["volume_id"]]=volume
        roots.append(dict(role=row["role"],path=row["path"],identity=row["identity"],
            stat_identity_before=before,stat_identity_after=after,logical_bytes=subtotal,
            accounting_scope=("entire accepted shared host-ledger directory including retained legacy control bytes; conservative, not exclusive producer-only"
                if row["role"]=="host_ledger" else "entire explicitly inventoried root")))
    return dict(utc=_utc(),logical_bytes=total,entries=entries,roots=roots,volumes=list(volumes.values()),
        entitlement_bytes=REFERENCE_ENTITLEMENT_BYTES,all_stage_files_counted_once=True,
        sampled=True,non_atomic=True,complete_lifetime_peak=False,
        observed_scan_anomalies=anomalies,observed_bytes_are_lower_bound_if_scan_anomalies=bool(anomalies),
        hard_cap_exists=value["host_policy"]=="main-q87-fixed500",
        shared_job_transport=first.reference_transport_observation,
        actual_writer_path_coverage=reference_writer_coverage(first,lease),
        actual_terminal_metadata_files=terminal_metadata)


def _reference_stop_name(first):
    return "n43-stop-"+first._once_key()+".json"


@contextmanager
def _reference_lock(lease):
    if lease.ledger.lock_active:
        yield
    else:
        with lease.ledger.locked(create_binding=False):
            yield


def reference_stopped(first, lease):
    with _reference_lock(lease):
        return _reference_stop_name(first) in lease.ledger.control.names()


def require_reference_not_stopped(first, lease):
    if reference_stopped(first,lease):
        raise CachePolicyError("N43 persistent STOP: no completed/reference promotion; partial files retained")


def latch_reference_stop(first, lease, reason, snapshot):
    name=_reference_stop_name(first)
    with _reference_lock(lease):
        if name not in lease.ledger.control.names():
            lease.ledger.control.write_once(name,_encoded(dict(format="darsia-n43-reference-producer-stop-v1",
                request=first.binding,provider=first.provider_binding,scope=first.provider["scope"],
                reason=reason,observation=snapshot,no_retry=True,no_reference_promotion=True,
                partial_products_retained=True)))
    return request_existing_frame_boundary_stop(first,lease)


def fence_reference_publication(first, lease):
    """Read the same persistent STOP and fresh aggregate at existing I/O seams."""
    with lease.ledger.locked(create_binding=False):
        try:
            snapshot=reference_scope_snapshot(first,lease)
        except BaseException as error:
            latch_reference_stop(first,lease,"AGGREGATE_OBSERVATION_UNAVAILABLE",
                dict(error_type=type(error).__name__,message=str(error)))
            raise
        if snapshot["observed_scan_anomalies"]:
            latch_reference_stop(first,lease,"PUBLICATION_SCAN_INCOMPLETE",snapshot)
        if snapshot["logical_bytes"]>REFERENCE_ENTITLEMENT_BYTES:
            latch_reference_stop(first,lease,"AGGREGATE_ENTITLEMENT_EXCEEDED",snapshot)
        if any(v["free_bytes"]<=PHYSICAL_RESERVE_BYTES for v in snapshot["volumes"]):
            latch_reference_stop(first,lease,"OPERATIONAL_RESERVE_CROSSED",snapshot)
        require_reference_not_stopped(first,lease)
    return snapshot


class ReferencePublicationFence:
    """Delegate the actual publisher; refuse observed STOP before promotion."""
    def __init__(self, original, first, lease):
        self.original,self.first,self.lease=original,first,lease

    def __call__(self, *args, **kwargs):
        success=kwargs.get("success",args[2] if len(args)>2 else None)
        if type(success) is not bool:
            raise CachePolicyError("Original publication success boolean required")
        if success:
            fence_reference_publication(self.first,self.lease)
        result=self.original(*args,**kwargs)
        if success:
            # Original shared prepare files stay diagnostic on STOP.
            fence_reference_publication(self.first,self.lease)
        # A plain new descriptor dict: no mutation or delayed dict side effect.
        # Failed JSON retains the original traceback; it is diagnostic only.
        return dict(result,n43_reference_guard=reference_publication_binding(self.first,self.lease))

    def publish_progress(self, *args, **kwargs):
        fence_reference_publication(self.first,self.lease)
        return self.original.publish_progress(*args,**kwargs)


def reference_publication_binding(first, lease):
    return dict(format="darsia-n43-reference-publication-guard-v1",request=first.binding,
        provider=first.provider_binding,stores_source=first.stores_source,
        task=first.request["task"],attempt_id=first.task["attempt_id"],commit=first.task["commit"],
        job_root=next(row["path"] for row in first.reference_administration["owned_roots"] if row["role"]=="job"),
        namespace_admission=lease.document["cache_bootstrap"]["namespace_admission"],
        aggregate_namespace_sha256=_digest(first.reference_administration["owned_roots"]))


def check_reference_publication_guard(guard, job_root, task, *, success):
    """Initial and recovery publication need the same actual managed context."""
    if type(success) is not bool:
        raise CachePolicyError("Original publication success boolean required")
    # Ordinary records/default paths retain their old route, without opening a
    # stores/provider or silently bootstrapping any managed service.
    if guard is None and REQUEST_ENV not in os.environ:
        return None
    from .bootstrap import current_import_lease
    lease=current_import_lease()  # Missing managed context is a refusal, never fallback.
    first=getattr(lease,"first_qualification",None)
    if guard is None:
        if first is not None and first.reference_build:
            actual=next(row["path"] for row in first.reference_administration["owned_roots"] if row["role"]=="job")
            if _same_path(actual,job_root):
                raise CachePolicyError("N43 publication/recovery record is missing its typed guard")
        return None
    _exact_keys(guard,{"format","request","provider","stores_source","task","attempt_id",
        "commit","job_root","namespace_admission","aggregate_namespace_sha256"},
        "Exact N43 publication guard required")
    if (first is None or not first.reference_build or lease.entry_kind!="worker" or lease.purpose!="jobs"
            or guard!=reference_publication_binding(first,lease)
            or lease.source!=first.stores_source or task!=first.task
            or guard["attempt_id"]!=task.get("attempt_id") or guard["commit"]!=task.get("commit")
            or not _same_path(guard["job_root"],job_root)):
        raise CachePolicyError("N43 publication/recovery actual source/request/task/namespace differs")
    first._namespace(lease)
    first.require_task(lease,task)
    if success:
        return fence_reference_publication(first,lease)
    # Preserve the primary traceback in original failed.json even after STOP.
    # This is not scientific product, completed/done or reference publication.
    return dict(state="DIAGNOSTIC_FAILURE_ONLY",no_reference_or_profile_promotion=True)
