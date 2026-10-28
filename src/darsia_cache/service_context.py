"""Resolve an original service from exact metadata, never from dead handles.

Resolution is read-only. It neither starts/attaches a producer nor establishes
retirement or ownership of externally generated files. A later explicitly
invoked lifecycle review may write only its exact observer eligibility record.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

from .bootstrap import _explicit_document, _outside, _owner, select_entry_role, _exact_authority_list
from .capacity import _json_bytes, read_bound
from .control import CacheBudgetLedger, _encoded, _json, _owner as checked_owner, _token
from .policy import CachePolicyError, _descriptor, cache_settings, nonnegative
from .windows import WindowsNamespace, _parts


def _same_path(left, right):
    return str(PureWindowsPath(left)).casefold() == str(PureWindowsPath(right)).casefold()


def _host_binding(admitted, actual):
    return (admitted.get("host") == actual["host"]
        and actual["service_account_sid"] in admitted.get("authorized_sids", [])
        and _same_path(admitted["host_ledger"]["path"], actual["common_root"]))


def _verify_original_directory(path, identity):
    namespace = WindowsNamespace(path, identity)  # Existing IDs, no create/enumeration.
    namespace.close()


class _MetadataOnce:
    def __init__(self, package_file, *, authority_reader=None):
        self.package_file, self.seen = package_file, {}
        self.authority_reader = authority_reader

    def remember(self, pin, document):
        key = str(PureWindowsPath(pin["path"])).casefold()
        if key in self.seen and self.seen[key][0] != pin:
            raise CachePolicyError("One original metadata path has conflicting descriptors")
        self.seen[key] = (deepcopy(pin), deepcopy(document))

    def read(self, binding):
        pin = _descriptor(binding)
        key = str(PureWindowsPath(pin["path"])).casefold()
        if key in self.seen:
            original, document = self.seen[key]
            if original != pin:
                raise CachePolicyError("Original metadata descriptor changed during resolution")
            return deepcopy(document)
        _outside(pin["path"], self.package_file)
        document, _ = read_bound(pin)
        self.remember(pin, document)
        return document


    def read_authorities(self, binding):
        pin = _descriptor(binding)
        key = str(PureWindowsPath(pin["path"])).casefold()
        if key in self.seen:
            original, document = self.seen[key]
            if original != pin:
                raise CachePolicyError("Original authority descriptor changed during resolution")
            if not isinstance(document, list) or any(not isinstance(row, dict) for row in document):
                raise CachePolicyError("Authority metadata must be a list of record objects")
            return deepcopy(document)
        document = (_exact_authority_list(pin, self.package_file) if self.authority_reader is None
            else self.authority_reader(pin))
        self.remember(pin, document)
        return deepcopy(document)


class _EligibilityMetadataOnly:
    """No producer records or host-binding initialization through this view."""
    def __init__(self, namespace, binding, observer):
        self.namespace, self.binding, self.observer = namespace, binding, observer

    def names(self):
        return self.namespace.names()

    def read_once(self, name, maximum=1024**2):
        return self.namespace.read_once(name, maximum=maximum)

    def write_once(self, name, raw, maximum=1024**2):
        row = _json_bytes(raw)
        value = row.get("value", {})
        if (row.get("format") != "darsia-cache-service-lifecycle-v1"
                or row.get("kind") != "eligibility" or row.get("binding") != self.binding
                or row.get("actor") != self.observer or value.get("observer") != self.observer
                or value.get("cleanup_authorized") is not False
                or value.get("service_budget_closed") is not False):
            raise CachePolicyError("Observer context permits only original-token eligibility evidence")
        key = hashlib.sha256(_encoded(value)).hexdigest()
        expected = "cache-l-" + self.binding["reservation"]["token"] + "-eligibility-" + key + ".json"
        if name != expected or row.get("key") != key:
            raise CachePolicyError("Eligibility evidence differs from original service/actual observer")
        return self.namespace.write_once(name, raw, maximum=maximum)


class _ExistingObserverLedger(CacheBudgetLedger):
    """Fresh existing-binding transactions; never initialize or allocate."""
    def __init__(self, *args, service_binding, observer, **kwargs):
        super().__init__(*args, **kwargs)
        self.service_binding, self.observer = deepcopy(service_binding), deepcopy(observer)

    @contextmanager
    def locked(self, *, create_binding=False):
        if create_binding is not False:
            raise CachePolicyError("Observer context cannot initialize a host binding")
        with super().locked(create_binding=False):
            namespace = self.control
            self.control = _EligibilityMetadataOnly(namespace, self.service_binding, self.observer)
            try:
                yield self
            finally:
                self.control = namespace  # Original finally closes this fresh handle.

    def _producer_operation_refused(self, *args, **kwargs):
        raise CachePolicyError("Read-only observer is not a producer/allocation/cleanup context")

    reserve = admit = bind_scope = close = _producer_operation_refused


@dataclass(frozen=True)
class OriginalServiceObserverContext:
    owner: dict
    context: dict
    document: dict
    source: dict
    policy: object
    profile: dict
    profile_binding: dict
    admission: dict
    manifest: dict
    reservation: dict
    root: Path
    directories: dict
    maximum: int
    entry_limit: int
    entry_kind: str
    role_identifier: str
    purpose: str
    receipt: dict
    ledger: object
    original_metadata: tuple
    resolver_is_not_retirement_proof: bool = True


def resolve_original_service_context(receipt_binding, *, package_file, first_request=None, retained_close=None,
                                     historical_authorities=None):
    """Reopen exact metadata once; no ImportLease constructor or live handles.

    Requires CacheBudgetLedger.locked(create_binding=False). The original
    policy/profile/namespace are metadata bindings, not fresh producer/runtime
    qualification. The actual native observer is distinct from the original
    primary owner. Full registered-owner/launch/reader retirement is still a
    separate mandatory provider; this function does not inspect/kill processes.
    """
    if historical_authorities is not None:
        from .bootstrap import _HistoricalAuthorityReader
        if not isinstance(historical_authorities, _HistoricalAuthorityReader):
            raise CachePolicyError("Exact Root-bound historical authority reader required")
    reader = None if historical_authorities is None else historical_authorities.for_original(first_request)
    reads = _MetadataOnce(package_file, authority_reader=reader)
    receipt_binding = _descriptor(receipt_binding)
    receipt = reads.read(receipt_binding)
    if receipt.get("format") != "darsia-cache-import-service-receipt-v1":
        raise CachePolicyError("Exact original import/service receipt required")
    policy_binding = _descriptor(receipt["policy"])
    policy_key = str(PureWindowsPath(policy_binding["path"])).casefold()
    if policy_key in reads.seen:
        raise CachePolicyError("Policy and service receipt cannot alias one metadata file")
    document, actual_source = _explicit_document(policy_binding["path"], package_file)
    if actual_source != policy_binding:
        raise CachePolicyError("Original service policy SHA/size/path differs")
    reads.remember(policy_binding, document)
    policy = cache_settings(document, required=True)
    role = select_entry_role(document, entry_kind=receipt["entry_kind"],
                             role_identifier=receipt["role_identifier"])
    if (receipt["profile"] != role["profile"]
            or receipt["namespace"] != document["cache_bootstrap"]["namespace_admission"]):
        raise CachePolicyError("Original service role/profile/namespace metadata differs")
    profile = reads.read(role["profile"])
    first=None
    if first_request is not None:
        from .first_qualification import FirstQualification, PROVIDER
        first_type = FirstQualification
        if profile.get("format") == "darsia-common-phase-qualified-provider-v1":
            from .common_qualification import CommonQualification
            first_type = CommonQualification
        elif profile.get("format") == "darsia-setup-service-provider-v1":
            from .setup_qualification import PositiveSetupServiceQualification
            first_type = PositiveSetupServiceQualification
        elif profile.get("format") == "darsia-setup-job-service-provider-v1":
            from .setup_session import SetupJobService
            first_type = SetupJobService
        elif profile.get("format") == "darsia-preset-wasserstein-first-provider-v1":
            from .preset_qualification import PresetQualification
            first_type = PresetQualification
        elif profile.get("format")!=PROVIDER:
            raise CachePolicyError("Explicit first observer requires the original immutable provider")
        first=first_type(first_request,provider_binding=role["profile"],
            stores_source=policy_binding,namespace=receipt["namespace"],read=reads.read,
            read_authorities=reads.read_authorities, historical_authorities=historical_authorities)
        admin=receipt.get("first_qualification",{})
        entitlement=admin.get("administrative_entitlement",{})
        if (admin.get("request")!=first.binding or admin.get("provider")!=role["profile"]
                or entitlement.get("measured_profile") is not False
                or entitlement.get("maximum_bytes")!=receipt["manifest"]["maximum_bytes"]):
            raise CachePolicyError("Original first service administrative receipt differs")
        maximum=nonnegative(entitlement["maximum_bytes"],"Original first service entitlement")
        if not maximum:raise CachePolicyError("Positive original first entitlement required")
        profile=dict(format="darsia-cache-first-entry-contract-v1",
            python_executable=first.provider["python_executable"],
            module_closure=first.provider["module_closure"],
            native_supervision=dict(startup=first.provider["startup"]))
    elif profile.get("format") in {"darsia-cache-qualified-profile-provider-v1",
            "darsia-cache-volume-bound-import-profile-v1", "darsia-cache-volume-bound-job-profile-v1",
            "darsia-common-phase-qualified-provider-v1", "darsia-common-phase-import-profile-v1",
            "darsia-common-phase-job-profile-v1", "darsia-setup-service-provider-v1",
            "darsia-setup-service-import-profile-v1", "darsia-preset-wasserstein-first-provider-v1",
            "darsia-preset-wasserstein-bounded-profile-v1"}:
        from .first_qualification import resolve_profile
        profile=resolve_profile(role["profile"],"import",reads.read,stores_source=policy_binding,
            read_authorities=reads.read_authorities)
    from .first_qualification import BOUND_IMPORT_PROFILE, checked_bound_import_context
    bounded = first is None and profile.get("format") == BOUND_IMPORT_PROFILE
    common = first is None and profile.get("format") == "darsia-common-phase-import-profile-v1"
    setup = first is None and profile.get("format") == "darsia-setup-service-import-profile-v1"
    preset = first is None and profile.get("format") == "darsia-preset-wasserstein-bounded-profile-v1"
    if preset:
        from .preset_qualification import checked_import_context
        checked_import_context(profile, purpose=role["purpose"], role_identifier=role["role_identifier"],
            entry_kind=role["entry_kind"], policy=policy, namespace=receipt["namespace"],
            machine_id=document.get("machine", {}).get("id"))
    if setup:
        from .setup_qualification import checked_import_context
        checked_import_context(profile, purpose=role["purpose"], role_identifier=role["role_identifier"],
            entry_kind=role["entry_kind"], policy=policy, namespace=receipt["namespace"])
    if common:
        from .common_qualification import checked_import_context
        checked_import_context(profile, purpose=role["purpose"], role_identifier=role["role_identifier"],
            entry_kind=role["entry_kind"], policy=policy, namespace=receipt["namespace"])
    if bounded:
        checked_bound_import_context(profile, purpose=role["purpose"], role_identifier=role["role_identifier"],
            entry_kind=role["entry_kind"], policy=policy, namespace=receipt["namespace"])
    if first is None and not bounded and not common and not setup and not preset and (profile.get("format") != "darsia-cache-import-profile-v1"
            or profile.get("purpose") != role["purpose"]
            or profile.get("role_identifier") != role["role_identifier"]
            or profile.get("entry_kind") != role["entry_kind"]
            or profile.get("writer_contract") != "cooperative-start-admission-v1"):
        raise CachePolicyError("Original import/service profile is not applicable")
    if first is None:
        maximum = nonnegative(profile["maximum_cache_bytes"], "Original service maximum")
        completed = reads.read(profile["completion"])
        if not maximum or completed.get("complete") is not True or completed.get("passed") is not True:
            raise CachePolicyError("Original completed import/service profile is required")
    admission = reads.read(receipt["namespace"])
    observer, actual = _owner()
    if not _host_binding(admission, actual):
        raise CachePolicyError("Actual observer host/account/common authority is not admitted")
    manifest = deepcopy(receipt["manifest"])
    original_owner = checked_owner(manifest["owner"])
    if original_owner.get("machine") != actual["host"]:
        raise CachePolicyError("Original service owner belongs to a different physical host")
    if ((observer["pid"], observer["process_create_time"])
            == (original_owner["pid"], original_owner["process_create_time"])):
        raise CachePolicyError("Original live primary is not an independent observer")
    reservation = deepcopy(manifest["reservation"])
    token = _token(reservation["token"])
    purpose = role["purpose"]
    container = policy.purpose(purpose) / ("current" if purpose == "tests" else "")
    root = container / token
    directories = deepcopy(receipt["directories"])
    if (manifest.get("format") != "darsia-cache-external-import-scope-v1"
            or manifest.get("token") != token or not _same_path(manifest["root"], root)
            or manifest["purpose"] != purpose or manifest["maximum_bytes"] != maximum
            or manifest["entry_kind"] != role["entry_kind"]
            or manifest["role_identifier"] != role["role_identifier"]
            or directories.get("") != manifest["root_identity"]
            or not _same_path(admission["purposes"][purpose]["path"], container)):
        raise CachePolicyError("Original service scope/purpose/root identity differs")
    for relative in directories:
        if relative:
            _parts(relative)
    expected_receipt = Path(admission["receipts"]["path"]) / (token + ".import-service.json")
    if not _same_path(receipt_binding["path"], expected_receipt):
        raise CachePolicyError("Service receipt is not in its original admitted receipt namespace")
    _verify_original_directory(admission["receipts"]["path"], admission["receipts"]["identity"])
    _verify_original_directory(root, manifest["root_identity"])
    entry_limit = nonnegative(document["cache_bootstrap"]["accounting_entry_limit"], "Original entry limit")
    if not entry_limit:
        raise CachePolicyError("Original positive accounting limit is required")
    binding = dict(reservation=reservation, policy=policy_binding,
        profile=role["profile"], namespace=receipt["namespace"], manifest=manifest)
    def no_cache_measurement(*args, **kwargs):
        raise CachePolicyError("Observer resolver does not measure or admit producer cache")
    ledger = _ExistingObserverLedger(policy, SimpleNamespace(**actual),
        measurement_provider=no_cache_measurement, namespace_admission=admission,
        verify_host_admission=_host_binding, service_binding=binding, observer=observer)
    with ledger.locked(create_binding=False):
        if retained_close is None:
            ledger.require_reservation(reservation, original_owner, maximum)
        else:
            # Q123 post-close observation only. The ordinary resolver continues
            # refusing closed reservations; this path cannot allocate or reopen.
            pin = _descriptor(retained_close)
            expected = Path(ledger.bound["path"]) / ("cache-c-" + token + ".json")
            if not _same_path(pin["path"], expected):
                raise CachePolicyError("Retained close is outside the original canonical ledger")
            row, raw = _json(ledger.control, "cache-r-" + token + ".json")
            close, close_raw = _json(ledger.control, expected.name)
            if (reservation != dict(token=token, pool=ledger.key, sha256=hashlib.sha256(raw).hexdigest())
                    or row.get("format") != "darsia-cache-budget-reservation-v2"
                    or row.get("owner") != original_owner or row.get("token") != token
                    or row.get("pool") != ledger.key or row.get("maximum_bytes") != maximum
                    or pin["sha256"] != hashlib.sha256(close_raw).hexdigest() or pin["size_bytes"] != len(close_raw)
                    or close.get("owner") != original_owner or close.get("token") != token
                    or close.get("reservation_sha256") != reservation["sha256"]
                    or close.get("no_further_writes") is not True):
                raise CachePolicyError("Actual retained reservation/close owner/token/SHA differs")
        original_scope, _ = _json(ledger.control, "cache-s-" + token + ".json")
        if original_scope != manifest:
            raise CachePolicyError("Original bound service manifest differs from current ledger")
    if ledger.control is not None or ledger.lock_active:
        raise CachePolicyError("Resolver did not close its freshly opened control handles")
    return OriginalServiceObserverContext(owner=deepcopy(observer), context=deepcopy(actual),
        document=deepcopy(document), source=deepcopy(policy_binding), policy=policy,
        profile=deepcopy(profile), profile_binding=deepcopy(role["profile"]), admission=deepcopy(admission),
        manifest=manifest, reservation=reservation, root=root, directories=directories,
        maximum=maximum, entry_limit=entry_limit, entry_kind=role["entry_kind"],
        role_identifier=role["role_identifier"], purpose=purpose, receipt=deepcopy(receipt_binding),
        ledger=ledger, original_metadata=tuple(deepcopy(pin) for pin, _ in reads.seen.values()))


class _RetainedCloseMetadataOnly:
    """Only the original canonical close record; never cleanup or allocation."""
    def __init__(self, namespace, reservation, owner):
        self.namespace,self.reservation,self.owner=namespace,reservation,owner

    def names(self):
        return self.namespace.names()

    def read_once(self, name, maximum=1024**2):
        return self.namespace.read_once(name,maximum=maximum)

    def write_once(self, name, raw, maximum=1024**2):
        row=_json_bytes(raw)
        if (name!="cache-c-"+self.reservation["token"]+".json"
                or set(row)!={"owner","token","utc","no_further_writes","reservation_sha256"}
                or row["owner"]!=self.owner or row["token"]!=self.reservation["token"]
                or row["reservation_sha256"]!=self.reservation["sha256"]
                or row["no_further_writes"] is not True or not isinstance(row["utc"],str)):
            raise CachePolicyError("Retained close permits only the exact original reservation close")
        return self.namespace.write_once(name,raw,maximum=maximum)


class _RetainedRegistryControl:
    """Read the old registry only while its current canonical lock is held."""
    def __init__(self, ledger):
        self.ledger = ledger

    def names(self):
        self.ledger._require_lock()
        prefix = "cache-l-" + self.ledger.original.reservation["token"] + "-"
        return [name for name in self.ledger.current.control.names()
                if name.casefold().startswith(prefix.casefold())]

    def read_once(self, name, maximum=1024**2):
        self.ledger._require_lock()
        prefix = "cache-l-" + self.ledger.original.reservation["token"] + "-"
        if not name.casefold().startswith(prefix.casefold()):
            raise CachePolicyError("Retained registry view only reads the original service registry")
        return self.ledger.current.control.read_once(name, maximum=maximum)

    def write_once(self, *args, **kwargs):
        raise CachePolicyError("Retained registry view cannot write or allocate")


class _RetainedRegistryLedger:
    """Post-close reads; never reopen the original reservation or its ledger."""
    def __init__(self, original, current, close_record):
        self.original, self.current = original, current
        self.close_record = _descriptor(close_record)
        self.bound, self.key = deepcopy(original.ledger.bound), original.ledger.key
        self.control = _RetainedRegistryControl(self)
        self._require_lock()

    def _require_lock(self):
        old, ledger = self.original, self.current
        if not isinstance(ledger, CacheBudgetLedger):
            raise CachePolicyError("Actual current canonical cache ledger required")
        ledger._require_lock()
        if (ledger.control is None or ledger is old.ledger
                or ledger.bound != self.bound or ledger.key != self.key
                or not _same_path(ledger.policy.root, old.policy.root)
                or ledger.capacity.host != old.context["host"]
                or ledger.admission.get("host") != old.context["host"]
                or ledger.admission.get("host_ledger") != old.ledger.admission.get("host_ledger")
                or ledger.admission.get("cache_identity") != old.ledger.admission.get("cache_identity")):
            raise CachePolicyError("Post-close registry requires the same locked canonical host/cache identity")
        token = old.reservation["token"]
        name = "cache-c-" + token + ".json"
        if not _same_path(self.close_record["path"], Path(self.bound["path"]) / name):
            raise CachePolicyError("Pinned old close is outside the current canonical ledger")
        close, raw = _json(ledger.control, name)
        if (self.close_record["sha256"] != hashlib.sha256(raw).hexdigest()
                or self.close_record["size_bytes"] != len(raw)
                or set(close) != {"owner", "token", "utc", "no_further_writes", "reservation_sha256"}
                or close["owner"] != old.manifest["owner"] or close["token"] != token
                or close["reservation_sha256"] != old.reservation["sha256"]
                or close["no_further_writes"] is not True):
            raise CachePolicyError("Pinned original close owner/token/reservation SHA differs")
        from .control import _time
        _time(close["utc"])

    def require_reservation(self, reservation, owner, needed):
        self._require_lock()
        old = self.original
        if (reservation != old.reservation or checked_owner(owner) != old.manifest["owner"]
                or needed != old.maximum):
            raise CachePolicyError("Post-close view can only read the exact original service reservation")
        row, raw = _json(self.current.control, "cache-r-" + reservation["token"] + ".json")
        if (set(row) != {"format", "pool", "utc", "owner", "token", "maximum_bytes"}
                or row["format"] != "darsia-cache-budget-reservation-v2"
                or row["owner"] != old.manifest["owner"] or row["token"] != reservation["token"]
                or row["pool"] != self.key or row["maximum_bytes"] != old.maximum
                or reservation != dict(token=row["token"], pool=self.key, sha256=hashlib.sha256(raw).hexdigest())):
            raise CachePolicyError("Closed original reservation bytes/schema differ")
        from .control import _time
        _time(row["utc"])
        scope, _ = _json(self.current.control, "cache-s-" + row["token"] + ".json")
        if scope != old.manifest:
            raise CachePolicyError("Closed original service manifest differs")
        return row, raw

    def _mutator_refused(self, *args, **kwargs):
        raise CachePolicyError("Retained registry view cannot write or allocate")

    locked = reserve = admit = bind_scope = close = project = _mutator_refused


class RetainedServiceCloseContext:
    """Explicit Q123 append-only use of the existing close API, no ImportLease.

    Every original cache file is retained. This is not cleanup eligibility,
    scientific success, rearm authority or an assertion of zero retained bytes.
    """
    def __init__(self, binding, *, package_file, retained_close=None, historical_authorities=None):
        from .first_qualification import _digest, _exact_keys
        self.reads=_MetadataOnce(package_file)
        self.binding=_descriptor(binding)
        self.request=self.reads.read(self.binding)
        roles={"original_service_receipt","original_request","original_native_retirement",
            "original_lifecycle_eligibility","recorded_owner_retirement","retained_cache_inventory",
            "answers_source","root_release"}
        _exact_keys(self.request,roles|{"format","authority_ids"},"Exact retained-service close request required")
        if (self.request["format"]!="darsia-cache-retained-service-close-request-v1"
                or self.request["authority_ids"]!=["Q123"]):
            raise CachePolicyError("Explicit technical retained close authority required")
        for role in roles:_descriptor(self.request[role])
        release=self.reads.read(self.request["root_release"])
        if (release.get("format")!="darsia-cache-retained-service-close-root-release-v1"
                or release.get("execution_authorized") is not True
                or release.get("request_payload_sha256")!=_digest({key:value for key,value in self.request.items() if key!="root_release"})):
            raise CachePolicyError("Actual Root release for this retained close required")
        if "historical_authority_resolution" in release:
            from .bootstrap import _HistoricalAuthorityReader
            if historical_authorities is not None:
                if not isinstance(historical_authorities, _HistoricalAuthorityReader):
                    raise CachePolicyError("Exact Root-bound historical authority reader required")
                historical_authorities.for_original(self.binding)
            historical_authorities = _HistoricalAuthorityReader(
                release["historical_authority_resolution"], package_file)
            self.reads.authority_reader = historical_authorities.current_authorities
        elif historical_authorities is not None:
            from .bootstrap import _HistoricalAuthorityReader
            if not isinstance(historical_authorities, _HistoricalAuthorityReader):
                raise CachePolicyError("Exact Root-bound historical authority reader required")
            self.reads.authority_reader = historical_authorities.for_original(self.binding)
        answers=self.reads.read_authorities(self.request["answers_source"])
        required={"Q123-answer-option1-append-only-closeout-and-new-empty-exclusive-scope-retained-bytes-counted"}
        if not required.issubset({row.get("id") for row in answers}):
            raise CachePolicyError("Actual Q123 authority required")
        self.original=resolve_original_service_context(self.request["original_service_receipt"],
            package_file=package_file,first_request=self.request["original_request"],retained_close=retained_close,
            historical_authorities=historical_authorities)
        if self.original.entry_kind!="worker" or self.original.purpose!="jobs":
            raise CachePolicyError("Retained close is only the original worker/jobs service")
        self._seal=_digest(self.request)

    def retired_after_close_locked(self, ledger, close_record):
        """Revalidate retirement through the current held lock, without writes."""
        view = _RetainedRegistryLedger(self.original, ledger, close_record)
        reservation, _ = view.require_reservation(self.original.reservation,
            self.original.manifest["owner"], self.original.maximum)
        return self._retired_locked(reservation, registry_ledger=view)

    def _retired_locked(self, reservation, *, registry_ledger=None):
        from .service_lifecycle import ServiceLifecycle, _digest as lifecycle_digest
        from .first_qualification import _digest
        import psutil
        context=self.original
        if registry_ledger is not None and (not isinstance(registry_ledger, _RetainedRegistryLedger)
                or registry_ledger.original is not context):
            raise CachePolicyError("Only the exact read-only retained registry view may observe a closed service")
        if _digest(self.request)!=self._seal:
            raise CachePolicyError("Retained-close request changed in memory")
        if (reservation["owner"]!=context.manifest["owner"]
                or reservation["token"]!=context.reservation["token"]):
            raise CachePolicyError("Retained close owner/token differ")
        retirement=self.reads.read(self.request["original_native_retirement"])
        observed=self.reads.read(self.request["recorded_owner_retirement"])
        actual=observed.get("actual",observed)
        rows=actual.get("recorded_owned_epochs",[])
        native=actual.get("original_whole_native_family_retirement",{})
        if (retirement.get("format")!="darsia-native-owned-worker-retirement-v1"
                or retirement.get("service_receipt")!=self.request["original_service_receipt"]
                or retirement.get("primary")!=context.manifest["owner"]
                or retirement.get("whole_family_kernel_observation") is not True
                or retirement.get("no_process_kill_after_resume") is not True
                or retirement.get("terminal",{}).get("accounting",{}).get("active_processes")!=0
                or retirement.get("terminal",{}).get("accounting",{}).get("terminated_by_limit_processes")!=0
                or retirement.get("terminal",{}).get("job_flags")!=0
                or {key:native.get(key) for key in ("path","sha256","size_bytes")}!=self.request["original_native_retirement"]
                or native.get("document")!=retirement or not rows):
            raise CachePolicyError("Original whole-kernel and recorded-owner retirement required")
        known=set()
        for row in rows:
            epoch=row.get("create_time")
            if (type(row.get("pid")) is not int or type(epoch) not in {int,float,str}
                    or row.get("recorded_epoch_absent") is not True):
                raise CachePolicyError("Every original exact owner epoch must be retired")
            known.add((row["pid"],str(epoch)))
            try:
                if str(psutil.Process(row["pid"]).create_time())==str(epoch):
                    raise CachePolicyError("An original retained-service owner is still live")
            except psutil.NoSuchProcess:pass
        registry_context = context if registry_ledger is None else replace(context, ledger=registry_ledger)
        lifecycle=ServiceLifecycle(registry_context);records=lifecycle._records()
        sealed=records.get(("seal","service"))
        owners=sorted((row for (kind,_),row in records.items() if kind=="owner"),key=lambda row:row["binding"]["path"])
        intents={key for kind,key in records if kind=="intent"}
        if (sealed is None or not owners or any(("observed",key) not in records or ("retired",key) not in records for key in intents)
                or any(key not in intents for kind,key in records if kind in {"observed","retired"})
                or any((row["document"]["value"]["owner"]["pid"],
                    str(row["document"]["value"]["owner"]["process_create_time"])) not in known for row in owners)):
            raise CachePolicyError("Original registered owner/launch retirement is incomplete")
        snapshot=dict(binding=lifecycle.binding,seal=sealed,owners=owners,
            launches=[dict(intent=records[("intent",key)],observed=records[("observed",key)],retired=records[("retired",key)]) for key in sorted(intents)],
            legacy_launches=[row for (kind,_),row in sorted(records.items()) if kind in {"legacy_intent","legacy_observed"}],
            child_coverage=dict(entry_kind=context.entry_kind,registered_launches_are_not_exhaustive_proof=True,
                complete_external_child_coverage_provider_required=True))
        eligibility=self.reads.read(self.request["original_lifecycle_eligibility"])
        value=eligibility.get("value",{})
        if (snapshot["legacy_launches"] or eligibility.get("binding")!=lifecycle.binding
                or eligibility.get("kind")!="eligibility"
                or records.get(("eligibility",eligibility.get("key")),{}).get("binding")!=self.request["original_lifecycle_eligibility"]
                or value.get("registry_sha256")!=lifecycle_digest(snapshot)
                or value.get("retirement")!=self.request["original_native_retirement"]
                or value.get("original_retirement_verifier_accepted") is not True
                or value.get("cleanup_authorized") is not False or value.get("service_budget_closed") is not False):
            raise CachePolicyError("Original sealed eligibility differs; no retained close")
        self._retained_files()
        return True

    def _retained_files(self):
        return verify_retained_cache_inventory(
            self.reads.read(self.request["retained_cache_inventory"]),self.original.policy.root)

    def close_retained(self):
        """One original cache-c append; a failure never deletes or retries it."""
        ledger=self.original.ledger
        with ledger.locked(create_binding=False):
            original=ledger.control
            # The original observer remains read-only everywhere else.
            ledger.control=_RetainedCloseMetadataOnly(original.namespace,
                self.original.reservation,self.original.manifest["owner"])
            try:
                CacheBudgetLedger.close(ledger,self.original.reservation,
                    owner=self.original.manifest["owner"],verify_no_further_writes=self._retired_locked)
                name="cache-c-"+self.original.reservation["token"]+".json"
                raw,_=ledger.control.read_once(name)
                close_record=dict(path=str(Path(ledger.bound["path"])/name),
                    sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
            finally:
                ledger.control=original
        return dict(format="darsia-cache-retained-service-close-result-v1",request=self.binding,
            reservation=deepcopy(self.original.reservation),close_record=close_record,
            original_service_receipt=self.request["original_service_receipt"],
            original_lifecycle_eligibility=self.request["original_lifecycle_eligibility"],
            original_native_retirement=self.request["original_native_retirement"],
            retained_cache_inventory=self.request["retained_cache_inventory"],all_files_retained=True,
            retained_bytes_are_still_used=True,cleanup_authorized=False,rearm_authorized=False)


def verify_retained_cache_inventory(inventory, root):
    """Exact old files remain; fresh admission separately counts the WHOLE root."""
    import stat
    if (inventory.get("format")!="darsia-cache-retained-file-inventory-v1"
            or not _same_path(inventory.get("root",""),root)
            or inventory.get("no_cleanup") is not True or not isinstance(inventory.get("files"),list)):
        raise CachePolicyError("Exact preserved cache-file inventory required")
    seen=set();total=0
    for row in inventory["files"]:
        relative=_parts(row["relative_path"])
        if tuple(relative) in seen:raise CachePolicyError("Duplicate retained file inventory")
        seen.add(tuple(relative));path=Path(root).joinpath(*relative)
        for ancestor in (path,*path.parents):
            info=ancestor.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info,"st_file_attributes",0)&0x400:
                raise CachePolicyError("Retained cache path is a reparse entry")
        info=path.lstat()
        actual=dict(dev=info.st_dev,ino=info.st_ino,size_bytes=info.st_size,mtime_ns=info.st_mtime_ns)
        if not stat.S_ISREG(info.st_mode) or actual!={key:row[key] for key in actual}:
            raise CachePolicyError("Retained cache-file identity/version changed")
        total+=info.st_size
    if total!=inventory.get("logical_bytes"):
        raise CachePolicyError("Retained cache-byte declaration differs")
    return deepcopy(inventory)
