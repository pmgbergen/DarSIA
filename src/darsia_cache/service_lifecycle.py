"""Durable service registration and proof-gated retention eligibility.

No deletion, source/data hashing, reservation release, process control or inferred
retirement. The original external observer must verify every registered owner
and owned launch tree. External-writer ownership remains a separate contract.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import re

from .control import _encoded, _json, _now, _owner, _time, failure_cleanup_due
from .policy import CachePolicyError, _descriptor


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _incarnation(value):
    owner = _owner(value)
    return {key: owner[key] for key in ("pid", "process_create_time")}


class ServiceLifecycle:
    """Use only the original service token and admitted host control namespace."""

    def __init__(self, lease):
        self.lease = lease
        self.token = lease.reservation["token"]
        self.prefix = "cache-l-" + self.token + "-"
        self.binding = dict(reservation=deepcopy(lease.reservation),
            policy=deepcopy(lease.source), profile=deepcopy(lease.profile_binding),
            namespace=deepcopy(lease.document["cache_bootstrap"]["namespace_admission"]),
            manifest=deepcopy(lease.manifest))
        if (lease.manifest["reservation"] != lease.reservation
                or lease.manifest["root"] != str(lease.root)
                or lease.manifest["entry_kind"] != lease.entry_kind
                or lease.manifest["role_identifier"] != lease.role_identifier):
            raise CachePolicyError("Lifecycle differs from original service binding")

    def _records(self):
        ledger = self.lease.ledger
        ledger.require_reservation(self.lease.reservation,
            self.lease.manifest["owner"], self.lease.maximum)
        names = [name for name in ledger.control.names() if name.casefold().startswith(self.prefix.casefold())]
        if len(names) > self.lease.entry_limit:
            raise CachePolicyError("Explicit service accounting entry limit exceeded")
        records = {}
        for name in names:
            row, raw = _json(ledger.control, name)
            if (row.get("format") != "darsia-cache-service-lifecycle-v1"
                    or row.get("binding") != self.binding
                    or row.get("kind") not in {"owner", "intent", "observed", "retired", "legacy_intent", "legacy_observed", "seal", "eligibility"}
                    or name != self.prefix + row["kind"] + "-" + row["key"] + ".json"):
                raise CachePolicyError("Foreign or malformed service record preserved")
            if row["kind"] == "owner":
                if (row["key"] != _digest(_incarnation(row["value"]["owner"]))
                        or row["actor"] != row["value"]["owner"]):
                    raise CachePolicyError("Registered owner record differs from original incarnation")
            if row["kind"] in {"intent", "observed", "retired"}:
                if (row["key"] != _digest({key: row["value"][key] for key in ("slot", "generation")})
                        or _incarnation(row["actor"]) != _incarnation(self.lease.manifest["owner"])):
                    raise CachePolicyError("Launch record differs from original owner/slot/generation")
            if row["kind"] in {"legacy_intent", "legacy_observed"}:
                if (row["key"] != _digest(dict(legacy_worker_id=row["value"]["worker_id"],
                        launcher=_incarnation(row["actor"])))
                        or _incarnation(row["actor"]) != _incarnation(self.lease.manifest["owner"])):
                    raise CachePolicyError("Legacy launch record differs from original launcher/worker ID")
            if row["kind"] == "eligibility" and row["actor"] != row["value"].get("observer"):
                raise CachePolicyError("Eligibility record actor differs from actual observer binding")
            records[(row["kind"], row["key"])] = dict(document=row,
                binding=dict(path=str(Path(ledger.bound["path"]) / name),
                             sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)))
        return records

    def _put(self, records, kind, key, value, *, actor=None):
        actor = _owner(self.lease.owner if actor is None else actor)
        if re.fullmatch(r"[0-9a-f]{64}|service", key) is None:
            raise CachePolicyError("Exact lifecycle record key required")
        previous = records.get((kind, key))
        if previous is not None:
            row = previous["document"]
            if row["value"] != value or row["actor"] != actor:
                raise CachePolicyError("A different lifecycle record already exists")
            return previous["binding"]
        if len(records) >= self.lease.entry_limit:
            raise CachePolicyError("Explicit service accounting entry limit reached; no new record")
        row = dict(format="darsia-cache-service-lifecycle-v1", binding=self.binding,
            kind=kind, key=key, utc=_now(), actor=deepcopy(actor), value=deepcopy(value))
        raw = _encoded(row)
        name = self.prefix + kind + "-" + key + ".json"
        self.lease.ledger.control.write_once(name, raw)
        return dict(path=str(Path(self.lease.ledger.bound["path"]) / name),
                    sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))

    def _primary(self):
        if _incarnation(self.lease.owner) != _incarnation(self.lease.manifest["owner"]):
            raise CachePolicyError("Only original service owner may seal or register launches")

    def register_owner(self):
        """Atomic with seal; the lease owner came from the native early bootstrap."""
        with self.lease.ledger.locked():
            records = self._records()
            if ("seal", "service") in records:
                raise CachePolicyError("Service is sealed; no new owner attachment")
            value = dict(owner=deepcopy(self.lease.owner), registration_is_not_retirement=True)
            if "native_launch_context" in self.lease.manifest:
                from .owned_process import current_native_membership
                binding = current_native_membership()
                if binding != self.lease.manifest["native_launch_context"]:
                    raise CachePolicyError("Current owner differs from original native Job")
                value["native_job_membership"] = binding
            return self._put(records, "owner", _digest(_incarnation(self.lease.owner)), value)

    def record_launch(self, kind, details):
        """Every allowed intent shares the seal; legacy scope stays unqualified."""
        self._primary()
        aliases = {"worker_spawn_intent": "intent", "worker_spawn_observed": "observed",
                   "worker_children_retired": "retired", "legacy_spawn_intent": "legacy_intent",
                   "legacy_spawn_observed": "legacy_observed"}
        if kind not in aliases or not isinstance(details, dict):
            raise CachePolicyError("Original producer launch observation required")
        legacy = kind.startswith("legacy_")
        if legacy:
            worker_id = details.get("worker_id")
            if not isinstance(worker_id, str) or not worker_id:
                raise CachePolicyError("Exact original legacy worker ID required")
            key = _digest(dict(legacy_worker_id=worker_id, launcher=_incarnation(self.lease.owner)))
            intent_kind = "legacy_intent"
        else:
            slot, generation = details.get("slot"), details.get("generation")
            if type(slot) is not int or slot < 0 or re.fullmatch(r"[0-9a-f]{32}", str(generation)) is None:
                raise CachePolicyError("Exact original slot and generation required")
            key = _digest(dict(slot=slot, generation=generation))
            intent_kind = "intent"
        stage = aliases[kind]
        with self.lease.ledger.locked():
            records = self._records()
            if ("owner", _digest(_incarnation(self.lease.manifest["owner"]))) not in records:
                raise CachePolicyError("Original launch owner was not registered")
            if stage in {"intent", "legacy_intent"}:
                if ("seal", "service") in records:
                    raise CachePolicyError("Service is sealed; no new producer launch intent")
            elif (intent_kind, key) not in records:
                raise CachePolicyError("No original owned producer launch intent")
            if stage in {"observed", "legacy_observed"}:
                child = details.get("original_native_identity")
                if child is not None:
                    _incarnation(child)
                    if ("native_launch_context" in self.lease.manifest
                            and (not str(child.get("creation_filetime_100ns", "")).isdecimal()
                                 or child.get("native_launch_context") !=
                                    self.lease.manifest["native_launch_context"])):
                        raise CachePolicyError("Original native launch FILETIME/Job binding missing")
                elif details.get("epoch_unavailable") is not True:
                    raise CachePolicyError("Unbound launch must remain explicitly unresolved")
            if stage == "retired":
                observed = records.get(("observed", key))
                if observed is None or observed["document"]["value"].get("original_native_identity") is None:
                    raise CachePolicyError("Unknown launch incarnation; preserve unresolved intent")
                identities = details.get("original_child_identities")
                if not isinstance(identities, list) or not identities:
                    raise CachePolicyError("Original actual child identity observations required")
                original = _incarnation(observed["document"]["value"]["original_native_identity"])
                if (original not in [_incarnation(value) for value in identities]
                        or details.get("owned_launch_handle_retired") is not True):
                    raise CachePolicyError("Original launch handle/tree is not observed retired")
            return self._put(records, stage, key, details)

    def seal(self, *, logical_gate, outcome, utc, traceback_text=None, publication=None):
        """Prevent new attachments/launches; existing owners may still be alive."""
        self._primary()
        if not isinstance(logical_gate, str) or not logical_gate or outcome not in {"success", "failed"}:
            raise CachePolicyError("Explicit actual gate and success/failure outcome required")
        _time(utc)
        if outcome == "failed" and not isinstance(traceback_text, str):
            raise CachePolicyError("Literal failed-service evidence must survive retention")
        if outcome == "success":
            publication = _descriptor(publication)
        value = dict(logical_gate=logical_gate, outcome=outcome, utc=utc,
            traceback=traceback_text, publication=publication,
            process_retirement_claimed=False, service_budget_closed=False,
            external_writer_ownership_not_established=True)
        with self.lease.ledger.locked():
            records = self._records()
            owner = ("owner", _digest(_incarnation(self.lease.manifest["owner"])))
            if owner not in records:
                raise CachePolicyError("Original service owner was not registered")
            return self._put(records, "seal", "service", value)

    def retention_eligibility(self, *, now_utc, retirement, verify_retirement,
                              verify_publication=None, same_gate_success=None,
                              verify_same_gate_success=None):
        """Eligibility for a separate ownership finalizer, NOT permission to delete.

        verify_retirement must use the original actual process observer, not
        checkpoint.retirement_document alone. It receives the complete sealed
        registry and exact external evidence descriptor, and must prove every
        owner, pending launch, descendant and external-reader boundary closed.
        A live owner cannot certify its own retirement. Success/next-success
        verification must resolve original bound publication metadata itself.
        No supplied callback is a qualified production provider by definition.
        Legacy launches and uninstrumented runtime/tests children refuse final
        eligibility explicitly. Even an empty worker launch set still requires
        the original complete external child-coverage/retirement provider.
        Callbacks must be read-only and cannot acquire result/cache locks or
        invoke the supervisor's mutating retirement path inside this transaction.
        """
        retirement = _descriptor(retirement)
        if not callable(verify_retirement):
            raise CachePolicyError("Original external retirement verifier is required")
        with self.lease.ledger.locked():
            records = self._records()
            sealed = records.get(("seal", "service"))
            if sealed is None:
                raise CachePolicyError("Service must be durably sealed before retention review")
            owners = [row for (kind, _), row in records.items() if kind == "owner"]
            primary = ("owner", _digest(_incarnation(self.lease.manifest["owner"])))
            if not owners or primary not in records:
                raise CachePolicyError("Original registered service owner is missing")
            from .bootstrap import _owner as current_native_owner
            actual, _ = current_native_owner()
            if _incarnation(actual) in [_incarnation(row["document"]["value"]["owner"]) for row in owners]:
                raise CachePolicyError("A registered live owner cannot certify its own retirement")
            intents = {key for kind, key in records if kind == "intent"}
            if any(key not in intents for kind, key in records if kind in {"observed", "retired"}):
                raise CachePolicyError("Orphan launch observation preserved; no final eligibility")
            for key in intents:
                if ("observed", key) not in records or ("retired", key) not in records:
                    raise CachePolicyError("Unresolved launch intent preserved; no final eligibility")
            snapshot = dict(binding=self.binding, seal=sealed,
                owners=sorted(owners, key=lambda row: row["binding"]["path"]),
                launches=[dict(intent=records[("intent", key)],
                    observed=records[("observed", key)], retired=records[("retired", key)])
                    for key in sorted(intents)],
                legacy_launches=[row for (kind, _), row in sorted(records.items())
                    if kind in {"legacy_intent", "legacy_observed"}],
                child_coverage=dict(entry_kind=self.lease.entry_kind,
                    registered_launches_are_not_exhaustive_proof=True,
                    complete_external_child_coverage_provider_required=True))
            refusal = None
            if snapshot["legacy_launches"]:
                refusal = "Recorded legacy child lifecycle is UNQUALIFIED; no retention eligibility"
            elif self.lease.entry_kind in {"runtime", "tests"}:
                refusal = "Runtime/tests child coverage is UNQUALIFIED; empty registry is not no-child proof"
            if refusal is not None:
                error = CachePolicyError(refusal)
                error.registry_snapshot = deepcopy(snapshot)  # Includes every recorded legacy launch.
                raise error
            if verify_retirement(deepcopy(snapshot), dict(retirement)) is not True:
                raise CachePolicyError("Actual complete original owner/launch/reader retirement not proven")
            outcome = sealed["document"]["value"]
            later = None
            if same_gate_success is not None:
                same_gate_success = _descriptor(same_gate_success)
                if not callable(verify_same_gate_success):
                    raise CachePolicyError("Original same-gate success verifier is required")
                later = verify_same_gate_success(deepcopy(snapshot), dict(same_gate_success))
                if not isinstance(later, dict):
                    raise CachePolicyError("Verified exact same-gate success document required")
            if outcome["outcome"] == "success":
                if (not callable(verify_publication)
                        or verify_publication(deepcopy(snapshot), outcome["publication"]) is not True):
                    raise CachePolicyError("Original required publication remains unverified")
                _time(now_utc)
                if _time(now_utc) < _time(outcome["utc"]):
                    raise CachePolicyError("Clock precedes sealed service outcome")
                due = True
            else:
                due = failure_cleanup_due(outcome, now_utc=now_utc,
                    retention_hours=self.lease.policy.failed_retention_hours, same_gate_success=later)
            value = dict(seal=sealed["binding"], registry_sha256=_digest(snapshot),
                observer=deepcopy(actual),
                retirement=retirement, same_gate_success=same_gate_success, now_utc=now_utc,
                retention_due=due, original_retirement_verifier_accepted=True,
                cleanup_authorized=False, service_budget_closed=False,
                external_writer_owned_inventory_and_final_cleanup_still_required=True)
            record = self._put(records, "eligibility", _digest(value), value, actor=actual)
            return dict(value, receipt=record)
