"""Source preparation: handle-bound scratch and one admitted host ledger.

Only explicit write_bytes calls are bounded by this allocator. Third-party
writers are NOT covered merely by moving TEMP/NUMBA/MPL paths. No historical
cache adoption, path deletion, ACL installation or filesystem hard quota.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re

from .policy import CachePolicyError, budget_report, cache_settings, nonnegative
from .windows import (WindowsNamespace, CacheCleanupIncomplete,
    CacheObjectIncomplete, CacheHandleCloseIncomplete, _parts)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _time(value):
    when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if when.tzinfo is None:
        raise CachePolicyError("UTC offset required")
    return when.astimezone(timezone.utc)


def _owner(value):
    if (not isinstance(value, dict) or type(value.get("pid")) is not int or value["pid"] < 1
            or not isinstance(value.get("process_create_time"), str) or not value["process_create_time"]):
        raise CachePolicyError("Actual PID and string creation epoch required")
    return dict(value)


def _token(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise CachePolicyError("Exact fresh 32-hex allocation token required")
    return value


def _encoded(document):
    value = (json.dumps(document, sort_keys=True, ensure_ascii=False,
                        allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    if len(value) > 1024**2:
        raise CachePolicyError("Control metadata exceeds one MiB")
    return value


def _json(namespace, name):
    raw, _ = namespace.read_once(name)
    return json.loads(raw.decode("utf-8", "strict")), raw


def failure_cleanup_due(failed, *, now_utc, retention_hours, same_gate_success=None):
    if type(retention_hours) is not int or not 0 < retention_hours <= 48:
        raise CachePolicyError("Failed retention must be in (0,48] hours")
    when, now = _time(failed["utc"]), _time(now_utc)
    if now < when:
        raise CachePolicyError("Clock precedes retained failure")
    due = now >= when + timedelta(hours=retention_hours)
    if same_gate_success is not None:
        if (same_gate_success.get("complete") is not True
                or same_gate_success.get("logical_gate") != failed["logical_gate"]
                or not when <= _time(same_gate_success["utc"]) <= now):
            raise CachePolicyError("A later complete success of this exact gate is required")
        due = True
    return due


class OwnedScratch:
    """Fresh files only; every mutation is relative to pinned handles.

    Admission supplies accepted kernel IDs for the existing purpose and receipt
    roots. Receipts survive cleanup. No root creation or automatic migration.
    Namespace/ACL admission remains a separately reviewed host operation.
    """
    def __init__(self, policy, purpose, token, owner, logical_gate, file_caps,
                 *, namespace_admission, ledger, reservation):
        if purpose == "clones":
            raise CachePolicyError("Reusable Git trees are never disposable scratch")
        self.policy, self.token, self.owner = policy, _token(token), _owner(owner)
        if not isinstance(logical_gate, str) or not logical_gate:
            raise CachePolicyError("Named actual consumer required")
        if not isinstance(file_caps, dict) or not 0 < len(file_caps) <= 4096:
            raise CachePolicyError("Bounded exact file caps required")
        self.caps = {"/".join(_parts(k)): nonnegative(v, "File cap") for k, v in file_caps.items()}
        if len({k.casefold() for k in self.caps}) != len(self.caps):
            raise CachePolicyError("Case-ambiguous file declarations")
        if namespace_admission != ledger.admission or reservation.get("token") != token:
            raise CachePolicyError("Scratch namespace/token differs from original host reservation")
        ledger.require_reservation(reservation, self.owner, sum(self.caps.values()))
        ledger.require_unbound_scope(token)
        self.ledger, self.reservation = ledger, dict(reservation)
        bound = namespace_admission["purposes"][purpose]
        container = policy.purpose(purpose) / "current" if purpose == "tests" else policy.purpose(purpose)
        self.root = container / token
        if str(PureWindowsPath(bound["path"])).casefold() != str(PureWindowsPath(self.root.parent)).casefold():
            raise CachePolicyError("Purpose root differs from host admission")
        self.parent, self.control = None, None
        self.tree, self.outcome = None, None
        intent_written = False
        try:
            self.parent = WindowsNamespace(bound["path"], bound["identity"])
            self.control = WindowsNamespace(namespace_admission["receipts"]["path"],
                                            namespace_admission["receipts"]["identity"])
            self._record("intent", dict(token=token, owner=self.owner, logical_gate=logical_gate,
                reservation=dict(reservation), proposed_root=str(self.root),
                admitted_parent=bound, file_caps=self.caps, creation_not_yet_observed=True))
            intent_written = True
            self.tree = self.parent.scope(self.root.name, create=True)
            self.manifest = dict(format="darsia-owned-scratch-v2", token=token,
                owner=self.owner, logical_gate=logical_gate, purpose=purpose, root=str(self.root),
                temporary_file_caps=self.caps, root_identity=self.tree.directories[""],
                reservation=dict(reservation), native_products_are_temporaries=False)
            self._record("declared", self.manifest)
            ledger.bind_scope(reservation, self.manifest)
        except BaseException as error:
            record_error = None
            if intent_written:
                try:
                    self._record("create-incomplete", dict(complete=False, error=str(error),
                        created_evidence=getattr(error, "created_evidence", None)))
                except BaseException as failed_record:
                    record_error = failed_record
            try:
                self.close_handles()  # Preserve every partial object/reservation.
            except CacheHandleCloseIncomplete as close_error:
                raise error from close_error
            if record_error is not None:
                raise error from record_error
            raise

    def _record(self, stage, document):
        return self.control.write_once(self.token + "." + stage + ".json", _encoded(document))

    def write_bytes(self, relative, value):
        with self.ledger.locked():
            return self._write_bytes_locked(relative, value)

    def _write_bytes_locked(self, relative, value):
        if self.outcome is not None or relative not in self.caps:
            raise CachePolicyError("Closed or undeclared scratch write")
        self.ledger.require_reservation(self.reservation, self.owner, sum(self.caps.values()))
        self.ledger.fresh_free_guard()
        key = hashlib.sha256(relative.encode("utf-8")).hexdigest()
        self._record("write-intent-" + key, dict(path=relative, maximum_bytes=self.caps[relative],
                     parent_scope_identity=self.manifest["root_identity"]))
        try:
            identity = self.tree.write_once(relative, value, self.caps[relative])
            self.ledger.fresh_free_guard()
        except BaseException as error:
            self._record("write-incomplete-" + key, dict(complete=False, path=relative,
                literal_error=str(error), created_evidence=getattr(error, "created_evidence", None),
                unresolved_creates=self.tree.unresolved_creates,
                known_partial_identity=self.tree.files.get(relative)))
            raise
        self._record("write-" + key,
                     dict(path=relative, identity=identity))
        return self.root / relative

    def finish(self, outcome, *, utc, traceback_text=None, publication=None):
        with self.ledger.locked():
            return self._finish_locked(outcome, utc=utc, traceback_text=traceback_text, publication=publication)

    def _finish_locked(self, outcome, *, utc, traceback_text=None, publication=None):
        if self.outcome is not None or outcome not in {"success", "failed"}:
            raise CachePolicyError("Exactly one lifecycle outcome required")
        _time(utc)
        if outcome == "failed" and not isinstance(traceback_text, str):
            raise CachePolicyError("Literal failure traceback must survive cleanup")
        value = dict(outcome=outcome, utc=utc, logical_gate=self.manifest["logical_gate"],
                     owner=self.owner, traceback=traceback_text, publication=publication)
        self._record("frozen", self.tree.frozen())
        self._record("outcome", value)
        self.outcome = value
        self.tree.close()

    @classmethod
    def reopen(cls, policy, token, *, namespace_admission, ledger):
        """Reopen trusted receipts after a later process/retention interval."""
        self = cls.__new__(cls)
        self.policy, self.token, self.tree = policy, _token(token), None
        if ledger.admission != namespace_admission:
            raise CachePolicyError("Cleanup must use the original admitted host ledger")
        self.ledger = ledger
        self.parent = None
        self.control = WindowsNamespace(namespace_admission["receipts"]["path"],
                                        namespace_admission["receipts"]["identity"])
        try:
            self.manifest, _ = _json(self.control, token + ".declared.json")
            self.outcome, _ = _json(self.control, token + ".outcome.json")
            purpose = self.manifest["purpose"]
            self.owner, self.caps = _owner(self.manifest["owner"]), self.manifest["temporary_file_caps"]
            container = policy.purpose(purpose) / "current" if purpose == "tests" else policy.purpose(purpose)
            self.root = container / token
            if (self.manifest["format"] != "darsia-owned-scratch-v2" or self.manifest["token"] != token
                    or self.manifest["root"] != str(self.root) or purpose == "clones"
                    or self.outcome["owner"] != self.owner
                    or self.outcome["logical_gate"] != self.manifest["logical_gate"]):
                raise CachePolicyError("Frozen scope/consumer binding differs")
            bound = namespace_admission["purposes"][purpose]
            if str(PureWindowsPath(bound["path"])).casefold() != str(PureWindowsPath(self.root.parent)).casefold():
                raise CachePolicyError("Admitted purpose differs")
            self.parent = WindowsNamespace(bound["path"], bound["identity"])
            return self
        except BaseException:
            self.close_handles()
            raise

    def cleanup(self, *, now_utc, retirement, verify_retirement, verify_publication,
                same_gate_success=None, verify_same_gate_success=None):
        with self.ledger.locked():
            return self._cleanup_locked(now_utc=now_utc, retirement=retirement,
                verify_retirement=verify_retirement, verify_publication=verify_publication,
                same_gate_success=same_gate_success, verify_same_gate_success=verify_same_gate_success)

    def _cleanup_locked(self, *, now_utc, retirement, verify_retirement, verify_publication,
                        same_gate_success=None, verify_same_gate_success=None):
        if self.outcome is None or verify_retirement(self.manifest, retirement) is not True:
            raise CachePolicyError("Actual original owner and every reader must be retired")
        if self.outcome["outcome"] == "success":
            if verify_publication(self.manifest, self.outcome["publication"]) is not True:
                raise CachePolicyError("Original required collect/publication is incomplete")
        else:
            if same_gate_success is not None and (not callable(verify_same_gate_success)
                    or verify_same_gate_success(self.manifest, same_gate_success) is not True):
                raise CachePolicyError("Later success must bind a verified actual receipt of this gate")
            if self.outcome["outcome"] != "failed" or not failure_cleanup_due(
                    self.outcome, now_utc=now_utc, retention_hours=self.policy.failed_retention_hours,
                    same_gate_success=same_gate_success):
                raise CachePolicyError("Failure diagnostics remain retained")
        frozen, _ = _json(self.control, self.token + ".frozen.json")
        if (frozen["root"] != self.manifest["root_identity"]
                or any(k not in self.caps or v["bytes"] > self.caps[k] for k, v in frozen["files"].items())):
            raise CachePolicyError("Frozen ownership differs from declaration")
        self.tree = self.parent.scope(self.root.name, expected=frozen["root"])
        try:
            result = self.tree.cleanup(frozen)
        except CacheCleanupIncomplete as error:
            # A partial refusal is durable and cannot be mislabelled complete.
            self._record("cleanup-incomplete", dict(complete=False, utc=now_utc,
                         removed_owned=list(error.removed_owned), mutations=list(error.mutations),
                         close_failures=list(error.close_failures), error=str(error)))
            raise
        result.update(utc=now_utc, token=self.token, preserved_receipts=True)
        self._record("cleanup", result)
        return result

    def close_handles(self):
        failures = []
        for resource in (getattr(self, "tree", None), getattr(self, "parent", None), getattr(self, "control", None)):
            if resource is not None:
                try:
                    resource.close()
                except CacheHandleCloseIncomplete as error:
                    failures.extend(error.close_failures)
        if failures:
            raise CacheHandleCloseIncomplete(failures)


def reusable_checkout_admission(policy, *, checkout, expected_head, observation):
    expected = policy.purpose("clones") / "current"
    if (Path(checkout) != expected or observation.get("managed_root") != str(expected)
            or observation.get("registered_as_own_reusable_checkout") is not True
            or observation.get("clean") is not True or observation.get("readers_retired") is not True
            or not re.fullmatch(r"[0-9a-f]{40}", str(observation.get("head", "")))
            or not re.fullmatch(r"[0-9a-f]{40}", expected_head)):
        raise CachePolicyError("Foreign, dirty or live checkout preserved")
    return dict(root=str(expected), current_head=observation["head"], target_head=expected_head,
                update_needed=observation["head"] != expected_head,
                new_full_tree_copy_allowed=False, execution_released=False)


def require_volume_cap_binding(policy, admission):
    """Check the same accepted root/cap binding before launch and under lock."""
    cap = policy.volume_cap
    if cap is not None and (
            admission.get("volume_cap") != cap.document()
            or any(admission["cache_identity"].get(k) != v for k, v in cap.cache_identity.items())):
        raise CachePolicyError("Accepted host namespace must bind the exact volume cap/root")


class CacheBudgetLedger:
    """Same physical host authority, independent of result disk and account.

    Accepted host admission binds the OS CommonAppData root, kernel IDs and an
    explicit account set. A caller-selected alternate common_root is refused.
    No claim is admitted on mere self-reported profile or quota booleans.
    """
    def __init__(self, policy, capacity_ledger, *, measurement_provider,
                 namespace_admission, verify_host_admission):
        from .capacity import _capacity_host_context
        actual = _capacity_host_context()
        if verify_host_admission(namespace_admission, actual) is not True:
            raise CachePolicyError("Actual accepted host namespace/ACL admission required")
        bound = namespace_admission["host_ledger"]
        if (capacity_ledger.host != actual["host"] or namespace_admission["host"] != actual["host"]
                or actual["service_account_sid"] not in namespace_admission["authorized_sids"]
                or str(PureWindowsPath(capacity_ledger.common_root)).casefold()
                   != str(PureWindowsPath(actual["common_root"])).casefold()
                or str(PureWindowsPath(bound["path"])).casefold()
                   != str(PureWindowsPath(actual["common_root"])).casefold()):
            raise CachePolicyError("Different physical host/common authority or unapproved account")
        self.policy, self.capacity, self.admission = policy, capacity_ledger, namespace_admission
        self.measure = measurement_provider
        self.key = hashlib.sha256((actual["host"] + "\0cache-host-pool-v2").encode()).hexdigest()
        self.bound, self.control, self.lock_active = bound, None, False
        self.volume_cap_observation = None

    @contextmanager
    def locked(self, *, create_binding=True):
        from .capacity import _LocalVolumeLock
        if self.lock_active:
            raise CachePolicyError("Nested cache transaction refused")
        if (not create_binding and os.name != "nt"
                and not (Path(self.bound["path"]) / ("mutex-" + self.key)).is_file()):
            raise CachePolicyError("Read-only projection has no existing local mutex file")
        with _LocalVolumeLock(self.bound["path"], self.key):
            self.control = WindowsNamespace(self.bound["path"], self.bound["identity"])
            cap_namespace = None
            self.volume_cap_observation = None
            try:
                cap = self.policy.volume_cap
                if cap is not None:
                    require_volume_cap_binding(self.policy, self.admission)
                    cap_namespace = WindowsNamespace(self.policy.root, self.admission["cache_identity"])
                    self.volume_cap_observation = cap_namespace.verify_volume_cap(cap)
                expected = dict(format="darsia-one-host-cache-binding-v2", host=self.capacity.host,
                    authorized_sids=sorted(self.admission["authorized_sids"]), cache_root=self.policy.root,
                    cache_identity=self.admission["cache_identity"], quota_bytes=self.policy.quota_bytes)
                if cap is not None:
                    expected["volume_cap"] = cap.document()
                name = "cache-" + self.key + ".json"
                if name in self.control.names():
                    if _json(self.control, name)[0] != expected:
                        raise CachePolicyError("This host already binds a different cache/identity/quota/account set")
                elif create_binding:
                    self.control.write_once(name, _encoded(expected))
                else:
                    raise CachePolicyError("Read-only projection requires the existing accepted cache binding")
                self.lock_active = True
                yield self
            finally:
                self.lock_active = False
                self.volume_cap_observation = None
                failures = []
                for namespace in (cap_namespace, self.control):
                    if namespace is not None:
                        try: namespace.close()
                        except CacheHandleCloseIncomplete as failed: failures.extend(failed.close_failures)
                self.control = None
                if failures: raise CacheHandleCloseIncomplete(failures)

    def _require_lock(self):
        if not self.lock_active:
            raise CachePolicyError("Same host transaction required")

    def remaining(self, allocated_by_token):
        self._require_lock()
        names = self.control.names()
        reservations = [n for n in names if re.fullmatch(r"cache-r-[0-9a-f]{32}\.json", n)]
        if len(reservations) > 4096:
            raise CachePolicyError("Explicit history compaction required before allocation")
        if any(n.startswith("cache-r-") and n not in reservations for n in names):
            raise CachePolicyError("Unknown cache reservation record preserved")
        total = 0
        for name in reservations:
            row, raw = _json(self.control, name)
            token = _token(row["token"])
            if (row["format"] != "darsia-cache-budget-reservation-v2"
                    or name != "cache-r-" + token + ".json" or row["pool"] != self.key):
                raise CachePolicyError("Foreign reservation preserved")
            closed = "cache-c-" + token + ".json"
            if closed in names:
                proof, _ = _json(self.control, closed)
                if (proof.get("owner") != row["owner"] or proof.get("token") != token
                        or proof.get("no_further_writes") is not True
                        or proof.get("reservation_sha256") != hashlib.sha256(raw).hexdigest()):
                    raise CachePolicyError("Allocation closure differs from original reservation")
            else:
                maximum = nonnegative(row["maximum_bytes"], "Reserved maximum")
                written = nonnegative(allocated_by_token[token], "Observed token bytes")
                if written > maximum:
                    raise CachePolicyError("Actual writer exceeded reservation")
                total += maximum - written
        return total

    def _observed_report(self, maximum_bytes, *, noncache_bytes=0):
        self._require_lock()
        observation = self.measure(self.policy.root)
        if (observation.get("root") != self.policy.root or observation.get("fresh") is not True
                or observation.get("root_identity") != self.admission["cache_identity"]
                or observation.get("no_reparse_traversal") is not True):
            raise CachePolicyError("Fresh actual admitted-root accounting required")
        logical_remaining = self.remaining(observation["active_scope_bytes"])
        report = budget_report(self.policy, used_bytes=observation["logical_bytes"],
            reserved_unwritten_bytes=logical_remaining,
            request_bytes=maximum_bytes, physical_free_bytes=observation["free_bytes"],
            required_noncache_bytes=noncache_bytes,
            eligible_reclaimable_bytes=observation.get("eligible_reclaimable_bytes", 0))
        report.update(measured_utc=observation["measured_utc"], written_scope_bytes_counted_once=True)
        cap = getattr(self, "volume_cap_observation", None)
        if cap is not None:
            report.update(hard_filesystem_quota_verified=True,
                enforcement="verified fixed-volume filesystem capacity plus logical admission; writer routing separate",
                verified_volume_maximum_bytes=cap["virtual_size_bytes"], volume_cap_observation=cap)
        return report

    def project(self, maximum_bytes, *, noncache_bytes=0):
        """Read fresh numbers under an existing-binding-only transaction."""
        result = self._observed_report(maximum_bytes, noncache_bytes=noncache_bytes)
        result.update(projection_only=True, reservation_created=False,
            reclaimable_semantics="only explicitly eligible logical bytes in this observation; never credited as free")
        return result

    def admit(self, maximum_bytes, *, noncache_bytes=0):
        report = self._observed_report(maximum_bytes, noncache_bytes=noncache_bytes)
        if not report["sufficient"]:
            raise CachePolicyError("Admission refused before claim: " + str(report))
        return report

    def reserve(self, token, owner, maximum_bytes):
        self._require_lock()
        token = _token(token)
        # This is the cooperative allocation boundary, not just a receipt
        # writer. Even a direct caller cannot skip fresh quota admission.
        self.admit(nonnegative(maximum_bytes, "Maximum"))
        row = dict(format="darsia-cache-budget-reservation-v2", pool=self.key, utc=_now(),
                   owner=_owner(owner), token=token, maximum_bytes=nonnegative(maximum_bytes, "Maximum"))
        raw = _encoded(row)
        self.control.write_once("cache-r-" + token + ".json", raw)
        return dict(token=token, pool=self.key, sha256=hashlib.sha256(raw).hexdigest())

    def require_reservation(self, reservation, owner, needed):
        self._require_lock()
        token = _token(reservation["token"])
        row, raw = _json(self.control, "cache-r-" + token + ".json")
        if (reservation != dict(token=token, pool=self.key, sha256=hashlib.sha256(raw).hexdigest())
                or row["owner"] != _owner(owner) or row["pool"] != self.key
                or row["maximum_bytes"] < nonnegative(needed, "Declared caps")
                or "cache-c-" + token + ".json" in self.control.names()):
            raise CachePolicyError("Closed, foreign or insufficient exact reservation")
        return row, raw

    def fresh_free_guard(self):
        """Actual free-space observation, not a theorem about physical lengths.

        This is repeated before and after each cooperative immutable write.
        Bypassing writers and allocation within one write can cross the floor;
        this guard is not an atomic filesystem quota or a reclamation forecast.
        """
        from .capacity import volume_info
        from .policy import PHYSICAL_RESERVE_BYTES
        self._require_lock()
        actual = volume_info(self.policy.root)
        if actual["free_bytes"] < PHYSICAL_RESERVE_BYTES:
            raise CachePolicyError("Fresh actual cache-volume free space is below the 40 GiB reserve")
        return actual

    def require_unbound_scope(self, token):
        self._require_lock()
        if "cache-s-" + _token(token) + ".json" in self.control.names():
            raise CachePolicyError("One reservation cannot fund a second scratch scope")

    def bind_scope(self, reservation, manifest):
        self.require_reservation(reservation, manifest["owner"], sum(manifest["temporary_file_caps"].values()))
        self.require_unbound_scope(reservation["token"])
        self.control.write_once("cache-s-" + reservation["token"] + ".json", _encoded(manifest))

    def close(self, reservation, *, owner, verify_no_further_writes):
        row, raw = self.require_reservation(reservation, owner, 0)
        if verify_no_further_writes(row) is not True:
            raise CachePolicyError("Actual original owner/readers and writes must be retired")
        self.control.write_once("cache-c-" + row["token"] + ".json", _encoded(dict(
            owner=row["owner"], token=row["token"], utc=_now(), no_further_writes=True,
            reservation_sha256=hashlib.sha256(raw).hexdigest())))


def applicable_job_cache_profile(policy, adapter, task, estimate, *, projection=False):
    """One original exact-v1 reader, shared by claim and read-only projection.

    A typed volume-bound profile covers every full task on its qualified head
    and route (N188 D2); the old measured profile stays exact in case count and commit.
    A native-phase-n1 profile also covers a full phase-one statistics task and
    a full phase-two n1 task on its qualified head (commit, CPU, float32),
    whose stages it measured.
    """
    from .capacity import read_bound
    refit = task.get("pipeline") == "n1-calibration-refit"
    if refit:
        from darsia.fluidflower.pipeline.refit_capacity import require_estimate
        require_estimate(task, estimate)
    verify = adapter.verify_profile_projection if projection else adapter.verify_profile
    binding = policy.profiles.get("jobs")
    if binding is None:
        if task.get("pipeline") == "preset-wasserstein":
            raise CachePolicyError("No measured original preset comparison job-cache profile; no claim")
        raise CachePolicyError("No measured applicable job-cache profile")
    if task.get("pipeline") == "preset-wasserstein":
        from darsia.fluidflower.preset_wasserstein_capacity import require_estimate, resolve_profile
        if getattr(getattr(adapter, "lease", None), "first_qualification", None) is not None:
            raise CachePolicyError("Preset comparison cannot reuse a first/native/common qualification")
        raw, _ = read_bound(binding)
        from .preset_qualification import PROVIDER
        if raw.get("format") == PROVIDER:
            from .first_qualification import resolve_profile as resolve_provider
            from darsia.fluidflower.preset_wasserstein_capacity import require_bounded_estimate
            profile = resolve_provider(binding, "jobs", lambda pin: read_bound(pin)[0],
                stores_source=adapter.lease.source)
            require_bounded_estimate(task, estimate, profile, binding)
            measured, _ = read_bound(profile["measurement"])
            complete, _ = read_bound(profile["completion"])
            if verify(task, estimate, binding, profile, measured, complete) is not True:
                raise CachePolicyError("Preset bounded service/claim/source administration differs")
            return 0, dict(binding), profile, measured, complete
        require_estimate(task, estimate)
        profile, measured, complete = resolve_profile(raw, task, estimate.scope, reader=read_bound)
        if (profile["measurement"] != estimate.measurement or profile["completion"] != estimate.completion
                or verify(task, estimate, binding, profile, measured, complete) is not True):
            raise CachePolicyError("Preset result/cache measurements or actual guarded writer scope differ")
        return profile["maximum_cache_bytes"], dict(binding), profile, measured, complete
    from .first_qualification import resolve_profile
    from .bootstrap import _exact_authority_list
    profile = resolve_profile(binding, "jobs", lambda pin:read_bound(pin)[0],
        stores_source=getattr(getattr(adapter,"lease",None),"source",None),
        read_authorities=lambda pin:_exact_authority_list(pin, adapter.lease.package_file))
    if profile.get("format") == "darsia-common-phase-job-profile-v1":
        from .common_qualification import require_applicable
        require_applicable(profile, task)
        measured, _ = read_bound(profile["measurement"])
        complete, _ = read_bound(profile["completion"])
        if verify(task, estimate, binding, profile, measured, complete) is not True:
            raise CachePolicyError("Common physical-bound source/stage profile is not applicable")
        return 0, dict(binding), profile, measured, complete
    from .first_qualification import BOUND_JOB_PROFILE, BOUND_WRITER_CONTRACT
    if profile.get("format") == BOUND_JOB_PROFILE:
        # The service-only writers add no claim bytes, so the qualified profile
        # covers every full task on its qualified head and route, whichever run
        # it is. The qualified task itself is covered as before.
        from darsia.fluidflower.storage_profile import declared_storage_profile
        route = ("commit", "pipeline", "backend", "precision")
        qualified = profile.get("scope")
        # A phase-one statistics task and a phase-two n1 task run the qualified
        # stages on another route: always another run, so a full task only.
        staged = (task.get("pipeline") in {"native-phase-statistics", "n1", "n1-calibration-refit"} and isinstance(qualified, dict)
                  and qualified.get("pipeline") == "native-phase-n1")
        keys = ("commit", "backend", "precision") if staged else route
        other_run = (isinstance(qualified, dict)
                     and {key: task.get(key) for key in (*route, "run", "ordered_case_ids")} != qualified)
        if (not isinstance(qualified, dict) or any(task.get(key) != qualified.get(key) for key in keys)
                or (other_run and declared_storage_profile(task) != "full")
                or profile.get("purpose") != "jobs"
                or profile.get("writer_contract") != BOUND_WRITER_CONTRACT
                or profile.get("additional_claim_cache_bytes") != 0
                or profile.get("allocation_model") != "shared-service-only-no-claim-files-v1"
                or profile.get("result_estimate_excludes_cache") is not True):
            raise CachePolicyError("Bound cache profile is outside its qualified head and route")
        measured, _ = read_bound(profile["measurement"])
        complete, _ = read_bound(profile["completion"])
        if verify(task, estimate, binding, profile, measured, complete) is not True:
            raise CachePolicyError("Typed physical-bound writer profile is not applicable")
        return 0, dict(binding), profile, measured, complete
    fields = {"format", "purpose", "pipeline", "precision", "case_count", "commit",
              "maximum_cache_bytes", "measurement", "completion",
              "result_estimate_excludes_cache", "writer_contract"}
    if (set(profile) != fields or profile["format"] != "darsia-cache-measured-profile-v1"
            or profile["purpose"] != "jobs" or profile["pipeline"] != task.get("pipeline")
            or profile["precision"] != task.get("precision")
            or profile["case_count"] != len(task["ordered_case_ids"])
            or profile["commit"] != task["commit"] or profile["result_estimate_excludes_cache"] is not True
            or profile["writer_contract"] != "all-cache-writers-bounded-and-lifecycle-bound-v1"):
        raise CachePolicyError("Actual source/profile/result-split applicability differs")
    maximum = nonnegative(profile["maximum_cache_bytes"], "Measured maximum")
    measured, _ = read_bound(profile["measurement"])
    complete, _ = read_bound(profile["completion"])
    if (measured.get("complete") is not True or measured.get("maximum_observed_cache_bytes") is None
            or nonnegative(measured["maximum_observed_cache_bytes"], "Observed peak") > maximum
            or complete.get("complete") is not True or complete.get("passed") is not True
            or complete.get("commit") != task["commit"]
            or verify(task, estimate, binding, profile, measured, complete) is not True):
        raise CachePolicyError("Measured source-bound cooperative writer coverage is not accepted")
    return maximum, dict(binding), profile, measured, complete


class ClaimCacheBudget:
    """The real ClaimCapacity call boundary; no permissive default adapter.

    Source preparation retains a concrete hook but does not pretend the base
    worker supplies the missing all-writer/hard-limit/retirement integration.
    A late hook cannot substitute for the separate pre-import host environment.
    """
    def __init__(self, runtime, capacity_ledger, owner):
        from .policy import PHYSICAL_RESERVE_BYTES
        self.runtime, self.capacity, self.owner = runtime, capacity_ledger, _owner(owner)
        if capacity_ledger.reserve < PHYSICAL_RESERVE_BYTES:
            raise CachePolicyError("Keep the 40 GiB physical reserve")
        self.policy = cache_settings(runtime.stores.document, required=True)
        self.adapter = getattr(runtime, "cache_adapter", None)
        methods = ("measure", "verify_profile", "verify_host_admission", "verify_preimport",
                   "register_claim_allocation", "budget_ledger")
        if self.adapter is None or any(not callable(getattr(self.adapter, x, None)) for x in methods):
            raise CachePolicyError("No qualified actual writer/lifecycle adapter; no new claim")
        if self.adapter.verify_preimport(runtime.stores, self.policy) is not True:
            raise CachePolicyError("Worker bootstrap environment/actual producer coverage is unqualified")
        # Import-time producers and claim-time service writers share exactly
        # one reservation/accounting ledger. Do not nest another cache lock or
        # call ImportLease.measure against an inactive ledger control handle.
        self.ledger = self.adapter.budget_ledger(capacity_ledger)
        if (not isinstance(self.ledger, CacheBudgetLedger)
                or self.ledger.policy != self.policy
                or self.ledger.admission != self.adapter.namespace_admission):
            raise CachePolicyError("Claim must reuse its actual admitted service cache ledger")
        self.profiles, self.lock = {}, None

    def prepare(self, task, estimate):
        if getattr(getattr(self.adapter,"lease",None), "first_qualification", None) is not None:
            if task.get("pipeline") == "preset-wasserstein":
                from .preset_qualification import PresetQualification
                if not isinstance(self.adapter.lease.first_qualification, PresetQualification):
                    raise CachePolicyError("Preset comparison requires its own exact first service route")
            binding=self.adapter.prepare_first_claim(task, estimate)
            # Enforced no additional claim scope/allocation; this is NOT a
            # measured zero-byte profile or a second service reservation.
            self.profiles[task["attempt_id"]]=(0,binding)
            return
        maximum, binding, _, _, _ = applicable_job_cache_profile(self.policy, self.adapter, task, estimate)
        self.profiles[task["attempt_id"]] = (maximum, binding)

    def acquire(self):
        self.lock = self.ledger.locked()
        self.lock.__enter__()

    def check(self, task, estimate, other_noncache_bytes):
        first=getattr(getattr(self.adapter,"lease",None),"first_qualification",None)
        if first is not None and first.reference_build:
            from .first_qualification import _same_path, reference_root_for_path
            if not _same_path(self.capacity.directory,self.adapter.lease.document["capacity"]["state_directory"]):
                raise CachePolicyError("N43 actual result-capacity state differs from its declared path")
            reference_root_for_path(first,self.capacity.directory)
            first.require_task(self.adapter.lease,task)
            self.ledger.require_reservation(self.adapter.lease.reservation,
                self.adapter.lease.manifest["owner"], self.adapter.lease.maximum)
            return first.reference_admission(self.adapter.lease,other_remaining=other_noncache_bytes)
        from .capacity import volume_info
        cache_volume, result_volume = volume_info(self.policy.root), self.capacity.fresh_volume()
        same = self.capacity.remote_scope is None and cache_volume["volume_id"] == result_volume["volume_id"]
        extra = estimate.products_estimated_bytes + estimate.transient_estimated_bytes + other_noncache_bytes if same else 0
        maximum, _ = self.profiles[task["attempt_id"]]
        report = self.ledger.admit(maximum, noncache_bytes=extra)
        report.update(same_cache_and_results_volume=same, result_cache_allocations_declared_disjoint=True)
        return report

    def bind(self, claim):
        maximum, profile = self.profiles[claim.token]
        reservation = self.ledger.reserve(claim.token, self.owner, maximum)
        # Failure after reserve deliberately leaves it active. Only actual
        # no-further-writes/retirement evidence may close it; no blind rollback.
        self.adapter.register_claim_allocation(claim, self.ledger, reservation, self.owner, maximum, profile)

    def unlock(self):
        if self.lock is not None:
            lock, self.lock = self.lock, None
            lock.__exit__(None, None, None)
