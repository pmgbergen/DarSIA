"""Explicit Q83 finalization of a proven fresh service scope, never old caches.

No automatic invocation, process observer, quota/profile or producer launcher is
provided here. Original sealed-registry retention and actual complete retirement
must already be accepted by ServiceLifecycle's external callbacks. Filenames may
be discovered after that proof; reparse, linked, open or unresolved branches stay
in place, with the complete original service reservation still charged.
"""
from __future__ import annotations

from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .control import CacheBudgetLedger, _encoded, _json, _time, _token
from .environment import WRITER_ENVIRONMENT_KEYS, WRITER_SUBDIRECTORIES
from .policy import CachePolicyError
from .service_context import resolve_original_service_context, _host_binding
from .service_lifecycle import ServiceLifecycle
from .windows import WindowsNamespace, CacheObjectIncomplete, _parts


class FreshScopeAuditIncomplete(CachePolicyError):
    def __init__(self, message, finalization_state):
        super().__init__(message)
        self.finalization_state = deepcopy(finalization_state)


def _fresh_provenance(context):
    manifest = context.manifest
    token = _token(context.reservation["token"])
    expected = context.policy.purpose(context.purpose)
    if context.purpose == "tests":
        expected /= "current"
    fresh = manifest.get("fresh_scope_provenance")
    writer_directories, writer_keys = WRITER_SUBDIRECTORIES, WRITER_ENVIRONMENT_KEYS
    if getattr(context, "profile", {}).get("format") in {
            "darsia-cache-volume-bound-import-profile-v1", "darsia-common-phase-import-profile-v1",
            "darsia-preset-wasserstein-bounded-profile-v1"}:
        from .environment import FIRST_WRITER_ENVIRONMENT_KEYS, FIRST_WRITER_SUBDIRECTORIES
        writer_directories, writer_keys = FIRST_WRITER_SUBDIRECTORIES, FIRST_WRITER_ENVIRONMENT_KEYS
    required = {"format", "token", "owner", "purpose", "root_identity",
        "creation_disposition", "exclusive_root_handle", "initial_names",
        "initial_empty_observed_utc", "observation_precedes_writer_subdirectories_and_imports",
        "writer_subdirectories", "writer_environment_keys"}
    if (not isinstance(fresh, dict) or set(fresh) != required
            or fresh.get("format") != "darsia-fresh-exclusive-empty-service-scope-v1"
            or fresh.get("token") != token or fresh.get("owner") != manifest["owner"]
            or fresh.get("purpose") != context.purpose
            or fresh.get("root_identity") != manifest["root_identity"]
            or fresh.get("creation_disposition") != "NtCreateFile.FILE_CREATE"
            or fresh.get("exclusive_root_handle") is not True
            or fresh.get("initial_names") != []
            or fresh.get("observation_precedes_writer_subdirectories_and_imports") is not True
            or fresh.get("writer_subdirectories") != list(writer_directories)
            or fresh.get("writer_environment_keys") != list(writer_keys)
            or context.root != expected / token
            or manifest.get("root") != str(context.root)
            or manifest.get("reservation") != context.reservation
            or context.admission["purposes"][context.purpose]["path"] != str(expected)):
        raise CachePolicyError("Original immutable fresh-exclusive-empty service provenance required; no old-scope adoption")
    _time(fresh["initial_empty_observed_utc"])
    return deepcopy(fresh)


def _error_evidence(error):
    created = deepcopy(getattr(error, "created_evidence", None))
    failures = deepcopy(getattr(error, "close_failures", []))
    if isinstance(created, dict):
        failures.extend(deepcopy(created.get("close_failures", [])))
    return dict(type=type(error).__name__, message=str(error),
        close_failures=failures, created_evidence=created)


def _cleanup_current_fresh_tree(tree, *, entry_limit):
    """Freeze safe handles first; retain unsafe branches and required ancestors.

    The caller, not this low-level function, establishes original fresh ownership
    and genuine retirement. Every open is relative to a pinned parent. Mutation
    uses the original disposition/checked-close primitives, with no path delete.
    """
    if tree.closed or set(tree.handles) != {""}:
        raise CachePolicyError("One exclusively reopened original root handle is required")
    if type(entry_limit) is not int or not 0 < entry_limit <= 8192:
        raise CachePolicyError("Positive existing accounting limit within the 8192-handle backend bound required")
    api = tree.api
    root_identity = api.identity(tree.handles[""])
    handles = tree.handles
    identities = {"": root_identity}
    directories = {""}
    snapshots, retained, blocked, mutations, close_failures = {}, [], set(), [], []
    ids = {(root_identity.volume_serial, root_identity.file_id_128)}
    removed_files, removed_dirs, removed_bytes = [], [], 0
    whole_preflight_refusal = None

    def keep(relative, reason, error=None):
        row = dict(path=relative, reason=reason)
        if error is not None:
            row["error"] = _error_evidence(error)
        retained.append(row)
        parts = relative.split("/") if relative else []
        for count in range(len(parts) + 1):
            blocked.add("/".join(parts[:count]))

    def close_one(relative):
        handle = handles.pop(relative)  # Never repeat an uncertain close.
        try:
            api.close(handle)
            return True
        except BaseException as error:
            close_failures.extend(deepcopy(getattr(error, "close_failures", []))
                or [_error_evidence(error)])
            keep(relative, "handle_close_unconfirmed", error)
            return False

    def walk(relative):
        handle = handles[relative]
        rows = api.names(handle, with_types=True)
        snapshots[relative] = rows
        for row in sorted(rows, key=lambda item: item["name"]):
            name = row["name"]
            _parts(name)
            child_relative = relative + "/" + name if relative else name
            if row["reparse"]:
                keep(child_relative, "reparse_entry_not_opened_or_followed")
                continue
            # Recursion may consume capacity before a later parent sibling.
            if len(identities) >= entry_limit:
                keep(child_relative, "existing_accounting_entry_limit_unopened_subtree")
                continue
            try:
                child, identity = api.child(handle, name,
                    directory=row["directory"], mutable=True)
            except CacheObjectIncomplete as error:
                evidence = error.created_evidence
                if (evidence.get("handle_returned") is not False
                        and evidence.get("handle_close_confirmed") is not True):
                    raise  # Unknown native handle state blocks every mutation.
                keep(child_relative, "exclusive_nofollow_open_or_safe_identity_refused", error)
                continue
            handles[child_relative] = child
            identities[child_relative] = identity
            key = (identity.volume_serial, identity.file_id_128)
            if (identity.volume_serial != root_identity.volume_serial
                    or not identity.file_id_128 or set(identity.file_id_128) == {"0"}
                    or key in ids):
                keep(child_relative, "outside_volume_or_duplicate_or_unknown_identity")
                close_one(child_relative)
                continue
            ids.add(key)
            if row["directory"]:
                directories.add(child_relative)
                walk(child_relative)

    try:
        walk("")
        # Retired original writers do not make a changed or incomplete snapshot
        # acceptable. All safe target handles are pinned before first mutation.
        for relative, handle in handles.items():
            current = api.identity(handle)
            frozen = identities[relative]
            if (not current.same_object(frozen.document())
                    or (relative not in directories and current != frozen)):
                raise CachePolicyError("Pinned target changed before first disposition")
        for relative, snapshot in snapshots.items():
            if (sorted(api.names(handles[relative], with_types=True), key=lambda item: item["name"])
                    != sorted(snapshot, key=lambda item: item["name"])):
                raise CachePolicyError("Fresh scope names changed before first disposition")
    except BaseException as error:
        whole_preflight_refusal = _error_evidence(error)
        close_failures.extend(deepcopy(whole_preflight_refusal["close_failures"]))
        keep("", "whole_scope_preflight_refused", error)

    if whole_preflight_refusal is None:
        for relative in sorted(set(handles) - directories):
            if relative in blocked:
                continue
            handle = handles[relative]
            try:
                if api.identity(handle) != identities[relative]:
                    raise CachePolicyError("Pinned file changed after preflight")
                api.dispose(handle)
            except BaseException as error:
                keep(relative, "file_disposition_or_identity_refused", error)
                continue
            mutation = dict(path=relative, directory=False, disposition_set=True,
                            handle_close_confirmed=False)
            mutations.append(mutation)
            if close_one(relative):
                mutation["handle_close_confirmed"] = True
                removed_files.append(relative)
                removed_bytes += identities[relative].bytes
        for relative in sorted(set(handles) & directories,
                               key=lambda key: (key.count("/"), len(key)), reverse=True):
            if relative in blocked:
                continue
            handle = handles[relative]
            try:
                if (not api.identity(handle).same_object(identities[relative].document())
                        or api.names(handle, with_types=True)):
                    raise CachePolicyError("Directory changed or retains entries after safe-child cleanup")
                api.dispose(handle)
            except BaseException as error:
                keep(relative, "directory_disposition_identity_or_nonempty_refused", error)
                continue
            mutation = dict(path=relative, directory=True, disposition_set=True,
                            handle_close_confirmed=False)
            mutations.append(mutation)
            if close_one(relative):
                mutation["handle_close_confirmed"] = True
                removed_dirs.append(relative)
    # No retry, fallback unlink or adoption. Unsafe targets remain and close
    # failures remain incomplete even if other safe siblings were removed.
    for relative in sorted(list(handles), key=lambda key: (key.count("/"), len(key)), reverse=True):
        close_one(relative)
    tree.closed = True
    complete = "" in removed_dirs and not retained and not close_failures
    return dict(complete=complete, removed_files=removed_files,
        removed_directories=removed_dirs, removed_logical_bytes=removed_bytes,
        retained=retained, mutations=mutations, close_failures=close_failures,
        preflight_refusal=whole_preflight_refusal, physical_reclaimed_bytes=None,
        reservation_must_remain_charged=not complete)


def _write_result_chunks(ledger, prefix, result):
    """Retain complete outcome bytes without raising the one-MiB control limit.

    These are post-cleanup audit chunks, not per-created-file ownership records.
    No budget closes until every immutable chunk and compact receipt is durable.
    A partial audit failure leaves the original intent, chunks and reservation.
    """
    chunks, buffer, digest, total = [], bytearray(), hashlib.sha256(), 0

    def emit(payload):
        index = len(chunks)
        name = prefix + "-result-part-" + f"{index:08d}" + ".json"
        document = dict(format="darsia-fresh-service-result-byte-chunk-v1",
            index=index, raw_offset=sum(row["raw_bytes"] for row in chunks),
            raw_bytes=len(payload), raw_sha256=hashlib.sha256(payload).hexdigest(),
            payload_base64=base64.b64encode(payload).decode("ascii"))
        raw = _encoded(document)  # Each 512-KiB payload encodes below one MiB.
        ledger.control.write_once(name, raw)
        chunks.append(dict(path=str(Path(ledger.bound["path"]) / name),
            sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw),
            raw_bytes=len(payload)))

    encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=False, allow_nan=False,
                               separators=(",", ":"))
    for text in encoder.iterencode(result):
        encoded = text.encode("utf-8")
        digest.update(encoded)
        total += len(encoded)
        buffer.extend(encoded)
        while len(buffer) >= 512 * 1024:
            emit(bytes(buffer[:512 * 1024]))
            del buffer[:512 * 1024]
    if buffer:
        emit(bytes(buffer))
    receipt = dict(format="darsia-fresh-service-finalization-result-chunks-v1",
        complete=result["complete"], service_budget_closed=False,
        logical_result_sha256=digest.hexdigest(), logical_result_bytes=total,
        chunks=chunks)
    raw = _encoded(receipt)
    name = prefix + "-result.json"
    ledger.control.write_once(name, raw)
    return dict(path=str(Path(ledger.bound["path"]) / name),
        sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def finalize_original_fresh_scope(receipt_binding, *, package_file, now_utc,
        retirement, verify_retirement, verify_publication=None,
        same_gate_success=None, verify_same_gate_success=None):
    """Explicit original-receipt finalization, with no automatic caller yet.

    Actual complete retirement/publication verification is mandatory and is not
    implemented by this module. A metadata resolver, seal, empty registry or
    caller-supplied complete=True never substitutes for that external evidence.
    """
    context = resolve_original_service_context(receipt_binding, package_file=package_file)
    fresh = _fresh_provenance(context)
    if not callable(verify_retirement):
        raise CachePolicyError("Actual original complete retirement verifier is required")
    lifecycle = ServiceLifecycle(context)
    eligibility = lifecycle.retention_eligibility(now_utc=now_utc,
        retirement=retirement, verify_retirement=verify_retirement,
        verify_publication=verify_publication, same_gate_success=same_gate_success,
        verify_same_gate_success=verify_same_gate_success)
    if (eligibility.get("retention_due") is not True
            or eligibility.get("original_retirement_verifier_accepted") is not True):
        raise CachePolicyError("Original sealed service is not actually eligible for cleanup")

    def no_measurement(*args, **kwargs):
        raise CachePolicyError("Fresh finalization neither measures nor admits a producer")

    ledger = CacheBudgetLedger(context.policy, SimpleNamespace(**context.context),
        measurement_provider=no_measurement, namespace_admission=context.admission,
        verify_host_admission=_host_binding)
    token = context.reservation["token"]
    prefix = "cache-f-" + token
    parent = tree = None
    result = None
    try:
        with ledger.locked(create_binding=False):
            ledger.require_reservation(context.reservation, context.manifest["owner"], context.maximum)
            original, _ = _json(ledger.control, "cache-s-" + token + ".json")
            if original != context.manifest:
                raise CachePolicyError("Original immutable scope differs before finalization")
            if any(name.casefold().startswith((prefix + "-").casefold()) for name in ledger.control.names()):
                raise CachePolicyError("Existing finalization evidence preserved; no automatic retry")
            # Bind the genuine eligibility row, not an arbitrary positive dictionary.
            value = {key: deepcopy(value) for key, value in eligibility.items() if key != "receipt"}
            key = hashlib.sha256(_encoded(value)).hexdigest()
            name = "cache-l-" + token + "-eligibility-" + key + ".json"
            if Path(eligibility["receipt"]["path"]) != Path(ledger.bound["path"]) / name:
                raise CachePolicyError("Eligibility receipt is outside its original control namespace")
            row, raw = _json(ledger.control, name)
            if (row.get("format") != "darsia-cache-service-lifecycle-v1"
                    or row.get("key") != key
                    or row.get("binding") != lifecycle.binding or row.get("kind") != "eligibility"
                    or row.get("value") != value or row.get("actor") != context.owner
                    or hashlib.sha256(raw).hexdigest() != eligibility["receipt"]["sha256"]
                    or len(raw) != eligibility["receipt"]["size_bytes"]):
                raise CachePolicyError("Original accepted eligibility receipt differs")
            intent = dict(format="darsia-fresh-service-finalization-intent-v1",
                original_receipt=deepcopy(context.receipt), fresh_scope_provenance=fresh,
                eligibility=deepcopy(eligibility), reservation=deepcopy(context.reservation),
                root=str(context.root), root_identity=deepcopy(context.manifest["root_identity"]),
                observer=deepcopy(context.owner), utc=now_utc,
                unknown_names_owned_only_by_original_fresh_empty_scope=True)
            ledger.control.write_once(prefix + "-intent.json", _encoded(intent))
            try:
                bound = context.admission["purposes"][context.purpose]
                parent = WindowsNamespace(bound["path"], bound["identity"])
                tree = parent.scope(token, expected=context.manifest["root_identity"])
                result = _cleanup_current_fresh_tree(tree,
                    entry_limit=min(context.entry_limit, 8192))
            except BaseException as error:
                result = dict(complete=False, error=_error_evidence(error),
                    reservation_must_remain_charged=True)
            finally:
                for item in (tree, parent):
                    if item is not None:
                        try:
                            item.close()
                        except BaseException as error:
                            if result is None:
                                result = dict(complete=False)
                            result.update(complete=False, reservation_must_remain_charged=True)
                            result.setdefault("outer_close_failures", []).append(_error_evidence(error))
            result.update(format="darsia-fresh-service-finalization-result-v1",
                original_receipt=deepcopy(context.receipt), eligibility=deepcopy(eligibility),
                root=str(context.root), reservation=deepcopy(context.reservation),
                service_budget_closed=False)
            try:
                result_receipt = _write_result_chunks(ledger, prefix, result)
            except BaseException as error:
                raise FreshScopeAuditIncomplete("Outcome audit incomplete; original reservation is not closed",
                    dict(complete=False, cleanup_result=result, service_budget_closed=False,
                         error=_error_evidence(error))) from error
            result["durable_result_receipt"] = result_receipt
            if result["complete"]:
                # This closure consumes already accepted original retirement, after
                # every scoped target handle closed and the entire scope disappeared.
                try:
                    ledger.close(context.reservation, owner=context.manifest["owner"],
                        verify_no_further_writes=lambda row: row["owner"] == context.manifest["owner"]
                            and eligibility["original_retirement_verifier_accepted"] is True
                            and result["complete"] is True)
                except BaseException as error:
                    raise FreshScopeAuditIncomplete("Original reservation closure is unconfirmed; do not assume held or released",
                        dict(complete=False, cleanup_result=result, service_budget_closed=None,
                             error=_error_evidence(error))) from error
                result["service_budget_closed"] = True
                try:
                    closed, raw = _json(ledger.control, "cache-c-" + token + ".json")
                except BaseException as error:
                    raise FreshScopeAuditIncomplete("Budget is closed; closure receipt readback incomplete",
                        dict(complete=False, cleanup_result=result, service_budget_closed=True,
                             error=_error_evidence(error))) from error
                result["budget_closure"] = dict(path=str(Path(ledger.bound["path"]) /
                    ("cache-c-" + token + ".json")), sha256=hashlib.sha256(raw).hexdigest(),
                    size_bytes=len(raw))
                try:
                    ledger.control.write_once(prefix + "-budget-closed.json", _encoded(dict(
                        original_receipt=deepcopy(context.receipt),
                        budget_closure=deepcopy(result["budget_closure"]),
                        complete_scope_cleanup=True, service_budget_closed=True)))
                except BaseException as error:
                    # The already durable cache-c record is the budget truth.
                    raise FreshScopeAuditIncomplete("Budget is closed; final audit publication incomplete",
                        dict(complete=False, cleanup_result=result, service_budget_closed=True,
                             budget_closure=result["budget_closure"], error=_error_evidence(error))) from error
            return result
    except FreshScopeAuditIncomplete:
        raise
    except BaseException as error:
        raise FreshScopeAuditIncomplete("Finalization incomplete; inspect original outcome and reservation receipts",
            dict(complete=False, cleanup_result=result,
                 service_budget_closed=result.get("service_budget_closed") if result else None,
                 original_receipt=deepcopy(context.receipt), error=_error_evidence(error))) from error
