"""Concrete operational adapters; no numerical or GUI import.

No automatic cleanup or fabricated child retirement. The display writer returns
its original scope so an actual consumer-lifecycle owner can later close it.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

from .bootstrap import (current_import_lease, _owner, select_entry_role,
                        RECEIPT_ENV, POLICY_ENV, EXPECTED_STORES_ENV)
from .control import OwnedScratch
from .policy import CachePolicyError
from .windows import WindowsNamespace


def worker_identity_epoch(pid):
    """Return the process_create_time string the worker queue records for pid.

    The queue's worker identity stores str(psutil.Process(pid).create_time()).
    psutil divides the 100 ns FILETIME count as a C double, so for about one
    process start in five its string differs by one rounding step from the
    exact value that _owner() records for the same process. Compare a worker
    identity only with this form, and bind the kernel identity separately and
    exactly (pid, creation_filetime_100ns).
    """
    import psutil
    return str(psutil.Process(pid).create_time())


def bind_actual_worker_owner(owner, *, slot=None, generation=None, role=None):
    """Attach the existing semantic worker identity to its early native owner.

    Call from the existing non-GUI worker boundary; the early registration has
    already happened before spawn imports. This is not a retirement callback.
    """
    from .folders import current_folder_context
    context = current_folder_context()
    if context is not None:
        if (context.pid != os.getpid() or not isinstance(owner, dict)
                or owner.get("pid") != os.getpid()
                or owner.get("process_create_time") != worker_identity_epoch(os.getpid())
                or not owner.get("machine", "").casefold().startswith(context.machine.casefold() + "-")):
            raise CachePolicyError("Folder worker PID, epoch or configured machine differs from actual owner")
        return None  # The real shared queue records its owner at atomic claim.
    lease=current_import_lease()
    actual,_=_owner()
    expected_epoch=actual["process_create_time"]
    if lease.manifest.get("native_launch_context") is not None:
        from .owned_process import current_native_membership
        expected_epoch=worker_identity_epoch(actual["pid"])
        if (any(actual.get(key)!=lease.owner.get(key)
                for key in ("pid","creation_filetime_100ns"))
                or current_native_membership()!=lease.manifest["native_launch_context"]
                or (isinstance(owner,dict) and "creation_filetime_100ns" in owner
                    and owner["creation_filetime_100ns"]!=actual["creation_filetime_100ns"])):
            raise CachePolicyError("Current native FILETIME/Job differs from original lease owner")
    if (not isinstance(owner,dict) or owner.get("pid")!=actual["pid"]
            or owner.get("process_create_time")!=expected_epoch):
        raise CachePolicyError("Semantic worker PID/epoch differs from current native owner")
    identity=hashlib.sha256(json.dumps(owner,sort_keys=True).encode()).hexdigest()
    return lease._record(lease.reservation["token"]+".worker-"+identity+".json",
        dict(owner=owner,cache_native_owner=actual,slot=slot,generation=generation,role=role,
             reservation=lease.reservation,role_identifier=lease.role_identifier,
             entry_kind=lease.entry_kind,retirement=False))


def require_same_startup_stores(stores):
    """Supplement original startup validation; never replace its Git/lock checks."""
    from .folders import current_folder_context
    context = current_folder_context()
    if context is not None:
        if (context.pid != os.getpid() or stores.machine_id != context.machine
                or any(stores.source.get(k) != context.source[k] for k in ("path", "sha256", "size_bytes"))):
            raise CachePolicyError("Worker startup stores differs from pre-import folder selection")
        return context
    lease=current_import_lease()
    if lease.entry_kind!="worker" or lease.purpose!="jobs":
        raise CachePolicyError("Worker startup requires its configured worker/jobs role")
    if any(stores.source.get(k)!=lease.source[k] for k in ("path","sha256","size_bytes")):
        raise CachePolicyError("Worker startup stores differs from pre-import cache binding")
    return lease


def record_claim_cache_requirement(claim, *, applicable_profile, maximum_bytes, reservation, owner):
    """Record an already reserved task supplement; not a permissive adapter.

    The existing ClaimCacheBudget must perform admission/reservation first. The
    caller supplies that original binding, not a self-reported coverage boolean.
    """
    lease=current_import_lease()
    if type(maximum_bytes) is not int or maximum_bytes<0 or not isinstance(applicable_profile,dict):
        raise CachePolicyError("Original applicable cache profile/maximum required")
    if reservation.get("token")!=claim.token:
        raise CachePolicyError("Original claim and cache reservation differ")
    with lease.ledger.locked():
        lease.ledger.require_reservation(reservation,owner,maximum_bytes)
    return lease._record(lease.reservation["token"]+".claim-"+claim.token+".json",
        dict(original_claim_token=claim.token,import_reservation=lease.reservation,
             task_cache_reservation=reservation,task_owner=owner,
             applicable_profile=applicable_profile,maximum_bytes=maximum_bytes,
             per_claim_admission_must_be_proven_separately=True))


def optional_import_lease():
    """Only the original missing-lease case means unmanaged ordinary import."""
    try:
        return current_import_lease()
    except CachePolicyError:
        return None


def independent_worker_environment(stores, *, machine, series_id,
                                   explicit_stores, parent_environment, job_root=None):
    """Select original stores/worker role; child creates its independent lease.

    No scope allocation here, no command rewriting, and no GUI retirement claim.
    The only removed environment variable is the parent service receipt.
    """
    if stores.machine_id.casefold()!=str(machine).casefold():
        raise CachePolicyError("Selected worker stores belongs to another machine")
    if series_id is not None:
        stores.roots(series_id)  # Original series alias validation, no discovery.
    source={key:stores.source[key] for key in ("path","sha256","size_bytes")}
    from .folders import folder_mode, folder_environment
    if series_id is not None and folder_mode(stores, series_id):
        result = folder_environment(stores.document, source, series_id,
            job_root=job_root, parent_environment=parent_environment)
        result[EXPECTED_STORES_ENV] = json.dumps(source,sort_keys=True,separators=(",",":"))
        return result
    if "cache" not in stores.document and "cache_bootstrap" not in stores.document:
        raise CachePolicyError("Choose a Cache folder in Start > Series folders before starting a worker")
    select_entry_role(stores.document,entry_kind="worker")
    result=dict(parent_environment)
    if not explicit_stores and POLICY_ENV in result:
        if Path(result[POLICY_ENV]).resolve()!=Path(source["path"]).resolve():
            raise CachePolicyError("Inherited cache policy differs from the original default worker stores")
    result.pop(RECEIPT_ENV,None)
    # Read by CPython before site/.pth/bootstrap; parent environment is untouched.
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    result[EXPECTED_STORES_ENV]=json.dumps(source,sort_keys=True,separators=(",",":"))
    return result


def cooperative_child_environment(*, parent_environment):
    """Inherit the exact admitted jobs family; workspace writers stay unqualified."""
    from .folders import current_folder_context
    if current_folder_context() is not None:
        return dict(parent_environment)
    lease=current_import_lease()
    if lease.purpose!="jobs":
        raise CachePolicyError("Legacy worker needs an explicitly configured jobs service role")
    result=dict(parent_environment)
    if RECEIPT_ENV not in result or json.loads(result[RECEIPT_ENV])!=lease.receipt:
        raise CachePolicyError("Child environment lacks the exact original service receipt")
    return result


def record_cache_observation(kind, details):
    """Record only an actual existing observation; never close a reservation."""
    allowed={"worker_spawn_intent","worker_spawn_observed","worker_children_retired",
             "legacy_spawn_intent","legacy_spawn_observed"}
    if kind not in allowed or not isinstance(details,dict):
        raise CachePolicyError("An exact supported operational observation is required")
    from .folders import current_folder_context
    if current_folder_context() is not None:
        return None  # Process supervision/retirement remains in the real worker.
    lease=current_import_lease();actual,_=_owner()
    from .service_lifecycle import ServiceLifecycle
    ServiceLifecycle(lease).record_launch(kind,details)  # Every allowed launch shares the seal transaction.
    return lease._record(lease.reservation["token"]+".observation-"+uuid.uuid4().hex+".json",
        dict(kind=kind,details=details,observer=actual,reservation=lease.reservation,
             service_or_family_retirement_claimed=False))


def bind_legacy_worker_identity(worker_id):
    lease=current_import_lease()
    if lease.purpose!="jobs":
        raise CachePolicyError("Legacy worker lacks its admitted jobs service role")
    actual,_=_owner();actual["worker_id"]=worker_id
    return bind_actual_worker_owner(actual,role="legacy_worker")


_DISPLAY_ALLOCATIONS=[]
_DISPLAY_REUSE={}


def reusable_display_png(*, logical_gate):
    """Reuse only this process's own closed immutable allocation for this lease.

    The key retains the original source signature, phase and display policy.
    No legacy directory, arbitrary existing PNG or previous lease is adopted.
    """
    lease=current_import_lease()
    result=_DISPLAY_REUSE.get((lease.reservation["token"],logical_gate))
    if result is None:return None
    allocation=result["allocation"];frozen=result["owned_frozen"]
    if (result["immutable_writer_closed"] is not True or allocation.outcome["outcome"]!="success"
            or not allocation.tree.closed or frozen["unresolved_creates"]
            or frozen["format"]!="darsia-windows-owned-tree-v2"
            or frozen["scope"]!=allocation.root.name
            or set(frozen["files"])!={"display.png","provenance.json"}
            or allocation.policy!=lease.policy or allocation.ledger.admission!=lease.admission):
        raise CachePolicyError("Display reuse requires the original closed owned allocation")
    namespace=WindowsNamespace(allocation.root,frozen["root"])
    try:
        if set(namespace.names())!=set(frozen["files"]):
            raise CachePolicyError("Display namespace changed; no reuse or legacy adoption")
        for name,expected in frozen["files"].items():
            handle,identity=namespace.api.child(namespace.handle,name,directory=False)
            try:
                if identity.document()!=expected:
                    raise CachePolicyError("Owned display immutable file version changed; no reuse")
            finally:namespace.api.close(handle)
    finally:namespace.close()
    return Path(result["path"])


def managed_display_png(png_bytes, provenance, *, logical_gate):
    """Keep local immutable allocation receipts until real process-reader closure.

    There is no function-return, widget or atexit cleanup. Exact future reader
    retirement and external writer closure still need qualification.
    """
    lease=current_import_lease();key=(lease.reservation["token"],logical_gate)
    existing=reusable_display_png(logical_gate=logical_gate)
    if existing is not None:
        result=_DISPLAY_REUSE[key]
        metadata=(json.dumps(provenance,ensure_ascii=False,indent=2)+"\n").encode("utf-8")
        if (hashlib.sha256(png_bytes).hexdigest()!=result["png_sha256"]
                or hashlib.sha256(metadata).hexdigest()!=result["provenance_sha256"]):
            raise CachePolicyError("Repeated display key has different encoded content/provenance")
        return existing
    result=write_display_png(png_bytes,provenance,logical_gate=logical_gate)
    _DISPLAY_ALLOCATIONS.append(result)
    _DISPLAY_REUSE[key]=result
    return Path(result["path"])


def write_display_png(png_bytes, provenance, *, logical_gate):
    """Encode before this call; reserve exact lengths and write immutable bytes.

    No Pillow import or pixel/mask changes. Native arrays/products are not an
    input contract. No publication, success cleanup or reader retirement is
    claimed merely by returning a local display path.
    """
    if not isinstance(png_bytes,bytes) or not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise CachePolicyError("Already encoded display PNG bytes required")
    if not isinstance(provenance,dict) or provenance.get("model_scores_changed") is not False:
        raise CachePolicyError("Explicit unchanged display provenance required")
    metadata=(json.dumps(provenance,ensure_ascii=False,indent=2)+"\n").encode("utf-8")
    lease=current_import_lease();owner,_=_owner();token=uuid.uuid4().hex
    caps={"display.png":len(png_bytes),"provenance.json":len(metadata)}
    with lease.ledger.locked():
        reservation=lease.ledger.reserve(token,owner,sum(caps.values()))
        allocation=OwnedScratch(lease.policy,"display",token,owner,logical_gate,caps,
            namespace_admission=lease.admission,ledger=lease.ledger,reservation=reservation)
    try:
        path=allocation.write_bytes("display.png",png_bytes)
        allocation.write_bytes("provenance.json",metadata)
        publication=dict(format="closed-local-display-png-v1",path=str(path),
            sha256=hashlib.sha256(png_bytes).hexdigest(),size_bytes=len(png_bytes),
            NAS_publication_claimed=False)
        allocation.finish("success",utc=datetime.now(timezone.utc).isoformat(),publication=publication)
        return dict(path=path,allocation=allocation,reservation=reservation,
            immutable_writer_closed=True,
            owned_frozen=allocation.tree.frozen(),
            png_sha256=hashlib.sha256(png_bytes).hexdigest(),
            provenance_sha256=hashlib.sha256(metadata).hexdigest(),reader_retirement_pending=True,
            created_utc=datetime.now(timezone.utc).isoformat())
    except BaseException:
        # Original create/write evidence and reservation remain. Do not unlink.
        allocation.close_handles()
        raise


def seal_current_service(*, logical_gate, outcome, utc, traceback_text=None, publication=None):
    """Explicit parent intent only; never called automatically from stopped/atexit."""
    from .service_lifecycle import ServiceLifecycle
    return ServiceLifecycle(current_import_lease()).seal(logical_gate=logical_gate,
        outcome=outcome,utc=utc,traceback_text=traceback_text,publication=publication)


def service_retention_eligibility(original_lease, **proofs):
    """External observer supplies original accepted lease/ledger and real proofs.

    No lease reconstruction, provider, root or quota is guessed by this API.
    Result is never authority to delete or to close the service reservation.
    """
    from .service_lifecycle import ServiceLifecycle
    return ServiceLifecycle(original_lease).retention_eligibility(**proofs)


def _worker_metadata(path, *, expected=None, maximum_bytes=16 * 1024**2):
    """Bounded original metadata only; no result array/image or product copy."""
    import stat
    from .policy import SETUP_CONTROL_MAX_BYTES
    if maximum_bytes not in {16 * 1024**2, SETUP_CONTROL_MAX_BYTES}:
        raise CachePolicyError("Only the original publication or explicit Setup control bound is supported")
    path = Path(path)
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink < 1
            or getattr(before, "st_reparse_tag", 0)
            or getattr(before, "st_file_attributes", 0) & 0x400
            or not 0 < before.st_size <= maximum_bytes):
        raise CachePolicyError("Original worker publication metadata is unsafe/unbounded")
    with path.open("rb") as stream:
        raw = stream.read(before.st_size + 1)
    after = path.lstat()
    if (len(raw) != before.st_size or any(getattr(before, key) != getattr(after, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_nlink"))):
        raise CachePolicyError("Original worker publication metadata changed during read")
    pin = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    if expected is not None and pin != expected:
        raise CachePolicyError("Original worker publication metadata bytes differ")
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise CachePolicyError("Original worker metadata document must be an object")
    return document, pin


def _worker_alias(root, binding):
    from pathlib import PurePosixPath
    from .policy import nonnegative
    if (not isinstance(binding, dict) or set(binding) != {"relative_path", "sha256", "bytes"}
            or not isinstance(binding["relative_path"], str)):
        raise CachePolicyError("Exact original alias publication reference required")
    relative = PurePosixPath(binding["relative_path"])
    if (not binding["relative_path"] or relative.is_absolute()
            or "\\" in binding["relative_path"] or ":" in binding["relative_path"]
            or any(part in {"", ".", ".."} for part in binding["relative_path"].split("/"))):
        raise CachePolicyError("Original alias publication escaped its job")
    target = Path(root).joinpath(*relative.parts)
    # This only resolves the producer's own immutable metadata, not native data.
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise CachePolicyError("Original alias publication resolved outside its job")
    pin = dict(path=str(target), sha256=binding["sha256"],
               size_bytes=nonnegative(binding["bytes"], "Original alias metadata bytes"))
    return _worker_metadata(target, expected=pin)


def seal_worker_service(*, job_root, exit_code, terminal_events, traceback_text=None):
    """Seal only the explicit native-managed original worker; never claim exit=retirement.

    Earlier cooperative and ordinary internal test/runtime entries have no
    native retirement provider and remain unqualified, not silently finalized.
    The public worker's original result/queue operations have already completed.
    """
    lease = optional_import_lease()
    if lease is None or "native_supervision" not in lease.profile:
        return None
    if lease.entry_kind != "worker" or "native_launch_context" not in lease.manifest:
        raise CachePolicyError("Native worker outcome has no original managed service")
    if (type(exit_code) is not int or not isinstance(terminal_events, list)
            or len(terminal_events) > lease.entry_limit):
        raise CachePolicyError("Bounded actual worker outcome is required")
    root = Path(job_root)
    job, job_pin = _worker_metadata(root / "job.json")
    if job.get("format") != "darsia-alias-job-v1":
        raise CachePolicyError("Native publication provider supports original alias jobs only")
    completed, failures = [], []
    for event in terminal_events:
        if not isinstance(event, dict):
            raise CachePolicyError("Actual terminal worker event is malformed")
        if event.get("kind") != "completed":
            failures.append(dict(event))
            continue
        run, attempt = event.get("run"), event.get("attempt_id")
        import re
        if (not isinstance(run, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run) is None
                or not isinstance(attempt, str) or re.fullmatch(r"[0-9a-f]{32}", attempt) is None):
            raise CachePolicyError("Actual completed event has no exact run/attempt")
        done, done_pin = _worker_metadata(root / "done" / (run + ".json"))
        if (done.get("format") != "darsia-alias-done-v1" or done.get("run") != run
                or done.get("attempt_id") != attempt):
            raise CachePolicyError("Original done publication differs from actual completed event")
        result, result_pin = _worker_alias(root, done["completed"])
        _, index_pin = _worker_alias(root, done["products_index"])
        if result.get("complete") is not True:
            raise CachePolicyError("Original completed metadata remains incomplete")
        completed.append(dict(run=run, attempt_id=attempt, done=done_pin,
                              completed=result_pin, products_index=index_pin))
    publication = dict(format="darsia-native-worker-service-publication-v1", job=job_pin,
        native_launch_context=lease.manifest["native_launch_context"],
        completed=completed, terminal_failures=failures,
        original_done_and_product_index_metadata_verified=True,
        native_payload_copied_or_rehashed=False, durable_results_not_cache=True)
    binding = lease._record(lease.reservation["token"] + ".worker-publication.json", publication)
    failed = exit_code != 0 or bool(failures) or traceback_text is not None
    literal = traceback_text
    if failed and literal is None:
        literal = json.dumps(dict(exit_code=exit_code, actual_terminal_failures=failures),
                             ensure_ascii=False, sort_keys=True)
    return seal_current_service(logical_gate="worker-job:" + job_pin["sha256"],
        outcome="failed" if failed else "success", utc=datetime.now(timezone.utc).isoformat(),
        traceback_text=literal, publication=None if failed else binding)


def verify_worker_publication(snapshot, binding):
    """Read the original immutable small publication closure, never native arrays."""
    from .policy import _descriptor
    pin = _descriptor(binding)
    manifest = snapshot["binding"]["manifest"]
    expected = Path(manifest["native_launch_context"]["path"]).parent / (
        manifest["token"] + ".worker-publication.json")
    if Path(pin["path"]) != expected:
        return False
    document, _ = _worker_metadata(pin["path"], expected=pin)
    if (document.get("format") != "darsia-native-worker-service-publication-v1"
            or document.get("native_launch_context") != manifest.get("native_launch_context")
            or document.get("terminal_failures")
            or document.get("original_done_and_product_index_metadata_verified") is not True):
        return False
    job, _ = _worker_metadata(document["job"]["path"], expected=document["job"])
    if job.get("format") != "darsia-alias-job-v1":
        return False
    root = Path(document["job"]["path"]).parent
    for row in document["completed"]:
        done, actual_done = _worker_metadata(root / "done" / (row["run"] + ".json"),
                                            expected=row["done"])
        result, result_pin = _worker_alias(root, done["completed"])
        _, index_pin = _worker_alias(root, done["products_index"])
        if (done.get("format") != "darsia-alias-done-v1"
                or done.get("run") != row["run"] or done.get("attempt_id") != row["attempt_id"]
                or actual_done != row["done"] or result_pin != row["completed"]
                or index_pin != row["products_index"] or result.get("complete") is not True):
            return False
    return True
