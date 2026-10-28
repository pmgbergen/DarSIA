"""Whole-run queue with atomic claims and attempt-specific publication leases.

The shared filesystem must provide atomic rename within one share. Progress is
written only for actual I/O or frame advances, never as a synthetic heartbeat.
An expired attempt keeps its output, but loses the right to publish a run.
Each attempt is retired by exactly one exclusive-create decision file
(publish, reclaimed or rejected). No lease directory is ever moved: a run-level
marker file binds the run to its published lease in place, and a marker file in
reclaimed/<run>/<attempt>/ names a reclaimed or rejected lease in place.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import threading
import traceback
import uuid
from dataclasses import dataclass
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath


class LeaseLost(RuntimeError):
    """This attempt has been reclaimed or already published."""


def utc() -> str:
    """Return an aware UTC timestamp for queue lifecycle evidence.
    """
    return datetime.now(timezone.utc).isoformat()


from darsia_cache.transient import (
    TRANSIENT_RETRY_DELAYS, _NETWORK_WINERRORS, transient_os_error,
    retry_transient as _retry_transient,
)


def retry_transient(action, *, delays=None, wait=None, on_retry=None, retry_if=None):
    """Keep the queue's original delay override API with the shared filesystem retry."""
    return _retry_transient(action,
        delays=TRANSIENT_RETRY_DELAYS if delays is None else delays,
        wait=wait, on_retry=on_retry, retry_if=retry_if)


def read_json(path: Path) -> dict:
    """Read UTF-8 JSON metadata, rejecting duplicate keys and nonfinite numbers.
    """
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(
        retry_transient(Path(path).read_bytes).decode("utf-8-sig"),
        object_pairs_hook=unique,
        parse_constant=invalid,
    )


def _windows_rename(source: Path, destination: Path) -> None:
    """Rename by one exclusive DELETE handle, without replacing a destination.

    Excluding FILE_SHARE_DELETE prevents concurrent source handles from moving
    the same file through several destinations. Ordinary readers may finish;
    the caller retains the existing bounded sharing-error retry policy.
    """
    import ctypes
    from ctypes import wintypes

    class RenameInfo(ctypes.Structure):
        _fields_ = [
            ("ReplaceIfExists", ctypes.c_ubyte),
            ("RootDirectory", wintypes.HANDLE),
            ("FileNameLength", wintypes.DWORD),
            ("FileName", wintypes.WCHAR * 1),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.SetFileInformationByHandle.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    ]
    kernel.SetFileInformationByHandle.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL

    # DELETE only; share reads/writes, but never a second delete/rename handle.
    handle = kernel.CreateFileW(
        os.path.abspath(source), 0x00010000, 0x00000003, None,
        3, 0x02000000, None,  # OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        target = os.path.abspath(destination).encode("utf-16-le")
        buffer = ctypes.create_string_buffer(ctypes.sizeof(RenameInfo) + len(target))
        info = RenameInfo.from_buffer(buffer)
        info.ReplaceIfExists = False
        info.RootDirectory = None
        info.FileNameLength = len(target)
        ctypes.memmove(ctypes.addressof(buffer) + RenameInfo.FileName.offset,
                       target, len(target))
        if not kernel.SetFileInformationByHandle(handle, 3, buffer, len(buffer)):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        if not kernel.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())


def rename(source: Path, destination: Path) -> None:
    """Keep original atomic moves inside any explicit Setup writer scope."""
    from .series_setup_sink import current_writer
    writer = current_writer()
    if writer is None:
        return _rename_without_setup(source, destination)
    return writer.rename(source, destination, rename_operation=_rename_without_setup)


def _rename_without_setup(source: Path, destination: Path) -> None:
    """Atomic move; briefly let concurrent Windows metadata readers close.

    A transient error is retried (N111). When a retry finds the source gone and
    the destination present, the interrupted try completed on the server.
    """
    interrupted = []

    def move():
        try:
            _rename_once(source, destination)
        except FileNotFoundError:
            if interrupted and Path(destination).exists() and not Path(source).exists():
                return
            raise

    retry_transient(move, on_retry=lambda attempt, delay, error: interrupted.append(error))


def _rename_once(source, destination):
    """One atomic move; a Windows reader handle gets one second to close."""
    deadline = time.monotonic() + 1.0
    while True:
        try:
            if os.name == "nt":
                _windows_rename(source, destination)
            else:
                os.rename(source, destination)
            return
        except PermissionError as exc:
            if (os.name != "nt" or getattr(exc, "winerror", None) not in {5, 32}
                    or time.monotonic() >= deadline):
                raise
            time.sleep(0.01)


def _immutable_bytes(document: dict) -> bytes:
    """Exact bytes write_immutable publishes for a document."""
    return (json.dumps(document, ensure_ascii=False, allow_nan=False) + "\n").encode()


def write_immutable(path: Path, document: dict, *, create_parents=True) -> None:
    """Publish complete UTF-8 bytes without ever replacing an existing file."""
    _publish_bytes(Path(path), _immutable_bytes(document), create_parents=create_parents)


def _publish_bytes(path: Path, data: bytes, *, create_parents: bool) -> None:
    """Exclusively create a file with exact bytes; retry a transient error (N111).

    A file with identical bytes found after an interrupted try is that try's
    own completed write. Otherwise an existing file raises FileExistsError.
    """
    # The optional lean-only context shares the existing exclusive publisher.
    # Unscoped full/legacy calls retain their original behavior and bytes.
    from .io_ops import check_metadata_bytes
    check_metadata_bytes(path, data)
    from .series_setup_sink import current_writer
    setup_writer = current_writer()
    interrupted = []

    def publish():
        if create_parents:
            if setup_writer is None:
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                setup_writer.mkdir(path.parent, parents=True, exist_ok=True)
        # Temporary IO names must not consume the semantic attempt/frame UUID
        # stream (the original full writer's identity sequence is preserved).
        from secrets import token_hex
        temporary = path.with_name(f".{path.name}.{token_hex(16)}.partial")
        if setup_writer is None:
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        else:
            register = getattr(setup_writer, "register_temporary", None)
            if register is not None:
                register(path, temporary)
            setup_writer.write_bytes(temporary, data)
        try:
            if os.name == "nt":
                rename(temporary, path)  # Windows refuses an existing destination.
            else:
                if setup_writer is None:
                    os.link(temporary, path)  # POSIX rename would replace a file.
                    temporary.unlink()  # Only the successful extra hardlink is removed.
                else:
                    def publish_link(source, destination):
                        os.link(source, destination)
                        source.unlink()
                    setup_writer.rename(temporary, path, rename_operation=publish_link)
        except FileExistsError:
            try:
                same = bool(interrupted) and path.read_bytes() == data
            except OSError:
                same = False
            if not same:
                raise

    retry_transient(publish, on_retry=lambda attempt, delay, error: interrupted.append(error))


def _ensure_directory(path: Path, *, parents=True, exist_ok=True) -> None:
    """Keep original mkdir semantics inside an active Setup output domain."""
    from .series_setup_sink import current_writer
    writer = current_writer()
    if writer is None:
        path.mkdir(parents=parents, exist_ok=exist_ok)
    else:
        writer.mkdir(path, parents=parents, exist_ok=exist_ok)


def _create_directory(path: Path, *, parents: bool = False) -> None:
    """Create one new directory; retry a transient error (N111).

    An empty directory found after an interrupted try is that try's own.
    Otherwise an existing directory raises FileExistsError, as mkdir does.
    """
    interrupted = []

    def create():
        try:
            from .series_setup_sink import current_writer
            writer = current_writer()
            if writer is None:
                path.mkdir(parents=parents)
            else:
                writer.mkdir(path, parents=parents)
        except FileExistsError:
            if not (interrupted and path.is_dir() and not any(path.iterdir())):
                raise

    retry_transient(create, on_retry=lambda attempt, delay, error: interrupted.append(error))


def descriptor(path: Path) -> dict:
    """Bind the exact bytes of a file to its path, SHA-256, and byte count.
    """
    path = Path(path)
    data = retry_transient(path.read_bytes)
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def require_unc(value: str) -> None:
    """Require a shared UNC input path and reject local drives and device paths.
    """
    if (not isinstance(value, str)
            or not PureWindowsPath(value).drive.startswith("\\\\")
            or value.replace("/", "\\").startswith(("\\\\?\\", "\\\\.\\"))):
        raise ValueError(f"UNC path required: {value!r}")


def name(value: str) -> str:
    """Validate and normalize a filesystem-safe run or machine identifier.
    """
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"Unsafe run or machine name: {value!r}")
    key = value.casefold()
    if key.endswith(".") or key.split(".")[0] in {
        "con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }:
        raise ValueError(f"Reserved filesystem name: {value!r}")
    return key


def validate_task(task: dict) -> None:
    """Validate whole-run metadata without reading scientific inputs.

    The task declares full code, calibration, and configuration identities, a
    UNC configuration path, and supported backend and working precision.
    """
    name(task["run"])
    if task.get("pipeline") == "native-phase-statistics":
        from .phase1_protocol import validate_phase1_task
        validate_phase1_task(task)
    elif task.get("pipeline") == "series-setup":
        from .series_setup_protocol import validate_setup_task
        validate_setup_task(task)
    elif task.get("pipeline") == "preset-wasserstein":
        from ..preset_wasserstein_protocol import validate_preset_wasserstein_task
        validate_preset_wasserstein_task(task)
    else:
        for key, length in (("commit", 40), ("calibration_sha256", 64)):
            if not re.fullmatch(f"[0-9a-f]{{{length}}}", task.get(key, "")):
                raise ValueError(f"Task {key} must be a full lowercase SHA")
    config = task["configuration"]
    from ..storage_profile import expected_format_version, job_storage_profile
    if task.get("format") == "darsia-alias-task-v1":
        from ..store_map import validate_input_reference
        from .job_inputs import validate_catalog

        # Lean tasks are version 2 (N55); full and legacy tasks stay version 1.
        if (task.get("format_version") != expected_format_version(task)
                or task.get("legacy_paths") is not False):
            raise ValueError("Explicit alias-task version/legacy flag required")
        validate_input_reference(config)
        validate_input_reference(task["series"])
        validate_input_reference(task["series_manifest"])
        catalog = validate_catalog(task["input_catalog"])
        if catalog["run"] != task["run"] or catalog["configuration"] != config:
            raise ValueError("Alias task/catalogue source identity differs")
        cases = task.get("ordered_case_ids")
        if (not isinstance(cases,list) or not cases or len({name(case) for case in cases}) != len(cases)):
            raise ValueError("Exact distinct selected case IDs required")
    elif task.get("format") is not None and "path" not in config:
        raise ValueError("Unsupported explicit task format")
    else:
        require_unc(config["path"])
    if not re.fullmatch(r"[0-9a-f]{64}", config.get("sha256", "")):
        raise ValueError("Scientific configuration SHA required")
    if task.get("backend", "any") not in {"any", "cpu", "cuda"}:
        raise ValueError("Task backend must be any, cpu or cuda")
    if (task.get("precision", "float32") not in {"float32", "float64"}
            and task.get("pipeline") != "preset-wasserstein"):
        raise ValueError("Task precision must be float32 or float64")
    job_storage_profile(task)  # Explicit 'full'/'lean'; absent is a legacy full task.


@dataclass(frozen=True)
class Claim:
    task: dict
    machine: str
    token: str
    claimed: Path
    lease: Path
    output: Path
    input_rows: tuple = ()
    publication_factory: object = None
    local_output: object = None


@dataclass(frozen=True)
class LocalOutputBinding:
    """In-process positive configured-results admission, never serialized as a task."""

    machine_id: str
    results_root: Path
    path: Path
    admission: object


def _own_asset(root, reference):
    """Resolve only this shared bundle's explicit relative metadata descriptor."""
    if (not isinstance(reference, dict)
            or set(reference) != {"relative_path", "sha256", "bytes"}
            or not isinstance(reference["relative_path"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", reference.get("sha256", ""))
            or type(reference["bytes"]) is not int or not 0 <= reference["bytes"] <= 128 * 1024 * 1024):
        raise ValueError("Exact bounded shared metadata reference required")
    relative = reference["relative_path"]
    if (not relative or "\\" in relative or PureWindowsPath(relative).drive
            or relative.startswith("/") or ":" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise ValueError("Shared metadata reference must stay inside its bundle")
    root = Path(root).resolve(strict=True)
    path = root.joinpath(*relative.split("/"))
    for entry in (path, *path.parents):
        if entry == root.parent:
            break
        info = entry.lstat()
        if entry.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Shared publication cannot follow symlink/reparse entries")
    if not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("Shared metadata escapes its original bundle")
    if path.stat().st_size != reference["bytes"]:
        raise ValueError("Shared metadata byte count differs before bounded hashing")
    actual = descriptor(path)
    if actual["sha256"] != reference["sha256"] or actual["size_bytes"] != reference["bytes"]:
        raise ValueError("Immutable shared publication metadata changed")
    return path


def _own_reference(root, path):
    path, root = Path(path), Path(root)
    relative = path.relative_to(root).as_posix()
    actual = descriptor(path)
    result = dict(relative_path=relative, sha256=actual["sha256"], bytes=actual["size_bytes"])
    _own_asset(root, result)
    return result


def _write_immutable_raw(path, raw):
    """Copy original completion bytes exactly, with the original exclusive fence."""
    _publish_bytes(Path(path), raw, create_parents=True)


DECISION_FORMAT = "darsia-attempt-decision-v1"
PUBLICATION_MARKER_FORMAT = "darsia-publication-marker-v1"
RETIRED_LEASE_FORMAT = "darsia-retired-lease-marker-v1"
OUTCOMES = frozenset({"publish", "reclaimed", "rejected"})


def decision_path(root: Path, run: str, token: str) -> Path:
    """Return the single decision file that retires one attempt."""
    if not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{32}", token):
        raise ValueError("Attempt decision requires a 32-hex attempt identity")
    return Path(root) / "decisions" / f"{name(run)}.{token}.json"


def read_decision(root: Path, run: str, token: str, *, reader=read_json) -> dict | None:
    """Read the standing decision of an attempt, or None while it is undecided."""
    path = decision_path(root, run, token)
    try:
        value = reader(path)
    except FileNotFoundError:
        return None
    run = name(run)
    if (not isinstance(value, dict) or value.get("format") != DECISION_FORMAT
            or value.get("run") != run or value.get("attempt_id") != token
            or value.get("outcome") not in OUTCOMES
            or value.get("lease_relative_path") != f"leases/{run}.{token}"):
        raise ValueError("Attempt decision differs from its run/attempt/outcome")
    return value


def decide(root: Path, run: str, token: str, outcome: str, **extra) -> tuple[bool, dict]:
    """Exclusively create the one decision of an attempt; a loser reads the winner.

    write_immutable is an exclusive create (POSIX link, Windows no-replace
    rename of a closed temporary file), so handles that remain open inside a
    lease directory cannot block this ownership transition.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"Unknown attempt decision outcome: {outcome!r}")
    path = decision_path(root, run, token)
    run = name(run)
    value = dict(format=DECISION_FORMAT, run=run, attempt_id=token, outcome=outcome,
                 lease_relative_path=f"leases/{run}.{token}", decided_utc=utc())
    if set(extra) & set(value):
        raise ValueError("Decision details cannot replace its identity fields")
    value.update(extra)
    try:
        write_immutable(path, value)
    except FileExistsError:
        existing = read_decision(root, run, token)
        if existing is None:
            raise RuntimeError("An exclusive attempt decision vanished") from None
        return False, existing
    return True, value


def publication_directory(root: Path, run: str) -> Path | None:
    """Return the published bundle of a run: a legacy moved directory or its marked lease."""
    root, run = Path(root), name(run)
    path = root / "publishing" / run
    if path.is_dir():
        return path  # Legacy layout: the whole lease directory was moved here.
    try:
        marker = read_json(path)
    except FileNotFoundError:
        return None
    token = marker.get("attempt_id") if isinstance(marker, dict) else None
    if (not isinstance(marker, dict)
            or set(marker) != {"format", "run", "attempt_id", "lease_relative_path"}
            or marker["format"] != PUBLICATION_MARKER_FORMAT or marker["run"] != run
            or not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{32}", token)
            or marker["lease_relative_path"] != f"leases/{run}.{token}"):
        raise ValueError("Publication marker differs from its run/attempt lease")
    decision = read_decision(root, run, token)
    if decision is None or decision["outcome"] != "publish":
        raise ValueError("Publication marker has no standing publish decision")
    return root / "leases" / f"{run}.{token}"


def retired_lease_directory(root: Path, run: str, token: str, *, reader=read_json) -> Path:
    """Return the lease directory of a reclaimed or rejected attempt.

    The lease stays in leases/<run>.<attempt>, and the marker file
    reclaimed/<run>/<attempt>/lease.json names it, so an open handle inside the
    lease can never block its retirement. Without a marker this is the legacy
    location reclaimed/<run>/<attempt>/lease, where older code moved the lease.
    """
    root, run = Path(root), name(run)
    archive = root / "reclaimed" / run / token
    try:
        marker = reader(archive / "lease.json")
    except (FileNotFoundError, NotADirectoryError):
        return archive / "lease"
    if (not isinstance(marker, dict)
            or set(marker) != {"format", "run", "attempt_id", "lease_relative_path"}
            or marker["format"] != RETIRED_LEASE_FORMAT or marker["run"] != run
            or marker["attempt_id"] != token
            or marker["lease_relative_path"] != f"leases/{run}.{token}"):
        raise ValueError("Retired-lease marker differs from its run/attempt lease")
    decision = read_decision(root, run, token, reader=reader)
    if decision is None or decision["outcome"] not in {"reclaimed", "rejected"}:
        raise ValueError("Retired-lease marker has no standing reclaim or reject decision")
    return root / "leases" / f"{run}.{token}"


def lease_retired(root: Path, run: str, token: str) -> bool:
    """Whether a reclaim or reject decision retired an attempt and its retry is recorded."""
    run = name(run)
    decision = read_decision(root, run, token)
    return (decision is not None and decision["outcome"] in {"reclaimed", "rejected"}
            and (Path(root) / "reclaimed" / run / token / "reclaimed.json").is_file())


def retired_claim_evidence(root, claimed, *, local_output=None, expected_stop=None,
                           expected_action="stop"):
    """Admit explicit cancellation from original metadata and exact owned epochs.

    No source/payload scan, process kill, timestamp shortcut or queue retry is
    involved. A Stop request or operator declaration alone is insufficient.
    """
    from ..job_control import read_control
    from ..job_launch_state import read_launch_requests
    from .worker import _same_process
    root, claimed = Path(root), Path(claimed)
    control = read_control(root)
    if expected_action not in {"stop", "pause"}:
        raise ValueError("Original Stop or Pause checkpoint command required")
    if (control["action"] != expected_action or not control.get("command")
            or expected_stop is not None and control["command"] != expected_stop):
        raise ValueError("Cancellation requires the exact recorded Stop command")
    claim_bound = descriptor(claimed)
    task = read_json(claimed)
    run, token = name(task["run"]), task["attempt_id"]
    decision_path(root, run, token)  # Validate the original attempt identity.
    lease = root / "leases" / (run + "." + token)
    owner_path = lease / "owner.json"
    owner_bound = descriptor(owner_path)
    owner = read_json(owner_path)
    service = name(owner["machine"])
    if (owner.get("task") != task or owner.get("token") != token
            or claimed != root / "claimed" / service / (run + ".json")
            or owner.get("actual_owner", {}).get("machine") != service):
        raise ValueError("Cancellation claim and original lease owner differ")
    if task.get("format") == "darsia-alias-task-v1":
        from .run_io import alias_resume_output, require_retired_source
        from ..job_filesystem import valid_results_admission
        if not isinstance(local_output, LocalOutputBinding) or not valid_results_admission(local_output.admission, local_output.results_root):
            raise ValueError("Cancellation requires this machine's admitted results root")
        output = alias_resume_output(owner, local_output, task, token)
    else:
        output = root / "results" / run / ".attempts" / token
        if PureWindowsPath(owner.get("output_path", "")) != PureWindowsPath(str(output)):
            raise ValueError("Cancellation output differs from the original canonical attempt")
    proof_path = output / "retirement.json"
    proof_bound = descriptor(proof_path)
    proof = read_json(proof_path)
    if task.get("format") == "darsia-alias-task-v1":
        require_retired_source(task, local_output, output, proof_path.read_bytes(), service)
    identities = proof.get("identities")
    supervisor = proof.get("supervisor", {})
    if (proof.get("complete") is not True or proof.get("all_absent") is not True
            or proof.get("worker_os_references_closed") is not True
            or proof.get("retirement_basis") != "owned_process_incarnations_absent"
            or proof.get("run") != run or proof.get("attempt_id") != token
            or PureWindowsPath(proof.get("attempt_directory", "")) != PureWindowsPath(str(output))
            or supervisor.get("machine") != service
            or not isinstance(identities, list) or not 0 < len(identities) <= 4096):
        raise ValueError("Original owned-process retirement evidence is incomplete")
    worker_epoch = lambda item: (item.get("pid"), item.get("process_create_time"))
    if worker_epoch(owner["actual_owner"]) not in {worker_epoch(item) for item in identities}:
        raise ValueError("Retirement does not bind the original claimed worker epoch")
    wrapper = proof.get("actual_launch_wrapper")
    if wrapper is not None and (proof.get("owned_launch_handle_retired") is not True
            or worker_epoch(wrapper) not in {worker_epoch(item) for item in identities}):
        raise ValueError("Original native launch wrapper has not retired")
    for original in [supervisor, *identities]:
        if (type(original.get("pid")) is not int or original["pid"] <= 0
                or not isinstance(original.get("process_create_time"), str)
                or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", original["process_create_time"]) is None):
            raise ValueError("Original process epoch is missing from retirement evidence")
        if _same_process(original) is not None:
            raise ValueError("An original worker or native launch owner is still running")
    requests = read_launch_requests(root)
    if not any(name(row["machine"] + "-" + row["backend"]) == service
               and row.get("terminal_report") for row in requests):
        raise ValueError("The original claim service has not reported stopping")
    if any(not row.get("terminal_report") and not row.get("preparation_failure")
           and row.get("state") != "Launcher did not create a process" for row in requests):
        raise ValueError("Another original service has not reported stopping")
    for log in (root / "logs").glob("*.log"):
        with log.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 128 * 1024))
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
        latest = None
        for line in lines:
            try:
                event = json.loads(line.partition(" ")[2])
            except ValueError:
                continue
            if event.get("kind") in {"started", "stopped", "startup_failure"}:
                latest = event["kind"]
        if latest == "started":
            raise ValueError("An original worker service has not reported stopping")
    if (descriptor(proof_path) != proof_bound or descriptor(claimed) != claim_bound
            or descriptor(owner_path) != owner_bound):
        raise ValueError("Original retirement evidence changed while read")
    return dict(claim=claim_bound, owner=owner_bound,
                retirement=proof_bound, stop=control["command"])


def cancel_retired_claim(root, claimed, *, evidence, actor, local_output=None):
    """Explicitly abandon a retired claim using the existing reject decision.

    This cancels operational ownership, never publishes a scientific failure
    or completion and never creates a retry. All job/output files are retained
    for the caller's separately confirmed deletion or inspection.
    """
    if not str(actor).strip():
        raise ValueError("An action name is required")
    current = retired_claim_evidence(root, claimed, local_output=local_output,
                                    expected_stop=evidence["stop"])
    if current != evidence:
        raise ValueError("Original claim retirement changed after review")
    task = read_json(claimed)
    won, decision = decide(Path(root), task["run"], task["attempt_id"], "rejected",
        reason="Explicit cancellation after original Stop and owned-process retirement",
        actor=str(actor).strip(), cancellation=True, original_retirement=evidence)
    if not won:
        raise LeaseLost("Another terminal decision already owns this claim")
    return decision_path(Path(root), task["run"], task["attempt_id"])


class JobQueue:
    """Filesystem protocol; workers and controllers share this implementation."""

    def __init__(self, root: Path, *, stale_seconds=600, clock=time.time, on_error=None,
                 server_clock=None, machine_id=None):
        """Open the shared queue; stale_seconds measures time since actual attempt progress."""
        self.root = Path(root)
        self.stale_seconds = stale_seconds
        self.clock = clock
        # clock remains a source-compatible argument, never a lease-time source.
        if type(stale_seconds) is not int or stale_seconds <= 0:
            raise ValueError("A positive integer actual-progress lease is required")
        self._server_clock = server_clock
        self._machine_id = name(machine_id) if machine_id is not None else None
        self._progress_lock = threading.Lock()
        self.on_error = on_error
        for directory in ("queue", "claimed", "leases", "decisions", "publishing", "reclaimed", "results", "done", "failed", "logs"):
            retry_transient(lambda path=self.root / directory: _ensure_directory(path))

    def _server(self, machine=None):
        if self._server_clock is None:
            identity = self._machine_id or (name(machine) if machine is not None else None)
            if identity is None:
                raise RuntimeError("An actual machine_id or injected server clock is required for lease timing")
            from .server_clock import ServerFileClock
            self._server_clock = ServerFileClock(self.root, identity)
        return self._server_clock

    def _old_progress(self, owner, lease):
        """Read only shared progress; never follow an alias owner's local output."""
        candidates = [lease / "owner.json", *lease.glob("progress_*.json")]
        if owner["task"].get("format") != "darsia-alias-task-v1":
            # Legacy output is proven canonical/shared before it is inspected.
            expected = self.root / "results" / name(owner["task"]["run"]) / ".attempts" / owner["token"]
            if PureWindowsPath(owner["output_path"]) != PureWindowsPath(str(expected)):
                raise ValueError("Legacy progress path differs from its canonical shared attempt")
            candidates.extend(expected.glob("progress_*.json"))
        from .server_clock import file_mtime_ns
        return max(candidates, key=file_mtime_ns)

    def _prior_resume_lineage(self, run, token):
        """Bind only the unique actual predecessor of this queue generation."""
        matches = []
        for path in (self.root / "reclaimed" / run).glob("*/reclaimed.json"):
            value = read_json(path)
            if value.get("next_attempt") == token:
                if value.get("run") != run:
                    raise ValueError("Reclaimed predecessor belongs to another run")
                matches.append(path)
        if len(matches) > 1:
            raise ValueError("Queue generation has more than one actual predecessor")
        return _own_reference(self.root, matches[0]) if matches else None

    def enqueue(self, task: dict) -> Path:
        """Reserve one whole-run output identity and publish its immutable pending task."""
        validate_task(task)
        run = name(task["run"])
        result_root = self.root / "results" / run
        output = ("results/" + run if task.get("format") == "darsia-alias-task-v1"
                  else task.get("output_directory", str(result_root)))
        value = dict(task, output_directory=output, attempt_id=uuid.uuid4().hex)
        self._validate_output(value)
        _create_directory(result_root)  # Also reserves the case-insensitive run identity.
        write_immutable(result_root / "task.json", value)
        destination = self.root / "queue" / (run + ".json")
        write_immutable(destination, value)
        return destination

    def _validate_output(self, task):
        actual = PureWindowsPath(task["output_directory"])
        expected = PureWindowsPath(str(self.root / "results" / name(task["run"])))
        if task.get("format") == "darsia-alias-task-v1":
            if task["output_directory"] != "results/" + name(task["run"]):
                raise ValueError("Alias task output must be its relative canonical run identity")
        else:
            if actual != expected or ".." in actual.parts:
                raise ValueError("Task output_directory must be its canonical job/results/run root")
            if PureWindowsPath(str(self.root)).drive.startswith("\\\\"):
                require_unc(task["output_directory"])
        job = self.root / "job.json"
        if job.exists():
            value = read_json(job)
            if "expected_runs" in value and task["run"] not in value["expected_runs"]:
                raise ValueError("Task is outside job.json expected_runs")

    def claim(self, machine: str, backend: str, owner: dict, *, availability=None,
              output_factory=None, publication_factory=None, exclude_runs=(),
              claim_controls=None, checkpoint=None) -> Claim | None:
        """Atomically claim one backend-compatible run and bind a fresh attempt lease to its owner."""
        machine = name(machine)
        if isinstance(exclude_runs, (str, bytes)):
            raise ValueError("exclude_runs must be an explicit worker-incarnation collection")
        excluded = {name(run) for run in exclude_runs}
        for pending in retry_transient(lambda: sorted((self.root / "queue").glob("*.json"))):
            if checkpoint is not None:
                checkpoint()
            try:
                task = read_json(pending)
                validate_task(task)
                self._validate_output(task)
                if pending.name != name(task["run"]) + ".json":
                    raise ValueError("Queue filename differs from task run")
                if not re.fullmatch(r"[0-9a-f]{32}", task.get("attempt_id", "")):
                    raise ValueError("Queue attempt identity is missing or malformed")
            except FileNotFoundError:
                continue
            except (ValueError, TypeError, KeyError):
                self._quarantine(pending, machine, owner, traceback.format_exc())
                continue
            except OSError as error:
                if not transient_os_error(error):
                    raise
                self._transient("claim", pending, error)  # Try the next task (N111).
                continue
            if task.get("backend", "any") not in {"any", backend}:
                continue
            if name(task["run"]) in excluded:
                continue
            alias = task.get("format") == "darsia-alias-task-v1"
            rows = ()
            before_task = deepcopy(task)
            if alias:
                job = read_json(self.root / "job.json")
                if (job.get("format") != "darsia-alias-job-v1"
                        or job.get("execution_ready") is not True
                        or job.get("runtime_support") != "darsia-role-io-runtime-v1"
                        or task.get("execution_ready") is not True
                        or task["input_catalog"].get("runtime_support") != "darsia-role-io-runtime-v1"):
                    raise RuntimeError("Prepared J2 metadata is not explicitly ready for J3 execution")
                if self._machine_id is None:
                    raise RuntimeError("Alias claim requires its explicit physical store machine_id")
                if not all(callable(callback) for callback in (availability, output_factory, publication_factory)):
                    raise RuntimeError("Alias claim requires availability, admitted local output and shared publication bindings")
                if not callable(getattr(publication_factory, "publish_progress", None)):
                    raise RuntimeError("Alias claim requires shared original frame-commit publication")
                # Availability includes original prefix/retirement admission.
                # None explicitly skips this machine; True is never admission.
                from .store_runtime import ResolvedInput, RunUnavailable
                try:
                    rows = availability(deepcopy(task))
                except RunUnavailable:
                    continue
                if checkpoint is not None:
                    checkpoint()
                if rows is None:
                    continue
                if (not isinstance(rows, tuple) or not rows
                        or any(not isinstance(row, ResolvedInput) for row in rows)
                        or [row.reference for row in rows] !=
                           [item["reference"] for item in task["input_catalog"]["inputs"]]
                        or [row.uses for row in rows] !=
                           [tuple(item["uses"]) for item in task["input_catalog"]["inputs"]]):
                    raise ValueError("Availability must bind the exact ordered role catalogue; boolean admission is forbidden")
            from contextlib import nullcontext
            if task.get("pipeline") == "series-setup" and not callable(claim_controls):
                raise RuntimeError("Setup needs its finite admitted claim-control scope")
            control_scope = nullcontext() if claim_controls is None else claim_controls(task, machine)
            with control_scope:
                if checkpoint is not None:
                    checkpoint()
                run, token = name(task["run"]), task["attempt_id"]
                if pending.name != run + ".json" or not re.fullmatch(r"[0-9a-f]{32}", token):
                    raise ValueError("Queue filename or attempt identity differs")
                claimed = self.root / "claimed" / machine / pending.name
                from .series_setup_sink import current_writer
                writer = current_writer()
                retry_transient(lambda: claimed.parent.mkdir(parents=True, exist_ok=True)
                    if writer is None else writer.mkdir(claimed.parent, parents=True, exist_ok=True))
                try:
                    if claimed.exists():
                        continue
                    rename(pending, claimed)
                except FileNotFoundError:
                    continue
                except OSError as error:
                    if not transient_os_error(error):
                        raise
                    self._transient("claim", pending, error)  # Try the next task (N111).
                    continue
                # Bind the generation actually renamed, not a pre-rename read.
                task = read_json(claimed)
                validate_task(task)
                self._validate_output(task)
                token = task["attempt_id"]
                if alias and task != before_task:
                    # A different generation won while availability was inspected.
                    # Never pair stale rows with the generation actually renamed.
                    rename(claimed, pending)
                    continue
                if task.get("backend", "any") not in {"any", backend}:
                    rename(claimed, pending)
                    continue
                lease = self.root / "leases" / (run + "." + token)
                try:
                    _create_directory(lease)
                except FileExistsError:
                    # This generation already has a lease (live, published or
                    # retired in place); a second queue file of it is set aside.
                    if task.get("pipeline") == "series-setup":
                        raise LeaseLost("Duplicate Setup generation retained at its original claimed path")
                    rename(claimed, claimed.with_name(f"{run}.duplicate.{uuid.uuid4().hex}.json"))
                    continue
                output_reference = None
                if alias:
                    binding = output_factory(deepcopy(task), token)
                    output, output_reference = self._admit_output(binding, task, machine, token)
                else:
                    output = self.root / "results" / run / ".attempts" / token
                _create_directory(output, parents=True)
                marker = lease / "claimed.json"
                write_immutable(marker, dict(run=run, attempt_id=token, machine_id=machine,
                                            event="actual_claim_bound"), create_parents=False)
                from .server_clock import file_mtime_ns
                metadata = dict(task=task, machine=machine, token=token, actual_owner=owner,
                                claimed_unix_ns=file_mtime_ns(marker), claimed_utc=utc(),
                                lease_clock_basis="shared-filesystem-mtime")
                if alias:
                    metadata.update(format="darsia-alias-lease-owner-v1",
                                    claimed=_own_reference(self.root, claimed), output=output_reference)
                else:
                    metadata.update(claimed_path=str(claimed), output_path=str(output))
                write_immutable(lease / "owner.json", metadata, create_parents=False)
                claim = Claim(task, machine, token, claimed, lease, output, rows,
                              publication_factory if alias else None, binding if alias else None)
                if read_json(claimed) != task:
                    raise LeaseLost("Queue generation changed during claim")
                self.assert_owner(claim)
                return claim
        return None

    def _admit_output(self, binding, task, machine, token):
        from ..job_filesystem import valid_results_admission
        if (not isinstance(binding, LocalOutputBinding)
                or not valid_results_admission(binding.admission, binding.results_root)
                or self._machine_id is None
                or name(binding.machine_id) != self._machine_id):
            raise ValueError("A positive configured result-root admission is required")
        root, output = Path(binding.results_root), Path(binding.path)
        relative = name(task["job_id"]) + "/" + name(task["run"]) + "/.attempts/" + token
        expected = root.joinpath(*relative.split("/"))
        if (not root.is_absolute() or not output.is_absolute()
                or output != expected or ".." in output.parts):
            raise ValueError("Output differs from its declared results root/job/run/token")
        checked = Path(binding.admission.checked_container).resolve(strict=True)
        actual_root = root.resolve()
        if (not actual_root.is_relative_to(checked)
                or not output.resolve().is_relative_to(actual_root)):
            raise ValueError("Output escaped its positively admitted container")
        return output, dict(machine_id=binding.machine_id, root_alias="results", relative_path=relative)

    def _quarantine(self, pending, machine, owner, text):
        try:
            run = name(pending.stem)
        except ValueError:
            run = "invalid-" + hashlib.sha256(pending.name.encode()).hexdigest()[:24]
        token = uuid.uuid4().hex
        archived = self.root / "claimed" / machine / f"{run}.invalid.{token}.json"
        retry_transient(lambda: _ensure_directory(archived.parent))
        try:
            rename(pending, archived)
        except FileNotFoundError:
            return
        value = dict(complete=False, run=run, attempt_id=token,
                     state="invalid_task", traceback=text,
                     original_queue_bytes=descriptor(archived), actual_worker=owner,
                     failed_utc=utc(), scientific_computation_started=False)
        attempt = self.root / "failed" / ".attempts" / run / (token + ".json")
        write_immutable(attempt, value)
        try:
            write_immutable(self.root / "failed" / (run + ".json"), value)
        except FileExistsError:
            pass  # Keep the first terminal failure and this distinct attempt.
        if self.on_error is not None:
            self.on_error(dict(kind="invalid_task", **value))

    def assert_owner(self, claim: Claim) -> None:
        """Require the exact current claim generation before writing or publishing."""
        try:
            owner = read_json(claim.lease / "owner.json")
            task = read_json(claim.claimed)
        except FileNotFoundError as exc:
            raise LeaseLost(f"Attempt {claim.token} no longer owns {claim.task['run']}") from exc
        if owner["token"] != claim.token or task != claim.task:
            raise LeaseLost("Claim identity changed")
        decision = read_decision(self.root, claim.task["run"], claim.token)
        if decision is not None and decision["outcome"] != "publish":
            raise LeaseLost(f"Attempt {claim.token} was {decision['outcome']} for {claim.task['run']}")

    def progress(self, claim: Claim, event: dict) -> None:
        """Record an actual I/O or frame-advance event with a nanosecond timestamp under ownership."""
        with self._progress_lock:
            self.assert_owner(claim)
            sequence = retry_transient(lambda: len(list(claim.output.glob("progress_*.json"))))
            value = dict(
                run=claim.task["run"], attempt_id=claim.token, sequence=sequence,
                event=event, written_unix_ns=self._server(claim.machine).now_ns(), written_utc=utc(),
            )
            # Keep the original local operational record; it is never read by
            # another machine as lease evidence or an implicit payload source.
            write_immutable(claim.output / f"progress_{sequence:08d}.json", value)
            shared = dict(value)
            if claim.task.get("format") == "darsia-alias-task-v1":
                # Preserve the original event as passive producer provenance.
                # No consumer needs to open its producer-local paths: actual
                # frame evidence is the separately verified shared byte copy.
                self._portable_event(event)  # Validate actual progress fields.
                if event["state"] == "frame_completed":
                    shared["published_frame"] = self._frame_progress_reference(claim, event)
                if event["state"] == "phase1_frame_completed":
                    from .phase1_protocol import shared_phase1_progress
                    shared["published_phase1_frame"] = shared_phase1_progress(self, claim, event)
                    shared["detail_scope"] = "Original phase-one event provenance; exact phase1_commit bytes are published_phase1_frame within this shared bundle"
                else:
                    shared["detail_scope"] = "Original event provenance; frame evidence is published_frame within this shared bundle"
            write_immutable(claim.lease / f"progress_{sequence:08d}.json", shared,
                            create_parents=False)
            self.assert_owner(claim)

    def _frame_progress_reference(self, claim, event):
        """Bind the original committed-frame metadata without foreign-local reads."""
        publisher = getattr(claim.publication_factory, "publish_progress", None)
        if not callable(publisher):
            raise RuntimeError("Alias frame progress requires an original commit byte-copy publisher")
        record = event.get("publication")
        case = event.get("case_id")
        if (event.get("complete") is not True or event.get("run") != claim.task["run"]
                or case not in claim.task["ordered_case_ids"] or not isinstance(record, dict)
                or record.get("frame_id") != case):
            raise ValueError("Frame progress differs from this claimed task and original publication")
        original = record["files"]["commit.json"]
        size = original.get("bytes", original.get("size_bytes"))
        if (not re.fullmatch(r"[0-9a-f]{64}", original.get("sha256", ""))
                or type(size) is not int or not 0 <= size <= 128 * 1024 * 1024):
            raise ValueError("Original committed-frame SHA and size are required")
        self.assert_owner(claim)
        reference = publisher(claim, deepcopy(event))
        self.assert_owner(claim)
        copied = _own_asset(claim.lease, reference)
        if reference["sha256"] != original["sha256"] or reference["bytes"] != size:
            raise ValueError("Published frame is not the original commit.json bytes")
        commit = read_json(copied)
        expected_files = {key: value for key, value in record["files"].items() if key != "commit.json"}
        if (commit.get("complete") is not True or commit.get("frame_id") != case
                or commit.get("manifest_sha256") != record.get("manifest_sha256")
                or commit.get("files") != expected_files):
            raise ValueError("Published original frame commit differs from its publication record")
        return deepcopy(reference)

    @staticmethod
    def _portable_event(event):
        if not isinstance(event, dict) or not isinstance(event.get("state"), str):
            raise ValueError("An actual named I/O/frame progress event is required")
        name(event["state"])
        result = {"state": event["state"]}
        for key in ("run", "case_id"):
            if key in event:
                name(event[key])
                result[key] = event[key]
        for key in ("frame_index", "bytes"):
            if key in event:
                if type(event[key]) is not int or event[key] < 0:
                    raise ValueError("Actual progress counters must be nonnegative integers")
                result[key] = event[key]
        if "complete" in event:
            if type(event["complete"]) is not bool:
                raise ValueError("Actual completion progress must be a boolean")
            result["complete"] = event["complete"]
        if "compute_seconds" in event:
            if type(event["compute_seconds"]) not in {int, float} or event["compute_seconds"] < 0:
                raise ValueError("Actual compute time must be a nonnegative number")
            result["compute_seconds"] = event["compute_seconds"]
        return result

    def reclaim_stale(self) -> list[dict]:
        """Archive expired claim generations and enqueue fresh attempts without allowing stale writers."""
        reclaimed = []
        for lease in retry_transient(lambda: sorted((self.root / "leases").iterdir())):
            try:
                owner = read_json(lease / "owner.json")
            except FileNotFoundError:
                continue
            except OSError as error:
                if not transient_os_error(error):
                    raise
                self._transient("reclaim", lease, error)  # A later pass reads it (N111).
                continue
            task, token = owner["task"], owner["token"]
            run = name(task["run"])
            decision = read_decision(self.root, run, token)
            outcome = None if decision is None else decision["outcome"]
            if outcome == "rejected" or (outcome == "publish" and (
                    (self.root / "publishing" / run).exists()
                    or not (lease / "publication.json").exists())):
                # Published leases stay in leases/ by design; a rejection
                # belongs to its rejector.
                continue
            if outcome == "reclaimed" and (
                    self.root / "reclaimed" / run / token / "reclaimed.json").exists():
                continue  # Retired in place, and its retry is recorded.
            try:
                progress = self._old_progress(owner, lease)
                from .server_clock import file_mtime_ns
                latest_ns = file_mtime_ns(progress)
                if outcome != "reclaimed" and not self._server().stale(progress, self.stale_seconds):
                    continue
            except FileNotFoundError:
                # Another reclaimer/rejection may have atomically archived this lease.
                continue
            claimed_path = (self.root.joinpath(*owner["claimed"]["relative_path"].split("/"))
                            if task.get("format") == "darsia-alias-task-v1"
                            else Path(owner["claimed_path"]))
            if task.get("format") == "darsia-alias-task-v1":
                expected = self.root / "claimed" / name(owner["machine"]) / (run + ".json")
                if claimed_path != expected:
                    raise ValueError("Portable claim reference differs from its canonical shared location")
            if outcome == "publish":
                # A stale publisher stopped between its decision and its run marker.
                self._write_publication_marker(run, token)
                self._finish_publication(lease)
                continue
            if decision is None:
                if (lease / "publication.json").exists():
                    # N67: the products of this attempt are saved. Finish it by
                    # publication; a published run is never re-run.
                    _won, decision = decide(self.root, run, token, "publish",
                                            decided_by="reclaim_stale_recovery")
                    if decision["outcome"] == "publish":
                        self._write_publication_marker(run, token)
                        self._finish_publication(lease)
                    continue
                won, decision = decide(self.root, run, token, "reclaimed",
                                       last_actual_progress_unix_ns=latest_ns,
                                       next_attempt=uuid.uuid4().hex)
                if not won:
                    continue  # A publisher or another reclaimer decided first.
            # A standing reclaim decision. The lease stays in leases/ and a
            # marker names it. Every step below is idempotent, so a pass that
            # was interrupted, or runs at the same time, simply repeats it.
            latest_ns = decision.get("last_actual_progress_unix_ns", latest_ns)
            archive = self._retire_lease(run, token)
            busy = False
            for old in (claimed_path, self.root / "queue" / (run + ".json")):
                try:
                    if read_json(old).get("attempt_id") == token:
                        rename(old, archive / "task.json")
                except (FileNotFoundError, FileExistsError):
                    pass  # Already archived by this or a concurrent pass.
                except OSError as error:
                    if not transient_os_error(error):
                        raise
                    self._transient("reclaim", old, error)
                    busy = True
            if busy:
                continue  # A later pass repeats these idempotent steps (N111).
            if (archive / "reclaimed.json").exists():
                continue  # An earlier pass already enqueued and recorded this retry.
            if (self.root / "publishing" / run).exists() and not (
                    decision.get("next_attempt")
                    and self._attempt_exists(run, decision["next_attempt"])):
                # Only this decision's own retry can have published the run.
                raise RuntimeError("Reclaimed run already has a publication owner")
            retry = self._reclaim_retry(task, token, decision)
            record = dict(run=task["run"], previous_attempt=token,
                          next_attempt=retry["attempt_id"], reclaimed_utc=utc(),
                          last_actual_progress_unix_ns=latest_ns)
            try:
                write_immutable(archive / "reclaimed.json", record)
            except FileExistsError:
                continue  # A concurrent pass recorded the same retry.
            reclaimed.append(record)
        # A process can die after the task rename but before owner.json exists.
        # No science starts before that owner document and the ownership check.
        for claimed in retry_transient(lambda: sorted((self.root / "claimed").glob("*/*.json"))):
            if ".invalid." in claimed.name or ".duplicate." in claimed.name:
                continue
            try:
                task = read_json(claimed)
                run, token = name(task["run"]), task["attempt_id"]
                lease = self.root / "leases" / (run + "." + token)
                if (lease / "owner.json").exists() or (self.root / "publishing" / run).exists():
                    continue
                # Rename can preserve an arbitrarily old queue mtime. Start
                # this ownerless window at the first shared observation instead.
                marker = self.root / "claim_observations" / (run + "." + token + ".json")
                try:
                    write_immutable(marker, dict(run=run, attempt_id=token,
                        claimed_relative_path=claimed.relative_to(self.root).as_posix(),
                        state="ownerless_claim_first_observed"))
                except FileExistsError:
                    pass
                observation = read_json(marker)
                if (observation.get("run") != run or observation.get("attempt_id") != token
                        or observation.get("claimed_relative_path") != claimed.relative_to(self.root).as_posix()):
                    raise ValueError("Ownerless observation belongs to another claim")
                from .server_clock import file_mtime_ns
                last_ns = file_mtime_ns(marker)
                if not self._server().stale(marker, self.stale_seconds):
                    continue
                prior_lineage = self._prior_resume_lineage(run, token)
                decision = read_decision(self.root, run, token)
                if decision is None:
                    won, decision = decide(self.root, run, token, "reclaimed",
                                           last_actual_progress_unix_ns=last_ns,
                                           next_attempt=uuid.uuid4().hex)
                    if not won:
                        continue  # A publisher or retirement decided first.
                if decision["outcome"] != "reclaimed":
                    continue
                # A standing reclaim decision whose claim was not archived yet,
                # also when an earlier reclaim pass stopped after its decision.
                # A lease that exists stays in place and is only marked.
                archive = self.root / "reclaimed" / run / token
                retry_transient(lambda: _ensure_directory(archive))
                if lease.exists():
                    self._retire_lease(run, token)
                rename(claimed, archive / "task.json")
            except FileNotFoundError:
                continue
            except FileExistsError:
                continue
            except OSError as error:
                if not transient_os_error(error):
                    raise
                self._transient("reclaim", claimed, error)  # A later pass repeats it (N111).
                continue
            if (self.root / "publishing" / run).exists():
                continue
            if (archive / "reclaimed.json").exists():
                continue  # An earlier pass already enqueued and recorded this retry.
            present = archive / "lease" if (archive / "lease").exists() else lease
            retry = self._reclaim_retry(task, token, decision)
            record = dict(run=task["run"], previous_attempt=token,
                          next_attempt=retry["attempt_id"], reclaimed_utc=utc(),
                          last_actual_progress_unix_ns=last_ns,
                          # An owner may have appeared between the initial
                          # observation and the atomic retirement. Preserve all
                          # ordinary prefix/OS-retirement guards in that case.
                          interrupted_claim_before_science=not (present / "owner.json").exists(),
                          prior_resume_lineage=prior_lineage,
                          preserve_prior_resume_lineage=True)
            try:
                write_immutable(archive / "reclaimed.json", record)
            except FileExistsError:
                continue  # A concurrent pass recorded the same retry.
            reclaimed.append(record)
        return reclaimed

    def _reclaim_retry(self, task, token, decision=None):
        """Enqueue one fresh generation; reuse the one an interrupted archive pass queued.

        A decision that names next_attempt fixes the identity of the retry, so
        a repeated or concurrent pass enqueues it only while that generation
        exists nowhere yet (queued, claimed, leased or decided).
        """
        run = name(task["run"])
        pending = self.root / "queue" / (run + ".json")
        try:
            existing = read_json(pending)
        except FileNotFoundError:
            existing = None
        if (isinstance(existing, dict) and existing.get("attempt_id") != token
                and {key: value for key, value in existing.items() if key != "attempt_id"}
                    == {key: value for key, value in task.items() if key != "attempt_id"}):
            return existing
        planned = (decision or {}).get("next_attempt")
        if planned is None:
            retry = dict(task, attempt_id=uuid.uuid4().hex)
            write_immutable(pending, retry)  # Any other queued file still refuses, as before.
            return retry
        retry = dict(task, attempt_id=planned)
        if not self._attempt_exists(run, planned):
            try:
                write_immutable(pending, retry)
            except FileExistsError:
                if not self._attempt_exists(run, planned):
                    raise  # Another generation is queued; refuse, as before.
        return retry

    def _attempt_exists(self, run, attempt):
        """Whether a queue generation exists anywhere: queued, claimed, leased or decided."""
        # A claim renames queue -> claimed and then creates the lease, so the
        # places are read in that order and a moving generation is still seen.
        try:
            if read_json(self.root / "queue" / (run + ".json")).get("attempt_id") == attempt:
                return True
        except FileNotFoundError:
            pass
        for claimed in (self.root / "claimed").glob("*/" + run + ".json"):
            try:
                if read_json(claimed).get("attempt_id") == attempt:
                    return True
            except FileNotFoundError:
                continue
        return ((self.root / "leases" / f"{run}.{attempt}").exists()
                or decision_path(self.root, run, attempt).exists())

    def _transient(self, step, path, error):
        """Report a transient file error after which a queue step was deferred (N111)."""
        if self.on_error is not None:
            self.on_error(dict(kind="transient_file_error", step=step, path=str(path),
                               error=repr(error), deferred=True, utc=utc()))

    def _retire_lease(self, run, token):
        """Mark a reclaimed or rejected lease in place; never move it. Returns its archive."""
        archive = self.root / "reclaimed" / run / token
        retry_transient(lambda: _ensure_directory(archive))
        marker = archive / "lease.json"
        if not (archive / "lease").is_dir() and not marker.exists():
            try:
                write_immutable(marker, dict(
                    format=RETIRED_LEASE_FORMAT, run=run, attempt_id=token,
                    lease_relative_path=f"leases/{run}.{token}"))
            except FileExistsError:
                pass  # A concurrent pass marked it.
        return archive

    def reject_before_science(self, claim: Claim, reason: str, owner: dict) -> dict:
        """Retire only an empty alias admission, then let another machine try it.

        Worker code must add the returned run to its incarnation's exclude_runs.
        No terminal failed/done record is produced. This is not OS retirement
        and cannot be used to waive an existing scientific prefix/retirement.
        """
        if claim.task.get("format") != "darsia-alias-task-v1":
            raise ValueError("Pre-science input rejection is only an alias admission path")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("The literal input-admission failure is required")
        self.assert_owner(claim)
        if any(claim.output.iterdir()) or list(claim.lease.glob("progress_*.json")):
            raise ValueError("Pre-science rejection cannot discard an attempt with output/progress")
        run = name(claim.task["run"])
        prior_lineage = self._prior_resume_lineage(run, claim.token)
        won, decision = decide(self.root, run, claim.token, "rejected",
                               next_attempt=uuid.uuid4().hex)
        if not won:
            raise LeaseLost("Reclaim/publication already retired this input admission")
        # The lease stays in leases/; a marker in the archive names it.
        archive = self._retire_lease(run, claim.token)
        if read_json(claim.claimed) != claim.task:
            raise LeaseLost("Input rejection lost its exact claim generation")
        rename(claim.claimed, archive / "task.json")
        if (self.root / "publishing" / run).exists():
            raise RuntimeError("Input-rejected run already has a publication owner")
        retry = dict(claim.task, attempt_id=decision["next_attempt"])
        record = dict(run=claim.task["run"], previous_attempt=claim.token,
                      next_attempt=retry["attempt_id"], reclaimed_utc=utc(),
                      interrupted_claim_before_science=True, input_admission_rejected=True,
                      prior_resume_lineage=prior_lineage,
                      preserve_prior_resume_lineage=True,
                      reason=reason, actual_worker=owner,
                      exclude_run_for_this_worker_incarnation=True)
        write_immutable(archive / "reclaimed.json", record)
        write_immutable(self.root / "queue" / (run + ".json"), retry)
        return record

    def _publish(self, claim: Claim, document: dict, success: bool) -> dict:
        alias = claim.task.get("format") == "darsia-alias-task-v1"
        if alias and not callable(claim.publication_factory):
            raise RuntimeError("Alias publication requires its original immutable shared-bundle binding")
        candidate = claim.output / ("candidate_completed.json" if success else "failed.json")
        try:
            write_immutable(candidate, document)
        except FileExistsError:
            # Only a success retry of this attempt reuses its identical original bytes.
            if not success or candidate.read_bytes() != _immutable_bytes(document):
                raise
        self.assert_owner(claim)
        run = name(claim.task["run"])
        existing_path = claim.lease / "publication.json"
        if not success and existing_path.exists():
            existing = read_json(existing_path)
            if existing.get("success") is True:
                # A failure after the prepared success must retain the original
                # success bundle for recovery, never rewrite publication.json
                # with a second failure publication or make another decision.
                if existing.get("attempt_id") != claim.token:
                    raise ValueError("Existing success publication belongs to another attempt")
                if alias:
                    if (existing.get("format") != "darsia-alias-publication-v1"
                            or existing.get("run") != claim.task["run"]):
                        raise ValueError("Existing success publication belongs to another run")
                    _own_asset(claim.lease, existing["candidate"])
                    index = _own_asset(claim.lease, existing["products_index"])
                    self._validate_alias_index(index, existing["candidate"], claim.task["run"], claim.token)
                    self._validate_alias_payload(claim.lease, existing.get("payload_publication"),
                        existing["products_index"], claim.task["run"], claim.token)
                else:
                    original = claim.output / "candidate_completed.json"
                    if existing.get("candidate") != descriptor(original):
                        raise ValueError("Existing success candidate identity differs")
                self._publication_diagnostic(claim, 0, "failure_after_prepared_success", dict(
                    original_failure=descriptor(candidate), preserved_success=descriptor(existing_path),
                    no_publication_rewrite=True, no_ownership_move=True))
                return document
        # Prepare recovery metadata before the ownership transition. A crash
        # after the decision can then finish publication without scientific replay.
        prepared = None
        if alias:
            original = descriptor(candidate)
            prepared = claim.publication_factory(claim, dict(original), success)
            self.assert_owner(claim)  # Copies never acquire publication rights.
            required = {"candidate", "products_index"} if success else {"candidate"}
            optional = ({"payload_publication"} if success else set()) | {"n43_reference_guard"}
            if (not isinstance(prepared, dict) or not required <= set(prepared)
                    or not set(prepared) <= required | optional):
                raise ValueError("Explicit shared candidate/index publication descriptors required")
            _own_asset(claim.lease, prepared["candidate"])
            if (prepared["candidate"]["sha256"] != original["sha256"]
                    or prepared["candidate"]["bytes"] != original["size_bytes"]):
                raise ValueError("Shared candidate is not the original Main completion/failure bytes")
            if success:
                index_path = _own_asset(claim.lease, prepared["products_index"])
                self._validate_alias_index(index_path, prepared["candidate"], claim.task["run"], claim.token)
                self._validate_alias_payload(claim.lease, prepared.get("payload_publication"),
                    prepared["products_index"], claim.task["run"], claim.token)
        if alias:
            self._check_n43_reference_publication(prepared,claim.task,success)
        try:
            publication = dict(attempt_id=claim.token, success=success,
                               candidate=descriptor(candidate))
            if alias:
                publication = dict(format="darsia-alias-publication-v1", run=claim.task["run"],
                                   attempt_id=claim.token, success=success, **prepared)
            write_immutable(existing_path, publication, create_parents=False)
        except FileNotFoundError as exc:
            raise LeaseLost("Reclaim closed the publication lease") from exc
        except FileExistsError:
            # A success retry of this attempt accepts only its identical record.
            if not success or existing_path.read_bytes() != _immutable_bytes(publication):
                raise
        self.assert_owner(claim)
        if alias:
            self._check_n43_reference_publication(publication,claim.task,success)
        # The ownership transition is one exclusive-create decision file. The
        # lease directory never moves, so handles open inside it cannot block it.
        self._shared_target(decision_path(self.root, run, claim.token))
        _won, decision = decide(self.root, run, claim.token, "publish")
        if decision["outcome"] != "publish":
            raise LeaseLost("Reclaim won before publication")
        self._write_publication_marker(run, claim.token)
        self._finish_publication(claim.lease)
        return document

    def _write_publication_marker(self, run, token):
        """Bind the run to its in-place published lease; an identical marker is a retry."""
        run = name(run)
        path = self.root / "publishing" / run
        marker = dict(format=PUBLICATION_MARKER_FORMAT, run=run, attempt_id=token,
                      lease_relative_path=f"leases/{run}.{token}")
        self._shared_target(path)
        if not path.exists():
            try:
                write_immutable(path, marker)
                return
            except FileExistsError:
                pass
        if path.is_dir():
            raise RuntimeError("Run already has a legacy publication directory")
        if read_json(path) != marker:
            raise RuntimeError("Run already has a publication marker of another attempt")

    def _publication_diagnostic(self, claim, attempt, phase, value):
        """Immutable owned-job metadata; never a stream left open on the lease."""
        run, token = name(claim.task["run"]), name(claim.token)
        path = self.root / "logs" / f"publication_{run}_{token}_{attempt:02}_{phase}.json"
        self._shared_target(path)
        write_immutable(path, dict(utc=utc(), run=claim.task["run"], attempt_id=claim.token,
            physical_attempt=attempt, phase=phase, **value))

    def _finish_publication(self, directory: Path) -> None:
        publication = read_json(directory / "publication.json")
        if publication.get("format") == "darsia-alias-publication-v1":
            self._finish_alias_publication(directory, publication)
            return
        candidate = publication["candidate"]
        actual = descriptor(Path(candidate["path"]))
        if actual != candidate:
            raise ValueError("Immutable candidate changed")
        value = read_json(Path(candidate["path"]))
        run = name(value["run"])
        target = (self.root / "results" / run / "completed.json"
                  if publication["success"] else self.root / "failed" / (run + ".json"))
        if target.exists():
            if read_json(target) != value:
                raise ValueError("A different terminal result already exists")
        else:
            try:
                write_immutable(target, value)
            except FileExistsError:
                if read_json(target) != value:
                    raise ValueError("A different terminal result already exists")
        if publication["success"]:
            done = dict(run=value["run"], attempt_id=publication["attempt_id"],
                        completed=descriptor(target))
            done_path = self.root / "done" / (run + ".json")
            if done_path.exists():
                if read_json(done_path) != done:
                    raise ValueError("Done identity changed")
            else:
                try:
                    write_immutable(done_path, done)
                except FileExistsError:
                    if read_json(done_path) != done:
                        raise ValueError("Done identity changed")

    def _finish_alias_publication(self, directory, publication):
        """Finish only the immutable published shared bundle, never producer C: paths."""
        owner = read_json(directory / "owner.json")
        task, token = owner["task"], owner["token"]
        run = name(task["run"])
        if (task.get("format") != "darsia-alias-task-v1"
                or publication.get("run") != task["run"]
                or publication.get("attempt_id") != token
                or type(publication.get("success")) is not bool):
            raise ValueError("Shared publication owner/run/attempt differs")
        source = _own_asset(directory, publication["candidate"])
        raw = source.read_bytes()
        if (len(raw) != publication["candidate"]["bytes"]
                or hashlib.sha256(raw).hexdigest() != publication["candidate"]["sha256"]):
            raise ValueError("Immutable shared candidate changed during recovery")
        value = read_json(source)
        if value.get("run") != task["run"] or value.get("complete") is not publication["success"]:
            raise ValueError("Shared candidate run/completion differs")
        if publication["success"]:
            if value.get("worker_references_closed") is not True:
                raise ValueError("Shared completion still has open scientific references")
            index_path = _own_asset(directory, publication["products_index"])
            self._validate_alias_index(index_path, publication["candidate"], task["run"], token)
            self._validate_alias_payload(directory, publication.get("payload_publication"),
                publication["products_index"], task["run"], token)
        target = (self.root / "results" / run / "completed.json"
                  if publication["success"] else self.root / "failed" / (run + ".json"))
        self._shared_target(target)
        self._check_n43_reference_publication(publication,task,publication["success"])
        if target.exists():
            if target.read_bytes() != raw:
                raise ValueError("A different terminal result already exists")
        else:
            try:
                _write_immutable_raw(target, raw)
            except FileExistsError:
                if target.read_bytes() != raw:
                    raise ValueError("A different terminal result already exists")
        if publication["success"]:
            done = dict(format="darsia-alias-done-v1", run=task["run"], attempt_id=token,
                        completed=_own_reference(self.root, target),
                        products_index=_own_reference(self.root, index_path))
            if "payload_publication" in publication:
                payload = _own_asset(directory, publication["payload_publication"])
                done["payload_publication"] = _own_reference(self.root, payload)
            done_path = self.root / "done" / (run + ".json")
            self._shared_target(done_path)
            self._check_n43_reference_publication(publication,task,True)
            try:
                write_immutable(done_path, done)
            except FileExistsError:
                if read_json(done_path) != done:
                    raise ValueError("Done identity changed")

    def _check_n43_reference_publication(self, publication, task, success):
        # The durable optional record also governs original recovery. Ordinary
        # records with no finite-first environment retain their exact route.
        guard=publication.get("n43_reference_guard")
        if "n43_reference_guard" in publication and not isinstance(guard,dict):
            raise ValueError("N43 publication guard must be its exact typed record")
        if guard is None and "DARSIA_CACHE_FIRST_QUALIFICATION" not in os.environ:
            return
        from darsia_cache.first_qualification import check_reference_publication_guard
        check_reference_publication_guard(guard,self.root,task,success=success)

    def _validate_alias_payload(self, directory, reference, index_reference, run, token):
        """Validate default-product receipt identity; never read large payloads here."""
        index_path = _own_asset(directory, index_reference)
        index = read_json(index_path)
        job = read_json(self.root / "job.json")
        kinds = job.get("publish", ["film", "lossless_export"])
        if (not isinstance(kinds, (list, tuple)) or len(kinds) != len(set(kinds))
                or any(not isinstance(kind, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", kind)
                       for kind in kinds)):
            raise ValueError("The immutable job publish kinds are malformed")
        selected = {row["product_id"]: row for row in index["products"] if row["kind"] in kinds}
        if not selected:
            if reference is not None:
                raise ValueError("A payload publication cannot claim absent matching products")
            return
        if reference is None:
            raise ValueError("The job's indexed publish kinds require their explicit payload publication")
        path = _own_asset(directory, reference)
        receipt = read_json(path)
        bound = receipt.get("products_index")
        size = bound.get("bytes", bound.get("size_bytes")) if isinstance(bound, dict) else None
        if (receipt.get("format") != "darsia-product-publication-v1"
                or not isinstance(bound, dict) or bound.get("sha256") != index_reference["sha256"]
                or size != index_reference["bytes"]
                or receipt.get("run") != run or receipt.get("original_attempt_id") != token
                or receipt.get("file_writers_closed") is not True
                or receipt.get("verification") != "Every explicitly copied product SHA-256 and byte count verified"
                or not isinstance(receipt.get("copies"), list)):
            raise ValueError("Payload publication is not bound to this original index/run/attempt")
        copies = receipt["copies"]
        if (len(copies) != len(selected)
                or {copy.get("product_id") for copy in copies} != set(selected)):
            raise ValueError("Payload publication differs from the exact job publish kinds")
        for copy in copies:
            row = selected[copy["product_id"]]
            if (copy.get("relative_path") != "products/" + copy["product_id"]
                    or any(copy.get(key) != row[key] for key in ("machine_id", "local_path", "sha256", "bytes"))):
                raise ValueError("Published payload receipt differs from its original indexed product")

    @staticmethod
    def _validate_alias_index(index_path, candidate, run, token):
        index = read_json(index_path)
        if (index.get("format") != "darsia-products-index-v1"
                or index.get("run") != run or index.get("attempt_id") != token):
            raise ValueError("Shared products index differs from the publication owner")
        from .products_publication import read_published_metadata
        raw = read_published_metadata(descriptor(index_path))
        if (len(raw) != candidate["bytes"]
                or hashlib.sha256(raw).hexdigest() != candidate["sha256"]):
            raise ValueError("Products index does not bind the original shared completion bytes")

    def _shared_target(self, path):
        root = self.root.resolve(strict=True)
        if not path.parent.resolve().is_relative_to(root):
            raise ValueError("Shared terminal target escaped the current job")
        for entry in (path, *path.parents):
            if entry == self.root.parent:
                break
            try:
                info = entry.lstat()
            except FileNotFoundError:
                continue
            if entry.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("Shared terminal target contains symlink/reparse entries")

    def recover_publications(self) -> None:
        """Finish pending publications while leaving matching completed/done documents untouched."""
        for entry in (self.root / "publishing").iterdir():
            if entry.name.startswith("."):
                continue  # A write_immutable temporary, never a publication.
            # A marker file names its in-place lease; a legacy directory is the moved lease.
            directory = publication_directory(self.root, entry.name)
            if directory is not None and (directory / "publication.json").exists():
                self._finish_publication(directory)

    def complete(self, claim: Claim, result: dict) -> dict:
        """Publish a genuine whole-run result only after scientific references are closed."""
        if result.get("complete") is not True or result.get("run") != claim.task["run"]:
            raise ValueError("A genuine complete per-run scientific result is required")
        if result.get("worker_references_closed") is not True:
            raise ValueError("Scientific references are still open")
        return self._publish(claim, result, True)

    def fail(self, claim: Claim, traceback_text: str, owner: dict) -> dict:
        """Publish the current attempt failure with its verbatim traceback and actual owner."""
        value = dict(complete=False, run=claim.task["run"], attempt_id=claim.token,
                     task=claim.task, actual_worker=owner, traceback=traceback_text,
                     failed_utc=utc())
        return self._publish(claim, value, False)

    def aggregate(self) -> dict:
        """Verify immutable completion bindings and report the expected whole-run results and failures."""
        results, failures = {}, {}
        for path in sorted((self.root / "done").glob("*.json")):
            done = read_json(path)
            if done.get("format") == "darsia-alias-done-v1":
                completed = _own_asset(self.root, done["completed"])
                _own_asset(self.root, done["products_index"])
                results[done["run"]] = read_json(completed)
            else:
                actual = descriptor(Path(done["completed"]["path"]))
                if actual != done["completed"]:
                    raise ValueError("Published result changed")
                results[done["run"]] = read_json(Path(actual["path"]))
        for path in sorted((self.root / "failed").glob("*.json")):
            failures[read_json(path)["run"]] = read_json(path)
        tasks = [read_json(p) for p in (self.root / "results").glob("*/task.json")]
        expected = {t["run"] for t in tasks}
        job = self.root / "job.json"
        if job.exists() and "expected_runs" in read_json(job):
            expected = set(read_json(job)["expected_runs"])
        return dict(complete=bool(expected) and set(results) == expected and not failures,
                    task_count=len(expected), completed_count=len(results),
                    failed_count=len(failures), results=results, failures=failures)
