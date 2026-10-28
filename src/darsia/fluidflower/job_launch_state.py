"""UI launch reservations, not process liveness, science or queue control.

All records are new immutable files in this explicit job's UI launch namespace.
Only matching original terminal service logs or an explicit operator statement
release a reservation. SSH exit, PID, elapsed time and completed frames do not.
"""

from datetime import datetime, timezone
import json
import ntpath
from pathlib import Path
import traceback
import uuid

from .pipeline.job_queue import descriptor, name, read_json, write_immutable


UNKNOWN = "Launch requested; worker state unknown"


def _path(value):
    return ntpath.normcase(ntpath.normpath(str(value)))


def _epoch(value):
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return stamp.timestamp() if stamp.tzinfo is not None else None
    except (ValueError, TypeError, AttributeError, OverflowError):
        return None


def launch_key(root, machine, backend):
    if backend not in {"cpu", "cuda"}:
        raise ValueError("Explicit backend required")
    return _path(root), name(machine), backend


def begin_launch_request(root, *, machine, backend, workers, kind, now=None):
    """Explicit Start only; reserve before invoking an existing launcher."""
    if _alias_job(root):
        from .portable_job_launch import begin
        return begin(root, machine=machine, backend=backend, workers=workers, kind=kind, now=now)
    launch_key(root, machine, backend)
    if kind not in {"local", "ssh"} or type(workers) is not int or workers < 1:
        raise ValueError("Explicit launch kind/count required")
    root = Path(root)
    identifier = uuid.uuid4().hex
    folder = root / "ui" / "launches"
    launch_directory = folder / identifier
    requested = (now or datetime.now(timezone.utc)).isoformat()
    if _epoch(requested) is None:
        raise ValueError("Aware launch request time required")
    log = root / "logs" / (name(machine + "-" + backend) + ".log")
    offset = log.stat().st_size if log.exists() else 0
    value = dict(
        format="darsia-job-ui-launch-request-v1",
        job=descriptor(root / "job.json"),
        job_root=str(root),
        machine=name(machine),
        backend=backend,
        workers=workers,
        kind=kind,
        requested_utc=requested,
        launch_directory=str(launch_directory),
        console=str(launch_directory / "console.log"),
        service_log=str(log),
        service_log_offset=offset,
        numerical_progress_claimed=False,
        worker_liveness_verified=False,
    )
    path = folder / (identifier + ".request.json")
    write_immutable(path, value)
    return dict(
        value,
        source=descriptor(path),
        state=UNKNOWN,
        released=False,
        error_traceback="",
    )


def record_launch_error(request, text):
    """Keep literal launch failure visible even when no transport was created."""
    if request.get("format") == "darsia-job-ui-launch-request-v2":
        from .portable_job_launch import error
        return error(request, text)
    path = Path(request["source"]["path"]).with_name(
        Path(request["source"]["path"]).name.replace(".request.json", ".error.json")
    )
    value = dict(
        format="darsia-job-ui-launch-error-v1",
        request=request["source"],
        traceback=text,
        console=request["console"],
        worker_liveness_verified=False,
        numerical_progress_claimed=False,
    )
    write_immutable(path, value)


def _read_bound(path, expected=None, reader=None):
    bound = descriptor(path)
    if expected is not None and bound != expected:
        raise ValueError("UI launch receipt SHA/path/size changed")
    if reader is not None:
        doc = reader(path, bound)
    else:
        doc = read_json(path)
        if descriptor(path) != bound:
            raise ValueError("UI launch receipt changed while read")
    return doc, bound


def _terminal_report(request):
    """Read only new bytes in this job's matching original service log."""
    log = Path(request["service_log"])
    offset = request["service_log_offset"]
    if not log.exists():
        return None
    if type(offset) is not int or offset < 0:
        raise ValueError("Invalid recorded service log position")
    with log.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        if size < offset:
            return None  # Rotated/truncated data is not a stop proof.
        stream.seek(max(offset, size - 128 * 1024))
        raw = stream.read()
    latest = None
    for line in raw.decode("utf-8", errors="replace").splitlines():
        timestamp, space, encoded = line.partition(" ")
        stamp = _epoch(timestamp)
        if not space or stamp is None or stamp < _epoch(request["requested_utc"]):
            continue
        try:
            event = json.loads(encoded)
        except (ValueError, TypeError):
            continue
        if event.get("kind") in {"started", "stopped", "startup_failure"}:
            latest = dict(timestamp=timestamp, event=event)
    return (
        latest
        if latest and latest["event"]["kind"] in {"stopped", "startup_failure"}
        else None
    )


def read_launch_requests(root, *, reader=None):
    if _alias_job(root):
        from .portable_job_launch import read
        return read(root, reader=reader)
    root = Path(root)
    job = descriptor(root / "job.json")
    result = []
    folder = root / "ui" / "launches"
    for path in sorted(folder.glob("*.request.json")):
        request, bound = _read_bound(path, reader=reader)
        identifier = path.name.removesuffix(".request.json")
        expected_directory = folder / identifier
        if (
            request.get("format") != "darsia-job-ui-launch-request-v1"
            or request.get("job") != job
            or _path(request.get("job_root", "")) != _path(root)
            or _path(request.get("launch_directory", "")) != _path(expected_directory)
            or _epoch(request.get("requested_utc")) is None
        ):
            raise ValueError(
                "UI launch request belongs to another job/directory or has no exact time"
            )
        launch_key(root, request["machine"], request["backend"])
        expected_log = (
            root
            / "logs"
            / (name(request["machine"] + "-" + request["backend"]) + ".log")
        )
        if _path(request.get("service_log", "")) != _path(expected_log) or _path(
            request.get("console", "")
        ) != _path(expected_directory / "console.log"):
            raise ValueError("UI launch log/console path differs")
        state = dict(
            request, source=bound, state=UNKNOWN, released=False, error_traceback=""
        )
        error_path = path.with_name(identifier + ".error.json")
        if error_path.exists():
            error, _ = _read_bound(error_path, reader=reader)
            if (
                error.get("format") != "darsia-job-ui-launch-error-v1"
                or error.get("request") != bound
                or not isinstance(error.get("traceback"), str)
            ):
                raise ValueError("UI launch error belongs to another request")
            state["error_traceback"] = error["traceback"]
        launch_path = expected_directory / "launch.json"
        if launch_path.exists():
            receipt, receipt_bound = _read_bound(launch_path, reader=reader)
            if receipt.get("format") not in {
                "darsia-job-local-launch-v1",
                "darsia-job-ssh-launch-v1",
            } or _path(receipt.get("job_root", "")) != _path(root):
                raise ValueError("Launcher receipt belongs to another job")
            state["launcher_receipt"] = dict(receipt, source=receipt_bound)
        terminal = _terminal_report(request)
        if terminal:
            state.update(
                released=True,
                state="Original service reported " + terminal["event"]["kind"],
                terminal_report=terminal,
            )
        for declaration_path in sorted(folder.glob(identifier + ".declaration.*.json")):
            declaration, declaration_bound = _read_bound(
                declaration_path, reader=reader
            )
            if (
                declaration.get("format") != "darsia-job-ui-operator-declaration-v1"
                or declaration.get("request") != bound
                or declaration.get("job") != job
                or declaration.get("operator_declares_previous_service_stopped")
                is not True
                or declaration.get("scientific_evidence") is not False
                or declaration.get("worker_liveness_verified") is not False
                or declaration.get("numerical_progress_claimed") is not False
                or _epoch(declaration.get("declared_utc")) is None
            ):
                raise ValueError(
                    "Operator declaration has no exact request binding or falsely claims scientific evidence"
                )
            state.update(
                released=True,
                state="Operator declared stopped; not independently verified",
                operator_declaration=declaration_bound,
            )
        result.append(state)
    return result


def read_registered_launch_requests(registrations):
    """Only explicitly registered roots; a read failure never releases Start.

    This is a UI reservation view, not a NAS history search or service-liveness
    check. Preserve each original job binding for declarations in that job.
    """
    result = []
    errors = []
    seen = {}
    for registration in registrations:
        try:
            root = registration["root"]
            key = _path(root)
            expected = registration.get("job")
            if key in seen:
                if seen[key] != expected:
                    raise ValueError("Conflicting registered job metadata bindings")
                continue
            seen[key] = expected
            if expected is not None and descriptor(Path(root) / "job.json") != expected:
                raise ValueError("Registered job metadata SHA/path/size changed")
            result.extend(
                dict(row, registered_job_name=str(registration.get("name") or root))
                for row in read_launch_requests(root)
            )
        except Exception:
            errors.append(
                dict(
                    root=str(registration.get("root", "Unknown registered job")),
                    error_traceback=traceback.format_exc(),
                )
            )
    return result, errors


def declare_requests_stopped(root, requests):
    """Explicit operator action only. Never a worker-retirement/science proof."""
    if _alias_job(root):
        from .portable_job_launch import declare
        return declare(root, requests)
    current = {row["source"]["path"]: row for row in read_launch_requests(root)}
    written = []
    for chosen in requests:
        row = current.get(chosen["source"]["path"])
        if row is None or row["source"] != chosen["source"]:
            raise ValueError("Launch request changed before operator declaration")
        if row["released"]:
            continue
        path = Path(row["source"]["path"])
        target = path.with_name(
            path.name.replace(
                ".request.json", ".declaration." + uuid.uuid4().hex + ".json"
            )
        )
        value = dict(
            format="darsia-job-ui-operator-declaration-v1",
            request=row["source"],
            job=row["job"],
            declared_utc=datetime.now(timezone.utc).isoformat(),
            operator_declares_previous_service_stopped=True,
            scientific_evidence=False,
            worker_liveness_verified=False,
            numerical_progress_claimed=False,
        )
        write_immutable(target, value)
        written.append(descriptor(target))
    return written


def _alias_job(root):
    return read_json(Path(root) / "job.json").get("format") == "darsia-alias-job-v1"
