"""Explicit SSH transport for the unchanged existing whole-run worker CLI.

Only Windows OpenSSH with an explicitly declared PowerShell default shell is
supported. This module does not discover a shell, install software, transfer
inputs, generate a start file, or treat its SSH client as a numerical worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
import re
import uuid

from darsia.fluidflower.pipeline.job_queue import name, require_unc, write_immutable


@dataclass(frozen=True)
class RemoteWorkerTarget:
    machine: str
    ssh_alias: str
    python: str
    checkout: str
    ssh_executable: str
    default_shell: str


@dataclass(frozen=True)
class RemoteWorkerLaunch:
    program: str
    arguments: tuple[str, ...]
    worker_argv: tuple[str, ...]
    remote_command: str
    job_root: str
    target: RemoteWorkerTarget
    backend: str
    workers: int


def _text(value, label):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(label + " must be an explicit nonempty string")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(label + " must not contain control characters")
    return value


def _local_windows_path(value, label, *, filename=None):
    value = _text(value, label)
    path = PureWindowsPath(value)
    if (not path.is_absolute() or not re.fullmatch(r"[A-Za-z]:", path.drive)
            or any(part in {".", ".."} for part in value.replace("/", "\\").split("\\"))
            or any(char in value[2:] for char in '<>:"|?*')):
        raise ValueError(label + " must be an unambiguous absolute Windows drive path")
    if filename is not None and path.name.casefold() != filename:
        raise ValueError(label + " must name " + filename)
    return value


def _powershell_literal(value):
    """One native-program argument as a PowerShell single-quoted literal."""
    return "'" + value.replace("'", "''") + "'"


def build_remote_worker_launch(job_root, *, target, backend, workers, pipeline=None):
    """Build one fixed worker invocation, without touching a file or process.

    ``default_shell`` is an operator/provider fact about this SSH server, not a
    guess based on its operating system. Unknown, cmd.exe, POSIX, forced-command
    wrappers and nonstandard command interpreters are unsupported here.
    """
    if not isinstance(target, RemoteWorkerTarget):
        raise TypeError("Explicit RemoteWorkerTarget facts required")
    if target.default_shell != "windows-powershell":
        raise ValueError("Confirm the SSH server's PowerShell default shell before remote start")
    machine = name(target.machine)
    alias = _text(target.ssh_alias, "SSH host alias")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", alias) is None:
        raise ValueError("Use one configured SSH host alias, without options or shell text")
    python = _local_windows_path(target.python, "Existing remote interpreter", filename="python.exe")
    _local_windows_path(target.checkout, "Existing remote checkout")
    ssh = _local_windows_path(target.ssh_executable, "Existing local SSH executable", filename="ssh.exe")
    job = _text(str(job_root), "Shared job folder")
    require_unc(job)
    if any(part in {".", ".."} for part in job.replace("/", "\\").split("\\")):
        raise ValueError("Shared job folder must not contain relative path components")
    if backend not in {"cpu", "cuda"} or type(workers) is not int or workers < 1:
        raise ValueError("Explicit CPU/CUDA backend and positive worker count required")
    comparison = pipeline == "preset-wasserstein"
    if comparison and (backend != "cpu" or workers != 1):
        raise ValueError("Original Comparison starts require CPU and exactly one worker per service")
    worker = (python, "-B", "-m", "darsia.fluidflower.worker", "--job", job,
              "--backend", backend, "--workers", str(workers), "--machine", machine,
              "--watchdog")
    if comparison:
        worker += ("--once",)
    # N109: the fixed stdlib supervisor of the same checkout starts this worker
    # once more when it stops before the service started.
    service_log = job.rstrip("\\/") + "\\logs\\" + name(machine + "-" + backend) + ".log"
    supervised = (python, "-B", "-m", "darsia_service.prestart", "--service-log", service_log,
                  "--", *worker)
    # This is a normal remote CLI invocation. No generated script, encoded
    # program, change-directory command or per-run launcher exists.
    command = "& " + " ".join(_powershell_literal(value) for value in supervised)
    arguments = ("-n", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                 "-o", "ConnectTimeout=10", "--", alias, command)
    return RemoteWorkerLaunch(ssh, arguments, worker, command, job, target, backend, workers)


def launch_remote_worker(job_root, *, target, backend, workers,
                         launch_directory=None, process_factory=None,
                         receipt_writer=write_immutable):
    """Explicit UI action only: detach the SSH transport and retain its console.

    Existing remote Python must already resolve the pinned repository. The
    unchanged worker's verify_startup checks its own clean Git HEAD and lockfile.
    Nothing here claims the remote service passed startup or completed a frame.
    A transport failure is read from this console; actual workers and frame
    progress are read only from the existing shared job's metadata.
    """
    job_path = Path(job_root) / "job.json"
    from .pipeline.job_queue import read_json
    selected_job = read_json(job_path) if job_path.is_file() else {}
    launch = build_remote_worker_launch(job_root, target=target, backend=backend, workers=workers,
        pipeline=selected_job.get("pipeline"))
    if not Path(launch.program).is_file():
        raise FileNotFoundError("Configured local SSH executable does not exist: " + launch.program)
    if process_factory is None:
        from PySide6.QtCore import QProcess
        process_factory = QProcess
    directory = (Path(launch_directory) if launch_directory is not None else
                 Path(launch.job_root) / "ui" / "launches" / uuid.uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    console = directory / "console.log"
    process = process_factory()
    process.setProgram(launch.program)
    process.setArguments(list(launch.arguments))
    process.setProcessChannelMode(process.ProcessChannelMode.MergedChannels)
    process.setStandardOutputFile(str(console))
    started, pid = process.startDetached()
    record = {
        "format": "darsia-job-ssh-launch-v1", "job_root": launch.job_root,
        "machine": target.machine, "ssh_alias": target.ssh_alias,
        "declared_checkout": target.checkout, "declared_default_shell": target.default_shell,
        "program": launch.program, "arguments": list(launch.arguments),
        "worker_argv": list(launch.worker_argv), "console": str(console),
        "requested_utc": datetime.now(timezone.utc).isoformat(),
        "started": bool(started), "transport_started": bool(started),
        "transport_pid": int(pid) if started else None, "detached": True,
        "worker_startup_verified": False, "numerical_progress_claimed": False,
        "error": "" if started else process.errorString(),
    }
    # New alias-job launch records bind only the immutable job identity. The
    # original command/checkout remain passive actual-launch provenance.
    if job_path.is_file():
        if read_json(job_path).get("format") == "darsia-alias-job-v1":
            from .pipeline.alias_job import job_metadata_reference
            record.update(format="darsia-job-ssh-launch-v2",
                          job=job_metadata_reference(Path(launch.job_root), job_path))
    receipt_writer(directory / "launch.json", record)
    if not started:
        raise RuntimeError(record["error"])
    return record
