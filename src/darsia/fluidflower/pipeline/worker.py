"""Persistent Git-pinned whole-run workers reading a shared UNC job directly."""

from __future__ import annotations

import argparse
from copy import deepcopy
import ctypes
import hashlib
import math
import multiprocessing as mp
import os
import queue
import socket
import subprocess
import sys
import tempfile
import time
import threading
import traceback
from pathlib import Path, PureWindowsPath

from .job_queue import (
    JobQueue,
    LeaseLost,
    lease_retired,
    name,
    read_json,
    read_decision,
    require_unc,
    retry_transient,
    transient_os_error,
    utc,
    write_immutable,
)

# N111: queue steps wait out a transient SMB error this long before each retry,
# after the file-level retries inside the queue itself.
QUEUE_RETRY_DELAYS = (5.0, 15.0, 30.0)


def _setup_operation_active(root):
    """Retained claim files are history once their original decision exists."""
    root = Path(root)
    for path in (root / "claimed").glob("*/*.json"):
        if ".invalid." in path.name or ".duplicate." in path.name:
            continue
        task = read_json(path)
        if read_decision(root, task["run"], task["attempt_id"]) is None:
            return True
    return False


def verify_startup(job_root: Path, *, stores_path=None, machine_id=None,
                   startup_context=None, expected_stores=None) -> tuple[dict, str]:
    """Check the running repository and its actual lockfile exactly once."""
    from ..job_filesystem import inspect_shared_folder

    transport = inspect_shared_folder(job_root)
    if not transport.allowed:
        raise ValueError("Shared queue transport refused: " + transport.reason + ": " + transport.evidence)
    job = read_json(job_root / "job.json")
    alias = job.get("format") == "darsia-alias-job-v1"
    from ..storage_profile import expected_format_version, job_storage_profile
    if not alias:
        require_unc(str(job_root))
    elif (job.get("format_version")!=expected_format_version(job) or job.get("legacy_paths") is not False
            or job.get("execution_ready") is not True
            or job.get("runtime_support")!="darsia-role-io-runtime-v1"):
        raise RuntimeError("Alias job is not explicitly ready for J3 runtime")
    repository = Path(__file__).resolve().parents[4]
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    if dirty or head != job["commit"]:
        raise RuntimeError("Startup requires a clean Git worktree at job.json commit")
    lock = job["environment_lock"]
    relative = Path(lock["path"])
    if (
        relative.is_absolute()
        or PureWindowsPath(lock["path"]).drive
        or ".." in relative.parts
    ):
        raise ValueError("environment_lock.path must be repository-relative")
    actual = (repository / relative).resolve(strict=True)
    if not actual.is_relative_to(repository):
        raise ValueError("Environment lock escapes the repository")
    git_relative = relative.as_posix()
    tracked = subprocess.check_output(
        ["git", "-C", str(repository), "ls-tree", "--name-only",
         head, "--", git_relative], text=True).splitlines()
    if tracked != [git_relative]:
        raise RuntimeError("Environment lock is not tracked at the admitted Git commit")
    original = subprocess.check_output(
        ["git", "-C", str(repository), "show", head + ":" + git_relative])
    if hashlib.sha256(original).hexdigest() != lock["sha256"]:
        raise RuntimeError("Actual repository environment lock SHA differs")
    before = actual.stat()
    raw = actual.read_bytes()
    after = actual.stat()
    stamp = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    if (stamp(before) != stamp(after)
            or raw.replace(b"\r\n", b"\n") != original.replace(b"\r\n", b"\n")):
        raise RuntimeError("Actual repository environment lock differs from its Git content")
    if startup_context is not None:
        startup_context["environment_lock_identity"] = dict(
            commit=head, repo_relative_path=git_relative,
            identity_algorithm="sha256-git-blob-content",
            sha256=hashlib.sha256(original).hexdigest(), size_bytes=len(original),
            working_tree_sha256=hashlib.sha256(raw).hexdigest(),
            working_tree_size_bytes=len(raw), EOL_only_representation=(raw != original))
    if (job.get("precision") not in {"float32", "float64"}
            and job.get("pipeline") != "preset-wasserstein"):
        raise ValueError("One explicit job precision is required")
    job_storage_profile(job)  # Explicit 'full'/'lean'; absent is a legacy full job.
    import re

    if job.get("pipeline") == "native-phase-statistics" and job.get("execution_mode") == "current_build":
        if not job.get("inputs") or not job.get("selection") or not job.get("results_root"):
            raise ValueError("Current Build P1 requires actual source and local result bindings")
    elif job.get("pipeline") == "native-phase-statistics":
        from .phase1_protocol import validate_phase1_job
        validate_phase1_job(job)
    elif job.get("pipeline") == "series-setup":
        from .series_setup_protocol import validate_setup_job
        validate_setup_job(job)
    elif job.get("pipeline") == "preset-wasserstein":
        from ..preset_wasserstein_protocol import validate_preset_wasserstein_job
        validate_preset_wasserstein_job(job)
    else:
        if not re.fullmatch(r"[0-9a-f]{64}", job.get("calibration_sha256", "")):
            raise ValueError("Explicit job calibration SHA required")
    if (
        not isinstance(job.get("expected_runs"), list)
        or not job["expected_runs"]
        or len({name(run) for run in job["expected_runs"]}) != len(job["expected_runs"])
    ):
        raise ValueError("Explicit distinct expected_runs required")
    if alias:
        from ..store_map import read_store_map
        stores=read_store_map(stores_path,forbidden_roots=(repository,))
        if name(stores.machine_id)!=name(machine_id or socket.gethostname()):
            raise ValueError("Store-map machine ID differs from --machine")
        stores.roots(job["series_id"])
        if expected_stores is not None and stores.source!=expected_stores.source:
            raise ValueError("Startup-admitted store map changed; running claims retain their original binding")
        if startup_context is not None:
            startup_context["stores"]=stores
    return job, head


class _JobControlBoundary(Exception):
    """A cooperative operational exit after a committed frame; never a task FAIL."""

    def __init__(self, state, command=None, traceback_text=None):
        self.state, self.command, self.traceback_text = state, command, traceback_text
        super().__init__(state)


def _job_control_boundary(root):
    """Use existing frame hooks; do not change the executor or scientific inputs."""
    from ..job_control import read_control

    try:
        command = read_control(root)
    except Exception:
        raise _JobControlBoundary("control_failure", traceback_text=traceback.format_exc())
    if command["action"] in {"pause", "stop"}:
        state = "paused" if command["action"] == "pause" else "control_stopped"
        raise _JobControlBoundary(state, command=command)


class ServiceLock:
    """One parent/log writer for this local machine, backend and UNC job."""

    def __init__(self, identity: str):
        """Bind a machine/backend identity to the sole shared-log ownership guard."""
        self.identity = hashlib.sha256(identity.encode()).hexdigest()
        self.handle = None
        self.file = None

    def __enter__(self):
        """Acquire the process ownership guard or reject a second active log writer."""
        if os.name == "nt":
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateMutexW.argtypes = [
                ctypes.c_void_p,
                ctypes.c_bool,
                ctypes.c_wchar_p,
            ]
            kernel.CreateMutexW.restype = ctypes.c_void_p
            ctypes.set_last_error(0)
            self.handle = kernel.CreateMutexW(
                None, True, "Global\\DarSIAWorker-" + self.identity
            )
            if not self.handle:
                raise ctypes.WinError(ctypes.get_last_error())
            if ctypes.get_last_error() == 183:
                kernel.CloseHandle(ctypes.c_void_p(self.handle))
                self.handle = None
                raise RuntimeError("A worker service already owns this machine log")
        else:
            import fcntl

            self.file = open(
                Path(tempfile.gettempdir()) / ("darsia-worker-" + self.identity), "a"
            )
            try:
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self.file.close()
                raise RuntimeError(
                    "A worker service already owns this machine log"
                ) from exc
        return self

    def __exit__(self, *args):
        """Release this process ownership guard without removing preserved log files."""
        if self.handle is not None:
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.ReleaseMutex(ctypes.c_void_p(self.handle))
            kernel.CloseHandle(ctypes.c_void_p(self.handle))
        if self.file is not None:
            self.file.close()


def _identity(machine: str) -> dict:
    import psutil

    process = psutil.Process(os.getpid())
    return dict(
        machine=machine, pid=process.pid, process_create_time=str(process.create_time())
    )


def _run_tasks(
    root: str,
    job: dict,
    commit: str,
    machine: str,
    backend: str,
    messages,
    stop,
    once: bool,
    exit_when_idle: bool,
    watchdog: bool = False,
    worker_slot=None,
    worker_generation=None,
    stores=None,
):
    from darsia._opencv_runtime import scientific_thread, require_ipp_off
    ipp_start = scientific_thread("pipeline-worker:start")
    def queue_error(event):
        messages.put(event)

    alias=job.get("format")=="darsia-alias-job-v1"
    io_runtime=None
    if alias:
        from .worker_io import WorkerRunIO
        io_runtime=WorkerRunIO(root,job,stores,machine,backend,commit)
        shared=JobQueue(Path(root),on_error=queue_error,machine_id=stores.machine_id)
    else:
        shared = JobQueue(Path(root), on_error=queue_error, machine_id=machine)
    excluded_runs=set()
    owner = _identity(machine)
    from darsia_cache.adapters import bind_actual_worker_owner
    bind_actual_worker_owner(owner,slot=worker_slot,generation=worker_generation,role="slot")
    next_control_read = 0.0
    def queue_checkpoint():
        nonlocal next_control_read
        if stop.is_set():
            raise _JobControlBoundary("control_stopped")
        now = time.monotonic()
        if now >= next_control_read:
            _job_control_boundary(root)
            next_control_read = now + 0.25
    if io_runtime is not None:
        io_runtime.checkpoint = queue_checkpoint
    while not stop.is_set():
        try:
            _job_control_boundary(root)
        except _JobControlBoundary as requested:
            messages.put(dict(kind=requested.state, command=requested.command,
                              traceback=requested.traceback_text, actual_worker=owner))
            return
        held = []

        def next_claim():
            queue_checkpoint()
            shared.recover_publications()
            for record in shared.reclaim_stale():
                messages.put(dict(kind="reclaimed", **record))
            if io_runtime is None:
                found = shared.claim(machine, backend, owner)
                held.extend([found] if found is not None else [])
                return found
            with io_runtime.capacity_claim(owner,queue_error) as capacity:
                capacity.checkpoint = queue_checkpoint
                found = shared.claim(machine,backend,owner,
                    availability=capacity.availability,
                    output_factory=io_runtime.output_factory,
                    publication_factory=io_runtime.publications,exclude_runs=excluded_runs,
                    checkpoint=queue_checkpoint,
                    **({"claim_controls": capacity.claim_controls}
                       if job.get("pipeline") == "series-setup" else {}))
                held.extend([found] if found is not None else [])
                try:
                    capacity.bind(found)
                except Exception:
                    if found is not None:
                        # An actual atomic claim exists even if local disk
                        # bookkeeping failed; the original parent must bind
                        # it for owned retirement before any replacement.
                        messages.put(dict(kind="claimed",run=found.task["run"],
                            attempt_id=found.token,actual_worker=owner))
                    raise
                return found

        def queue_retry(attempt, delay, error):
            messages.put(dict(kind="queue_transient_retry", attempt=attempt,
                              delay_seconds=delay, error=repr(error), actual_worker=owner))

        try:
            # N111: a transient SMB error waits and retries, never once a claim is held.
            claim = retry_transient(next_claim, delays=QUEUE_RETRY_DELAYS,
                                    wait=lambda seconds: stop.wait(seconds),
                                    on_retry=queue_retry, retry_if=lambda error: not held)
        except _JobControlBoundary as requested:
            messages.put(dict(kind=requested.state, command=requested.command,
                              traceback=requested.traceback_text, actual_worker=owner))
            return
        except Exception as error:
            if stop.is_set() and transient_os_error(error) and not held:
                return  # A stop request ended the wait for the share.
            text = traceback.format_exc()
            messages.put(dict(kind="queue_failure", traceback=text))
            raise SystemExit(1)
        if claim is None:
            if (exit_when_idle and not once and job.get("pipeline") == "series-setup"
                    and _setup_operation_active(root)):
                # Apply publishes Build children before its original claim
                # retires. Keep the other selected slots available meanwhile.
                stop.wait(1)
                continue
            if once or exit_when_idle:
                return
            stop.wait(1)
            continue
        messages.put(
            dict(
                kind="claimed",
                run=claim.task["run"],
                attempt_id=claim.token,
                actual_worker=owner,
            )
        )
        from contextlib import ExitStack
        writer_scope = ExitStack()
        setup = claim.task.get("pipeline") == "series-setup"
        retain_capacity = setup
        try:
            if io_runtime is not None and setup:
                writer_scope.enter_context(io_runtime.output_scope(claim))
            task = claim.task
            if (task.get("pipeline") == "native-phase-statistics"
                    or job.get("pipeline") == "native-phase-statistics"):
                from .phase1_protocol import validate_phase1_claim
                validate_phase1_claim(task, job)
                if task["commit"] != commit:
                    raise ValueError("Claimed phase-one commit differs from the actual worker")
            elif setup:
                from .series_setup_protocol import validate_setup_claim
                validate_setup_claim(task, job)
                if task["commit"] != commit:
                    raise ValueError("Claimed Setup commit differs from the actual worker")
            elif (task.get("pipeline") == "preset-wasserstein"
                    or job.get("pipeline") == "preset-wasserstein"):
                from ..preset_wasserstein_protocol import validate_preset_wasserstein_claim
                validate_preset_wasserstein_claim(task, job)
                if task["commit"] != commit:
                    raise ValueError("Claimed comparison commit differs from the actual worker")
            else:
                if (
                    task["commit"] != commit
                    or task["calibration_sha256"] != job["calibration_sha256"]
                ):
                    raise ValueError("Claimed commit/calibration differs from job.json")
            precision = task.get("precision", job["precision"])
            if precision != job["precision"]:
                raise ValueError("A job cannot mix working precisions")
            from ..storage_profile import require_task_matches_job
            require_task_matches_job(task, job)

            io_binding=None
            if io_runtime is not None:
                from .store_runtime import InputHashMismatch,RunUnavailable
                try:
                    io_binding=io_runtime.prepare(claim)
                    from .checkpoint import resume_source_for_claim
                    if task.get("pipeline") == "native-phase-statistics":
                        from .phase1_protocol import phase1_resume_source
                        prepared_resume=phase1_resume_source(shared,claim)
                    elif setup or task.get("pipeline") == "preset-wasserstein":
                        prepared_resume=None
                    else:
                        prepared_resume=resume_source_for_claim(shared,claim)
                except (InputHashMismatch,RunUnavailable):
                    text=traceback.format_exc()
                    record=shared.reject_before_science(claim,text,owner)
                    excluded_runs.add(name(record["run"]))
                    if not setup:
                        io_runtime.capacity_close(claim)
                    messages.put(dict(kind="input_rejected",run=task["run"],
                        attempt_id=claim.token,traceback=text,actual_worker=owner,
                        capacity_retained=retain_capacity))
                    if setup:
                        return
                    io_runtime.release(claim)
                    if once:
                        raise SystemExit(1)
                    continue

            def progress(event):
                if not isinstance(event, dict):
                    raise TypeError(
                        "Scientific progress must be an actual event dictionary"
                    )
                shared.progress(claim, event)
                if io_runtime is not None:
                    io_runtime.capacity_progress(claim,owner,event)
                messages.put(
                    dict(
                        kind="progress",
                        run=task["run"],
                        attempt_id=claim.token,
                        event=event,
                    )
                )
                # frame_completed is emitted only after the original atomic
                # publication. frame_started also permits a zero-frame pause.
                if event.get("state") == "phase1_frame_completed":
                    _job_control_boundary(root)
                elif event.get("state") == "setup_progress":
                    _job_control_boundary(root)
                elif event.get("state") == "preset_wasserstein_progress":
                    _job_control_boundary(root)
                elif event.get("state") in {"frame_started", "frame_completed"}:
                    _job_control_boundary(root)

            def checkpoint():
                """Check ownership and user Stop without writing progress files."""
                shared.assert_owner(claim)
                _job_control_boundary(root)

            progress.checkpoint = checkpoint

            from .checkpoint import resume_source_for_claim

            science_task = dict(task)
            if "environment_lock" in job:
                science_task["environment_lock"] = job["environment_lock"]
            resume = (None if setup or task.get("pipeline") == "preset-wasserstein" else
                      resume_source_for_claim(shared,claim) if io_binding is None else prepared_resume)
            if resume is not None:
                science_task["resume_from"] = resume
            pipeline = task.get("pipeline", "n1")
            if pipeline == "n1":
                from darsia.fluidflower.n1_pipeline import run_n1

                execute = run_n1
            elif pipeline == "native-phase-n1":
                from .runner import run_series

                execute = run_series
            elif pipeline == "lattice-phase-n1":
                from .lattice_runner import run_lattice_series

                execute = run_lattice_series
            elif pipeline == "native-phase-statistics":
                from .phase_statistics import run_phase1

                execute = run_phase1
            elif pipeline == "series-setup":
                from .series_setup_operations import execute_setup

                execute = execute_setup
            elif pipeline == "preset-wasserstein":
                from ..preset_wasserstein_runner import run_preset_wasserstein

                execute = run_preset_wasserstein
            elif pipeline == "n1-calibration-refit":
                from .refit_runner import run_refit

                execute = run_refit
            else:
                raise ValueError("Unknown explicit task pipeline: " + str(pipeline))
            # Setup image preparation is one task on one worker slot. Inside it, the
            # runs spread over this machine's cores within its memory bound (0 =
            # automatic); each run's own images stay in order.
            extra = ({"stage_workers": 0} if setup and task.get("operation") == "image-preparation" else {})
            if io_binding is None:
                result = execute(science_task, claim.output, backend, precision, progress, **extra)
            else:
                with io_binding.activate():
                    result = execute(science_task, claim.output, backend, precision, progress, **extra)
            if result.get("precision") != precision or result.get("backend") != backend:
                raise ValueError("Actual scientific backend/precision differs")
            if pipeline == "native-phase-statistics":
                from .phase1_protocol import worker_result
                result = worker_result(result, task=task, job=job, commit=commit,
                                       owner=owner, token=claim.token)
            elif setup:
                result = dict(result, actual_worker=owner, commit=commit,
                              attempt_id=claim.token, task=task)
                from .series_setup_results import validate_setup_result
                validate_setup_result(result, task=task, job=job)
                if task["operation"] == "apply":
                    from .series_setup_jobs import enqueue_setup_builds
                    with io_binding.activate():
                        enqueue_setup_builds(root, claim, result)
            elif pipeline == "preset-wasserstein":
                result = dict(result, actual_worker=owner, commit=commit,
                              attempt_id=claim.token, task=task)
                from ..preset_wasserstein_results import validate_preset_wasserstein_result
                validate_preset_wasserstein_result(result, task=task, job=job)
            else:
                result = dict(
                    result,
                    actual_worker=owner,
                    commit=commit,
                    calibration_sha256=task["calibration_sha256"],
                    attempt_id=claim.token,
                    task=task,
                )
            result["opencv_ipp_worker"] = dict(start=ipp_start, completed=require_ipp_off("pipeline-worker:complete"))
            shared.complete(claim, result)
            retain_capacity = False
            if io_runtime is not None:
                io_runtime.finalize_common(claim)
                if not setup:
                    io_runtime.capacity_close(claim)
            if not setup:
                messages.put(
                    dict(kind="completed", run=task["run"], attempt_id=claim.token)
                )
        except _JobControlBoundary as requested:
            # Keep this claim attached to the parent slot so the existing exact
            # child-retirement code binds its original manifest/frame receipts.
            # Never publish failed/completed for an interrupted whole run.
            messages.put(dict(kind=requested.state, run=claim.task["run"],
                              attempt_id=claim.token, command=requested.command,
                              traceback=requested.traceback_text))
            return
        except LeaseLost:
            text = traceback.format_exc()
            if io_runtime is not None and not setup:
                io_runtime.capacity_close(claim)
            messages.put(
                dict(
                    kind="lease_lost",
                    capacity_retained=retain_capacity,
                    run=claim.task["run"],
                    attempt_id=claim.token,
                    traceback=text,
                )
            )
        except Exception as original_error:
            text = traceback.format_exc()
            publication_error = None
            try:
                shared.fail(claim, text, owner)
            except LeaseLost:
                pass  # Its own attempt failure stays; only the current lease publishes.
            except Exception as error:
                publication_error = error
                original_error.add_note("Failure publication/diagnostic also failed: " + repr(error))
            if io_runtime is not None and not setup:
                io_runtime.capacity_close(claim)
            messages.put(
                dict(
                    kind="failed",
                    capacity_retained=retain_capacity,
                    run=claim.task["run"],
                    attempt_id=claim.token,
                    traceback=text,
                )
            )
            if publication_error is not None:
                raise original_error from publication_error
            if once:
                raise SystemExit(1)
        finally:
            writer_scope.close()
            if io_runtime is not None:
                if setup:
                    io_runtime.release(claim, retain_capacity=retain_capacity)
                else:
                    io_runtime.release(claim)
        if setup and not retain_capacity:
            messages.put(dict(kind="completed", run=task["run"], attempt_id=claim.token))
        if setup and retain_capacity:
            return  # Original parent proves owned process retirement before release.
        if once:
            return


class QueueStream:
    """Forward child Python stdout/stderr to the sole parent writer."""

    encoding = "utf-8"

    def __init__(self, messages, stream):
        """Bind a child console stream to the parent process message queue."""
        self.messages, self.stream = messages, stream

    def write(self, text):
        """Forward console text to the parent log writer and return the number of characters."""
        if text:
            self.messages.put(dict(kind="console", stream=self.stream, text=text))
        return len(text)

    def flush(self):
        """Perform no write because this stream retains no buffered child text."""
        pass

    def isatty(self):
        """Report a queued console stream rather than an interactive terminal."""
        return False


def _work(*args):
    slot = args[10] if len(args) > 10 else None
    generation = args[11] if len(args) > 11 else None
    forwarded = _SlotMessages(args[5], slot, generation)
    values = list(args)
    values[5] = forwarded
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = QueueStream(forwarded, "stdout"), QueueStream(
        forwarded, "stderr"
    )
    finished = threading.Event()
    heartbeat = None
    if len(args) > 9 and args[9]:
        heartbeat = threading.Thread(
            target=_heartbeat, args=(forwarded, finished, args[3]), daemon=True
        )
        heartbeat.start()
    try:
        return _run_tasks(*values)
    finally:
        finished.set()
        if heartbeat is not None:
            heartbeat.join()
        sys.stdout, sys.stderr = old_stdout, old_stderr


def emit(log, event):
    """The exact parent console bytes also go into its single log."""
    if event.get("kind") == "console":
        text = event["text"]
        stream = sys.stderr if event["stream"] == "stderr" else sys.stdout
    elif event.get("traceback"):
        text, stream = event["traceback"], sys.stderr
    else:
        import json

        text, stream = (
            utc() + " " + json.dumps(event, ensure_ascii=False) + "\n",
            sys.stdout,
        )
    print(text, file=stream, end="", flush=True)
    log.write(text)
    log.flush()


def main(argv=None):
    """Start the common CPU or CUDA queue worker with one verified code environment.

    Worker slots claim complete runs, report actual progress, and publish only
    while holding their attempt lease. Startup failures retain their traceback.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--machine", default=socket.gethostname())
    parser.add_argument("--stores",type=Path,
        help="Explicit host-local stores.toml (never written into job files)")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--exit-when-idle", action="store_true")
    parser.add_argument(
        "--watchdog",
        action="store_true",
        help="Parent restarts a silent owned worker only after exact retirement.",
    )
    parser.add_argument(
        "--watchdog-seconds",
        type=float,
        default=600.0,
        help="Operational liveness limit; actual-progress queue lease remains 600s.",
    )
    args = parser.parse_args(argv)
    args._cache_terminal_events = []
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not args.watchdog_seconds > 0 or not math.isfinite(args.watchdog_seconds):
        parser.error("--watchdog-seconds must be finite and positive")
    identity = name(args.machine + "-" + args.backend)
    from ..job_filesystem import inspect_shared_folder
    transport=inspect_shared_folder(args.job)
    if not transport.allowed:
        raise ValueError("Shared queue transport refused: "+transport.reason+": "+transport.evidence)
    with ServiceLock(identity):
        log_root = args.job / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        with (log_root / (identity + ".log")).open("a", encoding="utf-8") as log:
            try:
                startup_context={}
                # N111: a transient SMB error at startup waits and retries.
                job, commit = retry_transient(lambda: verify_startup(args.job,
                    stores_path=args.stores,machine_id=args.machine,
                    startup_context=startup_context), delays=QUEUE_RETRY_DELAYS)
                args._alias_stores=startup_context.get("stores")
                from darsia_cache.adapters import bind_actual_worker_owner, require_same_startup_stores
                if args._alias_stores is not None:
                    require_same_startup_stores(args._alias_stores)
                bind_actual_worker_owner(_identity(identity),role="service_parent")
                emit(log, dict(kind="startup_code_admission",
                    environment_lock=startup_context["environment_lock_identity"]))
            except Exception:
                literal = traceback.format_exc()
                emit(log, dict(kind="startup_failure", traceback=literal))
                from darsia_cache.adapters import seal_worker_service
                seal_worker_service(job_root=args.job, exit_code=1,
                    terminal_events=[], traceback_text=literal)
                return 1
            try:
                if job.get("execution_mode") == "current_build":
                    from .current_build_job import run_current_build_job
                    result = run_current_build_job(args, job, commit, log)
                else:
                    result = _parent(args, identity, job, commit, log)
            except BaseException:
                literal = traceback.format_exc()
                from darsia_cache.adapters import seal_worker_service
                seal_worker_service(job_root=args.job, exit_code=1,
                    terminal_events=args._cache_terminal_events, traceback_text=literal)
                raise
            from darsia_cache.adapters import seal_worker_service
            seal_worker_service(job_root=args.job, exit_code=result,
                terminal_events=args._cache_terminal_events)
            return result


class _SlotMessages:
    """Tag owned child messages; only the parent prints or writes the log."""

    def __init__(self, messages, slot, generation):
        self.messages, self.slot, self.generation = messages, slot, generation

    def put(self, event):
        self.messages.put(
            dict(event, worker_slot=self.slot, worker_generation=self.generation)
            if self.slot is not None
            else event
        )


def _heartbeat(messages, finished, machine):
    owner = _identity(machine)
    while not finished.is_set():
        messages.put(dict(kind="watchdog_heartbeat", actual_worker=owner))
        finished.wait(5.0)  # Control liveness only; never JobQueue.progress.


def _process_identity(process):
    value = dict(pid=process.pid, process_create_time=str(process.create_time()),
                 command_line=process.cmdline())
    from darsia_cache.owned_process import native_launch_identity
    try:
        native = native_launch_identity(process)
    except OSError as error:
        # Preserve missing native evidence without breaking original science.
        # Such an observation cannot satisfy automatic retirement eligibility.
        value["native_identity_unavailable"] = dict(
            exception_type=type(error).__name__, message=str(error))
    else:
        if native is not None:
            value.update(native)
    return value


def _same_process(identity):
    """PID reuse proves the old incarnation absent, never licenses a new kill."""
    import psutil

    if identity is None:
        return None
    try:
        process = psutil.Process(identity["pid"])
        return (
            process
            if process.create_time() == float(identity["process_create_time"])
            else None
        )
    except psutil.NoSuchProcess:
        return None


def retire_owned_processes(root_identity, observed=()):
    """Retire only the exact known child tree, then prove every incarnation absent."""
    import psutil

    identities = {
        (i["pid"], i["process_create_time"]): dict(i)
        for i in (*(() if root_identity is None else (root_identity,)), *observed)
    }
    root = _same_process(root_identity)
    if root is not None:
        for process in root.children(recursive=True):
            try:
                bound = _process_identity(process)
            except psutil.NoSuchProcess:
                continue
            identities[bound["pid"], bound["process_create_time"]] = bound
    # Known children were observed as descendants or arrived over this slot's
    # owned IPC. Do not include the watchdog/log parent's ancestors.
    live = []
    for bound in reversed(list(identities.values())):
        process = _same_process(bound)
        if process is not None:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                continue
            live.append(process)
    _, pending = psutil.wait_procs(live, timeout=10.0)
    for process in pending:
        bound = next(i for i in identities.values() if i["pid"] == process.pid)
        current = _same_process(bound)
        if current is not None:
            try:
                current.kill()
            except psutil.NoSuchProcess:
                continue
    psutil.wait_procs(pending, timeout=10.0)
    if any(_same_process(bound) is not None for bound in identities.values()):
        raise RuntimeError(
            "Owned child retirement is incomplete; no replacement may start"
        )
    return list(identities.values())


def _start_slot(context, args, job, commit, identity, messages, stop, number):
    import uuid

    generation = uuid.uuid4().hex
    process = context.Process(
        target=_work,
        args=(
            str(args.job),
            job,
            commit,
            identity,
            args.backend,
            messages,
            stop,
            args.once,
            args.exit_when_idle,
            args.watchdog,
            number,
            generation,
            getattr(args,"_alias_stores",None),
        ),
    )
    from darsia_cache.adapters import record_cache_observation
    record_cache_observation("worker_spawn_intent",dict(slot=number,generation=generation))
    process.start()
    import psutil

    try:
        bound = _process_identity(psutil.Process(process.pid))
    except psutil.NoSuchProcess:
        bound = None  # A completed launch has no invented psutil epoch/argv.
    # multiprocessing retains the actual original process HANDLE even when a
    # very short launch has exited; bind its native FILETIME before dropping it.
    from darsia_cache.owned_process import native_launch_identity, bind_retained_native_observation
    try:
        native = native_launch_identity(process, sentinel=getattr(process, "sentinel", None))
    except OSError as error:
        native = None
        if bound is not None:
            bound["native_identity_unavailable"] = dict(
                exception_type=type(error).__name__, message=str(error))
    if native is not None:
        bound = bind_retained_native_observation(native, bound)
    slot = dict(
        number=number,
        generation=generation,
        process=process,
        root=bound,
        observed=[],
        last_heartbeat=time.monotonic(),
        claim=None,
        retired=False,
        waiting_reclaim=False,
        finished=False,
        cache_observation_error=None,
    )
    try:
        record_cache_observation("worker_spawn_observed",dict(slot=number,generation=generation,
            original_native_identity=bound,epoch_unavailable=bound is None))
    except Exception as error:
        # This supplementary record must not discard the already-owned child.
        # Its mandatory native prelude attachment remains unchanged.
        slot["cache_observation_error"]=dict(exception_type=type(error).__name__,
            message=str(error),observation_complete=False)
        try:
            import logging
            logging.getLogger(__name__).error("Cache spawn observation incomplete: %s",
                slot["cache_observation_error"],exc_info=True)
        except BaseException:
            slot["cache_observation_error"]["logging_failed"]=True
    return slot


def _record_retirement(slot, args, identity, *, before_record=None):
    """The live parent publishes a separate genuine owned-child closure."""
    if slot["root"] is None and slot["process"].is_alive():
        raise RuntimeError(
            "Live launch has no bound epoch; preserve it and do not replace"
        )
    if before_record is not None:
        before_record()
    identities = retire_owned_processes(slot["root"], slot["observed"])
    slot["observed"] = identities
    slot["process"].join(timeout=1.0)
    if before_record is not None:
        proved = deepcopy(identities)
        before_record()
        if slot["observed"] != proved:
            identities = retire_owned_processes(slot["root"], slot["observed"])
            slot["observed"] = identities
            slot["process"].join(timeout=1.0)
    claim = slot["claim"]
    if claim is not None:
        from .checkpoint import retirement_document

        output = (
            args.job
            / "results"
            / name(claim["run"])
            / ".attempts"
            / claim["attempt_id"]
        )
        if claim["task"].get("format")=="darsia-alias-task-v1":
            from .worker_io import retirement_output
            stores=getattr(args,"_alias_stores",None)
            output,actual_owner=retirement_output(args.job,claim["task"],identity,stores)
        document = retirement_document(
            claim["task"], output, identities, _identity(identity)
        )
        if claim["task"].get("format")=="darsia-alias-task-v1":
            document["machine_id"]=stores.machine_id
            document["output_reference"]=actual_owner["output"]
        document["retired_utc"] = utc()
        document["actual_launch_wrapper"] = slot["root"]
        document["owned_launch_handle_retired"] = not slot["process"].is_alive()
        path = output / "retirement.json"
        setup_writer = None
        if claim["task"].get("pipeline") == "series-setup":
            from .series_setup_worker_admission import retirement_control_writer
            writers = slot.setdefault("setup_retirement_writers", {})
            setup_writer = writers.get(claim["attempt_id"])
            if setup_writer is None:
                setup_writer = retirement_control_writer(args.job, claim["task"], stores, output, actual_owner)
                writers[claim["attempt_id"]] = setup_writer
            elif (setup_writer.task != claim["task"] or setup_writer.owner != actual_owner
                    or setup_writer.output != output):
                raise ValueError("The retained supervisor writer belongs to another claim")
        if path.exists():
            old = read_json(path)
            if (
                old.get("attempt_id") != claim["attempt_id"]
                or old.get("all_absent") is not True
                or old.get("identities") != document["identities"]
            ):
                raise ValueError("A different retirement already exists; preserve it")
        else:
            if setup_writer is None:
                write_immutable(path, document)
            else:
                with setup_writer.activate():
                    write_immutable(path, document)
                setup_writer.close()
        if claim["task"].get("format")=="darsia-alias-task-v1":
            from .capacity import release_retired_capacity
            from .folder_capacity import CapacityLockBusy
            try:
                release_retired_capacity(args.job,claim["task"],stores,document, wait_seconds=0)
            except CapacityLockBusy:
                # Another child may still own admission. Its kernel lock must
                # not prevent this supervisor from retiring the other children.
                pending = getattr(args, "_pending_capacity_releases", None)
                if pending is None:
                    pending = args._pending_capacity_releases = {}
                pending[claim["attempt_id"]] = (deepcopy(claim["task"]), stores, deepcopy(document))
    from darsia_cache.adapters import record_cache_observation
    record_cache_observation("worker_children_retired",dict(slot=slot["number"],
        generation=slot["generation"],original_child_identities=identities,
        owned_launch_handle_retired=not slot["process"].is_alive(),
        service_parent_retirement_proven=False))
    slot["retired"] = True
    return identities


def _release_pending_capacity(args, *, final=False):
    """Retry genuine retirement releases without inventing process absence."""
    from .capacity import release_retired_capacity
    from .folder_capacity import CapacityLockBusy
    pending = getattr(args, "_pending_capacity_releases", {})
    for token, (task, stores, document) in list(pending.items()):
        try:
            release_retired_capacity(args.job, task, stores, document,
                wait_seconds=30 if final else 0)
        except CapacityLockBusy:
            continue
        del pending[token]
    return not pending


def _ready_setup_operations(root, job, backend):
    """Inspect queue membership using the admitted small operation index only."""
    if job.get("pipeline") != "series-setup" or backend != "cpu":
        return []
    operations = job.get("setup_operations")
    if not isinstance(operations, list):
        return []  # Older jobs keep their original worker allocation.
    from .setup_image_preparation import build_dependencies_ready
    return [row for row in operations
            if row.get("operation") in {"apply", "build", "image-preparation"}
            and (Path(root) / "queue" / (name(row["run"]) + ".json")).is_file()
            and (row.get("operation") != "image-preparation" or build_dependencies_ready(root, job))]


def _parent(args, identity, job, commit, log):
    """Own one log and persistent slots; optionally supervise exact child liveness."""
    import psutil

    if not hasattr(args, "_cache_terminal_events"):
        args._cache_terminal_events = []
    shared = (JobQueue(args.job,machine_id=args._alias_stores.machine_id)
              if job.get("format")=="darsia-alias-job-v1" else JobQueue(args.job,machine_id=identity))
    context = mp.get_context("spawn")
    messages, stop = context.Queue(), context.Event()
    setup_job = job.get("pipeline") == "series-setup"
    ready = _ready_setup_operations(args.job, job, args.backend)
    initial_count = (1 if len(ready) == 1 and ready[0]["operation"] == "image-preparation"
                    or args.once and len(ready) == 1 and ready[0]["operation"] == "apply"
                     else args.workers)
    slots = [
        _start_slot(context, args, job, commit, identity, messages, stop, i)
        for i in range(initial_count)
    ]
    emit(
        log,
        dict(kind="started", message=f"Worker service started at Git commit {commit}."),
    )
    fatal = False
    control_failed = False
    next_capacity_retry = 0.0
    batch_event_start = len(args._cache_terminal_events)

    def parent_retry(step):
        """Log one wait for a transient SMB error in a parent queue step (N111)."""
        return lambda attempt, delay, error: emit(log, dict(
            kind="watchdog_transient_retry", step=step, attempt=attempt,
            delay_seconds=delay, error=repr(error)))

    def maintain():
        shared.recover_publications()
        return shared.reclaim_stale()

    def handle_event(event):
        """Validate and attach one exact child event before retirement."""
        nonlocal control_failed
        number = event.get("worker_slot")
        slot = (
            slots[number]
            if isinstance(number, int) and 0 <= number < len(slots)
            else None
        )
        if (
            slot is not None
            and event.get("worker_generation") != slot["generation"]
        ):
            if event.get("kind") != "watchdog_heartbeat":
                emit(
                    log, event
                )  # Preserve old output, never attribute it to the replacement.
            return False
        owner = event.get("actual_worker")
        if slot is not None and owner is not None:
            if owner.get("machine") != identity:
                raise ValueError("Owned worker message machine differs")
            process = _same_process(owner)
            if process is not None:
                root = _same_process(slot["root"])
                try:
                    owned = root is not None and (
                        process.pid == root.pid
                        or any(
                            parent.pid == root.pid
                            and parent.create_time() == root.create_time()
                            for parent in process.parents()
                        )
                    )
                except psutil.NoSuchProcess:
                    owned = _same_process(owner) is None
                if not owned:
                    raise ValueError(
                        "Live worker is outside its owned launch tree"
                    )
            if owner not in slot["observed"]:
                slot["observed"].append(dict(owner))
        if slot is not None and event.get("kind") == "watchdog_heartbeat":
            slot["last_heartbeat"] = time.monotonic()
        elif event.get("kind") == "watchdog_heartbeat":
            raise ValueError("Watchdog message has no owned worker slot")
        else:
            if slot is not None and event.get("kind") == "claimed":
                path = (
                    args.job
                    / "claimed"
                    / identity
                    / (name(event["run"]) + ".json")
                )
                if job.get("format")=="darsia-alias-job-v1":
                    from .worker_io import claimed_task_after_archive
                    task=claimed_task_after_archive(args.job,identity,event)
                else:
                    task = read_json(path)
                if task["attempt_id"] != event["attempt_id"]:
                    raise ValueError(
                        "Claim message and actual task generation differ"
                    )
                slot["claim"] = dict(
                    run=event["run"], attempt_id=event["attempt_id"], task=task
                )
            if slot is not None and event.get("kind") in {
                "completed",
                "failed",
                "lease_lost",
                "input_rejected",
            }:
                # Terminal/revoked attempts do not need a numerical restart.
                if event.get("capacity_retained") is not True:
                    slot["claim"] = None
            emit(log, event)
            if event.get("kind") in {"completed", "failed", "lease_lost", "input_rejected",
                    "control_failure", "paused", "control_stopped"}:
                args._cache_terminal_events.append(dict(event))
            if event.get("kind") == "control_failure":
                control_failed = True

    def drain_pending_events():
        """Attach flushed child ownership events before their retirement record."""
        nonlocal control_failed
        while True:
            try:
                event = messages.get_nowait()
            except queue.Empty:
                return
            if setup_job:
                handle_event(event)
            else:
                if event.get("kind") != "watchdog_heartbeat":
                    emit(log, event)
                if event.get("kind") in {"completed", "failed", "lease_lost", "input_rejected",
                        "control_failure", "paused", "control_stopped"}:
                    args._cache_terminal_events.append(dict(event))
                if event.get("kind") == "control_failure":
                    control_failed = True

    try:
        while True:
            try:
                event = messages.get(timeout=0.2)
            except queue.Empty:
                event = None
            if event is not None:
                if handle_event(event) is False:
                    continue
            active = False
            for number, slot in enumerate(slots):
                if slot["finished"]:
                    continue
                process = slot["process"]
                stalled = (
                    args.watchdog
                    and process.is_alive()
                    and time.monotonic() - slot["last_heartbeat"]
                    >= args.watchdog_seconds
                )
                if process.is_alive() and not stalled:
                    active = True
                    continue
                if not slot["retired"]:
                    identifiers = _record_retirement(slot, args, identity,
                        **({"before_record": drain_pending_events} if setup_job else {}))
                    emit(
                        log,
                        dict(
                            kind=(
                                "watchdog_retired" if args.watchdog else "child_retired"
                            ),
                            slot=number,
                            identities=identifiers,
                            stalled=stalled,
                            exitcode=process.exitcode,
                        ),
                    )
                    clean_exit = process.exitcode == 0 and not stalled
                    if (
                        not args.watchdog
                        or clean_exit
                        or (args.once and slot["claim"] is None)
                    ):
                        slot["finished"] = True
                        fatal |= process.exitcode not in (0, None)
                        continue
                    slot["waiting_reclaim"] = slot["claim"] is not None
                if slot["waiting_reclaim"]:
                    # Keep the original 600-second actual-progress lease unchanged.
                    for record in retry_transient(maintain, delays=QUEUE_RETRY_DELAYS,
                                                  on_retry=parent_retry("reclaim")):
                        emit(log, dict(kind="reclaimed", **record))
                    claim = slot["claim"]
                    lease = (
                        args.job
                        / "leases"
                        / (name(claim["run"]) + "." + claim["attempt_id"])
                    )
                    # A published lease stays in leases/ (publishing/<run> is
                    # its marker file), so the publication is checked first.
                    if (args.job / "publishing" / name(claim["run"])).exists():
                        slot["finished"] = True
                        continue
                    # A reclaimed or rejected lease also stays in leases/;
                    # its decision and recorded retry end the wait.
                    if lease.exists() and not lease_retired(
                        args.job, claim["run"], claim["attempt_id"]
                    ):
                        active = True
                        continue
                    slot["waiting_reclaim"] = False
                # Re-verify the same startup admission before a replacement imports
                # checkout code. No per-file code guard or generated start code.
                if job.get("format")=="darsia-alias-job-v1":
                    same_job,same_commit=retry_transient(lambda: verify_startup(args.job,
                        stores_path=getattr(args,"stores",None),machine_id=args.machine,
                        expected_stores=args._alias_stores),
                        delays=QUEUE_RETRY_DELAYS, on_retry=parent_retry("restart_admission"))
                else:
                    same_job, same_commit = retry_transient(
                        lambda: verify_startup(args.job), delays=QUEUE_RETRY_DELAYS,
                        on_retry=parent_retry("restart_admission"))
                if same_job != job or same_commit != commit:
                    raise RuntimeError("Watchdog restart startup identity differs")
                slots[number] = _start_slot(
                    context, args, job, commit, identity, messages, stop, number
                )
                emit(
                    log,
                    dict(
                        kind="watchdog_restarted",
                        slot=number,
                        actual_launch_wrapper=slots[number]["root"],
                    ),
                )
                active = True
            if time.monotonic() >= next_capacity_retry:
                _release_pending_capacity(args)
                next_capacity_retry = time.monotonic() + 2.0
            if not active and all(slot["finished"] for slot in slots):
                # --once limits a child to one whole operation, not the
                # detached Setup service to one GUI-owned admission cycle.
                # Continue only after actual completion, never after a pure
                # capacity refusal or a failed/stopped batch.
                completed = any(event.get("kind") == "completed"
                                for event in args._cache_terminal_events[batch_event_start:])
                ready = _ready_setup_operations(args.job, job, args.backend)
                if (setup_job and completed and ready and not fatal and not control_failed
                        and not stop.is_set() and all(row["operation"] in {"build", "image-preparation"} for row in ready)):
                    try:
                        _job_control_boundary(args.job)
                    except _JobControlBoundary as boundary:
                        if boundary.state == "control_failure":
                            control_failed = True
                        emit(log, dict(kind=boundary.state, message="Setup stopped before the next Build batch."))
                        break
                    if job.get("format") == "darsia-alias-job-v1":
                        same_job, same_commit = verify_startup(args.job,
                            stores_path=getattr(args, "stores", None), machine_id=args.machine,
                            expected_stores=args._alias_stores)
                    else:
                        same_job, same_commit = verify_startup(args.job)
                    if same_job != job or same_commit != commit:
                        raise RuntimeError("Setup continuation startup identity differs")
                    batch_event_start = len(args._cache_terminal_events)
                    slots = [_start_slot(context, args, job, commit, identity, messages, stop, i)
                             for i in range(min(args.workers, len(ready)))]
                    emit(log, dict(kind="setup_build_batch_started", workers=len(slots),
                                   configured_workers=args.workers,
                                   message="Continuing queued Build operations in the detached worker service."))
                    continue
                break
    except KeyboardInterrupt:
        stop.set()  # Existing user interrupt behavior: finish the owned run.
    except Exception:
        fatal = True
        stop.set()
        emit(log, dict(kind="parent_failure", traceback=traceback.format_exc()))
    finally:
        for slot in slots:
            if fatal and slot["process"].is_alive():
                _record_retirement(slot, args, identity,
                    **({"before_record": drain_pending_events} if setup_job else {}))
            else:
                slot["process"].join()
                if setup_job:
                    drain_pending_events()
                    claim = slot["claim"]
                    if not slot["retired"] and claim is not None:
                        _record_retirement(slot, args, identity, before_record=drain_pending_events)
        drain_pending_events()
        if not _release_pending_capacity(args, final=True):
            fatal = True
            emit(log, dict(kind="capacity_release_pending",
                attempts=list(args._pending_capacity_releases),
                message="Workers retired; another admission still holds the capacity lock. Original retirement evidence and reservations are retained."))
        emit(log, dict(kind="stopped", message="Worker service stopped."))
    return 1 if fatal or control_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
