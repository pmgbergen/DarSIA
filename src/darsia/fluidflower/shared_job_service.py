"""Finite local admission cycles for jobs published in a shared folder.

The existing queue owns claims, calculations, progress and retirement. This
coordinator only starts the installed worker and publishes its GUI's presence.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import uuid

from .job_control import read_control
from .job_launch_state import UNKNOWN, _epoch, begin_launch_request, read_launch_requests, record_launch_error
from .job_management import JobActionLock
from .pipeline.job_queue import descriptor, name, read_json
from .shared_job_folder import SharedJobFolder
from .shared_job_capabilities import job_capabilities


def _message(error):
    lines = str(error).strip().splitlines()
    return (lines[-1] if lines else type(error).__name__)[:500]


def _same_machine(left, right):
    try:
        return name(str(left)) == name(str(right))
    except ValueError:
        return False


def reported_launch_message(request, observed_workers):
    """Describe already-read progress, without changing a launch reservation.

    This is a last-report presentation, never a liveness or retirement test.
    A report from another backend, or before this request, cannot describe it.
    """
    original = request.get("error_traceback") or request.get("state", UNKNOWN)
    if (request.get("error_traceback") or request.get("released")
            or request.get("terminal_report") or request.get("state") != UNKNOWN):
        return original
    requested = _epoch(request.get("requested_utc"))
    if requested is None:
        return original
    backend, machine = request["backend"], request["machine"]
    matches = []
    for worker in observed_workers:
        recorded_backend = worker.get("backend")
        same = _same_machine(worker.get("machine", ""), machine + "-" + backend)
        same |= recorded_backend == backend and _same_machine(worker.get("machine", ""), machine)
        stamp = _epoch(worker.get("last_progress_utc"))
        state, run = worker.get("state"), worker.get("run")
        if (not same or recorded_backend not in (None, backend) or stamp is None or stamp <= requested
                or not isinstance(state, str) or not state.strip() or not isinstance(run, str) or not run.strip()
                or state.casefold().startswith(("queued", "waiting", "pending", "prepared", "not reported", "unknown"))):
            continue
        matches.append((stamp, worker))
    if not matches:
        return original
    _, latest = max(matches, key=lambda item: item[0])
    state = latest["state"]
    description = f"{state} in {latest['run']}"
    if " · Last report: " in state:
        report = state.split(" · Last report: ", 1)[1].split(" / ", 2)
        if len(report) == 3 and all(report):
            description = f"{report[0].replace('-', ' ').capitalize()} in {report[1].upper()} · {report[2]}"
    detail = f"Worker reported {description} · last progress {latest['last_progress_utc']}"
    if len(matches) > 1:
        detail += f" · {len(matches)} worker reports"
    return detail[:500]


def _document(path):
    if Path(path).stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Job metadata exceeds its size limit")
    return read_json(Path(path))


def backend_blocker(job, backend, *, machine, master_machine):
    """Mirror installed runner capabilities without relaxing their admission."""
    pipeline = job.get("pipeline", "n1")
    if backend not in job_capabilities(job)["supported_backends"]:
        return f"{pipeline} does not support {backend.upper()} workers"
    if job.get("backend", "any") not in {"any", backend}:
        return f"This job requires {str(job['backend']).upper()} workers"
    if job.get("execution_mode") == "current_build":
        from .pipeline.current_build_job import SHARED_RUNS
        if job.get(SHARED_RUNS) is True:
            if backend != "cpu":
                return "Current Build P1 requires CPU workers"
        elif backend != "cpu" or not _same_machine(job.get("target_machine", ""), machine):
            return "Current Build P1 requires CPU workers on its Build machine"
    if pipeline == "series-setup" and not _same_machine(machine, master_machine):
        return "Setup runs on its master machine"
    if job.get("format") == "darsia-alias-job-v1" and (
        job.get("execution_ready") is not True
        or job.get("runtime_support") != "darsia-role-io-runtime-v1"
    ):
        return "This job has not been prepared for the installed worker runtime"
    return ""


def _shared_runs_remain(root, job):
    """Whether a shared phase-map job has runs left that a machine may claim.

    A failed run does not stop the job's other runs from starting elsewhere.
    """
    if job.get("execution_mode") != "current_build":
        return False
    from .pipeline.current_build_job import SHARED_RUNS, claimable_runs
    return job.get(SHARED_RUNS) is True and bool(claimable_runs(root, job))


def sample_owned_resources(root, machine, backend, snapshot):
    """Observe memory only for original claim owners with matching PID/birth."""
    import psutil
    identity = name(machine + "-" + backend)
    processes = []
    for row in snapshot.get("workers", []):
        if not _same_machine(row.get("machine", ""), identity):
            continue
        try:
            run = name(row["run"])
            task = _document(root / "claimed" / identity / (run + ".json"))
            token = task["attempt_id"]
            if not isinstance(token, str) or len(token) != 32 or any(char not in "0123456789abcdef" for char in token):
                continue
            owner_path = root / "leases" / (run + "." + token) / "owner.json"
            owner = _document(owner_path)
            actual = owner["actual_owner"]
            if (owner.get("task") != task or owner.get("token") != token
                    or not _same_machine(actual.get("machine", ""), identity)
                    or type(actual.get("pid")) is not int or actual["pid"] <= 0):
                continue
            process = psutil.Process(actual["pid"])
            birth = float(actual["process_create_time"])
            if process.create_time() != birth:
                continue
            rss = process.memory_info().rss
            if process.create_time() == birth:
                threads = task.get("threads")
                # These installed runners explicitly use task.get('threads', 1).
                # Other pipelines require their saved count; no hardware guess.
                if "threads" not in task and task.get("pipeline", "n1") in {
                    "n1", "native-phase-n1", "lattice-phase-n1", "native-phase-statistics"
                }:
                    threads = 1
                processes.append((process, birth, rss, threads, descriptor(owner_path)))
        except (OSError, ValueError, KeyError, TypeError, psutil.Error):
            continue
    if not processes:
        return {}
    largest = max(processes, key=lambda item: item[2])
    result = dict(peak_worker_rss_bytes=largest[2], threads=largest[3], source=largest[4],
                  available_bytes=psutil.virtual_memory().available)
    if backend == "cuda" and (program := shutil.which("nvidia-smi")):
        try:
            query = subprocess.run(
                [program, "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=2,
                creationflags=0x08000000 if sys.platform == "win32" else 0,
            )
            usage = {}
            for line in query.stdout.splitlines():
                pid, device, amount = line.split(",")
                usage[int(pid.strip())] = (device.strip(), int(amount.strip()) * 1024 * 1024)
            values = [usage[process.pid] for process, birth, *_ in processes
                      if process.pid in usage and process.create_time() == birth]
            if values and len({device for device, _ in values}) == 1:
                device = values[0][0]
                result["device_id"] = device
                result["peak_device_memory_bytes"] = max(amount for _, amount in values)
                available = subprocess.run(
                    [program, "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True, timeout=2,
                    creationflags=0x08000000 if sys.platform == "win32" else 0,
                )
                for line in available.stdout.splitlines():
                    identifier, amount = line.split(",")
                    if identifier.strip() == device:
                        result["available_device_bytes"] = int(amount.strip()) * 1024 * 1024
        except (OSError, ValueError, psutil.Error, subprocess.SubprocessError):
            pass
    return result


class SharedJobService:
    """Call tick in a background GUI task; it never waits for calculations."""

    def __init__(self, folder, machine, *, session_id=None, cpu_workers=0,
                 cuda_workers=0, role="worker", stores=None, launcher=None,
                 reader=None, memory_sampler=None, tuner=None):
        self.folder = SharedJobFolder(folder) if isinstance(folder, (str, Path)) else folder
        self.machine = name(machine)
        self.session_id = session_id or uuid.uuid4().hex
        if role not in {"master", "worker"}:
            raise ValueError("Choose master or worker role")
        self.role, self.stores = role, stores
        if launcher is None:
            from .jobs_panel import launch_local_worker
            launcher = launch_local_worker
        if reader is None:
            from .jobs_metadata import JobSnapshotReader
            reader = JobSnapshotReader()
        self.launcher, self.reader = launcher, reader
        self.memory_sampler = memory_sampler or sample_owned_resources
        self._guard = threading.Lock()
        self._accepting = True
        self._batches = {}
        self._samples = []
        self._observed_pending = set()
        self._last_jobs = []
        self.tuner = tuner
        self.update_capacity(cpu_workers, cuda_workers)

    def set_tuner(self, tuner):
        """An optional advisor changes allocations before the next admission."""
        with self._guard:
            self.tuner = tuner

    def update_capacity(self, cpu_workers, cuda_workers):
        """Change future batches only; existing processes keep their allocation."""
        if any(type(count) is not int or not 0 <= count <= 256
               for count in (cpu_workers, cuda_workers)):
            raise ValueError("Worker counts must be integers between 0 and 256")
        with self._guard:
            self.cpu_workers, self.cuda_workers = cpu_workers, cuda_workers

    @contextmanager
    def _admission(self):
        from .shared_admission_lock import admission
        with admission(self.folder, self.machine, self.session_id):
            yield

    def _queued_worker_count(self, root, job, backend, configured):
        """Keep capacity while allocating one child to the serial prerequisite."""
        if job.get("execution_mode") == "current_build":
            from .pipeline.current_build_job import SHARED_RUNS, claimable_runs
            # A shared job keeps no worker once every run is claimed or done.
            if job.get(SHARED_RUNS) is True and not claimable_runs(root, job):
                return 0
            return configured
        serial_apply = False
        for run in job.get("expected_runs", []):
            path = root / "queue" / (name(run) + ".json")
            try:
                task = _document(path)
            except FileNotFoundError:
                continue  # Another worker atomically claimed it.
            if task.get("run") != run:
                raise ValueError("Queued task differs from its expected run")
            pipeline = task.get("pipeline", job.get("pipeline", "n1"))
            if (backend in job_capabilities(dict(task, pipeline=pipeline))["supported_backends"]
                    and task.get("backend", "any") in {"any", backend}):
                if job.get("pipeline") == "preset-wasserstein":
                    return 1
                if (job.get("pipeline") != "series-setup" or pipeline != "series-setup"
                        or task.get("operation") != "apply" or serial_apply):
                    return configured
                serial_apply = True
        # The detached supervisor owns Apply -> Build and retains the selected
        # capacity even when this GUI's admission timer no longer exists.
        return configured if serial_apply else 0

    def _samples_from(self, snapshot, requests):
        for request in requests:
            key = request["source"]["path"]
            batch = self._batches.get(key)
            if batch is None or not request.get("released"):
                continue
            self._batches.pop(key)
            terminal = request.get("terminal_report") or {}
            if (terminal.get("event", {}).get("kind") != "stopped"
                    or request.get("error_traceback") or snapshot.get("errors")
                    or snapshot.get("failed_runs")):
                continue
            identity = name(self.machine + "-" + request["backend"])
            completed = [row for row in snapshot.get("runs", [])
                         if row.get("state") == "Complete"
                         and _same_machine(row.get("machine", ""), identity)
                         and batch["before"].get(row["run"], 0) == 0]
            frames = sum(row.get("committed", 0) for row in completed)
            try:
                elapsed = (datetime.fromisoformat(terminal["timestamp"].replace("Z", "+00:00"))
                           - datetime.fromisoformat(request["requested_utc"].replace("Z", "+00:00"))).total_seconds()
            except (KeyError, ValueError, TypeError):
                continue
            if frames > 0 and elapsed > 0:
                self._samples.append(dict(
                    sample_id=key, job_id=batch["job_id"], workload_key=batch["job_id"],
                    backend=request["backend"], workers=request["workers"],
                    completed_frames=frames, elapsed_seconds=elapsed,
                    job_facts=batch["job_facts"], configured_counts=batch["configured_counts"],
                    eligible_backends=batch["eligible_backends"],
                    started_utc=request["requested_utc"], completed_utc=terminal["timestamp"],
                    **batch["memory"],
                ))
        self._samples = self._samples[-64:]

    def _cycle(self, configured, accepting):
        entries = self.folder.jobs()
        rows, records, occupied, errors = [], [], {}, []
        new_batches = []
        # Read every published reservation before choosing a new job. A failed
        # read cannot silently free a backend that might already be computing.
        for entry in entries:
            root = Path(entry["job_root"])
            row = dict(job_id=entry["job_id"], job_root=str(root),
                       master_machine=entry["master_machine"], state=entry["state"],
                       progress={}, backends={})
            rows.append(row)
            try:
                requests = read_launch_requests(root)
                for request in requests:
                    if _same_machine(request["machine"], self.machine) and not request["released"]:
                        occupied.setdefault(request["backend"], []).append(request)
                        self._observed_pending.add(request["source"]["path"])
                job = _document(root / "job.json")
                row["series_id"] = job.get("series_id") or job.get("series")
                snapshot = self.reader(dict(root=str(root), name=root.name))
                from .job_lifecycle import recovery_state, user_status
                recovery = recovery_state(root)
                recovered = recovery["previous_launches"]
                row["capabilities"] = job_capabilities(job)
                for request in requests:
                    batch = self._batches.get(request["source"]["path"])
                    if batch is not None and not request["released"]:
                        memory = self.memory_sampler(root, self.machine, request["backend"], snapshot)
                        for key, value in memory.items():
                            if key.startswith("peak_"):
                                value = max(value, batch["memory"].get(key, 0))
                            batch["memory"][key] = value
                self._samples_from(snapshot, requests)
                row["state"] = snapshot.get("state", entry["state"])
                if (recovery["intent"] or {}).get("action") == "reset":
                    row["state"] = user_status(dict(snapshot, root=str(root)),
                        local_machine=self.machine, stores_path=self.stores)["state"]
                row["progress"] = {key: snapshot.get(key) for key in
                                   ("committed", "total_frames", "completed_runs", "run_count")}
                row["observed_workers"] = [
                    {key: worker.get(key) for key in ("machine", "backend", "worker", "run", "state", "last_progress_utc")}
                    for worker in snapshot.get("workers", [])
                ][:256]
                records.append((entry, root, job, snapshot, requests, row, recovered))
            except Exception as error:
                row.update(state="Metadata error", error=_message(error))
                errors.append(f"{root.name}: {_message(error)}")
        with self._guard:
            tuner = self.tuner
        if tuner is not None and accepting:
            tuner.observe(dict(machine=self.machine, jobs=rows, errors=errors,
                               samples=deepcopy(self._samples), completed_batches=deepcopy(self._samples)))
            self.update_capacity(*tuner.counts())
            configured.update(cpu=self.cpu_workers, cuda=self.cuda_workers)
        # Automatic measurements compare complete CPU/GPU allocations. Let a
        # faster backend finish without beginning another allocation while its
        # partner batch is still running. Compute once, so an empty cycle can
        # admit both backends. Manual pools remain independently scheduled.
        automatic_batch_pending = tuner is not None and bool(self._batches)
        for entry, root, job, snapshot, requests, row, recovered in records:
            for backend, count in configured.items():
                own = [request for request in requests
                       if _same_machine(request["machine"], self.machine)
                       and request["backend"] == backend]
                pending = [request for request in own if not request["released"]]
                status = row["backends"][backend] = dict(state="Idle", workers=0, configured_workers=count)
                if pending:
                    status.update(state="Launch requested", workers=sum(request["workers"] for request in pending),
                                  message="; ".join(reported_launch_message(request, row["observed_workers"])
                                                    for request in pending)[-500:])
                    continue
                if not accepting or count == 0:
                    status["state"] = "Disabled"
                    continue
                if errors:
                    status.update(state="Waiting", message="Resolve shared job metadata errors before new starts")
                    continue
                if automatic_batch_pending:
                    status.update(state="Waiting", message="Waiting for current automatic batch to finish")
                    continue
                if occupied.get(backend):
                    status.update(state="Waiting", message="This backend is reserved by another job")
                    continue
                if entry["state"] != "run" or snapshot.get("complete") or snapshot.get("archived"):
                    status["state"] = "Complete" if snapshot.get("complete") else entry["state"]
                    continue
                if snapshot.get("errors") or (snapshot.get("failed_runs") and not _shared_runs_remain(root, job)):
                    status.update(state="Blocked", message="Resolve this job's reported errors before restarting")
                    continue
                failed = [request for request in own
                          if request["source"]["path"] in self._observed_pending
                          and request["source"]["sha256"] not in recovered
                          and (request.get("error_traceback") or (request.get("terminal_report") or {}).get("event", {}).get("kind") == "startup_failure")]
                if failed:
                    event = (failed[-1].get("terminal_report") or {}).get("event", {})
                    detail = (failed[-1].get("error_traceback") or event.get("traceback")
                              or event.get("error") or "Worker startup failed")
                    status.update(state="Blocked", message=_message(detail))
                    continue
                blocker = backend_blocker(job, backend, machine=self.machine, master_machine=entry["master_machine"])
                if blocker:
                    status.update(state="Unavailable", message=blocker)
                    continue
                request = None
                try:
                    with JobActionLock(root):
                        # Serialize with ordinary GUI starts/deletes and re-read
                        # admission under that lock, including the master's stop.
                        if read_control(root)["action"] != "run":
                            status.update(state="Waiting", message="The master paused or stopped this job")
                            continue
                        fresh = [request for request in read_launch_requests(root)
                                 if _same_machine(request["machine"], self.machine)
                                 and request["backend"] == backend and not request["released"]]
                        if fresh:
                            occupied[backend] = fresh
                            status.update(state="Launch requested", workers=sum(item["workers"] for item in fresh))
                            continue
                        actual_count = self._queued_worker_count(root, job, backend, count)
                        if not actual_count:
                            status.update(state="Waiting", message="No queued work for this backend")
                            continue
                        with self._guard:
                            if not self._accepting:
                                status["state"] = "Disabled"
                                continue
                        request = begin_launch_request(root, machine=self.machine,
                                                       backend=backend, workers=actual_count, kind="local")
                        self._observed_pending.add(request["source"]["path"])
                        occupied[backend] = [request]
                        options = dict(backend=backend, workers=actual_count, machine=self.machine,
                                       launch_directory=request["launch_directory"], once=True)
                        if self.stores is not None:
                            options["stores"] = self.stores
                        receipt = self.launcher(root, **options)
                        if receipt.get("started") is not True:
                            raise RuntimeError(receipt.get("error") or "The local worker did not start")
                        self._batches[request["source"]["path"]] = dict(
                            job_id=entry["job_id"], memory={}, configured_counts=dict(configured), eligible_backends=[],
                            job_facts={key: value for key, value in dict(job, pipeline=job.get("pipeline", "n1")).items()
                                       if key in {"pipeline", "precision", "commit", "backend", "execution_mode", "target_machine"}},
                            before={item["run"]: item.get("committed", 0) for item in snapshot.get("runs", [])},
                        )
                        status.update(state="Launch requested", workers=actual_count,
                                      message="Waiting for the original worker's progress")
                        new_batches.append((request["source"]["path"], row))
                except Exception as error:
                    if request is not None:
                        try:
                            record_launch_error(request, str(error))
                        except Exception as recording_error:
                            errors.append(_message(recording_error))
                    status.update(state="Blocked", message=_message(error))
        for key, row in new_batches:
            self._batches[key]["eligible_backends"] = [
                backend for backend, status in row["backends"].items()
                if status["state"] == "Launch requested" and status["workers"] > 0
            ]
        return rows, errors

    def _report(self, rows, errors, configured, accepting):
        state = "online" if accepting else "offline"
        self.folder.report(self.machine, self.session_id, role=self.role,
                           cpu_workers=configured["cpu"], cuda_workers=configured["cuda"],
                           jobs=rows, state=state)
        return dict(machine=self.machine, session_id=self.session_id, role=self.role,
                    state=state, jobs=rows, workers=self.folder.workers(), errors=errors,
                    samples=deepcopy(self._samples), completed_batches=deepcopy(self._samples),
                    tuning_note=getattr(self.tuner, "note", ""))

    def tick(self):
        with self._guard:
            configured = dict(cpu=self.cpu_workers, cuda=self.cuda_workers)
            accepting = self._accepting
        try:
            with self._admission():
                rows, errors = self._cycle(configured, accepting)
                self._last_jobs = deepcopy(rows)
        except Exception as error:
            rows, errors = deepcopy(self._last_jobs), [_message(error)]
        return self._report(rows, errors, configured, accepting)

    def leave(self):
        """Stop accepting batches; never interrupt an existing external worker."""
        with self._guard:
            self._accepting = False
            configured = dict(cpu=self.cpu_workers, cuda=self.cuda_workers)
        return self._report(deepcopy(self._last_jobs), [], configured, False)
