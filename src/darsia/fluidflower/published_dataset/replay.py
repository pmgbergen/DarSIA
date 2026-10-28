"""Reproduce a dataset's runs on this machine: replay their prepared tasks with the production code.

The launcher's route needs no job queue and no job folder:

* the dataset carries each run's prepared task (``plan/tasks/<run>.json``), the
  production job's own input catalogue with every input by root, path and
  SHA-256;
* a store map in the per-user DarSIA folder binds the task's roots on this
  machine: the dataset folder for the series files, the unpacked raw images for
  each camera root, and ``outputs/`` in the results folder for the outputs;
* each run is replayed as the job layer's workers run a claimed task:
  ``resolve_catalog`` and ``verify_inputs`` admit every input by root, path and
  SHA-256, ``RunIOBinding`` binds the run's inputs and output, and the
  production ``run_series`` (or ``run_lattice_series``) computes the run into
  a fresh attempt folder. Nothing here computes science;
* no queue claim is made, no capacity or cache ledger is written, and no
  qualified profile is read;
* runs are spread over worker processes in order, and each worker replays its
  runs one after another;
* Pause stops each worker at its next frame boundary. Resume replays a paused
  run again from its first frame in a fresh attempt folder; earlier attempt
  folders are kept. The quick check stops each run after its first frames;
* every run record says it is a reproduction run, not qualified and not
  production, with the code commit, the environment identity and the
  host-neutral machine id ``reproduction``.

The results folder holds ``replay/`` (control, run records, worker settings and
logs), ``outputs/`` and ``scratch/``; the dataset is only read.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import traceback
import uuid

from .archives import extract_members
from .job_reader import LAYER_FILE
from .layout import REPLAY_PIPELINES, inside, raw_archive_name

MACHINE_ID = "reproduction"
ROUTE = "library replay"
RECORD_FORMAT = "darsia-reproduction-run-v1"
WORKER_FORMAT = "darsia-reproduction-worker-v1"
CONTROL_FORMAT = "darsia-reproduction-control-v1"
REPLAY, OUTPUTS, SCRATCH = "replay", "outputs", "scratch"
FINAL_STATES = ("done", "failed")


class PausedRun(Exception):
    """A worker stopped at a frame boundary because Pause was published."""


class EnoughFrames(Exception):
    """The quick check has its frames; the run stops after them."""


def _now() -> str:
    """The current UTC time as ISO text."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, value) -> None:
    """Write a small JSON document atomically (a temporary file, then a replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def raw_roots(root, manifest: dict, scratch: Path) -> dict[str, Path]:
    """Where each camera root of the dataset lies here: unpacked in the dataset, or in scratch."""
    roots = {}
    for alias, relative in (manifest.get("roots") or {}).items():
        if alias == "series":
            continue
        unpacked = inside(root, relative)
        roots[alias] = unpacked if unpacked.is_dir() else Path(scratch) / alias
    return roots


def _raw_items(manifest: dict, run: str) -> list[dict]:
    """A run's camera files: its frames and its reference images."""
    row = next(row for row in manifest["runs"] if row["run"] == run)
    return [frame["raw"] for frame in row["frames"]] + list(row.get("references") or ())


def unpack_run(root, manifest: dict, run: str, roots: dict[str, Path]) -> list[Path]:
    """Unpack one run's raw images that are not yet here into scratch, checking every SHA-256.

    Images already unpacked in the dataset (or left by an earlier attempt) are
    read where they are. Returns the files this call unpacked.
    """
    wanted = {}
    for item in _raw_items(manifest, run):
        alias, _separator, relative = item["path"].partition("/")
        if not roots[alias].joinpath(*relative.split("/")).is_file():
            wanted[item["path"]] = item["sha256"]
    if not wanted:
        return []
    alias = next(iter(wanted)).partition("/")[0]
    return extract_members(Path(root) / raw_archive_name(run), roots[alias].parent, names=sorted(wanted),
                           expected=wanted)


def remove_unpacked(manifest: dict, run: str, roots: dict[str, Path], scratch: Path) -> None:
    """Remove a run's raw images from scratch (never from the dataset)."""
    scratch = Path(scratch).resolve()
    for item in _raw_items(manifest, run):
        alias, _separator, relative = item["path"].partition("/")
        target = roots[alias].joinpath(*relative.split("/"))
        if target.resolve().is_relative_to(scratch):
            target.unlink(missing_ok=True)


def store_map_path(replay_id: str, *, environment=None) -> Path:
    """The local store map of one reproduction, beside the user's own store map."""
    from ..store_map import default_store_map_path

    return default_store_map_path(environment=environment).parent / "reproductions" / replay_id / "stores.toml"


def store_map_text(series_id: str, roots: dict[str, Path]) -> str:
    """A store map that binds one series' roots on this machine for the machine id ``reproduction``."""
    from ..series_paths import series_key

    key = series_key(series_id)
    lines = ["[machine]", f"id = {json.dumps(MACHINE_ID)}", "", f"[stores.{json.dumps(key)}]"]
    lines += [f"{alias} = {json.dumps(str(path))}" for alias, path in roots.items()]
    text = "\n".join(lines) + "\n"
    if tomllib.loads(text)["stores"][key] != {alias: str(path) for alias, path in roots.items()}:
        raise ValueError("The store map does not survive a TOML round trip")
    return text


def published_tasks(check) -> dict[str, dict]:
    """Each run's prepared task from the dataset, checked against the manifest's SHA-256."""
    tasks = {}
    for run, row in ((check.manifest.get("plan") or {}).get("tasks") or {}).items():
        raw = inside(check.root, row["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"] or len(raw) != row["size_bytes"]:
            raise ValueError(f"The prepared task of run {run} differs from the manifest")
        task = json.loads(raw)
        if task.get("run") != run or task.get("pipeline") not in REPLAY_PIPELINES:
            raise ValueError(f"The prepared task of run {run} cannot be replayed")
        tasks[run] = task
    return tasks


def replay_task(task: dict, *, attempt_id: str, job_id: str, threads: int | None = None,
                commit: str | None = None) -> dict:
    """The task this replay runs: the published one with a fresh attempt and the user's settings.

    Only the attempt and job identity, the threads per worker and (when the
    user overrides it) the commit change; every input stays the published one.
    """
    value = dict(task, attempt_id=attempt_id, job_id=job_id)
    if threads is not None:
        value["threads"] = int(threads)
    if commit is not None:
        value["commit"] = commit
    return value


def environment_identity() -> dict:
    """This installation's environment identity (every distribution's version and recorded hashes)."""
    from .environment import distribution_fingerprint

    fingerprint = distribution_fingerprint(hash_files=False)
    return {key: fingerprint[key] for key in ("format", "python", "file_digests", "sha256")}


def science(pipeline: str):
    """The production function that computes one run of ``pipeline``."""
    if pipeline == "native-phase-n1":
        from ..pipeline.runner import run_series

        return run_series
    if pipeline == "lattice-phase-n1":
        from ..pipeline.lattice_runner import run_lattice_series

        return run_lattice_series
    raise ValueError(f"The launcher cannot replay the pipeline {pipeline!r}")


def read_control(results) -> str:
    """The published action for the workers: ``run`` or ``pause``."""
    path = Path(results) / REPLAY / "control.json"
    if not path.is_file():
        return "run"
    value = json.loads(path.read_text(encoding="utf-8"))
    return "pause" if value.get("action") == "pause" else "run"


def publish_control(results, action: str) -> None:
    """Publish Pause or Run for this reproduction's workers."""
    if action not in {"run", "pause"}:
        raise ValueError("The action must be run or pause")
    _write_json(Path(results) / REPLAY / "control.json",
                {"format": CONTROL_FORMAT, "action": action, "published_utc": _now()})


def record_path(results, run: str) -> Path:
    """Where a run's record lies in the results folder."""
    from ..pipeline.job_queue import name

    return Path(results) / REPLAY / f"{name(run)}.json"


def read_record(results, run: str) -> dict | None:
    """A run's record (its attempts), or None before its first attempt."""
    path = record_path(results, run)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def run_state(results, run: str) -> str:
    """A run's state: queued, running, paused, done or failed (its latest attempt's)."""
    record = read_record(results, run)
    if not record or not record.get("attempts"):
        return "queued"
    return record["attempts"][-1]["state"]


def _new_record(spec: dict, run: str) -> dict:
    """A run record before its first attempt: what kind of run this is and what produced it."""
    return {"format": RECORD_FORMAT, "run": run, "route": ROUTE, "reproduction": True, "qualified": False,
            "production": False, "machine": MACHINE_ID,
            "code": {"commit": spec["commit"], "clean_checkout": spec.get("clean")},
            "environment": spec["environment"], "task": spec["tasks"][run], "attempts": []}


def replay_run(spec: dict, run: str, *, execute=None, checkout=None) -> dict:
    """Replay one run of a worker's settings and record its attempt; return the attempt.

    A new attempt folder is used every time. The quick check (``spec['frames']``)
    stops the run after its frames; Pause stops it at the next frame boundary.
    """
    from ..pipeline.job_queue import LocalOutputBinding
    from ..pipeline.run_io import RunIOBinding
    from ..pipeline.store_runtime import local_attempt_output, resolve_catalog, verify_inputs
    from ..series_paths import darsia_source_checkout
    from ..store_map import read_store_map
    from .manifest import load_manifest

    results = Path(spec["results"])
    record = read_record(results, run) or _new_record(spec, run)
    attempt = {"attempt_id": uuid.uuid4().hex, "state": "running", "started_utc": _now(), "frames_completed": 0}
    record["attempts"].append(attempt)
    _write_json(record_path(results, run), record)
    try:
        manifest = load_manifest(spec["dataset"])
        roots = raw_roots(spec["dataset"], manifest, results / SCRATCH)
        unpack_run(spec["dataset"], manifest, run, roots)
        raw = inside(spec["dataset"], spec["tasks"][run]["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec["tasks"][run]["sha256"]:
            raise ValueError(f"The prepared task of run {run} differs from the manifest")
        task = replay_task(json.loads(raw), attempt_id=attempt["attempt_id"], job_id=spec["job_id"],
                           threads=spec.get("threads"), commit=spec["commit"])
        checkout = Path(checkout) if checkout is not None else darsia_source_checkout()
        if checkout is None:
            raise ValueError("The replay needs a Git clone of DarSIA at the commit, not an installed copy")
        stores = read_store_map(spec["stores"], forbidden_roots=(checkout,))
        rows = resolve_catalog(task, stores, spec["series_id"])
        output, evidence = local_attempt_output(stores, spec["series_id"], task["job_id"], task["run"],
                                                task["attempt_id"])
        results_root = Path(stores.roots(spec["series_id"])["results"])
        attempt["output"] = output.resolve().relative_to(results.resolve()).as_posix()
        local = LocalOutputBinding(stores.machine_id, results_root, output, evidence)
        verified = verify_inputs(task, rows)
        binding = RunIOBinding.replay(task, verified, local, repository=checkout, admitted_commit=task["commit"])
        science_task = dict(task)
        if spec.get("environment_lock"):
            science_task["environment_lock"] = spec["environment_lock"]
        limit = (spec.get("frames") or {}).get(run)

        def progress(event):
            """Count committed frames; stop for Pause or after the quick check's frames."""
            if not isinstance(event, dict):
                raise TypeError("Progress must be an event dictionary")
            if event.get("state") == "frame_completed":
                attempt["frames_completed"] += 1
                _write_json(record_path(results, run), record)
                if limit and attempt["frames_completed"] >= limit:
                    raise EnoughFrames()
            if event.get("state") in {"frame_started", "frame_completed"} and read_control(results) == "pause":
                raise PausedRun()

        function = execute or science(task["pipeline"])
        precision = task["precision"]
        try:
            with binding.activate():
                result = function(science_task, output, spec["backend"], precision, progress)
        except EnoughFrames:
            result = {"precision": precision, "backend": spec["backend"], "frames": attempt["frames_completed"],
                      "quick_check": True}
        if result.get("precision") != precision or result.get("backend") != spec["backend"]:
            raise ValueError("The run's backend or precision differs from the replayed task")
        attempt.update(state="done", finished_utc=_now(), frames=result.get("frames", attempt["frames_completed"]),
                       quick_check=bool(result.get("quick_check")), wall_seconds=result.get("wall_seconds"))
        if spec.get("remove_scratch"):
            remove_unpacked(manifest, run, roots, results / SCRATCH)
    except PausedRun:
        attempt.update(state="paused", finished_utc=_now())
    except Exception as error:
        attempt.update(state="failed", finished_utc=_now(), error=f"{type(error).__name__}: {error}",
                       traceback=traceback.format_exc())
    _write_json(record_path(results, run), record)
    return attempt


def run_worker(spec_path, *, execute=None, checkout=None) -> int:
    """Replay a worker's runs one after another; stop when Pause is published. Return 0, or 1 on a failure."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if spec.get("format") != WORKER_FORMAT:
        raise ValueError("Not a reproduction worker's settings")
    failed = False
    for run in spec["runs"]:
        if read_control(spec["results"]) == "pause":
            break
        if run_state(spec["results"], run) in FINAL_STATES:
            continue
        attempt = replay_run(spec, run, execute=execute, checkout=checkout)
        failed = failed or attempt["state"] == "failed"
        if attempt["state"] == "paused":
            break
    return 1 if failed else 0


def main(argv=None) -> int:
    """``python -m darsia.fluidflower.published_dataset.replay --worker <settings.json>``."""
    import argparse

    parser = argparse.ArgumentParser(description="Replay a reproduction worker's runs")
    parser.add_argument("--worker", type=Path, required=True, help="The worker's settings (JSON)")
    arguments = parser.parse_args(argv)
    return run_worker(arguments.worker)


def checkout_problems(installed: dict, commit: str) -> list[str]:
    """Why this installation cannot replay ``commit``: no Git clone, another commit, local changes."""
    if not installed.get("commit"):
        return ["The replay needs a Git clone of DarSIA at the published commit (" + installed.get(
            "source", "no Git checkout") + ")"]
    if installed["commit"] != commit:
        return [f"The DarSIA checkout is at {installed['commit']}, not at {commit}"]
    if not installed.get("clean"):
        return ["The DarSIA checkout has local changes; the replay needs a clean clone at the commit"]
    return []


def memory_problems(manifest: dict, workers: int, inventory: dict, *, reserve_bytes: int) -> list[str]:
    """Refuse when the workers' measured memory does not fit the memory available now."""
    peak = (manifest.get("machine") or {}).get("peak_memory_bytes")
    available = ((inventory or {}).get("memory") or {}).get("available_bytes")
    if peak is None or available is None or workers < 1:
        return []
    needed = int(peak) * workers + reserve_bytes
    if needed <= available:
        return []
    gib = 1024 ** 3
    return [f"Memory: {workers} worker(s) × {int(peak) / gib:.1f} GiB measured on the reference machine "
            f"+ {reserve_bytes / gib:g} GiB reserve = {needed / gib:.1f} GiB, but {available / gib:.1f} GiB "
            "is available now. Choose fewer workers or close other programs."]


def disk_problems(check, scope, tasks: dict, results, *, scratch_bytes: int) -> list[str]:
    """Refuse when the job layer's output estimate and the unpacked raw images do not fit the free disk."""
    from ..storage_profile import estimate_prepared_job

    profile = (check.manifest.get("plan") or {}).get("storage_profile", "full")
    runs = [run for run in scope.runs if run in tasks]
    configs = []
    for run in runs:
        reference = tasks[run]["configuration"]
        if reference.get("root_alias") != "series":
            return []  # A configuration outside the series files: no estimate to check.
        configs.append(json.loads(inside(check.root, reference["relative_path"]).read_bytes()))
    estimate = estimate_prepared_job(profile, [tasks[run] for run in runs], configs)
    if not estimate.get("complete"):
        return []
    needed = 0
    for row in estimate["runs"]:
        share = 1.0
        if scope.frames.get(row["run"]) and row.get("frames"):
            share = min(1.0, scope.frames[row["run"]] / row["frames"])
        needed += int(row["needed_bytes"] * share)
    needed += scratch_bytes
    folder = Path(results)
    while not folder.exists() and folder != folder.parent:
        folder = folder.parent
    free = shutil.disk_usage(folder).free
    if needed <= free:
        return []
    return [f"Disk: the job layer estimates {needed / 1e9:.1f} GB (outputs and unpacked raw images), "
            f"but {free / 1e9:.1f} GB is free in the results folder's drive"]


def scratch_bytes(check, scope, workers: int) -> int:
    """Scratch for raw images: each worker unpacks one run at a time, so the largest archives, one per worker."""
    sizes = sorted((int((row.get("archive") or {}).get("size_bytes") or 0) for row in check.manifest["runs"]
                    if row["run"] in scope.runs and row["frames"]
                    and not inside(check.root, row["frames"][0]["raw"]["path"]).is_file()), reverse=True)
    return sum(sizes[:max(1, workers)])


def _launch_worker(spec_path: Path, log_path: Path):
    """Start one worker process with this Python; return its process handle."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    with log_path.open("ab") as log:
        return subprocess.Popen([sys.executable, "-m", "darsia.fluidflower.published_dataset.replay",
                                 "--worker", str(spec_path)], stdout=log, stderr=subprocess.STDOUT,
                                cwd=str(spec_path.parent), creationflags=flags)


class ReplayRunner:
    """The launcher's runner: replay the dataset's prepared tasks on this machine."""

    route = ROUTE

    def __init__(self, *, launch=None, poll_seconds: float = 5.0, sleep=time.sleep, installed=None,
                 inventory=None, environment=None):
        """``launch`` starts one worker (default: a process with this Python); the rest replace the machine in tests."""
        self.launch = launch or _launch_worker
        self.poll_seconds, self.sleep = poll_seconds, sleep
        self.installed, self.inventory, self.environment = installed, inventory, environment
        self.results = self.check = self.scope = None
        self.spec = None
        self.workers = []
        self.lock = threading.Lock()
        self.staged_runner = None

    @property
    def staged_evidence(self):
        return getattr(self.staged_runner, "evidence", None)

    def _staged(self):
        if self.staged_runner is None:
            from .staged_runner import StagedReplayRunner
            self.staged_runner = StagedReplayRunner(inventory=self.inventory)
        return self.staged_runner

    def _installed(self) -> dict:
        """The installed DarSIA commit (once)."""
        if self.installed is None:
            from .dataset_check import installed_commit

            self.installed = installed_commit()
        return self.installed

    def _commit(self, check, settings) -> str:
        """The commit this replay runs: the published one, or the installed one when overridden."""
        if settings is not None and settings.override_commit and self._installed().get("commit"):
            return self._installed()["commit"]
        return check.manifest["code"]["published_commit"]

    def preflight(self, results, *, check=None, scope=None, settings=None) -> list[str]:
        """Why this replay cannot start: tasks, checkout, memory and disk (empty when it can)."""
        if check is None or scope is None or settings is None:
            return []
        from .resources import LAUNCHER_RESERVE_BYTES, machine_inventory

        inventory = self.inventory if self.inventory is not None else machine_inventory()
        workers = min(int(settings.workers or 0), len(scope.runs))
        # This is the original recorded worker peak, not an inferred combined
        # P1/P2 peak. Missing measurements retain memory_problems' unknown case.
        problems = memory_problems(check.manifest, workers, inventory,
                                   reserve_bytes=LAUNCHER_RESERVE_BYTES)
        if check.manifest.get("staged_reproduction"):
            return problems + self._staged().preflight(results, check=check, scope=scope, settings=settings)
        try:
            tasks = published_tasks(check)
        except (OSError, ValueError, KeyError) as error:
            return [f"The dataset's prepared tasks: {error}"]
        missing = [run for run in scope.runs if run not in tasks]
        if missing:
            return [f"The dataset carries no prepared task for run(s) {', '.join(missing)}; "
                    "build the record again with this DarSIA version"]
        problems += checkout_problems(self._installed(), self._commit(check, settings))
        try:
            problems += disk_problems(check, scope, tasks, results,
                                      scratch_bytes=scratch_bytes(check, scope, max(1, workers)))
        except (OSError, ValueError, KeyError, TypeError) as error:
            problems.append(f"The job layer's output estimate failed: {error}")
        return problems

    def run(self, check, results, scope, settings, *, progress=None) -> dict:
        """Write the store map and the workers' settings, start the workers and wait for every run."""
        if check.manifest.get("staged_reproduction"):
            self.route = self._staged().route
            return self._staged().run(check, results, scope, settings, progress=progress)
        results = Path(results).resolve()
        self.results, self.check, self.scope = results, check, scope
        replay_id = f"reproduction-{uuid.uuid4().hex[:12]}"
        roots = raw_roots(check.root, check.manifest, results / SCRATCH)
        stores = store_map_path(replay_id)
        stores.parent.mkdir(parents=True, exist_ok=True)
        stores.write_text(store_map_text(check.manifest["identity"]["series_id"], {
            "series": check.root.resolve(), **{alias: path.resolve() for alias, path in roots.items()},
            "results": results / OUTPUTS}), encoding="utf-8")
        (results / OUTPUTS).mkdir(parents=True, exist_ok=True)
        plan = tomllib.loads(inside(check.root, check.manifest["plan"]["path"]).read_text(encoding="utf-8"))
        environment = self.environment if self.environment is not None else environment_identity()
        self.spec = {"format": WORKER_FORMAT, "dataset": str(check.root.resolve()), "results": str(results),
                     "stores": str(stores), "series_id": check.manifest["identity"]["series_id"],
                     "job_id": replay_id, "commit": self._commit(check, settings),
                     "clean": bool(self._installed().get("clean")), "threads": settings.threads,
                     "backend": settings.backend, "environment_lock": (plan.get("job") or {}).get("environment_lock"),
                     "environment": environment, "tasks": check.manifest["plan"]["tasks"],
                     "frames": dict(scope.frames), "remove_scratch": bool(settings.remove_scratch)}
        publish_control(results, "run")
        self.start(settings)
        while True:
            states = {run: run_state(results, run) for run in scope.runs}
            if progress is not None:
                progress(states)
            with self.lock:  # Resume publishes Run and starts its workers under the same lock.
                live = [worker for worker in self.workers if worker.poll() is None]
                paused = read_control(results) == "pause"
                if not live and not paused and not all(state in FINAL_STATES for state in states.values()):
                    self._fail_unfinished({run: run_state(results, run) for run in scope.runs})
                    continue
            if not live and all(state in FINAL_STATES for state in states.values()):
                break
            self.sleep(self.poll_seconds)
        failed = [run for run, state in states.items() if state == "failed"]
        if failed:
            raise RuntimeError(f"Runs failed: {', '.join(failed)}; see {REPLAY}/ in the results folder")
        return states

    def _fail_unfinished(self, states: dict) -> None:
        """Record a failed attempt for runs that no worker finished (a worker ended early)."""
        for run, state in states.items():
            if state in FINAL_STATES:
                continue
            record = read_record(self.results, run) or _new_record(self.spec, run)
            if state == "running":
                record["attempts"][-1].update(state="failed", finished_utc=_now(),
                                              error="The worker process ended before the run finished")
            else:
                record["attempts"].append({"attempt_id": None, "state": "failed", "started_utc": _now(),
                                           "error": "No worker replayed the run"})
            _write_json(record_path(self.results, run), record)

    def start(self, settings) -> None:
        """Start workers for the runs that are not done: runs spread over the workers in order."""
        with self.lock:
            self._start(settings)

    def _start(self, settings) -> None:
        """Start the workers (the caller holds the lock)."""
        runs = [run for run in self.scope.runs if run_state(self.results, run) not in FINAL_STATES]
        count = max(1, min(int(settings.workers or 1), len(runs)))
        for number in range(count):
            share = runs[number::count]
            if not share:
                continue
            path = self.results / REPLAY / "workers" / f"worker-{uuid.uuid4().hex[:8]}.json"
            _write_json(path, dict(self.spec, runs=share))
            self.workers.append(self.launch(path, self.results / REPLAY / "logs" / f"{path.stem}.log"))

    def pause(self) -> None:
        """Publish Pause: each worker stops at its next frame boundary."""
        if self.staged_runner is not None:
            return self.staged_runner.pause()
        publish_control(self.results, "pause")

    def resume(self, settings) -> None:
        """Publish Run and start workers again; a paused run is replayed from its first frame."""
        if self.staged_runner is not None:
            return self.staged_runner.resume(settings)
        with self.lock:
            publish_control(self.results, "run")
            self._start(settings)

    def _done_output(self, run: str) -> Path:
        """The attempt folder of a run's finished attempt."""
        record = read_record(self.results, run) or {}
        done = [attempt for attempt in record.get("attempts", ()) if attempt["state"] == "done"]
        if not done:
            raise ValueError(f"Run {run} has no finished attempt")
        return self.results.joinpath(*done[-1]["output"].split("/"))

    def reproduced(self, run: str, case: str) -> tuple[dict, dict]:
        """One reproduced frame: its masses and its stored layer's arrays."""
        if self.staged_runner is not None:
            return self.staged_runner.reproduced(run, case)
        from .comparison import load_layer

        output = self._done_output(run)
        result = json.loads((output / "frames" / case / "result.json").read_text(encoding="utf-8"))
        masses = {"total": result.get("total_mass_g"), "aqueous": result.get("aqueous_mass_g"),
                  "gas": result.get("gas_mass_g")}
        layers = [path for path in output.rglob(LAYER_FILE) if path.parent.name == case]
        return masses, (load_layer(layers[0]) if layers else {})


if __name__ == "__main__":
    raise SystemExit(main())
