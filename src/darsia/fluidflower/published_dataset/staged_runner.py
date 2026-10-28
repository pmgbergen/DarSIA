"""Local orchestration of the original, separately admitted P1 and P2 workers.

The dataset is read-only. Original inputs are copied into explicit local roots;
each stage gets a new owned attempt. No fit, plan adoption or queue is created.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import uuid
import zipfile

from .layout import inside, raw_archive_name


def _validate_settings(settings):
    """Check staged launch settings before publishing any execution state."""
    # Keep the worker/thread/backend rules aligned with Reproduction.preflight;
    # Resume cannot repeat that folder check against its existing results.
    if type(settings.workers) is not int or settings.workers < 1:
        raise ValueError("Choose at least one worker")
    if (not isinstance(settings.backend, str) or settings.backend not in {"cpu", "cuda"}
            or type(settings.threads) is not int or settings.threads < 1):
        raise ValueError("Choose the CPU or CUDA backend and a positive thread count")
    if settings.override_commit is not False:
        raise ValueError("Staged reproduction requires each original commit; overrides are unavailable")
    if type(settings.remove_scratch) is not bool:
        raise ValueError("Choose whether to remove the unpacked RAW scratch files")
    if not isinstance(settings.stage_runtimes, str) or not settings.stage_runtimes.strip():
        raise ValueError("Choose the exact Python and clean checkout for every original stage")


def _bound(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(path=str(path.resolve()), sha256=digest.hexdigest(), size_bytes=path.stat().st_size)


def _read(root, bound):
    path = inside(root, bound["path"])
    actual = _bound(path)
    if any(actual[key] != bound[key] for key in ("sha256", "size_bytes")):
        raise ValueError("Staged metadata changed: " + bound["path"])
    return json.loads(path.read_bytes())


def _copy_input(source, target, bound):
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.exists()
    if existing:
        actual = _bound(target)
        if any(actual[key] != bound[key] for key in ("sha256", "size_bytes")):
            raise ValueError("Conflicting staged input destination")
    digest, size = hashlib.sha256(), 0
    with (nullcontext(None) if existing else target.open("xb")) as stream:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            size += len(chunk)
            if size > bound["size_bytes"]:
                raise ValueError("Staged input exceeds its original byte count")
            digest.update(chunk)
            if stream is not None:
                stream.write(chunk)
    if size != bound["size_bytes"] or digest.hexdigest() != bound["sha256"]:
        raise ValueError("Staged input bytes differ from the original catalogue")


def materialize_inputs(dataset, section, selected_runs, destination):
    """Copy every exact input locator, including duplicate content at two paths."""
    roots = {alias: inside(destination, alias) for alias in section["roots"]}
    roots.setdefault("results", inside(destination, "results"))
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)
    camera = {}
    for row in section["input_assets"]:
        target = inside(roots[row["root_alias"]], row["relative_path"])
        if row.get("camera"):
            camera[row["path"]] = (row, target)
        else:
            with inside(dataset, row["path"]).open("rb") as stream:
                _copy_input(stream, target, row)
    # Archives may repeat a shared reference image. The first verified copy is
    # sufficient; distinct locators are never collapsed by acquisition SHA.
    for run in selected_runs:
        with zipfile.ZipFile(inside(dataset, raw_archive_name(run))) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate RAW archive member")
            if set(names) != set(section["camera_members"][run]):
                raise ValueError("RAW archive members differ from the exact selected input catalogue")
            for name in names:
                if name not in camera:
                    raise ValueError("Undeclared RAW archive member")
                row, target = camera[name]
                with archive.open(name) as stream:
                    _copy_input(stream, target, row)
    return roots


def historical_aliases(dataset, section, roots):
    """Map original descriptors to their explicit transport, never by hash alone."""
    inputs = {row["path"]: row for row in section["input_assets"]}
    by_reference = {(row["root_alias"], row["relative_path"]): row for row in section["input_assets"]}
    aliases = []
    for row in section["transport"]:
        asset, logical = row["asset"], row["logical"]
        source = inputs.get(asset["path"])
        if row.get("input_reference") is not None:
            reference = row["input_reference"]
            source = by_reference.get((reference["root_alias"], reference["relative_path"]))
            if (source is None or (source["sha256"], source["size_bytes"]) !=
                    (reference["sha256"], reference["bytes"])):
                raise ValueError("Historical transport differs from its exact input consumer")
        physical = (inside(roots[source["root_alias"]], source["relative_path"])
                    if source is not None else inside(dataset, asset["path"]))
        aliases.append(dict(logical=dict(logical), physical=dict(asset, path=str(physical.resolve()))))
    return aliases


class StagedReplayRunner:
    route = "original-raw-phase1-saved-calibration-phase2"

    def __init__(self, *, inventory=None):
        self.results = self.check = self.scope = self.settings = None
        self.inventory = inventory
        self.evidence = None
        self.outputs = {}
        self.control = None
        self._control_lock = threading.Lock()
        self._active_workers = 0
        self._resume = threading.Event()
        self._resume.set()

    def _jobs(self, check):
        result = {}
        for key, stage in check.manifest["staged_reproduction"]["stages"].items():
            result[key] = dict(_read(check.root, stage["job"]), software=stage["software"])
        return result

    def _require_memory(self, settings, check, scope, *, completed=None):
        """Admit both the still-active pool and the proposed next batch."""
        if check is None or scope is None:
            # Resume may precede setup. run() checks its effective settings
            # with the real context before publishing the first control file.
            return
        from .replay import memory_problems
        from .resources import LAUNCHER_RESERVE_BYTES, machine_inventory

        inventory = self.inventory if self.inventory is not None else machine_inventory()
        # outputs contains only runs whose original P2 process has completed.
        # Iterate the fixed selection, not the concurrently populated mapping.
        completed = self.outputs if completed is None else completed
        pending = sum(run not in completed for run in scope.runs)
        # Changing settings cannot resize a pool or retire its queued futures.
        # Keep its full allowance until shutdown has joined every worker.
        workers = max(self._active_workers, min(settings.workers, pending))
        problems = memory_problems(check.manifest, workers, inventory,
                                   reserve_bytes=LAUNCHER_RESERVE_BYTES)
        if problems:
            raise ValueError("\n".join(problems))

    def preflight(self, results, *, check, scope, settings):
        from .staged_package import validate_staged
        from .staged_runtime import load_runtimes
        try:
            validate_staged(check.manifest)
            _validate_settings(settings)
            if scope.frames:
                raise ValueError("Staged RAW reproduction requires the complete sealed frame selection")
            load_runtimes(settings.stage_runtimes, self._jobs(check))
            # Include every input copy plus the existing output estimate. This
            # is a lower bound, so an unknown worker peak is never invented.
            needed = sum(row["size_bytes"] for row in check.manifest["staged_reproduction"]["input_assets"])
            parent = Path(results).resolve()
            while not parent.exists():
                parent = parent.parent
            if shutil.disk_usage(parent).free <= needed:
                raise ValueError("Insufficient disk for the exact original staged inputs")
        except (OSError, ValueError, KeyError, TypeError) as error:
            return ["Staged RAW reproduction: " + str(error)]
        with self._control_lock:
            if self.control is None:
                # Start preflights before enabling Pause/Resume. Retain that
                # context so an early Resume is checked before run() begins.
                self.check, self.scope = check, scope
        return []

    def _stage(self, stage_id, execution, receipt):
        from .staged_runtime import launch_stage
        process = launch_stage(self.runtimes[stage_id], self.jobs[stage_id], execution,
            dataset_root=self.check.root, receipt_path=receipt, log_path=receipt.with_suffix(".log"))
        while process.poll() is None:
            time.sleep(.2)
        if not receipt.is_file():
            raise ValueError("The original stage ended without its runtime receipt")
        bound = _bound(receipt)
        value = json.loads(receipt.read_bytes())
        if value.get("state") == "paused":
            return value, bound
        if process.returncode != 0 or value.get("state") != "done":
            raise ValueError("Original stage failed: " + str(value.get("error") or stage_id))
        output = Path(value["output"]).resolve(strict=True)
        if not output.is_relative_to(self.roots["results"].resolve()):
            raise ValueError("Stage output escaped the owned local results root")
        return value, bound

    def _run_one(self, run, *, workers):
        from .replay import _write_json
        settings = self.settings
        pair = self.check.manifest["staged_reproduction"]["runs"][run]
        attempt = uuid.uuid4().hex
        folder = self.results / "staged-replay" / run / attempt
        folder.mkdir(parents=True, exist_ok=False)
        executions = {}
        for label in ("phase1", "phase2"):
            stage_id = pair[label]
            stage = self.check.manifest["staged_reproduction"]["stages"][stage_id]
            task = stage["task"]
            executions[label] = dict(task=dict(task, path=str(inside(self.check.root, task["path"]).resolve())),
                stores=str(self.stores), series_id=self.check.manifest["identity"]["series_id"],
                job_id="reproduction-" + uuid.uuid4().hex[:12], attempt_id=uuid.uuid4().hex,
                backend=settings.backend, threads=settings.threads,
                control=str(self.control))
        p1, p2 = pair["phase1"], pair["phase2"]
        helper = _bound(Path(__file__).with_name("staged_replay.py"))
        executions["phase1"]["coupling"] = dict(helper=helper, phase2=dict(
            runtime=self.runtimes[p2], expected_job=self.jobs[p2], execution=executions["phase2"]),
            original_aliases=historical_aliases(self.check.root,
                self.check.manifest["staged_reproduction"], self.roots))
        first, first_bound = self._stage(p1, executions["phase1"], folder / "phase1.json")
        if first["state"] == "paused":
            return dict(state="paused", run=run)
        if not first.get("coupling"):
            raise ValueError("The original P1 worker returned no verified coupling")
        executions["phase2"]["coupled_inputs"] = first_bound
        second, second_bound = self._stage(p2, executions["phase2"], folder / "phase2.json")
        if second["state"] == "paused":
            return dict(state="paused", run=run)
        self.outputs[run] = Path(second["output"])
        receipt = dict(run=run, state="done", receipt_paths_relative_to="results",
            settings=dict(workers=workers, threads=settings.threads, backend=settings.backend,
                          precision="float32"),
            phase1=dict(first_bound, path=Path(first_bound["path"]).relative_to(self.results).as_posix()),
            phase2=dict(second_bound, path=Path(second_bound["path"]).relative_to(self.results).as_posix()),
            ordered_case_ids=self.check.manifest["selection"]["runs"][run]["ordered_case_ids"],
            scientific_qualification=False, production=False)
        _write_json(folder / "coupled-run.json", receipt)
        return receipt

    def run(self, check, results, scope, settings, *, progress=None):
        from .replay import _write_json, store_map_text
        from .staged_package import readback_staged
        from .staged_runtime import load_runtimes, validate_runtime
        with self._control_lock:
            # The window may publish Pause before this worker thread starts.
            # Preserve that state and establish its file before slow setup.
            if self.control is None and self.settings is not None:
                if self.settings.stage_runtimes != settings.stage_runtimes:
                    raise ValueError("Keep the admitted stage runtimes when resuming")
                settings = self.settings
            _validate_settings(settings)
            # A new run queues the whole selection again; prior-run outputs
            # cannot reduce either its initial or subsequent Resume demand.
            self._require_memory(settings, check, scope, completed=())
            self.outputs, self.evidence = {}, None
            self.check, self.results, self.scope, self.settings = check, Path(results).resolve(), scope, settings
            self.control = self.results / "staged-control.json"
            _write_json(self.control, dict(action="run" if self._resume.is_set() else "pause"))
        readback_staged(check.root, check.manifest)
        self.jobs = self._jobs(check)
        self.runtimes = load_runtimes(settings.stage_runtimes, self.jobs)
        runtime_receipts = {stage_id: validate_runtime(runtime, self.jobs[stage_id], dataset_root=check.root)
                            for stage_id, runtime in self.runtimes.items()}
        _write_json(self.results / "staged-runtime-check.json", runtime_receipts)
        self.roots = materialize_inputs(check.root, check.manifest["staged_reproduction"], scope.runs,
                                       self.results / "staged-inputs")
        self.stores = self.results / "staged-stores.toml"
        self.stores.write_text(store_map_text(check.manifest["identity"]["series_id"], self.roots), encoding="utf-8")
        states, receipts = {run: "queued" for run in scope.runs}, {}
        while any(state != "done" for state in states.values()):
            pending = [run for run, state in states.items() if state != "done"]
            with self._control_lock:
                # Reserve the actual pool capacity atomically with selection.
                # Resume must see it even while construction is in progress.
                workers = min(self.settings.workers, len(pending))
                self._active_workers = workers
            try:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(self._run_one, run, workers=workers): run for run in pending}
                    for future in as_completed(futures):
                        run = futures[future]
                        receipts[run] = future.result()
                        states[run] = receipts[run]["state"]
                        if progress:
                            progress(dict(states))
            finally:
                # The context has joined every submitted/queued future before
                # releasing this allowance, including result/progress errors.
                # Never hold the control lock while waiting for retirement.
                with self._control_lock:
                    self._active_workers = 0
            if all(state == "done" for state in states.values()):
                break
            while not self._resume.wait(1):
                if progress:
                    progress(dict(states))
        self.evidence = dict(format="darsia-staged-reproduction-result-v1", complete=True,
            route=self.route, runs=receipts, scientific_qualification=False, production=False)
        _write_json(self.results / "staged-reproduction.json", self.evidence)
        if self.settings.remove_scratch:
            owned = (self.results / "staged-inputs").resolve()
            for item in check.manifest["staged_reproduction"]["input_assets"]:
                if item["camera"]:
                    path = inside(self.roots[item["root_alias"]], item["relative_path"])
                    if not path.resolve().is_relative_to(owned):
                        raise ValueError("RAW cleanup escaped the owned reproduction scratch folder")
                    path.unlink(missing_ok=True)
        return states

    def pause(self):
        from .replay import _write_json
        with self._control_lock:
            self._resume.clear()
            if self.control is not None:
                _write_json(self.control, dict(action="pause"))

    def resume(self, settings):
        from .replay import _write_json
        with self._control_lock:
            _validate_settings(settings)
            if self.settings is not None and settings.stage_runtimes != self.settings.stage_runtimes:
                raise ValueError("Keep the admitted stage runtimes when resuming")
            self._require_memory(settings, self.check, self.scope)
            self.settings = settings
            if self.control is not None:
                _write_json(self.control, dict(action="run"))
            self._resume.set()

    def reproduced(self, run, case):
        from .comparison import load_layer
        output = self.outputs[run]
        result = json.loads((output / "frames" / case / "result.json").read_bytes())
        layers = [path for path in output.rglob("storage_stored_frame.npz") if path.parent.name == case]
        if len(layers) != 1:
            raise ValueError("The original P2 output has no unique saved comparison layer")
        return ({"total": result.get("total_mass_g"), "aqueous": result.get("aqueous_mass_g"),
                 "gas": result.get("gas_mass_g")}, load_layer(layers[0]))
