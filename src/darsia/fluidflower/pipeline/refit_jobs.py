"""Create a refit job (pipeline ``n1-calibration-refit``) from finished phase-one runs.

The job has one task ``refit`` (fit on every fit run, then freeze) and, when
asked, one task ``loro-<run>`` per fit run (the same fit without that run, and
its predictions). Each task gets its own ``darsia-n1-refit-plan-v1`` plan: the
fit runs' per-frame statistics files, the baseline calibration and the rules.
The plans and an execution TOML that declares every task are written to a new
folder under the series root; the job and its tasks are then written as for
any alias job, with the same role catalogue, so the worker's replay verifies
every statistics file before the fit reads it.

Series values (run roles and rules) are arguments, read by the caller from the
series' own configuration. Nothing here computes a fit.

The API for a page that starts and follows a refit:

* :func:`job_identity` and :func:`create_refit_job` create the queued job;
* :func:`start_refit_job` starts workers exactly as the Jobs page's Start does;
* :func:`refit_status` reports every task's state and latest step;
* :func:`read_frozen_calibration` returns the published frozen calibration,
  verified by its SHA-256.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PureWindowsPath

STATISTICS_NAME = "native_statistics.json.gz"
ENVIRONMENT_LOCK = "requirements-worker.lock"
REFIT_FOLDER = ("Calibration", "refit")


def _descriptor(path, *, max_bytes=None):
    """Full path, SHA-256 and size of an existing file."""
    if max_bytes is None:
        data = Path(path).read_bytes()
    else:
        with Path(path).open("rb") as stream:
            data = stream.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError("Statistics file exceeds the metadata byte limit: " + str(path))
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def completed_attempt(job_root, run):
    """The attempt folder of a run's published completion in a finished job."""
    from .job_queue import name, read_json
    completion = read_json(Path(job_root) / "results" / name(run) / "completed.json")
    if completion.get("complete") is not True or completion.get("run") != run:
        raise ValueError("Run has no complete published result: " + run)
    manifest = completion["manifest"]
    if _descriptor(manifest["path"])["sha256"] != manifest["sha256"]:
        raise ValueError("Completed run manifest changed: " + run)
    return Path(manifest["path"]).parent


def statistics_entries(attempt, *, include_elapsed=False):
    """Every per-frame statistics file of one attempt, by frame index.

    Phase one keeps them under ``preparation/<case>``, the native-phase-n1
    route under ``frames/<case>``; an attempt uses exactly one of the two.
    """
    from .job_inputs import MAX_METADATA_BYTES
    from .refit_capacity import read_statistics_bound
    attempt = Path(attempt)
    found = [sorted(attempt.glob(folder + "/*/" + STATISTICS_NAME)) for folder in ("preparation", "frames")]
    paths = [paths for paths in found if paths]
    if len(paths) != 1:
        raise ValueError("Attempt needs per-frame statistics in exactly one layout: " + str(attempt))
    entries = []
    for path in paths[0]:
        statistics = _descriptor(path, max_bytes=MAX_METADATA_BYTES)
        row, _decoded_bytes = read_statistics_bound(path, statistics)
        if row.get("fit_candidate") is False:
            continue  # Kept by P1 for history, explicitly outside the original fit cohort.
        if row.get("selected_target_h") is None:
            raise ValueError("Statistics row lacks its selected target time: " + str(path))
        entries.append(dict(case_id=path.parent.name, frame_index=row["frame_index"],
                            selected_target_h=row["selected_target_h"], statistics=statistics))
        if include_elapsed:
            entries[-1]["elapsed_h"] = row["elapsed_h"]
    entries.sort(key=lambda entry: entry["frame_index"])
    if len({entry["frame_index"] for entry in entries}) != len(entries):
        raise ValueError("Duplicate frame indices in " + str(attempt))
    return entries


def build_plans(*, statistics, roles, rules, baseline_calibration, loro=True, variant_f=None, chemistry_model=None,
                calibration_scope="shared", image_selection=None):
    """One refit plan per task: 'refit' and, if asked, 'loro-<run>' for every fit run."""
    from .refit_runner import FOLD_PREFIX, GLOBAL_RUN, PLAN_FORMAT, validate_plan
    baseline = json.loads(Path(baseline_calibration["path"]).read_text(encoding="utf-8-sig"))
    if baseline.get("parameters") != rules["baseline"]:
        raise ValueError("The rules' baseline parameters differ from the baseline calibration")
    common = dict(format=PLAN_FORMAT, runs=deepcopy(roles), rules=deepcopy(rules),
                  statistics={run: deepcopy(statistics[run]) for run in roles["fit"]},
                  baseline=dict(calibration=dict(baseline_calibration)),
                  variant_f=dict(variant_f or {"enabled": False}),
                  colour_paths="as in the phase-one statistics")
    if image_selection is not None:
        common["image_selection"] = deepcopy(image_selection)
    if chemistry_model is not None:
        if chemistry_model != "titration":
            raise ValueError("This mass-calibration chemistry model is not available yet")
        common["chemistry_model"] = chemistry_model
    if calibration_scope not in {"shared", "individual"}:
        raise ValueError("Choose shared or individual calibration")
    if calibration_scope == "individual":
        from .refit_runner import INDIVIDUAL_PREFIX
        plans = {}
        for run in roles["fit"]:
            task = INDIVIDUAL_PREFIX + run
            plan = dict(deepcopy(common), run=task, fold=None, calibration_scope="individual",
                source_roles=deepcopy(roles), calibrated_run=run,
                runs=dict(fit=[run], heldout=[], descriptive=[], excluded=deepcopy(roles.get("excluded", []))),
                statistics={run: deepcopy(statistics[run])})
            validate_plan(plan)
            plans[task] = plan
        return plans
    names = [(GLOBAL_RUN, None)] + ([(FOLD_PREFIX + run, run) for run in roles["fit"]] if loro else [])
    plans = {}
    for run, fold in names:
        plan = dict(common, run=run, fold=fold)
        validate_plan(plan)
        plans[run] = plan
    return plans


def write_refit_inputs(folder, plans, *, job, job_id):
    """Write every plan and the execution TOML that declares them; return its path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    lines = ['format = "darsia-series-v1"', "adopted_for_execution = true", f'job_id = "{job_id}"', "",
             "[job]"]
    for key in ("commit", "calibration_sha256", "precision", "created_utc"):
        lines.append(f'{key} = "{job[key]}"')
    lines += ["", "[job.environment_lock]", f'path = "{job["environment_lock"]["path"]}"',
              f'sha256 = "{job["environment_lock"]["sha256"]}"']
    for run, plan in plans.items():
        path = folder / (run + "_plan.json")
        path.write_bytes(json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
        bound = _descriptor(path)
        lines += ["", "[[runs]]", f'run = "{run}"', "", "[runs.configuration]",
                  'path = "' + str(path).replace("\\", "\\\\") + '"', f'sha256 = "{bound["sha256"]}"',
                  f'size_bytes = {bound["size_bytes"]}', "", "[runs.resources]", 'backend = "cpu"', "threads = 1"]
    execution = folder / "execution.toml"
    execution.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return execution


def job_identity(repository, *, calibration_sha256, precision="float32"):
    """Commit, environment lock and time for a new job from a clean checkout.

    The worker starts a job only from a clean checkout at the job's commit, so
    the job takes the commit and tracked lock of the checkout that creates it.
    """
    import subprocess
    repository = Path(repository)
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).strip():
        raise ValueError("A mass calibration job is created from a clean checkout")
    lock = subprocess.check_output(["git", "-C", str(repository), "show", head + ":" + ENVIRONMENT_LOCK])
    return dict(commit=head, calibration_sha256=calibration_sha256, precision=precision,
                created_utc=datetime.now(timezone.utc).isoformat(),
                environment_lock=dict(path=ENVIRONMENT_LOCK, sha256=hashlib.sha256(lock).hexdigest()))


def create_refit_job(series_manifest, *, phase_job, roles, rules, baseline_calibration, job_id, jobs_root,
                     job, loro=True, chemistry_model=None):
    """Find the fit runs' statistics, write the plans and create the queued refit job.

    ``series_manifest`` is the bound series.toml; the plans go to
    ``Calibration/refit/<job id>`` by default, the job to ``jobs_root``.
    ``job`` comes from :func:`job_identity`. Returns the prepared job record.
    """
    from .job_queue import name
    name(job_id)
    statistics = {run: statistics_entries(completed_attempt(phase_job, run), include_elapsed=True)
                  for run in roles["fit"]}
    import tomllib
    from ..calibration_image_selection import calibration_image_count, calibration_statistics_selection
    document = tomllib.loads(Path(series_manifest["path"]).read_text(encoding="utf-8-sig"))
    statistics, rules, selection = calibration_statistics_selection(
        statistics, rules, calibration_image_count(document))
    plans = build_plans(statistics=statistics, roles=roles, rules=rules,
                        baseline_calibration=baseline_calibration, loro=loro, chemistry_model=chemistry_model,
                        image_selection=selection)
    from ..series_paths import series_folder_path
    folder = series_folder_path(Path(series_manifest["path"]).parent, "calibration_refit") / job_id
    execution = write_refit_inputs(folder, plans, job=job, job_id=job_id)
    return prepare_refit_job(execution, Path(jobs_root) / job_id, series_manifest=series_manifest)


def prepare_refit_job(execution, job_root, *, series_manifest, storage_profile=None):
    """Write job.json and enqueue one alias task per declared refit run.

    ``series_manifest`` is the bound series.toml descriptor. The tasks carry
    the same catalogue, series and manifest references as any alias task.
    """
    import tomllib
    from .. import storage_profile as storage
    from .alias_job import JOB_FORMAT, TASK_FORMAT, _manifest, shared_output_admission
    from .job_inputs import InputCatalog, validate_catalog
    from .job_queue import JobQueue, descriptor, name, validate_task, write_immutable
    from .refit_runner import PIPELINE
    profile = storage.new_job_profile(storage_profile)
    shared_output_admission(job_root)
    _, manifest, roots = _manifest(series_manifest)
    execution_bound = _descriptor(execution)
    value = tomllib.loads(Path(execution).read_text(encoding="utf-8"))
    job_id = name(value["job_id"])
    admission = InputCatalog(roots)
    series_reference = admission.add(execution_bound, "configuration", "execution_series")
    manifest_reference = admission.add(series_manifest, "configuration", "series_manifest")
    job = dict(value["job"])
    lock = job["environment_lock"]
    if PureWindowsPath(lock["path"]).is_absolute() or ".." in PureWindowsPath(lock["path"]).parts:
        raise ValueError("Environment lock must remain repository-relative")
    version = storage.FORMAT_VERSIONS[profile]
    tasks = []
    for row in value["runs"]:
        plan = json.loads(Path(row["configuration"]["path"]).read_text(encoding="utf-8"))
        if plan.get("run") != row["run"]:
            raise ValueError("Execution TOML and plan runs differ")
        catalog = InputCatalog(roots)
        catalog.add(execution_bound, "configuration", "run.execution_series")
        catalog.add(series_manifest, "configuration", "run.series_manifest")
        inputs = validate_catalog(catalog.collect(row["configuration"], plan))
        inputs["runtime_support"] = "darsia-role-io-runtime-v1"
        cases = [entry["case_id"] for run in sorted(plan["statistics"]) for entry in plan["statistics"][run]]
        task = dict(deepcopy(row["resources"]), pipeline=PIPELINE, run=row["run"],
                    commit=job["commit"], calibration_sha256=job["calibration_sha256"], precision=job["precision"],
                    job_id=job_id, format=TASK_FORMAT, format_version=version, legacy_paths=False,
                    execution_ready=True, execution_blocker="", configuration=inputs["configuration"],
                    series=series_reference, series_manifest=manifest_reference, input_catalog=inputs,
                    ordered_case_ids=cases, storage_profile=profile)
        validate_task(task)
        tasks.append(task)
    job = dict(job, expected_runs=[task["run"] for task in tasks], series=series_reference, format=JOB_FORMAT,
               format_version=version, legacy_paths=False, execution_ready=True, execution_blocker="",
               runtime_support="darsia-role-io-runtime-v1", series_manifest=manifest_reference,
               series_id=manifest["series_id"], input_reference_format="root-alias-relative-v1",
               storage_profile=profile, pipeline=PIPELINE, publish=[])
    root = Path(job_root)
    if (root / "job.json").exists():
        raise FileExistsError("A mass calibration job is created once: " + str(root))
    root.mkdir(parents=True, exist_ok=True)
    write_immutable(root / "job.json", job)
    queue = JobQueue(root)
    prepared = [dict(run=task["run"], task=descriptor(queue.enqueue(task))) for task in tasks]
    return dict(complete=True, job=descriptor(root / "job.json"), tasks=prepared, pipeline=PIPELINE,
                numerical_calls_started=False, created_utc=datetime.now(timezone.utc).isoformat())


def start_refit_job(job_root, *, machine, stores=None, workers=1, python=None, cwd=None):
    """Start CPU workers for a refit job exactly as the Jobs page's Start does.

    It needs the GUI's Qt process support and returns the launch record of
    ``jobs_panel.launch_local_worker``. The worker refuses a checkout other
    than the job's commit.
    """
    import sys
    import traceback
    from ..jobs_panel import launch_local_worker
    from ..job_launch_state import begin_launch_request, record_launch_error
    state = refit_start_state(job_root)
    if state["blocker"]:
        raise ValueError(state["blocker"])
    request = begin_launch_request(job_root, machine=machine, backend="cpu",
                                   workers=workers, kind="local")
    try:
        return launch_local_worker(job_root, backend="cpu", workers=workers, machine=machine,
                                   python=python or sys.executable, cwd=cwd, stores=stores,
                                   launch_directory=request["launch_directory"])
    except Exception as error:
        text = traceback.format_exc()
        try:
            record_launch_error(request, text)
        except Exception as receipt_error:
            error.add_note("The launch failure receipt could not be written: " + str(receipt_error))
        raise


def refit_start_state(job_root, *, status=None):
    """Reuse the Jobs control/launch reservations; unknown service state blocks Start.

    This is an operational view, not process liveness or scientific acceptance.
    The original worker still performs fresh capacity, source and code admission.
    """
    from ..job_control import read_control
    from ..job_launch_state import read_launch_requests
    status = refit_status(job_root) if status is None else status
    control = read_control(job_root)
    requests = read_launch_requests(job_root)
    tasks = status["tasks"]
    if control["action"] != "run":
        blocker = "The job requests pause or stop. Resume it on the Jobs page before Start."
    elif any(not row["released"] for row in requests):
        blocker = "A worker launch is already recorded. Follow it on the Jobs page; its retirement is not yet recorded."
    elif any(row["state"] == "running" for row in tasks.values()):
        blocker = "This mass calibration job already has a running task. Follow its progress; do not start it again."
    elif any(row["state"] == "failed" for row in tasks.values()):
        blocker = "This mass calibration job has a failed task. Read its failure before creating a new job."
    elif any(row.get("stop_required") is True for row in tasks.values()):
        blocker = "A mass calibration grid check requires a stop. Read the result before starting further work."
    elif not any(row["state"] == "waiting" for row in tasks.values()):
        blocker = "This mass calibration job has no waiting tasks."
    else:
        blocker = ""
    return dict(blocker=blocker, launch_requests=requests, control=control)


def _refit_job(job_root):
    """The job.json of a refit job; any other job is refused."""
    from .job_queue import read_json
    from .refit_runner import PIPELINE
    job = read_json(Path(job_root) / "job.json")
    if job.get("pipeline") != PIPELINE:
        raise ValueError("Not a mass calibration job: " + str(job_root))
    return job


def _refit_completion(root, job, run):
    """Read alias completion only through the original closed publication guard."""
    from .job_queue import read_json, _own_asset, descriptor
    if job.get("format") != "darsia-alias-job-v1":
        path = root / "results" / run / "completed.json"
        return (read_json(path), None) if path.exists() else (None, None)
    done_path = root / "done" / (run + ".json")
    if not done_path.exists():
        return None, None  # A copied candidate alone is not a closed alias result.
    from ..jobs_metadata import JobSnapshotReader
    from .common_calibration import refit_terminal
    reader = JobSnapshotReader().read
    done = reader(done_path)
    task = reader(root / "results" / run / "task.json")
    result, _machines, _lineage = refit_terminal(root, task, job, done, reader)
    return result, descriptor(_own_asset(root, done["products_index"]))


def refit_status(job_root):
    """Each task's state and latest progress step, and the refit's result once published.

    A task is 'waiting' while queued, 'running' once claimed, and 'completed'
    or 'failed' when published. The latest step comes from the task's own
    progress records, across attempts.
    """
    from .job_queue import name, read_json
    root = Path(job_root)
    job = _refit_job(root)
    failed = {}
    for path in sorted((root / "failed").glob("*.json")):
        record = read_json(path)
        failed[record.get("run")] = record
    tasks, completions = {}, {}
    for run in job["expected_runs"]:
        run = name(run)
        result, index = _refit_completion(root, job, run)
        records = [read_json(path) for path in sorted(root.glob("leases/" + name(run) + ".*/progress_*.json"))]
        latest = max(records, key=lambda record: record.get("written_unix_ns", 0))["event"] if records else None
        if result is not None:
            state = "completed"
        elif run in failed:
            state = "failed"
        elif (root / "queue" / (name(run) + ".json")).exists():
            state = "waiting"
        else:
            state = "running"
        completions[run] = (result, index)
        tasks[run] = dict(state=state, step=None if latest is None else latest.get("step"), latest_event=latest,
                          parameters=None if result is None else result.get("parameters"),
                          stop_required=None if result is None else result.get("stop_required"))
    refit = tasks.get("refit", {})
    frozen = origin = None
    if refit.get("state") == "completed":
        result, index = completions["refit"]
        origin = result.get("frozen_calibration")
        frozen = origin
        if origin is not None and index is not None:
            from .products_publication import published_metadata_descriptor
            shared = published_metadata_descriptor(index, "frozen_calibration.json")
            frozen = dict(path=shared["path"], sha256=shared["sha256"], size_bytes=shared["bytes"])
            if any(frozen[key] != origin[key] for key in ("sha256", "size_bytes")):
                raise ValueError("The shared frozen calibration differs from its original descriptor")
    return dict(job_root=str(root), tasks=tasks, complete=all(t["state"] == "completed" for t in tasks.values()),
                failed=sorted(run for run, task in tasks.items() if task["state"] == "failed"),
                parameters=refit.get("parameters"), stop_required=refit.get("stop_required"),
                frozen_calibration=frozen, frozen_calibration_origin=origin)


def read_frozen_calibration(job_root):
    """The published frozen calibration of a refit job and its descriptor, verified by SHA-256."""
    frozen = refit_status(job_root)["frozen_calibration"]
    if frozen is None:
        raise ValueError("No frozen calibration: mass calibration has not finished, was stopped by its grid check, or failed")
    data = Path(frozen["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != frozen["sha256"] or len(data) != frozen["size_bytes"]:
        raise ValueError("The frozen calibration changed after it was published")
    return json.loads(data), dict(frozen)
