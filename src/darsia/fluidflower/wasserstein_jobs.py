"""Explicit, inert preparation for genuine whole-run Wasserstein queue jobs.

Shared dispatch must be integrated before prepare_wasserstein_job can enqueue.
The existing queue validator is called before any job folder is written. No
worker is started, and no source plan is adopted or rewritten by this module.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import tomllib

from .wasserstein_job_protocol import (
    OPERATION_FORMAT, PIPELINE, comparisons, descriptor, operation_id,
    validate_operation, validate_wasserstein_claim,
)


def read_bound(bound):
    bound = descriptor(bound)
    from .pipeline.job_inputs import InputCatalog, _json_bytes
    raw = InputCatalog._read_metadata(bound["path"])
    if len(raw) != bound["size_bytes"] or hashlib.sha256(raw).hexdigest() != bound["sha256"]:
        raise ValueError("Saved Wasserstein metadata changed")
    return _json_bytes(raw)


def build_operations(project, plan_descriptor):
    """Revalidate the original saved plan, then return inert comparison documents.

The caller supplies the actual FluidFlowerProject and an existing ready plan;
there is no inference from current GUI selections or generated scientific plan.
"""
    from .wasserstein import validate_wasserstein_plan
    bound = descriptor(plan_descriptor)
    plan = read_bound(bound)
    rows = comparisons(plan)
    report = validate_wasserstein_plan(project, plan["configuration"])
    if (report.get("execution_ready") is not True
            or report.get("task_contract") != plan["task_contract"]):
        raise ValueError("Current original Wasserstein validation differs from the saved plan")
    return [dict(format=OPERATION_FORMAT, run=operation_id(row["order"]),
                 plan=deepcopy(bound), comparison=deepcopy(row)) for row in rows]


def _identity(*, job_id, commit, environment_lock):
    from .pipeline.job_queue import name
    if name(job_id) != job_id or re.fullmatch(r"[0-9a-f]{40}", str(commit)) is None:
        raise ValueError("Explicit canonical job ID and full source commit are required")
    lock = deepcopy(environment_lock)
    if not isinstance(lock, dict) or set(lock) != {"path", "sha256"}:
        raise ValueError("The actual repository environment lock is required")
    path = PureWindowsPath(lock["path"])
    if (not lock["path"] or path.drive or path.root or ".." in path.parts
            or re.fullmatch(r"[0-9a-f]{64}", str(lock["sha256"])) is None):
        raise ValueError("Environment lock must have its repository-relative path and SHA")
    return dict(commit=commit, environment_lock=lock, precision="float64", backend="cpu",
                created_utc=datetime.now(timezone.utc).isoformat())


def write_wasserstein_inputs(folder, operations, *, job_id, commit, environment_lock, actor, reason):
    """Save an explicitly requested job's metadata in one fresh folder; no queue.

Requires build_operations output. The record describes the human creation
action; it never marks an existing scientific source as adopted.
"""
    identity = _identity(job_id=job_id, commit=commit, environment_lock=environment_lock)
    if not isinstance(actor, str) or not actor.strip() or not isinstance(reason, str) or not reason.strip():
        raise ValueError("Job creation needs a human name and reason")
    if not operations:
        raise ValueError("The saved plan has no comparison operations")
    plan = read_bound(operations[0]["plan"])
    expected = comparisons(plan)
    if len(operations) != len(expected):
        raise ValueError("Job inputs must cover the whole saved plan exactly once")
    for index, config in enumerate(operations):
        validate_operation(config, plan)
        if config["comparison"] != expected[index] or config["plan"] != operations[0]["plan"]:
            raise ValueError("Job input operations are not in the original frozen order")
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    from .pipeline.job_queue import descriptor as file_descriptor, write_immutable
    quote = lambda value: json.dumps(str(value), ensure_ascii=False)
    lines = ['format = "darsia-series-v1"', 'execution_kind = "wasserstein-comparison"',
             "job_id = " + quote(job_id), "created_by = " + quote(actor.strip()),
             "creation_reason = " + quote(reason.strip()), "", "[job]"]
    for key in ("commit", "precision", "backend", "created_utc"):
        lines.append(key + " = " + quote(identity[key]))
    lines += ["", "[job.environment_lock]", "path = " + quote(identity["environment_lock"]["path"]),
              "sha256 = " + quote(identity["environment_lock"]["sha256"])]
    for config in operations:
        path = folder / (config["run"] + ".json")
        write_immutable(path, config)
        bound = file_descriptor(path)
        lines += ["", "[[runs]]", "run = " + quote(config["run"]), "", "[runs.configuration]",
                  "path = " + quote(bound["path"]), "sha256 = " + quote(bound["sha256"]),
                  "size_bytes = " + str(bound["size_bytes"])]
    execution = folder / "execution.toml"
    with execution.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
    return file_descriptor(execution)


def prepare_wasserstein_job(execution_descriptor, job_root, *, series_manifest):
    """Validate all explicit inputs, then enqueue via the original JobQueue.

The caller must integrate the typed shared validators/dispatch first. The
unchanged generic validator currently rejects this distinct non-calibration
pipeline; that rejection precedes all writes to job_root.
"""
    from .pipeline.alias_job import JOB_FORMAT, TASK_FORMAT, _manifest, shared_output_admission
    from .pipeline.job_inputs import InputCatalog
    from .pipeline.job_queue import JobQueue, descriptor as file_descriptor, validate_task, write_immutable
    from .wasserstein_job_inputs import collect_wasserstein_catalog
    source = descriptor(execution_descriptor)
    _, manifest, roots = _manifest(series_manifest)
    admission = InputCatalog(roots)
    admission.add(source, "configuration", "run.execution_series")
    raw = InputCatalog._read_metadata(source["path"])
    if len(raw) != source["size_bytes"] or hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError("Saved execution-series metadata changed")
    document = tomllib.loads(raw.decode("utf-8-sig"))
    if document.get("format") != "darsia-series-v1" or document.get("execution_kind") != PIPELINE:
        raise ValueError("An explicit Wasserstein execution series is required")
    identity = _identity(job_id=document["job_id"], commit=document["job"]["commit"],
                         environment_lock=document["job"]["environment_lock"])
    if document["job"].get("precision") != "float64" or document["job"].get("backend") != "cpu":
        raise ValueError("The saved comparison execution must explicitly select CPU float64")
    root = Path(job_root)
    if root.exists():
        raise FileExistsError("A Wasserstein queue job needs a fresh folder")
    shared_output_admission(root)
    tasks, plan = [], None
    for declaration in document["runs"]:
        catalog = InputCatalog(roots)
        series = catalog.add(source, "configuration", "run.execution_series")
        registered = catalog.add(series_manifest, "configuration", "run.series_manifest")
        # Root admission precedes every nested metadata open, including the
        # operation and the saved plan that it names.
        config = catalog.doc(declaration["configuration"], "configuration", "run.configuration")
        actual_plan = catalog.doc(config["plan"], "configuration", "wasserstein.plan")
        validate_operation(config, actual_plan)
        if (declaration["run"] != config["run"] or plan is not None and actual_plan != plan):
            raise ValueError("Execution series mixes comparison plans or operation identities")
        plan = actual_plan
        inputs = collect_wasserstein_catalog(catalog, declaration["configuration"], config)
        plan_ref = catalog.consumers["wasserstein.plan"]
        tasks.append(dict(format=TASK_FORMAT, format_version=1, legacy_paths=False, storage_profile="full",
            execution_ready=True, runtime_support="darsia-role-io-runtime-v1", pipeline=PIPELINE,
            backend="cpu", precision="float64", threads=1, commit=identity["commit"], job_id=document["job_id"],
            run=config["run"], ordered_case_ids=[config["run"]], configuration=inputs["configuration"],
            series=series, series_manifest=registered, input_catalog=inputs, wasserstein_plan=plan_ref,
            wasserstein_contract_sha256=plan["task_contract"]["content_sha256"], comparison=config["comparison"]))
    if plan is None or [task["comparison"] for task in tasks] != comparisons(plan):
        raise ValueError("Execution-series coverage differs from the complete saved plan")
    job = dict(identity, format=JOB_FORMAT, format_version=1, legacy_paths=False, storage_profile="full",
        pipeline=PIPELINE, job_id=document["job_id"], series_id=manifest["series_id"],
        execution_ready=True, execution_blocker="", runtime_support="darsia-role-io-runtime-v1",
        series=tasks[0]["series"], series_manifest=tasks[0]["series_manifest"],
        wasserstein_plan=tasks[0]["wasserstein_plan"], wasserstein_comparisons=comparisons(plan),
        wasserstein_contract_sha256=plan["task_contract"]["content_sha256"],
        expected_runs=[task["run"] for task in tasks], input_reference_format="root-alias-relative-v1", publish=[])
    for task in tasks:
        validate_wasserstein_claim(task, job)
        validate_task(task)  # Shared integration required; before root.mkdir.
    # Recheck metadata sources after preparing the catalogues, before writing.
    _manifest(series_manifest)
    if InputCatalog._read_metadata(source["path"]) != raw:
        raise ValueError("Execution series changed during preparation")
    root.mkdir(parents=True, exist_ok=False)
    write_immutable(root / "job.json", job)
    queue = JobQueue(root)
    prepared = [dict(run=task["run"], task=file_descriptor(queue.enqueue(task))) for task in tasks]
    return dict(complete=True, job=file_descriptor(root / "job.json"), tasks=prepared,
                pipeline=PIPELINE, numerical_calls_started=False, worker_started=False)
