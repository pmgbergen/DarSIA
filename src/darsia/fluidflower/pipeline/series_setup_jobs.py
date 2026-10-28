"""Prepare original-queue Setup tasks without capturing future Apply inputs."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path, PureWindowsPath
import re

from .alias_job import JOB_FORMAT, TASK_FORMAT, _manifest, shared_output_admission
from .capacity import read_bound
from .job_inputs import InputCatalog, MAX_METADATA_BYTES, _json_bytes, validate_catalog
from .job_queue import JobQueue, _immutable_bytes, name, validate_task, write_immutable
from .series_setup_inputs import _same
from .series_setup_protocol import (
    FORMAT, validate_setup_claim, validate_setup_job, validate_setup_operation,
    validate_setup_plan, validate_setup_task,
)

# Setup plans embed per-run request templates. Use the same finite metadata
# bound as the named-input catalogue and the worker that replay those plans.
CONTROL_LIMIT = MAX_METADATA_BYTES
APPLY_RUN = "setup-apply"


def _control(path):
    """Capture only one bounded current metadata control, preserving its version."""
    path = Path(path)
    before = path.stat()
    if before.st_size > CONTROL_LIMIT:
        raise ValueError("Setup control exceeds the shared job metadata limit")
    with path.open("rb") as stream:
        raw = stream.read(CONTROL_LIMIT + 1)
    after = path.stat()
    version = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if version(before) != version(after) or len(raw) != before.st_size:
        raise ValueError("Setup control changed while it was captured")
    bound = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    return _json_bytes(raw), bound


def _immutable(path, document):
    """Reuse identical controls and preserve every conflicting or partial file."""
    path = Path(path)
    raw = _immutable_bytes(document)
    if len(raw) > CONTROL_LIMIT:
        raise ValueError("Setup control exceeds the shared job metadata limit")
    if path.exists():
        old, bound = _control(path)
        if not _same(old, document):
            raise ValueError("Existing immutable Setup control differs")
        return bound
    write_immutable(path, document)
    _value, bound = _control(path)
    return bound


def _roots(plan, job_root):
    """Keep declared metadata away from protected acquisitions before any writer."""
    from ..rig_raw_write import _canonical
    roots = [_canonical(path) for path in plan["output_scope"]["metadata_roots"]]
    for path in (_canonical(job_root), _canonical(plan["control_root"])):
        if not any(path == root or path.is_relative_to(root) for root in roots):
            raise ValueError("Setup job and control roots must be declared metadata outputs")
    scopes = [plan["output_scope"], *[build["output_scope"] for build in plan["build_plans"]]]
    outputs = [_canonical(path) for scope in scopes
               for role in ("metadata_roots", "scratch_roots") for path in scope[role]]
    for build in plan["build_plans"]:
        protected = build["request_template"]["output"]["forbidden_raw_roots"]
        if not isinstance(protected, list) or not protected:
            raise ValueError("Explicit protected RAW roots are required")
        for raw_root in protected:
            source = _canonical(raw_root)
            if any(path == source or path.is_relative_to(source) or source.is_relative_to(path)
                   for path in outputs):
                raise ValueError("Setup metadata or scratch domain overlaps protected RAW acquisitions")


def _task(bound, config, job, roots, catalog_factory):
    """Collect only original named controls, using the real series for both roles."""
    catalog = catalog_factory(roots)
    series = config["series_manifest"]
    series_reference = catalog.add(series, "configuration", "run.execution_series")
    manifest_reference = catalog.add(series, "configuration", "run.series_manifest")
    actual = catalog.doc(bound, "configuration", "run.configuration")
    if not _same(actual, config):
        raise ValueError("The Setup operation changed before catalogue collection")
    inputs = validate_catalog(catalog.collect(bound, config))
    inputs["runtime_support"] = "darsia-role-io-runtime-v1"
    plans = [row["reference"] for row in inputs["inputs"] if "setup.plan" in row["uses"]]
    if len(plans) != 1:
        raise ValueError("The Setup catalogue needs exactly one immutable plan")
    task = dict(format=TASK_FORMAT, format_version=1, legacy_paths=False,
        execution_ready=True, execution_blocker="", runtime_support="darsia-role-io-runtime-v1",
        storage_profile="full", pipeline="series-setup", run=config["run"], operation=config["operation"],
        job_id=job["job_id"], commit=job["commit"], backend="cpu", precision=job["precision"],
        threads=job["threads"], target_machine=config["target_machine"],
        configuration=inputs["configuration"], input_catalog=inputs, setup_plan=plans[0],
        series=series_reference, series_manifest=manifest_reference, ordered_case_ids=[config["operation"]])
    validate_setup_task(task)
    validate_setup_operation(config, task=task)
    validate_task(task)
    return task


def _enqueue(queue, task):
    """Reuse only the same original queued task projection, never replace a run."""
    prior = queue.root / "results" / task["run"] / "task.json"
    if prior.exists():
        existing, bound = _control(prior)
        projected = {key: value for key, value in existing.items() if key not in {"attempt_id", "output_directory"}}
        if not _same(projected, task):
            raise ValueError("Existing Setup run task differs; retained work was preserved")
        return dict(run=task["run"], task=bound, already_enqueued=True)
    _value, bound = _control(queue.enqueue(task))
    return dict(run=task["run"], task=bound, already_enqueued=False)


def prepare_setup_job(plan_descriptor, job_root, *, job_id, commit, environment_lock,
                      precision="float32", threads=1, catalog_factory=None):
    """Register only Apply now; future Build configurations remain literal paths."""
    plan = validate_setup_plan(read_bound(plan_descriptor)[0])
    root = Path(job_root).expanduser().resolve()
    _roots(plan, root)
    shared_output_admission(root)
    name(job_id)
    if APPLY_RUN in plan["runs"]:
        raise ValueError("The coordinator run collides with an actual Setup run")
    if type(threads) is not int or threads < 1:
        raise ValueError("Setup threads must be an explicit positive integer")
    lock = deepcopy(environment_lock)
    if not isinstance(lock, dict) or not isinstance(lock.get("path"), str) or not lock["path"]:
        raise ValueError("The original environment lock needs its repository-relative path and SHA")
    relative = PureWindowsPath(lock["path"])
    if (relative.drive or relative.root or ".." in relative.parts
            or re.fullmatch(r"[0-9a-f]{64}", lock.get("sha256", "")) is None):
        raise ValueError("The original environment lock needs its repository-relative path and SHA")
    _source, manifest, roots = _manifest(plan["series_manifest"])
    config = dict(format=FORMAT, run=APPLY_RUN, operation="apply", setup_plan=deepcopy(plan_descriptor),
        **{key: deepcopy(plan[key]) for key in ("project_root", "project_manifest", "adopted_revision",
            "series_manifest", "native_parameters", "runs", "build_plans", "target_machine",
            "metadata_output", "output_scope")})
    validate_setup_operation(config)
    identity = dict(job_id=job_id, commit=commit, precision=precision, threads=threads)
    # Validate source/compute identity before any initial metadata directory.
    from ..store_map import input_reference
    provisional = dict(identity, format=JOB_FORMAT, pipeline="series-setup", backend="cpu",
        setup_plan=input_reference(plan_descriptor["path"], roots, sha256=plan_descriptor["sha256"],
                                  size_bytes=plan_descriptor["size_bytes"], role="configuration"),
        expected_runs=[APPLY_RUN, *plan["runs"]],
        setup_operations=[dict(run=APPLY_RUN, operation="apply"),
                          *[dict(run=run, operation="build") for run in plan["runs"]]])
    validate_setup_job(provisional)
    operation_bound = _immutable(Path(plan["control_root"]) / "apply-operation.json", config)
    task = _task(operation_bound, config, identity, roots, catalog_factory or InputCatalog)
    job = dict(provisional, format_version=1, legacy_paths=False, storage_profile="full",
        execution_ready=True, execution_blocker="", runtime_support="darsia-role-io-runtime-v1",
        environment_lock=lock, series=task["series"], series_manifest=task["series_manifest"],
        series_id=manifest["series_id"], input_reference_format="root-alias-relative-v1", publish=[])
    from .setup_image_preparation import independent_choice, configure_setup_image_preparation_job
    image = None
    if independent_choice(plan["series_manifest"])[1] is not None:
        image = configure_setup_image_preparation_job(job, plan_descriptor, plan, root)
        job = image["job"]
    validate_setup_claim(task, job, config=config)
    job_bound = _immutable(root / "job.json", job)
    queued = _enqueue(JobQueue(root), task)
    queued_image = [] if image is None else [_enqueue(JobQueue(root), image["task"])]
    return dict(complete=True, job=job_bound, tasks=[queued, *queued_image], pipeline="series-setup",
                execution_ready=True, numerical_calls_started=False)


def enqueue_setup_builds(root, claim, result, *, catalog_factory=None):
    """Enqueue captured children under the still-owned Apply claim and real sink."""
    from .run_io import for_task, task_without_resume
    from .series_setup_sink import SetupMetadataSink, current_sink
    queue = JobQueue(Path(root))
    queue.assert_owner(claim)
    job, _job_bound = _control(queue.root / "job.json")
    validate_setup_claim(claim.task, job)
    binding, sink = for_task(claim.task), current_sink()
    if (binding is None or not isinstance(sink, SetupMetadataSink)
            or not _same(task_without_resume(sink.admission.task), task_without_resume(claim.task))):
        raise ValueError("Setup child enqueue requires the exact active Apply admission")
    config = read_bound(binding.configuration_descriptor)[0]
    validate_setup_operation(config, task=claim.task)
    if not _same(config, sink.admission.configuration):
        raise ValueError("The Setup queue writer belongs to another admitted operation")
    plan = validate_setup_plan(binding.setup_controls["plan"])
    _roots(plan, queue.root)
    if (claim.task["run"] != APPLY_RUN or config["operation"] != "apply"
            or result.get("complete") is not True or result.get("pipeline") != "series-setup"
            or result.get("run") != APPLY_RUN or result.get("operation") != "apply"
            or result.get("Build_run") is not False
            or [row.get("run_id") for row in result.get("children", ())] != plan["runs"]):
        raise ValueError("Only complete original Apply controls can enqueue the ordered Build runs")
    _source, _manifest_value, roots = _manifest(plan["series_manifest"])
    from ..setup_completed_builds import retained
    completed = retained(queue.root, job)
    tasks = []
    records = []
    for child in result["children"]:
        operation = validate_setup_operation(read_bound(child["configuration"])[0])
        if (operation["operation"] != "build" or operation["run"] != child["run_id"]
                or not _same(operation["raw_build_request"], child["raw_build_request"])
                or operation["target_machine"] != child["target_machine"]):
            raise ValueError("A staged Build control differs from its actual Apply child")
        if child["run_id"] in completed:
            previous = completed[child["run_id"]]
            if (previous["task"]["configuration"]["sha256"] != child["configuration"]["sha256"]
                    or previous["result"]["raw_build_request"] != child["raw_build_request"]):
                raise ValueError("Retained Build belongs to another Apply child")
            records.append(dict(run=child["run_id"], already_completed=True))
            continue
        task = _task(child["configuration"], operation, job, roots, catalog_factory or InputCatalog)
        validate_setup_claim(task, job, config=operation)
        tasks.append(task)
    # All child catalogues are validated before the first new pending task.
    for task in tasks:
        queue.assert_owner(claim)
        sink.admission.require_write(queue.root / "results" / task["run"])
        sink.admission.require_write(queue.root / "queue" / (task["run"] + ".json"))
        records.append(_enqueue(queue, task))
    return records
