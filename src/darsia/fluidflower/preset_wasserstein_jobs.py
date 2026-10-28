"""Prepare original Comparison inputs and enqueue exact requested-time tasks."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from .preset_wasserstein_protocol import (
    OPERATION_FORMAT, PIPELINE, digest, expected_comparisons, operation_id,
    validate_inputs_contract, validate_operation, validate_preset_wasserstein_claim,
    validate_preset_wasserstein_job,
)


def _fresh_folder(path):
    path = Path(path).expanduser()
    if path.exists() or path.is_symlink():
        raise FileExistsError("Choose a fresh comparison metadata and job folder")
    return path.resolve()


def _execution_text(rows, *, job_id, identity, actor, reason):
    """Keep original configuration descriptors available to the role replay."""
    quote = lambda value: json.dumps(value, ensure_ascii=True)
    lines = ['format = "darsia-preset-wasserstein-series-v1"',
             'pipeline = "preset-wasserstein"', "job_id = " + quote(job_id),
             "created_by = " + quote(actor), "creation_reason = " + quote(reason),
             "commit = " + quote(identity["commit"]),
             "created_utc = " + quote(identity["created_utc"])]
    for row in rows:
        lines.extend(["", "[[runs]]", "run = " + quote(row["run"]),
                      "[runs.configuration]"])
        bound = row["configuration"]
        lines.extend(["path = " + quote(bound["path"]),
                      "sha256 = " + quote(bound["sha256"]),
                      "size_bytes = " + str(bound["size_bytes"])])
    return ("\n".join(lines) + "\n").encode("utf-8")


def prepare_preset_wasserstein_job(config_path, metadata_folder, job_root, *,
                                  series_root, series_id, job_id, actor, reason,
                                  repository):
    """Capture the current original recipe and create a job without a worker.

    Preparation parses original controls, selects existing mass products and
    binds their identities. It never loads a rig, decodes an image or calls the
    solver. A failed preparation preserves any written immutable metadata and
    creates no queue until every task has passed the existing admission checks.
    """
    from .analysis_preparation import job_identity
    from .pipeline.alias_job import _manifest, shared_output_admission
    from .pipeline.job_inputs import InputCatalog, MAX_METADATA_BYTES, validate_catalog
    from .pipeline.job_queue import (
        JobQueue, _immutable_bytes, _publish_bytes, descriptor, name,
        validate_task, write_immutable,
    )
    from .preset_wasserstein_inputs import compile_preset_wasserstein_inputs
    from .store_map import input_reference

    if (not isinstance(actor, str) or not actor.strip()
            or not isinstance(reason, str) or not reason.strip()):
        raise ValueError("Record your name and reason for this comparison job")
    actor, reason = actor.strip(), reason.strip()
    if not isinstance(job_id, str) or name(job_id) != job_id:
        raise ValueError("Choose a canonical job name")
    metadata, root = _fresh_folder(metadata_folder), _fresh_folder(job_root)
    if metadata == root or metadata.is_relative_to(root) or root.is_relative_to(metadata):
        raise ValueError("Comparison metadata and job folders must be separate")
    shared_output_admission(root)
    manifest = descriptor((Path(series_root) / "series.toml").resolve(strict=True))
    _source, registered, roots = _manifest(manifest)
    if registered.get("series_id") != series_id:
        raise ValueError("The selected registered series changed before preparation")
    identity = job_identity(repository)
    frozen = validate_inputs_contract(compile_preset_wasserstein_inputs(
        config_path, roots=roots, series_manifest=manifest))
    source_hash = digest(frozen)
    times = frozen["requested_times"]
    operations = [validate_operation(dict(format=OPERATION_FORMAT,
        run=operation_id(time["order"]), preset_inputs=deepcopy(frozen),
        comparison_time=deepcopy(time))) for time in times]
    paths = [metadata / (op["run"] + ".json") for op in operations]
    execution_path = metadata / "execution.toml"
    # Root admission precedes the first mkdir, including future controls.
    for path in [execution_path, *paths]:
        input_reference(str(path), roots, sha256="0" * 64, size_bytes=0, role="configuration")
    for op in operations:
        if len(_immutable_bytes(op)) > MAX_METADATA_BYTES:
            raise ValueError("Original comparison controls exceed the metadata ceiling")
    count = len(expected_comparisons(frozen, times[0]))
    if descriptor(Path(manifest["path"])) != manifest:
        raise ValueError("Registered series changed during input preparation")
    metadata.mkdir(parents=True, exist_ok=False)
    rows = []
    for path, op in zip(paths, operations):
        write_immutable(path, op)
        rows.append(dict(run=op["run"], configuration=descriptor(path)))
    raw = _execution_text(rows, job_id=job_id, identity=identity, actor=actor, reason=reason)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("Original execution declarations exceed the metadata ceiling")
    _publish_bytes(execution_path, raw, create_parents=False)
    execution = descriptor(execution_path)
    admission = InputCatalog(roots)
    series_ref = admission.add(execution, "configuration", "execution_series")
    manifest_ref = admission.add(manifest, "configuration", "series_manifest")
    common = dict(format_version=1, legacy_paths=False, storage_profile="full",
        execution_ready=True, execution_blocker="", runtime_support="darsia-role-io-runtime-v1",
        pipeline=PIPELINE, backend="cpu", precision="original", job_id=job_id,
        commit=identity["commit"], series=series_ref, series_manifest=manifest_ref,
        preset_inputs_sha256=source_hash)
    job = dict(deepcopy(common), format="darsia-alias-job-v1",
        environment_lock=deepcopy(identity["environment_lock"]),
        created_utc=identity["created_utc"], created_by=actor, creation_reason=reason,
        series_id=series_id, input_reference_format="root-alias-relative-v1", publish=[],
        preset_comparison_times=deepcopy(times), comparisons_per_time=count,
        reference_run=frozen["reference_run"], expected_runs=[row["run"] for row in rows])
    validate_preset_wasserstein_job(job)
    tasks = []
    for row, op in zip(rows, operations):
        catalog = InputCatalog(roots)
        catalog.add(execution, "configuration", "run.execution_series")
        catalog.add(manifest, "configuration", "run.series_manifest")
        if catalog.doc(row["configuration"], "configuration", "run.configuration") != op:
            raise ValueError("Prepared comparison configuration changed")
        inputs = validate_catalog(catalog.collect(row["configuration"], op))
        inputs["runtime_support"] = "darsia-role-io-runtime-v1"
        task = dict(deepcopy(common), format="darsia-alias-task-v1", run=row["run"],
            threads=1, ordered_case_ids=[row["run"]], comparison_time=deepcopy(op["comparison_time"]),
            expected_comparisons=count, configuration=inputs["configuration"], input_catalog=inputs)
        validate_task(task)
        validate_preset_wasserstein_claim(task, job, config=op)
        tasks.append(task)
    current = job_identity(repository)
    if (descriptor(Path(manifest["path"])) != manifest
            or any(current[key] != identity[key] for key in ("commit", "environment_lock"))):
        raise ValueError("Source or registered series changed during job preparation")
    root.mkdir(parents=True, exist_ok=False)
    write_immutable(root / "job.json", job)
    queue = JobQueue(root)
    for task in tasks:
        queue.enqueue(task)
    return dict(root=str(root), name=job_id, job=descriptor(root / "job.json"),
        pipeline=PIPELINE, comparison_count=count * len(times), requested_times=deepcopy(times),
        reference_run=frozen["reference_run"], worker_started=False)
