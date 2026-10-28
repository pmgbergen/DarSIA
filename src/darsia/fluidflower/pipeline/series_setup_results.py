"""Typed Setup results and shared metadata projections, without frame progress."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import traceback

from .job_queue import descriptor, name
from .series_setup_protocol import (
    PIPELINE, OPERATIONS, _descriptor, validate_setup_claim,
    validate_setup_job, validate_setup_task,
)

RESULT_FORMAT = "darsia-series-setup-result-v1"
RECEIPT = "operation_result.json"
_CORE = ("format", "pipeline", "complete", "run", "operation", "backend",
         "precision", "setup_plan", "worker_references_closed")


def _operations(job):
    """Read an optional literal operation map without deriving stages from ids."""
    rows = job.get("setup_operations")
    if rows is None:
        return {}
    if (not isinstance(rows, list) or any(not isinstance(row, dict)
            or set(row) != {"run", "operation"} or row["operation"] not in OPERATIONS
            for row in rows) or [row["run"] for row in rows] != job["expected_runs"]):
        raise ValueError("Setup operation mapping differs from its exact job order")
    return {row["run"]: row["operation"] for row in rows}


def validate_setup_result(result, task=None, job=None):
    """Validate a closed operation result without opening any declared source."""
    if (not isinstance(result, dict) or result.get("format") != RESULT_FORMAT
            or result.get("pipeline") != PIPELINE or result.get("complete") is not True
            or result.get("worker_references_closed") is not True
            or result.get("operation") not in OPERATIONS or result.get("backend") != "cpu"
            or result.get("precision") not in {"float32", "float64"}):
        raise ValueError("An actual closed typed Setup result is required")
    name(result.get("run"))
    _descriptor(result.get("setup_plan"), "Setup result plan")
    files = result.get("files")
    if not isinstance(files, dict) or set(files) != {RECEIPT}:
        raise ValueError("Setup results require exactly their operation receipt")
    _descriptor(files[RECEIPT], "Setup operation receipt")
    if any(key in result for key in ("frames", "records", "frame_producers", "case_count")):
        raise ValueError("Setup operation results cannot claim image or frame progress")
    if task is not None:
        validate_setup_task(task)
        if (any(result[key] != task[key] for key in ("run", "operation", "backend", "precision"))
                or result["setup_plan"]["sha256"] != task["setup_plan"]["sha256"]
                or result["setup_plan"]["size_bytes"] != task["setup_plan"]["bytes"]
                or ("commit" in result and result["commit"] != task["commit"])):
            raise ValueError("Setup result differs from its actual operation task")
    if job is not None:
        validate_setup_job(job)
        operations = _operations(job)
        if (result["run"] not in job["expected_runs"]
                or (operations and operations[result["run"]] != result["operation"])
                or result["backend"] != job["backend"] or result["precision"] != job["precision"]
                or result["setup_plan"]["sha256"] != job["setup_plan"]["sha256"]
                or result["setup_plan"]["size_bytes"] != job["setup_plan"]["bytes"]
                or ("commit" in result and result["commit"] != job["commit"])):
            raise ValueError("Setup result differs from its immutable job")
        if task is not None:
            validate_setup_claim(task, job)
    return result


def _envelope(result, task, token):
    """Require the runtime's exact claimed task, generation, source and owner."""
    validate_setup_result(result, task=task)
    actual = result.get("actual_worker")
    if (result.get("task") != task or result.get("attempt_id") != token
            or task.get("attempt_id") != token or result.get("commit") != task["commit"]
            or not isinstance(actual, dict)
            or name(actual.get("machine")) != name(task["target_machine"] + "-" + task["backend"])):
        raise ValueError("Setup completion differs from its actual runtime envelope")


def _receipt(result, document):
    """Join the immutable receipt to the producer fields in the completion."""
    if (not isinstance(document, dict) or not set(_CORE) <= document.keys()
            or "files" in document or any(result.get(key) != value for key, value in document.items())):
        raise ValueError("Setup operation receipt differs from its completed result")
    validate_setup_result(dict(document, files=result["files"]))


def extract_setup_products(result, claim, machine_id):
    """Publish only the explicit receipt inside this actual owned attempt."""
    from .products_publication import product, _bound, _verified_bytes, _json, MAX_METADATA_BYTES
    name(machine_id)
    _envelope(result, claim.task, claim.token)
    bound = _bound(result["files"][RECEIPT])
    root = Path(claim.output).resolve()
    path = root / RECEIPT
    if Path(bound["path"]).resolve() != path or path.is_symlink():
        raise ValueError("Setup receipt is outside its actual owned attempt")
    _receipt(result, _json(_verified_bytes(path, bound, limit=MAX_METADATA_BYTES)))
    row = product(RECEIPT, machine_id=machine_id, local_path=bound["path"],
                  sha256=bound["sha256"], bytes=bound["bytes"], kind="operational_receipt")
    return [row], [RECEIPT], {RECEIPT: path}


def setup_terminal(root, base_task, job, done, reader, *, display=None):
    """Verify original owner/index/result identities using shared copies only."""
    from ..published_jobs_snapshot import (
        _read_reference, _same_bytes as same, _task_identity, _lease_chain,
        _owner_machine, _original_metadata, _pure, _SnapshotMetadata,
    )
    from .products_publication import _read_index, published_metadata_descriptor
    from .job_queue import _own_asset
    root = Path(root)
    validate_setup_claim(base_task, job)
    if (done.get("format") != "darsia-alias-done-v1" or done.get("run") != base_task["run"]
            or done["completed"]["relative_path"] != "results/" + base_task["run"] + "/completed.json"):
        raise ValueError("Actual alias Setup done identity required")
    completed = _read_reference(root, done["completed"], reader)
    index_path = _own_asset(root, done["products_index"])
    index = descriptor(index_path)
    folder, document, products = _read_index(index)
    metadata = _SnapshotMetadata(index, folder, products)
    task = completed.get("task") or {}
    if (document["run"] != base_task["run"] or document["attempt_id"] != done["attempt_id"]
            or not same(document["job"], descriptor(root / "job.json"))
            or not same(published_metadata_descriptor(index), done["completed"])
            or not _task_identity(base_task, task)):
        raise ValueError("Original Setup index/task/source identity differs")
    _envelope(completed, task, done["attempt_id"])
    validate_setup_result(completed, task=task, job=job)
    owner = reader(index_path.parent.parent / "owner.json")
    if (owner.get("task") != task or owner.get("token") != done["attempt_id"]
            or owner.get("actual_owner") != completed.get("actual_worker")):
        raise ValueError("Setup publication owner differs from the actual claim")
    machine = _owner_machine(owner, base_task, done["attempt_id"])
    original = products["completed.json"]
    attempt = _pure(original["local_path"]).parent
    if (not same(original, done["completed"]) or name(original["machine_id"]) != name(machine)
            or _pure(original["local_path"]).name != "candidate_completed.json"
            or attempt.name != done["attempt_id"] or attempt.parent.name != ".attempts"
            or name(attempt.parent.parent.name) != name(task["run"])
            or name(attempt.parent.parent.parent.name) != name(task["job_id"])):
        raise ValueError("Setup producer is outside its actual owned job/run/attempt")
    if set(products) != {"completed.json", RECEIPT}:
        raise ValueError("Setup publication omits or adds an actual output")
    bound, product = completed["files"][RECEIPT], products[RECEIPT]
    if (not same(bound, product) or _pure(product["local_path"]) != attempt / RECEIPT
            or _pure(bound["path"]) != attempt / RECEIPT
            or name(product["machine_id"]) != name(machine)):
        raise ValueError("Indexed Setup receipt differs from its original descriptor")
    receipt = _original_metadata(index, products, bound, attempt, machine, metadata=metadata)
    _receipt(completed, receipt)
    lineage = _lease_chain(root, base_task["run"], done["attempt_id"], base_task, reader)
    metadata.finish()
    if display is not None:
        from ..jobs_progress import retain_execution
        retain_execution(display, task=task, owner=owner, result=completed)
    return completed, {}, lineage


def _reported_progress(logs):
    """Index bounded original service reports by exact task and attempt identity."""
    reports = {}
    for log in logs:
        service = Path(log["path"]).stem
        for line in log["tail"].splitlines():
            timestamp, space, encoded = line.partition(" ")
            if not space:
                continue
            try:
                stamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                envelope = json.loads(encoded)
            except (ValueError, TypeError):
                continue
            if stamp.tzinfo is None or not isinstance(envelope, dict):
                continue
            event = envelope.get("event")
            if (envelope.get("kind") != "progress" or not isinstance(event, dict)
                    or event.get("state") != "setup_progress"
                    or not isinstance(event.get("operation"), str)
                    or event.get("operation") not in OPERATIONS):
                continue
            task_run, attempt = envelope.get("run"), envelope.get("attempt_id")
            run = event.get("run", task_run)
            step, message = event.get("step"), event.get("message", "")
            if (not all(isinstance(value, str) and value for value in
                        (task_run, attempt, run, step)) or not isinstance(message, str)):
                continue
            key = service, task_run, attempt, event["operation"]
            reports[key] = dict(task_run=task_run, attempt_id=attempt,
                operation=event["operation"], run=run, step=step, message=message,
                reported_utc=timestamp, source_log=log["path"])
            if (type(event.get("run_index")) is int and type(event.get("run_count")) is int
                    and 1 <= event["run_index"] <= event["run_count"]):
                reports[key].update(run_index=event["run_index"], run_count=event["run_count"])
    return reports


def _snapshot(registration, job, ui, reader, *, prepared):
    """Project actual operation states without decoding any external product."""
    from ..published_jobs_snapshot import _task_identity, _owner_machine, _failed_publication
    from ..job_launch_state import read_launch_requests
    from .alias_job import job_metadata_reference
    from ..jobs_progress import retain_execution, retain_log_reports
    validate_setup_job(job)
    operations = _operations(job)
    root, rows, workers, errors = Path(registration["root"]), [], [], []
    logs = []
    for path in sorted((root / "logs").glob("*.log")):
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - 128 * 1024))
            tail = stream.read(128 * 1024).decode("utf-8", errors="replace")
            logs.append(dict(path=str(path), tail=tail))
    progress_reports = _reported_progress(logs)
    from ..setup_completed_builds import retained_operations
    retained_builds = retained_operations(root, job)
    for run in job["expected_runs"]:
        row = dict(run=run, operation=operations.get(run), state="Not queued", committed=0,
                   total=None, completed_operations=0, total_operations=1,
                   stage=operations.get(run, "").capitalize() or None,
                   machine="Not reported", worker="Not reported", error="", case_ids=[])
        try:
            task_path = root / "results" / name(run) / "task.json"
            if run in retained_builds:
                previous = retained_builds[run]
                row.update(previous["display"], state="Complete", completed_operations=1,
                           result=previous["result"], retained_from=previous["source_job"])
                rows.append(row)
                continue
            if not task_path.exists():
                if operations.get(run) == "build":
                    row["state"] = "Waiting for Apply"
                elif operations.get(run) == "image-preparation":
                    row["state"] = "Waiting for geometry Build"
                rows.append(row)
                continue
            task = reader(task_path)
            validate_setup_claim(task, job)
            if task["run"] != run or (operations and operations[run] != task["operation"]):
                raise ValueError("Setup task differs from its declared operation run")
            row["operation"] = task["operation"]
            row["stage"] = task["operation"].capitalize()
            done_path, failure_path = root / "done" / (run + ".json"), root / "failed" / (run + ".json")
            claims = list((root / "claimed").glob("*/" + run + ".json"))
            if prepared:
                if done_path.exists() or failure_path.exists() or claims:
                    raise ValueError("Prepared Setup metadata has unexpected execution artifacts")
                row["state"] = "Prepared (runtime pending)"
            elif done_path.exists():
                completed, _, _ = setup_terminal(root, task, job, reader(done_path), reader, display=row)
                row.update(state="Complete", completed_operations=1,
                           machine=row.get("machine", "Not reported"), result=completed)
            elif failure_path.exists():
                failure, counted, times, physical = _failed_publication(root, task, failure_path, reader, display=row)
                if counted or times:
                    raise ValueError("Setup failure cannot claim image progress")
                row.update(state="Failed", error=failure["traceback"], machine=physical)
            elif claims:
                if len(claims) != 1:
                    raise ValueError("More than one current Setup claim for a run")
                claimed = reader(claims[0])
                if not _task_identity(task, claimed):
                    raise ValueError("Current Setup claim differs from its task")
                owner_path = root / "leases" / (run + "." + claimed["attempt_id"]) / "owner.json"
                row["state"] = "Claiming (owner not published)"
                if owner_path.exists():
                    owner = reader(owner_path)
                    _owner_machine(owner, task, claimed["attempt_id"])
                    if owner.get("task") != claimed:
                        raise ValueError("Current Setup owner differs from its claim")
                    retain_execution(row, task=claimed, owner=owner)
                    actual = owner.get("actual_owner") or {}
                    row.update(state="Claimed (liveness not established)",
                               machine=row["machine"],
                               worker=row["worker"])
                    workers.append(dict(machine=row["machine"], worker=row["worker"], run=run,
                        state=row["state"], last_progress_utc="Not reported", error=""))
                    report = progress_reports.get((name(actual.get("machine", "Not reported")), run,
                        claimed["attempt_id"], task["operation"]))
                    if report is not None:
                        row["reported_progress"] = report
                        workers[-1].update(last_progress_utc=report["reported_utc"],
                            state=row["state"] + " · Last report: " + report["operation"]
                                + " / " + report["run"] + " / " + report["step"])
            elif (root / "publishing" / run).exists():
                row["state"] = "Publishing (not done)"
            elif (root / "queue" / (run + ".json")).exists():
                row["state"] = "Queued"
        except Exception:
            row.update(state="Metadata error", error=traceback.format_exc(), completed_operations=0)
            errors.append(row["error"])
        rows.append(row)
    completed = sum(row["completed_operations"] for row in rows)
    failed = sum(row["state"] == "Failed" for row in rows)
    complete = completed == len(rows) and not errors
    registered = []
    for path in sorted((root / "ui" / "machines").glob("*.json")):
        value = reader(path)
        if value.get("job") != job_metadata_reference(root, root / "job.json"):
            raise ValueError("Setup registered target belongs to another job")
        registered.append(dict(value, source=descriptor(path)))
    return retain_log_reports(dict(root=str(root), name=ui.get("name") or registration.get("name") or root.name,
        series=ui.get("series_label") or registration.get("series_label") or job.get("series_id"),
        created_utc=ui.get("created_utc") or job.get("created_utc") or "Not reported",
        master_machine=ui.get("machine") or "Not recorded", job_facts=dict(job), runs=rows,
        workers=workers, machines=[], logs=logs, registered_machines=registered,
        launch_requests=read_launch_requests(root, reader=reader), run_count=len(rows),
        completed_runs=completed, failed_runs=failed, committed=0, total_frames=None,
        remaining=None, frames_per_minute=None, eta_seconds=None, complete=complete,
        work_unit="operations", completed_operations=completed, total_operations=len(rows),
        remaining_operations=len(rows) - completed, errors=errors,
        state="Complete" if complete else "Metadata error" if errors else "Failed" if failed else "Claimed" if workers else "Waiting",
        rate_scope="Only verified shared Setup operation completions; no image progress or worker retirement inferred"))


def setup_snapshot(registration, job, ui, *, reader):
    """Read a runnable Setup job through the original shared publication layer."""
    return _snapshot(registration, job, ui, reader, prepared=False)


def prepared_setup_snapshot(registration, job, ui, *, reader):
    """Display only prepared operation metadata and explicit unstaged children."""
    return _snapshot(registration, job, ui, reader, prepared=True)
