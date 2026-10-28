"""Original comparison result counts from immutable shared publications only."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import traceback

from .preset_wasserstein_protocol import validate_preset_wasserstein_claim, validate_preset_wasserstein_job
from .preset_wasserstein_results import preset_wasserstein_terminal


def preset_wasserstein_snapshot(registration, job, ui, *, reader):
    """Project requested-time tasks without frame counts or scientific acceptance."""
    from .published_jobs_snapshot import _task_identity, _owner_machine, _failed_publication
    from .job_launch_state import read_launch_requests
    from .pipeline.alias_job import job_metadata_reference
    from .pipeline.job_queue import descriptor, name, _own_asset

    from .jobs_progress import retain_execution, retain_log_reports
    validate_preset_wasserstein_job(job)
    root, rows, workers, errors = Path(registration["root"]), [], [], []
    for run, time in zip(job["expected_runs"], job["preset_comparison_times"]):
        row = dict(run=run, comparison_time=dict(time), state="Not queued", committed=0,
                   total=None, completed_comparisons=0, total_comparisons=job["comparisons_per_time"],
                   stage="Comparison", machine="Not reported", worker="Not reported", error="", case_ids=[])
        try:
            task_path = root / "results" / name(run) / "task.json"
            if not task_path.exists():
                rows.append(row)
                continue
            task = reader(task_path)
            validate_preset_wasserstein_claim(task, job)
            if task["run"] != run or task["comparison_time"] != time:
                raise ValueError("Comparison task differs from its declared requested time")
            done_path, failure_path = root / "done" / (run + ".json"), root / "failed" / (run + ".json")
            claims = list((root / "claimed").glob("*/" + run + ".json"))
            if done_path.exists():
                done = reader(done_path)
                completed, _, _ = preset_wasserstein_terminal(root, task, job, done, reader)
                owner = reader(_own_asset(root, done["products_index"]).parent.parent / "owner.json")
                physical = _owner_machine(owner, task, done["attempt_id"])
                if owner.get("actual_owner") != completed["actual_worker"]:
                    raise ValueError("Comparison execution owner changed after terminal verification")
                retain_execution(row, task=task, owner=owner, result=completed)
                row.update(state="Complete", completed_comparisons=completed["comparison_count"],
                    machine=physical, service=completed["actual_worker"]["machine"], result=completed,
                    nonfinite_distances=completed["nonfinite_distances"],
                    nonfinite_normalized_distances=completed["nonfinite_normalized_distances"],
                    result_statuses=dict(completed["result_statuses"]))
            elif failure_path.exists():
                failure, counted, times, physical = _failed_publication(root, task, failure_path, reader, display=row)
                if counted or times:
                    raise ValueError("Comparison failure cannot claim image progress")
                row.update(state="Failed", error=failure["traceback"], machine=physical)
            elif claims:
                if len(claims) != 1:
                    raise ValueError("More than one current comparison claim for a requested time")
                claimed = reader(claims[0])
                if not _task_identity(task, claimed):
                    raise ValueError("Current comparison claim differs from its task")
                owner_path = root / "leases" / (run + "." + claimed["attempt_id"]) / "owner.json"
                row["state"] = "Claiming (owner not published)"
                if owner_path.exists():
                    owner = reader(owner_path)
                    physical = _owner_machine(owner, task, claimed["attempt_id"])
                    if owner.get("task") != claimed:
                        raise ValueError("Current comparison owner differs from its claim")
                    retain_execution(row, task=claimed, owner=owner)
                    actual = owner.get("actual_owner") or {}
                    if name(actual.get("machine")) != name(physical + "-" + task["backend"]):
                        raise ValueError("Claimed comparison service differs from its physical execution machine")
                    row.update(state="Claimed (liveness not established)",
                        machine=physical, service=actual["machine"],
                        worker=f"{actual['machine']} · PID {actual.get('pid', '?')}")
                    workers.append(dict(machine=row["machine"], worker=row["worker"], run=run,
                        state=row["state"], last_progress_utc="Not reported", error=""))
            elif (root / "publishing" / run).exists():
                row["state"] = "Publishing (not done)"
            elif (root / "queue" / (run + ".json")).exists():
                row["state"] = "Queued"
        except Exception:
            row.update(state="Metadata error", error=traceback.format_exc(), completed_comparisons=0)
            errors.append(row["error"])
        rows.append(row)
    completed_runs = sum(row["state"] == "Complete" for row in rows)
    completed = sum(row["completed_comparisons"] for row in rows)
    total = len(rows) * job["comparisons_per_time"]
    failed = sum(row["state"] == "Failed" for row in rows)
    complete = completed_runs == len(rows) and not errors
    statuses = Counter()
    for row in rows:
        statuses.update(row.get("result_statuses", {}))
    logs = []
    for path in sorted((root / "logs").glob("*.log")):
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - 128 * 1024))
            logs.append(dict(path=str(path), tail=stream.read().decode("utf-8", errors="replace")))
    registered = []
    for path in sorted((root / "ui" / "machines").glob("*.json")):
        value = reader(path)
        if value.get("job") != job_metadata_reference(root, root / "job.json"):
            raise ValueError("Comparison registered target belongs to another job")
        registered.append(dict(value, source=descriptor(path)))
    return retain_log_reports(dict(root=str(root), name=ui.get("name") or registration.get("name") or root.name,
        series=ui.get("series_label") or registration.get("series_label") or job.get("series_id"),
        created_utc=ui.get("created_utc") or job.get("created_utc") or "Not reported",
        master_machine=ui.get("machine") or "Not recorded", job_facts=dict(job), runs=rows,
        workers=workers, machines=[], logs=logs, registered_machines=registered,
        launch_requests=read_launch_requests(root, reader=reader), run_count=len(rows),
        completed_runs=completed_runs, failed_runs=failed, committed=0, total_frames=None,
        remaining=None, frames_per_minute=None, eta_seconds=None, complete=complete,
        work_unit="comparisons", completed_comparisons=completed, total_comparisons=total,
        remaining_comparisons=total - completed, errors=errors,
        nonfinite_distances=sum(row.get("nonfinite_distances", 0) for row in rows),
        nonfinite_normalized_distances=sum(row.get("nonfinite_normalized_distances", 0) for row in rows),
        result_statuses=dict(statuses),
        state="Complete" if complete else "Metadata error" if errors else "Failed" if failed else "Claimed" if workers else "Waiting",
        rate_scope="Verified original comparison result records; scientific validity is not scored"))
