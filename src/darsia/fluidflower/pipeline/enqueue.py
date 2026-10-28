"""Enqueue explicit whole-run task dictionaries in an existing UNC job."""
import argparse
import json
from pathlib import Path

from .job_queue import JobQueue, read_json, require_unc


def main(argv=None):
    """Enqueue explicit whole-run task metadata with the job's code and calibration pins.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True,
                        help="UNC JSON containing a task list or {'tasks': [...]}.")
    args = parser.parse_args(argv)
    require_unc(str(args.job))
    require_unc(str(args.tasks))
    job = read_json(args.job / "job.json")
    document = read_json(args.tasks)
    tasks = document if isinstance(document, list) else document["tasks"]
    shared = JobQueue(args.job)
    for original in tasks:
        task = dict(original)
        if job.get("pipeline") == "native-phase-statistics":
            from .phase1_protocol import validate_phase1_job, validate_phase1_claim
            validate_phase1_job(job)
            if task.get("pipeline") != "native-phase-statistics":
                raise ValueError("Each supplied task must already declare the positive phase-one stage")
            for key in ("commit", "precision", "phase1_configurations_sha256"):
                task.setdefault(key, job[key])
                if task[key] != job[key]:
                    raise ValueError(f"Task {key} differs from job.json")
            validate_phase1_claim(task, job)
            path = shared.enqueue(task)
            print(json.dumps(dict(run=task["run"], queued=str(path))), flush=True)
            continue
        for key in ("commit", "calibration_sha256", "precision"):
            task.setdefault(key, job[key])
            if task[key] != job[key]:
                raise ValueError(f"Task {key} differs from job.json")
        path = shared.enqueue(task)
        print(json.dumps(dict(run=task["run"], queued=str(path))), flush=True)


if __name__ == "__main__":
    main()
