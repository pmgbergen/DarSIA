"""Code-only recovery with preserved completed Builds and retired failures.

The successful Apply and its immutable inputs remain authoritative. Only
Unfinished attempts need their original ownership; completed results retain
their original publication and output bytes. Error wording grants no authority.
"""
from pathlib import Path
import re

from .pipeline.job_queue import descriptor, name, read_json

FORMAT = "darsia-setup-failure-before-build-output-v1"


def _pending_image_preparation(root, job, plan):
    """Permit only this job's exact typed, still-unstarted final Setup task."""
    from .pipeline.setup_image_preparation import IMAGE_RUN, OPERATION
    from .pipeline.series_setup_jobs import _control, _manifest, _task
    from .pipeline.series_setup_protocol import validate_setup_claim
    from .pipeline.job_inputs import InputCatalog
    if IMAGE_RUN not in job["expected_runs"]:
        return set()
    source = job.get("image_preparation", {})
    configuration = source.get("configuration", {})
    if Path(configuration.get("path", "")).resolve() != root / "setup-controls/image-operation.json":
        raise ValueError("Pending image preparation has no original job-owned configuration")
    config, actual = _control(configuration["path"])
    if (actual != configuration or config.get("run") != IMAGE_RUN
            or config.get("operation") != OPERATION or config.get("runs") != plan["runs"]
            or Path(config.get("job_root", "")).resolve() != root
            or config.get("choice_authority") != source.get("choice_authority")):
        raise ValueError("Pending image preparation differs from its original job declaration")
    task, _bound = _control(root / "results" / IMAGE_RUN / "task.json")
    validate_setup_claim(task, job, config=config)
    _, _, roots = _manifest(config["series_manifest"])
    expected = _task(configuration, config, job, roots, InputCatalog)
    projection = {key: value for key, value in task.items() if key not in {"attempt_id", "output_directory"}}
    if projection != expected or read_json(root / "queue" / (IMAGE_RUN + ".json")) != task:
        raise ValueError("Pending image preparation changed its exact original task or source catalogue")
    return {IMAGE_RUN + ".json"}


def _progress(lease, token, run, *, required=True):
    names, notification = set(), 0
    for sequence, path in enumerate(sorted(lease.glob("progress_*.json"))):
        value = read_json(path)
        event = value.get("event", {})
        keys = {"state", "operation", "run", "step"}
        if "run_index" in event or "run_count" in event:
            keys |= {"run_index", "run_count"}
            if (type(event.get("run_index")) is not int or event["run_index"] != 1
                    or type(event.get("run_count")) is not int or event["run_count"] != 1):
                raise ValueError("Early Build recovery found another geometry operation")
        if "message" in event:
            notification += 1
            keys |= {"message", "check_progress_sequence"}
            if (not isinstance(event["message"], str) or not event["message"].strip()
                    or event.get("check_progress_sequence") != notification):
                raise ValueError("Early Build source-check notifications changed")
        if (path.name != f"progress_{sequence:08d}.json" or value.get("run") != run
                or value.get("attempt_id") != token or value.get("sequence") != sequence
                or set(event) != keys or event.get("state") != "setup_progress"
                or event.get("operation") != "build" or event.get("run") != run
                or not isinstance(event.get("step"), str) or not event["step"].strip()):
            raise ValueError("Early Build recovery found scientific or unknown progress")
        names.add(path.name)
    if required and not names:
        raise ValueError("Early Build recovery lacks the original progress")
    return names


def review(root, job, plan, stores_path, *, allow_missing_retirement=False, series_refresh=None):
    from .job_management import _retirement_binding
    from .jobs_metadata import JobSnapshotReader
    from .published_jobs_snapshot import _failed_publication, _owner_machine, _task_identity
    from .pipeline.series_setup_jobs import APPLY_RUN, _control
    from .pipeline.series_setup_protocol import validate_setup_claim
    from .pipeline.series_setup_results import setup_terminal
    from .setup_failed_start import (
        _files, _completed_apply_progress, _completed_operation_result,
        completed_apply_snapshot,
    )
    root = Path(root)
    from .setup_completed_builds import retained, output_record
    retained_builds = retained(root, job)
    completed_builds = list(retained_builds.values())
    failures = _files(root / "failed")
    failed_runs = {Path(key).stem for key in failures}
    if (APPLY_RUN in failed_runs
            or set(failures) != {run + ".json" for run in failed_runs}
            or not failed_runs <= set(plan["runs"])):
        raise ValueError("Only original Build failures without output can update this Setup job")
    completed = completed_apply_snapshot(root, plan)
    if completed is None:
        raise ValueError("Early Build recovery requires the original completed Apply")
    reader = JobSnapshotReader()
    base, _ = _control(root / "results" / APPLY_RUN / "task.json")
    done_path = root / "done" / (APPLY_RUN + ".json")
    done = read_json(done_path)
    result, counted, lineage = setup_terminal(root, base, job, done, reader.read)
    if counted or lineage:
        raise ValueError("Early Build recovery refuses earlier operation attempts")
    token = result["attempt_id"]
    service = name(result["actual_worker"]["machine"])
    apply_claim = root / "claimed" / service / (APPLY_RUN + ".json")
    if read_json(apply_claim) != result["task"]:
        raise ValueError("Completed Apply claim differs from its published result")
    apply_lease = root / "leases" / (APPLY_RUN + "." + token)
    apply_progress = _completed_apply_progress(apply_lease, token, plan)
    attempts = []
    expected = {"claimed": {service + "/" + APPLY_RUN + ".json"},
                "leases": {APPLY_RUN + "." + token + "/" + filename for filename in
                    {"owner.json", "claimed.json", "publication.json", "artifacts/completed.json",
                     "artifacts/products_index.json", "artifacts/metadata/operation_result.json"} | apply_progress},
                "publishing": {APPLY_RUN}, "decisions": {APPLY_RUN + "." + token + ".json"},
                "failed": set(failures), "reclaimed": set()}

    def attempt(run, claim, progress, result_names, *, success=False, completed_result=None):
        binding = _retirement_binding(root, claim, stores_path)
        output = binding.path
        task = read_json(claim)
        expected_path = Path(binding.results_root) / job["job_id"] / run / ".attempts" / task["attempt_id"]
        if output.resolve() != expected_path.resolve():
            raise ValueError("Early Build attempt is outside its original results scope")
        files = _files(output)
        expected_files = result_names | progress
        if not success and "retirement.json" not in files and not allow_missing_retirement:
            from .job_local_stop import verified_local_stop_claim
            verified_local_stop_claim(root, claim, stores_path=stores_path, observe=False)
        if "retirement.json" in files:
            expected_files.add("retirement.json")
        if set(files) != expected_files:
            raise ValueError("Early Build recovery refuses product or checkpoint files")
        lease = root / "leases" / (run + "." + task["attempt_id"])
        for filename in progress:
            original = read_json(output / filename)
            shared = read_json(lease / filename)
            if shared != dict(original, detail_scope=
                    "Original event provenance; frame evidence is published_frame within this shared bundle"):
                raise ValueError("Original progress differs from its shared publication")
        row = dict(run=run, attempt_id=task["attempt_id"], claim=str(claim),
                   original_output=str(output), output_files=files)
        if success:
            row["completed"] = True
            row["actual_worker"] = (completed_result or result)["actual_worker"]
            row["completion"] = _files(root / "done")[run + ".json"]
        attempts.append(row)
        return output

    output = attempt(APPLY_RUN, apply_claim, apply_progress,
                     {"operation_result.json", "candidate_completed.json"}, success=True)
    _completed_operation_result(output / "operation_result.json", root, job, plan, completed)
    if read_json(output / "candidate_completed.json") != result:
        raise ValueError("Original completed Apply differs from its shared publication")
    # Atomic publication may retain identical temporary copies after retries.
    for filename in _files(root / "done"):
        if filename == APPLY_RUN + ".json" or Path(filename).stem in plan["runs"]:
            continue
        match = re.fullmatch(r"\.(.+)\.json\.[0-9a-f]{32}\.partial", filename)
        original = None if match is None else root / "done" / (match[1] + ".json")
        if (match is None or match[1] not in {APPLY_RUN, *plan["runs"]}
                or not original.is_file()
                or (root / "done" / filename).read_bytes() != original.read_bytes()):
            raise ValueError("Early Build recovery refuses other completed work")

    from .store_map import input_reference
    from .pipeline.series_setup_jobs import _manifest
    if series_refresh is None:
        _, _, roots = _manifest(plan["series_manifest"])
    else:
        from .setup_plan_refresh import original_manifest_roots
        roots = original_manifest_roots(plan, series_refresh)
    queued_runs = []
    for run in plan["runs"]:
        if run in retained_builds:
            continue
        base, _ = _control(root / "results" / run / "task.json")
        validate_setup_claim(base, job)
        expected_control = completed["build_controls"][str(root / "setup-controls/build" / run / "operation.json")]
        # The catalogue belongs to this immutable operation, not a new plan.
        expected_config = input_reference(expected_control["path"], roots,
            sha256=expected_control["sha256"], size_bytes=expected_control["size_bytes"], role="configuration")
        if base["operation"] != "build" or base["run"] != run or base["configuration"] != expected_config:
            raise ValueError("Early Build task differs from the original completed Apply controls")
        if (root / "done" / (run + ".json")).is_file():
            finished, counted, lineage = setup_terminal(root, base, job,
                read_json(root / "done" / (run + ".json")), reader.read)
            if counted or lineage or finished.get("operation") != "build" or run in failed_runs:
                raise ValueError("Completed Build has conflicting operation history")
            token = finished["attempt_id"]
            service = name(finished["actual_worker"]["machine"])
            claim = root / "claimed" / service / (run + ".json")
            if read_json(claim) != finished["task"]:
                raise ValueError("Completed Build claim differs from its original publication")
            progress = _progress(root / "leases" / (run + "." + token), token, run)
            for folder, filenames in {
                "claimed": {service + "/" + run + ".json"}, "publishing": {run},
                "decisions": {run + "." + token + ".json"},
                "leases": {run + "." + token + "/" + filename for filename in
                    {"owner.json", "claimed.json", "publication.json", "artifacts/completed.json",
                     "artifacts/products_index.json", "artifacts/metadata/operation_result.json"} | progress},
            }.items():
                expected[folder] |= filenames
            output = attempt(run, claim, progress, {"operation_result.json", "candidate_completed.json"},
                             success=True, completed_result=finished)
            if read_json(output / "candidate_completed.json") != finished:
                raise ValueError("Completed Build differs from its original worker result")
            row = output_record(run, finished, completed["build_controls"][
                str(root / "setup-controls/build" / run / "raw-build-request.json")])
            completed_builds.append(dict(row, metadata_root=str(root), source_job=descriptor(root / "job.json"),
                result=finished, attempt=attempts[-1]))
            continue
        if run not in failed_runs and (root / "queue" / (run + ".json")).is_file():
            if read_json(root / "queue" / (run + ".json")) != base:
                raise ValueError("Pending Build task differs from its original retained task")
            queued_runs.append(run)
            continue
        if run in failed_runs:
            failure, counted, times, _ = _failed_publication(
                root, base, root / "failed" / (run + ".json"), reader.read)
            if counted or times or failure["task"].get("operation") != "build":
                raise ValueError("Build recovery refuses published scientific progress")
            task, token = failure["task"], failure["attempt_id"]
            service = name(failure["actual_worker"]["machine"])
            claim = root / "claimed" / service / (run + ".json")
            if read_json(claim) != task:
                raise ValueError("Early Build claim differs from its original failure")
        else:
            claims = list((root / "claimed").glob("*/" + run + ".json"))
            if len(claims) != 1:
                raise ValueError("Stopped Build has no unique original claim")
            claim = claims[0]
            task = read_json(claim)
            validate_setup_claim(task, job)
            if not _task_identity(base, task):
                raise ValueError("Stopped Build differs from its original task")
            token, service = task["attempt_id"], name(claim.parent.name)
            owner = read_json(root / "leases" / (run + "." + token) / "owner.json")
            _owner_machine(owner, base, token)
            if owner.get("task") != task or owner.get("machine") != service:
                raise ValueError("Stopped Build differs from its original owner")
        lease = root / "leases" / (run + "." + token)
        published = run in failed_runs
        progress = _progress(lease, token, run, required=published)
        lease_files = {"owner.json", "claimed.json"} | progress
        if published:
            lease_files |= {"publication.json", "failure_artifacts/failed.json"}
        for folder, filenames in {
            "claimed": {service + "/" + run + ".json"}, "publishing": {run} if published else set(),
            "decisions": {run + "." + token + ".json"} if published else set(),
            "leases": {run + "." + token + "/" + filename for filename in
                lease_files},
        }.items():
            expected[folder] |= filenames
        output = attempt(run, claim, progress, {"failed.json"} if published else set())
        if published and (output / "failed.json").read_bytes() != (root / "failed" / (run + ".json")).read_bytes():
            raise ValueError("Early Build failure differs from its retained output")
    expected["queue"] = {run + ".json" for run in queued_runs} | _pending_image_preparation(root, job, plan)
    expected["done"] = set(_files(root / "done"))
    operational = {}
    for folder, names in expected.items():
        files = _files(root / folder)
        if set(files) != names:
            raise ValueError("Early Build recovery found other work in " + folder)
        operational[folder] = files
    return dict(format=FORMAT, attempt_id=attempts[0]["attempt_id"],
        claim=attempts[0]["claim"], original_output=attempts[0]["original_output"],
        output_files=attempts[0]["output_files"], attempts=attempts,
        operational=operational, completed_apply=completed, unapplied_export=None,
        queued_runs=queued_runs, completed_builds=completed_builds)
