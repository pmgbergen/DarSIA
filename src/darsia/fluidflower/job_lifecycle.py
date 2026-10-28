"""Explicit same-job recovery and conservative user-facing execution states."""
from pathlib import Path
import json
import time
import uuid

from .job_control import _bindings, publish_control, read_control
from .job_launch_state import read_launch_requests, _epoch
from .job_management import JobActionLock, _retirement_binding, require_inactive
from .pipeline.job_queue import (JobQueue, descriptor, name, read_json, read_decision,
                                 retired_claim_evidence, decide, utc, write_immutable)


def require_service_retirement(root):
    """A declaration/launcher is never evidence that a service has retired."""
    if any((Path(root) / "ui/launches").glob("*.local-stop.json")):
        from .job_local_stop import verified_local_stop
        verified_local_stop(root)
        return read_launch_requests(root)
    require_inactive(root)
    requests = read_launch_requests(root)
    if any(not row.get("terminal_report") and not row.get("preparation_failure")
           and row.get("state") != "Launcher did not create a process" for row in requests):
        raise ValueError("Waiting for the original worker service to report stopping; an operator declaration is insufficient.")
    return requests


def _claims(root):
    for path in sorted((Path(root) / "claimed").glob("*/*.json")):
        if ".invalid." not in path.name and ".duplicate." not in path.name:
            task = read_json(path)
            decision = read_decision(Path(root), task["run"], task["attempt_id"])
            if decision is None or decision["outcome"] == "reclaimed":
                yield path, task, decision


def recover_job(root, *, machine, expected_previous, stores_path=None, reset=False,
                publisher=publish_control):
    """Prepare Pending or a verified unfinished-work retry; never launch a worker.

    Completed products and all original job/source/code bindings are immutable.
    A full rewind with saved checkpoints/products requires a fresh prepared job.
    """
    root = Path(root)
    with JobActionLock(root):
        control = read_control(root)
        if control.get("command") != expected_previous:
            raise ValueError("Job control changed; refresh before recovering this job")
        reference, master_format, _ = _bindings(root)
        master = read_json(root / "ui" / "master.json")
        if (master.get("format") != master_format or master.get("machine") != machine
                or master.get("job") != reference(root / "job.json")):
            raise ValueError("Only the job's recorded master machine can recover it")
        requests = require_service_retirement(root)
        claims = list(_claims(root))
        if list((root / "failed").glob("*.json")):
            raise ValueError("A run has a published failure. Its publication and history are preserved; prepare a new job to retry that scientific failure.")
        if reset and (claims or any((root / "done").glob("*.json"))
                      or any((root / "reclaimed").glob("*/*/reclaimed.json"))):
            raise ValueError("Reset to Pending requires an unstarted job with no saved work. Use Restart unfinished work to retain completed work and checkpoints; a full rewind requires a new prepared job.")
        # Preflight every original claim before fencing or requeueing any one.
        retired = []
        for path, task, decision in claims:
            binding = _retirement_binding(root, path, stores_path)
            evidence = retired_claim_evidence(root, path, local_output=binding,
                expected_stop=expected_previous, expected_action=control["action"])
            retired.append((path, task, binding, evidence))
        for path, task, binding, evidence in retired:
            if retired_claim_evidence(root, path, local_output=binding,
                    expected_stop=expected_previous, expected_action=control["action"]) != evidence:
                raise ValueError("Original claim retirement changed during recovery")
            won, decision = decide(root, task["run"], task["attempt_id"], "reclaimed",
                next_attempt=uuid.uuid4().hex, reason="Explicit GUI restart after original owned-process retirement",
                original_retirement=evidence)
            if not won and decision["outcome"] != "reclaimed":
                raise ValueError("Another terminal decision owns this claim")
        if claims or any((root / "reclaimed").glob("*/*/lease.json")):
            # Standing reclaim decisions waive only the timer, never retirement,
            # prefix validation, source identity or the original queue fencing.
            JobQueue(root, machine_id=machine).reclaim_stale()
        if not list((root / "queue").glob("*.json")):
            raise ValueError("There is no pending unfinished work to restart")
        # Reset must also hold an already-published job away from polling
        # services. Only an explicit subsequent Start/Restart releases this.
        command = publisher(root, "pause" if reset else "run", machine=machine,
                            expected_previous=expected_previous)
        current = read_control(root)
        write_immutable(root / "ui" / "recovery" / f"{current['sequence']:020d}.json", dict(
            format="darsia-job-ui-recovery-v1", job=reference(root / "job.json"), command=command,
            sequence=current["sequence"], action="reset" if reset else "restart", requested_utc=utc(),
            previous_launches=sorted(row["source"]["sha256"] for row in requests)))
        return command


def recovery_state(root, *, control=None):
    """Read bound recovery intent/history without changing admission or liveness."""
    root = Path(root)
    control = read_control(root) if control is None else control
    reference, _, _ = _bindings(root)
    previous = set()
    intent = None
    for path in sorted((root / "ui" / "recovery").glob("*.json")):
        value = read_json(path)
        sequence = value.get("sequence")
        if (value.get("format") != "darsia-job-ui-recovery-v1"
                or value.get("job") != reference(root / "job.json")
                or type(sequence) is not int or not 0 < sequence <= control["sequence"]
                or value.get("command") != reference(root / "ui" / "commands" / f"{sequence:020d}.json")):
            raise ValueError("Recovery receipt differs from its immutable job/control chain")
        previous = set(value["previous_launches"])
        intent = value if sequence == control["sequence"] else None
    return dict(previous_launches=previous, intent=intent)


def _reported_remote_runs(root, claims, local_machine, requests, now):
    """Fresh original attempt progress is a report, never a remote liveness proof."""
    reported = set()
    if not local_machine:
        return reported
    local_services = {name(local_machine + "-cpu"), name(local_machine + "-cuda")}
    for claimed, task, decision in claims:
        service = name(claimed.parent.name)
        if decision is not None or service in local_services:
            continue
        lease = root / "leases" / (task["run"] + "." + task["attempt_id"])
        try:
            owner_path = lease / "owner.json"
            owner_bound = descriptor(owner_path)
            owner = read_json(owner_path)
            if (owner.get("task") != task or owner.get("token") != task["attempt_id"]
                    or name(owner.get("actual_owner", {}).get("machine", "")) != service):
                continue
            paths = sorted(lease.glob("progress_*.json"))
            if not paths:
                continue
            path = paths[-1]
            bound = descriptor(path)
            progress = read_json(path)
            stamp = _epoch(progress.get("written_utc"))
            sequence = progress.get("sequence")
            if (progress.get("run") != task["run"] or progress.get("attempt_id") != task["attempt_id"]
                    or type(sequence) is not int or sequence < 0 or path.name != f"progress_{sequence:08d}.json"
                    or stamp is None or not 0 <= now - stamp <= 120
                    or progress.get("event", {}).get("state") not in {
                        "frame_started", "frame_completed", "phase1_frame_completed", "setup_progress"}):
                continue
            if any(row["machine"] + "-" + row["backend"] == service
                   and (_epoch((row.get("terminal_report") or {}).get("timestamp")) or 0) >= stamp
                   for row in requests):
                continue
            if descriptor(owner_path) == owner_bound and descriptor(path) == bound:
                reported.add(task["run"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return reported


def user_status(snapshot, *, local_machine=None, stores_path=None, now=None):
    """Normalize display only. This grants no launch or execution admission."""
    result = dict(snapshot)
    root = Path(snapshot["root"])
    control = read_control(root)
    recovery = recovery_state(root, control=control)
    previous = recovery["previous_launches"]
    requests = snapshot.get("launch_requests")
    if requests is None:
        requests = read_launch_requests(root)
    current_requests = [row for row in requests if row["source"]["sha256"] not in previous]
    active_runs = set()
    claims = list(_claims(root))
    reported_runs = _reported_remote_runs(root, claims, local_machine, requests,
                                         time.time() if now is None else now)
    # Only a matching actual local process incarnation establishes Running.
    if local_machine:
        from .pipeline.worker import _same_process
        for path, task, decision in claims:
            if decision is not None or name(path.parent.name) not in {name(local_machine + "-cpu"), name(local_machine + "-cuda")}:
                continue
            try:
                owner = read_json(root / "leases" / (task["run"] + "." + task["attempt_id"]) / "owner.json")
                actual = owner.get("actual_owner", {})
                if (owner.get("task") == task and name(actual.get("machine", "")) == name(path.parent.name)
                        and _same_process(actual) is not None):
                    active_runs.add(task["run"])
            except (OSError, ValueError, KeyError):
                pass
    retired, retirement_reason = False, ""
    if control["action"] in {"stop", "pause"}:
        try:
            require_service_retirement(root)
            local_confirmation = None
            if any((root / "ui/launches").glob("*.local-stop.json")):
                from .job_local_stop import verified_local_stop, verified_local_stop_claim
                local_confirmation = verified_local_stop(root, machine=local_machine, stores_path=stores_path)
            for path, task, decision in claims:
                if decision is None:
                    if local_confirmation is not None:
                        verified_local_stop_claim(root, path, confirmation=local_confirmation,
                            expected_stop=control.get("command"))
                    else:
                        retired_claim_evidence(root, path,
                            local_output=_retirement_binding(root, path, stores_path),
                            expected_stop=control.get("command"), expected_action=control["action"])
            retired = True
        except (OSError, ValueError, KeyError) as error:
            retirement_reason = str(error)
    failed_start = any(row.get("preparation_failure") or row.get("error_traceback")
        or (row.get("terminal_report") or {}).get("event", {}).get("kind") == "startup_failure"
        for row in current_requests)
    unresolved = any(not row.get("terminal_report") and not row.get("preparation_failure")
        and row.get("state") != "Launcher did not create a process" for row in current_requests)
    for log in snapshot.get("logs", []):
        latest = None
        for line in log.get("tail", "").splitlines():
            try:
                event = json.loads(line.partition(" ")[2])
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("kind") in {"started", "stopped", "startup_failure"}:
                latest = event["kind"]
        unresolved |= latest == "started"
    if unresolved and not retirement_reason:
        retirement_reason = "Waiting for the original service to report stopping; a launcher or operator declaration does not establish retirement."
    reset_pending = (control["action"] == "pause" and retired and not claims
                     and (recovery["intent"] or {}).get("action") == "reset")
    state = ("Completed" if snapshot.get("complete") else "Failed" if snapshot.get("errors")
        else ("Cancelled" if retired else "Cancelling") if control["action"] == "stop"
        else "Running" if active_runs else "Running (reported)" if reported_runs
        else "Failed" if snapshot.get("failed_runs") or failed_start
        else "Pending" if reset_pending
        else "Paused" if control["action"] == "pause" and retired
        else "Starting" if unresolved or claims else "Pending")
    result.update(state=state, status_detail=retirement_reason, pending_reset=reset_pending,
                  recovery_available=retired or not unresolved and not claims,
                  reset_available=not claims and not snapshot.get("completed_runs")
                      and not snapshot.get("committed") and not snapshot.get("completed_operations"))
    rows = []
    for original in snapshot.get("runs", []):
        row = dict(original, detail_state=original.get("state"))
        row["state"] = ("Completed" if original.get("state") in {"Complete", "Completed"}
            else "Failed" if original.get("state") in {"Failed", "Metadata error"}
            else "Running" if original.get("run") in active_runs
            else "Running (reported)" if original.get("run") in reported_runs
            else state if state in {"Cancelled", "Cancelling", "Paused"}
            else "Starting" if str(original.get("state", "")).startswith(("Claim", "Publishing", "Working"))
            else "Pending")
        rows.append(row)
    result["runs"] = rows
    return result
