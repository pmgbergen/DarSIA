"""Explicit GUI verification of a stopped local Setup process family.

This is a later OS observation, not a worker retirement or scientific result.
It applies only to the unchanged Setup generation it records. Completed
geometry remains authoritative when only its failed image operation is retried.
"""
from datetime import datetime
import hashlib
import json
import math
import ntpath
from pathlib import Path
import re
import uuid

from .pipeline.job_queue import descriptor, name, read_json, utc, write_immutable

FORMAT = "darsia-job-local-process-stop-v1"


def _key(path):
    return ntpath.normcase(ntpath.normpath(str(path)))


def _epoch(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError, TypeError):
        raise ValueError("Original process time or launch time is missing") from None


def _identity(value):
    if (not isinstance(value, dict) or type(value.get("pid")) is not int or value["pid"] <= 0
            or not isinstance(value.get("process_create_time"), str)):
        raise ValueError("Original local process PID and creation time are required")
    birth = float(value["process_create_time"])
    if not math.isfinite(birth) or birth <= 0:
        raise ValueError("Original local process creation time is invalid")
    return {key: value[key] for key in ("machine", "pid", "process_create_time") if key in value}


def _read(pin):
    path = Path(pin["path"])
    if descriptor(path) != pin:
        raise ValueError("Original local-stop source changed: " + str(path))
    return read_json(path)


def _terminal(console, requested):
    last = None
    for line in console.decode("utf-8", errors="replace").splitlines():
        stamp, _, payload = line.partition(" ")
        try:
            event = json.loads(payload)
        except ValueError:
            continue
        if (event.get("kind") in {"started", "stopped", "startup_failure"}
                and _epoch(stamp) >= requested):
            last = event["kind"]
    return last in {"stopped", "startup_failure"}


def _observe_absent(root, identities, supervisors):
    from .pipeline.worker import _same_process
    for identity in identities:
        if _same_process(identity) is not None:
            raise ValueError("An original local worker or supervisor is still running")
    # Also reject unclaimed children or a new service for this exact job. This
    # never kills a process and never treats a reused PID as the old process.
    import psutil
    observer = psutil.Process().username()
    for process in psutil.process_iter(["pid", "username", "cmdline"]):
        info = process.info
        if info.get("username") != observer:
            continue
        arguments = info.get("cmdline") or []
        for position, value in enumerate(arguments[:-1]):
            if value == "--job" and _key(arguments[position + 1]) == _key(root):
                raise ValueError("A local service for this job is still running")
        text = " ".join(arguments)
        if any(re.search(r"parent_pid\s*=\s*" + str(pid) + r"\b", text) for pid in supervisors):
            raise ValueError("An original supervisor still has a live local child")


def _capture(root, *, machine, stores_path=None):
    from .job_control import _bindings, read_control
    from .job_launch_state import read_launch_requests
    from .pipeline.series_setup_protocol import validate_setup_claim
    from .published_jobs_snapshot import _owner_machine
    from .setup_failed_build import review
    from .job_management import _retirement_binding
    from .store_map import read_store_map
    root = Path(root).resolve(strict=True)
    if name(read_store_map(stores_path).machine_id) != name(machine):
        raise ValueError("Local process verification must run on the original machine")
    job = read_json(root / "job.json")
    if job.get("pipeline") != "series-setup":
        raise ValueError("Local stop verification supports only the original Setup workflow")
    reference, master_format, _ = _bindings(root)
    master = read_json(root / "ui/master.json")
    control = read_control(root)
    if (master.get("format") != master_format or master.get("job") != reference(root / "job.json")
            or name(master.get("machine")) != name(machine) or control["action"] not in {"stop", "pause"}):
        raise ValueError("The original local master must request Stop or Pause first")
    plan = read_json(root / "setup-controls/plan.json")
    image_retry = False
    if any(Path(build["request_template"]["output"]["destination"]).exists() for build in plan["build_plans"]):
        from .pipeline.setup_image_preparation import IMAGE_RUN
        claims = list((root / "claimed").glob("*/" + IMAGE_RUN + ".json"))
        if (IMAGE_RUN not in job["expected_runs"] or len(claims) != 1
                or not (root / "failed" / (IMAGE_RUN + ".json")).is_file()):
            raise ValueError("Local stop recovery refuses an existing scientific Build destination")
        task = read_json(claims[0])
        if not (root / "leases" / (IMAGE_RUN + "." + task["attempt_id"]) / "owner.json").is_file():
            raise ValueError("Local image stop verification requires its original published owner")
        from .setup_failed_image_preparation import review as review_image
        # This existing reader authenticates every completed native Build and
        # the owned image failure without requiring a service terminal report.
        recovery = review_image(root, job, plan, stores_path)
        image_retry = True
    else:
        # Missing worker receipts confer no permission to discard saved work.
        recovery = review(root, job, plan, stores_path, allow_missing_retirement=True)
    attempts = list(recovery["attempts"])
    for run, row in recovery.get("geometry_operations", {}).items():
        if Path(row["metadata_root"]).resolve() == root:
            result = row["result"]
            claim = root / "claimed" / name(result["actual_worker"]["machine"]) / (run + ".json")
            attempts.append(dict(claim=str(claim),
                original_output=str(_retirement_binding(root, claim, stores_path).path)))
    requests = read_launch_requests(root)
    if not requests:
        raise ValueError("No original local launch is recorded")
    sources = [descriptor(root / "job.json"), descriptor(root / "ui/master.json")]
    launch_rows = []
    for request in requests:
        launch = request.get("launcher_receipt") or {}
        if (request.get("kind") != "local" or name(request["machine"]) != name(machine)
                or launch.get("format") != "darsia-job-local-launch-v2"
                or launch.get("entrypoint") != "python -B -m darsia_service.prestart -- python -m darsia.fluidflower.worker"
                or launch.get("started") is not True or launch.get("launcher_pid") is not None):
            raise ValueError("The original local launch lacks the supported process binding")
        console = Path(request["console"])
        raw = console.read_bytes()
        pin = dict(path=str(console), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
        sources += [request["source"], launch["source"], pin]
        launch_rows.append(dict(request=request["source"], requested=_epoch(request["requested_utc"]),
                                workers=request["workers"], backend=request["backend"],
                                terminal=_terminal(raw, _epoch(request["requested_utc"])), claims=[]))
    launch_rows.sort(key=lambda row: row["requested"])
    identities, supervisors, claims = {}, set(), {}
    for attempt in attempts:
        claim = Path(attempt["claim"])
        task = read_json(claim)
        validate_setup_claim(task, job)
        owner_path = root / "leases" / (task["run"] + "." + task["attempt_id"]) / "owner.json"
        owner = read_json(owner_path)
        _owner_machine(owner, task, task["attempt_id"])
        actual = _identity(owner["actual_owner"])
        if actual.get("machine") != name(machine + "-" + task["backend"]):
            raise ValueError("A retained attempt belongs to a different machine")
        matching = [row for row in launch_rows if row["backend"] == task["backend"]
                    and row["requested"] <= float(actual["process_create_time"])]
        if not matching:
            raise ValueError("Original worker has no matching local launch")
        launch = matching[-1]
        launch["claims"].append(task["run"])
        claims[task["run"]] = dict(claim=descriptor(claim), owner=descriptor(owner_path),
            attempt_id=task["attempt_id"], actual_worker=actual)
        sources += [claims[task["run"]]["claim"], claims[task["run"]]["owner"]]
        identities[(actual["pid"], actual["process_create_time"])] = actual
        original_retirement = Path(attempt["original_output"]) / "retirement.json"
        if original_retirement.is_file():
            pin = descriptor(original_retirement)
            retired = _read(pin)
            if (retired.get("run") != task["run"] or retired.get("attempt_id") != task["attempt_id"]
                    or retired.get("complete") is not True or retired.get("all_absent") is not True
                    or not any(_identity(item) == actual for item in retired.get("identities", []))):
                raise ValueError("Retained supervisor receipt differs from the original claim")
            supervisor = _identity(retired.get("supervisor"))
            if (supervisor.get("machine") != actual["machine"]
                    or not launch["requested"] <= float(supervisor["process_create_time"]) <= float(actual["process_create_time"])):
                raise ValueError("Original supervisor belongs to a different launch")
            sources.append(pin)
            launch["supervisor"] = supervisor
            supervisors.add(supervisor["pid"])
            for original in [supervisor, *retired["identities"]]:
                item = _identity(original)
                identities[(item["pid"], item["process_create_time"])] = item
    for row in launch_rows:
        # Requested capacity is an upper bound. Once every geometry operation
        # is complete, this typed final operation can have only one claimant.
        required = (1 if image_retry and row["claims"] == [IMAGE_RUN] else row["workers"])
        if not row["terminal"] and ("supervisor" not in row or len(row["claims"]) < required):
            raise ValueError("The interrupted service lacks original supervisor/worker-family identities")
    values = sorted(identities.values(), key=lambda row: (row["pid"], row["process_create_time"]))
    _observe_absent(root, values, supervisors)
    return dict(format=FORMAT, job=descriptor(root / "job.json"), observer_machine=name(machine),
        stop=control["command"], stop_sequence=control["sequence"], claims=claims,
        identities=values, supervisors=sorted(supervisors), sources=sources,
        requests=[row["request"] for row in launch_rows],
        completed_apply=recovery["completed_apply"]["coordinator"],
        scientific_completion_claimed=False, worker_retirement_claimed=False)


def review_local_stop(root, *, machine, stores_path=None):
    return _capture(root, machine=machine, stores_path=stores_path)


def _verify(root, document, *, machine=None, observe=True):
    from .job_control import read_control
    root = Path(root).resolve(strict=True)
    if (document.get("format") != FORMAT or document.get("job") != descriptor(root / "job.json")
            or document.get("scientific_completion_claimed") is not False
            or document.get("worker_retirement_claimed") is not False
            or machine is not None and name(machine) != document.get("observer_machine")):
        raise ValueError("Local stop confirmation belongs to another job or machine")
    for pin in document["sources"]:
        if descriptor(Path(pin["path"])) != pin:
            raise ValueError("Original local-stop source has changed")
    identities = {(item["pid"], item["process_create_time"]): _identity(item)
                  for item in document["identities"]}
    if not identities or not document["claims"]:
        raise ValueError("Local stop confirmation has no original worker identities")
    expected_identities = {}
    for run, row in document["claims"].items():
        task, owner = _read(row["claim"]), _read(row["owner"])
        actual = _identity(owner["actual_owner"])
        if (task["run"] != run or task["attempt_id"] != row["attempt_id"]
                or owner.get("task") != task or row["actual_worker"] != actual
                or row["claim"] not in document["sources"] or row["owner"] not in document["sources"]):
            raise ValueError("Local stop confirmation changed its original worker binding")
        expected_identities[(actual["pid"], actual["process_create_time"])] = actual
    supervisors = set()
    for pin in document["sources"]:
        if Path(pin["path"]).name != "retirement.json":
            continue
        retired = _read(pin)
        row = document["claims"].get(retired.get("run"))
        if row is None or row["attempt_id"] != retired.get("attempt_id"):
            raise ValueError("Local stop confirmation names another original supervisor")
        supervisor = _identity(retired["supervisor"])
        supervisors.add(supervisor["pid"])
        for item in [supervisor, *retired["identities"]]:
            actual = _identity(item)
            expected_identities[(actual["pid"], actual["process_create_time"])] = actual
    if identities != expected_identities or set(document["supervisors"]) != supervisors:
        raise ValueError("Local stop confirmation omits or changes original process identities")
    expected = {row["claim"]["path"] for row in document["claims"].values()}
    if {str(path) for path in (root / "claimed").glob("*/*.json")} != expected:
        raise ValueError("The job has new or different process claims")
    expected_requests = {pin["path"] for pin in document["requests"]}
    if {str(path) for path in (root / "ui/launches").glob("*.request.json")} != expected_requests:
        raise ValueError("The job has new or different launches")
    control = read_control(root)
    if control["sequence"] < document["stop_sequence"] or control["action"] not in {"stop", "pause"}:
        raise ValueError("The job no longer has the verified Stop/Pause request")
    for path in (root / "ui/commands").glob("*.json"):
        command = read_json(path)
        if command["sequence"] > document["stop_sequence"] and command["action"] == "run":
            raise ValueError("A later Run command invalidated the local stop confirmation")
    if control["sequence"] == document["stop_sequence"] and control["command"] != document["stop"]:
        raise ValueError("The verified stop command changed")
    if observe:
        _observe_absent(root, document["identities"], document["supervisors"])
    return document


def verified_local_stop(root, *, machine=None, stores_path=None, observe=True):
    root = Path(root).resolve(strict=True)
    paths = sorted((root / "ui/launches").glob("*.local-stop.json"))
    if not paths:
        raise FileNotFoundError("No verified local process-stop confirmation is recorded")
    if observe:
        from .store_map import read_store_map
        actual_machine = name(read_store_map(stores_path).machine_id)
        if machine is not None and name(machine) != actual_machine:
            raise ValueError("Local process verification must run on the original machine")
        machine = actual_machine
    path = max(paths, key=lambda candidate: (_epoch(read_json(candidate).get("observed_utc")), candidate.name))
    pin = descriptor(path)
    document = _verify(root, _read(pin), machine=machine, observe=observe)
    return dict(document=document, source=pin, identities=document["identities"], claims=document["claims"])


def verified_local_stop_claim(root, claimed, *, machine=None, stores_path=None,
                             expected_stop=None, confirmation=None, observe=True):
    from .job_control import read_control
    result = confirmation or verified_local_stop(root, machine=machine, stores_path=stores_path, observe=observe)
    task = read_json(claimed)
    row = result["claims"].get(task["run"])
    control = read_control(root)
    if (row is None or row["claim"] != descriptor(Path(claimed))
            or expected_stop is not None and control["command"] != expected_stop):
        raise ValueError("The local process confirmation does not bind this claim and stop request")
    return dict(format="darsia-job-local-process-stop-claim-v1", confirmation=result["source"],
        job=result["document"]["job"], claim=row["claim"], owner=row["owner"],
        run=task["run"], attempt_id=task["attempt_id"], actual_worker=row["actual_worker"],
        identities=result["identities"], stop=control["command"], all_absent=True)


def confirm_local_stop(root, *, machine, actor, stores_path=None):
    from .job_management import JobActionLock
    if not str(actor).strip():
        raise ValueError("A named operator is required")
    root = Path(root).resolve(strict=True)
    with JobActionLock(root):
        document = _capture(root, machine=machine, stores_path=stores_path)
        _verify(root, document, machine=machine)
        document.update(actor=str(actor).strip(), observed_utc=utc())
        path = root / "ui/launches" / (uuid.uuid4().hex + ".local-stop.json")
        write_immutable(path, document)
    # No result or worker-retirement record is invented. This original
    # job-owned capacity cleanup consumes the separately typed observation.
    from .pipeline.series_setup_worker_admission import release_local_stopped_setup_capacity
    release_local_stopped_setup_capacity(root, machine=machine, stores_path=stores_path)
    return descriptor(path)


def apply_confirmations(root, requests):
    """Project saved process evidence into display state, never invent a log."""
    root = Path(root)
    if not any((root / "ui/launches").glob("*.local-stop.json")):
        return requests
    proof = verified_local_stop(root, observe=False)
    accepted = {pin["path"] for pin in proof["document"]["requests"]}
    for row in requests:
        if row["source"]["path"] in accepted:
            row.update(released=True, state="Local processes verified stopped", local_stop_confirmation=proof["source"])
    return requests
