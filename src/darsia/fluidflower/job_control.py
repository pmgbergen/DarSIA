"""Append-only UI commands for the existing whole-run queue.

This is operational metadata, not another queue. Tasks, attempt leases,
publication and verified resume remain with pipeline.job_queue/checkpoint.
"""

from __future__ import annotations

from pathlib import Path

from .pipeline.job_queue import descriptor, read_json, utc, write_immutable

FORMAT = "darsia-job-ui-control-v1"
ALIAS_FORMAT = "darsia-job-ui-control-v2"
ACTIONS = {"run", "pause", "stop"}


def ensure_setup_master(job_root, *, machine, actor):
    """Enable existing controls on an explicit Setup Start, preserving owners."""
    root = Path(job_root)
    job = read_json(root / "job.json")
    if job.get("pipeline") != "series-setup" or not isinstance(machine, str) or not machine.strip():
        raise ValueError("An explicit Setup job and launch machine are required")
    reference, master_format, _control_format = _bindings(root)
    expected = reference(root / "job.json")
    path = root / "ui" / "master.json"
    if not path.exists():
        value = dict(format=master_format, job=expected, machine=machine, created_utc=utc(),
            actor=str(actor).strip(), reason="Enable controls for the explicit Setup start")
        try:
            write_immutable(path, value)
        except FileExistsError:
            pass  # The original immutable winner remains authoritative.
    master = read_json(path)
    if (master.get("format") != master_format or master.get("job") != expected
            or not isinstance(master.get("machine"), str) or not master["machine"].strip()):
        raise ValueError("Job UI master is not bound to this immutable Setup job")
    return reference(path)


def read_control(job_root):
    """Read a complete immutable command chain; no directory is created."""
    root = Path(job_root)
    directory = root / "ui" / "commands"
    paths = sorted(directory.glob("*.json"))
    if not paths:
        return {"action": "run", "sequence": 0}
    reference, master_format, control_format = _bindings(root)
    job = reference(root / "job.json")
    master_asset = reference(root / "ui" / "master.json")
    master = read_json(root / "ui" / "master.json")
    if master.get("format") != master_format or master.get("job") != job:
        raise ValueError("Job UI master is not bound to this immutable job")
    creator = master.get("machine")
    if not isinstance(creator, str) or not creator.strip():
        raise ValueError("Job UI master machine is missing")
    previous = None
    result = None
    for sequence, path in enumerate(paths, 1):
        if path.name != f"{sequence:020d}.json":
            raise ValueError("Job UI control sequence is not contiguous")
        value = read_json(path)
        if (
            value.get("format") != control_format
            or type(value.get("sequence")) is not int
            or value["sequence"] != sequence
            or value.get("action") not in ACTIONS
            or value.get("job") != job
            or value.get("master") != master_asset
            or value.get("machine") != creator
            or value.get("previous") != previous
        ):
            raise ValueError("Job UI command identity or chain differs: " + str(path))
        result = dict(value, command=reference(path))
        previous = result["command"]
    return result


def publish_control(job_root, action, *, machine, expected_previous):
    """Explicit master action only; refuse stale UI state and never overwrite.

    Caller obtains the actual host name and an immutable master.json when the
    job is created. Merely opening a Jobs page must never call this function.
    """
    if action not in ACTIONS:
        raise ValueError("Unknown job action")
    root = Path(job_root)
    master_path = root / "ui" / "master.json"
    master = read_json(master_path)
    reference, master_format, control_format = _bindings(root)
    if master.get("machine") != machine:
        raise ValueError("Only the job's recorded master machine writes commands")
    if (
        master.get("format") != master_format
        or master.get("job") != reference(root / "job.json")
    ):
        raise ValueError("Job UI master is not bound to this immutable job")
    current = read_control(root)
    if current.get("command") != expected_previous:
        raise ValueError("Job control changed; refresh before applying this action")
    sequence = current["sequence"] + 1
    document = dict(
        format=control_format,
        sequence=sequence,
        action=action,
        job=reference(root / "job.json"),
        master=reference(master_path),
        machine=machine,
        previous=current.get("command"),
        requested_utc=utc(),
    )
    # One sequence filename is an atomic winner if two UI instances race.
    path = root / "ui" / "commands" / f"{sequence:020d}.json"
    write_immutable(path, document)
    return reference(path)


def _bindings(root):
    """Only own-job immutable references change syntax; control rules do not."""
    job = read_json(root / "job.json")
    if job.get("format") == "darsia-alias-job-v1":
        from .storage_profile import expected_format_version
        # Lean jobs are version 2 (N55); full and legacy jobs version 1.
        if job.get("format_version") != expected_format_version(job):
            raise ValueError("Unknown alias job version")
        from .pipeline.alias_job import job_metadata_reference
        return (lambda path: job_metadata_reference(root, path)), "darsia-job-ui-master-v2", ALIAS_FORMAT
    return descriptor, "darsia-job-ui-master-v1", FORMAT
