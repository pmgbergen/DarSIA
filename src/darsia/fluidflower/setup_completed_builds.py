"""Keep original completed Build publications across a code-only job retry."""
from pathlib import Path
from copy import deepcopy

from .pipeline.job_queue import descriptor, read_json

POINTER = "ui/retained-builds.json"
FORMAT = "darsia-retained-completed-builds-v1"


def receipt_row(row, *, metadata_root=None):
    """Reference the original completion once; do not copy its full catalogue."""
    from .pipeline.series_setup_jobs import _control
    value = {key: deepcopy(item) for key, item in row.items()
             if key not in {"display", "task", "result"}}
    source = Path(row["metadata_root"] if metadata_root is None else metadata_root)
    result, result_source = _control(source / "results" / row["run"] / "completed.json")
    if result != row["result"]:
        raise ValueError("Retained Build completion changed before receipt publication")
    value.update(metadata_root=str(source), source_job=descriptor(source / "job.json"),
                 result_source=result_source)
    return value


def verify_output(row):
    """Recheck stable file identities, without reopening scientific arrays."""
    from .setup_failed_start import _version_files
    path = Path(row["output"])
    if _version_files(path) != row["output_versions"]:
        raise ValueError("A retained completed Build output changed: " + row["run"])


def output_record(run, result, request):
    from .setup_failed_start import _version_files
    from .pipeline.series_setup_jobs import _control
    from .job_management import _identity
    raw, _ = _control(Path(request["path"]))
    output = Path(raw["output"]["destination"])
    owner, _ = _control(output / "raw_build_owner.json")
    saved, _ = _control(output / "raw_save_completed.json")
    certificate, pin = _control(output / "baseline_native_renderer.json")
    if (result.get("raw_build_request") != request or result.get("baseline_renderer") != pin
            or owner.get("request") != request or saved.get("request") != request
            or saved.get("complete") is not True or saved.get("no_clobber_save") is not True
            or saved.get("format") != "darsia-owned-raw-save-completed-v1"
            or saved.get("operation_id") != owner.get("operation_id")
            or saved.get("destination") != str(output)
            or saved.get("directory_identity") != list(_identity(output, directory=True))
            or certificate.get("inputs", {}).get("operation_request") != request
            or certificate.get("outputs") != result.get("produced_files")):
        raise ValueError("Completed Build lacks its original output/save identity: " + run)
    return dict(run=run, output=str(output), output_versions=_version_files(output))


def retained(root, job=None):
    """Read only explicitly retained original publications for this generation."""
    from .setup_job_reprepare import validate_setup_reprepare_transition
    from .pipeline.series_setup_results import setup_terminal
    from .jobs_metadata import JobSnapshotReader
    root = Path(root)
    pointer = root / POINTER
    if not pointer.exists():
        return {}
    value = read_json(pointer)
    if value.get("format") != FORMAT:
        raise ValueError("Unknown retained Build publication record")
    receipt_pin = value["receipt"]
    if descriptor(Path(receipt_pin["path"])) != receipt_pin:
        raise ValueError("Retained Build recovery receipt changed")
    receipt = read_json(receipt_pin["path"])
    if receipt.get("job") != descriptor(root / "job.json"):
        raise ValueError("Retained Build receipt belongs to another job generation")
    history = Path(receipt_pin["path"]).parent
    previous = read_json(history / "previous/job.json")
    current = job or read_json(root / "job.json")
    validate_setup_reprepare_transition(previous, current, receipt, history_root=history)
    rows, reader = {}, JobSnapshotReader().read
    for row in receipt.get("retained_completed_builds", []):
        run = row["run"]
        source_root = Path(row["metadata_root"])
        if not source_root.resolve().is_relative_to((root / "history").resolve()):
            raise ValueError("Retained publication is outside the original job history")
        source_job_pin = row["source_job"]
        if descriptor(source_root / "job.json") != source_job_pin:
            raise ValueError("Retained Build source job changed")
        source_job = read_json(source_root / "job.json")
        base = read_json(source_root / "results" / run / "task.json")
        display = {}
        result, _, lineage = setup_terminal(source_root, base, source_job,
            read_json(source_root / "done" / (run + ".json")), reader, display=display)
        result_source = row.get("result_source")
        if result_source is not None:
            if descriptor(source_root / "results" / run / "completed.json") != result_source:
                raise ValueError("Retained Build original completion changed")
        elif result != row.get("result"):
            raise ValueError("Retained Build differs from its original completion")
        if (run in rows or result.get("operation") != "build" or lineage
                or source_job["setup_plan"] != current["setup_plan"]
                or source_job["series_id"] != current["series_id"]
                or _geometry_runs(source_job) != _geometry_runs(current)):
            raise ValueError("Retained Build does not match this unchanged Setup plan")
        verify_output(row)
        rows[run] = dict(row, display=display, task=base, result=result)
    return rows


def save_pointer(root, receipt):
    from .pipeline.job_queue import write_immutable
    value = read_json(receipt["path"])
    if value.get("retained_completed_builds"):
        write_immutable(Path(root) / POINTER, dict(format=FORMAT, receipt=deepcopy(receipt)))


def _geometry_runs(job):
    """Only a typed trailing image operation may extend unchanged geometry."""
    operations = job.get("setup_operations")
    if operations is None:
        return job["expected_runs"]
    return [row["run"] for row in operations if row["operation"] in {"apply", "build"}]


def retained_operations(root, job=None):
    """Original Apply/Build completions retained by an explicit image upgrade.

    Build-only callers keep using retained(); the Apply is never synthesized
    from exported files or approval flags.
    """
    rows = retained(root, job)
    root = Path(root)
    if not (root / POINTER).exists():
        return rows
    pointer = read_json(root / POINTER)
    receipt = read_json(pointer["receipt"]["path"])
    row = receipt.get("retained_setup_apply")
    if row is None:
        return rows
    from .pipeline.series_setup_results import setup_terminal
    from .jobs_metadata import JobSnapshotReader
    from .setup_failed_start import verify_completed_apply_snapshot
    current = job or read_json(root / "job.json")
    source = Path(row["metadata_root"])
    if (not source.resolve().is_relative_to((root / "history").resolve())
            or descriptor(source / "job.json") != row["source_job"]):
        raise ValueError("Retained Apply source differs from its original history")
    original = read_json(source / "job.json")
    if (original["setup_plan"] != current["setup_plan"]
            or original["series_id"] != current["series_id"]
            or _geometry_runs(original) != _geometry_runs(current)):
        raise ValueError("Retained Apply changed its original geometry authority")
    run = "setup-apply"
    base = read_json(source / "results" / run / "task.json")
    display = {}
    result, counted, lineage = setup_terminal(source, base, original,
        read_json(source / "done" / (run + ".json")), JobSnapshotReader().read, display=display)
    if (counted or lineage or result.get("operation") != "apply"
            or descriptor(source / "results" / run / "completed.json") != row["result_source"]):
        raise ValueError("Retained Apply differs from its original completion")
    snapshot = receipt.get("retained_apply")
    if snapshot is None:
        raise ValueError("Retained Apply lacks its original completed controls")
    verify_completed_apply_snapshot(root, snapshot)
    rows[run] = dict(row, display=display, task=base, result=result)
    return rows
