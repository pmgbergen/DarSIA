"""Retain exclusively owned RAW output before Build qualification/completion."""
from pathlib import Path
import re

from .job_management import _identity
from .pipeline.job_queue import descriptor, read_json
from .rig_raw_build import CERTIFICATE_NAME
from .rig_raw_write import _canonical


def snapshot(directory):
    """Metadata-only identity snapshot; no RAW/image payload is read."""
    directory = Path(directory)
    directories = {".": list(_identity(directory, directory=True))}
    files = {}
    for path in directory.rglob("*"):
        key = path.relative_to(directory).as_posix()
        if path.is_dir():
            directories[key] = list(_identity(path, directory=True))
        else:
            files[key] = list(_identity(path, directory=False))
        if len(directories) + len(files) > 8192:
            raise ValueError("Incomplete Build exceeds the bounded recovery file inventory")
    return dict(directories=directories, files=files)


def review(root, plan, failed_start):
    if not failed_start or not failed_start.get("completed_apply"):
        return []
    attempted = {row.get("run") for row in failed_start.get("attempts", ())}
    controls = failed_start["completed_apply"]["build_controls"]
    result = []
    for build in plan["build_plans"]:
        run = build["run_id"]
        if run in {row["run"] for row in failed_start.get("completed_builds", ())}:
            continue
        authority = build["request_template"]["output"]
        source = Path(authority["destination"])
        if not source.exists():
            continue
        if run not in attempted or source != _canonical(source):
            raise ValueError("Existing Build is not this exact failed/stopped operation")
        intended = _canonical(authority["intended_root"])
        if source == intended or not source.is_relative_to(intended):
            raise ValueError("Incomplete Build escaped its original output root")
        if (source / CERTIFICATE_NAME).exists():
            raise ValueError("A qualified RAW Build is preserved and cannot be reset by code update")
        marker = source / "raw_build_owner.json"
        if not marker.is_file():
            raise ValueError("Incomplete Build has no original RAW output owner")
        _identity(marker, directory=False)
        owner = read_json(marker)
        request = controls[str(Path(root) / "setup-controls/build" / run / "raw-build-request.json")]
        if (owner.get("format") != "darsia-owned-raw-output-v1"
                or owner.get("request") != request or owner.get("destination") != str(source)
                or Path(owner.get("intended_root", "")) != intended
                or owner.get("directory_identity") != list(_identity(source, directory=True))
                or not isinstance(owner.get("operation_id"), str)
                or re.fullmatch(r"[0-9a-f]{32}", owner["operation_id"]) is None):
            raise ValueError("Incomplete Build has no exact original RAW output owner")
        for filename in ("raw_setup_started.json", "raw_save_started.json"):
            path = source / filename
            if path.exists():
                _identity(path, directory=False)
                if read_json(path) != dict(operation_id=owner["operation_id"]):
                    raise ValueError("Incomplete Build stage belongs to another output owner")
        row = dict(run=run, source=str(source), owner=descriptor(marker),
                   request=request, snapshot=snapshot(source))
        saved_path = source / "raw_save_completed.json"
        if saved_path.exists():
            _identity(saved_path, directory=False)
            saved = read_json(saved_path)
            if (saved.get("format") != "darsia-owned-raw-save-completed-v1"
                    or saved.get("complete") is not True or saved.get("no_clobber_save") is not True
                    or saved.get("operation_id") != owner["operation_id"]
                    or saved.get("request") != request
                    or saved.get("destination") != str(source)
                    or saved.get("directory_identity") != owner["directory_identity"]
                    or not (source / "raw_setup_started.json").is_file()
                    or not (source / "raw_save_started.json").is_file()):
                raise ValueError("Unqualified RAW save lacks its exact original operation ownership")
            # Rig.save is an inner stage. Without its native qualifier and
            # outer Build completion, preserve it as an unfinished attempt;
            # the original full producer must run again at the fresh target.
            row["completed_save"] = descriptor(saved_path)
        result.append(row)
    return result


def verify(row, *, directory=None):
    path = Path(row["source"] if directory is None else directory)
    if snapshot(path) != row["snapshot"]:
        raise ValueError("Incomplete Build changed during code update")
    marker = path / "raw_build_owner.json"
    actual = descriptor(marker)
    if any(actual[key] != row["owner"][key] for key in ("sha256", "size_bytes")):
        raise ValueError("Incomplete Build owner changed during code update")
    if row.get("completed_save") is not None:
        actual = descriptor(path / "raw_save_completed.json")
        if any(actual[key] != row["completed_save"][key] for key in ("sha256", "size_bytes")):
            raise ValueError("Unqualified RAW save changed during code update")
