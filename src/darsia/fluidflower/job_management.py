"""Explicit inactive-job management. No worker, RAW or external-product cleanup."""
from __future__ import annotations

from pathlib import Path
from contextlib import ExitStack
import os
import stat
import uuid

from .job_launch_state import read_launch_requests
from .pipeline.job_queue import (descriptor, read_json, read_decision, LocalOutputBinding,
                                 retired_claim_evidence, cancel_retired_claim)
from .native_jobs_registry import _path
from .job_control import _bindings, read_control, publish_control


class UnfinishedClaimError(ValueError):
    """Deletion is blocked by resumable work; inactive jobs may be archived."""


class JobActionLock:
    """Serialize GUI Start reservations with explicit inactive-job management."""
    def __init__(self, root):
        self.path = Path(root) / ".job-management.lock"
        self.token = uuid.uuid4().hex
        self.held = False

    def __enter__(self):
        if not (self.path.parent / "job.json").is_file():
            raise ValueError("The selected job folder no longer exists")
        try:
            with self.path.open("x", encoding="ascii") as stream:
                stream.write(self.token)
        except FileExistsError as error:
            raise ValueError("Another job action is in progress") from error
        self.held = True
        return self

    def release(self):
        if self.held:
            if self.path.read_text(encoding="ascii") != self.token:
                raise ValueError("The job action lock changed; it is preserved")
            self.path.unlink()
            self.held = False

    def __exit__(self, *args):
        self.release()


def registered_job(registry, series_root, series_id, job_root):
    row = next((row for row in registry.context(series_root, series_id)["jobs"]
                if _path(row["root"]) == _path(job_root)), None)
    if row is None or descriptor(Path(job_root) / "job.json") != row["job"]:
        raise ValueError("The selected registered job metadata changed")
    return row


def require_inactive(job_root):
    root = Path(job_root)
    if any(not row["released"] for row in read_launch_requests(root)):
        raise ValueError("The job has a running or unresolved start. Stop it or resolve its previous service state first.")
    # Original worker logs also cover starts made outside this GUI registry.
    for path in (root / "logs").glob("*.log"):
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 128 * 1024))
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
        latest = None
        import json
        for line in lines:
            try:
                event = json.loads(line.partition(" ")[2])
            except ValueError:
                continue
            if event.get("kind") in {"started", "stopped", "startup_failure"}:
                latest = event["kind"]
        if latest == "started":
            raise ValueError("The job's worker service has not reported stopping")


def archive_job(registry, series_root, series_id, job_root, *, actor, archived=True):
    with JobActionLock(job_root):
        registered_job(registry, series_root, series_id, job_root)
        if archived:
            require_inactive(job_root)
        return registry.set_archived(series_root, series_id, job_root, archived, actor=actor)


def _identity(path, *, directory):
    info = path.lstat()
    if (getattr(info, "st_reparse_tag", 0) or getattr(info, "st_file_attributes", 0) & 0x400
            or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))):
        raise ValueError("Job deletion refuses linked or non-regular entries: " + str(path))
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) if not directory else (info.st_dev, info.st_ino)


def _uses_job(value, root, *, same_series):
    portable = "jobs/" + Path(root).name
    if isinstance(value, dict):
        relative = value.get("relative_path")
        if (same_series and value.get("root_alias") == "series" and isinstance(relative, str)
                and (relative == portable or relative.startswith(portable + "/"))):
            return True
        return any(_uses_job(item, root, same_series=same_series) for item in value.values())
    if isinstance(value, list):
        return any(_uses_job(item, root, same_series=same_series) for item in value)
    if isinstance(value, str):
        return _path(value) == _path(root) or _path(value).startswith(_path(root) + "\\")
    return False


def _affected_jobs(registry, job_root):
    """Follow only explicit registered job metadata, including indirect consumers."""
    rows = registry.known_jobs()
    documents = {}
    for row in rows:
        path = Path(row["root"]) / "job.json"
        if descriptor(path) != row["job"]:
            raise ValueError("Registered job metadata changed: " + row["root"])
        documents[_path(row["root"])] = read_json(path)
        if descriptor(path) != row["job"]:
            raise ValueError("Registered job metadata changed: " + row["root"])
    affected = [next(row for row in rows if _path(row["root"]) == _path(job_root))]
    included = {_path(job_root)}
    for source in affected:
        for row in rows:
            if _path(row["root"]) in included:
                continue
            same_series = Path(row["root"]).parent.resolve() == Path(source["root"]).parent.resolve()
            if _uses_job(documents[_path(row["root"])], source["root"], same_series=same_series):
                affected.append(row)
                included.add(_path(row["root"]))
    return affected


def stop_delete_review(registry, series_root, series_id, job_root):
    """Review Stop targets without reading payloads or changing any job."""
    registered_job(registry, series_root, series_id, job_root)
    root = Path(job_root)
    _identity(root, directory=True)
    from .series_paths import series_folder_path
    if root.resolve(strict=True).parent != series_folder_path(series_root, "jobs").resolve(strict=True) or root.name.startswith("."):
        raise ValueError("Delete is limited to a job folder directly inside this series' jobs folder")
    affected = []
    for original in _affected_jobs(registry, root):
        row = dict(original)
        selected = Path(row["root"])
        try:
            reference, master_format, _ = _bindings(selected)
            master = read_json(selected / "ui" / "master.json")
            if master.get("format") != master_format or master.get("job") != reference(selected / "job.json"):
                raise ValueError("The recorded control master belongs to another job")
            row.update(master=reference(selected / "ui" / "master.json"),
                       master_machine=master["machine"], control=read_control(selected))
        except (OSError, ValueError, KeyError) as error:
            row["stop_error"] = str(error)
        affected.append(row)
    return dict(root=str(root), affected=affected)


def _require_reviewed_jobs(registry, review):
    current = _affected_jobs(registry, review["root"])
    identity = lambda rows: {(_path(row["root"]), row["job"]["sha256"]) for row in rows}
    if identity(current) != identity(review["affected"]):
        raise ValueError("Affected jobs changed. Review Stop and delete again.")


def stop_for_deletion(registry, review, *, machine, actor):
    """Publish only the original Stop command after an explicit reviewed action."""
    if not str(actor).strip():
        raise ValueError("An action name is required")
    _require_reviewed_jobs(registry, review)
    # Preflight every consumer before publishing any command.
    for row in review["affected"]:
        if row.get("stop_error"):
            raise ValueError("Stop is unavailable for " + row["name"] + ": " + row["stop_error"])
        reference, _, _ = _bindings(Path(row["root"]))
        if reference(Path(row["root"]) / "ui" / "master.json") != row["master"]:
            raise ValueError("Recorded control master changed. Review Stop and delete again.")
        if read_control(row["root"]) != row["control"]:
            raise ValueError("Job control changed. Review Stop and delete again.")
        if row["control"]["action"] != "stop" and row["master_machine"] != machine:
            raise ValueError("Request Stop on the recorded master machine for " + row["name"] + ": " + row["master_machine"])
    for row in review["affected"]:
        if row["control"]["action"] != "stop":
            publish_control(row["root"], "stop", machine=machine,
                            expected_previous=row["control"].get("command"))
    return dict(review, stops={row["root"]: read_control(row["root"])["command"] for row in review["affected"]})


def _retirement_binding(root, claimed, stores_path):
    task = read_json(claimed)
    if task.get("format") != "darsia-alias-task-v1":
        return None
    from .store_map import read_store_map
    from .pipeline.store_runtime import local_attempt_output
    stores = read_store_map(stores_path)
    job = read_json(Path(root) / "job.json")
    output, admission = local_attempt_output(stores, job["series_id"], task["job_id"],
                                            task["run"], task["attempt_id"])
    return LocalOutputBinding(stores.machine_id, Path(stores.roots(job["series_id"])["results"]), output, admission)


def stop_delete_status(registry, review, *, stores_path=None):
    """Read original service/claim evidence; never infer retirement from Stop."""
    _require_reviewed_jobs(registry, review)
    waiting, claims = [], []
    for row in review["affected"]:
        root = Path(row["root"])
        control = read_control(root)
        if (control["action"] != "stop" or control.get("command") != review.get("stops", {}).get(row["root"])):
            raise ValueError("A reviewed job no longer requests Stop. Review Stop and delete again.")
        for request in read_launch_requests(root):
            if (not request.get("terminal_report") and not request.get("preparation_failure")
                    and request.get("state") != "Launcher did not create a process"):
                waiting.append(row["name"] + ": waiting for the original service to report stopping")
        try:
            require_inactive(root)
        except ValueError as error:
            waiting.append(row["name"] + ": " + str(error))
        for path in (root / "claimed").glob("*/*.json"):
            if ".invalid." in path.name or ".duplicate." in path.name:
                continue
            task = read_json(path)
            if read_decision(root, task["run"], task["attempt_id"]) is None:
                try:
                    binding = _retirement_binding(root, path, stores_path)
                    evidence = retired_claim_evidence(root, path, local_output=binding,
                        expected_stop=review["stops"][row["root"]])
                    claims.append(dict(root=str(root), path=str(path), evidence=evidence))
                except (OSError, ValueError, KeyError) as error:
                    waiting.append(row["name"] + ": waiting for original claim retirement: " + str(error))
    if waiting:
        return dict(ready=False, waiting=waiting, can_wait=True)
    return dict(ready=True, waiting=[], can_wait=False, claims=claims)


def deletion_plan(registry, series_root, series_id, job_root, *, lock_token=None,
                  retirement_review=None, stores_path=None):
    row = registered_job(registry, series_root, series_id, job_root)
    root, series = Path(job_root), Path(series_root)
    _identity(root, directory=True)
    actual = root.resolve(strict=True)
    from .series_paths import series_folder_path
    parent = series_folder_path(series, "jobs").resolve(strict=True)
    if actual.parent != parent or actual == parent or actual.name.startswith("."):
        raise ValueError("Delete is limited to a job folder directly inside this series' jobs folder")
    require_inactive(root)
    status = None
    if retirement_review is not None:
        if _path(retirement_review["root"]) != _path(root):
            raise ValueError("Retirement review belongs to another delete target")
        status = stop_delete_status(registry, retirement_review, stores_path=stores_path)
        if not status["ready"]:
            raise ValueError("\n".join(status["waiting"]))
    for other in registry.known_jobs():
        if _path(other["root"]) == _path(root):
            continue
        other_root = Path(other["root"])
        same_series = other_root.parent.resolve() == parent
        if _uses_job(read_json(other_root / "job.json"), root, same_series=same_series):
            if retirement_review is None or not any(_path(item["root"]) == _path(other_root) for item in retirement_review["affected"]):
                raise ValueError("Another registered job uses this job's files. Archive it instead: " + other["name"])
    lock_path = root / ".job-management.lock"
    if lock_path.exists() and (not lock_token or lock_path.read_text(encoding="ascii") != lock_token):
        raise ValueError("Another job action is in progress")
    for path in (root / "claimed").glob("*/*.json"):
        if ".invalid." in path.name or ".duplicate." in path.name:
            continue
        task = read_json(path)
        if read_decision(root, task["run"], task["attempt_id"]) is None:
            if status is not None and any(_path(item["path"]) == _path(path) for item in status["claims"]):
                continue
            raise UnfinishedClaimError(
                "Deletion is blocked because this job has an unfinished claim. "
                "Stopping a job preserves unfinished work so it can be resumed. "
                "Archive this job to hide it while keeping all files.")
    # Another local series association must not become a dangling registration.
    for path in registry.root.glob("*.json"):
        if path == registry._file(series_root, series_id):
            continue
        if any(_path(item["root"]) == _path(root) for item in read_json(path).get("jobs", [])):
            raise ValueError("This job is also registered in another series. Archive it instead.")
    files, directories = {}, {"": _identity(root, directory=True)}
    for current, names, leaves in os.walk(root, followlinks=False):
        current = Path(current)
        for name in names:
            child = current / name
            directories[child.relative_to(root).as_posix()] = _identity(child, directory=True)
        for name in leaves:
            child = current / name
            if child == lock_path and lock_token:
                continue
            files[child.relative_to(root).as_posix()] = _identity(child, directory=False)
    plan = dict(root=str(root), job=row["job"], files=files, directories=directories,
                file_count=len(files), bytes=sum(value[2] for value in files.values()))
    if retirement_review is not None:
        plan.update(retirement_review=retirement_review, retired_claims=status["claims"])
    return plan


def delete_job(registry, series_root, series_id, plan, *, actor, stores_path=None):
    """Delete only the reviewed tree after a fresh inactive/identity check.

    Rename isolates the exact folder. Each unlink/rmdir revalidates its own
    identity; no recursive deletion follows a junction or an inserted path.
    """
    if not str(actor).strip():
        raise ValueError("An action name is required")
    root = Path(plan["root"])
    with ExitStack() as stack:
        review = plan.get("retirement_review")
        lock_roots = [row["root"] for row in review["affected"]] if review is not None else [str(root)]
        locks = { _path(path): stack.enter_context(JobActionLock(path)) for path in sorted(lock_roots, key=_path) }
        lock = locks[_path(root)]
        current = deletion_plan(registry, series_root, series_id, plan["root"], lock_token=lock.token,
                                retirement_review=review, stores_path=stores_path)
        if current != plan:
            raise ValueError("Job files changed after review. Review deletion again.")
        if review is not None:
            expected_files, expected_directories = dict(plan["files"]), dict(plan["directories"])
            for claim in plan["retired_claims"]:
                if _path(claim["root"]) != _path(root):
                    continue  # Dependent consumers are stopped, but keep their own claims/files.
                binding = _retirement_binding(root, claim["path"], stores_path)
                decision = cancel_retired_claim(root, claim["path"], evidence=claim["evidence"],
                                                actor=actor, local_output=binding)
                expected_files[decision.relative_to(root).as_posix()] = _identity(decision, directory=False)
                expected_directories["decisions"] = _identity(root / "decisions", directory=True)
            current = deletion_plan(registry, series_root, series_id, plan["root"], lock_token=lock.token,
                                    retirement_review=review, stores_path=stores_path)
            if current["files"] != expected_files or current["directories"] != expected_directories:
                raise ValueError("Job files changed during claim cancellation. Review deletion again.")
            plan = current
        from .series_paths import series_folder_path
        parent = series_folder_path(series_root, "jobs").resolve(strict=True)
        removed = parent / (".deleting-" + uuid.uuid4().hex)
        if root.resolve(strict=True).parent != parent or removed.resolve().parent != parent:
            raise ValueError("Deletion target escaped its reviewed jobs directory")
        root.rename(removed)
        lock.path = removed / lock.path.name
        try:
            registry.remove_job(series_root, series_id, root, actor=actor, deletion_path=removed)
        except Exception:
            removed.rename(root)  # No payload deletion has happened yet.
            lock.path = root / lock.path.name
            raise
        lock.release()
    try:
        for relative, expected in plan["files"].items():
            path = removed.joinpath(*relative.split("/"))
            if _identity(path, directory=False) != tuple(expected):
                raise ValueError("A reviewed job file changed during deletion")
            path.unlink()
        for relative in sorted(plan["directories"], key=lambda value: len(Path(value).parts), reverse=True):
            path = removed.joinpath(*relative.split("/")) if relative else removed
            if _identity(path, directory=True) != tuple(plan["directories"][relative]):
                raise ValueError("A reviewed job directory changed during deletion")
            path.rmdir()
    except Exception as error:
        raise RuntimeError("Deletion incomplete. Remaining files are at " + str(removed)) from error
    return dict(deleted=True, root=str(root), file_count=plan["file_count"], bytes=plan["bytes"])
