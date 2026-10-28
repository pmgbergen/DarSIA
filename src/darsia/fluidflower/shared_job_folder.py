"""Shared Jobs discovery and presence; computation remains in the original queue."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import ntpath
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import uuid

from .pipeline.job_queue import write_immutable


_LIMIT = 256 * 1024
_JOB_LIMIT = 4 * 1024 * 1024
_MAX_ROWS = 4096
_ID = re.compile(r"[a-f0-9]{64}\Z")
_FORMAT = "darsia-shared-job-v1"
_PRESENCE = "darsia-shared-worker-v1"
_CONTROL = "darsia-shared-job-admission-v1"


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _epoch(value):
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Shared Jobs timestamps require a timezone")
    return result.timestamp()


def _text(value, field):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"An explicit {field} is required")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"Invalid {field}")
    return value.strip()


def _count(value):
    if type(value) is not int or not 0 <= value <= 256:
        raise ValueError("Worker counts must be integers between 0 and 256")
    return value


def _decode(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate shared metadata key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite shared metadata value: " + value)

    doc = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique,
                     parse_constant=invalid)
    if not isinstance(doc, dict):
        raise ValueError("Shared Jobs metadata must be a JSON object")
    return doc


def _bytes(path, limit=_LIMIT):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if not raw or len(raw) > limit:
        raise ValueError("Shared Jobs metadata exceeds its size limit")
    return raw


def _read(path):
    return _decode(_bytes(path))


def _encoded(doc):
    raw = json.dumps(doc, ensure_ascii=False, allow_nan=False,
                     separators=(",", ":")).encode("utf-8")
    if len(raw) > _LIMIT:
        raise ValueError("Shared Jobs metadata exceeds its size limit")
    return raw


def _atomic(path, doc):
    raw = _encoded(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _files(directory):
    if not directory.exists():
        return []
    rows = []
    for path in directory.glob("*.json"):
        rows.append(path)
        if len(rows) > _MAX_ROWS:
            raise ValueError("Too many shared Jobs metadata records")
    return sorted(rows)


def _job_asset(root):
    raw = _bytes(root / "job.json", _JOB_LIMIT)
    _decode(raw)
    return dict(path="job.json", sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def current_worker_reports(reports):
    """One current presence row per machine; original session reports stay kept.

    Presence is only a GUI connection report. Selecting a fresh report neither
    proves a calculation is alive nor retires workers from older sessions.
    """
    grouped = {}
    for report in reports:
        grouped.setdefault(str(report.get("machine", "")).casefold(), []).append(report)
    result = []
    def timestamp(row):
        try:
            return _epoch(row.get("reported_utc", row.get("updated_utc")))
        except (ValueError, TypeError, OverflowError):
            return -math.inf  # A missing report is not evidence of a fresh connection.
    for machine in sorted(grouped):
        rows = grouped[machine]
        current = max(rows, key=lambda row: (
            row.get("state") == "online" and not row.get("stale", True),
            timestamp(row),
        ))
        result.append(dict(current, previous_session_count=len(rows) - 1))
    return result


class SharedJobFolder:
    """Read only explicitly published jobs and atomic GUI presence files.

    Presence timestamps describe the reporting GUI, never prove that a worker
    is alive or that a frame completed. Job admission does not issue queue
    pause/stop commands; those remain with the original master's controls.
    """

    def __init__(self, path):
        self.path = Path(path).resolve()
        if not self.path.is_dir():
            raise FileNotFoundError("Choose an existing shared Jobs folder")
        self.metadata = self.path / ".darsia-jobs"

    def _location(self, root):
        try:
            relative = root.relative_to(self.path).as_posix()
        except ValueError:
            text = str(root)
            if not PureWindowsPath(text).is_absolute() or not text.startswith(("\\\\", "//")):
                raise ValueError("Jobs outside the shared folder require an absolute UNC path")
            return dict(unc_path=text)
        return dict(relative_path=relative)

    def _root(self, location):
        if not isinstance(location, dict):
            raise ValueError("Missing shared job location")
        if set(location) == {"relative_path"}:
            value = location["relative_path"]
            if (not isinstance(value, str) or not value or "\\" in value
                    or ":" in value or PurePosixPath(value).is_absolute()
                    or ".." in PurePosixPath(value).parts):
                raise ValueError("Shared job path escapes its folder")
            root = (self.path / value).resolve()
            if not root.is_relative_to(self.path):
                raise ValueError("Shared job path escapes its folder")
            return root
        if set(location) == {"unc_path"}:
            value = location["unc_path"]
            if (not isinstance(value, str) or not value.startswith(("\\\\", "//"))
                    or not PureWindowsPath(value).is_absolute()
                    or ".." in PureWindowsPath(value).parts):
                raise ValueError("Invalid shared job UNC path")
            return Path(value).resolve()
        raise ValueError("Invalid shared job location")

    @staticmethod
    def _identifier(location):
        # Relative references remain portable when a share has another local mount.
        value = json.dumps(location, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(ntpath.normcase(value).encode("utf-8")).hexdigest()

    @staticmethod
    def _master(root, machine, *, create=False):
        from .job_control import _bindings

        reference, master_format, _ = _bindings(root)
        path = root / "ui" / "master.json"
        expected = reference(root / "job.json")
        if create and not path.exists():
            value = dict(format=master_format, job=expected, machine=machine,
                         created_utc=_utc(), actor=machine,
                         reason="Publish this job to the shared Jobs folder")
            try:
                write_immutable(path, value)
            except FileExistsError:
                pass
        master = _read(path)
        if (master.get("format") != master_format or master.get("job") != expected
                or _text(master.get("machine"), "master machine").casefold() != machine.casefold()):
            raise ValueError("Only the job's recorded master machine may publish or change admission")
        return master["machine"], reference(path)

    def publish(self, job_root, *, master_machine, cpu_workers=0, cuda_workers=0):
        machine = _text(master_machine, "master machine")
        cpu_workers, cuda_workers = _count(cpu_workers), _count(cuda_workers)
        root = Path(job_root).resolve()
        location = self._location(root)
        asset = _job_asset(root)
        machine, master = self._master(root, machine, create=True)
        if _job_asset(root) != asset:
            raise ValueError("Job metadata changed during publication")
        job_id = self._identifier(location)
        path = self.metadata / "jobs" / (job_id + ".json")
        value = dict(format=_FORMAT, job_id=job_id, location=location, job=asset,
                     master_machine=machine, master=master, published_utc=_utc(),
                     cpu_workers=cpu_workers, cuda_workers=cuda_workers)
        _encoded(value)
        try:
            write_immutable(path, value)
        except FileExistsError:
            previous, _source, _generation = self._publication(path)
            if any(previous.get(key) != value[key] for key in
                   ("format", "job_id", "location", "job", "master_machine", "master")):
                raise ValueError("Published job identity or original master changed")
        return self._job(path)

    @staticmethod
    def _publication_asset(path):
        raw = _bytes(path, _JOB_LIMIT)
        return dict(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))

    def _publication(self, path, *, pending_previous=None):
        """Fold append-only, receipt-bound unstarted Setup code transitions.

        The original v1 publication stays byte-identical. Older readers refuse
        its now-obsolete job binding instead of executing a different job.
        """
        value, source, generation = _read(path), path, 0
        directory = self.metadata / "job-revisions" / path.stem
        revisions = list(_files(directory))
        for generation, revision in enumerate(revisions, 1):
            root = self._root(value["location"])
            transition = _read(revision)
            if (revision.name != f"{generation:020d}.json"
                    or transition.get("format") != "darsia-shared-job-revision-v1"
                    or transition.get("generation") != generation
                    or transition.get("previous") != self._publication_asset(source)):
                raise ValueError("Shared job publication revision chain differs")
            receipt = transition.get("receipt") or {}
            relative = receipt.get("relative_path", "")
            if (not relative or PurePosixPath(relative).is_absolute() or "\\" in relative
                    or ".." in PurePosixPath(relative).parts):
                raise ValueError("Shared job revision requires its job-owned recovery receipt")
            receipt_path = root.joinpath(*PurePosixPath(relative).parts)
            if (not receipt_path.resolve().is_relative_to(root / "history")
                    or self._publication_asset(receipt_path) != {
                        key: receipt.get(key) for key in ("sha256", "size_bytes")}):
                raise ValueError("Shared job revision receipt changed")
            record = _decode(_bytes(receipt_path, _JOB_LIMIT))
            replacement = transition.get("publication") or {}
            retained_path = receipt_path.parent / "previous" / "job.json"
            retained = _decode(_bytes(retained_path, _JOB_LIMIT))
            replacement_pin = record.get("replacement_job")
            if replacement_pin is not None:
                replacement_path = Path(replacement_pin.get("path", ""))
                if (not replacement_path.resolve().is_relative_to(receipt_path.parent.resolve() / "replacement")
                        or replacement_path.name != "job.json"
                        or self._publication_asset(replacement_path) != {
                            key: replacement_pin.get(key) for key in ("sha256", "size_bytes")}):
                    raise ValueError("Shared job replacement snapshot changed")
            elif generation < len(revisions):
                # Older receipts predate replacement snapshots. The following
                # generation retained exactly the job this generation produced.
                following = _read(revisions[generation])
                relative_next = (following.get("receipt") or {}).get("relative_path", "")
                if (not relative_next or PurePosixPath(relative_next).is_absolute()
                        or "\\" in relative_next or ".." in PurePosixPath(relative_next).parts):
                    raise ValueError("Shared job revision requires its job-owned recovery receipt")
                next_receipt = root.joinpath(*PurePosixPath(relative_next).parts)
                if not next_receipt.resolve().is_relative_to(root / "history"):
                    raise ValueError("Shared job revision receipt escapes history")
                replacement_path = next_receipt.parent / "previous" / "job.json"
            else:
                replacement_path = pending_previous if pending_previous is not None else root / "job.json"
            if self._publication_asset(replacement_path) != {
                    key: replacement.get("job", {}).get(key) for key in ("sha256", "size_bytes")}:
                raise ValueError("Shared job replacement metadata changed")
            actual_job = _decode(_bytes(replacement_path, _JOB_LIMIT))
            if (record.get("format") != "darsia-setup-job-reprepare-v1" or record.get("complete") is not True
                    or retained.get("pipeline") != "series-setup"
                    or self._publication_asset(retained_path) != {
                        key: value.get("job", {}).get(key) for key in ("sha256", "size_bytes")}
                    or any(replacement.get(key) != value.get(key) for key in
                           ("format", "job_id", "location", "master_machine"))
                    or any(record.get("previous_job", {}).get(key) != value.get("job", {}).get(key)
                           or record.get("job", {}).get(key) != replacement.get("job", {}).get(key)
                           for key in ("sha256", "size_bytes"))):
                raise ValueError("Shared job revision differs from its original Setup transition")
            from .setup_job_reprepare import validate_setup_reprepare_transition
            validate_setup_reprepare_transition(retained, actual_job, record, history_root=receipt_path.parent)
            value, source = replacement, revision
        return value, source, generation

    def review_reprepare(self, job_root, *, master_machine):
        """Capture only this publication and admission for the locked recovery."""
        root = Path(job_root).resolve()
        identifier = self._identifier(self._location(root))
        path = self.metadata / "jobs" / (identifier + ".json")
        if not path.exists():
            return None
        row = self._job(path)
        if row["master_machine"].casefold() != master_machine.casefold():
            raise ValueError("Only the recorded publication master can reprepare this Setup job")
        value, source, generation = self._publication(path)
        admission = self._admission_path(identifier, generation)
        return dict(publication=value, source=str(source), binding=self._publication_asset(source),
                    generation=generation, admission=str(admission),
                    admission_bytes=admission.read_bytes() if admission.exists() else None)

    def _admission_path(self, identifier, generation):
        return (self.metadata / "admission" / identifier / f"{generation:020d}.json" if generation
                else self.metadata / "admission" / (identifier + ".json"))

    def reprepare_setup_publication(self, job_root, *, review, receipt, master_machine):
        """Append a paused publication generation after the real Setup producer."""
        from .job_control import read_control
        root = Path(job_root).resolve()
        if read_control(root)["action"] != "pause":
            raise ValueError("A reprepared Setup publication must remain paused until explicit Start")
        old = review["publication"]
        path = self.metadata / "jobs" / (old["job_id"] + ".json")
        previous_snapshot = Path(receipt["path"]).parent / "previous" / "job.json"
        if (not previous_snapshot.resolve().is_relative_to(root / "history")
                or self._publication_asset(previous_snapshot) != {
                    key: old.get("job", {}).get(key) for key in ("sha256", "size_bytes")}):
            raise ValueError("Pending Setup transition differs from the retained publication")
        current, source, generation = self._publication(path, pending_previous=previous_snapshot)
        admission = Path(review["admission"])
        if (current != old or generation != review["generation"]
                or self._publication_asset(source) != review["binding"]
                or (admission.read_bytes() if admission.exists() else None) != review["admission_bytes"]):
            raise ValueError("Shared job publication changed during reprepare")
        machine, master = self._master(root, master_machine)
        publication = dict(old, job=_job_asset(root), master=master, published_utc=_utc())
        relative = Path(receipt["path"]).resolve().relative_to(root).as_posix()
        value = dict(format="darsia-shared-job-revision-v1", generation=generation + 1,
            previous=review["binding"], publication=publication,
            receipt=dict(relative_path=relative, **{key: receipt[key] for key in ("sha256", "size_bytes")}))
        destination = self.metadata / "job-revisions" / old["job_id"] / f"{generation + 1:020d}.json"
        # The new original control is already paused before this becomes visible.
        write_immutable(destination, value)
        self._job(path)
        return destination

    def _job(self, path):
        from .job_control import read_control

        value, _source, generation = self._publication(path)
        job_id = value.get("job_id")
        if (value.get("format") != _FORMAT or not isinstance(job_id, str)
                or not _ID.fullmatch(job_id) or path.name != job_id + ".json"
                or self._identifier(value.get("location")) != job_id):
            raise ValueError("Invalid shared job identity")
        root = self._root(value["location"])
        asset = _job_asset(root)
        if asset != value.get("job"):
            raise ValueError("Published job metadata changed")
        machine = _text(value.get("master_machine"), "master machine")
        _, master = self._master(root, machine)
        if master != value.get("master"):
            raise ValueError("Published job master changed")
        _count(value.get("cpu_workers"))
        _count(value.get("cuda_workers"))
        _epoch(value.get("published_utc"))
        state = "run"
        admission = self._admission_path(job_id, generation)
        if admission.exists():
            control = _read(admission)
            if (control.get("format") != _CONTROL or control.get("job_id") != job_id
                    or control.get("master") != master
                    or control.get("master_machine") != machine
                    or control.get("state") not in {"run", "paused", "stopped"}):
                raise ValueError("Invalid shared job admission")
            _epoch(control.get("updated_utc"))
            state = control["state"]
        original = read_control(root)["action"]
        effective = {"pause": "paused", "stop": "stopped"}.get(original, state)
        return dict(value, job_root=str(root), state=effective,
                    folder_state=state, original_control_state=original, publication_generation=generation)

    def jobs(self):
        return [self._job(path) for path in _files(self.metadata / "jobs")]

    def update_job(self, job_id, *, master_machine, state):
        if not isinstance(job_id, str) or not _ID.fullmatch(job_id):
            raise ValueError("Invalid shared job identifier")
        if state not in {"run", "paused", "stopped"}:
            raise ValueError("Unknown shared job admission state")
        machine = _text(master_machine, "master machine")
        row = self._job(self.metadata / "jobs" / (job_id + ".json"))
        if row["master_machine"].casefold() != machine.casefold():
            raise ValueError("Only the recorded master may change shared job admission")
        value = dict(format=_CONTROL, job_id=job_id, master=row["master"],
                     master_machine=row["master_machine"], state=state, updated_utc=_utc())
        _atomic(self._admission_path(job_id, row["publication_generation"]), value)
        return self._job(self.metadata / "jobs" / (job_id + ".json"))

    @staticmethod
    def _worker(value):
        if value.get("format") != _PRESENCE or value.get("role") not in {"master", "worker"}:
            raise ValueError("Invalid shared worker role")
        machine = _text(value.get("machine"), "worker machine")
        session = _text(value.get("session_id"), "worker session")
        _count(value.get("cpu_workers"))
        _count(value.get("cuda_workers"))
        _epoch(value.get("reported_utc"))
        if value.get("state") not in {"online", "offline"}:
            raise ValueError("Invalid shared worker presence state")
        jobs = value.get("jobs")
        if (not isinstance(jobs, list) or len(jobs) > 256
                or any(not isinstance(row, dict) for row in jobs)):
            raise ValueError("Worker progress must be a bounded list of job records")
        return hashlib.sha256((machine.casefold() + "\0" + session).encode("utf-8")).hexdigest()

    def report(self, machine, session_id, *, role, cpu_workers, cuda_workers,
               jobs, state="online"):
        value = dict(format=_PRESENCE, machine=_text(machine, "worker machine"),
                     session_id=_text(session_id, "worker session"), role=role,
                     cpu_workers=cpu_workers, cuda_workers=cuda_workers, jobs=jobs,
                     state=state, reported_utc=_utc())
        identifier = self._worker(value)
        _atomic(self.metadata / "workers" / (identifier + ".json"), value)
        return value

    def workers(self, now=None, stale_after=30):
        if isinstance(now, (int, float)):
            current = float(now)
        else:
            current = _epoch(now if now is not None else _utc())
        if not math.isfinite(current) or not isinstance(stale_after, (int, float)) or not 0 < stale_after < math.inf:
            raise ValueError("A finite current time and positive presence timeout are required")
        result = []
        for path in _files(self.metadata / "workers"):
            value = _read(path)
            if path.name != self._worker(value) + ".json":
                raise ValueError("Shared worker identity differs from its filename")
            age = current - _epoch(value["reported_utc"])
            result.append(dict(value, age_seconds=max(0, age),
                               stale=value["state"] != "online" or age > stale_after or age < -stale_after))
        return result
