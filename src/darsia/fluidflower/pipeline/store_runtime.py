"""Resolve a role-bound run through this worker's local store map.

This module admits locations and exact input bytes only. It does not select
frames, alter parameter values, invoke a numerical producer or own a claim.
The shared task remains unchanged; physical paths exist in this process only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from .io_ops import retry_io
import stat

from ..job_filesystem import inspect_results_root
from ..store_map import require_input_containment
from .job_queue import name
from .job_inputs import validate_catalog, _json_bytes, MAX_METADATA_BYTES


class RunUnavailable(ValueError):
    """This machine must leave the pending whole run for another worker."""


class InputHashMismatch(ValueError):
    """An owned, not-yet-started claim must be retired with this reason."""


def _version(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


@dataclass(frozen=True)
class ResolvedInput:
    reference: dict
    path: Path
    uses: tuple[str, ...]


@dataclass(frozen=True)
class VerifiedInputs:
    """Admission evidence scoped to one run; never a replacement source SHA."""

    run: str
    rows: tuple[ResolvedInput, ...]
    versions: dict
    digests: dict

    def for_use(self, use):
        rows = [row for row in self.rows if use in row.uses]
        if len(rows) != 1:
            raise InputHashMismatch("Exactly one input is required for consumer " + use)
        return rows[0]

    @retry_io
    def metadata(self, use):
        row = self.for_use(use)
        before = row.path.stat()
        if _version(before) != self.versions[str(row.path)]:
            raise InputHashMismatch("Verified input changed before metadata read: " + use)
        with row.path.open("rb") as stream:
            raw = stream.read(MAX_METADATA_BYTES + 1)
        if len(raw) > MAX_METADATA_BYTES:
            raise InputHashMismatch("Input metadata exceeds its size limit: " + use)
        if _version(row.path.stat()) != _version(before) or len(raw) != row.reference["bytes"]:
            raise InputHashMismatch("Verified input changed during metadata read: " + use)
        return _json_bytes(raw)


def resolve_catalog(task, stores, series_id, *, checkpoint=None):
    """Availability before claim: metadata/stat only, no payload hashes or writes."""
    catalog = validate_catalog(task["input_catalog"])
    if catalog["run"] != task["run"] or catalog["configuration"] != task["configuration"]:
        raise ValueError("Task and role catalogue identities differ")
    roots = stores.roots(series_id)
    exact = {row["reference"]["relative_path"]: row for row in catalog.get("native_exact_inputs", [])}
    if exact and (task.get("pipeline") != "series-setup" or task.get("operation") != "build"):
        raise ValueError("Native exact input files belong only to the original Setup Build reader")
    from darsia_cache.folders import require_durable_path, folder_mode
    ordinary = folder_mode(stores, series_id)
    rows = []
    references = [item["reference"] for item in catalog["inputs"]]
    for field in ("series", "series_manifest"):
        if task[field] not in references:
            raise ValueError("Role catalogue omits task input " + field)
    for item in catalog["inputs"]:
        if checkpoint is not None:
            checkpoint()
        reference = item["reference"]
        try:
            if ordinary and reference["root_alias"] == "cache":
                raise RunUnavailable("Disposable Cache folder cannot supply a job input; prepare it under Jobs")
            if reference["root_alias"] == "native_exact":
                from ..storage_paths import resolve_storage_path
                row = exact[reference["relative_path"]]
                path = resolve_storage_path(row["descriptor"]["path"])
                info = path.stat()
                if not stat.S_ISREG(info.st_mode) or info.st_size != reference["bytes"]:
                    raise RunUnavailable("Native exact input byte count/type differs")
            else:
                path = _available_input(reference, stores, series_id, roots)
            if ordinary:
                require_durable_path(stores, series_id, path)
        except (OSError, ValueError) as error:
            raise RunUnavailable("Input unavailable for " + ", ".join(item["uses"]) +
                                 ": " + str(error)) from error
        rows.append(ResolvedInput(dict(reference), path, tuple(item["uses"])))
    return tuple(rows)


@retry_io
def _available_input(reference, stores, series_id, roots):
    path = require_input_containment(reference, stores, series_id)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise RunUnavailable("Input is not a regular file")
    if info.st_size != reference["bytes"]:
        raise RunUnavailable("Input byte count differs")
    return path


@retry_io
def _verify_input_bytes(path, *, checkpoint=None):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            if checkpoint is not None:
                checkpoint()
            digest.update(block)
    after = path.stat()
    if _version(before) != _version(after):
        raise InputHashMismatch("Input changed during admission: " + str(path))
    return _version(after), digest.hexdigest()


def verify_inputs(task, rows, *, previous=None, checkpoint=None):
    """Hash each distinct physical input once after claim, before any frame starts."""
    versions, digests = {}, {}
    reusable = (isinstance(previous, VerifiedInputs) and previous.run == task["run"]
                and previous.rows == tuple(rows))
    for row in rows:
        if checkpoint is not None:
            checkpoint()
        key = str(row.path)
        if key not in digests:
            if reusable and _version(row.path.stat()) == previous.versions.get(key):
                versions[key], digests[key] = previous.versions[key], previous.digests[key]
            else:
                versions[key], digests[key] = (_verify_input_bytes(row.path)
                    if checkpoint is None else _verify_input_bytes(row.path, checkpoint=checkpoint))
        if (digests[key] != row.reference["sha256"]
                or versions[key][2] != row.reference["bytes"]):
            raise InputHashMismatch("Input SHA/bytes differ for " + ", ".join(row.uses))
    return VerifiedInputs(task["run"], tuple(rows), versions, digests)


def local_attempt_output(stores, series_id, job_id, run, attempt_id):
    """Admit the explicit local or SMB/NFS result root without creating it."""
    roots = stores.roots(series_id)
    if "results" not in roots:
        raise RunUnavailable("No explicit results root is configured")
    root = Path(roots["results"])
    evidence = inspect_results_root(root)
    if not evidence.allowed:
        raise RunUnavailable("Results root is not admitted: " +
                             evidence.reason + ": " + evidence.evidence)
    from .capacity import remote_results_owner
    remote_results_owner(stores,series_id,root,admission=evidence)
    import re

    if re.fullmatch(r"[0-9a-f]{32}", attempt_id) is None:
        raise ValueError("A real queue attempt identity is required")
    output = root / name(job_id) / name(run) / ".attempts" / attempt_id
    from darsia_cache.folders import require_durable_path
    require_durable_path(stores, series_id, output)
    parent = output
    while not parent.exists():
        if parent == parent.parent:
            raise RunUnavailable("No existing configured result-root ancestor")
        parent = parent.parent
    actual_root = root.resolve()
    if not output.resolve().is_relative_to(actual_root):
        raise RunUnavailable("Output path escapes its configured results root")
    if not output.resolve().is_relative_to(parent.resolve()):
        raise RunUnavailable("Output ancestor changed during location admission")
    return output, evidence
