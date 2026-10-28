"""Explicit original P1 publication bindings for coupled RAW reproduction.

This module reads bounded shared metadata only. It does not discover a parent
job from a result path, open scientific inputs, or grant worker admission.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
import re


FORMAT = "darsia-phase1-reproduction-origin-v1"
_FILES = ("job", "task", "done", "products_index")


def validate_phase1_origin(origin):
    """Validate additive DATA without reading any publication or input file."""
    from .job_queue import name
    from .products_publication import MAX_INDEX_BYTES, MAX_METADATA_BYTES
    if (not isinstance(origin, dict) or set(origin) != {"format", "run", *_FILES}
            or origin.get("format") != FORMAT):
        raise ValueError("Explicit original phase-one reproduction origin required")
    name(origin["run"])
    for role in _FILES:
        bound = origin[role]
        limit = MAX_INDEX_BYTES if role == "products_index" else MAX_METADATA_BYTES
        # Producer locators retain their spelling independently of this host's OS.
        if (not isinstance(bound, dict) or set(bound) != {"path", "sha256", "size_bytes"}
                or not isinstance(bound["path"], str)
                or not (PurePosixPath(bound["path"]).is_absolute()
                        or PureWindowsPath(bound["path"]).is_absolute())
                or re.fullmatch(r"[0-9a-f]{64}", str(bound["sha256"])) is None
                or type(bound["size_bytes"]) is not int or not 0 <= bound["size_bytes"] <= limit):
            raise ValueError("Original phase-one " + role + " needs exact bounded metadata bytes")
    if PureWindowsPath(origin["job"]["path"]).name != "job.json":
        raise ValueError("Choose the explicit original phase-one job.json")
    return origin


def _pin(path, *, limit):
    """Bind one already named bounded metadata file, never an input payload."""
    path = Path(path)
    if not path.is_file() or not 0 <= path.stat().st_size <= limit:
        raise ValueError("Original publication metadata exceeds its bounded file contract")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Original publication metadata changed beyond its size limit")
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def bind_phase1_origin(job_bound, run, *, configuration, completion, expected_commit):
    """Bind a user-selected original job through its unchanged terminal reader.

    The caller must supply the actual job descriptor. Canonical children are
    resolved only inside this explicit job, never from a configuration/output.
    """
    from .job_queue import name, _own_asset
    from .products_publication import MAX_INDEX_BYTES, MAX_METADATA_BYTES
    from ..jobs_metadata import JobSnapshotReader
    root = Path(job_bound["path"]).parent
    run = name(run)
    actual = _pin(root / "job.json", limit=MAX_METADATA_BYTES)
    if actual != job_bound:
        raise ValueError("The explicitly selected original phase-one job changed")
    task = _pin(root / "results" / run / "task.json", limit=MAX_METADATA_BYTES)
    done = _pin(root / "done" / (run + ".json"), limit=MAX_METADATA_BYTES)
    closed = JobSnapshotReader().read(Path(done["path"]), done)
    index = _pin(_own_asset(root, closed["products_index"]), limit=MAX_INDEX_BYTES)
    origin = dict(format=FORMAT, run=run, job=deepcopy(job_bound), task=task,
                  done=done, products_index=index)
    return verify_phase1_origin(origin, configuration=configuration, completion=completion,
                                expected_commit=expected_commit)["origin"]


def verify_phase1_origin(origin, *, configuration, completion, expected_commit,
                         ordered_case_ids=None):
    """Re-read the original publication and return its exact task/catalogue.

    ``proof_files`` names every shared metadata file used by the original
    terminal verifier, plus its indexed shared metadata copies. Their relative
    paths permit an exporter to preserve the complete proof tree unchanged.
    The original configuration/catalogue still require normal input resolution
    and byte verification before export or execution.
    """
    from .job_queue import name, _own_asset, validate_task
    from .job_inputs import validate_catalog
    from .products_publication import (
        MAX_INDEX_BYTES, MAX_METADATA_BYTES, _read_index, published_metadata_descriptor,
    )
    from ..jobs_metadata import JobSnapshotReader
    from ..published_jobs_snapshot import _phase1_terminal, _same_bytes

    validate_phase1_origin(origin)
    if re.fullmatch(r"[0-9a-f]{40}", str(expected_commit)) is None:
        raise ValueError("The original phase-one code commit is required")
    root = Path(origin["job"]["path"]).parent
    run = name(origin["run"])
    observed, metadata = {}, JobSnapshotReader()

    def observe(path, expected=None):
        path = Path(path)
        if not path.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
            raise ValueError("Phase-one proof metadata escapes the explicitly selected job")
        actual = _pin(path, limit=MAX_METADATA_BYTES)
        if expected is not None and actual != expected:
            raise ValueError("Original phase-one proof metadata bytes changed")
        key = str(path.resolve(strict=True))
        if key in observed and observed[key] != actual:
            raise ValueError("Original phase-one proof changed during verification")
        observed[key] = actual
        return actual

    def read(path, expected=None):
        actual = observe(path, expected)
        return metadata.read(Path(path), actual)

    job = read(Path(origin["job"]["path"]), origin["job"])
    if (job.get("format") != "darsia-alias-job-v1"
            or job.get("pipeline") != "native-phase-statistics"
            or job.get("commit") != expected_commit or run not in job.get("expected_runs", ())):
        raise ValueError("The origin is not the actual original phase-one job/run/code")
    for role, expected_path in (("task", root / "results" / run / "task.json"),
                                ("done", root / "done" / (run + ".json"))):
        if Path(origin[role]["path"]) != expected_path:
            raise ValueError("Original phase-one " + role + " is outside its canonical job/run")
    task = read(Path(origin["task"]["path"]), origin["task"])
    done = read(Path(origin["done"]["path"]), origin["done"])
    validate_task(task)
    validate_catalog(task["input_catalog"])
    index_path = _own_asset(root, done["products_index"])
    if Path(origin["products_index"]["path"]) != index_path:
        raise ValueError("Original phase-one products index differs from its canonical done record")
    read(index_path, origin["products_index"])
    completed, machines, _lineage = _phase1_terminal(root, task, job, done, read)
    if (completed.get("configuration") != configuration
            or completed.get("phase1_completion") != completion
            or not _same_bytes(task["configuration"], configuration)
            or (ordered_case_ids is not None and task["ordered_case_ids"] != ordered_case_ids)):
        raise ValueError("Original phase-one publication differs from the sealed source or selection")
    _index_root, index, products = _read_index(origin["products_index"])
    for product_id, row in products.items():
        if "metadata_copy" in row:
            bound = published_metadata_descriptor(origin["products_index"], product_id)
            pin = dict(path=bound["path"], sha256=bound["sha256"], size_bytes=bound["bytes"])
            if pin["size_bytes"] > MAX_METADATA_BYTES:
                raise ValueError("Indexed original phase-one metadata exceeds its bound")
            observe(Path(pin["path"]), pin)
    proof = []
    for pin in observed.values():
        if _pin(Path(pin["path"]), limit=MAX_METADATA_BYTES) != pin:
            raise ValueError("Original phase-one proof changed before verification closed")
        relative = Path(pin["path"]).resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()
        proof.append(dict(relative_path=relative, **pin))
    return dict(origin=deepcopy(origin), job=job, task=task, completed=completed,
                products_index=index, frame_machines=machines,
                proof_files=sorted(proof, key=lambda row: row["relative_path"]),
                scientific_inputs_verified=False, worker_admission_granted=False)
