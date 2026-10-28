"""Per-run approvals in Calibration review (N141 spec 2.5, Part 2).

An approval is one append-only record in the series folder: the run, the mass
calibration that series.toml references (its path and SHA-256), the result
viewed, the reviewer and the UTC time. The reviewer is the series' reviewer
setting (config/phase_model_studio_reviewer.json); without one the record
names nobody, and the operating-system login is never used. Records are never
rewritten. Approving a run has no effect on any result, calibration, plan or job.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

APPROVAL_FORMAT = "darsia-calibration-run-approval-v1"
APPROVALS_PATH = Path("review") / "calibration_review" / "run_approvals.jsonl"
MAX_APPROVALS_BYTES = 16 * 1024 * 1024


def calibration_identity(series_root, reference):
    """The referenced mass calibration, as series.toml names it, and its SHA-256."""
    if not reference:
        raise ValueError("No mass calibration is referenced in series.toml.")
    path = Path(reference)
    if not path.is_absolute():
        path = Path(series_root) / path
    from .storage_paths import resolve_storage_path
    path = resolve_storage_path(path, series_roots=(Path(series_root),))
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"reference": str(reference), "sha256": digest}


def read_run_approvals(series_root):
    """Every approval record of the series, oldest first.

    A malformed line raises ValueError, so a damaged record is never read as
    "not approved".
    """
    path = Path(series_root) / APPROVALS_PATH
    if not path.exists():
        return []
    if path.stat().st_size > MAX_APPROVALS_BYTES:
        raise ValueError(f"Approval records exceed the display limit: {path}")
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or record.get("format") != APPROVAL_FORMAT:
            raise ValueError(f"Line {number} of {path} is not a run approval record.")
        records.append(record)
    return records


def run_approval(records, run, calibration_sha256, *, native_evidence=None):
    """The latest approval of this run and, when supplied, exact native evidence."""
    wanted = str(run).casefold()
    matches = [
        record
        for record in records
        if str(record.get("run", "")).casefold() == wanted
        and (record.get("calibration") or {}).get("sha256") == calibration_sha256
        and (native_evidence is None or record.get("native_evidence") == native_evidence)
    ]
    return matches[-1] if matches else None


def approve_run(series_root, run, calibration, *, result=None, reviewer=None, now=None, retry=None,
                native_evidence=None, before_append=None):
    """Append one approval record and return it; earlier records stay as they are.

    ``reviewer`` is the name from the series' reviewer setting; when it is
    empty the record's reviewer is None. The append is retried on a transient
    file error (N111) with the job queue's retry helper, unless ``retry``
    supplies another runner. Optional native evidence extends the same record;
    its caller must qualify it. ``before_append`` rechecks the captured review
    immediately before every append attempt, including retries.
    """
    record = {
        "format": APPROVAL_FORMAT,
        "run": str(run),
        "calibration": {"reference": str(calibration["reference"]), "sha256": str(calibration["sha256"])},
        "result": str(result) if result else None,
        "reviewer": str(reviewer).strip() or None if reviewer else None,
        "recorded_utc": (now or dt.datetime.now(dt.timezone.utc)).isoformat(),
    }
    if native_evidence is not None:
        # Detach evidence from the caller, rejecting non-JSON/non-finite data.
        record["native_evidence"] = json.loads(json.dumps(native_evidence, allow_nan=False))
    line = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    path = Path(series_root) / APPROVALS_PATH

    def append():
        if before_append is not None:
            before_append()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())

    if retry is None:
        from darsia.fluidflower.pipeline.job_queue import retry_transient as retry
    retry(append)
    return record
