"""Pure progress presentation from retained, admitted job metadata.

No filesystem, hardware, process or producer calls belong here. Counts describe
published work; reported stages and elapsed wall time do not establish liveness.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math


def epoch(value):
    """Accept an actual timezone-aware timestamp, never a selection time."""
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return stamp.timestamp() if stamp.tzinfo is not None else None
    except (ValueError, TypeError, OverflowError):
        return None


def _earliest(row, value):
    stamp = epoch(value)
    previous = epoch(row.get("started_utc"))
    if stamp is not None and (previous is None or stamp < previous):
        row["started_utc"] = value


def retain_execution(row, *, task=None, owner=None, result=None, current=True):
    """Retain already-validated identity/claim fields in a snapshot row."""
    owner, result, task = owner or {}, result or {}, task or {}
    _earliest(row, owner.get("claimed_utc"))
    for value in (result.get("started_utc"), result.get("claimed_utc")):
        _earliest(row, value)
    if not current:
        return
    actual = result.get("actual_worker") or owner.get("actual_owner") or {}
    if actual:
        row["actual_worker"] = dict(actual)
        row["worker"] = worker_text(actual)
    row["attempt_id"] = result.get("attempt_id") or owner.get("token") or task.get("attempt_id")
    if (owner.get("output") or {}).get("machine_id"):
        row["machine"] = owner["output"]["machine_id"]
    elif row.get("machine", "Not reported") == "Not reported" and actual.get("machine"):
        row["machine"] = actual["machine"]
    backend = result.get("backend") or task.get("backend")
    if actual and backend in {"cpu", "cuda"}:
        row["backend"] = backend
    for field in ("finished_utc", "completed_utc", "failed_utc"):
        if epoch(result.get(field)) is not None:
            row["finished_utc"] = result[field]
            break


def worker_text(actual):
    parts = []
    if actual.get("worker_slot") is not None:
        parts.append("slot " + str(actual["worker_slot"]))
    if actual.get("pid") is not None:
        parts.append("PID " + str(actual["pid"]))
    return " · ".join(parts) or "Not reported"


def retain_event(row, event, timestamp):
    """Keep original progress after the reader has checked run/attempt identity."""
    if epoch(timestamp) is None:
        return
    row["last_progress_utc"] = timestamp
    row["progress_event"] = dict(event)
    if event.get("state") in {"frame_started", "phase1_frame_started", "preparing_sources"}:
        _earliest(row, timestamp)


def retain_log_reports(snapshot):
    """Join bounded existing logs only to an admitted exact task/worker attempt."""
    rows = {row["run"]: row for row in snapshot.get("runs", [])
            if row.get("state") != "Metadata error"}
    for log in snapshot.get("logs", []):
        for line in log.get("tail", "").splitlines():
            timestamp, separator, encoded = line.partition(" ")
            if not separator or epoch(timestamp) is None:
                continue
            try:
                event = json.loads(encoded)
            except (ValueError, TypeError):
                continue
            if not isinstance(event, dict):
                continue
            row = rows.get(event.get("run"))
            if row is None or not row.get("attempt_id") or event.get("attempt_id") != row["attempt_id"]:
                continue
            actual = row.get("actual_worker") or {}
            service = str(log.get("path", "")).replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".log")
            if not actual.get("machine") or service.casefold() != str(actual["machine"]).casefold():
                continue
            if event.get("kind") == "claimed" and event.get("actual_worker") == actual:
                _earliest(row, timestamp)
                if type(event.get("worker_slot")) is int:
                    row["worker"] = worker_text(actual) + " · slot " + str(event["worker_slot"])
            if (row.get("state"), event.get("kind")) in {("Complete", "completed"), ("Failed", "failed")}:
                row["finished_utc"] = timestamp
    return snapshot


def duration_text(seconds):
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "Unknown"
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return (f"{days}d " if days else "") + f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _elapsed(row, now):
    start = epoch(row.get("started_utc"))
    end = epoch(row.get("finished_utc"))
    state = row.get("state", "")
    terminal = state in {"Complete", "Failed", "Cancelled", "Stopped", "Metadata error"} or state.startswith(("Stopped ·", "Paused"))
    if start is None or (terminal and end is None):
        return None
    end = now if end is None else end
    return end - start if end >= start else None


def _counts(row, unit):
    keys = {"operations": ("completed_operations", "total_operations"),
            "comparisons": ("completed_comparisons", "total_comparisons")}
    done, total = keys.get(unit, ("committed", "total"))
    completed, count = row.get(done), row.get(total)
    if type(completed) is not int or completed < 0 or type(count) is not int or count <= 0:
        return None, None
    return min(completed, count), count


def _percentage(done, total, complete):
    if done is None or total is None:
        return "Unknown"
    if complete:
        return "100%"
    percent = done * 100 / total
    # Publishing all units is not confirmation of the enclosing task/job.
    return "Estimated <100%" if percent >= 100 else f"Estimated {percent:.1f}%"


def _stage(row):
    event = row.get("progress_event") or row.get("reported_progress") or {}
    state = event.get("state", "")
    if state in {"raw_preparation", "raw_prepared"}:
        done, total = event.get("position"), event.get("total")
        if type(done) is int and type(total) is int and 0 <= done <= total and total > 0:
            return f"RAW preparation: {done}/{total} ({100 * done / total:.1f}% of this stage)"
    text = event.get("step") or state or row.get("stage") or ""
    index, count = event.get("run_index"), event.get("run_count")
    if type(index) is int and type(count) is int and 1 <= index <= count:
        text += f" · run {index}/{count}"
    if event.get("run") and event["run"] != row.get("run"):
        text += " · " + str(event["run"])
    if event.get("message"):
        text += " · " + str(event["message"])
    return text.replace("_", " ")


def progress_view(snapshot, *, now=None):
    """Build compact display cells; a timer can call this without doing IO."""
    if not snapshot:
        return dict(summary="", rows=[])
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    unit = snapshot.get("work_unit", "frames")
    rows, starts, ends, counts = [], [], [], []
    source_rows = snapshot.get("runs", [])
    for row in source_rows:
        done, total = _counts(row, unit)
        counts.append((done, total))
        start, end = epoch(row.get("started_utc")), epoch(row.get("finished_utc"))
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)
        failed = row.get("state") == "Metadata error"
        rows.append(dict(task="Image correction" if row.get("operation") == "image-preparation"
            else row.get("run", "Unknown"), operation="Image correction" if row.get("operation") == "image-preparation"
            else row.get("operation") or row.get("stage") or "Run",
            state=row.get("state", "Unknown"), percent="Unknown" if failed else _percentage(done, total, row.get("state") == "Complete"),
            work=f"{done}/{total} {unit}" if done is not None else "Unknown",
            elapsed=duration_text(_elapsed(row, now)), machine=row.get("machine") or "Not reported",
            worker=row.get("worker") or "Not reported", backend=str(row.get("backend") or "Unknown").upper(),
            stage=_stage(row), started_utc=row.get("started_utc"), finished_utc=row.get("finished_utc")))
    known = bool(counts) and all(total is not None for _, total in counts)
    done = sum(value for value, _ in counts) if known else None
    total = sum(value for _, value in counts) if known else None
    terminal = snapshot.get("complete") or snapshot.get("user_status", snapshot.get("state")) in {"Completed", "Failed", "Cancelled"}
    # Every terminal task needs an actual end; otherwise the final duration is unknown.
    end = max(ends) if terminal and len(ends) == len(starts) and ends else (None if terminal else now)
    elapsed = end - min(starts) if starts and end is not None and end >= min(starts) else None
    percent = "Unknown" if snapshot.get("errors") else _percentage(done, total, snapshot.get("complete") is True)
    count_text = f"{done}/{total} {unit}" if known else f"{unit}: Unknown"
    return dict(summary=f"{percent} · {count_text} · Elapsed: {duration_text(elapsed)}", rows=rows)
