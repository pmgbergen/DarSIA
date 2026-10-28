"""Bridge between project overflow revisions and the mature scale-image reader."""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from time import monotonic, sleep
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from darsia.fluidflower.protocol_schema import read_protocol_tables


REVIEWED_IMAGE_STATUSES = frozenset(
    {"Verified", "Corrected", "Partial", "Unreadable"}
)
EXACT_IMAGE_STATUSES = frozenset({"Verified", "Corrected"})


def _review_float(value: Any) -> float | None:
    from darsia.fluidflower.mass_inputs import _number as _canonical_finite_number
    return _canonical_finite_number(value)


@lru_cache(maxsize=256)
def _read_scale_reader_progress(
    path_text: str,
    _modified_ns: int,
    _size: int,
) -> dict[str, Any]:
    path = Path(path_text)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "available": True,
            "path": path_text,
            "error": f"{type(exc).__name__}: {exc}",
        }

    records = [
        dict(value)
        for value in payload.get("records") or []
        if isinstance(value, Mapping)
    ]
    reviewed = [
        record
        for record in records
        if str(record.get("status") or "") in REVIEWED_IMAGE_STATUSES
    ]
    selected_count = sum(
        bool(record.get("selected_for_review"))
        or str(record.get("status") or "") in REVIEWED_IMAGE_STATUSES
        for record in records
    )
    times_h = sorted(
        value
        for value in (
            _review_float(record.get("time_from_injection_h"))
            for record in reviewed
        )
        if value is not None
    )
    image_times = sorted(
        str(record.get("image_time") or "")
        for record in reviewed
        if str(record.get("image_time") or "").strip()
    )
    reviewers = sorted(
        {
            str(record.get("reviewer") or "").strip()
            for record in reviewed
            if str(record.get("reviewer") or "").strip()
        }
    )
    counts = {
        status: sum(
            str(record.get("status") or "") == status for record in reviewed
        )
        for status in sorted(REVIEWED_IMAGE_STATUSES)
    }
    metadata = dict(payload.get("metadata") or {})
    return {
        "available": True,
        "path": path_text,
        "error": "",
        "reviewed_count": len(reviewed),
        "selected_count": selected_count,
        "exact_count": sum(
            str(record.get("status") or "") in EXACT_IMAGE_STATUSES
            for record in reviewed
        ),
        "partial_count": counts.get("Partial", 0),
        "unreadable_count": counts.get("Unreadable", 0),
        "first_time_h": times_h[0] if times_h else None,
        "last_time_h": times_h[-1] if times_h else None,
        "first_image_time": image_times[0] if image_times else "",
        "last_image_time": image_times[-1] if image_times else "",
        "reviewers": reviewers,
        "updated_at": str(metadata.get("updated_at") or ""),
    }


def scale_reader_review_progress(project_root: Path, run_id: str) -> dict[str, Any]:
    """Summarize manually handled scale images without changing reader artifacts."""

    normalized = str(run_id).strip().casefold()
    path = (
        Path(project_root)
        / "results"
        / "overflow"
        / "scale_review"
        / normalized
        / f"{normalized}_overflow_scale_review.json"
    )
    try:
        stat = path.stat()
    except OSError:
        return {
            "available": False,
            "path": str(path),
            "error": "",
            "reviewed_count": 0,
            "selected_count": 0,
            "exact_count": 0,
            "partial_count": 0,
            "unreadable_count": 0,
            "first_time_h": None,
            "last_time_h": None,
            "first_image_time": "",
            "last_image_time": "",
            "reviewers": [],
            "updated_at": "",
        }
    return dict(
        _read_scale_reader_progress(str(path), stat.st_mtime_ns, stat.st_size)
    )


def _find_reader(explicit: Path | None = None) -> Path:
    if explicit is not None and str(explicit):
        selected = Path(explicit).expanduser().resolve()
        if not selected.is_file():
            raise FileNotFoundError(selected)
        return selected
    environment = os.environ.get("DARSIA_FLUIDFLOWER_OVERFLOW_READER")
    if environment:
        selected = Path(environment).expanduser().resolve()
        if selected.is_file():
            return selected
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "scripts" / "overflow_weight_ui.py"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "The scale-image reader was not found. Set "
        "DARSIA_FLUIDFLOWER_OVERFLOW_READER to overflow_weight_ui.py."
    )


def _datetime_value(value: Any, experiment_date: date | None) -> datetime | None:
    if value is None or str(value).strip().casefold() in {"", "none", "nan", "nat"}:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, time) and experiment_date is not None:
        return datetime.combine(experiment_date, value)
    if isinstance(value, date):
        return datetime.combine(value, time())
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.year == 1900 and experiment_date is not None:
            return datetime.combine(experiment_date, parsed.time())
        return parsed
    except ValueError:
        if experiment_date is None:
            return None
        try:
            return datetime.combine(experiment_date, time.fromisoformat(text))
        except ValueError:
            return None


def _protocol_window(project: Any, run_id: str) -> tuple[datetime, datetime] | None:
    selected_run = next(
        (
            dict(row)
            for row in project.manifest.get("runs") or []
            if str(row.get("run_id") or "").strip().casefold() == run_id
        ),
        {},
    )
    mapping = dict(selected_run.get("source_mapping") or {})
    paths = [
        str(value)
        for value in mapping.get("protocol_tables") or []
        if str(value).strip()
    ]
    if str(mapping.get("protocol_table") or "").strip():
        paths.insert(0, str(mapping["protocol_table"]))
    starts: list[datetime] = []
    ends: list[datetime] = []
    for value in dict.fromkeys(paths):
        path = Path(value).expanduser()
        if not path.is_file():
            continue
        try:
            rows = read_protocol_tables(path)
        except Exception:  # noqa: BLE001, S112 - optional reader context
            continue
        for row in rows:
            row_run = str(row.get("run_id") or row.get("run") or "").strip().casefold()
            if row_run and row_run != run_id:
                continue
            event_type = str(row.get("event_type") or row.get("type") or "").casefold()
            if not event_type.startswith("injection_"):
                continue
            experiment_date: date | None = None
            try:
                experiment_date = date.fromisoformat(
                    str(row.get("experiment_date"))[:10]
                )
            except ValueError:
                pass
            start = _datetime_value(
                row.get("start_datetime") or row.get("datetime"), experiment_date
            )
            end = _datetime_value(row.get("end_datetime"), experiment_date)
            if start is not None:
                starts.append(start)
            if end is not None:
                ends.append(end)
    if not starts:
        return None
    start = min(starts)
    end = max(ends) if ends else start + timedelta(hours=1)
    return start, max(start, end)


def _fallback_window(row: Mapping[str, Any]) -> tuple[datetime, datetime]:
    scan = dict(row.get("scan") or {})
    first = str(scan.get("first_timestamp") or "")
    try:
        start = datetime.fromisoformat(first)
    except ValueError:
        start = datetime.now().replace(microsecond=0)  # noqa: DTZ005
    return start, start + timedelta(hours=1)


def write_scale_reader_config(
    project: Any,
    config: Mapping[str, Any],
    *,
    path: Path | None = None,
    run_ids: Sequence[str] = (),
) -> Path:
    """Materialize a project-local adapter config without changing source files."""

    target = path or project.root / "cache" / "overflow_reader" / "config.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    selected = {
        str(value).strip().casefold() for value in run_ids if str(value).strip()
    }
    runs: dict[str, dict[str, Any]] = {}
    for raw in config.get("runs") or []:
        row = dict(raw)
        if str(row.get("mode") or "none") != "scale_images":
            continue
        folders = [
            Path(value).expanduser().resolve()
            for value in row.get("image_folders") or []
            if str(value).strip()
        ]
        if not folders:
            continue
        run_id = str(row.get("run_id") or "").strip().casefold()
        if selected and run_id not in selected:
            continue
        window = _protocol_window(project, run_id) or _fallback_window(row)
        try:
            image_root = Path(os.path.commonpath([str(value) for value in folders]))
        except ValueError:
            image_root = folders[0]
        runs[run_id] = {
            "image_root": str(image_root),
            "image_roots": [str(value) for value in folders],
            "gravity_feed_g_per_h": 0.0,
            "gravity_start": window[0].isoformat(),
            "gravity_duration_h": 48.0,
            "gravity_start_source": (
                "Project adapter fallback; confirm the overflow-feed schedule in review."
            ),
            "gravity_schedule_verified": False,
            "configuration_source": "darsia-fluidflower-project-adapter",
            "injection_start": window[0].isoformat(),
            "injection_end": window[1].isoformat(),
            "cells": None,
        }
    if not runs:
        raise ValueError("No experiment currently uses scale-display images.")
    payload = {
        "schema": 1,
        "project_id": project.project_id,
        "overflow_revision_id": str(
            (project.manifest.get("overflow") or {}).get("revision_id") or "working"
        ),
        "runs": runs,
    }
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def _reader_python(reader: Path) -> Path:
    """Resolve a Python runtime that includes the reader's numerical dependencies."""

    configured = os.environ.get("DARSIA_FLUIDFLOWER_OVERFLOW_PYTHON")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    virtual_environment = os.environ.get("VIRTUAL_ENV")
    if virtual_environment:
        environment = Path(virtual_environment).expanduser()
        candidates.extend(
            [environment / "Scripts" / "python.exe", environment / "bin" / "python"]
        )
    repository = reader.parents[1]
    candidates.extend(
        [
            repository / ".venv" / "Scripts" / "python.exe",
            repository / ".venv" / "bin" / "python",
            Path(sys.executable),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "No Python runtime is available for the scale-image reader. Set "
        "DARSIA_FLUIDFLOWER_OVERFLOW_PYTHON to an environment containing numpy, "
        "scipy, scikit-learn, Pillow, and openpyxl."
    )


def _reader_error(stderr_path: Path, *, limit: int = 16) -> str:
    try:
        lines = stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-limit:]).strip()


def _wait_for_reader(
    process: subprocess.Popen[Any],
    url: str,
    stderr_path: Path,
    *,
    timeout_s: float,
) -> None:
    """Return only after the reader API is reachable or fail with its real log."""

    deadline = monotonic() + max(1.0, float(timeout_s))
    health_url = url.rstrip("/") + "/api/options"
    while monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            detail = _reader_error(stderr_path)
            suffix = f"\n\n{detail}" if detail else ""
            raise RuntimeError(
                f"Scale-image reader exited during startup (code {return_code})."
                + suffix
            )
        try:
            with urlopen(health_url, timeout=0.75) as response:
                if 200 <= int(response.status) < 300:
                    return
        except (HTTPError, URLError, TimeoutError, OSError):
            sleep(0.1)
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
    detail = _reader_error(stderr_path)
    suffix = f"\n\n{detail}" if detail else ""
    raise TimeoutError(
        f"Scale-image reader did not become ready within {timeout_s:.0f} seconds."
        + suffix
    )


def launch_scale_image_reader(
    project: Any,
    config: Mapping[str, Any],
    *,
    program: Path | None = None,
    run_ids: Sequence[str] = (),
    startup_timeout_s: float = 45.0,
) -> dict[str, Any]:
    """Start the local reader and return only after its API is reachable."""

    reader = _find_reader(program)
    python = _reader_python(reader)
    adapter = write_scale_reader_config(project, config, run_ids=run_ids)
    output = project.root / "results" / "overflow" / "scale_review"
    output.mkdir(parents=True, exist_ok=True)
    logs = project.root / "history" / "overflow_reader"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005
    stdout_path = logs / f"reader-{stamp}.stdout.log"
    stderr_path = logs / f"reader-{stamp}.stderr.log"
    port = _available_port()
    command = [
        str(python),
        str(reader),
        "--config",
        str(adapter),
        "serve",
        "--output-dir",
        str(output),
        "--raw-root",
        str(project.root),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-browser",
        "--configured-runs-only",
    ]
    creation_flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=str(reader.parents[1]),
            stdout=stdout,
            stderr=stderr,
            creationflags=creation_flags,
        )
    url = f"http://127.0.0.1:{port}/"
    _wait_for_reader(
        process,
        url,
        stderr_path,
        timeout_s=startup_timeout_s,
    )
    persisted_result = {
        "pid": process.pid,
        "url": url,
        "config_path": str(adapter),
        "output_path": str(output),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "python": str(python),
    }
    session_path = project.root / "cache" / "overflow_reader" / "session.json"
    session_path.write_text(
        json.dumps(persisted_result, indent=2) + "\n", encoding="utf-8"
    )
    result = dict(persisted_result)
    result["_process"] = process
    return result


__all__ = [
    "launch_scale_image_reader",
    "scale_reader_review_progress",
    "write_scale_reader_config",
]
