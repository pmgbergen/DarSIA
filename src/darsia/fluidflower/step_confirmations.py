"""Explicit user confirmations of workflow steps, kept in the series folder.

A step's checkmark means that the user confirmed it. Each confirmation records
a digest of what the step's status found (its summary and saved files), so a
later change of that saved choice shows as changed since confirmed. The file
keeps every confirmation and reopening in order; nothing is removed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path

FORMAT = "darsia-step-confirmations-v1"
FILE_NAME = "step_confirmations.json"
MAX_BYTES = 4 * 1024 * 1024
# Windows can hold a just-written file briefly (indexers, antivirus, SMB).
_REPLACE_DELAYS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)


class ConfirmationsChangedError(RuntimeError):
    """The confirmations file changed between reading and replacing it."""


def confirmations_path(series_root) -> Path:
    """The series' confirmations file, in its configuration folder."""
    from .series_paths import series_folder_path
    return series_folder_path(series_root, "config") / FILE_NAME


def step_key(logical) -> str:
    """Name a step by its sidebar action and item, e.g. ``method:use_for_series``."""
    action, item = logical
    return f"{action}:{item}"


# Prefix of a basis in which a file enters by what the step uses from it (CONTENT_BASIS).
CONTENT_BASIS = "content-v1:"


def step_basis(state) -> str:
    """Digest of what a step's status found: state, summary and each saved file.

    The saved files enter with path, kind, size and modification time as the
    status read them. A file whose status also says what the step uses from it
    (its ``content``, such as an injection protocol's schedule without the
    positions) enters by that instead of its size and time, and the basis then
    starts with CONTENT_BASIS. series.toml is left out: it holds many steps'
    choices, and a step that saves its choice there names that choice in its
    summary.
    """
    if state is None:
        return ""
    files, by_content = [], False
    for item in state.artifacts:
        if Path(item.path).name.casefold() == "series.toml":
            continue
        content = getattr(item, "content", "")
        by_content = by_content or bool(content)
        files.append((item.kind, str(item.path), content) if content
                     else (item.kind, str(item.path), item.size_bytes, item.modified))
    files.sort(key=lambda row: (row[0], row[1]))
    text = json.dumps(dict(state=state.state, summary=state.summary, files=files), sort_keys=True)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return CONTENT_BASIS + digest if by_content else digest


def _document(raw: bytes) -> dict:
    if not raw:
        return dict(format=FORMAT, events=[])
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict) or value.get("format") != FORMAT or not isinstance(value.get("events"), list):
        raise ValueError("Not a DarSIA step confirmations file")
    return value


def _read_bytes(path: Path) -> bytes:
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
    except FileNotFoundError:
        return b""
    if len(raw) > MAX_BYTES:
        raise ValueError(f"Step confirmations exceed {MAX_BYTES} bytes: {path}")
    return raw


def read_confirmations(path) -> dict[str, dict]:
    """Return the latest event of each step: ``{step: {action, basis, utc}}``."""
    latest = {}
    for event in _document(_read_bytes(Path(path)))["events"]:
        if isinstance(event, dict) and isinstance(event.get("step"), str):
            latest[event["step"]] = event
    return latest


def record_step(path, step: str, action: str, basis: str = "", *, now=None) -> dict[str, dict]:
    """Append one ``confirm`` or ``reopen`` event and return the latest events.

    The file is replaced only if it still has the bytes this call read, so a
    confirmation saved meanwhile on another computer is never lost.
    """
    if action not in {"confirm", "reopen"}:
        raise ValueError(f"Unknown confirmation action: {action}")
    path = Path(path)
    original = _read_bytes(path)
    document = _document(original)
    moment = now or dt.datetime.now(dt.timezone.utc)
    document["events"].append(dict(step=step, action=action, basis=basis,
                                   utc=moment.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
    data = (json.dumps(document, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_bytes(data)
    try:
        for delay in (*_REPLACE_DELAYS, None):
            if _read_bytes(path) != original:
                raise ConfirmationsChangedError("Step confirmations changed while saving; try again.")
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if delay is None:
                    raise
                time.sleep(delay)
    finally:
        temporary.unlink(missing_ok=True)
    return read_confirmations(path)


def confirmation_state(event, basis: str) -> str:
    """``confirmed``, ``changed`` (confirmed for other saved output), ``earlier`` or ``open``.

    ``earlier`` is a confirmation recorded before the step's files entered by
    what the step uses from them (step_basis): it cannot be compared with the
    current basis, so the step is confirmed once more.
    """
    if not isinstance(event, dict) or event.get("action") != "confirm":
        return "open"
    saved = str(event.get("basis", ""))
    if saved == basis:
        return "confirmed"
    if basis.startswith(CONTENT_BASIS) and not saved.startswith(CONTENT_BASIS):
        return "earlier"
    return "changed"
