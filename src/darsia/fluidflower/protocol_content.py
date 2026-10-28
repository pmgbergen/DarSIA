"""What Setup uses from an injection protocol, as a digest that survives edits it does not use.

A Setup Build pins every protocol file by its bytes. For an injection protocol
it also records the digest of what Setup and its later consumers use from the
run's sheet: the interval ids, start and end times and mass rates, read by the
ordinary ``InjectionProtocol`` reader. The injection positions are not part of
it, so correcting a position does not make a completed Setup stale. A consumer
then reads the current file, records its current bytes as consumed and
continues; any other change refuses as before. Rows without a digest (Setups
made before it) keep their exact byte check.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

FORMAT = "darsia-injection-schedule-v1"
# The columns of InjectionProtocol's table that no Setup step uses.
POSITION_COLUMNS = ("location_x", "location_y")
_digests: dict = {}


def _descriptor(path):
    raw = Path(path).read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _physical(path):
    """The file this machine reads for a recorded path (configured storage aliases)."""
    from .storage_paths import resolve_storage_path
    return resolve_storage_path(path)


def injection_schedule_digest(path, sheet=None):
    """SHA-256 of an injection protocol's ids, start and end times and mass rates.

    ``path`` is the file this machine reads and ``sheet`` the run's sheet of a
    workbook (None for a CSV file). The positions are left out.
    """
    from darsia import InjectionProtocol
    source = Path(path)
    raw = source.read_bytes()
    key = (str(source), sheet, hashlib.sha256(raw).hexdigest())
    if key not in _digests:
        table = InjectionProtocol((source, sheet) if sheet is not None else source).df
        rows = [dict(id=int(row["id"]), start=row["start"].isoformat(), end=row["end"].isoformat(),
                     rate_kg_s=repr(float(row["rate_kg_s"])))
                for _, row in table.drop(columns=list(POSITION_COLUMNS)).iterrows()]
        _digests[key] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()
    return _digests[key]


def with_content(rows, *, physical=_physical):
    """Protocol input rows with the injection schedule digest added to each injection row.

    ``physical`` maps a row's recorded path to the file this machine reads.
    Other roles stay as they are, and so does an injection row whose file the
    InjectionProtocol reader cannot parse: it keeps its exact byte check.
    """
    result = []
    for row in rows:
        row = deepcopy(row)
        if row["role"] == "injection":
            try:
                digest = injection_schedule_digest(physical(row["source"]["path"]), row["sheet"])
            except (AssertionError, KeyError, TypeError, ValueError):
                digest = None
            if digest is not None:
                row["content"] = dict(format=FORMAT, sha256=digest)
        result.append(row)
    return result


def without_content(rows):
    """Protocol input rows as the file/sheet/folder binding alone, for exact comparisons."""
    return [{key: deepcopy(value) for key, value in row.items() if key != "content"} for row in rows]


def require_content(row):
    """Check a protocol row's optional content field; return the row's keys without it."""
    content = row.get("content")
    if content is not None and (row.get("role") != "injection" or not isinstance(content, dict)
                                or set(content) != {"format", "sha256"} or content["format"] != FORMAT
                                or not isinstance(content["sha256"], str) or len(content["sha256"]) != 64):
        raise ValueError("A protocol row's content must be the injection schedule digest")
    return set(row) - {"content"}


def current_source(row, *, check=None, physical=_physical):
    """The descriptor of the protocol file a consumer reads for a pinned protocol row.

    A row without a schedule digest returns its recorded pin, so the consumer's
    own exact check applies as before. For a row with one, ``check`` (the
    consumer's own verification of a pin; it raises ValueError) is tried first
    and the pin is returned when it passes. Otherwise the current file's
    descriptor at the same recorded path is returned when that file still has
    the recorded schedule; any other change raises ValueError. ``physical`` maps
    the recorded path to the file this machine reads.
    """
    pin = row["source"]
    if row.get("content") is None:
        return pin
    if check is not None:
        try:
            check(pin)
            return pin
        except ValueError:
            pass
    path = Path(physical(pin["path"]))
    current = _descriptor(path)
    if (current["sha256"], current["size_bytes"]) == (pin["sha256"], pin["size_bytes"]):
        return pin
    require_content(row)
    if injection_schedule_digest(path, row["sheet"]) != row["content"]["sha256"]:
        raise ValueError("The protocol file changed beyond what its Setup records allow: " + str(pin["path"]))
    return dict(pin, sha256=current["sha256"], size_bytes=current["size_bytes"])
