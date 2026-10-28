"""Size-bounded full-state recovery journals shared by review interfaces."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

RECOVERY_JOURNAL_MAX_BYTES = 4 * 1024 * 1024


def append_bounded_snapshot_journal(
    path: Path,
    payload: Mapping[str, Any],
    *,
    max_bytes: int = RECOVERY_JOURNAL_MAX_BYTES,
) -> None:
    """Append a recoverable snapshot without allowing journals to grow forever."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    encoded_bytes = encoded.encode("utf-8")
    compact = (
        path.is_file()
        and path.stat().st_size + len(encoded_bytes) > max(1, int(max_bytes))
    )
    if compact:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            stream.write(encoded_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return
    with path.open("ab") as stream:
        stream.write(encoded_bytes)
        stream.flush()
        os.fsync(stream.fileno())
