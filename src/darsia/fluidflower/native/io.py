"""Small content-bound readers independent of review applications."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from collections.abc import Mapping
from ..pipeline.io_ops import retry_io


@retry_io
def sha256_file(path: str | Path) -> str:
    """Hash the exact bytes of a file without loading its image into memory."""
    from ..pipeline.run_io import current_binding
    binding = current_binding()
    if binding is not None:
        admitted = binding.native_input_digest(path)
        if admitted is not None:
            return admitted
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@retry_io
def read_bound_json(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    """Read a JSON object only after its declared content digest matches."""
    source = Path(path)
    from ..pipeline.run_io import current_binding
    binding = current_binding()
    raw = (binding.native_metadata_bytes(source, expected_sha256)
           if binding is not None else None)
    if raw is None:
        raw = source.read_bytes()
        if (not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or hashlib.sha256(raw).hexdigest() != expected_sha256):
            raise ValueError("Source artifact changed: " + str(source))
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Bound JSON must contain an object: " + str(source))
    return value


def resolve_asset_path(path: str | Path, aliases: Mapping[str, str]) -> Path:
    """Resolve an explicitly listed relocation, leaving unlisted paths intact.

    Aliases identify the same immutable content at a new location. Callers
    still verify each scientific asset against its declared digest.
    """
    from ..pipeline.run_io import current_binding
    binding = current_binding()
    if binding is not None:
        return binding.asset_path(path)
    source = Path(path)
    resolved = str(source.resolve()).casefold()
    for original, destination in aliases.items():
        if (str(Path(original)).casefold() == str(source).casefold()
                or str(Path(original).resolve()).casefold() == resolved):
            return Path(destination)
    return source
