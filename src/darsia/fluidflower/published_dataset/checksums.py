"""The record's checksum list, ``CHECKSUMS.sha256`` (one ``<sha256>  <name>`` per line).

The format is the one ``sha256sum --check`` reads, so a user can verify a
download without DarSIA. The list covers every record file except itself.
"""

from __future__ import annotations

from pathlib import Path
import re

from .archives import file_sha256
from .layout import CHECKSUMS, record_relative

_LINE = re.compile(r"([0-9a-f]{64}) [ *](.+)")


def write_checksums(root: str | Path, digests: dict[str, str]) -> Path:
    """Write the checksum list of ``digests`` ({record name: SHA-256}) into ``root``."""
    lines = []
    for name in sorted(digests):
        record_relative(name)
        if name == CHECKSUMS or re.fullmatch(r"[0-9a-f]{64}", digests[name]) is None:
            raise ValueError(f"Invalid checksum entry: {name}")
        lines.append(f"{digests[name]}  {name}\n")
    path = Path(root) / CHECKSUMS
    with path.open("x", encoding="ascii", newline="\n") as stream:
        stream.writelines(lines)
    return path


def read_checksums(root: str | Path) -> dict[str, str]:
    """Read ``CHECKSUMS.sha256`` of a record or dataset folder."""
    path = Path(root) / CHECKSUMS
    result = {}
    for number, line in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        if not line.strip():
            continue
        match = _LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"{CHECKSUMS} line {number} is not '<sha256>  <name>'")
        name = record_relative(match.group(2))
        if name in result:
            raise ValueError(f"{CHECKSUMS} lists {name} twice")
        result[name] = match.group(1)
    return result


def verify_checksums(root: str | Path, *, sizes: dict[str, int] | None = None,
                     full: bool = True) -> list[str]:
    """Return the problems of a record folder against its checksum list.

    With ``full`` the SHA-256 of every listed file is recomputed; without it,
    only presence and the ``sizes`` from the manifest are checked (fast).
    """
    base = Path(root)
    problems = []
    for name, digest in read_checksums(base).items():
        path = base.joinpath(*name.split("/"))
        if not path.is_file():
            problems.append(f"Missing file: {name}")
            continue
        if sizes is not None and name in sizes and path.stat().st_size != sizes[name]:
            problems.append(f"Size differs from the manifest: {name}")
            continue
        if full and file_sha256(path) != digest:
            problems.append(f"Checksum differs: {name}")
    return problems
