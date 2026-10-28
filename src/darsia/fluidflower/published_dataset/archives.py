"""Stored (uncompressed) ZIP archives of a published record.

Raw images, NumPy archives and configurations are stored byte for byte, with
no recompression. Member order, timestamps and attributes are fixed, so the
same members always give the same archive bytes. Every member's SHA-256 is
checked against its expected value while it is copied.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
from typing import Callable
import zipfile

from .layout import record_relative

CHUNK_BYTES = 8 * 1024 * 1024
FIXED_TIME = (1980, 1, 1, 0, 0, 0)
FILE_MODE = 0o100644


@dataclass(frozen=True)
class Member:
    """One archive member: its name in the archive and its bytes (a file or in memory)."""

    name: str
    source: Path | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    data: bytes | None = None
    generate: Callable[[], bytes] | None = None

    def size(self) -> int:
        """The member's size in bytes."""
        if self.data is not None:
            return len(self.data)
        if self.generate is not None:
            if self.size_bytes is None or self.sha256 is None:
                raise ValueError("Generated members need their checked size and SHA-256")
            return self.size_bytes
        return Path(self.source).stat().st_size if self.size_bytes is None else self.size_bytes

    def chunks(self):
        """The member's bytes, in chunks."""
        if self.data is not None:
            yield self.data
            return
        if self.generate is not None:
            yield self.generate()
            return
        with Path(self.source).open("rb") as stream:
            yield from iter(lambda: stream.read(CHUNK_BYTES), b"")


def file_sha256(path: str | Path) -> str:
    """The SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stored_size(members) -> int:
    """The exact byte size of a stored archive of these members, before writing it.

    Local headers, ZIP64 extra fields, the central directory and its end
    records are counted as Python's writer lays them out.
    """
    offset = central = count = 0
    for member in members:
        name = len(member.name.encode("utf-8"))
        size = member.size()
        header_offset = offset
        # The writer reserves a ZIP64 local field when the size may pass the limit.
        offset += 30 + name + (20 if size * 1.05 > zipfile.ZIP64_LIMIT else 0) + size
        extras = (2 if size > zipfile.ZIP64_LIMIT else 0) + (1 if header_offset > zipfile.ZIP64_LIMIT else 0)
        central += 46 + name + (4 + 8 * extras if extras else 0)
        count += 1
    end = 22
    if count > 0xFFFF or offset > zipfile.ZIP64_LIMIT or central > zipfile.ZIP64_LIMIT:
        end += 56 + 20
    return offset + central + end


def write_stored_archive(path: str | Path, members) -> dict:
    """Write a new stored archive (refusing an existing file) and describe it.

    Each source is streamed into the archive while its SHA-256 and size are
    computed; a source that differs from its expected SHA-256 or size stops
    the write and the partial archive is removed. Returns the archive's name,
    SHA-256, size and its members' SHA-256 and sizes.
    """
    target = Path(path)
    ordered = sorted(members, key=lambda member: member.name)
    names = set()
    for member in ordered:
        record_relative(member.name)
        key = member.name.casefold()
        if key in names:
            raise ValueError(f"Duplicate archive member: {member.name}")
        names.add(key)
    rows = []
    created = False
    try:
        with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            created = True
            for member in ordered:
                size = member.size() if member.data is not None or member.generate is not None else Path(member.source).stat().st_size
                if member.size_bytes is not None and size != member.size_bytes:
                    raise ValueError(f"Source size differs from its record: {member.name}")
                info = zipfile.ZipInfo(member.name, date_time=FIXED_TIME)
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = FILE_MODE << 16
                info.file_size = size
                digest = hashlib.sha256()
                written = 0
                with archive.open(info, "w") as output:
                    for chunk in member.chunks():
                        digest.update(chunk)
                        output.write(chunk)
                        written += len(chunk)
                if written != size:
                    raise ValueError(f"Source changed while it was archived: {member.name}")
                actual = digest.hexdigest()
                if member.sha256 is not None and actual != member.sha256:
                    raise ValueError(f"Source SHA-256 differs from its record: {member.name}")
                rows.append(dict(name=member.name, sha256=actual, size_bytes=size))
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise
    return dict(name=target.name, sha256=file_sha256(target), size_bytes=target.stat().st_size,
                members=rows)


def archive_members(path: str | Path) -> list[dict]:
    """The members of an archive: name, size and whether each is stored uncompressed."""
    with zipfile.ZipFile(path) as archive:
        return [dict(name=info.filename, size_bytes=info.file_size,
                     stored=info.compress_type == zipfile.ZIP_STORED)
                for info in archive.infolist()]


def extract_members(path: str | Path, destination: str | Path, *, names=None,
                    expected=None) -> list[Path]:
    """Extract archive members below ``destination``, checking each SHA-256.

    ``expected`` maps member names to SHA-256 values; a member that differs is
    removed again and the extraction stops. Existing files are never replaced.
    """
    target_root = Path(destination)
    written = []
    with zipfile.ZipFile(path) as archive:
        wanted = archive.namelist() if names is None else list(names)
        for name in wanted:
            relative = record_relative(name)
            target = target_root.joinpath(*relative.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            output = target.open("xb")  # Never replaces a file; only this new file is removed on failure.
            try:
                with archive.open(name) as stream, output:
                    for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
                        digest.update(chunk)
                        output.write(chunk)
                if expected is not None and expected.get(name) not in (None, digest.hexdigest()):
                    raise ValueError(f"Extracted member differs from its manifest: {name}")
            except BaseException:
                output.close()
                target.unlink(missing_ok=True)
                raise
            written.append(target)
    return written


def copy_verified(source: str | Path, target: str | Path, *, sha256: str | None = None) -> str:
    """Copy one file to a new path and return its SHA-256, checking it if given."""
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    output = destination.open("xb")  # Never replaces a file; only this new file is removed on failure.
    try:
        with Path(source).open("rb") as stream, output:
            for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
                digest.update(chunk)
                output.write(chunk)
        if sha256 is not None and digest.hexdigest() != sha256:
            raise ValueError(f"Copied file differs from its expected SHA-256: {source}")
    except BaseException:
        output.close()
        destination.unlink(missing_ok=True)
        raise
    shutil.copystat(source, destination, follow_symlinks=True)
    return digest.hexdigest()
