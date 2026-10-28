"""Read the identity fields of a camera file's EXIF, without changing the file.

Raw formats such as ARW, NEF, DNG and CR2 are TIFF containers; JPEG keeps its
EXIF in an APP1 segment. Only the fields that can name a person or a machine
are read: Artist, Copyright, the Windows author tag, HostComputer and the
camera owner name. Nothing else is decoded and nothing is ever written.
"""

from __future__ import annotations

from pathlib import Path
import struct

# Tag id -> field name, in the first image directory (IFD0) and the EXIF directory.
IFD0_FIELDS = {0x013B: "Artist", 0x8298: "Copyright", 0x013C: "HostComputer",
               0x9C9D: "XPAuthor"}
EXIF_FIELDS = {0xA430: "CameraOwnerName"}
EXIF_POINTER = 0x8769
MAX_ENTRIES = 4096
MAX_DIRECTORIES = 16
MAX_VALUE_BYTES = 64 * 1024


def identity_fields(path: str | Path) -> dict[str, str]:
    """The nonempty identity fields of a TIFF-based or JPEG camera file.

    Unknown or damaged files give an empty result: this is a report, never a
    gate, and the file itself is only read.
    """
    try:
        with Path(path).open("rb") as stream:
            head = stream.read(4)
            stream.seek(0)
            if head[:2] == b"\xff\xd8":
                return _jpeg_fields(stream)
            if head in (b"II*\x00", b"MM\x00*"):
                return _tiff_fields(stream, 0)
    except (OSError, ValueError, struct.error):
        return {}
    return {}


def _jpeg_fields(stream) -> dict[str, str]:
    """The identity fields of a JPEG's APP1 EXIF segment."""
    stream.seek(2)
    while True:
        marker = stream.read(2)
        if len(marker) < 2 or marker[0] != 0xFF or marker[1] in (0xD9, 0xDA):
            return {}
        size = struct.unpack(">H", stream.read(2))[0]
        start = stream.tell()
        if marker[1] == 0xE1 and stream.read(6) == b"Exif\x00\x00":
            return _tiff_fields(stream, start + 6)
        stream.seek(start + size - 2)


def _tiff_fields(stream, base: int) -> dict[str, str]:
    """The identity fields of a TIFF structure starting at ``base``: IFD0 and the EXIF IFD."""
    stream.seek(base)
    order = stream.read(2)
    if order not in (b"II", b"MM"):
        return {}
    endian = "<" if order == b"II" else ">"
    stream.read(2)
    offset = struct.unpack(endian + "I", stream.read(4))[0]
    fields: dict[str, str] = {}
    exif_offset = None
    visited = set()
    for _ in range(MAX_DIRECTORIES):
        if not offset or offset in visited:
            break
        visited.add(offset)
        entries, offset = _directory(stream, base, offset, endian)
        for tag, kind, count, value_offset, inline in entries:
            if tag in IFD0_FIELDS:
                text = _value(stream, base, endian, tag, kind, count, value_offset, inline)
                if text:
                    fields.setdefault(IFD0_FIELDS[tag], text)
            elif tag == EXIF_POINTER and exif_offset is None:
                exif_offset = struct.unpack(endian + "I", inline)[0]
    if exif_offset:
        entries, _next = _directory(stream, base, exif_offset, endian)
        for tag, kind, count, value_offset, inline in entries:
            if tag in EXIF_FIELDS:
                text = _value(stream, base, endian, tag, kind, count, value_offset, inline)
                if text:
                    fields.setdefault(EXIF_FIELDS[tag], text)
    return fields


def _directory(stream, base, offset, endian):
    """One TIFF directory's entries and the offset of the next directory."""
    stream.seek(base + offset)
    count = struct.unpack(endian + "H", stream.read(2))[0]
    if count > MAX_ENTRIES:
        raise ValueError("Implausible TIFF directory size")
    entries = []
    for _ in range(count):
        raw = stream.read(12)
        tag, kind, number = struct.unpack(endian + "HHI", raw[:8])
        inline = raw[8:12]
        entries.append((tag, kind, number, struct.unpack(endian + "I", inline)[0], inline))
    following = struct.unpack(endian + "I", stream.read(4))[0]
    return entries, following


def _value(stream, base, endian, tag, kind, count, value_offset, inline) -> str:
    """A text tag's value (ASCII, BYTE or UNDEFINED), stripped; empty for other types."""
    unit = {1: 1, 2: 1, 7: 1}.get(kind)
    if unit is None or count == 0 or count > MAX_VALUE_BYTES:
        return ""
    size = unit * count
    if size <= 4:
        raw = inline[:size]
    else:
        stream.seek(base + value_offset)
        raw = stream.read(size)
    if tag == 0x9C9D:  # Windows XP tag: UTF-16LE in BYTE values
        text = raw.decode("utf-16-le", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    return text.replace("\x00", " ").strip()
