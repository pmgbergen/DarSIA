"""EXIF identity fields are read from TIFF-based and JPEG camera files, never written."""

import struct

from darsia.fluidflower.published_dataset.exif import identity_fields
from tests.unit.published_dataset.synthetic import tiff


def _tiff_with_exif(endian=b"II"):
    """A TIFF whose IFD0 has XPAuthor and an EXIF pointer to CameraOwnerName."""
    order = "<" if endian == b"II" else ">"
    author = "Fictional Person".encode("utf-16-le") + b"\x00\x00"
    owner = b"Sample Owner\x00"
    ifd0 = 8
    ifd0_size = 2 + 2 * 12 + 4
    exif = ifd0 + ifd0_size
    exif_size = 2 + 12 + 4
    data = exif + exif_size
    head = endian + struct.pack(order + "HI", 42, ifd0)
    body = struct.pack(order + "H", 2)
    body += struct.pack(order + "HHII", 0x9C9D, 1, len(author), data)
    body += struct.pack(order + "HHII", 0x8769, 4, 1, exif) + struct.pack(order + "I", 0)
    body += struct.pack(order + "H", 1) + struct.pack(order + "HHII", 0xA430, 2, len(owner), data + len(author))
    body += struct.pack(order + "I", 0)
    return head + body + author + owner


def test_tiff_fields_in_both_byte_orders(tmp_path):
    for endian in (b"II", b"MM"):
        path = tmp_path / f"{endian.decode()}.NEF"
        path.write_bytes(_tiff_with_exif(endian))
        before = path.read_bytes()
        assert identity_fields(path) == {"XPAuthor": "Fictional Person", "CameraOwnerName": "Sample Owner"}
        assert path.read_bytes() == before


def test_short_inline_value_and_artist(tmp_path):
    path = tmp_path / "a.ARW"
    path.write_bytes(tiff(b"payload", artist="ab"))
    assert identity_fields(path) == {"Artist": "ab"}
    path.write_bytes(tiff(b"payload"))
    assert identity_fields(path) == {}


def test_jpeg_app1_exif(tmp_path):
    exif = b"Exif\x00\x00" + tiff(b"", artist="Fictional Person")
    segment = b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif
    path = tmp_path / "a.jpg"
    path.write_bytes(b"\xff\xd8" + b"\xff\xe0" + struct.pack(">H", 4) + b"JF" + segment + b"\xff\xd9")
    assert identity_fields(path) == {"Artist": "Fictional Person"}


def test_unknown_or_damaged_files_give_nothing(tmp_path):
    for name, data in (("a.txt", b"plain text"), ("b.ARW", b"II*\x00" + b"\xff" * 4),
                       ("c.jpg", b"\xff\xd8\xff\xda")):
        path = tmp_path / name
        path.write_bytes(data)
        assert identity_fields(path) == {}
    assert identity_fields(tmp_path / "missing.ARW") == {}
