"""The builder's name scan finds names and absolute paths; validator names are allowed."""

import io
import json
import struct
import zlib

import numpy as np

from darsia.fluidflower.published_dataset import scan
from darsia.fluidflower.published_dataset.scan import NameScanner, ScanName

HOST = ScanName("example-host", "build machine host name")
USER = ScanName("sample_user", "build machine user name")
MACHINE = ScanName("Fixture", "production receipt")


def kinds(hits):
    return sorted({(hit.kind, hit.source, hit.allowed) for hit in hits})


def test_names_match_as_words_with_any_separator_and_ignore_identifiers():
    scanner = NameScanner([HOST, USER, MACHINE], deny=["Fictional Person"])
    text = "made on Example_Host by fictional.person; fixture_mass stays; FIXTURE ran"
    hits = scanner.text(text, file="note.txt", where="line 1")
    assert sorted(hit.source for hit in hits) == ["build machine host name", "deny-list", "production receipt"]
    assert not scanner.text("fixture_mass_eligible and refixture", file="a", where="b")
    assert scanner.text("folder sample_user\\x", file="a", where="b")[0].source == "build machine user name"


def test_absolute_paths_are_found_but_web_links_are_not():
    scanner = NameScanner()
    for value, label in (("C:\\Users\\x\\file", "drive letter"), ("X:/data", "drive letter"),
                         ("\\\\server\\share\\x", "UNC path"), ("\\\\\\\\server\\\\share", "UNC path"),
                         ("//server/share/x", "UNC path"), ("/home/someone/data", "home folder"),
                         ("/Users/someone", "home folder")):
        assert label in {hit.source for hit in scanner.text(value, file="f", where="w")}, value
    assert not scanner.text("see https://doi.org/10.5281/zenodo.1 and m/s or 1/2", file="f", where="w")


def test_reviewer_fields_may_name_the_validator_and_nothing_else():
    scanner = NameScanner(deny=["Fictional Person"])
    document = {"format": "record", "reviewer_name": "Fictional Person",
                "reviews": [{"reviewer": "Fictional Person", "note": "checked by Fictional Person"}],
                "path_in_reviewer": {"reviewer": "C:\\Users\\x"}}
    hits = scanner.document(document, file="record.json")
    allowed = [hit for hit in hits if hit.allowed]
    assert [hit.where for hit in allowed] == ["key /reviewer_name", "key /reviews/0/reviewer"]
    refused = scan.refused(hits)
    assert {hit.where for hit in refused} == {"key /reviews/0/note", "key /path_in_reviewer/reviewer"}
    assert "allowed (validation record)" in allowed[0].describe()
    assert refused[0].as_dict()["allowed"] is False


def test_content_reads_json_toml_text_npz_png_npy_and_binary():
    scanner = NameScanner([HOST])
    hits = scanner.content(b'{"a": {"example-host": 1}}', file="a.json", suffix=".json")
    assert hits[0].where == "key /a/example-host"
    assert scanner.content(b'name = "example-host"\n', file="a.toml", suffix=".toml")[0].where == "key /name"
    assert scanner.content(b"x\nexample-host\n", file="a.csv", suffix=".csv")[0].where == "line 2"
    assert scanner.content(b"{not json example-host", file="a.json", suffix=".json")[0].where == "line 1"
    stream = io.BytesIO()
    np.savez(stream, **{"values": np.arange(3), "labels": np.array(["fine", "example-host"])})
    hits = scanner.content(stream.getvalue(), file="a.npz", suffix=".npz")
    assert [hit.where for hit in hits] == ["array labels[1]"]
    named = io.BytesIO()
    np.savez(named, **{"example-host": np.arange(2)})
    assert scanner.content(named.getvalue(), file="b.npz", suffix=".npz")[0].where == "array name"
    array = io.BytesIO()
    np.save(array, np.array(["example-host"]))
    assert scanner.content(array.getvalue(), file="a.npy", suffix=".npy")[0].where == "value 0"
    text = b"Author\x00example-host"
    chunk = struct.pack(">I", len(text)) + b"tEXt" + text + struct.pack(">I", zlib.crc32(b"tEXt" + text))
    png = b"\x89PNG\r\n\x1a\n" + chunk + struct.pack(">I", 0) + b"IEND" + struct.pack(">I", 0)
    assert scanner.content(png, file="a.png", suffix=".png")[0].where == "PNG tEXt chunk"
    blob = bytes(range(256)) + b"\x00\x01made-by-example-host\x00" + bytes(64)
    assert scanner.content(blob, file="a.bin", suffix=".bin")[0].where.startswith("byte ")
    assert not scanner.content(bytes(range(256)) * 4, file="a.bin", suffix=".bin")


def test_file_names_and_raw_images(tmp_path):
    from tests.unit.published_dataset.synthetic import tiff

    raw = tmp_path / "IMG_1.ARW"
    raw.write_bytes(tiff(b"pixels", artist="Fictional Person"))
    scanner = NameScanner([HOST], deny=["Fictional Person"])
    hits, fields = scanner.raw_image(raw, file="raw_r1.zip:raw/example-host/IMG_1.ARW")
    assert [hit.where for hit in hits] == ["file name"]
    assert fields == [{"file": "raw_r1.zip:raw/example-host/IMG_1.ARW", "field": "Artist",
                       "value": "Fictional Person"}]
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"by": "Fictional Person"}))
    assert {hit.where for hit in scanner.file(config, file="config/example-host.json")} == {
        "file name", "key /by"}
    assert scanner.path("a/example-host/b", file="x.zip", where="member name")[0].where == "member name"


def test_machine_and_receipt_names(monkeypatch):
    monkeypatch.setattr(scan.socket, "gethostname", lambda: "example-host.local")
    monkeypatch.setattr(scan.getpass, "getuser", lambda: "sample_user")
    names = {(item.text, item.source) for item in scan.machine_names()}
    assert ("example-host.local", "build machine host name") in names
    assert ("example-host", "build machine host name") in names
    assert ("sample_user", "build machine user name") in names
    found = scan.receipt_names({"actual_worker": {"machine": "fixture-cpu", "pid": 3},
                                "owner": {"host": "fixture"}, "rows": [{"user": "sample_user"}],
                                "backend": "cpu"})
    assert sorted(item.text for item in found) == ["fixture", "fixture-cpu", "sample_user"]


def test_short_automatic_names_are_skipped_but_deny_list_names_are_not():
    scanner = NameScanner([ScanName("ab", "build machine user name")], deny=["xy"])
    assert [hit.source for hit in scanner.text("ab xy", file="f", where="w")] == ["deny-list"]
    assert scan.record_suffix("a/b.tar.npz") == ".npz"
