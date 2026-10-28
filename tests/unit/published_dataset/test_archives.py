"""Stored archives are byte-for-byte deterministic, checked while written, and sized exactly."""

import hashlib
import zipfile

import pytest

from darsia.fluidflower.published_dataset import archives
from darsia.fluidflower.published_dataset.archives import Member


def _members(tmp_path):
    (tmp_path / "b.bin").write_bytes(b"b" * 1000)
    (tmp_path / "a.bin").write_bytes(b"a" * 10)
    sha = lambda data: hashlib.sha256(data).hexdigest()
    return [Member("folder/b.bin", tmp_path / "b.bin", sha(b"b" * 1000), 1000),
            Member("a.bin", tmp_path / "a.bin"), Member("generated/c.json", data=b"{}\n")]


def test_same_members_give_the_same_stored_bytes_and_exact_size(tmp_path):
    members = _members(tmp_path)
    first = archives.write_stored_archive(tmp_path / "one.zip", members)
    second = archives.write_stored_archive(tmp_path / "two.zip", list(reversed(members)))
    assert (tmp_path / "one.zip").read_bytes() == (tmp_path / "two.zip").read_bytes()
    assert first["sha256"] == second["sha256"] == archives.file_sha256(tmp_path / "one.zip")
    assert first["size_bytes"] == archives.stored_size(members) == (tmp_path / "one.zip").stat().st_size
    assert [row["name"] for row in first["members"]] == ["a.bin", "folder/b.bin", "generated/c.json"]
    listed = archives.archive_members(tmp_path / "one.zip")
    assert all(row["stored"] for row in listed) and [row["size_bytes"] for row in listed] == [10, 1000, 3]


def test_a_changed_source_stops_the_write_and_leaves_no_archive(tmp_path):
    members = _members(tmp_path)
    (tmp_path / "b.bin").write_bytes(b"x" * 1000)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        archives.write_stored_archive(tmp_path / "bad.zip", members)
    assert not (tmp_path / "bad.zip").exists()
    with pytest.raises(ValueError, match="size differs"):
        archives.write_stored_archive(tmp_path / "bad.zip", [Member("a.bin", tmp_path / "a.bin", size_bytes=11)])
    assert not (tmp_path / "bad.zip").exists()


def test_member_names_must_be_unique_and_relative(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"a")
    for members in ([Member("a.bin", tmp_path / "a.bin"), Member("A.BIN", tmp_path / "a.bin")],
                    [Member("../a.bin", tmp_path / "a.bin")], [Member("C:/a.bin", tmp_path / "a.bin")]):
        with pytest.raises(ValueError):
            archives.write_stored_archive(tmp_path / "x.zip", members)
    (tmp_path / "exists.zip").write_bytes(b"")
    with pytest.raises(FileExistsError):
        archives.write_stored_archive(tmp_path / "exists.zip", [Member("a.bin", tmp_path / "a.bin")])
    assert (tmp_path / "exists.zip").read_bytes() == b""


def test_extract_checks_each_member_and_never_overwrites(tmp_path):
    members = _members(tmp_path)
    archives.write_stored_archive(tmp_path / "one.zip", members)
    target = tmp_path / "out"
    written = archives.extract_members(tmp_path / "one.zip", target, names=["folder/b.bin"],
                                       expected={"folder/b.bin": members[0].sha256})
    assert written == [target / "folder" / "b.bin"] and written[0].read_bytes() == b"b" * 1000
    with pytest.raises(FileExistsError):
        archives.extract_members(tmp_path / "one.zip", target, names=["folder/b.bin"])
    assert written[0].read_bytes() == b"b" * 1000
    with pytest.raises(ValueError, match="differs"):
        archives.extract_members(tmp_path / "one.zip", tmp_path / "other", names=["a.bin"],
                                 expected={"a.bin": "0" * 64})
    assert not (tmp_path / "other" / "a.bin").exists()


def test_size_follows_the_zip64_layout_of_large_members(tmp_path, monkeypatch):
    # A small ZIP64 limit lets tiny files take the large-file paths of the writer.
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 1000)
    members = []
    for index, size in enumerate((10, 960, 1001, 1500)):
        (tmp_path / f"m{index}").write_bytes(bytes([index]) * size)
        members.append(Member(f"m{index}", tmp_path / f"m{index}"))
    row = archives.write_stored_archive(tmp_path / "big.zip", members)
    assert row["size_bytes"] == archives.stored_size(members)
    with zipfile.ZipFile(tmp_path / "big.zip") as archive:
        assert [archive.read(f"m{index}") == bytes([index]) * size
                for index, size in enumerate((10, 960, 1001, 1500))] == [True] * 4


def test_copy_verified(tmp_path):
    (tmp_path / "a").write_bytes(b"abc")
    digest = archives.copy_verified(tmp_path / "a", tmp_path / "x" / "b", sha256=hashlib.sha256(b"abc").hexdigest())
    assert digest == hashlib.sha256(b"abc").hexdigest() and (tmp_path / "x" / "b").read_bytes() == b"abc"
    with pytest.raises(ValueError):
        archives.copy_verified(tmp_path / "a", tmp_path / "x" / "c", sha256="0" * 64)
    assert not (tmp_path / "x" / "c").exists()
    with pytest.raises(FileExistsError):
        archives.copy_verified(tmp_path / "x" / "b", tmp_path / "a")
    assert (tmp_path / "a").read_bytes() == b"abc"
