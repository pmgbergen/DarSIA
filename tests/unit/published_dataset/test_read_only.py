"""A protected dataset folder refuses every write route of the process and still reads."""

import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pytest

from darsia.fluidflower.published_dataset import read_only
from darsia.fluidflower.published_dataset.layout import MANIFEST


def snapshot(folder):
    return {str(path): (path.stat().st_size, path.stat().st_mtime_ns, path.is_dir())
            for path in Path(folder).rglob("*")}


@pytest.fixture
def protected(tmp_path):
    root = tmp_path / "dataset"
    (root / "config").mkdir(parents=True)
    (root / MANIFEST).write_text("{}", encoding="utf-8")
    (root / "config" / "a.json").write_text('{"a": 1}', encoding="utf-8")
    np.save(root / "config" / "layer.npy", np.arange(4))
    (tmp_path / "outside.json").write_text("{}", encoding="utf-8")
    (tmp_path / "source").mkdir()
    (tmp_path / "source" / "b.json").write_text("{}", encoding="utf-8")
    read_only.protect(root)
    try:
        yield root
    finally:
        read_only.unprotect(root)


WRITES = {
    "open for writing": lambda root: open(root / "config" / "a.json", "w"),
    "append": lambda root: open(root / "config" / "a.json", "a"),
    "exclusive create": lambda root: open(root / "new.txt", "x"),
    "update in place": lambda root: open(root / "config" / "a.json", "r+b"),
    "write_text": lambda root: (root / "new.txt").write_text("x"),
    "os.open with create": lambda root: os.open(root / "new.bin", os.O_WRONLY | os.O_CREAT),
    "new folder": lambda root: (root / "folder").mkdir(),
    "remove": lambda root: os.remove(root / "config" / "a.json"),
    "remove a folder": lambda root: os.rmdir(root / "config"),
    "rename out": lambda root: os.rename(root / "config" / "a.json", root.parent / "moved.json"),
    "replace in": lambda root: os.replace(root.parent / "outside.json", root / "config" / "a.json"),
    "move in": lambda root: shutil.move(root.parent / "outside.json", root / "moved.json"),
    "remove a tree": lambda root: shutil.rmtree(root / "config"),
    "copy in": lambda root: shutil.copyfile(root.parent / "outside.json", root / "copy.json"),
    "copy a tree in": lambda root: shutil.copytree(root.parent / "source", root / "tree"),
    "time stamp": lambda root: os.utime(root / "config" / "a.json"),
    "touch": lambda root: (root / "config" / "a.json").touch(),
    "mode": lambda root: os.chmod(root / "config" / "a.json", 0o444),
    "numpy": lambda root: np.save(root / "array.npy", np.zeros(2)),
}


@pytest.mark.parametrize("name", sorted(WRITES))
def test_every_write_route_is_refused_inside_a_protected_folder(protected, name):
    before = snapshot(protected)
    with pytest.raises(read_only.ReadOnlyDatasetError):
        WRITES[name](protected)
    assert snapshot(protected) == before


def test_a_temporary_file_cannot_be_made_inside(protected):
    before = snapshot(protected)
    with pytest.raises(OSError):  # tempfile retries a refused name on Windows, then gives up
        tempfile.NamedTemporaryFile(dir=protected)
    assert snapshot(protected) == before


def test_reads_existing_folders_and_other_folders_stay_usable(protected, tmp_path):
    assert (protected / "config" / "a.json").read_text(encoding="utf-8") == '{"a": 1}'
    with open(protected / "config" / "a.json", "rb") as stream:
        assert stream.read() == b'{"a": 1}'
    assert np.load(protected / "config" / "layer.npy").tolist() == [0, 1, 2, 3]
    assert sorted(os.listdir(protected)) == ["config", MANIFEST]
    (protected / "config").mkdir(exist_ok=True)  # An existing folder: nothing is created.
    (tmp_path / "results" / "report").mkdir(parents=True)
    (tmp_path / "results" / "report" / "report.json").write_text("{}", encoding="utf-8")
    shutil.copyfile(protected / "config" / "a.json", tmp_path / "results" / "copy.json")
    assert (tmp_path / "results" / "copy.json").read_bytes() == b'{"a": 1}'


def test_unprotect_ends_the_refusals_and_roots_are_listed(protected):
    key = os.path.normcase(str(protected.resolve()))
    assert key in read_only.protected_roots()
    read_only.unprotect(protected)
    assert key not in read_only.protected_roots()
    (protected / "new.txt").write_text("x", encoding="utf-8")
    assert (protected / "new.txt").read_text(encoding="utf-8") == "x"


def test_a_sibling_with_the_same_prefix_is_not_protected(protected, tmp_path):
    sibling = tmp_path / (protected.name + "-results")
    sibling.mkdir()
    (sibling / "report.json").write_text("{}", encoding="utf-8")
    assert (sibling / "report.json").is_file()


def test_dataset_root_finds_the_manifest_folder_from_inside(protected, tmp_path):
    assert read_only.dataset_root(protected / "config" / "a.json") == protected.resolve()
    assert read_only.dataset_root(protected) == protected.resolve()
    assert read_only.dataset_root(tmp_path) is None
