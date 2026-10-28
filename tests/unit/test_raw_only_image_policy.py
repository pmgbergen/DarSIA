"""Strict image-file policies and camera RAW metadata (RAW consistency A).

A series uses one kind of file. "raw_only" and "processed_only" select that
family and never fall back to the other; a folder that lacks it is reported.
RAW files give their size and capture time to the integrity scan, series
creation and protocol set-up. LibRaw is replaced by a fake; synthetic TIFF
files named .dng carry the EXIF time. Set DARSIA_TEST_DISPLAY_ARW and
DARSIA_TEST_DISPLAY_NEF to real files to run the last test on them.
"""

import os
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from darsia.fluidflower import image_formats as formats
from darsia.fluidflower import photograph_io
from darsia.fluidflower.data_integrity import IntegrityScanner, IntegrityScanOptions
from darsia.fluidflower.series_creation import _image_time
from darsia.fluidflower.source_discovery import inspect_run_structure
from darsia.presets.workflows.setup.setup_protocols import _extract_exif_datetime

TAKEN = "2022:10:05 15:56:12"


def _jpeg(path, colour=(10, 20, 30)):
    Image.new("RGB", (16, 12), colour).save(path)
    return path


def _dng(path, taken=TAKEN):
    """A small TIFF named like a camera RAW file, with an EXIF DateTime."""

    extratags = [(306, "s", 0, taken, True)] if taken else []
    tifffile.imwrite(path, np.zeros((12, 16), np.uint16), extratags=extratags)
    return path


class _FakeRaw:
    def __init__(self, sizes):
        self.sizes = sizes

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def libraw(monkeypatch):
    """LibRaw reports 120 x 90 for every RAW file."""

    class _Error(Exception):
        pass

    fake = types.SimpleNamespace(
        LibRawError=_Error,
        imread=lambda path: _FakeRaw(types.SimpleNamespace(width=120, height=90, flip=0)),
    )
    monkeypatch.setitem(sys.modules, "rawpy", fake)
    return fake


@pytest.fixture
def folders(tmp_path):
    """A paired folder, a JPEG-only folder and a RAW-only folder."""

    paired = tmp_path / "run" / "paired"
    jpeg_only = tmp_path / "run" / "jpeg_only"
    raw_only = tmp_path / "run" / "raw_only"
    for folder in (paired, jpeg_only, raw_only):
        folder.mkdir(parents=True)
    _jpeg(paired / "DSC00001.JPG")
    _dng(paired / "DSC00001.dng")
    _jpeg(jpeg_only / "DSC00002.JPG")
    _dng(raw_only / "DSC00003.dng")
    return {"root": tmp_path / "run", "paired": paired, "jpeg_only": jpeg_only, "raw_only": raw_only}


def _names(paths):
    return sorted(Path(path).name for path in paths)


# ------------------------------------------------------------- the policies


def test_the_strict_policies_select_one_family_and_never_fall_back(folders):
    paired = list(folders["paired"].iterdir())
    jpeg_only = list(folders["jpeg_only"].iterdir())
    raw_only = list(folders["raw_only"].iterdir())
    select = formats.select_preferred_images
    assert _names(select(paired, formats.IMAGE_FILE_POLICY_RAW_ONLY)) == ["DSC00001.dng"]
    assert _names(select(paired, formats.IMAGE_FILE_POLICY_PROCESSED_ONLY)) == ["DSC00001.JPG"]
    assert select(jpeg_only, formats.IMAGE_FILE_POLICY_RAW_ONLY) == []
    assert select(raw_only, formats.IMAGE_FILE_POLICY_PROCESSED_ONLY) == []


def test_the_preferring_policies_still_fall_back(folders):
    jpeg_only = list(folders["jpeg_only"].iterdir())
    raw_only = list(folders["raw_only"].iterdir())
    paired = list(folders["paired"].iterdir())
    select = formats.select_preferred_images
    assert _names(select(jpeg_only, formats.IMAGE_FILE_POLICY_RAW)) == ["DSC00002.JPG"]
    assert _names(select(raw_only, formats.IMAGE_FILE_POLICY_PROCESSED)) == ["DSC00003.dng"]
    assert _names(select(paired, formats.DEFAULT_IMAGE_FILE_POLICY)) == ["DSC00001.JPG"]
    assert len(select(paired, formats.IMAGE_FILE_POLICY_ALL)) == 2


def test_a_folder_without_the_required_family_is_named(folders):
    missing = formats.missing_image_family
    reason = missing(list(folders["jpeg_only"].iterdir()), formats.IMAGE_FILE_POLICY_RAW_ONLY)
    assert "no camera RAW image" in reason and "1 JPEG/TIFF/PNG" in reason
    reason = missing(list(folders["raw_only"].iterdir()), formats.IMAGE_FILE_POLICY_PROCESSED_ONLY)
    assert "no JPEG/TIFF/PNG image" in reason
    assert missing(list(folders["paired"].iterdir()), formats.IMAGE_FILE_POLICY_RAW_ONLY) == ""
    assert missing([], formats.IMAGE_FILE_POLICY_RAW_ONLY) == ""
    assert missing(list(folders["jpeg_only"].iterdir()), formats.IMAGE_FILE_POLICY_RAW) == ""


def test_the_new_policies_are_known_and_labelled():
    for policy in (formats.IMAGE_FILE_POLICY_RAW_ONLY, formats.IMAGE_FILE_POLICY_PROCESSED_ONLY):
        assert formats.normalize_image_file_policy(policy.upper()) == policy
        assert formats.IMAGE_FILE_POLICY_LABELS[policy].startswith("Only ")
    assert formats.DEFAULT_IMAGE_FILE_POLICY == formats.IMAGE_FILE_POLICY_PROCESSED
    with pytest.raises(ValueError):
        formats.normalize_image_file_policy("raw_preferred_somehow")


# --------------------------------------------------------- RAW capture time


def test_the_raw_capture_time_comes_from_its_exif(tmp_path):
    assert photograph_io.raw_capture_time(_dng(tmp_path / "a.dng")) == datetime(2022, 10, 5, 15, 56, 12)
    assert photograph_io.raw_capture_time(_dng(tmp_path / "b.dng", taken=None)) is None
    broken = tmp_path / "c.arw"
    broken.write_bytes(b"not a tiff")
    assert photograph_io.raw_capture_time(broken) is None


def test_series_creation_and_protocols_read_the_raw_time(tmp_path):
    path = _dng(tmp_path / "frame.dng")
    expected = datetime(2022, 10, 5, 15, 56, 12)
    assert _image_time(path) == expected.timestamp()
    assert _extract_exif_datetime(path) == expected
    untimed = _dng(tmp_path / "untimed.dng", taken=None)
    os.utime(untimed, (1_700_000_000, 1_700_000_000))
    assert _image_time(untimed) == 1_700_000_000
    assert _extract_exif_datetime(untimed) is None


# --------------------------------------------------- discovery and integrity


def test_discovery_names_folders_a_strict_policy_rejects(folders):
    records = {
        Path(row["absolute_path"]).name: row
        for row in inspect_run_structure(folders["root"], image_file_policy=formats.IMAGE_FILE_POLICY_RAW_ONLY)
    }
    assert records["jpeg_only"]["image_count"] == 0
    assert "no camera RAW image" in records["jpeg_only"]["missing_image_family"]
    assert records["paired"]["missing_image_family"] == ""
    assert records["paired"]["image_count"] == 1
    default = inspect_run_structure(folders["root"])
    assert all(row["missing_image_family"] == "" for row in default)


class _Project:
    project_id = "test"

    def __init__(self, sources):
        self._sources = sources

    def current_source_records(self):
        return self._sources


def test_the_integrity_scan_reads_raw_metadata_and_reports_missing_raw(folders, libraw):
    source = {
        "source_id": "images",
        "kind": "clearwater_images",
        "path": str(folders["root"]),
        "image_file_policy": formats.IMAGE_FILE_POLICY_RAW_ONLY,
    }
    scanner = IntegrityScanner(_Project([source]), IntegrityScanOptions(parallel_reads=1))
    paths, _attempts, missing = scanner._enumerate_source(source)
    assert _names(paths) == ["DSC00001.dng", "DSC00003.dng"]
    assert [folder.name for folder, _reason in missing] == ["jpeg_only"]
    row, issue = scanner._inspect_entry(source, folders["raw_only"] / "DSC00003.dng")
    assert issue is None
    assert (row["width"], row["height"]) == (120, 90)
    assert row["exif_datetime"] == "2022-10-05T15:56:12"
    assert row["image_format"] == "RAW"


def test_the_default_policy_still_skips_raw_metadata(folders, libraw):
    source = {"source_id": "images", "kind": "clearwater_images", "path": str(folders["root"])}
    scanner = IntegrityScanner(_Project([source]), IntegrityScanOptions(parallel_reads=1))
    paths, _attempts, missing = scanner._enumerate_source(source)
    assert missing == []
    assert "DSC00003.dng" in _names(paths)
    row, _issue = scanner._inspect_entry(source, folders["raw_only"] / "DSC00003.dng")
    assert row["width"] is None and row["exif_datetime"] == ""


# ------------------------------------------------------------- real files


@pytest.mark.parametrize("variable", ["DARSIA_TEST_DISPLAY_ARW", "DARSIA_TEST_DISPLAY_NEF"])
def test_real_raw_files_give_their_capture_time(variable):
    value = os.environ.get(variable)
    if not value or not Path(value).is_file():
        pytest.skip(f"{variable} does not name a RAW file here")
    taken = photograph_io.raw_capture_time(value)
    assert isinstance(taken, datetime) and taken.year >= 2000
