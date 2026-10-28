"""TESTONLY: the storage-layer and phase-one writers publish through the N111 retry.

Each writer now serializes into a fresh temporary and commits once, retrying
a transient SMB error. The bytes it writes are the same as the old direct
exclusive write: each test compares them with that old serialization.
Synthetic data only.
"""
import ast
import gzip
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import darsia._opencv_runtime as opencv_runtime
from darsia.fluidflower import storage_n1_outputs, storage_original_photo, storage_outputs
from darsia.fluidflower.pipeline import io_ops, job_queue, phase_statistics

SOURCES = [Path(module.__file__) for module in (storage_outputs, storage_n1_outputs,
                                                phase_statistics, storage_original_photo)]


@pytest.fixture(autouse=True)
def no_real_wait(monkeypatch):
    monkeypatch.setattr(job_queue, "TRANSIENT_RETRY_DELAYS", (0, 0, 0))
    monkeypatch.setattr(time, "time", lambda: 1_700_000_000.0)


def sharing_error():
    value = PermissionError("synthetic sharing violation")
    value.winerror = 32
    return value


def fail_first_fsync(monkeypatch):
    """One transient error at the first commit flush; later flushes succeed."""
    original, calls = io_ops.os.fsync, []

    def fsync(fd):
        calls.append(fd)
        if len(calls) == 1:
            raise sharing_error()
        return original(fd)
    monkeypatch.setattr(io_ops.os, "fsync", fsync)
    return calls


def old_json_TESTONLY(value, *, indent=None):
    """The old direct writers' exact JSON bytes."""
    if indent is None:
        return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                           separators=(",", ":")) + "\n").encode("utf8")
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                       indent=indent) + "\n").encode("utf8")


def old_npz_TESTONLY(path, arrays):
    """The old direct writer's exact NPZ, into an exclusive file."""
    with Path(path).open("xb") as stream:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for name, array in arrays.items():
                with archive.open(name + ".npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, np.asarray(array), allow_pickle=False)
    return Path(path).read_bytes()


def png_TESTONLY(pixels, mode=None):
    stream = io.BytesIO()
    Image.fromarray(pixels, **({} if mode is None else dict(mode=mode))).save(stream, format="PNG")
    return stream.getvalue()


def test_the_json_writer_keeps_its_bytes_and_retries(tmp_path, monkeypatch):
    calls = fail_first_fsync(monkeypatch)
    value = dict(b=[1, 2.5], a="é", c=None)
    storage_outputs._write_json(tmp_path / "receipt.json", value)
    assert (tmp_path / "receipt.json").read_bytes() == old_json_TESTONLY(value) and len(calls) == 2
    assert len(list(tmp_path.glob(".receipt.json.*.partial"))) == 1


def test_the_npz_writer_keeps_its_bytes_and_retries(tmp_path, monkeypatch):
    arrays = dict(a=np.arange(12, dtype=np.float32).reshape(3, 4), b=np.ones((2, 2), bool))
    expected = old_npz_TESTONLY(tmp_path / "old.npz", arrays)
    calls = fail_first_fsync(monkeypatch)
    storage_outputs._write_npz(tmp_path / "frame.npz", arrays)
    assert (tmp_path / "frame.npz").read_bytes() == expected and len(calls) == 2


def test_an_existing_target_is_still_refused(tmp_path):
    (tmp_path / "peer.json").write_bytes(b"peer")
    with pytest.raises(FileExistsError):
        storage_outputs._write_json(tmp_path / "peer.json", {})
    assert (tmp_path / "peer.json").read_bytes() == b"peer"


def test_the_phase_one_copy_keeps_its_bytes_checks_its_source_and_retries(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_bytes(b"phase-one bytes" * 1000)
    bound = dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 size_bytes=source.stat().st_size)
    reader = type("Reader", (), dict(path=lambda self, value: value["path"]))()
    calls = fail_first_fsync(monkeypatch)
    copied = storage_n1_outputs._copy(bound, tmp_path / "copy.json", tmp_path / "final", reader, kind="receipt")
    assert (tmp_path / "copy.json").read_bytes() == source.read_bytes() and len(calls) == 2
    assert copied == dict(path=str(tmp_path / "final" / "copy.json"), sha256=bound["sha256"],
                          size_bytes=bound["size_bytes"], kind="receipt")
    with pytest.raises(ValueError, match="changed during exact storage copy"):
        storage_n1_outputs._copy(dict(bound, sha256="0" * 64), tmp_path / "other.json", tmp_path, reader,
                                 kind="receipt")


def test_the_phase_one_before_png_keeps_its_bytes_and_retries(tmp_path, monkeypatch):
    capture = dict(g3_pixel=np.linspace(0, 1, 6, dtype=np.float32).reshape(2, 3),
                   flatfield_valid=np.ones((2, 3), bool), source_box_xywh=[1, 1, 3, 2])
    la, _ = storage_outputs.encode_before(capture, (4, 5))
    calls = fail_first_fsync(monkeypatch)
    result = phase_statistics._save_before(tmp_path, dict(source=dict(TESTONLY="source")), capture, (4, 5))
    png = tmp_path / "storage_gas_probability_before_rules.png"
    assert png.read_bytes() == png_TESTONLY(la, "LA") and len(calls) == 3  # One retry, PNG and receipt.
    assert result["state"] == "complete" and (tmp_path / "phase_before.json").is_file()


def test_the_phase_one_statistics_keep_their_gzip_bytes_and_name(tmp_path, monkeypatch):
    monkeypatch.setattr(opencv_runtime, "require_ipp_off", lambda stage: dict(stage=stage))
    statistics = dict(regions=[dict(name="TESTONLY", mass=1.5)])
    calls = fail_first_fsync(monkeypatch)
    phase_statistics._save_statistics(tmp_path, dict(statistics=statistics, sandmap=dict(TESTONLY=True)))
    path = tmp_path / "native_statistics.json.gz"
    old = tmp_path / "old" / "native_statistics.json.gz"
    old.parent.mkdir()
    with old.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=3) as compressed:
            compressed.write(json.dumps(statistics, allow_nan=False, separators=(",", ":")).encode("utf-8"))
    assert path.read_bytes() == old.read_bytes() and len(calls) == 3  # One retry, gzip and report.
    assert b"native_statistics.json\x00" in path.read_bytes()[:64]


def test_the_original_photograph_keeps_its_png_and_receipt_bytes(tmp_path, monkeypatch):
    rgb = np.random.default_rng(2).integers(0, 256, (4, 6, 3), dtype=np.uint8)
    rgba = np.zeros((5, 7, 4), np.uint8)
    rgba[1:5, 1:7, :3], rgba[1:5, 1:7, 3] = rgb, 255
    calls = fail_first_fsync(monkeypatch)
    result = storage_original_photo._publish(tmp_path / "registered_original.png", tmp_path / "receipt.json",
                                             rgb, rgba, (0, 0, 6, 4), dict(TESTONLY="fields"))
    assert (tmp_path / "registered_original.png").read_bytes() == png_TESTONLY(rgba) and len(calls) == 3
    receipt = json.loads((tmp_path / "receipt.json").read_bytes())
    assert (tmp_path / "receipt.json").read_bytes() == old_json_TESTONLY(receipt, indent=2)
    assert result["photograph"] == receipt["photograph"]


def test_the_frame_moves_use_the_retried_rename(monkeypatch):
    source = Path(storage_outputs.__file__).read_text(encoding="utf-8")
    write_frame = next(node for node in ast.parse(source).body
                       if isinstance(node, ast.FunctionDef) and node.name == "write_frame")
    calls = [node for node in ast.walk(write_frame) if isinstance(node, ast.Call)]
    assert not any(isinstance(call.func, ast.Attribute) and call.func.attr == "rename" for call in calls)
    assert any(isinstance(call.func, ast.Name) and call.func.id == "rename" for call in calls)


@pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.name)
def test_no_direct_exclusive_write_remains_in_these_writers(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "open":
            modes = [arg.value for arg in node.args if isinstance(arg, ast.Constant)]
            assert "xb" not in modes, path.name + ":" + str(node.lineno)
