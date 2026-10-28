"""Retry only immutable file steps; preserve scientific bytes and decisions."""

import ast
import io
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from darsia.fluidflower.pipeline import io_ops, job_queue, products_publication
from darsia.fluidflower.native import io as native_io, sidecar
from darsia.fluidflower.mass import n1_pipeline


@pytest.fixture(autouse=True)
def no_real_wait(monkeypatch):
    monkeypatch.setattr(job_queue, "TRANSIENT_RETRY_DELAYS", (0, 0, 0))


def sharing_error():
    value = PermissionError("synthetic sharing violation")
    value.winerror = 32
    return value


def test_native_writer_retries_serialization_and_keeps_partial_without_recomputing_values(tmp_path, monkeypatch):
    values = np.array([0x7fc01234, 0x80000000, 0x3f800000], np.uint32).view(np.float32)
    original = np.save
    calls = []
    def save(stream, data, **kwargs):
        calls.append(data)
        if len(calls) == 1:
            stream.write(b"partial")
            raise sharing_error()
        return original(stream, data, **kwargs)
    monkeypatch.setattr(np, "save", save)
    path = tmp_path / "native.npy"
    digest = sidecar.write_log_linear(path, values)
    expected = io.BytesIO()
    original(expected, values, allow_pickle=False)
    assert path.read_bytes() == expected.getvalue()
    assert digest == native_io.sha256_file(path) and len(calls) == 2
    assert len(list(tmp_path.glob("*.partial"))) == 1
    np.testing.assert_array_equal(np.load(path).view(np.uint32), values.view(np.uint32))


def test_existing_target_is_a_protocol_answer_and_never_calls_the_writer(tmp_path):
    path = tmp_path / "peer"
    path.write_bytes(b"peer")
    with pytest.raises(FileExistsError):
        io_ops.publish_file(path, lambda stream: pytest.fail("Writer ran for an existing peer"))
    assert path.read_bytes() == b"peer"


def test_missing_file_is_not_retried(tmp_path, monkeypatch):
    calls = []
    def missing(path, *args, **kwargs):
        calls.append(path)
        raise FileNotFoundError(path)
    monkeypatch.setattr(Path, "open", missing)
    with pytest.raises(FileNotFoundError):
        native_io.sha256_file(tmp_path / "missing")
    assert len(calls) == 1


def test_nested_readers_have_three_retries_total(tmp_path, monkeypatch):
    path = tmp_path / "source"
    path.write_bytes(b"immutable")
    expected = dict(bytes=9, sha256=native_io.sha256_file(path))
    original = Path.open
    calls = []
    def locked(source, *args, **kwargs):
        if source == path:
            calls.append(source)
            raise sharing_error()
        return original(source, *args, **kwargs)
    monkeypatch.setattr(Path, "open", locked)
    with pytest.raises(PermissionError):
        products_publication._verified_bytes(path, expected, limit=100)
    assert len(calls) == 4


def test_product_copy_waits_for_one_locked_source_and_preserves_exact_bytes(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "copy"
    source.write_bytes(bytes(range(256)) * 20)
    expected = dict(bytes=source.stat().st_size, sha256=native_io.sha256_file(source))
    original = Path.open
    calls = []
    def once_locked(path, *args, **kwargs):
        if path == source:
            calls.append(path)
            if len(calls) == 1:
                raise sharing_error()
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", once_locked)
    products_publication._copy(source, expected, destination)
    assert destination.read_bytes() == source.read_bytes() and len(calls) == 3


def test_wrong_copy_hash_is_never_published(tmp_path):
    path = tmp_path / "target"
    with pytest.raises(ValueError, match="frozen source"):
        io_ops.publish_file(path, lambda stream: stream.write(b"wrong"), expected_sha256="0" * 64)
    assert not path.exists() and len(list(tmp_path.glob("*.partial"))) == 1


def test_n1_NPZ_bytes_match_the_preserved_pre_retry_writer(tmp_path):
    root = Path(__file__).resolve().parents[2]
    code = subprocess.check_output(["git", "show", "88cc65d91b83d6f6961f3172bd71722b58aa7f8a:src/darsia/fluidflower/mass/n1_pipeline.py"], cwd=root)
    definition = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == "_write_npz")
    namespace = dict(Path=Path, os=os, asset=n1_pipeline.asset)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "preserved_writer", "exec"), namespace)
    values = dict(signal=np.array([0x7fc01234, 0x80000000, 0x3f800000], np.uint32).view(np.float32),
                  mask=np.array([True, False], bool))
    old, current = tmp_path / "old.npz", tmp_path / "current.npz"
    namespace["_write_npz"](old, values)
    n1_pipeline._write_npz(current, values)
    assert current.read_bytes() == old.read_bytes()


def test_n1_json_retains_canonical_bytes(tmp_path):
    path = tmp_path / "row.json"
    n1_pipeline._write_json(path, dict(b=2, a=[1., None]))
    assert path.read_bytes() == b'{"a":[1.0,null],"b":2}'


def test_input_hash_admission_retries_only_file_errors_and_retains_exact_hash_gate(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import store_runtime
    path = tmp_path / "input.json"
    path.write_bytes(b'{"source":"raw"}')
    bound = n1_pipeline.asset(path)
    reference = dict(sha256=bound["sha256"], bytes=bound["size_bytes"])
    row = store_runtime.ResolvedInput(reference, path, ("input",))
    original = Path.open
    calls = []
    def once_locked(source, *args, **kwargs):
        if source == path:
            calls.append(source)
            if len(calls) == 1:
                raise sharing_error()
        return original(source, *args, **kwargs)
    monkeypatch.setattr(Path, "open", once_locked)
    verified = store_runtime.verify_inputs(dict(run="fixture"), [row])
    assert verified.digests[str(path)] == bound["sha256"] and len(calls) == 2
    changed = store_runtime.ResolvedInput(dict(reference, sha256="0" * 64), path, ("input",))
    with pytest.raises(store_runtime.InputHashMismatch):
        store_runtime.verify_inputs(dict(run="fixture"), [changed])
    assert len(calls) == 3


def test_verified_metadata_waits_for_a_transient_file_handle(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import store_runtime
    path = tmp_path / "input.json"
    path.write_bytes(b'{"source":"raw"}')
    bound = n1_pipeline.asset(path)
    row = store_runtime.ResolvedInput(dict(sha256=bound["sha256"], bytes=bound["size_bytes"]), path, ("input",))
    verified = store_runtime.verify_inputs(dict(run="fixture"), [row])
    original = Path.open
    calls = []
    def once_locked(source, *args, **kwargs):
        if source == path:
            calls.append(source)
            if len(calls) == 1:
                raise sharing_error()
        return original(source, *args, **kwargs)
    monkeypatch.setattr(Path, "open", once_locked)
    assert verified.metadata("input") == dict(source="raw") and len(calls) == 2
