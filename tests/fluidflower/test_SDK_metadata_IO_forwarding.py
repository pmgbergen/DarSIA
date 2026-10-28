"""Shared SDK metadata I/O preserves exact bytes and failure side effects."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

import pytest

from darsia.fluidflower import project, mass_artifacts
from darsia.fluidflower.tools.aqueous_noise_gate import file_sha256


ORIGINAL_COMMIT = "65c0e85015700f764b2c4f5fed6bb558490ef7f0"


def _original(relative_path, name):
    repo = Path(__file__).resolve().parents[2]
    raw = subprocess.check_output([
        "git", "show", ORIGINAL_COMMIT + ":src/darsia/fluidflower/" + relative_path,
    ], cwd=repo).decode("utf-8")
    function = copy.deepcopy(next(node for node in ast.parse(raw).body
                                  if isinstance(node, ast.FunctionDef) and node.name == name))
    function.returns = None
    for argument in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs):
        argument.annotation = None
    namespace = {"json": json, "os": os, "uuid": uuid, "hashlib": hashlib, "Path": Path}
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, relative_path + ":" + name, "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("copy_payload", [False, True])
def test_atomic_json_keeps_original_utf8_indentation_nan_and_newline(tmp_path, copy_payload):
    relative = "analysis_domains.py" if copy_payload else "project.py"
    original = _original(relative, "_atomic_json")
    expected, actual = tmp_path / "old" / "metadata.json", tmp_path / "new" / "metadata.json"
    payload = {"unicode": "æλ", "nan": float("nan"), "rows": [1, True, None]}
    original(expected, payload)
    project._atomic_json(actual, payload, copy_payload=copy_payload)
    assert actual.read_bytes() == expected.read_bytes()
    assert b"NaN" in actual.read_bytes() and "æλ" in actual.read_text(encoding="utf-8")
    assert actual.read_bytes().endswith(os.linesep.encode())
    assert list(actual.parent.glob(".*.tmp")) == []


def test_dict_copy_failure_occurs_after_original_directory_and_append_open(tmp_path):
    atomic_original = _original("analysis_domains.py", "_atomic_json")
    append_original = _original("analysis_domains.py", "_append_jsonl")
    for name, old, new in (
        ("atomic", atomic_original, project._atomic_json),
        ("append", append_original, project._append_jsonl),
    ):
        old_path, new_path = tmp_path / (name + "old") / "bad.json", tmp_path / (name + "new") / "bad.json"
        with pytest.raises(TypeError) as original_error:
            old(old_path, object())
        with pytest.raises(TypeError) as current_error:
            new(new_path, object(), copy_payload=True)
        assert str(current_error.value) == str(original_error.value)
        assert old_path.parent.exists() and new_path.parent.exists()
        assert old_path.exists() == new_path.exists() == (name == "append")
        if name == "append":
            assert old_path.read_bytes() == new_path.read_bytes() == b""
        assert list(new_path.parent.glob(".*.tmp")) == []


@pytest.mark.parametrize("copy_payload", [False, True])
def test_jsonl_keeps_original_utf8_lf_append_and_fsync(tmp_path, monkeypatch, copy_payload):
    original = _original("analysis_domains.py" if copy_payload else "project.py", "_append_jsonl")
    expected, actual = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
    real_fsync, calls = os.fsync, []
    def observed_fsync(fd):
        calls.append(fd)
        return real_fsync(fd)
    monkeypatch.setattr(os, "fsync", observed_fsync)
    for payload in ({"unicode": "æλ", "nan": float("nan")}, {"next": True}):
        before = len(calls)
        original(expected, payload)
        project._append_jsonl(actual, payload, copy_payload=copy_payload)
        assert len(calls) == before + 2
    assert actual.read_bytes() == expected.read_bytes()
    assert b"\r\n" not in actual.read_bytes()
    assert actual.read_bytes().count(b"\n") == 2


def test_pid_atomic_json_preserves_cleanup_on_replace_failure(tmp_path, monkeypatch):
    original = _original("mass_artifacts.py", "_atomic_json")
    payload = {"unicode": "æλ", "nan": float("nan")}
    old_path, new_path = tmp_path / "old" / "metadata.json", tmp_path / "new" / "metadata.json"
    original(old_path, payload)
    mass_artifacts._atomic_json(new_path, payload)
    assert old_path.read_bytes() == new_path.read_bytes()
    def fail_replace(source, target):
        raise OSError("fixture replacement failure")
    monkeypatch.setattr(os, "replace", fail_replace)
    for function, path in ((original, tmp_path / "failold" / "metadata.json"),
                           (mass_artifacts._atomic_json, tmp_path / "failnew" / "metadata.json")):
        with pytest.raises(OSError, match="fixture replacement failure"):
            function(path, payload)
        assert path.parent.exists() and not path.exists()
        assert list(path.parent.glob(".*.tmp")) == []


def test_uncached_sha_preserves_path_conversion_and_missing_file_errors(tmp_path):
    data = b"local metadata fixture\x00\xff" * 50000
    path = tmp_path / "fixture.bin"
    path.write_bytes(data)
    original = _original("project.py", "_sha256_file_uncached")
    assert project._sha256_file_uncached(path) == original(path) == hashlib.sha256(data).hexdigest()
    assert file_sha256(str(path)) == project._sha256_file_uncached(path)
    for function in (project._sha256_file_uncached, original):
        with pytest.raises(FileNotFoundError):
            function(tmp_path / "absent.bin")
        with pytest.raises(AttributeError):
            function(str(path))
