"""Configuration is explicit, content-bound and has no method defaults."""

import hashlib
import json

import pytest

from darsia.fluidflower.native.config import (
    configure_parameters, load_parameters, require_configuration, require_parameters, reset_parameters,
)
from darsia.fluidflower.native.io import read_bound_json, resolve_asset_path, sha256_file


def test_bound_reader_preserves_exact_numeric_types(tmp_path):
    path = tmp_path / "parameters.json"
    raw = b'\xef\xbb\xbf{"sensor": {"width": 3, "gain": 1.25}}'
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert sha256_file(path) == digest
    document = load_parameters(path, digest)
    assert type(document["sensor"]["width"]) is int
    assert type(document["sensor"]["gain"]) is float
    assert read_bound_json(path, digest) == document
    path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="Source artifact changed"):
        load_parameters(path, digest)


def test_bound_reader_rejects_nonobject(tmp_path):
    path = tmp_path / "parameters.json"
    raw = b"[]"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="must contain an object"):
        read_bound_json(path, hashlib.sha256(raw).hexdigest())


def test_explicit_sections_and_nested_context_restore(monkeypatch):
    monkeypatch.delenv("DARSIA_NATIVE_CONFIGURATION", raising=False)
    monkeypatch.delenv("DARSIA_NATIVE_CONFIGURATION_SHA256", raising=False)
    with pytest.raises(ValueError, match="explicit native"):
        require_parameters("camera")
    first = configure_parameters({"camera": {"width": 3}})
    try:
        assert require_parameters("camera")["width"] == 3
        assert require_configuration()["camera"]["width"] == 3
        second = configure_parameters({"camera": {"width": 7}})
        try:
            assert require_parameters("camera")["width"] == 7
            assert require_parameters("camera", configuration={"camera": {"width": 5}})["width"] == 5
            with pytest.raises(ValueError, match="section is absent"):
                require_parameters("missing")
        finally:
            reset_parameters(second)
        assert require_parameters("camera")["width"] == 3
    finally:
        reset_parameters(first)


def test_legacy_environment_configuration_is_bound(tmp_path, monkeypatch):
    path = tmp_path / "parameters.json"
    raw = json.dumps({"camera": {"width": 9}}).encode()
    path.write_bytes(raw)
    monkeypatch.setenv("DARSIA_NATIVE_CONFIGURATION", str(path))
    monkeypatch.setenv("DARSIA_NATIVE_CONFIGURATION_SHA256", hashlib.sha256(raw).hexdigest())
    assert require_parameters("camera")["width"] == 9
    monkeypatch.setenv("DARSIA_NATIVE_CONFIGURATION_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="Source artifact changed"):
        require_parameters("camera")


def test_asset_relocation_requires_an_explicit_entry(tmp_path):
    original = tmp_path / "original.json"
    relocated = tmp_path / "copied.json"
    original.write_bytes(b"{}")
    relocated.write_bytes(b"{}")
    assert resolve_asset_path(original, {}) == original
    assert resolve_asset_path(original, {str(original): str(relocated)}) == relocated
    assert sha256_file(resolve_asset_path(original, {str(original): str(relocated)})) == sha256_file(original)
