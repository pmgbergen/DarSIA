"""Behavioral tests for five shared SDK helpers; no NAS or campaign inputs."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest


def test_image_rgb_conversion_preserves_dtype_channel_and_scale():
    from darsia.fluidflower.phase_training.model import _as_rgb_float

    pixels = np.array([[[255, 128, 0, 19]]], dtype=np.uint8)
    result = _as_rgb_float(type("Image", (), {"img": pixels})())
    expected = np.array([[[1, np.float32(128) / np.float32(255), 0]]], dtype=np.float32)
    assert result.dtype == np.float32
    assert np.array_equal(result, expected)
    assert np.array_equal(pixels, np.array([[[255, 128, 0, 19]]], dtype=np.uint8))


def test_image_rgb_conversion_threshold_empty_nan_and_shape():
    from darsia.fluidflower.phase_training.model import _as_rgb_float

    result = _as_rgb_float(np.array([[[1.5, 0.5, -2]]], dtype=np.float64))
    assert np.array_equal(result, np.array([[[1, 0.5, 0]]], dtype=np.float32))
    empty = _as_rgb_float(np.empty((0, 2, 3), dtype=np.float64))
    assert empty.shape == (0, 2, 3) and empty.dtype == np.float32
    result = _as_rgb_float(np.array([[[np.nan, 255, -10]]], dtype=np.float64))
    assert np.isnan(result[0, 0, 0])
    assert np.array_equal(result[0, 0, 1:], np.array([1, 0], dtype=np.float32))
    with pytest.raises(ValueError, match=r"Expected an RGB image, got \(2, 3\)\."):
        _as_rgb_float(np.zeros((2, 3)))


@pytest.mark.parametrize("module_name", ["darsia.fluidflower.phase_learning", "darsia.fluidflower.facies_learning"])
def test_file_reference_records_full_file_and_modified_ns(tmp_path, module_name):
    _file_reference = importlib.import_module(module_name)._file_reference

    path = tmp_path / "value.bin"
    path.write_bytes(b"first\x00second")
    reference = _file_reference(path)
    assert list(reference) == ["path", "size_bytes", "modified_ns", "sha256"]
    assert reference == {
        "path": str(path.resolve()),
        "size_bytes": 12,
        "modified_ns": path.stat().st_mtime_ns,
        "sha256": hashlib.sha256(b"first\x00second").hexdigest(),
    }
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError) as caught:
        _file_reference(missing)
    assert caught.value.args == (missing.resolve(),)


def test_rig_class_asset_preserves_relative_path_and_digest_guards(tmp_path):
    from darsia.fluidflower.phase_pixel_area import _asset

    path = tmp_path / "asset.bin"
    data = b"labels-only"
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    selected, actual, signature = _asset({"path": path.name, "sha256": digest}, tmp_path)
    assert selected == path and actual == data and signature == digest
    assert _asset({"path": str(path), "sha256": digest}, tmp_path / "unused")[0] == path
    with pytest.raises(ValueError, match="Pixel-area asset requires an exact path and SHA256"):
        _asset({"path": path.name}, tmp_path)
    with pytest.raises(ValueError, match="Pixel-area asset SHA256 mismatch"):
        _asset({"path": path.name, "sha256": "0" * 64}, tmp_path)


def test_saved_json_write_preserves_default_ascii_nan_newline_and_replace(tmp_path):
    from darsia.fluidflower.tools.phase_reference_maps import _write

    target = tmp_path / "sub" / "saved.json"
    payload = {"label": "ø", "not_finite": float("nan"), "order": 1}
    _write(target, payload)
    # The preserved writer uses Path.write_text with native newline translation.
    expected = (json.dumps(payload, indent=2) + "\n").replace("\n", os.linesep)
    assert target.read_bytes() == expected.encode("utf-8")
    _write(target, {"second": 2})
    assert target.read_bytes() == ('{\n  "second": 2\n}\n'.replace("\n", os.linesep)).encode("utf-8")
    assert list(target.parent.iterdir()) == [target]


def test_workflow_color_integer_conversion_and_exact_rejection():
    from darsia.presets.workflows.config.video import _to_rgb

    assert _to_rgb([0.9, "12", 255.9], "contour") == (0, 12, 255)
    with pytest.raises(ValueError, match=r"contour must have exactly 3 entries \[R, G, B\]\."):
        _to_rgb([1, 2], "contour")
    with pytest.raises(ValueError, match=r"contour entries must be in \[0, 255\]\."):
        _to_rgb([-1, 2, 3], "contour")
    with pytest.raises(ValueError):
        _to_rgb(["invalid", 2, 3], "contour")


@pytest.mark.parametrize(
    "client_name, client_function, owner_name, owner_function, arguments",
    [
        ("darsia.fluidflower.phase_training.model", "_as_rgb_float",
         "darsia.fluidflower.phase_features", "_as_rgb_float", (object(),)),
        ("darsia.fluidflower.phase_learning", "_file_reference",
         "darsia.fluidflower.phase_interpretation", "_file_reference", (Path("value"),)),
        ("darsia.fluidflower.facies_learning", "_file_reference",
         "darsia.fluidflower.phase_interpretation", "_file_reference", (Path("value"),)),
        ("darsia.fluidflower.phase_pixel_area", "_asset",
         "darsia.fluidflower.phase_saved_review_metadata", "_rig_class_asset", ({}, Path("."))),
        ("darsia.fluidflower.tools.phase_reference_maps", "_write",
         "darsia.fluidflower.phase_saved_review_metadata", "_write", (Path("value"), {})),
        ("darsia.presets.workflows.config.video", "_to_rgb",
         "darsia.presets.workflows.config.analysis", "_to_rgb", ([1, 2, 3], "color")),
    ],
)
def test_legacy_helpers_delegate_to_single_owner(
    monkeypatch, client_name, client_function, owner_name, owner_function, arguments
):
    client = importlib.import_module(client_name)
    owner = importlib.import_module(owner_name)
    seen = []
    marker = object()

    def selected(*values):
        seen.append(values)
        return marker

    monkeypatch.setattr(owner, owner_function, selected)
    assert getattr(client, client_function)(*arguments) is marker
    assert seen == [arguments]
