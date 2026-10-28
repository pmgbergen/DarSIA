"""Local-fixture checks for the explicit UI crop adapter."""

from copy import deepcopy
import json
from pathlib import Path
import tomllib

import numpy as np
from PIL import Image
import pytest
import toml

from darsia.fluidflower import setup_geometry_apply as apply
from darsia.fluidflower.camera_registration import CameraRegistration
from darsia.fluidflower.series_geometry import CROP_CORNERS, NATIVE_SETUP_EXPORT_FORMAT


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(toml.dumps(data), encoding="utf-8")


def _series(tmp_path):
    root = tmp_path / "series"
    images = tmp_path / "photographs"
    images.mkdir()
    for name in ("ac1", "ac2", "other"):
        Image.new("RGB", (80, 50), (50, 60, 70)).save(images / f"{name}.png")
    manifest = {
        "schema_version": 1,
        "series_id": "local",
        "paths": {
            "common_config": "config/common.toml",
            "run_configs": "config/runs",
            "merged_configs": "config/merged",
        },
        "runs": {
            "ids": ["ac1", "ac2"],
            "config_files": {
                "ac1": "config/runs/ac1.toml",
                "ac2": "config/runs/ac2.toml",
            },
        },
    }
    _write(root / "series.toml", manifest)
    common = {
        "rig": {"dim": 2, "width": 0.9, "height": 0.49},
        "keep": {"names": ["Albus", "Blå"], "number": 7},
        "corrections": {
            "active": ["type", "resize"],
            "curvature": {
                "active": ["init", "stretch"],
                "init": {"horizontal_bulge": 0.2},
                "stretch": {"vertical_stretch": 0.1},
            },
        },
    }
    _write(root / "config/common.toml", common)
    for run in ("ac1", "ac2"):
        run_data = {
            "data": {
                "folders": [str(images)],
                "baseline": f"{run}.png",
                "results": str(root / "results" / run),
            }
        }
        _write(root / f"config/runs/{run}.toml", run_data)
        merged = deepcopy(common)
        merged.update(run_data)
        merged["only_in_merged"] = {"custom": run}
        _write(root / f"config/merged/{run}.toml", merged)
    export = root / "geometry/export.json"
    export.parent.mkdir()
    corners = {
        "top_left": [3.25, 4.25],
        "bottom_left": [44.25, 6.25],
        "bottom_right": [43.25, 73.25],
        "top_right": [4.25, 72.25],
    }
    export.write_text(
        json.dumps(
            {
                "format": NATIVE_SETUP_EXPORT_FORMAT,
                "revision_id": "revision-1",
                "reference_run": "ac1",
                "reference_baseline": str(images / "ac1.png"),
                "rig_frame": {
                    "corners": corners,
                    "run_id": "ac1",
                    "image": str(images / "ac1.png"),
                    "image_size": [80, 50],
                    "width_m": 0.9,
                    "height_m": 0.49,
                },
            }
        ),
        encoding="utf-8",
    )
    return root, export, images, corners


def _registration(source, target, corners):
    return CameraRegistration(
        np.array([[1.0, 0.0, -3.5], [0.0, 1.0, 2.5], [0.0, 0.0, 1.0]]),
        "affine",
        30,
        30,
        0.3,
        0.5,
        (80, 50),
        (80, 50),
        {"top": 10, "left": 10, "right": 10},
    )


def test_native_corner_order_and_final_rounding_no_writes(tmp_path):
    root, export, images, corners = _series(tmp_path)
    before = {path: path.read_bytes() for path in root.rglob("*.toml")}
    calls = []

    def register(*args):
        calls.append(args)
        return _registration(*args)

    plan = apply.prepare_crop_plan(
        root, export, expected_revision_id="revision-1", register=register
    )
    assert len(calls) == 1
    assert calls[0][0] == images / "ac1.png"
    assert calls[0][1] == images / "ac2.png"
    expected = {
        name: [round(corners[name][0] + 2.5), round(corners[name][1] - 3.5)]
        for name in CROP_CORNERS
    }
    actual = plan.runs[1]["crop"]
    assert {name: actual[name] for name in CROP_CORNERS} == expected
    assert (
        actual["width"] == 0.9
        and actual["height"] == 0.49
        and actual["in_meters"] is True
    )
    assert plan.runs[0]["registration"]["kind"] == "same_raw_photograph"
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_source_photo_is_export_frame_image_even_for_reference_run(tmp_path):
    root, export, images, _ = _series(tmp_path)
    value = json.loads(export.read_bytes())
    value["rig_frame"]["image"] = str(images / "other.png")
    export.write_text(json.dumps(value), encoding="utf-8")
    calls = []

    def register(*args):
        calls.append(args)
        return _registration(*args)

    apply.prepare_crop_plan(
        root, export, expected_revision_id="revision-1", register=register
    )
    assert [call[1].name for call in calls] == ["ac1.png", "ac2.png"]
    assert all(call[0] == images / "other.png" for call in calls)


def test_preserves_absent_active_lists_and_existing_values():
    original = b'\xef\xbb\xbf[corrections.curvature.init]\r\nhorizontal_bulge = 0.4\r\n[unknown]\r\ntext = "bl\\u00e5"\r\nitems = [1, 3, 7]\r\n'
    crop = {
        "top_left": [3, 4],
        "bottom_left": [10, 4],
        "bottom_right": [10, 20],
        "top_right": [3, 20],
        "width": 0.9,
        "height": 0.49,
        "in_meters": True,
    }
    after = apply.crop_config_bytes(original, crop)
    data = tomllib.loads(after.decode("utf-8-sig"))
    assert "active" not in data["corrections"]
    assert "active" not in data["corrections"]["curvature"]
    assert data["corrections"]["curvature"]["init"]["horizontal_bulge"] == 0.4
    assert data["unknown"] == tomllib.loads(original.decode("utf-8-sig"))["unknown"]
    assert after.startswith(b"\xef\xbb\xbf") and b"\r\n" in after


def test_inherited_active_lists_and_merged_custom_values_preserved(tmp_path):
    root, export, _, _ = _series(tmp_path)
    common_before = (root / "config/common.toml").read_bytes()
    plan = apply.prepare_crop_plan(
        root, export, expected_revision_id="revision-1", register=_registration
    )
    for change in plan.changes:
        data = tomllib.loads(change.after.decode("utf-8"))
        assert data["corrections"]["active"] == ["type", "resize", "curvature"]
        assert data["corrections"]["curvature"]["active"] == ["init", "stretch", "crop"]
        if "merged" in str(change.path):
            assert data["only_in_merged"]["custom"] == change.run_id
    receipt = apply.apply_crop_plan(plan, receipt_directory=tmp_path / "receipts")
    assert json.loads(receipt.read_bytes())["state"] == "complete"
    assert (root / "config/common.toml").read_bytes() == common_before
    for record in json.loads(receipt.read_bytes())["files"]:
        backup = Path(record["backup"])
        assert apply._sha(backup.read_bytes()) == record["before_sha256"]


def test_registration_failure_preserves_all_configs(tmp_path):
    root, export, _, _ = _series(tmp_path)
    before = {path: path.read_bytes() for path in root.rglob("*.toml")}

    def broken(*args):
        raise ValueError("The rig frame gives no consistent camera move.")

    with pytest.raises(ValueError, match="no consistent camera move"):
        apply.prepare_crop_plan(
            root, export, expected_revision_id="revision-1", register=broken
        )
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_mismatched_merged_baseline_is_not_silently_relabelled(tmp_path):
    root, export, images, _ = _series(tmp_path)
    merged = root / "config/merged/ac2.toml"
    data = tomllib.loads(merged.read_text(encoding="utf-8"))
    data["data"]["baseline"] = "other.png"
    _write(merged, data)
    before = merged.read_bytes()
    with pytest.raises(ValueError, match="different raw baselines"):
        apply.prepare_crop_plan(
            root, export, expected_revision_id="revision-1", register=_registration
        )
    assert merged.read_bytes() == before


def test_config_race_does_not_overwrite_new_user_content(tmp_path):
    root, export, _, _ = _series(tmp_path)
    plan = apply.prepare_crop_plan(
        root, export, expected_revision_id="revision-1", register=_registration
    )
    changed = plan.changes[0].path
    changed.write_bytes(b"# user edit\n")
    with pytest.raises(RuntimeError, match="File changed"):
        apply.apply_crop_plan(plan, receipt_directory=tmp_path / "receipts")
    assert changed.read_bytes() == b"# user edit\n"
    assert not (tmp_path / "receipts").exists()


def test_second_replace_failure_records_partial_truth_and_backups(
    monkeypatch, tmp_path
):
    root, export, _, _ = _series(tmp_path)
    plan = apply.prepare_crop_plan(
        root, export, expected_revision_id="revision-1", register=_registration
    )
    original_replace = apply.os.replace
    count = 0

    def fail_second(source, target):
        nonlocal count
        count += 1
        if count == 2:
            raise PermissionError("simulated second replace failure")
        original_replace(source, target)

    monkeypatch.setattr(apply.os, "replace", fail_second)
    with pytest.raises(apply.CropApplyError) as caught:
        apply.apply_crop_plan(plan, receipt_directory=tmp_path / "receipts")
    error = caught.value
    assert error.applied == (plan.changes[0].path,)
    assert plan.changes[0].path.read_bytes() == plan.changes[0].after
    assert plan.changes[1].path.read_bytes() == plan.changes[1].before
    receipt = json.loads(error.receipt.read_bytes())
    assert receipt["state"] == "failed"
    assert receipt["files"][0]["state"] == "applied"
    assert all(Path(item["backup"]).read_bytes() for item in receipt["files"])
