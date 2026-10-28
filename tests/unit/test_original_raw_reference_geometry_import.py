"""Compile-only test text: exact original raw frame/metric protocol metadata.

No Store, image decoder, corrected raster, Build, real series or NAS is used.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import toml

from darsia.fluidflower import geometry_import as subject
from darsia.fluidflower.series_geometry import (
    create_series_geometry_project,
    native_series_link,
    save_rig_frame,
    SeriesRunInputs,
    RigFrame,
)


def descriptor(path):
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw), "mtime_ns": path.stat().st_mtime_ns}


@pytest.fixture
def raw_metadata(tmp_path):
    root = tmp_path / "series"
    (root / "config/runs").mkdir(parents=True)
    raw = root / "raw"
    raw.mkdir()
    photo = raw / "original.JPG"
    photo.write_bytes(b"explicit raw photograph identity; no image decode")
    protocol = root / "injection.csv"
    protocol.write_bytes(b"location_x,location_y\n0.21,0.04\n0.21,0.04\n0.72,0.03\n")
    manifest = root / "series.toml"
    manifest.write_text(toml.dumps({"schema_version": 1, "series_id": "synthetic",
        "runs": {"ids": ["reference"], "config_files": {"reference": "config/runs/reference.toml"}}}))
    common = root / "config/common.toml"
    common.write_text(toml.dumps({"rig": {"width": .9, "height": .49}}))
    config = {"data": {"folders": [str(raw)], "baseline": photo.name,
                       "results": str(root / "results/reference")},
              "protocols": {"injection": str(protocol)},
              "corrections": {"curvature": {"crop": {
                  "top_left": [1, 2], "bottom_left": [29, 3],
                  "bottom_right": [28, 52], "top_right": [1, 51],
                  "width": .9, "height": .49, "in_meters": True}}}}
    partial = root / "config/runs/reference.toml"
    partial.write_text(toml.dumps(config))
    reference = SeriesRunInputs("reference", partial, baseline_image=photo,
                               injection_protocol=protocol)
    project = create_series_geometry_project(series_id="synthetic", series_name="synthetic",
        series_root=root, rig_type="custom", reference=reference, created_by="synthetic")
    bindings = {"manifest": descriptor(manifest), "common_configuration": descriptor(common),
                "run_configuration": descriptor(partial), "raw_baseline": descriptor(photo),
                "injection_protocol": descriptor(protocol)}
    return dict(root=root, project=project, config=config, partial=partial,
                protocol=protocol, photo=photo, bindings=bindings)


def invoke(data):
    return subject.save_original_raw_reference_geometry(data["project"],
        series_root=data["root"], run_id="reference", source_bindings=data["bindings"],
        actor="explicit synthetic metadata import")


def replace_config(data):
    data["partial"].write_text(toml.dumps(data["config"]))
    data["bindings"]["run_configuration"] = descriptor(data["partial"])


def test_real_original_APIs_save_exact_raw_corners_and_parser_metric_points(raw_metadata):
    data = raw_metadata
    before = {key: Path(row["path"]).read_bytes() for key, row in data["bindings"].items()}
    result = invoke(data)
    assert result["raw_metadata_complete"] and result["raw_frame_saved"] and result["protocol_points_saved"]
    assert result["frame_scope"] == "raw_photograph" and not result["corrected_canvas_frame_changed"]
    assert not result["revision_or_current_written"] and not result["raster_transformed"]
    link = native_series_link(data["project"])
    frame = link["rig_frames"]["reference"]
    assert frame["corners"] == {key: [float(value[0]), float(value[1])]
                               for key, value in data["config"]["corrections"]["curvature"]["crop"].items()
                               if key in ("top_left", "bottom_left", "bottom_right", "top_right")}
    assert frame["image"] == str(data["photo"]) and frame["source"] == "series configuration"
    assert frame["width_m"] == .9 and frame["height_m"] == .49
    points = link["injection_points"]
    assert [(p["location_id"], p["x_m"], p["y_m"], p["pixel"])
            for p in points] == [("ip1", .21, .04, None), ("ip2", .72, .03, None)]
    assert result["protocol_points"] == points and not result["missing"]
    assert set(result["sources"]) == set(data["bindings"])
    assert not (data["project"].root / "overrides/geometry_facies/current.json").exists()
    assert all(Path(row["path"]).read_bytes() == before[key] for key, row in data["bindings"].items())
    # An exact second explicit import reuses, rather than overwrites, metadata.
    saved = data["project"].manifest_path.read_bytes()
    again = invoke(data)
    assert again["existing_metadata_reused"] and not again["raw_frame_saved"] and not again["protocol_points_saved"]
    assert data["project"].manifest_path.read_bytes() == saved


@pytest.mark.parametrize("problem", ["missing_corner", "invalid_coordinate", "nonfinite_coordinate", "no_dimensions"])
def test_missing_original_coordinates_are_reported_without_any_save(raw_metadata, problem):
    data = raw_metadata
    crop = data["config"]["corrections"]["curvature"]["crop"]
    if problem == "missing_corner":
        del crop["top_left"]
    elif problem == "invalid_coordinate":
        crop["top_left"] = ["not a coordinate", 2]
    elif problem == "nonfinite_coordinate":
        crop["top_left"] = [float("nan"), 2]
    else:
        crop["in_meters"] = False
    replace_config(data)
    saved = data["project"].manifest_path.read_bytes()
    result = invoke(data)
    assert result["missing"] and not result["raw_metadata_complete"]
    assert not result["raw_frame_saved"] and not result["protocol_points_saved"]
    assert data["project"].manifest_path.read_bytes() == saved


@pytest.mark.parametrize("text", [b"location_x,location_y\n", b"location_x,location_y\nmissing,nan\n"])
def test_no_genuine_protocol_points_never_creates_points_or_Done(raw_metadata, text):
    data = raw_metadata
    data["protocol"].write_bytes(text)
    data["bindings"]["injection_protocol"] = descriptor(data["protocol"])
    saved = data["project"].manifest_path.read_bytes()
    result = invoke(data)
    assert result["missing"] and "no finite recorded" in result["missing"][0]
    assert not result["raw_metadata_complete"] and not result["raw_frame_saved"]
    assert not result["protocol_points_saved"] and "done" not in result and "state" not in result
    assert data["project"].manifest_path.read_bytes() == saved


def test_changed_source_SHA_is_a_hard_error_with_no_save(raw_metadata):
    data = raw_metadata
    data["protocol"].write_bytes(b"external change")
    saved = data["project"].manifest_path.read_bytes()
    with pytest.raises(ValueError, match="source changed"):
        invoke(data)
    assert data["project"].manifest_path.read_bytes() == saved


def test_corrected_NPZ_is_not_accepted_as_the_configured_raw_photo(raw_metadata):
    data = raw_metadata
    corrected = data["root"] / "shape_corrected_baseline.npz"
    corrected.write_bytes(b"saved corrected archive identity, not a raw photograph")
    data["bindings"]["raw_baseline"] = descriptor(corrected)
    with pytest.raises(ValueError, match="original reader's raw_baseline"):
        invoke(data)
    assert not native_series_link(data["project"]).get("rig_frames")


def test_raw_metadata_import_does_not_read_or_rewrite_corrected_revision(raw_metadata, monkeypatch):
    data = raw_metadata
    current = data["project"].root / "overrides/geometry_facies/current.json"
    current.parent.mkdir(parents=True)
    original = b'{"fixture":"existing corrected revision must remain byte-identical"}\n'
    current.write_bytes(original)
    original_open = Path.open

    def forbid_current(path, *args, **kwargs):
        if path.resolve() == current.resolve():
            pytest.fail("Raw frame import must not access the corrected Store revision")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as local:
        local.setattr(Path, "open", forbid_current)
        result = invoke(data)
    assert result["raw_metadata_complete"] and current.read_bytes() == original


def test_existing_different_human_frame_is_preserved(raw_metadata):
    data = raw_metadata
    frame = RigFrame(corners={name: (5., 6.) for name in
        ("top_left", "bottom_left", "bottom_right", "top_right")},
        width_m=.9, height_m=.49, run_id="reference", source="marked in geometry workspace",
        image=str(data["photo"]))
    save_rig_frame(data["project"], frame, actor="human fixture")
    saved = data["project"].manifest_path.read_bytes()
    result = invoke(data)
    assert result["missing"] and "existing raw rig frame differs" in result["missing"][0]
    assert not result["raw_metadata_complete"] and data["project"].manifest_path.read_bytes() == saved


def test_point_save_error_preserves_actual_frame_and_original_error(raw_metadata, monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("exact synthetic protocol save failure")
    monkeypatch.setattr(subject, "save_injection_points", fail)
    data = raw_metadata
    with pytest.raises(RuntimeError, match="^exact synthetic protocol save failure$"):
        invoke(data)
    link = native_series_link(data["project"])
    assert link["rig_frames"]["reference"]["image"] == str(data["photo"])
    assert not link.get("injection_points")
