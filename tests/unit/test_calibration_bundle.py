import json
import zipfile
from pathlib import Path

import pytest

from darsia.presets.workflows.utils.calibration_bundle import (
    export_calibration_bundle,
    import_calibration_bundle,
)


def _write_file(path: Path, content: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _write_minimal_config(tmp_path: Path) -> Path:
    data_folder = tmp_path / "data"
    data_folder.mkdir(parents=True, exist_ok=True)
    (data_folder / "baseline.jpg").touch()
    (data_folder / "dummy.jpg").touch()

    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[data]
folder = "{data_folder}"
baseline = "baseline.jpg"
results = "{tmp_path / "results"}"

[data.path.baseline_imgs]
paths = ["dummy.jpg"]

[data.path.cal_imgs]
paths = ["dummy.jpg"]

[color_paths]
basis = "labels"
baseline = "baseline_imgs"
data = "cal_imgs"

[color_to_mass]
basis = "labels"
data = "cal_imgs"
"""
    )
    return config


def _create_calibration_artifacts(tmp_path: Path) -> Path:
    config_path = _write_minimal_config(tmp_path)
    results = tmp_path / "results"
    _write_file(
        results / "calibration" / "color_paths" / "from_labels" / "color_path_0.json"
    )
    _write_file(results / "calibration" / "color_paths" / "from_labels" / "metadata.json")
    _write_file(
        results
        / "calibration"
        / "color_to_mass"
        / "from_labels"
        / "metadata.json",
        '{"basis":"labels","label_ids":[0]}',
    )
    _write_file(
        results
        / "calibration"
        / "baseline_color_spectrum"
        / "color_spectrum_0.json"
    )
    _write_file(
        results / "calibration" / "color_range.json",
        '{"min_color":[0,0,0],"max_color":[1,1,1],"color_mode":"RELATIVE"}',
    )
    return config_path


def test_export_calibration_bundle(tmp_path: Path):
    config_path = _create_calibration_artifacts(tmp_path)
    bundle = tmp_path / "bundle.zip"

    exported = export_calibration_bundle(config_path, bundle=bundle)

    assert exported == bundle
    assert bundle.exists()
    with zipfile.ZipFile(bundle, "r") as zf:
        names = set(zf.namelist())
        assert "calibration_bundle/color_paths/color_path_0.json" in names
        assert "calibration_bundle/color_to_mass/metadata.json" in names
        assert "calibration_bundle/baseline_color_spectrum/color_spectrum_0.json" in names
        assert "calibration_bundle/color_range/color_range.json" in names
        manifest = json.loads(zf.read("calibration_bundle/manifest.json"))
        assert "artifacts" in manifest


def test_import_calibration_bundle(tmp_path: Path):
    config_path = _create_calibration_artifacts(tmp_path)
    bundle = export_calibration_bundle(config_path, bundle=tmp_path / "bundle.zip")

    imported = import_calibration_bundle(
        config_path,
        bundle=bundle,
        target_folder=tmp_path / "imported",
    )

    assert (tmp_path / "imported" / "CONFIG_SNIPPET.toml").exists()
    assert imported["color_paths"].exists()
    assert imported["color_to_mass"].exists()
    assert imported["color_range"].exists()


def test_import_calibration_bundle_overwrite_guard(tmp_path: Path):
    config_path = _create_calibration_artifacts(tmp_path)
    bundle = export_calibration_bundle(config_path, bundle=tmp_path / "bundle.zip")
    target = tmp_path / "imported"
    target.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileExistsError):
        import_calibration_bundle(config_path, bundle=bundle, target_folder=target)
