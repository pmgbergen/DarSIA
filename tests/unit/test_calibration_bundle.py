import json
import zipfile
from pathlib import Path

import pytest

from darsia.presets.workflows.utils.calibration_bundle import (
    export_calibration_bundle,
    import_calibration_bundle,
    preview_calibration_bundle_import_conflicts,
)


def _write_file(path: Path, content: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _write_minimal_config(tmp_path: Path) -> Path:
    data_folder = tmp_path / "data"
    results_folder = tmp_path / "results"
    data_folder.mkdir(parents=True, exist_ok=True)
    (data_folder / "baseline.jpg").touch()
    (data_folder / "dummy.jpg").touch()

    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[data]
baseline = "baseline.jpg"
folder = "{data_folder.as_posix()}"
results = "{results_folder.as_posix()}"
"""
    )
    return config


def _create_calibration_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    config_path = _write_minimal_config(tmp_path)
    root = tmp_path / "results" / "calibration" / "color" / "my_colorpath"
    _write_file(root / "color_paths" / "from_labels" / "color_path_0.json")
    _write_file(root / "color_paths" / "from_labels" / "metadata.json")
    _write_file(root / "color_to_mass" / "from_labels" / "metadata.json")
    _write_file(root / "baseline_color_spectrum" / "color_spectrum_0.json")
    _write_file(root / "color_range.json")
    return config_path, root


def test_export_calibration_bundle(tmp_path: Path):
    config_path, _ = _create_calibration_artifacts(tmp_path)
    bundle = tmp_path / "bundle.zip"

    exported = export_calibration_bundle(config_path, bundle=bundle)

    assert exported == bundle
    assert bundle.exists()
    with zipfile.ZipFile(bundle, "r") as zf:
        names = set(zf.namelist())
        assert (
            "calibration/my_colorpath/color_paths/from_labels/color_path_0.json"
            in names
        )
        manifest = json.loads(zf.read("calibration/manifest.json"))
        assert manifest["format_version"] == 2
        assert manifest["embeddings"] == ["my_colorpath"]


def test_import_calibration_bundle(tmp_path: Path):
    config_path, _ = _create_calibration_artifacts(tmp_path)
    bundle = export_calibration_bundle(config_path, bundle=tmp_path / "bundle.zip")

    imported = import_calibration_bundle(
        config_path,
        bundle=bundle,
        target_folder=tmp_path / "imported",
    )

    assert (tmp_path / "imported" / "CONFIG_SNIPPET.toml").exists()
    assert imported["my_colorpath"].exists()
    assert (
        tmp_path
        / "imported"
        / "my_colorpath"
        / "color_paths"
        / "from_labels"
        / "metadata.json"
    ).exists()


def test_import_calibration_bundle_overwrite_guard(tmp_path: Path):
    config_path, _ = _create_calibration_artifacts(tmp_path)
    bundle = export_calibration_bundle(config_path, bundle=tmp_path / "bundle.zip")
    target = tmp_path / "imported"
    conflict_file = (
        target / "my_colorpath" / "color_paths" / "from_labels" / "metadata.json"
    )
    conflict_file.parent.mkdir(parents=True, exist_ok=True)
    conflict_file.write_text("{}")

    with pytest.raises(FileExistsError):
        import_calibration_bundle(config_path, bundle=bundle, target_folder=target)


def test_preview_calibration_bundle_import_conflicts(tmp_path: Path):
    config_path, _ = _create_calibration_artifacts(tmp_path)
    bundle = export_calibration_bundle(config_path, bundle=tmp_path / "bundle.zip")
    target = tmp_path / "imported"
    conflict = (
        target / "my_colorpath" / "color_to_mass" / "from_labels" / "metadata.json"
    )
    conflict.parent.mkdir(parents=True, exist_ok=True)
    conflict.write_text("{}")

    conflicts = preview_calibration_bundle_import_conflicts(
        config_path, bundle=bundle, target_folder=target
    )
    assert conflict in conflicts
