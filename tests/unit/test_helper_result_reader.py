from pathlib import Path
from types import SimpleNamespace

import numpy as np

import darsia
from darsia.presets.workflows.config.format_registry import FormatRegistry
from darsia.presets.workflows.helper.helper_result_reader import (
    _collect_result_files,
    _compute_statistics,
    _resolve_result_format,
)


def test_resolve_result_format_accepts_registry_key(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[format.csv.csv_default]
name = "stem"
""".strip()
    )
    registry = FormatRegistry().load(config_path)
    config = SimpleNamespace(format_registry=registry)

    spec = _resolve_result_format(config, "csv_default")

    assert spec.type == "csv"
    assert spec.folder_name == "csv_default"


def test_collect_result_files_prefers_stem_match(tmp_path: Path) -> None:
    result_folder = tmp_path / "results"
    result_folder.mkdir(parents=True)
    (result_folder / "img_b.csv").write_text("1")
    (result_folder / "img_a.csv").write_text("2")

    source_paths = [Path("/tmp/none/img_a.jpg"), Path("/tmp/none/img_b.jpg")]
    matched = _collect_result_files(source_paths, result_folder, "csv")

    assert [path.name for path in matched] == ["img_a.csv", "img_b.csv"]


def test_compute_statistics_uses_geometry_integral() -> None:
    image = darsia.ScalarImage(
        np.array([[1.0, 2.0], [3.0, 4.0]]), dimensions=[2.0, 2.0]
    )
    geometry = darsia.Geometry(space_dim=2, num_voxels=(2, 2), dimensions=(2.0, 2.0))

    minimum, maximum, integral = _compute_statistics(image, geometry=geometry)

    assert minimum == 1.0
    assert maximum == 4.0
    assert integral == 10.0
