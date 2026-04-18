from pathlib import Path
from types import SimpleNamespace

import numpy as np

import darsia
from darsia.presets.workflows.analysis.image_export_formats import ImageExportFormats
from darsia.presets.workflows.config.format_registry import FormatRegistry


def _write(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


def test_export_naming_options(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[format.npy.opt_time_hms]
name = "time_HH:MM:SS"

[format.npy.opt_time_ms]
name = "time_MM:SS"

[format.npy.opt_name_time]
name = "name_time_HH:MM:SS"

[format.npy.opt_name_stem]
name = "name_stem"
""".strip(),
    )
    registry = FormatRegistry().load(config_path)
    config = SimpleNamespace(format_registry=registry, analysis=None)
    exporter = ImageExportFormats(config, list(registry.keys()))
    image = darsia.ScalarImage(
        np.zeros((3, 3), dtype=np.float32),
        dimensions=[1.0, 1.0],
        time=9045.0,
        name="mass",
    )

    exporter.export_image(image, tmp_path, "DSC01621", supported_types={"npy"})

    assert (tmp_path / "npy_opt_time_hms" / "time_02_30_45_hrs.npy").exists()
    assert (tmp_path / "npy_opt_time_ms" / "time_150_45_hrs.npy").exists()
    assert (tmp_path / "npy_opt_name_time" / "mass_2_30_45_hrs.npy").exists()
    assert (tmp_path / "npy_opt_name_stem" / "mass_DSC01621.npy").exists()
