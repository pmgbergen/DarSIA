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
[format.npy.opt_time_hh]
name = "time_HH"

[format.npy.opt_time_hh_mm]
name = "time_HH:MM"

[format.npy.opt_time_hh_mm_ss]
name = "time_HH:MM:SS"

[format.npy.opt_time_mm_ss]
name = "time_MM:SS"

[format.npy.opt_time_dd_hh]
name = "time_DD:HH"

[format.npy.opt_time_dd_hh_mm]
name = "time_DD:HH:MM"

[format.npy.opt_stem_time_hh_mm]
name = "stem_time_HH:MM"

[format.npy.opt_custom]
name = "spatial_map_hh_mm_h"
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

    assert (tmp_path / "opt_time_hh" / "time_02_hrs.npy").exists()
    assert (tmp_path / "opt_time_hh_mm" / "time_02_30_hrs.npy").exists()
    assert (tmp_path / "opt_time_hh_mm_ss" / "time_02_30_45_hrs.npy").exists()
    assert (tmp_path / "opt_time_mm_ss" / "time_150_45_min.npy").exists()
    assert (tmp_path / "opt_time_dd_hh" / "time_00_02_days_hrs.npy").exists()
    assert (tmp_path / "opt_time_dd_hh_mm" / "time_00_02_30_days_hrs.npy").exists()
    assert (tmp_path / "opt_stem_time_hh_mm" / "DSC01621_02_30_hrs.npy").exists()
    assert (tmp_path / "opt_custom" / "spatial_map_02_30_h_hrs.npy").exists()
