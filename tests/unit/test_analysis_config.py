from pathlib import Path

import pytest

from darsia.presets.workflows.config.analysis import AnalysisConfig


def _write(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


def test_analysis_cropping_formats_are_loaded(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
[analysis.cropping]
formats = ["npz", "jpg"]
""".strip(),
    )

    config = AnalysisConfig().load(
        path=config_path,
        data=tmp_path,
        results=tmp_path,
    )

    assert config.cropping is not None
    assert config.cropping.formats == ["npz", "jpg"]


def test_analysis_cropping_formats_reject_invalid_entries(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
[analysis.cropping]
formats = ["npz", "png"]
""".strip(),
    )

    with pytest.raises(
        ValueError, match=r"Unsupported \[analysis\.cropping\]\.formats entries"
    ):
        AnalysisConfig().load(
            path=config_path,
            data=tmp_path,
            results=tmp_path,
        )


def test_analysis_thresholding_modes_and_thresholds_are_loaded(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
[analysis.thresholding]
modes = ["concentration_aq", "saturation_g", "mass_total", "mass_g", "mass_aq"]
[analysis.thresholding.thresholds]
concentration_aq = 0.05
saturation_g = 0.15
mass_total = 0.0
mass_g = 0.01
mass_aq = 0.02
[analysis.thresholding.legend]
show = true
font_scale = 0.8
text_color = [255, 255, 255]
position = [10, 20]
box_enabled = true
box_color = [0, 0, 0]
box_alpha = 0.5
box_padding = 8
""".strip(),
    )

    config = AnalysisConfig().load(
        path=config_path,
        data=tmp_path,
        results=tmp_path,
    )

    assert config.thresholding is not None
    assert config.thresholding.modes == [
        "concentration_aq",
        "saturation_g",
        "mass_total",
        "mass_g",
        "mass_aq",
    ]
    assert config.thresholding.thresholds == {
        "concentration_aq": 0.05,
        "saturation_g": 0.15,
        "mass_total": 0.0,
        "mass_g": 0.01,
        "mass_aq": 0.02,
    }


def test_analysis_thresholding_rejects_invalid_mode(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
[analysis.thresholding]
modes = ["concentration_aq", "not_supported"]
""".strip(),
    )

    with pytest.raises(
        ValueError, match=r"Unsupported \[analysis\.thresholding\]\.modes entries"
    ):
        AnalysisConfig().load(
            path=config_path,
            data=tmp_path,
            results=tmp_path,
        )
