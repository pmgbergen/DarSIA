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


def test_analysis_thresholding_layers_and_formats_are_loaded(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
[analysis.thresholding]
formats = ["jpg", "npz"]
[analysis.thresholding.layers.gas]
mode = "saturation_g"
threshold_min = 0.15
label = "Gas plume"
fill = [255, 0, 0]
stroke = [255, 255, 255]
[analysis.thresholding.layers.aq]
mode = "concentration_aq"
threshold_max = 0.05
label = "Aqueous plume"
fill = [0, 0, 255]
stroke = [255, 255, 255]
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
    assert config.thresholding.formats == ["jpg", "npz"]
    assert set(config.thresholding.layers.keys()) == {"gas", "aq"}
    assert config.thresholding.layers["gas"].mode == "saturation_g"
    assert config.thresholding.layers["gas"].threshold_min == 0.15
    assert config.thresholding.layers["gas"].label == "Gas plume"
    assert config.thresholding.layers["aq"].mode == "concentration_aq"
    assert config.thresholding.layers["aq"].threshold_max == 0.05


def test_analysis_thresholding_rejects_invalid_layer_mode(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
[analysis.thresholding]
[analysis.thresholding.layers.bad]
mode = "not_supported"
threshold_min = 0.1
""".strip(),
    )

    with pytest.raises(ValueError, match=r"Unsupported analysis\.thresholding\.layers"):
        AnalysisConfig().load(
            path=config_path,
            data=tmp_path,
            results=tmp_path,
        )


def test_analysis_thresholding_rejects_invalid_formats(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
[analysis.thresholding]
formats = ["jpg", "png"]
[analysis.thresholding.layers.gas]
mode = "saturation_g"
threshold_min = 0.1
""".strip(),
    )

    with pytest.raises(
        ValueError, match=r"Unsupported \[analysis\.thresholding\]\.formats entries"
    ):
        AnalysisConfig().load(
            path=config_path,
            data=tmp_path,
            results=tmp_path,
        )


def test_analysis_thresholding_accepts_extended_modes(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
[analysis.thresholding]
[analysis.thresholding.layers.red]
mode = "colorchannel.rgb.r"
threshold_min = 0.2
[analysis.thresholding.layers.green_band]
mode = "colorrange.custom_range"
threshold_min = 0.5
""".strip(),
    )

    config = AnalysisConfig().load(
        path=config_path,
        data=tmp_path,
        results=tmp_path,
    )
    assert config.thresholding is not None
    assert config.thresholding.layers["red"].mode == "colorchannel.rgb.r"
    assert config.thresholding.layers["green_band"].mode == "colorrange.custom_range"
