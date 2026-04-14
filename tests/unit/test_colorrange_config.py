from pathlib import Path

import pytest

from darsia.presets.workflows.config.colorrange import ColorRangeConfig


def _write(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


def test_colorrange_config_loads_none_bounds(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[colorrange.custom_range]
color_space = "HSV"
range = [[0.2, 0.4], [0.5, "none"], [0.8, "none"]]
""".strip(),
    )

    cfg = ColorRangeConfig().load(config_path)
    assert "custom_range" in cfg.ranges
    assert cfg.ranges["custom_range"].color_space == "HSV"
    assert cfg.ranges["custom_range"].ranges == [(0.2, 0.4), (0.5, None), (0.8, None)]


def test_colorrange_config_rejects_invalid_shape(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[colorrange.bad]
color_space = "HSV"
range = [[0.2, 0.4], [0.5, 0.9]]
""".strip(),
    )
    with pytest.raises(ValueError, match="must be a list of 3"):
        ColorRangeConfig().load(config_path)
