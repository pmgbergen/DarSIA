from pathlib import Path

import pytest

from darsia.presets.workflows.config.colorchannel_registry import ColorChannelRegistry


def _write_config(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


def test_colorchannel_registry_normalizes_case(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.toml",
        """
[colorchannel.my_channel]
color_space = "rgb"
channel = "R"
""".strip(),
    )

    cfg = ColorChannelRegistry().load(config_path)
    resolved = cfg.resolve("my_channel")["my_channel"]
    assert resolved.color_space == "RGB"
    assert resolved.channel == "r"


def test_colorchannel_registry_rejects_invalid_channel(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.toml",
        """
[colorchannel.bad]
color_space = "HSV"
channel = "x"
""".strip(),
    )

    with pytest.raises(ValueError, match="Unsupported colorchannel.bad.channel"):
        ColorChannelRegistry().load(config_path)


def test_colorchannel_registry_resolve_missing_key_lists_available(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path / "config.toml",
        """
[colorchannel.red]
color_space = "RGB"
channel = "r"
""".strip(),
    )

    registry = ColorChannelRegistry().load(config_path)
    with pytest.raises(KeyError, match="Available keys: \\['red'\\]"):
        registry.resolve("green")
