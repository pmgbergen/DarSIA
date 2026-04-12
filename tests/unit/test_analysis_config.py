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
