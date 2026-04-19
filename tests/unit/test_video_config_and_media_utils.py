import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.presets.workflows.config.video import VideoConfig
from darsia.presets.workflows.utils.utils_media import (
    _apply_overlay,
    _protocol_sort_frames,
)


def _write_toml(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(textwrap.dedent(content))
    return path


def test_video_config_defaults_and_results_video_folder(tmp_path: Path) -> None:
    config_path = _write_toml(
        tmp_path,
        """
        [video]

        [video.source]
        folder = "segmentation"
        """,
    )
    results = tmp_path / "results"
    cfg = VideoConfig().load(config_path, results=results)
    assert cfg.source.folder == Path("segmentation")
    assert cfg.output.formats == ["mp4"]
    assert cfg.folder == results / "videos"


def test_video_config_requires_source_folder(tmp_path: Path) -> None:
    config_path = _write_toml(
        tmp_path,
        """
        [video]
        """,
    )
    with pytest.raises(KeyError, match="source"):
        VideoConfig().load(config_path, results=tmp_path / "results")


def test_video_config_rejects_empty_source_folder(tmp_path: Path) -> None:
    config_path = _write_toml(
        tmp_path,
        """
        [video.source]
        folder = "   "
        """,
    )
    with pytest.raises(ValueError, match="must not be empty"):
        VideoConfig().load(config_path, results=tmp_path / "results")


def test_video_config_loads_source_folder_path(tmp_path: Path) -> None:
    config_path = _write_toml(
        tmp_path,
        """
        [video]
        [video.source]
        folder = "custom/roi/stream"
        """,
    )
    cfg = VideoConfig().load(config_path, results=tmp_path / "results")
    assert cfg.source.folder == Path("custom/roi/stream")


class _FakeImagingProtocol:
    def __init__(self) -> None:
        self._base = datetime(2026, 1, 1, 0, 0, 0)

    def is_blacklisted(self, path: Path) -> bool:
        return path.stem == "img_00003"

    def get_datetime(self, path: Path) -> datetime:
        if path.stem == "noise":
            raise ValueError("not in protocol")
        image_id = int(path.stem.split("_")[-1])
        return self._base + timedelta(hours=image_id)


class _FakeExperiment:
    def __init__(self) -> None:
        self._base = datetime(2026, 1, 1, 0, 0, 0)
        self._imaging_protocol = _FakeImagingProtocol()

    def is_blacklisted(self, path: Path) -> bool:
        return self._imaging_protocol.is_blacklisted(path)

    def get_datetime(self, path: Path) -> datetime:
        return self._imaging_protocol.get_datetime(path)

    def time_since_start(self, dt: datetime) -> float:
        return (dt - self._base).total_seconds() / 3600.0


def test_protocol_sort_frames_filters_blacklist_and_invalid_and_sorts() -> None:
    exp = _FakeExperiment()
    paths = [
        Path("img_00004.png"),
        Path("img_00002.png"),
        Path("noise.png"),
        Path("img_00003.png"),
    ]
    ordered = _protocol_sort_frames(exp, paths, "protocol")
    assert [p.name for p, _, _ in ordered] == ["img_00002.png", "img_00004.png"]
    assert [round(elapsed, 2) for _, _, elapsed in ordered] == [2.0, 4.0]


def test_apply_overlay_writes_elapsed_time_and_note_on_frame() -> None:
    frame = np.zeros((120, 240, 3), dtype=np.uint8)
    overlay_cfg = SimpleNamespace(
        show_elapsed_time=True,
        elapsed_time_format="Elapsed: {:.2f} h",
        show_note=True,
        note="Experiment #1 - Segmentation",
        font_scale=0.6,
        thickness=1,
        line_spacing=6,
        box_padding=6,
        position=(10, 10),
        box_enabled=True,
        box_alpha=0.5,
        text_color=(255, 255, 255),
        box_color=(0, 0, 0),
    )
    out = _apply_overlay(frame.copy(), elapsed_time_h=1.25, overlay_config=overlay_cfg)
    assert np.any(out != frame)
