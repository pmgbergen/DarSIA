"""Unit tests for setup protocol CSV generation."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from darsia.presets.workflows.config.protocol import ProtocolConfig
from darsia.presets.workflows.setup.setup_protocols import (
    preview_protocol_setup_conflicts,
    setup_imaging_protocol,
)


def _write_config(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def _create_image(path: Path, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10), color=(255, 255, 255)).save(path)
    os.utime(path, (mtime, mtime))


def _base_config(tmp_path: Path) -> str:
    return f"""
[data]
folder = "{tmp_path / "images"}"
baseline = "img_0001.JPG"
results = "{tmp_path / "results"}"

[protocols]
imaging = "{tmp_path / "protocols" / "imaging_protocol.csv"}"
injection = "{tmp_path / "protocols" / "injection_protocol.csv"}"
pressure_temperature = "{tmp_path / "protocols" / "pressure_temperature_protocol.csv"}"
imaging_mode = "ctime"
"""


def test_setup_imaging_protocol_generates_all_csv_templates(tmp_path: Path) -> None:
    now = 1_700_000_000
    image_1 = tmp_path / "images" / "img_0001.JPG"
    image_2 = tmp_path / "images" / "img_0002.JPG"
    _create_image(image_1, now)
    _create_image(image_2, now + 60)

    config_path = tmp_path / "config.toml"
    _write_config(config_path, _base_config(tmp_path))

    setup_imaging_protocol(config_path, force=False)

    imaging_path = tmp_path / "protocols" / "imaging_protocol.csv"
    injection_path = tmp_path / "protocols" / "injection_protocol.csv"
    pressure_path = tmp_path / "protocols" / "pressure_temperature_protocol.csv"
    assert imaging_path.exists()
    assert injection_path.exists()
    assert pressure_path.exists()

    imaging_df = pd.read_csv(imaging_path)
    assert set(imaging_df.columns) == {"path", "image_id", "datetime"}
    assert imaging_df["image_id"].tolist() == [1, 2]

    injection_df = pd.read_csv(injection_path)
    assert set(injection_df.columns) == {
        "id",
        "location_x",
        "location_y",
        "start",
        "end",
        "rate_kg/s",
    }
    assert injection_df["rate_kg/s"].iloc[0] == 0.0

    pressure_df = pd.read_csv(pressure_path)
    assert set(pressure_df.columns) == {
        "datetime",
        "pressure_bar",
        "temperature_celsius",
        "pressure_gradient_bar",
        "temperature_gradient_celsius",
    }
    assert pressure_df["pressure_bar"].iloc[0] == 1.013
    assert pressure_df["temperature_celsius"].iloc[0] == 20.0


def test_setup_imaging_protocol_requires_force_for_existing_files(
    tmp_path: Path,
) -> None:
    now = 1_700_000_000
    _create_image(tmp_path / "images" / "img_0001.JPG", now)
    _create_image(tmp_path / "images" / "img_0002.JPG", now + 60)
    config_path = tmp_path / "config.toml"
    _write_config(config_path, _base_config(tmp_path))

    setup_imaging_protocol(config_path, force=False)
    with pytest.raises(FileExistsError, match="Use --force"):
        setup_imaging_protocol(config_path, force=False)

    setup_imaging_protocol(config_path, force=True)


def test_preview_protocol_setup_conflicts_lists_existing_targets(
    tmp_path: Path,
) -> None:
    now = 1_700_000_000
    _create_image(tmp_path / "images" / "img_0001.JPG", now)
    config_path = tmp_path / "config.toml"
    _write_config(config_path, _base_config(tmp_path))

    setup_imaging_protocol(config_path, force=False)
    conflicts = preview_protocol_setup_conflicts(config_path)

    assert len(conflicts) == 3
    assert all(path.exists() for path in conflicts)


def test_protocol_config_defaults_to_exif_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        """
[protocols]
imaging = "imaging_protocol.csv"
""",
    )

    config = ProtocolConfig().load(config_path)
    assert config.imaging_mode == "exif"


def test_protocol_config_supports_per_folder_imaging_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    folder_a = tmp_path / "images_a"
    folder_b = tmp_path / "images_b"
    _write_config(
        config_path,
        f"""
[protocols.imaging]
"{folder_a}" = "{tmp_path / "protocols" / "imaging_a.csv"}"
"{folder_b}" = ["{tmp_path / "protocols" / "imaging_b.xlsx"}", "Sheet1"]
""",
    )

    config = ProtocolConfig().load(config_path)
    assert isinstance(config.imaging, dict)
    assert config.imaging[folder_a] == tmp_path / "protocols" / "imaging_a.csv"
    assert config.imaging[folder_b] == (
        tmp_path / "protocols" / "imaging_b.xlsx",
        "Sheet1",
    )


def test_setup_imaging_protocol_with_multiple_folders(
    tmp_path: Path,
) -> None:
    now = 1_700_000_000
    _create_image(tmp_path / "images_a" / "img_0001.JPG", now)
    _create_image(tmp_path / "images_b" / "img_0001.JPG", now + 60)

    config_path = tmp_path / "config_multi.toml"
    _write_config(
        config_path,
        f"""
[data]
folders = ["{tmp_path / "images_a"}", "{tmp_path / "images_b"}"]
baseline = "img_0001.JPG"
results = "{tmp_path / "results"}"

[protocols]
injection = "{tmp_path / "protocols" / "injection_protocol.csv"}"
pressure_temperature = "{tmp_path / "protocols" / "pressure_temperature_protocol.csv"}"
imaging_mode = "ctime"

[protocols.imaging]
"{tmp_path / "images_a"}" = "{tmp_path / "protocols" / "imaging_a.csv"}"
"{tmp_path / "images_b"}" = "{tmp_path / "protocols" / "imaging_b.csv"}"
""",
    )

    setup_imaging_protocol(config_path, force=False)

    imaging_a = pd.read_csv(tmp_path / "protocols" / "imaging_a.csv")
    imaging_b = pd.read_csv(tmp_path / "protocols" / "imaging_b.csv")
    assert imaging_a["path"].tolist() == ["img_0001.JPG"]
    assert imaging_b["path"].tolist() == ["img_0001.JPG"]


def test_setup_imaging_protocol_requires_mapping_for_multiple_folders(
    tmp_path: Path,
) -> None:
    now = 1_700_000_000
    _create_image(tmp_path / "images_a" / "img_0001.JPG", now)
    _create_image(tmp_path / "images_b" / "img_0001.JPG", now + 60)

    config_path = tmp_path / "config_multi_invalid.toml"
    _write_config(
        config_path,
        f"""
[data]
folders = ["{tmp_path / "images_a"}", "{tmp_path / "images_b"}"]
baseline = "img_0001.JPG"
results = "{tmp_path / "results"}"

[protocols]
imaging = "{tmp_path / "protocols" / "imaging_protocol.csv"}"
imaging_mode = "ctime"
""",
    )

    with pytest.raises(ValueError, match="per-folder table"):
        setup_imaging_protocol(config_path, force=False)
