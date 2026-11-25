"""The pressure/temperature protocol: stored by the protocol setup from the inputs in
[protocols] (constant values or a file of the user), and read from there only."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

import darsia
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.heterogeneous_color_to_mass_analysis import (
    HeterogeneousColorToMassAnalysis,
)
from darsia.presets.workflows.results_folder import (
    PRESSURE_TEMPERATURE,
    _has_configured_protocol_files,
    protocol_csv,
)
from darsia.presets.workflows.setup.setup_protocols import setup_imaging_protocol

START = datetime(2026, 1, 1, 12, 0, 0)
"""Start of the experiment."""


def _write_protocol(path: Path, rows: list[tuple[datetime, float, float]]) -> Path:
    """Protocol file of (date, pressure, temperature) rows with gradients 0.1, 0.2."""
    pd.DataFrame(
        {
            "datetime": [date.isoformat() for date, _, _ in rows],
            "pressure_bar": [pressure for _, pressure, _ in rows],
            "temperature_celsius": [temperature for _, _, temperature in rows],
            "pressure_gradient_bar": [0.1] * len(rows),
            "temperature_gradient_celsius": [0.2] * len(rows),
        }
    ).to_csv(path, index=False)
    return path


def _two_rows(path: Path) -> Path:
    """Protocol from (1 bar, 20 C) to (2 bar, 30 C) over two hours."""
    return _write_protocol(
        path, [(START, 1.0, 20.0), (START + timedelta(hours=2), 2.0, 30.0)]
    )


def test_protocol_of_one_row_is_constant(tmp_path: Path) -> None:
    protocol = darsia.PressureTemperatureProtocol(
        _write_protocol(tmp_path / "pt.csv", [(START, 1.2, 22.0)])
    )
    assert protocol.is_constant
    for date in [None, START - timedelta(days=1), START + timedelta(days=100)]:
        assert protocol.get_state(date) == darsia.ThermodynamicState(1.2, 22.0)
        assert protocol.get_gradient(date) == darsia.ThermodynamicState(0.1, 0.2)


def test_protocol_is_interpolated_and_needs_a_date(tmp_path: Path) -> None:
    protocol = darsia.PressureTemperatureProtocol(_two_rows(tmp_path / "pt.csv"))

    assert not protocol.is_constant
    assert protocol.get_state(START + timedelta(hours=1)) == darsia.ThermodynamicState(
        1.5, 25.0
    )
    assert protocol.get_gradient(START) == darsia.ThermodynamicState(0.1, 0.2)
    with pytest.raises(ValueError, match="needs a date"):
        protocol.get_state()
    with pytest.raises(ValueError, match="needs a date"):
        protocol.get_gradient()


def _config(tmp_path: Path, protocols: str) -> Path:
    """Config of one folder of two images with the given pressure/temperature keys."""
    images = tmp_path / "images"
    images.mkdir()
    now = START.timestamp()
    for i, name in enumerate(["img_0001.JPG", "img_0002.JPG"]):
        Image.new("RGB", (10, 10)).save(images / name)
        os.utime(images / name, (now + 60 * i, now + 60 * i))
    path = tmp_path / "config.toml"
    path.write_text(
        f"""
[data]
folder = "{images.as_posix()}"
baseline = "img_0001.JPG"
results = "{(tmp_path / "results").as_posix()}"

[protocols]
injection = "{(tmp_path / "protocols" / "injection.csv").as_posix()}"
imaging_mode = "ctime"
{protocols}

[protocols.imaging]
"{images.as_posix()}" = "{(tmp_path / "protocols" / "imaging.csv").as_posix()}"
""",
        encoding="utf-8",
    )
    return path


def _protocol(path: Path) -> darsia.PressureTemperatureProtocol:
    """The protocol of the experiment of the config."""
    config = FluidFlowerConfig(path, require_data=False, require_results=False)
    return darsia.ProtocolledExperiment.init_from_config(
        config
    ).pressure_temperature_protocol


def test_constant_mode_stores_the_config_values(tmp_path: Path) -> None:
    user_file = tmp_path / "user.csv"
    path = _config(
        tmp_path,
        f"""pressure_temperature_mode = "constant"
pressure_bar = 1.5
temperature_celsius = 25.0
pressure_temperature = "{user_file.as_posix()}"
""",
    )
    setup_imaging_protocol(path)

    # The file of the user is not written (nor needed) in constant mode
    assert not user_file.exists()
    assert protocol_csv(tmp_path / "results", PRESSURE_TEMPERATURE).exists()
    protocol = _protocol(path)
    assert protocol.is_constant
    assert protocol.get_state() == darsia.ThermodynamicState(1.5, 25.0)
    assert protocol.get_gradient() == darsia.ThermodynamicState(0.0, 0.0)


def test_load_from_csv_mode_stores_the_file_of_the_user(tmp_path: Path) -> None:
    user_file = _two_rows(tmp_path / "user.csv")
    path = _config(
        tmp_path,
        f"""pressure_temperature_mode = "Load from CSV"
pressure_temperature = "{user_file.as_posix()}"
""",
    )
    setup_imaging_protocol(path)

    # The analysis reads the stored protocol, not the file of the user
    user_file.unlink()
    protocol = _protocol(path)
    assert not protocol.is_constant
    assert protocol.get_state(START + timedelta(hours=1)) == darsia.ThermodynamicState(
        1.5, 25.0
    )


def test_stored_protocol_is_rewritten_by_the_setup(tmp_path: Path) -> None:
    path = _config(tmp_path, "pressure_bar = 1.5")
    setup_imaging_protocol(path)
    path.write_text(
        path.read_text(encoding="utf-8").replace("1.5", "2.5"), encoding="utf-8"
    )
    setup_imaging_protocol(path, force=True)
    assert _protocol(path).get_state() == darsia.ThermodynamicState(2.5, 20.0)


def test_load_from_csv_mode_needs_the_file(tmp_path: Path) -> None:
    path = _config(tmp_path, 'pressure_temperature_mode = "Load from CSV"')
    with pytest.raises(ValueError, match="requires"):
        setup_imaging_protocol(path)


def test_analysis_needs_the_protocol_setup(tmp_path: Path) -> None:
    path = _config(tmp_path, "")
    with pytest.raises(FileNotFoundError, match="protocol setup"):
        _protocol(path)


def test_protocol_status_follows_the_stored_protocol(tmp_path: Path) -> None:
    injection = tmp_path / "injection.csv"
    injection.touch()
    merged = {
        "data": {"results": (tmp_path / "results").as_posix()},
        "protocols": {"injection": injection.as_posix()},
    }
    assert not _has_configured_protocol_files(merged)
    stored = protocol_csv(tmp_path / "results", PRESSURE_TEMPERATURE)
    stored.parent.mkdir(parents=True)
    stored.touch()
    assert _has_configured_protocol_files(merged)


def test_fluid_state_of_a_constant_protocol_needs_no_date(tmp_path: Path) -> None:
    states: list[dict] = []
    analysis = SimpleNamespace(
        pressure_temperature_protocol=darsia.PressureTemperatureProtocol(
            _write_protocol(tmp_path / "pt.csv", [(START, 1.5, 25.0)])
        ),
        fluid_model=SimpleNamespace(update_state=lambda **state: states.append(state)),
    )
    HeterogeneousColorToMassAnalysis.update_state(analysis, None)
    assert states == [
        {
            "atmospheric_pressure": 1.5,
            "atmospheric_temperature": 25.0,
            "atmospheric_pressure_gradient": 0.1,
            "atmospheric_temperature_gradient": 0.2,
        }
    ]
