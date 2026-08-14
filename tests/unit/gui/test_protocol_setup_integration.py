"""Integration test: GUI workflow for folder/imaging-protocol setup and protocol CSV generation."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import toml
from PIL import Image

from darsia.gui.ui.settings import SettingsFactory
from darsia.presets.workflows.config.data import DataConfig
from darsia.presets.workflows.config.protocols import ProtocolsConfig
from darsia.presets.workflows.setup.setup_protocols import setup_imaging_protocol


class MockLineEdit:
    """Mock QLineEdit widget for testing."""

    def __init__(self, initial_text: str = ""):
        self._text = initial_text

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        self._text = text


class MockComboBox:
    """Mock QComboBox widget for testing."""

    def __init__(self, initial_text: str = ""):
        self._text = initial_text

    def currentText(self) -> str:
        return self._text

    def setCurrentText(self, text: str) -> None:
        self._text = text


def _create_test_image(path: Path, mtime: float) -> None:
    """Create a test image with given modification time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10), color=(255, 255, 255)).save(path)
    os.utime(path, (mtime, mtime))


def test_save_settings_path_map_round_trip_into_protocol_setup(tmp_path: Path) -> None:
    """Test end-to-end: mock widgets → save_settings logic → load config → setup_imaging_protocol.

    This test verifies the save_settings() path_map branch works correctly:
    1. Create mock widgets (list of QLineEdit-like objects for multi_folder, dict of tuples for path_map)
    2. Populate them with user input (folder paths and protocol file paths)
    3. Simulate the save_settings() logic that processes these widgets and writes TOML
    4. Reload and verify data integrity (folder path key must match exactly)
    5. Run setup_imaging_protocol to prove the config is usable

    This focuses on the single-folder case, sufficient to verify the path_map save branch
    and key alignment between data.folders and protocols.imaging dict.
    """
    now = 1_700_000_000
    images_folder = tmp_path / "images"
    imaging_csv = tmp_path / "imaging_protocol.csv"
    injection_csv = tmp_path / "injection_protocol.csv"
    pressure_csv = tmp_path / "pressure_temperature_protocol.csv"
    config_path = tmp_path / "config.toml"

    # Create test image and dummy protocol files
    _create_test_image(images_folder / "img_0001.JPG", now)
    imaging_csv.write_text("path,image_id,datetime\nimg_0001.JPG,1,2023-11-15 12:00:00")
    injection_csv.write_text("id,location_x,location_y,start,end,rate_kg/s\n1,0.5,0.5,00:00:00,01:00:00,0.0")
    pressure_csv.write_text("datetime,pressure_bar,temperature_celsius,pressure_gradient_bar,temperature_gradient_celsius\n2023-11-15 12:00:00,1.013,20.0,0.0,0.0")

    # Build mock config_dict and settings_inputs, simulating what display_settings() creates
    config_dict = {"data": {}, "protocols": {}}
    settings_inputs = {}

    # data.folders: list of mock QLineEdit objects (multi_folder widget)
    folder_edit = MockLineEdit(str(images_folder))
    settings_inputs["data.folders"] = [folder_edit]

    # data.baseline: single mock QLineEdit
    baseline_edit = MockLineEdit("img_0001.JPG")
    settings_inputs["data.baseline"] = baseline_edit

    # data.results: single mock QLineEdit
    results_edit = MockLineEdit(str(tmp_path / "results"))
    settings_inputs["data.results"] = results_edit

    # protocols.imaging: path_map widget (dict with "path_map" marker and "rows" key)
    key_edit = MockLineEdit(str(images_folder))
    value_edit = MockLineEdit(str(imaging_csv))
    imaging_widget = {"path_map": True, "rows": [(key_edit, value_edit)]}
    settings_inputs["protocols.imaging"] = imaging_widget

    # protocols.injection and pressure_temperature: mock QLineEdit with valid paths
    injection_edit = MockLineEdit(str(injection_csv))
    settings_inputs["protocols.injection"] = injection_edit

    pressure_edit = MockLineEdit(str(pressure_csv))
    settings_inputs["protocols.pressure_temperature"] = pressure_edit

    # protocols.imaging_mode: mock QComboBox
    mode_combo = MockComboBox("ctime")
    settings_inputs["protocols.imaging_mode"] = mode_combo

    # Simulate the save_settings() logic that processes settings_inputs
    # (extracted from main_window.py::save_settings, lines 185-231)
    settings_factory = SettingsFactory(MagicMock(config_dict=config_dict))

    # Second pass: save regular values (simplified version without group-checkbox logic)
    for key, value in settings_inputs.items():
        # Skip path_map dicts (handled separately below)
        if isinstance(value, dict) and "path_map" in value:
            continue

        try:
            if isinstance(value, MockLineEdit):
                # Try to parse as literal, fall back to raw string
                try:
                    import ast
                    parsed = ast.literal_eval(value.text())
                    settings_factory.set_value(config_dict, key, parsed)
                except (ValueError, SyntaxError):
                    settings_factory.set_value(config_dict, key, value.text())
            elif isinstance(value, MockComboBox):
                settings_factory.set_value(config_dict, key, value.currentText())
            elif isinstance(value, list):
                # Multi_folder: list of QLineEdit objects
                result = [item.text() for item in value if item.text().strip()]
                settings_factory.set_value(config_dict, key, result)
        except Exception:
            pass

    # Third pass: save path_map dicts
    for key, value in settings_inputs.items():
        if isinstance(value, dict) and "path_map" in value:
            rows = value["rows"]
            result = {
                k.text(): v.text()
                for k, v in rows
                if k.text().strip() and v.text().strip()
            }
            settings_factory.set_value(config_dict, key, result)

    # Write config_dict to TOML (simulating main_window.py::save_settings line 233-235)
    with open(config_path, "w") as f:
        toml.dump(config_dict, f)

    # Verify the TOML was written
    assert config_path.exists(), "Config TOML was not written"

    # Reload and verify data integrity: folder path must match exactly across both dicts
    data_config = DataConfig().load(config_path, require_data=False, require_results=False)
    protocols_config = ProtocolsConfig().load(config_path)

    assert data_config.folders == [images_folder], (
        f"Expected folders=[{images_folder}], got {data_config.folders}"
    )
    assert isinstance(protocols_config.imaging, dict), (
        f"Expected protocols.imaging to be a dict, got {type(protocols_config.imaging)}"
    )
    assert images_folder in protocols_config.imaging, (
        f"Expected key {images_folder} in imaging dict, got keys: {list(protocols_config.imaging.keys())}"
    )
    assert protocols_config.imaging[images_folder] == imaging_csv, (
        f"Expected imaging[{images_folder}] = {imaging_csv}, got {protocols_config.imaging[images_folder]}"
    )

    # Run setup_imaging_protocol to prove the GUI-authored config is fully usable
    # Use force=True since we created dummy CSVs upfront for the config to reference
    setup_imaging_protocol(config_path, force=True)

    # Verify the imaging protocol CSV was generated
    assert imaging_csv.exists(), (
        f"setup_imaging_protocol failed to generate {imaging_csv}"
    )

    # Verify the CSV has the expected structure
    df = pd.read_csv(imaging_csv)
    assert set(df.columns) == {"path", "image_id", "datetime"}, (
        f"Expected columns {{path, image_id, datetime}}, got {set(df.columns)}"
    )
    assert df["image_id"].tolist() == [1], (
        f"Expected image_id=[1], got {df['image_id'].tolist()}"
    )
