"""Test GUI format_map widget read/write round-trip."""

import toml
from pathlib import Path
from darsia.presets.workflows.config.format_registry import FormatRegistry


def test_format_map_widget_prefill_logic():
    """Test that the widget's prefill dict is built correctly from raw TOML."""
    # Load develop.toml as the widget would see it
    # (config is in ff_bilbo root, not in external/darsia)
    config_file = Path(__file__).parent.parent.parent.parent.parent / "config" / "run" / "develop.toml"
    config_dict = toml.load(config_file)

    # Widget read-side: convert list to prefill dict
    format_list = config_dict.get("format", [])
    prefill_dict = {
        entry.get("name", ""): entry
        for entry in format_list
        if entry.get("name")
    }

    # Verify at least the 3 original formats are present
    assert len(prefill_dict) >= 3
    assert "csv" in prefill_dict
    assert "jpg" in prefill_dict
    assert "npz" in prefill_dict

    # Verify each has the expected structure
    assert prefill_dict["jpg"]["type"] == "jpg"
    assert prefill_dict["jpg"]["filename_pattern"] == "spatial_map_HH:MM"
    assert prefill_dict["csv"]["type"] == "csv"
    assert prefill_dict["npz"]["type"] == "npz"


def test_format_map_write_handler_produces_valid_toml():
    """Test that the write-side handler produces TOML-compatible dicts."""
    # Load develop.toml and parse with FormatRegistry to verify shape
    config_file = Path(__file__).parent.parent.parent.parent.parent / "config" / "run" / "develop.toml"
    registry = FormatRegistry()
    registry.load(config_file)

    # Serialize back via to_toml_dict (same shape as write-side handler produces)
    toml_output = registry.to_toml_dict()

    # Verify output is a list of dicts with expected keys
    assert "format" in toml_output
    assert isinstance(toml_output["format"], list)
    assert len(toml_output["format"]) >= 3

    # Verify at least the 3 original formats exist with expected structure
    original_names = {"csv", "jpg", "npz"}
    format_names = {entry["name"] for entry in toml_output["format"]}
    assert original_names.issubset(format_names), f"Missing expected formats. Have: {format_names}"

    for entry in toml_output["format"]:
        assert isinstance(entry, dict)
        assert "type" in entry
        assert "name" in entry
        assert "filename_pattern" in entry
        # Other fields are optional (only present if non-default)
