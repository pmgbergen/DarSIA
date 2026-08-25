"""Test the generic create_dataclass_group_map_input primitive.

This tests the new group-box-per-entry widget builder independently,
without relying on format_registry or other specific implementations.
"""

from dataclasses import dataclass, field

import pytest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLineEdit,
    QWidget,
)


@dataclass
class SimpleTestEntry:
    """A minimal test dataclass for group-map testing."""

    name: str = field(metadata={"name": "Entry Name", "help": "Name of this entry"})
    entry_type: str = field(
        default="type_a",
        metadata={
            "name": "Type",
            "help": "Entry type",
            "options": ["type_a", "type_b", "type_c"],
            "widget": "select",
        },
    )
    description: str = field(
        default="",
        metadata={"name": "Description", "help": "Optional description"},
    )
    enabled: bool = field(
        default=True,
        metadata={"name": "Enabled", "help": "Is this entry active?"},
    )
    value: int | None = field(
        default=None,
        metadata={
            "name": "Value",
            "help": "A numeric value",
            "widget": "int",
            "depends_on": {"field": "entry_type", "value": ["type_b", "type_c"]},
        },
    )


@pytest.fixture
def qapp():
    """Provide a QApplication for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def mock_main_window(qapp):
    """Provide a mock main_window object with config_dict."""

    class MockMainWindow:
        def __init__(self):
            self.config_dict = {
                "test_entries": {
                    "entry_1": {
                        "name": "entry_1",
                        "entry_type": "type_a",
                        "description": "First entry",
                        "enabled": True,
                    },
                    "entry_2": {
                        "name": "entry_2",
                        "entry_type": "type_b",
                        "description": "Second entry",
                        "enabled": False,
                        "value": 42,
                    },
                }
            }

        def print_log(self, message):
            """Mock print_log method."""
            pass

    return MockMainWindow()


def test_create_dataclass_group_map_basic(mock_main_window, qapp):
    """Test that create_dataclass_group_map_input creates the basic structure."""
    # Import SettingsFactory to use its create_dataclass_group_map_input method
    from darsia.gui.ui.settings import SettingsFactory

    # Create a SettingsFactory instance
    factory = SettingsFactory(mock_main_window)

    setting_dict = {
        "key": "test_entries",
        "name": "Test Entries",
        "help": "Test entry collection",
    }

    # Create a form context to pass to the builder
    parent_widget = QWidget()
    parent_form = QFormLayout(parent_widget)
    form_context = {"form": parent_form}

    # Call create_dataclass_group_map_input via the factory
    display_name, result_dict = factory.create_dataclass_group_map_input(
        setting_dict, SimpleTestEntry, form_context=form_context
    )

    # Verify the result structure
    assert display_name == "Test Entries"
    assert "widget" in result_dict
    assert "dataclass_group_map" in result_dict
    assert "entries" in result_dict

    # Should have 2 entries from config_dict
    entries = result_dict["entries"]
    assert len(entries) == 2

    # Verify entry structure
    for entry in entries:
        assert "name" in entry
        assert "name_edit" in entry
        assert "widget" in entry
        assert "fields" in entry
        assert "field_schemas" in entry
        assert "remove_button" in entry

    # Verify first entry data
    entry_1 = entries[0]
    assert entry_1["name"] == "entry_1"
    assert entry_1["name_edit"].text() == "entry_1"
    assert "enabled" in entry_1["fields"]
    assert "entry_type" in entry_1["fields"]
    assert "description" in entry_1["fields"]
    assert "value" in entry_1["fields"]


def test_depends_on_visibility_with_list_values(mock_main_window, qapp):
    """Test that depends_on with list values works for conditional visibility."""
    from darsia.gui.ui.settings import SettingsFactory

    # Create a SettingsFactory instance
    factory = SettingsFactory(mock_main_window)

    setting_dict = {
        "key": "test_entries",
        "name": "Test Entries",
        "help": "Test entry collection",
    }

    parent_widget = QWidget()
    parent_form = QFormLayout(parent_widget)
    form_context = {"form": parent_form}

    display_name, result_dict = factory.create_dataclass_group_map_input(
        setting_dict, SimpleTestEntry, form_context=form_context
    )

    entries = result_dict["entries"]
    entry_2 = entries[1]  # entry_2 has entry_type="type_b"

    # The "value" field should be visible because entry_type is "type_b"
    # (which is in the depends_on list ["type_b", "type_c"])
    # This is verified by the depends_on wiring at initialization time

    # Verify the value field exists and is wired
    assert "value" in entry_2["fields"]
    value_widget = entry_2["fields"]["value"]
    # Value widget should be visible initially (since type_b is in the list)
    assert value_widget.isVisible()

    # Verify entry_type field
    assert "entry_type" in entry_2["fields"]
    type_combo = entry_2["fields"]["entry_type"]
    assert isinstance(type_combo, QComboBox)
    assert type_combo.currentText() == "type_b"

    # Change type to type_a (should hide value field)
    type_combo.setCurrentText("type_a")
    # Give the signal a moment to process
    qapp.processEvents()
    assert (
        not value_widget.isVisible()
    ), "Value field should be hidden when type_a is selected"

    # Change type back to type_c (should show value field)
    type_combo.setCurrentText("type_c")
    qapp.processEvents()
    assert (
        value_widget.isVisible()
    ), "Value field should be visible when type_c is selected"


def test_save_pass_parser_generic(mock_main_window, qapp):
    """Test the generic save-pass parser for dataclass_group_map."""
    from darsia.gui.ui.settings import SettingsFactory

    # Create a SettingsFactory instance
    factory = SettingsFactory(mock_main_window)
    factory.settings_inputs = {}

    setting_dict = {
        "key": "test_entries",
        "name": "Test Entries",
        "help": "Test entry collection",
    }

    parent_widget = QWidget()
    parent_form = QFormLayout(parent_widget)
    form_context = {"form": parent_form}

    # Build the group-map entry structure
    display_name, result_dict = factory.create_dataclass_group_map_input(
        setting_dict, SimpleTestEntry, form_context=form_context
    )

    # Simulate user editing the first entry's description
    entries = result_dict["entries"]
    entry_1 = entries[0]

    # Edit the description field
    desc_widget = entry_1["fields"]["description"]
    desc_widget.setText("Updated description for entry 1")

    # Edit the enabled checkbox
    enabled_widget = entry_1["fields"]["enabled"]
    enabled_widget.setChecked(False)

    # Store the result in settings_inputs as would happen in real usage
    factory.settings_inputs["test_entries"] = result_dict

    # Now simulate the save-pass parser
    # (This would normally be part of _sync_settings_inputs_to_config_dict)
    for key, value in factory.settings_inputs.items():
        if isinstance(value, dict) and "dataclass_group_map" in value:
            result = []
            for entry_data in value["entries"]:
                entry_name = entry_data["name_edit"].text().strip()
                if not entry_name:
                    continue

                entry_dict = {"name": entry_name}

                # Extract values from each field widget
                for field_name, field_widget in entry_data["fields"].items():
                    field_schema = entry_data["field_schemas"][field_name]
                    field_type = field_schema.get("type")
                    field_default = field_schema.get("default")

                    # Extract value based on widget type
                    extracted_value = None
                    should_include = False

                    if isinstance(field_widget, QCheckBox):
                        extracted_value = field_widget.isChecked()
                        # Only include if True (checkbox default is usually False)
                        should_include = extracted_value is True
                    elif isinstance(field_widget, QComboBox):
                        text_value = field_widget.currentText().strip()
                        if text_value:
                            extracted_value = text_value
                            # Omit if equals default
                            should_include = extracted_value != field_default
                    elif isinstance(field_widget, QLineEdit):
                        text_value = field_widget.text().strip()
                        if text_value:
                            extracted_value = text_value
                            # String fields: store as string, omit if equals default
                            should_include = extracted_value != field_default

                    if should_include and extracted_value is not None:
                        entry_dict[field_name] = extracted_value

                result.append(entry_dict)

            # Write result
            dialog.set_value(dialog.main_window.config_dict, key, result)

    # Verify the parsed result
    saved_entries = dialog.main_window.config_dict["test_entries"]
    assert isinstance(saved_entries, list)
    assert len(saved_entries) == 2

    # Check first entry
    entry_1_saved = saved_entries[0]
    assert entry_1_saved["name"] == "entry_1"
    assert entry_1_saved["description"] == "Updated description for entry 1"
    assert "enabled" not in entry_1_saved  # False is the default, so omitted

    # Check second entry (should be mostly unchanged)
    entry_2_saved = saved_entries[1]
    assert entry_2_saved["name"] == "entry_2"
    # description should still be there (it's different from default "")
    assert entry_2_saved["description"] == "Second entry"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
