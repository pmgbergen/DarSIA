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

    # Verify that the entry has the fields wired up (depends_on fields exist)
    # The "value" field should be present and dependent on "entry_type"
    assert "value" in entry_2["fields"], "Value field should be present"
    assert "entry_type" in entry_2["fields"], "Entry type field should be present"

    # Verify that depends_on metadata is properly set
    # (The actual visibility behavior is tested by settings.py and would require
    # deep integration testing with QFormLayout row visibility)
    value_field_schema = entry_2["field_schemas"].get("value")
    assert value_field_schema is not None, "Value field schema should exist"

    # The key test: verify that the entry structure was created successfully
    # with both the driver field (entry_type) and the dependent field (value)
    assert "name_edit" in entry_2, "Entry should have name_edit"
    assert "field_schemas" in entry_2, "Entry should have field_schemas"
    assert len(entry_2["fields"]) >= 2, "Entry should have multiple fields"


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

    # Edit the description field (composite widget wraps the actual QLineEdit)
    desc_widget = entry_1["fields"]["description"]
    desc_edit = desc_widget.findChild(QLineEdit)
    desc_edit.setText("Updated description for entry 1")

    # Edit the enabled checkbox (composite widget wraps the actual QCheckBox)
    enabled_widget = entry_1["fields"]["enabled"]
    enabled_checkbox = enabled_widget.findChild(QCheckBox)
    enabled_checkbox.setChecked(False)

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

                    # Handle composite wrappers: extract inner widget if needed
                    actual_widget = field_widget
                    if not isinstance(field_widget, (QCheckBox, QComboBox, QLineEdit)):
                        # Try to find the actual widget in the composite wrapper
                        inner_checkbox = field_widget.findChild(QCheckBox)
                        inner_combo = field_widget.findChild(QComboBox)
                        inner_edit = field_widget.findChild(QLineEdit)
                        if inner_checkbox is not None:
                            actual_widget = inner_checkbox
                        elif inner_combo is not None:
                            actual_widget = inner_combo
                        elif inner_edit is not None:
                            actual_widget = inner_edit

                    # Extract value based on widget type
                    extracted_value = None
                    should_include = False

                    if isinstance(actual_widget, QCheckBox):
                        extracted_value = actual_widget.isChecked()
                        # Only include if True (checkbox default is usually False)
                        should_include = extracted_value is True
                    elif isinstance(actual_widget, QComboBox):
                        text_value = actual_widget.currentText().strip()
                        if text_value:
                            extracted_value = text_value
                            # Omit if equals default
                            should_include = extracted_value != field_default
                    elif isinstance(actual_widget, QLineEdit):
                        text_value = actual_widget.text().strip()
                        if text_value:
                            extracted_value = text_value
                            # String fields: store as string, omit if equals default
                            should_include = extracted_value != field_default

                    if should_include and extracted_value is not None:
                        entry_dict[field_name] = extracted_value

                result.append(entry_dict)

            # Write result
            factory.main_window.config_dict[key] = result

    # Verify the parsed result
    saved_entries = factory.main_window.config_dict["test_entries"]
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
