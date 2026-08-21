"""Unit test for scalar (single-value) widget save/sync with composite wrappers.

Regression test for the bug where scalar field values (data.baseline, protocols.injection,
etc.) would revert to their previously-loaded value on save instead of using the edited
value the user entered in the GUI. Root cause: composite wrapper QWidgets (containing
the real control + type label + help button) were not being unwrapped before value
extraction in _sync_settings_inputs_to_config_dict.
"""

import pytest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QWidget,
)
from unittest.mock import MagicMock

from darsia.gui.ui.settings import SettingsFactory, unwrap_composite_widget


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for the entire test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestUnwrapCompositeWidget:
    """Test the unwrap_composite_widget helper function."""

    def test_unwrap_bare_qlineedit_unchanged(self):
        """A bare QLineEdit should be returned unchanged."""
        edit = QLineEdit()
        result = unwrap_composite_widget(edit)
        assert result is edit

    def test_unwrap_bare_qcombobox_unchanged(self):
        """A bare QComboBox should be returned unchanged."""
        combo = QComboBox()
        result = unwrap_composite_widget(combo)
        assert result is combo

    def test_unwrap_bare_qcheckbox_unchanged(self):
        """A bare QCheckBox should be returned unchanged."""
        checkbox = QCheckBox()
        result = unwrap_composite_widget(checkbox)
        assert result is checkbox

    def test_unwrap_non_widget_unchanged(self):
        """Non-widget values (dict, list, etc.) should be returned unchanged."""
        test_dict = {"key": "value"}
        result = unwrap_composite_widget(test_dict)
        assert result is test_dict

        test_list = [1, 2, 3]
        result = unwrap_composite_widget(test_list)
        assert result is test_list

    def test_unwrap_composite_qlineedit(self, qapp):
        """Unwrap a composite wrapper containing a QLineEdit."""
        # Build a composite widget exactly as create_simple_input does
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        edit.setText("test_value")
        type_label = QLabel("(str)")
        layout.addWidget(edit, stretch=1)
        layout.addWidget(type_label)

        # Store reference to the real control for unwrapping (as production code does)
        wrapper.setProperty("value_widget", edit)

        # Unwrap should find and return the QLineEdit
        result = unwrap_composite_widget(wrapper)
        assert result is edit
        assert result.text() == "test_value"

    def test_unwrap_composite_qcombobox(self, qapp):
        """Unwrap a composite wrapper containing a QComboBox."""
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        combo = QComboBox()
        combo.addItems(["option1", "option2"])
        combo.setCurrentText("option2")
        type_label = QLabel("(str)")
        layout.addWidget(combo, stretch=1)
        layout.addWidget(type_label)

        # Store reference to the real control (as production code does)
        wrapper.setProperty("value_widget", combo)

        result = unwrap_composite_widget(wrapper)
        assert result is combo
        assert result.currentText() == "option2"

    def test_unwrap_composite_qcheckbox(self, qapp):
        """Unwrap a composite wrapper containing a QCheckBox."""
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        checkbox = QCheckBox()
        checkbox.setChecked(True)
        type_label = QLabel("(bool)")
        layout.addWidget(checkbox)
        layout.addWidget(type_label)
        layout.addStretch()

        # Store reference to the real control (as production code does)
        wrapper.setProperty("value_widget", checkbox)

        result = unwrap_composite_widget(wrapper)
        assert result is checkbox
        assert result.isChecked()


class TestScalarWidgetSaveRoundtrip:
    """Test that scalar widget edits are saved correctly, not reverted."""

    def test_string_field_roundtrip(self, qapp):
        """Edit a string field, sync, and verify config reflects the new value."""
        # Build config_dict and settings_inputs manually (MagicMock pattern)
        config_dict = {"data": {}, "protocols": {}}
        settings_inputs = {}

        # Create a composite wrapper widget containing a QLineEdit, as production code does
        edit = QLineEdit()
        edit.setText("/new/path/to/baseline.jpg")
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, stretch=1)
        layout.addWidget(QLabel("(str)"))
        wrapper.setProperty("value_widget", edit)

        settings_inputs["data.baseline"] = wrapper

        # Create mock main_window with the two dicts that _sync_settings_inputs_to_config_dict needs
        mock_main_window = MagicMock(config_dict=config_dict, settings_inputs=settings_inputs)

        # Drive SettingsFactory to sync the edited value
        factory = SettingsFactory(mock_main_window)
        factory._sync_settings_inputs_to_config_dict()

        # Assert config_dict was updated with the edited value
        actual_value = factory.get_value(config_dict, "data.baseline")
        assert actual_value == "/new/path/to/baseline.jpg", (
            f"Expected '/new/path/to/baseline.jpg', but config_dict has {actual_value}. "
            f"Bug: edited value not saved!"
        )

    def test_dropdown_field_roundtrip(self, qapp):
        """Edit a dropdown field, sync, and verify config reflects the new value."""
        # Build config_dict and settings_inputs manually (MagicMock pattern)
        config_dict = {"protocols": {}}
        settings_inputs = {}

        # Create a composite wrapper widget containing a QComboBox, as production code does
        combo = QComboBox()
        combo.addItems(["exif", "ctime"])
        combo.setCurrentText("ctime")
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(combo, stretch=1)
        layout.addWidget(QLabel("(str)"))
        wrapper.setProperty("value_widget", combo)

        settings_inputs["protocols.imaging_mode"] = wrapper

        # Create mock main_window with the two dicts that _sync_settings_inputs_to_config_dict needs
        mock_main_window = MagicMock(config_dict=config_dict, settings_inputs=settings_inputs)

        # Drive SettingsFactory to sync the edited value
        factory = SettingsFactory(mock_main_window)
        factory._sync_settings_inputs_to_config_dict()

        # Assert config_dict was updated with the edited value
        actual_value = factory.get_value(config_dict, "protocols.imaging_mode")
        assert actual_value == "ctime", (
            f"Expected 'ctime', but config_dict has {actual_value}. "
            f"Bug: edited dropdown value not saved!"
        )
