"""Unit test for scalar (single-value) widget save/sync with composite wrappers.

Regression test for the bug where scalar field values (data.baseline, protocols.injection,
etc.) would revert to their previously-loaded value on save instead of using the edited
value the user entered in the GUI. Root cause: composite wrapper QWidgets (containing
the real control + type label + help button) were not being unwrapped before value
extraction in _sync_settings_inputs_to_config_dict.
"""

import pytest
import toml
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QLineEdit

from darsia.gui.ui.main_window import MainWindow, _unwrap_composite_widget


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for the entire test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def temp_config(tmp_path):
    """Create a minimal test TOML config with scalar fields."""
    config = {
        "data": {
            "folders": ["D:/test/folder"],
            "results": "results_dir",
            "baseline": "baseline.jpg",
        },
        "protocols": {
            "injection": "injection.csv",
            "imaging_mode": "ctime",
        },
        "rig": {
            "dim": 2,
        },
    }
    config_path = tmp_path / "test_config.toml"
    with open(config_path, "w") as f:
        toml.dump(config, f)
    return config_path


class TestUnwrapCompositeWidget:
    """Test the _unwrap_composite_widget helper function."""

    def test_unwrap_bare_qlineedit_unchanged(self):
        """A bare QLineEdit should be returned unchanged."""
        edit = QLineEdit()
        result = _unwrap_composite_widget(edit)
        assert result is edit

    def test_unwrap_bare_qcombobox_unchanged(self):
        """A bare QComboBox should be returned unchanged."""
        combo = QComboBox()
        result = _unwrap_composite_widget(combo)
        assert result is combo

    def test_unwrap_bare_qcheckbox_unchanged(self):
        """A bare QCheckBox should be returned unchanged."""
        checkbox = QCheckBox()
        result = _unwrap_composite_widget(checkbox)
        assert result is checkbox

    def test_unwrap_non_widget_unchanged(self):
        """Non-widget values (dict, list, etc.) should be returned unchanged."""
        test_dict = {"key": "value"}
        result = _unwrap_composite_widget(test_dict)
        assert result is test_dict

        test_list = [1, 2, 3]
        result = _unwrap_composite_widget(test_list)
        assert result is test_list

    def test_unwrap_composite_qlineedit(self, qapp):
        """Unwrap a composite wrapper containing a QLineEdit."""
        from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

        # Build a composite widget exactly as create_simple_input does
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        edit.setText("test_value")
        type_label = QLabel("(str)")
        layout.addWidget(edit, stretch=1)
        layout.addWidget(type_label)

        # Unwrap should find and return the QLineEdit
        result = _unwrap_composite_widget(wrapper)
        assert result is edit
        assert result.text() == "test_value"

    def test_unwrap_composite_qcombobox(self, qapp):
        """Unwrap a composite wrapper containing a QComboBox."""
        from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        combo = QComboBox()
        combo.addItems(["option1", "option2"])
        combo.setCurrentText("option2")
        type_label = QLabel("(str)")
        layout.addWidget(combo, stretch=1)
        layout.addWidget(type_label)

        result = _unwrap_composite_widget(wrapper)
        assert result is combo
        assert result.currentText() == "option2"

    def test_unwrap_composite_qcheckbox(self, qapp):
        """Unwrap a composite wrapper containing a QCheckBox."""
        from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        checkbox = QCheckBox()
        checkbox.setChecked(True)
        type_label = QLabel("(bool)")
        layout.addWidget(checkbox)
        layout.addWidget(type_label)
        layout.addStretch()

        result = _unwrap_composite_widget(wrapper)
        assert result is checkbox
        assert result.isChecked()


class TestScalarWidgetSaveRoundtrip:
    """Test that scalar widget edits are saved correctly, not reverted."""

    def test_string_field_roundtrip(self, qapp, temp_config):
        """Edit a string field, sync, and verify config reflects the new value."""
        window = MainWindow()
        window.load_config(str(temp_config))

        # Manually access and edit the data.baseline widget as if the user did via GUI
        data_baseline_key = "data.baseline"
        if data_baseline_key in window.settings_inputs:
            widget_or_wrapper = window.settings_inputs[data_baseline_key]
            # Unwrap if needed
            widget = _unwrap_composite_widget(widget_or_wrapper)
            if isinstance(widget, QLineEdit):
                new_value = "/new/path/to/baseline.jpg"
                widget.setText(new_value)

                # Sync the edit to config_dict
                window._sync_settings_inputs_to_config_dict()

                # Assert config_dict was updated
                from darsia.gui.ui.settings import SettingsFactory

                factory = SettingsFactory(window)
                actual_value = factory.get_value(window.config_dict, data_baseline_key)
                assert actual_value == new_value, (
                    f"Expected {new_value}, but config_dict has {actual_value}. "
                    f"Bug: edited value not saved!"
                )

    def test_dropdown_field_roundtrip(self, qapp, temp_config):
        """Edit a dropdown field, sync, and verify config reflects the new value."""
        window = MainWindow()
        window.load_config(str(temp_config))

        imaging_mode_key = "protocols.imaging_mode"
        if imaging_mode_key in window.settings_inputs:
            widget_or_wrapper = window.settings_inputs[imaging_mode_key]
            widget = _unwrap_composite_widget(widget_or_wrapper)
            if isinstance(widget, QComboBox):
                # Ensure the combo has options; if not, skip this part of the test
                if widget.count() > 1:
                    new_option = widget.itemText(1) if widget.count() > 1 else "ctime"
                    widget.setCurrentText(new_option)

                    window._sync_settings_inputs_to_config_dict()

                    from darsia.gui.ui.settings import SettingsFactory

                    factory = SettingsFactory(window)
                    actual_value = factory.get_value(
                        window.config_dict, imaging_mode_key
                    )
                    assert actual_value == new_option, (
                        f"Expected {new_option}, but config_dict has {actual_value}. "
                        f"Bug: edited dropdown value not saved!"
                    )
