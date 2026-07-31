"""Settings and input widget factory for DarSIA GUI."""

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QWidget,
)

from .file_dialog import FileDialogHelper
from .help import HelpButton


class SettingsFactory:
    """Factory for creating settings input widgets and managing settings."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.file_dialog = FileDialogHelper(main_window)

    def get_value(self, dictionary, key_path):
        """Get a value from nested dict using dot notation (e.g., 'a.b.c')"""
        keys = key_path.split(".")
        value = dictionary
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def set_value(self, dictionary, key_path, value):
        """Set a value in nested dict using dot notation, creating keys as needed"""
        keys = key_path.split(".")
        current = dictionary
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def get_relevant_settings(self, action, checked_ids):
        """Get the relevant settings based on the action and checked checkboxes.

        Parameters
        ----------
        action : str
            The workflow action (setup, calibration, analysis, etc.)
        checked_ids : list
            List of checked checkbox IDs

        Returns
        -------
        list
            List of setting dictionaries with name and type information.
        """
        settings = []
        for checked_id in checked_ids:
            try:
                settings += self.main_window.settings_mapping["settings"][action][
                    checked_id
                ]["content"]
            except KeyError:
                self.main_window.print_log(
                    f"No settings found for {action} with id {checked_id}"
                )
        return settings

    def wrap_setting_with_help(self, setting_container, setting_dict):
        """Wrap a setting container with a dedicated help button column."""
        help_text = setting_dict.get("help")
        link_url = setting_dict.get("link")

        wrapper = QWidget()
        wrapper_layout = QHBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(8)

        # Left: setting container with stretch
        wrapper_layout.addWidget(setting_container, stretch=1)

        # Right: fixed-width column for help button (or empty space)
        right_column = QWidget()
        right_layout = QHBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)

        if help_text:
            help_button = HelpButton(help_text, link_url)
            right_layout.addWidget(help_button)
        else:
            right_layout.addStretch()

        right_column.setFixedWidth(40)
        wrapper_layout.addWidget(right_column)

        return wrapper

    def create_setting_edit(self, setting_dict):
        """Create a new setting edit based on the setting type."""
        setting_type = setting_dict["type"]

        if setting_type == "int":
            return self.create_simple_input(setting_dict)
        elif setting_type == "float":
            return self.create_simple_input(setting_dict)
        elif setting_type == "string":
            return self.create_string_input(setting_dict)
        elif setting_type == "bool":
            return self.create_bool_input(setting_dict)
        elif setting_type == "fixed_list" and setting_dict["list_type"] == "string":
            return self.create_fixed_list_string_input(setting_dict)
        elif setting_type == "file":
            return self.file_dialog.create_file_chooser(
                setting_dict["key"], None, False
            )
        elif setting_type == "folder":
            return self.file_dialog.create_file_chooser(setting_dict["key"], None, True)
        elif setting_type == "multi_file":
            return self.file_dialog.create_multi_file_input(setting_dict)
        else:
            self.main_window.print_log(
                f"Setting type {setting_type} not supported yet, using simple input"
            )
            return self.create_simple_input(setting_dict)

    def create_simple_input(self, setting_dict):
        """Create a line edit input for numeric or string values."""
        setting = setting_dict["key"]

        value = self.get_value(self.main_window.config_dict, setting)
        setting_container = QWidget()
        setting_layout = QHBoxLayout(setting_container)
        setting_label = QLabel(setting)
        setting_edit = QLineEdit()
        if value is not None:
            setting_edit.setText(str(value))
        if setting_dict["type"] == "list":
            type_label = QLabel(
                f"({setting_dict['type']}, {setting_dict['list_type']})"
            )
        else:
            type_label = QLabel(f"({setting_dict['type']})")
        setting_layout.addWidget(setting_label)
        setting_layout.addWidget(setting_edit)
        setting_layout.addWidget(type_label)
        return setting_container, setting_edit

    def create_bool_input(self, setting_dict):
        """Create a checkbox input for boolean values."""
        setting = setting_dict["key"]
        value = self.get_value(self.main_window.config_dict, setting)
        setting_container = QWidget()
        setting_layout = QHBoxLayout(setting_container)
        setting_label = QLabel(setting)
        setting_checkbox = QCheckBox()
        if value is not None:
            setting_checkbox.setChecked(bool(value))
        setting_layout.addWidget(setting_label)
        setting_layout.addWidget(setting_checkbox)
        setting_layout.addWidget(QLabel("(bool)"))
        return setting_container, setting_checkbox

    def create_string_input(self, setting_dict):
        """Create a combobox input with predefined string options."""
        setting = setting_dict["key"]
        value = self.get_value(self.main_window.config_dict, setting)
        setting_container = QWidget()
        setting_layout = QHBoxLayout(setting_container)
        setting_label = QLabel(setting)
        options = setting_dict["options"]
        setting_combo = QComboBox()
        setting_combo.addItems([str(option) for option in options])

        if value is not None:
            value = str(value)
            index = setting_combo.findText(value)
            if index >= 0:
                setting_combo.setCurrentIndex(index)

        setting_layout.addWidget(setting_label)
        setting_layout.addWidget(setting_combo)

        return setting_container, setting_combo

    def create_fixed_list_string_input(self, setting_dict):
        """Create a checkbox list for selecting from predefined options."""
        setting = setting_dict["key"]
        values = self.get_value(self.main_window.config_dict, setting)
        setting_container = QWidget()
        setting_layout = QHBoxLayout(setting_container)
        setting_label = QLabel(setting)
        setting_layout.addWidget(setting_label)
        options = setting_dict["options"]
        check_boxes = []
        for option in options:
            check_box = QCheckBox(option)
            check_boxes.append(check_box)
            if values is not None:
                if option in values:
                    check_box.setChecked(True)
            setting_layout.addWidget(check_box)
        return setting_container, check_boxes
