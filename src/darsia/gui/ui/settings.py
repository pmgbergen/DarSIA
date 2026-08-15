"""Settings and input widget factory for DarSIA GUI."""

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QWidget,
)

from .file_dialog import FileDialogHelper
from .help import build_help_column
from .schema.dataclass_introspection import get_section_fields
from .schema.section_registry import get_required_sections


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
        """Get the relevant settings grouped by section.

        Parameters
        ----------
        action : str
            The workflow action (setup, calibration, analysis, etc.)
        checked_ids : list
            List of checked checkbox IDs

        Returns
        -------
        dict[str, list[dict]]
            Dictionary mapping section name to list of setting dictionaries.
            Preserves section order from get_required_sections().
        """
        settings_by_section = {}
        seen_sections = set()

        for checked_id in checked_ids:
            # Get the required sections for this checkbox
            try:
                sections = get_required_sections(action, checked_id)
            except ValueError as e:
                self.main_window.print_log(
                    f"Error resolving sections for {action}.{checked_id}: {e}"
                )
                continue

            if sections is None:
                self.main_window.print_log(
                    f"No settings mapping found for {action}.{checked_id}"
                )
                continue

            # For each required section, get its fields (avoiding duplicates)
            for section in sections:
                if section in seen_sections:
                    continue
                seen_sections.add(section)

                section_fields = get_section_fields(section)
                if section_fields is None:
                    self.main_window.print_log(
                        f"No dataclass found for section '{section}'"
                    )
                    continue

                settings_by_section[section] = section_fields

        return settings_by_section

    def _get_or_create_group_form(self, group_forms, parent_form, key, title):
        """Get or create a group QGroupBox with its own QFormLayout.

        key may be a str (top-level group name or standalone multi-row name) or a tuple
        (nested multi-row box, keyed by (outer_group_name, own_name) to avoid colliding
        with top-level keys).
        """
        if key not in group_forms:
            box = QGroupBox(title)
            form = QFormLayout(box)
            group_forms[key] = form
            parent_form.addRow(
                box
            )  # Spanning row in whichever form this box belongs to
        return group_forms[key]

    def build_tab_form(self, tab_form, settings_list, form_context=None):
        """Build rows in a QFormLayout, handling grouping via group_name metadata.

        Multi-row types (multi_file/multi_folder/path_map) are automatically grouped in
        their own titled QGroupBox using the field's display name, ensuring both the header
        and data rows render inside the same box.

        When a multi-row field carries an explicit group_name (e.g., "Input"), the behavior
        is nested: the outer "Input" group box is created (or reused) at the top level,
        and the multi-row field's own "Folders" box is created as a spanning row INSIDE
        the "Input" box's form (not as a sibling at the top level). This creates a
        visual hierarchy: Input > Folders + Format + Baseline.

        Parameters
        ----------
        tab_form : QFormLayout
            The target form layout to populate
        settings_list : list[dict]
            List of setting dicts from get_section_fields or similar
        form_context : dict, optional
            Context dict passed to create_setting_edit
                (contains "form" for multi_file/path_map)
        """
        group_forms = {}  # key (str | tuple) -> QFormLayout, first-occurrence cache

        MULTI_ROW_TYPES = {"multi_file", "multi_folder", "path_map"}

        for setting in settings_list:
            setting_type = setting["type"]
            explicit_group = setting.get("group_name")

            if setting_type in MULTI_ROW_TYPES:
                own_name = setting.get("name", setting["key"].rsplit(".", 1)[-1])
                auto_grouped = True
                if explicit_group:
                    # Multi-row field with explicit group: create nested structure
                    # Outer explicit-group box at top level (shared with scalar siblings)
                    outer_form = self._get_or_create_group_form(
                        group_forms, tab_form, explicit_group, explicit_group
                    )
                    # Inner box for this multi-row field, NESTED inside the outer box's form
                    target_form = self._get_or_create_group_form(
                        group_forms, outer_form, (explicit_group, own_name), own_name
                    )
                else:
                    # No explicit group: standalone top-level box
                    target_form = self._get_or_create_group_form(
                        group_forms, tab_form, own_name, own_name
                    )
            else:
                group_name = explicit_group
                auto_grouped = False
                target_form = (
                    self._get_or_create_group_form(
                        group_forms, tab_form, group_name, group_name
                    )
                    if group_name
                    else tab_form
                )

            # Resolve form_context against the actual destination form (KEY FIX)
            local_form_context = {"form": target_form}
            label_text, field_or_result = self.create_setting_edit(
                setting, local_form_context
            )

            # Handle type:"group" (Optional[dataclass]) fields — label_text is None,
            # field_or_result is dict with "widget"
            if (
                label_text is None
                and isinstance(field_or_result, dict)
                and "widget" in field_or_result
            ):
                # This is a group_box result dict
                group_widget = field_or_result.get("widget")
                if group_widget:
                    target_form.addRow(group_widget)  # Spanning row
                self.main_window.settings_inputs[setting["key"]] = field_or_result
                for sub_key, sub_widget in field_or_result.get(
                    "sub_inputs", {}
                ).items():
                    self.main_window.settings_inputs[sub_key] = sub_widget
            # Handle grouped scalar fields and auto-grouped multi-row fields
            elif auto_grouped and isinstance(field_or_result, dict):
                # Multi-row field that created its own result dict (backward compat)
                # Skip adding to form
                self.main_window.settings_inputs[setting["key"]] = field_or_result
            elif auto_grouped or explicit_group:
                # Blank the label for auto-grouped multi-row fields
                # (box title already shows name)
                row_label = "" if auto_grouped else label_text
                target_form.addRow(row_label, field_or_result)
                self.main_window.settings_inputs[setting["key"]] = field_or_result
            # Handle ungrouped scalar fields
            elif isinstance(field_or_result, dict):
                # This is a result dict from multi_file/path_map in fallback mode
                # (backward compat)
                # Skip adding to form; the field already created its own layout
                self.main_window.settings_inputs[setting["key"]] = field_or_result
            else:
                target_form.addRow(label_text, field_or_result)
                self.main_window.settings_inputs[setting["key"]] = field_or_result

    def wrap_setting_with_help(self, setting_container, setting_dict):
        """Wrap a setting container with a dedicated help button column."""
        wrapper = QWidget()
        wrapper_layout = QHBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(8)

        # Left: setting container with stretch
        wrapper_layout.addWidget(setting_container, stretch=1)

        # Right: fixed-width column for help button (or empty space)
        right_column = build_help_column(setting_dict)
        wrapper_layout.addWidget(right_column)

        return wrapper

    def create_setting_edit(self, setting_dict, form_context=None):
        """Create a new setting edit based on the setting type and options.

        Parameters
        ----------
        setting_dict : dict
            Setting configuration dictionary
        form_context : dict, optional
            Context dict with "form" (QFormLayout) for dynamic row insertion
                (multi_file/path_map).

        Returns
        -------
        tuple
            (label_text, field_widget) for scalar types,
            (None, result_dict) for type:'group' (result_dict carries widget and metadata),
            (label_text, edit_widget_or_dict) for file types.
        """
        setting_type = setting_dict["type"]
        options = setting_dict.get("options")
        free_value_types = ("int", "float", "string", "list")

        # If field has options, use dropdown regardless of type
        if options and setting_type in free_value_types:
            return self.create_dropdown_input(setting_dict)
        # Otherwise, dispatch based on type
        elif setting_type in free_value_types:
            return self.create_simple_input(setting_dict)
        elif setting_type == "bool":
            return self.create_bool_input(setting_dict)
        elif setting_type == "group":
            return self.create_group_input(setting_dict)
        elif setting_type == "fixed_list" and setting_dict.get("list_type") == "string":
            return self.create_fixed_list_string_input(setting_dict)
        elif setting_type == "file":
            display_name = setting_dict.get("name", setting_dict["key"])
            return self.file_dialog.create_file_chooser(
                display_name, None, False, setting_dict
            )
        elif setting_type == "folder":
            display_name = setting_dict.get("name", setting_dict["key"])
            return self.file_dialog.create_file_chooser(
                display_name, None, True, setting_dict
            )
        elif setting_type == "multi_file":
            return self.file_dialog.create_multi_file_input(
                setting_dict, form_context=form_context
            )
        elif setting_type == "multi_folder":
            return self.file_dialog.create_multi_file_input(
                setting_dict, is_directory=True, form_context=form_context
            )
        elif setting_type == "path_map":
            key_is_directory = setting_dict.get("key_is_directory", False)
            value_is_directory = setting_dict.get("value_is_directory", False)
            return self.file_dialog.create_path_map_input(
                setting_dict,
                key_is_directory=key_is_directory,
                value_is_directory=value_is_directory,
                form_context=form_context,
            )
        else:
            self.main_window.print_log(
                f"Setting type {setting_type} not supported yet, using simple input"
            )
            return self.create_simple_input(setting_dict)

    def create_simple_input(self, setting_dict):
        """Create a line edit input for numeric or string values.

        Returns (label_text, field_widget) where field_widget is a composite HBox:
        [setting_edit (stretch=1), type_label, help_button_or_spacer (fixed 40px)]
        """
        key = setting_dict["key"]
        display_name = setting_dict.get("name", key)
        value = self.get_value(self.main_window.config_dict, key)
        if value is None:
            value = setting_dict.get("default")

        setting_edit = QLineEdit()
        if value is not None:
            setting_edit.setText(str(value))

        # Build composite field widget with type label and help button
        field_widget = QWidget()
        field_layout = QHBoxLayout(field_widget)
        field_layout.setContentsMargins(0, 0, 0, 0)
        field_layout.setSpacing(4)

        # Type annotation label
        if setting_dict["type"] == "list":
            type_label = QLabel(
                f"({setting_dict['type']}, {setting_dict['list_type']})"
            )
        else:
            type_label = QLabel(f"({setting_dict['type']})")

        field_layout.addWidget(setting_edit, stretch=1)
        field_layout.addWidget(type_label)

        # Right column: help button or spacer (fixed 40px)
        field_layout.addWidget(build_help_column(setting_dict))

        return display_name, field_widget

    def create_bool_input(self, setting_dict):
        """Create a checkbox input for boolean values.

        Returns (label_text, field_widget) where field_widget is a composite HBox:
        [setting_checkbox, type_label, help_button_or_spacer (fixed 40px)]
        """
        key = setting_dict["key"]
        display_name = setting_dict.get("name", key)
        value = self.get_value(self.main_window.config_dict, key)
        if value is None:
            value = setting_dict.get("default")

        setting_checkbox = QCheckBox()
        if value is not None:
            setting_checkbox.setChecked(bool(value))

        # Build composite field widget with type label and help button
        field_widget = QWidget()
        field_layout = QHBoxLayout(field_widget)
        field_layout.setContentsMargins(0, 0, 0, 0)
        field_layout.setSpacing(4)

        field_layout.addWidget(setting_checkbox)
        field_layout.addWidget(QLabel("(bool)"))
        field_layout.addStretch()

        # Right column: help button or spacer (fixed 40px)
        field_layout.addWidget(build_help_column(setting_dict))

        return display_name, field_widget

    def create_dropdown_input(self, setting_dict):
        """Create a combobox input with predefined options.

        Returns (label_text, field_widget) where field_widget is a composite HBox:
        [setting_combo (stretch=1), help_button_or_spacer (fixed 40px)]
        """
        key = setting_dict["key"]
        display_name = setting_dict.get("name", key)
        value = self.get_value(self.main_window.config_dict, key)
        if value is None:
            value = setting_dict.get("default")

        options = setting_dict["options"]
        setting_combo = QComboBox()
        setting_combo.addItems([str(option) for option in options])

        if value is not None:
            value = str(value)
            index = setting_combo.findText(value)
            if index >= 0:
                setting_combo.setCurrentIndex(index)

        # Build composite field widget with help button
        field_widget = QWidget()
        field_layout = QHBoxLayout(field_widget)
        field_layout.setContentsMargins(0, 0, 0, 0)
        field_layout.setSpacing(4)

        field_layout.addWidget(setting_combo, stretch=1)

        # Right column: help button or spacer (fixed 40px)
        field_layout.addWidget(build_help_column(setting_dict))

        return display_name, field_widget

    def create_fixed_list_string_input(self, setting_dict):
        """Create a checkbox list for selecting from predefined options.

        Returns (label_text, field_widget) where field_widget is a composite HBox:
        [checkbox1, checkbox2, ..., help_button_or_spacer (fixed 40px)]
        """
        key = setting_dict["key"]
        display_name = setting_dict.get("name", key)
        values = self.get_value(self.main_window.config_dict, key)
        if values is None:
            values = setting_dict.get("default")

        options = setting_dict["options"]

        # Build composite field widget with checkboxes and help button
        field_widget = QWidget()
        field_layout = QHBoxLayout(field_widget)
        field_layout.setContentsMargins(0, 0, 0, 0)
        field_layout.setSpacing(4)

        check_boxes = []
        for option in options:
            check_box = QCheckBox(option)
            check_boxes.append(check_box)
            if values is not None:
                if option in values:
                    check_box.setChecked(True)
            field_layout.addWidget(check_box)

        field_layout.addStretch()

        # Right column: help button or spacer (fixed 40px)
        field_layout.addWidget(build_help_column(setting_dict))

        return display_name, field_widget

    def create_group_input(self, setting_dict):
        """Create a group box for an Optional[dataclass] field.

        If the field declares `metadata={"active_list_key": "<name>"}`, the group is
        checkable and its checked state is driven by `<section>.<name>` in the loaded
        config (falling back to "checked iff the sub-table is present" for configs
        predating that list). Fields without this metadata render as a plain
        non-checkable box — their data layer doesn't support enable/disable, so no
        checkbox is shown and nothing is written to the TOML for them.

        Returns the group_box widget (to be added as a spanning row in a QFormLayout)
        and a result dict with sub_inputs and optional checkbox/active_list_key/name.
        """
        key = setting_dict["key"]  # e.g. "corrections.resize"
        name = key.rsplit(".", 1)[-1]
        display_name = setting_dict.get("name", name)
        active_list_name = setting_dict.get("active_list_key")

        group_box = QGroupBox(display_name)
        result = {}

        if active_list_name is not None:
            section = key.rsplit(".", 1)[0]
            active_list_key = f"{section}.{active_list_name}"
            active_list = self.get_value(self.main_window.config_dict, active_list_key)
            if active_list is not None:
                is_active = name in active_list
            else:
                is_active = (
                    self.get_value(self.main_window.config_dict, key) is not None
                )

            group_box.setCheckable(True)
            group_box.setChecked(is_active)
            result.update(
                {
                    "checkbox": group_box,
                    "active_list_key": active_list_key,
                    "name": name,
                }
            )

        # Use QFormLayout for sub-fields (same structure as top-level tabs)
        group_form = QFormLayout(group_box)
        sub_inputs = {}
        for sub_setting in setting_dict["fields"]:
            label_text, field_widget = self.create_setting_edit(sub_setting)
            group_form.addRow(label_text, field_widget)
            sub_inputs[sub_setting["key"]] = field_widget
        result["sub_inputs"] = sub_inputs

        # Return (None, group_dict) where group_dict carries the group_box and metadata.
        # display_settings will check if the label is None to detect this case.
        # Mark as a group result so save_settings can identify it.
        result["widget"] = group_box
        result["is_group_result"] = True
        return None, result
