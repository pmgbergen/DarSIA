"""Settings and input widget factory for DarSIA GUI."""

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
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

        MULTI_ROW_TYPES = {
            "multi_file",
            "multi_folder",
            "path_map",
            "int_group_list",
            "int_list_map",
        }

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
            elif (
                auto_grouped
                and isinstance(field_or_result, dict)
                and "widget" in field_or_result
            ):
                # Multi-row field (form_context mode): form gets the header widget as the
                # spanning row; settings_inputs gets the row-tracking payload save_settings
                # expects.
                target_form.addRow("", field_or_result["widget"])
                if "path_map" in field_or_result:
                    self.main_window.settings_inputs[setting["key"]] = {
                        "path_map": True,
                        "rows": field_or_result["rows"],
                    }
                elif "int_group_list" in field_or_result:
                    self.main_window.settings_inputs[setting["key"]] = {
                        "int_group_list": True,
                        "rows": field_or_result["rows"],
                    }
                elif "int_list_map" in field_or_result:
                    self.main_window.settings_inputs[setting["key"]] = {
                        "int_list_map": True,
                        "flatten_in_section": field_or_result.get(
                            "flatten_in_section", False
                        ),
                        "section": field_or_result.get("section"),
                        "rows": field_or_result["rows"],
                    }
                else:
                    self.main_window.settings_inputs[setting["key"]] = field_or_result[
                        "rows"
                    ]
            elif auto_grouped and isinstance(field_or_result, dict):
                # Multi-row field that created its own result dict
                # (backward compat, non-form_context)
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
        elif setting_type == "int_group_list":
            return self.create_int_group_list_input(
                setting_dict, form_context=form_context
            )
        elif setting_type == "int_list_map":
            return self.create_int_list_map_input(
                setting_dict, form_context=form_context
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

    def create_int_group_list_input(self, setting_dict, form_context=None):
        """Create a multi-row editor for list[list[int]] fields (groups of label IDs).

        Each row is a single QLineEdit holding a comma/whitespace-separated list of ints,
        e.g. "3, 5, 8". Mirrors create_multi_file_input's form_context row-management
        pattern (add/remove via form.insertRow/removeRow). No fallback branch — form_context
        is always provided by the live app (display_settings).

        Returns (display_name, enriched_dict) where enriched_dict has "widget" for form
        insertion and "rows" (list of QLineEdit) for save_settings to parse.
        """
        key = setting_dict["key"]
        display_name = setting_dict.get("name", key)
        values = self.main_window.settings_factory.get_value(
            self.main_window.config_dict, key
        )
        if values is None:
            values = setting_dict.get("default")

        row_edits = []  # List of QLineEdit widgets for each group
        row_data_list = []  # Track row data (widget, remove_button)

        def refresh_remove_buttons():
            show_remove = len(row_data_list) > 1
            for row in row_data_list:
                row["remove_button"].setVisible(show_remove)

        add_button = QPushButton("Add group")

        if form_context:
            form = form_context["form"]

            # Build composite header widget:
            # [add_button (stretch=1)][help_button_or_spacer (fixed 40px)]
            header_widget = QWidget()
            header_layout = QHBoxLayout(header_widget)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(4)
            header_layout.addWidget(add_button, stretch=1)
            header_layout.addWidget(build_help_column(setting_dict))

            def add_row(initial_value=""):
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(4)

                text_edit = QLineEdit()
                placeholder = setting_dict.get("placeholder")
                if placeholder:
                    text_edit.setPlaceholderText(placeholder)
                if initial_value:
                    text_edit.setText(initial_value)
                remove_button = QPushButton("Remove")
                remove_button.setMaximumWidth(80)

                def remove():
                    self.file_dialog._remove_form_row(
                        form,
                        row_widget,
                        row_data,
                        row_data_list,
                        text_edit,
                        row_edits,
                        refresh_remove_buttons,
                    )

                remove_button.clicked.connect(remove)

                row_layout.addWidget(text_edit, stretch=1)
                row_layout.addWidget(remove_button)

                # Find the correct insertion index: after the header_widget header row
                header_idx, _ = form.getWidgetPosition(header_widget)
                if row_data_list:
                    last_idx, _ = form.getWidgetPosition(row_data_list[-1]["widget"])
                    insert_idx = last_idx + 1
                else:
                    # Insert right after header row
                    insert_idx = header_idx + 1

                form.insertRow(insert_idx, "", row_widget)

                row_data = {
                    "widget": row_widget,
                    "remove_button": remove_button,
                }
                row_data_list.append(row_data)
                row_edits.append(text_edit)
                refresh_remove_buttons()

            # Connect add_button to add_row closure
            add_button.clicked.connect(lambda: add_row())

            # Defer pre-fill until after header row is added to form
            from PySide6.QtCore import QTimer

            def deferred_prefill():
                if isinstance(values, list) and values:
                    for group in values:
                        # Join list of ints with ", "
                        group_str = ", ".join(str(x) for x in group)
                        add_row(group_str)
                else:
                    add_row("")

            QTimer.singleShot(0, deferred_prefill)

            # Return enriched dict: widget for form insertion, rows for save_settings
            return display_name, {
                "widget": header_widget,
                "int_group_list": True,
                "rows": row_edits,
            }

        else:
            # Fallback (should not be reached in the current app, but kept for compatibility)
            setting_container = QWidget()
            setting_layout = QVBoxLayout(setting_container)
            setting_layout.setContentsMargins(0, 0, 0, 0)

            setting_layout.addWidget(add_button)

            rows_container = QWidget()
            rows_layout = QVBoxLayout(rows_container)
            rows_layout.setContentsMargins(0, 0, 0, 0)
            setting_layout.addWidget(rows_container)

            def add_row(initial_value=""):
                row_container = QWidget()
                row_layout = QHBoxLayout(row_container)
                row_layout.setContentsMargins(0, 0, 0, 0)

                text_edit = QLineEdit()
                placeholder = setting_dict.get("placeholder")
                if placeholder:
                    text_edit.setPlaceholderText(placeholder)
                if initial_value:
                    text_edit.setText(initial_value)
                remove_button = QPushButton("Remove")
                remove_button.setMaximumWidth(80)

                def remove():
                    row_container.deleteLater()
                    if row_data in row_data_list:
                        row_data_list.remove(row_data)
                    if text_edit in row_edits:
                        row_edits.remove(text_edit)
                    refresh_remove_buttons()

                remove_button.clicked.connect(remove)

                row_layout.addWidget(text_edit)
                row_layout.addWidget(remove_button)
                rows_layout.addWidget(row_container)

                row_data = {
                    "container": row_container,
                    "remove_button": remove_button,
                }
                row_data_list.append(row_data)
                row_edits.append(text_edit)
                refresh_remove_buttons()

            add_button.clicked.connect(lambda: add_row())

            if isinstance(values, list) and values:
                for group in values:
                    group_str = ", ".join(str(x) for x in group)
                    add_row(group_str)
            else:
                add_row("")

            return display_name, {"int_group_list": True, "rows": row_edits}

    def create_int_list_map_input(self, setting_dict, form_context=None):
        """Create a multi-row editor for dict[int, list[int]] fields
        (int key → int-list value).

        Each row has two QLineEdits: a narrow key edit (facies ID, int) and a value edit
        (comma/whitespace-separated int labels, e.g. "3, 5, 8"). If flatten_in_section=True,
        reads from and writes to the parent section's sub-tables (e.g., [facies.0], [facies.1])
        rather than a nested field key (to maintain compatibility with existing TOML layouts).
        Mirrors create_int_group_list_input's form_context row-management pattern (add/remove
        via form.insertRow/removeRow).

        Returns (display_name, enriched_dict) where enriched_dict has "widget" for form
        insertion and "rows" (list of (key_edit, value_edit) tuples) for save_settings
        to parse.
        """
        key = setting_dict["key"]
        display_name = setting_dict.get("name", key)
        flatten = setting_dict.get("flatten_in_section", False)

        # Fetch values: either from the field's own key (plain dict[int, list[int]])
        # or from the flattened section root (if flatten_in_section=True).
        if flatten:
            section = key.rsplit(".", 1)[0]
            section_dict = self.get_value(self.main_window.config_dict, section)
            values = {}
            if section_dict and isinstance(section_dict, dict):
                for k, v in section_dict.items():
                    if isinstance(v, dict) and "labels" in v:
                        try:
                            values[int(k)] = v["labels"]
                        except (ValueError, TypeError):
                            pass
        else:
            values = self.main_window.settings_factory.get_value(
                self.main_window.config_dict, key
            )
            if values is None:
                values = setting_dict.get("default")

        row_pairs = []  # List of (key_edit, value_edit) tuples
        row_data_list = []  # Track row data (widget, remove_button)

        def refresh_remove_buttons():
            show_remove = len(row_data_list) > 1
            for row in row_data_list:
                row["remove_button"].setVisible(show_remove)

        add_button = QPushButton("Add row")

        if form_context:
            form = form_context["form"]

            # Build composite header widget:
            # [add_button (stretch=1)][help_button_or_spacer (fixed 40px)]
            header_widget = QWidget()
            header_layout = QHBoxLayout(header_widget)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(4)
            header_layout.addWidget(add_button, stretch=1)
            header_layout.addWidget(build_help_column(setting_dict))

            def add_row(initial_key="", initial_value=""):
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(4)

                # Key column (facies ID)
                key_edit = QLineEdit()
                key_edit.setPlaceholderText("Facies ID")
                key_edit.setMaximumWidth(100)
                if initial_key != "":
                    key_edit.setText(str(initial_key))

                # Value column (comma/space-separated labels)
                value_edit = QLineEdit()
                placeholder = setting_dict.get("placeholder")
                if placeholder:
                    value_edit.setPlaceholderText(placeholder)
                if initial_value:
                    value_edit.setText(initial_value)

                # Remove button
                remove_button = QPushButton("Remove")
                remove_button.setMaximumWidth(80)

                def remove():
                    self.file_dialog._remove_form_row(
                        form,
                        row_widget,
                        row_data,
                        row_data_list,
                        (key_edit, value_edit),
                        row_pairs,
                        refresh_remove_buttons,
                    )

                remove_button.clicked.connect(remove)

                row_layout.addWidget(key_edit)
                row_layout.addWidget(value_edit, stretch=1)
                row_layout.addWidget(remove_button)

                # Find the correct insertion index: after the header_widget header row
                header_idx, _ = form.getWidgetPosition(header_widget)
                if row_data_list:
                    last_idx, _ = form.getWidgetPosition(row_data_list[-1]["widget"])
                    insert_idx = last_idx + 1
                else:
                    # Insert right after header row
                    insert_idx = header_idx + 1

                form.insertRow(insert_idx, "", row_widget)

                row_data = {
                    "widget": row_widget,
                    "remove_button": remove_button,
                }
                row_data_list.append(row_data)
                row_pairs.append((key_edit, value_edit))
                refresh_remove_buttons()

            # Connect add_button to add_row closure
            add_button.clicked.connect(lambda: add_row())

            # Defer pre-fill until after header row is added to form
            from PySide6.QtCore import QTimer

            def deferred_prefill():
                if isinstance(values, dict) and values:
                    for k, v in values.items():
                        # Join list of ints with ", "
                        value_str = ", ".join(str(x) for x in v)
                        add_row(k, value_str)
                else:
                    add_row("", "")

            QTimer.singleShot(0, deferred_prefill)

            # Return enriched dict: widget for form insertion, rows for save_settings
            return display_name, {
                "widget": header_widget,
                "int_list_map": True,
                "flatten_in_section": flatten,
                "section": section if flatten else None,
                "rows": row_pairs,
            }

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
