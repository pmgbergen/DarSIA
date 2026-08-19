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
from .schema.dataclass_introspection import ALL_SECTIONS, get_section_fields
from .schema.section_registry import get_required_sections


class SettingsFactory:
    """Factory for creating settings input widgets and managing settings."""

    # Multi-row widget types that need special handling (auto-grouped into their own box)
    MULTI_ROW_TYPES = {
        "multi_file",
        "multi_folder",
        "path_map",
        "int_group_list",
        "int_list_map",
        "time_interval_map",
        "time_window_map",
        "time_data_map",
        "path_data_map",
        "format_map",
        "format_key_list",
        "registry_key_list",
    }

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

    def get_all_settings(self):
        """Get all fixed-schema sections for full-config view.

        Returns
        -------
        dict[str, list[dict]]
            Dictionary mapping each section name to its field list,
            in the canonical order defined by ALL_SECTIONS.
        """
        settings_by_section = {}

        for section in ALL_SECTIONS:
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

        for setting in settings_list:
            setting_type = setting["type"]
            explicit_group = setting.get("group_name")

            if setting_type in self.MULTI_ROW_TYPES:
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
                elif "time_interval_map" in field_or_result:
                    self.main_window.settings_inputs[setting["key"]] = {
                        "time_interval_map": True,
                        "rows": field_or_result["rows"],
                    }
                elif "time_window_map" in field_or_result:
                    self.main_window.settings_inputs[setting["key"]] = {
                        "time_window_map": True,
                        "rows": field_or_result["rows"],
                    }
                elif "time_data_map" in field_or_result:
                    self.main_window.settings_inputs[setting["key"]] = {
                        "time_data_map": True,
                        "rows": field_or_result["rows"],
                    }
                elif "path_data_map" in field_or_result:
                    self.main_window.settings_inputs[setting["key"]] = {
                        "path_data_map": True,
                        "rows": field_or_result["rows"],
                    }
                elif "format_map" in field_or_result:
                    self.main_window.settings_inputs[setting["key"]] = {
                        "format_map": True,
                        "rows": field_or_result["rows"],
                    }
                elif "registry_key_list" in field_or_result:
                    self.main_window.settings_inputs[setting["key"]] = {
                        "registry_key_list": True,
                        "rows": field_or_result["rows"],
                    }
                elif "format_key_list" in field_or_result:
                    result_dict = {
                        "format_key_list": True,
                        "rows": field_or_result["rows"],
                    }
                    if "max_rows" in field_or_result:
                        result_dict["max_rows"] = field_or_result["max_rows"]
                    self.main_window.settings_inputs[setting["key"]] = result_dict
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
        elif setting_type == "time_interval_map":
            return self.create_time_interval_map_input(
                setting_dict, form_context=form_context
            )
        elif setting_type == "time_window_map":
            return self.create_time_window_map_input(
                setting_dict, form_context=form_context
            )
        elif setting_type == "time_data_map":
            return self.create_time_data_map_input(
                setting_dict, form_context=form_context
            )
        elif setting_type == "path_data_map":
            return self.create_path_data_map_input(
                setting_dict, form_context=form_context
            )
        elif setting_type == "format_map":
            return self.create_format_map_input(setting_dict, form_context=form_context)
        elif setting_type == "registry_key_list":
            return self.create_registry_key_list_input(
                setting_dict, form_context=form_context
            )
        elif setting_type == "format_key_list":
            return self.create_format_key_list_input(
                setting_dict, form_context=form_context
            )
        else:
            self.main_window.print_log(
                f"Setting type {setting_type} not supported yet, using simple input"
            )
            return self.create_simple_input(setting_dict)

    def create_time_interval_map_input(self, setting_dict, form_context=None):
        """Create a hardcoded dict[str, TimeInterval] editor with name + start/end/num/tol
        per row.

        Each row is: name (entry name) + start (HH:MM:SS) + end (HH:MM:SS) +
        num (image count) + tol (tolerance HH:MM:SS) + Remove button.

        Returns (display_name, enriched_dict) where enriched_dict has "widget"
        for form insertion and "rows" (list of 5-tuples of QLineEdits) for
        save_settings to parse.
        """
        key = setting_dict["key"]
        display_name = setting_dict.get("name", key)
        values = self.get_value(self.main_window.config_dict, key)
        if values is None:
            values = {}

        row_data_list = (
            []
        )  # Track (widget, remove_button, (name, start, end, num, tol))
        # List of 5-tuples: (name_edit, start_edit, end_edit, num_edit, tol_edit)
        row_edits = []

        def refresh_remove_buttons():
            show_remove = len(row_data_list) > 1
            for row_data in row_data_list:
                row_data["remove_button"].setVisible(show_remove)

        add_button = QPushButton("Add interval")

        if form_context:
            form = form_context["form"]

            # Build header widget
            header_widget = QWidget()
            header_layout = QHBoxLayout(header_widget)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(4)
            header_layout.addWidget(add_button, stretch=1)
            header_layout.addWidget(build_help_column(setting_dict))

            # Add header row to form
            form.addRow("", header_widget)

            def add_row(entry_name="", entry_data=None):
                """Add a row with name + start/end/num/tol fields."""
                if entry_data is None:
                    entry_data = {}

                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(4, 4, 4, 4)
                row_layout.setSpacing(4)
                row_widget.setMinimumHeight(28)

                # Name field
                name_edit = QLineEdit()
                name_edit.setPlaceholderText("Entry name")
                name_edit.setMaximumWidth(100)
                if entry_name:
                    name_edit.setText(str(entry_name))
                row_layout.addWidget(name_edit, 0)

                # Start field
                start_edit = QLineEdit()
                start_edit.setPlaceholderText("Start HH:MM:SS")
                start_edit.setMaximumWidth(100)
                if "start" in entry_data:
                    start_edit.setText(str(entry_data["start"]))
                row_layout.addWidget(start_edit, 0)

                # End field
                end_edit = QLineEdit()
                end_edit.setPlaceholderText("End HH:MM:SS")
                end_edit.setMaximumWidth(100)
                if "end" in entry_data:
                    end_edit.setText(str(entry_data["end"]))
                row_layout.addWidget(end_edit, 0)

                # Num field
                num_edit = QLineEdit()
                num_edit.setPlaceholderText("Num images")
                num_edit.setMaximumWidth(80)
                if "num" in entry_data:
                    num_edit.setText(str(entry_data["num"]))
                row_layout.addWidget(num_edit, 0)

                # Tolerance field
                tol_edit = QLineEdit()
                tol_edit.setPlaceholderText("Tol HH:MM:SS")
                tol_edit.setMaximumWidth(100)
                if "tol" in entry_data:
                    tol_edit.setText(str(entry_data["tol"]))
                row_layout.addWidget(tol_edit, 0)

                # Remove button
                remove_button = QPushButton("Remove")
                remove_button.setMaximumWidth(80)

                def remove():
                    row_idx, _ = form.getWidgetPosition(row_widget)
                    form.removeRow(row_idx)
                    if row_data in row_data_list:
                        row_data_list.remove(row_data)
                    if edits in row_edits:
                        row_edits.remove(edits)
                    refresh_remove_buttons()

                remove_button.clicked.connect(remove)
                row_layout.addWidget(remove_button, 0)

                # Add row to form
                header_idx, _ = form.getWidgetPosition(header_widget)
                if row_data_list:
                    last_idx, _ = form.getWidgetPosition(row_data_list[-1]["widget"])
                    insert_idx = last_idx + 1
                else:
                    insert_idx = header_idx + 1

                form.insertRow(insert_idx, "", row_widget)

                edits = (name_edit, start_edit, end_edit, num_edit, tol_edit)
                row_data = {
                    "widget": row_widget,
                    "remove_button": remove_button,
                }
                row_data_list.append(row_data)
                row_edits.append(edits)
                refresh_remove_buttons()

            # Connect add button
            add_button.clicked.connect(lambda: add_row())

            # Prefill existing rows
            for entry_name, entry_data in (values or {}).items():
                add_row(entry_name, entry_data if isinstance(entry_data, dict) else {})

            # Return enriched dict
            return display_name, {
                "widget": header_widget,
                "time_interval_map": True,
                "rows": row_edits,
            }

        else:
            # Fallback (should not be reached in current app)
            return display_name, {"time_interval_map": True, "rows": []}

    def create_time_window_map_input(self, setting_dict, form_context=None):
        """Create a hardcoded dict[str, TimeWindow] editor with name + start/end per row.

        Each row is: name (entry name) + start (HH:MM:SS) + end (HH:MM:SS) + Remove button.

        Returns (display_name, enriched_dict) with "widget" and "rows" (list of
        3-tuples of QLineEdits: (name_edit, start_edit, end_edit)).
        """
        key = setting_dict["key"]
        display_name = setting_dict.get("name", key)
        values = self.get_value(self.main_window.config_dict, key)
        if values is None:
            values = {}

        row_data_list = []
        row_edits = []  # List of 3-tuples: (name_edit, start_edit, end_edit)

        def refresh_remove_buttons():
            show_remove = len(row_data_list) > 1
            for row_data in row_data_list:
                row_data["remove_button"].setVisible(show_remove)

        add_button = QPushButton("Add window")

        if form_context:
            form = form_context["form"]

            # Build header widget
            header_widget = QWidget()
            header_layout = QHBoxLayout(header_widget)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(4)
            header_layout.addWidget(add_button, stretch=1)
            header_layout.addWidget(build_help_column(setting_dict))
            form.addRow("", header_widget)

            def add_row(entry_name="", entry_data=None):
                """Add a row with name + start/end fields."""
                if entry_data is None:
                    entry_data = {}

                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(4, 4, 4, 4)
                row_layout.setSpacing(4)
                row_widget.setMinimumHeight(28)

                # Name field
                name_edit = QLineEdit()
                name_edit.setPlaceholderText("Entry name")
                name_edit.setMaximumWidth(100)
                if entry_name:
                    name_edit.setText(str(entry_name))
                row_layout.addWidget(name_edit, 0)

                # Start field
                start_edit = QLineEdit()
                start_edit.setPlaceholderText("Start HH:MM:SS")
                start_edit.setMaximumWidth(100)
                if "start" in entry_data:
                    start_edit.setText(str(entry_data["start"]))
                row_layout.addWidget(start_edit, 0)

                # End field
                end_edit = QLineEdit()
                end_edit.setPlaceholderText("End HH:MM:SS")
                end_edit.setMaximumWidth(100)
                if "end" in entry_data:
                    end_edit.setText(str(entry_data["end"]))
                row_layout.addWidget(end_edit, 0)

                # Remove button
                remove_button = QPushButton("Remove")
                remove_button.setMaximumWidth(80)

                def remove():
                    row_idx, _ = form.getWidgetPosition(row_widget)
                    form.removeRow(row_idx)
                    if row_data in row_data_list:
                        row_data_list.remove(row_data)
                    if edits in row_edits:
                        row_edits.remove(edits)
                    refresh_remove_buttons()

                remove_button.clicked.connect(remove)
                row_layout.addWidget(remove_button, 0)

                # Add row to form
                header_idx, _ = form.getWidgetPosition(header_widget)
                if row_data_list:
                    last_idx, _ = form.getWidgetPosition(row_data_list[-1]["widget"])
                    insert_idx = last_idx + 1
                else:
                    insert_idx = header_idx + 1

                form.insertRow(insert_idx, "", row_widget)

                edits = (name_edit, start_edit, end_edit)
                row_data = {
                    "widget": row_widget,
                    "remove_button": remove_button,
                }
                row_data_list.append(row_data)
                row_edits.append(edits)
                refresh_remove_buttons()

            # Connect add button
            add_button.clicked.connect(lambda: add_row())

            # Prefill existing rows
            for entry_name, entry_data in (values or {}).items():
                add_row(entry_name, entry_data if isinstance(entry_data, dict) else {})

            # Return enriched dict
            return display_name, {
                "widget": header_widget,
                "time_window_map": True,
                "rows": row_edits,
            }

        else:
            return display_name, {"time_window_map": True, "rows": []}

    def create_time_data_map_input(self, setting_dict, form_context=None):
        """Create a hardcoded dict[str, ImageTimeData] editor with name + times per row.

        Each row is: name (entry name) + times (comma-separated list) + Remove button.

        Returns (display_name, enriched_dict) with "widget" and "rows" (list of
        2-tuples of QLineEdits: (name_edit, times_edit)).
        """
        key = setting_dict["key"]
        display_name = setting_dict.get("name", key)
        values = self.get_value(self.main_window.config_dict, key)
        if values is None:
            values = {}

        row_data_list = []
        row_edits = []  # List of 2-tuples: (name_edit, times_edit)

        def refresh_remove_buttons():
            show_remove = len(row_data_list) > 1
            for row_data in row_data_list:
                row_data["remove_button"].setVisible(show_remove)

        add_button = QPushButton("Add entry")

        if form_context:
            form = form_context["form"]

            # Build header widget
            header_widget = QWidget()
            header_layout = QHBoxLayout(header_widget)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(4)
            header_layout.addWidget(add_button, stretch=1)
            header_layout.addWidget(build_help_column(setting_dict))
            form.addRow("", header_widget)

            def add_row(entry_name="", entry_data=None):
                """Add a row with name + times fields."""
                if entry_data is None:
                    entry_data = {}

                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(4, 4, 4, 4)
                row_layout.setSpacing(4)
                row_widget.setMinimumHeight(28)

                # Name field
                name_edit = QLineEdit()
                name_edit.setPlaceholderText("Entry name")
                name_edit.setMaximumWidth(100)
                if entry_name:
                    name_edit.setText(str(entry_name))
                row_layout.addWidget(name_edit, 0)

                # Times field (comma-separated)
                times_edit = QLineEdit()
                times_edit.setPlaceholderText("Times (comma-separated HH:MM:SS)")
                times_edit.setMinimumWidth(300)
                if "times" in entry_data:
                    times_list = entry_data["times"]
                    if isinstance(times_list, list):
                        times_edit.setText(", ".join(str(t) for t in times_list))
                    else:
                        times_edit.setText(str(times_list))
                row_layout.addWidget(times_edit, 1)

                # Remove button
                remove_button = QPushButton("Remove")
                remove_button.setMaximumWidth(80)

                def remove():
                    row_idx, _ = form.getWidgetPosition(row_widget)
                    form.removeRow(row_idx)
                    if row_data in row_data_list:
                        row_data_list.remove(row_data)
                    if edits in row_edits:
                        row_edits.remove(edits)
                    refresh_remove_buttons()

                remove_button.clicked.connect(remove)
                row_layout.addWidget(remove_button, 0)

                # Add row to form
                header_idx, _ = form.getWidgetPosition(header_widget)
                if row_data_list:
                    last_idx, _ = form.getWidgetPosition(row_data_list[-1]["widget"])
                    insert_idx = last_idx + 1
                else:
                    insert_idx = header_idx + 1

                form.insertRow(insert_idx, "", row_widget)

                edits = (name_edit, times_edit)
                row_data = {"widget": row_widget, "remove_button": remove_button}
                row_data_list.append(row_data)
                row_edits.append(edits)
                refresh_remove_buttons()

            # Connect add button
            add_button.clicked.connect(lambda: add_row())

            # Prefill existing rows
            for entry_name, entry_data in (values or {}).items():
                add_row(entry_name, entry_data if isinstance(entry_data, dict) else {})

            # Return enriched dict
            return display_name, {
                "widget": header_widget,
                "time_data_map": True,
                "rows": row_edits,
            }

        else:
            return display_name, {"time_data_map": True, "rows": []}

    def create_path_data_map_input(self, setting_dict, form_context=None):
        """Create a hardcoded dict[str, PathData] editor with name + paths per row.

        Each row is: name (entry name) + paths (comma-separated list) + Remove button.

        Returns (display_name, enriched_dict) with "widget" and "rows" (list of
        2-tuples of QLineEdits: (name_edit, paths_edit)).
        """
        key = setting_dict["key"]
        display_name = setting_dict.get("name", key)
        values = self.get_value(self.main_window.config_dict, key)
        if values is None:
            values = {}

        row_data_list = []
        row_edits = []  # List of 2-tuples: (name_edit, paths_edit)

        def refresh_remove_buttons():
            show_remove = len(row_data_list) > 1
            for row_data in row_data_list:
                row_data["remove_button"].setVisible(show_remove)

        add_button = QPushButton("Add entry")

        if form_context:
            form = form_context["form"]

            # Build header widget
            header_widget = QWidget()
            header_layout = QHBoxLayout(header_widget)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(4)
            header_layout.addWidget(add_button, stretch=1)
            header_layout.addWidget(build_help_column(setting_dict))
            form.addRow("", header_widget)

            def add_row(entry_name="", entry_data=None):
                """Add a row with name + paths fields."""
                if entry_data is None:
                    entry_data = {}

                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(4, 4, 4, 4)
                row_layout.setSpacing(4)
                row_widget.setMinimumHeight(28)

                # Name field
                name_edit = QLineEdit()
                name_edit.setPlaceholderText("Entry name")
                name_edit.setMaximumWidth(100)
                if entry_name:
                    name_edit.setText(str(entry_name))
                row_layout.addWidget(name_edit, 0)

                # Paths field (comma-separated)
                paths_edit = QLineEdit()
                paths_edit.setPlaceholderText(
                    "Paths (comma-separated, supports glob *)"
                )
                paths_edit.setMinimumWidth(300)
                if "paths" in entry_data:
                    paths_list = entry_data["paths"]
                    if isinstance(paths_list, list):
                        paths_edit.setText(", ".join(str(p) for p in paths_list))
                    else:
                        paths_edit.setText(str(paths_list))
                row_layout.addWidget(paths_edit, 1)

                # Remove button
                remove_button = QPushButton("Remove")
                remove_button.setMaximumWidth(80)

                def remove():
                    row_idx, _ = form.getWidgetPosition(row_widget)
                    form.removeRow(row_idx)
                    if row_data in row_data_list:
                        row_data_list.remove(row_data)
                    if edits in row_edits:
                        row_edits.remove(edits)
                    refresh_remove_buttons()

                remove_button.clicked.connect(remove)
                row_layout.addWidget(remove_button, 0)

                # Add row to form
                header_idx, _ = form.getWidgetPosition(header_widget)
                if row_data_list:
                    last_idx, _ = form.getWidgetPosition(row_data_list[-1]["widget"])
                    insert_idx = last_idx + 1
                else:
                    insert_idx = header_idx + 1

                form.insertRow(insert_idx, "", row_widget)

                edits = (name_edit, paths_edit)
                row_data = {"widget": row_widget, "remove_button": remove_button}
                row_data_list.append(row_data)
                row_edits.append(edits)
                refresh_remove_buttons()

            # Connect add button
            add_button.clicked.connect(lambda: add_row())

            # Prefill existing rows
            for entry_name, entry_data in (values or {}).items():
                add_row(entry_name, entry_data if isinstance(entry_data, dict) else {})

            # Return enriched dict
            return display_name, {
                "widget": header_widget,
                "path_data_map": True,
                "rows": row_edits,
            }

        else:
            return display_name, {"path_data_map": True, "rows": []}

    def create_format_map_input(self, setting_dict, form_context=None):
        """Create a dict[str, ImageExportFormat] editor with type-conditional field visibility.

        Each row has 13 widgets: name, type (combo, driver for visibility), filename_pattern,
        resolution, keep_ratio (checkbox), then type-specific fields that toggle visibility:
        - dpi, cmap, quality, compression for jpg/png
        - dtype for npz/npy/csv
        - delimiter, header, float_format for csv

        Returns (display_name, enriched_dict) with "widget" and "rows" (list of 13-tuples).
        """
        from darsia.presets.workflows.config.format_registry import (
            SUPPORTED_EXPORT_FORMATS,
        )

        key = setting_dict["key"]
        display_name = setting_dict.get("name", key)

        # Read directly from config_dict["format"] (list of dicts) instead of the
        # dotted key path, since raw TOML has [[format]] array-of-tables, not
        # [format_registry.formats] nested tables. Keyed-dict for prefill.
        format_list = self.main_window.config_dict.get("format", [])
        value = {
            entry.get("name", ""): entry for entry in format_list if entry.get("name")
        }

        row_data_list = []
        row_edits = []

        def refresh_remove_buttons():
            show_remove = len(row_data_list) > 1
            for row_data in row_data_list:
                row_data["remove_button"].setVisible(show_remove)

        add_button = QPushButton("Add format")

        if form_context:
            form = form_context["form"]

            # Build header widget
            header_widget = QWidget()
            header_layout = QHBoxLayout(header_widget)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(4)
            header_layout.addWidget(add_button, stretch=1)
            header_layout.addWidget(build_help_column(setting_dict))
            form.addRow("", header_widget)

            def add_row(entry_name="", entry_data=None):
                """Add one format entry row with 13 widgets and type-conditional visibility."""
                if entry_data is None:
                    entry_data = {}

                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(4, 4, 4, 4)
                row_layout.setSpacing(4)
                row_widget.setMinimumHeight(28)

                # 1. name (entry key for the registry)
                name_edit = QLineEdit()
                name_edit.setPlaceholderText("Entry name")
                name_edit.setMaximumWidth(80)
                if entry_name:
                    name_edit.setText(str(entry_name))
                row_layout.addWidget(name_edit, 0)

                # 2. type (combo, driver field for visibility)
                type_combo = QComboBox()
                type_combo.setEditable(False)
                type_combo.addItems(sorted(SUPPORTED_EXPORT_FORMATS))
                type_combo.setMaximumWidth(80)
                if "type" in entry_data:
                    type_combo.setCurrentText(str(entry_data["type"]))
                row_layout.addWidget(type_combo, 0)

                # 3. filename_pattern (preset dropdown)
                filename_pattern_preset = [
                    "stem",
                    "stem_HH",
                    "stem_HH:MM",
                    "stem_HH:MM:SS",
                    "stem_MM:SS",
                    "stem_DD:HH",
                    "stem_DD:HH:MM",
                    "spatial_map_HH",
                    "spatial_map_HH:MM",
                    "spatial_map_HH:MM:SS",
                    "spatial_map_MM:SS",
                    "spatial_map_DD:HH",
                    "spatial_map_DD:HH:MM",
                ]
                filename_pattern_combo = QComboBox()
                filename_pattern_combo.setEditable(False)
                filename_pattern_combo.addItems(filename_pattern_preset)
                filename_pattern_combo.setMaximumWidth(140)

                current_pattern = entry_data.get("filename_pattern", "stem")
                # Add stale value if not in preset list (so it doesn't get silently clobbered)
                if current_pattern not in filename_pattern_preset:
                    filename_pattern_combo.addItem(current_pattern)
                    filename_pattern_combo.setCurrentText(current_pattern)
                else:
                    filename_pattern_combo.setCurrentText(current_pattern)

                row_layout.addWidget(filename_pattern_combo, 0)

                # 4. resolution (comma-separated "rows,cols")
                resolution_edit = QLineEdit()
                resolution_edit.setPlaceholderText("rows,cols")
                resolution_edit.setMaximumWidth(100)
                if "resolution" in entry_data and entry_data["resolution"]:
                    res = entry_data["resolution"]
                    resolution_edit.setText(f"{res[0]},{res[1]}")
                row_layout.addWidget(resolution_edit, 0)

                # 5. keep_ratio (checkbox)
                keep_ratio_check = QCheckBox("Keep ratio")
                keep_ratio_check.setMaximumWidth(100)
                if "keep_ratio" in entry_data:
                    keep_ratio_check.setChecked(bool(entry_data["keep_ratio"]))
                row_layout.addWidget(keep_ratio_check, 0)

                # Type-specific fields: jpg/png only
                # 6. dpi
                dpi_edit = QLineEdit()
                dpi_edit.setPlaceholderText("dpi")
                dpi_edit.setMaximumWidth(60)
                if "dpi" in entry_data and entry_data["dpi"]:
                    dpi_edit.setText(str(entry_data["dpi"]))
                row_layout.addWidget(dpi_edit, 0)

                # 7. cmap
                cmap_edit = QLineEdit()
                cmap_edit.setPlaceholderText("cmap")
                cmap_edit.setMaximumWidth(100)
                if "cmap" in entry_data and entry_data["cmap"]:
                    cmap_edit.setText(str(entry_data["cmap"]))
                row_layout.addWidget(cmap_edit, 0)

                # 8. quality (jpg/png)
                quality_edit = QLineEdit()
                quality_edit.setPlaceholderText("quality")
                quality_edit.setMaximumWidth(70)
                if "quality" in entry_data and entry_data["quality"]:
                    quality_edit.setText(str(entry_data["quality"]))
                row_layout.addWidget(quality_edit, 0)

                # 9. compression (jpg/png)
                compression_edit = QLineEdit()
                compression_edit.setPlaceholderText("compression")
                compression_edit.setMaximumWidth(90)
                if "compression" in entry_data and entry_data["compression"]:
                    compression_edit.setText(str(entry_data["compression"]))
                row_layout.addWidget(compression_edit, 0)

                # Type-specific fields: npz/npy/csv
                # 10. dtype
                dtype_edit = QLineEdit()
                dtype_edit.setPlaceholderText("dtype")
                dtype_edit.setMaximumWidth(80)
                if "dtype" in entry_data and entry_data["dtype"]:
                    dtype_edit.setText(str(entry_data["dtype"]))
                row_layout.addWidget(dtype_edit, 0)

                # Type-specific fields: csv only
                # 11. delimiter
                delimiter_edit = QLineEdit()
                delimiter_edit.setPlaceholderText("delimiter")
                delimiter_edit.setMaximumWidth(70)
                if "delimiter" in entry_data:
                    delimiter_edit.setText(str(entry_data["delimiter"]))
                else:
                    delimiter_edit.setText(",")
                row_layout.addWidget(delimiter_edit, 0)

                # 12. header
                header_edit = QLineEdit()
                header_edit.setPlaceholderText("header")
                header_edit.setMaximumWidth(70)
                if "header" in entry_data and entry_data["header"]:
                    header_edit.setText(str(entry_data["header"]))
                row_layout.addWidget(header_edit, 0)

                # 13. float_format
                float_format_edit = QLineEdit()
                float_format_edit.setPlaceholderText("float_format")
                float_format_edit.setMaximumWidth(90)
                if "float_format" in entry_data:
                    float_format_edit.setText(str(entry_data["float_format"]))
                else:
                    float_format_edit.setText("{:.2e}")
                row_layout.addWidget(float_format_edit, 0)

                # Remove button
                remove_button = QPushButton("Remove")
                remove_button.setMaximumWidth(80)

                def remove():
                    row_idx, _ = form.getWidgetPosition(row_widget)
                    form.removeRow(row_idx)
                    if row_data in row_data_list:
                        row_data_list.remove(row_data)
                    if edits in row_edits:
                        row_edits.remove(edits)
                    refresh_remove_buttons()

                remove_button.clicked.connect(remove)
                row_layout.addWidget(remove_button, 0)

                # Insert row into form
                header_idx, _ = form.getWidgetPosition(header_widget)
                if row_data_list:
                    last_idx, _ = form.getWidgetPosition(row_data_list[-1]["widget"])
                    insert_idx = last_idx + 1
                else:
                    insert_idx = header_idx + 1

                form.insertRow(insert_idx, "", row_widget)

                # Wire up type-conditional visibility using the depends_on pattern
                type_specific_widgets = {
                    "jpg_png": [dpi_edit, cmap_edit, quality_edit, compression_edit],
                    "npz_npy_csv": [dtype_edit],
                    "csv": [delimiter_edit, header_edit, float_format_edit],
                }

                def make_visibility_handler():
                    def handler(current_type):
                        # Always show: name, type, filename_pattern, resolution, keep_ratio
                        # Conditionally show type-specific fields
                        is_jpg_png = current_type in {"jpg", "png"}
                        is_npz_npy_csv = current_type in {"npz", "npy", "csv"}
                        is_csv = current_type == "csv"

                        for widget in type_specific_widgets["jpg_png"]:
                            widget.setVisible(is_jpg_png)
                        for widget in type_specific_widgets["npz_npy_csv"]:
                            widget.setVisible(is_npz_npy_csv)
                        for widget in type_specific_widgets["csv"]:
                            widget.setVisible(is_csv)

                    return handler

                # Connect type combo to visibility handler
                handler = make_visibility_handler()
                type_combo.currentTextChanged.connect(handler)

                # Set initial visibility based on current type
                if "type" in entry_data:
                    handler(entry_data["type"])
                else:
                    handler(type_combo.currentText())

                # Store row data and edits
                row_data = {"widget": row_widget, "remove_button": remove_button}
                edits = (
                    name_edit,
                    type_combo,
                    filename_pattern_combo,
                    resolution_edit,
                    keep_ratio_check,
                    dpi_edit,
                    cmap_edit,
                    quality_edit,
                    compression_edit,
                    dtype_edit,
                    delimiter_edit,
                    header_edit,
                    float_format_edit,
                )
                row_data_list.append(row_data)
                row_edits.append(edits)
                refresh_remove_buttons()

            # Connect add button
            add_button.clicked.connect(lambda: add_row())

            # Prefill existing entries
            if value:
                for entry_name, entry_data in value.items():
                    add_row(
                        entry_name, entry_data if isinstance(entry_data, dict) else {}
                    )
            else:
                # At least one empty row
                add_row()

            # Return enriched dict
            return display_name, {
                "widget": header_widget,
                "format_map": True,
                "rows": row_edits,
            }

        else:
            return display_name, {"format_map": True, "rows": []}

    def create_registry_key_list_input(self, setting_dict, form_context=None):
        """Create a multi-row registry-key selector with dropdowns.

        Each row is a QComboBox (non-editable) populated with available registry keys.
        On save, the union of all selected keys becomes data_selection as a list[str].

        Returns (display_name, enriched_dict) with "widget" and "rows" (list of
        QComboBox widgets).
        """
        key = setting_dict["key"]
        display_name = setting_dict.get("name", key)

        # Gather available registry keys
        registry = self.main_window.config_dict.get("registry", {})
        available_keys = sorted(
            set(registry.get("interval_registry", {}))
            | set(registry.get("window_registry", {}))
            | set(registry.get("time_registry", {}))
            | set(registry.get("path_registry", {}))
        )

        # Get current value and normalize to list
        value = self.get_value(self.main_window.config_dict, key)
        if value is None:
            current_keys = []
        elif isinstance(value, str):
            current_keys = [value]
        elif isinstance(value, list):
            current_keys = value
        else:
            current_keys = []

        row_data_list = []  # Track (widget, combo, remove_button)
        row_combos = []  # List of QComboBox widgets for save_settings

        def refresh_remove_buttons():
            show_remove = len(row_data_list) > 1
            for row_data in row_data_list:
                row_data["remove_button"].setVisible(show_remove)

        add_button = QPushButton("Add key")

        if form_context:
            form = form_context["form"]

            # Build header widget
            header_widget = QWidget()
            header_layout = QHBoxLayout(header_widget)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(4)
            header_layout.addWidget(add_button, stretch=1)
            header_layout.addWidget(build_help_column(setting_dict))
            form.addRow("", header_widget)

            def add_row(selected_key=""):
                """Add a row with a registry-key dropdown."""
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(4, 4, 4, 4)
                row_layout.setSpacing(4)
                row_widget.setMinimumHeight(28)

                # Dropdown with available keys + any stale current selection
                combo = QComboBox()
                combo.setEditable(False)
                all_options = list(available_keys)
                # Add selected_key if it's not already in the list (stale/deleted entry)
                if selected_key and selected_key not in all_options:
                    all_options.append(selected_key)
                    all_options.sort()
                combo.addItems(all_options)
                if selected_key:
                    combo.setCurrentText(selected_key)
                elif all_options:
                    combo.setCurrentIndex(0)
                row_layout.addWidget(combo, 1)

                # Remove button
                remove_button = QPushButton("Remove")
                remove_button.setMaximumWidth(80)

                def remove():
                    row_idx, _ = form.getWidgetPosition(row_widget)
                    form.removeRow(row_idx)
                    if row_data in row_data_list:
                        row_data_list.remove(row_data)
                    if combo in row_combos:
                        row_combos.remove(combo)
                    refresh_remove_buttons()

                remove_button.clicked.connect(remove)
                row_layout.addWidget(remove_button, 0)

                # Add row to form
                header_idx, _ = form.getWidgetPosition(header_widget)
                if row_data_list:
                    last_idx, _ = form.getWidgetPosition(row_data_list[-1]["widget"])
                    insert_idx = last_idx + 1
                else:
                    insert_idx = header_idx + 1

                form.insertRow(insert_idx, "", row_widget)

                row_data = {
                    "widget": row_widget,
                    "remove_button": remove_button,
                }
                row_data_list.append(row_data)
                row_combos.append(combo)
                refresh_remove_buttons()

            # Connect add button
            add_button.clicked.connect(lambda: add_row())

            # Prefill existing selections
            if current_keys:
                for key_name in current_keys:
                    add_row(key_name)
            else:
                # Always at least one empty row
                add_row()

            # Return enriched dict
            return display_name, {
                "widget": header_widget,
                "registry_key_list": True,
                "rows": row_combos,
            }

        else:
            return display_name, {"registry_key_list": True, "rows": []}

    def create_format_key_list_input(self, setting_dict, form_context=None):
        """Create a multi-row format-registry-key selector with dropdowns.

        Similar to create_registry_key_list_input but reads from the [[format]]
        array-of-tables TOML shape and filters dropdown options by format type.
        Each row is a QComboBox (non-editable) populated with available registry
        entries whose type is in the allowed set (or all if unrestricted).

        Returns (display_name, enriched_dict) with "widget" and "rows".
        """
        key = setting_dict["key"]
        display_name = setting_dict.get("name", key)

        # Gather available format-registry keys from the raw [[format]] array-of-tables,
        # filtering by format type if restricted
        format_list = self.main_window.config_dict.get("format", [])
        supported_types = setting_dict.get("format_types")

        if supported_types is None:
            # Unrestricted: all registry names, any type
            available_keys = sorted(
                {entry.get("name", "") for entry in format_list if entry.get("name")}
            )
        else:
            # Restricted: only names whose entry has a type in the allowed set
            available_keys = sorted(
                {
                    entry.get("name", "")
                    for entry in format_list
                    if entry.get("name") and entry.get("type") in supported_types
                }
            )

        # Get current value and normalize to list
        value = self.get_value(self.main_window.config_dict, key)
        if value is None:
            current_keys = []
        elif isinstance(value, str):
            current_keys = [value]
        elif isinstance(value, list):
            current_keys = value
        else:
            current_keys = []

        # For single-value fields (max_rows=1), cap the selection
        max_rows = setting_dict.get("max_rows")
        if max_rows == 1:
            current_keys = current_keys[:1]

        row_data_list = []
        row_combos = []

        def refresh_remove_buttons():
            show_remove = len(row_data_list) > 1
            for row_data in row_data_list:
                row_data["remove_button"].setVisible(show_remove)

        add_button = QPushButton("Add format")

        if form_context:
            form = form_context["form"]

            # Build header widget
            header_widget = QWidget()
            header_layout = QHBoxLayout(header_widget)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(4)

            if max_rows != 1:
                header_layout.addWidget(add_button, stretch=1)
            header_layout.addWidget(build_help_column(setting_dict))
            form.addRow("", header_widget)

            def add_row(selected_key=""):
                """Add a row with a format-key dropdown."""
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(4, 4, 4, 4)
                row_layout.setSpacing(4)
                row_widget.setMinimumHeight(28)

                # Dropdown with available keys + stale/appended values
                combo = QComboBox()
                combo.setEditable(False)
                all_options = list(available_keys)

                # Add selected_key if not already in options
                # (stale/filtered-out/raw-type-string)
                if selected_key and selected_key not in all_options:
                    all_options.append(selected_key)
                    all_options.sort()

                combo.addItems(all_options)
                if selected_key:
                    combo.setCurrentText(selected_key)
                elif all_options:
                    combo.setCurrentIndex(0)
                row_layout.addWidget(combo, 1)

                # Remove button
                remove_button = QPushButton("Remove")
                remove_button.setMaximumWidth(80)

                def remove():
                    row_idx, _ = form.getWidgetPosition(row_widget)
                    form.removeRow(row_idx)
                    if row_data in row_data_list:
                        row_data_list.remove(row_data)
                    if combo in row_combos:
                        row_combos.remove(combo)
                    refresh_remove_buttons()

                remove_button.clicked.connect(remove)
                row_layout.addWidget(remove_button, 0)

                # Insert row into form
                header_idx, _ = form.getWidgetPosition(header_widget)
                if row_data_list:
                    last_idx, _ = form.getWidgetPosition(row_data_list[-1]["widget"])
                    insert_idx = last_idx + 1
                else:
                    insert_idx = header_idx + 1

                form.insertRow(insert_idx, "", row_widget)

                # Store row data
                row_data = {"widget": row_widget, "remove_button": remove_button}
                row_data_list.append(row_data)
                row_combos.append(combo)
                refresh_remove_buttons()

            # Connect add button if not single-value field
            if max_rows != 1:
                add_button.clicked.connect(lambda: add_row())

            # Prefill existing selections
            if current_keys:
                for key_name in current_keys:
                    add_row(key_name)
            else:
                # Always at least one empty row
                add_row()

            # Return enriched dict with format_key_list marker and max_rows if set
            result_dict = {
                "widget": header_widget,
                "format_key_list": True,
                "rows": row_combos,
            }
            if max_rows is not None:
                result_dict["max_rows"] = max_rows

            return display_name, result_dict

        else:
            return display_name, {"format_key_list": True, "rows": []}

    def create_simple_input(self, setting_dict):
        """Create a line edit input for numeric or string values.

        Returns (label_text, field_widget) where field_widget is a composite HBox:
        [setting_edit (stretch=1), type_label, help_button_or_spacer (fixed 40px)]
        """
        key = setting_dict["key"]
        display_name = setting_dict.get("name", key)
        value = self.get_value(self.main_window.config_dict, key)

        # Fallback to legacy source field if not found (for backward compatibility)
        if value is None and setting_dict.get("legacy_source"):
            legacy_source = setting_dict.get("legacy_source")
            legacy_index = setting_dict.get("legacy_index")
            # Build sibling key path: e.g., "corrections.curvature.crop.pts_src"
            # from key "corrections.curvature.crop.top_left"
            parent_key = ".".join(key.split(".")[:-1])
            sibling_key = f"{parent_key}.{legacy_source}"
            legacy_value = self.get_value(self.main_window.config_dict, sibling_key)
            if (
                legacy_value is not None
                and isinstance(legacy_value, (list, tuple))
                and legacy_index is not None
                and legacy_index < len(legacy_value)
            ):
                value = legacy_value[legacy_index]

        if value is None:
            value = setting_dict.get("default")

        setting_edit = QLineEdit()
        if value is not None:
            setting_edit.setText(str(value))

        # Set placeholder text if provided
        placeholder = setting_dict.get("placeholder")
        if placeholder:
            setting_edit.setPlaceholderText(placeholder)

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

        # Store reference to the QComboBox for signal wiring in create_group_input
        field_widget.setProperty("value_widget", setting_combo)

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

        If the field declares `metadata={"active_list_key": "<name>"}` or has
        `active_bool_key`, the group box is made checkable with no custom QSS,
        allowing Qt to render a native checkmark in the title area. Fields without
        these metadata render as a plain non-toggleable box.

        Returns the group_box widget (to be added as a spanning row in a QFormLayout)
        and a result dict with sub_inputs and optional checkbox/active_list_key/name.
        """
        key = setting_dict["key"]  # e.g. "corrections.resize"
        name = key.rsplit(".", 1)[-1]
        display_name = setting_dict.get("name", name)
        active_list_name = setting_dict.get("active_list_key")
        active_bool_key = setting_dict.get("active_bool_key")

        group_box = QGroupBox(display_name)
        result = {}

        # Handle active_bool_key (section-level toggle, simple boolean)
        if active_bool_key is not None:
            is_active = self.get_value(self.main_window.config_dict, active_bool_key)
            if is_active is None:
                is_active = setting_dict.get("active_bool_default", True)

            group_box.setCheckable(True)
            group_box.setChecked(is_active)
            result.update(
                {
                    "checkbox": group_box,
                    "bool_key": active_bool_key,
                }
            )

        # Handle active_list_key (field-level toggles within a section, list membership)
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
        group_form.setContentsMargins(8, 10, 8, 8)  # Padding: left, top, right, bottom
        sub_inputs = {}
        field_row_map = {}  # Map unqualified_key -> (row_index, field_widget)
        group_forms = (
            {}
        )  # For nested multi-row fields, reuse _get_or_create_group_form helper

        for sub_setting in setting_dict["fields"]:
            sub_setting_type = sub_setting.get("type")
            own_name = sub_setting.get("name", sub_setting["key"].rsplit(".", 1)[-1])

            # Handle multi-row fields (multi_file, path_map, etc.) — create a sub-box inside
            # this group
            if sub_setting_type in self.MULTI_ROW_TYPES:
                # Create a sub-group box for this multi-row field inside the parent group
                multi_row_form = self._get_or_create_group_form(
                    group_forms, group_form, own_name, own_name
                )
                local_form_context = {"form": multi_row_form}
                label_text, field_or_result = self.create_setting_edit(
                    sub_setting, local_form_context
                )

                # Multi-row fields return {"widget": header_widget, "rows": [...]}
                if isinstance(field_or_result, dict) and "widget" in field_or_result:
                    multi_row_form.addRow("", field_or_result["widget"])
                    # Register the row-tracking payload so save_settings can find it
                    if "path_map" in field_or_result:
                        sub_inputs[sub_setting["key"]] = {
                            "path_map": True,
                            "rows": field_or_result["rows"],
                        }
                    elif "int_group_list" in field_or_result:
                        sub_inputs[sub_setting["key"]] = {
                            "int_group_list": True,
                            "rows": field_or_result["rows"],
                        }
                    elif "int_list_map" in field_or_result:
                        sub_inputs[sub_setting["key"]] = {
                            "int_list_map": True,
                            "flatten_in_section": field_or_result.get(
                                "flatten_in_section", False
                            ),
                            "section": field_or_result.get("section"),
                            "rows": field_or_result["rows"],
                        }
                    else:
                        sub_inputs[sub_setting["key"]] = field_or_result["rows"]
                continue

            # Handle nested groups (e.g., curvature correction stages)
            # These return (None, result_dict) and must be added as spanning rows
            label_text, field_widget = self.create_setting_edit(sub_setting)
            if (
                label_text is None
                and isinstance(field_widget, dict)
                and "widget" in field_widget
            ):
                # Nested group — add as spanning row, not label+field
                nested_widget = field_widget.get("widget")
                if nested_widget:
                    group_form.addRow(nested_widget)
                sub_inputs[sub_setting["key"]] = field_widget
                # Flatten nested sub_inputs so they appear in parent's sub_inputs
                for nested_key, nested_widget_or_result in field_widget.get(
                    "sub_inputs", {}
                ).items():
                    sub_inputs[nested_key] = nested_widget_or_result
                # Skip depends_on wiring for nested groups (only scalar fields supported)
                continue

            # Handle scalar fields normally
            group_form.addRow(label_text, field_widget)
            sub_inputs[sub_setting["key"]] = field_widget
            # Store row index for depends_on wiring
            unqualified_key = sub_setting["key"].rsplit(".", 1)[-1]
            row_index = group_form.rowCount() - 1
            field_row_map[unqualified_key] = (row_index, field_widget, sub_setting)
        result["sub_inputs"] = sub_inputs

        # Wire up depends_on visibility: for each field with a depends_on constraint,
        # connect the driver field's value-change signal to show/hide this row.
        for unqualified_key, (
            row_index,
            field_widget,
            sub_setting,
        ) in field_row_map.items():
            depends_on = sub_setting.get("depends_on")
            if depends_on is None:
                continue

            driver_field_key = depends_on.get("field")
            driver_value = depends_on.get("value")
            if driver_field_key is None or driver_value is None:
                continue

            # Find the driver field's widget
            if driver_field_key not in field_row_map:
                continue  # Driver field not found in this group, skip
            driver_row_index, driver_widget, driver_setting = field_row_map[
                driver_field_key
            ]

            # Extract the QComboBox from the composite field_widget via the stored property
            driver_combo = driver_widget.property("value_widget")
            if driver_combo is None:
                continue  # Driver is not a dropdown, skip

            # Create a closure to capture the row_index and driver_value
            def make_visibility_handler(row_idx, required_val):
                def handler(current_text):
                    group_form.setRowVisible(row_idx, current_text == required_val)

                return handler

            # Connect the driver's value-changed signal to show/hide this row
            handler = make_visibility_handler(row_index, driver_value)
            driver_combo.currentTextChanged.connect(handler)

            # Set initial visibility based on driver's current value
            handler(driver_combo.currentText())

        # Return (None, group_dict) where group_dict carries the group_box and metadata.
        # display_settings will check if the label is None to detect this case.
        # Mark as a group result so save_settings can identify it.
        result["widget"] = group_box
        result["is_group_result"] = True
        return None, result
