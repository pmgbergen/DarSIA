"""Settings and input widget factory for DarSIA GUI."""

import ast

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from darsia.gui.ui.schema.dataclass_introspection import _build_fields
from darsia.presets.workflows.config.format_registry import (
    _format_entry_to_dict,
)

from .file_dialog import NO_FILE_CHOSEN, FileDialogHelper
from .help import build_help_column
from .schema.dataclass_introspection import (
    ALL_SECTIONS,
    SECTION_LOADABLE,
    get_section_fields,
)
from .schema.section_registry import get_required_sections


def unwrap_composite_widget(value):
    """Extract the real editable control from a composite field wrapper.

    Composite field widgets built by create_simple_input(), create_bool_input(),
    create_dropdown_input(), and create_file_chooser() are QWidget wrappers that
    store their real control via setProperty("value_widget", ...). If value isn't
    such a wrapper, it is returned unchanged.
    """
    if type(value) is QWidget:
        unwrapped = value.property("value_widget")
        if unwrapped is not None:
            return unwrapped
    return value


def _parse_list_text(text, list_type=None):
    """Parse user-friendly list text (space/comma-separated or bracket syntax)
    into a Python list.

    Accepts multiple input formats:
    - Python literal: [1, 2, 3] or (1, 2, 3) or 1, 2, 3
    - Space-separated: 1 2 3
    - Comma-separated: 1, 2, 3
    - Mixed: 1, 2 3 (commas take precedence)

    Args:
        text: User-entered text (e.g., "0.1, 0.2" or "0.1 0.2" or "[0.1, 0.2]").
        list_type: Element type for coercion ("int", "float", "string", "file").
                   If None, no coercion is applied.

    Returns:
        A Python list of parsed and (optionally) coerced values.

    Raises:
        ValueError: If parsing fails or coercion is impossible.
        SyntaxError: If literal_eval encounters invalid syntax.
    """
    text = text.strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            result = list(parsed)
        else:
            result = [parsed]
    except (ValueError, SyntaxError):
        if "," in text:
            parts = text.split(",")
        else:
            parts = text.split()

        result = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            try:
                result.append(ast.literal_eval(part))
            except (ValueError, SyntaxError):
                result.append(part)

    if list_type:
        coerced = []
        for item in result:
            if list_type in ("int", "float"):
                coerced.append(float(item) if list_type == "float" else int(item))
            elif list_type in ("string", "file"):
                coerced.append(str(item))
            else:
                coerced.append(item)
        result = coerced

    return result


def _format_list_text(value):
    """Format a list as bracket-free, comma-separated text for canonical display.

    Args:
        value: A list or tuple to format.

    Returns:
        A string like "0.1, 0.2, 0.3" (no brackets).
    """
    if not value:
        return ""
    return ", ".join(str(v) for v in value)


class SettingsFactory:
    """Factory for creating settings input widgets and managing settings."""

    # Multi-row widget types that need special handling (auto-grouped into their own box)
    MULTI_ROW_TYPES = {
        "multi_file",
        "multi_folder",
        "path_map",
        "int_group_list",
        "int_list_map",
        "dataclass_group_map",
        "format_key_list",
        "registry_key_list",
        "roi_key_list",
        "color_key_list",
    }

    @staticmethod
    def _wrap_multi_row_result(field_or_result):
        """Wrap a multi-row result dict with type tag for save-time identification.

        Multi-row create_setting_edit calls return dicts with a "widget" (header) and
        "rows" (list of QComboBox/QLineEdit/etc). This helper preserves the type tag
        so _sync_settings_inputs_to_config_dict can identify which kind of multi-row
        field it is and serialize it correctly. Fixes bug where nested multi-row fields
        (inside Optional[dataclass] groups) were stored as untagged bare lists and
        silently dropped on save.

        Parameters
        ----------
        field_or_result : dict
            Result from create_setting_edit for a MULTI_ROW_TYPES field,
            containing "widget" and "rows" plus a type tag.

        Returns
        -------
        dict
            Tagged result dict ready for settings_inputs, or bare "rows" list
            if no type tag is recognized.
        """
        if "path_map" in field_or_result:
            return {
                "path_map": True,
                "rows": field_or_result["rows"],
            }
        elif "int_group_list" in field_or_result:
            return {
                "int_group_list": True,
                "rows": field_or_result["rows"],
            }
        elif "int_list_map" in field_or_result:
            return {
                "int_list_map": True,
                "flatten_in_section": field_or_result.get("flatten_in_section", False),
                "section": field_or_result.get("section"),
                "rows": field_or_result["rows"],
            }
        elif "dataclass_group_map" in field_or_result:
            entry = {
                "dataclass_group_map": True,
                "entries": field_or_result["entries"],
            }
            if "array_key" in field_or_result:
                entry["array_key"] = field_or_result["array_key"]
            return entry
        elif "registry_key_list" in field_or_result:
            return {
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
            return result_dict
        elif "roi_key_list" in field_or_result:
            result_dict = {
                "roi_key_list": True,
                "rows": field_or_result["rows"],
            }
            if "max_rows" in field_or_result:
                result_dict["max_rows"] = field_or_result["max_rows"]
            return result_dict
        elif "color_key_list" in field_or_result:
            result_dict = {
                "color_key_list": True,
                "rows": field_or_result["rows"],
            }
            if "max_rows" in field_or_result:
                result_dict["max_rows"] = field_or_result["max_rows"]
            return result_dict
        else:
            return field_or_result["rows"]

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

                # Check if this is a dotted section (e.g., "calibration.color")
                # Try to resolve as a top-level section first, then as a base.group pair
                section_fields = get_section_fields(section)

                if section_fields is None and "." in section:
                    # Split and try resolving as base_section with only_group filter
                    base_section, group = section.split(".", 1)
                    section_fields = get_section_fields(base_section, only_group=group)
                    # Store under base_section so it renders as one "Calibration" tab,
                    # not multiple per group (since in normal GUI use, only one of
                    # "calibration.color" or "calibration.mass" is active at a time)
                    if section_fields is not None:
                        section = base_section

                if section_fields is None:
                    self.main_window.print_log(
                        f"No dataclass found for section '{section}'"
                    )
                    continue

                settings_by_section[section] = section_fields

        # Append the activity-specific options group if the action is an activity
        ACTIVITY_OPTIONS_GROUPS = {"setup", "calibration", "analysis", "helper"}
        if action in ACTIVITY_OPTIONS_GROUPS:
            options_fields = get_section_fields("options", only_group=action)
            if options_fields is not None:
                settings_by_section["options"] = options_fields

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
        field_row_map = {}  # key -> (row_index, field_or_result, setting, target_form)

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
                row_index = target_form.rowCount() - 1
                self.main_window.settings_inputs[setting["key"]] = field_or_result
                for sub_key, sub_widget in field_or_result.get(
                    "sub_inputs", {}
                ).items():
                    self.main_window.settings_inputs[sub_key] = sub_widget
                # Track this group for depends_on wiring
                unqualified_key = setting["key"].rsplit(".", 1)[-1]
                field_row_map[unqualified_key] = (
                    row_index,
                    field_or_result,
                    setting,
                    target_form,
                )
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
                self.main_window.settings_inputs[setting["key"]] = (
                    self._wrap_multi_row_result(field_or_result)
                )
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

        # Wire up depends_on visibility for top-level group fields (e.g., restoration
        # method -> volume_averaging_options / tvd_options)
        for unqualified_key, (
            row_index,
            field_or_result,
            setting,
            target_form,
        ) in field_row_map.items():
            depends_on = setting.get("depends_on")
            if depends_on is None:
                continue

            driver_field_key = depends_on.get("field")
            driver_value = depends_on.get("value")
            if driver_field_key is None or driver_value is None:
                continue

            # Find the driver field's widget — must be a scalar field in the same form
            # (typically at the top level; handle via full key lookup in settings_inputs)
            section = setting["key"].rsplit(".", 1)[0]
            driver_full_key = f"{section}.{driver_field_key}"

            # Look up the driver widget in settings_inputs
            if driver_full_key not in self.main_window.settings_inputs:
                continue

            driver_widget = self.main_window.settings_inputs[driver_full_key]
            # Unwrap composite widget to get the real QComboBox
            driver_combo = unwrap_composite_widget(driver_widget)
            if not isinstance(driver_combo, QComboBox):
                continue

            # Get the group widget to control visibility directly
            group_widget = field_or_result.get("widget")
            if group_widget is None:
                continue

            # Create visibility handler supporting both single and list values
            def make_visibility_handler(widget, required_val):
                def handler(current_text):
                    is_visible = (
                        current_text in required_val
                        if isinstance(required_val, (list, set, tuple))
                        else current_text == required_val
                    )
                    widget.setVisible(is_visible)

                return handler

            handler = make_visibility_handler(group_widget, driver_value)
            driver_combo.currentTextChanged.connect(handler)

            # Set initial visibility
            handler(driver_combo.currentText())

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
        elif setting_type == "time":
            return self.create_time_input(setting_dict)
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
        elif setting_type == "dataclass_group_map":
            entry_dataclass = setting_dict.get("entry_dataclass")
            assert entry_dataclass is not None
            return self.create_dataclass_group_map_input(
                setting_dict,
                entry_dataclass=entry_dataclass,
                form_context=form_context,
            )
        elif setting_type == "registry_key_list":
            return self.create_registry_key_list_input(
                setting_dict, form_context=form_context
            )
        elif setting_type == "format_key_list":
            return self.create_format_key_list_input(
                setting_dict, form_context=form_context
            )
        elif setting_type == "roi_key_list":
            return self.create_roi_key_list_input(
                setting_dict, form_context=form_context
            )
        elif setting_type == "color_key_list":
            return self.create_color_key_list_input(
                setting_dict, form_context=form_context
            )
        else:
            self.main_window.print_log(
                f"Setting type {setting_type} not supported yet, using simple input"
            )
            return self.create_simple_input(setting_dict)

    def create_dataclass_group_map_input(
        self, setting_dict, entry_dataclass, form_context=None
    ):
        r"""Create a dict[str, entry_dataclass] editor with one QGroupBox per entry.

        Each entry is rendered as a checkable QGroupBox containing a QFormLayout with
        labeled sub-fields (generated via create_setting_edit per field).

        Parameters
        ----------
        setting_dict : dict
            Setting configuration dictionary with "key", "name", "help", "widget".
        entry_dataclass : type
            The dataclass type for each dict value (e.g., ImageExportFormat).
        form_context : dict, optional
            Context dict with "form" (QFormLayout) for row insertion.

        Returns
        -------
        tuple
            (display_name, enriched_dict) where enriched_dict has "widget" (header widget
            for insertion into parent form), "dataclass_group_map": True (marker), and
            "entries" (list of dicts, one per entry, with "name", "widget", "fields").
        """
        try:
            key = setting_dict["key"]
            display_name = setting_dict.get("name", key)
        except Exception as e:
            self.main_window.print_log(f"Error extracting setting dict keys: {e}")
            return setting_dict.get("name", "Unknown"), {
                "dataclass_group_map": True,
                "entries": [],
            }

        # Read existing entries from config_dict
        # For array-of-tables TOML (marked by metadata), read from config_dict[array_key]
        # For nested table-style (default), use the dotted key path
        try:
            array_key = setting_dict.get("array_key")
            if array_key:
                # Array-of-tables style: [[format]], [[roi]], etc.
                entry_list = self.main_window.config_dict.get(array_key, [])
                value = {
                    entry.get("name", ""): entry
                    for entry in entry_list
                    if entry.get("name")
                }
            else:
                # Generic nested-table style
                value = self.get_value(self.main_window.config_dict, key)
                if value is None:
                    value = {}

            entries_data = []  # List of {name, widget, fields, field_widgets}
            entry_schema_list = _build_fields(
                entry_dataclass, "entry"
            )  # Build schema for entry
        except Exception as e:
            self.main_window.print_log(
                f"Error building schema for {entry_dataclass.__name__}: {e}"
            )
            import traceback

            self.main_window.print_log(traceback.format_exc())
            return display_name, {"dataclass_group_map": True, "entries": []}

        try:
            add_button = QPushButton("Add Entry")
        except Exception as e:
            self.main_window.print_log(f"Error creating add button: {e}")
            return display_name, {"dataclass_group_map": True, "entries": []}

        if form_context:
            try:
                form = form_context["form"]

                # Build header widget (load button added later after entries_data exists)
                header_widget = QWidget()
                header_layout = QHBoxLayout(header_widget)
                header_layout.setContentsMargins(0, 0, 0, 0)
                header_layout.setSpacing(4)
                header_layout.addWidget(add_button, stretch=1)
                header_layout.addWidget(build_help_column(setting_dict))
                form.addRow("", header_widget)
            except Exception as e:
                self.main_window.print_log(f"Error building header widget: {e}")
                import traceback

                self.main_window.print_log(traceback.format_exc())
                return display_name, {"dataclass_group_map": True, "entries": []}

            def add_entry(entry_name="", entry_data=None):
                """Add one entry group box with labeled sub-fields."""
                if entry_data is None:
                    entry_data = {}

                # Create group box for this entry
                entry_name_display = entry_name or "(new entry)"
                group_box = QGroupBox(entry_name_display)
                if setting_dict.get("checkable", False):
                    group_box.setCheckable(True)
                    group_box.setChecked(True)
                group_layout = QFormLayout(group_box)
                group_layout.setContentsMargins(8, 10, 8, 8)

                # Editable entry-name field (the dict key)
                name_edit = QLineEdit()
                name_edit.setPlaceholderText("Entry name (dict key)")
                if entry_name:
                    name_edit.setText(str(entry_name))
                group_layout.addRow("Name:", name_edit)

                # Wire name changes to update group box title
                def update_group_title(text):
                    group_box.setTitle(text or "(new entry)")

                name_edit.textChanged.connect(update_group_title)

                # Build one widget per schema field using create_setting_edit
                field_widgets = {}  # name -> composite widget dict
                field_row_map = {}
                # name -> (row_index, composite_widget, field_dict, target_form)
                group_forms = {}  # group_name -> QFormLayout, scoped to this entry

                for field_schema in entry_schema_list:
                    field_name = field_schema["key"].split(".", 1)[
                        -1
                    ]  # Unqualified name

                    if field_name == "name":
                        continue

                    # Use create_setting_edit for all other fields (returns composite
                    # wrapper or group result)
                    label_text, field_widget = self.create_setting_edit(field_schema)

                    # Handle nested groups: label_text is None, field_widget is dict with
                    # "widget"
                    if (
                        label_text is None
                        and isinstance(field_widget, dict)
                        and "widget" in field_widget
                    ):
                        # Nested group (dataclass field) — add as spanning row, not
                        # label+field
                        # Note: dataclass-typed fields cannot carry "group" metadata
                        # (guarded in dataclass_introspection.py:242-247), so target_form
                        # is always group_layout
                        nested_widget = field_widget.get("widget")
                        if nested_widget:
                            group_layout.addRow(nested_widget)
                        # Store the entire group result dict (contains sub_inputs,
                        # is_group_result)
                        field_widgets[field_name] = field_widget
                        row_index = group_layout.rowCount() - 1
                        field_row_map[field_name] = (
                            row_index,
                            field_widget,
                            field_schema,
                            group_layout,
                        )
                        # Skip scalar prefill and depends_on wiring for nested groups
                        continue

                    # Scalar field: resolve target form (group-aware)
                    group_name = field_schema.get("group_name")
                    target_form = (
                        self._get_or_create_group_form(
                            group_forms, group_layout, group_name, group_name
                        )
                        if group_name
                        else group_layout
                    )

                    # Add label+widget row to the resolved target form
                    target_form.addRow(label_text, field_widget)

                    # Override prefilled value from entry_data (not config_dict)
                    if field_name in entry_data and entry_data[field_name] is not None:
                        unwrapped = unwrap_composite_widget(field_widget)
                        if isinstance(unwrapped, QCheckBox):
                            unwrapped.setChecked(bool(entry_data[field_name]))
                        elif isinstance(unwrapped, QComboBox):
                            unwrapped.setCurrentText(str(entry_data[field_name]))
                        else:  # QLineEdit
                            # Use canonical format for list fields
                            if unwrapped.property("darsia_is_list"):
                                value_str = _format_list_text(entry_data[field_name])
                            else:
                                value_str = str(entry_data[field_name])
                            unwrapped.setText(value_str)

                    field_widgets[field_name] = field_widget
                    row_index = target_form.rowCount() - 1
                    field_row_map[field_name] = (
                        row_index,
                        field_widget,
                        field_schema,
                        target_form,
                    )

                # Wire up depends_on visibility (same pattern as create_group_input)
                # Note: only scalar fields support depends_on; skip group results
                for (
                    row_index,
                    field_widget,
                    field_schema,
                    field_target_form,
                ) in field_row_map.values():
                    depends_on = field_schema.get("depends_on")
                    if depends_on is None:
                        continue

                    # Skip group result dicts (only scalar fields support depends_on)
                    if isinstance(field_widget, dict) and field_widget.get(
                        "is_group_result"
                    ):
                        continue

                    driver_field_key = depends_on.get("field")
                    driver_value = depends_on.get("value")
                    if driver_field_key is None or driver_value is None:
                        continue

                    # Find the driver field's widget in this entry
                    if driver_field_key not in field_row_map:
                        continue
                    driver_row_index, driver_widget, driver_schema, _driver_form = (
                        field_row_map[driver_field_key]
                    )

                    # Unwrap composite widget to get the real control
                    unwrapped_driver = unwrap_composite_widget(driver_widget)
                    if isinstance(unwrapped_driver, QComboBox):
                        driver_combo = unwrapped_driver
                    else:
                        continue

                    # Create visibility handler supporting both single and list values
                    def make_visibility_handler(row_idx, required_val, form):
                        def handler(current_text):
                            is_visible = (
                                current_text in required_val
                                if isinstance(required_val, (list, set, tuple))
                                else current_text == required_val
                            )
                            form.setRowVisible(row_idx, is_visible)

                        return handler

                    handler = make_visibility_handler(
                        row_index, driver_value, field_target_form
                    )
                    driver_combo.currentTextChanged.connect(handler)

                    # Set initial visibility
                    handler(driver_combo.currentText())

                # Remove button
                remove_button = QPushButton("Remove")

                def remove():
                    row_idx, _ = form.getWidgetPosition(group_box)
                    form.removeRow(row_idx)
                    if entry in entries_data:
                        entries_data.remove(entry)
                    refresh_remove_buttons()

                remove_button.clicked.connect(remove)
                group_layout.addRow(remove_button)

                # Insert group box into parent form
                header_idx, _ = form.getWidgetPosition(header_widget)
                if entries_data:
                    last_idx, _ = form.getWidgetPosition(entries_data[-1]["widget"])
                    insert_idx = last_idx + 1
                else:
                    insert_idx = header_idx + 1

                form.insertRow(insert_idx, group_box)

                # Track this entry
                entry = {
                    "name": entry_name,
                    "widget": group_box,
                    "name_edit": name_edit,
                    "fields": field_widgets,
                    "field_schemas": {
                        fs["key"].split(".", 1)[-1]: fs for fs in entry_schema_list
                    },
                    "remove_button": remove_button,
                }
                entries_data.append(entry)
                refresh_remove_buttons()

            def refresh_remove_buttons():
                for entry in entries_data:
                    entry["remove_button"].setVisible(True)

            try:
                # Connect add button
                add_button.clicked.connect(lambda: add_entry())

                # Add load button if this dataclass_group_map is loadable (e.g., formats)
                loadable_type = setting_dict.get("loadable")
                if loadable_type:

                    def on_apply_list(name, entry_dict):
                        # For list-of-dataclass fields, check for duplicate then append
                        if any(e["name"] == name for e in entries_data):
                            self.main_window.print_log(
                                f"{loadable_type.capitalize()} '{name}' already exists, "
                                "skipped."
                            )
                            return
                        add_entry(name, entry_dict)

                    load_button = self._create_load_button(setting_dict, on_apply_list)
                    if load_button:
                        # Insert load button into header layout before help column
                        header_layout.insertWidget(1, load_button, stretch=0)

                # Prefill existing entries
                if value:
                    for entry_name, entry_data in value.items():
                        add_entry(
                            entry_name,
                            entry_data if isinstance(entry_data, dict) else {},
                        )
                elif setting_dict.get("auto_add_empty", False):
                    # Auto-create one empty entry if list is empty (unless disabled via
                    # metadata)
                    add_entry()

                # Return enriched dict (thread array_key through for save-pass)
                result = {
                    "widget": header_widget,
                    "dataclass_group_map": True,
                    "entries": entries_data,
                }
                if array_key:
                    result["array_key"] = array_key
                return display_name, result
            except Exception as e:
                self.main_window.print_log(f"Error in add_entry logic: {e}")
                import traceback

                self.main_window.print_log(traceback.format_exc())
                fallback = {
                    "widget": header_widget,
                    "dataclass_group_map": True,
                    "entries": entries_data,
                }
                if array_key:
                    fallback["array_key"] = array_key
                return display_name, fallback

        else:
            return display_name, {"dataclass_group_map": True, "entries": []}

    def create_registry_key_list_input(self, setting_dict, form_context=None):
        """Create a multi-row registry-key selector with dropdowns.

        Each row is a QComboBox (non-editable) populated with available registry keys.
        On save, the union of all selected keys becomes data_selection as a list[str].

        Returns (display_name, enriched_dict) with "widget" and "rows" (list of
        QComboBox widgets).
        """
        key = setting_dict["key"]
        display_name = setting_dict.get("name", key)

        # Gather available registry keys from top-level arrays
        def extract_names(array):
            if isinstance(array, list):
                return {
                    entry.get("name")
                    for entry in array
                    if isinstance(entry, dict) and entry.get("name")
                }
            return set()

        available_keys = sorted(
            extract_names(self.main_window.config_dict.get("data_interval", []))
            | extract_names(self.main_window.config_dict.get("data_window", []))
            | extract_names(self.main_window.config_dict.get("data_time", []))
            | extract_names(self.main_window.config_dict.get("data_path", []))
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
            for row_data in row_data_list:
                row_data["remove_button"].setVisible(True)

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
            for row_data in row_data_list:
                row_data["remove_button"].setVisible(True)

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

    def create_roi_key_list_input(self, setting_dict, form_context=None):
        """Create a multi-row ROI-registry-key selector with dropdowns.

        Reads from the [[roi]] array-of-tables TOML shape.
        Each row is a QComboBox populated with available ROI registry entry names.

        Returns (display_name, enriched_dict) with "widget" and "rows".
        """
        key = setting_dict["key"]
        display_name = setting_dict.get("name", key)

        # Gather available ROI-registry keys from the raw [[roi]] array-of-tables
        roi_list = self.main_window.config_dict.get("roi", [])
        available_keys = sorted(
            {entry.get("name", "") for entry in roi_list if entry.get("name")}
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
            for row_data in row_data_list:
                row_data["remove_button"].setVisible(True)

        add_button = QPushButton("Add ROI")

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
                """Add a row with a ROI-key dropdown."""
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(4, 4, 4, 4)
                row_layout.setSpacing(4)
                row_widget.setMinimumHeight(28)

                # Dropdown with available keys + stale/appended values
                combo = QComboBox()
                combo.setEditable(False)
                all_options = list(available_keys)

                # Add selected_key if not already in options (stale/filtered-out value)
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

            # Return enriched dict with roi_key_list marker and max_rows if set
            result_dict = {
                "widget": header_widget,
                "roi_key_list": True,
                "rows": row_combos,
            }
            if max_rows is not None:
                result_dict["max_rows"] = max_rows

            return display_name, result_dict

        else:
            return display_name, {"roi_key_list": True, "rows": []}

    def create_color_key_list_input(self, setting_dict, form_context=None):
        """Create a multi-row color-embedding-key selector with dropdowns.

        Reads from the [[color]] array-of-tables TOML shape.
        Each row is a QComboBox populated with available color embedding names.

        Returns (display_name, enriched_dict) with "widget" and "rows".
        """
        key = setting_dict["key"]
        display_name = setting_dict.get("name", key)

        # Gather available color-embedding keys from the raw [[color]] array-of-tables
        color_list = self.main_window.config_dict.get("color", [])
        available_keys = sorted(
            {entry.get("name", "") for entry in color_list if entry.get("name")}
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
            for row_data in row_data_list:
                row_data["remove_button"].setVisible(True)

        add_button = QPushButton("Add Color")

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
                """Add a row with a color-embedding-key dropdown."""
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(4, 4, 4, 4)
                row_layout.setSpacing(4)
                row_widget.setMinimumHeight(28)

                # Dropdown with available keys + stale/appended values
                combo = QComboBox()
                combo.setEditable(False)
                all_options = list(available_keys)

                # Add selected_key if not already in options (stale/filtered-out value)
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

            # Return enriched dict with color_key_list marker and max_rows if set
            result_dict = {
                "widget": header_widget,
                "color_key_list": True,
                "rows": row_combos,
            }
            if max_rows is not None:
                result_dict["max_rows"] = max_rows

            return display_name, result_dict

        else:
            return display_name, {"color_key_list": True, "rows": []}

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
            if setting_dict["type"] == "list":
                setting_edit.setText(_format_list_text(value))
            else:
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
            if setting_dict.get("fixed_length") is not None:
                # Fixed-size tuple: show the arity (e.g., "2 x float" for tuple[float, float])
                arity = setting_dict["fixed_length"]
                elem_type = setting_dict["list_type"]
                type_label = QLabel(f"({arity} x {elem_type})")
            else:
                # Variable-length list
                type_label = QLabel(
                    f"({setting_dict['type']}, {setting_dict['list_type']})"
                )
        else:
            type_label = QLabel(f"({setting_dict['type']})")

        field_layout.addWidget(setting_edit, stretch=1)
        field_layout.addWidget(type_label)

        # Right column: help button or spacer (fixed 40px)
        field_layout.addWidget(build_help_column(setting_dict))

        # Store reference to the real control for unwrapping in sync
        field_widget.setProperty("value_widget", setting_edit)

        # Tag list fields so the save pass can identify and parse them specially
        if setting_dict["type"] == "list":
            list_type = setting_dict.get("list_type")
            fixed_length = setting_dict.get("fixed_length")

            setting_edit.setProperty("darsia_is_list", True)
            setting_edit.setProperty("darsia_list_type", list_type)
            # Tag fixed-length fields (derived from tuple arity)
            if fixed_length is not None:
                setting_edit.setProperty("darsia_fixed_length", fixed_length)

                def normalize_list(se=setting_edit, lt=list_type, fl=fixed_length):
                    text = se.text()
                    if not text.strip():
                        se.setStyleSheet("")
                        se.setToolTip("")
                        return
                    try:
                        parsed = _parse_list_text(text, lt)
                    except (ValueError, SyntaxError) as e:
                        se.setStyleSheet("border: 1px solid #d32f2f;")
                        se.setToolTip(f"Invalid list value: {e}")
                        return

                    # Truncate to fixed length if too long
                    if len(parsed) > fl:
                        parsed = parsed[:fl]
                        se.setText(_format_list_text(parsed))

                    # Mark red if too short
                    if len(parsed) < fl:
                        se.setStyleSheet("border: 1px solid #d32f2f;")
                        se.setToolTip(
                            f"Expected exactly {fl} entries, got {len(parsed)}."
                        )
                    else:
                        se.setStyleSheet("")
                        se.setToolTip("")

                setting_edit.editingFinished.connect(normalize_list)

        return display_name, field_widget

    def create_time_input(self, setting_dict):
        """Create a time input field with HH:MM:SS normalization.

        Returns (label_text, field_widget) where field_widget is a composite HBox:
        [setting_edit (stretch=1), type_label, help_button_or_spacer (fixed 40px)]

        The input is normalized on blur (editingFinished) to canonical HH:MM:SS format.
        Invalid input shows a red border and tooltip error, but leaves the text intact
        for the user to fix.
        """
        from darsia.presets.workflows.config.utils import _normalize_time_string

        key = setting_dict["key"]
        display_name = setting_dict.get("name", key)
        value = self.get_value(self.main_window.config_dict, key)
        if value is None:
            value = setting_dict.get("default")

        setting_edit = QLineEdit()
        if value is not None:
            setting_edit.setText(str(value))

        placeholder = setting_dict.get("placeholder")
        if placeholder:
            setting_edit.setPlaceholderText(placeholder)

        # Wire normalization on blur
        def normalize_time():
            text = setting_edit.text().strip()
            if not text:
                return  # Allow blank for optional fields
            try:
                normalized = _normalize_time_string(text)
                setting_edit.setText(normalized)
                # Clear error styling on success
                setting_edit.setStyleSheet("")
                setting_edit.setToolTip("")
            except (ValueError, AssertionError) as e:
                # Show error state but don't crash or lose the text
                setting_edit.setStyleSheet("border: 1px solid #d32f2f;")
                setting_edit.setToolTip(f"Invalid time format: {e}")

        setting_edit.editingFinished.connect(normalize_time)

        # Build composite field widget with type label and help button
        field_widget = QWidget()
        field_layout = QHBoxLayout(field_widget)
        field_layout.setContentsMargins(0, 0, 0, 0)
        field_layout.setSpacing(4)

        type_label = QLabel(f"({setting_dict['type']})")
        field_layout.addWidget(setting_edit, stretch=1)
        field_layout.addWidget(type_label)
        field_layout.addWidget(build_help_column(setting_dict))

        field_widget.setProperty("value_widget", setting_edit)
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

        # Store reference to the real control for unwrapping in sync
        field_widget.setProperty("value_widget", setting_checkbox)

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
            for row in row_data_list:
                row["remove_button"].setVisible(True)

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
            for row in row_data_list:
                row["remove_button"].setVisible(True)

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

    def _create_load_button(self, setting_dict, on_apply):
        """Create a 'Load' menu button for a loadable registry (e.g., curvature or formats).

        Returns a QToolButton with a QMenu offering:
        - "From TOML..." → file picker, parse entries, call on_apply per entry
        - "From catalogue > <preset name>" → call on_apply with catalogue entry

        Args:
            setting_dict: The setting dict (must have "loadable" key).
            on_apply: Callable(name: str, entry_dict: dict) -> None,
                called once per loaded item (name as dict key, entry_dict as TOML dict).

        Returns:
            A QToolButton with QMenu, or None if not loadable.
        """
        loadable_type = setting_dict.get("loadable")
        if not loadable_type:
            return None

        # Map loadable_type to config class, catalogue class, catalogue filename
        from pathlib import Path

        loadable_config = {
            "curvature": {
                "config_class": "darsia.presets.workflows.config.corrections:CurvatureCorrectionConfig",  # noqa: E501
                "catalogue_class": "darsia.presets.workflows.config.catalogue.corrections:CurvatureCatalogue",  # noqa: E501
                "catalogue_file": "corrections.toml",
                "toml_key": "corrections",
                "toml_subkey": "curvature",
                "is_single_section": True,
            },
            "format": {
                "config_class": "darsia.presets.workflows.config.format_registry:ImageExportFormat",  # noqa: E501
                "catalogue_class": "darsia.presets.workflows.config.catalogue.formats:FormatCatalogue",  # noqa: E501
                "catalogue_file": "formats.toml",
                "toml_key": "format",
                "toml_subkey": None,
                "is_single_section": False,
            },
            "rig": {
                "config_class": "darsia.presets.workflows.config.rig:RigConfig",  # noqa: E501
                "catalogue_class": "darsia.presets.workflows.config.catalogue.rig:RigCatalogue",  # noqa: E501
                "catalogue_file": "rig.toml",
                "toml_key": "rig",
                "toml_subkey": None,
                "is_single_section": True,
            },
        }

        if loadable_type not in loadable_config:
            self.main_window.print_log(f"Unknown loadable type: {loadable_type}")
            return None

        config_meta = loadable_config[loadable_type]

        button = QToolButton()
        button.setText("Load ▾")
        button.setPopupMode(QToolButton.InstantPopup)

        menu = QMenu(button)
        button.setMenu(menu)

        # "From TOML..." action
        def load_from_toml():
            file, _ = QFileDialog.getOpenFileName(
                self.main_window,
                f"Load {loadable_type} preset from TOML",
                "",
                "TOML Files (*.toml);;All Files (*)",
            )
            if not file:
                return
            try:
                import tomllib

                # For curvature: extract section.subsection
                # For formats: extract [[format]] array-of-tables
                is_single_section = config_meta["is_single_section"]
                with open(file, "rb") as f:
                    data = tomllib.load(f)

                if is_single_section:
                    # Curvature: extract section.subsection and call on_apply once
                    section = config_meta["toml_key"]
                    subkey = config_meta["toml_subkey"]
                    if section not in data:
                        self.main_window.print_log(
                            f"Section '{section}' not found in {file}"
                        )
                        return
                    sec = data[section]
                    if subkey not in sec:
                        self.main_window.print_log(
                            f"Sub-section '{section}.{subkey}' not found in {file}"
                        )
                        return
                    sub_sec = sec[subkey]

                    # Validate and normalize via the config class
                    module_path, class_name = config_meta["config_class"].rsplit(":", 1)
                    mod = __import__(module_path, fromlist=[class_name])
                    config_class = getattr(mod, class_name)
                    config = config_class().load(sub_sec)
                    preset_dict = config.to_dict()

                    on_apply("_singleton", preset_dict)
                    self.main_window.print_log(f"Loaded preset from {file}")
                else:
                    # Formats/etc: extract [[key]] array-of-tables and call on_apply per entry
                    toml_key = config_meta["toml_key"]
                    if toml_key not in data:
                        self.main_window.print_log(
                            f"No [[{toml_key}]] entries found in {file}"
                        )
                        return
                    entry_list = data[toml_key]
                    if not isinstance(entry_list, list):
                        self.main_window.print_log(
                            f"[[{toml_key}]] must be array-of-tables, not nested dict"
                        )
                        return

                    # Use the registry's parsing to validate each entry
                    module_path, class_name = config_meta["config_class"].rsplit(":", 1)
                    mod = __import__(module_path, fromlist=[class_name])

                    for entry_dict in entry_list:
                        try:
                            name = entry_dict.get("name", "")
                            if not name:
                                self.main_window.print_log(
                                    f"Skipped unnamed entry in {file}"
                                )
                                continue
                            # For ImageExportFormat, validate via the registry's load logic
                            # (simplified: just pass the dict to on_apply and let GUI handle
                            # errors)
                            on_apply(name, entry_dict)
                        except Exception as e:
                            self.main_window.print_log(f"Error loading entry: {e}")

                    self.main_window.print_log(
                        f"Loaded {len(entry_list)} presets from {file}"
                    )

            except Exception as e:
                self.main_window.print_log(f"Error loading preset from TOML: {e}")

        action_from_toml = menu.addAction("From TOML...")
        action_from_toml.triggered.connect(load_from_toml)

        # "From catalogue" submenu
        try:
            # Load the bundled catalogue
            catalogue_path = (
                Path(__file__).parent.parent
                / "config"
                / "catalogue"
                / config_meta["catalogue_file"]
            )
            if not catalogue_path.exists():
                # Fallback: try relative to darsia installation
                import darsia

                darsia_root = Path(darsia.__file__).parent
                catalogue_path = (
                    darsia_root
                    / "presets"
                    / "workflows"
                    / "config"
                    / "catalogue"
                    / config_meta["catalogue_file"]
                )

            # Dynamically import the catalogue class
            module_path, class_name = config_meta["catalogue_class"].rsplit(":", 1)
            mod = __import__(module_path, fromlist=[class_name])
            catalogue_class = getattr(mod, class_name)

            catalogue = catalogue_class().load(catalogue_path)
            preset_names = catalogue.names()

            if preset_names:
                submenu = menu.addMenu("From catalogue")
                for preset_name in preset_names:

                    def make_preset_loader(name):
                        def load_preset():
                            try:
                                config = catalogue.get(name)
                                # For curvature: call to_dict(); for formats: pass the object
                                # and let on_apply handle it
                                if config_meta["is_single_section"]:
                                    preset_dict = config.to_dict()
                                else:
                                    preset_dict = _format_entry_to_dict(config)

                                on_apply(name, preset_dict)
                                self.main_window.print_log(f"Loaded preset '{name}'")
                            except Exception as e:
                                self.main_window.print_log(
                                    f"Error loading preset '{name}': {e}"
                                )

                        return load_preset

                    action = submenu.addAction(preset_name)
                    action.triggered.connect(make_preset_loader(preset_name))
        except Exception as e:
            self.main_window.print_log(
                f"Warning: could not load {loadable_type} catalogue: {e}"
            )

        return button

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

        # Add Load button if this group is loadable (e.g., curvature correction)
        key_path = setting_dict["key"]  # e.g. "corrections.curvature"

        def on_apply_group(name, preset_dict):
            # For group fields (curvature), full replace via apply_partial_preset
            self.main_window.config_controller.apply_partial_preset(
                key_path, preset_dict
            )

        load_button = self._create_load_button(setting_dict, on_apply_group)
        if load_button:
            load_button_wrapper = QHBoxLayout()
            load_button_wrapper.addWidget(load_button)
            load_button_wrapper.addStretch()
            load_button_wrapper.setContentsMargins(0, 0, 0, 0)
            load_button_wrapper_widget = QWidget()
            load_button_wrapper_widget.setLayout(load_button_wrapper)
            group_form.addRow(load_button_wrapper_widget)

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
                    # Use shared helper to ensure consistent tagging with top-level loop
                    sub_inputs[sub_setting["key"]] = self._wrap_multi_row_result(
                        field_or_result
                    )
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
                row_index = group_form.rowCount() - 1
                sub_inputs[sub_setting["key"]] = field_widget
                # Flatten nested sub_inputs so they appear in parent's sub_inputs
                for nested_key, nested_widget_or_result in field_widget.get(
                    "sub_inputs", {}
                ).items():
                    sub_inputs[nested_key] = nested_widget_or_result
                # Track nested groups for depends_on wiring (same as scalars)
                unqualified_key = sub_setting["key"].rsplit(".", 1)[-1]
                field_row_map[unqualified_key] = (row_index, field_widget, sub_setting)
            else:
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
            _, driver_widget, _ = field_row_map[driver_field_key]

            # Extract the QComboBox from the composite field_widget via the stored property
            driver_combo = driver_widget.property("value_widget")
            if driver_combo is None:
                continue  # Driver is not a dropdown, skip

            # Create a closure to capture the target widget and driver_value
            # Support both single-value (string) and multi-value (list/set/tuple) comparisons
            # Handle both scalar widgets (use setRowVisible) and group result dicts
            # (setVisible)
            if isinstance(field_widget, dict) and field_widget.get("is_group_result"):
                # Nested group result: call widget.setVisible() directly
                target_widget = field_widget.get("widget")
                if target_widget is None:
                    continue

                def make_group_visibility_handler(widget, required_val):
                    def handler(current_text):
                        is_visible = (
                            current_text in required_val
                            if isinstance(required_val, (list, set, tuple))
                            else current_text == required_val
                        )
                        widget.setVisible(is_visible)

                    return handler

                handler = make_group_visibility_handler(target_widget, driver_value)
            else:
                # Scalar widget: use setRowVisible
                def make_scalar_visibility_handler(row_idx, required_val):
                    def handler(current_text):
                        is_visible = (
                            current_text in required_val
                            if isinstance(required_val, (list, set, tuple))
                            else current_text == required_val
                        )
                        group_form.setRowVisible(row_idx, is_visible)

                    return handler

                handler = make_scalar_visibility_handler(row_index, driver_value)

            # Connect the driver's value-changed signal to show/hide this row
            driver_combo.currentTextChanged.connect(handler)

            # Set initial visibility based on driver's current value
            handler(driver_combo.currentText())

        # Return (None, group_dict) where group_dict carries the group_box and metadata.
        # display_settings will check if the label is None to detect this case.
        # Mark as a group result so save_settings can identify it.
        result["widget"] = group_box
        result["is_group_result"] = True
        return None, result

    def _sync_settings_inputs_to_config_dict(self):
        """Flush all live settings widget values into main_window.config_dict.

        Shared by save_settings() (before writing to disk) and
        _render_settings_tabs() (before it deletes the current widgets and
        resets settings_inputs), so in-progress edits are never silently
        discarded by navigating between tabs before saving.
        """
        import re

        from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit

        # First pass: collect group checkbox states and determine which sub-inputs to skip
        group_active_names: dict[str, set[str]] = {}
        skip_keys: set[str] = set()

        for key, value in self.main_window.settings_inputs.items():
            if isinstance(value, dict) and "checkbox" in value:
                # Handle section-level boolean toggle (active_bool_key)
                if "bool_key" in value:
                    self.set_value(
                        self.main_window.config_dict,
                        value["bool_key"],
                        value["checkbox"].isChecked(),
                    )
                    if not value["checkbox"].isChecked():
                        skip_keys.update(value["sub_inputs"].keys())
                    continue

                # Handle field-level toggles within a section (active_list_key)
                # This is a group dict; record its active state
                active_list_key = value["active_list_key"]
                group_active_names.setdefault(active_list_key, set())
                if value["checkbox"].isChecked():
                    group_active_names[active_list_key].add(value["name"])
                else:
                    # Group is unchecked; skip saving any of its sub-inputs
                    skip_keys.update(value["sub_inputs"].keys())

            # Handle depends_on for group results: if a group is hidden (driver doesn't
            # match), skip its sub-inputs to avoid clobbering the active group's data
            elif (
                isinstance(value, dict)
                and value.get("is_group_result")
                and "depends_on" in value
            ):
                depends_on = value.get("depends_on")
                if depends_on is not None:
                    driver_field_key = depends_on.get("field")
                    driver_value = depends_on.get("value")
                    if driver_field_key is not None and driver_value is not None:
                        # Look up the driver field's current value in settings_inputs
                        # (key is something like "restoration.volume_averaging_options",
                        # driver key is "restoration.method")
                        section = key.rsplit(".", 1)[0]
                        driver_full_key = f"{section}.{driver_field_key}"
                        if driver_full_key in self.main_window.settings_inputs:
                            driver_widget = self.main_window.settings_inputs[
                                driver_full_key
                            ]
                            # Unwrap composite widget to get the real control
                            driver_control = unwrap_composite_widget(driver_widget)
                            if isinstance(driver_control, QComboBox):
                                current_val = driver_control.currentText()
                                # Check if the current value matches the dependency
                                is_active = (
                                    current_val in driver_value
                                    if isinstance(driver_value, (list, set, tuple))
                                    else current_val == driver_value
                                )
                                # If not active, skip this group's sub-inputs
                                if not is_active:
                                    skip_keys.update(value.get("sub_inputs", {}).keys())

        # Second pass: save all regular values (non-group dicts),
        # skipping unchecked group sub-inputs.
        for key, value in self.main_window.settings_inputs.items():
            # Skip group result dicts (marked with is_group_result)
            if isinstance(value, dict) and value.get("is_group_result"):
                continue
            # Skip group dicts with checkboxes (already handled above)
            if isinstance(value, dict) and "checkbox" in value:
                continue
            # Skip path_map, int_group_list, int_list_map, registry_key_list,
            # dataclass_group_map, roi_map, roi_key_list, color_key_list dicts
            # (handled below)
            if isinstance(value, dict) and (
                "path_map" in value
                or "int_group_list" in value
                or "int_list_map" in value
                or "registry_key_list" in value
                or "dataclass_group_map" in value
                or "format_key_list" in value
                or "roi_key_list" in value
                or "color_key_list" in value
            ):
                continue
            # Skip sub-inputs of unchecked groups
            if key in skip_keys:
                continue

            # Unwrap composite field widgets (wrapper widget containing type label +
            # help button) to extract the actual editable control (QLineEdit, QComboBox,
            # QCheckBox)
            value = unwrap_composite_widget(value)

            try:
                if isinstance(value, QLineEdit):
                    # Skip writing file/folder fields that are empty or still have the
                    # placeholder
                    if value.text() == NO_FILE_CHOSEN or value.text().strip() == "":
                        continue
                    # Check if this is a list field (tagged during widget creation)
                    if value.property("darsia_is_list"):
                        list_type = value.property("darsia_list_type")
                        parsed_value = _parse_list_text(value.text(), list_type)
                        self.set_value(self.main_window.config_dict, key, parsed_value)
                    else:
                        self.set_value(
                            self.main_window.config_dict,
                            key,
                            ast.literal_eval(value.text()),
                        )
                elif isinstance(value, QComboBox):
                    self.set_value(
                        self.main_window.config_dict, key, value.currentText()
                    )
                elif isinstance(value, QCheckBox):
                    self.set_value(self.main_window.config_dict, key, value.isChecked())
                elif isinstance(value, list):
                    if len(value) > 0:
                        if isinstance(value[0], QCheckBox):
                            self.set_value(
                                self.main_window.config_dict,
                                key,
                                [item.text() for item in value if item.isChecked()],
                            )
                        elif isinstance(value[0], QLineEdit):
                            self.set_value(
                                self.main_window.config_dict,
                                key,
                                [item.text() for item in value if item.text().strip()],
                            )
            except (ValueError, SyntaxError):
                if hasattr(value, "text"):
                    # Skip writing file/folder fields that are empty or still have
                    # the placeholder
                    if value.text() == NO_FILE_CHOSEN or value.text().strip() == "":
                        continue
                    self.set_value(self.main_window.config_dict, key, value.text())

        # Third pass: save path_map dicts (key -> value mappings)
        for key, value in self.main_window.settings_inputs.items():
            if isinstance(value, dict) and "path_map" in value:
                rows = value["rows"]
                result = {
                    k.text(): v.text()
                    for k, v in rows
                    if k.text().strip() and v.text().strip()
                }
                self.set_value(self.main_window.config_dict, key, result)

        # Fourth pass: parse int_group_list rows into list[list[int]]
        for key, value in self.main_window.settings_inputs.items():
            if isinstance(value, dict) and "int_group_list" in value:
                groups = []
                for edit in value["rows"]:
                    text = edit.text().strip()
                    if not text:
                        continue
                    tokens = [t for t in re.split(r"[,\s]+", text) if t]
                    try:
                        groups.append([int(t) for t in tokens])
                    except ValueError:
                        self.main_window.print_log(
                            f"Skipping invalid group '{text}' for {key}: not all-integer."
                        )
                self.set_value(self.main_window.config_dict, key, groups)

        # Fifth pass: parse int_list_map rows into dict[int, list[int]]
        for key, value in self.main_window.settings_inputs.items():
            if isinstance(value, dict) and "int_list_map" in value:
                result = {}
                for key_edit, value_edit in value["rows"]:
                    key_text = key_edit.text().strip()
                    value_text = value_edit.text().strip()
                    if not key_text or not value_text:
                        continue
                    tokens = [t for t in re.split(r"[,\s]+", value_text) if t]
                    try:
                        result[int(key_text)] = [int(t) for t in tokens]
                    except ValueError:
                        self.main_window.print_log(
                            f"Skipping invalid row '{key_text}: {value_text}' for {key}: "
                            "key and values must all be integers."
                        )
                if value.get("flatten_in_section"):
                    section = value["section"]
                    section_dict = self.main_window.config_dict.setdefault(section, {})
                    # Drop id sub-tables removed in the GUI (has "labels", not in new result)
                    stale_ids = [
                        k
                        for k, v in section_dict.items()
                        if isinstance(v, dict)
                        and "labels" in v
                        and k not in {str(i) for i in result}
                    ]
                    for k in stale_ids:
                        del section_dict[k]
                    for facies_id, labels in result.items():
                        section_dict[str(facies_id)] = {"labels": labels}
                else:
                    self.set_value(self.main_window.config_dict, key, result)

        # Tenth pass: parse registry_key_list rows into list[str]
        for key, value in self.main_window.settings_inputs.items():
            if isinstance(value, dict) and "registry_key_list" in value:
                result = []
                seen = set()
                for combo in value["rows"]:
                    text = combo.currentText().strip()
                    if text and text not in seen:
                        result.append(text)
                        seen.add(text)
                # Write as list[str] (empty list if no selections, never None)
                if result:
                    self.set_value(self.main_window.config_dict, key, result)
                else:
                    # Empty selection: delete the key (or set to empty list)
                    # For now, set to empty list to match the field's Optional nature
                    self.set_value(self.main_window.config_dict, key, None)
        # Tenth-and-a-half pass (v2): parse dataclass_group_map entries into list[dict]
        # This is the new generic handler for group-box-per-entry collections.
        for key, value in self.main_window.settings_inputs.items():
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

                        # Handle nested group results (dataclass fields)
                        if isinstance(field_widget, dict) and field_widget.get(
                            "is_group_result"
                        ):
                            # Recursively extract nested group's sub_inputs
                            nested_dict = {}
                            for sub_key, sub_widget in field_widget.get(
                                "sub_inputs", {}
                            ).items():
                                # Extract the sub-field value using the same logic as scalars
                                sub_unwrapped = unwrap_composite_widget(sub_widget)
                                sub_schema = None
                                # Try to find the schema for this sub-field
                                for fs in field_schema.get("fields", []):
                                    if (
                                        fs["key"].split(".", 1)[-1]
                                        == sub_key.split(".", 1)[-1]
                                    ):
                                        sub_schema = fs
                                        break
                                sub_default = (
                                    sub_schema.get("default") if sub_schema else None
                                )

                                if isinstance(sub_unwrapped, QCheckBox):
                                    val = sub_unwrapped.isChecked()
                                    if val is True:
                                        nested_dict[sub_key.split(".", 1)[-1]] = val
                                elif isinstance(sub_unwrapped, QComboBox):
                                    val = sub_unwrapped.currentText().strip()
                                    if val and val != sub_default:
                                        nested_dict[sub_key.split(".", 1)[-1]] = val
                                elif isinstance(sub_unwrapped, QLineEdit):
                                    val = sub_unwrapped.text().strip()
                                    if val and val != sub_default:
                                        nested_dict[sub_key.split(".", 1)[-1]] = val
                            if nested_dict:
                                entry_dict[field_name] = nested_dict
                            continue

                        # Unwrap composite widget to get the real control
                        unwrapped_widget = unwrap_composite_widget(field_widget)

                        # Extract value based on widget type
                        extracted_value = None
                        should_include = False

                        if isinstance(unwrapped_widget, QCheckBox):
                            extracted_value = unwrapped_widget.isChecked()
                            # Only include if True (checkbox default is usually False)
                            should_include = extracted_value is True
                        elif isinstance(unwrapped_widget, QComboBox):
                            text_value = unwrapped_widget.currentText().strip()
                            if text_value:
                                extracted_value = text_value
                                # Omit if equals default
                                should_include = extracted_value != field_default
                        elif isinstance(unwrapped_widget, QLineEdit):
                            text_value = unwrapped_widget.text().strip()
                            if text_value:
                                # Special handling for numeric fields
                                # (dpi, quality, compression)
                                if field_type == "int":
                                    try:
                                        extracted_value = int(text_value)
                                        # Omit if equals default
                                        should_include = (
                                            extracted_value != field_default
                                        )
                                    except ValueError:
                                        pass
                                elif field_type == "list":
                                    # Parse list fields with per-element-type coercion
                                    try:
                                        list_type = field_schema.get("list_type")
                                        extracted_value = _parse_list_text(
                                            text_value, list_type
                                        )
                                        should_include = (
                                            extracted_value != field_default
                                        )
                                    except (ValueError, SyntaxError):
                                        pass
                                elif field_type == "time":
                                    # Time fields: normalize and store as canonical string
                                    try:
                                        from darsia.presets.workflows.config.utils import (
                                            _normalize_time_string,
                                        )

                                        extracted_value = _normalize_time_string(
                                            text_value
                                        )
                                        should_include = (
                                            extracted_value != field_default
                                        )
                                    except (ValueError, AssertionError):
                                        pass
                                else:
                                    # String fields: store as string, omit if equals default
                                    extracted_value = text_value
                                    should_include = extracted_value != field_default

                        if should_include and extracted_value is not None:
                            entry_dict[field_name] = extracted_value

                    result.append(entry_dict)

                # Write result based on whether this is array-of-tables TOML
                array_key = value.get("array_key")
                if array_key:
                    # Write to array-of-tables style (e.g. [[format]], [[roi]])
                    self.main_window.config_dict[array_key] = result
                else:
                    # Write to nested table style (default)
                    self.set_value(self.main_window.config_dict, key, result)

        # Twelfth pass: parse format_key_list rows into list[str] (or single str)
        for key, value in self.main_window.settings_inputs.items():
            if isinstance(value, dict) and "format_key_list" in value:
                result = []
                seen = set()
                for combo in value["rows"]:
                    text = combo.currentText().strip()
                    if text and text not in seen:
                        result.append(text)
                        seen.add(text)
                if value.get("max_rows") == 1:
                    self.set_value(
                        self.main_window.config_dict, key, result[0] if result else None
                    )
                else:
                    self.set_value(
                        self.main_window.config_dict, key, result if result else None
                    )

        # Fourteenth pass: parse roi_key_list rows into list[str] (or single str)
        for key, value in self.main_window.settings_inputs.items():
            if isinstance(value, dict) and "roi_key_list" in value:
                result = []
                seen = set()
                for combo in value["rows"]:
                    text = combo.currentText().strip()
                    if text and text not in seen:
                        result.append(text)
                        seen.add(text)
                if value.get("max_rows") == 1:
                    self.set_value(
                        self.main_window.config_dict, key, result[0] if result else None
                    )
                else:
                    self.set_value(
                        self.main_window.config_dict, key, result if result else None
                    )

        # Fourteenth-and-a-half pass: parse color_key_list rows into list[str] (or single str)
        for key, value in self.main_window.settings_inputs.items():
            if isinstance(value, dict) and "color_key_list" in value:
                result = []
                seen = set()
                for combo in value["rows"]:
                    text = combo.currentText().strip()
                    if text and text not in seen:
                        result.append(text)
                        seen.add(text)
                if value.get("max_rows") == 1:
                    self.set_value(
                        self.main_window.config_dict, key, result[0] if result else None
                    )
                else:
                    self.set_value(
                        self.main_window.config_dict, key, result if result else None
                    )

        # Fifteenth pass: write all active lists
        for active_list_key, names in group_active_names.items():
            self.set_value(self.main_window.config_dict, active_list_key, sorted(names))

    def refresh_current_view(self):
        """Refresh the currently-displayed settings tab/view, preserving tab index.

        Replays the last-shown view (full settings or filtered action view) and
        restores the tab index so the user stays on the same tab. Called after
        config_dict mutations to reflect those changes on screen (e.g., by Load
        button or Save).
        """
        if not self.main_window._last_settings_view:
            return

        from PySide6.QtWidgets import QTabWidget

        # Capture the currently active tab index before rebuild
        current_tab_index = None
        if self.main_window.settings_layout.count() > 0:
            widget = self.main_window.settings_layout.itemAt(0).widget()
            if isinstance(widget, QTabWidget):
                current_tab_index = widget.currentIndex()

        # Replay the last view to refresh
        if self.main_window._last_settings_view[0] == "full":
            self.display_full_settings()
        elif self.main_window._last_settings_view[0] == "action":
            _, action, checked_ids = self.main_window._last_settings_view
            self.display_settings(action, checked_ids)

        # Restore the tab index if possible
        if current_tab_index is not None:
            widget = self.main_window.settings_layout.itemAt(0).widget()
            if isinstance(widget, QTabWidget):
                if 0 <= current_tab_index < widget.count():
                    widget.setCurrentIndex(current_tab_index)

    def save_settings(self):
        """Save the current settings to the loaded config file."""
        import toml

        self._sync_settings_inputs_to_config_dict()

        if self.main_window.config_file != "":
            with open(self.main_window.config_file, "w") as f:
                toml.dump(self.main_window.config_dict, f)
            self.main_window.print_log(
                f"Settings saved to {self.main_window.config_file}"
            )

            # Refresh the currently-displayed settings panel if one is open, so
            # newly added registry entries immediately appear in dropdowns etc.
            self.refresh_current_view()
        else:
            self.main_window.print_log(
                "Settings not saved, please choose a config file"
            )

    def _render_settings_tabs(self, settings_by_section):
        """Render settings_by_section dict into tabbed layout.

        Clears settings_layout and builds QTabWidget with one tab per section,
        populating form inputs via settings_factory.build_tab_form.
        Shared with both display_settings and display_full_settings.
        """
        from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout

        # Flush any pending edits into config_dict before the widgets holding
        # them are destroyed below, so switching tabs never silently drops
        # unsaved changes (e.g. newly added registry rows).
        if self.main_window.settings_inputs:
            self._sync_settings_inputs_to_config_dict()

        while self.main_window.settings_layout.count():
            child = self.main_window.settings_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.main_window.settings_inputs = {}

        # If no sections, add a message and return
        if not settings_by_section:
            self.main_window.settings_layout.addWidget(QLabel("No settings available"))
            self.main_window.settings_layout.addStretch()
            return

        # Create a tabbed interface with one tab per section
        tabs = QTabWidget()

        section_titles = {
            "registry": "Data registry",
            "format_registry": "Format registry",
            "roi": "Roi registry",
            "color": "Color registry",
        }

        # Iterate through sections in order (dict preserves insertion order in Python 3.7+)
        for section, settings_list in settings_by_section.items():
            # Create a scroll area and container for this section's fields
            tab_container = QWidget()
            tab_form = QFormLayout(tab_container)
            tab_form.setContentsMargins(8, 8, 8, 8)  # Padding on all sides

            # If this is a loadable flat section, add a Load button as the first row
            if section in SECTION_LOADABLE:
                loadable_type = SECTION_LOADABLE[section]

                def on_apply_section(_name, preset_dict):
                    self.main_window.config_controller.apply_partial_preset(
                        section, preset_dict
                    )

                setting_dict = {"loadable": loadable_type}
                load_button = self._create_load_button(setting_dict, on_apply_section)
                if load_button:
                    load_button_wrapper = QHBoxLayout()
                    load_button_wrapper.addWidget(load_button)
                    load_button_wrapper.addStretch()
                    load_button_wrapper.setContentsMargins(0, 0, 0, 0)
                    load_button_wrapper_widget = QWidget()
                    load_button_wrapper_widget.setLayout(load_button_wrapper)
                    tab_form.addRow(load_button_wrapper_widget)

            # Create form_context for multi_file/path_map dynamic row insertion
            form_context = {"form": tab_form}

            # Build rows in the form, handling grouping via group_name metadata
            self.build_tab_form(tab_form, settings_list, form_context=form_context)

            # Add stretch at the end of the section (QFormLayout doesn't auto-stretch rows)
            tab_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
            # Push content to top by adding a stretch row at the end
            tab_form.addItem(QVBoxLayout())

            # Add this section as a tab using the section_titles map
            tab_title = section_titles.get(section, section.capitalize())
            tabs.addTab(tab_container, tab_title)

        self.main_window.settings_layout.addWidget(tabs)
        self.main_window.settings_layout.addStretch()

    def display_settings(self, action, checked_ids):
        """Display the relevant settings based on the action being used."""
        self.main_window._last_settings_view = ("action", action, checked_ids)
        relevant_settings = self.get_relevant_settings(action, checked_ids)
        self._render_settings_tabs(relevant_settings)

    def display_full_settings(self):
        """Display all fixed-schema sections in the settings panel."""
        self.main_window._last_settings_view = ("full",)
        all_settings = self.get_all_settings()
        self._render_settings_tabs(all_settings)
