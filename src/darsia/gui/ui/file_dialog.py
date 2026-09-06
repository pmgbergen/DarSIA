"""File dialog and file selection utilities for DarSIA GUI."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from .help import build_help_column
from .table_viewer import TABLE_LOADERS, TableViewerDialog
from .theme import danger_color, theme_signal

NO_FILE_CHOSEN = "No file chosen"


def make_remove_button(on_click) -> QPushButton:
    """Build a styled destructive 'Remove' button, quiet until hovered.

    Shared by every entry/row editor (registries, multi-file/path-map rows) so
    a destructive action never looks identical to the neighboring 'Add'
    button: a muted danger-colored outline instead of the default flat fill,
    filling solid only on hover.
    """
    button = QPushButton("Remove")
    button.setCursor(Qt.PointingHandCursor)

    def refresh_style():
        color = danger_color(QApplication.instance().palette()).name()
        button.setStyleSheet(
            f"QPushButton {{ color: {color}; border: 1px solid {color}; "
            f"background: transparent; border-radius: 3px; }}"
            f"QPushButton:hover {{ background-color: {color}; color: white; }}"
        )

    refresh_style()
    theme_signal.theme_changed.connect(refresh_style)
    button.clicked.connect(on_click)
    return button


class FileDialogHelper:
    """Helper for creating file/folder selection UI components."""

    def __init__(self, main_window):
        self.main_window = main_window

    def _browse_for_path(self, is_directory, title, line_edit):
        """Open a file/folder dialog and write the selected path into line_edit."""
        if is_directory:
            selected = QFileDialog.getExistingDirectory(
                self.main_window, title, line_edit.text() if line_edit.text() else ""
            )
        else:
            selected, _ = QFileDialog.getOpenFileName(
                self.main_window,
                title,
                line_edit.text() if line_edit.text() else "",
                "All Files (*)",
            )
        if selected:
            line_edit.setText(selected)

    def browse_file(self, key):
        """Open file/folder dialog and store selected path."""
        file_info = self.main_window.chosen_files[key]
        is_directory = file_info["is_directory"]
        file_filter = file_info["filter"]

        if is_directory:
            selected_path = QFileDialog.getExistingDirectory(
                self.main_window, f"Select {key.replace('_', ' ')}"
            )
        else:
            selected_path, _ = QFileDialog.getOpenFileName(
                self.main_window, f"Select {key.replace('_', ' ')}", "", file_filter
            )

        if selected_path:
            self.main_window.chosen_files[key]["path"] = selected_path
            file_info["label"].setText(selected_path)
            file_info["label"].setStyleSheet("color: white;")
            self.main_window.print_log(f"Selected {key}: {selected_path}")

    def _remove_form_row(
        self,
        form,
        row_widget,
        row_data,
        row_data_list,
        removed_value,
        value_list,
        refresh_fn,
    ):
        """Remove a dynamically-added row from a QFormLayout and its tracking lists."""
        row_idx, _ = form.getWidgetPosition(row_widget)
        form.removeRow(row_idx)
        if row_data in row_data_list:
            row_data_list.remove(row_data)
        if removed_value in value_list:
            value_list.remove(removed_value)
        refresh_fn()

    def create_file_chooser(
        self, display_name, file_filter, is_directory, setting_dict=None
    ):
        """Create a file/folder chooser UI element (browse button + path edit).

        Parameters
        ----------
        display_name : str
            Display name for the button and dialog
        file_filter : str
            File filter for the dialog (e.g., "TOML Files (*.toml);;All Files (*)")
        is_directory : bool
            If True, opens directory selection dialog; if False, opens file dialog
        setting_dict : dict, optional
            Setting configuration dict with "key" and "default"; when provided,
            pre-fills the path edit from the loaded config or default value.

        Returns
        -------
        tuple
            (label_text, field_widget) where field_widget is a composite HBox:
            [browse_button, path_edit (stretch=1), help_button_or_spacer (fixed 40px)]
        """
        if not file_filter:
            file_filter = "All Files (*)"

        # Browse button
        browse_button = QPushButton("Browse")
        browse_button.setMaximumWidth(100)

        # Path edit to display/edit selected path
        path_edit = QLineEdit()

        # Set placeholder text (greyed-out hint) from metadata or default
        if setting_dict and "metadata" in setting_dict:
            placeholder = setting_dict["metadata"].get("placeholder")
            if placeholder:
                path_edit.setPlaceholderText(placeholder)

        if not path_edit.placeholderText():
            # Fallback placeholder if none provided in metadata
            placeholder = (
                "Select a folder or type a path"
                if is_directory
                else "Select a file or type a path"
            )
            path_edit.setPlaceholderText(placeholder)

        # Pre-fill from config or default if setting_dict is provided
        if setting_dict is not None:
            value = self.main_window.settings_factory.get_value(
                self.main_window.config_dict, setting_dict["key"]
            )
            if value is None:
                value = setting_dict.get("default")
            if value:
                path_edit.setText(str(value))

            # Store label reference for updating (backward compatibility with browse_file)
            key = display_name.lower().replace(" ", "_")
            self.main_window.chosen_files[key] = {
                "path": "",
                "label": path_edit,
                "is_directory": is_directory,
                "filter": file_filter,
            }
            browse_button.clicked.connect(lambda: self.browse_file(key))

        # Build composite field widget
        field_widget = QWidget()
        field_layout = QHBoxLayout(field_widget)
        field_layout.setContentsMargins(0, 0, 0, 0)
        field_layout.setSpacing(4)

        field_layout.addWidget(browse_button)
        field_layout.addWidget(path_edit, stretch=1)

        # Optional "View" button for table viewers
        if setting_dict and setting_dict.get("table_viewer"):
            view_button = QPushButton("View")
            view_button.setMaximumWidth(80)

            def open_table_viewer():
                path_text = path_edit.text()
                loader_key = setting_dict.get("table_viewer")
                if loader_key not in TABLE_LOADERS:
                    self.main_window.print_log(
                        f"Unknown table viewer format: {loader_key}"
                    )
                    return
                loader = TABLE_LOADERS[loader_key]
                try:
                    df = None
                    if path_text:
                        df = loader(Path(path_text))
                    title = setting_dict.get("name", loader_key)
                    dialog = TableViewerDialog(
                        parent=self.main_window, title=title, dataframe=df
                    )
                    dialog.exec()
                except Exception as e:
                    self.main_window.print_log(f"Error opening table viewer: {e}")
                    import traceback

                    self.main_window.print_log(traceback.format_exc())

            view_button.clicked.connect(open_table_viewer)
            field_layout.addWidget(view_button)

        # Right column: help button or spacer (fixed 40px)
        field_layout.addWidget(build_help_column(setting_dict))

        # Store reference to the real control for unwrapping in sync
        field_widget.setProperty("value_widget", path_edit)

        return display_name, field_widget

    def create_multi_file_input(
        self, setting_dict, is_directory=False, form_context=None
    ):
        """Create a variable-size file/folder list input with add/remove buttons.

        Parameters
        ----------
        setting_dict : dict
            Setting configuration dictionary with 'key' field
        is_directory : bool, optional
            If True, opens directory selection dialog; if False, opens file dialog
        form_context : dict
            Contains "form" (QFormLayout) for dynamic row insertion/removal.

        Returns
        -------
        tuple
            (label_text, result_dict) where result_dict carries "widget" (header)
            and "rows" (list of QLineEdit) for save_settings.
        """
        key = setting_dict["key"]
        display_name = setting_dict.get("name", key)
        values = self.main_window.settings_factory.get_value(
            self.main_window.config_dict, key
        )
        if values is None:
            values = setting_dict.get("default")

        file_edits = []
        file_rows = []  # Track row data (widget, remove_button)

        def refresh_remove_buttons():
            show_remove = len(file_rows) > 1
            for row in file_rows:
                row["remove_button"].setVisible(show_remove)

        add_button_text = "Add folder" if is_directory else "Add file"
        add_button = QPushButton(add_button_text)

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

            browse_button = QPushButton("Browse")
            browse_button.setMaximumWidth(80)
            path_edit = QLineEdit()
            placeholder = (
                "Select a folder or type a path"
                if is_directory
                else "Select a file or type a path"
            )
            path_edit.setPlaceholderText(placeholder)
            if initial_value:
                path_edit.setText(str(initial_value))

            def remove():
                self._remove_form_row(
                    form,
                    row_widget,
                    row_data,
                    file_rows,
                    path_edit,
                    file_edits,
                    refresh_remove_buttons,
                )

            remove_button = make_remove_button(remove)
            remove_button.setMaximumWidth(80)

            browse_button.clicked.connect(
                lambda: self._browse_for_path(
                    is_directory,
                    f"Select {'folder' if is_directory else 'file'} for {display_name}",
                    path_edit,
                )
            )

            row_layout.addWidget(browse_button)
            row_layout.addWidget(path_edit, stretch=1)
            row_layout.addWidget(remove_button)

            # Find the correct insertion index: after the header_widget header row,
            # then after last data row
            header_idx, _ = form.getWidgetPosition(header_widget)
            if file_rows:
                last_idx, _ = form.getWidgetPosition(file_rows[-1]["widget"])
                insert_idx = last_idx + 1
            else:
                # Insert right after header row
                insert_idx = header_idx + 1

            form.insertRow(insert_idx, "", row_widget)

            row_data = {
                "widget": row_widget,
                "remove_button": remove_button,
            }
            file_rows.append(row_data)
            file_edits.append(path_edit)
            refresh_remove_buttons()

        add_button.clicked.connect(lambda: add_row())

        # The header row isn't in the form yet, so data rows can't be pre-filled
        # here. Defer the pre-fill via QTimer so it runs after build_tab_form has
        # inserted the header row.
        from PySide6.QtCore import QTimer

        def deferred_prefill():
            if isinstance(values, list) and values:
                for value in values:
                    add_row(value)
            else:
                add_row("")

        QTimer.singleShot(0, deferred_prefill)

        # Return enriched dict: widget for form insertion, rows for save_settings
        return display_name, {"widget": header_widget, "rows": file_edits}

    def create_path_map_input(
        self,
        setting_dict,
        value_is_directory=False,
        key_source=None,
        form_context=None,
    ):
        """Create a per-folder dict[Path, Path] editor with two-column rows.

        The key column is locked: rows are generated 1:1 from the settings list
        named by ``key_source`` (read-only, no Browse button, no manual
        add/remove). When that list is empty, the field shows a placeholder
        prompting the user to add folders in the Data tab.

        Parameters
        ----------
        setting_dict : dict
            Setting configuration dictionary with 'key' field
        value_is_directory : bool, optional
            If True, value column opens directory selection; if False, file selection
        key_source : str, optional
            Dotted settings-key (e.g. "data.folders") of the list-valued setting
            that supplies the (read-only) key column.
        form_context : dict
            Contains "form" (QFormLayout) for dynamic row insertion/removal.

        Returns
        -------
        tuple
            (label_text, result_dict) where result_dict carries "widget" (header),
            the "path_map" tag and "rows" ([(key_edit, value_edit), ...]).
        """
        key = setting_dict["key"]
        display_name = setting_dict.get("name", key)
        values = self.main_window.settings_factory.get_value(
            self.main_window.config_dict, key
        )
        if values is None:
            values = setting_dict.get("default")

        row_pairs = []  # List of (key_edit, value_edit) tuples
        row_data_list = []  # Track row data (widget)

        locked_keys = []
        if key_source:
            source_values = self.main_window.settings_factory.get_value(
                self.main_window.config_dict, key_source
            )
            locked_keys = [str(v) for v in source_values] if source_values else []

        form = form_context["form"]

        # Header widget: the first locked row (and the help button) are folded
        # directly into the header, so the field takes exactly len(locked_keys)
        # rows rather than reserving an extra, otherwise-empty header row.
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)

        def insert_row(row_widget):
            # Find the correct insertion index: after the header_widget header row
            header_idx, _ = form.getWidgetPosition(header_widget)
            if row_data_list:
                last_idx, _ = form.getWidgetPosition(row_data_list[-1]["widget"])
                insert_idx = last_idx + 1
            else:
                # Insert right after header row
                insert_idx = header_idx + 1
            form.insertRow(insert_idx, "", row_widget)

        def build_locked_value_field(initial_value=""):
            """Browse button + editable value field shared by every locked row,
            including the one embedded directly in the header."""
            value_browse_button = QPushButton("Browse")
            value_browse_button.setMaximumWidth(80)
            value_edit = QLineEdit()
            value_placeholder = (
                "Select folder or type path"
                if value_is_directory
                else "Select file or type path"
            )
            value_edit.setPlaceholderText(value_placeholder)
            if initial_value:
                value_edit.setText(str(initial_value))
            value_browse_button.clicked.connect(
                lambda: self._browse_for_path(
                    value_is_directory,
                    (
                        """Select value """
                        f"""({"folder" if value_is_directory else "file"}) """
                        f"""for {display_name}"""
                    ),
                    value_edit,
                )
            )
            return value_browse_button, value_edit

        def add_locked_row(folder, initial_value=""):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            # Key column: read-only, sourced from key_source — not user-editable.
            key_edit = QLineEdit(str(folder))
            key_edit.setReadOnly(True)

            value_browse_button, value_edit = build_locked_value_field(initial_value)

            row_layout.addWidget(key_edit, stretch=1)
            row_layout.addWidget(value_browse_button)
            row_layout.addWidget(value_edit, stretch=1)

            insert_row(row_widget)

            row_data_list.append({"widget": row_widget})
            row_pairs.append((key_edit, value_edit))

        # Defer pre-fill until after header row is added to form
        from PySide6.QtCore import QTimer

        if locked_keys:
            # Fold the first locked row into the header itself, so the field
            # takes exactly len(locked_keys) rows, not one extra blank row.
            saved = values if isinstance(values, dict) else {}
            first_folder = locked_keys[0]
            key_edit = QLineEdit(str(first_folder))
            key_edit.setReadOnly(True)
            value_browse_button, value_edit = build_locked_value_field(
                saved.get(first_folder, "")
            )
            header_layout.addWidget(key_edit, stretch=1)
            header_layout.addWidget(value_browse_button)
            header_layout.addWidget(value_edit, stretch=1)
            header_layout.addWidget(build_help_column(setting_dict))
            row_data_list.append({"widget": header_widget})
            row_pairs.append((key_edit, value_edit))

            def deferred_prefill():
                for folder in locked_keys[1:]:
                    add_locked_row(folder, saved.get(folder, ""))

        else:
            # No data folders configured yet.
            header_layout.addWidget(
                QLabel("No data folders configured — add folders in the Data tab."),
                stretch=1,
            )
            header_layout.addWidget(build_help_column(setting_dict))

            def deferred_prefill():
                return None

        QTimer.singleShot(0, deferred_prefill)

        # Return enriched dict: widget for form insertion, rows for save_settings
        return display_name, {
            "widget": header_widget,
            "path_map": True,
            "rows": row_pairs,
        }
