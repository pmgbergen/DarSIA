"""File dialog and file selection utilities for DarSIA GUI."""

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .help import build_help_column


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
        path_edit = QLineEdit("No file chosen")

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
            browse_button.clicked.connect(lambda: self.main_window.browse_file(key))

        # Build composite field widget
        field_widget = QWidget()
        field_layout = QHBoxLayout(field_widget)
        field_layout.setContentsMargins(0, 0, 0, 0)
        field_layout.setSpacing(4)

        field_layout.addWidget(browse_button)
        field_layout.addWidget(path_edit, stretch=1)

        # Right column: help button or spacer (fixed 40px)
        field_layout.addWidget(build_help_column(setting_dict))

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
        form_context : dict, optional
            If provided, contains "form" (QFormLayout) for dynamic row insertion/removal.
            If None, uses internal QVBoxLayout (fallback for backward compatibility).

        Returns
        -------
        tuple
            (label_text, field_widget) if form_context is None (internal layout),
            or (label_text, file_edits_list) if form_context is provided (form-based).
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

        # If form_context provided, use QFormLayout-based insertion
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
                remove_button = QPushButton("Remove")
                remove_button.setMaximumWidth(80)

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

                browse_button.clicked.connect(
                    lambda: self._browse_for_path(
                        is_directory,
                        f"Select {'folder' if is_directory else 'file'} for {display_name}",
                        path_edit,
                    )
                )
                remove_button.clicked.connect(remove)

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

            # Connect add_button to add_row closure (which will handle form insertion)
            add_button.clicked.connect(lambda: add_row())

            # Return the add_button as the field_widget; build_tab_form will insert
            # the header row. Once the header is in the form, we can insert data rows
            # (in the deferred pre-fill below).
            # NOTE: We can't pre-fill data rows here because the header row isn't in
            # the form yet. Instead, we'll pre-fill them lazily when the GUI is shown
            # (by connecting to a deferred callback). For now, just return the add_button
            # and file_edits list for save_settings.

            # Actually, to pre-fill, we can use a deferred insertion that runs after the
            # header is added. The simplest: call add_row() after build_tab_form has had
            # a chance to add the header. We'll use a QTimer to defer the pre-fill:
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

        # Fallback: use internal QVBoxLayout (backward compat)
        else:
            setting_container = QWidget()
            setting_layout = QVBoxLayout(setting_container)
            setting_layout.setContentsMargins(0, 0, 0, 0)

            header_container = QWidget()
            header_layout = QHBoxLayout(header_container)
            header_layout.setContentsMargins(0, 0, 0, 0)
            add_button = QPushButton(add_button_text)
            header_layout.addStretch()
            header_layout.addWidget(add_button)
            setting_layout.addWidget(header_container)

            rows_container = QWidget()
            rows_layout = QVBoxLayout(rows_container)
            rows_layout.setContentsMargins(0, 0, 0, 0)
            setting_layout.addWidget(rows_container)

            def add_row(initial_value=""):
                row_container = QWidget()
                row_layout = QHBoxLayout(row_container)
                row_layout.setContentsMargins(0, 0, 0, 0)

                browse_button = QPushButton("Browse")
                browse_button.setMinimumWidth(100)
                path_edit = QLineEdit()
                placeholder = (
                    "Select a folder or type a path"
                    if is_directory
                    else "Select a file or type a path"
                )
                path_edit.setPlaceholderText(placeholder)
                if initial_value:
                    path_edit.setText(str(initial_value))
                remove_button = QPushButton("Remove")
                remove_button.setMaximumWidth(80)

                def remove():
                    row_container.deleteLater()
                    if row_data in file_rows:
                        file_rows.remove(row_data)
                    if path_edit in file_edits:
                        file_edits.remove(path_edit)
                    refresh_remove_buttons()

                browse_button.clicked.connect(
                    lambda: self._browse_for_path(
                        is_directory,
                        f"Select {'folder' if is_directory else 'file'} for {display_name}",
                        path_edit,
                    )
                )
                remove_button.clicked.connect(remove)

                row_layout.addWidget(browse_button)
                row_layout.addWidget(path_edit)
                row_layout.addWidget(remove_button)

                rows_layout.addWidget(row_container)

                row_data = {
                    "container": row_container,
                    "remove_button": remove_button,
                }
                file_rows.append(row_data)
                file_edits.append(path_edit)
                refresh_remove_buttons()

            add_button.clicked.connect(lambda: add_row())

            if isinstance(values, list) and values:
                for value in values:
                    add_row(value)
            else:
                add_row("")

            return display_name, file_edits

    def create_path_map_input(
        self,
        setting_dict,
        key_is_directory=False,
        value_is_directory=False,
        form_context=None,
    ):
        """Create a dict[Path, Path] editor with two-column rows (key, value).

        Parameters
        ----------
        setting_dict : dict
            Setting configuration dictionary with 'key' field
        key_is_directory : bool, optional
            If True, key column opens directory selection; if False, file selection
        value_is_directory : bool, optional
            If True, value column opens directory selection; if False, file selection
        form_context : dict, optional
            If provided, contains "form" (QFormLayout) for dynamic row insertion/removal.
            If None, uses internal QVBoxLayout (fallback for backward compatibility).

        Returns
        -------
        tuple
            (label_text, row_pairs_list) where row_pairs_list is [(key_edit, value_edit), ...].
        """
        key = setting_dict["key"]
        display_name = setting_dict.get("name", key)
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

        # If form_context provided, use QFormLayout-based insertion
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

                # Key column
                key_browse_button = QPushButton("Browse")
                key_browse_button.setMaximumWidth(80)
                key_edit = QLineEdit()
                key_placeholder = (
                    "Select folder or type path"
                    if key_is_directory
                    else "Select file or type path"
                )
                key_edit.setPlaceholderText(key_placeholder)
                if initial_key:
                    key_edit.setText(str(initial_key))

                # Value column
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

                # Remove button
                remove_button = QPushButton("Remove")
                remove_button.setMaximumWidth(80)

                def remove():
                    self._remove_form_row(
                        form,
                        row_widget,
                        row_data,
                        row_data_list,
                        (key_edit, value_edit),
                        row_pairs,
                        refresh_remove_buttons,
                    )

                key_browse_button.clicked.connect(
                    lambda: self._browse_for_path(
                        key_is_directory,
                        (
                            """Select key """
                            f"""({"folder" if key_is_directory else "file"}) """
                            f"""for {display_name}"""
                        ),
                        key_edit,
                    )
                )
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
                remove_button.clicked.connect(remove)

                row_layout.addWidget(key_browse_button)
                row_layout.addWidget(key_edit, stretch=1)
                row_layout.addWidget(value_browse_button)
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
                        add_row(k, v)
                else:
                    add_row("")

            QTimer.singleShot(0, deferred_prefill)

            # Return enriched dict: widget for form insertion, rows for save_settings
            return display_name, {
                "widget": header_widget,
                "path_map": True,
                "rows": row_pairs,
            }

        # Fallback: use internal QVBoxLayout (backward compat)
        else:
            setting_container = QWidget()
            setting_layout = QVBoxLayout(setting_container)
            setting_layout.setContentsMargins(0, 0, 0, 0)

            header_container = QWidget()
            header_layout = QHBoxLayout(header_container)
            header_layout.setContentsMargins(0, 0, 0, 0)
            setting_label = QLabel(display_name)
            add_button = QPushButton("Add row")
            header_layout.addWidget(setting_label)
            header_layout.addStretch()
            header_layout.addWidget(add_button)
            setting_layout.addWidget(header_container)

            rows_container = QWidget()
            rows_layout = QVBoxLayout(rows_container)
            rows_layout.setContentsMargins(0, 0, 0, 0)
            setting_layout.addWidget(rows_container)

            def add_row(initial_key="", initial_value=""):
                row_container = QWidget()
                row_layout = QHBoxLayout(row_container)
                row_layout.setContentsMargins(0, 0, 0, 0)

                # Key column
                key_browse_button = QPushButton("Browse")
                key_browse_button.setMaximumWidth(80)
                key_edit = QLineEdit()
                key_placeholder = (
                    "Select folder or type path"
                    if key_is_directory
                    else "Select file or type path"
                )
                key_edit.setPlaceholderText(key_placeholder)
                if initial_key:
                    key_edit.setText(str(initial_key))

                # Value column
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

                # Remove button
                remove_button = QPushButton("Remove")
                remove_button.setMaximumWidth(80)

                def remove():
                    row_container.deleteLater()
                    if row_data in row_data_list:
                        row_data_list.remove(row_data)
                    if (key_edit, value_edit) in row_pairs:
                        row_pairs.remove((key_edit, value_edit))
                    refresh_remove_buttons()

                key_browse_button.clicked.connect(
                    lambda: self._browse_for_path(
                        key_is_directory,
                        (
                            """Select key """
                            """({'folder' if key_is_directory else 'file'}) """
                            f"""for {display_name}"""
                        ),
                        key_edit,
                    )
                )
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
                remove_button.clicked.connect(remove)

                row_layout.addWidget(key_browse_button)
                row_layout.addWidget(key_edit)
                row_layout.addWidget(value_browse_button)
                row_layout.addWidget(value_edit)
                row_layout.addWidget(remove_button)

                rows_layout.addWidget(row_container)

                row_data = {
                    "container": row_container,
                    "remove_button": remove_button,
                }
                row_data_list.append(row_data)
                row_pairs.append((key_edit, value_edit))
                refresh_remove_buttons()

            add_button.clicked.connect(lambda: add_row())

            if isinstance(values, dict) and values:
                for k, v in values.items():
                    add_row(k, v)
            else:
                add_row("")

            return display_name, {"path_map": True, "rows": row_pairs}
