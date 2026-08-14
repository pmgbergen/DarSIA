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


class FileDialogHelper:
    """Helper for creating file/folder selection UI components."""

    def __init__(self, main_window):
        self.main_window = main_window

    def create_file_chooser(
        self, display_name, file_filter, is_directory, setting_dict=None
    ):
        """Create a file/folder chooser UI element (button + path label).

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
            pre-fills the path label from the loaded config or default value.

        Returns
        -------
        tuple
            (chooser_container, path_label) - the UI widget and the label to update
        """
        if not file_filter:
            file_filter = "All Files (*)"

        chooser_container = QWidget()
        chooser_layout = QHBoxLayout(chooser_container)
        chooser_layout.setContentsMargins(0, 5, 0, 5)

        # Browse button
        browse_button = QPushButton(f"Browse {display_name}")
        browse_button.setMinimumWidth(200)

        # Path label to display selected path
        path_label = QLineEdit("No file chosen")
        path_label.setStyleSheet("color: white;")

        # Pre-fill from config or default if setting_dict is provided
        if setting_dict is not None:
            value = self.main_window.settings_factory.get_value(
                self.main_window.config_dict, setting_dict["key"]
            )
            if value is None:
                value = setting_dict.get("default")
            if value:
                path_label.setText(str(value))
                path_label.setStyleSheet("color: white;")

        # Store label reference for updating
        key = display_name.lower().replace(" ", "_")
        self.main_window.chosen_files[key] = {
            "path": "",
            "label": path_label,
            "is_directory": is_directory,
            "filter": file_filter,
        }

        # Connect button to file dialog
        browse_button.clicked.connect(lambda: self.main_window.browse_file(key))

        chooser_layout.addWidget(browse_button)
        chooser_layout.addWidget(path_label)
        chooser_layout.addStretch()
        return chooser_container, path_label

    def create_multi_file_input(self, setting_dict, is_directory=False):
        """Create a variable-size file/folder list input with add/remove buttons.

        Parameters
        ----------
        setting_dict : dict
            Setting configuration dictionary with 'key' field
        is_directory : bool, optional
            If True, opens directory selection dialog; if False, opens file dialog

        Returns
        -------
        tuple
            (setting_container, file_edits) - the UI widget and list of line edits
        """
        key = setting_dict["key"]
        display_name = setting_dict.get("name", key)
        values = self.main_window.settings_factory.get_value(
            self.main_window.config_dict, key
        )
        if values is None:
            values = setting_dict.get("default")

        setting_container = QWidget()
        setting_layout = QVBoxLayout(setting_container)
        setting_layout.setContentsMargins(0, 0, 0, 0)

        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        setting_label = QLabel(display_name)
        add_button_text = "Add folder" if is_directory else "Add file"
        add_button = QPushButton(add_button_text)
        header_layout.addWidget(setting_label)
        header_layout.addStretch()
        header_layout.addWidget(add_button)
        setting_layout.addWidget(header_container)

        rows_container = QWidget()
        rows_layout = QVBoxLayout(rows_container)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        setting_layout.addWidget(rows_container)

        file_edits = []
        file_rows = []

        def refresh_remove_buttons():
            show_remove = len(file_rows) > 1
            for row in file_rows:
                row["remove_button"].setVisible(show_remove)

        def add_row(initial_value=""):
            row_container = QWidget()
            row_layout = QHBoxLayout(row_container)
            row_layout.setContentsMargins(0, 0, 0, 0)

            browse_button = QPushButton("Browse")
            browse_button.setMinimumWidth(100)
            path_edit = QLineEdit()
            placeholder = "Select a folder or type a path" if is_directory else "Select a file or type a path"
            path_edit.setPlaceholderText(placeholder)
            if initial_value:
                path_edit.setText(str(initial_value))
            remove_button = QPushButton("Remove")
            remove_button.setMaximumWidth(80)

            def browse():
                if is_directory:
                    selected_path = QFileDialog.getExistingDirectory(
                        self.main_window,
                        f"Select folder for {display_name}",
                        path_edit.text() if path_edit.text() else "",
                    )
                else:
                    selected_path, _ = QFileDialog.getOpenFileName(
                        self.main_window,
                        f"Select file for {display_name}",
                        path_edit.text() if path_edit.text() else "",
                        "All Files (*)",
                    )
                if selected_path:
                    path_edit.setText(selected_path)

            def remove():
                row_container.deleteLater()
                if row_data in file_rows:
                    file_rows.remove(row_data)
                if path_edit in file_edits:
                    file_edits.remove(path_edit)
                refresh_remove_buttons()

            browse_button.clicked.connect(browse)
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

        return setting_container, file_edits
