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

    def create_file_chooser(self, display_name, file_filter, is_directory):
        """Create a file/folder chooser UI element (button + path label).

        Parameters
        ----------
        display_name : str
            Display name for the button and dialog
        file_filter : str
            File filter for the dialog (e.g., "TOML Files (*.toml);;All Files (*)")
        is_directory : bool
            If True, opens directory selection dialog; if False, opens file dialog

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

    def create_multi_file_input(self, setting_dict):
        """Create a variable-size file list input with add/remove buttons.

        Parameters
        ----------
        setting_dict : dict
            Setting configuration dictionary with 'key' field

        Returns
        -------
        tuple
            (setting_container, file_edits) - the UI widget and list of line edits
        """
        setting = setting_dict["key"]
        values = self.main_window.settings_factory.get_value(
            self.main_window.config_dict, setting
        )

        setting_container = QWidget()
        setting_layout = QVBoxLayout(setting_container)
        setting_layout.setContentsMargins(0, 0, 0, 0)

        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        setting_label = QLabel(setting)
        add_button = QPushButton("Add file")
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
            path_edit.setPlaceholderText("Select a file or type a path")
            if initial_value:
                path_edit.setText(str(initial_value))
            remove_button = QPushButton("Remove")
            remove_button.setMaximumWidth(80)

            def browse():
                selected_path, _ = QFileDialog.getOpenFileName(
                    self.main_window,
                    f"Select file for {setting}",
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
