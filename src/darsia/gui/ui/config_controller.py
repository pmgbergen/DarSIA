"""Config file lifecycle management for DarSIA GUI."""

from pathlib import Path

import toml
from PySide6.QtWidgets import QFileDialog

from .recent_files import add_recent_config, remove_recent_config


class ConfigController:
    """Manages config file operations: new, open, save, recent files."""

    def __init__(self, main_window):
        self.main_window = main_window

    def new_config(self):
        """Create a new empty config file at a chosen path and open it."""
        file, _ = QFileDialog.getSaveFileName(
            self.main_window,
            "New Config File",
            "",
            "TOML Files (*.toml);;All Files (*)",
        )
        if not file:
            return
        try:
            with open(file, "w") as f:
                toml.dump({}, f)
        except Exception as e:
            self.main_window.print_log(f"Error creating config file: {e}")
            return
        self.main_window.config_file = file
        self.main_window.config_path_label.setText(file)
        self.main_window.config_dict = {}
        add_recent_config(file)
        self.main_window.print_log(f"New config created and opened: {file}")

    def open_config(self):
        """Open a config file via dialog and load it immediately."""
        file, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Open Config File",
            "",
            "TOML Files (*.toml);;All Files (*)",
        )
        if not file:
            return
        self.main_window.config_path_label.setText(file)
        self.load_config()

    def save_config_as(self):
        """Save current settings to a new config file chosen via dialog."""
        file, _ = QFileDialog.getSaveFileName(
            self.main_window, "Save Config As", "", "TOML Files (*.toml);;All Files (*)"
        )
        if not file:
            return
        self.main_window.config_file = file
        self.main_window.config_path_label.setText(file)
        add_recent_config(file)
        self.main_window.settings_factory.save_settings()

    def open_recent_config(self, path):
        """Open a config file from the recent-files list."""
        if not Path(path).exists():
            self.main_window.print_log(f"Recent config file no longer exists: {path}")
            remove_recent_config(path)
            return
        self.main_window.config_path_label.setText(path)
        self.load_config()

    def load_config(self):
        """Load the config file chosen in the GUI."""
        file = self.main_window.config_path_label.text()
        if not file:
            self.main_window.print_log(
                """No config file selected. """
                """Use <b><i>File > Open Config</i></b> to select a config file."""
            )
            return
        try:
            with open(file, "r") as f:
                self.main_window.config_dict = toml.load(f)
        except Exception as e:
            self.main_window.print_log(f"Error loading config file: {e}")
            return
        self.main_window.config_file = file
        add_recent_config(file)
        self.main_window.print_log("Config loaded")
