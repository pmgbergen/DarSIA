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
        self.main_window.sidebar.deselect_all()
        self.main_window.settings_factory.display_full_settings()

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
        self.main_window.print_log(f"Config saved as: {file}")

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
        self.main_window.print_log(f"Config loaded: {file}")
        self.main_window.sidebar.deselect_all()
        self.main_window.settings_factory.display_full_settings()

    def apply_partial_preset(self, key_path: str, preset_dict: dict) -> None:
        """Apply a partial preset (e.g. curvature correction config) to the current config.

        Fully replaces the target sub-dict in config_dict, then updates the
        'active' list within that sub-dict to match exactly the stages present
        in the preset. Preserves the currently-displayed tab/view and tab index.

        Args:
            key_path: Dot-separated config path (e.g. "corrections.curvature").
            preset_dict: Normalized dict from a preset (e.g. CurvatureCorrectionConfig.to_dict()).
        """
        # Flush any pending edits from other tabs/sections *before* mutating config_dict,
        # so in-progress edits elsewhere are preserved (not lost to sync-on-rebuild).
        self.main_window.settings_factory._sync_settings_inputs_to_config_dict()

        keys = key_path.split(".")
        config = self.main_window.config_dict

        # Navigate to parent and key name
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        target_key = keys[-1]

        # Full replace: set the entire section to the preset
        config[target_key] = preset_dict.copy()

        # Update the 'active' list within the target to match exactly the stages present
        # (required so stage checkboxes reflect the loaded preset, not stale lists).
        # For curvature: add 'active' key with list of stages present.
        if target_key == "curvature":
            stages = [k for k in preset_dict.keys() if k in ["init", "crop", "bulge", "stretch"]]
            if stages:
                config[target_key]["active"] = stages
            elif "active" in config[target_key]:
                # No stages in preset, so clear the active list
                del config[target_key]["active"]

        # Clear settings_inputs so the upcoming rebuild (refresh_current_view) doesn't
        # sync stale widgets back into config_dict, overwriting the preset we just applied.
        # Nothing is lost: these widgets are about to be destroyed anyway, and
        # _render_settings_tabs will reset this dict immediately after.
        self.main_window.settings_factory.main_window.settings_inputs = {}

        # Refresh the current view (full or filtered) while preserving tab index.
        self.main_window.settings_factory.refresh_current_view()
