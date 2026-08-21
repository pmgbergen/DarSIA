"""Toolbar builder for icon buttons (New, Open, Save, Play, Stop) below the menu bar."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QStyle

from .icons import qta_icon

_ICON_DIR = Path(__file__).parent.parent / "icons"

_STANDARD_ICONS = {
    "new": QStyle.StandardPixmap.SP_FileIcon,
    "open": QStyle.StandardPixmap.SP_DirOpenIcon,
    "save": QStyle.StandardPixmap.SP_DriveFDIcon,
    "settings": QStyle.StandardPixmap.SP_FileDialogDetailedView,
}

_QTA_ICONS = {
    "play": "fa5s.play",
    "stop": "fa5s.stop",
}


class ToolbarBuilder:
    """Builds the main toolbar, reusing the QActions MenuBuilder created.

    Custom icons: drop new.png / open.png / save.png into gui/icons/ and they
    are used automatically in place of Qt's built-in standard icons - no code
    changes required.
    """

    def __init__(self, main_window, menu_builder):
        self.main_window = main_window
        self.menu_builder = menu_builder

    def build(self):
        toolbar = self.main_window.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)

        self._configure(self.menu_builder.new_action, "new", "New (Ctrl+N)")
        self._configure(
            self.menu_builder.open_action, "open", "Open Config... (Ctrl+O)"
        )
        self._configure(self.menu_builder.save_action, "save", "Save Config (Ctrl+S)")
        self._configure(
            self.menu_builder.open_full_config_action,
            "settings",
            "Open Full Config",
        )

        toolbar.addAction(self.menu_builder.new_action)
        toolbar.addAction(self.menu_builder.open_action)
        toolbar.addAction(self.menu_builder.save_action)
        toolbar.addAction(self.menu_builder.open_full_config_action)

        # Add separator and Play/Stop actions for workflow control
        toolbar.addSeparator()

        self.play_action = QAction("Play")
        self.play_action.triggered.connect(self._on_play)
        self._configure(self.play_action, "play", "Run Selected Workflow")
        toolbar.addAction(self.play_action)

        self.stop_action = QAction("Stop")
        self.stop_action.triggered.connect(self._on_stop)
        self.stop_action.setEnabled(False)
        self._configure(self.stop_action, "stop", "Stop Workflow")
        toolbar.addAction(self.stop_action)

    def _configure(self, action, key, tooltip):
        action.setIcon(self._load_icon(key))
        action.setToolTip(tooltip)

    def _on_play(self):
        """Handle Play button: dispatch to selected workflow."""
        if self.main_window.selected_action is None:
            self.main_window.print_log("Select an item in the sidebar first.")
            return

        tab_manager = self.main_window.action_dispatch.get(
            self.main_window.selected_action
        )
        if tab_manager:
            tab_manager.on_run_clicked()

    def _on_stop(self):
        """Handle Stop button: abort selected workflow."""
        if self.main_window.selected_action is None:
            self.main_window.print_log("No workflow running.")
            return

        tab_manager = self.main_window.action_dispatch.get(
            self.main_window.selected_action
        )
        if tab_manager:
            tab_manager.on_abort_clicked()

    def _load_icon(self, key):
        custom_path = _ICON_DIR / f"{key}.png"
        if custom_path.exists():
            return QIcon(str(custom_path))

        # Try qtawesome icon
        if key in _QTA_ICONS:
            return qta_icon(_QTA_ICONS[key])

        # Fallback to standard Qt icon
        return self.main_window.style().standardIcon(_STANDARD_ICONS[key])
