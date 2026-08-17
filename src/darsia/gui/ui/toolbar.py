"""Toolbar builder for icon buttons (New, Open, Save) below the menu bar."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QStyle

_ICON_DIR = Path(__file__).parent.parent / "icons"

_STANDARD_ICONS = {
    "new": QStyle.StandardPixmap.SP_FileIcon,
    "open": QStyle.StandardPixmap.SP_DirOpenIcon,
    "save": QStyle.StandardPixmap.SP_DriveFDIcon,
    "settings": QStyle.StandardPixmap.SP_FileDialogDetailedView,
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

    def _configure(self, action, key, tooltip):
        action.setIcon(self._load_icon(key))
        action.setToolTip(tooltip)

    def _load_icon(self, key):
        custom_path = _ICON_DIR / f"{key}.png"
        if custom_path.exists():
            return QIcon(str(custom_path))
        return self.main_window.style().standardIcon(_STANDARD_ICONS[key])
