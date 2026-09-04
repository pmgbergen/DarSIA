"""Toolbar builder for icon buttons (New, Open, Save, Play, Stop) below the menu bar."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from .icons import themed_icon
from .theme import theme_signal

_ICON_DIR = Path(__file__).parent.parent / "icons"

_QTA_ICONS = {
    "new": "fa5s.file",
    "open": "fa5s.folder-open",
    "save": "fa5s.save",
    "settings": "fa5s.sliders-h",
    "play": "fa5s.play",
    "stop": "fa5s.stop",
    "stream": "fa5s.stream",
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
        self._themed_actions = {}  # key -> QAction (for live refresh)

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

        # Add separator and Play/Stop actions for workflow control (created by
        # MenuBuilder so they also appear, with shortcuts, in the Run menu).
        toolbar.addSeparator()

        self.play_action = self.menu_builder.play_action
        self._configure(self.play_action, "play", "Run Selected Workflow (Ctrl+Return)")
        toolbar.addAction(self.play_action)

        self.stop_action = self.menu_builder.stop_action
        self._configure(self.stop_action, "stop", "Stop Workflow (Ctrl+Escape)")
        toolbar.addAction(self.stop_action)

        # Reuse the dock's own checkable toggleViewAction so this button stays
        # in sync with Ctrl+P/View menu/edge button with no extra bookkeeping.
        toolbar.addSeparator()
        self._configure(
            self.menu_builder.streaming_toggle_action,
            "stream",
            "Toggle Streaming Preview (Ctrl+P)",
        )
        toolbar.addAction(self.menu_builder.streaming_toggle_action)

        theme_signal.theme_changed.connect(self.refresh_icons)

    def _configure(self, action, key, tooltip):
        action.setIcon(self._load_icon(key))
        action.setToolTip(tooltip)
        if key in _QTA_ICONS:
            self._themed_actions[key] = action

    def refresh_icons(self):
        """Rebuild all qtawesome-backed icons from current palette."""
        for key, action in self._themed_actions.items():
            action.setIcon(self._load_icon(key))

    def _load_icon(self, key):
        custom_path = _ICON_DIR / f"{key}.png"
        if custom_path.exists():
            return QIcon(str(custom_path))

        # Try qtawesome icon (palette-aware)
        if key in _QTA_ICONS:
            return themed_icon(_QTA_ICONS[key])

        return QIcon()
