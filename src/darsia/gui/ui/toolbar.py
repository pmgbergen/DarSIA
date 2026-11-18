"""Toolbar builder below the menu: icon buttons (New, Open, Save, Play, Stop,
Results, View, Logging) and the light/dark theme switch."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPalette

from .icons import themed_icon
from .theme import is_dark, theme_signal
from .widgets import ToggleSwitch

_ICON_DIR = Path(__file__).parent.parent / "icons"

_QTA_ICONS = {
    "new": "fa5s.file",
    "open": "fa5s.folder-open",
    "save": "fa5s.save",
    "settings": "fa5s.sliders-h",
    "wizard": "fa5s.magic",
    "play": "fa5s.play",
    "stop": "fa5s.stop",
    "results": "fa5s.folder",
    "stream": "fa5s.stream",
    "log": "fa5s.align-left",
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

        # Wizard-wand button: opens the guided wizard for whichever sidebar
        # category is currently selected. Greyed out (see icons.themed_icon's
        # baked-in disabled color) whenever that category has none registered.
        toolbar.addSeparator()
        self.wizard_action = self.menu_builder.current_wizard_action
        self._configure(self.wizard_action, "wizard", "Run Wizard for Current Category")
        toolbar.addAction(self.wizard_action)

        # Add separator and Play/Stop actions for workflow control (created by
        # MenuBuilder so they also appear, with shortcuts, in the Run menu).
        toolbar.addSeparator()

        self.play_action = self.menu_builder.play_action
        self._configure(self.play_action, "play", "Run Selected Workflow (Ctrl+Return)")
        toolbar.addAction(self.play_action)

        self.stop_action = self.menu_builder.stop_action
        self._configure(self.stop_action, "stop", "Stop Workflow (Ctrl+Escape)")
        toolbar.addAction(self.stop_action)

        toolbar.addSeparator()
        self._configure(
            self.menu_builder.open_results_action, "results", "Open Results Folder"
        )
        toolbar.addAction(self.menu_builder.open_results_action)

        # Reuse each dock's own checkable toggleViewAction so these buttons
        # stay in sync with their shortcuts and View-menu items.
        toolbar.addSeparator()
        self._configure(
            self.menu_builder.streaming_toggle_action,
            "stream",
            "Toggle View (Ctrl+P)",
        )
        toolbar.addAction(self.menu_builder.streaming_toggle_action)

        self._configure(
            self.menu_builder.show_logging_action,
            "log",
            "Toggle Logging (Ctrl+L)",
        )
        toolbar.addAction(self.menu_builder.show_logging_action)

        # Light/dark switch (knob left = light, right = dark). It shows the
        # current look, so from the System theme it reflects the system colors.
        toolbar.addSeparator()
        self.theme_switch = ToggleSwitch(checked=is_dark(), inner=self._theme_icons())
        self.theme_switch.setToolTip("Switch Light / Dark Theme")
        self.theme_switch.toggled.connect(
            lambda dark: self.main_window.set_theme("Dark" if dark else "Light")
        )
        toolbar.addWidget(self.theme_switch)

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
        # Follow theme changes from anywhere (View menu, or the switch itself).
        self.theme_switch.set_inner(*self._theme_icons())
        self.theme_switch.set_checked(is_dark(), emit=False)

    @staticmethod
    def _theme_icons():
        """Sun and moon inside the switch track, in the knob (window) color."""
        return (
            themed_icon("fa5s.sun", role=QPalette.Window),
            themed_icon("fa5s.moon", role=QPalette.Window),
        )

    def _load_icon(self, key):
        custom_path = _ICON_DIR / f"{key}.png"
        if custom_path.exists():
            return QIcon(str(custom_path))

        # Try qtawesome icon (palette-aware)
        if key in _QTA_ICONS:
            return themed_icon(_QTA_ICONS[key])

        return QIcon()
