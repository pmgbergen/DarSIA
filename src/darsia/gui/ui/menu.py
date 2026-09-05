from functools import partial

from PySide6.QtGui import QAction, QActionGroup, QKeySequence

from .recent_files import clear_recent_configs, get_recent_configs
from .theme import get_theme


class MenuBuilder:
    """Builds the main window's menu bar from a declarative structure.

    To add a new menu, add an entry to build(); to add an action to an existing
    menu, add an _add_action call. No other wiring is required.
    """

    def __init__(self, main_window):
        self.main_window = main_window

    def build(self):
        menu_bar = self.main_window.menuBar()

        file_menu = menu_bar.addMenu("&File")
        self.new_action = self._add_action(
            file_menu, "&New", self.main_window.config_controller.new_config, "Ctrl+N"
        )
        file_menu.addSeparator()
        self.open_action = self._add_action(
            file_menu,
            "&Open Config...",
            self.main_window.config_controller.open_config,
            "Ctrl+O",
        )
        self.recent_menu = file_menu.addMenu("Open &Recent")
        self.recent_menu.aboutToShow.connect(self._populate_recent_menu)
        file_menu.addSeparator()
        self.save_action = self._add_action(
            file_menu,
            "&Save Config",
            self.main_window.settings_factory.save_settings,
            "Ctrl+S",
        )
        self._add_action(
            file_menu,
            "Save Config &As...",
            self.main_window.config_controller.save_config_as,
            "Ctrl+Shift+S",
        )
        file_menu.addSeparator()
        self._add_action(file_menu, "&Close", self.main_window.close, "Ctrl+W")
        # Also bind Ctrl+Q to close (common shortcut)
        quit_action = QAction(self.main_window)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.main_window.close)
        self.main_window.addAction(quit_action)

        settings_menu = menu_bar.addMenu("&Settings")
        self.open_full_config_action = self._add_action(
            settings_menu,
            "Open &Full Config",
            self.main_window._on_open_full_config,
            "Ctrl+E",
        )

        run_menu = menu_bar.addMenu("&Run")
        self.play_action = self._add_action(
            run_menu,
            "&Run Selected Workflow",
            self.main_window.run_selected_workflow,
        )
        # Cover both the main-keyboard Enter/Return key and the numpad Enter key.
        self.play_action.setShortcuts(
            [QKeySequence("Ctrl+Return"), QKeySequence("Ctrl+Enter")]
        )
        self.stop_action = self._add_action(
            run_menu,
            "&Stop Workflow",
            self.main_window.abort_selected_workflow,
            "Ctrl+Escape",
        )
        self.stop_action.setEnabled(False)

        view_menu = menu_bar.addMenu("&View")
        theme_menu = view_menu.addMenu("Switch &Theme")
        theme_group = QActionGroup(self.main_window)
        theme_group.setExclusive(True)

        current_theme = get_theme()
        for theme_name in ["System", "Light", "Dark"]:
            action = self._add_action(
                theme_menu,
                f"&{theme_name}",
                partial(self.main_window.set_theme, theme_name),
            )
            action.setCheckable(True)
            action.setChecked(theme_name == current_theme)
            theme_group.addAction(action)

        # Logging dock's built-in toggle action (checkable, tracks the dock's
        # shown/hidden state automatically) — same pattern as the View dock.
        self.show_logging_action = self.main_window.log_dock.toggleViewAction()
        self.show_logging_action.setText("Show &Logging")
        self.show_logging_action.setShortcut(QKeySequence("Ctrl+L"))
        view_menu.addAction(self.show_logging_action)

        # View dock's built-in toggle action (checkable, tracks the dock's
        # shown/hidden state automatically).
        self.streaming_toggle_action = (
            self.main_window.streaming_dock.toggleViewAction()
        )
        self.streaming_toggle_action.setText("Show &View")
        self.streaming_toggle_action.setShortcut(QKeySequence("Ctrl+P"))
        view_menu.addAction(self.streaming_toggle_action)

        help_menu = menu_bar.addMenu("&Help")
        self._add_action(help_menu, "&About", self.main_window.show_about_dialog)

    def _add_action(self, menu, text, handler, shortcut=None):
        action = QAction(text, self.main_window)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(handler)
        menu.addAction(action)
        return action

    def _populate_recent_menu(self):
        self.recent_menu.clear()
        try:
            recent = get_recent_configs()
        except Exception as e:
            self.main_window.print_log(f"Error loading recent files: {e}")
            recent = []
        if not recent:
            empty = QAction("No recent files", self.main_window)
            empty.setEnabled(False)
            self.recent_menu.addAction(empty)
            return
        for path in recent:
            action = QAction(path, self.main_window)
            action.triggered.connect(
                partial(self.main_window.config_controller.open_recent_config, path)
            )
            self.recent_menu.addAction(action)
        self.recent_menu.addSeparator()
        clear_action = QAction("Clear Recent", self.main_window)
        clear_action.triggered.connect(clear_recent_configs)
        self.recent_menu.addAction(clear_action)
