from functools import partial

from PySide6.QtGui import QAction

from .recent_files import clear_recent_configs, get_recent_configs


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
        self._add_action(file_menu, "&New", self.main_window.new_config)
        file_menu.addSeparator()
        self._add_action(file_menu, "&Open Config...", self.main_window.open_config)
        self.recent_menu = file_menu.addMenu("Open &Recent")
        self.recent_menu.aboutToShow.connect(self._populate_recent_menu)
        file_menu.addSeparator()
        self._add_action(file_menu, "&Save Config", self.main_window.save_settings)
        self._add_action(
            file_menu, "Save Config &As...", self.main_window.save_config_as
        )
        file_menu.addSeparator()
        self._add_action(file_menu, "&Close", self.main_window.close)

        help_menu = menu_bar.addMenu("&Help")
        self._add_action(help_menu, "&About", self.main_window.show_about_dialog)

    def _add_action(self, menu, text, handler):
        action = QAction(text, self.main_window)
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
            action.triggered.connect(partial(self.main_window.open_recent_config, path))
            self.recent_menu.addAction(action)
        self.recent_menu.addSeparator()
        clear_action = QAction("Clear Recent", self.main_window)
        clear_action.triggered.connect(clear_recent_configs)
        self.recent_menu.addAction(clear_action)
