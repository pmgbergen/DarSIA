from PySide6.QtGui import QAction


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
