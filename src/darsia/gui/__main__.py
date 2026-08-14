"""Entry point for the GUI application.

Run the GUI using one of the following methods:

1. From anywhere (recommended):
    python -m darsia.gui

2. From the darsia.gui directory:
    python __main__.py
"""

from PySide6.QtWidgets import QApplication

from .ui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication()
    app.setOrganizationName("DarSIA")
    app.setApplicationName("DarSIA GUI")
    window = MainWindow()
    window.show()
    app.exec()
