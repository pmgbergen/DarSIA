"""Entry point for the GUI application.

Run the GUI using one of the following methods:

1. From anywhere (recommended):
    python -m darsia.gui

2. From the darsia.gui directory:
    python __main__.py
"""

from .gui_main import MainWindow
from PySide6.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication()
    window = MainWindow()
    window.show()
    app.exec()
