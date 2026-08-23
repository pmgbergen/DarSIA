"""Entry point for the GUI application.

Run the GUI using one of the following methods:

1. From anywhere (recommended):
    python -m darsia.gui
    or
    uv run darsia

2. From the darsia.gui directory:
    python __main__.py
"""

from PySide6.QtWidgets import QApplication

from .ui.main_window import MainWindow
from .ui.theme import apply_theme, get_theme


def main() -> None:
    """Initialize and run the DarSIA GUI application.

    Creates a QApplication, applies the theme, instantiates MainWindow,
    and starts the event loop.
    """
    app = QApplication()
    app.setOrganizationName("DarSIA")
    app.setApplicationName("DarSIA GUI")
    apply_theme(app, get_theme())
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
