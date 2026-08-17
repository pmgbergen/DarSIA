"""Theme switcher for Light/Dark/System modes with persistence."""

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor, QPalette

_original_style = None
_original_palette = None


def apply_theme(app, mode: str) -> None:
    """Apply a theme (System/Light/Dark) to the application.

    Parameters
    ----------
    app : QApplication
        The application instance to theme.
    mode : str
        One of "System", "Light", or "Dark".
    """
    global _original_style, _original_palette

    if mode == "System":
        # Restore original state
        if _original_style is not None:
            app.setStyle(_original_style)
        if _original_palette is not None:
            app.setPalette(_original_palette)
        app.styleHints().setColorScheme(Qt.ColorScheme.Unknown)
    elif mode == "Light":
        # Save original state on first call
        if _original_style is None:
            _original_style = app.style().objectName()
        if _original_palette is None:
            _original_palette = app.palette()

        app.setStyle("Fusion")
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, Qt.black)
        palette.setColor(QPalette.Base, Qt.white)
        palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.black)
        palette.setColor(QPalette.Text, Qt.black)
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, Qt.black)
        palette.setColor(QPalette.BrightText, Qt.white)
        palette.setColor(QPalette.Link, QColor(0, 0, 255))
        palette.setColor(QPalette.Highlight, QColor(76, 163, 224))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        app.setPalette(palette)
        app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    elif mode == "Dark":
        # Save original state on first call
        if _original_style is None:
            _original_style = app.style().objectName()
        if _original_palette is None:
            _original_palette = app.palette()

        app.setStyle("Fusion")
        palette = QPalette()
        dark_color = QColor(53, 53, 53)
        disabled_color = QColor(127, 127, 127)
        palette.setColor(QPalette.Window, dark_color)
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, dark_color)
        palette.setColor(QPalette.ToolTipBase, dark_color)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, dark_color)
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.white)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        palette.setColor(QPalette.Disabled, QPalette.Text, disabled_color)
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, disabled_color)
        app.setPalette(palette)
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark)


def get_theme() -> str:
    """Get the saved theme preference.

    Returns
    -------
    str
        One of "System" (default), "Light", or "Dark".
    """
    settings = QSettings()
    value = settings.value("theme", "System")
    if isinstance(value, str):
        return value
    return "System"


def set_theme(mode: str) -> None:
    """Save the theme preference.

    Parameters
    ----------
    mode : str
        One of "System", "Light", or "Dark".
    """
    settings = QSettings()
    settings.setValue("theme", mode)
