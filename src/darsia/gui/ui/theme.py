"""Theme switcher for Light/Dark/System modes with persistence."""

from PySide6.QtCore import QObject, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

_original_style = None
_original_palette = None


class _ThemeSignal(QObject):
    """Singleton signal emitted when theme changes."""

    theme_changed = Signal(str)


theme_signal = _ThemeSignal()


def muted_text_color(pal: QPalette, weight: float = 0.55) -> QColor:
    """Blend WindowText toward Window background for theme-agnostic 'muted' text.

    QPalette has no built-in secondary/muted-text role. This blends the foreground
    text color toward the background color at a fixed ratio, self-inverting correctly
    in both Light and Dark modes without per-theme branching.

    Parameters
    ----------
    pal : QPalette
        The application palette (from QApplication.instance().palette()).
    weight : float
        Blend weight (0-1) toward background; 0.55 is a good default for readable muted text.

    Returns
    -------
    QColor
        Blended color.
    """
    fg = pal.color(QPalette.WindowText)
    bg = pal.color(QPalette.Window)
    r = int(fg.red() * (1 - weight) + bg.red() * weight)
    g = int(fg.green() * (1 - weight) + bg.green() * weight)
    b = int(fg.blue() * (1 - weight) + bg.blue() * weight)
    return QColor(r, g, b)


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
        palette.setColor(QPalette.Button, QColor(200, 200, 200))
        palette.setColor(QPalette.ButtonText, Qt.black)
        palette.setColor(QPalette.BrightText, Qt.white)
        palette.setColor(QPalette.Link, QColor(0, 0, 255))
        palette.setColor(QPalette.Highlight, QColor(76, 163, 224))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        palette.setColor(QPalette.Mid, QColor(200, 200, 200))
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
        palette.setColor(QPalette.Button, QColor(80, 80, 80))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.white)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        palette.setColor(QPalette.Disabled, QPalette.Text, disabled_color)
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, disabled_color)
        palette.setColor(QPalette.Mid, QColor(100, 100, 100))
        app.setPalette(palette)
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark)

    # Emit theme-change signal so already-built widgets can refresh (especially for
    # icon rebuilding)
    theme_signal.theme_changed.emit(mode)

    # Force full repaint of every widget to avoid containers (Sidebar, QScrollArea, etc.)
    # lagging one theme-switch behind. This is centralized, dynamic, and self-healing for
    # any future widgets added without explicit theme-change wiring.
    for widget in QApplication.instance().allWidgets():
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()


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
