"""Shared qtawesome icon helper for the GUI."""

import qtawesome as qta
from PySide6.QtGui import QIcon, QPalette
from PySide6.QtWidgets import QApplication


def qta_icon(name: str, **kwargs) -> QIcon:
    """Build a QIcon from a qtawesome icon name (e.g. "fa5s.cogs")."""
    return qta.icon(name, **kwargs)


def themed_icon(name: str, *, role=None, **kwargs) -> QIcon:
    """Build a qtawesome icon colored from the current app palette.

    This reads a color from the application's current QPalette and passes it to
    qtawesome. Must be called fresh whenever the theme changes — qtawesome icons
    are baked bitmaps with the color burned in; there is no live re-tinting.

    Parameters
    ----------
    name : str
        qtawesome icon name (e.g. "fa5s.play")
    role : QPalette.ColorRole, optional
        Palette role to read the color from. Defaults to WindowText.
    **kwargs
        Additional arguments forwarded to qta_icon (scale_factor, etc).

    Returns
    -------
    QIcon
        A QIcon with the palette-derived color baked in.
    """
    if role is None:
        role = QPalette.WindowText
    pal = QApplication.instance().palette()
    color = pal.color(role)
    return qta_icon(name, color=color, **kwargs)
