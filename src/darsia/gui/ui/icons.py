"""Shared qtawesome icon helper for the GUI."""

import qtawesome as qta
from PySide6.QtGui import QIcon, QPalette
from PySide6.QtWidgets import QApplication

from .theme import muted_text_color


def qta_icon(name: str, **kwargs) -> QIcon:
    """Build a QIcon from a qtawesome icon name (e.g. "fa5s.cogs")."""
    return qta.icon(name, **kwargs)


def themed_icon(name: str, *, role=None, **kwargs) -> QIcon:
    """Build a qtawesome icon colored from the current app palette.

    This reads a color from the application's current QPalette and passes it to
    qtawesome. Must be called fresh whenever the theme changes — qtawesome icons
    are baked bitmaps with the color burned in; there is no live re-tinting.

    Also bakes an explicit, theme-agnostic muted color for QIcon's Disabled mode
    (via muted_text_color), so a toolbar action's icon actually greys out when
    action.setEnabled(False) is called. Left to qtawesome's own fallback, this
    would read QPalette's Disabled/Text color group — which this app's Light
    theme never sets distinctly from Active (see apply_theme), so a disabled
    icon would otherwise render in the same color as an enabled one.

    Parameters
    ----------
    name : str
        Qtawesome icon name (e.g. "fa5s.play")
    role : QPalette.ColorRole, optional
        Palette role to read the color from. Defaults to WindowText.
    **kwargs
        Additional arguments forwarded to qta_icon (scale_factor, etc).

    Returns
    -------
    QIcon
        A QIcon with the palette-derived color (and a muted disabled-color)
        baked in.
    """
    if role is None:
        role = QPalette.WindowText
    pal = QApplication.instance().palette()
    color = pal.color(role)
    disabled_color = muted_text_color(pal)
    return qta_icon(name, color=color, color_disabled=disabled_color, **kwargs)
