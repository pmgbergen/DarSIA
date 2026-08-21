"""Shared qtawesome icon helper for the GUI."""

from PySide6.QtGui import QIcon
import qtawesome as qta


def qta_icon(name: str, **kwargs) -> QIcon:
    """Build a QIcon from a qtawesome icon name (e.g. "fa5s.cogs")."""
    return qta.icon(name, **kwargs)
