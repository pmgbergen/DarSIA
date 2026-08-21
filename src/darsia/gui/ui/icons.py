"""Shared qtawesome icon helper for the GUI."""

import qtawesome as qta
from PySide6.QtGui import QIcon


def qta_icon(name: str, **kwargs) -> QIcon:
    """Build a QIcon from a qtawesome icon name (e.g. "fa5s.cogs")."""
    return qta.icon(name, **kwargs)
