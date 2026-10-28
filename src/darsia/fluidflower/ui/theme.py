"""Shared visual theme for the FluidFlower Qt applications."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QStyle, QWidget


APPLICATION_STYLESHEET = """
QMainWindow, QWidget { background:#171a1d; color:#e6eaed; font-size:14px; }
QLabel#homeTitle { font-size:30px; font-weight:650; }
QLabel#workspaceTitle { font-size:20px; font-weight:650; }
QLabel#pageTitle { font-size:24px; font-weight:650; }
QLabel#sectionTitle { font-size:18px; font-weight:600; margin-top:8px; }
QLabel#secondaryText { color:#aeb8c0; }
QLabel#pathPreview { color:#8fb4c9; font-size:12px; }
QLineEdit#pathPreview { color:#8fb4c9; font-size:12px; }
QLabel#moduleStatus { font-size:16px; }
QLabel#integrationNotice { background:#202a30; border-left:4px solid #4d91c4; padding:10px; color:#c8d4dc; }
QLabel#savedState { background:#183128; border-left:4px solid #66c69a; padding:10px; color:#d8f0e5; }
QLabel#pendingState { background:#352d1d; border-left:4px solid #e3b65c; padding:10px; color:#f1e4c4; }
QLabel#incompleteState { background:#382323; border-left:4px solid #d36b6b; padding:10px; color:#f2d4d4; }
QPushButton { background:#263039; border:1px solid #4b5964; padding:7px 12px; }
QPushButton:hover { background:#303d47; border-color:#6c8190; }
QPushButton:disabled { color:#6f777d; background:#1c2023; border-color:#343a3f; }
QPushButton#scopeModeButton:checked { background:#294d63; border-color:#5aa6d6; color:#ffffff; }
QLineEdit, QComboBox, QTextEdit { background:#101316; border:1px solid #46525b; padding:6px; selection-background-color:#27668d; }
QTableWidget, QListWidget { background:#121518; border:1px solid #363f46; alternate-background-color:#191e22; }
QHeaderView::section { background:#252d33; color:#dfe4e8; border:0; border-right:1px solid #3b454d; padding:7px; }
QListWidget#moduleNavigation { background:#14181b; border:0; border-right:1px solid #343c42; padding-top:6px; }
QListWidget#moduleNavigation::item { padding:10px 12px; }
QListWidget#moduleNavigation::item:selected { background:#273944; color:#ffffff; border-left:4px solid #5aa6d6; }
QToolTip { background:#f5f5f5; color:#111; border:1px solid #777; }
"""


def apply_application_theme(widget: QWidget) -> None:
    """Apply the shared DarSIA FluidFlower visual contract to a top-level widget."""

    widget.setStyleSheet(APPLICATION_STYLESHEET)


def application_icon() -> QIcon:
    """Return the common FluidFlower application icon."""

    icon_path = Path(__file__).with_name("assets") / "fluidflower_run10_4h_icon.png"
    icon = QIcon(str(icon_path))
    if not icon.isNull():
        return icon
    return QApplication.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)


__all__ = [
    "APPLICATION_STYLESHEET",
    "application_icon",
    "apply_application_theme",
]
