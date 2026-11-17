"""Color swatch for RGB fields: shows the color of a text field and opens a color dialog.

The text field ("r, g, b", values 0-255) stays the editable source of the value; the
swatch follows it and writes the color chosen in Qt's color dialog (color field, RGB and
HSV spin boxes, hex code, screen picker) back into it.

"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QColorDialog, QLineEdit, QToolButton, QWidget

SWATCH_SIZE = QSize(36, 16)
"""Size of the colored box."""


def parse_rgb(text: str) -> tuple[int, int, int] | None:
    """RGB color from text such as "255, 113, 107" or "[255, 113, 107]".

    Parameters
    ----------
    text : str
        Three integers in [0, 255], separated by commas.

    Returns
    -------
    tuple[int, int, int] or None
        The color; None if the text is not a valid RGB color.

    """
    parts = [part.strip() for part in text.strip().strip("[]()").split(",")]
    if len(parts) != 3:
        return None
    try:
        values = tuple(int(part) for part in parts)
    except ValueError:
        return None
    if any(value < 0 or value > 255 for value in values):
        return None
    return values


def format_rgb(color: QColor) -> str:
    """Text of a color as "r, g, b".

    Parameters
    ----------
    color : QColor
        The color.

    Returns
    -------
    str
        Red, green and blue in [0, 255].

    """
    return f"{color.red()}, {color.green()}, {color.blue()}"


class ColorSwatchButton(QToolButton):
    """Button showing the color of an RGB text field; a click opens a color dialog.

    Parameters
    ----------
    line_edit : QLineEdit
        Text field holding the color as "r, g, b".
    parent : QWidget, optional
        Parent widget.

    """

    def __init__(self, line_edit: QLineEdit, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.line_edit = line_edit
        self.setIconSize(SWATCH_SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(self.choose_color)
        line_edit.textChanged.connect(self.update_swatch)
        self.update_swatch()

    def update_swatch(self, *args) -> None:
        """Paint the box in the color of the text field (crossed out if invalid)."""
        rgb = parse_rgb(self.line_edit.text())
        pixmap = QPixmap(SWATCH_SIZE)
        pixmap.fill(QColor(*rgb) if rgb is not None else Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        outline = self.palette().text().color()
        painter.setPen(QPen(outline, 1))
        width, height = SWATCH_SIZE.width() - 1, SWATCH_SIZE.height() - 1
        painter.drawRect(0, 0, width, height)
        if rgb is None:
            painter.drawLine(0, 0, width, height)
            painter.drawLine(0, height, width, 0)
        painter.end()
        self.setIcon(QIcon(pixmap))
        if rgb is None:
            self.setToolTip(
                "No valid color: enter r, g, b (0-255), or click to choose."
            )
        else:
            color = QColor(*rgb)
            self.setToolTip(f"{format_rgb(color)} ({color.name()}): click to choose.")

    def choose_color(self) -> None:
        """Open the color dialog; the chosen color replaces the text."""
        rgb = parse_rgb(self.line_edit.text())
        initial = QColor(*rgb) if rgb is not None else QColor(255, 255, 255)
        color = QColorDialog.getColor(
            initial,
            self.window(),
            "Choose color",
            QColorDialog.ColorDialogOption.DontUseNativeDialog,
        )
        if not color.isValid():
            return  # cancelled
        self.line_edit.setText(format_rgb(color))
        # Same validation as after typing
        self.line_edit.editingFinished.emit()
