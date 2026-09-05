"""Generic scale-to-fit image display widget for the DarSIA GUI."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class ImageCanvas(QWidget):
    """Load-from-path, scale-to-fit, keep-aspect-ratio image display, with a
    plain-text placeholder state for "nothing to show yet"."""

    def __init__(self, placeholder_text: str = ""):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel(placeholder_text)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setWordWrap(True)
        self._label.setMinimumSize(150, 150)
        layout.addWidget(self._label, stretch=1)
        self._original_pixmap: QPixmap | None = None

    def set_image_path(self, path: Path | None) -> None:
        """Display the image at path, or clear to an empty state if None."""
        if path is None:
            self.set_message("")
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.set_message(f"Could not load image:\n{path}")
            return
        self._original_pixmap = pixmap
        self._rescale()

    def set_message(self, text: str) -> None:
        """Clear any image and show a plain-text placeholder/status message."""
        self._original_pixmap = None
        self._label.setPixmap(QPixmap())
        self._label.setText(text)

    def _rescale(self) -> None:
        if self._original_pixmap is None:
            return
        self._label.setPixmap(
            self._original_pixmap.scaled(
                self._label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()
