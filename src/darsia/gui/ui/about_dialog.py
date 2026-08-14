import webbrowser
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout

GITHUB_URL = "https://github.com/pmgbergen/DarSIA"


class AboutDialog(QDialog):
    """About dialog: logo, brief description, and GitHub link."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About DarSIA")
        self.setFixedWidth(360)

        layout = QVBoxLayout(self)

        logo_path = Path(__file__).parent.parent / "logo.png"
        if logo_path.exists():
            logo_label = QLabel()
            logo_label.setPixmap(
                QPixmap(str(logo_path)).scaledToWidth(280, Qt.SmoothTransformation)
            )
            logo_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(logo_label)

        text_label = QLabel(
            "DarSIA is an open-source Python library for quantitative image "
            "analysis of experimental data, developed for porous-media flow "
            "and related applications."
        )
        text_label.setWordWrap(True)
        layout.addWidget(text_label)

        link_button = QPushButton(GITHUB_URL)
        link_button.setFlat(True)
        link_button.setStyleSheet("color: #1565c0; text-align: left;")
        link_button.setCursor(Qt.PointingHandCursor)
        link_button.clicked.connect(lambda: webbrowser.open(GITHUB_URL))
        layout.addWidget(link_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)
