import webbrowser

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class HelpPopup(QFrame):
    """Custom styled help popup that appears on hover."""

    def __init__(self, help_text):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet(
            """
            HelpPopup {
                background-color: #f0f0f0;
                border: 1px solid #888;
                border-radius: 4px;
                padding: 8px;
            }
            HelpPopup QLabel {
                color: #333;
            }
        """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Help text label
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        help_label.setMaximumWidth(300)
        layout.addWidget(help_label)

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.hide()


class HelpButton(QPushButton):
    """Custom button that shows a help popup on hover and opens link on click."""

    def __init__(self, help_text, link_url=None):
        super().__init__("?")
        self.help_text = help_text
        self.link_url = link_url
        self.popup = None
        self.hover_timer = None

        # Style the help button as a square
        self.setFixedSize(32, 32)

        # Apply different styles based on whether a link is provided
        if link_url:
            self.setStyleSheet(
                """
                QPushButton {
                    background-color: #0d47a1;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 16px;
                    padding: 0px;
                }
                QPushButton:hover {
                    background-color: #1565c0;
                }
                QPushButton:pressed {
                    background-color: #0c3aa3;
                }
            """
            )
            self.setEnabled(True)
        else:
            self.setStyleSheet(
                """
                QPushButton {
                    background-color: #999999;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 16px;
                    padding: 0px;
                }
                QPushButton:hover {
                    background-color: #555555;
                }
                QPushButton:pressed {
                    background-color: #555555;
                }
            """
            )
            self.setEnabled(False)

        self.setFocusPolicy(Qt.NoFocus)
        self.clicked.connect(self.on_click)

    def on_click(self):
        """Handle button click: open link if available."""
        if self.link_url:
            webbrowser.open(self.link_url)
            if self.popup:
                self.popup.hide()

    def enterEvent(self, event):
        """Show popup after 1 second of hovering."""
        if self.hover_timer is None:
            self.hover_timer = QTimer()
            self.hover_timer.setSingleShot(True)
            self.hover_timer.timeout.connect(self.show_popup)
        self.hover_timer.start(1000)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Hide popup when mouse leaves."""
        if self.hover_timer:
            self.hover_timer.stop()
        if self.popup:
            self.popup.hide()
        super().leaveEvent(event)

    def show_popup(self):
        """Display the help popup near the button."""
        if self.popup is None:
            self.popup = HelpPopup(self.help_text)

        # Position popup below and to the right of the button
        pos = self.mapToGlobal(self.rect().bottomRight())
        self.popup.move(pos.x() - 200, pos.y() + 5)
        self.popup.show()


def build_help_column(setting_dict=None) -> QWidget:
    """Fixed-width (40px) column: a HelpButton if setting_dict has 'help', else a spacer.

    Used as the right-hand column of every field/header row so Browse buttons,
    value editors, and Add buttons all line up regardless of whether a field has help text.
    """
    right_column = QWidget()
    right_layout = QHBoxLayout(right_column)
    right_layout.setContentsMargins(0, 0, 0, 0)

    help_text = setting_dict.get("help") if setting_dict else None
    if help_text:
        link_url = setting_dict.get("link")
        right_layout.addWidget(HelpButton(help_text, link_url))
    else:
        right_layout.addStretch()

    right_column.setFixedWidth(40)
    return right_column
