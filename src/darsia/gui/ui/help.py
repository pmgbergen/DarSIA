import webbrowser

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .theme import theme_signal


class HelpPopup(QFrame):
    """Custom styled help popup that appears on hover."""

    def __init__(self, help_text):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self._help_text = help_text

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Help text label
        self._help_label = QLabel(help_text)
        self._help_label.setWordWrap(True)
        self._help_label.setMaximumWidth(300)
        layout.addWidget(self._help_label)

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.refresh_style()
        theme_signal.theme_changed.connect(self.refresh_style)
        self.hide()

    def refresh_style(self):
        """Rebuild styling from current palette."""
        pal = QApplication.instance().palette()
        bg = pal.color(QPalette.ToolTipBase).name()
        border = pal.color(QPalette.Mid).name()
        text = pal.color(QPalette.ToolTipText).name()
        self.setStyleSheet(
            f"""
            HelpPopup {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 8px;
            }}
            HelpPopup QLabel {{
                color: {text};
            }}
        """
        )
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()


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
        self.setFocusPolicy(Qt.NoFocus)
        self.clicked.connect(self.on_click)

        self.setEnabled(bool(link_url))
        self.refresh_style()
        theme_signal.theme_changed.connect(self.refresh_style)

    def refresh_style(self):
        """Rebuild styling from current palette."""
        pal = QApplication.instance().palette()
        if self.link_url:
            base = pal.color(QPalette.Highlight).name()
            hover = pal.color(QPalette.Highlight).lighter(115).name()
            pressed = pal.color(QPalette.Highlight).darker(115).name()
            self.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {base};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 16px;
                    padding: 0px;
                }}
                QPushButton:hover {{ background-color: {hover}; }}
                QPushButton:pressed {{ background-color: {pressed}; }}
            """
            )
        else:
            bg = pal.color(QPalette.Mid).name()
            text = pal.color(QPalette.ButtonText).name()
            self.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {bg};
                    color: {text};
                    border: none;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 16px;
                    padding: 0px;
                }}
                QPushButton:hover {{ background-color: {bg}; }}
                QPushButton:pressed {{ background-color: {bg}; }}
            """
            )
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

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
