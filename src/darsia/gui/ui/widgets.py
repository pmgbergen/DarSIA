"""Small reusable custom Qt widgets shared across the GUI."""

from PySide6.QtWidgets import QComboBox


class NoScrollComboBox(QComboBox):
    """QComboBox that ignores mouse-wheel events.

    Prevents accidentally changing a dropdown's value by scrolling the page/panel
    when the cursor happens to pass over it — click-based selection (the dropdown
    button and its popup list) is unaffected.
    """

    def wheelEvent(self, event):
        event.ignore()
