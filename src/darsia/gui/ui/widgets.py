"""Small reusable custom Qt widgets shared across the GUI."""

import numpy as np
from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import QComboBox


class NoScrollComboBox(QComboBox):
    """QComboBox that ignores mouse-wheel events.

    Prevents accidentally changing a dropdown's value by scrolling the page/panel
    when the cursor happens to pass over it — click-based selection (the dropdown
    button and its popup list) is unaffected.
    """

    def wheelEvent(self, event):
        event.ignore()


CMAP_ICON_SIZE = QSize(64, 12)
"""Size of the color band shown next to each colormap option."""


def cmap_icon(cmap, size: QSize = CMAP_ICON_SIZE) -> QIcon:
    """Color band (left: low values, right: high values) of a matplotlib colormap."""
    row = (cmap(np.linspace(0.0, 1.0, size.width())) * 255).astype(np.uint8)
    rgba = np.ascontiguousarray(np.repeat(row[np.newaxis], size.height(), axis=0))
    image = QImage(
        rgba.data, size.width(), size.height(), 4 * size.width(), QImage.Format_RGBA8888
    )
    return QIcon(QPixmap.fromImage(image.copy()))


class CmapComboBox(NoScrollComboBox):
    """Editable colormap dropdown with a color band per option.

    The options are recomputed whenever the list is opened, so color paths calibrated
    during the session show up without reloading the config. The first entry is empty
    (default colormap); values not in the list can be typed in.

    Parameters
    ----------
    options_source : callable
        Returns ``(options, color_registry)``: the colormap names to offer and the
        registry needed to resolve ``color_path.*`` names (or None).
    """

    def __init__(self, options_source, parent=None):
        super().__init__(parent)
        self._options_source = options_source
        self.options: list[str] = []
        self.setEditable(True)
        self.setIconSize(CMAP_ICON_SIZE)
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the option list (keeping the current text) with color bands."""
        from darsia.presets.workflows.analysis.colormaps import resolve_cmap

        options, registry = self._options_source()
        text = self.currentText()
        self.blockSignals(True)
        self.clear()
        self.addItem("")
        for name in options:
            cmap = resolve_cmap(name, registry)
            if cmap is None:
                self.addItem(name)
            else:
                self.addItem(cmap_icon(cmap), name)
        self.options = list(options)
        self.setCurrentText(text)
        self.blockSignals(False)

    def setCurrentText(self, text: str) -> None:
        """Select the matching option (showing its color band), else show the text."""
        index = self.findText(text)
        if index >= 0:
            self.setCurrentIndex(index)
        else:
            super().setCurrentText(text)

    def showPopup(self):
        self.refresh()
        super().showPopup()
