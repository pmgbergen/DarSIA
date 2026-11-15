"""Small reusable custom Qt widgets shared across the GUI."""

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSlider, QWidget


class NoScrollComboBox(QComboBox):
    """QComboBox that ignores mouse-wheel events.

    Prevents accidentally changing a dropdown's value by scrolling the page/panel
    when the cursor happens to pass over it — click-based selection (the dropdown
    button and its popup list) is unaffected.
    """

    def wheelEvent(self, event):
        event.ignore()


class _NoScrollSlider(QSlider):
    """QSlider that ignores mouse-wheel events (see :class:`NoScrollComboBox`)."""

    def wheelEvent(self, event):
        event.ignore()


class DiscreteSlider(QWidget):
    """Slider over a few fixed values (e.g. 1, 2, 4, 8, 16), with the value shown.

    The slider positions are equally spaced, independent of the values.

    Parameters
    ----------
    options : list
        The selectable values, in the order of the slider positions.

    """

    valueChanged = Signal(object)

    def __init__(self, options, parent=None):
        super().__init__(parent)
        self._options = list(options)
        self._slider = _NoScrollSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, len(self._options) - 1)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(1)
        self._slider.setTickInterval(1)
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._label = QLabel()
        self._label.setMinimumWidth(32)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._slider, stretch=1)
        layout.addWidget(self._label)
        self._slider.valueChanged.connect(self._on_position_changed)
        self._on_position_changed(self._slider.value())

    def _on_position_changed(self, position: int) -> None:
        self._label.setText(str(self._options[position]))
        self.valueChanged.emit(self._options[position])

    def options(self) -> list:
        """The selectable values."""
        return list(self._options)

    def value(self):
        """The selected value."""
        return self._options[self._slider.value()]

    def setValue(self, value) -> bool:
        """Select a value; return False (and keep the selection) if it is no option."""
        if value not in self._options:
            return False
        self._slider.setValue(self._options.index(value))
        return True


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
    """Editable dropdown with a color band per option.

    Used for colormap names and for the labels of a calibrated color path. The options
    are recomputed whenever the list is opened, so color paths calibrated during the
    session show up without reloading the config. Values not in the list can be typed.

    Parameters
    ----------
    options_source : callable
        Returns a list of ``(text, cmap)`` pairs: the option text and the matplotlib
        colormap shown as its band (None: no band).
    empty_option : bool
        Whether the list starts with an empty entry (e.g. "default colormap").
    icon_size : QSize
        Size of the color bands.
    """

    def __init__(
        self,
        options_source,
        empty_option: bool = True,
        icon_size: QSize = CMAP_ICON_SIZE,
        parent=None,
    ):
        super().__init__(parent)
        self._options_source = options_source
        self._empty_option = empty_option
        self._icon_size = icon_size
        self.options: list[str] = []
        self.setEditable(True)
        self.setIconSize(icon_size)
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the option list (keeping the current text) with color bands."""
        options = self._options_source()
        text = self.currentText()
        self.blockSignals(True)
        self.clear()
        if self._empty_option:
            self.addItem("")
        for name, cmap in options:
            if cmap is None:
                self.addItem(name)
            else:
                self.addItem(cmap_icon(cmap, self._icon_size), name)
        self.options = [name for name, _ in options]
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
