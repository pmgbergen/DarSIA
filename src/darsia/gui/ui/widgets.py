"""Small reusable custom Qt widgets shared across the GUI."""

import numpy as np
from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import QIcon, QImage, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QWidget,
)

from .theme import muted_text_color


class ToggleSwitch(QWidget):
    """Pill-shaped two-state switch with a sliding knob.

    Unchecked = knob left, checked = knob right. Optional labels outside the
    track (``left``, ``right``: text) name the two states; the label of the
    current state is drawn in the normal text color, the other one muted.
    Optional icons inside the track (``inner``: (left icon, right icon)) sit
    under the knob's two positions, so the free side shows the other state's
    icon. The track takes the text color and the knob the window color, so the
    switch inverts with the theme; inner icons should be built in the window
    color (their colors are baked in, pass new ones via ``set_inner`` after a
    theme change). A click anywhere flips the state and emits ``toggled``.
    """

    toggled = Signal(bool)

    _TRACK_WIDTH = 44
    _TRACK_HEIGHT = 24
    _KNOB_MARGIN = 3
    _GAP = 6
    _ICON_SIZE = 12

    def __init__(self, left=None, right=None, checked=False, inner=None, parent=None):
        super().__init__(parent)
        self._left = left
        self._right = right
        self._inner = inner or (None, None)
        self._checked = bool(checked)
        self._position = 1.0 if self._checked else 0.0
        self._animation = QPropertyAnimation(self, b"knob_position", self)
        self._animation.setDuration(150)
        self._animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    # Animated knob position: 0 = left (unchecked), 1 = right (checked).
    def _get_position(self) -> float:
        return self._position

    def _set_position(self, value: float) -> None:
        self._position = value
        self.update()

    knob_position = Property(float, _get_position, _set_position)

    def is_checked(self) -> bool:
        return self._checked

    def set_checked(self, checked: bool, *, emit: bool = True, animate: bool = True):
        """Set the state; slides the knob and emits ``toggled`` if it changed."""
        checked = bool(checked)
        if checked == self._checked:
            return
        self._checked = checked
        target = 1.0 if checked else 0.0
        self._animation.stop()
        if animate and self.isVisible():
            self._animation.setStartValue(self._position)
            self._animation.setEndValue(target)
            self._animation.start()
        else:
            self._set_position(target)
        if emit:
            self.toggled.emit(checked)

    def set_inner(self, left=None, right=None) -> None:
        """Replace the icons inside the track (e.g. rebuilt for a new theme)."""
        self._inner = (left, right)
        self.update()

    def _item_width(self, item) -> int:
        if item is None:
            return 0
        return self.fontMetrics().horizontalAdvance(str(item))

    def sizeHint(self) -> QSize:
        widths = [self._item_width(self._left), self._item_width(self._right)]
        width = self._TRACK_WIDTH + sum(w + self._GAP for w in widths if w)
        height = max(self._TRACK_HEIGHT, self.fontMetrics().height()) + 4
        return QSize(width, height)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def _draw_item(self, painter, item, x, active) -> int:
        """Draw a text label at x (vertically centered); returns its width."""
        width = self._item_width(item)
        if not width:
            return 0
        pal = self.palette()
        color = pal.color(QPalette.WindowText) if active else muted_text_color(pal)
        painter.setPen(color)
        painter.drawText(
            QRectF(x, 0, width, self.height()),
            Qt.AlignVCenter | Qt.AlignLeft,
            str(item),
        )
        return width

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pal = self.palette()

        x = 0.0
        left_width = self._draw_item(painter, self._left, x, not self._checked)
        if left_width:
            x += left_width + self._GAP

        track = QRectF(
            x,
            (self.height() - self._TRACK_HEIGHT) / 2,
            self._TRACK_WIDTH,
            self._TRACK_HEIGHT,
        )
        radius = self._TRACK_HEIGHT / 2
        painter.setPen(Qt.NoPen)
        painter.setBrush(muted_text_color(pal, 0.2))
        painter.drawRoundedRect(track, radius, radius)

        diameter = self._TRACK_HEIGHT - 2 * self._KNOB_MARGIN
        travel = self._TRACK_WIDTH - 2 * self._KNOB_MARGIN - diameter
        top = track.top() + self._KNOB_MARGIN
        for index, icon in enumerate(self._inner):
            if icon is None:
                continue
            size = self._ICON_SIZE
            centre_x = track.left() + self._KNOB_MARGIN + index * travel + diameter / 2
            pixmap = icon.pixmap(QSize(size, size))
            painter.drawPixmap(
                int(centre_x - size / 2), int(top + (diameter - size) / 2), pixmap
            )

        knob = QRectF(
            track.left() + self._KNOB_MARGIN + self._position * travel,
            top,
            diameter,
            diameter,
        )
        painter.setBrush(pal.color(QPalette.Window))
        painter.drawEllipse(knob)

        x = track.right() + self._GAP
        self._draw_item(painter, self._right, x, self._checked)
        painter.end()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.rect().contains(
            event.position().toPoint()
        ):
            self.set_checked(not self._checked)
            event.accept()
            return
        super().mouseReleaseEvent(event)


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
