"""Two image panels with one shared, pixel-aligned view of a registered grid.

The source images are kept as pixmaps. Zooming and panning only change a small
view transform; neither operation rescales or reloads the source arrays.
"""

from __future__ import annotations

from math import isfinite

from PySide6.QtCore import QObject, QPointF, Qt, QSize
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QSizePolicy, QWidget


class _ImageView(QWidget):
    """A viewport whose image coordinates are controlled by its pair."""

    def __init__(self, pair: "SynchronizedImagePair", side: int):
        super().__init__()
        self.pair = pair
        self.side = side
        self._drag_at: QPointF | None = None
        self.setMinimumSize(1, 1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111820"))
        image = self.pair._images[self.side]
        if image is None or image.isNull() or self.pair._center is None:
            return
        zoom = self.pair.zoom
        center = self.pair._center
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, zoom != 1.0)
        painter.translate(self.width() / 2 - center.x() * zoom,
                          self.height() / 2 - center.y() * zoom)
        painter.scale(zoom, zoom)
        painter.drawPixmap(0, 0, image)

    def wheelEvent(self, event: QWheelEvent):
        degrees = event.angleDelta().y() / 120.0
        if degrees and self.pair.has_images:
            self.pair.zoom_by(1.2 ** degrees, side=self.side,
                              viewport_position=event.position())
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag_at = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_at is not None:
            delta = event.position() - self._drag_at
            self._drag_at = event.position()
            self.pair.pan_pixels(delta)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._drag_at is not None and event.button() in (
                Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag_at = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pair._fit_mode and self.pair.has_images:
            self.pair.fit()


class SynchronizedImagePair(QObject):
    """Own two views sharing one scale and one center in source-pixel units.

    Add ``views[0]`` and ``views[1]`` to the two pane layouts. Call
    :meth:`set_images` on the GUI thread with RGB ``QImage`` objects from the
    same registered pixel grid. Subsequent same-grid images retain the current
    view by default, and :meth:`swap_images` never changes it.
    """

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._images: list[QPixmap | None] = [None, None]
        self._grid: QSize | None = None
        self._center: QPointF | None = None
        self._zoom: float | None = None
        self._fit_mode = True
        self.views = [_ImageView(self, 0), _ImageView(self, 1)]

    @property
    def has_images(self) -> bool:
        return self._grid is not None

    @property
    def zoom(self) -> float:
        return self._zoom if self._zoom is not None else 1.0

    @property
    def center(self) -> QPointF | None:
        return QPointF(self._center) if self._center is not None else None

    @property
    def grid_size(self) -> QSize | None:
        return QSize(self._grid) if self._grid is not None else None

    def set_images(self, left: QImage, right: QImage, *, preserve_view: bool = True):
        """Display two images; reject a mismatched grid before changing either."""
        if left.isNull() or right.isNull() or left.size() != right.size():
            raise ValueError("Comparison photographs must share the exact registered pixel grid")
        if left.width() < 1 or left.height() < 1:
            raise ValueError("Comparison photographs cannot be empty")
        new_images = [QPixmap.fromImage(left), QPixmap.fromImage(right)]
        same_grid = self._grid is not None and self._grid == left.size()
        self._images = new_images
        self._grid = QSize(left.size())
        if preserve_view and same_grid and self._center is not None:
            if self._fit_mode:
                self.fit()
            else:
                self._update_views()
        else:
            self.fit()

    def swap_images(self):
        """Exchange source pixmaps without changing scale or image coordinates."""
        self._images[0], self._images[1] = self._images[1], self._images[0]
        self._update_views()

    def clear(self, *, preserve_view: bool = False):
        self._images = [None, None]
        if not preserve_view:
            self._grid = None
            self._center = None
            self._zoom = None
            self._fit_mode = True
        self._update_views()

    def fit(self):
        """Fit the complete common grid inside *both* panel viewports."""
        if self._grid is None:
            return
        widths = [max(1, view.width()) for view in self.views]
        heights = [max(1, view.height()) for view in self.views]
        self._zoom = min(*(width / self._grid.width() for width in widths),
                         *(height / self._grid.height() for height in heights))
        self._center = QPointF(self._grid.width() / 2, self._grid.height() / 2)
        self._fit_mode = True
        self._update_views()

    def actual_size(self):
        """Display one screen pixel per source pixel, retaining the center."""
        if self._grid is None:
            return
        if self._center is None:
            self._center = QPointF(self._grid.width() / 2, self._grid.height() / 2)
        self._zoom = 1.0
        self._fit_mode = False
        self._update_views()

    def zoom_by(self, factor: float, *, side: int = 0,
                viewport_position: QPointF | None = None):
        """Zoom both views about one source pixel under the cursor of ``side``."""
        if self._grid is None or self._center is None or not isfinite(factor) or factor <= 0:
            return
        old_zoom = self.zoom
        new_zoom = max(0.01, min(64.0, old_zoom * factor))
        view = self.views[side]
        position = (viewport_position if viewport_position is not None else
                    QPointF(view.width() / 2, view.height() / 2))
        offset = position - QPointF(view.width() / 2, view.height() / 2)
        image_at_cursor = self._center + offset / old_zoom
        self._center = image_at_cursor - offset / new_zoom
        self._zoom = new_zoom
        self._fit_mode = False
        self._update_views()

    def pan_pixels(self, delta: QPointF):
        """Move the same image-coordinate center in both views."""
        if self._center is None:
            return
        self._center -= delta / self.zoom
        self._fit_mode = False
        self._update_views()

    def image_coordinate(self, side: int, viewport_position: QPointF) -> QPointF:
        """Map a panel location into the common registered source grid."""
        if self._center is None:
            raise ValueError("No photographs are loaded")
        view = self.views[side]
        offset = viewport_position - QPointF(view.width() / 2, view.height() / 2)
        return self._center + offset / self.zoom

    def _update_views(self):
        for view in self.views:
            view.update()
