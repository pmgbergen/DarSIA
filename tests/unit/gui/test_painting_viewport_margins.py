"""Photo margins permit edge centring without extending drawing coordinates."""

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from darsia.fluidflower.series_layer_view import LayerCanvas


@pytest.fixture
def canvas(qtbot):
    widget = LayerCanvas()
    qtbot.addWidget(widget)
    widget.resize(740, 540)
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_photo(np.full((80, 120, 3), 80, np.uint8), (1200, 800),
                     token={"source": "synthetic"}, reset_view=True)
    return widget


def _wheel(canvas, direction):
    position = canvas.viewport().rect().center()
    event = QWheelEvent(QPointF(position), QPointF(canvas.viewport().mapToGlobal(position)),
                        QPoint(), QPoint(0, direction * 120), Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(canvas.viewport(), event)


def _margins(canvas):
    image = canvas.pixmap_item.boundingRect()
    scene = canvas.scene().sceneRect()
    visible = canvas.mapToScene(canvas.viewport().rect()).boundingRect()
    assert image.left() - scene.left() >= visible.width() / 2. - 1.e-9
    assert scene.right() - image.right() >= visible.width() / 2. - 1.e-9
    assert image.top() - scene.top() >= visible.height() / 2. - 1.e-9
    assert scene.bottom() - image.bottom() >= visible.height() / 2. - 1.e-9


def _centre_all_edges(canvas):
    bounds = canvas.pixmap_item.boundingRect()
    for point in (bounds.topLeft(), bounds.topRight(), bounds.bottomLeft(), bounds.bottomRight(),
                  QPointF(bounds.center().x(), bounds.top()),
                  QPointF(bounds.center().x(), bounds.bottom()),
                  QPointF(bounds.left(), bounds.center().y()),
                  QPointF(bounds.right(), bounds.center().y())):
        canvas.centerOn(point)
        shown = canvas.mapFromScene(point)
        centre = canvas.viewport().rect().center()
        # Scrollbars use integer screen positions, not subpixel centres.
        assert abs(shown.x() - centre.x()) <= 2
        assert abs(shown.y() - centre.y()) <= 2


@pytest.mark.parametrize("steps", [-8, 0, 8])
@pytest.mark.parametrize("size", [(620, 430), (980, 720)])
def test_fit_zoom_and_resize_allow_every_edge_and_corner_at_viewport_centre(canvas, qtbot, steps, size):
    _margins(canvas)
    _centre_all_edges(canvas)
    for _ in range(abs(steps)):
        _wheel(canvas, 1 if steps > 0 else -1)
    _margins(canvas)
    _centre_all_edges(canvas)
    canvas.resize(*size)
    qtbot.waitUntil(lambda: (canvas.width(), canvas.height()) == size)
    _margins(canvas)
    _centre_all_edges(canvas)


def test_fit_targets_whole_pixmap_rather_than_the_padded_scene(canvas):
    for _ in range(8):
        _wheel(canvas, 1)
    canvas.centerOn(canvas.pixmap_item.boundingRect().bottomRight())
    canvas.fit_image()
    image = canvas.viewportTransform().mapRect(canvas.pixmap_item.boundingRect())
    viewport = QRectF(canvas.viewport().rect())
    assert viewport.adjusted(-2., -2., 2., 2.).contains(image)
    # Fitting the padded scene would leave the photograph about half-size.
    assert max(image.width() / viewport.width(), image.height() / viewport.height()) > .95
    assert not canvas._user_zoomed
    _margins(canvas)


def test_raw_scene_roundtrip_pixel_origin_and_edit_clamp_are_unchanged(canvas):
    points = [(-.5, -.5), (0., 0.), (12.1234567, 25.7654321), (799., 1199.)]
    originals = [canvas.raw_to_scene(*point) for point in points]
    assert canvas._scale() == (.1, .1)
    assert canvas.pixmap_item.boundingRect() == QRectF(0., 0., 120., 80.)
    for _ in range(6):
        _wheel(canvas, 1)
    canvas.resize(900, 650)
    _centre_all_edges(canvas)
    for raw, before in zip(points, originals):
        after = canvas.raw_to_scene(*raw)
        assert after == before
        assert after.x() == pytest.approx((raw[1] + .5) * .1 - .5)
        assert after.y() == pytest.approx((raw[0] + .5) * .1 - .5)
        assert canvas.scene_to_raw(after) == pytest.approx(raw, abs=1.e-12)
    assert canvas._clamped_scene(QPointF(-100., -100.)) == QPointF(0., 0.)
    assert canvas._clamped_scene(QPointF(1000., 1000.)) == QPointF(119., 79.)
    assert canvas.pixmap_item.boundingRect() == QRectF(0., 0., 120., 80.)


@pytest.mark.parametrize("tool", ["polygon", "paint", "fill"])
def test_pan_margin_is_never_an_area_for_new_marks_or_pixel_fill(canvas, tool):
    strokes, polygons, fills = [], [], []
    canvas.stroke_finished.connect(strokes.append)
    canvas.polygon_finished.connect(polygons.append)
    canvas.fill_requested.connect(lambda row, col: fills.append((row, col)))
    canvas.paint_enabled = True
    canvas.set_layer(1, QColor("#ff0000"))
    canvas.set_tool(tool)
    canvas.centerOn(QPointF(0., 0.))
    margin = QPointF(-10., -10.)
    assert canvas.scene().sceneRect().contains(margin)
    assert not canvas.pixmap_item.boundingRect().contains(margin)
    position = canvas.mapFromScene(margin)
    assert canvas.viewport().rect().contains(position)
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, position)
    assert not canvas.drafting() and not canvas._points
    assert strokes == [] and polygons == [] and fills == []


def test_photo_refresh_updates_margins_without_resetting_zoom_and_clear_restores_policies(canvas):
    policies = canvas._photo_scrollbar_policies
    for _ in range(4):
        _wheel(canvas, 1)
    scale = canvas.transform().m11()
    canvas.set_photo(np.full((80, 120, 3), 120, np.uint8), (1200, 800),
                     token={"source": "refreshed"}, reset_view=False)
    assert canvas.transform().m11() == scale
    _margins(canvas)
    canvas.clear_photo()
    assert canvas.raw_size is None and canvas._photo_scrollbar_policies is None
    assert (canvas.horizontalScrollBarPolicy(), canvas.verticalScrollBarPolicy()) == policies
