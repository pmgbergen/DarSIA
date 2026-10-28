"""Expanded photographs display unchanged camera-U masks and gestures."""
from copy import deepcopy

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower.series_layer_view import LayerCanvas


class Translation:
    def __init__(self, offset=(20.25, 15.75)):
        self.offset = np.asarray(offset)

    def camera_u_to_display_xy(self, points):
        return np.asarray(points) + self.offset

    def display_to_camera_u_xy(self, points):
        return np.asarray(points) - self.offset


@pytest.fixture
def canvas(qtbot):
    widget = LayerCanvas()
    qtbot.addWidget(widget)
    widget.resize(900, 700)
    widget.set_photo(np.zeros((50, 60, 3), np.uint8), (80, 60),
        token={}, reset_view=True, display_mapping=Translation(),
        display_size=(120, 100), valid_support=np.ones((100, 120), bool))
    widget.show()
    qtbot.waitExposed(widget)
    return widget


def xy(point):
    return point.x(), point.y()


@pytest.mark.parametrize("shape", [(60, 80), (30, 40)])
def test_proposal_pixel_centres_follow_original_u_with_independent_display_reduction(canvas, shape):
    rows, cols = shape
    rgba = np.empty((rows, cols, 4), dtype=np.uint8)
    rgba[:] = (30, 160, 220, 255)
    rgba[11, 17] = (250, 10, 80, 255)
    original = rgba.copy()
    canvas.set_proposal(rgba)
    assert canvas.proposal_item.pixmap().size().toTuple() == (cols, rows)
    assert canvas.proposal_item.pixmap().toImage().pixelColor(17, 11).getRgb() == (250, 10, 80, 255)
    for row, col in ((0, 0), (11, 17), (rows-1, cols-1)):
        raw_row = (row + .5) * 60. / rows - .5
        raw_col = (col + .5) * 80. / cols - .5
        shown = canvas.proposal_item.mapToScene(QPointF(col + .5, row + .5))
        expected = canvas.raw_to_scene(raw_row, raw_col)
        np.testing.assert_allclose(xy(shown), xy(expected), atol=1e-12)
    np.testing.assert_array_equal(rgba, original)


def test_photo_change_without_mapping_null_overlay_and_clear_reset_raster_transform(canvas):
    rgba = np.full((30, 40, 4), 255, np.uint8)
    canvas.set_proposal(rgba)
    assert not canvas.proposal_item.transform().isIdentity()
    # Keep the display size identical, so the existing pixmap is repositioned.
    canvas.set_photo(np.zeros((50, 60, 3), np.uint8), (60, 50), token={}, reset_view=False)
    assert not canvas.proposal_item.pixmap().isNull()
    assert canvas.proposal_item.transform().isIdentity()
    assert canvas.proposal_item.pos() == QPointF()
    canvas.set_proposal(rgba)
    assert canvas.proposal_item.transform().isIdentity()
    canvas.set_photo(np.zeros((50, 60, 3), np.uint8), (80, 60), token={}, reset_view=False,
        display_mapping=Translation(), display_size=(120, 100))
    assert not canvas.proposal_item.transform().isIdentity()
    canvas.set_proposal(None)
    assert canvas.proposal_item.pixmap().isNull() and canvas.proposal_item.transform().isIdentity()
    assert canvas.proposal_item.pos() == QPointF()
    canvas.set_proposal(rgba)
    canvas.clear_photo()
    assert canvas.proposal_item.pixmap().isNull() and canvas.proposal_item.transform().isIdentity()
    assert canvas.photo_display_mapping is None


def test_polygon_and_stroke_rendering_share_mapped_camera_u_without_changing_data(canvas):
    polygon = lp.Polygon(1, [(-5., -8.), (20., 15.), (45., 60.)])
    stroke = lp.Stroke(1, 4., [(10., 10.), (10., 40.)])
    before = deepcopy((polygon.to_list(), stroke.to_list()))
    colors = {1: QColor("#ff6633")}
    canvas.render_polygons([polygon], colors)
    canvas.render_strokes([stroke], colors)
    path = canvas._polygon_items[0].path()
    for index, point in enumerate(polygon.points):
        np.testing.assert_allclose((path.elementAt(index).x, path.elementAt(index).y),
            xy(canvas.raw_to_scene(*point)), atol=1e-12)
    scene = canvas.raw_to_scene(10., 25.)
    pixel = canvas._stroke_image.pixelColor(round(scene.x()), round(scene.y()))
    assert pixel.alpha() > 200 and pixel.red() > 240
    assert canvas._stroke_image.pixelColor(2, 2).alpha() == 0
    assert (polygon.to_list(), stroke.to_list()) == before


def test_expanded_photo_click_polygon_and_brush_gestures_return_original_camera_u(canvas, qtbot):
    def position(row, col):
        point = canvas.mapFromScene(canvas.raw_to_scene(row, col))
        assert canvas.viewport().rect().contains(point)
        return point

    selected = []
    canvas.raw_point_selected.connect(lambda row, col: selected.append((row, col)))
    canvas.place_mode = True
    qtbot.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=position(-5., -8.))
    np.testing.assert_allclose(selected, [[-5., -8.]], atol=.3)
    canvas.place_mode = False
    canvas.paint_enabled = True
    canvas.follow_edges = False
    canvas.set_layer(3, QColor("#66aaee"))
    canvas.set_tool("polygon")
    polygons = []
    canvas.polygon_finished.connect(polygons.append)
    points = [(-5., -8.), (15., 40.), (40., 5.)]
    for row, col in points:
        qtbot.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=position(row, col))
    assert canvas.close_polygon()
    assert len(polygons) == 1 and polygons[0].layer_id == 3
    np.testing.assert_allclose(polygons[0].points, points, atol=.3)

    canvas.set_tool("paint")
    strokes = []
    canvas.stroke_finished.connect(strokes.append)
    qtbot.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=position(-3., -6.))
    qtbot.mouseMove(canvas.viewport(), pos=position(20., 30.))
    qtbot.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=position(20., 30.))
    assert len(strokes) == 1 and strokes[0].layer_id == 3
    np.testing.assert_allclose(strokes[0].points[0], [-3., -6.], atol=.3)
    np.testing.assert_allclose(strokes[0].points[-1], [20., 30.], atol=.3)
