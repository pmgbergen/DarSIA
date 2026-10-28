"""Photo navigation and white internal interfaces are display-only operations."""
import numpy as np
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QPainterPath, QPalette
from PySide6.QtWidgets import QVBoxLayout, QWidget

from darsia.fluidflower.painting_gap_review_ui import _Photo
from darsia.fluidflower.painting_transfer_overlays import (
    full_material_display, sand_boundary_path, sand_boundary_segments,
)


def test_full_material_reaches_measured_frame_beyond_native_raster():
    source = np.full((5, 6), 3, np.uint8)
    source[2:] = 4
    original = source.copy()
    photo = np.array([[1., 0., 5.], [0., 1., 4.], [0., 0., 1.]])
    frame = [[-2., -1.], [8., -1.], [8., 7.], [-2., 7.]]
    displayed = full_material_display(source, np.eye(3), photo, frame, (14, 16))
    assert np.all(displayed[4, 3:14] == 3)
    assert np.all(displayed[10, 3:14] == 4)
    assert not displayed[:, :3].any() and not displayed[:, 14:].any()
    segments = sand_boundary_segments(displayed, [3, 4])
    assert len(segments) == 1
    np.testing.assert_array_equal(segments[0], [[2.5, 5.5], [13.5, 5.5]])
    np.testing.assert_array_equal(source, original)


def test_internal_sand_boundary_has_no_outer_frame_or_contour_end_caps():
    # Two actual layers meet the left fixture and right water/chart region.
    labels = np.array([[0, 0, 0, 0, 0, 0],
                       [18, 3, 3, 3, 2, 17],
                       [18, 3, 3, 3, 2, 17],
                       [18, 4, 4, 4, 2, 17],
                       [0, 0, 0, 0, 0, 0]], dtype=np.int32)
    before = labels.copy()
    segments = sand_boundary_segments(labels, [3, 4])
    assert len(segments) == 1
    # The genuine layer reaches the boundary of its last porous pixel pair.
    np.testing.assert_array_equal(segments[0], [[.5, 2.5], [3.5, 2.5]])
    path = sand_boundary_path(labels, [3, 4], np.eye(3))
    assert path.elementCount() == 2  # One open line; no cap/perimeter/return.
    assert path.elementAt(0).type == QPainterPath.ElementType.MoveToElement
    assert path.elementAt(1).type == QPainterPath.ElementType.LineToElement
    np.testing.assert_array_equal(labels, before)


def test_separate_real_interfaces_keep_gaps_and_exact_photo_transform():
    labels = np.array([[3, 4], [3, 4], [18, 18], [3, 4]], dtype=np.uint8)
    matrix = np.array([[2., 0., 5.], [0., 3., 7.], [0., 0., 1.]])
    path = sand_boundary_path(labels, [3, 4], matrix)
    assert path.elementCount() == 4
    actual = [(path.elementAt(i).x, path.elementAt(i).y) for i in range(4)]
    np.testing.assert_array_equal(actual, [[6., 5.5], [6., 11.5], [6., 14.5], [6., 17.5]])
    assert path.elementAt(2).type == QPainterPath.ElementType.MoveToElement


def test_undefined_photo_transform_is_not_silently_replaced():
    matrix = np.array([[1., 0., 0.], [0., 1., 0.], [1., 0., -.5]])
    with pytest.raises(ValueError, match="undefined"):
        sand_boundary_path(np.array([[3, 4]]), [3, 4], matrix)


def _photo(qtbot):
    photo = _Photo()
    qtbot.addWidget(photo)
    photo.resize(760, 520)
    photo.show()
    photo.show_rgb(np.full((600, 1000, 3), 30, np.uint8))
    photo.scale(3., 3.)
    return photo


def _assert_edge_can_be_centred(photo, x, y):
    photo.centerOn(x, y)
    actual = photo.mapFromScene(QPointF(x, y))
    centre = photo.viewport().rect().center()
    assert abs(actual.x() - centre.x()) <= 2
    assert abs(actual.y() - centre.y()) <= 2


def test_zoomed_photo_all_edges_can_reach_centre_and_resize_keeps_zoom(qtbot):
    photo = _photo(qtbot)
    for corner in ((0., 0.), (1000., 0.), (0., 600.), (1000., 600.)):
        _assert_edge_can_be_centred(photo, *corner)
    zoom = photo.transform()
    photo.resize(930, 700)
    qtbot.waitUntil(lambda: photo.viewport().width() > 900)
    assert photo.transform() == zoom
    _assert_edge_can_be_centred(photo, 0., 0.)
    _assert_edge_can_be_centred(photo, 1000., 600.)


def test_reactive_photo_adjustment_replaces_pixels_without_refit_or_pan_change(qtbot, monkeypatch):
    photo = _photo(qtbot)
    _assert_edge_can_be_centred(photo, 0., 300.)
    transform = photo.transform()
    pan = photo.horizontalScrollBar().value(), photo.verticalScrollBar().value()
    def forbidden_fit():
        pytest.fail("A same-photo adjustment must not refit the viewport")
    monkeypatch.setattr(photo, "fit", forbidden_fit)
    photo.show_rgb(np.full((600, 1000, 3), 60, np.uint8), refit=False)
    assert photo.transform() == transform
    assert (photo.horizontalScrollBar().value(), photo.verticalScrollBar().value()) == pan
    assert photo.item.pixmap().toImage().pixelColor(20, 20).red() == 60


def _assert_complete_photo_visible(photo):
    rectangle = photo.mapFromScene(photo.item.boundingRect()).boundingRect()
    viewport = photo.viewport().rect().adjusted(-2, -2, 2, 2)
    assert viewport.contains(rectangle)


def test_review_photo_refits_after_parent_layout_and_resize_until_user_zooms(qtbot):
    parent = QWidget()
    qtbot.addWidget(parent)
    layout = QVBoxLayout(parent)
    photo = _Photo()
    photo.auto_fit_on_resize = True
    layout.addWidget(photo)
    # The reader can finish before the parent's final visible layout.
    photo.show_rgb(np.full((600, 1000, 3), 30, np.uint8))
    parent.resize(440, 320)
    parent.show()
    qtbot.waitUntil(lambda: photo.viewport().isVisible())
    _assert_complete_photo_visible(photo)
    for width, height in ((1100, 760), (480, 340)):
        parent.resize(width, height)
        qtbot.waitUntil(lambda: abs(photo.width() - width) < 30)
        _assert_complete_photo_visible(photo)
    photo.scale(3., 3.)
    zoom = photo.transform()
    parent.resize(900, 650)
    qtbot.waitUntil(lambda: photo.viewport().width() > 800)
    assert photo.transform() == zoom
    _assert_edge_can_be_centred(photo, 0., 0.)
    _assert_edge_can_be_centred(photo, 1000., 600.)
    photo.fit()
    parent.resize(440, 320)
    qtbot.waitUntil(lambda: photo.viewport().width() < 440)
    _assert_complete_photo_visible(photo)


def test_rig_view_masks_only_the_display_and_keeps_complete_photo_pixels(qtbot):
    photo = _Photo()
    qtbot.addWidget(photo)
    photo.resize(900, 500)
    photo.show()
    photo.show_rgb(np.full((80, 100, 3), [4, 50, 200], np.uint8))
    photo.set_view_rect(QRectF(20., 20., 60., 40.))
    assert photo.item.pixmap().size().toTuple() == (100, 80)
    outside = photo.mapFromScene(QPointF(18., 40.))
    assert photo.viewport().rect().contains(outside)
    shown = photo.viewport().grab().toImage()
    assert shown.pixelColor(outside) == photo.palette().color(QPalette.ColorRole.Base)
    photo.set_view_rect(None)
    _assert_complete_photo_visible(photo)
    assert photo.item.pixmap().toImage().pixelColor(18, 40).blue() == 200
