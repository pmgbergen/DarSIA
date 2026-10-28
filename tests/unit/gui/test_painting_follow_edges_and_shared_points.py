"""Layer painting tab: Draw follows the edges of shown polygons, and Edit moves
shared points together.

With Follow edges on (the default, a setting of the user), a Draw click near a
point or edge of a shown polygon snaps onto it, a point on an edge becomes a
point of that polygon too, and between two such clicks the new polygon
follows the outline, the shorter way or, with Tab, the other way. Points of
several polygons at one position are one shared point in Edit. Step 3 offers
whether the classifier may move shared edges. Real mouse and key events are
sent to the canvas. A synthetic 80 x 60 photograph and a project in a
temporary folder; no real series, rig model or NAS input is used.
"""

import json

import pytest
from PIL import Image
from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsEllipseItem,
    QGraphicsRectItem,
    QMenu,
)

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import series_layer_view as slv
from darsia.fluidflower.series_geometry import SeriesRunInputs
from darsia.fluidflower.series_layer_view import LayerPainterView

NO = Qt.KeyboardModifier.NoModifier
ALT = Qt.KeyboardModifier.AltModifier
LEFT = Qt.MouseButton.LeftButton
# Polygons in raw pixels (row, col) of the 80 x 60 photograph.
POLYGON_A = [(5.0, 5.0), (5.0, 35.0), (30.0, 35.0), (30.0, 5.0)]
POLYGON_B = [(10.0, 45.0), (10.0, 75.0), (50.0, 75.0), (50.0, 45.0)]
# Two layers side by side that share the edge from (5, 30) to (40, 30).
LEFT_SIDE = [(5.0, 5.0), (5.0, 30.0), (40.0, 30.0), (40.0, 5.0)]
RIGHT_SIDE = [(5.0, 30.0), (5.0, 55.0), (40.0, 55.0), (40.0, 30.0)]


class _Project:
    def __init__(self, root):
        self.root = root
        self.manifest = {"rig": {"type": "medium"}}

    def update_metadata(self, *args, **kwargs):
        pass


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """The user's settings in a private INI file, not the registry.

    QSettings(organization, application) always uses the native format, which
    is the registry on Windows, so the view's QSettings is replaced here.
    """

    path = str(tmp_path / "painting.ini")
    monkeypatch.setattr(
        slv, "QSettings", lambda *args: QSettings(path, QSettings.Format.IniFormat)
    )
    return path


@pytest.fixture
def make_view(qtbot, tmp_path, monkeypatch, settings):
    # Menus are built as in the application but not shown.
    monkeypatch.setattr(QMenu, "popup", lambda self, *args, **kwargs: None)
    folder = tmp_path / "series"
    folder.mkdir()
    baseline = folder / "raw.png"
    Image.new("RGB", (80, 60), (30, 40, 50)).save(baseline)
    config = folder / "run.toml"
    config.write_text('[data]\nfolder="."\nbaseline="raw.png"\n', encoding="utf-8")
    reference = SeriesRunInputs(
        "ac52", config, baseline_image=baseline, width_m=0.9, height_m=0.49
    )

    def make():
        widget = LayerPainterView(_Project(tmp_path / "project"), lambda: reference)
        qtbot.addWidget(widget)
        widget.resize(1400, 900)
        widget.show()
        qtbot.waitExposed(widget)
        # The photograph is shown at full size: one display pixel per raw pixel.
        assert widget.canvas._scale() == (1.0, 1.0)
        return widget

    return make


@pytest.fixture
def view(make_view):
    return make_view()


def _at(canvas, row, col):
    """Viewport position of a raw pixel."""

    return canvas.mapFromScene(canvas.raw_to_scene(row, col))


def _raw(canvas, position):
    """Raw (row, col) under a viewport position."""

    return canvas.scene_to_raw(canvas.mapToScene(position))


def _press(canvas, position, modifiers=NO):
    QTest.mousePress(canvas.viewport(), LEFT, modifiers, position)


def _move(canvas, position):
    QTest.mouseMove(canvas.viewport(), position)


def _release(canvas, position, modifiers=NO):
    QTest.mouseRelease(canvas.viewport(), LEFT, modifiers, position)


def _settle():
    # The hint below the canvas changes its height when a polygon is started
    # or closed, and the photograph is fitted again: positions are taken after.
    QApplication.processEvents()


def _click(canvas, position, modifiers=NO):
    QTest.mouseClick(canvas.viewport(), LEFT, modifiers, position)
    _settle()


def _drag(canvas, start, end, modifiers=NO, steps=4):
    _press(canvas, start, modifiers)
    for step in range(1, steps + 1):
        _move(
            canvas,
            QPoint(
                round(start.x() + (end.x() - start.x()) * step / steps),
                round(start.y() + (end.y() - start.y()) * step / steps),
            ),
        )
    _release(canvas, end, modifiers)
    _settle()


def _double_click(canvas, position):
    # What the window system sends: press, release, double click, release.
    _click(canvas, position)
    QTest.mouseDClick(canvas.viewport(), LEFT, NO, position)
    _release(canvas, position)


def _key(canvas, key, modifiers=NO):
    QTest.keyClick(canvas, key, modifiers)
    _settle()


def _record(canvas):
    edits = []
    canvas.polygon_edited.connect(lambda index, _points: edits.append(("one", index)))
    canvas.polygons_edited.connect(
        lambda changes: edits.append(("several", sorted(changes)))
    )
    return edits


def _scene_xy(point):
    """(x, y) of a QPointF or of a QPainterPath element, rounded."""

    x, y = point.x, point.y
    if callable(x):
        x, y = x(), y()
    return round(x, 6), round(y, 6)


# ---------------------------------------------------------------------------
# Follow edges: the box


def test_follow_edges_is_on_by_default_and_kept_as_a_setting_of_the_user(
    make_view, settings
):
    view = make_view()
    box = view.follow_edges
    assert box.text() == "Follow edges" and box.isChecked()
    assert view.canvas.follow_edges
    assert "snaps onto it" in box.toolTip() and "Alt+click" in box.toolTip()
    # It sits after the mark tools, before the brush.
    erase = view.tool_buttons["erase"]
    assert box.mapToGlobal(QPoint(0, 0)).x() > erase.mapToGlobal(QPoint(0, 0)).x()
    top = box.mapToGlobal(QPoint(0, 0)).y()
    assert abs(top - erase.mapToGlobal(QPoint(0, 0)).y()) < 12
    box.click()
    assert not box.isChecked() and not view.canvas.follow_edges
    stored = QSettings(settings, QSettings.Format.IniFormat)
    assert stored.value(slv._FOLLOW_EDGES_KEY) in (False, "false")
    # The choice is the user's: the painting session does not hold it.
    view.save_session()
    session_file = lp.session_path(view.project, "ac52")
    if session_file.exists():
        assert "follow" not in session_file.read_text(encoding="utf-8").lower()
    assert not make_view().follow_edges.isChecked()
    box.click()
    assert make_view().follow_edges.isChecked()


# ---------------------------------------------------------------------------
# Follow edges: snapping


def test_a_click_near_a_point_or_an_edge_snaps_onto_it_and_adds_the_edge_point(view):
    canvas = view.canvas
    view.add_polygon(lp.Polygon(1, list(POLYGON_A)))
    view.select_layer_row(2)  # Coarse sand
    edits = _record(canvas)
    steps = len(view._undo_stack)
    # Within the point-hit distance of A's point (5, 35): exactly onto it.
    _click(canvas, _at(canvas, 5.0, 35.0) + QPoint(4, 3))
    assert canvas.draft_points == [(5.0, 35.0)]
    # Near the middle of A's right edge, from outside A: onto the edge.
    near_edge = _at(canvas, 17.5, 35.0) + QPoint(5, 0)
    clicked = _raw(canvas, near_edge)
    _click(canvas, near_edge)
    row, col = canvas.draft_points[1]
    assert col == 35.0 and row == pytest.approx(clicked[0], abs=1e-6)
    assert 16.0 < row < 19.0
    # Both on A's right edge: the straight edge is the shorter way.
    assert len(canvas.draft_points) == 2
    for point in ((17.5, 50.0), (5.0, 50.0)):
        _click(canvas, _at(canvas, *point))
    _key(canvas, Qt.Key.Key_Return)
    assert len(view.session.polygons) == 2
    new = view.session.polygons[1]
    assert new.points[:2] == [(5.0, 35.0), (row, 35.0)]
    # The point on A's edge is A's point too, added in the same change.
    assert view.session.polygons[0].points == [
        (5.0, 5.0),
        (5.0, 35.0),
        (row, 35.0),
        (30.0, 35.0),
        (30.0, 5.0),
    ]
    assert edits == [] and len(view._undo_stack) == steps + 1
    view.undo()
    assert [polygon.points for polygon in view.session.polygons] == [POLYGON_A]
    view.redo()
    assert len(view.session.polygons) == 2
    assert len(view.session.polygons[0].points) == 5


def test_alt_click_and_follow_edges_off_place_the_point_where_clicked(view):
    canvas = view.canvas
    view.add_polygon(lp.Polygon(1, list(POLYGON_A)))
    view.select_layer_row(2)
    position = _at(canvas, 5.0, 35.0) + QPoint(4, 3)
    clicked = _raw(canvas, position)
    _click(canvas, position, ALT)
    assert canvas.draft_points[0] == pytest.approx(clicked)
    assert canvas.draft_points[0] != (5.0, 35.0)
    view.follow_edges.click()
    _settle()
    assert not canvas.follow_edges
    edge = _at(canvas, 17.5, 35.0) + QPoint(5, 0)
    on_edge = _raw(canvas, edge)
    _click(canvas, edge)
    assert canvas.draft_points[1] == pytest.approx(on_edge)
    # The hint names Tab and Alt only while Follow edges is on.
    assert "Tab" not in view.canvas_hint.text()
    _click(canvas, _at(canvas, 20.0, 50.0))
    _key(canvas, Qt.Key.Key_Return)
    # As before 0004: no point is added to A.
    assert view.session.polygons[0].points == POLYGON_A
    assert view.session.polygons[1].points[1] == pytest.approx(on_edge)


def test_polygons_of_hidden_layers_are_not_snapped_to(view):
    canvas = view.canvas
    view.add_polygon(lp.Polygon(1, list(POLYGON_A)))
    view.select_layer_row(2)
    view.set_layer_visible(1, False)
    _settle()
    position = _at(canvas, 5.0, 35.0) + QPoint(4, 3)
    assert canvas.snap_point(QPointF(position)) is None
    clicked = _raw(canvas, position)
    _click(canvas, position)
    assert canvas.draft_points[0] == pytest.approx(clicked)
    canvas.cancel_polygon()
    view.set_layer_visible(1, True)
    _settle()
    position = _at(canvas, 5.0, 35.0) + QPoint(4, 3)
    assert canvas.snap_point(QPointF(position)) == (5.0, 35.0)


# ---------------------------------------------------------------------------
# Follow edges: following an outline


def test_the_draft_follows_the_outline_the_shorter_way_and_tab_takes_the_other(view):
    canvas = view.canvas
    view.add_polygon(lp.Polygon(2, list(POLYGON_B)))
    view.select_layer_row(2)
    _click(canvas, _at(canvas, 10.0, 45.0))
    assert canvas.draft_points == [(10.0, 45.0)]
    # The preview shows the way a click on B's bottom edge would follow:
    # backwards past (50, 45), shorter than forwards past (10, 75) and (50, 75).
    bottom = _at(canvas, 50.0, 60.0)
    _move(canvas, bottom)
    assert canvas.snap_item.isVisible()
    rubber = canvas.rubber_item.path()
    assert rubber.elementCount() == 3
    assert _scene_xy(rubber.elementAt(1)) == _scene_xy(canvas.raw_to_scene(50.0, 45.0))
    _click(canvas, bottom)
    end = canvas.draft_points[-1]
    assert end[0] == 50.0 and end[1] == pytest.approx(60.0, abs=0.2)
    assert canvas.draft_points == [(10.0, 45.0), (50.0, 45.0), end]
    assert "Tab: follow the other way round" in view.canvas_hint.text()
    # Snapped points are round, and the outline point followed is a small dot.
    handles = canvas._draft_handles
    assert all(isinstance(item, QGraphicsEllipseItem) for item in handles)
    assert handles[1].rect().width() < handles[2].rect().width()
    # Tab takes the other way round, Shift+Tab (Backtab) comes back.
    _key(canvas, Qt.Key.Key_Tab)
    assert canvas.draft_points == [(10.0, 45.0), (10.0, 75.0), (50.0, 75.0), end]
    assert canvas.hasFocus()
    _key(canvas, Qt.Key.Key_Backtab)
    assert canvas.draft_points == [(10.0, 45.0), (50.0, 45.0), end]
    _key(canvas, Qt.Key.Key_Tab)
    # Backspace removes the last point with the outline followed to it.
    _key(canvas, Qt.Key.Key_Backspace)
    assert canvas.draft_points == [(10.0, 45.0)]
    _click(canvas, bottom)
    assert canvas.draft_points == [(10.0, 45.0), (50.0, 45.0), end]
    # A free point far from B is square, and the polygon closes as before.
    _click(canvas, _at(canvas, 57.0, 55.0))
    assert isinstance(canvas._draft_handles[-1], QGraphicsRectItem)
    _key(canvas, Qt.Key.Key_Return)
    new = view.session.polygons[1]
    assert new.points[:3] == [(10.0, 45.0), (50.0, 45.0), end]
    # The point on B's bottom edge is B's point too.
    assert view.session.polygons[0].points == [
        (10.0, 45.0),
        (10.0, 75.0),
        (50.0, 75.0),
        end,
        (50.0, 45.0),
    ]


def test_a_new_layer_drawn_along_a_neighbour_shares_its_outline(view):
    canvas = view.canvas
    view.add_polygon(lp.Polygon(1, list(LEFT_SIDE)))
    view.select_layer_row(2)
    # Along the neighbour's whole right side, then round the new layer.
    for point in ((5.0, 30.0), (40.0, 30.0), (40.0, 55.0), (5.0, 55.0)):
        _click(canvas, _at(canvas, *point))
    _key(canvas, Qt.Key.Key_Return)
    new = view.session.polygons[1]
    # Exactly on the neighbour's points; the free points where clicked.
    assert new.points[:2] == [(5.0, 30.0), (40.0, 30.0)]
    assert new.points[2] == pytest.approx((40.0, 55.0), abs=0.1)
    assert new.points[3] == pytest.approx((5.0, 55.0), abs=0.1)
    assert lp.shared_points({0: LEFT_SIDE, 1: new.points}) == [
        [(0, 1), (1, 0)],
        [(0, 2), (1, 1)],
    ]
    assert lp.shared_edges(view.session.polygons) == [((5.0, 30.0), (40.0, 30.0))]


# ---------------------------------------------------------------------------
# Shared points in Edit


def _points(view):
    return [polygon.points for polygon in view.session.polygons]


def _two_sides(view):
    view.add_polygon(lp.Polygon(1, list(LEFT_SIDE)))
    view.add_polygon(lp.Polygon(2, list(RIGHT_SIDE)))
    view.set_tool("edit")
    return view.canvas


def test_shared_points_have_round_handles_and_are_selected_together(view):
    canvas = _two_sides(view)
    handles = canvas._handle_items
    assert isinstance(handles[0][1], QGraphicsEllipseItem)  # (5, 30)
    assert isinstance(handles[1][3], QGraphicsEllipseItem)  # (40, 30)
    assert isinstance(handles[0][0], QGraphicsRectItem)
    assert handles[0][1].rect().width() > handles[0][0].rect().width()
    assert handles[0][1].pen().color().lightness() > 200  # A white rim.
    _click(canvas, _at(canvas, 5.0, 30.0))
    assert canvas.selected_points == [(0, 1), (1, 0)]
    # Selected, a shared handle looks like any selected handle, but round.
    assert handles[0][1].rect().width() == slv._SELECTED_HANDLE_PX
    # Ctrl+click takes the whole shared point out again.
    _click(canvas, _at(canvas, 5.0, 30.0), Qt.KeyboardModifier.ControlModifier)
    assert canvas.selected_points == []
    # A rubber band takes shared points whole.
    _drag(canvas, _at(canvas, 38.0, 28.0), _at(canvas, 42.0, 32.0))
    assert canvas.selected_points == [(0, 2), (1, 3)]
    # Hidden layers share no point.
    view.set_layer_visible(2, False)
    assert canvas.shared_with(0, 1) == [(0, 1)]
    assert isinstance(canvas._handle_items[0][1], QGraphicsRectItem)


def test_dragging_nudging_and_deleting_a_shared_point_act_on_all_as_one_change(view):
    canvas = _two_sides(view)
    edits = _record(canvas)
    steps = len(view._undo_stack)
    corner = _at(canvas, 5.0, 30.0)
    target = corner + QPoint(30, 40)
    _drag(canvas, corner, target)
    start, end = _raw(canvas, corner), _raw(canvas, target)
    moved = (5.0 + end[0] - start[0], 30.0 + end[1] - start[1])
    left, right = (polygon.points for polygon in view.session.polygons)
    assert left[1] == pytest.approx(moved) and right[0] == left[1]
    assert edits == [("several", [0, 1])] and len(view._undo_stack) == steps + 1
    assert canvas.selected_points == [(0, 1), (1, 0)]
    view.undo()
    assert _points(view) == [LEFT_SIDE, RIGHT_SIDE]

    _click(canvas, corner)
    _key(canvas, Qt.Key.Key_Right)
    left, right = (polygon.points for polygon in view.session.polygons)
    assert left[1][1] > 30.0 and right[0] == left[1]
    assert len(view._undo_stack) == steps + 1
    view.undo()

    _click(canvas, corner)
    _key(canvas, Qt.Key.Key_Delete)
    assert view.session.polygons[0].points == [LEFT_SIDE[0], *LEFT_SIDE[2:]]
    assert view.session.polygons[1].points == RIGHT_SIDE[1:]
    assert len(view._undo_stack) == steps + 1
    view.undo()
    assert _points(view) == [LEFT_SIDE, RIGHT_SIDE]


def test_alt_drag_moves_only_the_point_of_the_polygon_pressed_into(view):
    canvas = _two_sides(view)
    edits = _record(canvas)
    steps = len(view._undo_stack)
    # Pressed a little inside the left polygon, near the shared point.
    press = _at(canvas, 5.0, 30.0) + QPoint(-4, 4)
    target = press + QPoint(-30, 30)
    _drag(canvas, press, target, ALT)
    left, right = (polygon.points for polygon in view.session.polygons)
    assert right == RIGHT_SIDE
    assert left[1] != (5.0, 30.0) and left[2:] == LEFT_SIDE[2:]
    assert edits == [("one", 0)] and len(view._undo_stack) == steps + 1
    # No longer shared: square handles again.
    assert isinstance(canvas._handle_items[1][0], QGraphicsRectItem)
    assert canvas.shared_with(1, 0) == [(1, 0)]
    view.undo()
    assert _points(view) == [LEFT_SIDE, RIGHT_SIDE]


def test_a_point_added_on_a_shared_edge_goes_into_both_polygons(view):
    canvas = _two_sides(view)
    steps = len(view._undo_stack)
    _double_click(canvas, _at(canvas, 20.0, 30.0))
    left, right = (polygon.points for polygon in view.session.polygons)
    assert len(left) == 5 and len(right) == 5
    assert left[2] == right[4] and left[2][1] == 30.0
    assert canvas.selected_points == [(0, 2), (1, 4)]
    assert len(view._undo_stack) == steps + 1
    view.undo()
    # The same from the right-click menu's Add point here.
    assert view.add_polygon_point(1, 25.0, 30.5) == 4
    left, right = (polygon.points for polygon in view.session.polygons)
    assert left[2] == right[4] == (25.0, 30.0)
    assert len(view._undo_stack) == steps + 1
    # Remove this point on a shared point removes it from both, as one change.
    view.remove_polygon_point(0, 2)
    assert _points(view) == [LEFT_SIDE, RIGHT_SIDE]
    assert len(view._undo_stack) == steps + 2


# ---------------------------------------------------------------------------
# Step 3: may the classifier move shared edges?


def test_the_choice_whether_the_classifier_may_move_shared_edges(make_view):
    view = make_view()
    box = view.move_shared_edges
    assert [box.itemText(index) for index in range(box.count())] == ["Yes", "No"]
    assert box.currentText() == "Yes" and view.session.move_shared_edges
    # The shared-edge choice has its own short row in step 3, so the
    # sidebar remains readable when the painting canvas takes most of the view.
    detail = view.detail.mapTo(view, QPoint(0, view.detail.height() // 2))
    choice = box.mapTo(view, QPoint(0, box.height() // 2))
    assert choice.y() > detail.y()
    assert view.step_propose.content.isAncestorOf(box)
    assert "exactly on the shared edge" in box.toolTip()
    session_file = lp.session_path(view.project, "ac52")
    box.setCurrentIndex(1)
    assert view.session.move_shared_edges is False
    view.save_session()
    saved = json.loads(session_file.read_text(encoding="utf-8"))
    assert saved["proposal_settings"]["move_shared_edges"] is False
    assert make_view().move_shared_edges.currentText() == "No"
    box.setCurrentIndex(0)
    view.save_session()
    saved = json.loads(session_file.read_text(encoding="utf-8"))
    assert "move_shared_edges" not in saved["proposal_settings"]
