"""Layer painting tab: separate Draw and Edit tools for polygons.

Draw only adds new polygons. Edit selects points of the shown polygons and
moves, nudges, removes or adds them; every change is one undo step. Real
mouse and key events are sent to the canvas. A synthetic 80 x 60 photograph
and a project in a temporary folder; no real series, rig model or NAS input
is used.
"""

import json

import pytest
from PIL import Image
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QShortcut
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMenu

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower.series_geometry import SeriesRunInputs
from darsia.fluidflower.series_layer_view import LayerCanvas, LayerPainterView

NO = Qt.KeyboardModifier.NoModifier
CTRL = Qt.KeyboardModifier.ControlModifier
SHIFT = Qt.KeyboardModifier.ShiftModifier
LEFT = Qt.MouseButton.LeftButton
RIGHT = Qt.MouseButton.RightButton
# Polygons in raw pixels (row, col) of the 80 x 60 photograph.
POLYGON_A = [(5.0, 5.0), (5.0, 35.0), (30.0, 35.0), (30.0, 5.0)]
POLYGON_B = [(10.0, 45.0), (10.0, 75.0), (50.0, 75.0), (50.0, 45.0)]
PENTAGON = [(35.0, 5.0), (35.0, 20.0), (45.0, 25.0), (55.0, 20.0), (55.0, 5.0)]
TOO_FEW = (
    "A polygon needs at least three points; use Delete polygon in its "
    "right-click menu."
)


class _Project:
    def __init__(self, root):
        self.root = root
        self.manifest = {"rig": {"type": "medium"}}

    def update_metadata(self, *args, **kwargs):
        pass


@pytest.fixture
def view(qtbot, tmp_path, monkeypatch):
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
    widget = LayerPainterView(_Project(tmp_path / "project"), lambda: reference)
    qtbot.addWidget(widget)
    widget.resize(1400, 900)
    widget.show()
    qtbot.waitExposed(widget)
    # The default layers: 1 Free water, 2 Fine sand, 3 Coarse sand, 4 Excluded.
    assert [layer.layer_id for layer in widget.session.layers] == [1, 2, 3, 4]
    # The photograph is shown at full size: one display pixel per raw pixel.
    assert widget.canvas._scale() == (1.0, 1.0)
    return widget


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


def _click(canvas, position, modifiers=NO):
    QTest.mouseClick(canvas.viewport(), LEFT, modifiers, position)


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


def _double_click(canvas, position):
    # What the window system sends: press, release, double click, release.
    _click(canvas, position)
    QTest.mouseDClick(canvas.viewport(), LEFT, NO, position)
    _release(canvas, position)


def _key(canvas, key, modifiers=NO):
    QTest.keyClick(canvas, key, modifiers)


def _record(canvas):
    edits = []
    canvas.polygon_edited.connect(lambda index, _points: edits.append(("one", index)))
    canvas.polygons_edited.connect(
        lambda changes: edits.append(("several", sorted(changes)))
    )
    return edits


def _menu_texts(view):
    return [action.text() for action in view._polygon_menu_widget.actions()]


def _menu_action(view, text):
    (action,) = [
        row for row in view._polygon_menu_widget.actions() if row.text() == text
    ]
    return action


# ---------------------------------------------------------------------------
# The two tools


def test_draw_and_edit_buttons_tooltips_and_shortcuts(view):
    marks = ("pan", "polygon", "edit", "paint", "erase")
    corrections = ("correct", "restore", "lasso", "fill")
    assert LayerCanvas.TOOLS == marks + corrections
    buttons = view.tool_buttons
    assert list(buttons) == list(marks + corrections)
    assert [buttons[key].text() for key in buttons] == [
        "Pan",
        "Draw",
        "Edit",
        "Paint",
        "Erase",
        "Correct",
        "Restore",
        "Lasso",
        "Fill",
    ]
    # Edit sits right after Draw; the tools that correct the proposal have
    # their own row below.
    lefts = [buttons[key].x() for key in marks]
    assert lefts == sorted(lefts)
    def tops(keys):
        return {buttons[key].mapToGlobal(QPoint(0, 0)).y() for key in keys}

    assert len(tops(marks)) == 1 and len(tops(corrections)) == 1
    assert min(tops(corrections)) > max(tops(marks))
    assert buttons["polygon"].toolTip() == (
        "Draw a new polygon for the selected layer (L): left click adds a point, "
        "right click or Enter closes it"
    )
    assert buttons["edit"].toolTip() == (
        "Edit existing polygons (V): click a point to select it and drag to move; "
        "Ctrl+click selects more points; drag on empty space selects points in a "
        "rectangle; Delete removes the selected points; double-click or "
        "right-click an edge to add a point"
    )
    assert view.canvas.tool == "polygon" and buttons["polygon"].isChecked()
    shortcuts = {
        shortcut.key().toString(): shortcut
        for shortcut in view.canvas.findChildren(QShortcut)
    }
    shortcuts["V"].activated.emit()
    assert view.canvas.tool == "edit" and buttons["edit"].isChecked()
    assert not view.brush_size.isEnabled()
    shortcuts["L"].activated.emit()
    assert view.canvas.tool == "polygon" and buttons["polygon"].isChecked()
    buttons["edit"].click()
    assert view.canvas.tool == "edit"


def test_draw_shows_no_handles_and_a_click_on_a_point_adds_a_draft_point(view):
    canvas = view.canvas
    view.add_polygon(lp.Polygon(1, list(POLYGON_A)))
    view.select_layer_row(2)  # Coarse sand
    assert canvas.tool == "polygon" and canvas.layer_id == 3
    assert canvas._handle_items == []
    edits = _record(canvas)
    corner = _at(canvas, 5.0, 35.0)
    assert canvas._hit_vertex(QPointF(corner)) == (0, 1)
    # Pressing on the existing point and moving the mouse drags nothing: the
    # press adds the first point of a new polygon there, exactly on the
    # existing point (Follow edges is on by default).
    _press(canvas, corner)
    _move(canvas, corner + QPoint(30, 20))
    _release(canvas, corner + QPoint(30, 20))
    assert len(canvas.draft_points) == 1
    assert canvas.draft_points[0] == pytest.approx(_raw(canvas, corner), abs=0.1)
    assert canvas.draft_points[0] == (5.0, 35.0)
    assert view.session.polygons[0].points == POLYGON_A
    assert edits == [] and canvas.selected_points == []
    assert canvas._handle_items == []
    # A new polygon on the other side of the shared edge, through A's points.
    for row, col in ((30.0, 35.0), (30.0, 42.0), (5.0, 42.0)):
        _click(canvas, _at(canvas, row, col))
    assert len(canvas.draft_points) == 4
    _key(canvas, Qt.Key.Key_Return)
    assert not canvas.drafting()
    assert len(view.session.polygons) == 2
    assert view.session.polygons[0].points == POLYGON_A
    added = view.session.polygons[1]
    assert added.layer_id == 3 and len(added.points) == 4
    assert added.points[1] == pytest.approx(
        _raw(canvas, _at(canvas, 30.0, 35.0)), abs=0.1
    )
    assert added.points[:2] == [(5.0, 35.0), (30.0, 35.0)]
    assert edits == [] and canvas._handle_items == []


# ---------------------------------------------------------------------------
# Edit: selecting and moving


def test_edit_click_selects_a_point_and_a_drag_moves_it_as_one_change(view):
    canvas = view.canvas
    view.add_polygon(lp.Polygon(2, list(POLYGON_B)))
    view.set_tool("edit")
    assert [len(handles) for handles in canvas._handle_items] == [4]
    edits = _record(canvas)
    steps = len(view._undo_stack)
    corner = _at(canvas, 10.0, 45.0)
    _click(canvas, corner)
    assert canvas.selected_points == [(0, 0)]
    chosen, other = canvas._handle_items[0][0], canvas._handle_items[0][1]
    assert chosen.rect().width() > other.rect().width()
    assert chosen.brush().color() == QColor(255, 255, 255)
    assert chosen.pen().widthF() > other.pen().widthF()
    assert chosen.pen().color().lightness() < 60
    assert other.brush().color() == QColor(view.session.layer(2).color)
    # A click, or a press that moves less than a click, changes nothing.
    _drag(canvas, corner, corner + QPoint(2, 1))
    assert view.session.polygons[0].points == POLYGON_B
    assert len(view._undo_stack) == steps and edits == []

    target = corner + QPoint(30, 20)
    _press(canvas, corner)
    _move(canvas, corner + QPoint(15, 10))
    _move(canvas, target)
    start, end = _raw(canvas, corner), _raw(canvas, target)
    moved = (10.0 + end[0] - start[0], 45.0 + end[1] - start[1])
    # Live: the handle and the outline follow the mouse; nothing is stored yet.
    handle = canvas._handle_items[0][0].pos()
    assert (handle.x(), handle.y()) == pytest.approx(
        (canvas.raw_to_scene(*moved).x(), canvas.raw_to_scene(*moved).y())
    )
    first = canvas._polygon_items[0].path().elementAt(0)
    assert (first.x, first.y) == pytest.approx((handle.x(), handle.y()))
    assert view.session.polygons[0].points == POLYGON_B
    _release(canvas, target)
    points = view.session.polygons[0].points
    assert points[0] == pytest.approx(moved)
    assert points[1:] == POLYGON_B[1:]
    assert len(view._undo_stack) == steps + 1 and edits == [("one", 0)]
    assert canvas.selected_points == [(0, 0)]

    view.undo()
    assert view.session.polygons[0].points == POLYGON_B
    assert canvas.selected_points == []
    view.redo()
    assert view.session.polygons[0].points == points


def test_ctrl_click_selects_points_of_two_polygons_and_a_drag_moves_both(view):
    canvas = view.canvas
    view.add_polygon(lp.Polygon(1, list(POLYGON_A)))
    view.add_polygon(lp.Polygon(2, list(POLYGON_B)))
    view.set_tool("edit")
    edits = _record(canvas)
    steps = len(view._undo_stack)
    a, b = _at(canvas, 30.0, 35.0), _at(canvas, 10.0, 45.0)
    _click(canvas, a)
    _click(canvas, b, CTRL)
    assert canvas.selected_points == [(0, 2), (1, 0)]
    # Ctrl+click on a selected point takes it out of the selection.
    _click(canvas, a, CTRL)
    assert canvas.selected_points == [(1, 0)]
    # A plain click on another point selects only that point ...
    _click(canvas, a)
    assert canvas.selected_points == [(0, 2)]
    # ... and Ctrl+press adds a point; the drag that follows moves both.
    target = b + QPoint(-25, 40)
    _drag(canvas, b, target, CTRL)
    start, end = _raw(canvas, b), _raw(canvas, target)
    rows, cols = end[0] - start[0], end[1] - start[1]
    assert view.session.polygons[0].points[2] == pytest.approx(
        (30.0 + rows, 35.0 + cols)
    )
    assert view.session.polygons[1].points[0] == pytest.approx(
        (10.0 + rows, 45.0 + cols)
    )
    assert view.session.polygons[0].points[:2] == POLYGON_A[:2]
    assert view.session.polygons[1].points[1:] == POLYGON_B[1:]
    assert edits == [("several", [0, 1])]
    assert len(view._undo_stack) == steps + 1
    # A plain press on a selected point keeps the whole selection.
    _click(canvas, _at(canvas, 30.0 + rows, 35.0 + cols))
    assert canvas.selected_points == [(0, 2), (1, 0)]

    view.undo()  # one step puts both points back
    assert view.session.polygons[0].points == POLYGON_A
    assert view.session.polygons[1].points == POLYGON_B
    assert len(view._undo_stack) == steps


def test_a_dragged_selection_stops_together_at_the_edge_of_the_photograph(view):
    canvas = view.canvas
    view.add_polygon(lp.Polygon(1, list(POLYGON_A)))
    view.set_tool("edit")
    corner = _at(canvas, 5.0, 5.0)
    _click(canvas, corner)
    _click(canvas, _at(canvas, 30.0, 35.0), CTRL)
    # Far above and left of the photograph: the first point stops at (0, 0)
    # and the other point moves by the same offset.
    _drag(canvas, corner, _at(canvas, -40.0, -30.0))
    points = view.session.polygons[0].points
    assert points[0] == pytest.approx((0.0, 0.0))
    assert points[2] == pytest.approx((25.0, 30.0))
    assert points[1] == POLYGON_A[1] and points[3] == POLYGON_A[3]


def test_rubber_band_selects_the_points_inside_and_ctrl_adds_to_them(view):
    canvas = view.canvas
    view.add_polygon(lp.Polygon(1, list(POLYGON_A)))
    view.add_polygon(lp.Polygon(2, list(POLYGON_B)))
    view.set_tool("edit")
    edits = _record(canvas)
    steps = len(view._undo_stack)
    start, end = _at(canvas, 2.0, 2.0), _at(canvas, 8.0, 38.0)
    _press(canvas, start)
    _move(canvas, end)
    assert canvas.band_item.isVisible()
    _release(canvas, end)
    assert not canvas.band_item.isVisible()
    assert canvas.selected_points == [(0, 0), (0, 1)]
    assert canvas._handle_items[0][0].rect().width() > (
        canvas._handle_items[0][2].rect().width()
    )
    _drag(canvas, _at(canvas, 47.0, 42.0), _at(canvas, 55.0, 78.0), CTRL)
    assert canvas.selected_points == [(0, 0), (0, 1), (1, 2), (1, 3)]
    # Without Ctrl a new rectangle replaces the selection.
    _drag(canvas, _at(canvas, 27.0, 32.0), _at(canvas, 33.0, 38.0))
    assert canvas.selected_points == [(0, 2)]
    # A click on empty space clears it.
    _click(canvas, _at(canvas, 40.0, 20.0))
    assert canvas.selected_points == []
    assert edits == [] and len(view._undo_stack) == steps
    assert view.session.polygons[0].points == POLYGON_A
    assert view.session.polygons[1].points == POLYGON_B


# ---------------------------------------------------------------------------
# Edit: keys


def test_delete_removes_the_selected_points_as_one_change(view):
    canvas = view.canvas
    view.add_polygon(lp.Polygon(3, list(PENTAGON)))
    view.add_polygon(lp.Polygon(2, list(POLYGON_B)))
    view.set_tool("edit")
    notices = []
    canvas.notice.connect(notices.append)
    steps = len(view._undo_stack)
    # Two points of the pentagon (it keeps three) and two of the square (it
    # would keep two).
    _click(canvas, _at(canvas, 35.0, 20.0))
    for row, col in ((45.0, 25.0), (10.0, 45.0), (10.0, 75.0)):
        _click(canvas, _at(canvas, row, col), CTRL)
    assert canvas.selected_points == [(0, 1), (0, 2), (1, 0), (1, 1)]
    _key(canvas, Qt.Key.Key_Delete)
    assert view.session.polygons[0].points == [PENTAGON[0], PENTAGON[3], PENTAGON[4]]
    assert view.session.polygons[1].points == POLYGON_B
    assert notices == [TOO_FEW]
    assert view.canvas_hint.text() == TOO_FEW
    assert len(view._undo_stack) == steps + 1
    # The square's points stay selected; nothing more can be taken from it.
    assert canvas.selected_points == [(1, 0), (1, 1)]
    _key(canvas, Qt.Key.Key_Delete)
    assert view.session.polygons[1].points == POLYGON_B
    assert len(view._undo_stack) == steps + 1 and notices == [TOO_FEW] * 2

    view.undo()
    assert view.session.polygons[0].points == PENTAGON
    assert canvas.selected_points == []
    # Backspace removes the selected points too; a polygon may keep three.
    _click(canvas, _at(canvas, 55.0, 20.0))
    _click(canvas, _at(canvas, 55.0, 5.0), CTRL)
    _key(canvas, Qt.Key.Key_Backspace)
    assert view.session.polygons[0].points == PENTAGON[:3]
    assert canvas.selected_points == [] and notices == [TOO_FEW] * 2


def test_escape_clears_the_selection_and_arrow_keys_nudge_it(view):
    canvas = view.canvas
    view.add_polygon(lp.Polygon(2, list(POLYGON_B)))
    view.set_tool("edit")
    steps = len(view._undo_stack)
    _click(canvas, _at(canvas, 10.0, 45.0))
    _click(canvas, _at(canvas, 10.0, 75.0), CTRL)
    # One screen pixel in raw pixels (the photograph is shown at full size).
    step = 1.0 / canvas.transform().m11()
    assert 0.0 < step < 1.0
    _key(canvas, Qt.Key.Key_Right)
    points = view.session.polygons[0].points
    assert points[0] == pytest.approx((10.0, 45.0 + step))
    assert points[1] == pytest.approx((10.0, 75.0 + step))
    assert points[2:] == POLYGON_B[2:]
    assert len(view._undo_stack) == steps + 1
    _key(canvas, Qt.Key.Key_Down, SHIFT)
    points = view.session.polygons[0].points
    assert points[0] == pytest.approx((10.0 + 10.0 * step, 45.0 + step))
    assert len(view._undo_stack) == steps + 2
    _key(canvas, Qt.Key.Key_Left)
    _key(canvas, Qt.Key.Key_Up)
    assert view.session.polygons[0].points[1] == pytest.approx(
        (10.0 + 9.0 * step, 75.0)
    )
    assert len(view._undo_stack) == steps + 4
    assert canvas.selected_points == [(0, 0), (0, 1)]

    _key(canvas, Qt.Key.Key_Escape)
    assert canvas.selected_points == []
    before = list(view.session.polygons[0].points)
    _key(canvas, Qt.Key.Key_Right)  # nothing selected: nothing moves
    _key(canvas, Qt.Key.Key_Delete)
    assert view.session.polygons[0].points == before
    assert len(view._undo_stack) == steps + 4
    # Esc during a drag puts the point back and changes nothing.
    corner = _at(canvas, *before[2])
    _press(canvas, corner)
    _move(canvas, corner + QPoint(-40, -30))
    assert canvas._handle_items[0][2].pos() != canvas.raw_to_scene(*before[2])
    _key(canvas, Qt.Key.Key_Escape)
    assert canvas._handle_items[0][2].pos() == canvas.raw_to_scene(*before[2])
    _move(canvas, corner + QPoint(-60, -30))
    _release(canvas, corner + QPoint(-60, -30))
    assert view.session.polygons[0].points == before
    assert canvas.selected_points == [] and len(view._undo_stack) == steps + 4


# ---------------------------------------------------------------------------
# Edit: adding points


def test_double_click_on_an_edge_inserts_a_point_and_selects_it(view):
    canvas = view.canvas
    view.add_polygon(lp.Polygon(2, list(POLYGON_B)))
    view.set_tool("edit")
    steps = len(view._undo_stack)
    # Three screen pixels below the top edge, between its two points.
    spot = _at(canvas, 10.0, 60.0) + QPoint(0, 3)
    _double_click(canvas, spot)
    points = view.session.polygons[0].points
    assert len(points) == 5
    assert points[0] == POLYGON_B[0] and points[2:] == POLYGON_B[1:]
    assert points[1] == pytest.approx((10.0, _raw(canvas, spot)[1]))
    assert canvas.selected_points == [(0, 1)]
    assert canvas._handle_items[0][1].brush().color() == QColor(255, 255, 255)
    assert len(view._undo_stack) == steps + 1

    view.undo()
    assert view.session.polygons[0].points == POLYGON_B
    assert canvas.selected_points == []
    # On a point, or far from every edge, a double click adds nothing.
    _double_click(canvas, _at(canvas, 10.0, 45.0))
    _double_click(canvas, _at(canvas, 30.0, 60.0))
    _double_click(canvas, _at(canvas, 30.0, 20.0))
    assert view.session.polygons[0].points == POLYGON_B
    assert len(view._undo_stack) == steps


def test_right_click_menu_in_edit_removes_the_selected_points(view):
    canvas = view.canvas
    view.add_polygon(lp.Polygon(3, list(PENTAGON)))
    view.set_tool("edit")
    steps = len(view._undo_stack)
    _click(canvas, _at(canvas, 35.0, 20.0))
    _click(canvas, _at(canvas, 45.0, 25.0), CTRL)
    QTest.mouseClick(canvas.viewport(), RIGHT, NO, _at(canvas, 45.0, 25.0))
    assert _menu_texts(view) == [
        "Polygon of Coarse sand",
        "Remove this point",
        "Remove 2 selected points",
        "Join to neighbour's edge…",
        "Move to layer",
        "",
        "Delete polygon",
    ]
    # Not on a selected point: only that point can be removed.
    QTest.mouseClick(canvas.viewport(), RIGHT, NO, _at(canvas, 55.0, 5.0))
    assert "Remove this point" in _menu_texts(view)
    assert not any(text.endswith("selected points") for text in _menu_texts(view))
    QTest.mouseClick(canvas.viewport(), RIGHT, NO, _at(canvas, 35.0, 20.0))
    _menu_action(view, "Remove 2 selected points").trigger()
    assert view.session.polygons[0].points == [PENTAGON[0], PENTAGON[3], PENTAGON[4]]
    assert len(view._undo_stack) == steps + 1
    assert canvas.selected_points == []

    # Right-clicking just outside an edge offers "Add point here"; the new
    # point is selected.
    view.undo()
    outside = _at(canvas, 45.0, 5.0) + QPoint(-4, 0)
    assert canvas._hit_polygon(QPointF(outside)) is None
    QTest.mouseClick(canvas.viewport(), RIGHT, NO, outside)
    assert "Add point here" in _menu_texts(view)
    _menu_action(view, "Add point here").trigger()
    points = view.session.polygons[0].points
    assert len(points) == 6 and points[5] == pytest.approx(
        (_raw(canvas, outside)[0], 5.0)
    )
    assert canvas.selected_points == [(0, 5)]


def test_remove_this_point_keeps_the_rest_of_the_selection(view):
    canvas = view.canvas
    view.add_polygon(lp.Polygon(3, list(PENTAGON)))
    view.set_tool("edit")
    _click(canvas, _at(canvas, 55.0, 20.0))
    _click(canvas, _at(canvas, 55.0, 5.0), CTRL)
    view.remove_polygon_point(0, 1)
    assert view.session.polygons[0].points == [
        PENTAGON[0],
        PENTAGON[2],
        PENTAGON[3],
        PENTAGON[4],
    ]
    # The same two points are selected, one place earlier in the polygon.
    assert canvas.selected_points == [(0, 2), (0, 3)]


# ---------------------------------------------------------------------------
# Edit: what takes part and what clears the selection


def test_hidden_layers_points_cannot_be_selected_and_hiding_drops_them(view):
    canvas = view.canvas
    view.add_polygon(lp.Polygon(1, list(POLYGON_A)))
    view.add_polygon(lp.Polygon(2, list(POLYGON_B)))
    view.set_tool("edit")
    _click(canvas, _at(canvas, 5.0, 5.0))
    _click(canvas, _at(canvas, 10.0, 45.0), CTRL)
    assert canvas.selected_points == [(0, 0), (1, 0)]
    view.set_layer_visible(2, False)
    assert canvas.selected_points == [(0, 0)]
    assert [len(handles) for handles in canvas._handle_items] == [4, 0]
    # A point of the hidden layer is empty space: Ctrl+click keeps the
    # selection, a click clears it, a rectangle over it selects nothing.
    _click(canvas, _at(canvas, 10.0, 45.0), CTRL)
    assert canvas.selected_points == [(0, 0)]
    _click(canvas, _at(canvas, 10.0, 45.0))
    assert canvas.selected_points == []
    _drag(canvas, _at(canvas, 7.0, 42.0), _at(canvas, 53.0, 78.0))
    assert canvas.selected_points == []
    _double_click(canvas, _at(canvas, 10.0, 60.0))
    assert view.session.polygons[1].points == POLYGON_B
    canvas.set_selection([(1, 0), (0, 1)])
    assert canvas.selected_points == [(0, 1)]

    view.set_layer_visible(2, True)
    _drag(canvas, _at(canvas, 7.0, 42.0), _at(canvas, 53.0, 78.0))
    assert canvas.selected_points == [(1, 0), (1, 1), (1, 2), (1, 3)]


def test_undo_tools_layers_and_reload_clear_the_selection(view):
    canvas = view.canvas
    view.add_polygon(lp.Polygon(1, list(POLYGON_A)))
    view.add_polygon(lp.Polygon(2, list(POLYGON_B)))
    view.set_tool("edit")

    def select():
        canvas.set_selection([(0, 0), (1, 0)])
        assert canvas.selected_points == [(0, 0), (1, 0)]

    select()
    view.set_tool("polygon")
    assert canvas.selected_points == [] and canvas._handle_items == []
    canvas.set_selection([(0, 0)])  # Draw has no selection
    assert canvas.selected_points == []
    view.set_tool("edit")
    select()
    view.set_tool("edit")  # the same tool again keeps it
    assert canvas.selected_points == [(0, 0), (1, 0)]

    view.delete_polygon(1)
    assert canvas.selected_points == []
    view.undo()
    select()
    view.move_polygon(0, 3)
    assert canvas.selected_points == []
    select()
    view.undo()
    assert canvas.selected_points == []
    select()
    view.redo()
    assert canvas.selected_points == []
    # Undoing a stroke leaves the polygons, and the selection, as they are.
    view.add_stroke(lp.Stroke(4, 3.0, [(55.0, 70.0)]))
    select()
    view.undo()
    assert canvas.selected_points == [(0, 0), (1, 0)]

    view.select_layer_row(3)  # Excluded, which has no polygon
    view.remove_layer(confirm=False)
    assert canvas.selected_points == []
    select()
    # Coarse sand holds A since the redo above: A goes and B moves up.
    assert view.session.polygons[0].layer_id == 3
    view.select_layer_row(2)
    view.clear_layer_marks(confirm=False)
    assert canvas.selected_points == [] and len(view.session.polygons) == 1
    canvas.set_selection([(0, 0)])
    view.reload()
    assert canvas.selected_points == [] and canvas.tool == "polygon"


def test_edit_never_starts_a_polygon_a_stroke_or_a_draft(view):
    canvas = view.canvas
    view.add_polygon(lp.Polygon(1, list(POLYGON_A)))
    view.set_tool("edit")
    started = []
    canvas.draft_changed.connect(lambda count: started.append(("draft", count)))
    canvas.polygon_finished.connect(lambda _polygon: started.append("polygon"))
    canvas.stroke_finished.connect(lambda _stroke: started.append("stroke"))
    for row, col in ((40.0, 60.0), (20.0, 20.0), (5.0, 5.0), (-5.0, -5.0)):
        _click(canvas, _at(canvas, row, col))
        assert not canvas.drafting() and canvas._points == []
    _drag(canvas, _at(canvas, 40.0, 50.0), _at(canvas, 55.0, 70.0))
    _double_click(canvas, _at(canvas, 40.0, 60.0))
    _double_click(canvas, _at(canvas, 20.0, 20.0))
    for key in (Qt.Key.Key_Return, Qt.Key.Key_Backspace, Qt.Key.Key_Escape):
        _key(canvas, key)
    assert started == []
    assert canvas.draft_points == [] and canvas._points == []
    assert canvas.active_item.path().isEmpty()
    assert len(view.session.polygons) == 1 and view.session.strokes == []


def test_the_session_saved_after_edits_round_trips(view):
    canvas = view.canvas
    view.add_polygon(lp.Polygon(1, list(POLYGON_A)))
    view.add_polygon(lp.Polygon(3, list(PENTAGON)))
    view.add_stroke(lp.Stroke(2, 3.0, [(40.0, 50.0), (40.0, 70.0)]))
    view.save_session()
    path = lp.session_path(view.project, "ac52")
    keys = set(json.loads(path.read_text(encoding="utf-8")))
    view.set_tool("edit")
    corner = _at(canvas, 5.0, 35.0)
    _click(canvas, corner)
    _click(canvas, _at(canvas, 35.0, 20.0), CTRL)
    _drag(canvas, corner, corner + QPoint(17, -9))
    _key(canvas, Qt.Key.Key_Left, SHIFT)
    _double_click(canvas, _at(canvas, 17.0, 5.0))
    _click(canvas, _at(canvas, 55.0, 20.0))
    _key(canvas, Qt.Key.Key_Delete)
    assert [len(row.points) for row in view.session.polygons] == [5, 4]
    view.save_session()

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert set(saved) == keys and saved["format"] == lp.LAYER_PAINTING_FORMAT
    assert saved["polygons"] == [row.to_list() for row in view.session.polygons]
    assert saved["strokes"] == [row.to_list() for row in view.session.strokes]
    loaded = lp.PaintingSession.from_dict(saved)
    assert loaded.to_dict() == saved
    view.reload()
    assert [row.to_list() for row in view.session.polygons] == saved["polygons"]
    assert [row.to_list() for row in view.session.strokes] == saved["strokes"]
