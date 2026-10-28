"""Layer painting tab: per-layer visibility (display only) and "Add point here".

A synthetic 80 x 60 photograph and a project in a temporary folder; no real
series, rig model or NAS input is used.
"""

import json
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMenu, QStyle, QStyleOptionViewItem

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower.series_geometry import SeriesRunInputs
from darsia.fluidflower.series_layer_view import (
    LayerCanvas,
    LayerPainterView,
    insert_point_on_nearest_edge,
)

CHECKED = Qt.CheckState.Checked
UNCHECKED = Qt.CheckState.Unchecked
# Polygons in raw pixels (row, col) of the 80 x 60 photograph.
POLYGON_A = [(5.0, 5.0), (5.0, 35.0), (30.0, 35.0), (30.0, 5.0)]
POLYGON_B = [(10.0, 45.0), (10.0, 75.0), (50.0, 75.0), (50.0, 45.0)]
COLORS = {1: QColor("#3d8bd9"), 2: QColor("#f2c14e")}


# ---------------------------------------------------------------------------
# The pure helper


def test_insert_point_projects_onto_the_middle_of_an_edge():
    triangle = [(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)]
    points, index = insert_point_on_nearest_edge(triangle, 6.0, 4.0)
    assert index == 1
    assert points == [(0.0, 0.0), (5.0, 5.0), (10.0, 10.0), (20.0, 0.0)]
    assert triangle == [(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)]


def test_insert_point_on_the_closing_edge_goes_after_the_last_point():
    square = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
    points, index = insert_point_on_nearest_edge(square, 5.0, 1.0)
    assert index == 4
    assert points == square + [(5.0, 0.0)]


def test_insert_point_beyond_the_end_of_an_edge_is_clamped_to_it():
    triangle = [(0.0, 0.0), (0.0, 10.0), (10.0, 0.0)]
    points, index = insert_point_on_nearest_edge(triangle, -3.0, 14.0)
    # The projection onto the first edge's line would be (0, 14); it stops at
    # the edge's end point.
    assert index == 1
    assert points[index] == (0.0, 10.0)
    assert points == [(0.0, 0.0), (0.0, 10.0), (0.0, 10.0), (10.0, 0.0)]


def test_insert_point_uses_the_nearest_of_several_edges():
    rectangle = [(0.0, 0.0), (0.0, 20.0), (10.0, 20.0), (10.0, 0.0)]
    points, index = insert_point_on_nearest_edge(rectangle, 7.0, 10.0)
    assert index == 3
    assert points == [(0.0, 0.0), (0.0, 20.0), (10.0, 20.0), (10.0, 10.0), (10.0, 0.0)]


def test_insert_point_needs_two_points_and_a_finite_position():
    assert insert_point_on_nearest_edge([(1.0, 2.0)], 0.0, 0.0) == ([(1.0, 2.0)], -1)
    square = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
    assert insert_point_on_nearest_edge(square, float("nan"), 1.0) == (square, -1)


# ---------------------------------------------------------------------------
# The canvas


@pytest.fixture
def canvas(qtbot):
    widget = LayerCanvas()
    qtbot.addWidget(widget)
    widget.resize(640, 480)
    widget.set_photo(
        np.zeros((60, 80, 3), np.uint8), (80, 60), token={}, reset_view=True
    )
    widget.paint_enabled = True
    widget.set_tool("polygon")
    return widget


def _spot(canvas, row, col):
    """Viewport position of a raw pixel."""

    return QPointF(canvas.mapFromScene(canvas.raw_to_scene(row, col)))


def _alpha(canvas, row, col):
    # The test photograph is shown at full size: display pixel = raw pixel.
    return canvas._stroke_image.pixelColor(int(col), int(row)).alpha()


def test_canvas_hidden_layers_polygons_are_not_drawn_handled_or_hit(canvas):
    canvas.set_tool("edit")  # point handles are shown by the Edit tool (0002)
    polygons = [lp.Polygon(1, list(POLYGON_A)), lp.Polygon(2, list(POLYGON_B))]
    canvas.render_polygons(polygons, COLORS)
    corner_b, inside_b = _spot(canvas, 10.0, 45.0), _spot(canvas, 30.0, 60.0)
    inside_a = _spot(canvas, 20.0, 20.0)
    assert [item.isVisible() for item in canvas._polygon_items] == [True, True]
    assert [len(handles) for handles in canvas._handle_items] == [4, 4]
    assert canvas._hit_vertex(corner_b) == (1, 0)
    assert canvas.polygon_at(inside_b) == (1, -1)

    canvas.set_hidden_layers({2})
    assert canvas.hidden_layers == {2}
    assert [item.isVisible() for item in canvas._polygon_items] == [True, False]
    assert [len(handles) for handles in canvas._handle_items] == [4, 0]
    assert canvas._hit_vertex(corner_b) is None
    assert canvas.polygon_at(corner_b) is None
    assert canvas.polygon_at(inside_b) is None
    assert canvas.polygon_at(inside_a) == (0, -1)
    contexts = []
    canvas.polygon_context.connect(lambda *args: contexts.append(args))
    canvas._right_click(NS(position=lambda: inside_b))
    assert contexts == []

    # The global marks toggle and a new rendering keep working with it.
    canvas.set_overlays_visible(False, True)
    assert [item.isVisible() for item in canvas._polygon_items] == [False, False]
    canvas.set_overlays_visible(True, True)
    assert [item.isVisible() for item in canvas._polygon_items] == [True, False]
    canvas.render_polygons(polygons, COLORS)
    assert [item.isVisible() for item in canvas._polygon_items] == [True, False]
    assert [len(handles) for handles in canvas._handle_items] == [4, 0]

    canvas.set_hidden_layers(())
    assert [item.isVisible() for item in canvas._polygon_items] == [True, True]
    assert [len(handles) for handles in canvas._handle_items] == [4, 4]
    assert canvas._hit_vertex(corner_b) == (1, 0)
    assert canvas.polygon_at(inside_b) == (1, -1)


def test_canvas_hidden_layers_strokes_are_left_out_of_the_stroke_image(canvas):
    stroke_a = lp.Stroke(1, 3.0, [(20.0, 10.0), (20.0, 30.0)])
    stroke_b = lp.Stroke(2, 3.0, [(40.0, 50.0), (40.0, 70.0)])
    canvas.render_strokes([stroke_a, stroke_b], COLORS)
    assert _alpha(canvas, 20, 20) > 0 and _alpha(canvas, 40, 60) > 0

    canvas.set_hidden_layers({2})
    assert _alpha(canvas, 20, 20) > 0 and _alpha(canvas, 40, 60) == 0
    # Eraser strokes still apply to the strokes shown; a new stroke of a
    # hidden layer is not drawn.
    canvas.add_stroke_to_image(lp.Stroke(lp.ERASER, 4.0, [(20.0, 20.0)]), COLORS)
    canvas.add_stroke_to_image(lp.Stroke(2, 3.0, [(52.0, 52.0)]), COLORS)
    assert _alpha(canvas, 20, 20) == 0 and _alpha(canvas, 20, 12) > 0
    assert _alpha(canvas, 52, 52) == 0

    canvas.set_hidden_layers(())
    assert _alpha(canvas, 40, 60) > 0 and _alpha(canvas, 52, 52) > 0
    assert _alpha(canvas, 20, 20) == 0 and _alpha(canvas, 20, 12) > 0


def test_canvas_right_click_records_the_raw_position_before_the_menu_signal(canvas):
    canvas.render_polygons([lp.Polygon(1, list(POLYGON_A))], COLORS)
    seen = []
    canvas.polygon_context.connect(
        lambda index, vertex, _where: seen.append(
            (index, vertex, canvas.last_context_raw)
        )
    )
    canvas._right_click(NS(position=lambda: _spot(canvas, 20.0, 12.0)))
    assert len(seen) == 1 and seen[0][:2] == (0, -1)
    row, col = seen[0][2]
    assert abs(row - 20.0) < 0.5 and abs(col - 12.0) < 0.5


# ---------------------------------------------------------------------------
# The view


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
    return widget


def _shows(view):
    table = view.layer_table
    return [table.item(row, 4).checkState() for row in range(table.rowCount())]


def _check_box_center(table, item):
    option = QStyleOptionViewItem()
    table.initViewItemOption(option)
    option.rect = table.visualItemRect(item)
    table.itemDelegate().initStyleOption(option, table.indexFromItem(item))
    style = table.style()
    return style.subElementRect(
        QStyle.SubElement.SE_ItemViewItemCheckIndicator, option, table
    ).center()


def test_show_column_hides_a_layer_on_the_canvas_only(view):
    view.set_tool("edit")  # point handles are shown by the Edit tool (0002)
    table = view.layer_table
    assert table.columnCount() == 6
    assert [table.horizontalHeaderItem(column).text() for column in range(6)] == [
        "",
        "Layer",
        "Polygons",
        "Strokes",
        "Show",
        "Top",
    ]
    for row in range(table.rowCount()):
        item = table.item(row, 4)
        assert item.flags() & Qt.ItemFlag.ItemIsUserCheckable
        assert item.checkState() == CHECKED
        assert item.toolTip() == (
            "Show or hide this layer using its own colour (By layer, display only)"
        )
    view.add_polygon(lp.Polygon(2, list(POLYGON_B)))
    view.add_stroke(lp.Stroke(2, 3.0, [(40.0, 50.0), (40.0, 70.0)]))
    view.save_session()
    path = lp.session_path(view.project, "ac52")
    saved = path.read_bytes()
    session = view.session
    before = session.to_dict()
    digest = lp.proposal_digest(session, view.current_frame())
    changes = []
    view.changed.connect(lambda: changes.append(True))

    table.item(1, 4).setCheckState(UNCHECKED)  # Fine sand
    assert view.hidden_layers == {2} and view.canvas.hidden_layers == {2}
    assert not view.layer_visible(2)
    assert not view.canvas._polygon_items[0].isVisible()
    assert view.canvas._handle_items == [[]]
    assert _alpha(view.canvas, 40, 60) == 0
    assert view._dirty is False and not view._save_timer.isActive()
    assert changes == []
    view.save_session()
    assert path.read_bytes() == saved
    assert session.to_dict() == before
    assert lp.proposal_digest(session, view.current_frame()) == digest
    # Saving while the layer is hidden writes the same session.
    view._dirty = True
    view.save_session()
    first, second = json.loads(saved), json.loads(path.read_bytes())
    for document in (first, second):
        document.pop("updated_utc")
        document.pop("updated_by")
    assert second == first

    table.item(1, 4).setCheckState(CHECKED)
    assert view.hidden_layers == set()
    assert view.canvas._polygon_items[0].isVisible()
    assert _alpha(view.canvas, 40, 60) > 0
    assert view._dirty is False and changes == []


def test_clicking_a_show_box_keeps_the_layer_being_painted(view):
    table = view.layer_table
    view.select_layer_row(0)
    assert view.selected_layer_id() == 1 and view.canvas.layer_id == 1
    item = table.item(2, 4)  # Coarse sand
    QTest.mouseClick(
        table.viewport(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        _check_box_center(table, item),
    )
    assert view.hidden_layers == {3}
    assert table.item(2, 4).checkState() == UNCHECKED
    assert view.selected_layer_id() == 1 and view.canvas.layer_id == 1
    # Selecting a row does not change what is shown.
    view.select_layer_row(2)
    assert view.canvas.layer_id == 3 and view.hidden_layers == {3}


def test_hidden_layers_survive_rebuilds_undo_and_redo(view):
    view.add_polygon(lp.Polygon(2, list(POLYGON_B)))
    view.set_layer_visible(2, False)
    view._refresh_layers()
    assert view.hidden_layers == {2}
    assert _shows(view) == [CHECKED, UNCHECKED, CHECKED, CHECKED]
    view.add_polygon(lp.Polygon(3, list(POLYGON_A)))
    assert view.hidden_layers == {2}
    view.undo()
    assert len(view.session.polygons) == 1 and view.hidden_layers == {2}
    assert [item.isVisible() for item in view.canvas._polygon_items] == [False]
    view.redo()
    assert view.hidden_layers == {2}
    assert [item.isVisible() for item in view.canvas._polygon_items] == [False, True]
    assert _shows(view) == [CHECKED, UNCHECKED, CHECKED, CHECKED]


def test_removed_layer_is_forgotten_and_reload_shows_every_layer(view):
    view.set_layer_visible(4, False)
    view.set_layer_visible(2, False)
    view.select_layer_row(3)
    view.remove_layer(confirm=False)
    assert view.hidden_layers == {2}
    view.add_layer()  # gets the removed layer's ID again
    assert view.session.layers[-1].layer_id == 4 and view.layer_visible(4)
    assert _shows(view) == [CHECKED, UNCHECKED, CHECKED, CHECKED]
    view.reload()
    assert view.hidden_layers == set() and view.canvas.hidden_layers == set()
    assert _shows(view) == [CHECKED] * 4


def test_layer_table_menu_shows_only_one_layer_or_all_layers(view):
    table = view.layer_table
    assert table.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
    position = table.visualItemRect(table.item(2, 1)).center()
    table.customContextMenuRequested.emit(position)
    actions = {
        action.text(): action for action in view._layer_table_menu_widget.actions()
    }
    assert list(actions) == [
        "Show only this layer",
        "Show only this facies",
        "Highlight this facies",
        "Clear facies highlight",
        "Show all layers",
        "",  # separator
        "Snap distance of this layer…",
        "Use the common snap distance",
    ]
    assert not actions["Show all layers"].isEnabled()
    assert not actions["Use the common snap distance"].isEnabled()
    actions["Show only this layer"].trigger()
    assert view.hidden_layers == {1, 2, 4}
    assert _shows(view) == [UNCHECKED, UNCHECKED, CHECKED, UNCHECKED]
    table.customContextMenuRequested.emit(position)
    actions = {
        action.text(): action for action in view._layer_table_menu_widget.actions()
    }
    assert actions["Show all layers"].isEnabled()
    actions["Show all layers"].trigger()
    assert view.hidden_layers == set() and _shows(view) == [CHECKED] * 4
    assert view._dirty is False


def _overlay_without_hiding(view):
    """The proposal overlay as it was built before layers could be hidden."""

    height, width = view.image.display.shape[:2]
    classes = cv2.resize(
        view.proposal.classes, (width, height), interpolation=cv2.INTER_NEAREST
    )
    alpha = int(round(2.55 * view.opacity.value()))
    table = np.zeros((256, 4), dtype=np.uint8)
    table[lp.OUTSIDE] = (0, 0, 0, 110)
    for layer in view.session.layers:
        color = QColor(layer.color)
        table[layer.layer_id] = (color.red(), color.green(), color.blue(), alpha)
    rgba = table[classes]
    boundary = np.zeros(classes.shape, dtype=bool)
    boundary[:, 1:] |= classes[:, 1:] != classes[:, :-1]
    boundary[1:, :] |= classes[1:, :] != classes[:-1, :]
    rgba[boundary] = (255, 255, 255, 235)
    uncertain = cv2.resize(
        view.proposal.uncertain().astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    rows, cols = np.indices(classes.shape)
    rgba[uncertain & (((rows + cols) // 3) % 2 == 0)] = (255, 0, 255, 200)
    return rgba


def test_proposal_overlay_is_transparent_for_hidden_layers(view, monkeypatch):
    classes = np.zeros((60, 80), np.uint8)  # outside the rig frame
    classes[5:55, 5:40] = 2
    classes[5:55, 40:60] = 3
    classes[5:55, 60:75] = 4
    confidence = np.full((60, 80), 255, np.uint8)
    confidence[20:30, 10:70] = 10  # uncertain in all three layers
    view.proposal = lp.LayerProposal(classes, confidence, (80, 60), "synthetic")
    view.show_uncertain.setChecked(True)
    overlays = []
    original = view.canvas.set_proposal
    monkeypatch.setattr(
        view.canvas,
        "set_proposal",
        lambda rgba: (
            overlays.append(None if rgba is None else rgba.copy()),
            original(rgba),
        ),
    )
    view._render_proposal()
    shown = overlays[-1]
    assert np.array_equal(shown, _overlay_without_hiding(view))
    hatch = (255, 0, 255, 200)
    assert (shown[20:30, 45:55] == hatch).all(axis=-1).any()

    view.set_layer_visible(3, False)
    overlay = overlays[-1]
    assert (overlay[6:54, 41:60, 3] == 0).all()  # no fill, no hatch
    # The boundaries of the shown layers 2 and 4 against layer 3 stay (in
    # layer 4, the hatch of uncertain pixels covers them as before).
    assert (overlay[6:54, 40] == (255, 255, 255, 235)).all()
    assert (overlay[6:20, 60] == (255, 255, 255, 235)).all()
    assert (overlay[30:54, 60] == (255, 255, 255, 235)).all()
    unchanged = np.ones(classes.shape, dtype=bool)
    unchanged[5:55, 40:60] = False
    assert np.array_equal(overlay[unchanged], shown[unchanged])

    view.set_layer_visible(4, False)
    overlay = overlays[-1]
    assert (overlay[6:54, 60, 3] == 0).all()  # between two hidden layers
    assert (overlay[6:54, 61:75, 3] == 0).all()
    assert (overlay[6:54, 75] == (255, 255, 255, 235)).all()  # rig frame edge
    assert (overlay[0:5, :] == (0, 0, 0, 110)).all()  # outside colour
    assert np.array_equal(overlay[6:54, 6:40], shown[6:54, 6:40])  # layer 2

    view.show_all_layers()
    assert np.array_equal(overlays[-1], shown)
    assert view._dirty is False
    assert view.proposal.digest == "synthetic"


def test_add_point_here_inserts_one_point_and_undo_restores_the_polygon(view):
    view.set_tool("edit")  # point handles are shown by the Edit tool (0002)
    view.add_polygon(lp.Polygon(2, list(POLYGON_B)))
    original = list(view.session.polygons[0].points)
    canvas = view.canvas
    # Inside the polygon, near its left edge (the closing edge), off its points.
    spot = _spot(canvas, 30.0, 48.0)
    assert canvas.polygon_at(spot) == (0, -1)
    canvas._right_click(NS(position=lambda: spot))
    actions = {action.text(): action for action in view._polygon_menu_widget.actions()}
    assert "Add point here" in actions and "Remove this point" not in actions
    actions["Add point here"].trigger()
    points = view.session.polygons[0].points
    assert len(points) == len(original) + 1
    assert points[:4] == original
    row, col = points[4]
    assert col == 45.0 and abs(row - 30.0) < 0.5
    assert [len(handles) for handles in canvas._handle_items] == [5]

    view.undo()
    assert view.session.polygons[0].points == original
    view.redo()
    assert view.session.polygons[0].points == points

    # On a point, the menu removes that point instead.
    canvas._right_click(NS(position=lambda: _spot(canvas, 10.0, 75.0)))
    actions = {action.text(): action for action in view._polygon_menu_widget.actions()}
    assert "Remove this point" in actions and "Add point here" not in actions


def test_a_new_mark_on_a_hidden_layer_shows_that_layer(view):
    canvas = view.canvas
    view.set_layer_visible(2, False)
    view.select_layer_row(1)  # Fine sand
    assert canvas.layer_id == 2 and view.hidden_layers == {2}
    for row, col in POLYGON_B[:3]:
        canvas._add_draft_point(canvas.raw_to_scene(row, col))
    assert canvas.close_polygon()
    assert len(view.session.polygons) == 1
    assert view.hidden_layers == set()
    assert _shows(view) == [CHECKED] * 4
    assert canvas._polygon_items[0].isVisible()

    view.set_layer_visible(3, False)
    view.add_stroke(lp.Stroke(3, 3.0, [(20.0, 20.0)]))
    assert view.hidden_layers == set()
    assert _alpha(canvas, 20, 20) > 0

    view.set_layer_visible(4, False)
    view.move_polygon(0, 4)
    assert view.hidden_layers == set() and canvas._polygon_items[0].isVisible()
