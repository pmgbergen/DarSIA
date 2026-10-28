"""The gap-fill drawing tool edits shared outlines in one undo transaction."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import toml
from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QHBoxLayout, QWidget
from shapely.geometry import Polygon

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower.painting_fill_region_tool import FillRegionTool
from darsia.fluidflower.series_geometry import RigFrame

from .test_painting_corrections import view, _proposal, _show  # noqa: F401


@pytest.fixture(autouse=True)
def private_settings(tmp_path):
    old = QSettings.defaultFormat()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    yield
    QSettings.setDefaultFormat(old)


@pytest.fixture
def tool(view, monkeypatch, qtbot):
    frame = RigFrame(corners={"top_left": (0., 0.), "top_right": (0., 79.),
                              "bottom_right": (59., 79.), "bottom_left": (59., 0.)},
                     width_m=.079, height_m=.059, run_id=view.session.run_id,
                     image=view.session.image_path, image_size=(80, 60))
    monkeypatch.setattr(view, "current_frame", lambda: frame)
    controller = FillRegionTool(view)
    bar = QWidget(view)
    layout = QHBoxLayout(bar)
    for widget in (controller.button, controller.cut_button, controller.clear_cut_button, controller.note):
        layout.addWidget(widget)
    view.layout().addWidget(bar)
    bar.show()
    qtbot.waitExposed(view)
    return controller


def _at(view, row, col):
    return view.canvas.mapFromScene(view.canvas.raw_to_scene(row, col))


def test_moved_reference_photo_keeps_its_saved_frame_editable(view, tool, tmp_path, monkeypatch):
    old = Path(view.session.image_path)
    new = tmp_path / "moved" / old.name
    new.parent.mkdir()
    old.rename(new)
    registry = tmp_path / "private-registry.toml"
    registry.write_text(toml.dumps(dict(schema_version=1,
        storage=dict(path_aliases={str(old): str(new)}))), encoding="utf-8")
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(registry))
    view.session.image_path = str(new)
    assert tool._problem() == ''
    view.session.image_path = str(new.with_name('different.png'))
    assert 'another reference photograph' in tool._problem()


def _click(view, row, col):
    QTest.mouseClick(view.canvas.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, _at(view, row, col))


def _draw_cut(view, tool, first=(8., 30.), last=(22., 30.)):
    tool.cut_button.click()
    QTest.mousePress(view.canvas.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, _at(view, *first))
    QTest.mouseMove(view.canvas.viewport(), _at(view, *last))
    QTest.mouseRelease(view.canvas.viewport(), Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, _at(view, *last))


def _bands(view):
    view.session.polygons = [
        lp.Polygon(2, [(0., 0.), (0., 79.), (10., 79.), (10., 0.)]),
        lp.Polygon(4, [(20., 0.), (20., 79.), (59., 79.), (59., 0.)]),
    ]
    view._render_marks()


def test_actual_click_adds_shared_frame_intersections_with_one_undo_and_keeps_products(view, tool):
    donor = lp.Polygon(2, [(-5., 40.), (30., 20.), (65., 40.), (65., 90.), (-5., 90.)])
    view.session.polygons = [donor]
    view.session.strokes = [lp.Stroke(2, 1., [(40., 8.)])]
    proposal = _proposal()
    _show(view, proposal)
    view.corrections = [lp.Correction("brush", 2, [(5., 5.)], radius_px=1.)]
    view.select_layer_row(2)
    view._render_marks()
    before = deepcopy(view.session.polygons), deepcopy(view.session.strokes), deepcopy(view.corrections)
    arrays = proposal.classes.copy(), proposal.confidence.copy()
    undo_count = len(view._undo_stack)
    tool.button.click()
    assert tool.active and tool.button.isChecked() and view.canvas.tool == "pan"
    assert sum(button.isChecked() for button in view.tool_group.buttons()) == 1
    _click(view, 30., 8.)
    assert len(view.session.polygons) == 2
    updated, filled = view.session.polygons
    assert filled.layer_id == 3
    assert [point for point in updated.points if point in donor.points] == donor.points
    assert Polygon(updated.points).boundary.hausdorff_distance(Polygon(donor.points).boundary) < 1.e-10
    assert Polygon(updated.points).symmetric_difference(Polygon(donor.points)).area < 1.e-10
    inserted = set(updated.points) - set(donor.points)
    assert len(inserted) == 2 and inserted.issubset(filled.points)
    assert len(view._undo_stack) == undo_count + 1
    assert view.session.strokes == before[1] and view.corrections == before[2]
    assert view.proposal is proposal
    assert np.array_equal(proposal.classes, arrays[0]) and np.array_equal(proposal.confidence, arrays[1])
    view.undo()
    assert view.session.polygons == before[0]
    view.redo()
    assert len(view.session.polygons) == 2 and set(inserted).issubset(view.session.polygons[0].points)


def test_fill_needs_no_proposal_and_hidden_polygons_of_all_roles_bound_gap(view, tool):
    _bands(view)
    assert view.proposal is None
    view._set_hidden_layers({2, 4})
    view.select_layer_row(2)
    tool.button.click()
    _click(view, 15., 40.)
    assert len(view.session.polygons) == 3
    filled = Polygon(view.session.polygons[-1].points)
    assert filled.equals(Polygon([(10., 0.), (10., 79.), (20., 79.), (20., 0.)]))
    assert view.proposal is None


def test_dragged_cut_is_temporary_dashed_and_limits_next_fill(view, tool):
    _bands(view)
    view.select_layer_row(2)
    before = deepcopy(view.session), deepcopy(view.corrections), len(view._undo_stack)
    _draw_cut(view, tool)
    assert tool.cut_line and tool.button.isChecked()
    assert tool._cut_item.pen().style() == Qt.PenStyle.DashLine
    assert view.session == before[0] and view.corrections == before[1]
    assert len(view._undo_stack) == before[2]
    cut_col = tool.cut_line[0][1]
    _click(view, 15., 60.)
    assert not tool.cut_line and tool._cut_item is None
    filled = Polygon(view.session.polygons[-1].points)
    assert filled.area == pytest.approx((79. - cut_col) * 10.)
    assert filled.bounds[1] == pytest.approx(cut_col)
    for polygon, row in zip(view.session.polygons[:2], (10., 20.)):
        assert any(point[0] == row and point[1] == pytest.approx(cut_col) for point in polygon.points)
    assert len(view._undo_stack) == before[2] + 1
    view.undo()
    assert view.session.polygons == before[0].polygons


def test_clear_and_escape_remove_only_temporary_cut(view, tool):
    _bands(view)
    before = deepcopy(view.session), len(view._undo_stack)
    _draw_cut(view, tool)
    tool.clear_cut_button.click()
    assert not tool.cut_line and tool._cut_item is None
    _draw_cut(view, tool)
    QTest.keyClick(view.canvas, Qt.Key.Key_Escape)
    assert not tool.active and not tool.cut_line and tool._cut_item is None
    assert view.tool_buttons["pan"].isChecked()
    assert view.session == before[0] and len(view._undo_stack) == before[1]


def test_selecting_existing_tool_deactivates_custom_left_interception(view, tool):
    _bands(view)
    _draw_cut(view, tool)
    view.tool_buttons["polygon"].click()
    assert not tool.active and not tool.cut_line
    assert view.canvas.tool == "polygon" and view.tool_buttons["polygon"].isChecked()
    tool.button.click()
    # The shortcut's set_tool path checks an existing button too.
    view.set_tool("edit")
    assert not tool.active and view.canvas.tool == "edit"
    assert sum(button.isChecked() for button in view.tool_group.buttons()) == 1


@pytest.mark.parametrize("reason", ["busy", "other_photo", "missing_frame", "readonly"])
def test_unavailable_context_refuses_without_any_mark_or_undo_change(view, tool, monkeypatch, reason):
    _bands(view)
    view.select_layer_row(2)
    before = deepcopy(view.session), deepcopy(view.corrections), deepcopy(view._undo_stack)
    if reason == "busy":
        monkeypatch.setattr(view, "busy", lambda: True)
    elif reason == "other_photo":
        monkeypatch.setattr(view, "_other_photo", Path("another.png"))
    elif reason == "missing_frame":
        monkeypatch.setattr(view, "current_frame", lambda: None)
    else:
        monkeypatch.setattr(view.run_preview, "is_reference", lambda: False)
    assert not tool.fill_at(15., 40.)
    tool.refresh()
    assert not tool.button.isEnabled()
    assert view.session == before[0] and view.corrections == before[1]
    assert view._undo_stack == before[2]


def test_actual_other_run_preview_is_not_watched_or_editable(view, tool):
    controller = view.run_preview
    controller.run_inputs = lambda: [SimpleNamespace(run_id=view.session.run_id), SimpleNamespace(run_id="target")]
    controller._index = ""
    controller.refresh()
    before = deepcopy(view.session), deepcopy(view._undo_stack)
    controller.runs.setCurrentIndex(controller.runs.findData("target"))
    assert not tool.fill_at(15., 40.)
    assert view.session == before[0] and view._undo_stack == before[1]
    assert tool.canvas is view.canvas and tool.canvas is not controller.canvas


def test_covered_click_and_unrepresentable_hole_do_not_create_undo(view, tool):
    view.session.polygons = [lp.Polygon(2, [(10., 10.), (10., 20.), (20., 20.), (20., 10.)])]
    before = deepcopy(view.session), deepcopy(view._undo_stack)
    assert not tool.fill_at(15., 15.)
    assert "covered" in tool.note.text()
    assert not tool.fill_at(30., 40.)
    assert "enclosed" in tool.note.text()
    assert view.session == before[0] and view._undo_stack == before[1]


def test_failed_add_rolls_back_insertions_undo_redo_and_dirty_flags(view, tool, monkeypatch):
    view.session.polygons = [lp.Polygon(2, [(-5., 40.), (30., 20.), (65., 40.), (65., 90.), (-5., 90.)])]
    view.select_layer_row(2)
    before = deepcopy(view.session), deepcopy(view.corrections), deepcopy(view._undo_stack), deepcopy(view._redo_stack)
    flags = view._dirty, view._export_stale

    def refuse(_polygon):
        raise ValueError("Synthetic application failure")

    monkeypatch.setattr(view.session, "add_polygon", refuse)
    assert not tool.fill_at(30., 8.)
    assert view.session == before[0] and view.corrections == before[1]
    assert view._undo_stack == before[2] and view._redo_stack == before[3]
    assert (view._dirty, view._export_stale) == flags
    assert "Nothing was changed" in tool.note.text()


def test_canvas_destruction_and_late_events_do_not_call_deleted_qt_objects(qtbot):
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QButtonGroup, QToolButton
    from shiboken6 import isValid

    from darsia.fluidflower.series_layer_view import LayerCanvas

    host = QWidget()
    canvas = LayerCanvas()
    canvas.setParent(host)
    host.canvas = canvas
    host.session = lp.PaintingSession.default("run", "synthetic.png", (10, 10))
    host.image = object()
    host._other_photo = None
    host.busy = lambda: False
    host._canvas_notice = lambda _message: None
    host.current_frame = lambda: RigFrame(
        corners={"top_left": (0., 0.), "top_right": (0., 9.),
                 "bottom_right": (9., 9.), "bottom_left": (9., 0.)},
        image="synthetic.png", image_size=(10, 10))
    host.tool_group = QButtonGroup(host)
    pan = QToolButton(host)
    pan.setCheckable(True)
    host.tool_group.addButton(pan)
    host.tool_buttons = {"pan": pan}
    host.set_tool = lambda name: (canvas.set_tool(name), pan.setChecked(True))
    canvas.set_photo(np.zeros((10, 10, 3), np.uint8), (10, 10), token="teardown", reset_view=True)
    canvas.paint_enabled = True
    canvas.layer_id = host.session.layers[0].layer_id
    controller = FillRegionTool(host)
    assert not controller._problem()
    controller.button.setChecked(True)
    assert controller.active
    try:
        canvas.deleteLater()
        qtbot.waitUntil(lambda: not isValid(canvas), timeout=3000)
        assert not controller.active
        controller.refresh()
        assert controller.eventFilter(host, QEvent(QEvent.Type.Paint)) is False
    finally:
        host.deleteLater()
        qtbot.waitUntil(lambda: not isValid(host), timeout=3000)
