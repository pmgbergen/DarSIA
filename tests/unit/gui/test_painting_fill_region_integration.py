"""The main painting view owns one visible, context-bound drawing fill tool."""

from copy import deepcopy
from types import SimpleNamespace

from PIL import Image
import pytest
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtTest import QTest
from shapely.geometry import Polygon

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower.painting_fill_region_tool import FillRegionTool
from darsia.fluidflower.series_geometry import RigFrame

from .test_painting_corrections import view  # noqa: F401


@pytest.fixture(autouse=True)
def private_settings(tmp_path):
    previous = QSettings.defaultFormat()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      str(tmp_path / "settings"))
    yield
    QSettings.setDefaultFormat(previous)


@pytest.fixture
def integrated(view, monkeypatch):
    frame = RigFrame(
        corners={"top_left": (0., 0.), "top_right": (0., 79.),
                 "bottom_right": (59., 79.), "bottom_left": (59., 0.)},
        width_m=.079, height_m=.059, run_id=view.session.run_id,
        image=view.session.image_path, image_size=(80, 60))
    monkeypatch.setattr(view, "current_frame", lambda: frame)
    view._refresh_status()
    return view.fill_region_tool


def _at(view, row, col):
    return view.canvas.mapFromScene(view.canvas.raw_to_scene(row, col))


def _bands(view):
    view.session.polygons = [
        lp.Polygon(2, [(0., 0.), (0., 79.), (10., 79.), (10., 0.)]),
        lp.Polygon(4, [(20., 0.), (20., 79.), (59., 79.), (59., 0.)]),
    ]
    view._render_marks()
    view.select_layer_row(2)


def _cut(view, tool):
    tool.cut_button.click()
    QTest.mousePress(view.canvas.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, _at(view, 8., 30.))
    QTest.mouseMove(view.canvas.viewport(), _at(view, 22., 30.))
    QTest.mouseRelease(view.canvas.viewport(), Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, _at(view, 22., 30.))
    assert tool.cut_line and tool.active


def test_main_view_constructs_one_visible_drawing_fill_tool_without_a_proposal(view, integrated):
    tool = integrated
    assert view.findChildren(FillRegionTool) == [tool]
    assert tool.canvas is view.canvas and tool.canvas is not view.run_preview.canvas
    assert view.proposal is None
    assert view.canvas.tool == "polygon"  # Integrating the tool preserves the default.
    for widget in (tool.button, tool.cut_button, tool.clear_cut_button, tool.note):
        assert widget.isVisible()
    assert tool.button in view.tool_group.buttons()
    assert tool.cut_button in view.tool_group.buttons()
    assert tool.button.isEnabled() and tool.cut_button.isEnabled()
    assert not tool.clear_cut_button.isEnabled()
    assert "polygon" in tool.note.text()
    assert tool.button is not view.tool_buttons["fill"]
    view.tool_buttons["paint"].click()
    assert view.brush_size.isEnabled()
    tool.button.click()
    assert tool.active and not view.brush_size.isEnabled()
    assert sum(button.isChecked() for button in view.tool_group.buttons()) == 1


def test_status_and_busy_signal_update_the_visible_tool_and_clear_temporary_cut(
        view, integrated, monkeypatch):
    tool = integrated
    frame = view.current_frame()
    monkeypatch.setattr(view, "current_frame", lambda: None)
    view._refresh_status()
    assert not tool.button.isEnabled()
    assert "complete rig frame" in tool.note.text()
    monkeypatch.setattr(view, "current_frame", lambda: frame)
    view._refresh_status()
    assert tool.button.isEnabled() and "complete rig frame" not in tool.note.text()
    _bands(view)
    _cut(view, tool)
    before = deepcopy(view.session), deepcopy(view._undo_stack)
    busy = {"value": True}
    monkeypatch.setattr(view, "busy", lambda: busy["value"])
    view.busy_changed.emit(True)
    assert not tool.active and not tool.cut_line and tool._cut_item is None
    assert not tool.button.isEnabled() and "finish" in tool.note.text()
    assert view.session == before[0] and view._undo_stack == before[1]
    busy["value"] = False
    view.busy_changed.emit(False)
    assert tool.button.isEnabled() and not tool.active


def test_visible_cut_and_fill_share_one_undo_step_for_all_polygon_edits(view, integrated):
    tool = integrated
    _bands(view)
    view.session.strokes = [lp.Stroke(2, 1., [(5., 5.)])]
    view.corrections = [lp.Correction("brush", 2, [(6., 6.)], radius_px=1.)]
    before = deepcopy(view.session), deepcopy(view.corrections), len(view._undo_stack)
    _cut(view, tool)
    assert view.session == before[0] and len(view._undo_stack) == before[2]
    cut_column = tool.cut_line[0][1]
    QTest.mouseClick(view.canvas.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, _at(view, 15., 60.))
    assert len(view.session.polygons) == 3
    filled = view.session.polygons[-1]
    assert filled.layer_id == 3
    assert Polygon(filled.points).area == pytest.approx((79. - cut_column) * 10.)
    for index in (0, 1):
        inserted = set(view.session.polygons[index].points) - set(before[0].polygons[index].points)
        assert inserted and inserted.issubset(filled.points)
    assert view.session.strokes == before[0].strokes and view.corrections == before[1]
    assert view.proposal is None
    assert len(view._undo_stack) == before[2] + 1
    assert not tool.cut_line and tool._cut_item is None
    view.undo_button.click()
    assert view.session.polygons == before[0].polygons
    view.redo_button.click()
    assert len(view.session.polygons) == 3


def test_actual_run_chooser_clears_cut_and_blocks_hidden_reference_canvas(view, integrated):
    tool = integrated
    chooser = view.run_preview
    chooser.run_inputs = lambda: [SimpleNamespace(run_id=view.session.run_id),
                                 SimpleNamespace(run_id="target")]
    chooser._index = ""  # A missing index gives a read-only message without a worker.
    chooser.refresh()
    _bands(view)
    _cut(view, tool)
    before = deepcopy(view.session), deepcopy(view._undo_stack)
    chooser.runs.setCurrentIndex(chooser.runs.findData("target"))
    assert not chooser.is_reference() and chooser.stack.currentWidget() is chooser.canvas
    assert not tool.active and not tool.cut_line and tool._cut_item is None
    assert not tool.button.isEnabled() and "read only" in tool.note.text()
    assert not tool.fill_at(15., 60.)
    assert view.session == before[0] and view._undo_stack == before[1]
    chooser.runs.setCurrentIndex(chooser.runs.findData(view.session.run_id))
    assert chooser.is_reference() and tool.button.isEnabled()
    assert not tool.active and not tool.cut_line


def test_other_photo_and_reload_drop_cut_without_changing_drawing_tool_defaults(
        view, integrated, tmp_path):
    tool = integrated
    _bands(view)
    _cut(view, tool)
    before = deepcopy(view.session), deepcopy(view._undo_stack)
    other = tmp_path / "other.png"
    Image.new("RGB", (80, 60), (50, 40, 30)).save(other)
    assert view.show_frame_photo(other)
    assert not tool.active and not tool.cut_line and tool._cut_item is None
    assert not tool.button.isEnabled() and "reference photograph" in tool.note.text()
    assert view.session == before[0] and view._undo_stack == before[1]
    view.show_reference_photo()
    assert tool.button.isEnabled()
    _cut(view, tool)
    view.reload()
    assert view.fill_region_tool is tool
    assert not tool.active and not tool.cut_line and tool._cut_item is None
    assert tool.button.isEnabled() and view.canvas.tool == "polygon"


def test_readonly_saved_grid_cannot_fill_even_with_matching_frame(view, integrated):
    tool = integrated
    before = deepcopy(view.session), deepcopy(view._undo_stack)
    view.canvas.paint_enabled = False
    view._refresh_status()
    assert not tool.button.isEnabled() and "read only" in tool.note.text()
    assert not tool.fill_at(15., 60.)
    assert view.session == before[0] and view._undo_stack == before[1]


def test_closing_main_view_removes_temporary_cut(view, integrated):
    tool = integrated
    _bands(view)
    _cut(view, tool)
    before = deepcopy(view.session), deepcopy(view._undo_stack)
    view.close()
    assert not tool.active and not tool.cut_line and tool._cut_item is None
    assert view.session == before[0] and view._undo_stack == before[1]


def test_proposal_controls_keep_left_column_narrow_and_version_table_accessible(view):
    view.splitter.setSizes([580, 820])
    # Method and shared-edge controls no longer add their minimum widths to
    # the original Snap/Detail row, which used to clip the State column.
    origin = view.proposal_method.parentWidget()
    detail_y = view.detail.mapTo(origin, QPoint()).y()
    method_y = view.proposal_method.mapTo(origin, QPoint()).y()
    shared_y = view.move_shared_edges.mapTo(origin, QPoint()).y()
    assert detail_y < method_y < shared_y
    assert view.steps_scroll.widget().minimumSizeHint().width() <= 560
    assert view.versions_table.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
