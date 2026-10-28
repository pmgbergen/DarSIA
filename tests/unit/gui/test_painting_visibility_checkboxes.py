"""One-click display filters on private synthetic paintings never change saved maps."""
from copy import deepcopy
import threading

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import series_layer_view as slv
from darsia.fluidflower import painting_run_preview_ui as preview_ui
from darsia.fluidflower.painting_run_preview import RunPreview
from .test_painting_photo_saturation import settings, make_view, view  # noqa: F401
from .test_painting_corrections import _proposal, _show
from .test_painting_facies_display import _group, _files
from .test_layer_visibility_and_add_point import _check_box_center
from .test_painting_manual_map import LEFT, RIGHT, MIDDLE, _drawn
from .test_painting_run_preview import private_settings  # noqa: F401
from .test_painting_run_preview_status import _chooser, _select


def _facies_item(view, name):
    return next(view.facies_table.item(row, 0) for row in range(view.facies_table.rowCount())
                if view.facies_table.item(row, 0).data(Qt.ItemDataRole.UserRole) == name)


def _click_item(table, item):
    painting = table.window()
    column = item.column()
    if table is painting.facies_table:
        target = item.data(Qt.ItemDataRole.UserRole)
        painting.step_facies.set_expanded(True)
    else:
        target = painting.session.layers[item.row()].layer_id
        painting.step_paint.set_expanded(True)
    painting.steps_scroll.ensureWidgetVisible(table)
    QTest.qWait(1)
    if table is painting.facies_table:
        item = _facies_item(painting, target)
    else:
        row = next(index for index, layer in enumerate(painting.session.layers) if layer.layer_id == target)
        item = table.item(row, column)
    table.scrollToItem(item)
    QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier, _check_box_center(table, item))


def _toggle_facies(view, name):
    _click_item(view.facies_table, _facies_item(view, name))


def _button_states(view, on, off, *, enabled=True):
    for owner in ("layer", "facies"):
        for suffix, active in (("on", on), ("off", off)):
            button = getattr(view, f"{owner}_all_{suffix}_button")
            assert button.isCheckable() and button.isChecked() == (active and owner == view._color_mode)
            assert button.isEnabled() == enabled


def test_facies_table_keeps_named_checkboxes_without_duplicate_photo_controls(view):
    assert view.color_mode.currentText() == "By layer"
    assert not hasattr(view, "facies_visibility") and not hasattr(view, "facies_checks")
    assert not hasattr(view, "legend_all_on_button")
    assert not any(label.text() == "Facies visibility · display only" for label in view.findChildren(QLabel))
    for name in view.session.facies_order():
        item = _facies_item(view, name)
        members = view.session.layers_in_facies(name)
        assert item.text() == name and item.flags() & Qt.ItemFlag.ItemIsUserCheckable
        if members:
            assert item.checkState() == Qt.CheckState.Checked
            assert not item.icon().isNull()
            icon = item.icon().pixmap(14, 14).toImage()
            assert icon.pixelColor(7, 7) == view._layer_colors()[members[0].layer_id]
            assert "display only" in item.toolTip()
    assert view.layer_table.columnCount() == 6
    assert view.layer_table.horizontalHeaderItem(5).text() == "Top"
    assert all(view.layer_table.item(row, 4).checkState() == Qt.CheckState.Checked
               for row in range(view.layer_table.rowCount()))
    for owner in ("layer", "facies"):
        for suffix in ("on", "off"):
            assert getattr(view, f"{owner}_all_{suffix}_button").isVisibleTo(view)
    assert all(button.isVisibleTo(view) for button in view.tool_buttons.values())


@pytest.mark.parametrize("mode", ["layer", "facies"])
def test_facies_show_selects_group_colours_from_either_prior_mode(view, mode):
    name, first, second = _group(view)
    other = next(layer.layer_id for layer in view.session.layers if layer.facies != name)
    view.session.polygons = [
        lp.Polygon(first, [(5., 5.), (5., 35.), (25., 35.), (25., 5.)]),
        lp.Polygon(second, [(30., 45.), (30., 75.), (55., 75.), (55., 45.)]),
        lp.Polygon(other, [(35., 5.), (35., 25.), (55., 25.), (55., 5.)]),
    ]
    view.session.strokes = [lp.Stroke(first, 3., [(20., 20.)]),
                            lp.Stroke(second, 3., [(40., 60.)]),
                            lp.Stroke(other, 3., [(45., 15.)])]
    view._render_marks()
    view.set_color_mode(mode)
    view.show_strokes.setChecked(True)
    classes = np.tile(np.repeat(np.array([first, second, other], np.uint8), 8), (12, 1))
    before = deepcopy(view.session.to_dict())
    assert [item.isVisible() for item in view.canvas._polygon_items] == [True, True, True]
    _toggle_facies(view, name)
    assert view._color_mode == "facies"
    assert view.hidden_layers == {first, second} == view.canvas.hidden_layers
    assert [item.isVisible() for item in view.canvas._polygon_items] == [False, False, True]
    image = view.canvas._stroke_image
    assert image.pixelColor(10, 10).alpha() == 0
    assert image.pixelColor(30, 20).alpha() == 0
    assert image.pixelColor(7, 22).alpha() > 0
    rgba, shown = view._layer_overlay(classes)
    assert not shown[:, :16].any() and shown[:, 16:].all()
    assert rgba[5, 4, 3] == rgba[5, 12, 3] == 0 and rgba[5, 20, 3] > 0
    assert view.session.to_dict() == before
    _toggle_facies(view, name)
    assert not view.hidden_layers
    assert all(item.isVisible() for item in view.canvas._polygon_items)


def test_table_show_clicks_select_independent_modes_and_preserve_the_other_choices(view, monkeypatch):
    name, first, second = _group(view)
    before = deepcopy(view.session.to_dict())
    def forbidden(*args, **kwargs):
        pytest.fail("A visibility click tried to rename a facies")
    monkeypatch.setattr(view, "rename_facies", forbidden)
    _toggle_facies(view, name)
    assert view._color_mode == "facies" and view.hidden_layers == {first, second}
    row = next(index for index, layer in enumerate(view.session.layers) if layer.layer_id == first)
    assert view.layer_table.item(row, 4).checkState() == Qt.CheckState.Checked
    _click_item(view.layer_table, view.layer_table.item(row, 4))
    assert view._color_mode == "layer" and view.hidden_layers == {first}
    assert _facies_item(view, name).checkState() == Qt.CheckState.Unchecked
    _toggle_facies(view, name)
    assert view._color_mode == "facies" and not view.hidden_layers
    assert view._hidden_layers == {first}
    assert view.layer_table.item(row, 4).checkState() == Qt.CheckState.Unchecked
    view.set_color_mode("layer")
    assert view.hidden_layers == {first}
    _click_item(view.layer_table, view.layer_table.item(row, 4))
    assert not view.hidden_layers
    assert view.session.to_dict() == before


@pytest.mark.parametrize("owner", ["layer", "facies"])
def test_all_on_all_off_choose_their_colour_mode_and_existing_show_only_actions(view, owner):
    name, first, second = _group(view)
    view.highlight_facies(name)
    getattr(view, f"{owner}_all_off_button").click()
    assert view.hidden_layers == {layer.layer_id for layer in view.session.layers}
    if owner != "layer":
        assert all(view.facies_table.item(row, 0).checkState() == Qt.CheckState.Unchecked
                   for row in range(view.facies_table.rowCount()))
    else:
        assert _facies_item(view, name).checkState() == Qt.CheckState.Checked
    assert view._highlight_facies is None and view._color_mode == owner
    getattr(view, f"{owner}_all_on_button").click()
    assert not view.hidden_layers and view._highlight_facies is None
    view.show_only_facies(name)
    assert view.hidden_layers == {layer.layer_id for layer in view.session.layers if layer.facies != name}
    view.show_only_layer(first)
    assert view.hidden_layers == {layer.layer_id for layer in view.session.layers if layer.layer_id != first}
    assert _facies_item(view, name).checkState() == Qt.CheckState.Checked
    assert not view.layer_visible(second)


@pytest.mark.parametrize("off_owner,on_owner", [("facies", "layer"), ("layer", "facies"),
                                               ("facies", "facies"), ("layer", "layer")])
def test_all_on_selects_its_mode_and_preserves_the_other_filter_without_saved_changes(view, off_owner, on_owner):
    name, first, _second = _group(view)
    _show(view, _proposal())
    view.highlight_facies(name)
    view.set_layer_visible(first, False)
    view.set_facies_visible(name, False)
    before = (deepcopy(view.session.to_dict()), view.proposal.classes.copy(), _files(view),
              deepcopy(view._undo_stack), deepcopy(view.versions), view._export_state_key())
    masters = view.show_strokes.isChecked(), view.show_proposal.isChecked()
    getattr(view, f"{off_owner}_all_off_button").click()
    assert view.hidden_layers == {row.layer_id for row in view.session.layers}
    _button_states(view, False, True)
    getattr(view, f"{on_owner}_all_on_button").click()
    assert not view.hidden_layers and view._color_mode == on_owner
    if on_owner == "layer":
        assert not view._hidden_layers and view._hidden_facies
    else:
        assert not view._hidden_facies and view._hidden_layers
    _button_states(view, True, False)
    # Clicking the active action again retains the effective state.
    getattr(view, f"{on_owner}_all_on_button").click()
    _button_states(view, True, False)
    assert view._highlight_facies is None
    assert masters == (view.show_strokes.isChecked(), view.show_proposal.isChecked())
    assert view.session.to_dict() == before[0]
    np.testing.assert_array_equal(view.proposal.classes, before[1])
    assert (_files(view), view._undo_stack, view.versions, view._export_state_key()) == before[2:]


def test_visibility_buttons_follow_effective_individual_filters_and_empty_or_loaded_session(view):
    name, first, _second = _group(view)
    _button_states(view, True, False)
    view.set_layer_visible(first, False)
    _button_states(view, False, False)
    view.set_facies_visible(name, False)
    _button_states(view, False, False)
    view.facies_all_off_button.click()
    _button_states(view, False, True)
    # Switching to layer colours uses only the saved individual-layer choices.
    view.set_layer_visible(first, True)
    assert view._color_mode == "layer" and view._hidden_facies
    _button_states(view, True, False)
    view.set_facies_visible(name, True)
    _button_states(view, False, False)
    saved = deepcopy(view.session.to_dict())
    view.session = None
    view._refresh_layers()
    view._refresh_facies_table()
    _button_states(view, False, False, enabled=False)
    view.session = lp.PaintingSession.from_dict(saved)
    view._refresh_layers()
    view._refresh_facies_table()
    _button_states(view, True, False)


def test_checkbox_clicks_preserve_session_proposal_digest_export_key_files_and_undo(view, monkeypatch):
    name, first, _second = _group(view)
    _show(view, _proposal())
    view.save_session()
    before = deepcopy(view.session.to_dict()), view.proposal.classes.copy(), view._export_state_key(), _files(view)
    digest, undo = lp.proposal_digest(view.session, view.current_frame()), deepcopy(view._undo_stack)
    changes = []
    view.changed.connect(lambda: changes.append(True))
    def forbidden(*args, **kwargs):
        pytest.fail("A visibility checkbox tried to save a painting")
    monkeypatch.setattr(lp, "save_painting_session", forbidden)
    for mode in ("facies", "layer"):
        view.set_color_mode(mode)
        _toggle_facies(view, name)
        _click_item(view.facies_table, _facies_item(view, name))
        view.set_layer_visible(first, False)
        view.facies_all_off_button.click()
        view.facies_all_on_button.click()
    assert view.session.to_dict() == before[0]
    np.testing.assert_array_equal(view.proposal.classes, before[1])
    assert view._export_state_key() == before[2] and _files(view) == before[3]
    assert lp.proposal_digest(view.session, view.current_frame()) == digest
    assert view._undo_stack == undo and changes == []
    assert not view._dirty and not view._save_timer.isActive()


def test_preview_style_seam_uses_supplied_session_and_emits_without_saved_change(view):
    name, first, second = _group(view)
    alternate = deepcopy(view.session)
    other = next(layer for layer in alternate.layers if layer.facies != name)
    other.facies = name
    snapshot = deepcopy(alternate.to_dict())
    styles, saved = [], []
    view.display_styles_changed.connect(lambda: styles.append(True))
    view.changed.connect(lambda: saved.append(True))
    view.set_layer_visible(first, False)
    _toggle_facies(view, name)
    view.set_color_mode("facies")
    colors, hidden = view.display_layer_styles(alternate)
    assert {first, second, other.layer_id} <= hidden
    assert colors[other.layer_id] == slv._facies_color(name)
    _toggle_facies(view, name)
    assert not view.hidden_layers and view._hidden_layers == {first}
    view.set_color_mode("layer")
    assert view.hidden_layers == {first}
    assert styles and saved == [] and alternate.to_dict() == snapshot


def test_saved_manual_version_remains_identical_while_checkbox_changes_display(view, qtbot):
    name, _first, _second = _group(view)
    for polygon in (LEFT, RIGHT, MIDDLE):
        view.add_polygon(deepcopy(polygon))
    view.map_shown.setCurrentIndex(view.map_shown.findText("Manual map"))
    _drawn(view, qtbot)
    record = deepcopy(view.save_version("Drawn map"))
    assert record is not None
    displayed_record = deepcopy(next(row for row in view.versions if row["name"] == record["name"]))
    classes, key, files = view._manual.classes.copy(), view._manual_key, _files(view)
    _toggle_facies(view, name)
    view.set_color_mode("facies")
    view.facies_all_on_button.click()
    assert view._manual_thread is None and view._manual_key == key
    np.testing.assert_array_equal(view._manual.classes, classes)
    assert _files(view) == files
    assert next(row for row in view.versions if row["name"] == record["name"]) == displayed_record


def test_empty_and_renamed_facies_have_safe_controls_and_new_session_resets_visibility(view, qtbot):
    name, first, _second = _group(view)
    view.set_layer_visible(first, False)
    _toggle_facies(view, name)
    view.rename_facies(name, "Fine <sand> & water")
    qtbot.waitUntil(lambda: any(view.facies_table.item(row, 0).text() == "Fine <sand> & water"
                               for row in range(view.facies_table.rowCount())))
    assert _facies_item(view, "Fine <sand> & water").text() == "Fine <sand> & water"
    assert _facies_item(view, "Fine <sand> & water").checkState() == Qt.CheckState.Unchecked
    empty = view.add_facies("Unused")
    assert _facies_item(view, empty).checkState() == Qt.CheckState.Unchecked
    assert "No layers" in _facies_item(view, empty).toolTip()
    view.session = lp.PaintingSession.from_dict(view.session.to_dict())
    view._refresh_layers()
    view._refresh_facies_table()
    assert not view.hidden_layers and view._highlight_facies is None
    assert _facies_item(view, "Fine <sand> & water").checkState() == Qt.CheckState.Checked


def _preview_packet(session, options, first, second):
    labels = np.empty((60, 80), np.uint8)
    labels[:, :40], labels[:, 40:] = first, second
    return RunPreview(options["target_run"], np.full((60, 80, 3), (120, 90, 60), np.uint8),
        labels, np.zeros(labels.shape, bool), options["target_run"] + ".png",
        session.run_id, options["source_name"])


@pytest.mark.parametrize("mode", ["layer", "facies"])
def test_other_run_controls_redraw_cached_map_without_reading_photo_or_restarting_worker(view, qtbot, monkeypatch, mode):
    name, first, second = _group(view)
    controller = _chooser(view, qtbot, monkeypatch)
    loaded = []
    def prepare(session, values, **options):
        loaded.append(options["target_run"])
        return _preview_packet(session, options, first, second)
    monkeypatch.setattr(preview_ui.preview, "prepare_preview", prepare)
    view.set_color_mode(mode)
    _select(controller)
    qtbot.waitUntil(lambda: controller.thread is None and controller.last_result is not None, timeout=5000)
    QTest.qWait(1)  # Settle the initial photo/sidebar layout before recording its zoom.
    assert not controller.progress_bar.isVisible()
    assert not hasattr(view, "facies_visibility")
    assert view.steps_scroll.isEnabled() and view.facies_table.isEnabled() and view.layer_table.isEnabled()
    assert all(getattr(view, f"{owner}_all_{suffix}_button").isEnabled()
               for owner in ("layer", "facies") for suffix in ("on", "off"))
    assert not view.add_layer_button.isEnabled() and not view.remove_layer_button.isEnabled()
    assert not view.add_facies_button.isEnabled() and not view.remove_facies_button.isEnabled()
    assert not view.step_frame.isEnabled() and not view.step_propose.isEnabled()
    assert not view.step_use_map.isEnabled()
    for table in (view.layer_table, view.facies_table):
        assert table.contextMenuPolicy() == Qt.ContextMenuPolicy.NoContextMenu
        assert not table.editTriggers()
        for row in range(table.rowCount()):
            for column in range(table.columnCount()):
                item = table.item(row, column)
                if item is not None:
                    assert not item.flags() & Qt.ItemFlag.ItemIsEditable
                widget = table.cellWidget(row, column)
                if widget is not None:
                    assert not widget.isEnabled()
    assert not view.layer_table.item(0, 5).flags() & Qt.ItemFlag.ItemIsUserCheckable
    result, cached = controller.last_result, controller._last_session
    before = deepcopy(view.session.to_dict()), deepcopy(cached.to_dict()), result.layers.copy(), result.photo.copy()
    arrays = id(result.layers), id(result.photo), id(result.uncovered)
    controller.canvas.wheelEvent(QWheelEvent(QPointF(30, 30), QPointF(30, 30),
        QPoint(), QPoint(0, 120), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False))
    note, transform = controller.note.text(), controller.canvas.transform()
    captured = []
    original = controller.canvas.set_proposal
    def overlay(value):
        captured.append(value.copy())
        original(value)
    monkeypatch.setattr(controller.canvas, "set_proposal", overlay)
    def forbidden(*args, **kwargs):
        pytest.fail("A visibility click attempted to load or place the preview again")
    monkeypatch.setattr(controller.canvas, "set_photo", forbidden)
    monkeypatch.setattr(controller, "request", forbidden)
    monkeypatch.setattr(preview_ui.preview, "prepare_preview", forbidden)
    view.set_layer_visible(first, False)
    _toggle_facies(view, name)
    assert np.all(captured[-1][..., 3] == 0)
    _toggle_facies(view, name)
    assert captured[-1][20, 10, 3] == captured[-1][20, 60, 3] == round(255 * view.opacity.value() / 100)
    np.testing.assert_array_equal(captured[-1][20, 10, :3], captured[-1][20, 60, :3])
    assert view._hidden_layers == {first} and view._color_mode == "facies"
    view.facies_all_off_button.click()
    assert np.all(captured[-1][..., 3] == 0)
    _button_states(view, False, True)
    view.layer_all_on_button.click()
    assert not view.hidden_layers
    _button_states(view, True, False)
    view.layer_all_off_button.click()
    assert np.all(captured[-1][..., 3] == 0)
    _button_states(view, False, True)
    view.facies_all_on_button.click()
    assert not view.hidden_layers
    _button_states(view, True, False)
    assert view._hidden_layers == {row.layer_id for row in view.session.layers}
    view.layer_all_on_button.click()
    view.set_layer_visible(first, False)
    _button_states(view, False, False)
    def no_colour_dialog(*args, **kwargs):
        pytest.fail("Read-only table double-click opened a layer colour editor")
    monkeypatch.setattr(slv.QColorDialog, "getColor", no_colour_dialog)
    view._layer_double_clicked(0, 0)
    color = view._layer_colors()[second]
    np.testing.assert_array_equal(captured[-1][20, 60, :3], [color.red(), color.green(), color.blue()])
    assert view.hidden_layers == {first}
    assert loaded == ["target_a"] and controller.last_result is result and controller._last_session is cached
    assert arrays == (id(result.layers), id(result.photo), id(result.uncovered))
    assert view.session.to_dict() == before[0] and cached.to_dict() == before[1]
    np.testing.assert_array_equal(result.layers, before[2])
    np.testing.assert_array_equal(result.photo, before[3])
    assert controller.note.text() == note and controller.canvas.transform() == transform
    controller.runs.setCurrentIndex(controller.runs.findData(view.session.run_id))
    _button_states(view, False, False)
    assert controller.is_reference() and view.add_layer_button.isEnabled()
    assert view.add_facies_button.isEnabled() and view.step_propose.isEnabled()
    assert view.layer_table.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable
    assert view.facies_table.item(0, 0).flags() & Qt.ItemFlag.ItemIsEditable
    assert view.layer_table.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
    assert view.facies_table.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu


def test_cached_colour_redraw_keeps_partial_progress_and_later_measurement_error(view, qtbot, monkeypatch):
    name, first, second = _group(view)
    controller = _chooser(view, qtbot, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    calls = []
    def prepare(session, values, **options):
        calls.append(1)
        options["on_preview"](_preview_packet(session, options, first, second))
        options["on_progress"]("Measuring boundaries on the target photograph")
        entered.set()
        assert release.wait(10)
        raise OSError("boundary measurements unavailable")
    monkeypatch.setattr(preview_ui.preview, "prepare_preview", prepare)
    try:
        _select(controller)
        qtbot.waitUntil(entered.is_set, timeout=3000)
        qtbot.waitUntil(lambda: controller.last_result is not None and "Measuring boundaries" in controller.note.text(), timeout=3000)
        stage = controller._stage
        _toggle_facies(view, name)
        assert controller._stage == stage and controller._loading_status
        assert controller.progress_bar.isVisible() and controller.thread is not None
        assert not controller.status_overlay.isVisible()
        release.set()
        qtbot.waitUntil(lambda: controller.thread is None, timeout=5000)
        error = controller.note.text()
        assert "boundary measurements unavailable" in error
        _toggle_facies(view, name)
        assert controller.note.text() == error and not controller._loading_status
        assert not controller.progress_bar.isVisible()
        assert calls == [1] and controller.last_result is not None
    finally:
        release.set()
        controller.wait()


@pytest.mark.parametrize("partial", [False, True])
def test_reference_return_hides_shared_progress_while_same_preview_worker_finishes(view, qtbot, monkeypatch, partial):
    _name, first, second = _group(view)
    controller = _chooser(view, qtbot, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    calls = []
    def prepare(session, values, **options):
        calls.append(options["target_run"])
        result = _preview_packet(session, options, first, second)
        if partial:
            options["on_preview"](result)
        options["on_progress"]("Reading the selected photograph" if not partial else "Checking the shown map")
        entered.set()
        assert release.wait(10)
        return result
    monkeypatch.setattr(preview_ui.preview, "prepare_preview", prepare)
    try:
        _select(controller)
        qtbot.waitUntil(entered.is_set, timeout=3000)
        qtbot.waitUntil(lambda: controller.progress_bar.isVisible(), timeout=3000)
        if partial:
            qtbot.waitUntil(lambda: controller.last_result is not None, timeout=3000)
            assert not controller.status_overlay.isVisible()
        worker = controller.worker
        controller.runs.setCurrentIndex(controller.runs.findData(view.session.run_id))
        assert controller.is_reference() and not controller.progress_bar.isVisible()
        assert "Reference run" in controller.note.text()
        assert controller.worker is worker and controller.thread is not None
        release.set()
        qtbot.waitUntil(lambda: controller.thread is None, timeout=5000)
        assert not controller.progress_bar.isVisible() and not controller._elapsed_timer.isActive()
        assert calls == ["target_a"]
    finally:
        release.set()
        controller.wait()
        assert not controller.progress_bar.isVisible()


def test_visibility_signal_refuses_stale_selection_and_deleted_target_canvas(view, qtbot, monkeypatch):
    name, first, second = _group(view)
    controller = _chooser(view, qtbot, monkeypatch)
    monkeypatch.setattr(preview_ui.preview, "prepare_preview", lambda session, values, **options:
        _preview_packet(session, options, first, second))
    _select(controller)
    qtbot.waitUntil(lambda: controller.thread is None and controller.last_result is not None, timeout=5000)
    rendered = []
    monkeypatch.setattr(controller.canvas, "set_proposal", lambda values: rendered.append(values))
    controller.runs.blockSignals(True)
    controller.runs.setCurrentIndex(controller.runs.findData("target_b"))
    controller.runs.blockSignals(False)
    _toggle_facies(view, name)
    assert rendered == []
    controller.canvas.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    _toggle_facies(view, name)
    controller.redraw_display()
    assert controller._disposed and rendered == []
