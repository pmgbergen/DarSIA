"""Facies colours and focus change only the displayed painting and settings."""
from copy import deepcopy

import numpy as np
import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QCheckBox, QColorDialog, QMenu

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import series_layer_view as slv
from .test_painting_photo_saturation import settings, make_view, view, _shown, _grey  # noqa: F401
from .test_painting_corrections import _proposal, _show
from .test_painting_manual_map import _old_proposal_overlay, _drawn, LEFT, RIGHT, MIDDLE


def _group(view):
    """Two differently coloured layers share one facies in the saved fixture."""
    name = view.session.layers[1].facies
    view.set_layer_facies(view.session.layers[2].layer_id, name)
    view.save_session()
    return name, view.session.layers[1].layer_id, view.session.layers[2].layer_id


def _actions(menu):
    return {action.text(): action for action in menu.actions()}


def _capture(view, monkeypatch):
    overlays = []
    original = view.canvas.set_proposal
    def draw(rgba):
        overlays.append(None if rgba is None else rgba.copy())
        original(rgba)
    monkeypatch.setattr(view.canvas, "set_proposal", draw)
    return overlays


def _files(view):
    folder = lp.painting_folder(view.project)
    return {str(path.relative_to(folder)): path.read_bytes()
            for path in folder.rglob("*") if path.is_file()} if folder.exists() else {}


def test_default_layer_colours_and_full_proposal_pixels_stay_exact(view, monkeypatch):
    assert view.color_mode.currentText() == "By layer"
    assert view._highlight_facies is None
    assert not view.facies_legend.isVisibleTo(view)
    assert view._layer_colors() == {row.layer_id: QColor(row.color) for row in view.session.layers}
    shown = _capture(view, monkeypatch)
    proposal = _proposal()
    proposal.confidence[20:30, 10:70] = 10
    _show(view, proposal)
    view.show_uncertain.setChecked(True)
    view.opacity.setValue(70)
    for hidden in ((), (view.session.layers[2].layer_id,)):
        view._set_hidden_layers(hidden)
        view._render_proposal()
        np.testing.assert_array_equal(shown[-1], _old_proposal_overlay(view))


def test_facies_colours_share_marks_cursor_and_map_while_layer_swatches_keep_own_colours(view):
    facies, first, second = _group(view)
    assert QColor(view.session.layer(first).color) != QColor(view.session.layer(second).color)
    view.session.polygons = [deepcopy(LEFT), deepcopy(RIGHT)]
    view.session.strokes = [lp.Stroke(first, 2., [(10., 10.)]), lp.Stroke(second, 2., [(10., 60.)])]
    snapshot = deepcopy(view.session.to_dict())
    original = {row.layer_id: QColor(row.color) for row in view.session.layers}
    view.facies_all_on_button.click()
    assert view._color_mode == "facies"
    color = slv._facies_color(facies)
    assert view._layer_colors()[first] == view._layer_colors()[second] == color
    assert view.canvas._polygon_colors[first] == view.canvas._polygon_colors[second] == color
    assert view.canvas._stroke_colors[first] == view.canvas._stroke_colors[second] == color
    view.select_layer_row(2)
    assert view.canvas.layer_color == color
    for layer_id in (first, second):
        row = next(number for number, layer in enumerate(view.session.layers) if layer.layer_id == layer_id)
        assert view.layer_table.item(row, 0).background().color() == original[layer_id]
    classes = np.tile(np.repeat(np.array([first, second], np.uint8), 8), (12, 1))
    rgba, shown = view._layer_overlay(classes)
    expected = (color.red(), color.green(), color.blue(), int(round(2.55 * view.opacity.value())))
    assert shown.all() and np.all(rgba == expected)
    assert facies in view.facies_legend.text()
    assert view.facies_legend.isVisibleTo(view)
    row = next(number for number in range(view.facies_table.rowCount())
               if view.facies_table.item(number, 0).data(Qt.ItemDataRole.UserRole) == facies)
    icon = view.facies_table.item(row, 0).icon().pixmap(14, 14).toImage()
    # Pixmaps store RGB pixels; compare their displayed 8-bit colour.
    assert icon.pixelColor(7, 7).name() == color.name()
    # A different layer order never changes a facies's colour.
    view.session.layers.reverse()
    assert view._layer_colors()[first] == view._layer_colors()[second] == color
    view.session.layers.reverse()
    view.layer_all_on_button.click()
    assert view._color_mode == "layer" and view._layer_colors() == original
    rgba, shown = view._layer_overlay(classes)
    assert shown.all()
    assert tuple(rgba[5, 3, :3]) == original[first].getRgb()[:3]
    assert tuple(rgba[5, 12, :3]) == original[second].getRgb()[:3]
    assert not np.array_equal(rgba[5, 3, :3], rgba[5, 12, :3])
    view.highlight_facies(facies)
    view.color_mode.activated.emit(view.color_mode.currentIndex())
    assert view._highlight_facies is None and view._layer_colors() == original
    assert view.session.to_dict() == snapshot


@pytest.mark.parametrize("highlight", [False, True])
def test_facies_display_keeps_each_layer_colour_in_picker_and_membership_menu(view, monkeypatch, highlight):
    facies, _first, _second = _group(view)
    view.set_color_mode("facies")
    if highlight:
        view.highlight_facies(facies)
    original = {row.layer_id: QColor(row.color) for row in view.session.layers}
    picked = []
    monkeypatch.setattr(
        QColorDialog, "getColor",
        lambda initial, *_args: (picked.append(initial), QColor())[1],
    )
    for row, layer in enumerate(view.session.layers):
        assert view.layer_table.item(row, 0).background().color() == original[layer.layer_id]
        view._layer_double_clicked(row, 0)
        assert picked[-1] == original[layer.layer_id]
    menu = QMenu(view)
    view._fill_layer_menu(menu, facies)
    for layer, action in zip(view.session.layers, menu.actions()):
        box = action.defaultWidget().findChild(QCheckBox)
        icon = box.icon().pixmap(14, 14).toImage()
        assert icon.pixelColor(7, 7).name() == original[layer.layer_id].name()
    view._layer_menu_closed()


@pytest.mark.parametrize("mode", ["layer", "facies"])
def test_highlight_is_one_strong_colour_in_both_modes_and_keeps_hidden_choices(view, mode):
    facies, first, second = _group(view)
    view.set_layer_visible(first, False)
    view.set_color_mode(mode)
    hidden = view.hidden_layers
    baseline = view._layer_colors()
    view.highlight_facies(facies)
    assert view.hidden_layers == hidden == view.canvas.hidden_layers
    colors = view._layer_colors()
    assert colors[first] == colors[second] == slv._facies_color(facies)
    assert colors[first].saturation() >= 180
    for layer in view.session.layers:
        if layer.facies != facies:
            assert colors[layer.layer_id] == QColor("#999999")
    for row, layer in enumerate(view.session.layers):
        assert view.layer_table.item(row, 0).background().color() == QColor(layer.color)
    assert view.clear_highlight_button.isVisibleTo(view)
    view.clear_highlight_button.click()
    assert view._highlight_facies is None and view.hidden_layers == hidden
    assert view._layer_colors() == baseline
    assert not view.clear_highlight_button.isVisibleTo(view)


def test_both_context_menus_focus_the_entire_clicked_facies(view):
    facies, first, second = _group(view)
    layer_table = view.layer_table
    # Right-click the second member, not the first layer of this facies.
    row = next(number for number, layer in enumerate(view.session.layers) if layer.layer_id == second)
    position = layer_table.visualItemRect(layer_table.item(row, 1)).center()
    layer_table.customContextMenuRequested.emit(position)
    actions = _actions(view._layer_table_menu_widget)
    actions["Show only this facies"].trigger()
    assert view.hidden_layers == {layer.layer_id for layer in view.session.layers if layer.layer_id not in {first, second}}
    assert view.canvas.hidden_layers == view.hidden_layers
    view.show_all_layers()
    actions["Highlight this facies"].trigger()
    assert view._highlight_facies == facies
    # The facies table offers the same actions on its own row.
    table = view.facies_table
    assert table.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
    row = next(number for number in range(table.rowCount())
               if table.item(number, 0).data(Qt.ItemDataRole.UserRole) == facies)
    table.customContextMenuRequested.emit(table.visualItemRect(table.item(row, 0)).center())
    actions = _actions(view._facies_table_menu_widget)
    assert actions["Clear facies highlight"].isEnabled()
    actions["Clear facies highlight"].trigger()
    assert view._highlight_facies is None
    actions["Show only this facies"].trigger()
    assert view.hidden_layers == {layer.layer_id for layer in view.session.layers if layer.facies != facies}
    actions["Highlight this facies"].trigger()
    assert view._highlight_facies == facies


def test_empty_facies_context_does_not_offer_a_fabricated_target(view):
    name = view.add_facies("Empty category")
    view.save_session()
    menu = QMenu(view)
    view._add_facies_display_actions(menu, name)
    assert not _actions(menu)["Show only this facies"].isEnabled()
    assert not _actions(menu)["Highlight this facies"].isEnabled()
    view.highlight_facies(name)
    view.show_only_facies(name)
    assert view._highlight_facies is None and not view.hidden_layers


def test_display_actions_change_no_session_proposal_export_or_undo_data(view, monkeypatch):
    facies, first, _second = _group(view)
    _show(view, _proposal())
    view.session.polygons = [deepcopy(LEFT)]
    view._mark_dirty()
    view.save_session()
    snapshot = deepcopy(view.session.to_dict())
    classes = view.proposal.classes.copy()
    export_key = view._export_state_key()
    files = _files(view)
    undo = list(view._undo_stack)
    changes = []
    view.changed.connect(lambda: changes.append(True))
    def forbidden(*args, **kwargs):
        pytest.fail("Display controls attempted a session or version write")
    monkeypatch.setattr(lp, "save_painting_session", forbidden)
    view.set_layer_visible(first, False)
    view.set_color_mode("facies")
    view.highlight_facies(facies)
    view.show_only_facies(facies)
    view.highlight_facies(None)
    view.show_proposal_boundaries.setChecked(False)
    view.show_proposal_boundaries.setChecked(True)
    view.set_color_mode("layer")
    view.show_all_layers()
    assert view.session.to_dict() == snapshot
    np.testing.assert_array_equal(view.proposal.classes, classes)
    assert view._export_state_key() == export_key and view._undo_stack == undo
    assert _files(view) == files and view._dirty is False and changes == []
    assert not view._save_timer.isActive()


def test_colour_preference_is_only_a_user_setting_and_combines_with_photo_saturation(view, make_view, settings):
    _group(view)
    view.set_color_mode("facies")
    view.saturation.setValue(0)
    view.apply_saturation()
    assert all(_grey(array) for array in _shown(view))
    stored = QSettings(settings, QSettings.Format.IniFormat)
    assert stored.value(slv._COLOR_MODE_KEY) == "facies"
    assert "color_mode" not in str(view.session.to_dict())
    again = make_view()
    assert again.color_mode.currentText() == "By facies" and again.saturation.value() == 0
    assert again._highlight_facies is None and not again.hidden_layers
    view.set_color_mode("layer")
    assert view.color_mode.currentText() == "By layer"


def test_manual_map_version_reuses_the_same_facies_colours_without_rebuilding_data(view, qtbot, monkeypatch):
    facies, first, second = _group(view)
    for polygon in (LEFT, RIGHT, MIDDLE):
        view.add_polygon(deepcopy(polygon))
    view.map_shown.setCurrentIndex(view.map_shown.findText("Manual map"))
    _drawn(view, qtbot)
    record = view.save_version("Drawn map")
    classes = view._manual.classes.copy()
    files = _files(view)
    digest = view._manual_key
    overlays = _capture(view, monkeypatch)
    view.set_color_mode("facies")
    view.highlight_facies(facies)
    assert view._manual_thread is None and view._manual_key == digest
    np.testing.assert_array_equal(view._manual.classes, classes)
    assert record["kind"] == "manual map" and _files(view) == files
    color = slv._facies_color(facies)
    assert view._layer_colors()[first] == view._layer_colors()[second] == color
    # The interior of each drawn layer has the same RGB colour in the actual
    # manual-map overlay; diagnostic frame-edge hatching stays independent.
    image = overlays[-1]
    np.testing.assert_array_equal(image[10, 10, :3], [color.red(), color.green(), color.blue()])
    np.testing.assert_array_equal(image[10, 35, :3], image[10, 10, :3])


def test_legend_escapes_facies_names_and_marks_hidden_categories(view):
    name, first, second = _group(view)
    view.rename_facies(name, "Fine <sand> & water")
    view.save_session()
    view.set_color_mode("facies")
    view.set_facies_visible("Fine <sand> & water", False)
    assert "Fine &lt;sand&gt; &amp; water (hidden)" in view.facies_legend.text()
    assert "Fine <sand>" not in view.facies_legend.text()
