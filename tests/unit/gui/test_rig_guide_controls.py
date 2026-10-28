"""Support lines stay visual, photograph-bound and explicitly saved."""
from copy import deepcopy

import numpy as np
from PySide6.QtCore import Qt

from darsia.fluidflower import rig_static_reference_ui as ui

from .test_rig_static_reference_ui import context, define, packet


def draw(context, first=(0., 0.), second=(40., 30.)):
    context.ui.draw_guide_button.click()
    for x, y in (first, second):
        context.page.canvas.raw_point_selected.emit(y, x)
    return context.ui.document["guides"][-1]


def guide_items(panel):
    return [item for item in panel._items if hasattr(item, "pen")
            and item.pen().color().name() == "#ffd166"
            and item.pen().style() == Qt.PenStyle.DashLine]


def endpoints(item):
    path = item.path()
    return np.array([[path.elementAt(index).x, path.elementAt(index).y]
                     for index in (0, path.elementCount()-1)])


def test_two_native_clicks_add_dashed_overlay_without_changing_image_or_references(context, monkeypatch):
    define(context)
    panel, canvas = context.ui, context.page.canvas
    result = panel.document["runs"]["run1"]
    result.update(model={"status": "valid"}, plane_models={"rig": {"status": "valid"}})
    before = deepcopy({key: panel.document[key] for key in ("regions", "runs")})
    image, support = canvas.preview["image_rgb"].copy(), canvas.preview["valid_support"].copy()
    generation = panel._generation
    monkeypatch.setattr(context.rig, "_store_frame", lambda: (_ for _ in ()).throw(
        AssertionError("A guide must not change the rig frame")))
    panel.draw_guide_button.click()
    canvas.raw_point_selected.emit(0., 0.)
    assert panel._mode == "guide" and canvas.place_mode
    pending = [item for item in panel._items if hasattr(item, "pen")
               and item.pen().color().name() == "#ffd166"]
    assert len(pending) == 1 and pending[0].path().elementCount() == 5
    canvas.raw_point_selected.emit(30., 40.)
    guide = panel.document["guides"][0]
    assert guide["points_u_xy"] == [[0., 0.], [40., 30.]]
    assert guide["photo_identity"] == "run1" and guide["extend"]
    assert not panel._mode and not canvas.place_mode and not panel.draw_guide_button.isChecked()
    items = guide_items(panel)
    assert len(items) == 1 and items[0].pen().isCosmetic()
    np.testing.assert_allclose(endpoints(items[0]), [[0., 5.], [119., 94.25]])
    original = endpoints(items[0])
    canvas.scale(3., 3.)
    canvas.translate(80., -10.)
    panel._draw_overlay()
    np.testing.assert_allclose(endpoints(guide_items(panel)[0]), original)
    assert {key: panel.document[key] for key in ("regions", "runs")} == before
    assert panel._generation == generation and panel.dirty and not context.backend.saved
    np.testing.assert_array_equal(canvas.preview["image_rgb"], image)
    np.testing.assert_array_equal(canvas.preview["valid_support"], support)


def test_guide_reference_and_frame_modes_are_exclusive_and_scope_switch_cancels_pending(context):
    define(context)
    panel, canvas = context.ui, context.page.canvas
    panel.table.selectRow(0)
    panel.define_button.click()
    canvas.raw_point_selected.emit(0., 0.)
    panel.draw_guide_button.click()
    assert panel._mode == "guide" and not panel.define_button.isChecked() and not panel._clicks
    canvas.raw_point_selected.emit(3., 5.)
    panel.correct_button.click()
    assert panel._mode == "correct" and not panel.draw_guide_button.isChecked() and not panel._clicks
    panel.draw_guide_button.click()
    canvas.raw_point_selected.emit(3., 5.)
    context.rig._set_mode("corners")
    assert not panel._mode and not panel.draw_guide_button.isChecked() and not panel._clicks
    panel.draw_guide_button.click()
    assert not context.rig._mode and panel._mode == "guide"
    canvas.raw_point_selected.emit(3., 5.)
    context.page.ready = False
    panel.context_changed()
    assert not panel._mode and not canvas.place_mode and not panel._clicks
    assert not panel.draw_guide_button.isEnabled()
    canvas.raw_point_selected.emit(30., 40.)
    assert not panel.document.get("guides") and not guide_items(panel)


def test_guides_follow_photo_identity_and_current_photo_undo_clear_and_explicit_save(context, qtbot, monkeypatch):
    panel, page, backend = context.ui, context.page, context.backend
    original_binding = backend.photo_binding

    def binding(value):
        result = original_binding(value)
        result["identity"] = repr(ui._identity(value))
        return result

    monkeypatch.setattr(backend, "photo_binding", binding)
    first = draw(context)
    for part, key, value in (("source_raw", "path", "different.CR2"),
                             ("source_raw", "sha256", "new-content"),
                             ("profile", "sha256", "new-profile")):
        page.canvas.preview = packet("run1")
        page.canvas.preview[part][key] = value
        panel.context_changed()
        assert not guide_items(panel)
    page.current_run, page.canvas.preview = "run2", packet("run2")
    panel.context_changed()
    assert not guide_items(panel)
    second = draw(context)
    assert len(guide_items(panel)) == 1
    panel.undo_guide_button.click()
    assert panel.document["guides"] == [first] and not guide_items(panel)
    draw(context)
    panel.clear_guides_button.click()
    assert panel.document["guides"] == [first]
    page.current_run, page.canvas.preview = "run1", packet("run1")
    panel.context_changed()
    assert len(guide_items(panel)) == 1
    panel.guides_visible.setChecked(False)
    assert not guide_items(panel) and panel.document["guides_visible"] is False
    assert not backend.saved and panel.save()
    qtbot.waitUntil(lambda: not panel.busy(), timeout=5000)
    saved = backend.saved[0][0]
    assert saved["guides"] == [first] and saved["guides_visible"] is False
    assert not panel.dirty
    monkeypatch.setattr(backend, "load_references", lambda root: deepcopy(saved))
    panel.document = backend.empty_document()
    assert panel.load()
    qtbot.waitUntil(lambda: not panel.busy(), timeout=5000)
    assert panel.document["guides"] == [first] and not panel.guides_visible.isChecked()
    panel.guides_visible.setChecked(True)
    assert len(guide_items(panel)) == 1 and panel.dirty
    assert second["run_id"] == "run2"


def test_horizontal_vertical_and_extent_controls_use_first_point_and_photo_edges(context):
    panel = context.ui
    panel.guide_mode.setCurrentIndex(panel.guide_mode.findData("horizontal"))
    panel.guide_extend.setChecked(False)
    horizontal = draw(context, (10., 10.), (50., 40.))
    assert horizontal["points_u_xy"] == [[10., 10.], [50., 10.]]
    assert horizontal["mode"] == "horizontal" and horizontal["extend"] is False
    np.testing.assert_allclose(endpoints(guide_items(panel)[-1]), [[30., 30.], [70., 30.]])
    panel.guide_mode.setCurrentIndex(panel.guide_mode.findData("vertical"))
    panel.guide_extend.setChecked(True)
    vertical = draw(context, (25., 20.), (60., 50.))
    assert vertical["points_u_xy"] == [[25., 20.], [25., 50.]]
    assert vertical["mode"] == "vertical" and vertical["extend"] is True
    np.testing.assert_allclose(endpoints(guide_items(panel)[-1]), [[45., 0.], [45., 119.]])
