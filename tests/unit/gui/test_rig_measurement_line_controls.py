"""Measured distances have two endpoints and remain separate from display guides."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt

from darsia.fluidflower import rig_static_reference_ui as ui
from darsia.fluidflower import rig_static_references as refs

from .test_rig_static_reference_ui import context
from ..test_rig_static_references import packet, scene


@pytest.fixture
def measurements(context, monkeypatch, tmp_path):
    source, _, _ = scene(tmp_path)
    target = packet(tmp_path, source["image_rgb"].copy(), run="target")
    for value in (source, target):
        value["mapping"] = SimpleNamespace(camera_u_to_display_xy=lambda points: np.asarray(points)+[100., 60.])
    monkeypatch.setattr(ui, "_backend", lambda: refs)
    panel, page = context.ui, context.page
    # Exercise retained legacy annotation APIs explicitly; this editor is
    # hidden in the current four-reference-point workflow.
    panel.measurement_box.show()
    panel.document = refs.empty_document()
    page.runs = [SimpleNamespace(run_id=name, baseline_image=None) for name in ("source", "target")]
    page.current_run, page.canvas.preview = "source", source
    page.canvas.scene().setSceneRect(0., 0., 800., 600.)
    page.read_static_reference_packet = lambda reference: deepcopy(source if reference.run_id == "source" else target)
    panel.context_changed()
    return SimpleNamespace(panel=panel, page=page, rig=context.rig, source=source, target=target, root=tmp_path)


def draw(measurements, *, name="Frame distance", length=125.5, points=((0., 40.), (575., 60.))):
    panel = measurements.panel
    panel.measurement_name.setText(name)
    panel.measurement_length_mm.setValue(length)
    panel.add_measurement_button.click()
    for x, y in points:
        measurements.page.canvas.raw_point_selected.emit(y, x)
    assert panel.document.get("measurement_lines"), panel.note.text()
    return panel.document["measurement_lines"][-1]


def line_items(panel):
    return [item for item in panel._items if item.data(0) == "measurement-line"]


def corners(rig):
    return rig.reference_photo_corners()


def test_two_clicks_create_finite_solid_measured_line_and_no_point_geometry(measurements):
    panel, page = measurements.panel, measurements.page
    before = corners(measurements.rig)
    panel.measurement_name.setText("Frame height")
    panel.measurement_length_mm.setValue(800.)
    panel.add_measurement_button.click()
    page.canvas.raw_point_selected.emit(40., 0.)
    assert panel._mode == "measurement" and page.canvas.place_mode
    assert not panel.document.get("measurement_lines")
    assert "second measurement endpoint" in panel.note.text()
    page.canvas.raw_point_selected.emit(60., 575.)
    line = panel.document["measurement_lines"][0]
    assert line["name"] == "Frame height" and line["length_mm"] == 800.
    assert line["points_u_xy"] == [[0., 40.], [575., 60.]]
    assert not panel._mode and not page.canvas.place_mode
    assert panel.measurement_table.item(0, 1).text() == "800.000"
    assert len(line_items(panel)) == 1
    item = line_items(panel)[0]
    assert item.pen().style() == Qt.PenStyle.SolidLine and item.pen().isCosmetic()
    path = item.path()
    assert path.elementCount() == 2
    np.testing.assert_allclose([[path.elementAt(i).x, path.elementAt(i).y] for i in (0, 1)],
                               [[20., 60.], [595., 80.]])
    assert any(item.data(0) == "measurement-endpoints" for item in panel._items)
    assert any(hasattr(item, "text") and "800 mm" in item.text() for item in panel._items)
    assert panel.document["regions"] == [] and panel.document["runs"]["source"]["observations"] == {}
    assert not panel.document["runs"]["source"].get("model") and not panel.use_button.isEnabled()
    assert corners(measurements.rig) == before
    assert panel.point_area_box.isHidden() and not panel.point_area_toggle.isChecked()
    assert panel.find_button.isEnabled() and panel.find_all_button.isEnabled()


def test_edit_name_length_plane_preserves_endpoints_and_remove_cleans_observations(measurements):
    panel = measurements.panel
    line = draw(measurements, length=0.)
    assert "needs measurement" in panel.measurement_table.item(0, 3).text()
    source_observation = deepcopy(panel.document["runs"]["source"]["measurement_lines"][line["id"]])
    identity = line["tracking_identity"]
    for column, value in ((0, "Measured diagonal"), (1, "231.25"), (2, "bolt-heads")):
        panel.measurement_table.item(0, column).setText(value)
    assert (line["name"], line["length_mm"], line["plane_id"]) == ("Measured diagonal", 231.25, "bolt-heads")
    assert line["tracking_identity"] == identity
    assert panel.document["runs"]["source"]["measurement_lines"][line["id"]] == source_observation
    assert panel._measurement_observations()[line["id"]]["length_mm"] == 231.25
    assert "needs measurement" not in panel.measurement_table.item(0, 3).text()
    panel.measurement_table.item(0, 1).setText("-1")
    assert line["length_mm"] == 231.25 and "nonnegative" in panel.note.text()
    panel.measurement_table.selectRow(0)
    panel.remove_measurement_button.click()
    assert panel.document["measurement_lines"] == [] and panel.measurement_table.rowCount() == 0
    assert panel.document["runs"]["source"]["measurement_lines"] == {}
    assert not line_items(panel)


def test_support_guides_and_measurement_lines_are_separate_and_survive_explicit_save(measurements, qtbot):
    panel, page = measurements.panel, measurements.page
    line = draw(measurements)
    for offset in (0., 5., 10.):
        panel.draw_guide_button.click()
        for x, y in ((0., 40.+offset), (575., 60.+offset)):
            page.canvas.raw_point_selected.emit(y, x)
    first, middle, last = deepcopy(panel.document["guides"])
    panel.guide_selection.setCurrentIndex(panel.guide_selection.findData(middle["id"]))
    panel.remove_guide_button.click()
    assert panel.document["guides"] == [first, last]
    assert panel.guide_selection.count() == 2
    assert len(panel.document["measurement_lines"]) == 1
    assert any(hasattr(item, "pen") and item.pen().style() == Qt.PenStyle.DashLine
               and item.pen().color().name() == "#ffd166" for item in panel._items)
    panel.guides_visible.setChecked(False)
    assert len(line_items(panel)) == 1
    assert panel.save()
    qtbot.waitUntil(lambda: not panel.busy(), timeout=5000)
    saved = refs.load_references(measurements.root)
    assert saved["measurement_lines"][0]["id"] == line["id"]
    assert saved["guides"] == [first, last] and saved["guides_visible"] is False
    assert saved["regions"] == []
    panel.measurement_table.selectRow(0)
    panel.remove_measurement_button.click()
    assert panel.document["guides"] == [first, last]
    assert refs.load_references(measurements.root)["measurement_lines"][0]["id"] == line["id"]
    panel.clear_guides_button.click()
    assert panel.document["guides"] == [] and panel.guide_selection.count() == 0


def test_find_line_only_target_run_and_redraw_preserve_manual_correction(measurements, qtbot):
    panel, page = measurements.panel, measurements.page
    line = draw(measurements)
    source_observation = deepcopy(panel.document["runs"]["source"]["measurement_lines"][line["id"]])
    page.current_run, page.canvas.preview = "target", measurements.target
    panel.context_changed()
    assert not line_items(panel)
    assert panel.find_regions(False)
    qtbot.waitUntil(lambda: not panel.busy(), timeout=10000)
    assert panel.measurement_table.item(0, 3).text() == "recognized", panel.note.text()
    assert float(panel.measurement_table.item(0, 4).text()) > .8
    assert len(line_items(panel)) == 1
    panel.measurement_table.selectRow(0)
    panel.redraw_measurement_button.click()
    for x, y in ((1., 41.), (576., 61.)):
        page.canvas.raw_point_selected.emit(y, x)
    corrected = deepcopy(panel.document["runs"]["target"]["measurement_lines"][line["id"]])
    assert corrected["points_u_xy"] == [[1., 41.], [576., 61.]] and corrected["status"] == "manual"
    assert panel.measurement_table.item(0, 3).text() == "manual"
    assert panel.document["runs"]["source"]["measurement_lines"][line["id"]] == source_observation
    assert panel.find_regions(False)
    qtbot.waitUntil(lambda: not panel.busy(), timeout=10000)
    assert panel.document["runs"]["target"]["measurement_lines"][line["id"]] == corrected
    page.current_run, page.canvas.preview = "source", measurements.source
    panel.context_changed()
    assert panel._measurement_observations()[line["id"]]["points_u_xy"] == [[0., 40.], [575., 60.]]
    page.canvas.preview = deepcopy(measurements.source)
    page.canvas.preview["source_raw"]["sha256"] = "0"*64
    panel.context_changed()
    assert not line_items(panel)


def test_measurement_guide_point_modes_are_exclusive_and_run_switch_cancels_second_click(measurements):
    panel, page = measurements.panel, measurements.page
    panel.add_measurement_button.click()
    page.canvas.raw_point_selected.emit(40., 0.)
    panel.draw_guide_button.click()
    assert panel._mode == "guide" and not panel.add_measurement_button.isChecked() and not panel._clicks
    panel.add_measurement_button.click()
    assert panel._mode == "measurement" and not panel.draw_guide_button.isChecked()
    panel.define_button.click()
    assert panel._mode == "define" and not panel.add_measurement_button.isChecked()
    panel.add_measurement_button.click()
    page.canvas.raw_point_selected.emit(40., 0.)
    page.current_run, page.canvas.preview = "target", measurements.target
    panel.context_changed()
    assert not panel._mode and not panel._clicks and not page.canvas.place_mode
    page.canvas.raw_point_selected.emit(60., 575.)
    assert not panel.document.get("measurement_lines") and not panel.document.get("guides")
    assert panel.document["regions"] == []
