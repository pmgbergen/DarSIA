"""Layer painting tab: the manual map, a version of the polygons exactly as drawn.

The map shown under the photograph is the proposal or the manual map. The
manual map is drawn at full resolution in a worker thread, saved as a version
without a proposal, approved only when the polygons cover the rig frame, and
placed and prepared for adoption as drawn. A layer marks where it and the
proposal disagree. A synthetic 80 x 60 photograph and a project in a temporary
folder; no real series is used.
"""

import json
import threading
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QMessageBox

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import layer_painting_manual as lpm
from darsia.fluidflower import series_layer_view as view_module

from .test_painting_corrections import COARSE, FINE, _proposal, _show, view  # noqa: F401
from .test_painting_map_placement import _files, _index

# Fine sand on the left and coarse sand on the right leave columns 50-59 open;
# a third, smaller polygon closes them and wins where it overlaps the others.
LEFT = lp.Polygon(FINE, [(0.0, 0.0), (0.0, 50.0), (60.0, 50.0), (60.0, 0.0)])
RIGHT = lp.Polygon(COARSE, [(0.0, 60.0), (0.0, 80.0), (60.0, 80.0), (60.0, 60.0)])
MIDDLE = lp.Polygon(COARSE, [(0.0, 45.0), (0.0, 62.0), (60.0, 62.0), (60.0, 45.0)])


def _drawn(view, qtbot, *, compared=False):
    """Wait until the manual map (and the proposal's map to compare) are those
    of the marks shown."""

    def done():
        if view._manual_thread is not None or view._manual_timer.isActive():
            return False
        if not view._manual_current():
            return False
        return not compared or view._compare_key == view._compare_inputs_key() != ""

    qtbot.waitUntil(done, timeout=30000)


def _rendered(view, monkeypatch):
    """The overlays drawn over the photograph, in order."""

    shown = []
    original = view.canvas.set_proposal

    def record(rgba):
        shown.append(None if rgba is None else rgba.copy())
        original(rgba)

    monkeypatch.setattr(view.canvas, "set_proposal", record)
    return shown


def _in_worker(monkeypatch, name):
    """Record for each call of a manual-map function whether it ran in the
    GUI thread."""

    calls = []
    original = getattr(lpm, name)

    def spy(*args, **kwargs):
        calls.append(threading.current_thread() is threading.main_thread())
        return original(*args, **kwargs)

    monkeypatch.setattr(lpm, name, spy)
    return calls


def test_any_pixel_keeps_thin_areas_on_a_coarser_grid():
    mask = np.zeros((100, 100), dtype=bool)
    mask[37, 81] = True
    small = view_module._any_pixel(mask, (10, 10))
    assert small.shape == (10, 10) and small.sum() == 1 and small[3, 8]
    assert np.array_equal(view_module._any_pixel(mask, (100, 100)), mask)


def test_a_manual_map_is_drawn_saved_and_approved_without_a_proposal(view, qtbot, monkeypatch):
    drawn_in_gui = _in_worker(monkeypatch, "manual_map")
    rendered = _rendered(view, monkeypatch)
    view.add_polygon(LEFT)
    view.add_polygon(RIGHT)
    # The proposal is shown first; without one there is nothing to save.
    assert view.map_shown.currentText() == "Proposal" and view.show_proposal.text() == "Proposal"
    assert view.save_version("A") is None
    assert "Propose the boundaries first" in view.versions_note.text()
    assert view._manual_thread is None and not drawn_in_gui
    assert not view.manual_status.isVisibleTo(view)
    view.map_shown.setCurrentIndex(view.map_shown.findText("Manual map"))
    assert view.show_proposal.text() == "Manual map" and view.manual_status.isVisibleTo(view)
    _drawn(view, qtbot)
    assert drawn_in_gui and not any(drawn_in_gui)
    manual = view._manual
    assert np.array_equal(
        manual.classes[manual.inside],
        lp.rasterize_polygons([LEFT, RIGHT], (60, 80), (1.0, 1.0))[manual.inside],
    )
    gap = manual.uncovered_pixels()
    assert gap > 0 and manual.uncovered()[30, 55]
    text = view.manual_status.text()
    assert f"{gap:,} px inside the rig frame" in text and "magenta" in text
    # The open columns are drawn in magenta; the polygons are outlines only.
    assert tuple(rendered[-1][30, 55]) == view_module._UNCOVERED_COLOR
    assert not view.canvas._polygon_fill
    # Corrections are drawn on the proposal only.
    assert not view.tool_buttons["correct"].isEnabled()
    assert view.default_version_name().startswith("Manual map ")
    record = view.save_version("")
    assert record["kind"] == "manual map" and record["name"].startswith("Manual map ")
    assert record["uncovered_pixels"] == gap
    assert "cover them, save a new version" in view.versions_note.text()
    assert not lp.proposal_path(view.project, "ac52").exists()
    assert view.versions_table.item(0, 1).text() == "manual map (gaps)"
    # It cannot be approved while pixels are uncovered.
    view.versions_table.selectRow(0)
    asked = []

    def yes(*args, **kwargs):
        asked.append(args[1])
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", yes)
    view.approve_version_button.click()
    assert asked == [] and "cannot be approved" in view.versions_note.text()
    assert view.approve_version(record) is None
    assert "could not be approved" in view.versions_note.text()
    assert lp.read_map_approvals(view.project, "ac52") == []
    # A polygon over the open columns; drawn again once the marks rest.
    view.add_polygon(MIDDLE)
    assert not view._manual_current()
    _drawn(view, qtbot)
    assert view._manual.uncovered_pixels() == 0
    assert view._manual.classes[30, 47] == COARSE and view._manual.classes[30, 40] == FINE
    assert "every pixel inside the rig frame has a layer" in view.manual_status.text()
    complete = view.save_version("Complete")
    assert complete["kind"] == "manual map" and complete["sha256"] != record["sha256"]
    assert complete["sha256"] == lpm.manual_map_digest(view.session, view._manual)
    assert view.versions_table.item(0, 1).text() == "manual map"
    view.versions_table.selectRow(0)
    view.approve_version_button.click()
    assert asked == ["Approve version"]
    approval = lp.map_approval(view.map_approvals, complete["sha256"])
    assert approval is not None and approval["kind"] == "manual map"
    assert view.versions_table.item(0, 5).text().startswith("Approved ")
    view.versions_table.selectRow(0)
    assert view.place_version_button.isEnabled() and view.prepare_adoption_button.isEnabled()
    # Back to the proposal: the corrections tools wait for one again.
    view.map_shown.setCurrentIndex(view.map_shown.findText("Proposal"))
    assert view.show_proposal.text() == "Proposal" and not view.manual_status.isVisibleTo(view)
    assert view.save_version("B") is None


def test_the_differences_from_the_proposal_are_marked_and_measured(view, qtbot, monkeypatch):
    compared_in_gui = _in_worker(monkeypatch, "painted_layers")
    rendered = _rendered(view, monkeypatch)
    _show(view, _proposal())
    for polygon in (LEFT, RIGHT, MIDDLE):
        view.add_polygon(polygon)
    view.show_differences.setChecked(True)
    assert view.manual_status.isVisibleTo(view)
    _drawn(view, qtbot, compared=True)
    assert compared_in_gui and not any(compared_in_gui)
    layers = lpm.painted_layers(
        view.session, view.proposal, frame=view.current_frame(), corrections=view.corrections
    )
    expected = lpm.manual_map_differences(view._manual, layers)
    count = int(expected.sum())
    # The proposal has coarse sand from column 40, the polygons from column 45.
    assert count > 0 and expected[30, 42] and not expected[30, 20]
    area = lpm.pixel_area_cm2(view.current_frame(), (80, 60))
    text = view.manual_status.text()
    assert f"disagree on {count * area:.1f} cm²" in text and f"({count:,} px, red)" in text
    # Marked in red over the proposal, and over the manual map.
    red = np.all(rendered[-1] == view_module._DIFFERENCE_COLOR, axis=-1)
    assert view.map_shown.currentText() == "Proposal" and np.array_equal(red, expected)
    view.map_shown.setCurrentIndex(view.map_shown.findText("Manual map"))
    red = np.all(rendered[-1] == view_module._DIFFERENCE_COLOR, axis=-1)
    assert np.array_equal(red, expected)
    view.show_differences.setChecked(False)
    assert not np.all(rendered[-1] == view_module._DIFFERENCE_COLOR, axis=-1).any()
    # A correction changes the proposal's map: compared again once it rests.
    view.show_differences.setChecked(True)
    view.map_shown.setCurrentIndex(view.map_shown.findText("Proposal"))
    view.select_layer_row(1)  # Fine sand
    assert view.add_correction(lp.Correction("lasso", FINE, [(0.0, 38.0), (0.0, 46.0), (60.0, 46.0), (60.0, 38.0)]))
    _drawn(view, qtbot, compared=True)
    layers = lpm.painted_layers(
        view.session, view.proposal, frame=view.current_frame(), corrections=view.corrections
    )
    fewer = int(lpm.manual_map_differences(view._manual, layers).sum())
    assert fewer < count and f"({fewer:,} px, red)" in view.manual_status.text()


def test_restoring_a_manual_map_brings_back_its_polygons_and_keeps_the_proposal(view, qtbot):
    _show(view, _proposal())
    view.proposal.save(lp.proposal_path(view.project, "ac52"))
    proposal_file = lp.proposal_path(view.project, "ac52").read_bytes()
    for polygon in (LEFT, RIGHT, MIDDLE):
        view.add_polygon(polygon)
    view.map_shown.setCurrentIndex(view.map_shown.findText("Manual map"))
    _drawn(view, qtbot)
    first = view.save_version("Three polygons")
    view.undo()  # The middle polygon goes.
    assert len(view.session.polygons) == 2
    _drawn(view, qtbot)
    assert view.restore_version(first)
    # The drawing shown was kept as a version first.
    names = [row["name"] for row in view.versions]
    assert names == ["Three polygons", "Before restoring Three polygons"]
    assert view.versions[1]["kind"] == "manual map" and view.versions[1]["uncovered_pixels"] > 0
    assert [[tuple(point) for point in row.points] for row in view.session.polygons] == [
        [tuple(point) for point in row.points] for row in (LEFT, RIGHT, MIDDLE)
    ]
    # The proposal stays; the manual map drawn again is the restored version.
    assert view.proposal is not None and view.proposal.digest == "first"
    assert lp.proposal_path(view.project, "ac52").read_bytes() == proposal_file
    _drawn(view, qtbot)
    assert lpm.manual_map_digest(view.session, view._manual) == first["sha256"]
    # The map shown equals the saved version: restoring it again keeps no copy.
    assert view.restore_version(first) and len(view.versions) == 2


def test_an_approved_manual_map_is_placed_as_drawn_but_unbound_adoption_is_refused(view, qtbot, tmp_path):
    for polygon in (LEFT, RIGHT, MIDDLE):
        view.add_polygon(polygon)
    view.map_shown.setCurrentIndex(view.map_shown.findText("Manual map"))
    _drawn(view, qtbot)
    record = view.save_version("Manual")
    assert view.approve_version(record) is not None
    view.versions_table.selectRow(0)
    assert view.place_version_button.isEnabled()
    index = _index(tmp_path / "geometry")
    painting_before = _files(lp.painting_folder(view.project))

    def photo(run, geometry):
        return np.full(tuple(geometry.shape_hw) + (3,), 90, dtype=np.uint8)

    assert view.place_version(record, index, runs=["ac52", "ac53"], photo_loader=photo)
    qtbot.waitUntil(lambda: view.last_placement is not None and not view.busy(), timeout=30000)
    result = view.last_placement
    assert result["map_version"]["kind"] == "manual map"
    assert result["map_version"]["sha256"] == record["sha256"]
    assert result["map_version"]["approval"]["sha256"] == record["sha256"]
    with np.load(f"{result['folder']}/ac52/layers.npz") as saved:
        assert np.array_equal(saved["layers"], view._manual.classes)
    view._placement_dialog.close()
    # Each run's label grid: its camera grid without the top 5 rows.
    for run in ("ac52", "ac53"):
        path = tmp_path / "geometry" / run / "reference_model.json"
        model = json.loads(path.read_text(encoding="utf-8"))
        model["baseline_native_grid"] = {
            "shape_hw": [55, 80],
            "H_baseline_camera_u_to_rig_corrected_native": [[1, 0, 0], [0, 1, -5], [0, 0, 1]],
        }
        path.write_text(json.dumps(model), encoding="utf-8")

    def current(run):
        return {"metadata": {"run": run}, "shape_hw": [55, 80], "source": {"path": f"{run}.npz"}}

    assert view.prepare_adoption_of(record, index, runs=["ac52"], current_labels=current)
    # Native thread exit can precede delivery of the queued GUI refusal/retirement.
    qtbot.waitUntil(
        lambda: view._adoption_thread is None
        and not view.busy()
        and "The rigs need their RAW basis before the map can be adopted" in view.use_map_progress.text(),
        timeout=30000,
    )
    # This placement-only fixture has no Store/source/profile/renderer proof.
    # A matrix with the right dimensions cannot authorize native adoption.
    assert view.last_adoption is None
    assert "The rigs need their RAW basis before the map can be adopted" in view.use_map_progress.text()
    assert not (lp.painting_folder(view.project).parent / "map_adoption").exists()
    assert _files(lp.painting_folder(view.project)) == painting_before


def _old_proposal_overlay(view):
    """The proposal's overlay as the painting tab drew it before the manual
    map (frozen copy), for the state shown."""

    import cv2
    from PySide6.QtGui import QColor

    height, width = view.image.display.shape[:2]
    classes = cv2.resize(view.proposal.classes, (width, height), interpolation=cv2.INTER_NEAREST)
    corrected = None
    if view.corrections and view.show_corrections.isChecked():
        layer_ids = frozenset(view._layer_ids())
        classes, corrected = view._display_corrections.get(
            (id(view.proposal), view.proposal.digest, classes.shape, layer_ids),
            classes,
            view.corrections,
            (height / view.image.image_size[1], width / view.image.image_size[0]),
            layer_ids,
        )
    alpha = int(round(2.55 * view.opacity.value()))
    table = np.zeros((256, 4), dtype=np.uint8)
    table[lp.OUTSIDE] = (0, 0, 0, 110)
    for layer in view.session.layers:
        color = QColor(layer.color)
        table[layer.layer_id] = (color.red(), color.green(), color.blue(), alpha)
    hidden = np.zeros(256, dtype=bool)
    for layer_id in view._hidden_layers:
        if 0 < layer_id < 256:
            hidden[layer_id] = True
    table[hidden] = 0
    rgba = table[classes]
    shown = ~hidden[classes]
    boundary = np.zeros(classes.shape, dtype=bool)
    boundary[:, 1:] |= (classes[:, 1:] != classes[:, :-1]) & (shown[:, 1:] | shown[:, :-1])
    boundary[1:, :] |= (classes[1:, :] != classes[:-1, :]) & (shown[1:, :] | shown[:-1, :])
    rgba[boundary] = (255, 255, 255, 235)
    if view.show_uncertain.isChecked():
        uncertain = cv2.resize(
            view.proposal.uncertain().astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        rows, cols = np.indices(classes.shape)
        hatch = ((rows + cols) // 3) % 2 == 0
        if corrected is not None:
            uncertain &= ~corrected
        rgba[uncertain & hatch & shown] = (255, 0, 255, 200)
    if corrected is not None and np.any(corrected):
        inner = cv2.erode(corrected.astype(np.uint8), np.ones((3, 3), np.uint8))
        rgba[corrected & ~inner.astype(bool)] = view_module._CORRECTED_EDGE
    return rgba


def test_the_proposal_is_drawn_as_before_while_the_differences_are_off(view, monkeypatch):
    rendered = _rendered(view, monkeypatch)
    proposal = _proposal()
    proposal.confidence[20:30, 10:70] = 10  # Uncertain pixels, hatched.
    _show(view, proposal)
    view.add_polygon(LEFT)
    view.show_uncertain.setChecked(True)
    view.select_layer_row(2)  # Coarse sand
    assert view.add_correction(lp.Correction("brush", COARSE, [(30.0, 20.0)], radius_px=4.0))
    view.opacity.setValue(70)
    for hide in ((), (COARSE,)):
        for layer_id in hide:
            view.set_layer_visible(layer_id, False)
        view._render_proposal()
        assert np.array_equal(rendered[-1], _old_proposal_overlay(view))
    assert view._manual_thread is None and view._manual is None
