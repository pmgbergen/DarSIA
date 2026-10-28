"""Explicit layer priority preserves old defaults and the drawing beneath it.

All host tests use the synthetic project and strict private USER INI factory
from the series-frame fixture. No real project, photo, setting or run is used.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import layer_painting_manual as manual
from darsia.fluidflower import painting_layer_priority as priority
from darsia.fluidflower import painting_run_preview as preview

from .test_painting_series_frame_models import context  # noqa: F401


# A fixed pre-priority session, including the original compact mark lines.
# This fixture is independent of the serializer under test.
_LEGACY_SESSION = """{
  "schema": 1,
  "format": "darsia-fluidflower-layer-painting-v1",
  "run_id": "private_run",
  "image": {
    "path": "private.png",
    "size": [
      80,
      60
    ],
    "stamp": "private-stamp"
  },
  "layers": [
    {
      "layer_id": 7,
      "name": "First layer",
      "color": "#123456",
      "facies": "First material"
    }
  ],
  "facies": {
    "First material": {
      "role": "porous_medium",
      "porosity": 0.3,
      "permeability": 100.0
    }
  },
  "permeability_unit": "darcy",
  "proposal_settings": {
    "smoothing": "medium",
    "min_region_cm2": 2.0,
    "snap_mm": 10.0,
    "edge_fill_mm": 5.0,
    "detail": "normal"
  },
  "updated_utc": "2020-01-01T00:00:00Z",
  "updated_by": "private fixture",
  "polygons": [
    [7,[[1.0,1.0],[1.0,30.0],[20.0,30.0],[20.0,1.0]]]
  ],
  "strokes": []
}
"""


def _session(polygons):
    return lp.PaintingSession(
        run_id="private_run", image_size=(80, 60), edge_fill_mm=0.,
        layers=[lp.PaintedLayer(7, "First layer", "#123456", "First material"),
                lp.PaintedLayer(11, "Second layer", "#654321", "Second material"),
                lp.PaintedLayer(13, "Third layer", "#aabbcc", "Third material")],
        polygons=deepcopy(polygons),
    )


def _rectangle(layer_id, top, left, bottom, right):
    return lp.Polygon(layer_id, [(top, left), (top, right),
                                 (bottom, right), (bottom, left)])


def test_legacy_session_bytes_and_positional_layer_constructor_are_unchanged():
    payload = json.loads(_LEGACY_SESSION)
    session = lp.PaintingSession.from_dict(payload)
    assert all(not row.always_on_top for row in session.layers)
    assert session.to_dict() == payload
    assert lp._session_text(session).encode("utf8") == _LEGACY_SESSION.encode("utf8")
    # Existing fifth positional arguments still mean snap_mm.
    layer = lp.PaintedLayer(9, "Other layer", "#aabbcc", "Other material", 1.5)
    assert layer.snap_mm == 1.5 and layer.always_on_top is False
    session.layer(7).always_on_top = True
    flagged = session.to_dict()
    assert flagged["layers"][0]["always_on_top"] is True
    assert lp.PaintingSession.from_dict(flagged).layer(7).always_on_top is True
    session.layer(7).always_on_top = False
    assert lp._session_text(session).encode("utf8") == _LEGACY_SESSION.encode("utf8")
    explicit_false = deepcopy(payload)
    explicit_false["layers"][0]["always_on_top"] = False
    assert lp._session_text(lp.PaintingSession.from_dict(explicit_false)) == _LEGACY_SESSION


@pytest.mark.parametrize("shape,scale", [((60, 80), (1., 1.)), ((30, 40), (.5, .5))])
def test_all_off_keeps_exact_original_raster_at_each_grid(shape, scale):
    session = _session([_rectangle(7, 2., 2., 50., 70.),
                        _rectangle(11, 10., 10., 20., 20.)])
    before = deepcopy(session)
    old = lp.rasterize_polygons(session.polygons, shape, scale)
    actual = priority.rasterize_polygons(session, shape, scale)
    assert actual.dtype == old.dtype and actual.shape == old.shape
    assert actual.tobytes() == old.tobytes()
    assert session == before


def test_a_large_priority_polygon_wins_without_removing_the_small_underlay():
    session = _session([_rectangle(7, 2., 2., 50., 70.),
                        _rectangle(11, 10., 10., 20., 20.)])
    points = deepcopy(session.polygons)
    original = lp.rasterize_polygons(session.polygons, (60, 80), (1., 1.))
    assert original[15, 15] == 11
    session.layer(7).always_on_top = True
    actual = priority.rasterize_polygons(session, (60, 80), (1., 1.))
    assert actual[15, 15] == 7 and actual[30, 40] == 7
    assert actual[55, 75] == lp.OUTSIDE
    assert session.polygons == points
    assert len(session.polygons) == 2 and session.polygons[1].layer_id == 11
    session.layer(7).always_on_top = False
    assert priority.rasterize_polygons(session, (60, 80), (1., 1.)).tobytes() == original.tobytes()


def test_priority_group_retains_smallest_area_and_later_equal_area_rules():
    session = _session([_rectangle(7, 2., 2., 50., 70.),
                        _rectangle(11, 10., 10., 30., 30.),
                        _rectangle(13, 15., 15., 35., 35.)])
    for layer in session.layers:
        layer.always_on_top = True
    before = deepcopy(session.polygons)
    actual = priority.rasterize_polygons(session, (60, 80), (1., 1.))
    assert actual[5, 5] == 7
    assert actual[12, 12] == 11
    assert actual[20, 20] == 13, "Later equal-area priority polygon must win"
    assert session.polygons == before
    # Reversing only the equal-area drawing order reverses their tie.
    session.polygons[1:] = list(reversed(session.polygons[1:]))
    assert priority.rasterize_polygons(session, (60, 80), (1., 1.))[20, 20] == 11



def test_current_drawing_preview_prioritizes_polygons_without_changing_strokes():
    session = _session([_rectangle(7, 2., 2., 50., 70.)])
    session.strokes = [lp.Stroke(11, 2., [(15., 15.), (55., 75.)])]
    before_polygons, before_strokes = deepcopy(session.polygons), deepcopy(session.strokes)
    ordinary = preview.drawing_layers(session, None)
    assert ordinary[15, 15] == 11 and ordinary[55, 75] == 11
    session.layer(7).always_on_top = True
    marked = preview.drawing_layers(session, None)
    assert marked[15, 15] == 7, "Priority polygon must remain above an ordinary stroke"
    assert marked[55, 75] == 11, "Priority must not erase marks beyond its polygon"
    assert session.polygons == before_polygons and session.strokes == before_strokes
    session.layer(7).always_on_top = False
    assert preview.drawing_layers(session, None).tobytes() == ordinary.tobytes()


def _overlapping_host(context):
    view = context.view
    view.add_polygon(_rectangle(2, 0., 0., 119., 159.))
    view.add_polygon(_rectangle(3, 20., 20., 40., 40.))
    view.map_shown.setCurrentIndex(view.map_shown.findData(manual.MANUAL_MAP_KIND))
    return view


def _drawn(view, qtbot):
    qtbot.waitUntil(lambda: (not view.busy()
        and view._manual_thread is None and not view._manual_timer.isActive()
        and view._manual_current()), timeout=30000)


def _row(view, layer_id):
    return next(index for index, layer in enumerate(view.session.layers)
                if layer.layer_id == layer_id)


def _polygon_state(view):
    return [(polygon.layer_id, list(polygon.points)) for polygon in view.session.polygons]


def test_actual_top_checkbox_has_one_undo_and_preserves_show_and_proposal(context, qtbot):
    view = _overlapping_host(context)
    _drawn(view, qtbot)
    assert view._manual.classes[30, 30] == 3
    old_key = view._manual_inputs_key()
    polygons = _polygon_state(view)
    proposal = lp.LayerProposal(np.full((120, 160), 3, dtype=np.uint8),
        np.full((120, 160), 255, dtype=np.uint8), (160, 120), "private-proposal")
    view._proposal_ready(proposal)
    original_arrays = (proposal.classes.tobytes(), proposal.confidence.tobytes(), proposal.digest)
    view.map_shown.setCurrentIndex(view.map_shown.findData("proposal"))
    table = view.layer_table
    row = _row(view, 2)
    assert table.horizontalHeaderItem(4).text() == "Show"
    assert table.horizontalHeaderItem(5).text() == "Top"
    shown_before = table.item(row, 4).checkState()
    top = table.item(row, 5)
    assert top.flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert top.checkState() == Qt.CheckState.Unchecked
    undo_count = len(view._undo_stack)
    # This actual table item emits itemChanged through the installed Qt slot.
    top.setCheckState(Qt.CheckState.Checked)
    assert view.session.layer(2).always_on_top is True
    assert len(view._undo_stack) == undo_count + 1
    assert view._map_kind() == manual.MANUAL_MAP_KIND
    assert view.show_proposal.text() == "Manual map"
    assert any("manual-map version" in problem for problem in view.export_problems_now())
    _drawn(view, qtbot)
    assert view._manual.classes[30, 30] == 2
    assert view._manual_inputs_key() != old_key
    assert _polygon_state(view) == polygons
    assert table.item(_row(view, 2), 4).checkState() == shown_before
    pairs = list(zip(view.canvas._polygons, view.canvas._polygon_items))
    assert min(item.zValue() for polygon, item in pairs if polygon.layer_id == 2) > max(
        item.zValue() for polygon, item in pairs if polygon.layer_id == 3)
    assert (proposal.classes.tobytes(), proposal.confidence.tobytes(), proposal.digest) == original_arrays
    view.set_layer_always_on_top(2, True)  # Repeating a value adds no undo.
    assert len(view._undo_stack) == undo_count + 1
    view.undo()
    _drawn(view, qtbot)
    assert view.session.layer(2).always_on_top is False
    assert len(view._undo_stack) == undo_count
    assert view.layer_table.item(_row(view, 2), 5).checkState() == Qt.CheckState.Unchecked
    assert view._manual_inputs_key() == old_key and view._manual.classes[30, 30] == 3
    assert _polygon_state(view) == polygons
    view.redo()
    _drawn(view, qtbot)
    assert view.session.layer(2).always_on_top is True
    assert view._manual.classes[30, 30] == 2 and _polygon_state(view) == polygons
    assert (proposal.classes.tobytes(), proposal.confidence.tobytes(), proposal.digest) == original_arrays


def test_manual_versions_restore_both_priority_and_original_underlay(context, qtbot):
    view = _overlapping_host(context)
    _drawn(view, qtbot)
    polygons = _polygon_state(view)
    unprioritized = view.save_version("Original overlap")
    assert unprioritized is not None, view.versions_note.text()
    view.set_layer_always_on_top(2, True)
    _drawn(view, qtbot)
    expected_classes = view._manual.classes.copy()
    prioritized = view.save_version("Priority overlap")
    assert prioritized is not None, view.versions_note.text()
    assert prioritized["sha256"] != unprioritized["sha256"]
    loaded, stored = manual.load_manual_version(prioritized)
    assert loaded.layer(2).always_on_top is True and stored.classes[30, 30] == 2
    assert np.array_equal(stored.classes, expected_classes)
    saved_files = {p: p.read_bytes() for record in (unprioritized, prioritized)
                   for p in Path(record["folder"]).iterdir() if p.is_file()}
    view.set_layer_always_on_top(2, False)
    _drawn(view, qtbot)
    assert view._manual.classes[30, 30] == 3
    assert view.restore_version(prioritized), view.versions_note.text()
    _drawn(view, qtbot)
    assert view.session.layer(2).always_on_top is True
    assert np.array_equal(view._manual.classes, expected_classes)
    assert manual.manual_map_digest(view.session, view._manual) == prioritized["sha256"]
    assert _polygon_state(view) == polygons
    assert view.restore_version(unprioritized), view.versions_note.text()
    _drawn(view, qtbot)
    assert view.session.layer(2).always_on_top is False
    assert view._manual.classes[30, 30] == 3 and _polygon_state(view) == polygons
    assert manual.manual_map_digest(view.session, view._manual) == unprioritized["sha256"]
    assert all(path.read_bytes() == raw for path, raw in saved_files.items())


def test_live_outer_photo_context_keeps_priority_and_original_underlay():
    from darsia.fluidflower.painting_drawing_coverage import add_explicit_outside
    from darsia.fluidflower.series_geometry import RigFrame
    session = _session([_rectangle(7, 5., 2., 50., 50.),
                        _rectangle(11, 10., 5., 20., 15.)])
    session.image_path = "private.png"
    frame = RigFrame(run_id=session.run_id, image=session.image_path,
        image_size=session.image_size, width_m=.06, height_m=.059,
        corners=dict(top_left=(0.,20.), top_right=(0.,79.),
                     bottom_right=(59.,79.), bottom_left=(59.,20.)))
    native = manual.manual_map(session, session.image_size, frame=frame)
    ordinary, _ = add_explicit_outside(session, frame, native.classes)
    assert ordinary[15, 10] == 11
    session.layer(7).always_on_top = True
    before = deepcopy(session.to_dict())
    native = manual.manual_map(session, session.image_size, frame=frame)
    marked, count = add_explicit_outside(session, frame, native.classes)
    assert count > 0 and marked[15, 10] == 7
    inside = lp.frame_mask(frame, native.classes.shape, (1., 1.))
    assert np.array_equal(marked[inside], native.classes[inside])
    assert session.to_dict() == before
    session.layer(7).always_on_top = False
    restored, _ = add_explicit_outside(session, frame, manual.manual_map(session, session.image_size, frame=frame).classes)
    assert np.array_equal(restored, ordinary)
