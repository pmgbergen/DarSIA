"""Live drawing context has photo bounds; saved/scientific maps keep their bytes.

All hosts/settings/files are the synthetic private series fixture. Guide tests
use its exact native photo/model grid and explicitly supplied synthetic camera
profile metadata; they make no RAW-rendering qualification claim.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import threading

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QTimer, Qt
from PySide6.QtWidgets import QGraphicsView, QWidget
from shiboken6 import isValid

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import layer_painting_manual as manual
from darsia.fluidflower import painting_drawing_coverage as coverage
from darsia.fluidflower import painting_other_frames_ui as guides
from darsia.fluidflower import painting_photo_continuation as continuation
from darsia.fluidflower import painting_preview_frame_domain as domain
from darsia.fluidflower import painting_run_preview as preview_core
from darsia.fluidflower.series_geometry import RigFrame

from .test_painting_series_frame_models import context  # noqa: F401
from .test_painting_series_frame_models import _files


def _frame(context, run, *, left=20., right=140., top=5., bottom=110.):
    return RigFrame(corners=dict(top_left=(top, left), top_right=(top, right),
        bottom_right=(bottom, right), bottom_left=(bottom, left)), width_m=.12,
        height_m=.105, run_id=run, source="private saved frame",
        image=str(context.photos[run]), image_size=(160, 120))


def _prepare(context, qtbot):
    view = context.view
    view.rig_panel.frame = _frame(context, "run_a")
    view.session.edge_fill_mm = 0.
    view.add_polygon(lp.Polygon(3, [(25., -20.), (25., 10.), (45., 10.), (45., -20.)]))
    view.map_shown.setCurrentIndex(view.map_shown.findData(manual.MANUAL_MAP_KIND))
    view._update_manual_map()
    qtbot.waitUntil(lambda: (view._manual_thread is None and view._manual_current()), timeout=15000)
    view.save_session()
    for timer in (view._save_timer, view._manual_timer, view._saturation_timer):
        timer.stop()
    view._dirty = False
    return view


def _before(context):
    context.settings.sync()
    return dict(files=_files(context.project.root, context.series),
        session=deepcopy(context.view.session.to_dict()), project=deepcopy(context.project.manifest),
        settings=context.settings_path.read_bytes())


def _unchanged(context, before):
    assert _files(context.project.root, context.series) == before["files"]
    assert context.view.session.to_dict() == before["session"]
    assert context.project.manifest == before["project"]
    context.settings.sync()
    assert context.settings_path.read_bytes() == before["settings"]


def _other(context, qtbot):
    preview = context.view.run_preview
    preview.runs.setCurrentIndex(preview.runs.findData("run_b"))
    qtbot.waitUntil(lambda: (not preview.busy() and preview._waiting_key is None
        and not preview._pending), timeout=30000)
    assert preview.last_result is not None, preview.note.text()
    return preview


def test_live_manual_context_keeps_explicit_outer_polygons_without_changing_saved_map(context, qtbot):
    view = _prepare(context, qtbot)
    native = view._manual.classes.copy()
    assert native[35, 5] == lp.OUTSIDE
    record = manual.save_manual_version(context.project, view.session, view._manual, name="Private frozen map")
    before = _before(context)
    display = view._manual_display_layers()
    assert display["classes"][35, 5] == 3
    assert display["classes"][35, 15] == 2, "Only the polygon actually covering that pixel is retained"
    assert display["classes"][95, 5] == lp.OUTSIDE, "Undrawn outer pixels are never continued"
    assert np.array_equal(view._manual.classes, native)
    saved_session, saved_map = manual.load_manual_version(record)
    assert np.array_equal(saved_map.classes, native)
    assert saved_session.to_dict() == view.session.to_dict()
    view._refresh_manual_status()
    assert "Saved manual versions keep their original" in view.manual_status.text()
    _unchanged(context, before)


@pytest.mark.parametrize("source_kind", ["manual", "drawing"])
def test_live_placement_carries_outer_drawing_but_never_claims_pixels_beyond_source_photo(context, qtbot, monkeypatch, source_kind):
    view = _prepare(context, qtbot)
    if source_kind == "drawing":
        source = view.run_preview.sources
        source.setCurrentIndex(next(i for i in range(source.count())
            if source.itemData(i) == ("drawing", None)))
    before = _before(context)
    captured = []
    original = view.run_preview.canvas.set_proposal
    def observe(rgba):
        captured.append(None if rgba is None else rgba.copy())
        return original(rgba)
    monkeypatch.setattr(view.run_preview.canvas, "set_proposal", observe)
    preview = _other(context, qtbot)
    result = preview.last_result
    assert result.layers[35, 10] == 3, "The exact target translation carries the explicit source polygon"
    assert result.measurements[coverage.KEY]["display_only"] is True
    value = result.measurements[domain.KEY]
    assert value["status"] == "available" and not value["display_inside"][35, 10]
    assert value["projected_frame_native_xy"][0] == [25., 5.]
    rgba = next(array for array in reversed(captured) if array is not None)
    assert rgba[35, 10, 3] > 0, "Explicit outer context remains visible"
    continued = result.measurements[continuation.KEY]
    # The configured target's own frame spans the complete 160 x 120 photo.
    # N373 permits separate geometric continuation there, while the unchanged
    # original source-photo coverage still records that this pixel was filled.
    assert continued["display_inside"][35, 1] and continued["display_continued"][35, 1]
    assert result.uncovered[35, 1], "Continuation must not rewrite original source-photo provenance"
    assert np.array_equal(rgba[35, 1], [0, 220, 255, 150])
    assert "Target own frame" in preview.note.text()
    assert "original filled/support provenance is retained" in preview.note.text()
    assert "placed-reference-frame diagnostic is unchanged" in preview.note.text()
    _unchanged(context, before)


@pytest.mark.parametrize("source_kind", ["manual", "drawing"])
def test_live_continuation_stays_hidden_outside_explicit_saved_target_own_frame(context, qtbot, monkeypatch, source_kind):
    """A saved target frame excludes replicated pixels but keeps real context."""
    view = _prepare(context, qtbot)
    target = _frame(context, "run_b", left=20., right=159., top=0., bottom=119.)
    target.width_m, target.height_m = .139, .119
    link = context.project.manifest["tools"]["native_series"]
    link.setdefault("rig_frames", {})["run_b"] = target.to_dict()
    (context.project.root / "series.json").write_text(json.dumps(context.project.manifest), encoding="utf8")
    if source_kind == "drawing":
        source = view.run_preview.sources
        source.setCurrentIndex(next(i for i in range(source.count())
            if source.itemData(i) == ("drawing", None)))
    before = _before(context)
    captured = []
    original = view.run_preview.canvas.set_proposal
    def observe(rgba):
        captured.append(None if rgba is None else rgba.copy())
        return original(rgba)
    monkeypatch.setattr(view.run_preview.canvas, "set_proposal", observe)
    preview = _other(context, qtbot)
    result = preview.last_result
    continued = result.measurements[continuation.KEY]
    rgba = next(array for array in reversed(captured) if array is not None)
    assert continued["authority"] == "saved rig frame"
    assert not continued["display_inside"][35, 1] and not continued["display_continued"][35, 1]
    assert result.uncovered[35, 1] and rgba[35, 1, 3] == 0
    assert not continued["display_inside"][35, 10] and not result.uncovered[35, 10]
    assert result.layers[35, 10] == 3 and rgba[35, 10, 3] > 0, "The exact explicit outer polygon remains visible"
    _unchanged(context, before)


def test_a_current_proposal_is_not_extended_or_replaced_by_live_polygon_context(context, qtbot):
    view = _prepare(context, qtbot)
    proposal = lp.LayerProposal(np.full((120, 160), 2, dtype=np.uint8),
        np.full((120, 160), 255, dtype=np.uint8), (160, 120),
        lp.proposal_digest(view.session, view.current_frame()))
    view._proposal_ready(proposal)
    view.map_shown.setCurrentIndex(view.map_shown.findData("proposal"))
    view.save_session()
    for timer in (view._save_timer, view._manual_timer, view._saturation_timer):
        timer.stop()
    before = _before(context)
    classes, confidence = proposal.classes.copy(), proposal.confidence.copy()
    result = _other(context, qtbot).last_result
    assert coverage.KEY not in result.measurements
    assert np.array_equal(proposal.classes, classes)
    assert np.array_equal(proposal.confidence, confidence)
    assert proposal.digest == lp.proposal_digest(view.session, view.current_frame())
    _unchanged(context, before)


def test_explicit_saved_manual_version_remains_exact_and_does_not_gain_live_polygons(context, qtbot):
    view = _prepare(context, qtbot)
    record = manual.save_manual_version(context.project, view.session, view._manual, name="Private frozen map")
    view.versions = view._read_versions(view.session.run_id)
    view.run_preview.refresh()
    sources = view.run_preview.sources
    selected = next(i for i in range(sources.count()) if sources.itemData(i)[1] is not None
        and sources.itemData(i)[1].get("folder") == record["folder"])
    sources.setCurrentIndex(selected)
    before = _before(context)
    result = _other(context, qtbot).last_result
    assert coverage.KEY not in result.measurements
    assert result.layers[35, 10] == lp.OUTSIDE
    _saved_session, native = manual.load_manual_version(record)
    assert native.classes[35, 5] == lp.OUTSIDE
    _unchanged(context, before)


def test_current_drawing_keeps_strokes_but_manual_context_still_uses_only_polygons(context, qtbot):
    view = _prepare(context, qtbot)
    session = deepcopy(view.session)
    session.strokes.append(lp.Stroke(3, 2., [(80., 5.)]))
    native = view._manual.classes.copy()
    before = deepcopy(session.to_dict())
    only_polygons, _ = coverage.add_explicit_outside(session, view.current_frame(), native)
    drawing, _ = coverage.add_explicit_outside(session, view.current_frame(), native, drawing=True)
    assert only_polygons[80, 5] == lp.OUTSIDE and drawing[80, 5] == 3
    assert drawing.shape == native.shape and session.to_dict() == before
    assert np.array_equal(view._manual.classes, native)


@pytest.mark.parametrize("failure", ["run", "photo", "grid", "collapsed"])
def test_extra_coverage_refuses_wrong_frame_without_touching_input(context, qtbot, failure):
    view = _prepare(context, qtbot)
    frame = deepcopy(view.current_frame())
    if failure == "run":
        frame.run_id = "another_run"
    elif failure == "photo":
        frame.image = str(context.photos["run_b"])
    elif failure == "grid":
        frame.image_size = (80, 60)
    else:
        frame.corners = {key: (4., 4.) for key in frame.corners}
    native = view._manual.classes.copy()
    before = native.tobytes()
    with pytest.raises(ValueError):
        coverage.add_explicit_outside(view.session, frame, native)
    assert native.tobytes() == before


def _guide_context(context):
    view = context.view
    first, second = _frame(context, "run_a"), _frame(context, "run_b", left=0., right=159.)
    view.rig_panel.frame = first
    link = context.project.manifest["tools"]["native_series"]
    link["rig_frames"] = {"run_a": first.to_dict(), "run_b": second.to_dict()}
    # Explicit synthetic profile for this private native-grid model. No RAW
    # renderer is replaced and no real profile/settings file is read or written.
    view.image.camera_u = SimpleNamespace(path="private-profile.json", sha256="1" * 64,
                                         profile_id="private-profile", camera_wh=(160, 120))
    return view.other_frames


def test_other_frames_default_off_then_uses_target_own_frame_and_warns_beyond_photo(context, qtbot):
    controller = _guide_context(context)
    assert not controller.show_frames.isChecked() and not controller.busy()
    controller.context_changed()
    assert not controller.busy() and not controller._items
    before = _before(context)
    controller.show_frames.setChecked(True)
    qtbot.waitUntil(lambda: not controller.busy(), timeout=15000)
    assert controller.result is not None, controller.note.text()
    assert len(controller.result.guides) == 1 and not controller.result.missing
    guide = controller.result.guides[0]
    assert guide["run_id"] == "run_b" and guide["source"] == "private saved frame"
    assert guide["corners_rc"] == ((5., -5.), (5., 154.), (110., 154.), (110., -5.))
    assert guide["extends_outside_photo"] and "outside this photograph" in controller.note.text()
    assert len(controller._items) == 1
    assert controller._items[0].acceptedMouseButtons() == Qt.MouseButton.NoButton
    assert controller._items[0].pen().isCosmetic()
    assert guide["corners_rc"] != tuple(context.view.current_frame().corners.values())
    controller.show_frames.setChecked(False)
    assert not controller._items and controller.result is not None
    _unchanged(context, before)


@pytest.mark.parametrize("failure", ["run", "photo", "grid", "collapsed"])
def test_invalid_saved_own_frame_is_reported_and_never_replaced_by_configuration(context, qtbot, failure):
    controller = _guide_context(context)
    own = context.project.manifest["tools"]["native_series"]["rig_frames"]["run_b"]
    if failure == "run":
        own["run_id"] = "another_run"
    elif failure == "photo":
        own["image"] = str(context.photos["run_a"])
    elif failure == "grid":
        own["image_size"] = [80, 60]
    else:
        own["corners"] = {key: [4., 4.] for key in own["corners"]}
    before = _before(context)
    controller.show_frames.setChecked(True)
    qtbot.waitUntil(lambda: not controller.busy(), timeout=15000)
    assert controller.result is not None, controller.note.text()
    assert not controller.result.guides and len(controller.result.missing) == 1
    assert controller.result.missing[0]["run_id"] == "run_b"
    assert "Unavailable" in controller.note.text() and not controller._items
    _unchanged(context, before)


def test_guide_metadata_worker_keeps_gui_alive_blocks_close_and_discards_changed_context(context, qtbot, monkeypatch):
    controller = _guide_context(context)
    entered, release = threading.Event(), threading.Event()
    original = guides.load_other_frames
    def blocked(**arguments):
        assert threading.get_ident() != context.gui_thread
        entered.set()
        assert release.wait(10.), "Private test gate was not released"
        return original(**arguments)
    monkeypatch.setattr(guides, "load_other_frames", blocked)
    ticks = []
    timer = QTimer(context.view)
    timer.setInterval(1)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start()
    try:
        controller.show_frames.setChecked(True)
        qtbot.waitUntil(entered.is_set, timeout=5000)
        qtbot.waitUntil(lambda: len(ticks) >= 3, timeout=5000)
        assert controller.busy() and not controller.can_close() and not context.view.can_close()
        context.view._reference_token = {"private": "changed token"}
        controller.refresh()
        assert not controller._items
    finally:
        release.set()
        controller.wait_for_workers()
        timer.stop()
    assert not controller.busy() and controller.result is None and not controller._items
    assert "stale" in controller.note.text().lower() or "bound reference" in controller.note.text().lower()


def test_controller_late_event_after_private_canvas_destruction_is_safe(qtbot):
    host = QWidget()
    host.canvas = QGraphicsView(host)
    qtbot.addWidget(host)
    controller = guides.PaintingOtherFramesController(host)
    canvas, viewport = controller.canvas, controller._viewport
    canvas.deleteLater()
    qtbot.waitUntil(lambda: not isValid(canvas), timeout=5000)
    assert controller._disposed
    assert controller.eventFilter(viewport, QEvent(QEvent.Type.Resize)) is False
    controller.refresh()
