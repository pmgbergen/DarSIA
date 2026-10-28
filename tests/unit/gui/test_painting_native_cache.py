"""Cold native corrections keep Qt responsive and stale fills cannot undo Undo."""

from dataclasses import replace
from pathlib import Path
import threading

import numpy as np
from PIL import Image
import pytest
from PySide6.QtCore import QSettings, QTimer

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import layer_painting_polygons_only as polygons
from darsia.fluidflower.painting_native_cache import ASYNC_PIXELS
from darsia.fluidflower.series_geometry import RigFrame

from .test_painting_corrections import view  # noqa: F401


@pytest.fixture(autouse=True)
def private_settings(tmp_path):
    old = QSettings.defaultFormat()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    yield
    QSettings.setDefaultFormat(old)


def test_large_cold_cache_keeps_qt_alive_and_undo_reuses_exact_state_before_stale_fill(view, qtbot, tmp_path, monkeypatch):
    side = 1001
    assert side * side > ASYNC_PIXELS
    photo = tmp_path / "large-synthetic.png"
    Image.new("RGB", (side, side), (40, 60, 80)).save(photo)
    reference = replace(view.reference(), baseline_image=photo, width_m=1., height_m=1.)
    monkeypatch.setattr(view, "reference", lambda: reference)
    view.session.image_path = str(photo)
    view.session.image_size = (side, side)
    view.session.image_stamp = lp.image_stamp(photo)
    view.session.polygons = [lp.Polygon(2, [(5., 450.), (5., 550.), (995., 550.), (995., 450.)])]
    view.session.strokes = []
    view.corrections = []
    display = np.full((101, 101, 3), .25, np.float32)
    view.image = lp.PaintingImage(photo, (side, side), display, display.copy())
    view.canvas.set_photo(np.full((101, 101, 3), 64, np.uint8), (side, side), token="native-cache-synthetic", reset_view=True)
    frame = RigFrame(corners={"top_left": (5., 5.), "top_right": (5., 995.),
                              "bottom_right": (995., 995.), "bottom_left": (995., 5.)},
                     width_m=.99, height_m=.99, run_id=view.session.run_id,
                     image=str(photo), image_size=(side, side))
    monkeypatch.setattr(view, "current_frame", lambda: frame)
    proposal = polygons.make_proposal((side, side), view.session, frame=frame)
    original_arrays = proposal.classes.copy(), proposal.inside.copy()
    view.select_layer_row(2)
    for timer in (view._save_timer, view._manual_timer, view._saturation_timer):
        timer.stop()

    entered = {0: threading.Event(), 2: threading.Event()}
    released = {0: threading.Event(), 2: threading.Event()}
    blocked = set()
    worker_calls = []
    original_corrected = polygons.corrected_classes

    def observed(proposal_value, corrections, *, layer_ids=None):
        in_gui = threading.current_thread() is threading.main_thread()
        worker_calls.append((len(corrections), in_gui))
        if in_gui:
            raise AssertionError("Native correction work ran in the GUI thread")
        count = len(corrections)
        if count in entered and count not in blocked:
            blocked.add(count)
            entered[count].set()
            if not released[count].wait(20):
                raise AssertionError("The synthetic native worker was not released")
        return original_corrected(proposal_value, corrections, layer_ids=layer_ids)

    monkeypatch.setattr(polygons, "corrected_classes", observed)
    beats = []
    heartbeat = QTimer(view)
    heartbeat.setInterval(5)
    heartbeat.timeout.connect(lambda: beats.append(1))
    heartbeat.start()
    try:
        view._proposal_ready(proposal)
        qtbot.waitUntil(entered[0].is_set, timeout=10000)
        assert view._native_async.busy()
        baseline = len(beats)
        qtbot.waitUntil(lambda: len(beats) >= baseline + 4, timeout=3000)
        assert view._large_native_result() is None
        view._refresh_status()
        assert "Computing" in " ".join(view.export_problems_now())
        assert not view.save_version_button.isEnabled() and not view.export_button.isEnabled()
        view.show_differences.blockSignals(True)
        try:
            view.show_differences.setChecked(True)
            assert view._compare_inputs_key() == ""
            view._refresh_manual_status()
            assert "Computing exact native corrections" in view.manual_status.text()
            view.show_differences.setChecked(False)
        finally:
            view.show_differences.blockSignals(False)
        assert not view.fill_region_at(500., 200.)
        assert view.corrections == []
        released[0].set()
        qtbot.waitUntil(lambda: not view._native_async.busy()
                        and view._large_native_result() is not None, timeout=30000)
        initial = view._large_native_result()
        assert not initial.error
        expected, decided, skipped = original_corrected(proposal, (), layer_ids=view._layer_ids())
        assert np.array_equal(initial.classes, expected)
        assert np.array_equal(initial.decided, decided) and initial.skipped == tuple(skipped)
        assert np.array_equal(initial.gap_mask, proposal.inside & (expected == lp.OUTSIDE))
        assert initial.gap_count == int(np.count_nonzero(initial.gap_mask))
        assert initial.display_classes.shape == (101, 101)

        # Fill the left gap through the actual GUI method, then wait for its
        # complete exact state rather than reading an in-progress fallback.
        assert view.fill_region_at(500., 200.)
        view._save_timer.stop()
        qtbot.waitUntil(lambda: not view._native_async.busy()
                        and view._large_native_result() is not None, timeout=30000)
        filled = view._large_native_result()
        assert not filled.error and len(view.corrections) == 1
        expected, decided, skipped = original_corrected(proposal, tuple(view.corrections),
                                                        layer_ids=view._layer_ids())
        assert np.array_equal(filled.classes, expected)
        assert np.array_equal(filled.decided, decided) and filled.skipped == tuple(skipped)
        assert filled.classes[500, 200] == 3 and filled.classes[500, 800] == lp.OUTSIDE
        assert np.all(filled.classes[~proposal.inside] == lp.OUTSIDE)
        assert not filled.decided[~proposal.inside].any()
        assert filled.gap_count == int(np.count_nonzero(proposal.inside & (expected == lp.OUTSIDE)))

        # A second fill is deliberately held in the worker. Undo must select
        # the previous cached state immediately, before that worker finishes.
        view.select_layer_row(3)
        assert view.fill_region_at(500., 800.)
        view._save_timer.stop()
        qtbot.waitUntil(entered[2].is_set, timeout=10000)
        assert view._native_async.busy() and len(view.corrections) == 2
        baseline = len(beats)
        qtbot.waitUntil(lambda: len(beats) >= baseline + 4, timeout=3000)
        view.undo()
        view._save_timer.stop()
        assert len(view.corrections) == 1
        assert view._large_native_result() is filled
        assert np.array_equal(view._corrected_native_state()[0], expected)
        assert view._native_async.busy()
        released[2].set()
        qtbot.waitUntil(lambda: not view._native_async.busy(), timeout=30000)
        assert view._large_native_result() is filled
        assert view._large_native_result().classes[500, 800] == lp.OUTSIDE
        assert np.array_equal(view._corrected_native_state()[0], expected)
        assert np.array_equal(proposal.classes, original_arrays[0])
        assert np.array_equal(proposal.inside, original_arrays[1])
        assert worker_calls and all(not in_gui for _count, in_gui in worker_calls)
        assert not view._native_async.busy()
    finally:
        heartbeat.stop()
        for gate in released.values():
            gate.set()
        view._native_async.wait()
