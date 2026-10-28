"""Slow or failed previews stay visible, responsive and scoped to selection."""
from copy import deepcopy
from types import SimpleNamespace
import threading

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QThread, QTimer
from PySide6.QtGui import QColor
from shiboken6 import isValid

from darsia.fluidflower import painting_run_preview_ui as ui
from darsia.fluidflower.painting_run_preview import RunPreview

from .test_painting_corrections import view  # noqa: F401
from .test_painting_run_preview import private_settings, _files, _mock_policy_unrelated_preview  # noqa: F401


def _chooser(view, qtbot, monkeypatch):
    # Shared lifecycle-only packets intentionally have no scientific source binding.
    _mock_policy_unrelated_preview(monkeypatch)
    for timer in (view._save_timer, view._manual_timer, view._saturation_timer):
        timer.stop()
    qtbot.waitUntil(lambda: not view.busy(), timeout=10000)
    controller = view.run_preview
    controller.run_inputs = lambda: [SimpleNamespace(run_id=view.session.run_id),
                                    SimpleNamespace(run_id="target_a"), SimpleNamespace(run_id="target_b")]
    controller._index = ""
    controller.refresh()
    controller._index = str(view.project.root / "preview-geometry-index.json")
    monkeypatch.setattr(ui.preview, "preview_layers", lambda session, *args, **kwargs:
                        (session, np.full((60, 80), 2, np.uint8)))
    return controller


def _result(session, values, options):
    return RunPreview(options["target_run"], np.full((60, 80, 3), (120, 90, 60), np.uint8),
        values.copy(), np.zeros(values.shape, bool), options["target_run"] + ".png",
        session.run_id, options["source_name"])


def _select(controller, name="target_a"):
    controller.runs.setCurrentIndex(controller.runs.findData(name))


def test_slow_loading_has_canvas_message_stage_elapsed_and_gui_heartbeat(view, qtbot, monkeypatch):
    controller = _chooser(view, qtbot, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    gui_thread = view.thread()
    beats = []
    timer = QTimer(view)
    timer.setInterval(5)
    timer.timeout.connect(lambda: beats.append(1))

    def prepare(session, values, **options):
        assert QThread.currentThread() != gui_thread
        options["on_progress"]("Loading target_a photograph on its camera grid")
        entered.set()
        assert release.wait(10)
        return _result(session, values, options)

    monkeypatch.setattr(ui.preview, "prepare_preview", prepare)
    before = deepcopy(view.session), _files(view)
    timer.start()
    try:
        _select(controller)
        qtbot.waitUntil(entered.is_set, timeout=3000)
        qtbot.waitUntil(lambda: "Loading target_a photograph" in controller.status_text.text(), timeout=3000)
        assert controller.status_overlay.isVisible() and controller.progress_bar.isVisible()
        assert controller.status_overlay.geometry() == controller.canvas.viewport().rect()
        assert " s" in controller.status_text.text()
        qtbot.waitUntil(lambda: len(beats) >= 3, timeout=3000)
        release.set()
        qtbot.waitUntil(lambda: controller.thread is None, timeout=5000)
        assert controller.last_result is not None and not controller.status_overlay.isVisible()
        assert not controller._elapsed_timer.isActive()
        assert view.session == before[0] and _files(view) == before[1]
    finally:
        release.set()
        controller.wait()
        timer.stop()


def test_failed_loader_shows_reason_on_canvas_and_status_without_silent_black(view, qtbot, monkeypatch):
    controller = _chooser(view, qtbot, monkeypatch)

    def fail(*args, **kwargs):
        raise OSError("photograph is unavailable")

    monkeypatch.setattr(ui.preview, "prepare_preview", fail)
    _select(controller)
    qtbot.waitUntil(lambda: controller.thread is None, timeout=5000)
    assert controller.last_result is None
    assert controller.status_overlay.isVisible() and not controller.progress_bar.isVisible()
    assert "photograph is unavailable" in controller.status_text.text()
    assert "Preview unavailable" in controller.note.text()
    assert not controller._elapsed_timer.isActive()


def test_gui_render_exception_is_a_visible_failure_and_worker_is_retired(view, qtbot, monkeypatch):
    controller = _chooser(view, qtbot, monkeypatch)
    monkeypatch.setattr(ui.preview, "prepare_preview", lambda session, values, **options:
                        _result(session, values, options))
    def broken_overlay(*args):
        raise ValueError("overlay could not be painted")
    monkeypatch.setattr(controller.canvas, "set_proposal", broken_overlay)
    _select(controller)
    qtbot.waitUntil(lambda: controller.thread is None, timeout=5000)
    assert controller.worker is None and controller.last_result is None
    assert controller.status_overlay.isVisible()
    assert "Could not display the preview" in controller.note.text()
    assert "overlay could not be painted" in controller.status_text.text()


@pytest.mark.parametrize("hidden", [False, True])
def test_optional_display_styles_honor_color_and_hidden_gates(view, qtbot, monkeypatch, hidden):
    controller = _chooser(view, qtbot, monkeypatch)
    received, captured = [], []

    def styles(session=None):
        received.append(session)
        return {2: QColor("#112233")}, frozenset({2}) if hidden else frozenset()

    monkeypatch.setattr(view, "display_layer_styles", styles, raising=False)
    monkeypatch.setattr(ui.preview, "prepare_preview", lambda session, values, **options:
                        _result(session, values, options))
    original = controller.canvas.set_proposal
    def record(rgba):
        captured.append(rgba.copy())
        original(rgba)
    monkeypatch.setattr(controller.canvas, "set_proposal", record)
    _select(controller)
    qtbot.waitUntil(lambda: controller.thread is None and controller.last_result is not None, timeout=5000)
    assert received and received[0].run_id == view.session.run_id
    assert np.array_equal(captured[-1][20, 30], [0, 0, 0, 0] if hidden else [17, 34, 51, 100])


def test_early_own_photo_map_visible_while_offsets_wait_and_retained_on_offset_failure(view, qtbot, monkeypatch):
    controller = _chooser(view, qtbot, monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def prepare(session, values, **options):
        result = _result(session, values, options)
        options["on_preview"](result)
        options["on_progress"]("Measuring boundaries on target_a")
        entered.set()
        assert release.wait(10)
        raise OSError("reference offsets unavailable")

    monkeypatch.setattr(ui.preview, "prepare_preview", prepare)
    try:
        _select(controller)
        qtbot.waitUntil(entered.is_set, timeout=3000)
        qtbot.waitUntil(lambda: controller.last_result is not None, timeout=3000)
        assert controller.thread is not None and not controller.status_overlay.isVisible()
        assert controller.last_result.photo_path == "target_a.png"
        assert np.array_equal(controller.last_result.photo,
                              np.full((60, 80, 3), (120, 90, 60), np.uint8))
        release.set()
        qtbot.waitUntil(lambda: controller.thread is None, timeout=5000)
        assert controller.last_result is not None and not controller.status_overlay.isVisible()
        assert "retained" in controller.note.text() and "reference offsets unavailable" in controller.note.text()
    finally:
        release.set()
        controller.wait()


def test_busy_host_retains_latest_selection_and_retries_once_after_idle(view, qtbot, monkeypatch):
    controller = _chooser(view, qtbot, monkeypatch)
    occupied, seen = [True], []
    monkeypatch.setattr(view, "busy", lambda: occupied[0])
    def prepare(session, values, **options):
        seen.append(options["target_run"])
        return _result(session, values, options)
    monkeypatch.setattr(ui.preview, "prepare_preview", prepare)
    _select(controller, "target_a")
    assert controller.thread is None and controller._retry_timer.isActive()
    _select(controller, "target_b")
    assert "target_b" in controller.status_text.text()
    occupied[0] = False
    view.busy_changed.emit(False)
    qtbot.waitUntil(lambda: controller.thread is None and controller.last_result is not None, timeout=5000)
    assert seen == ["target_b"] and controller.last_result.run_id == "target_b"
    assert not controller._retry_timer.isActive() and controller._waiting_key is None


def test_busy_retry_is_bounded_and_cancelled_on_reference_return(view, qtbot, monkeypatch):
    controller = _chooser(view, qtbot, monkeypatch)
    monkeypatch.setattr(view, "busy", lambda: True)
    _select(controller)
    controller._retry_timer.stop()
    controller._retry_attempts = 120
    controller._retry_waiting()
    assert not controller._retry_timer.isActive() and not controller._elapsed_timer.isActive()
    assert "still busy" in controller.status_text.text()
    controller.runs.setCurrentIndex(controller.runs.findData(view.session.run_id))
    assert controller._waiting_key is None and not controller._retry_timer.isActive()
    assert controller.thread is None and not controller.status_overlay.isVisible()


def test_selection_switch_during_slow_work_never_publishes_old_run(view, qtbot, monkeypatch):
    controller = _chooser(view, qtbot, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    seen = []
    def prepare(session, values, **options):
        seen.append(options["target_run"])
        if options["target_run"] == "target_a":
            entered.set()
            assert release.wait(10)
            options["on_preview"](_result(session, values, options))
        return _result(session, values, options)
    monkeypatch.setattr(ui.preview, "prepare_preview", prepare)
    try:
        _select(controller)
        qtbot.waitUntil(entered.is_set, timeout=3000)
        _select(controller, "target_b")
        release.set()
        qtbot.waitUntil(lambda: controller.thread is None and controller.last_result is not None, timeout=5000)
        assert seen == ["target_a", "target_b"]
        assert controller.last_result.run_id == "target_b"
        assert "target_a" not in controller.note.text()
    finally:
        release.set()
        controller.wait()


def test_refresh_during_same_running_preview_does_not_launch_a_duplicate(view, qtbot, monkeypatch):
    controller = _chooser(view, qtbot, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    seen = []

    def prepare(session, values, **options):
        seen.append(options["target_run"])
        entered.set()
        assert release.wait(10)
        return _result(session, values, options)

    monkeypatch.setattr(ui.preview, "prepare_preview", prepare)
    try:
        _select(controller)
        qtbot.waitUntil(entered.is_set, timeout=3000)
        started = controller._started
        controller.request()
        controller.request()
        assert not controller._pending and controller._started == started
        release.set()
        qtbot.waitUntil(lambda: controller.thread is None and controller.last_result is not None, timeout=5000)
        assert seen == ["target_a"]
    finally:
        release.set()
        controller.wait()


def test_deleted_canvas_cancels_queued_retry_and_late_callbacks(view, qtbot, monkeypatch):
    controller = _chooser(view, qtbot, monkeypatch)
    monkeypatch.setattr(view, "busy", lambda: True)
    _select(controller)
    canvas = controller.canvas
    canvas.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qtbot.waitUntil(lambda: not isValid(canvas))
    controller._retry_waiting()
    controller._elapsed()
    assert controller._disposed and not controller._retry_timer.isActive()
    assert not controller._elapsed_timer.isActive()
    assert controller.eventFilter(canvas, QEvent(QEvent.Type.Resize)) is False
    assert controller.thread is None
