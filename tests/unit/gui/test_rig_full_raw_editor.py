"""Full RAW rig display keeps saved U coordinates, sources and GUI responsiveness."""
from dataclasses import replace
from pathlib import Path
from threading import Event, Timer

import numpy as np
from PySide6.QtCore import Qt, QTimer

from darsia.fluidflower import series_rig_frame_panel as rig_panel
from .test_painting_raw_reference_on_u import (  # noqa: F401
    raw_view, raw_file, profile, renders, settings,
)
from .test_painting_manual_step4_route import _saved_manual


class Translation:
    def camera_u_to_display_xy(self, points):
        return np.asarray(points) + [20., 15.]

    def display_to_camera_u_xy(self, points):
        return np.asarray(points) - [20., 15.]


def packet(path):
    image = np.full((95, 120, 3), [50, 100, 150], np.uint8)
    support = np.ones(image.shape[:2], bool)
    support[:3, :] = False
    return dict(image_rgb=image, display_plain=image, display_enhanced=image,
        mapping=Translation(), valid_support=support, camera_u_wh=(80, 60),
        source_raw={'path': str(path)})


def loaded(view):
    return view.canvas.photo_display_mapping is not None


def click_u(qtbot, view, row, col):
    point = view.canvas.mapFromScene(view.canvas.raw_to_scene(row, col))
    assert view.canvas.viewport().rect().contains(point)
    qtbot.mouseClick(view.canvas.viewport(), Qt.MouseButton.LeftButton, pos=point)


def test_full_sensor_margin_clicks_save_original_u_and_reject_padding(raw_view, qtbot, monkeypatch):
    view = raw_view
    view._rig_photo_loader.reader = lambda path, profile: packet(path)
    before = view.session.to_dict()
    saved = []
    monkeypatch.setattr(rig_panel, 'save_rig_frame', lambda project, frame, **kwargs: saved.append(frame))
    view.set_workflow_scope('rig_frame')
    qtbot.waitUntil(lambda: loaded(view))
    assert view.canvas.raw_size == view.image.image_size == (80, 60)
    assert view.canvas.pixmap_item.pixmap().size().toTuple() == (120, 95)
    assert view.rig_panel.isEnabled() and view.rig_panel.mark_corners.isEnabled()
    assert not view.canvas.paint_enabled and not view.canvas._marks_visible
    assert view.rig_panel._scene_point(10, -10) == view.canvas.raw_to_scene(10, -10)
    view.rig_panel.mark_corners.setChecked(True)
    click_u(qtbot, view, -14, -10)  # Expanded canvas, outside the actual sensor support.
    assert view.rig_panel._corner_index == 0
    for row, col in ((-5, -10), (-5, 70), (50, 70), (50, -10)):
        click_u(qtbot, view, row, col)
    assert len(saved) == 1 and saved[0].image_size == (80, 60)
    assert saved[0].corners['top_left'][0] < 0 and saved[0].corners['top_left'][1] < 0
    assert np.allclose(saved[0].corners['top_left'], (-5, -10), atol=.3)
    assert view.session.to_dict() == before
    # The injection-point path uses the same inverse display map, including U<0.
    points = []
    monkeypatch.setattr(view.rig_panel, '_store_points', lambda: points.extend(view.rig_panel.current_points()))
    view.rig_panel._add_point()
    view.rig_panel._set_mode('point')
    click_u(qtbot, view, 10, -8)
    picked = view.rig_panel.points.item(0, 0).data(Qt.ItemDataRole.UserRole)
    assert np.allclose([picked['row'], picked['col']], [10, -8], atol=.3)
    assert points
    view.set_workflow_scope('layers')
    assert view.canvas.photo_display_mapping is not None
    assert view.canvas.pixmap_item.pixmap().size().toTuple() == (120, 95)
    assert view.canvas.paint_enabled


def test_rig_scope_rejects_late_saved_version_and_keeps_selected_identity(raw_view, qtbot, monkeypatch):
    from darsia.fluidflower.painting_run_preview_ui import _SavedVersionWorker
    view = raw_view
    hold, started = Event(), Event()
    original = _SavedVersionWorker.run

    def held(worker):
        started.set()
        hold.wait(5)
        original(worker)

    monkeypatch.setattr(_SavedVersionWorker, 'run', held)
    view._rig_photo_loader.reader = lambda path, profile: packet(path)
    try:
        record = _saved_manual(view, qtbot, 'Saved ninth map')
        qtbot.waitUntil(started.is_set)
        selected = view.run_preview.sources.currentData()
        source_bytes = {p.name: p.read_bytes() for p in Path(record['folder']).iterdir()}
        view.set_workflow_scope('rig_frame')
        qtbot.waitUntil(lambda: loaded(view))
        assert view.run_preview.stack.currentIndex() == 0
        hold.set()
        qtbot.waitUntil(lambda: not view.run_preview.busy())
        assert loaded(view) and view.run_preview.stack.currentIndex() == 0
        assert view.rig_panel.mark_corners.isEnabled() and not view._display_only_preview
        assert view._selected_version()['folder'] == record['folder']
        assert view.run_preview.sources.currentData() == selected
        view.set_workflow_scope('layers')
        qtbot.waitUntil(lambda: not view.run_preview.busy())
        assert view.run_preview.stack.currentWidget() is view.run_preview.canvas
        assert view.run_preview.sources.currentData() == selected
        assert {p.name: p.read_bytes() for p in Path(record['folder']).iterdir()} == source_bytes
    finally:
        hold.set()
        qtbot.waitUntil(lambda: not view.run_preview.busy())


def test_slow_raw_reader_heartbeat_and_latest_run_scope_win(raw_view, raw_file, qtbot, monkeypatch):
    view = raw_view
    hold, started = Event(), Event()
    calls, applied, beats = [], [], []
    def reader(path, profile):
        calls.append(Path(path).name)
        if len(calls) == 1:
            started.set()
            hold.wait(5)
        return packet(path)
    view._rig_photo_loader.reader = reader
    original_apply = view._apply_rig_photo
    def apply(value, path, **kwargs):
        applied.append(Path(path).name)
        return original_apply(value, path, **kwargs)
    monkeypatch.setattr(view, '_apply_rig_photo', apply)
    timer = QTimer(view)
    timer.timeout.connect(lambda: beats.append(True))
    timer.start(10)
    try:
        view.set_workflow_scope('rig_frame')
        qtbot.waitUntil(started.is_set)
        qtbot.waitUntil(lambda: len(beats) >= 3)
        view.set_workflow_scope('layers')
        reference = view.reference()
        other = raw_file.with_name('new-run.ARW')
        other.write_bytes(b'private second RAW')
        monkeypatch.setattr(view, '_reference', lambda: replace(reference, run_id='ac53', baseline_image=other))
        view.reload()
        view.set_workflow_scope('rig_frame')
        assert not loaded(view)
        hold.set()
        qtbot.waitUntil(lambda: loaded(view))
        assert calls == [raw_file.name, other.name] and applied == [other.name]
        assert view.canvas.preview['source_raw']['path'] == str(other)
        assert view.session.run_id == 'ac53'
    finally:
        hold.set()
        timer.stop()
        qtbot.waitUntil(lambda: view._rig_photo_loader.thread is None)


def test_raw_read_error_cannot_restore_old_canvas_and_retry_is_available(raw_view, qtbot):
    view = raw_view
    def failed(path, profile):
        raise ValueError('bound profile mismatch')
    view._rig_photo_loader.reader = failed
    view.set_workflow_scope('rig_frame')
    qtbot.waitUntil(lambda: 'bound profile mismatch' in view.photo_note.text())
    assert view.canvas.preview is None and not view.rig_panel.isEnabled()
    view._rig_photo_loader.reader = lambda path, profile: packet(path)
    view.show_reference_photo()
    qtbot.waitUntil(lambda: loaded(view))
    assert view.rig_panel.mark_corners.isEnabled()


def test_real_close_waits_for_finite_raw_reader_without_applying_late_photo(raw_view, qtbot):
    view = raw_view
    hold, started = Event(), Event()
    def reader(path, profile):
        started.set()
        hold.wait(5)
        return packet(path)
    view._rig_photo_loader.reader = reader
    try:
        view.set_workflow_scope('rig_frame')
        qtbot.waitUntil(started.is_set)
        assert not view.can_close()
        view.close()
        assert view.isVisible() and view._rig_photo_loader.busy()
        release = Timer(.05, hold.set)
        release.start()
        view.wait_for_workers()  # Same finite join used by the application close path.
        release.join()
        assert not view._rig_photo_loader.busy() and view.can_close()
        assert not loaded(view) and view.canvas.preview is None
        view.close()
        assert not view.isVisible()
    finally:
        hold.set()
        qtbot.waitUntil(lambda: not view._rig_photo_loader.busy())
