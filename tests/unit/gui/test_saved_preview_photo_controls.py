"""Saved maps keep shared photo tools, stable cached displays and truthful status."""
from copy import deepcopy
from pathlib import Path
from threading import Event

import numpy as np
from PySide6.QtGui import QColor

from darsia.fluidflower import rig_photo_loader
from darsia.fluidflower.series_layer_view import LayerPainterView
from .test_painting_manual_step4_route import _saved_manual
from .test_painting_raw_reference_on_u import raw_view, raw_file, profile, renders, settings  # noqa: F401
from .test_rig_full_raw_editor import packet


def archive(record):
    return {path: path.read_bytes() for path in Path(record['folder']).iterdir() if path.is_file()}


def ready(view, qtbot):
    qtbot.waitUntil(lambda: view.run_preview._version_thread is None
        and view.run_preview._reference_version_key is not None
        and view.run_preview.canvas.photo_display_mapping is not None)


def test_cold_saved_raw_waits_then_exposes_shared_tools_without_mutating_version(raw_view, qtbot, monkeypatch):
    original = raw_view
    record = _saved_manual(original, qtbot, 'V9')
    qtbot.waitUntil(lambda: not original.run_preview.busy())
    before = archive(record), deepcopy(original.session.to_dict())
    hold, started = Event(), Event()
    calls = []
    loader = rig_photo_loader.RigPhotoLoader
    def read(path, profile):
        calls.append(path)
        started.set()
        assert hold.wait(5)
        return packet(path)
    monkeypatch.setattr(rig_photo_loader, 'RigPhotoLoader', lambda: loader(reader=read))
    view = LayerPainterView(original.project, original.reference)
    qtbot.addWidget(view)
    view.resize(1400, 900)
    view.show()
    try:
        view.set_workflow_scope('layers')
        qtbot.waitUntil(started.is_set)
        qtbot.waitUntil(lambda: view.run_preview._version_thread is None)
        preview = view.run_preview
        assert preview.canvas.preview is None
        assert 'Loading' in preview.note.text()
        hold.set()
        ready(view, qtbot)
        assert 'Loading' not in preview.note.text() and 'read only' in preview.note.text()
        assert preview.canvas.pixmap_item.pixmap().size().toTuple() == (120, 95)
        assert preview.active_canvas() is preview.canvas
        assert all(group.isVisible() for group in view._painting_photo_groups)
        assert all(not preview.stack.isAncestorOf(group) for group in view._painting_photo_groups)
        assert preview.read_only_tools.isVisible() and preview.edit_current_map.isEnabled()
        assert view.map_fit.isEnabled() and view.tool_buttons['pan'].isEnabled()
        assert not view.tool_buttons['polygon'].isEnabled()
        preview.canvas.scale(1.7, 1.7)
        zoom = preview.canvas.transform().m11()
        view.map_fit.click()
        assert preview.canvas.transform().m11() != zoom
        preview.edit_current_map.click()
        assert preview.stack.currentIndex() == 0 and preview.active_canvas() is view.canvas
        assert not preview.read_only_tools.isVisible()
        assert view.tool_buttons['polygon'].isEnabled() and view.canvas.paint_enabled
        assert view.session.to_dict() == before[1]
        assert archive(record) == before[0]
        assert len(calls) == 1
    finally:
        hold.set()
        view.wait_for_workers()


def test_saved_cached_display_survives_alignment_key_change_and_keeps_zoom(raw_view, qtbot, monkeypatch):
    view = raw_view
    record = _saved_manual(view, qtbot, 'V9')
    qtbot.waitUntil(lambda: not view.run_preview.busy())
    calls = []
    def read(path, profile):
        calls.append(path)
        return packet(path)
    view._rig_photo_loader.reader = read
    view.set_workflow_scope('layers')
    ready(view, qtbot)
    preview = view.run_preview
    before = archive(record), deepcopy(view.session.to_dict())
    cached_result, cached_session = preview.last_result, preview._last_session
    # A completed alignment changes the generic preview key without changing
    # this already-loaded saved map or its reference photograph.
    monkeypatch.setattr(view, 'reviewed_point_selection', lambda base=None: object())
    assert preview._result_key != preview._key()
    preview._stage = 'Loading an earlier run'
    preview._loading_status = True
    preview._elapsed_timer.start()
    preview._selected()
    assert not preview._elapsed_timer.isActive() and 'Loading' not in preview.note.text()
    hidden = set()
    monkeypatch.setattr(view, 'display_layer_styles', lambda session=None: (
        {layer.layer_id: QColor(layer.color) for layer in (session or view.session).layers}, frozenset(hidden)))
    canvas = preview.canvas
    canvas.scale(1.6, 1.6)
    transform = canvas.transform()
    hidden.update(layer.layer_id for layer in cached_session.layers)
    preview.redraw_display()
    image = canvas.proposal_item.pixmap().toImage()
    assert all(image.pixelColor(col, row).alpha() == 0
        for row, col in np.argwhere(cached_result.layers > 0))
    hidden.clear()
    view.opacity.setValue(72)
    row, col = np.argwhere(cached_result.layers > 0)[0]
    assert canvas.proposal_item.pixmap().toImage().pixelColor(int(col), int(row)).alpha() == round(255 * .72)
    view.show_proposal.setChecked(False)
    assert not canvas.proposal_item.isVisible()
    view.show_proposal.setChecked(True)
    assert canvas.proposal_item.isVisible()
    view.enhance.setChecked(not view.enhance.isChecked())
    assert canvas.transform() == transform
    assert preview.last_result is cached_result and preview._last_session is cached_session
    assert len(calls) == 1 and 'Loading' not in preview.note.text()
    assert archive(record) == before[0] and view.session.to_dict() == before[1]


def test_saved_raw_error_clears_only_after_successful_retry_and_ignores_old_result(raw_view, qtbot):
    view = raw_view
    record = _saved_manual(view, qtbot, 'V9')
    qtbot.waitUntil(lambda: not view.run_preview.busy())
    before = archive(record)
    def fail(path, profile):
        raise ValueError('Synthetic RAW read failure')
    view._rig_photo_loader.reader = fail
    view.set_workflow_scope('layers')
    qtbot.waitUntil(lambda: 'Synthetic RAW read failure' in view.run_preview.note.text())
    assert view.run_preview.canvas.preview is None
    failed_key = view._rig_photo_key
    view._rig_photo_loader.reader = lambda path, profile: packet(path)
    view._invalidate_rig_photo()
    view.show_reference_photo()
    ready(view, qtbot)
    message = view.run_preview.note.text()
    assert 'Synthetic RAW read failure' not in message and 'Loading' not in message
    view._rig_photo_ready(failed_key, None, 'Old delayed failure')
    assert view.run_preview.note.text() == message
    assert view.run_preview.canvas.preview is not None and archive(record) == before
