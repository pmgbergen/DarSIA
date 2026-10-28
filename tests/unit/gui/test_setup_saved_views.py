"""Immutable Setup pictures survive navigation; optional boxes own their edits."""
from copy import deepcopy
from pathlib import Path
from threading import Event

from .test_painting_raw_reference_on_u import raw_view, raw_file, profile, renders, settings  # noqa: F401
from .test_rig_full_raw_editor import packet
from .test_painting_manual_step4_route import _saved_manual
from .test_painting_analysis_boxes_integration import context  # noqa: F401


def test_initial_relocated_saved_version_receives_late_full_raw_photo(raw_view, qtbot, monkeypatch, tmp_path):
    """V10 may bind an old path while today's identical RAW uses its alias."""
    from darsia.fluidflower import layer_painting_manual as manual
    from darsia.fluidflower import storage_paths
    from .test_painting_manual_step4_route import _select
    view = raw_view
    first = _saved_manual(view, qtbot, 'Original version')
    qtbot.waitUntil(lambda: not view.run_preview.busy())
    saved, mapping = manual.load_manual_version(first)
    current_path = Path(view.session.image_path)
    legacy_path = tmp_path / 'old-project-name' / current_path.name
    saved.image_path = str(legacy_path)
    relocated = manual.save_manual_version(view.project, saved, mapping, name='V10 relocated')
    original_resolver = storage_paths.resolve_storage_path
    monkeypatch.setattr(storage_paths, 'resolve_storage_path', lambda value, *args, **kwargs:
        current_path if Path(value) == legacy_path else original_resolver(value, *args, **kwargs))
    assert legacy_path.resolve() != current_path.resolve()
    view.versions = view._read_versions(view.session.run_id)
    view._refresh_versions()
    hold, entered = Event(), Event()
    def read(path, profile):
        entered.set()
        hold.wait(5)
        return packet(path)
    view._rig_photo_loader.reader = read
    files = {path:path.read_bytes() for path in Path(relocated['folder']).iterdir() if path.is_file()}
    try:
        view.set_workflow_scope('layers')
        _select(view, relocated)
        qtbot.waitUntil(entered.is_set)
        qtbot.waitUntil(lambda: view.run_preview._version_thread is None
            and view.run_preview.last_result is not None
            and view.run_preview.last_result.source_name == 'V10 relocated')
        assert view.run_preview._has_saved_result()
        assert view.run_preview._last_session.image_path == str(legacy_path)
        hold.set()
        qtbot.waitUntil(lambda: view._rig_photo_packet is not None and view.run_preview.canvas.preview is not None)
        assert not view.run_preview.canvas.pixmap_item.pixmap().isNull()
        assert 'Saved filled-map preview' in view.run_preview.note.text()
        assert not view.canvas.paint_enabled and view._display_only_preview
        assert {path:path.read_bytes() for path in files} == files
    finally:
        hold.set()
        qtbot.waitUntil(lambda: not view._rig_photo_loader.busy())


def test_cached_saved_raw_snapshot_repaints_after_navigation_clears_canvas(raw_view, qtbot):
    view = raw_view
    view._rig_photo_loader.reader = lambda path, profile: packet(path)
    view.set_workflow_scope('layers')
    record = _saved_manual(view, qtbot, 'Completed Setup snapshot')
    preview = view.run_preview
    qtbot.waitUntil(lambda: preview._version_thread is None and preview.last_result is not None
        and view._rig_photo_packet is not None)
    before = deepcopy(view.session.to_dict())
    files = {path: path.read_bytes() for path in Path(record['folder']).iterdir() if path.is_file()}
    # The same immutable version key remains selected after another page owns
    # the photo. Reopening it must repaint, not merely update its status line.
    preview.canvas.clear_photo()
    assert preview.canvas.preview is None
    preview._show_reference_version(record)
    assert preview.canvas.preview is not None
    assert not preview.canvas.pixmap_item.pixmap().isNull()
    assert view._display_only_preview and not view.canvas.paint_enabled
    view._invalidate_rig_photo()
    preview.canvas.clear_photo()
    preview._show_reference_version(record)
    qtbot.waitUntil(lambda: view._rig_photo_packet is not None and preview.canvas.preview is not None)
    assert not view.canvas.paint_enabled
    preview._last_session.image_stamp = None
    preview._show_reference_version(record)
    qtbot.waitUntil(lambda: preview._version_thread is None and preview._has_saved_result())
    assert preview.canvas.preview is not None
    view.set_workflow_scope('rig_frame')
    qtbot.waitUntil(lambda: view._rig_photo_packet is not None)
    view.set_workflow_scope('layers')
    qtbot.waitUntil(lambda: preview._version_thread is None and preview.canvas.preview is not None)
    assert preview.stack.currentWidget() is preview.canvas
    assert not preview.canvas.pixmap_item.pixmap().isNull()
    view.set_workflow_scope('apply')
    view.set_workflow_scope('layers')
    qtbot.waitUntil(lambda: preview.canvas.preview is not None and not preview.busy())
    assert preview.stack.currentWidget() is preview.canvas and view._display_only_preview
    assert view.session.to_dict() == before
    assert {path: path.read_bytes() for path in files} == files


def test_analysis_box_scope_reuses_editor_without_unlocking_setup(context, qtbot):
    view, controller = context.view, context.view.analysis_boxes
    qtbot.waitUntil(lambda: not controller.busy())
    before = deepcopy(view.session.to_dict())
    view.set_workflow_scope('analysis_boxes')
    assert view.step_analysis_boxes.isVisibleTo(view)
    assert controller.widget.isVisibleTo(view)
    assert controller.widget.parentWidget() is view.analysis_boxes_host
    assert all(section.isHidden() for section in (view.step_frame, view.step_injection,
        view.step_point_review, view.step_paint, view.step_facies, view.step_propose, view.step_use_map))
    assert view.run_preview.stack.currentIndex() == 0 and view.canvas.preview is view._reference_token
    assert not view.canvas.paint_enabled and not view.rig_panel.isEnabled()
    assert controller.draw_button.isEnabled(), controller.status.text()
    view.set_tool('polygon')
    assert view.canvas.tool == 'pan' and not view.canvas.paint_enabled
    assert view.session.to_dict() == before
    view.set_workflow_scope('rig_frame')
    assert view.step_analysis_boxes.isHidden() and not controller.widget.isVisibleTo(view)
    assert view.rig_panel.isEnabled()
    view.set_workflow_scope('layers')
    assert view.step_analysis_boxes.isHidden() and not controller.widget.isVisibleTo(view)
    assert view.step_paint.isVisibleTo(view)
