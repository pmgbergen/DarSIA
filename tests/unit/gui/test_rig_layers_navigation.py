"""Reference review navigation preserves map identity and restores the editor."""
from copy import deepcopy
from pathlib import Path

from PySide6.QtWidgets import QWidget

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import rig_point_review as review
from .test_painting_manual_map import _drawn
from .test_painting_raw_reference_on_u import raw_view, raw_file, profile, renders, settings  # noqa: F401
from .test_rig_point_review_ui import context  # noqa: F401


def complete_review(view, panel, qtbot):
    panel.auto_next.setChecked(False)
    for index in range(panel.runs.rowCount()):
        panel.runs.selectRow(index)
        qtbot.waitUntil(lambda: panel._viewed_identity is not None)
        panel.checked.setChecked(True)
        assert panel.approve.isEnabled()
        panel.approve.click()
    qtbot.waitUntil(lambda: view._reference_point_review_blocker() is None)


def layers_layout(view):
    assert view._workflow_scope == 'layers'
    assert not view._point_review_active and not view._transfer_review_active
    assert not view.step_frame.is_expanded()
    assert view.step_paint.is_expanded()
    assert all(not section.isHidden() for section in (view.step_paint, view.step_facies, view.step_propose))
    assert view.step_point_review.isHidden() and view.rig_photo_toolbar.isHidden()
    assert not hasattr(view, '_rig_photo_visibility')
    assert view.run_preview.sources.isVisible() and view.run_preview.sources.isEnabled()


def current_editor(view, qtbot):
    view.run_preview.sources.setCurrentIndex(0)
    qtbot.waitUntil(lambda: view.run_preview.stack.currentIndex() == 0 and view.canvas.paint_enabled)
    assert not getattr(view, '_display_only_preview', False)
    assert all(group.isVisible() for group in view._painting_photo_groups)
    assert view.tool_buttons['polygon'].isEnabled()
    view.tool_buttons['polygon'].click()
    assert view.canvas.tool == 'polygon'
    assert view.layer_table.isEnabled() and view.facies_table.isEnabled()


def saved_map(view, qtbot):
    current_editor(view, qtbot)
    view.add_polygon(lp.Polygon(2, [(0., 0.), (0., 80.), (60., 80.), (60., 0.)]))
    view.map_shown.setCurrentIndex(view.map_shown.findText('Manual map'))
    _drawn(view, qtbot)
    record = view.save_version('V9')
    assert record is not None
    view.versions_table.selectRow(0)
    qtbot.waitUntil(lambda: view.run_preview._version_thread is None
        and view.run_preview._last_session is not None and view.run_preview.last_result.source_name == 'V9')
    return record


def assert_saved(view, record, qtbot):
    qtbot.waitUntil(lambda: view.run_preview._version_thread is None
        and view.run_preview._reference_version_key is not None)
    assert view._selected_version()['folder'] == record['folder']
    assert view.run_preview._reference_version()['folder'] == record['folder']
    assert view.run_preview.stack.currentWidget() is view.run_preview.canvas
    assert view._display_only_preview
    assert view.restore_version_button.isEnabled()


def test_completed_review_returns_to_layers_twice_without_hiding_current_map_tools(context, qtbot):
    view, panel, _ = context
    complete_review(view, panel, qtbot)
    approved = deepcopy(review.load_review(view.project))
    view.close_reference_point_review()
    view.set_workflow_scope('layers')
    layers_layout(view)
    record = saved_map(view, qtbot)
    version_files = {path: path.read_bytes() for path in Path(record['folder']).iterdir() if path.is_file()}
    working = deepcopy(view.session.to_dict())
    for cycle in range(2):
        view.open_reference_point_review()
        qtbot.waitUntil(lambda: panel._viewed_identity is not None)
        if cycle == 0:
            view.close_reference_point_review()
        view.set_workflow_scope('layers')
        layers_layout(view)
        assert_saved(view, record, qtbot)
        current_editor(view, qtbot)
        assert view._selected_version()['folder'] == record['folder']
        assert view.session.to_dict() == working
        assert review.load_review(view.project) == approved
        assert all(path.read_bytes() == content for path, content in version_files.items())


def test_background_alignment_restore_and_nested_transfer_preserve_layers_route(context, qtbot):
    view, panel, _ = context
    complete_review(view, panel, qtbot)
    approved = deepcopy(review.load_review(view.project))
    view.set_workflow_scope('layers')
    record = saved_map(view, qtbot)
    # Restore the completed selection as on series load, with no visible review.
    old_panel = panel
    old_panel.wait_for_workers()
    view._point_review_panel = None
    view._reviewed_point_selection = None
    view._restore_reviewed_point_selection()
    panel = view._point_review_panel
    qtbot.waitUntil(lambda: view.reviewed_point_selection() is not None)
    layers_layout(view)
    assert panel.isHidden() and view.run_preview.stack.currentWidget() is view.run_preview.canvas
    assert_saved(view, record, qtbot)
    current_editor(view, qtbot)
    for _ in range(2):
        transfer = QWidget(view)
        view.show_transfer_review(transfer)
        view.set_workflow_scope('rig_frame')
        view.open_reference_point_review()
        qtbot.waitUntil(lambda: panel._viewed_identity is not None)
        view.set_workflow_scope('layers')
        layers_layout(view)
        assert_saved(view, record, qtbot)
        current_editor(view, qtbot)
    assert review.load_review(view.project) == approved


def test_immediate_native_apply_waits_for_deferred_approved_point_geometry(
        context, qtbot, monkeypatch):
    from threading import Event
    from types import SimpleNamespace

    from PySide6.QtCore import QTimer
    from darsia.fluidflower import rig_point_review_ui as point_ui
    from darsia.fluidflower import painting_chart_transfer_reader as chart_reader
    from darsia.fluidflower.native_darsia_app import NativeFluidFlowerMainWindow
    from .test_painting_chart_step2 import _fake_review

    view, panel, _ = context
    complete_review(view, panel, qtbot)
    view.set_workflow_scope('layers')
    record = saved_map(view, qtbot)
    view.approve_version(record)
    panel.wait_for_workers()
    qtbot.waitUntil(lambda: not view.busy() and not view.run_preview.busy())
    expected = view.reviewed_point_selection()
    assert expected is not None
    approved = deepcopy(review.load_review(view.project))
    monkeypatch.setattr(view, '_series_runs', lambda: list(approved['runs']))
    retained = {path: path.read_bytes() for path in Path(record['folder']).iterdir() if path.is_file()}
    retained[Path(expected.index_path)] = Path(expected.index_path).read_bytes()

    entered, release = Event(), Event()
    original_align = point_ui._Align.run

    def held_align(worker):
        entered.set()
        release.wait(10)
        original_align(worker)

    monkeypatch.setattr(point_ui._Align, 'run', held_align)
    calls = _fake_review(monkeypatch)
    monkeypatch.setattr(chart_reader, 'saved_chart_review_paths', lambda root: ['saved review'])
    selections = []
    monkeypatch.setattr(chart_reader, 'selected_chart_session',
        lambda root, version, selection, sources: selections.append(selection) or 'saved session')
    view._point_review_panel = None
    view._reviewed_point_selection = None
    controller = view.adoption_controller
    native = SimpleNamespace(geometry_workspace=SimpleNamespace(simple_view=view),
        _select_row=lambda *_args: None)
    try:
        # Native page construction routes Apply before reload's deferred restore.
        QTimer.singleShot(0, view._restore_reviewed_point_selection)
        NativeFluidFlowerMainWindow._set_geometry_workflow_scope(native, 'export')
        qtbot.waitUntil(entered.is_set)
        qtbot.waitUntil(lambda: not controller._chart_readers_busy())
        assert view._reference_point_review_blocker().startswith('Preparing approved point geometry')
        assert controller._automatic_chart_request is not None
        assert calls == [] and selections == []
        opening_message = view.use_map_progress.text()
        assert opening_message
        release.set()
        qtbot.waitUntil(lambda: len(calls) == 1)
        assert view._workflow_scope == 'apply'
        assert view.reviewed_point_selection() == expected
        assert controller._chart_context['selection'] == expected
        assert controller._automatic_chart_request is None
        assert calls[0][1]['session_factory'](calls[0][0]) == 'saved session'
        assert selections == [expected]
        calls[0][2].review_loaded.emit()
        assert not view.use_map_progress.text()
        view.use_map_progress.setText('Transferring saved map to the next run…')
        calls[0][2].review_loaded.emit()
        assert view.use_map_progress.text() == 'Transferring saved map to the next run…'
        assert view.run_preview.stack.currentWidget() is calls[0][2]
        assert review.load_review(view.project) == approved
        assert all(path.read_bytes() == content for path, content in retained.items())
    finally:
        release.set()
        if view._point_review_panel is not None:
            view._point_review_panel.wait_for_workers()
