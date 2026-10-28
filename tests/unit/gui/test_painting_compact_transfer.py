"""Compact series controls use only the active review's actual run state."""
from copy import deepcopy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from darsia.fluidflower import painting_chart_review_ui as chart_ui
from darsia.fluidflower.painting_chart_review_ui import ChartReviewDialog
from .test_painting_version_selection import settings, make_view, view, _ready  # noqa: F401
from .test_painting_manual_step4_route import _saved_manual
from .test_painting_chart_review_ui import Session, preview, approve_current


class SavedReviewSession(Session):
    def __init__(self, source):
        super().__init__(source)
        self.excluded_runs = [dict(run_id=run, reason='Saved exclusion for ' + run)
                              for run in ('ac20', 'ac24', 'ac27', 'ac51')]


def saved_review_panel(source, **options):
    options.update(session_factory=SavedReviewSession, preview_reader=preview,
                   review_validator=None, registered_runs=None)
    return ChartReviewDialog(source, **options)


def _row(view, run):
    return next(row for row in range(view.use_map_runs.rowCount())
        if view.use_map_runs.item(row, 0).data(Qt.ItemDataRole.UserRole) == run)


def test_compact_run_list_selects_same_photo_and_tracks_actual_approvals_and_exclusions(view, qtbot, monkeypatch):
    record = _saved_manual(view, qtbot)
    approval = view.approve_version(record)
    _ready(view, qtbot, 'Selected map')
    controller = view.adoption_controller
    context = dict(project=view.project, project_root=str(view.project.root), version=deepcopy(record),
                   approval=approval, frame={}, selection=None)
    monkeypatch.setattr(controller, '_findings_context_snapshot', lambda: context)
    controller._registered_gap_reviews = [dict(project_root=str(view.project.root), binding=dict(path='fixture'))]
    monkeypatch.setattr(chart_ui, 'ChartReviewDialog', saved_review_panel)
    windows = {widget for widget in QApplication.topLevelWidgets() if widget.isVisible()}
    before = deepcopy(view.session.to_dict()), deepcopy(view.map_approvals)
    view.set_workflow_scope('apply')
    qtbot.waitUntil(lambda: controller.chart_dialog is not None and not controller.chart_dialog.busy()
                   and controller.chart_dialog.proposal is not None, timeout=5000)
    review = controller.chart_dialog
    assert review.runs.isHidden() and review.run_title.isVisibleTo(view)
    assert review.counter.isHidden() and review.excluded.isHidden()
    assert view._steps_layout.stretch(view._steps_layout.indexOf(view.step_use_map)) == 1
    view.resize(1400, 1250)
    QApplication.processEvents()
    assert view.use_map_runs.height() > 320
    assert not hasattr(view, 'version_snapshot_table')
    assert view.step_use_map.content.layout().itemAt(0).widget() is view.use_map_version
    assert view.use_map_version.currentData() == record['folder']
    assert view.use_map_runs.rowCount() == len(review.expected_runs) + 4
    assert view.recorded_review_button.isVisibleTo(view)
    first = review.runs.currentData()
    assert view.use_map_runs.item(_row(view, first), 1).text() == 'Waiting for approval'
    approve_current(qtbot, review)
    assert view.use_map_runs.item(_row(view, first), 1).text() == 'Approved'
    assert review.run_title.text() == first.upper() + ' · Approved'
    saved_approvals = deepcopy(review.approvals)
    second = review.runs.itemData(1)
    view.use_map_runs.selectRow(_row(view, second))
    qtbot.waitUntil(lambda: not review.busy() and review.proposal is not None)
    assert review.runs.currentData() == second and review.packet['run_id'] == second
    review.next.click()
    qtbot.waitUntil(lambda: not review.busy() and review.proposal is not None)
    selected_row = view.use_map_runs.selectionModel().selectedRows()[0].row()
    assert view.use_map_runs.item(selected_row, 0).data(Qt.ItemDataRole.UserRole) == review.runs.currentData()
    view.use_map_runs.selectRow(_row(view, 'ac20'))
    assert review.photo.isHidden() and review.runs.currentIndex() == -1
    assert review.excluded_notice.isVisibleTo(view)
    assert 'Saved exclusion for ac20' in review.excluded_notice.text()
    assert 'No photograph' in review.excluded_notice.text()
    assert review.proposal is None and not review.approve.isEnabled() and not review.right.isEnabled()
    assert review.approvals == saved_approvals
    view.use_map_runs.selectRow(_row(view, first))
    qtbot.waitUntil(lambda: not review.busy() and review.proposal is not None)
    assert review.packet['run_id'] == first and not review.photo.isHidden()
    assert review.sand_valid.isChecked() and review.chart_valid.isChecked()
    review.right.setValue(1.)
    assert view.use_map_runs.item(_row(view, first), 1).text() == 'Waiting for approval'
    assert not review.reuse.isEnabled()
    qtbot.waitUntil(lambda: not review.busy() and review.proposal is not None)
    assert before == (view.session.to_dict(), view.map_approvals)
    assert {widget for widget in QApplication.topLevelWidgets() if widget.isVisible()} == windows
    # A linked series' active chooser is authoritative: archived exclusions
    # must not reappear in its transfer list.
    monkeypatch.setattr(view, '_series_root', lambda: view.project.root)
    blocked = view.run_preview.runs.blockSignals(True)
    view.run_preview.runs.clear()
    for run in sorted(review.expected_runs):
        view.run_preview.runs.addItem(run.upper(), run)
    view.run_preview.runs.blockSignals(blocked)
    view._refresh_transfer_runs()
    listed = {view.use_map_runs.item(row, 0).data(Qt.ItemDataRole.UserRole)
              for row in range(view.use_map_runs.rowCount())}
    assert listed == review.expected_runs
    view.set_workflow_scope('layers')
    assert view.use_map_runs.maximumHeight() == 320


def test_apply_without_matching_review_keeps_photo_and_preparation_controls_together(view, qtbot, monkeypatch):
    from darsia.fluidflower import painting_chart_transfer_reader as reader
    record = _saved_manual(view, qtbot)
    approval = view.approve_version(record)
    _ready(view, qtbot, 'Selected map')
    controller = view.adoption_controller
    context = dict(project=view.project, project_root=str(view.project.root), version=deepcopy(record),
                   approval=approval, frame={}, selection=None)
    monkeypatch.setattr(controller, '_findings_context_snapshot', lambda: context)
    controller._registered_gap_reviews = [dict(project_root=str(view.project.root), binding=dict(path='fixture'))]
    def unavailable(*args):
        raise ValueError('No saved transfer review matches this selected map and series geometry.')
    monkeypatch.setattr(reader, 'selected_chart_session', unavailable)
    before = deepcopy(view.map_approvals), deepcopy(view.session.to_dict())
    view.set_workflow_scope('apply')
    qtbot.waitUntil(lambda: controller._chart_failed)
    assert view.run_preview.stack.currentWidget() is view.run_preview.canvas
    assert view._transfer_review_prompt.isVisibleTo(view)
    assert 'No saved transfer review matches' in view._transfer_review_message.text()
    assert view._transfer_review_prepare.isEnabled()
    view.run_preview.runs.blockSignals(True)
    view.run_preview.runs.addItem('Other run', 'other')
    view.run_preview.runs.setCurrentIndex(view.run_preview.runs.count()-1)
    view.run_preview.runs.blockSignals(False)
    view._refresh_transfer_runs()
    assert view.use_map_runs.rowCount() == 1
    assert view.use_map_runs.item(0, 0).data(Qt.ItemDataRole.UserRole) == record['run_id']
    assert view.use_map_runs.item(0, 1).text() == 'Reference · not transferred'
    view.set_reference_display_mode(False)
    view._refresh_use_map()
    assert view.step_use_map.isEnabled() and view.use_map_version.isEnabled()
    assert view.place_version_button.isEnabled() and view.chart_review_button.isEnabled()
    assert before == (view.map_approvals, view.session.to_dict())
    view.set_workflow_scope('layers')
    assert not view._transfer_review_active
    assert not view.run_preview.sources.isHidden()


def test_transfer_list_does_not_reuse_another_versions_runs(view, qtbot, monkeypatch):
    from types import SimpleNamespace
    record = _saved_manual(view, qtbot)
    _ready(view, qtbot, 'Selected map')
    controller = view.adoption_controller
    controller.chart_dialog = SimpleNamespace(session=object())
    controller._chart_context = {'version': dict(record, sha256='different-version')}
    view._refresh_transfer_runs()
    assert view.use_map_runs.rowCount() == 1
    assert view.use_map_runs.item(0, 0).data(Qt.ItemDataRole.UserRole) == record['run_id']
    controller.chart_dialog = None
    monkeypatch.setattr(view, '_selected_version', lambda: None)
    view._refresh_transfer_runs()
    assert view.use_map_runs.rowCount() == 0
    assert view.use_map_run_note.text() == 'Select a saved map version.'


def test_transfer_progress_shows_real_stage_counts_and_elapsed_time(view):
    view._start_placement_progress(39)
    assert view.use_map_progress_bar.minimum() == view.use_map_progress_bar.maximum() == 0
    assert '39 runs' in view.use_map_progress.text()
    view._placement_run_progress(dict(phase='transfer', completed=12, total=39, run_id='ac31'))
    assert view.use_map_progress_bar.value() == 12
    assert view.use_map_progress_bar.maximum() == 39
    assert '12 of 39 complete' in view.use_map_progress.text()
    assert 'AC31' in view.use_map_progress.text()
    view._placement_started -= 65
    view._refresh_use_map()
    view._update_placement_progress()
    assert view.use_map_elapsed.text().startswith('Elapsed: 1:')
    assert view.use_map_progress_bar.value() == 12
    view._placement_run_progress(dict(phase='native_review', completed=0, total=39, run_id='ac17'))
    assert 'Preparing run review: 0 of 39' in view.use_map_progress.text()
    assert view.use_map_progress_bar.value() == 0
    view._placement_failed('Read failed')
    assert not view._placement_timer.isActive()
    assert view.use_map_progress_bar.isHidden()
    assert 'Read failed' in view.use_map_progress.text()


def test_prepared_transfer_opens_its_inline_review_without_an_extra_window(view, monkeypatch, tmp_path):
    from darsia.fluidflower import map_placement_view
    opened = []
    monkeypatch.setattr(view.adoption_controller, 'transfer_review_ready', opened.append)
    def separate_window(*args, **kwargs):
        raise AssertionError('Prepared transfer approval belongs in the current photograph area')
    monkeypatch.setattr(map_placement_view, 'MapPlacementDialog', separate_window)
    binding = {'path': str(tmp_path / 'gap_review.json'), 'sha256': 'a' * 64, 'size_bytes': 10}
    view._placement_finished({'folder': str(tmp_path), 'runs': [{'status': 'placed'}], 'gap_review': binding})
    assert opened == [binding]
    assert 'Placed in 1 of 1 runs' in view.use_map_progress.text()


def test_transfer_retirement_uses_retained_review_for_inline_approval(view, monkeypatch, tmp_path):
    from darsia.fluidflower import map_placement_view
    controller = view.adoption_controller
    opened = []
    monkeypatch.setattr(controller, 'open_chart_review', lambda **options: opened.append(options))
    def unrelated_window(*args, **kwargs):
        raise AssertionError('Transfer completion must stay in the inline approval workspace')
    monkeypatch.setattr(controller, 'open_gap_review', unrelated_window)
    monkeypatch.setattr(map_placement_view, 'MapPlacementDialog', unrelated_window)
    binding = {'path': str(tmp_path / 'gap_review.json'), 'sha256': 'a' * 64, 'size_bytes': 10}
    view._workflow_scope = 'apply'
    view._placement_finished({'folder': str(tmp_path), 'runs': [{'status': 'placed'}], 'gap_review': binding})
    view._placement_thread_finished()
    assert opened == [{'automatic': True}]
    assert view._last_gap_review_source == binding
    assert controller._placement_review_source is None


def test_completed_transfer_populates_inline_run_approval_without_another_click(view, qtbot, monkeypatch, tmp_path):
    from darsia.fluidflower import map_placement_view
    record = _saved_manual(view, qtbot)
    approval = view.approve_version(record)
    _ready(view, qtbot, 'Selected map')
    controller = view.adoption_controller
    context = dict(project=view.project, project_root=str(view.project.root), version=deepcopy(record),
                   approval=approval, frame={}, selection=None)
    monkeypatch.setattr(controller, '_findings_context_snapshot', lambda: context)
    monkeypatch.setattr(chart_ui, 'ChartReviewDialog', saved_review_panel)
    def unrelated_window(*args, **kwargs):
        raise AssertionError('No second review window is needed')
    monkeypatch.setattr(map_placement_view, 'MapPlacementDialog', unrelated_window)
    view._workflow_scope = 'apply'
    binding = {'path': str(tmp_path / 'gap_review.json'), 'sha256': 'a' * 64, 'size_bytes': 10}
    view._placement_finished({'folder': str(tmp_path), 'runs': [{'status': 'placed'}], 'gap_review': binding})
    view._placement_thread_finished()
    qtbot.waitUntil(lambda: controller.chart_dialog is not None and not controller.chart_dialog.busy()
                   and controller.chart_dialog.proposal is not None, timeout=5000)
    review = controller.chart_dialog
    assert view.run_preview.stack.currentWidget() is review
    assert review.approve_all.isVisibleTo(view) and review.approve_all.isEnabled()
    assert review.sand_valid.isVisibleTo(view) and review.chart_valid.isVisibleTo(view)
    assert view.use_map_runs.rowCount() == len(review.expected_runs) + len(review.excluded_runs)
    assert not review.approvals and controller.gap_dialog is None


def test_reopened_transfer_finds_nested_review_without_in_memory_binding(view, qtbot, monkeypatch):
    record = _saved_manual(view, qtbot)
    approval = view.approve_version(record)
    _ready(view, qtbot, 'Selected map')
    controller = view.adoption_controller
    context = dict(project=view.project, project_root=str(view.project.root), version=deepcopy(record),
                   approval=approval, frame={}, selection=None)
    monkeypatch.setattr(controller, '_findings_context_snapshot', lambda: context)
    monkeypatch.setattr(chart_ui, 'ChartReviewDialog', saved_review_panel)
    path = view.project.root / 'annotations' / 'map_placement' / 'saved' / 'native_gap_review' / 'gap_review.json'
    path.parent.mkdir(parents=True)
    path.write_text('{}', encoding='utf8')
    assert not controller._registered_gap_reviews
    view.set_workflow_scope('apply')
    qtbot.waitUntil(lambda: controller.chart_dialog is not None and not controller.chart_dialog.busy()
                   and controller.chart_dialog.proposal is not None, timeout=5000)
    assert view.run_preview.stack.currentWidget() is controller.chart_dialog
    assert view.use_map_runs.rowCount() > 1
