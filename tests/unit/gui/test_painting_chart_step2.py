"""The series workflow opens its own saved all-run review without a path picker."""
from copy import deepcopy
import json
import os
from types import SimpleNamespace

import pytest
import toml
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QPushButton, QWidget

from darsia.fluidflower import painting_chart_review_ui as chart_ui
from darsia.fluidflower import painting_chart_transfer_reader as reader
from darsia.fluidflower.coloured_map_ui import NativeColouredMapWorkflow
from darsia.fluidflower.series_geometry import RigFrame

from .test_painting_corrections import view  # noqa: F401
from .test_painting_manual_step4_route import _saved_manual


def test_chart_is_only_step2_action_and_import_starts_workflow(view, qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    workflow = NativeColouredMapWorkflow(window)
    workflow.install(view)
    assert view.steps_scroll.widget().layout().indexOf(view.import_map_start) == 0
    assert view.import_map_start.isAncestorOf(view.import_coloured_map_button)
    assert not view.step_propose.content.isAncestorOf(view.import_coloured_map_button)
    layout = view.step_use_map.content.layout()
    second = next(layout.itemAt(i).layout() for i in range(layout.count())
        if layout.itemAt(i).layout() is not None
        and layout.itemAt(i).layout().indexOf(view.chart_review_button) >= 0)
    assert second.itemAt(0).widget() is view.chart_review_button
    assert layout.itemAt(0).widget() is view.use_map_version
    assert layout.indexOf(view.place_version_button) < layout.indexOf(view.use_map_runs)
    assert layout.indexOf(view.use_map_runs) < layout.indexOf(second)
    buttons = view.step_use_map.content.findChildren(QPushButton)
    assert [button.text() for button in buttons].count('Review the transfer…') == 1
    assert not {'Review gaps and findings…', 'Review chart masks and reuse saved maps…', 'Open other review…'} & {
        button.text() for button in buttons}


def _fake_review(monkeypatch):
    calls = []
    class Review(QWidget):
        approvals_changed = Signal(object)
        reuse_requested = Signal(object)
        review_loaded = Signal()
        def __init__(self, sources, **options):
            super().__init__(options['parent'])
            self._closing = False
            self.approvals = {}
            calls.append((sources, options, self))
        def wait_for_workers(self):
            self._closing = True
        def closeEvent(self, event):
            self.wait_for_workers()
            super().closeEvent(event)
    monkeypatch.setattr(chart_ui, 'ChartReviewDialog', Review)
    def forbidden(*args, **kwargs):
        raise AssertionError('The series review must never open a file picker.')
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', forbidden)
    return calls


def test_findings_action_is_reachable_separately_from_chart_review(view, qtbot, monkeypatch):
    record = _saved_manual(view, qtbot)
    view.approve_version(record)
    opened = []
    monkeypatch.setattr(view.adoption_controller, 'open_latest_gap_review', lambda: opened.append('findings'))
    assert view.gap_review_button is not view.chart_review_button
    assert view.gap_review_button.isEnabled()
    view.gap_review_button.click()
    assert opened == ['findings']


def test_approved_chart_reuse_survives_required_findings_check(view, qtbot, monkeypatch, tmp_path):
    record = _saved_manual(view, qtbot)
    approval = view.approve_version(record)
    selection = SimpleNamespace(index_path=str(tmp_path / 'geometry.json'))
    monkeypatch.setattr(view.run_preview, 'geometry_selection', lambda: selection)
    monkeypatch.setattr(view, 'current_frame', lambda: RigFrame({}))
    calls = _fake_review(monkeypatch)
    controller = view.adoption_controller
    view.chart_review_button.click()
    request = dict(review=dict(map_version=record, geometry_index=dict(path=selection.index_path),
        source=dict(painting_facies=dict(approval=approval))), approvals={'sample': {'mask': 'approved'}})
    started, opened = [], []
    monkeypatch.setattr(controller, 'start', lambda *a, **kw: started.append(kw) or False)
    monkeypatch.setattr(controller, 'findings_block_reason', lambda: 'Review the findings first.')
    monkeypatch.setattr(controller, 'open_latest_gap_review', lambda: opened.append('findings'))
    calls[0][2].reuse_requested.emit(request)
    assert opened == ['findings']
    assert started[0]['local_reuse'] == request
    controller.context = dict(selection=selection)
    controller.gap_dialog = SimpleNamespace(busy=lambda: False, _acquisition_actions=[],
        _warning_actions={}, review={}, close=lambda: None)
    monkeypatch.setattr(controller, '_allow_findings_check', lambda: True)
    monkeypatch.setattr(controller, '_current', lambda: True)
    monkeypatch.setattr(controller, '_findings_input_bindings', lambda _review: {'target_bindings': []})
    controller._recheck_gaps({'areas': 'confirmed'})
    assert started[-1]['local_reuse'] == request
    controller.gap_dialog = None


def test_step2_opens_selected_saved_map_directly_without_picker(view, qtbot, monkeypatch, tmp_path):
    record = _saved_manual(view, qtbot)
    view.approve_version(record)
    selection = SimpleNamespace(index_path=str(tmp_path / 'geometry.json'))
    monkeypatch.setattr(view.run_preview, 'geometry_selection', lambda: selection)
    monkeypatch.setattr(view, 'current_frame', lambda: RigFrame({}))
    calls = _fake_review(monkeypatch)
    windows = {widget for widget in QApplication.topLevelWidgets() if widget.isVisible()}
    view.chart_review_button.click()
    assert len(calls) == 1 and calls[0][2].isVisible(), view.use_map_progress.text()
    assert calls[0][1]['registered_runs'] == view._series_runs()
    assert view.adoption_controller._chart_context['version']['folder'] == record['folder']
    assert view._adoption_thread is None
    assert calls[0][2].parentWidget() is view.run_preview.stack
    assert not calls[0][2].isWindow()
    assert {widget for widget in QApplication.topLevelWidgets() if widget.isVisible()} == windows
    assert view.run_preview.stack.currentWidget() is calls[0][2]
    assert view.run_preview.runs.isHidden() and view.run_preview.sources.isHidden()


@pytest.mark.parametrize('new_transfer', [False, True])
def test_transfer_review_binding_retires_only_a_different_chart_session(
        view, qtbot, monkeypatch, tmp_path, new_transfer):
    from darsia.fluidflower.painting_chart_transfer import RUNS
    record = _saved_manual(view, qtbot)
    view.approve_version(record)
    selection = SimpleNamespace(index_path=str(tmp_path / 'geometry.json'))
    monkeypatch.setattr(view.run_preview, 'geometry_selection', lambda: selection)
    monkeypatch.setattr(view, 'current_frame', lambda: RigFrame({}))
    monkeypatch.setattr(view, '_refresh_transfer_runs', lambda *_args: None)
    calls = _fake_review(monkeypatch)
    controller = view.adoption_controller
    monkeypatch.setattr(controller, '_chart_readers_busy', lambda: False)
    view._workflow_scope = 'apply'
    old_path = tmp_path / 'old-transfer/gap_review.json'
    old_path.parent.mkdir()
    old_path.write_text('{"original":"transfer"}', encoding='utf8')
    old_binding = reader._bound(old_path)
    old_review = dict(review_binding=old_binding, map_version=record,
        geometry_index=dict(path=selection.index_path))
    view._last_gap_review_source = deepcopy(old_review)
    old = controller.open_chart_review(automatic=True)
    old.session = SimpleNamespace(review=deepcopy(old_review))
    old.approvals = {run: {'historical fixture decision': run} for run in RUNS}
    old.approvals_changed.emit(old.approvals)
    pending = (controller._findings_context_snapshot(), {'source': old_binding})
    controller._pending_local_reuse = pending
    sidecar = old_path.parent / 'chart_review_decisions_v3.json'
    sidecar.write_text('{"historical_decisions":39}', encoding='utf8')
    retained = sidecar.read_bytes()
    new_path = tmp_path / 'new-transfer/gap_review.json'
    new_path.parent.mkdir()
    new_path.write_text('{"new":"transfer"}', encoding='utf8')
    source = dict(old_review, review_binding=reader._bound(new_path)) if new_transfer else deepcopy(old_review)
    before = controller._chart_context
    old_generation = controller._chart_generation
    controller.transfer_review_ready(source)
    if new_transfer:
        assert old._closing and not old.isVisible()
        assert controller.chart_dialog is None and controller._chart_context is None
        assert controller._chart_approvals == {} and controller._pending_local_reuse is None
        # A queued result from the retired reader must not restore old state.
        old.approvals_changed.emit(old.approvals)
        old.reuse_requested.emit({'must not be reused': old_binding})
        assert controller._chart_approvals == {}
    else:
        assert not old._closing
        assert controller.chart_dialog is old and controller._chart_context == before
        assert controller._chart_approvals == old.approvals
        assert controller._pending_local_reuse == pending
    controller.placement_retired()
    current = controller.chart_dialog
    assert current is not None
    if new_transfer:
        assert current is not old and current.approvals == {} and len(calls) == 2
        assert source in calls[-1][0]
        assert controller._chart_review_binding == source['review_binding']
        controller.chart_review_unavailable.emit((old_generation, before), 'Retired reader failed late')
        assert not controller._chart_failed and current.isVisible()
    else:
        assert current is old and len(calls) == 1 and len(current.approvals) == 39
    assert sidecar.read_bytes() == retained
    assert not (new_path.parent / 'chart_review_decisions_v3.json').exists()


def test_step2_preselects_approved_map_when_no_row_is_selected(view, qtbot, monkeypatch, tmp_path):
    record = _saved_manual(view, qtbot)
    view.approve_version(record)
    view.versions_table.setCurrentCell(-1, -1)
    view.versions_table.clearSelection()
    assert view._selected_version() is None
    monkeypatch.setattr(view.run_preview, 'geometry_selection',
                        lambda: SimpleNamespace(index_path=str(tmp_path / 'geometry.json')))
    monkeypatch.setattr(view, 'current_frame', lambda: RigFrame({}))
    calls = _fake_review(monkeypatch)
    view._refresh_use_map()
    assert view.chart_review_button.isEnabled()
    view.chart_review_button.click()
    assert len(calls) == 1, view.use_map_progress.text()
    assert view._selected_version()['folder'] == record['folder']


def test_real_transfer_panel_shares_main_photo_area_and_returns_to_mark_layers(view, qtbot):
    from .test_painting_chart_review_ui import Session, preview, approve_current
    windows = {widget for widget in QApplication.topLevelWidgets() if widget.isVisible()}
    previous = view.run_preview.stack.currentWidget()
    panel = chart_ui.ChartReviewDialog('fixture', actor='Reviewer', parent=view.run_preview.stack,
        session_factory=Session, preview_reader=preview)
    view.show_transfer_review(panel)
    qtbot.waitUntil(lambda: not panel.busy() and panel.proposal is not None)
    assert panel.parentWidget() is view.run_preview.stack and not panel.isWindow()
    assert panel.window() is view.window()
    assert {widget for widget in QApplication.topLevelWidgets() if widget.isVisible()} == windows
    assert view.run_preview.stack.currentWidget() is panel
    assert view.run_preview.runs.isHidden() and view.run_preview.sources.isHidden()
    approve_current(qtbot, panel)
    approvals = deepcopy(panel.approvals)
    view.run_preview.stack.setCurrentIndex(0)
    assert view.run_preview.stack.currentWidget() is panel
    view.step_paint.toggle.click()
    assert view.run_preview.stack.currentWidget() is previous
    assert not view.run_preview.runs.isHidden() and not view.run_preview.sources.isHidden()
    view.show_transfer_review(panel)
    assert panel.approvals == approvals and panel.sand_valid.isChecked() and panel.chart_valid.isChecked()
    view.show_mark_layers()
    panel.close()


def test_real_chart_dialog_loads_aliased_saved_index_and_keeps_exact_map_approval(view, qtbot, monkeypatch, tmp_path):
    from .test_painting_chart_review_ui import Session, preview
    from darsia.fluidflower.storage_paths import refresh_storage_paths

    monkeypatch.setenv('DARSIA_SERIES_REGISTRY', str(tmp_path / 'registry.toml'))
    record = _saved_manual(view, qtbot)
    approval = view.approve_version(record)
    run = record['run_id']
    series = tmp_path / 'series'
    old = series / 'geometry_facies'
    index = old / 'frames/index.json'
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps(dict(format='darsia-measured-run-geometry-v1', runs={run: {}})), encoding='utf-8')
    bound = reader._bound(index)
    moved = tmp_path / 'Results/setup/Geometry'
    moved.parent.mkdir(parents=True)
    old.rename(moved)
    (series / 'series.toml').write_text(toml.dumps(dict(schema_version=1, series_id='sample',
        storage=dict(path_aliases={str(old): str(moved)}))), encoding='utf-8')
    refresh_storage_paths(series)
    selection = SimpleNamespace(index_path=str(moved / 'frames/index.json'),
        sha256=bound['sha256'], size_bytes=bound['size_bytes'])
    monkeypatch.setattr(view.run_preview, 'geometry_selection', lambda: selection)
    monkeypatch.setattr(view, 'current_frame', lambda: RigFrame({}))
    monkeypatch.setattr(view, '_series_runs', lambda: [run])
    saved = dict(map_version=dict(record, approval=approval), geometry_index=bound,
        source=dict(reference_run=run, painting_facies=dict(approval=approval)), runs=[dict(run_id=run)])

    class RetainedSession(Session):
        def __init__(self, source):
            super().__init__(source)
            self.review = deepcopy(saved)

    original_dialog = chart_ui.ChartReviewDialog
    monkeypatch.setattr(reader, 'selected_chart_session', lambda *args: RetainedSession('saved review'))
    monkeypatch.setattr(chart_ui, 'ChartReviewDialog', lambda source, **options:
        original_dialog(source, preview_reader=preview, **options))
    view.adoption_controller._chart_opening_note('Loading the saved transfer review…')
    view.use_map_progress.setText('Transferring saved map to the next run…')
    panel = view.adoption_controller.open_chart_review()
    assert panel is not None
    qtbot.waitUntil(lambda: not panel.busy() and panel.session is not None, timeout=5000)
    assert view.use_map_progress.text() == 'Transferring saved map to the next run…'
    assert panel.session.review == saved
    assert panel.runs.count() == 1 and panel.runs.currentData() == run
    changed = deepcopy(saved)
    changed['source']['painting_facies']['approval']['reviewer'] = 'different reviewer'
    with pytest.raises(ValueError, match='selected approved map'):
        panel._validator(changed)
    changed = deepcopy(saved)
    changed['geometry_index']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='selected approved map'):
        panel._validator(changed)
    panel.close()


@pytest.mark.parametrize('layout', ('map_adoption/gap_review.json',
                                    'map_placement/native_gap_review/gap_review.json'))
def test_saved_review_discovery_needs_no_index_and_matches_version_and_geometry(tmp_path, monkeypatch, layout):
    record = dict(folder=str(tmp_path / 'selected'), sha256='a' * 64)
    selection = SimpleNamespace(index_path=str(tmp_path / 'geometry.json'), sha256='b' * 64, size_bytes=100)
    selected = dict(map_version=record, geometry_index=dict(path=selection.index_path,
        sha256=selection.sha256, size_bytes=selection.size_bytes))
    paths = []
    for number, document in enumerate((selected,
            dict(selected, map_version=dict(record, folder=str(tmp_path / 'other'))),
            dict(selected, geometry_index=dict(selected['geometry_index'], sha256='c' * 64)))):
        kind, relative = layout.split('/', 1)
        path = tmp_path / 'annotations' / kind / str(number) / relative
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(document), encoding='utf8')
        os.utime(path, (100 + number, 100 + number))
        paths.append(path)
    from darsia.fluidflower import painted_analysis_exclusions as gaps
    def read(path):
        document = json.loads(path.read_text(encoding='utf8')) if hasattr(path, 'read_text') else json.loads(open(path, encoding='utf8').read())
        return dict(document, review_binding=reader._bound(path))
    monkeypatch.setattr(gaps, '_read_scientific_gap_review', read)
    def current_session(review, **options):
        from darsia.fluidflower.painting_chart_transfer import DETECTED_FORMAT
        assert options['proposal_format'] == DETECTED_FORMAT
        return SimpleNamespace(review=deepcopy(review))
    monkeypatch.setattr(reader, 'ChartTransferSession', current_session)
    session = reader.selected_chart_session(tmp_path, record, selection)
    assert session.review['review_binding']['path'] == str(paths[0])
