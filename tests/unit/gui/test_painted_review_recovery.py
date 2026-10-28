"""Retain authenticated failures and distinguish pending Prepare from display Check."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QMessageBox

from darsia.fluidflower import map_adoption_view as adoption_ui
from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import painted_review_comparison as display
from darsia.fluidflower import painted_adoption_table as table

from .test_painting_gap_review_ui import _controller, _review
from .test_painting_corrections import view, _show, _proposal  # noqa: F401
from .test_painting_map_placement import _index
from ..test_geometry_registered_export_apply import registered_setup  # noqa: F401
from ..test_painted_adoption_table import imported, synthetic_payload_admission  # noqa: F401
from ..test_painted_display_findings_backend import display_provider  # noqa: F401


@pytest.mark.parametrize('error_type', [ValueError, ImportError])
def test_preference_normalization_failure_still_runs_real_check_and_keeps_original_issues(qtbot, monkeypatch, tmp_path, error_type):
    candidate = dict(provenance=dict(preparation={'path': 'synthetic'}))
    original = [dict(severity='warning', code='original-frame', message='Actual original frame issue'),
                dict(severity='error', code='original-error', message='Actual original error')]
    monkeypatch.setattr(adoption_ui.ma, 'adoption_candidate', lambda *_args, **_kwargs: (candidate, {}))
    monkeypatch.setattr(table, '_checked', lambda *_args, **_kwargs: dict(gap_review={'path': 'synthetic-review'}))
    def missing(*_args, **_kwargs):
        raise error_type('current display source unavailable')
    monkeypatch.setattr(display, 'read_review', missing)
    checks = []
    def check(value, **options):
        checks.append((deepcopy(value), deepcopy(options)))
        return dict(issues=deepcopy(original), summary=dict(error_count=1, warning_count=1))
    monkeypatch.setattr(geometry, 'validate_geometry_config', check)
    worker = adoption_ui._CheckWorker({}, tmp_path, None, display_preferences={})
    results, failures = [], []
    worker.finished.connect(results.append); worker.failed.connect(failures.append); worker.run()
    assert not failures and checks == [(candidate, {})]
    report = results[0]['validation_report']
    assert report['issues'][:2] == original and report['summary']['error_count'] == 2
    assert report['issues'][-1]['severity'] == 'error' and error_type.__name__ in report['issues'][-1]['message']
    assert len(results[0]['errors']) == 2


@pytest.mark.parametrize('route', ['analysis-exclusion', 'target-product'])
def test_real_candidate_precheck_keeps_typed_original_findings_without_claiming_validation(
        qtbot, registered_setup, display_provider, monkeypatch, tmp_path, route):
    from darsia.fluidflower import map_adoption as adoption
    from darsia.fluidflower import painted_analysis_exclusions as core
    from darsia.fluidflower import painted_target_integration as target
    from ..test_painted_analysis_exclusions import _make_gap_case, _approve, _fresh, _paint_component, ACTION
    original_prepare = adoption.prepare_adoption
    def with_display(*args, **options):
        return original_prepare(*args, **dict(options, display_context=display_provider.context))
    monkeypatch.setattr(adoption, 'prepare_adoption', with_display)
    case = _make_gap_case(registered_setup, tmp_path, monkeypatch)
    if route == 'analysis-exclusion':
        prepared = _fresh(case, _approve(case.review))
        owner, name = core, 'exclusion_sources'
    else:
        _, _, binding = _paint_component(case)
        prepared = case.actual(case.root / 'fresh-target', *case.captured['args'][1:],
            **dict(case.captured['kwargs'], target_corrections=[binding], review_findings=True))
        owner, name = target, 'effective_product'
    declaration = core.make_facies_source_acquisition(prepared, 'indicator', **ACTION)
    original_issues = deepcopy(case.review['issues'])
    assert original_issues, 'The original tiny source must have genuine frame findings'
    real_precheck = getattr(owner, name)
    calls = []
    def precheck(*args, **options):
        calls.append(name)
        return real_precheck(*args, **options)
    monkeypatch.setattr(owner, name, precheck)
    def stale(*_args, **_kwargs):
        raise ValueError('Actual bound display metadata changed')
    monkeypatch.setattr(display_provider.module, 'validate_display_context', stale)
    monkeypatch.setattr(geometry, 'validate_geometry_config', lambda *_args, **_kwargs: pytest.fail('No candidate exists for validation'))
    worker = adoption_ui._CheckWorker(case.data.revision, Path(prepared['folder']), None,
        facies_source_acquisition=declaration)
    results, failures = [], []
    worker.finished.connect(results.append); worker.failed.connect(failures.append); worker.run()
    assert calls and not failures and len(results) == 1
    result = results[0]
    assert result['candidate'] is None and result['validation_not_executed'] is True
    assert result['validation_report']['issues'][:-1] == original_issues
    assert result['validation_report']['issues'][-1]['severity'] == 'error' and result['errors']
    host, _, controller = _controller(qtbot, monkeypatch)
    findings = controller.open_gap_review(_review())
    qtbot.waitUntil(lambda: not findings.busy(), timeout=5000)
    findings.set_check_result(None, result['validation_report'])
    assert not findings.adopt.isEnabled() and findings._report['validation_not_executed'] is True


def test_generic_scientific_candidate_error_keeps_original_failed_boundary(qtbot, monkeypatch, tmp_path):
    def invalid(*_args, **_kwargs):
        raise ValueError('Scientific source SHA mismatch')
    monkeypatch.setattr(adoption_ui.ma, 'adoption_candidate', invalid)
    worker = adoption_ui._CheckWorker({}, tmp_path, None)
    results, failures = [], []
    worker.finished.connect(results.append); worker.failed.connect(failures.append); worker.run()
    assert results == [] and failures == ['ValueError: Scientific source SHA mismatch']


@pytest.mark.parametrize('method,worker_name,thread_name', [
    ('place_version', '_PlacementWorker', '_placement_thread'),
    ('prepare_adoption_of', '_AdoptionWorker', '_adoption_thread')])
def test_native_series_public_default_preserves_legacy_call_without_display_capture(
        view, qtbot, monkeypatch, tmp_path, method, worker_name, thread_name):
    from darsia.fluidflower import series_layer_view as module
    _show(view, _proposal())
    record = view.save_version('synthetic legacy default')
    view.approve_version(record)
    view.project.manifest.setdefault('tools', {})['native_series'] = {'series_root': str(tmp_path)}
    index = _index(tmp_path / 'geometry')
    monkeypatch.setattr(view.run_preview, 'geometry_selection', lambda: SimpleNamespace(index_path=str(index)))
    monkeypatch.setattr(view, '_capture_review_display_context', lambda: pytest.fail('Default V1 route captured display context'))
    monkeypatch.setattr(view, '_refresh_versions', lambda: None)
    monkeypatch.setattr(QMessageBox, 'information', lambda *_args, **_kwargs: None)
    calls = []
    def stop_before_science(worker):
        calls.append(set(worker.options))
        worker.failed.emit('Synthetic entrypoint-boundary stop before any scientific operation')
    monkeypatch.setattr(getattr(module, worker_name), 'run', stop_before_science)
    options = dict(runs=['ac52'])
    options['photo_loader' if method == 'place_version' else 'current_labels'] = lambda *_: None
    assert getattr(view, method)(record, index, **options)
    qtbot.waitUntil(lambda: getattr(view, thread_name) is None, timeout=5000)
    assert len(calls) == 1 and 'display_context' not in calls[0] and 'continue_photo_edges' not in calls[0]


def _prepared():
    review = _review()
    return dict(folder='synthetic-successful-prepare', map_version=deepcopy(review['map_version']),
        reference_run='ac17', painting_facies={}, geometry_index=deepcopy(review['geometry_index']))


@pytest.mark.parametrize('completion', ['normal', 'shutdown'])
def test_pending_target_survives_reopen_then_successful_prepare_allows_preference_only_check(qtbot, monkeypatch, completion):
    from darsia.fluidflower import painted_analysis_exclusions as core
    from .test_findings_before_check import _literal_review
    host, _, controller = _controller(qtbot, monkeypatch)
    requests = []
    controller.catalogue_refresher = requests.append  # Finite worker-result DATA; no scientific replay is claimed.
    first = controller.open_gap_review(_literal_review())
    qtbot.waitUntil(lambda: not first.busy(), timeout=5000)
    assert controller.prepared(_prepared()) and not controller._preparation_requires_refresh()

    def install_last(dialog, review):
        """Use the original pure catalogue admission with structurally exact fixture DATA."""
        request = requests[-1]
        fresh = deepcopy(review)
        fresh['review_binding'] = dict(path=f"C:/synthetic/residual-{request['generation']}.json",
            sha256=f"{request['generation']:064x}", size_bytes=101)
        assert controller.install_findings_result(dict(generation=request['generation'], review=fresh,
            report=dict(issues=[], input_bindings=deepcopy(request['input_bindings']))))
        qtbot.waitUntil(lambda: not dialog.busy(), timeout=5000)
        return fresh

    def refresh_display(dialog):
        count = len(requests)
        assert dialog._display_preferences_dirty and not dialog.recheck.isEnabled()
        dialog.refresh_findings.click()
        assert len(requests) == count + 1
        return install_last(dialog, dialog.review)

    checks, source_choices = [], []
    class Checked:
        candidate = checked_candidate = review_resolutions = {'old': 'checked'}
        def busy(self): return False
        def close(self): pass
        def wait_for_workers(self): pass
        def _update_buttons(self): pass
        def set_display_preferences(self, _value):
            self.candidate = self.checked_candidate = self.review_resolutions = None
            return True
        def check(self):
            checks.append('check'); return True
        def choose_facies_source_state(self, action):
            source_choices.append(deepcopy(action))
            return self.check()
    controller.dialog = Checked()
    binding = dict(version={'path': 'exact-human-target-version'}, approval={'path': 'exact-human-approval'})
    result = dict(run_id='ac18', component_id='c' * 64, binding=binding)
    retained_approval = dict(format='darsia-painted-gap-exclusion-approval-v1', runs=[
        dict(run_id=run['run_id'], component_id=component['component_id'],
             review_row_sha256=component['review_row_sha256'],
             actor='Synthetic reviewer', approved_utc='2026-10-22T10:00:00Z',
             decision_ref='N438', classification='Vann over sanden', other_text='', reason=reason)
        for run, reason in zip(first.review['runs'][:2], ['old affected area', 'keep the other area'])
        for component in run['components']])
    first.approval = deepcopy(retained_approval)
    controller._area_approval = deepcopy(retained_approval)
    first._refresh_rows()
    source_row = next(index for index, row in enumerate(first._rows) if row['kind'] == 'source_state')
    first.runs.setCurrentCell(source_row, 0)
    qtbot.waitUntil(lambda: not first.busy(), timeout=5000)
    first.source_state.setCurrentIndex(first.source_state.findData('unknown'))
    first.reason.setText('Synthetic human reviewed the source water state')
    first.set_state.click()
    qtbot.waitUntil(lambda: bool(requests) and not first.busy(), timeout=5000)
    original = install_last(first, first.review)
    target = SimpleNamespace(run_id='ac18', component_id='c' * 64, binding=None, retired=False,
        review=deepcopy(first.review))
    def finish_explicit_save():
        target.binding = deepcopy(binding)
        target.retired = True
    target.wait_for_workers = finish_explicit_save
    host._target_correction_dialog = target
    generation = controller._preparation_action_generation
    if completion == 'shutdown':
        controller.wait_for_workers()  # Save completes naturally while queued GUI delivery is suspended.
        assert target.retired
    else:
        target.wait_for_workers()
        controller._target_saved(result)
        first.close()
    assert controller._preparation_action_generation == generation + 1
    assert controller._preparation_requires_refresh()
    assert controller.dialog.candidate is controller.dialog.checked_candidate is controller.dialog.review_resolutions is None
    assert retained_approval['runs'][0]['reason'] == 'old affected area'  # Caller/history metadata is untouched.
    remaining_approval = dict(retained_approval, runs=[deepcopy(retained_approval['runs'][1])])
    assert controller._area_approval == first.approval == remaining_approval
    qtbot.waitUntil(lambda: not first.isVisible(), timeout=5000)
    reopened = controller.open_gap_review(original)
    qtbot.waitUntil(lambda: not reopened.busy(), timeout=5000)
    assert checks == [] and host.calls == [] and controller.findings_block_reason()
    count = len(requests)
    assert controller.refresh_findings()
    assert len(requests) == count + 1
    residual = deepcopy(original)
    changed = residual['runs'][0]
    changed['components'], changed['gap']['pixels'] = [], 0
    changed['target_correction_bindings'] = [deepcopy(binding)]
    changed['review_row_sha256'] = core._row_digest(changed)
    residual = install_last(reopened, residual)
    reopened.set_check_result(None, {'issues': []})  # Catalogue rendering must not reset the controller generation.
    reopened._water_tolerances(dict(height_tolerance_mm=4., angle_tolerance_degrees=.6))
    reopened.recheck.click()
    assert checks == [] and host.calls == []  # Changed display choices need a fresh catalogue first.
    residual = refresh_display(reopened)
    reopened.recheck.click()
    assert checks == [] and len(host.calls) == 1
    assert host.calls[0][2]['target_corrections'] == [binding]
    assert host.calls[0][2]['analysis_exclusions'] == remaining_approval
    successful = dict(_prepared(), target_corrections=dict(bindings=[binding]),
        gui_preparation_actions=dict(analysis_exclusions=remaining_approval, target_corrections=[binding]))
    assert controller.prepared(successful) and not controller._preparation_requires_refresh()
    controller.dialog = Checked()
    final = controller.open_gap_review(residual)
    qtbot.waitUntil(lambda: not final.busy(), timeout=5000)
    final.set_check_result(None, {'issues': []})
    generation = controller._preparation_action_generation
    controller._target_saved(deepcopy(result))  # A delayed duplicate signal must not dirty a consumed binding.
    assert controller._preparation_action_generation == generation and not controller._preparation_requires_refresh()
    assert controller.dialog.candidate == {'old': 'checked'}
    controller.wait_for_workers()  # A second shutdown ingestion of the same completed Save is also a no-op.
    assert controller._preparation_action_generation == generation and not controller._preparation_requires_refresh()
    assert controller.dialog.candidate == {'old': 'checked'}
    final = controller.open_gap_review(residual)
    qtbot.waitUntil(lambda: not final.busy(), timeout=5000)
    final.set_check_result(None, {'issues': []})
    final._water_tolerances(dict(height_tolerance_mm=5., angle_tolerance_degrees=.7))
    final.recheck.click()
    assert checks == [] and len(host.calls) == 1
    refresh_display(final)
    final.recheck.click()
    assert checks == ['check'] and len(host.calls) == 1  # Binding is already consumed; no permanent dirty state.
    assert source_choices == [controller._source_choices[-1]]
    final.set_check_result(None, {'issues': []})  # The finite Check spy delivers no worker signal.
    final._recheck()
    assert checks == ['check', 'check'] and len(host.calls) == 1
    assert source_choices == [controller._source_choices[-1], controller._source_choices[-1]]
    controller.dialog = None
    host._target_correction_dialog = None
