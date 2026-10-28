"""Ordinary GUI handoff of display DATA; every photograph/operation is synthetic."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import map_adoption_view as adoption_ui
from darsia.fluidflower import painted_review_comparison as display

from .test_painting_gap_review_ui import _Backend, _controller, _dialog, _report, _review


@pytest.mark.parametrize('worker_name', ['_AdoptionWorker', '_PlacementWorker'])
@pytest.mark.parametrize('has_context', [False, True])
@pytest.mark.parametrize('positive_contracts', [False, True])
def test_ordinary_prepare_transfer_and_native_review_keep_absent_or_snapshotted_context(qtbot, monkeypatch, tmp_path,
        worker_name, has_context, positive_contracts):
    from darsia.fluidflower import series_layer_view as view
    from darsia.fluidflower import map_adoption, map_placement, raw_shape_basis_reader, painting_frame_models
    from darsia.fluidflower import painted_analysis_exclusions as core
    row = SimpleNamespace(layer_id=1, name='sand', facies='porous')
    session = SimpleNamespace(run_id='ac17', layers=[row], facies_of=lambda _: SimpleNamespace(role='porous'))
    identity = dict(path='synthetic-index', sha256='a' * 64, size_bytes=1)
    monkeypatch.setattr(painting_frame_models, 'read_adoption_cohort',
        lambda *_: dict(summary='synthetic', runs=['ac17'], geometry_index=identity))
    monkeypatch.setattr(painting_frame_models, 'verify_adoption_cohort', lambda *_: None)
    monkeypatch.setattr(painting_frame_models, 'verify_frame_model_selection', lambda *_: None)
    monkeypatch.setattr(raw_shape_basis_reader, 'require_basis_index', lambda *_: None)
    monkeypatch.setattr(view.lpm, 'version_layers', lambda *_args, **_kwargs: (session, np.ones((2, 2), np.uint8)))
    monkeypatch.setattr(view, '_new_placement_contracts', lambda *_args, **_kwargs:
        ({'ac17': {}}, {'same': 'source'}) if positive_contracts else (None, None))
    calls = []
    def native(*_args, **options):
        calls.append(('native', deepcopy(options)))
        return dict(synthetic='review')
    monkeypatch.setattr(core, 'build_transfer_gap_review', native)
    def sink(*_args, **options):
        if options.get('native_review_builder'):
            options['native_review_builder'](tmp_path)
        calls.append(('operation', {key: deepcopy(value) for key, value in options.items()
            if key not in {'native_review_builder', 'photo_loader', 'progress'}}))
        return dict(geometry_index=identity)
    monkeypatch.setattr(map_adoption, 'prepare_adoption', sink)
    monkeypatch.setattr(map_placement, 'place_map_in_runs', sink)
    context = dict(declared_run_order=['ac17', 'ac20', 'ac18'])
    options = dict(destination='unused', geometry_index='synthetic-index', runs=['ac17'], photo_loader=lambda *_: None)
    if has_context:
        options['display_context'] = context
    worker = getattr(view, worker_name)(dict(run_id='ac17'), None, options)
    context['declared_run_order'].append('caller mutation')
    failures = []
    worker.failed.connect(failures.append)
    worker.run()
    assert not failures and any(kind == 'operation' for kind, _ in calls)
    for _, values in calls:
        assert ('display_context' in values) is (has_context and positive_contracts)
        if has_context and positive_contracts:
            assert values['display_context']['declared_run_order'] == ['ac17', 'ac20', 'ac18']
    assert any(kind == 'native' for kind, _ in calls) is (worker_name == '_PlacementWorker' and positive_contracts)


def test_check_normalizes_snapshotted_preferences_off_the_candidate_and_save_forwards_exact_attachment(qtbot, monkeypatch, tmp_path):
    from darsia.fluidflower import geometry_facies as geometry
    from darsia.fluidflower import painted_adoption_table as table
    from darsia.fluidflower import painted_analysis_exclusions as core
    prep = dict(path='synthetic-preparation', sha256='a' * 64, size_bytes=1)
    review_binding = dict(path='synthetic-review', sha256='b' * 64, size_bytes=2)
    candidate = dict(provenance=dict(preparation=prep))
    preferences = dict(position_policy=dict(method=display.METHOD, shift_tolerance_mm=4., angle_tolerance_degrees=.2),
        water_policy=dict(height_tolerance_mm=5., angle_tolerance_degrees=.3), water_actions={})
    monkeypatch.setattr(adoption_ui.ma, 'adoption_candidate', lambda *_args, **_kwargs: (deepcopy(candidate), {}))
    monkeypatch.setattr(table, '_checked', lambda value, **_: dict(gap_review=review_binding) if value == prep else pytest.fail('unexpected metadata'))
    monkeypatch.setattr(display, 'read_review', lambda path: dict(review_binding=review_binding))
    snapshots = []
    def make(review, **options):
        snapshots.append(deepcopy(options))
        return dict(synthetic='normalized compact attachment', preferences=deepcopy(options))
    monkeypatch.setattr(display, 'make_display_attachments', make)
    def validate(value, **options):
        assert value == candidate and 'display_attachments' not in value
        return dict(issues=[], display_attachments=deepcopy(options['display_attachments']), display_findings_sha256='c' * 64)
    monkeypatch.setattr(geometry, 'validate_geometry_config', validate)
    worker = adoption_ui._CheckWorker({}, tmp_path, None, display_preferences=preferences)
    preferences['position_policy']['shift_tolerance_mm'] = 999.
    results, failures = [], []
    worker.finished.connect(results.append); worker.failed.connect(failures.append)
    worker.run()
    assert not failures and snapshots[0]['position_policy']['shift_tolerance_mm'] == 4.
    attachment = results[0]['validation_report']['display_attachments']
    seals, saves = [], []
    def seal(value, report, records, **options):
        seals.append(deepcopy(options))
        assert report['display_attachments'] == options['display_attachments']
        return dict(records=deepcopy(records), **options)
    class Store:
        def __init__(self, project):
            pass
        def save_revision(self, value, **options):
            saves.append(deepcopy(options))
            return dict(revision_id='synthetic')
    monkeypatch.setattr(core, 'seal_findings_resolutions', seal)
    monkeypatch.setattr(geometry, 'GeometryRevisionStore', Store)
    save = adoption_ui._SaveRevisionWorker(object(), candidate, 'synthetic human', 'reason',
        validation_report=results[0]['validation_report'], review_resolutions=dict(warnings=[], acquisition=[], analysis_exclusions=None),
        display_attachments=attachment)
    attachment['preferences']['position_policy']['shift_tolerance_mm'] = 777.
    save.failed.connect(failures.append); save.run()
    assert not failures and seals[0]['display_attachments']['preferences']['position_policy']['shift_tolerance_mm'] == 4.
    assert saves[0]['preserve_failed_artifacts'] is True


def test_changed_display_preferences_invalidate_both_dialogs_and_recheck_without_preparing_again(qtbot, monkeypatch):
    from .test_findings_before_check import _literal_review
    host, backend, controller = _controller(qtbot, monkeypatch)
    requests = []
    controller.catalogue_refresher = requests.append  # Finite GUI handoff DATA, not scientific acceptance.
    findings = controller.open_gap_review(_literal_review())
    qtbot.waitUntil(lambda: not findings.busy(), timeout=5000)
    findings.classification.setCurrentIndex(findings.classification.findData('air'))
    findings.reason.setText('Synthetic human reviewed both displayed non-sand areas')
    findings.approve_all.click()
    source_row = next(index for index, row in enumerate(findings._rows) if row['kind'] == 'source_state')
    findings.runs.setCurrentCell(source_row, 0)
    qtbot.waitUntil(lambda: not findings.busy(), timeout=5000)
    findings.source_state.setCurrentIndex(findings.source_state.findData('unknown'))
    findings.reason.setText('Synthetic human reviewed the source water state')
    findings.set_state.click()
    qtbot.waitUntil(lambda: bool(requests) and not findings.busy(), timeout=5000)
    request = requests[-1]
    assert controller.install_findings_result(dict(generation=request['generation'],
        review=deepcopy(request['review']), report=dict(issues=[], input_bindings=deepcopy(request['input_bindings']))))
    qtbot.waitUntil(lambda: not findings.busy(), timeout=5000)
    approvals = deepcopy(backend.approvals)
    calls = []
    class Existing:
        def __init__(self):
            self.candidate = self.checked_candidate = {'old': 'candidate'}
            self.validation_report = {'old': 'report'}
            self.review_resolutions = {'old': 'acknowledgments'}
        def busy(self):
            return False
        def wait_for_workers(self):
            pass  # This finite Check spy owns no worker.
        def set_display_preferences(self, value):
            calls.append(('preferences', deepcopy(value)))
            self.candidate = self.checked_candidate = self.validation_report = self.review_resolutions = None
            return True
        def check(self):
            calls.append(('check', None)); return True
        def choose_facies_source_state(self, action):
            calls.append(('source-choice', deepcopy(action)))
            return self.check()
    controller.dialog = Existing()
    controller.preparation = {'folder': 'existing completed preparation'}
    controller._completed_preparation_generation = controller._preparation_action_generation
    issue = dict(severity='warning', code='advisory', message='same actual issue', details={})
    _report(findings, [issue])
    qtbot.waitUntil(lambda: not findings.busy(), timeout=5000)
    findings._water_tolerances(dict(height_tolerance_mm=6., angle_tolerance_degrees=.8))
    assert findings._candidate is None and controller.dialog.candidate is None
    assert findings._warning_actions == {} and not findings.adopt.isEnabled()
    findings.recheck.click()
    assert not any(kind == 'check' for kind, _ in calls) and host.calls == []
    findings.refresh_findings.click()
    request = requests[-1]
    assert controller.install_findings_result(dict(generation=request['generation'],
        review=deepcopy(request['review']), report=dict(issues=[issue], input_bindings=deepcopy(request['input_bindings']))))
    qtbot.waitUntil(lambda: not findings.busy(), timeout=5000)
    warning_row = next(index for index, row in enumerate(findings._rows) if row['kind'] == 'warning')
    findings.runs.setCurrentCell(warning_row, 0)
    qtbot.waitUntil(lambda: not findings.busy(), timeout=5000)
    findings.reason.setText('Synthetic human accepts this exact refreshed advisory')
    findings.approve_warning.click()
    findings.recheck.click()
    assert [kind for kind, _ in calls].count('check') == 1 and host.calls == []
    preferences = [value for kind, value in calls if kind == 'preferences']
    assert preferences[-1]['water_policy']['height_tolerance_mm'] == 6.
    assert [value for kind, value in calls if kind == 'source-choice'] == [controller._source_choices[-1]]
    assert backend.approvals == approvals  # Display changes added no area action or human approval.
    # This synthetic dialog is not a Qt worker owner and needs no shutdown hook.
    controller.dialog = None


def test_human_can_remove_unavailable_stale_water_action_without_touching_history(qtbot, monkeypatch):
    host, backend, controller = _controller(qtbot, monkeypatch)
    old = dict(action=dict(run_id='ac99', basis_sha256='a' * 64, actor='past human'), measurements={'old': 'data'})
    controller._water_surface_confirmations = {'ac99': deepcopy(old)}
    findings = controller.open_gap_review(_review())
    qtbot.waitUntil(lambda: not findings.busy(), timeout=5000)
    assert findings.clear_water_confirmations.isVisible() and findings.clear_water_confirmations.isEnabled()
    assert controller._display_preferences()['water_actions']['ac99'] == old['action']
    findings.clear_water_confirmations.click()
    assert controller._water_surface_confirmations == {} and findings.water_surface_confirmations == {}
    assert old['action']['basis_sha256'] == 'a' * 64 and backend.approvals == [] and host.calls == []
    assert findings._display_preferences_dirty and 'Saved history is retained' in findings.status.text()


@pytest.mark.parametrize('change', ['human-line-edit', 'actual-photo-basis'])
def test_changed_line_or_actual_photo_never_reuses_old_human_confirmation(qtbot, change):
    from .test_painting_water_surface_ui import _api, _packet
    backend = _Backend()
    packet = backend.packets['ac18']
    packet.update(display_grid={'synthetic': 'same-explicit-plane'}, water_surface=_packet(),
        image_choices=[dict(image_id='clearwater', available=True)])
    api, _ = _api()
    widget, _ = _dialog(qtbot, backend=backend, water_surface_api=api,
        comparison_reader=lambda *_: dict(packet))
    widget.images.setCurrentIndex(widget.images.findData('clearwater'))
    qtbot.waitUntil(lambda: not widget.busy(), timeout=5000)
    widget.water_surface.confirm.click()
    old = deepcopy(widget.water_surface_confirmations['ac18'])
    if change == 'human-line-edit':
        widget.water_surface.move.setChecked(True)
        widget.water_surface.move_to_photo_point([5., 15.])
        assert widget.water_surface.action is None and 'ac18' not in widget.water_surface_confirmations
        calls = []
        widget.display_recheck_requested.connect(lambda: calls.append('check'))
        widget.recheck.click()
        assert calls == [] and widget._unconfirmed_water_edits == {'ac18'}
    else:
        fresh = deepcopy(packet['water_surface'])
        fresh['basis']['photo']['sha256'] = 'f' * 64
        widget.packet['water_surface'] = fresh
        widget._set_water_surface(widget.packet)
        assert widget.water_surface.action is None and 'The previous confirmation' in widget.water_surface.status.text()
        assert widget.water_surface_confirmations['ac18'] == old  # Viewing does not clear or reconfirm it.
    widget.water_surface.confirm.click()
    fresh = widget.water_surface_confirmations['ac18']
    assert fresh['action'] != old['action'] and widget._unconfirmed_water_edits == set()
    if change == 'actual-photo-basis':
        assert old['action']['basis_sha256'] != fresh['action']['basis_sha256']
    else:
        assert old['action']['line_photo_xy'] != fresh['action']['line_photo_xy']
