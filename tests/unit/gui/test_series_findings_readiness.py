"""Exact current findings gate new preparation without running scientific workers."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QDialog, QLabel, QMessageBox, QPushButton, QWidget

from darsia.fluidflower import layer_painting as painting
from darsia.fluidflower.map_adoption_view import AdoptionDialog
from darsia.fluidflower.series_setup_jobs_ui import SeriesSetupJobsController

from .test_painting_gap_review_ui import _controller, _review


@pytest.fixture
def current_list(qtbot, monkeypatch):
    """Load the real findings dialog through a finite injected metadata reader."""
    host, backend, controller = _controller(qtbot, monkeypatch)
    requests = []
    controller.catalogue_refresher = requests.append
    review = _review()
    review['review_binding'] = dict(path='C:/synthetic/review.json', sha256='f' * 64, size_bytes=101)
    controller.transfer_review_ready(review)
    controller.placement_retired()
    findings = controller.gap_dialog
    qtbot.waitUntil(lambda: not findings.busy(), timeout=5000)
    assert findings.review == review

    def install(fresh, report):
        """Install reader-double DATA for controller tests, not native qualification."""
        findings._catalogue(deepcopy(fresh))
        findings.set_catalogue_result(deepcopy(report))
        return True

    findings.install_catalogue = install
    return SimpleNamespace(host=host, backend=backend, controller=controller, findings=findings, requests=requests)


def _reply(owner, *, review=None):
    """Deliver synthetic authenticated-reader DATA to test only the GUI join."""
    request = owner.requests[-1]
    fresh = deepcopy(request['review'] if review is None else review)
    fresh['review_binding'] = dict(path=f"C:/synthetic/residual-{request['generation']}.json",
        sha256=f"{request['generation']:064x}", size_bytes=101)
    return dict(generation=request['generation'], review=fresh,
        report=dict(issues=[], input_bindings=deepcopy(request['input_bindings'])))


def _resolve(qtbot, owner, *, retain_resolved_components=False):
    """Exercise choices and inject collector DATA without claiming replay qualification."""
    findings = owner.findings
    findings.classification.setCurrentIndex(findings.classification.findData('air'))
    findings.reason.setText('The displayed non-sand areas were reviewed explicitly')
    findings.approve_all.click()
    qtbot.waitUntil(lambda: not findings.busy(), timeout=5000)
    assert owner.requests == [] and owner.controller.findings_block_reason()
    index = next(index for index, row in enumerate(findings._rows) if row['kind'] == 'source_state')
    findings.runs.setCurrentCell(index, 0)
    qtbot.waitUntil(lambda: not findings.busy(), timeout=5000)
    findings.reason.setText('The source water state was reviewed explicitly')
    findings.source_state.setCurrentIndex(findings.source_state.findData('clearwater'))
    findings.set_state.click()
    assert owner.controller.findings_block_reason()
    residual = deepcopy(findings.review)
    if not retain_resolved_components:
        for run in residual['runs']:
            run['components'] = []
    assert owner.controller.install_findings_result(_reply(owner, review=residual))
    qtbot.waitUntil(lambda: not findings.busy(), timeout=5000)
    assert findings.unresolved_findings_reason() == ''


def test_historical_route_without_transfer_keeps_original_preparation_readiness(qtbot, monkeypatch):
    """No new list requirement is inferred when this view has no Transfer context."""
    host, _, controller = _controller(qtbot, monkeypatch)
    assert controller.findings_block_reason() == ''
    assert controller.start(host.record, host.selection)
    assert len(host.calls) == 1 and controller.gap_dialog is None


def test_direct_checks_and_water_choice_cannot_bypass_unresolved_or_closed_list(current_list, qtbot):
    """The source choice and every direct entry leave unresolved areas unprocessed."""
    owner = current_list
    checks, changes = [], []
    owner.controller.findings_changed.connect(lambda: changes.append(True))
    owner.controller.preparation = {'synthetic': 'completed preparation'}
    owner.controller._completed_preparation_generation = owner.controller._preparation_action_generation
    owner.controller.dialog = SimpleNamespace(
        busy=lambda: False, wait_for_workers=lambda: None,
        set_display_preferences=lambda *_: checks.append('preferences') or True,
        check=lambda: checks.append('check') or True,
        choose_facies_source_state=lambda *_: checks.append('state-check') or True)
    index = next(index for index, row in enumerate(owner.findings._rows) if row['kind'] == 'source_state')
    owner.findings.runs.setCurrentCell(index, 0)
    qtbot.waitUntil(lambda: not owner.findings.busy(), timeout=5000)
    owner.findings.reason.setText('The source photo was reviewed explicitly')
    owner.findings.source_state.setCurrentIndex(owner.findings.source_state.findData('clearwater'))
    owner.findings.set_state.click()
    assert owner.controller._source_choices == owner.findings._acquisition_actions
    assert checks == [] and 'Finish the current gap choices' in owner.findings.status.text()
    assert owner.requests == []
    owner.controller._recheck_display()
    owner.controller._recheck_gaps(None)
    assert not owner.controller.start(owner.host.record, owner.host.selection)
    assert owner.host.calls == checks == []
    before = owner.controller.findings_block_reason()
    owner.findings.close()
    assert owner.controller.context is None
    assert owner.controller.findings_block_reason() == before
    assert not owner.controller.start(owner.host.record, owner.host.selection)
    assert owner.host.calls == checks == [] and changes


@pytest.mark.parametrize('changed', ['project', 'map', 'approval', 'frame', 'geometry'])
def test_resolved_readiness_is_bound_to_exact_current_gui_context(current_list, qtbot, changed):
    """A prior all-resolved list cannot authorize another selected source context."""
    owner = current_list
    _resolve(qtbot, owner)
    assert owner.controller.findings_block_reason() == ''
    if changed == 'project':
        owner.host.project = SimpleNamespace(root=owner.host.project.root)
    elif changed == 'map':
        owner.host.record['kind'] = 'different-kind'
    elif changed == 'approval':
        owner.host.approval = {'version': 'new-explicit-approval'}
    elif changed == 'frame':
        owner.host.frame = SimpleNamespace(to_dict=lambda: {'frame': 'changed'})
    else:
        owner.host.selection = replace(owner.host.selection, sha256='0' * 64)
    assert 'current project, map, approval and geometry' in owner.controller.findings_block_reason()
    assert not owner.controller.start(owner.host.record, owner.host.selection)
    assert owner.host.calls == []


def test_restart_status_never_uses_an_old_adopted_revision_as_findings_proof(qtbot, monkeypatch):
    """Only exact completed passive metadata can establish a known Transfer input."""
    host, _, controller = _controller(qtbot, monkeypatch)
    folder = host.project.root / 'paintings' / 'session'
    monkeypatch.setattr(painting, 'painting_folder', lambda project: folder)
    expected = json.dumps([str(folder.parent),
        str(host.project.root / 'overrides/geometry_facies/current.json'),
        host.record['folder'], host.record['sha256'], host.approval], sort_keys=True)
    status = host.series_use_status = SimpleNamespace(desired=expected, cached=expected,
        record=deepcopy(host.record), approval=deepcopy(host.approval),
        result=dict(transfer={'recorded': True}, preparation=None, gap_reviews=[], errors=[],
                    adopted={'revision_id': 'older-revision'}))
    assert 'Open Review gaps and findings' in controller.findings_block_reason()
    assert not controller.start(host.record, host.selection) and host.calls == []
    status.cached = None
    assert 'finish loading' in controller.findings_block_reason()
    status.cached = expected
    status.record['kind'] = 'another-record'
    assert 'finish loading' in controller.findings_block_reason()
    status.record = deepcopy(host.record)
    status.result = dict(transfer=None, preparation=None, gap_reviews=[{'recorded': 'review'}], errors=[],
                         adopted={'revision_id': 'older-revision'})
    assert 'Open Review gaps and findings' in controller.findings_block_reason()
    status.result = dict(transfer=None, preparation=None, gap_reviews=[], errors=[],
                         adopted={'revision_id': 'older-revision'})
    assert controller.findings_block_reason() == ''


def test_failed_cached_history_remains_unavailable_in_a_fresh_controller(qtbot, monkeypatch):
    """A completed failed read cannot establish that no Transfer was recorded."""
    host, _, controller = _controller(qtbot, monkeypatch)
    folder = host.project.root / 'paintings' / 'session'
    monkeypatch.setattr(painting, 'painting_folder', lambda project: folder)
    expected = json.dumps([str(folder.parent),
        str(host.project.root / 'overrides/geometry_facies/current.json'),
        host.record['folder'], host.record['sha256'], host.approval], sort_keys=True)
    status = host.series_use_status = SimpleNamespace(desired=expected, cached=expected,
        record=deepcopy(host.record), approval=deepcopy(host.approval),
        result=dict(transfer=None, preparation=None, adopted=None,
                    errors=['PermissionError: Saved status metadata could not be read']))
    reason = controller.findings_block_reason()
    assert 'history is unavailable' in reason and 'status read error' in reason
    assert not controller.start(host.record, host.selection)
    assert host.use_map_progress.text() == reason and host.calls == []
    status.result = dict(transfer=None, preparation=None, adopted=None, gap_reviews=[], errors=[])
    assert controller.findings_block_reason() == ''


@pytest.mark.parametrize('malformed', [None, {}, {'errors': []}, [],
    dict(transfer=None, preparation=None, adopted=None, gap_reviews=[], errors=''),
    dict(transfer=None, preparation=None, adopted=None, gap_reviews=None, errors=[]),
    dict(transfer=False, preparation=None, adopted=None, gap_reviews=[], errors=[]),
    dict(transfer=None, preparation=None, adopted=None, gap_reviews=['unbound'], errors=[])])
def test_malformed_cached_history_cannot_establish_transfer_absence(qtbot, monkeypatch, malformed):
    """Only the actual successful passive reader shape can describe no Transfer."""
    host, _, controller = _controller(qtbot, monkeypatch)
    folder = host.project.root / 'paintings' / 'session'
    monkeypatch.setattr(painting, 'painting_folder', lambda project: folder)
    expected = json.dumps([str(folder.parent),
        str(host.project.root / 'overrides/geometry_facies/current.json'),
        host.record['folder'], host.record['sha256'], host.approval], sort_keys=True)
    host.series_use_status = SimpleNamespace(desired=expected, cached=expected,
        record=deepcopy(host.record), approval=deepcopy(host.approval), result=malformed)
    assert 'history is unavailable' in controller.findings_block_reason()
    assert not controller.start(host.record, host.selection) and host.calls == []


class _ReadinessSignals(QObject):
    """Supply only existing view and workspace notifications for Setup refresh."""

    busy_changed = Signal(bool)
    reference_changed = Signal(str)


def test_findings_signals_and_jobs_refresh_use_loaded_approval_without_reader(current_list, monkeypatch):
    """Passive GUI readiness never calls the fresh-action approval I/O boundary."""
    owner = current_list
    reads = []

    def forbidden_reader(run):
        reads.append(run)
        raise AssertionError('Readiness must use already loaded approval metadata')

    monkeypatch.setattr(owner.host, '_read_approvals', forbidden_reader)
    signals = _ReadinessSignals(owner.host)
    workspace = QWidget(owner.host)
    workspace.project, workspace._loaded_root = owner.host.project, owner.host.project.root
    workspace.reference_changed = signals.reference_changed
    owner.host.busy_changed = signals.busy_changed
    owner.host.apply_build_button = QPushButton(owner.host)
    owner.host.apply_build_reason = QLabel(owner.host)
    jobs = SeriesSetupJobsController(workspace, owner.host, context_provider=lambda _: None)
    assert owner.controller.findings_block_reason()
    owner.findings.readiness_changed.emit()
    owner.controller.findings_changed.emit()
    signals.busy_changed.emit(False)
    signals.reference_changed.emit('current-reference')
    jobs.refresh()
    assert reads == [] and not owner.host.apply_build_button.isEnabled()
    assert 'Review gaps and findings' in owner.host.apply_build_reason.text()
    assert owner.host.calls == []


def _saved_target(owner):
    """Expose a completed explicit Save from the currently owned target dialog."""
    run = owner.findings.review['runs'][0]
    component = run['components'][0]
    result = dict(run_id=run['run_id'], component_id=component['component_id'],
        binding=dict(version={'path': 'C:/synthetic/target/version.json'},
                     approval={'path': 'C:/synthetic/target/approval.json'}))
    target = SimpleNamespace(run_id=result['run_id'], component_id=result['component_id'],
        binding=deepcopy(result['binding']), review=deepcopy(owner.findings.review),
        wait_for_workers=lambda: None)
    owner.host._target_correction_dialog = target
    return result, target, component


def _draft_payload(owner):
    """Describe an actual current row without manufacturing any C approval."""
    run = owner.findings.review['runs'][0]
    component = run['components'][0]
    return dict(review_binding=deepcopy(owner.findings.review['review_binding']), run_id=run['run_id'],
        component_id=component['component_id'], review_row_sha256=component['review_row_sha256'], pending=True)


def test_draft_lock_survives_close_and_false_signal_until_confirmed_callback(current_list):
    """Only the separate trusted confirmation seam clears this exact draft context."""
    owner = current_list
    payload = _draft_payload(owner)
    pending = [deepcopy(payload)]
    owner.findings.pending_draft_choices = lambda: deepcopy(pending)
    assert owner.controller.set_findings_draft_pending(owner.findings, payload)
    owner.controller._findings_required = False
    assert 'Unconfirmed area choices' in owner.controller.findings_block_reason()
    cleared = dict(payload, pending=False)
    assert not owner.controller.set_findings_draft_pending(owner.findings, cleared)
    owner.findings.close()
    assert 'Unconfirmed area choices' in owner.controller.findings_block_reason()
    assert not owner.controller.start(owner.host.record, owner.host.selection)
    pending.clear()
    assert owner.controller.set_findings_draft_pending(owner.findings, cleared, confirmed=True)
    assert owner.controller._findings_draft_contexts == {} and owner.host.calls == []


def test_new_draft_invalidates_inflight_result_without_discarding_existing_actions(current_list):
    """Older DATA cannot overwrite a fresh unconfirmed edit of the same literal inputs."""
    owner = current_list
    assert owner.controller.refresh_findings()
    reply = _reply(owner)
    inputs = deepcopy(owner.requests[-1]['input_bindings'])
    payload = _draft_payload(owner)
    owner.findings.pending_draft_choices = lambda: [deepcopy(payload)]
    assert owner.controller.set_findings_draft_pending(owner.findings, payload)
    assert owner.controller._catalogue_request is None
    assert not owner.controller.install_findings_result(reply)
    owner.findings.operation_waiting = owner.findings._catalogue_refresh_pending = True
    assert owner.controller._consume_findings_result(
        ('findings-catalogue', reply['generation']), reply, owner=owner.findings)
    assert not owner.findings.operation_waiting and not owner.findings._catalogue_refresh_pending
    assert 'Unconfirmed area choices' in owner.controller.findings_block_reason()
    owner.controller._queue_findings_refresh()
    assert not owner.controller.refresh_findings()
    assert len(owner.requests) == 1
    assert owner.controller._findings_input_bindings(owner.findings.review) == inputs
    owner.findings.pending_draft_choices = lambda: []
    assert owner.controller.set_findings_draft_pending(owner.findings, dict(payload, pending=False), confirmed=True)
    owner.controller._queue_findings_refresh()
    assert len(owner.requests) == 2 and owner.requests[-1]['generation'] > reply['generation']
    assert not owner.controller.install_findings_result(reply) and owner.host.calls == []


@pytest.mark.parametrize('wrong', ['review', 'run', 'component', 'row', 'dialog', 'pending-type'])
def test_draft_notification_requires_the_exact_owned_current_row(current_list, wrong):
    """A stale signal cannot lock or clear another review's current policy state."""
    owner = current_list
    payload = _draft_payload(owner)
    dialog = owner.findings
    if wrong == 'review':
        payload['review_binding']['sha256'] = '0' * 64
    elif wrong == 'run':
        payload['run_id'] = 'other-run'
    elif wrong == 'component':
        payload['component_id'] = '0' * 64
    elif wrong == 'row':
        payload['review_row_sha256'] = '0' * 64
    elif wrong == 'dialog':
        dialog = SimpleNamespace(review=owner.findings.review)
    else:
        payload['pending'] = 1
    generation = owner.controller._catalogue_generation
    assert not owner.controller.set_findings_draft_pending(dialog, payload)
    assert owner.controller._findings_draft_contexts == {}
    assert owner.controller._catalogue_generation == generation and owner.requests == []


@pytest.mark.parametrize('mismatch', ['bare', 'run', 'component', 'binding', 'review', 'row'])
def test_only_exact_current_target_save_can_enter_collector_inputs(current_list, mismatch):
    """A pending binding or another target/review row never clears the new gate."""
    owner = current_list
    result, target, _ = _saved_target(owner)
    if mismatch == 'bare':
        owner.host._target_correction_dialog = None
    elif mismatch == 'run':
        result['run_id'] = 'other-run'
    elif mismatch == 'component':
        result['component_id'] = '0' * 64
    elif mismatch == 'binding':
        result['binding']['version']['path'] += '.changed'
    elif mismatch == 'review':
        target.review['review_binding']['sha256'] = '0' * 64
    else:
        target.review['runs'][0]['components'][0]['review_row_sha256'] = '0' * 64
    generation = owner.controller._preparation_action_generation
    owner.controller._target_saved(result)
    assert owner.controller._target_bindings == {}
    assert owner.findings._resolved_target_rows == set()
    assert owner.controller._preparation_action_generation == generation
    assert owner.controller.findings_block_reason() and owner.host.calls == []


def test_verified_target_save_requires_original_replay_residual_review_before_check(current_list, qtbot):
    """Save removes old approval but only fresh collector DATA changes the gap list."""
    owner = current_list
    result, _, component = _saved_target(owner)
    original = deepcopy(owner.findings.review)
    action = dict(run_id=result['run_id'], component_id=result['component_id'],
        review_row_sha256=component['review_row_sha256'], classification='air', reason='Original explicit choice')
    owner.findings.approval = dict(runs=[deepcopy(action)])
    owner.controller._target_saved(result)
    assert owner.findings.approval is None and action['reason'] == 'Original explicit choice'
    assert owner.findings._resolved_target_rows == set()
    assert len(owner.findings.pending_corrections) == 1
    qtbot.waitUntil(lambda: not owner.findings.busy(), timeout=5000)
    assert 'Updating findings' in owner.controller.findings_block_reason()
    assert not owner.findings.can_recheck() and owner.findings.review == original
    assert owner.requests[-1]['input_bindings']['target_bindings'] == [result['binding']]
    assert owner.requests[-1]['input_bindings']['analysis_exclusions'] is None
    generation = owner.controller._preparation_action_generation
    owner.controller._target_saved(deepcopy(result))
    assert owner.controller._preparation_action_generation == generation
    residual = deepcopy(original)
    residual['runs'][0]['components'] = []
    reply = _reply(owner, review=residual)
    assert owner.controller.install_findings_result(reply)
    qtbot.waitUntil(lambda: not owner.findings.busy(), timeout=5000)
    assert owner.findings.review == reply['review']
    assert owner.findings._resolved_target_rows == set()
    assert owner.findings.unresolved_findings_reason() and owner.host.calls == []


def test_cold_review_retains_complete_target_prefix_before_a_new_save_request(current_list, qtbot):
    """Old reviewed descriptors keep cohort order and are not silently reapproved."""
    owner = current_list
    first = dict(version={'path': 'C:/synthetic/prior-first.json'}, approval={'path': 'C:/synthetic/prior-first-approval.json'})
    second = dict(version={'path': 'C:/synthetic/prior-second.json'}, approval={'path': 'C:/synthetic/prior-second-approval.json'})
    third = dict(version={'path': 'C:/synthetic/prior-third.json'}, approval={'path': 'C:/synthetic/prior-third-approval.json'})
    owner.findings.review['runs'][0]['target_correction_bindings'] = [deepcopy(first), deepcopy(second)]
    owner.findings.review['runs'][1]['target_correction_bindings'] = [deepcopy(third)]
    original = deepcopy(owner.findings.review)
    result, _, _ = _saved_target(owner)
    owner.controller._target_bindings[('retained', 'same-binding')] = deepcopy(second)
    owner.controller._target_saved(result)
    qtbot.waitUntil(lambda: bool(owner.requests) and not owner.findings.busy(), timeout=5000)
    captured = owner.requests[-1]['input_bindings']['target_bindings']
    assert captured == [first, second, third, result['binding']]
    assert owner.findings.review == original and owner.findings.approval is None
    captured[0]['version']['path'] = 'changed returned request'
    assert owner.findings.review == original
    assert owner.controller._findings_input_bindings(owner.findings.review)['target_bindings'] == [
        first, second, third, result['binding']]
    assert owner.findings._resolved_target_rows == set() and owner.host.calls == []


def test_same_complete_target_cohort_is_forwarded_to_original_prepare(current_list, qtbot):
    """The existing Prepare dispatcher receives the collector's literal target order."""
    owner = current_list
    _resolve(qtbot, owner)
    prior = dict(version={'path': 'C:/synthetic/prior.json'}, approval={'path': 'C:/synthetic/prior-approval.json'})
    added = dict(version={'path': 'C:/synthetic/added.json'}, approval={'path': 'C:/synthetic/added-approval.json'})
    owner.findings.review['runs'][0]['target_correction_bindings'] = [deepcopy(prior)]
    owner.controller._target_bindings[('retained', 'duplicate')] = deepcopy(prior)
    owner.controller._target_bindings[('current', 'added')] = deepcopy(added)
    assert owner.controller.refresh_findings()
    cohort = deepcopy(owner.requests[-1]['input_bindings']['target_bindings'])
    assert cohort == [prior, added]
    assert owner.controller.install_findings_result(_reply(owner))
    qtbot.waitUntil(lambda: not owner.findings.busy(), timeout=5000)
    owner.controller._recheck_gaps(owner.findings.approval)
    assert len(owner.host.calls) == 1
    assert owner.host.calls[0][2]['target_corrections'] == cohort


@pytest.mark.parametrize('stale', ['generation', 'context', 'target', 'review', 'report-inputs'])
def test_collector_result_cannot_install_after_its_exact_inputs_change(current_list, qtbot, stale):
    """No stale callback may replace the list or clear its original-replay lock."""
    owner = current_list
    result, _, _ = _saved_target(owner)
    owner.controller._target_saved(result)
    qtbot.waitUntil(lambda: bool(owner.requests) and not owner.findings.busy(), timeout=5000)
    reply = _reply(owner)
    before = deepcopy(owner.findings.review)
    installs = []
    owner.findings.install_catalogue = lambda *args: installs.append(args)
    if stale == 'generation':
        reply['generation'] -= 1
    elif stale == 'context':
        owner.host.approval = {'version': 'changed-current-approval'}
    elif stale == 'target':
        key = (result['run_id'], result['component_id'])
        owner.controller._target_bindings[key]['version']['path'] += '.changed'
    elif stale == 'review':
        owner.findings.review['review_binding']['sha256'] = '0' * 64
        before = deepcopy(owner.findings.review)
    else:
        reply['report']['input_bindings']['source_choices'] = [{'acquisition_state': 'unrequested'}]
    assert not owner.controller.install_findings_result(reply)
    assert installs == [] and owner.findings.review == before
    assert owner.controller.findings_block_reason() and owner.host.calls == []


def test_collector_error_keeps_lock_and_old_generation_cannot_complete_retry(current_list, qtbot):
    """An explicit retry requires a fresh matching result after a collector failure."""
    owner = current_list
    result, _, _ = _saved_target(owner)
    owner.controller._target_saved(result)
    qtbot.waitUntil(lambda: bool(owner.requests) and not owner.findings.busy(), timeout=5000)
    old_reply = _reply(owner)
    request = owner.requests[-1]
    assert not owner.controller.install_findings_result(dict(generation=request['generation'],
        input_bindings=deepcopy(request['input_bindings']), error='The original replay source is unavailable'))
    assert 'original replay source is unavailable' in owner.controller.findings_block_reason()
    assert owner.controller.refresh_findings()
    assert owner.requests[-1]['generation'] > request['generation']
    assert not owner.controller.install_findings_result(old_reply)
    assert 'Updating findings' in owner.controller.findings_block_reason()
    assert owner.findings._candidate is None and owner.host.calls == []


def test_dispatch_failure_keeps_original_replay_required_and_reports_the_reason(current_list):
    """An unavailable dispatcher cannot leave a false updating or ready result."""
    owner = current_list

    def unavailable(request):
        raise RuntimeError('The owned metadata worker could not start')

    owner.controller.catalogue_refresher = unavailable
    assert not owner.controller.refresh_findings()
    assert 'RuntimeError: The owned metadata worker could not start' in owner.controller.findings_block_reason()
    assert owner.controller._catalogue_request is None
    assert not owner.findings.can_recheck() and owner.host.calls == []


def test_old_checked_candidate_cannot_bypass_independent_owner_update_lock(current_list, qtbot):
    """Adopt observes the same owner reason even when the old view candidate exists."""
    owner = current_list
    _resolve(qtbot, owner)
    owner.controller.preparation = {'synthetic': 'completed preparation'}
    owner.controller._completed_preparation_generation = owner.controller._preparation_action_generation
    owner.findings.set_check_result({'synthetic': 'checked candidate'}, {'issues': []})
    qtbot.waitUntil(lambda: not owner.findings.busy(), timeout=5000)
    assert owner.findings.can_adopt()
    forwarded = []
    owner.controller.dialog = SimpleNamespace(busy=lambda: False, wait_for_workers=lambda: None,
        set_review_resolutions=lambda *_: forwarded.append('resolutions'), show=lambda: forwarded.append('show'))
    owner.controller._findings_update_reason = 'The original replay for the current choices is still required.'
    owner.controller._findings_adopt({'synthetic': 'previous resolutions'})
    assert forwarded == [] and not owner.controller._confirmation_pending
    assert owner.host.calls == []


class _FinalQuestionDialog(QDialog):
    """Use the ORIGINAL question/save guards with UI-only candidate sentinels.

    No preparation or scientific acceptance is manufactured. A Save worker is
    forbidden by the tests, so a regression cannot launch any Store operation.
    """
    confirmation_decided = Signal(bool)
    _save_clicked = AdoptionDialog._save_clicked
    save = AdoptionDialog.save

    def __init__(self, owner):
        super().__init__(owner.host)
        self.project, self.actor = owner.host.project, 'Synthetic reviewer'
        self.candidate = {'synthetic': 'UI-only checked candidate'}
        self.checked_candidate = deepcopy(self.candidate)
        self.review_resolutions = {'synthetic': 'UI-only resolutions'}
        self.validation_report = {'issues': []}
        self.preparation = {'map_version': {}}
        self.automatic, self.findings_managed = False, True
        self.adoption_scope_note = ''
        self.display_attachments = self.revision = self._thread = None
        self.status = QLabel(self)
        self.confirmation_decided.connect(owner.controller._confirmation_decided)

    def busy(self):
        return False

    def wait_for_workers(self):
        pass  # This finite presentation fixture never creates a worker.


def _final_question(qtbot, owner, monkeypatch):
    from darsia.fluidflower import map_adoption_view

    workers = []
    def forbidden_worker(*args, **kwargs):
        workers.append((args, kwargs))
        pytest.fail('Changed final-question inputs must not construct a Store worker')
    monkeypatch.setattr(map_adoption_view, '_SaveRevisionWorker', forbidden_worker)
    dialog = _FinalQuestionDialog(owner)
    qtbot.addWidget(dialog)
    owner.controller.dialog = owner.host._adoption_dialog = dialog
    dialog.show()
    return dialog, workers


def test_unconfirmed_draft_before_confirmation_timer_never_asks_or_saves(current_list, qtbot, monkeypatch):
    """A resolved list becoming dirty must block the actual zero-delay timer."""
    owner = current_list
    _resolve(qtbot, owner, retain_resolved_components=True)
    assert owner.controller.findings_block_reason() == ''
    dialog, workers = _final_question(qtbot, owner, monkeypatch)
    questions = []
    monkeypatch.setattr(QMessageBox, 'question', lambda *args:
        questions.append(args[2]) or QMessageBox.StandardButton.Yes)
    payload = _draft_payload(owner)
    owner.findings.pending_draft_choices = lambda: [deepcopy(payload)]
    owner.controller._confirmation_pending = True
    owner.controller._confirm_timer.start(0)
    assert owner.controller.set_findings_draft_pending(owner.findings, payload)

    qtbot.waitUntil(lambda: not owner.controller._confirmation_pending, timeout=5000)

    assert questions == workers == [] and owner.host.calls == []
    assert dialog.candidate is not None  # The timer gate itself refused the old candidate.
    assert 'Unconfirmed area choices' in owner.host.use_map_progress.text()
    assert owner.controller._findings_draft_contexts


def test_source_choice_during_accepted_final_question_clears_candidate_before_original_save(
        current_list, qtbot, monkeypatch):
    """A nested-question edit is still a new choice after the initial gate passed."""
    owner = current_list
    _resolve(qtbot, owner)
    assert owner.controller.findings_block_reason() == ''
    dialog, workers = _final_question(qtbot, owner, monkeypatch)
    questions, decisions = [], []
    dialog.confirmation_decided.connect(decisions.append)
    before_requests = len(owner.requests)
    def accepted_after_changed_source(*args):
        questions.append(args[2])
        owner.findings.reason.setText('A queued source-water choice changed during the final question')
        owner.findings.source_state.setCurrentIndex(owner.findings.source_state.findData('indicator'))
        assert owner.findings.set_state.isEnabled()
        owner.findings.set_state.click()
        assert owner.controller.findings_block_reason()
        return QMessageBox.StandardButton.Yes
    monkeypatch.setattr(QMessageBox, 'question', accepted_after_changed_source)
    owner.controller._confirmation_pending = True

    owner.controller._confirm()

    assert len(questions) == 1 and decisions == [True]
    assert len(owner.requests) == before_requests + 1
    assert owner.requests[-1]['input_bindings']['source_choices'][-1]['acquisition_state'] == 'indicator'
    assert dialog.candidate is None and owner.controller.context is None
    assert 'sources or findings changed' in dialog.status.text()
    assert workers == [] and owner.host.calls == [] and dialog.revision is None


def test_a_changed_source_choice_refreshes_findings_even_after_all_rows_were_resolved(current_list, qtbot):
    """The last human choice still asks for DATA instead of automatically checking."""
    owner = current_list
    _resolve(qtbot, owner)
    checks = []
    owner.controller.dialog = SimpleNamespace(busy=lambda: False, wait_for_workers=lambda: None,
        choose_facies_source_state=lambda *_: checks.append('check') or True)
    owner.controller.preparation = {'synthetic': 'completed preparation'}
    previous = len(owner.requests)
    owner.findings.reason.setText('The same source photo received a new explicit water-state choice')
    owner.findings.source_state.setCurrentIndex(owner.findings.source_state.findData('indicator'))
    owner.findings.set_state.click()
    assert len(owner.requests) == previous + 1 and checks == []
    assert owner.requests[-1]['input_bindings']['source_choices'][-1]['acquisition_state'] == 'indicator'
    assert 'Updating findings' in owner.controller.findings_block_reason()
    assert not owner.findings.can_recheck() and owner.host.calls == []


def test_human_choices_batch_one_refresh_without_waiting_for_diagnostic_errors(current_list, qtbot):
    """Gap/source completion schedules one replay; diagnostic errors keep their semantics."""
    owner = current_list
    diagnostic = dict(severity='error', code='unmapped_labels', message='Original diagnostic error')
    owner.findings.set_check_result({'synthetic': 'old candidate'}, {'issues': [diagnostic]})
    qtbot.waitUntil(lambda: not owner.findings.busy(), timeout=5000)
    owner.findings.classification.setCurrentIndex(owner.findings.classification.findData('air'))
    owner.findings.reason.setText('The first displayed area was explicitly reviewed')
    generation = owner.controller._catalogue_generation
    owner.findings.approve_one.click()
    assert owner.controller._catalogue_generation > generation
    assert owner.requests == [] and owner.findings._candidate is None
    source = next(index for index, row in enumerate(owner.findings._rows) if row['kind'] == 'source_state')
    owner.findings.runs.setCurrentCell(source, 0)
    qtbot.waitUntil(lambda: not owner.findings.busy(), timeout=5000)
    owner.findings.reason.setText('The source water state was explicitly reviewed')
    owner.findings.source_state.setCurrentIndex(owner.findings.source_state.findData('clearwater'))
    owner.findings.set_state.click()
    assert owner.requests == []
    last = next(index for index, row in enumerate(owner.findings._rows)
                if row['kind'] == 'gap' and not owner.findings._resolved(row))
    owner.findings.runs.setCurrentCell(last, 0)
    qtbot.waitUntil(lambda: not owner.findings.busy(), timeout=5000)
    owner.findings.reason.setText('The remaining displayed area was explicitly reviewed')
    owner.findings.approve_one.click()
    qtbot.waitUntil(lambda: bool(owner.requests) and not owner.findings.busy(), timeout=5000)
    assert len(owner.requests) == 1 and owner.controller._choices_ready_for_refresh()
    assert owner.findings._issues[0]['issue'] == diagnostic
    assert not owner.findings.can_recheck() and owner.host.calls == []


def test_refresh_dependency_is_explicit_without_starting_check_or_preparation(current_list, qtbot):
    """The unavailable collector leaves dirty choices blocked and refresh retryable."""
    owner = current_list
    owner.controller.catalogue_refresher = None
    owner.findings._invalidate_display_report()
    qtbot.waitUntil(lambda: not owner.findings.busy(), timeout=5000)
    owner.findings.refresh_findings.click()
    assert 'original-check findings collector is not connected' in owner.findings.status.text()
    assert owner.findings.refresh_findings.isEnabled()
    assert owner.findings._display_preferences_dirty
    assert not owner.findings.can_recheck() and owner.host.calls == []
