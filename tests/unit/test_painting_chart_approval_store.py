"""Exact small file-bound decisions; no real run or scientific approval."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import toml

from darsia.fluidflower import painting_chart_transfer as chart
from darsia.fluidflower.painting_chart_approval_store import ChartApprovalStore
from darsia.fluidflower.painted_adoption_table import _bound
from .test_painting_chart_transfer import inputs


@pytest.fixture
def bound_session(tmp_path):
    def document(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding='utf8')
        return _bound(path)
    geometry = document('geometry.json', {'fixture': 'original grid'})
    painting = document('painting.json', {'fixture': 'saved version'})
    model = document('model.json', {'fixture': 'measured model'})
    product = document('product.json', {'fixture': 'original map'})
    detection = document('detection.json', {'fixture': 'original detection'})
    manifest = document('manifest.json', {'fixture': 'selected detections'})
    record = dict(folder=str(tmp_path / 'saved-version'), sha256='c' * 64)
    review = dict(source=dict(reference_run='ac17', painting_facies=painting), geometry_index=geometry,
        map_version=record,
        runs=[dict(run_id=run, reference_model=model, target_model=model, product=product) for run in chart.RUNS])
    review['review_binding'] = document('review.json', review)
    class Session:
        def __init__(self, _source=None):
            self.review = deepcopy(review)
            self.manifest_binding = deepcopy(manifest)
            self.roles = {2: 'free_water', 3: 'porous_medium', 17: 'colorchecker', 18: 'excluded'}
        def proposal(self, run, *, correction_mm=None, proposal_format=chart.FORMAT):
            data = inputs()
            data['source_binding'] = dict(review=deepcopy(review['review_binding']),
                painting_facies=deepcopy(painting), run_id=run, reference_model=deepcopy(model),
                target_model=deepcopy(model), base_product=deepcopy(product), detection_manifest=deepcopy(manifest),
                reference_detection=dict(inputs=[deepcopy(detection)]), detection=dict(inputs=[deepcopy(detection)]))
            return chart.propose_chart_transfer(run, **data, correction_mm=correction_mm, proposal_format=proposal_format)
    return SimpleNamespace(factory=Session, root=tmp_path, model=model, record=record,
        selection=SimpleNamespace(index_path=geometry['path'], sha256=geometry['sha256'], size_bytes=geometry['size_bytes']))


@pytest.mark.parametrize('proposal_format', [chart.LEGACY_FORMAT, chart.FORMAT, chart.DETECTED_FORMAT])
def test_exact_v1_v2_v3_decisions_reopen_without_new_human_records(bound_session, proposal_format):
    session = bound_session.factory()
    store = ChartApprovalStore(session)
    assert not store.load()['approvals']
    proposal = session.proposal('ac35', proposal_format=proposal_format)
    record = chart.confirm_chart_transfer(proposal, actor='Reviewer', reason='Finite fixture confirmation')
    store.save({'ac35': record}, {'ac35': proposal.payload.decode('utf8')}, {}, {'ac35': [True, True]})
    reopened = ChartApprovalStore(bound_session.factory()).load()
    assert reopened['approvals'] == {'ac35': record}
    assert reopened['payloads']['ac35'].encode() == proposal.payload


@pytest.mark.parametrize('fault', ['source-bytes', 'geometry', 'map-version', 'payload'])
def test_source_binding_or_approved_payload_tamper_cannot_restore_an_approval(bound_session, fault):
    session = bound_session.factory()
    store = ChartApprovalStore(session)
    store.load()
    proposal = session.proposal('ac17')
    record = chart.confirm_chart_transfer(proposal, actor='Reviewer', reason='Finite fixture confirmation')
    store.save({'ac17': record}, {'ac17': proposal.payload.decode('utf8')}, {}, {'ac17': [True, True]})
    current = bound_session.factory()
    if fault == 'source-bytes':
        from pathlib import Path
        Path(bound_session.model['path']).write_text('changed source', encoding='utf8')
    elif fault == 'geometry':
        current.review['geometry_index']['sha256'] = 'f' * 64
    elif fault == 'map-version':
        current.review['source']['painting_facies']['sha256'] = 'f' * 64
    else:
        value = json.loads(store.path.read_bytes())
        payload = json.loads(value['payloads']['ac17'])
        payload['correction_mm'] = [1., 0.]
        value['payloads']['ac17'] = chart._json(payload).decode()
        store.path.write_bytes(chart._json(value))
    with pytest.raises(ValueError):
        ChartApprovalStore(current).load()


def test_a_second_open_view_cannot_overwrite_saved_decisions(bound_session):
    first, second = (ChartApprovalStore(bound_session.factory()) for _ in range(2))
    first.load(); second.load()
    first.save({}, {}, {'ac17': [1., 0.]}, {'ac17': [False, False]})
    with pytest.raises(ValueError, match='another view'):
        second.save({}, {}, {}, {})


def test_all_39_retained_decisions_restore_after_physical_relocation_without_reapproval(bound_session, tmp_path, monkeypatch):
    from darsia.fluidflower.storage_paths import refresh_storage_paths, resolve_storage_path
    from darsia.fluidflower.painting_chart_transfer_reader import proposal_for_approval

    monkeypatch.setenv('DARSIA_SERIES_REGISTRY', str(tmp_path.parent / (tmp_path.name + '-registry.toml')))
    session = bound_session.factory()
    session.decision_proposal_format = chart.DETECTED_FORMAT
    store = ChartApprovalStore(session)
    store.load()
    proposals = {run: session.proposal(run, proposal_format=chart.DETECTED_FORMAT) for run in chart.RUNS}
    approvals = {run: chart.confirm_chart_transfer(proposal, actor='synthetic reviewer',
        reason='Retained fixture review') for run, proposal in proposals.items()}
    payloads = {run: proposal.payload.decode() for run, proposal in proposals.items()}
    confirmations = {run: [True, True] for run in chart.RUNS}
    store.save(approvals, payloads, {}, confirmations)
    original = store.path.read_bytes()
    old = bound_session.root
    moved = tmp_path.parent / (tmp_path.name + '-Results') / 'setup/Geometry'
    moved.parent.mkdir(parents=True)
    old.rename(moved)
    series = tmp_path.parent / (tmp_path.name + '-DarSIA')
    series.mkdir()
    (series / 'series.toml').write_text(toml.dumps(dict(schema_version=1, series_id='sample',
        storage=dict(path_aliases={str(old): str(moved)}))), encoding='utf-8')
    refresh_storage_paths(series)
    current = bound_session.factory()
    current.decision_proposal_format = chart.DETECTED_FORMAT
    current.review['review_binding'] = _bound(current.review['review_binding']['path'])
    current.manifest_binding = _bound(current.manifest_binding['path'])
    reopened = ChartApprovalStore(current)
    loaded = reopened.load()
    assert loaded == dict(approvals=approvals, payloads=payloads, adjustments={}, confirmations=confirmations)
    assert len(loaded['approvals']) == 39
    assert reopened.path.read_bytes() == original

    # The small session fixture closes over authored descriptors. Present the
    # current reader's physical locators while keeping its actual replay result.
    authored_proposal = current.proposal
    def located(value):
        if isinstance(value, dict):
            if {'path', 'sha256', 'size_bytes'} <= value.keys():
                return dict(value, path=str(resolve_storage_path(value['path'])))
            return {key: located(item) for key, item in value.items()}
        return [located(item) for item in value] if isinstance(value, list) else value
    def physical_proposal(run, **options):
        fresh = authored_proposal(run, **options)
        metadata = fresh.metadata()
        metadata['source_binding'] = located(metadata['source_binding'])
        return chart.ChartTransferProposal(chart._json(metadata), fresh.result, fresh.display_outlines)
    current.proposal = physical_proposal
    current._approval_store, current._approval_state = reopened, loaded
    for run in chart.RUNS:
        fresh = physical_proposal(run, proposal_format=chart.DETECTED_FORMAT)
        assert fresh.payload != payloads[run].encode()
        confirmed = proposal_for_approval(current, run, approvals[run])
        assert confirmed.payload == payloads[run].encode()
        assert confirmed.metadata()['result'] == proposals[run].metadata()['result']
        chart.apply_chart_transfer(confirmed, approvals[run])
    assert reopened.path.read_bytes() == original

    # An ordinary replay consumer without the dialog's cached state also uses
    # the original decision policy recorded by that exact human approval.
    del current._approval_store, current._approval_state, current.decision_proposal_format
    assert proposal_for_approval(current, chart.RUNS[0], approvals[chart.RUNS[0]]).payload == payloads[chart.RUNS[0]].encode()
    fresh = physical_proposal(chart.RUNS[0], proposal_format=chart.DETECTED_FORMAT)
    metadata = fresh.metadata()
    metadata['motion']['dx_mm'] += 1.
    changed_result = chart.ChartTransferProposal(chart._json(metadata), fresh.result, fresh.display_outlines)
    with pytest.raises(ValueError, match='changed beyond'):
        reopened.confirmed_proposal(changed_result, approvals[chart.RUNS[0]], saved_state=loaded)

    foreign = series / 'same-bytes-different-review.json'
    foreign.write_bytes((moved / 'review.json').read_bytes())
    wrong_review = bound_session.factory()
    wrong_review.decision_proposal_format = chart.DETECTED_FORMAT
    wrong_review.review['review_binding'] = _bound(foreign)
    with pytest.raises(ValueError, match='another review'):
        ChartApprovalStore(wrong_review, path=reopened.path).load()
    changed = bound_session.factory()
    changed.decision_proposal_format = chart.DETECTED_FORMAT
    changed.review['runs'][0]['run_id'] = 'foreign-run'
    with pytest.raises(ValueError, match='another review'):
        ChartApprovalStore(changed).load()
    assert reopened.path.read_bytes() == original


def test_new_detected_review_keeps_historical_decisions_separate_and_needs_new_approval(bound_session):
    legacy_session = bound_session.factory()
    legacy = ChartApprovalStore(legacy_session)
    legacy.load()
    old_proposal = legacy_session.proposal('ac18')
    old_approval = chart.confirm_chart_transfer(old_proposal, actor='Reviewer', reason='Historical fixture review')
    legacy.save({'ac18': old_approval}, {'ac18': old_proposal.payload.decode()}, {}, {'ac18': [True, True]})
    old_bytes = legacy.path.read_bytes()

    session = bound_session.factory()
    session.decision_proposal_format = chart.DETECTED_FORMAT
    current = ChartApprovalStore(session)
    assert current.path != legacy.path
    assert current.load()['approvals'] == {}
    with pytest.raises(ValueError, match='Historical chart decisions'):
        current.save({'ac18': old_approval}, {'ac18': old_proposal.payload.decode()}, {}, {'ac18': [True, True]})
    proposal = session.proposal('ac18', proposal_format=chart.DETECTED_FORMAT)
    assert not old_proposal.metadata()['moved'] and proposal.metadata()['moved']
    assert not (old_proposal.result['chart_mask'] == proposal.result['chart_mask']).all()
    approval = chart.confirm_chart_transfer(proposal, actor='Reviewer', reason='New detected whole-chart fixture review')
    current.save({'ac18': approval}, {'ac18': proposal.payload.decode()}, {}, {'ac18': [True, True]})
    assert ChartApprovalStore(session).load()['approvals'] == {'ac18': approval}
    assert ChartApprovalStore(legacy_session).load()['approvals'] == {'ac18': old_approval}
    assert legacy.path.read_bytes() == old_bytes


@pytest.mark.parametrize('fault', [None, 'source-bytes', 'foreign-geometry'])
def test_fresh_findings_reopen_original_v3_source_and_keep_all_39_decisions(bound_session, monkeypatch, fault):
    from pathlib import Path
    from darsia.fluidflower import painting_chart_transfer_reader as reader
    from darsia.fluidflower import painted_analysis_exclusions as gaps

    session = bound_session.factory()
    session.proposal_format = session.decision_proposal_format = chart.DETECTED_FORMAT
    store = ChartApprovalStore(session)
    store.load()
    proposals = {run: session.proposal(run, proposal_format=chart.DETECTED_FORMAT) for run in chart.RUNS}
    approvals = {run: chart.confirm_chart_transfer(proposal, actor='synthetic human',
        reason='Explicit whole-chart fixture review') for run, proposal in proposals.items()}
    store.save(approvals, {run: proposal.payload.decode() for run, proposal in proposals.items()}, {},
        {run: [True, True] for run in proposals})
    saved_bytes = store.path.read_bytes()
    original = session.review
    current = {key: deepcopy(value) for key, value in original.items() if key != 'review_binding'}
    current['chart_source_review'] = deepcopy(original['review_binding'])
    if fault == 'source-bytes':
        path = Path(original['review_binding']['path'])
        path.write_bytes(path.read_bytes() + b' ')
    elif fault == 'foreign-geometry':
        foreign = {key: deepcopy(value) for key, value in original.items() if key != 'review_binding'}
        foreign['geometry_index']['sha256'] = 'f' * 64
        path = bound_session.root / 'foreign-review.json'
        path.write_text(json.dumps(foreign), encoding='utf8')
        current['chart_source_review'] = _bound(path)
    fresh = bound_session.root / 'annotations/map_adoption/fresh/gap_review.json'
    fresh.parent.mkdir(parents=True)
    fresh.write_text(json.dumps(current), encoding='utf8')
    binding = _bound(fresh)

    def metadata(path):
        bound = _bound(path)
        value = gaps._checked_gap_review(bound)
        value['review_binding'] = bound
        return value

    def construct(review, *, proposal_format):
        assert review == original
        result = bound_session.factory()
        result.proposal_format = result.decision_proposal_format = proposal_format
        return result

    monkeypatch.setattr(gaps, '_read_scientific_gap_review', metadata)
    monkeypatch.setattr(reader, 'ChartTransferSession', construct)
    if fault:
        with pytest.raises(ValueError, match='source review changed|approved map, geometry'):
            reader.selected_chart_session(bound_session.root, bound_session.record,
                bound_session.selection, sources=[binding])
    else:
        reopened = reader.selected_chart_session(bound_session.root, bound_session.record,
            bound_session.selection, sources=[binding])
        assert reopened.review['review_binding'] == original['review_binding']
        assert reopened.proposal_format == chart.DETECTED_FORMAT
        assert ChartApprovalStore(reopened).load()['approvals'] == approvals
        assert len(approvals) == 39
    assert store.path.read_bytes() == saved_bytes
