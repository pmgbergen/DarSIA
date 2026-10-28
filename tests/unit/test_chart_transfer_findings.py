"""Current approved chart masks survive findings refresh and the genuine Check."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import painted_adoption_table as table
from darsia.fluidflower import painted_analysis_exclusions as gaps
from darsia.fluidflower import painted_transfer_findings as findings
from darsia.fluidflower import painting_chart_transfer as chart
from darsia.fluidflower import painting_chart_transfer_reader as reader
from darsia.fluidflower.local_map_adoption import prepare_local_adoption, check_local_adoption
from darsia.fluidflower.painted_target_integration import effective_product
from darsia.fluidflower.painted_transfer_replay import replay_transfer_review

from .test_painted_analysis_exclusions import ACTION, _approve
from .test_painted_adoption_table import (  # noqa: F401
    imported, prepared_painting, registered_setup, synthetic_payload_admission,
)
from .test_painted_transfer_replay import _partial_target, _retained, _unchanged


def test_exact_manual_map_approval_replaces_advisory_reapproval(
        registered_setup, tmp_path, monkeypatch):
    from darsia.fluidflower.manual_map_approval import verify_manual_map_approval, approved_candidate
    _, _, prepared, _ = _current_chart_case(registered_setup, tmp_path, monkeypatch)
    # No findings catalogue or RAW decoder is needed to recognize the saved
    # human decision. The actual Check still verifies all numeric products.
    monkeypatch.setattr(findings, 'collect_transfer_findings',
                        lambda *a, **k: pytest.fail('A second findings catalogue is not needed'))
    assert verify_manual_map_approval(prepared)
    candidate, _, checked = check_local_adoption(registered_setup.revision,
        prepared['folder'], facies_source_acquisition=None)
    assert approved_candidate(candidate)
    assert not gaps.findings_resolutions_required(candidate)
    report = checked.report_for(candidate)
    assert not report['summary']['error_count']
    assert any(row['severity'] == 'warning' for row in report['issues'])
    # A missing or altered approval never acquires authority from eligibility.
    changed = deepcopy(prepared)
    run = changed['runs'][0]['run_id']
    changed['chart_review']['runs'][run]['approval']['proposal_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='confirmation'):
        verify_manual_map_approval(changed)
    changed = deepcopy(prepared)
    del changed['chart_review']['runs'][run]
    assert not verify_manual_map_approval(changed)


def test_approved_exact_maps_do_not_repeat_advisory_boundary_screen(
        registered_setup, tmp_path, monkeypatch):
    from darsia.fluidflower import geometry_facies as geometry
    from darsia.fluidflower.map_adoption import adoption_candidate
    _, _, prepared, _ = _current_chart_case(registered_setup, tmp_path, monkeypatch)
    candidate, _ = adoption_candidate(registered_setup.revision, prepared['folder'])
    called = []
    original_screen = geometry.screen_layer_boundary_change

    def screen(*args, **kwargs):
        called.append(True)
        return original_screen(*args, **kwargs)

    monkeypatch.setattr(geometry, 'screen_layer_boundary_change', screen)
    report = geometry._geometry_input_report(candidate)
    assert called == []
    decisions = report['summary']['run_geometry_change_diagnostics']
    assert len(decisions) == len(prepared['runs'])
    assert all(row['status'] == 'manually_approved' for row in decisions)
    assert all(row['authority']['review'] == prepared['gap_review'] for row in decisions)
    assert report['summary']['screened_run_geometry_count'] == 0

    # Structural checks still reject invalid physical properties.
    invalid = deepcopy(candidate)
    porous = next(row for row in invalid['regions'] if row['analysis_role'] == 'porous_medium')
    porous['porosity'] = 0
    assert any(issue['code'] == 'invalid_porosity'
               for issue in geometry._geometry_input_report(invalid)['issues'])
    assert called == []

    # Without the bound manual approval this remains the ordinary screen.
    ordinary = deepcopy(candidate)
    del ordinary['painted_adoption_table']
    geometry._geometry_input_report(ordinary)
    assert len(called) == len(prepared['runs'])

    # Same bytes at a different, unapproved run-map locator are not authority.
    called.clear()
    changed = deepcopy(candidate)
    row = changed['run_geometry']['runs'][0]
    copied = tmp_path / 'unapproved-copy.npz'
    copied.write_bytes(Path(row['labels_path']).read_bytes())
    row['labels_path'] = str(copied)
    geometry._geometry_input_report(changed)
    assert len(called) == 1


def _current_chart_case(data, tmp_path, monkeypatch, *, legacy=False, keep_gap=False, prior_target=False,
                        nested_guard=False):
    from types import SimpleNamespace
    from darsia.fluidflower.layer_painting_fixtures import read_underlay
    from darsia.fluidflower.rig_point_alignment import effective_model_matrix

    with pytest.raises(gaps.GapReviewRequired) as stopped:
        prepared_painting(data, tmp_path, fixture=True, continuation=True,
            outside=True, outside_targets=True)
    original = stopped.value.review
    if prior_target:
        from .test_painted_transfer_replay import build_residual_transfer_review
        _, _, binding, *_ = _partial_target(SimpleNamespace(review=original, root=tmp_path))
        replay = replay_transfer_review(original, source_context=original['preparation_context'], target_bindings=[binding])
        original = build_residual_transfer_review(replay, tmp_path / 'prior_target_review')
    retained = _retained(SimpleNamespace(review=original))
    drawing, *_ = table.verify_version(original['source']['painting_facies'])
    source = read_underlay(original['preparation_context']['source_fixture_underlay'])
    material = source.analysis_layers.copy()
    material[source.fixture_mask] = source.drawn_layers[source.fixture_mask]
    centres = np.asarray([[x, -y] for y in range(6) for x in range(4)], np.float64)
    rows = {row['run_id']: row for row in original['runs']}
    nested = tmp_path / 'nested-chart-input.bin'
    if nested_guard:
        nested.write_bytes(b'original bound nested chart source')

    class Session:
        proposal_format = chart.DETECTED_FORMAT

        def __init__(self, *args, **kwargs):
            pass

        def proposal(self, run, *, correction_mm=None, proposal_format=chart.DETECTED_FORMAT):
            row = rows[run]
            fields = gaps._arrays(row['product'])
            model = gaps._checked(row['target_model'], metadata=True)
            native = np.asarray(model['baseline_native_grid']['H_baseline_camera_u_to_rig_corrected_native'])
            metric = np.linalg.inv(effective_model_matrix(model, original['geometry_index'])) @ np.linalg.inv(native)
            binding = dict(review=original['review_binding'], painting_facies=original['source']['painting_facies'],
                target_model=row['target_model'])
            if nested_guard:
                binding['nested_sources'] = dict(original['source_map'], child=gaps._bound(nested))
            return chart.propose_chart_transfer(run, fields['array'], fields['inside'], fields['array'] == 8,
                material, source.fixture_mask, np.linalg.inv(row['native_to_painting_h']), metric,
                centres, centres, binding, chart_id=8,
                known_ids=[layer.layer_id for layer in drawing.layers], correction_mm=correction_mm,
                proposal_format=proposal_format)

    monkeypatch.setattr(reader, 'ChartTransferSession', Session)
    approvals = {run: chart.confirm_chart_transfer(Session().proposal(run,
        correction_mm=[10000., 0.] if keep_gap else None), actor='synthetic human',
        reason='Explicit new whole-chart fixture review') for run in rows}
    with monkeypatch.context() as patch:
        if legacy:
            # Reproduce the former writer at publication; never rebind an
            # already published review to manufacture another authority.
            dump = json.dump
            def former_writer(value, stream, **kwargs):
                if isinstance(value, dict) and value.get('chart_source_review'):
                    value = {key: item for key, item in value.items() if key != 'chart_preparation'}
                return dump(value, stream, **kwargs)
            patch.setattr(json, 'dump', former_writer)
        try:
            prepared = prepare_local_adoption(original['review_binding'],
                source_context=original['preparation_context'], destination=tmp_path / 'current_chart',
                chart_transfers=approvals)
        except gaps.GapReviewRequired as error:
            assert keep_gap
            prepared, current = None, error.review
        else:
            current = gaps.read_gap_review(prepared['gap_review']['path'])
    return original, retained, prepared, current


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('prior_target', [False, True])
def test_current_chart_findings_source_choice_and_check_keep_the_current_masks(
        registered_setup, tmp_path, monkeypatch, legacy, prior_target):
    data = registered_setup
    original, retained, prepared, current = _current_chart_case(data, tmp_path, monkeypatch,
        legacy=legacy, prior_target=prior_target)
    assert sum(row['gap']['pixels'] for row in original['runs']) > 0
    assert sum(row['gap']['pixels'] for row in current['runs']) == 0
    revision = table._source(data.revision)['revision']
    unknown = findings.collect_transfer_findings(revision, current['review_binding'],
        source_context=current['preparation_context'], residual_destination=tmp_path / 'unknown_inputs')
    action = dict(ACTION, acquisition_state='indicator')
    selected = findings.collect_transfer_findings(revision, current['review_binding'],
        source_context=current['preparation_context'], explicit_actions=dict(source_choices=[action]),
        residual_destination=tmp_path / 'indicator_inputs')
    assert selected['review'] == unknown['review'] == current
    assert any(issue['code'] == 'run_geometry_acquisition_unknown' for issue in unknown['report']['issues'])
    assert not any(issue['code'] == 'run_geometry_acquisition_unknown' for issue in selected['report']['issues'])
    assert any(issue['code'] == 'indicator_facies_carryover_risk' for issue in selected['report']['issues'])
    assert selected['report']['input_bindings']['source_choices'] == [action]
    assert selected['report']['unresolved_domains'] == []
    assert selected['report']['table_data'] == unknown['report']['table_data']
    assert [row['target_correction_bindings'] for row in current['runs']] == [
        row['target_correction_bindings'] for row in original['runs']]
    copies = replay_transfer_review(current, source_context=current['preparation_context'], retain_source_arrays=False)
    for entry, replayed, row in zip(prepared['runs'], copies['runs'], current['runs']):
        assert replayed['base_arrays'] is None and replayed['reviewed_arrays'] is None
        effective, bound = effective_product(prepared, prepared['folder'], entry)
        assert bound == row['product']
        assert {key: gaps._identity(value) for key, value in replayed['effective_arrays'].items()} == row['scientific_arrays']
        for field in effective:
            np.testing.assert_array_equal(replayed['effective_arrays'][field], effective[field])
        assert not replayed['residual_mask'].any()
    pins = {item['path'] for item in selected['report']['source_bindings']['replay']['files']}
    assert current['review_binding']['path'] in pins and original['review_binding']['path'] in pins
    assert all(row['product']['path'] in pins for row in current['runs'])
    if legacy:
        assert str(Path(prepared['folder']) / 'index.json') in pins
    declaration = gaps.make_facies_source_acquisition(prepared, 'indicator', **ACTION)
    candidate, _, checked = check_local_adoption(data.revision, prepared['folder'],
        facies_source_acquisition=declaration)
    report = checked.report_for(candidate)
    assert not report['summary']['error_count']
    assert not any(issue['code'] == 'run_geometry_acquisition_unknown' for issue in report['issues'])
    assert candidate['provenance']['preparation']['path'] == str(Path(prepared['folder']) / 'index.json')
    _unchanged(retained)


def test_current_chart_review_before_index_keeps_real_unresolved_gaps(registered_setup, tmp_path, monkeypatch):
    original, retained, prepared, current = _current_chart_case(registered_setup, tmp_path, monkeypatch, keep_gap=True)
    assert prepared is None and current['chart_preparation']
    assert not (Path(current['review_binding']['path']).parent / 'index.json').exists()
    result = findings.collect_transfer_findings(table._source(registered_setup.revision)['revision'],
        current['review_binding'], source_context=current['preparation_context'])
    assert result['review'] == current
    actual = {(row['run_id'], item['component_id']) for row in current['runs'] for item in row['components']}
    assert actual
    assert {(row['run_id'], row['component_id']) for row in result['report']['unresolved_domains']
            if row['kind'] == 'analysis_gap'} == actual
    assert result['report']['summary']['error_count'] > 0
    approval = _approve(current)
    confirmed = findings.collect_transfer_findings(table._source(registered_setup.revision)['revision'],
        current['review_binding'], source_context=current['preparation_context'],
        explicit_actions=dict(analysis_exclusions=approval), residual_destination=tmp_path / 'confirmed_current_gaps')
    assert confirmed['review'] == current
    assert confirmed['report']['unresolved_domains'] == []
    assert not confirmed['report']['summary']['error_count']
    _unchanged(retained)


def test_fresh_chart_chain_never_substitutes_the_older_base_mask_review(
        registered_setup, tmp_path, monkeypatch):
    from darsia.fluidflower.photo_edge_continuation import FRAME_RASTER

    original, retained, prepared, current = _current_chart_case(
        registered_setup, tmp_path, monkeypatch)
    assert all(row['contract']['frame_raster'] == FRAME_RASTER for row in original['runs'])
    assert all(not row['target_correction_bindings'] for row in original['runs'])
    assert current['chart_source_review'] == original['review_binding']
    assert current['chart_preparation']
    # Exercise the table's real restored-layer case, not a chart move that
    # happens to leave the base map's label-value list valid.
    changed = [row for row in current['runs']
               if set(np.unique(gaps._arrays(row['product'])['array']))
               != set(np.unique(gaps._arrays(row['base_product'])['array']))]
    assert changed

    replay = replay_transfer_review(current['review_binding'],
        source_context=current['preparation_context'])
    for row, replayed in zip(current['runs'], replay['runs']):
        assert {key: gaps._identity(value) for key, value
                in replayed['effective_arrays'].items()} == row['scientific_arrays']
    result = findings.collect_transfer_findings(table._source(registered_setup.revision)['revision'],
        current['review_binding'], source_context=current['preparation_context'],
        explicit_actions=dict(source_choices=[dict(ACTION, acquisition_state='indicator')]),
        residual_destination=tmp_path / 'fresh_current_findings')
    assert result['review'] == current
    assert not result['report']['summary']['error_count']
    assert result['report']['unresolved_domains'] == []

    declaration = gaps.make_facies_source_acquisition(prepared, 'indicator', **ACTION)
    candidate, _, checked = check_local_adoption(registered_setup.revision, prepared['folder'],
        facies_source_acquisition=declaration)
    assert not checked.report_for(candidate)['summary']['error_count']

    # Both reviews remain authentic and readable. A stale pointer must still
    # fail: old base-mask authority cannot validate the newly restored labels.
    index = Path(prepared['folder']) / 'index.json'
    published = index.read_bytes()
    mixed = json.loads(published)
    mixed['gap_review'] = original['review_binding']
    index.write_text(json.dumps(mixed), encoding='utf8')
    try:
        with pytest.raises(ValueError, match='effective chart labels differ from their exact reviewed pixels'):
            check_local_adoption(registered_setup.revision, prepared['folder'],
                facies_source_acquisition=declaration)
    finally:
        index.write_bytes(published)
    _unchanged(retained)


@pytest.mark.parametrize('fault', ['original_review', 'chart_product', 'legacy_recipe', 'nested_descriptor'])
def test_current_chart_replay_refuses_changed_bound_inputs(registered_setup, tmp_path, monkeypatch, fault):
    original, _, prepared, current = _current_chart_case(registered_setup, tmp_path, monkeypatch,
        legacy=fault == 'legacy_recipe', nested_guard=fault == 'nested_descriptor')
    if fault == 'original_review':
        path = Path(original['review_binding']['path'])
    elif fault == 'chart_product':
        path = Path(next(row['product']['path'] for row in current['runs'] if row['product'] != row['base_product']))
    elif fault == 'nested_descriptor':
        path = tmp_path / 'nested-chart-input.bin'
    else:
        path = Path(prepared['folder']) / 'index.json'
        replacement = json.loads(path.read_text(encoding='utf8'))
        replacement['gap_review'] = original['review_binding']
        path.write_text(json.dumps(replacement), encoding='utf8')
    if fault != 'legacy_recipe':
        path.write_bytes(path.read_bytes() + b' changed bound scientific source')
    with pytest.raises(ValueError, match='changed|differ|bind'):
        replay_transfer_review(current, source_context=current['preparation_context'])


def test_new_target_edit_cannot_silently_rebase_an_existing_exact_chart_approval(
        registered_setup, tmp_path, monkeypatch):
    from types import SimpleNamespace
    original, retained, _, current = _current_chart_case(registered_setup, tmp_path, monkeypatch, keep_gap=True)
    _, _, binding, *_ = _partial_target(SimpleNamespace(review=current, root=tmp_path))
    destination = tmp_path / 'must_not_publish_rebased_chart'
    with pytest.raises(ValueError, match='source|prior|supported|ColorChecker|changed'):
        findings.collect_transfer_findings(table._source(registered_setup.revision)['revision'],
            current['review_binding'], source_context=current['preparation_context'],
            target_correction_bindings=[binding], residual_destination=destination)
    assert not destination.exists()
    _unchanged(retained)
