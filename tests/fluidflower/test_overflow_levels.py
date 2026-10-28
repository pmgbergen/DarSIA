"""Finite role/group/F5 policy, not actual current measurement qualification."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from darsia.fluidflower.mass import overflow_levels as levels, overflow_method as method
from darsia.fluidflower.mass.n1_pipeline import DataReader, asset, _write_json

# The test series' recorded method settings: pinned sources and one authorization.
SETTINGS = method.method_settings(dict(overflow_method=dict(
    protocol=dict(sha256='0535b5e15dc1a9318c92560c36d28fa5f1f4a7d97cdcebb5c59cd9b61d318dff', size_bytes=14141),
    addendum=dict(sha256='01c45bb70cc9a755dde65d9e2084cb776f6c874b31535496a4f010f061fa1c55', size_bytes=7321),
    overflow_review=dict(sha256='f39dd5568d110c4dbbf45855d93573198838b7b47171b3b42dc4251bdebba8ee', size_bytes=32518),
    records={'decision-0345-overflow-method': '0d86f03cd087f5acb7d4203210156d8d60221c87982b0053165591cea042d682'},
    authorizations=[dict(actor='Test decider', confirmed_utc='2026-10-21T10:51:53Z',
                         reason='Use each admitted overflow measurement for its run.')])))


def inputs():
    analysis = sorted(set(method.USABLE_RUNS) | set(method.NO_USABLE_OVERFLOW))
    roles = dict(heldout=list(method.HELDOUT_RUNS), descriptive=['ac38'],
                 excluded=list(method.EXCLUDED_RUNS))
    roles['fit'] = [run for run in analysis if run not in roles['heldout']+roles['descriptive']]
    own = [run for run in method.USABLE_RUNS if run not in method.CONDITIONAL_ANCHOR_RUNS]
    observations = {run:dict(ratio=.6+.01*index, sigma=.01, selected_cases=[run+'-current-1h'])
                    for index,run in enumerate(own)}
    fallback = {run:dict(code=levels.NO_OVERFLOW, reason='Original reviewed data are unusable.')
                for run in method.NO_USABLE_OVERFLOW}
    fallback.update({run:dict(code=levels.ANCHOR_UNPROVED, reason='No authenticated post-sampling anchor.')
                     for run in method.CONDITIONAL_ANCHOR_RUNS})
    return observations, fallback, roles, analysis


def test_conditional_anchor_exclusion_records_actual13_without_fabricating_fourteenth():
    observations, fallback, roles, analysis = inputs()
    original = deepcopy((observations, fallback, roles, analysis))
    result = levels.observations_to_levels(observations, fallback, roles=roles, analysis_runs=analysis)
    assert result['historical_group_candidate_count'] == 14
    assert result['actual_group_fit_count'] == len(result['group_fit_runs']) == 13
    assert result['group_fit_exclusions'] == [dict(run='ac34', **fallback['ac34'])]
    assert set(result['group_fit_runs']) == set(method.GROUP_CANDIDATES)-{'ac34'}
    assert not set(result['group_fit_runs']) & set(roles['heldout']+roles['excluded']+roles['descriptive'])
    assert result['runs']['ac34']['group_fallback'] and result['runs']['ac41']['group_fallback']
    assert all(result['runs'][run]['ratio'] is None for run in method.CONDITIONAL_ANCHOR_RUNS)
    assert len(result['runs']) == 39 and result['group_recomputed_per_fold'] is False
    assert result['calibration_admitted'] is result['production_admitted'] is False
    assert (observations, fallback, roles, analysis) == original


def test_heldout_measurement_never_changes_global_group_or_any_fit_run_level():
    observations, fallback, roles, analysis = inputs()
    before = levels.observations_to_levels(observations, fallback, roles=roles, analysis_runs=analysis)
    observations['ac19']['ratio'] = 100.
    after = levels.observations_to_levels(observations, fallback, roles=roles, analysis_runs=analysis)
    assert after['group_ratio'] == before['group_ratio']
    assert after['between_run_variance'] == before['between_run_variance']
    assert all(after['runs'][run] == before['runs'][run] for run in roles['fit'])
    assert after['runs']['ac19']['level'] == 1. and after['runs']['ac19']['ratio_above_one']


def test_actual_unusable_current_volume_is_explicitly_excluded_without_replacing_it():
    observations, fallback, roles, analysis = inputs()
    observations.pop('ac28')
    fallback['ac28'] = dict(code=levels.NO_CURRENT_VOLUME, reason='No current selected gas volume reaches 3 mL.')
    result = levels.observations_to_levels(observations, fallback, roles=roles, analysis_runs=analysis)
    assert result['actual_group_fit_count'] == 12
    assert {row['run'] for row in result['group_fit_exclusions']} == {'ac28', 'ac34'}
    assert result['runs']['ac28']['group_fallback'] is True
    assert result['historical_group_candidates'] == list(method.GROUP_CANDIDATES)


@pytest.mark.parametrize('fault', ['invented_anchor', 'unexplained_omission', 'wrong_fallback'])
def test_ordinary_qa_or_missing_membership_cannot_bypass_the_actual_fallback(fault):
    observations, fallback, roles, analysis = inputs()
    if fault == 'invented_anchor':
        observations['ac34'] = dict(ratio=.8, sigma=.1, selected_cases=['ac34-current'])
        fallback.pop('ac34')
    elif fault == 'unexplained_omission':
        fallback.pop('ac17')
    else:
        fallback['ac34']['code'] = levels.NO_OVERFLOW
    with pytest.raises(ValueError):
        levels.observations_to_levels(observations, fallback, roles=roles, analysis_runs=analysis)


def test_unrecognised_claimed_anchor_refuses_before_any_source_can_be_opened():
    _observations, _fallback, roles, analysis = inputs()
    with pytest.raises(ValueError, match='supported original post-sampling anchor'):
        levels.read_measurements(DataReader(), {}, {}, {}, roles=roles, analysis_runs=analysis,
            sampling_anchor_evidence={'ac34': {'post_sampling': True, 'qa_passed': True}})


def test_actual_method_sources_do_not_admit_a_different_run_scale_document(tmp_path):
    _observations, _fallback, roles, analysis = inputs()
    fixture = Path(__file__).parents[1]/'fixtures'/'overflow_method'
    value = method.method_document(protocol=asset(fixture/'round5_protocol.md'),
        addendum=asset(fixture/'overflow_addendum.md'), review=asset(fixture/'overflow_review.json'),
        settings=SETTINGS)
    method_bound = _write_json(tmp_path/'method.json', value)
    scale = _write_json(tmp_path/'different-scale.json', dict(metadata=dict(run='another-run',
        injection_start='2023-01-01T00:00:00'), records=[]))
    with pytest.raises(ValueError, match='belongs to another run'):
        levels.read_measurements(DataReader(), method_bound,
            {run:scale for run in method.USABLE_RUNS}, {run:[] for run in method.USABLE_RUNS},
            roles=roles, analysis_runs=analysis)


def test_actual_reviewed_scale_join_retains_original_qa_and_empty_current_selection_fallback():
    # Genuine reviewed metadata/feed, with explicitly finite current-volume
    # specimens. These volumes are not a completed P1/campaign authority.
    fixture = Path(__file__).parents[1]/'fixtures'/'overflow_method'
    scale_path = fixture/'ac19_overflow_scale_review.json'
    review_path = fixture/'overflow_review.json'
    before = {path:path.read_bytes() for path in (scale_path, review_path)}
    scale, review = asset(scale_path), asset(review_path)
    frames = [dict(case_id='finite-current-1h', elapsed_h=1., baseline_gas_volume_m3=100e-6),
              dict(case_id='finite-current-2h', elapsed_h=2., baseline_gas_volume_m3=100e-6)]
    result = levels.read_run_measurement(DataReader(), scale, frames, review=review, run='ac19')
    original = next(row for row in json.loads(before[review_path])['rows'] if row['run'] == 'ac19')
    assert scale['sha256'] == 'bdbbc2868e0e172447dbb3af9b3b23c9ef16fbe9ac46e194e6d7a6b348097bc6'
    assert result['source']['scale_review'] == scale
    assert result['source']['original_feed_qa'] == {
        key:original[key] for key in ('t_end', 'late_sd', 'drift', 'vA_pre', 'q_pre')}
    assert result['source']['original_feed_anchor']['time_from_injection_h'] < 0
    assert result['measurement']['selected_cases'] == [row['case_id'] for row in frames]
    assert result['measurement']['ratio'] > 0 and result['measurement']['sigma'] > 0
    assert result['fallback'] is None
    empty = levels.read_run_measurement(DataReader(), scale, [], review=review, run='ac19')
    assert empty['source'] == result['source'] and empty['measurement'] is None
    assert empty['fallback']['code'] == levels.NO_CURRENT_VOLUME
    with pytest.raises(ValueError, match='explicit frame list'):
        levels.read_run_measurement(DataReader(), scale, (), review=review, run='ac19')
    assert all(path.read_bytes() == raw for path,raw in before.items())


def test_newly_bound_same_run_scale_cannot_change_original_reviewed_feed_qa(tmp_path):
    fixture = Path(__file__).parents[1]/'fixtures'/'overflow_method'
    original_path = fixture/'ac19_overflow_scale_review.json'
    original_raw = original_path.read_bytes()
    changed = json.loads(original_raw)
    for row in changed['records']:
        if row.get('status') in ('Verified', 'Corrected') and row.get('weight_g') is not None:
            row['weight_g'] *= .5
    scale = _write_json(tmp_path/'newly-bound-scale.json', changed)
    assert scale['sha256'] != hashlib.sha256(original_raw).hexdigest()
    with pytest.raises(ValueError, match='original accepted feed QA'):
        levels.read_run_measurement(DataReader(), scale, [],
            review=asset(fixture/'overflow_review.json'), run='ac19')
    assert original_path.read_bytes() == original_raw
