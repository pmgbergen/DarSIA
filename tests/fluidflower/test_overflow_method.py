"""Exact recorded method/roles only; no source campaign, F preparation or fit.

The small historical documents are source fixtures with neutral wording. The
test series names who authorized the method and pins its sources; private
answer fixtures only qualify conversion to the ordinary human-form method
record and are never included in that product record or an actual publication.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from darsia.fluidflower.mass import overflow_method as method
from darsia.fluidflower.mass.n1_pipeline import DataReader, asset, _write_json

FIXTURES = Path(__file__).parents[1]/'fixtures'/'overflow_method'
PROTOCOL_PIN = dict(sha256='0535b5e15dc1a9318c92560c36d28fa5f1f4a7d97cdcebb5c59cd9b61d318dff', size_bytes=14141)
DECISIONS = dict(overflow_method=dict(
    protocol=PROTOCOL_PIN,
    addendum=dict(sha256='01c45bb70cc9a755dde65d9e2084cb776f6c874b31535496a4f010f061fa1c55', size_bytes=7321),
    overflow_review=dict(sha256='f39dd5568d110c4dbbf45855d93573198838b7b47171b3b42dc4251bdebba8ee', size_bytes=32518),
    records={'decision-0345-overflow-method': '0d86f03cd087f5acb7d4203210156d8d60221c87982b0053165591cea042d682',
             'decision-0444-overflow-variant': 'a3c9aac1807b7b193384cde7025844a19d963934b6187bd4d22ec19137a0e353'},
    authorizations=[
        dict(actor='Test decider', confirmed_utc='2026-10-21T10:51:53Z',
             reason='Use each admitted overflow measurement for its run, and the fit-run group otherwise.',
             conditional_anchor_rule='Use the group for the conditional anchor runs unless the feed anchor is proved later.'),
        dict(actor='Test decider', confirmed_utc='2026-10-22T08:36:22Z',
             reason='Count phase-labelled gas without its own probability as certain gas.')]))
SETTINGS = method.method_settings(DECISIONS)


def recorded_method():
    return method.method_document(protocol=asset(FIXTURES/'round5_protocol.md'),
        addendum=asset(FIXTURES/'overflow_addendum.md'), review=asset(FIXTURES/'overflow_review.json'),
        settings=SETTINGS)


def roles():
    analysis = sorted(set(method.USABLE_RUNS) | set(method.NO_USABLE_OVERFLOW))
    result = dict(heldout=list(method.HELDOUT_RUNS), descriptive=['ac38'],
                  excluded=list(method.EXCLUDED_RUNS))
    result['fit'] = [run for run in analysis if run not in result['heldout']+result['descriptive']]
    return result, analysis


def test_private_existing_decisions_produce_only_plain_human_method_record(tmp_path):
    decisions = json.loads((FIXTURES/'private_recorded_decisions.json').read_bytes())
    original = deepcopy(decisions)
    assert method.check_recorded_decisions(decisions, SETTINGS['records']) is None
    value = recorded_method()
    assert method.validate_method(value) == value
    assert method.validate_method(value, settings=SETTINGS) == value
    assert value['authorizations'][0]['actor'] == 'Test decider'
    assert value['authorizations'][1]['confirmed_utc'] == '2026-10-22T08:36:22Z'
    raw = json.dumps(value, ensure_ascii=False)
    for private in decisions:
        assert private['id'] not in raw and private['answer'] not in raw
    assert 'decided_by' not in raw and 'private_record' not in raw
    assert value['recompute_group_per_fold'] is False
    assert value['fit_gas_gain'] is False and value['phase_labels_modified'] is False
    assert value['calibration_admitted'] is False and value['production_admitted'] is False
    assert decisions == original
    # Source-only conversion never implicitly publishes even the clean record.
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('fault', ['missing', 'changed', 'duplicate'])
def test_private_decision_drift_is_refused_instead_of_inferred(fault):
    rows = json.loads((FIXTURES/'private_recorded_decisions.json').read_bytes())
    if fault == 'missing':
        rows.pop()
    elif fault == 'changed':
        rows[0]['answer'] += ' '
    else:
        rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError, match='unchanged recorded'):
        method.check_recorded_decisions(rows, SETTINGS['records'])


def test_clean_method_rechecks_exact_raw_markdown_and_review_sources(tmp_path):
    value = recorded_method()
    bound = _write_json(tmp_path/'ordinary_method.json', value)
    actual = method.checked_method(DataReader(), bound)
    assert actual == value == method.checked_method(DataReader(), bound, settings=SETTINGS)
    assert actual['protocol']['sha256'] == PROTOCOL_PIN['sha256']
    assert actual['historical_group_candidates'] == list(method.GROUP_CANDIDATES)
    assert len(actual['historical_group_candidates']) == 14
    assert actual['conditional_anchor_runs'] == ['ac34', 'ac41']
    assert actual['certain_gas_probability'] == 1.


@pytest.mark.parametrize('fault', ['human', 'rule', 'source', 'extra'])
def test_method_cannot_grant_new_rule_or_changed_human_authority(fault):
    value = recorded_method()
    if fault == 'human':
        value['authorizations'][0]['actor'] = 'another person'
    elif fault == 'rule':
        value['recompute_group_per_fold'] = True
    elif fault == 'source':
        value['protocol']['sha256'] = 'f'*64
    else:
        value['approve_production'] = True
    with pytest.raises(ValueError):
        method.validate_method(value, settings=SETTINGS)


@pytest.mark.parametrize('fault', ['actor', 'time', 'extra', 'none'])
def test_saved_method_must_name_who_authorized_it_when_and_why(fault):
    value = recorded_method()
    if fault == 'actor':
        value['authorizations'][0]['actor'] = ' '
    elif fault == 'time':
        value['authorizations'][1]['confirmed_utc'] = '2026-10-22 08:36'
    elif fault == 'extra':
        value['authorizations'][1]['decided_by'] = 'somebody'
    else:
        value['authorizations'] = []
    with pytest.raises(ValueError):
        method.validate_method(value)


def test_series_without_overflow_method_settings_names_the_missing_key():
    with pytest.raises(ValueError, match='decisions.overflow_method'):
        method.method_settings({})
    broken = deepcopy(DECISIONS)
    broken['overflow_method']['records'] = {'decision': 'not-a-digest'}
    with pytest.raises(ValueError, match='records'):
        method.method_settings(broken)


def test_original_reader_rejects_bound_protocol_tamper_even_same_byte_count(tmp_path):
    value = recorded_method()
    local = tmp_path/'protocol.md'
    local.write_bytes((FIXTURES/'round5_protocol.md').read_bytes())
    value['protocol'] = asset(local)
    bound = _write_json(tmp_path/'method.json', value)
    raw = local.read_bytes()
    local.write_bytes(bytes([raw[0] ^ 1])+raw[1:])
    assert local.stat().st_size == value['protocol']['size_bytes']
    with pytest.raises(ValueError, match='Data SHA'):
        method.checked_method(DataReader(), bound)


def test_declared_storage_move_preserves_exact_overflow_sources(tmp_path, monkeypatch):
    from darsia.fluidflower import storage_paths
    from darsia.fluidflower.pipeline.current_inputs import CurrentInputReader
    logical, physical = tmp_path/'old', tmp_path/'moved'
    physical.mkdir()
    value = recorded_method()
    for key, filename in (('protocol', 'round5_protocol.md'),
                          ('addendum', 'overflow_addendum.md'),
                          ('overflow_review', 'overflow_review.json')):
        source = physical/filename
        source.write_bytes((FIXTURES/filename).read_bytes())
        value[key] = dict(asset(source), path=str(logical/filename))
    bound = _write_json(physical/'method.json', value)
    logical_bound = dict(bound, path=str(logical/'method.json'))
    monkeypatch.setattr(storage_paths, 'storage_path_remapper',
                        lambda: lambda path: str(physical/Path(path).name)
                        if Path(path).parent == logical else path)
    assert method.checked_method(DataReader(), logical_bound) == value
    assert method.checked_method(CurrentInputReader(), logical_bound) == value
    changed = physical/'round5_protocol.md'
    raw = changed.read_bytes()
    changed.write_bytes(bytes([raw[0] ^ 1])+raw[1:])
    with pytest.raises(ValueError, match='Data SHA'):
        method.checked_method(DataReader(), logical_bound)


def test_metadata_reader_still_refuses_an_unverified_reader():
    with pytest.raises(TypeError, match='original DATA reader'):
        method.checked_metadata(object(), {}, limit=1024)


def test_roles_keep_actual_orders_and_no_heldout_or_excluded_run_enters_group():
    values, analysis = roles()
    original = deepcopy((values, analysis))
    assert len(values['fit']) == 26 and len(analysis) == 39
    assert method.validate_roles(values, analysis) == values
    assert (values, analysis) == original
    assert set(method.GROUP_CANDIDATES) <= set(values['fit'])
    assert not set(method.GROUP_CANDIDATES) & set(values['heldout']+values['excluded']+values['descriptive'])


def test_already_removed_exclusions_stay_outside_the_same_active39_cohort():
    values, analysis = roles()
    values['excluded'] = []
    assert method.validate_roles(values, analysis) == values
    assert not set(method.EXCLUDED_RUNS) & set(analysis)
    values['excluded'] = ['ac20']
    with pytest.raises(ValueError):
        method.validate_roles(values, analysis)


@pytest.mark.parametrize('fault', ['heldout_in_fit', 'missing', 'unknown', 'wrong_type'])
def test_role_drift_never_becomes_overflow_eligibility(fault):
    values, analysis = roles()
    if fault == 'heldout_in_fit':
        values['fit'][0] = values['heldout'][0]
    elif fault == 'missing':
        analysis.pop()
    elif fault == 'unknown':
        values['fit'][0] = 'unregistered'
    else:
        values['fit'][0] = {'run': 'ac17'}
    with pytest.raises(ValueError):
        method.validate_roles(values, analysis)
