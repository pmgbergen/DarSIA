"""Genuine tiny saved N1 and probability products; no queue/campaign claim."""
from copy import deepcopy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower.mass import n1_pipeline, n1_statistics, overflow_statistics
from darsia.fluidflower.mass.production_precision import PixelMath
from darsia.fluidflower.phase import probability_storage
from tests.fixtures.tiny_saved_model_fixture import tiny_config
from tests.unit.test_phase_probability_storage import _models


@pytest.fixture
def steering_case(tiny_config, tmp_path):
    config, _ = tiny_config
    config = deepcopy(config)
    config['geometry_fields'] = dict(source_to_native='H_ac28_to_frame_native',
        original_domain_to_native='H_ac28_to_frame_native',
        carried_homography_provenance='original_H_ac28_to_native')
    context = n1_pipeline.N1Pipeline(config, precision='float32',
                                     pixel_math=PixelMath('float32'), statistics_only=True)
    spec = config['frames'][0]
    extracted = context.extract_statistics(spec)
    path = tmp_path/'completed_original_statistics.json.gz'
    path.write_bytes(gzip.compress(json.dumps(n1_pipeline._jsonable(extracted['statistics']),
        allow_nan=False, separators=(',', ':')).encode('utf8'), mtime=0))
    bound = n1_pipeline.asset(path)
    # The actual tiny native labels are read through the original exact reader.
    frame, *_ = context.load_frame(spec)
    labels = frame.phase_labels
    support = labels != 255
    p = np.where(labels == 2, .8, np.where(labels == 1, .2, 0)).astype(np.float32)
    regional = np.where(labels != 0, .8, 0).astype(np.float32)
    part = {}
    probability_storage.pair(part, p, regional, support, np.isin(labels, (1, 2)),
        labels == 2, branch='placement', pixel_threshold=.7, regional_threshold=.6)
    capture = dict(edge={}, placement=part, postprocessing={})
    classes = dict(modelled=support, filled_observed=np.zeros(labels.shape, bool),
                   filled_unobserved=np.zeros(labels.shape, bool))
    probability_storage.finish(capture, labels, classes, (0, 0, labels.shape[1], labels.shape[0]))
    source = dict(run=config['run'], source_binding=deepcopy(spec['source_binding']),
                  source_sha256=spec['source_binding']['source_sha256'])
    producer = tmp_path/'probability'; producer.mkdir()
    phase_config = n1_pipeline._write_json(producer/'phase.json', dict(model_bundles=_models()))
    roles = probability_storage.write_product(producer, dict(case_id=spec['case_id'], source=source),
        capture, model_bindings=_models(), phase_configuration=phase_config,
        labels=labels, label_descriptor=spec['labels'])
    return dict(context=context, spec=spec, source=source, statistics=bound,
                roles=roles, original=extracted['statistics'], root=tmp_path)


def prepared(case):
    return overflow_statistics.prepare_frame(case['context'], case['spec'],
        probability_roles=case['roles'], statistics_descriptor=case['statistics'],
        phase_source=case['source'], case_id=case['spec']['case_id'])


def test_original_native_summary_and_physical_reader_precede_f_derivation(steering_case):
    case = steering_case
    original_files = {path: path.read_bytes() for path in case['root'].iterdir() if path.is_file()}
    frame = prepared(case)
    result = overflow_statistics.derive_statistics(frame, level=.5, uncapped_level=.5)
    row = result['statistics']
    # Every interval/exclusion/anchor/source field remains the literal original.
    changed = {'main_aqueous_histogram', 'main_gas_mass_g', 'main_gas_water_mass_g',
               'baseline_undiluted_main_mass_g', 'contract', 'overflow_steering'}
    assert {key:value for key,value in row.items() if key not in changed} == {
        key:value for key,value in case['original'].items() if key not in changed}
    assert row['contract'] == overflow_statistics.STATISTICS_FORMAT != case['original']['contract']
    assert row['overflow_steering']['original_statistics'] == case['statistics']
    assert result['distribution']['actual_volume_m3'] == pytest.approx(
        .5*overflow_statistics.baseline_volume(frame), rel=2e-7)
    assert result['report']['gas_change_g'] < 0
    assert result['report']['gas_on_non_gas_label_g'] > 0
    assert not result['raw_decoded'] and not result['phase_inference_run']
    assert not result['optical_model_training_run'] and not result['source_admission_or_job_created']
    assert all(path.read_bytes() == raw for path,raw in original_files.items())
    # Original summary evaluator consumes the derived capacities, not a gas gain.
    anchor = dict(elapsed_h=frame.prepared['frame'].elapsed_h,
                  affected_pore_volume_m3=row['affected_pore_volume_m3'])
    prediction = n1_statistics.evaluate_summary(row, dict(candidate='N1', gain=1.1, beta=0.), anchor)
    assert prediction['gas_mass_g'] == row['main_gas_mass_g']
    assert prediction['aqueous_mass_g'] > row['main_gas_water_mass_g']


@pytest.mark.parametrize('fault', ['statistics-bytes', 'probability-bytes', 'wrong-source'])
def test_original_saved_source_and_probability_guards_cannot_be_bypassed(steering_case, fault):
    case = steering_case
    if fault == 'statistics-bytes':
        path = Path(case['statistics']['path']); path.write_bytes(path.read_bytes()+b' ')
    elif fault == 'probability-bytes':
        path = Path(case['roles']['archive']['path']); path.write_bytes(path.read_bytes()+b' ')
    else:
        case['source']['source_sha256'] = 'f'*64
    with pytest.raises(ValueError):
        prepared(case)


def test_a_rebound_incorrect_summary_still_fails_independent_original_replay(steering_case):
    case = steering_case
    row = deepcopy(case['original']); row['main_gas_mass_g'] += 1.
    path = case['root']/'wrong_statistics.json.gz'
    path.write_bytes(gzip.compress(json.dumps(n1_pipeline._jsonable(row), allow_nan=False).encode(), mtime=0))
    case['statistics'] = n1_pipeline.asset(path)
    with pytest.raises(ValueError, match='differs from the completed P1'):
        prepared(case)


def test_first_pass_matches_original_geometry_without_loading_optical_models(steering_case):
    case = steering_case
    expected = overflow_statistics.baseline_volume(prepared(case))
    config = deepcopy(case['context'].config)
    config['shared']['optics'] = {'unavailable': 'not opened by original frame-loading-only mode'}
    context = n1_pipeline.N1Pipeline(config, precision='float32', pixel_math=PixelMath('float32'),
                                     frame_loading_only=True)
    assert not hasattr(context, 'models') and not hasattr(context, 'calibration')
    result = overflow_statistics.baseline_volume_from_saved_frame(context, case['spec'])
    assert result['baseline_gas_volume_m3'] == expected
    assert not result['raw_decoded'] and not result['optical_model_evaluated']


def test_own_measured_volume_changes_only_phase_terms_and_never_injected_mass(steering_case):
    from darsia.fluidflower.mass.overflow_phase_split import own_target
    frame = prepared(steering_case)
    original = deepcopy(frame.prepared['statistics'])
    baseline = overflow_statistics.baseline_volume(frame)
    time = frame.prepared['frame'].elapsed_h
    target = own_target(dict(t=[time-1., time+1.], vC=[baseline*1e6*.7]*2),
        run=frame.prepared['frame'].run_id, elapsed_h=time, source={'path': 'own-reviewed-overflow.json'})
    result = overflow_statistics.derive_phase_split_statistics(frame, target)
    row = result['statistics']
    assert row['injected_mass_g'] == original['injected_mass_g']
    assert row['source_binding'] == original['source_binding']
    assert row['no_outflow_evidence'] == original['no_outflow_evidence']
    assert result['distribution']['actual_volume_m3'] == pytest.approx(target['target_volume_m3'], rel=2e-7)
    assert row['overflow_steering']['phase_split_target'] == target
    assert row['overflow_steering']['labels_modified'] is False
    assert 'level' not in row['overflow_steering']
