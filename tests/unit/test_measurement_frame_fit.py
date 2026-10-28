"""Measured spans contribute geometry only after explicit physical placement."""
from copy import deepcopy

import numpy as np
import pytest

from darsia.fluidflower import rig_measurement_lines as lines
from darsia.fluidflower import rig_static_references as refs
from .test_rig_static_references import packet


H = np.array([[.65, .03, 70.], [.01, -.7, 450.], [.00008, .00004, 1.]])
SPANS = [([0., 0.], 899., 'x+'), ([0., 390.], 899., 'x+'),
         ([0., 0.], 390., 'y+'), ([185., 0.], 390., 'y+'),
         ([899., 390.], 390., 'y-')]


def placed_document(tmp_path):
    source = packet(tmp_path, np.full((600, 800, 3), 90, np.uint8), origin=(0., 0.))
    document = refs.empty_document()
    document['measurement_lines'] = []
    observations = {}
    for index, (start, length, direction) in enumerate(SPANS):
        placement = dict(start_xy_mm=start, direction=direction)
        physical = np.array([start, np.asarray(start)+length*np.asarray(lines._DIRECTIONS[direction])])
        points = refs._project(H, physical)
        line = lines.capture_measurement_line(source, line_id=f'span-{index}', name=f'Span {index}',
            points_u_xy=points.tolist(), length_mm=length)
        line = lines.update_measurement_line(line, placement=placement)
        document['measurement_lines'].append(line)
        observations[line['id']] = lines.manual_measurement_observation(line, source, points)
    return source, document, observations


def fit(document, observations):
    return lines.fit_measurement_reference_model(document, {}, observations)


def test_placed_horizontal_and_vertical_spans_recover_perspective_and_recognition_model(tmp_path):
    source, document, observations = placed_document(tmp_path)
    model = fit(document, observations)
    assert model['status'] == 'valid', model
    assert model['used_line_ids'] == [line['id'] for line in document['measurement_lines']]
    assert model['used_region_ids'] == []
    assert len(model['residuals']) == 10
    assert {r['endpoint_index'] for r in model['residuals']} == {0, 1}
    probes = [[0., 490.], [899., 490.], [400., 200.], [-10., 100.]]
    np.testing.assert_allclose(refs._project(model['H_metric_mm_to_camera_u'], probes),
                               refs._project(H, probes), atol=.0001)
    result = refs.recognize_regions(document, source)
    assert result['status'] == 'proposal'
    assert result['plane_models']['rig']['status'] == 'valid'
    assert result['model']['used_line_ids'] == model['used_line_ids']


def test_legacy_distance_only_lines_guides_and_unknown_lengths_do_not_invent_geometry(tmp_path):
    _, document, observations = placed_document(tmp_path)
    for line in document['measurement_lines']:
        line.pop('placement')
    model = fit(document, observations)
    assert model['status'] == 'insufficient'
    assert model['used_line_ids'] == []
    assert len(model['unplaced_line_ids']) == 5
    assert 'start x/y' in model['reason']
    assert model['H_metric_mm_to_camera_u'] is None
    line = document['measurement_lines'][0]
    line = lines.update_measurement_line(line, placement=dict(start_xy_mm=[0, 0], direction='x+'), length_mm=0.)
    assert lines.measurement_metric_points(line) is None
    document['measurement_lines'] = [line]
    assert fit(document, observations)['status'] == 'insufficient'
    # Guides never enter either physical-anchor collection.
    document['guides'] = [dict(points_u_xy=[[0, 0], [800, 600]], metric_xy_mm=[899, 490])]
    assert fit(document, observations)['used_anchor_ids'] == []


@pytest.mark.parametrize('case', ['one_line', 'collinear', 'shared_corner'])
def test_underdetermined_or_degenerate_spans_never_generate_a_frame(tmp_path, case):
    _, document, observations = placed_document(tmp_path)
    if case == 'one_line':
        document['measurement_lines'] = document['measurement_lines'][:1]
    elif case == 'shared_corner':
        document['measurement_lines'] = [document['measurement_lines'][i] for i in (0, 2)]
    else:
        document['measurement_lines'] = document['measurement_lines'][:2]
        document['measurement_lines'][1] = lines.update_measurement_line(document['measurement_lines'][1],
            placement=dict(start_xy_mm=[1000., 0.], direction='x+'))
    model = fit(document, observations)
    assert model['status'] in ('insufficient', 'degenerate'), model
    assert model['H_metric_mm_to_camera_u'] is None


@pytest.mark.parametrize('damage', ['wrong_length', 'wrong_start', 'wrong_direction'])
def test_incompatible_redundant_measurements_are_rejected(tmp_path, damage):
    _, document, observations = placed_document(tmp_path)
    line = document['measurement_lines'][3]
    if damage == 'wrong_length':
        line = lines.update_measurement_line(line, length_mm=310.)
    elif damage == 'wrong_start':
        line = lines.update_measurement_line(line, placement=dict(start_xy_mm=[260., 0.], direction='y+'))
    else:
        line = lines.update_measurement_line(line, placement=dict(start_xy_mm=[185., 0.], direction='y-'))
    document['measurement_lines'][3] = line
    model = fit(document, observations)
    assert model['status'] in ('inconsistent', 'invalid_orientation'), model
    assert model['H_metric_mm_to_camera_u'] is None


def test_signed_directions_define_endpoint_order_without_changing_tracking(tmp_path):
    _, document, _ = placed_document(tmp_path)
    original = document['measurement_lines'][0]
    reversed_line = lines.update_measurement_line(original,
        placement=dict(start_xy_mm=[899., 10.], direction='x-'))
    assert reversed_line['tracking_identity'] == original['tracking_identity']
    assert lines.measurement_metric_points(reversed_line) == [[899., 10.], [0., 10.]]
    assert lines.measurement_metric_points(document['measurement_lines'][4]) == [[899., 390.], [899., 0.]]
    cleared = lines.update_measurement_line(reversed_line, placement=None)
    assert 'placement' not in cleared
    assert lines.measurement_metric_points(cleared) is None


def test_real_point_references_and_placed_line_endpoints_share_one_model(tmp_path):
    source, document, observations = placed_document(tmp_path)
    document['measurement_lines'] = document['measurement_lines'][:1]
    points = [[0., 390.], [899., 390.]]
    document['regions'] = [dict(id=f'point-{i}', metric_xy_mm=xy, plane_id='rig') for i, xy in enumerate(points)]
    region_observations = {region['id']: dict(status='manual', anchor_u_xy=uv.tolist(),
        photo_identity=refs.photo_binding(source)['identity'])
        for region, uv in zip(document['regions'], refs._project(H, points))}
    model = lines.fit_measurement_reference_model(document, region_observations, observations)
    assert model['status'] == 'valid', model
    assert model['used_region_ids'] == ['point-0', 'point-1']
    assert model['used_line_ids'] == ['span-0']


def test_stale_tracking_missing_bindings_and_other_depth_planes_cannot_supply_anchors(tmp_path):
    source, document, observations = placed_document(tmp_path)
    for index in (0, 1, 2):
        document['measurement_lines'][index]['plane_id'] = 'bolt-heads'
    observations['span-3']['tracking_identity'] = '0'*64
    observations['span-4'].pop('photo_identity')
    model = fit(document, observations)
    assert model['status'] == 'insufficient', model
    assert model['used_line_ids'] == []
    assert set(model['excluded_line_ids']) == {'span-3', 'span-4'}
    result = refs.recognize_regions(document, source)
    assert set(result['plane_models']) == {'rig', 'bolt-heads'}
    assert result['plane_models']['bolt-heads']['used_line_ids'] == ['span-0', 'span-1', 'span-2']
    assert result['plane_models']['rig']['used_line_ids'] == ['span-3', 'span-4']
    # Even perfectly fitting endpoints from two photos cannot form one frame.
    _, full, observed = placed_document(tmp_path)
    observed['span-0']['photo_identity'] = '1'*64
    mixed = fit(full, observed)
    assert mixed['status'] == 'stale' and mixed['H_metric_mm_to_camera_u'] is None


def test_physical_edits_keep_manual_pixels_but_invalidate_saved_geometry(tmp_path):
    source, document, observations = placed_document(tmp_path)
    document.update(rig_origin_xy_mm=[0., 0.], model_plane_id='rig')
    result = dict(run_id=source['run_id'], photo=refs.photo_binding(source), observations={},
        measurement_lines=observations, model=fit(document, observations), status='proposal')
    document['runs'][source['run_id']] = result
    saved = refs.save_references(tmp_path/'project', document, actor="Reviewer")
    restored = refs.load_references(tmp_path/'project')
    assert restored['rig_origin_xy_mm'] == [0., 0.] and restored['model_plane_id'] == 'rig'
    assert restored['runs']['source']['model']['status'] == 'valid'
    line = restored['measurement_lines'][0]
    previous = deepcopy(line)
    restored['measurement_lines'][0] = lines.update_measurement_line(line, name='Known width', length_mm=898.,
        placement=dict(start_xy_mm=[1., 0.], direction='x+'))
    refs.save_references(tmp_path/'project', restored, actor="Reviewer")
    edited = refs.load_references(tmp_path/'project')
    assert edited['runs']['source']['model'] is None
    assert edited['runs']['source']['status'] == 'stale'
    assert edited['measurement_lines'][0]['tracking_identity'] == previous['tracking_identity']
    manual = edited['runs']['source']['measurement_lines']['span-0']
    assert manual['status'] == 'manual' and manual['points_u_xy'] == observations['span-0']['points_u_xy']
    assert lines.measurement_for_photo(edited['measurement_lines'][0], source, edited['runs']['source'])['length_mm'] == 898.
    assert saved['measurement_lines'][0]['length_mm'] == 899.
    # A changed RAW invalidates both stored line observations and the model.
    from pathlib import Path
    Path(source['source_raw']['path']).write_bytes(b'changed RAW')
    stale = refs.load_references(tmp_path/'project')
    assert stale['runs']['source']['measurement_lines']['span-0']['status'] == 'stale'
    assert stale['runs']['source']['measurement_lines']['span-0']['points_u_xy'] is None


@pytest.mark.parametrize('placement', [dict(start_xy_mm=[0, 0], direction='diagonal'),
    dict(start_xy_mm=[float('nan'), 0], direction='x+'),
    dict(start_xy_mm=[1000001., 0], direction='x+'),
    dict(start_xy_mm=[999999., 0], direction='x+')])
def test_placement_requires_finite_bounded_coordinates_and_known_axis(tmp_path, placement):
    _, document, _ = placed_document(tmp_path)
    with pytest.raises(ValueError, match='(direction|finite|coordinates|endpoints)'):
        lines.update_measurement_line(document['measurement_lines'][0], placement=placement)


@pytest.mark.parametrize('field,value', [('rig_origin_xy_mm', [0., float('nan')]),
    ('rig_origin_xy_mm', [0.]), ('rig_origin_xy_mm', [1000001., 0.]), ('model_plane_id', 'wrong plane')])
def test_saved_frame_origin_and_plane_settings_are_validated(tmp_path, field, value):
    document = refs.empty_document()
    document[field] = value
    with pytest.raises(ValueError):
        refs.save_references(tmp_path/'invalid', document, actor="Reviewer")
