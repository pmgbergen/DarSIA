import json
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.phase.train_ffum_raw_model import content_input, optical_record
from darsia.fluidflower.phase.ffum_raw_numerics import sha
from darsia.fluidflower.phase.ffum_review_mapping import UmReviewMapping
from darsia.fluidflower.phase.transfer_ffum_raw_masks import sparse_region


def prepared(root, support):
    folder = root/'metric'/'one'; folder.mkdir(parents=True)
    values = dict(rgb_unit=np.full((*support.shape, 3), .4, np.float32),
                  source_support=support, training_domain=support.copy())
    receipt = {}
    for name, value in values.items():
        np.save(folder/(name+'.npy'), value)
        receipt[name+'_sha256'] = sha(folder/(name+'.npy'))
    (folder/'receipt.json').write_text(json.dumps(receipt))
    return folder


def test_missing_source_coverage_cannot_become_black_training_evidence(tmp_path):
    support = np.ones((3, 5), bool); support[0, 0] = False
    prepared(tmp_path, support)
    with pytest.raises(ValueError, match='Incomplete finite'):
        content_input(tmp_path, 'one')


def test_corrupted_prepared_values_are_rejected_before_training(tmp_path):
    folder = prepared(tmp_path, np.ones((3, 5), bool))
    np.save(folder/'rgb_unit.npy', np.zeros((3, 5, 3), np.float32))
    with pytest.raises(ValueError, match='checksum'):
        content_input(tmp_path, 'one')


@pytest.mark.parametrize('maximum,accepted', [(np.nextafter(np.float32(1), np.float32(2)), True), (1.00001, False)])
def test_convex_interpolation_roundoff_does_not_hide_photometric_overflow(tmp_path, maximum, accepted):
    folder = prepared(tmp_path, np.ones((3, 5), bool))
    np.save(folder/'rgb_unit.npy', np.full((3, 5, 3), maximum, np.float32))
    receipt = json.loads((folder/'receipt.json').read_text())
    receipt['rgb_unit_sha256'] = sha(folder/'rgb_unit.npy')
    (folder/'receipt.json').write_text(json.dumps(receipt))
    if accepted:
        assert content_input(tmp_path, 'one')[0].max() == 1
    else:
        with pytest.raises(ValueError, match='Incomplete finite'):
            content_input(tmp_path, 'one')


def test_region_policy_changes_class_without_rewriting_original_geometry():
    payload = dict(x=.3, y=.2, width=.1, height=.05, category='missed_visible_aqueous_co2', reviewer_id='test-reviewer')
    row = dict(region_id='run10:manual:abc', phase_class='aqueous',
        original_inventory=dict(run='run10', dataset='ffum', frame_index='3', elapsed_h='1.5',
            confidence='high', source='manual', reviewer='Test Reviewer'),
        origin=dict(original_annotation=payload))
    record = optical_record(row)
    assert record['payload'] == payload
    assert record['phase_class'] == 'aqueous'
    assert record['region_id'] == row['region_id']
    assert record['frame_index'] == 3
    assert record['payload']['reviewer_id'] == 'test-reviewer'
    assert set(record) == {'region_id', 'run', 'dataset', 'frame_index', 'elapsed_h',
                           'phase_class', 'confidence', 'source', 'reviewer', 'payload'}


def test_sparse_transfer_keeps_nonrectangular_raster_and_original_domain():
    h, w = 24, 32
    yy, xx = np.mgrid[:h, :w]
    mapper = UmReviewMapping(np.stack((yy.ravel(), xx.ravel())), (h, w),
        np.eye(3)[:2], (w, h), (w, h), (0, 0))
    profile = SimpleNamespace(shape_hw=(h, w), sensor_wh=(w, h), display_to_sensor=lambda p, c: p)
    old = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0]], bool)
    domain = np.ones((h, w), bool); domain[9, 10] = False
    rasterizer = lambda *args, **kwargs: ((10, 8, 14, 11), old.copy())
    detail, mask, training = sparse_region({}, dict(canvas_wh=(w, h), content_box_xywh=(0, 0, w, h)),
        mapper, profile, domain, rasterizer)
    detail = json.loads(json.dumps(detail, allow_nan=False))
    result = np.zeros((h, w), bool); result_training = result.copy()
    x0, y0, x1, y1 = detail['bbox_xyxy']
    result[y0:y1, x0:x1] = mask; result_training[y0:y1, x0:x1] = training
    expected = np.zeros((h, w), bool); expected[8:11, 10:14] = old
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result_training, expected & domain)
