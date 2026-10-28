"""Probability output guards for the UM model-bound full-field inference."""
import json
import numpy as np
import pytest

from darsia.fluidflower.phase.ffum_raw_model import (bound_output_scores,
    predict_new_raw_frame, validate_temporal_request)
from darsia.fluidflower.phase.package_ffum_raw_model import snapshot_native_support


def test_score_roundoff_keeps_unknown_pixels_and_conditional_scores():
    above = np.nextafter(np.float32(1), np.float32(2))
    values = np.array([[[above, 0.9], [np.nan, np.nan]], [[0.8, 0.7], [0., 1.]]], np.float32)
    support = np.array([[True, False], [True, True]])
    report = bound_output_scores(values, support)
    assert report['corrected_float32_roundoff_elements'] == 1
    assert values[0, 0, 0] == 1
    assert np.isnan(values[0, 1]).all()
    assert values[1, 0].sum() > 1  # These two heads are not additive phase fractions.


@pytest.mark.parametrize('invalid', [np.nan, np.inf, -0.01, 1.001])
def test_supported_score_errors_are_not_hidden(invalid):
    values = np.array([[[invalid, 0.5]]], np.float32)
    with pytest.raises(ValueError):
        bound_output_scores(values, np.ones((1, 1), bool))


def test_context_only_preparation_does_not_claim_a_false_first_frame():
    validate_temporal_request(None, None, False, True)
    with pytest.raises(ValueError, match='first-frame claim'):
        validate_temporal_request(None, None, True, True)
    with pytest.raises(ValueError, match='first temporal frame'):
        validate_temporal_request(None, None, False, False)


def test_a_context_only_record_cannot_be_predicted_directly(tmp_path):
    preparation = tmp_path/'previous'; preparation.mkdir()
    (preparation/'preparation.json').write_text(json.dumps(dict(temporal_context_only=True)))
    with pytest.raises(ValueError, match='context-only'):
        predict_new_raw_frame(tmp_path/'model', preparation, tmp_path/'prediction')
    assert not (tmp_path/'prediction').exists()


def test_package_keeps_native_io_import_support_at_fluidflower_level(tmp_path):
    source, target = tmp_path/'approved/fluidflower', tmp_path/'bundle/darsia/fluidflower'
    (source/'native').mkdir(parents=True); (source/'pipeline').mkdir()
    target.mkdir(parents=True)
    (source/'pipeline/run_io.py').write_bytes(b'approved reader binding')
    for name in ('store_map.py', 'series_paths.py', 'job_filesystem.py'):
        (source/name).write_bytes(('approved '+name).encode())
    (target/'series_paths.py').write_bytes(b'older runtime base')
    snapshot_native_support(source/'native', target)
    assert (target/'pipeline/run_io.py').read_bytes() == b'approved reader binding'
    for name in ('store_map.py', 'series_paths.py', 'job_filesystem.py'):
        assert (target/name).read_bytes() == (source/name).read_bytes()
