import numpy as np
import pytest

from darsia.fluidflower.phase.um_bm_sampling import (
    project_to_full_frame,
    sample_full_frame,
)


def test_fractional_sampling_preserves_linear_rgb_and_input_units():
    y, x = np.mgrid[:9, :12]
    source = np.stack((2*x+3*y, 4*x-y, 7+x), axis=-1).astype(np.float32)
    original = source.copy()
    result = sample_full_frame(source, np.ones((9, 12), bool),
        lambda p: p+[.25, .625], (8, 11), chunk_rows=3)
    yy, xx = np.mgrid[:8, :11].astype(float)
    xx += .25
    yy += .625
    expected = np.stack((2*xx+3*yy, 4*xx-yy, 7+xx), axis=-1)
    np.testing.assert_allclose(result.values, expected)
    assert result.valid_support.all()
    np.testing.assert_array_equal(source, original)


def test_invalid_neighbours_never_become_evidence_but_zero_weight_is_harmless():
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    support = np.ones((3, 4), bool)
    support[1, 2] = False
    source[1, 2] = np.nan
    mapping = np.array([[1., 1.], [1.25, 1.], [3., 2.], [-.01, 1.], [np.nan, 0]])
    result = sample_full_frame(source, support, lambda p: mapping, (1, 5))
    np.testing.assert_array_equal(result.valid_support, [[True, False, True, False, False]])
    np.testing.assert_equal(result.values[0, [0, 2]], [5, 11])
    assert np.isnan(result.values[~result.valid_support]).all()


def test_separate_frame_drift_maps_register_same_physical_baseline():
    y, x = np.mgrid[:12, :15]
    baseline = x+4*y
    current = (x-2)+4*(y-1)
    support = np.ones(baseline.shape, bool)
    a = sample_full_frame(baseline, support, lambda p: p, (8, 10))
    b = sample_full_frame(current, support, lambda p: p+[2, 1], (8, 10))
    np.testing.assert_array_equal(a.values, b.values)
    assert a.valid_support.all() and b.valid_support.all()


def test_reduced_prediction_returns_full_raw_canvas_without_extending_rig_domain():
    values = np.ones((2, 3, 2), np.float32)*[.2, .8]
    source_support = np.ones((2, 3), bool)
    raw_support = np.ones((8, 10), bool)
    raw_support[2, 3] = False
    result = project_to_full_frame(values, source_support, lambda p: p-[2, 1],
                                    (4, 6), raw_support, chunk_rows=2)
    expected_support = np.zeros((8, 10), bool)
    expected_support[1:5, 2:8] = True
    expected_support[2, 3] = False
    np.testing.assert_array_equal(result.valid_support, expected_support)
    np.testing.assert_allclose(result.values[result.valid_support],
                               np.tile([.2, .8], (expected_support.sum(), 1)))
    assert np.isnan(result.values[~result.valid_support]).all()


def test_mapping_contract_and_hidden_nonfinite_source_are_checked():
    source = np.ones((2, 2, 3), np.float32)
    source[0, 0, 1] = np.nan
    support = np.ones((2, 2), bool)
    result = sample_full_frame(source, support, lambda p: p, (2, 2))
    assert not result.valid_support[0, 0]
    with pytest.raises(ValueError, match="xy pair"):
        sample_full_frame(source, support, lambda p: p[:, 0], (2, 2))


def test_phase_projection_cannot_invent_a_third_phase_between_water_and_gas():
    labels = np.array([[0, 2], [0, 2]], dtype=np.uint8)
    result = project_to_full_frame(labels, np.ones((2, 2), bool),
        lambda p: p/[4, 1], (2, 2), np.ones((2, 5), bool), interpolation="nearest")
    np.testing.assert_array_equal(result.values, [[0, 0, 2, 2, 2], [0, 0, 2, 2, 2]])
    assert result.valid_support.all()
    assert set(np.unique(result.values)) == {0, 2}
