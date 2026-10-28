import numpy as np
import pytest

from darsia.fluidflower.phase.bm_raw_photometry import encode_training_rgb


def test_fixed_camera_transfer_retains_precision_and_invalid_support():
    linear = np.array([[[.001, .0031308, 1.], [.123456, .2, .4]], [[np.nan]*3, [0, 0, 0]]])*10000
    support = np.array([[1, 1], [0, 1]], bool)
    actual = encode_training_rgb(linear, support, white_range_dn=10000,
                                 white_balance=[1, 1, 1], exposure=1, matrix=None,
                                 transfer='srgb', chunk_rows=1)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual[0, 0], [.01292, .040449936, 1.], atol=1e-7)
    assert np.isnan(actual[1, 0]).all()
    assert actual[0, 1, 0]*255 != round(actual[0, 1, 0]*255)


def test_no_implicit_white_balance_or_new_colour_matrix():
    with pytest.raises(ValueError, match='explicit fixed'):
        encode_training_rgb(np.ones((2, 2, 3)), np.ones((2, 2), bool),
                            white_range_dn=10000, white_balance=[1, 1, 1],
                            exposure=1, matrix=np.eye(3), transfer='srgb')
