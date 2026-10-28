import numpy as np
import pytest

from darsia.fluidflower.phase.bm_raw_regions import BMReviewMapping, CurvatureInverse, sparse_region_mask


def curved_grid():
    y, x = np.mgrid[:41, :61].astype(float)
    return {"shape": (41, 61), "grid": np.vstack(((y+.002*x*x+7).ravel(), (x+.001*y*y+3).ravel()))}


def test_inverse_saved_nonlinear_grid_preserves_interior_and_boundary():
    inverse = CurvatureInverse(curved_grid(), seed_stride=8)
    points = np.array([[0., 0.], [60, 40], [30.2, 17.8], [1.2, 39], [58, .5]])
    np.testing.assert_allclose(inverse(inverse.forward(points)), points, atol=1e-4)
    assert np.isnan(inverse(np.array([[-20., -20.], [np.nan, 2]]))).all()


def test_noninvertible_grid_is_not_assumed_identity():
    with pytest.raises(ValueError, match="invertible"):
        CurvatureInverse({"shape": (2, 2), "grid": np.zeros((2, 4))})


class RawMapping:
    def display_to_camera_u_xy(self, xy):
        return np.asarray(xy)+[-10, -12]

    def camera_u_to_sensor_xy(self, xy):
        return np.asarray(xy)*[2, 3]

    def sensor_to_camera_u_xy(self, xy):
        return np.asarray(xy)/[2, 3]

    def camera_u_to_display_xy(self, xy):
        return np.asarray(xy)-[-10, -12]


def test_complete_chain_keeps_tiff_offset_separate_from_actual_drift():
    mapping = BMReviewMapping(RawMapping(), CurvatureInverse(curved_grid()),
                              np.array([[1, 0, 5], [0, 1, -3]]), (8, 424), (100, 80), (100, 80))
    points = np.array([[7., 14.], [25.8, 31.]])
    raw = mapping.content_to_display(points)
    np.testing.assert_allclose(mapping(raw), points, atol=1e-4)
    registered = mapping.curvature.forward(points)
    expected = ((registered-[5, -3]+[8, 424])/[2, 3])-[-10, -12]
    np.testing.assert_allclose(raw, expected)
    assert np.isnan(mapping(np.array([[0., 0.]]))).all()


def test_metric_reports_physical_anisotropy_instead_of_reusing_one_mm():
    y, x = np.mgrid[:10, :12].astype(float)
    grid = {"shape": (10, 12), "grid": np.vstack((y.ravel(), x.ravel()))}
    mapping = BMReviewMapping(RawMapping(), CurvatureInverse(grid),
                              np.array([[1, 0, 0], [0, 1, 0]]), (8, 424), (100, 80), (100, 80))
    scales = mapping.projected_metric(np.array([[5., 5.]]), old_spacing_m=(.001, .001))
    np.testing.assert_allclose(scales, [[.003, .002]], rtol=1e-10)


def test_missing_actual_drift_fails():
    with pytest.raises(ValueError, match="actual frame drift"):
        BMReviewMapping(RawMapping(), CurvatureInverse(curved_grid()), np.eye(3), (8, 424), (100, 80), (100, 80))


def test_historical_resize_uses_pixel_centres_and_roundtrips():
    mapping = BMReviewMapping(RawMapping(), CurvatureInverse(curved_grid()),
                              np.array([[1, 0, 0], [0, 1, 0]]), (8, 424), (200, 160), (100, 80))
    points = np.array([[20., 20.]])
    registered = mapping.curvature.forward(points)
    expected = (((registered+.5)*2-.5+[8, 424])/[2, 3])-[-10, -12]
    np.testing.assert_allclose(mapping.content_to_display(points), expected)
    np.testing.assert_allclose(mapping(expected), points, atol=1e-4)


def test_sparse_storage_retains_holes_and_unsupported_pixels():
    old = np.array([[1, 0], [0, 1]], bool)
    indices = np.array([[-1, 0, 1, -1], [-1, 2, 3, -1]], np.int32)
    stored = sparse_region_mask(old, indices)
    np.testing.assert_array_equal(stored['bbox_xywh'], [1, 0, 2, 2])
    np.testing.assert_array_equal(np.unpackbits(stored['packed_mask'])[:4].reshape(2, 2), old)
    assert stored['pixels'] == 2
