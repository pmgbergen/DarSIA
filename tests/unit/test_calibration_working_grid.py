"""Computing-grid physics on tiny arrays; no real experiment or RAW reads."""
import numpy as np
import pytest

from darsia.fluidflower.calibration_working_grid import calibration_working_grid


def grid(shape=(7, 11), factor=2, **changes):
    basis = dict(shape_hw=list(shape), native_origin_xy=[-3, 7],
                 native_to_camera_h=[[1., .02, 10.], [0., 1., -2.], [0., .001, 1.]],
                 build_sha256="a" * 64, photometry_sha256="b" * 64, metric_sha256="c" * 64)
    basis.update(changes)
    return calibration_working_grid(basis, factor=factor)


def points(matrix, values):
    homogeneous = np.column_stack([values, np.ones(len(values))]) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def test_full_preserves_actual_buffers_and_legacy_model():
    current = grid(factor=1)
    rgb = np.ones((7, 11, 3), np.float32)
    area = np.ones((7, 11), np.float32)
    labels = np.ones((7, 11), np.uint8)
    assert current.linear_rgb(rgb) is rgb
    assert current.extensive(area) is area
    assert current.intensive(area, area) is area
    assert current.training_labels(labels) is labels
    current.require_trained_model({})


@pytest.mark.parametrize("factor", [2, 4])
def test_nondivisible_footprint_area_and_mass_are_conserved(factor):
    current = grid(factor=factor)
    area = np.arange(1., 78.).reshape(7, 11) / 100
    pore_volume = area * np.linspace(.2, .6, 77).reshape(7, 11) * .01
    concentration = np.arange(77.).reshape(7, 11) / 7
    reduced_area = current.extensive(area)
    reduced_pore_volume = current.extensive(pore_volume)
    reduced_concentration = current.intensive(concentration, pore_volume)
    assert reduced_area.shape == current.shape_hw
    assert reduced_area.sum() == pytest.approx(area.sum(), rel=1e-13)
    assert reduced_pore_volume.sum() == pytest.approx(pore_volume.sum(), rel=1e-13)
    assert (reduced_pore_volume * reduced_concentration).sum() == pytest.approx(
        (pore_volume * concentration).sum(), rel=1e-13)


def test_linear_rgb_is_averaged_before_features_not_a_storage_resize():
    current = grid(shape=(2, 2))
    rgb = np.array([[[.1] * 3, [.9] * 3], [[.2] * 3, [.8] * 3]])
    np.testing.assert_allclose(current.linear_rgb(rgb), .5)
    assert not np.isclose(np.log(current.linear_rgb(rgb))[0, 0, 0], np.log(rgb).mean())


def test_reduced_camera_mapping_keeps_same_physical_crop_footprint():
    current = grid()
    ox, oy = current.source_basis["native_origin_xy"]
    native = np.array([[ox - .5, oy - .5], [ox + 10.5, oy + 6.5], [2., 9.]])
    reduced = points(current.native_to_working_h, native)
    np.testing.assert_allclose(reduced[:2], [[-.5, -.5], [current.shape_hw[1] - .5,
                                                              current.shape_hw[0] - .5]])
    np.testing.assert_allclose(points(current.working_to_camera_h, reduced),
        points(np.array(current.source_basis["native_to_camera_h"]), native))


def test_history_transform_recomputed_between_real_grid_origins():
    previous = grid(shape=(9, 13), native_origin_xy=[2, 5])
    current = grid()
    transform = np.array([[1., .02, 1.], [.03, 1., -2.], [0., .001, 1.]])
    native = np.array([[3., 6.], [8., 7.]])
    np.testing.assert_allclose(points(current.transport_from(previous, transform),
        points(previous.native_to_working_h, native)),
        points(current.native_to_working_h, points(transform, native)))
    with pytest.raises(ValueError, match="history"):
        current.transport_from(grid(factor=4), transform)


def test_mixed_training_footprints_do_not_invent_labels():
    current = grid(shape=(2, 4))
    labels = np.array([[1, 1, 2, 2], [1, 1, 2, 255]], np.uint8)
    np.testing.assert_array_equal(current.training_labels(labels), [[1, 255]])


def test_full_model_cannot_silently_run_in_changed_resolution_or_photometry():
    current = grid()
    with pytest.raises(ValueError, match="Retrain"):
        current.require_trained_model({})
    model = dict(calibration_working_grid_bases=[current.binding()])
    current.require_trained_model(model)
    with pytest.raises(ValueError, match="Retrain"):
        grid(photometry_sha256="d" * 64).require_trained_model(model)
    with pytest.raises(ValueError, match="Retrain"):
        grid(factor=4).require_trained_model(model)


@pytest.mark.parametrize("factor", [0, .5, True, 3])
def test_invalid_resolution_is_rejected(factor):
    with pytest.raises(ValueError):
        grid(factor=factor)


def test_missing_source_identity_and_log_rgb_are_rejected():
    with pytest.raises(ValueError, match="identities"):
        grid(build_sha256="unknown")
    with pytest.raises(ValueError, match="linear RGB"):
        grid().linear_rgb(np.full((7, 11, 3), -1., np.float32))
