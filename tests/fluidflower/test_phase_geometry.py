"""Reference warps and source-bound curves on tiny synthetic rasters."""

import numpy as np
import pytest

from darsia.fluidflower.phase import geometry


@pytest.fixture
def parameters():
    return dict(
        source_origin_xy=[0, 0],
        unit_denominator=65535.0,
        curve_homography_floor=1e-10,
        mask_border_value=0,
    )


@pytest.fixture
def feature_parameters():
    return dict(
        rig_decoder=dict(denominator=65535.0),
        alignment=dict(homography_floor=1e-12, coverage_floor=1e-6),
    )


def test_reference_alignment_translates_support_without_changing_input(
    parameters, feature_parameters
):
    signal = np.full((3, 2, 3), 4.0, np.float32)
    valid = np.ones((2, 3), bool)
    H = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], np.float64)
    result, support = geometry.align_scientific(
        signal,
        valid,
        H,
        target_shape_hw=(2, 4),
        parameters=parameters,
        feature_parameters=feature_parameters,
    )
    assert result.dtype == np.float32 and result.shape == (3, 2, 4)
    assert not support[:, 0].any() and support[:, 1:].all()
    assert np.isnan(result[:, :, 0]).all()
    assert np.array_equal(result[:, :, 1:], signal)
    assert np.array_equal(signal, np.full(signal.shape, 4.0, np.float32))


def test_reference_alignment_rejects_fractional_bilinear_support(
    parameters, feature_parameters
):
    signal = np.full((3, 2, 3), 4.0, np.float32)
    valid = np.ones((2, 3), bool)
    valid[:, 0] = False
    H = np.array([[1, 0, 0.5], [0, 1, 0], [0, 0, 1]], np.float64)
    result, support = geometry.align_scientific(
        signal,
        valid,
        H,
        target_shape_hw=(2, 3),
        parameters=parameters,
        feature_parameters=feature_parameters,
    )
    assert not support[:, :2].any() and support[:, 2].all()
    assert np.isnan(result[:, :, :2]).all()


def test_curve_transport_preserves_gaps_and_explicit_origin_offsets(parameters):
    curve = np.array([1.0, 1.0, np.nan, 3.0, 3.0])
    result = geometry.transport_curve(
        curve,
        H_source_to_target=np.eye(3),
        source_origin_xy=(1, 2),
        target_origin_xy=(0, 0),
        target_width=7,
        parameters=parameters,
    )
    expected = np.array([np.nan, 3.0, 3.0, np.nan, 5.0, 5.0, np.nan])
    assert np.array_equal(result, expected, equal_nan=True)
    assert result.dtype == np.float64


def test_floor_holds_endpoints_but_rejects_interior_missing_evidence(parameters):
    result = geometry.transported_floor(
        np.array([np.nan, 2.0, 2.0, np.nan]),
        H_source_to_target=np.eye(3),
        source_origin_xy=(0, 0),
        target_origin_xy=(0, 0),
        target_shape_hw=(4, 6),
        parameters=parameters,
    )
    assert np.array_equal(result, np.full(6, 2.0))
    with pytest.raises(ValueError, match="interior gap"):
        geometry.transported_floor(
            np.array([1.0, 1.0, np.nan, 3.0, 3.0]),
            H_source_to_target=np.eye(3),
            source_origin_xy=(0, 0),
            target_origin_xy=(0, 0),
            target_shape_hw=(4, 5),
            parameters=parameters,
        )
    with pytest.raises(ValueError, match="no valid"):
        geometry.transported_floor(
            np.full(3, np.nan),
            H_source_to_target=np.eye(3),
            source_origin_xy=(0, 0),
            target_origin_xy=(0, 0),
            target_shape_hw=(4, 3),
            parameters=parameters,
        )


def test_zone_warp_places_native_mask_and_holds_its_bottom_row(parameters):
    zones = np.zeros((3, 4), np.uint8)
    zones[:, 2] = 4
    result = geometry.native_l1(
        zones,
        H_source_to_target=np.eye(3),
        native_shape_hw=(3, 4),
        source_box_xywh=(1, 1, 4, 3),
        phase_shape_hw=(5, 6),
        zone_id=4,
        parameters=parameters,
    )
    expected = np.zeros((5, 6), bool)
    expected[1:, 3] = True
    assert result.dtype == bool and np.array_equal(result, expected)
    assert not result[0].any() and not result[:, 0].any()
