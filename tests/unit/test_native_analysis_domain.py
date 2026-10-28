"""Native incomplete-domain accounting with unchanged original known arithmetic."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import darsia
from darsia.fluidflower.native_analysis_domain import AnalysisDomainGeometry


def _geometry():
    """Create original weighted geometry with one genuinely unknown property."""
    porosity = np.array([[0.25, np.nan, 0.0], [0.5, 0.75, 0.125]], dtype=np.float64)
    depth = np.array([[0.02, 0.03, 0.04], [0.05, 0.06, 0.07]], dtype=np.float64)
    return darsia.ExtrudedPorousGeometry(porosity, depth, space_dim=2,
                                        num_voxels=(2, 3), dimensions=[2.0, 3.0])


def _masks():
    """Supply distinct mask causes with one overlap and one property-only pixel."""
    gap = np.array([[False, True, False], [True, False, False]])
    unknown = np.array([[False, True, False], [False, False, True]])
    return gap, unknown


def _wrapped():
    """Keep original physical properties separate from declared analysis gaps."""
    gap, unknown = _masks()
    return AnalysisDomainGeometry(_geometry(), gap, {"approval": {"sha256": "a" * 64}},
                                  unknown_properties_mask=unknown)


def test_union_counts_do_not_turn_physical_zero_into_unknown():
    """Count explicit mask causes once and retain literal zero porosity as known."""
    geometry = _geometry()
    before = geometry.porosity.tobytes(), geometry.depth.tobytes(), geometry.voxel_volume.tobytes()
    gap, unknown = _masks()
    wrapped = AnalysisDomainGeometry(geometry, gap, {"approved": False}, unknown_properties_mask=unknown)
    values = np.array([[2.0, 900.0, 7.0], [800.0, 5.0, 600.0]])
    result = wrapped.integrate_quantified(values)
    original_products = np.multiply(geometry.voxel_volume, values)
    literal = np.sum(np.sum(np.where(gap | unknown, 0.0, original_products), axis=0), axis=0)
    assert np.asarray(result.quantified_value).tobytes() == np.asarray(literal).tobytes()
    assert result.approved_gap_pixels == 2
    assert result.unknown_properties_pixels == 2
    assert result.unquantified_pixels == 3
    assert not result.complete and result.full_value is None
    assert result.unknown_mass is None and result.unknown_volume is None
    assert not wrapped.unquantified_mask[0, 2]
    assert geometry.porosity[0, 2] == 0.0
    assert before == (geometry.porosity.tobytes(), geometry.depth.tobytes(), geometry.voxel_volume.tobytes())
    with pytest.raises(TypeError):
        float(result)


@pytest.mark.parametrize("operation", ["integrate", "normalize", "make_extensive"])
def test_complete_operations_refuse_before_using_incomplete_values(operation):
    """A normal scalar operation cannot mistake an incomplete domain for zero."""
    wrapped = _wrapped()
    image = darsia.Image(np.zeros((2, 3)), dimensions=[2.0, 3.0], scalar=True)
    with pytest.raises(ValueError, match="incomplete"):
        if operation == "normalize":
            wrapped.normalize(image, image)
        else:
            getattr(wrapped, operation)(image)


def test_extensive_image_cannot_bypass_guard_and_quantifies_without_second_weight():
    """Honor the original extensive sum while retaining incomplete support."""
    wrapped = _wrapped()
    values = np.array([[1.0, np.nan, 3.0], [100.0, 5.0, 200.0]])
    image = darsia.ExtensiveImage(values, dimensions=[2.0, 3.0])
    with pytest.raises(ValueError, match="incomplete"):
        wrapped.integrate(image)
    result = wrapped.integrate_quantified(image)
    assert result.quantified_value == 9.0
    assert result.full_value is None and result.unquantified_pixels == 3


def test_all_unknown_zero_is_not_a_complete_zero_mass():
    """An all-gap canvas has a zero known contribution and an unknown full total."""
    geometry = darsia.Geometry(2, (2, 3), dimensions=[2.0, 3.0])
    wrapped = AnalysisDomainGeometry(geometry, np.ones((2, 3), bool), {})
    result = wrapped.integrate_quantified(np.zeros((2, 3)))
    assert result.quantified_value == 0.0
    assert result.unquantified_pixels == 6
    assert not result.complete and result.full_value is None
    with pytest.raises(ValueError, match="incomplete"):
        wrapped.integrate(np.zeros((2, 3)))


def test_empty_masks_keep_original_integral_normalization_and_extensive_bytes():
    """The explicit empty scope uses each original numerical implementation."""
    geometry = darsia.Geometry(2, (2, 3), dimensions=[2.0, 3.0])
    wrapped = AnalysisDomainGeometry(geometry, np.zeros((2, 3), bool), {})
    values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    image = darsia.Image(values, dimensions=[2.0, 3.0], scalar=True)
    reference = darsia.Image(values * 2, dimensions=[2.0, 3.0], scalar=True)
    assert np.asarray(wrapped.integrate(image)).tobytes() == np.asarray(geometry.integrate(image)).tobytes()
    assert wrapped.make_extensive(image).img.tobytes() == geometry.make_extensive(image).img.tobytes()
    assert wrapped.normalize(image, reference).img.tobytes() == geometry.normalize(image, reference).img.tobytes()
    result = wrapped.integrate_quantified(image)
    assert result.complete and result.unquantified_pixels == 0
    assert np.asarray(result.full_value).tobytes() == np.asarray(geometry.integrate(image)).tobytes()


def test_masks_provenance_and_array_result_are_detached_and_immutable():
    """Caller mutation cannot change retained support, provenance or array sums."""
    gap, unknown = _masks()
    provenance = {"approval": {"accepted": False}, "flags": [False]}
    wrapped = AnalysisDomainGeometry(darsia.Geometry(2, (2, 3), dimensions=[2, 3]),
                                    gap, provenance, unknown_properties_mask=unknown)
    gap[:] = False
    unknown[:] = False
    provenance["approval"]["accepted"] = True
    provenance["flags"].append(True)
    result = wrapped.integrate_quantified(np.ones((2, 3, 2)))
    assert result.unquantified_pixels == 3
    assert result.provenance == {"approval": {"accepted": False}, "flags": [False]}
    detached = result.provenance
    detached["flags"].append(True)
    assert result.provenance["flags"] == [False]
    for array in (wrapped.approved_gap_mask, wrapped.unknown_properties_mask,
                  wrapped.unquantified_mask, result.quantified_value):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        wrapped._provenance_json = "{}"


def test_unknown_property_without_gap_still_prevents_complete_total():
    """A missing property does not need approval-gap membership to stay unknown."""
    unknown = np.zeros((2, 3), bool)
    unknown[1, 2] = True
    wrapped = AnalysisDomainGeometry(darsia.Geometry(2, (2, 3), dimensions=[2, 3]),
                                    np.zeros((2, 3), bool), {}, unknown_properties_mask=unknown)
    result = wrapped.integrate_quantified(np.ones((2, 3)))
    assert result.approved_gap_pixels == 0 and result.unknown_properties_pixels == 1
    assert result.unquantified_pixels == 1 and result.full_value is None


@pytest.mark.parametrize("bad_mask", [np.zeros((3, 2), bool), np.zeros((2, 3), np.uint8)])
def test_masks_require_exact_native_shape_and_boolean_dtype(bad_mask):
    """A mask cannot be resized or interpreted from integer label values."""
    with pytest.raises(ValueError, match="exact boolean native canvas"):
        AnalysisDomainGeometry(_geometry(), bad_mask, {})
    with pytest.raises(ValueError, match="exact boolean native canvas"):
        AnalysisDomainGeometry(_geometry(), np.zeros((2, 3), bool), {}, unknown_properties_mask=bad_mask)


def test_quantified_data_cannot_resize_native_support_or_hide_known_nan():
    """Reject a foreign canvas and retain an unmasked nonfinite contribution."""
    wrapped = _wrapped()
    with pytest.raises(ValueError, match="exact native canvas"):
        wrapped.integrate_quantified(np.ones((4, 6)))
    values = np.ones((2, 3))
    values[0, 0] = np.nan
    assert np.isnan(wrapped.integrate_quantified(values).quantified_value)


def test_masked_nonfinite_weights_do_not_multiply_under_strict_error_state():
    """Unknown zero-times-infinity contributions never execute or suppress known arithmetic."""
    porosity = np.array([[0.25, np.inf, np.nan], [0.5, 0.75, 0.0]])
    geometry = darsia.ExtrudedPorousGeometry(porosity, np.ones((2, 3)), 2, (2, 3),
                                           dimensions=[2.0, 3.0])
    gap = np.zeros((2, 3), bool)
    gap[0, 1] = True
    unknown = np.zeros((2, 3), bool)
    unknown[0, 2] = True
    wrapped = AnalysisDomainGeometry(geometry, gap, {}, unknown_properties_mask=unknown)
    values = np.array([[2.0, 0.0, 0.0], [3.0, 4.0, 5.0]])
    before = geometry.voxel_volume.tobytes(), geometry.porosity.tobytes()
    expected = np.sum(np.sum(np.multiply(np.where(gap | unknown, 0.0, geometry.voxel_volume),
                                         values), axis=0), axis=0)
    with np.errstate(invalid="raise", over="raise"):
        result = wrapped.integrate_quantified(values)
    assert np.asarray(result.quantified_value).tobytes() == np.asarray(expected).tobytes()
    assert result.unquantified_pixels == 2 and result.full_value is None
    assert before == (geometry.voxel_volume.tobytes(), geometry.porosity.tobytes())


def test_unmasked_invalid_weight_product_still_raises():
    """A strict caller's error state still applies to invalid known contributions."""
    geometry = darsia.WeightedGeometry(np.array([[np.inf, 1.0], [1.0, 1.0]]),
                                     2, (2, 2), dimensions=[2.0, 2.0])
    gap = np.array([[False, False], [False, True]])
    wrapped = AnalysisDomainGeometry(geometry, gap, {})
    values = np.array([[0.0, 1.0], [1.0, 0.0]])
    with np.errstate(invalid="raise"):
        with pytest.raises(FloatingPointError):
            wrapped.integrate_quantified(values)


def test_subregion_keeps_original_roi_masks_and_removes_only_outside_causes():
    """Known and incomplete ROIs retain original slicing without mask resampling."""
    geometry = darsia.WeightedGeometry(np.ones((2, 3)), 2, (2, 3), dimensions=[2.0, 3.0])
    gap = np.zeros((2, 3), bool)
    gap[0, 0] = True
    unknown = np.zeros((2, 3), bool)
    unknown[1, 2] = True
    wrapped = AnalysisDomainGeometry(geometry, gap, {"source": "retained"}, unknown_properties_mask=unknown)
    roi = darsia.CoordinateArray([[1.0, 0.0], [3.0, 2.0]])
    native = darsia.Image(gap, dimensions=[2.0, 3.0], scalar=True).subregion(roi)
    sub = wrapped.subregion(roi)
    np.testing.assert_array_equal(sub.approved_gap_mask, native.img)
    assert sub.provenance == {"source": "retained"}
    assert sub.scope["original_native_canvas"] == wrapped.scope["original_native_canvas"]
    assert sub.scope["num_voxels"] == list(native.num_voxels)
    assert sub.scope["dimensions"] == list(native.dimensions)
    assert sub.scope["origin"] == native.origin.tolist()
    assert sub.scope["subregions"][0]["num_voxels"] == list(native.num_voxels)
    assert sub.integrate_quantified(np.ones(tuple(sub.num_voxels))).unquantified_pixels == 1
    with pytest.raises(ValueError, match="incomplete"):
        sub.integrate(np.ones(tuple(sub.num_voxels)))
    known_roi = darsia.CoordinateArray([[1.0, 1.0], [2.0, 2.0]])
    known = wrapped.subregion(known_roi)
    known_data = np.ones(tuple(known.num_voxels))
    known_value = geometry.subregion(known_roi).integrate(known_data)
    assert known.integrate(known_data) == known_value
    accounted = known.integrate_quantified(known_data)
    assert accounted.complete and accounted.full_value == known_value
    assert accounted.scope["num_voxels"] == [1, 1]
    assert accounted.scope["dimensions"] == [1.0, 1.0]
    assert accounted.scope["origin"] == [1.0, 2.0]
    assert accounted.scope["original_native_canvas"]["num_voxels"] == [2, 3]
    assert accounted.scope["original_native_canvas"]["dimensions"] == [2.0, 3.0]


def test_update_requires_explicit_reconstruction_and_finite_provenance():
    """An update cannot silently discard retained unknown support or provenance."""
    wrapped = _wrapped()
    with pytest.raises(NotImplementedError, match="Rebuild"):
        wrapped.update(depth=np.ones((2, 3)))
    with pytest.raises(ValueError):
        AnalysisDomainGeometry(_geometry(), np.zeros((2, 3), bool), {"value": float("nan")})
