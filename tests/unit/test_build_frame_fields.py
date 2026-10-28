"""Numeric transport contracts only; no Build, phase or scientific qualification."""
from dataclasses import replace

import numpy as np
import pytest

from darsia.fluidflower.mass.build_frame_fields import (
    registered_build_fields, registered_scientific_reference,
)
from darsia.fluidflower.mass.native_optical_source import BuildPhysicalContext
from darsia.fluidflower.mass.production_precision import PixelMath


PARAMETERS = dict(determinant_floor=1e-12)


def shift(x, y=0):
    return np.array([[1., 0., x], [0., 1., y], [0., 0., 1.]])


@pytest.fixture
def context():
    facies = np.tile(np.array([1, 9, 2, 5], np.uint16), (3, 1))
    depth = np.tile(np.array([1., 5., 9., 13.], np.float32), (3, 1))
    return BuildPhysicalContext(
        run="sample", baseline_rgb=np.ones((3, 4, 3), np.float32),
        labels=facies.copy(), facies=facies, depth_m=depth,
        porosity=np.tile(np.array([.2, .4, .6, .8], np.float32), (3, 1)),
        grid=dict(num_voxels=[3, 4], dimensions=[.3, .4], origin=[0., .3], voxel_size=[.1, .1]),
        provenance=dict(scope="numeric fixture only"))


def fields(context, h=None, origin=(0, 0), domain=None, source_domain=None):
    shape = context.facies.shape
    return registered_build_fields(context, np.eye(3) if h is None else h, origin,
        shape, np.ones(shape, bool) if domain is None else domain, PixelMath("float32"),
        PARAMETERS, source_domain=np.ones(shape, bool) if source_domain is None else source_domain)


def reference(rgb, valid, source=None, source_origin=(0, 0), current=None,
              current_origin=(0, 0), **kwargs):
    return registered_scientific_reference(rgb, valid,
        np.eye(3) if source is None else source, source_origin,
        np.eye(3) if current is None else current, current_origin,
        valid.shape, PixelMath("float32"), geometry_parameters=PARAMETERS, **kwargs)


def test_build_registration_uses_measured_build_h_and_negative_canvas_origin(context):
    build_to_frame, photometry_to_frame = shift(1), shift(-2)
    result = fields(context, build_to_frame, origin=(-1, 0))
    np.testing.assert_array_equal(result["facies"], [[0, 0, 1, 9]] * 3)
    np.testing.assert_array_equal(result["depth_m"], [[0, 0, 1, 5]] * 3)
    np.testing.assert_array_equal(result["source_to_frame_h"], shift(2))
    assert not np.array_equal(result["source_to_frame_h"], photometry_to_frame)
    assert result["unknown_physical_mask"][:, :2].all()
    assert result["physical_supported_mask"][:, 2:].all()


def test_facies_remain_nearest_categories_while_continuous_fields_and_coverage_interpolate(context):
    result = fields(context, shift(.25))
    np.testing.assert_array_equal(result["facies"][:, 1], 9)
    np.testing.assert_allclose(result["depth_m"][:, 1], 4.)
    np.testing.assert_allclose(result["porosity"][:, 1], .35)
    assert result["domain"][:, 0].all()
    assert not result["physical_supported_mask"][:, 0].any()
    assert result["unknown_physical_mask"][:, 0].all()


def test_unknown_geometry_stays_unknown_only_inside_target_domain_without_mutation(context):
    context.depth_m[1, 1] = 0
    source_domain, domain = np.ones((3, 4), bool), np.ones((3, 4), bool)
    source_domain[0, 0] = False
    domain[2, 3] = False
    originals = [value.copy() for value in (context.facies, context.depth_m, context.porosity,
                                          source_domain, domain)]
    result = fields(context, source_domain=source_domain, domain=domain)
    assert result["unknown_physical_mask"][0, 0] and result["unknown_physical_mask"][1, 1]
    assert not result["unknown_physical_mask"][2, 3]
    assert not result["physical_supported_mask"][2, 3]
    assert result["depth_m"][1, 1] == 0 and result["facies"][1, 1] == 9
    for actual, original in zip((context.facies, context.depth_m, context.porosity,
                                 source_domain, domain), originals):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("category", [-1, 256, 1.5])
def test_invalid_facies_refuse_before_original_uint8_transport(context, category):
    values = context.facies.astype(np.float64 if category == 1.5 else np.int32)
    values[0, 0] = category
    with pytest.raises(ValueError, match="integer categories"):
        fields(replace(context, facies=values))


@pytest.mark.parametrize("fault", ["saved-grid", "field-grid", "source-domain", "target-domain"])
def test_physical_fields_reject_different_grids_and_nonboolean_domains(context, fault):
    source_domain, domain = np.ones((3, 4), bool), np.ones((3, 4), bool)
    if fault == "saved-grid":
        context = replace(context, grid=dict(context.grid, num_voxels=[3, 5]))
    elif fault == "field-grid":
        context = replace(context, depth_m=context.depth_m[:, :3])
    elif fault == "source-domain":
        source_domain = source_domain.astype(np.uint8)
    else:
        domain = domain[:, :3]
    with pytest.raises(ValueError, match="grid|boolean"):
        fields(context, source_domain=source_domain, domain=domain)


def test_reference_combines_both_measured_build_transforms_and_canvas_origins():
    rgb = np.arange(1, 37, dtype=np.float32).reshape(3, 4, 3)
    valid = np.ones((3, 4), bool)
    result = reference(rgb, valid, source=shift(10, 4), source_origin=(-2, -1),
                       current=shift(11, 5), current_origin=(-1, 0))
    np.testing.assert_array_equal(result["rgb"], rgb)
    np.testing.assert_array_equal(result["source_to_current_h"], np.eye(3))
    assert result["valid"].all() and (result["coverage"] == 1).all()


def test_reference_fractional_border_is_not_promoted_to_full_support():
    rgb, valid = np.ones((3, 4, 3), np.float32), np.ones((3, 4), bool)
    result = reference(rgb, valid, current=shift(.25), require_positive=True)
    np.testing.assert_array_equal(result["coverage"][:, 0], .75)
    np.testing.assert_array_equal(result["rgb"][:, 0], .75)
    assert not result["valid"][:, 0].any() and result["valid"][:, 1:].all()


@pytest.mark.parametrize("positive,zero_supported", [(False, True), (True, False)])
def test_baseline_zero_and_positive_clearwater_have_separate_explicit_support(positive, zero_supported):
    rgb, valid = np.ones((3, 4, 3), np.float32), np.ones((3, 4), bool)
    rgb[1, 1] = 0
    result = reference(rgb, valid, require_positive=positive)
    assert bool(result["valid"][1, 1]) is zero_supported
    assert result["valid"][0].all()
    np.testing.assert_array_equal(rgb[1, 1], 0)


def test_invalid_source_values_are_zeroed_before_warp_without_mutation():
    rgb, valid = np.ones((3, 4, 3), np.float32), np.ones((3, 4), bool)
    rgb[1, 0] = np.nan
    valid[1, 0] = False
    original_rgb, original_valid = rgb.copy(), valid.copy()
    result = reference(rgb, valid, current=shift(.5))
    assert np.isfinite(result["rgb"]).all()
    np.testing.assert_array_equal(result["rgb"][1, 1], .5)
    assert result["coverage"][1, 1] == .5 and not result["valid"][1, 1]
    np.testing.assert_array_equal(rgb, original_rgb)
    np.testing.assert_array_equal(valid, original_valid)


@pytest.mark.parametrize("fault", ["rgb-dtype", "valid-grid", "valid-dtype", "singular-h"])
def test_reference_rejects_incompatible_storage_grids_and_transforms(fault):
    rgb, valid, current = np.ones((3, 4, 3), np.float32), np.ones((3, 4), bool), np.eye(3)
    if fault == "rgb-dtype":
        rgb = rgb.astype(np.float64)
    elif fault == "valid-grid":
        valid = valid[:, :3]
    elif fault == "valid-dtype":
        valid = valid.astype(np.uint8)
    else:
        current[1] = current[0]
    with pytest.raises((ValueError, TypeError), match="float32|boolean|singular"):
        reference(rgb, valid, current=current)
