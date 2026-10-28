"""Categorical support and source-selected boundary populations on tiny grids."""

from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.phase import domain


@pytest.fixture
def parameters():
    return dict(
        coarse_coverage_threshold=0.999,
        upper_bound_rejection_threshold=0.5,
        nominal_rule_b_pixel_area_mm2=0.027053,
        gas_minimum_area_mm2=2.5,
        gas_connectivity=4,
        interpolation_contract="interpolation",
        domain_reduction_description="explicit coverage reduction",
        upper_bound_definition="explicit interpolation upper bound",
        tie_fill_contract="equal-distance",
        prior_fill_contract="prior-fill",
        seam_contract="scoped-seam",
    )


@pytest.fixture
def rule_parameters():
    return dict(
        barrier=dict(rule_contract="barrier"),
        literal_values=dict(
            validate_components_int_255=255,
            apply_barrier_rule_b_int_255=255,
            apply_barrier_rule_b_int_4=4,
            apply_barrier_rule_b_int_32000000=32_000_000,
            apply_barrier_rule_b_float_100_0=100.0,
        ),
    )


def retained(labels, eligible):
    shape = labels.shape
    return domain.RetainedSourceInferenceCache(
        labels.copy(),
        np.zeros(shape, bool),
        np.zeros(shape, bool),
        eligible.copy(),
        np.ones(shape, bool),
        None,
        dict(
            run="example",
            frame_index=0,
            current_source_sha256="a" * 64,
            prepared_before_fresh_inference=True,
            cache_enabled=True,
        ),
        np.zeros(shape, bool),
    )


def test_outer_ring_excludes_a_closed_internal_support_hole():
    mask = np.ones((5, 5), bool)
    mask[2, 2] = False
    ring = domain.outer_domain_ring(mask)
    assert ring[0].all() and ring[-1].all()
    assert ring[:, 0].all() and ring[:, -1].all()
    assert not ring[1:4, 1:4].any()


def test_interpolation_release_uses_source_exterior_not_internal_holes(parameters):
    closed_hole = np.ones((9, 9), bool)
    closed_hole[4, 4] = False
    footprint, audit = domain.old_domain_interpolation_footprint(
        closed_hole, (3, 3), parameters=parameters
    )
    assert not footprint.any()
    assert audit["exterior_rejected_coarse_cells"] == 0
    edge = np.ones((9, 9), bool)
    edge[0] = False
    footprint, audit = domain.old_domain_interpolation_footprint(
        edge, (3, 3), parameters=parameters
    )
    assert footprint[1].any() and not footprint[0].any()
    assert audit["exterior_rejected_coarse_cells"] == 3
    assert not audit["eligibility_selected_from_output_difference"]


def test_nearest_tie_preference_cannot_choose_a_farther_category(parameters):
    labels = np.array([[0, 255, 255, 255, 2]], np.uint8)
    grid = np.ones(labels.shape, bool)
    observed = labels != 255
    preferred = np.full(labels.shape, 2, np.uint8)
    eligible = ~observed
    audit = {}
    result, classes = domain.fill_rig_labels(
        labels,
        grid,
        observed,
        tie_break_reference=preferred,
        tie_break_mask=eligible,
        audit=audit,
        parameters=parameters,
    )
    assert result.tolist() == [[0, 0, 2, 2, 2]]
    assert np.array_equal(classes["modelled"], observed)
    assert audit["equal_distance_preferences"] == 1
    assert not classes["filled_observed"].any()
    assert np.array_equal(classes["filled_unobserved"], eligible)
    with pytest.raises(ValueError, match="No modelled"):
        domain.fill_rig_labels(
            np.full(labels.shape, 255, np.uint8),
            grid,
            np.zeros(labels.shape, bool),
            parameters=parameters,
        )


def test_prior_final_fill_is_preserved_without_demoting_newly_modelled(parameters):
    labels = np.ones((2, 3), np.uint8)
    cache = retained(labels, np.ones(labels.shape, bool))
    assert not cache.labels.flags.writeable
    fresh = np.full(labels.shape, 2, np.uint8)
    modelled = np.zeros(labels.shape, bool)
    modelled[0, 0] = True
    observed = np.zeros(labels.shape, bool)
    observed[0, 1:] = True
    unobserved = ~modelled & ~observed
    result, populations, audit = domain.preserve_prior_categorical_fill(
        cache,
        fresh,
        modelled=modelled,
        filled_observed=observed,
        filled_unobserved=unobserved,
        parameters=parameters,
    )
    assert result[0, 0] == 2 and (result[~modelled] == 1).all()
    assert np.array_equal(fresh, np.full(labels.shape, 2, np.uint8))
    assert populations["prior_fill_eligible_now_modelled_category_changed"][0, 0]
    assert audit["modelled_categories_unchanged"]
    with pytest.raises(ValueError, match="disjoint"):
        domain.preserve_prior_categorical_fill(
            cache,
            fresh,
            modelled=modelled,
            filled_observed=np.ones(labels.shape, bool),
            filled_unobserved=unobserved,
            parameters=parameters,
        )


def test_crop_embed_and_homography_preserve_coordinates_and_dtype():
    native = np.arange(6, dtype=np.float32).reshape(2, 3)
    box = (2, 1, 3, 2)
    full = domain.embed_native_canvas(
        native, source_box_xywh=box, phase_shape_hw=(5, 7), fill=np.nan
    )
    crop = domain.crop_native_canvas(full, box)
    assert full.dtype == np.float32 and np.array_equal(crop, native)
    assert np.shares_memory(crop, full) and np.isnan(full[0]).all()
    matrix = np.diag([2.0, 3.0, 1.0])
    result = domain.canvas_homography(matrix, source_box_xywh=box)
    assert np.array_equal(
        result, np.array([[2, 0, -2], [0, 3, -2], [0, 0, 1]], np.float64)
    )


def test_seam_filters_only_incident_gas_components(parameters, rule_parameters):
    old = np.zeros((5, 8), np.uint8)
    old[2, 1:3] = 2
    old[2, 6] = 2
    fresh = old.copy()
    fresh[2, 1] = 1
    release = np.zeros(old.shape, bool)
    release[2, 1] = True
    grid = np.ones(old.shape, bool)
    result, audit, masks = domain.revise_seam_labels(
        old,
        fresh,
        release,
        grid,
        old == 2,
        fresh == 2,
        grid,
        np.full(old.shape, 0.1, np.float32),
        open_components=np.ones(old.shape, np.uint8),
        source_component_ids=[1],
        barrier_mask=np.zeros(old.shape, bool),
        parameters=parameters,
        rule_parameters=rule_parameters,
    )
    assert result[2, 1:3].tolist() == [1, 1]
    assert result[2, 6] == 2  # This tiny unrelated component is not refiltered.
    assert not masks["affected_gas_graph"][2, 6]
    assert audit["co2_outside_ring_exact"]
    assert np.array_equal(old[2, 1:3], np.array([2, 2], np.uint8))


def test_seam_rejects_new_reachability_changes_outside_declared_scope(
    parameters, rule_parameters
):
    old = np.zeros((5, 7), np.uint8)
    old[1, 1:6] = 1
    old[3, 5] = 2
    fresh = old.copy()
    fresh[1, 3] = 0
    release = np.zeros(old.shape, bool)
    release[1, 3] = True
    components = np.zeros(old.shape, np.uint8)
    components[:, :3] = 1
    components[:, 4:] = 2
    grid = np.ones(old.shape, bool)
    with pytest.raises(ValueError, match="wider declared gas scope"):
        domain.revise_seam_labels(
            old,
            fresh,
            release,
            grid,
            old == 2,
            fresh == 2,
            grid,
            1.0,
            open_components=components,
            source_component_ids=[1],
            barrier_mask=components == 0,
            parameters=parameters,
            rule_parameters=rule_parameters,
        )


def test_merge_retains_old_fill_but_release_uses_a_separate_fresh_fill(
    parameters, rule_parameters
):
    shape = (3, 7)
    box = (2, 1, 3, 1)
    centre = SimpleNamespace(
        labels=np.full((1, 3), 2, np.uint8),
        domain=np.ones((1, 3), bool),
        flatfield_valid=np.array([[False, True, False]]),
        preminimum_gas=np.array([[False, True, False]]),
    )
    fresh = np.full(shape, 255, np.uint8)
    fresh[0, 0] = 0
    support = fresh != 255
    grid = np.ones(shape, bool)
    cache = retained(np.ones((1, 3), np.uint8), np.ones((1, 3), bool))
    release = np.zeros(shape, bool)
    release[1, 2] = True
    old_domain = domain.embed_native_canvas(
        centre.domain, source_box_xywh=box, phase_shape_hw=shape, fill=False
    )
    result, classes, audit, _ = domain.merge_native_and_rig(
        centre,
        fresh,
        support,
        grid,
        support,
        release=release,
        old_domain=old_domain,
        original_area=np.ones(shape, np.float32),
        open_components=np.ones(shape, np.uint8),
        barrier_mask=np.zeros(shape, bool),
        source_ids=[1],
        fresh_pre_minimum=np.zeros(shape, bool),
        source_box_xywh=box,
        parameters=parameters,
        rule_parameters=rule_parameters,
        prior_cache=cache,
    )
    assert result[1, 2:5].tolist() == [0, 2, 1]
    assert classes["modelled"][1, 3] and not classes["modelled"][1, 2]
    assert audit["inherited_fill"]["final_prior_fill_categories_exact"]
    assert np.array_equal(fresh[support], np.array([0], np.uint8))
