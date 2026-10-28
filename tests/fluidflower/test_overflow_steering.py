"""Finite analytical examples only; no source or campaign qualification."""
from copy import deepcopy

import numpy as np
import pytest

from darsia.fluidflower.mass.overflow_steering import (
    apply_gas_fields, gas_population, gas_report, redistribute_gas, shrink_run_levels,
)


def population(*, unscored_gas=False):
    arrays = dict(pixel_probability=np.array([[.25, 1]], np.float32),
                  regional_probability=np.array([[.8, .8]], np.float32),
                  branch_support=np.ones((1, 2), np.uint8),
                  later_rule_changes=np.zeros((1, 2), np.uint16))
    if unscored_gas:
        arrays["pixel_probability"][0, 1] = np.nan
        arrays["regional_probability"][0, 1] = np.nan
        arrays["branch_support"][0, 1] = 0
    labels = np.array([[1, 2]], np.uint8)
    keep = np.ones((1, 2), bool)
    volume = np.ones((1, 2), np.float64)
    swi = np.zeros((1, 2), np.float64)
    return gas_population(arrays, labels, keep, volume, swi), arrays, labels


def test_probability_threshold_has_hand_computed_volume_and_signed_report():
    pop, arrays, labels = population()
    original_arrays = deepcopy(arrays)
    original_labels = labels.copy()
    result = redistribute_gas(pop, target_volume_m3=.5)
    # Both pixels are below p*=2.5: .25/2.5 + 1/2.5 = .5 m3.
    assert result["threshold"] == 2.5
    np.testing.assert_array_equal(result["saturation_free_gas"], np.array([[.1, .4]], np.float32))
    assert abs(result["actual_volume_m3"] - .5) <= np.finfo(np.float32).eps
    assert result["float32_volume_residual_m3"] == result["actual_volume_m3"] - .5
    report = gas_report(pop, result, gas_density=np.full((1, 2), 2, np.float32))
    assert report["baseline_gas_mass_g"] == 2000
    assert report["gas_mass_g"] == pytest.approx(1000, rel=1e-7)
    assert report["gas_change_g"] == pytest.approx(-1000, rel=1e-7)
    assert report["gas_change_share"] == pytest.approx(-1, rel=1e-7)
    assert report["gas_on_non_gas_label_g"] == pytest.approx(200, rel=1e-7)
    assert report["gas_on_non_gas_label_share"] == pytest.approx(.2, rel=1e-7)
    assert not result["labels_modified"] and not result["geometry_modified"]
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], original_arrays[key])
    np.testing.assert_array_equal(labels, original_labels)


@pytest.mark.parametrize("unscored", [False, True])
def test_gas_without_its_own_positive_probability_uses_one(unscored):
    pop, arrays, labels = population(unscored_gas=unscored)
    if not unscored:
        arrays["pixel_probability"][0, 1] = 0
        pop = gas_population(arrays, labels, np.ones((1, 2), bool),
                             np.ones((1, 2), np.float64), np.zeros((1, 2), np.float64))
    assert pop.probability[0, 1] == 1
    assert pop.certain_without_probability.tolist() == [[False, True]]
    assert redistribute_gas(pop, target_volume_m3=.5)["certain_without_probability_pixels"] == 1


def test_cap_and_zero_are_explicit_without_rescaling_input_geometry():
    pop, _arrays, _labels = population()
    capped = redistribute_gas(pop, target_volume_m3=3)
    assert capped["available_volume_m3"] == capped["target_volume_m3"] == 2
    assert capped["capped_volume_m3"] == 1 and capped["actual_volume_m3"] == 2
    np.testing.assert_array_equal(capped["saturation_free_gas"], np.ones((1, 2), np.float32))
    empty = redistribute_gas(pop, target_volume_m3=0)
    assert empty["threshold"] is None and empty["mode"] == "zero"
    assert empty["actual_volume_m3"] == 0
    report = gas_report(pop, empty, gas_density=np.ones((1, 2), np.float32))
    assert report["gas_change_share"] is None and report["gas_on_non_gas_label_share"] is None
    assert not pop.pore_volume_m3.flags.writeable


def test_masked_physical_unknowns_and_background_cannot_receive_gas():
    _pop, arrays, labels = population()
    arrays["pixel_probability"][0, 0] = 0
    arrays["regional_probability"][0, 0] = 0
    labels[0, 0] = 0
    pop = gas_population(arrays, labels, np.array([[True, False]]),
                         np.array([[1, np.nan]], np.float64), np.array([[0, np.nan]], np.float64))
    result = redistribute_gas(pop, target_volume_m3=1)
    assert result["available_volume_m3"] == 0 and result["capped_volume_m3"] == 1
    assert not result["saturation_free_gas"].any()


def test_freed_water_uses_original_local_concentration_and_float32_order():
    pop, _arrays, _labels = population()
    result = redistribute_gas(pop, target_volume_m3=.5)
    original = dict(aqueous=np.array([[.2, 1]], np.float32),
                    gas=np.array([[0, 1]], np.float32),
                    concentration=np.array([[.2, 1]], np.float32),
                    mass_aqueous=np.array([[.2, 0]], np.float32),
                    mass_gas=np.array([[0, 2]], np.float32),
                    mass_total=np.array([[.2, 2]], np.float32))
    before = {name: value.tobytes() for name, value in original.items()}
    changed = apply_gas_fields(original, result, population=pop,
                               solubility=np.ones((1, 2), np.float32),
                               gas_density=np.full((1, 2), 2, np.float32))
    np.testing.assert_allclose(changed["mass_aqueous"], [[.18, .6]], rtol=1e-7)
    np.testing.assert_allclose(changed["mass_gas"], [[.2, .8]], rtol=1e-7)
    np.testing.assert_allclose(changed["mass_total"], [[.38, 1.4]], rtol=1e-7)
    assert all(value.dtype == np.float32 for value in changed.values())
    assert {name: value.tobytes() for name, value in original.items()} == before
    assert changed["concentration"] is original["concentration"]


@pytest.mark.parametrize("fault", ["dtype", "unscored", "range", "rule", "physical"])
def test_original_probability_and_physical_guards_remain_required(fault):
    _pop, arrays, labels = population()
    volume = np.ones((1, 2), np.float64)
    if fault == "dtype":
        arrays["pixel_probability"] = arrays["pixel_probability"].astype(np.float64)
    elif fault == "unscored":
        arrays["branch_support"][0, 0] = 0
    elif fault == "range":
        arrays["pixel_probability"][0, 0] = 1.01
    elif fault == "rule":
        arrays["later_rule_changes"][0, 0] = 32768
    else:
        volume[0, 0] = -1
    with pytest.raises(ValueError):
        gas_population(arrays, labels, np.ones((1, 2), bool), volume,
                       np.zeros((1, 2), np.float64))


def test_group_uses_fit_measurements_only_and_preserves_own_heldout_measurement():
    observed = {"fit-a": dict(ratio=.5, sigma=.1), "fit-b": dict(ratio=1., sigma=.2),
                "heldout": dict(ratio=100., sigma=.001)}
    result = shrink_run_levels(observed, fit_runs=("fit-a", "fit-b"),
                               runs=("fit-a", "fit-b", "heldout", "missing"))
    # Precision weights are 100,25: group=.6, Q=5, C=40, tau2=(5-1)/40=.1.
    assert result["group_ratio"] == pytest.approx(.6)
    assert result["between_run_variance"] == pytest.approx(.1)
    assert result["group_fit_runs"] == ["fit-a", "fit-b"]
    assert result["runs"]["fit-a"]["level"] == pytest.approx(28/55)
    assert result["runs"]["fit-b"]["level"] == pytest.approx(31/35)
    assert result["runs"]["missing"]["level"] == pytest.approx(.6)
    assert result["runs"]["missing"]["group_fallback"]
    assert result["runs"]["heldout"]["level"] == 1
    assert result["runs"]["heldout"]["ratio_above_one"]


def test_no_between_run_variance_shrinks_to_the_group_and_caps_levels():
    result = shrink_run_levels({"a": dict(ratio=2, sigma=.1), "b": dict(ratio=2, sigma=.2)},
                               fit_runs=("a", "b"), runs=("a", "b", "missing"))
    assert result["between_run_variance"] == 0 and result["group_ratio"] == 2
    assert result["group_level"] == 1
    assert all(row["level"] == 1 and row["weight"] == 0 for row in result["runs"].values())


def test_finite_measurements_cannot_publish_an_overflowed_between_run_variance():
    # Each squared deviation, Q and C are finite, but Q/C exceeds float64.
    with pytest.raises(ValueError, match="between-run measurement variance"):
        shrink_run_levels({"a": dict(ratio=0., sigma=10.), "b": dict(ratio=2.6e154, sigma=10.)},
                          fit_runs=("a", "b"), runs=("a", "b"))


def test_field_application_cannot_exceed_the_same_physical_swi_capacity():
    _pop, arrays, labels = population()
    pop = gas_population(arrays, labels, np.ones((1, 2), bool),
                         np.ones((1, 2), np.float64), np.full((1, 2), .5, np.float64))
    original = dict(aqueous=np.array([[.2, 1]], np.float32),
                    gas=np.array([[0, .5]], np.float32),
                    concentration=np.array([[.2, 1]], np.float32),
                    mass_aqueous=np.array([[.2, .5]], np.float32),
                    mass_gas=np.array([[0, 1]], np.float32),
                    mass_total=np.array([[.2, 1.5]], np.float32))
    wrong = dict(saturation_free_gas=np.array([[.1, .75]], np.float32))
    with pytest.raises(ValueError, match="physical gas saturation"):
        apply_gas_fields(original, wrong, population=pop,
                         solubility=np.ones((1, 2), np.float32),
                         gas_density=np.full((1, 2), 2, np.float32))


@pytest.mark.parametrize("sigma", [0, -1, float("nan"), 1e-300])
def test_invalid_measurement_uncertainty_is_not_replaced_by_a_floor(sigma):
    with pytest.raises(ValueError):
        shrink_run_levels({"a": dict(ratio=.5, sigma=sigma), "b": dict(ratio=.7, sigma=.2)},
                          fit_runs=("a", "b"), runs=("a", "b"))
