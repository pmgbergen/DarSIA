"""Small provider-exact mass and source-column cases, with no external IO."""

import numpy as np
import pytest

from darsia.fluidflower.mass.pixels import evaluate_pixels
from darsia.fluidflower.mass.production_precision import PixelMath


def _case(labels, precision="float32"):
    labels = np.asarray(labels, np.uint8)
    shape = labels.shape
    dtype = np.dtype(precision)
    return dict(
        domain=np.ones(shape, bool), visible=np.ones(shape, bool),
        geometry_known=np.ones(shape, bool), colour_available=np.ones(shape, bool),
        phase_labels=labels, relative_aqueous=np.full(shape, .23, dtype),
        swi=np.full(shape, .12, dtype), solubility=np.full(shape, 1.4, dtype),
        gas_density=np.full(shape, 2.3, dtype),
        transport_weight_l=np.full(shape, .25, np.float64),
        y_m=np.broadcast_to(np.arange(shape[0], 0, -1)[:, None] * .01, shape).copy(),
        max_distance_m=.020, gain=1.7, frozen_dilution_factor=.6,
        force_estimated=np.zeros(shape, bool), pixel_math=PixelMath(precision),
        unknown_without_capacity=np.zeros(shape, bool),
    )


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_own_fields_are_exact_original_provider_outputs(precision):
    case = _case([[1, 2], [1, 2]], precision)
    case["relative_aqueous"][:] = [[.123456789, .8], [1.9, -.2]]
    own = case["phase_labels"] == 1
    gas = case["phase_labels"] == 2
    expected = case["pixel_math"].selected_n1_fields(
        case["relative_aqueous"], gain=case["gain"], factor=case["frozen_dilution_factor"],
        swi=case["swi"], aqueous_known=own, gas_known=gas,
        solubility=case["solubility"], gas_density=case["gas_density"],
    )
    before = {key: value.copy() for key, value in case.items() if isinstance(value, np.ndarray)}
    result = evaluate_pixels(**case)
    for output, native in [("concentration_aqueous", "concentration"), ("saturation_free_gas", "gas"),
                           ("mass_total", "mass_total"), ("mass_aqueous", "mass_aqueous"),
                           ("mass_free_gas", "mass_gas")]:
        np.testing.assert_array_equal(result["maps"][output], expected[native])
        assert result["maps"][output].dtype == np.dtype(precision)
    assert result["M_measured_g"] == case["pixel_math"].integrate(
        expected["mass_total"], case["transport_weight_l"], case["domain"]
    )
    assert result["M_edge_g"] == result["U_g"] == 0.
    for key, value in before.items():
        np.testing.assert_array_equal(case[key], value)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_gas_donor_supplies_actual_concentration_once_with_target_geometry(precision):
    case = _case([[2], [1]], precision)
    case["visible"][1] = False
    case["solubility"][:] = [[1.75], [1.25]]
    case["transport_weight_l"][:] = [[.2], [.7]]
    case["relative_aqueous"][:] = .01
    result = evaluate_pixels(**case)
    concentration = result["maps"]["concentration_aqueous"]
    assert concentration[0, 0] == concentration[1, 0] == case["solubility"][0, 0]
    assert result["maps"]["saturation_free_gas"][1, 0] == 0
    assert result["maps"]["mass_free_gas"][1, 0] == 0
    assert result["M_edge_g"] == float(np.float64(concentration[0, 0]) * .7)
    assert result["donor_rows"][1, 0] == 0
    assert result["audit"]["donor_concentration_recalibrated"] is False
    np.testing.assert_array_equal(result["maps"]["phase_labels"], [[2], [1]])


def test_nearest_measured_donor_wins_and_estimates_do_not_chain():
    case = _case([[1, 1], [1, 1], [1, 1], [1, 1]])
    case["visible"][:] = False
    case["visible"][0, 0] = True
    case["visible"][1, 0] = True
    case["relative_aqueous"][0, 0], case["relative_aqueous"][1, 0] = .1, .8
    case["y_m"][:] = [[.06, .06], [.04, .04], [.025, .025], [.01, .01]]
    result = evaluate_pixels(**case)
    assert result["donor_rows"][2, 0] == 1
    assert result["maps"]["concentration_aqueous"][2, 0] == result["maps"]["concentration_aqueous"][1, 0]
    assert result["masks"]["concentration_unknown"][3, 0]
    assert result["masks"]["concentration_unknown"][:, 1].all()
    assert result["donor_rows"][3, 0] == -1
    assert not result["masks"]["donors"][2:].any()


def test_visible_pixel_without_a_colour_path_uses_only_an_in_range_column_donor():
    case = _case([[1], [1], [1]])
    case["colour_available"][1:] = False
    case["y_m"][:] = [[.04], [.02], [.019999999]]
    result = evaluate_pixels(**case)
    assert result["masks"]["column_estimated_aqueous"][1, 0]
    assert result["donor_distance_m"][1, 0] == .020
    assert result["maps"]["concentration_aqueous"][1, 0] == result["maps"]["concentration_aqueous"][0, 0]
    assert result["masks"]["concentration_unknown"][2, 0]
    assert not result["masks"]["measured_aqueous"][1:].any()


def test_estimated_gas_is_not_a_donor_for_another_unknown_pixel():
    case = _case([[2], [1]])
    case["visible"][:] = False
    result = evaluate_pixels(**case)
    assert result["masks"]["estimated_gas"][0, 0]
    assert not result["masks"]["donors"].any()
    assert result["masks"]["concentration_unknown"][1, 0]
    assert result["donor_rows"][1, 0] == -1


def test_forced_pixel_uses_column_instead_of_own_colour_and_cannot_donate():
    case = _case([[1], [1], [1]])
    case["force_estimated"][1] = True
    case["visible"][2] = False
    case["relative_aqueous"][:] = [[.1], [.9], [.8]]
    case["y_m"][:] = [[.04], [.025], [.01]]
    result = evaluate_pixels(**case)
    assert result["masks"]["concentration_estimated"][1, 0]
    assert result["maps"]["concentration_aqueous"][1, 0] == result["maps"]["concentration_aqueous"][0, 0]
    assert not result["masks"]["donors"][1, 0]
    assert result["masks"]["concentration_unknown"][2, 0]


def test_target_gas_uses_its_own_swi_and_density_even_when_unseen_or_forced():
    case = _case([[2, 2], [2, 2]])
    case["visible"][:] = [[True, False], [True, True]]
    case["force_estimated"][1, 1] = True
    case["swi"][:] = [[.32, .12], [.14, .2]]
    case["gas_density"][:] = [[1., 2.], [3., 4.]]
    native = case["pixel_math"].selected_n1_fields(
        case["relative_aqueous"], gain=case["gain"], factor=case["frozen_dilution_factor"],
        swi=case["swi"], aqueous_known=np.zeros((2, 2), bool), gas_known=np.ones((2, 2), bool),
        solubility=case["solubility"], gas_density=case["gas_density"],
    )
    result = evaluate_pixels(**case)
    np.testing.assert_array_equal(result["maps"]["saturation_free_gas"], native["gas"])
    np.testing.assert_array_equal(result["maps"]["mass_total"], native["mass_total"])
    np.testing.assert_array_equal(result["masks"]["measured_gas"], [[True, False], [True, False]])
    np.testing.assert_array_equal(result["masks"]["estimated_gas"], [[False, True], [False, True]])
    assert result["U_g"] == 0


def test_unknown_capacity_uses_target_phase_and_phase255_never_becomes_water():
    case = _case([[1, 2, 255, 0]])
    case["visible"][:] = False
    case["swi"][0, 1] = np.nan
    case["solubility"][:] = [[1., 3., 1., 1.]]
    case["gas_density"][:] = [[5., 2., 4., 1.]]
    case["transport_weight_l"][:] = [[.1, .2, .3, .4]]
    result = evaluate_pixels(**case)
    np.testing.assert_array_equal(result["masks"]["concentration_unknown"], [[True, True, True, False]])
    np.testing.assert_array_equal(result["masks"]["phase_unknown"], [[False, False, True, False]])
    np.testing.assert_array_equal(result["maps"]["phase_labels"], case["phase_labels"])
    assert result["U_g"] == np.sum(np.array([1., 3., 4.], np.float64) * [.1, .2, .3])
    assert np.isnan(result["maps"]["mass_total"][0, :3]).all()
    assert result["maps"]["mass_total"][0, 3] == 0
    assert result["M_measured_g"] == result["M_edge_g"] == 0


def test_undefined_geometry_returns_none_capacity_without_global_fallback():
    case = _case([[1, 255, 0]])
    case["geometry_known"][:] = False
    case["transport_weight_l"][:] = np.nan
    case["solubility"][:] = np.nan
    case["gas_density"][:] = np.nan
    result = evaluate_pixels(**case)
    assert result["U_g"] is None
    assert np.isnan(result["per_pixel_mass_g"]["unknown_capacity"][0, :2]).all()
    assert result["per_pixel_mass_g"]["unknown_capacity"][0, 2] == 0
    assert result["M_measured_g"] == result["M_edge_g"] == 0


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_explicit_phase_unknown_without_capacity_keeps_labels_and_cannot_donate(precision):
    case = _case([[1, 255], [255, 255], [1, 255]], precision)
    case["unknown_without_capacity"][1, 0] = True
    case["geometry_known"][1, 0] = False
    for name in ("transport_weight_l", "solubility", "gas_density"):
        case[name][1, 0] = np.nan
    case["solubility"][:, 1] = 1.
    case["gas_density"][:, 1] = 4.
    case["visible"][2, 0] = False
    original_labels = case["phase_labels"].copy()
    result = evaluate_pixels(**case)
    np.testing.assert_array_equal(result["maps"]["phase_labels"], original_labels)
    np.testing.assert_array_equal(case["phase_labels"], original_labels)
    assert result["masks"]["phase_unknown"][1, 0]
    assert result["masks"]["concentration_unknown"][1, 0]
    assert result["masks"]["unknown_without_capacity"][1, 0]
    assert not result["masks"]["measured"][1, 0]
    assert not result["masks"]["concentration_estimated"][1, 0]
    assert not result["masks"]["donors"][1, 0]
    assert np.isnan(result["maps"]["concentration_aqueous"][1, 0])
    assert np.isnan(result["maps"]["mass_total"][1, 0])
    assert result["per_pixel_mass_g"]["unknown_capacity"][1, 0] == 0.
    assert result["donor_rows"][2, 0] == 0  # passes the unknown pixel, never borrows it
    assert result["U_g"] == 3.  # other three phase-255 pixels retain local capacity
    assert result["audit"]["unknown_without_capacity_pixels"] == 1


def test_only_explicit_unknown_mask_can_remove_undefined_geometry_capacity():
    case = _case([[255, 255]])
    case["geometry_known"][:] = False
    case["transport_weight_l"][:] = np.nan
    case["solubility"][:] = np.nan
    case["gas_density"][:] = np.nan
    case["unknown_without_capacity"][0, 0] = True
    result = evaluate_pixels(**case)
    assert result["U_g"] is None
    assert result["per_pixel_mass_g"]["unknown_capacity"][0, 0] == 0.
    assert np.isnan(result["per_pixel_mass_g"]["unknown_capacity"][0, 1])
    case["unknown_without_capacity"][:] = True
    result = evaluate_pixels(**case)
    assert result["U_g"] == 0.
    assert result["M_measured_g"] == result["M_edge_g"] == 0.
    assert result["masks"]["concentration_unknown"].all()
    assert np.isnan(result["maps"]["concentration_aqueous"]).all()


@pytest.mark.parametrize("phase", [0, 1, 2])
def test_unknown_capacity_exception_cannot_discard_a_known_phase(phase):
    case = _case([[phase]])
    case["unknown_without_capacity"][:] = True
    with pytest.raises(ValueError, match="subset of in-domain phase 255"):
        evaluate_pixels(**case)


def test_unknown_capacity_exception_cannot_include_outside_domain():
    case = _case([[255]])
    case["domain"][:] = False
    case["unknown_without_capacity"][:] = True
    with pytest.raises(ValueError, match="subset of in-domain phase 255"):
        evaluate_pixels(**case)


def test_hardware_and_outside_pixels_never_supply_mass_or_donors():
    case = _case([[2], [1]])
    case["domain"][0, 0] = False
    case["visible"][1, 0] = False
    result = evaluate_pixels(**case)
    assert not result["masks"]["donors"].any()
    assert result["masks"]["concentration_unknown"][1, 0]
    assert result["maps"]["mass_total"][0, 0] == 0
    assert result["M_measured_g"] == result["M_edge_g"] == 0


def test_float64_weights_are_applied_before_reduction():
    case = _case([[1, 1, 1]])
    case["transport_weight_l"][:] = [[1e8 + .125, 1e-8, .123456789]]
    result = evaluate_pixels(**case)
    native = result["maps"]["mass_total"]
    expected = np.sum(native.astype(np.float64) * case["transport_weight_l"], dtype=np.float64)
    rounded = np.sum(native * case["transport_weight_l"].astype(np.float32), dtype=np.float64)
    assert result["M_measured_g"] == expected
    assert result["M_measured_g"] != rounded


@pytest.mark.parametrize(
    "phase,visible,field", [(1, True, "M_measured_g"), (2, False, "M_edge_g"), (255, False, "U_g")]
)
def test_scalar_reductions_keep_original_axis_zero_order_for_positive_masses(phase, visible, field):
    case = _case(np.full((2, 3), phase, np.uint8), "float64")
    case["visible"][:] = visible
    case["relative_aqueous"][:] = 1.
    case["swi"][:] = 0.
    case["solubility"][:] = 1.
    case["gas_density"][:] = 1.
    case["gain"] = case["frozen_dilution_factor"] = 1.
    case["transport_weight_l"][:] = [[1e16, 1., 1.], [1., 1., 1.]]
    result = evaluate_pixels(**case)
    expected = case["pixel_math"].integrate(
        np.ones((2, 3), np.float64), case["transport_weight_l"], case["domain"]
    )
    assert expected == 10000000000000004.
    assert expected != np.sum(case["transport_weight_l"], dtype=np.float64)
    assert result[field] == expected


@pytest.mark.parametrize("field", ["visible", "geometry_known", "colour_available", "force_estimated", "unknown_without_capacity"])
def test_masks_must_remain_boolean_on_the_same_grid(field):
    case = _case([[1]])
    case[field] = np.ones((1, 1), np.uint8)
    with pytest.raises(TypeError):
        evaluate_pixels(**case)


@pytest.mark.parametrize("field", ["relative_aqueous", "swi", "solubility", "gas_density"])
def test_working_pixel_precision_is_not_silently_changed(field):
    case = _case([[1]])
    case[field] = case[field].astype(np.float64)
    with pytest.raises(TypeError):
        evaluate_pixels(**case)


@pytest.mark.parametrize("field,value", [("gain", 0.), ("gain", np.nan), ("frozen_dilution_factor", 1.01)])
def test_invalid_frozen_parameters_fail(field, value):
    case = _case([[1]])
    case[field] = value
    with pytest.raises(ValueError):
        evaluate_pixels(**case)


def test_nonphysical_weights_swi_and_measured_signal_fail():
    case = _case([[2]])
    case["swi"][:] = 1.1
    with pytest.raises(ValueError):
        evaluate_pixels(**case)
    case = _case([[1]])
    case["transport_weight_l"][:] = 0
    with pytest.raises(ValueError):
        evaluate_pixels(**case)
    case = _case([[1]])
    case["relative_aqueous"][:] = np.nan
    with pytest.raises(ValueError):
        evaluate_pixels(**case)
def test_n16_no_adjacency_is_zero_without_inventing_a_capacity():
    from darsia.fluidflower.mass.pixels import hidden_mass_by_adjacency
    hidden = np.array([[True, False]], bool)
    result = hidden_mass_by_adjacency(hidden_mask=hidden, elapsed_h=2.25,
        adjacency=dict(co2_adjacent=False, touching_length_mm=0., distance_limit_mm=20.))
    assert result["U_g"] == 0
    assert result["U_layer"] is None
    assert result["receipt"]["status"] == "zero by adjacency rule"
    assert result["receipt"]["elapsed_h"] == 2.25
    np.testing.assert_array_equal(result["hidden_mask"], hidden)
    np.testing.assert_array_equal(result["U_per_pixel_g"], [[0., 0.]])


def test_n16_adjacency_uses_only_supplied_hidden_capacity_and_keeps_time_length():
    from darsia.fluidflower.mass.pixels import hidden_mass_by_adjacency
    hidden = np.array([[True, False]], bool)
    capacity = np.array([[.25, 99.]], np.float64)
    result = hidden_mass_by_adjacency(hidden_mask=hidden, elapsed_h=24.,
        adjacency=dict(co2_adjacent=True, touching_length_mm=7.5, distance_limit_mm=20.),
        round3_capacity_g=capacity)
    assert result["U_g"] == .25
    assert result["receipt"]["touching_length_mm"] == 7.5
    assert result["receipt"]["elapsed_h"] == 24.
    assert result["capacity_formula_created"] is False
    np.testing.assert_array_equal(result["U_per_pixel_g"], [[.25, 0.]])
    np.testing.assert_array_equal(capacity, [[.25, 99.]])


@pytest.mark.parametrize("capacity", [None, [[np.nan]], [[-1.]]])
def test_n16_adjacent_hidden_capacity_cannot_be_guessed_or_replaced(capacity):
    from darsia.fluidflower.mass.pixels import hidden_mass_by_adjacency
    with pytest.raises(ValueError, match="round-3 capacity"):
        hidden_mass_by_adjacency(hidden_mask=np.ones((1, 1), bool), elapsed_h=2.25,
            adjacency=dict(co2_adjacent=True, touching_length_mm=0., distance_limit_mm=20.),
            round3_capacity_g=capacity)
