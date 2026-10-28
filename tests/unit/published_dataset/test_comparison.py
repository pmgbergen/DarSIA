"""Verdicts under the cross-machine rule: zero, half a limit and just beyond it."""

from copy import deepcopy
import math

import numpy as np
import pytest

from darsia.fluidflower.published_dataset import comparison
from darsia.fluidflower.published_dataset.comparison import IDENTICAL, OUTSIDE, WITHIN
from darsia.fluidflower.published_dataset.manifest import DEFAULT_RULE
from tests.unit.published_dataset.synthetic import layer_arrays

MASSES = {"total": 1.0, "aqueous": 0.6, "gas": 0.4}


def _frame(masses=None, layer=None, case="r_f1"):
    expected_layer = layer_arrays(0, 2)
    return comparison.compare_frame("r", case, MASSES, masses or dict(MASSES), expected_layer,
                                    layer or deepcopy(expected_layer), DEFAULT_RULE)


def test_identical_frames_and_their_zero_margins():
    frame = _frame()
    assert frame.verdict == IDENTICAL and frame.first_failure is None
    assert all(check.share == 0 for check in frame.checks)
    as_dict = frame.as_dict()
    assert as_dict["verdict"] == IDENTICAL and {row["kind"] for row in as_dict["checks"]} == {
        "label", "mass", "map"}


@pytest.mark.parametrize("factor, verdict", [(0.5, WITHIN), (1.2, OUTSIDE)])
def test_masses_at_half_a_limit_and_beyond(factor, verdict):
    limit = DEFAULT_RULE["masses"]["relative"] * 0.4 + DEFAULT_RULE["masses"]["absolute"]
    frame = _frame(masses=dict(MASSES, gas=0.4 + factor * limit))
    assert frame.verdict == verdict
    gas = next(check for check in frame.checks if check.quantity == "gas mass")
    assert gas.share == pytest.approx(factor)


@pytest.mark.parametrize("factor, verdict", [(0.5, WITHIN), (2.0, OUTSIDE)])
def test_maps_at_half_a_limit_and_beyond(factor, verdict):
    expected = layer_arrays(0, 2)
    layer = deepcopy(expected)
    per_cell = DEFAULT_RULE["maps"]["relative_to_frame_maximum"] * float(np.max(expected["cell_mass_gas_g"]))
    layer["cell_mass_gas_g"] = expected["cell_mass_gas_g"].astype(np.float64)
    layer["cell_mass_gas_g"][1, 1] += factor * per_cell
    layer["cell_mass_gas_g"] = layer["cell_mass_gas_g"].astype(np.float32)
    frame = _frame(layer=layer)
    assert frame.verdict == verdict
    check = next(check for check in frame.checks if check.quantity == "cell_mass_gas_g")
    if verdict == OUTSIDE:
        assert frame.first_failure["quantity"] == "cell_mass_gas_g" and math.isinf(check.share) is False
    else:
        assert check.passed and check.share == pytest.approx(0.5, rel=1e-3)


def test_a_label_changed_cell_is_exempt_from_the_map_limit_and_counts_as_label_change():
    expected = layer_arrays(0, 2)
    layer = deepcopy(expected)
    layer["phase_dissolved_area_fraction"] = expected["phase_dissolved_area_fraction"].copy()
    layer["phase_dissolved_area_fraction"][0, 0] += 0.5
    layer["concentration_aqueous"] = expected["concentration_aqueous"].copy()
    layer["concentration_aqueous"][0, 0] += 100.0
    frame = _frame(layer=layer)
    assert frame.verdict == WITHIN
    co2 = next(check for check in frame.checks if check.quantity == "changed CO2 area")
    assert co2.difference == pytest.approx(0.5 * 1e-6 * 1e4, rel=1e-5) and co2.passed
    concentration = next(check for check in frame.checks if check.quantity == "concentration_aqueous")
    assert concentration.passed and concentration.difference == 0.0
    layer["concentration_aqueous"][1, 0] += 100.0  # Not label-changed: the map limit applies.
    assert _frame(layer=layer).first_failure["quantity"] == "concentration_aqueous"


def test_a_label_change_beyond_the_limit_is_outside():
    expected = layer_arrays(0, 2)
    layer = deepcopy(expected)
    layer["phase_gas_area_fraction"] = np.full(expected["phase_gas_area_fraction"].shape, 1.0, np.float32)
    frame = _frame(layer=layer)
    assert frame.verdict == OUTSIDE and frame.first_failure["quantity"] == "changed CO2 area"


def test_missing_arrays_and_masses_are_outside():
    layer = layer_arrays(0, 2)
    del layer["cell_mass_total_g"]
    assert _frame(layer=layer).first_failure["detail"] == "missing or of another shape"
    frame = _frame(masses=dict(MASSES, total=None))
    assert frame.verdict == OUTSIDE and frame.first_failure["quantity"] == "total mass"
    without_labels = {key: value for key, value in layer_arrays(0, 2).items() if not key.startswith("phase_")}
    assert comparison.label_checks(without_labels, without_labels, DEFAULT_RULE) == ([], None)


def test_summary_takes_the_worst_verdict_and_names_the_first_divergence():
    limit = DEFAULT_RULE["masses"]["relative"] * 0.4 + DEFAULT_RULE["masses"]["absolute"]
    frames = [_frame(case="r_f1"), _frame(masses=dict(MASSES, gas=0.4 + 0.5 * limit), case="r_f2"),
              _frame(masses=dict(MASSES, gas=0.4 + 3 * limit), case="r_f3"),
              comparison.compare_frame("s", "s_f1", MASSES, dict(MASSES), layer_arrays(1, 0),
                                       layer_arrays(1, 0), DEFAULT_RULE)]
    summary = comparison.summarize(frames)
    assert summary["dataset"] == OUTSIDE and summary["reproduced"] is False
    assert summary["runs"] == {"r": OUTSIDE, "s": IDENTICAL}
    assert summary["frames"] == {IDENTICAL: 2, WITHIN: 1, OUTSIDE: 1}
    assert summary["first_divergence"]["case_id"] == "r_f3"
    assert summary["first_divergence"]["quantity"] == "gas mass"
    assert summary["largest_share_of_limit"]["mass"] == pytest.approx(3.0)
    assert comparison.summarize(frames[:2])["reproduced"] is True
    assert comparison.worst([IDENTICAL, WITHIN]) == WITHIN
    with pytest.raises(ValueError):
        comparison.worst([])


def test_layers_load_without_pickles(tmp_path):
    arrays = layer_arrays(0, 1)
    np.savez(tmp_path / "layer.npz", **arrays)
    loaded = comparison.load_layer(tmp_path / "layer.npz")
    assert set(loaded) == set(arrays) and np.array_equal(loaded["cell_mass_gas_g"], arrays["cell_mass_gas_g"])
