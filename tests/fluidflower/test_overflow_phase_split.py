from copy import deepcopy

import numpy as np
import pytest

from darsia.fluidflower.mass import overflow_phase_split as subject
from darsia.fluidflower.mass.overflow_steering import GasPopulation


def state(run, pressure, volume, *, time=1., dose=100., density=2.):
    return dict(run=run, elapsed_h=time, pressure_bar=pressure, temperature_c=23.,
        target_volume_m3=volume, injected_mass_g=dose, gas_density_kg_m3=density,
        stage=("rest", 1, 0), compatibility={"rig": "same", "chemistry": "same"},
        kind="own_measurement", estimated=False, source={"path": run+"-measured.json"})


def test_own_observations_are_exact_and_interpolation_is_only_within_that_run():
    curve = dict(t=np.array([0., 1., 2.]), vC=np.array([0., 12., 20.]))
    value = subject.own_target(curve, run="one", elapsed_h=1., source={"path": "own.json"})
    assert value["target_volume_m3"] == pytest.approx(12e-6)
    assert value["kind"] == "own_measurement" and not value["estimated"]
    assert not value["pooled"] and not value["shrunk"]
    assert subject.own_target(curve, run="one", elapsed_h=1.5, source={"path": "own.json"})["target_volume_m3"] == pytest.approx(16e-6)
    with pytest.raises(subject.TargetUnavailable, match="outside"):
        subject.own_target(curve, run="one", elapsed_h=3., source={"path": "own.json"})
    with pytest.raises(subject.TargetUnavailable, match="negative"):
        subject.own_target(dict(t=[0., 1.], vC=[-2., -1.]), run="one", elapsed_h=.5, source={"path": "own.json"})


def test_nearest_pressure_estimate_uses_amount_normalisation_and_fit_donors_only():
    recipient = state("recipient", 1.01, 0., dose=200., density=4.)
    donors = [state("near", 1., .01), state("far", 1.2, .02), state("holdout", 1.01, .04)]
    value = subject.pressure_matched_target(recipient, donors, fit_runs=["recipient", "near", "far"])
    assert value["kind"] == "pressure_matched_estimate" and value["estimated"]
    assert value["target_volume_m3"] == pytest.approx(.01)
    assert value["estimated_gas_fraction_of_injected_mass"] == pytest.approx(.2)
    assert [row["run"] for row in value["donors"]] == ["near"]
    assert value["pressure_difference_bar"] == pytest.approx(.01)
    changed = deepcopy(donors)
    changed[-1]["target_volume_m3"] = 1e5
    assert subject.pressure_matched_target(recipient, changed, fit_runs=["recipient", "near", "far"]) == value


def test_comparable_donor_interpolation_cannot_cross_injection_or_physical_context():
    recipient = state("recipient", 1., 0., time=2.)
    donors = [state("donor", 1., .01, time=1.), state("donor", 1., .02, time=3.)]
    result = subject.pressure_matched_target(recipient, donors, fit_runs=["donor", "recipient"])
    assert result["target_volume_m3"] == pytest.approx(.015)
    donors[-1]["stage"] = ("injecting", 1, 1)
    with pytest.raises(subject.TargetUnavailable, match="comparable"):
        subject.pressure_matched_target(recipient, donors, fit_runs=["donor", "recipient"])
    donors[-1]["stage"] = donors[0]["stage"]
    donors[-1]["compatibility"] = {"rig": "different"}
    with pytest.raises(subject.TargetUnavailable):
        subject.pressure_matched_target(recipient, donors, fit_runs=["donor", "recipient"])


def test_measurement_can_exceed_model_gas_mask_and_zero_scores_without_capping():
    population = GasPopulation(probability=np.array([[1., 0.]], np.float32),
        maximum_saturation=np.array([[.8, .8]], np.float32),
        baseline_saturation=np.array([[.8, 0.]], np.float32),
        pore_volume_m3=np.ones((1, 2), np.float64), eligible=np.ones((1, 2), bool),
        phase_gas=np.array([[True, False]]), certain_without_probability=np.zeros((1, 2), bool))
    value = subject.distribute_target(population, 1.2)
    assert value["actual_volume_m3"] == pytest.approx(1.2, rel=2e-7)
    assert value["capped_volume_m3"] == 0
    assert value["saturation_free_gas"][0, 1] > 0
    assert value["extended_zero_score_pixels"] == 1
    with pytest.raises(ValueError, match="physical CO2 pore capacity"):
        subject.distribute_target(population, 2.)


def test_injection_phase_handles_repeated_injections():
    stages = [dict(start_h=0., end_h=1.), dict(start_h=3., end_h=4.)]
    assert subject.injection_phase(stages, .5) == ("injecting", 0, 1)
    assert subject.injection_phase(stages, 2.) == ("rest", 1, 0)
    assert subject.injection_phase(stages, 3.5) == ("injecting", 1, 1)
    assert subject.injection_phase(stages, 5.) == ("rest", 2, 0)
