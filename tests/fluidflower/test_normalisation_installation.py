"""Small synthetic optical inputs protect the saved method and donor scope."""

from copy import deepcopy
from pathlib import Path
import subprocess

import numpy as np
import pytest

from darsia.fluidflower.native import normalisation_installation as installation
from darsia.fluidflower.native import normalisation_residual as residual
from darsia.fluidflower.native import photometry
from tests.fluidflower.test_native_photometry import originals, parameters


@pytest.fixture
def inputs(parameters):
    shape = (48, 64)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    signal = (np.array([5., 5.2, 4.9])[:, None, None] + .0001 * xx + .0002 * yy).astype(np.float32)
    runs = ("ac19", "ac28", "ac50", "ac60")
    xy = np.column_stack((np.tile(np.linspace(2, 10, 4), 6), np.repeat(np.linspace(2, 12, 6), 4)))
    chart = np.broadcast_to(np.array([5., 5.2, 4.9], np.float32)[:, None], (3, 24)).copy()
    return dict(
        baseline_rgb={run: signal.copy() for run in runs},
        fit_support={run: np.ones(shape, np.int16) for run in runs},
        chart_xy={run: xy.copy() for run in runs},
        chart_log_rgb={run: chart.copy() for run in runs},
        chart_clipped={run: np.zeros(24, bool) for run in runs},
        grid=photometry.ReferenceGrid(shape, (0, 0), shape, parameters=parameters),
        old_donors=["ac19", "ac28"], new_donors=["ac50"], new_light_runs=["ac50", "ac60"],
        reference_run="ac28", matrix=np.eye(3), common_old=np.ones(shape, bool), common_new=np.ones(shape, bool),
        parameters=dict(chart_offset_xy=[0., 0.], virtual_reference_name="ac28_virtual_old_donor_consensus",
                        maximum_asset_iterations=12, previous_weight=.5, next_weight=.5),
        convergence=dict(minimum_iteration=2, facies_minimum_pixels=200,
            raw_flat_fixedpoint_residual_median_abs_log_over_donor_sand_all_channels=.01,
            raw_flat_fixedpoint_residual_p95_abs_log_over_donor_sand_all_channels=.05,
            raw_consensus_fixedpoint_residual_median_abs_log_over_donor_sand_all_channels=.005,
            donor_corrected_baseline_change_median_abs_log_per_run_and_facies=.01,
            donor_corrected_baseline_change_p95_abs_log_per_run_and_facies=.05),
        residual_parameters=dict(colour_sigma=3, division_floor=1e-6, minimum_centring_pixels=100,
            residual_colour_limit=.08, fraction_tolerance=1e-6, minimum_support_pixels=1000,
            sigma_reference_cells=8.0, minimum_smooth_support=.05),
        photometry_parameters=parameters,
    )


def test_saved_residual_matches_original_bits(inputs, originals):
    original = originals["ffac_v5_photometry"]
    path = "src/darsia/fluidflower/tools/ffac_v5_local_old_residual.py"
    source = subprocess.check_output(["git", "show", "f0d338aa:" + path], cwd=Path(__file__).resolve().parents[2])
    scope = dict(__name__=original.__package__ + ".residual", __package__=original.__package__)
    exec(compile(source, path, "exec"), scope)
    result = installation.fit_installation(**inputs)
    fit, reference = result["fits"]["ac19"], result["reference"]
    corrected = photometry.apply_grid(inputs["baseline_rgb"]["ac19"], fit, reference, inputs["grid"],
                                     parameters=inputs["photometry_parameters"])
    actual, details = residual.estimate_old_B_field(fit, reference, result["consensus"], corrected,
        parameters=inputs["residual_parameters"], photometry_parameters=inputs["photometry_parameters"])
    expected, old_details = scope["estimate_old_B_field"](fit, reference, result["consensus"], corrected)
    assert actual.tobytes() == expected.tobytes() and actual.dtype == expected.dtype
    assert details == old_details
    assert residual.apply_old_field(corrected, actual).tobytes() == scope["apply_old_field"](corrected, expected).tobytes()


def test_equal_inputs_converge_and_non_donors_cannot_change_installation(inputs):
    result = installation.fit_installation(**inputs)
    assert len(result["iteration_history"]) == 2
    assert result["convergence_pass"]
    changed = deepcopy(inputs)
    changed["baseline_rgb"]["ac60"] += np.float32(.3)
    changed["chart_log_rgb"]["ac60"] += np.float32(.3)
    other = installation.fit_installation(**changed)
    for name in ("consensus", "flat"):
        assert result[name].tobytes() == other[name].tobytes()
    for run in ("ac19", "ac28"):
        assert result["old_fields"][run].tobytes() == other["old_fields"][run].tobytes()
    assert result["iteration_history"] == other["iteration_history"]
    assert not np.array_equal(result["fits"]["ac60"].logB, other["fits"]["ac60"].logB)


def test_cancellation_and_nonconvergence_do_not_return_partial_assets(inputs):
    class Cancelled(Exception):
        pass
    def cancel():
        raise Cancelled()
    with pytest.raises(Cancelled):
        installation.fit_installation(**inputs, check_cancelled=cancel)
    inputs["parameters"]["maximum_asset_iterations"] = 1
    with pytest.raises(ValueError, match="did not converge"):
        installation.fit_installation(**inputs)


def test_missing_original_donor_is_not_silently_removed(inputs):
    del inputs["baseline_rgb"]["ac19"]
    with pytest.raises(ValueError, match="Every original donor"):
        installation.fit_installation(**inputs)
