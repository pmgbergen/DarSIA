"""Before/after gas accounting uses saved split products, without another fit."""
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.mass.gas_contribution_review import (
    capture_gas_contributions, contribution_scales, contribution_summary,
    read_gas_contributions, save_gas_contributions,
)
from darsia.fluidflower.mass.n1_pipeline import DataReader


def frame():
    return SimpleNamespace(
        population=SimpleNamespace(eligible=np.ones((2, 2), bool),
            pore_volume_m3=np.array([[1., 1.], [2., 4.]])*1e-6),
        prepared={"frame": SimpleNamespace(dimensions_m=(1., 2.), frame_index=7, elapsed_h=2.5)},
        statistics_descriptor={"path": "original.json.gz", "sha256": "original"},
        probability_roles={"archive": {"path": "original-probabilities.npz"}},
        phase_source={"run": "run", "case_id": "image"})


def test_native_addition_and_removal_are_not_cancelled_by_coarse_pooling(tmp_path):
    source = frame()
    before = np.array([[3., 0.], [0., 1.]], np.float32)
    after = np.array([[1., 2.], [1., 1.]], np.float32)
    old_bytes, new_bytes = before.tobytes(), after.tobytes()
    capture = capture_gas_contributions(source, before, after, max_axis=1)
    assert capture["grid"]["shape"] == [1, 1]
    assert capture["arrays"]["cell_model_retained_g"].item() == pytest.approx(.005)
    assert capture["arrays"]["cell_overflow_added_g"].item() == pytest.approx(.004)
    assert capture["arrays"]["cell_overflow_removed_g"].item() == pytest.approx(.002)
    assert before.tobytes() == old_bytes and after.tobytes() == new_bytes
    result = dict(report=dict(baseline_gas_mass_g=.007, gas_mass_g=.009),
                  statistics={}, gas_contribution_maps=capture)
    record = save_gas_contributions(tmp_path, source, result,
        dict(kind="own_measurement", estimated=False))
    rows = [dict(run="run", case_id="image", gas_contribution=record)]
    document = dict(frames=rows, gas_contribution_review=contribution_scales(rows))
    loaded = read_gas_contributions(document, "run", "image", reader=DataReader(unc_only=False))
    assert loaded["status"] == "available"
    assert loaded["summary"]["frame_index"] == 7 and loaded["summary"]["elapsed_h"] == 2.5
    assert loaded["maps"]["cell_model_before_g"].sum() == pytest.approx(.007)
    assert loaded["maps"]["cell_final_gas_g"].sum() == pytest.approx(.009)
    assert loaded["maps"]["cell_overflow_change_g"].sum() == pytest.approx(.002)
    assert loaded["summary"]["overflow_added_g"] == pytest.approx(.004)
    assert loaded["summary"]["overflow_removed_g"] == pytest.approx(.002)
    assert loaded["summary"]["reference_label"] == "Measured overflow"
    assert loaded["display_scale_g_per_cell"]["positive"] == pytest.approx(.009)
    assert loaded["grid"]["display_only"]


@pytest.mark.parametrize("before,after,kind,estimated", [
    (2., 3., "own_measurement", False),
    (3., 2., "own_measurement", False),
    (2., 0., "own_measurement", False),
    (0., 0., "model_phase_split", False),
    (2., 3., "pressure_matched_estimate", True),
])
def test_signed_mass_change_and_zero_denominator_are_explicit(before, after, kind, estimated):
    summary = contribution_summary(dict(baseline_gas_mass_g=before, gas_mass_g=after),
        dict(kind=kind, estimated=estimated))
    assert summary["model_gas_mass_g"] == before
    assert summary["final_gas_mass_g"] == after
    assert summary["overflow_change_g"] == after-before
    assert summary["overflow_change_share_of_final"] == (None if after == 0 else (after-before)/after)
    assert summary["estimated"] is estimated
    if estimated:
        assert "ESTIMATED" in summary["reference_label"]


def test_old_report_keeps_actual_numbers_but_never_invents_spatial_data():
    document = dict(frames=[dict(run="old", case_id="photo", report=dict(
        baseline_gas_mass_g=4., gas_mass_g=3., gas_change_g=-1.),
        phase_split_target=dict(kind="own_measurement", estimated=False))])
    result = read_gas_contributions(document, "old", "photo")
    assert result["status"] == "spatial_unavailable" and result["maps"] is None
    assert result["summary"]["overflow_change_g"] == -1.
    assert result["summary"]["reference_kind"] == "own_measurement"
    assert read_gas_contributions(document, "other", "photo")["status"] == "unavailable"


def test_no_reference_preserves_model_only_mass_and_zero_correction():
    summary = contribution_summary(dict(baseline_gas_mass_g=5., gas_mass_g=5.), model_only=True)
    assert summary["overflow_change_g"] == 0.
    assert summary["reference_kind"] == "model_phase_split"
    assert summary["estimated"] is False
