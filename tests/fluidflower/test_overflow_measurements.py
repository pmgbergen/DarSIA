"""Literal original feed/QA port and finite current-volume examples only."""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower.mass import overflow_measurements as measurements


def test_original_feed_and_qa_numerical_bodies_are_literal_ports():
    tree = ast.parse(Path(measurements.__file__).read_bytes().decode("utf8"))
    bodies = {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
              for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert bodies["feed"] == "39c1eebee25c84a057ca31c3a38316d141ac7d8d970084d39522ce98aa084171"
    assert bodies["qa"] == "7c1ab67a49cad0df8110bf28d8aa551daffb1cdfcd6f55bd639ba21e44e61e20"


def curve():
    # The observed overflow is 8,12,20 mL at the selected times. A/B/C
    # spread contributes 2 mL and the common late scatter contributes 1 mL.
    central = np.array([0, 8, 12, 20, 0], np.float64)
    return dict(t=np.array([0, 1, 2, 4, 10], np.float64), vA=central+2,
                vB=central-2, vC=central, vC_late_sd=1.)


def test_ratio_uses_current_selected_volumes_and_shared_measurement_error():
    frames = [dict(case_id="outside", elapsed_h=.5, baseline_gas_volume_m3=10e-6),
              dict(case_id="one", elapsed_h=1., baseline_gas_volume_m3=10e-6),
              dict(case_id="two", elapsed_h=2., baseline_gas_volume_m3=20e-6),
              dict(case_id="four", elapsed_h=4., baseline_gas_volume_m3=40e-6),
              dict(case_id="too-little", elapsed_h=2., baseline_gas_volume_m3=2e-6)]
    for row in frames:
        row["historical_model_volume_m3"] = 999  # Deliberately unrelated old geometry.
    result = measurements.measured_ratio(curve(), frames)
    assert result["ratio"] == pytest.approx(.6)  # median(.8,.6,.5)
    assert result["sigma"] == pytest.approx(.15)  # median(3/10,3/20,3/40), not /sqrt(3)
    assert result["selected_cases"] == ["one", "two", "four"]


def test_repeated_selected_frame_cannot_receive_extra_ratio_weight():
    row = dict(case_id="one", elapsed_h=1., baseline_gas_volume_m3=10e-6)
    with pytest.raises(ValueError, match="twice"):
        measurements.measured_ratio(curve(), [row, deepcopy(row)])


def test_missing_observation_interval_is_not_silently_extrapolated():
    observed = curve()
    observed["t"] = np.array([1.5, 2, 4, 8, 10], np.float64)
    with pytest.raises(ValueError, match="outside"):
        measurements.measured_ratio(observed, [dict(case_id="one", elapsed_h=1., baseline_gas_volume_m3=10e-6)])


def review():
    return dict(metadata=dict(weight_data_valid=True), records=[
        dict(time_from_injection_h=float(time), weight_g=float(1000+100*time),
             status="Verified", image_time=f"2023-02-17T00:{index:02}:00")
        for index, time in enumerate(range(-2, 30))])


def test_original_manual_correction_is_local_and_review_document_is_unchanged():
    document = review()
    corrected = document["records"][7]
    corrected.update(image_time="2023-02-17T17:45:29", weight_g=1500.)
    before = deepcopy(document)
    result = measurements.reviewed_curve(document, run="ac33")
    assert result["usable"] and result["reasons"] == []
    assert result["curve"]["w"][7] == 1584.7
    assert document == before
    assert result["curve"]["t0_anchor"] == -1


@pytest.mark.parametrize("fault", ["invalid", "short", "no-pre"])
def test_original_unusable_measurements_are_explicit_group_candidates(fault):
    document = review()
    if fault == "invalid":
        document["metadata"]["weight_data_valid"] = False
    elif fault == "short":
        document["records"] = document["records"][:20]
    else:
        for row in document["records"]:
            row["time_from_injection_h"] += 3
    result = measurements.reviewed_curve(document, run="ac19")
    assert not result["usable"] and result["curve"] is None and result["reasons"]
