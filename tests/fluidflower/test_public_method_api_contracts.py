"""Public joint-mass and queue metadata contracts on tiny local fixtures."""
from datetime import datetime
import hashlib

import numpy as np
import pytest

from darsia.fluidflower.mass import concentration_joint as joint
from darsia.fluidflower.mass.phase_mass_joint_optics import (
    JointReference, METHODS, srgb_to_linear,
)
from darsia.fluidflower.pipeline import job_queue


def _summaries():
    signal = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    av = np.full(signal.shape, 0.001, dtype=np.float32)
    gv = np.full(signal.shape, 0.0005, dtype=np.float32)
    solubility = np.full(signal.shape, 2.0, dtype=np.float32)
    density = np.full(signal.shape, 1.0, dtype=np.float32)
    return [joint.summarize_frame(
        signal, av, gv, solubility, density, run="run", elapsed_h=time,
        injected_g=dose, bins=16, max_signal=1.0,
        expected_final_injected_mass_g=2.0,
    ) for time, dose in ((0.0, 0.0), (1.0, 2.0), (2.0, 2.0))]


def test_joint_summary_public_chain_preserves_timeline_and_mass_bounds():
    summaries = _summaries()
    anchor = joint.anchor_from_summaries(summaries)
    assert anchor["frame_index"] == 1
    assert anchor["elapsed_h"] == 1.0
    assert anchor["affected_pore_volume_m3"] == summaries[1]["affected_pore_volume_m3"]
    endpoint = joint.endpoint_from_summaries(summaries, [0, 1])
    assert 0 < endpoint <= 1.0
    parameters = joint.JointConcentrationParameters(
        endpoint=endpoint, amplitude=0.8, gamma=1.2,
        gas_saturation=0.5, beta=0.3,
    )
    direct = [joint.evaluate_joint_summary(row, parameters, anchor) for row in summaries]
    predictions = joint.predict_joint_summaries(
        {"run": summaries}, {"run": parameters}, {"run": anchor},
    )
    assert predictions["run"] == direct
    assert [row["elapsed_h"] for row in direct] == [0.0, 1.0, 2.0]
    for row in direct:
        assert row["total_mass_g"] == row["aqueous_mass_g"] + row["gas_mass_g"]
        assert row["numerical_lower_g"] <= row["total_mass_g"] <= row["numerical_upper_g"]
        assert row["summary_mass_error_bound_g"] >= 0
    with pytest.raises(ValueError, match="strictly ordered"):
        joint.anchor_from_summaries(summaries[::-1])
    with pytest.raises(ValueError, match="training observations"):
        joint.endpoint_from_summaries(summaries, [])


def test_joint_metrics_exclude_zero_dose_and_detect_nonzero_mass_error():
    exact = {"run": [
        {"nominal_injected_mass_g": 0.0, "total_mass_g": 100.0},
        {"nominal_injected_mass_g": 1.0, "total_mass_g": 1.0},
        {"nominal_injected_mass_g": 2.0, "total_mass_g": 2.0},
    ]}
    metrics = joint.joint_mass_metrics(exact)
    assert metrics["observations"] == 2
    assert metrics["rmse_g"] == metrics["nmae"] == metrics["nrmse"] == 0.0
    exact["run"][2]["total_mass_g"] = 3.0
    assert joint.joint_mass_metrics(exact)["rmse_g"] > 0
    assert joint.joint_mass_metrics(exact, {"run": [0]})["observations"] == 0


def test_linear_light_endpoints_and_zero_optical_change_with_fixed_pore_volumes():
    decoded = srgb_to_linear(np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float32))
    assert decoded.dtype == np.float32
    assert decoded[0] == 0.0 and decoded[-1] == 1.0
    assert np.all(np.diff(decoded) > 0)
    rgb = np.full((2, 2, 3), 0.5, dtype=np.float32)
    baseline_log = np.log(srgb_to_linear(rgb))
    reference = JointReference(
        baseline_log=baseline_log,
        denominators={method: np.ones((2, 2), dtype=np.float32) for method in METHODS},
        reliability={}, direction=np.array([0.5, 0.5, -1.0], dtype=np.float32),
        diagnostics={}, sigma=0.0, linear_floor=0.001, usable=True,
    )
    av = np.full((2, 2), 0.001, dtype=np.float32)
    gv = np.zeros((2, 2), dtype=np.float32)
    solubility = np.full((2, 2), 2.0, dtype=np.float32)
    density = np.ones((2, 2), dtype=np.float32)
    parameters = joint.JointConcentrationParameters(
        endpoint=1.0, amplitude=1.0, gamma=1.0, gas_saturation=0.5, beta=0.0,
    )
    arrays, record = joint.evaluate_joint_fields(
        rgb, reference, av, gv, solubility, density, parameters, elapsed_h=0.0,
        anchor={"elapsed_h": 0.0, "affected_pore_volume_m3": float(av.sum())},
    )
    assert np.array_equal(arrays["joint_signal"], np.zeros((2, 2), dtype=np.float32))
    assert record["total_mass_g"] == record["aqueous_mass_g"] == record["gas_mass_g"] == 0.0
    assert np.array_equal(av, np.full((2, 2), 0.001, dtype=np.float32))


def test_queue_public_metadata_helpers_bind_bytes_without_scientific_io(tmp_path):
    path = tmp_path / "metadata.json"
    raw = b'{"purpose":"local metadata"}'
    path.write_bytes(raw)
    assert job_queue.read_json(path) == {"purpose": "local metadata"}
    desc = job_queue.descriptor(path)
    assert desc["sha256"] == hashlib.sha256(raw).hexdigest()
    assert desc["size_bytes"] == len(raw)
    assert datetime.fromisoformat(job_queue.utc()).utcoffset().total_seconds() == 0
    assert job_queue.name("Run_1") == "run_1"
    assert job_queue.require_unc(r"\\server\share\configuration.json") is None
    task = {
        "run": "Run_1", "commit": "a" * 40, "calibration_sha256": "b" * 64,
        "configuration": {"path": r"\\server\share\configuration.json", "sha256": "c" * 64},
        "backend": "cpu", "precision": "float32",
    }
    job_queue.validate_task(task)
    with pytest.raises(ValueError, match="Reserved filesystem"):
        job_queue.name("CON")
    with pytest.raises(ValueError):
        job_queue.require_unc("C:/configuration.json")
    with pytest.raises(ValueError, match="backend"):
        job_queue.validate_task(dict(task, backend="unknown"))


@pytest.mark.parametrize("module", ["aggregate", "enqueue", "worker"])
def test_queue_cli_requires_an_explicit_job_before_io(module):
    import importlib
    cli = importlib.import_module("darsia.fluidflower.pipeline." + module)
    with pytest.raises(SystemExit) as error:
        cli.main([])
    assert error.value.code == 2
