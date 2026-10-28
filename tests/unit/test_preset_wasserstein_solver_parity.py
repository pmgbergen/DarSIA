"""UNRUN original-solver parity candidate for tiny private fixture inputs.

Both routes call the actual original mass processing, reference calculations,
Bregman solver and optional VTK writer. Only the reused job fixture's Git and
filesystem admission are synthetic. No native analysis-domain proof, shared
transport, process lifecycle or scientific acceptance is claimed by this test.
"""
from dataclasses import fields
import json
import math
from pathlib import Path
import shutil

import darsia
import numpy as np
import pytest

from darsia.fluidflower.pipeline.job_queue import read_json
from darsia.fluidflower.preset_wasserstein_runner import (
    _thread_limits, run_preset_wasserstein,
)
from darsia.presets.workflows.comparison.comparison_wasserstein import (
    WassersteinDistanceResult, _compute_wasserstein_distances, _load_and_process_mass,
)
from darsia.presets.workflows.config.multi_fluidflower_config import MultiFluidFlowerConfig
from darsia.presets.workflows.rig import Rig
from .test_preset_wasserstein_jobs import original_job_source, _binding, _prepare


def _write_solver_inputs(case):
    """Keep real protocol/config readers and provide a nondegenerate small grid."""
    height, width = 8, 12
    y, x = np.indices((height, width))
    dimensions = [1.0, 1.5]
    rig_folder = case.fixture["rig_folder"]
    baseline = darsia.OpticalImage(
        np.full((height, width, 3), 0.5, dtype=np.float32),
        dimensions=dimensions, color_space="RGB",
    )
    baseline.save(rig_folder / "baseline.npz")
    scalar_fields = dict(
        depth=(0.02 + 0.005 * y / height).astype(np.float32),
        labels=np.ones((height, width), dtype=np.int32),
        facies=np.ones((height, width), dtype=np.int32),
        porosity=(0.35 + 0.05 * x / width).astype(np.float32),
        # Unit-test numbers, not measured permeability: the unchanged original
        # routine uses np.isclose(..., 0), so these must not all be treated as zero.
        permeability=(1.0 + x / width).astype(np.float32),
        image_porosity=np.ones((height, width), dtype=np.float32),
    )
    for name, array in scalar_fields.items():
        darsia.ScalarImage(array, dimensions=dimensions).save(rig_folder / (name + ".npz"))

    # Only the first selected run (z) supplies reference geometry. Give a a
    # distinct saved rig so accidentally selecting the lexical first run changes
    # the real mass weighting and cannot hide behind equal reference fixtures.
    second_rig = case.root / "rig-a"
    shutil.copytree(rig_folder, second_rig)
    darsia.ScalarImage(np.full((height, width), 0.8, dtype=np.float32),
        dimensions=dimensions).save(second_rig / "porosity.npz")
    run_config = case.root / "a.toml"
    run_config.write_text(run_config.read_text(encoding="utf-8").replace(
        'path = "rig"', 'path = "rig-a"'), encoding="utf-8")

    # Positive, different distributions survive area reduction and the cropped
    # ROI. Original preprocessing performs all weighting and normalization.
    distributions = {
        "a": (1.0 + 0.5 * (x % 4 < 2) + 0.25 * (y < 4)).astype(np.float32),
        "z": (1.0 + 0.5 * (x % 4 >= 2) + 0.25 * (y >= 4)).astype(np.float32),
    }
    for run, array in distributions.items():
        darsia.ScalarImage(array, dimensions=dimensions, time=3600.0).save(
            case.fixture["mass_paths"][run])

    config_path = case.fixture["config_path"]
    text = config_path.read_text(encoding="utf-8")
    text = text.replace('[wasserstein]\n',
        '[[roi]]\nname = "middle"\ncorner_1 = [0.25, 0.75]\n'
        'corner_2 = [1.25, 0.25]\n[wasserstein]\n')
    text = text.replace('roi = ["whole"]', 'roi = ["whole", "middle"]')
    config_path.write_text(text, encoding="utf-8")
    return second_rig


def _assert_scientific_values(actual, expected):
    """Tight same-environment tolerance; exact types, structure and nonfinite kind."""
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key in expected:
            _assert_scientific_values(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_scientific_values(left, right)
    elif isinstance(expected, float):
        if math.isnan(expected):
            assert math.isnan(actual)
        elif math.isinf(expected):
            assert actual == expected
        else:
            assert math.isfinite(actual)
            assert actual == pytest.approx(expected, rel=1e-12, abs=1e-14)
    else:
        assert actual == expected


def test_actual_original_computation_matches_bound_runner(original_job_source, tmp_path):
    """UNRUN: actual solver twice, with identical original settings and one thread."""
    case = original_job_source
    second_rig = _write_solver_inputs(case)
    _prepare(case)
    task = read_json(next((case.job_root / "queue").glob("*.json")))
    config = MultiFluidFlowerConfig(case.fixture["config_path"],
        require_data=False, require_results=True)
    assert config.wasserstein.runs == ["z", "a"]
    assert config.wasserstein.times == [(1.0, 0.1)]
    assert config.wasserstein.resize_factor == 0.5
    assert config.wasserstein.relative_tol is None  # Keep the original solver options.

    original_output = tmp_path / "original-computation"
    original_output.mkdir()
    original_paths = []
    with _thread_limits():
        # Confirm the genuine original preprocessing sees a coarser grid and a
        # nonempty cropped domain, rather than testing raw fixture arrays only.
        reference = Rig.load(config.runs.config["z"].rig.path)
        coarse = _load_and_process_mass("a", config, 1.0, 0.1, reference.geometry.weight)
        assert coarse.img.shape == (4, 6)
        assert coarse.subregion(config.wasserstein.roi["middle"].roi).img.shape == (2, 4)
        # Reference rig, protocol and mass reader are all the original defaults.
        _compute_wasserstein_distances(Rig, config, False,
            results_dir=original_output, vtk_root=original_output / "vtk",
            on_result=lambda result, path: original_paths.append(Path(path)))

    binding = _binding(case, task, tmp_path)
    frozen = binding.configuration["preset_inputs"]
    assert frozen["reference_run"] == "z"
    assert not any(Path(row["descriptor"]["path"]).is_relative_to(second_rig)
                   for row in frozen["files"])
    events = []
    with binding.activate():
        actual = run_preset_wasserstein(task, binding.output, "cpu", "original", events.append)

    identity = [(row["run_1"], row["run_2"], row["roi_name"]) for row in actual["comparisons"]]
    assert identity == [("a", "z", " whole "), ("a", "z", "middle")]
    assert [Path(row["file"]["path"]).name for row in actual["comparisons"]] == [
        path.name for path in original_paths]
    assert actual["comparison_count"] == actual["expected_count"] == 2
    assert [event["completed_count"] for event in events] == [1, 2]
    assert actual["validation"]["state"] == "not_scored"
    schema = {field.name for field in fields(WassersteinDistanceResult)}
    for path, record in zip(original_paths, actual["comparisons"]):
        expected = json.loads(path.read_text(encoding="utf-8"))
        observed = json.loads(Path(record["file"]["path"]).read_text(encoding="utf-8"))
        assert set(observed) == set(expected) == schema
        # Positive finite distance proves these fixtures reached the genuine
        # solver branch, rather than an original zero/missing/tolerance shortcut.
        assert expected["status"] == observed["status"] == "success"
        assert math.isfinite(expected["distance"]) and expected["distance"] > 0
        for row in (expected, observed):
            assert isinstance(row["timestamp"], str) and row["timestamp"]
            assert math.isfinite(row["computation_time"]) and row["computation_time"] >= 0
            row.pop("timestamp")
            row.pop("computation_time")
        _assert_scientific_values(observed, expected)

    observed_dtypes = actual["runtime"]["input_dtypes"]
    assert {row["consumer"] for row in observed_dtypes if row["consumer"].startswith("mass:")} == {"mass:z", "mass:a"}
    assert all(row["dtype"] == "float32" and row["shape"] == [8, 12]
               for row in observed_dtypes if row["consumer"].startswith("mass:"))
    assert next(row["dtype"] for row in observed_dtypes
                if row["consumer"] == "reference_rig.baseline") == "float32"
    assert next(row["dtype"] for row in observed_dtypes
                if row["consumer"] == "reference_rig.permeability") == "float32"
    for path in case.fixture["mass_paths"].values():
        assert darsia.imread(path).img.dtype == np.float32
    # The unchanged optional writer may produce no files when pyevtk is absent;
    # compare actual inventories without inventing a successful VTK artifact.
    assert actual["vtk"]["files"] == sorted(
        path.relative_to(original_output).as_posix()
        for path in (original_output / "vtk").rglob("*") if path.is_file())
