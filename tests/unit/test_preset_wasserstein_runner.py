"""Ownership and orchestration checks; these stubs do not establish science parity."""

from contextlib import contextmanager
from copy import deepcopy
from importlib import import_module
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from darsia.fluidflower import preset_wasserstein_runner as runner
from darsia.presets.workflows.comparison.comparison_wasserstein import (
    WassersteinDistanceResult,
)


def _image(dtype="float32"):
    return SimpleNamespace(img=SimpleNamespace(dtype=dtype, shape=(2, 3)))


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    """Replace worker admission and calculation, never call a scientific solver."""
    protocol = import_module("darsia.fluidflower.preset_wasserstein_protocol")
    inputs = import_module("darsia.fluidflower.preset_wasserstein_inputs")
    results = import_module("darsia.fluidflower.preset_wasserstein_results")
    run_io = import_module("darsia.fluidflower.pipeline.run_io")
    original = import_module(
        "darsia.presets.workflows.comparison.comparison_wasserstein"
    )
    output = tmp_path / "attempt"
    selected = dict(order=2, time_hours=1.5, tolerance_hours=0.25)
    task = dict(pipeline="preset-wasserstein", backend="cpu", precision="original",
                run="comparison-time-2", job_id="comparison", comparison_time=selected,
                format="darsia-alias-task-v1", attempt_id="a" * 32)
    run_configs = {name: object() for name in ("z", "a", "m")}
    config = SimpleNamespace(
        runs=SimpleNamespace(config=run_configs),
        wasserstein=SimpleNamespace(
            runs=["z", "a", "m"], times=[(1.5, 0.25)],
            roi={"lower": SimpleNamespace(name="lower"),
                 "upper": SimpleNamespace(name="upper")},
        ),
    )
    rig = SimpleNamespace(
        baseline=_image(), permeability=_image("float64"),
        geometry=SimpleNamespace(weight=_image(), depth=_image()),
    )
    image, experiment = _image(), object()
    consumption = {"format": "test-consumption", "verified": True}
    state = {"entered": False, "closed": False, "verified": False}

    def verify_consumed():
        state["verified"] = True
        return deepcopy(consumption)

    runtime = SimpleNamespace(
        config=config, reference_rig=rig, reference_experiment=experiment,
        load_mass=lambda *args, **kwargs: image, verify_consumed=verify_consumed,
    )

    @contextmanager
    def load_runtime(actual_task):
        assert actual_task is task
        state["entered"] = True
        try:
            yield runtime
        finally:
            state["closed"] = True

    @contextmanager
    def limits():
        yield {"input_dtypes": [], "requested_threads": 1}

    def require_output(path):
        assert Path(path) == output

    def receipt(result, *, task, output_root):
        assert state["closed"] and state["verified"]
        assert output_root == output
        return result

    monkeypatch.setattr(protocol, "validate_preset_wasserstein_task", lambda value: None)
    monkeypatch.setattr(run_io, "for_task", lambda value: SimpleNamespace(
        require_output=require_output
    ))
    monkeypatch.setattr(inputs, "load_preset_wasserstein_runtime", load_runtime)
    monkeypatch.setattr(results, "write_preset_wasserstein_receipt", receipt)
    monkeypatch.setattr(runner, "_thread_limits", limits)
    return SimpleNamespace(
        task=task, output=output, runtime=runtime, original=original,
        image=image, state=state, consumption=consumption,
    )


def _record(declaration):
    """A real original result record with NaNs, without performing a solve."""
    return WassersteinDistanceResult(
        run_1=declaration["run_1"], run_2=declaration["run_2"],
        time=declaration["time_hours"], time_1=None, time_2=None,
        roi_name=declaration["roi_name"], roi_exact_mass=float("nan"),
        roi_detected_mass_1=float("nan"), roi_detected_mass_2=float("nan"),
        distance=float("nan"), normalized_distance={"plume_based": float("nan"),
            "rig_based": float("nan"), "mass_based": float("nan")},
        computation_time=0.0, timestamp="2026-01-01T00:00:00", status="success",
    )


def _stub_compute(admitted, monkeypatch, *, after_result=None, original_missing=False):
    def compute(cls, config, skip_existing, **kwargs):
        assert config is admitted.runtime.config
        assert config.wasserstein.runs == ["z", "a", "m"]
        assert skip_existing is False
        assert kwargs["reference_rig"] is admitted.runtime.reference_rig
        assert kwargs["reference_experiment"] is admitted.runtime.reference_experiment
        assert kwargs["results_dir"] == admitted.output
        assert kwargs["vtk_root"] == admitted.output / "vtk"
        loaded = kwargs["mass_reader"](
            config.runs.config["a"], data="mass", time=1.5, tol=0.25
        )
        assert loaded is (None if original_missing else admitted.image)
        for declaration in runner._expected_comparisons(
            config, admitted.output, admitted.task["comparison_time"]
        ):
            record = _record(declaration)
            if original_missing:
                record.status = "missing"
            path = record.save_to_dir(admitted.output)
            kwargs["on_result"](record, path)
            if after_result is not None:
                after_result(path)

    monkeypatch.setattr(admitted.original, "_compute_wasserstein_distances", compute)


def test_runner_keeps_original_order_nan_records_and_reports_actual_outputs(
    admitted, monkeypatch
):
    _stub_compute(admitted, monkeypatch)
    events = []

    result = runner.run_preset_wasserstein(
        admitted.task, admitted.output, "cpu", "original", events.append
    )

    assert [(row["run_1"], row["run_2"], row["roi_name"])
            for row in result["comparisons"]] == [
        ("a", "z", "lower"), ("a", "z", "upper"),
        ("a", "m", "lower"), ("a", "m", "upper"),
        ("m", "z", "lower"), ("m", "z", "upper"),
    ]
    assert result["expected_count"] == result["comparison_count"] == 6
    assert result["output_count"] == 6
    assert result["vtk"] == dict(files=[], actual_count=0, state="no_files_written")
    assert result["source_consumption"] == admitted.consumption
    assert result["precision"] == "original"
    assert result["validation"]["state"] == "not_scored"
    assert {entry["dtype"] for entry in result["runtime"]["input_dtypes"]} == {
        "float32", "float64"
    }
    for comparison in result["comparisons"]:
        original = json.loads(Path(comparison["file"]["path"]).read_text())
        assert original["status"] == "success"
        assert original["distance"] != original["distance"]  # The original NaN remains.
    assert [event["completed_count"] for event in events] == list(range(1, 7))
    assert all(event["state"] == "preset_wasserstein_progress" for event in events)
    assert admitted.state == {"entered": True, "closed": True, "verified": True}


def test_result_callback_boundary_propagates_after_original_file_write(
    admitted, monkeypatch
):
    _stub_compute(admitted, monkeypatch)

    class PauseBoundary(Exception):
        pass

    def pause(event):
        assert Path(event["comparison"]["file"]["path"]).is_file()
        raise PauseBoundary

    with pytest.raises(PauseBoundary):
        runner.run_preset_wasserstein(
            admitted.task, admitted.output, "cpu", "original", pause
        )

    assert len(list(admitted.output.glob("*.json"))) == 1
    assert admitted.state == {"entered": True, "closed": True, "verified": False}


def test_actual_queue_progress_is_retained_but_excluded_from_original_scientific_inventory(admitted, monkeypatch):
    """Use the existing queue writer in private folders, without a queue/service."""
    from darsia.fluidflower.pipeline.job_queue import JobQueue

    _stub_compute(admitted, monkeypatch)
    lease = admitted.output.parent / "lease"
    lease.mkdir()
    claim = SimpleNamespace(task=admitted.task, token=admitted.task["attempt_id"],
        output=admitted.output, lease=lease, machine="fixture-cpu")
    queue = object.__new__(JobQueue)
    queue._progress_lock = threading.RLock()
    queue.assert_owner = lambda actual: None
    queue._server = lambda machine: SimpleNamespace(now_ns=lambda: 123456789)
    result = runner.run_preset_wasserstein(admitted.task, admitted.output, "cpu", "original",
        lambda event: queue.progress(claim, event))
    assert len(result["queue_progress"]) == 6 and result["output_count"] == 6
    assert not any(key.startswith("progress_") for key in result["files"])
    assert len(list(admitted.output.glob("progress_*.json"))) == 6
    assert len(list(lease.glob("progress_*.json"))) == 6


@pytest.mark.parametrize("filename", ["undeclared.json", "progress_00000000.json"])
def test_runner_rejects_undeclared_files_and_unbound_queue_progress(admitted, monkeypatch, filename):
    _stub_compute(admitted, monkeypatch,
        after_result=lambda path: (admitted.output / filename).write_text("{}"))
    with pytest.raises((ValueError, KeyError)):
        runner.run_preset_wasserstein(admitted.task, admitted.output, "cpu", "original")


@pytest.mark.parametrize("missing", ["reference_rig", "reference_experiment", "config"])
def test_runner_refuses_missing_bound_dependency(admitted, missing):
    setattr(admitted.runtime, missing, None)

    with pytest.raises(ValueError, match="bound original dependencies"):
        runner.run_preset_wasserstein(admitted.task, admitted.output, "cpu", "original")

    assert admitted.state["closed"] is True


def test_runner_preserves_declared_original_missing_mass(admitted, monkeypatch):
    _stub_compute(admitted, monkeypatch, original_missing=True)
    admitted.runtime.load_mass = lambda *args, **kwargs: None

    result = runner.run_preset_wasserstein(admitted.task, admitted.output, "cpu", "original")

    assert result["comparison_count"] == 6
    assert all(row["status"] == "missing" for row in result["comparisons"])
    assert not any(row["consumer"].startswith("mass:")
                   for row in result["runtime"]["input_dtypes"])


def test_runner_refuses_a_nonfresh_attempt_before_loading_inputs(admitted):
    admitted.output.mkdir()
    (admitted.output / "existing.json").write_text("{}")

    with pytest.raises(FileExistsError, match="fresh"):
        runner.run_preset_wasserstein(admitted.task, admitted.output, "cpu", "original")

    assert admitted.state["entered"] is False


def test_runner_refuses_output_changes_after_original_completion_event(admitted, monkeypatch):
    _stub_compute(admitted, monkeypatch, after_result=lambda path: path.write_text("{}"))

    with pytest.raises(ValueError, match="changed after its completion event"):
        runner.run_preset_wasserstein(admitted.task, admitted.output, "cpu", "original")


def test_runner_refuses_original_filename_collisions(admitted, monkeypatch):
    admitted.runtime.config.wasserstein.roi = {
        "one": SimpleNamespace(name="lower area"),
        "two": SimpleNamespace(name="lower_area"),
    }

    with pytest.raises(ValueError, match="filenames collide"):
        runner.run_preset_wasserstein(admitted.task, admitted.output, "cpu", "original")

    assert not list(admitted.output.iterdir())


def test_runner_refuses_a_runtime_with_another_time_selection(admitted):
    admitted.runtime.config.wasserstein.times = [(2.5, 0.25)]

    with pytest.raises(ValueError, match="exact selected time"):
        runner.run_preset_wasserstein(admitted.task, admitted.output, "cpu", "original")

    assert not list(admitted.output.iterdir())
