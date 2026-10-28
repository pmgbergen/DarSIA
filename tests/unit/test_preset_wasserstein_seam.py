"""Original-reader and completion-hook behavior, without a numerical solve."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def missing_comparison(tmp_path, monkeypatch):
    original = import_module(
        "darsia.presets.workflows.comparison.comparison_wasserstein"
    )
    roi = SimpleNamespace(name="whole", roi=object())
    config = SimpleNamespace(
        runs=SimpleNamespace(config={
            name: SimpleNamespace(rig=SimpleNamespace(path=Path("rig-" + name)))
            for name in ("z", "a")
        }),
        wasserstein=SimpleNamespace(
            runs=["z", "a"], times=[(1.5, 0.25)], results=tmp_path,
            roi={"whole": roi}, resize_factor=0.25,
        ),
    )
    rig = SimpleNamespace(
        baseline=object(), permeability=SimpleNamespace(img=np.array([1.0])),
        geometry=SimpleNamespace(
            weight=object(), depth=SimpleNamespace(img=np.array([1.0])),
            dimensions=(2.0, 1.0),
        ),
    )
    state = SimpleNamespace(pressure=1.0, temperature=1.0)
    experiment = SimpleNamespace(
        experiment_start=object(),
        injection_protocol=SimpleNamespace(injected_mass=lambda **kwargs: 1.0),
        pressure_temperature_protocol=SimpleNamespace(
            get_state=lambda start: state, get_gradient=lambda start: state
        ),
    )
    monkeypatch.setattr(original.darsia, "full_like", lambda *args: object())
    monkeypatch.setattr(original.darsia, "CO2MassAnalysis", lambda **kwargs:
                        SimpleNamespace(solubility_co2=np.array([1.0])))
    loads = []

    def missing(**kwargs):
        loads.append(kwargs)
        return None

    monkeypatch.setattr(original, "_load_and_process_mass", missing)
    return SimpleNamespace(
        original=original, config=config, rig=rig, experiment=experiment,
        loads=loads, output=tmp_path,
    )


def test_local_defaults_still_use_first_selected_reference_run(
    missing_comparison, monkeypatch
):
    case = missing_comparison
    rig_paths, experiment_configs = [], []

    class Rig:
        @classmethod
        def load(cls, path):
            rig_paths.append(path)
            return case.rig

    def experiment(config):
        experiment_configs.append(config)
        return case.experiment

    monkeypatch.setattr(case.original.darsia.ProtocolledExperiment,
                        "init_from_config", experiment)

    case.original._compute_wasserstein_distances(Rig, case.config, False)

    assert rig_paths == [Path("rig-z")]
    assert experiment_configs == [case.config.runs.config["z"]]
    assert [call["run_name"] for call in case.loads] == ["z", "a", "z"]
    assert all(call["mass_reader"] is None for call in case.loads)


def test_injected_dependencies_reach_every_original_mass_call_and_missing_callback(
    missing_comparison, monkeypatch
):
    case = missing_comparison
    destination = case.output / "owned"
    destination.mkdir()

    def unexpected(*args, **kwargs):
        pytest.fail("Injected original dependencies must not use a fallback reader")

    monkeypatch.setattr(case.original.darsia.ProtocolledExperiment,
                        "init_from_config", unexpected)
    rig_type = SimpleNamespace(load=unexpected)
    reader = object()
    events = []

    def completed(result, path):
        assert path.is_file()
        events.append((result, path))

    case.original._compute_wasserstein_distances(
        rig_type, case.config, False, reference_rig=case.rig,
        reference_experiment=case.experiment, mass_reader=reader,
        results_dir=destination, vtk_root=destination / "vtk", on_result=completed,
    )

    assert [call["run_name"] for call in case.loads] == ["z", "a", "z"]
    assert all(call["mass_reader"] is reader for call in case.loads)
    assert len(events) == 1
    result, path = events[0]
    assert (result.run_1, result.run_2, result.status) == ("a", "z", "missing")
    assert path.parent == destination
    assert np.isnan(result.distance)


def test_original_skip_existing_still_reports_its_existing_success_record(
    missing_comparison
):
    case = missing_comparison
    record = case.original.WassersteinDistanceResult(
        run_1="a", run_2="z", time=1.5, time_1=None, time_2=None,
        roi_name="whole", roi_exact_mass=0.0, roi_detected_mass_1=0.0,
        roi_detected_mass_2=0.0, distance=float("nan"),
        normalized_distance={"mass_based": float("nan")},
        computation_time=0.0, timestamp="2026-01-01T00:00:00", status="success",
    )
    path = record.save_to_dir(case.output)
    before = path.read_bytes()
    reported = []

    case.original._compute_wasserstein_distances(
        None, case.config, True, reference_rig=case.rig,
        reference_experiment=case.experiment,
        on_result=lambda result, actual: reported.append((result, actual)),
    )

    assert path.read_bytes() == before
    assert len(reported) == 1
    assert reported[0][0].status == "success"
    assert reported[0][1] == path


def test_original_result_callback_exception_is_not_caught(missing_comparison):
    case = missing_comparison

    class Boundary(Exception):
        pass

    def stop(result, path):
        assert path.is_file()
        raise Boundary

    with pytest.raises(Boundary):
        case.original._compute_wasserstein_distances(
            None, case.config, False, reference_rig=case.rig,
            reference_experiment=case.experiment, on_result=stop,
        )


def test_mass_reader_injection_keeps_the_original_missing_sentinel(monkeypatch):
    original = import_module(
        "darsia.presets.workflows.comparison.comparison_wasserstein"
    )
    config = SimpleNamespace(runs=SimpleNamespace(config={"a": object()}))

    def unexpected(*args, **kwargs):
        pytest.fail("The supplied reader must be used")

    monkeypatch.setattr(original, "load_data", unexpected)

    assert original._load_and_process_mass(
        "a", config, 1.5, 0.25, object(), mass_reader=lambda *args, **kwargs: None
    ) is None
