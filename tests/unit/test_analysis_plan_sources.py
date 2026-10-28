"""A staged plan's inputs are found where the series keeps them, never outside it."""
import json
from pathlib import Path

import pytest

from darsia.fluidflower import analysis_plan_sources as sources
from darsia.fluidflower import analysis_preparation

RUNS = ("r1", "r2")


def _series(root, paths=""):
    root.mkdir(parents=True)
    (root / "series.toml").write_text(
        'schema_version = 1\nseries_id = "demo"\n'
        '[roots]\nseries = "."\nresults = "../results"\n'
        '[paths]\nresults = "../results"\n' + paths
        + '[runs]\nids = ["r1", "r2"]\n', encoding="utf-8")
    return root


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return dict(path=str(path))


def _complete(series, *, models=None, method=None, frame_rows=True):
    """Save every phase-one input; ``models``/``method`` may name another folder."""
    results = series.parent / "results"
    inputs = sources.analysis_inputs_folder(series)
    method = _json(Path(method or results) / "method.json", {})
    for run in RUNS:
        _json(inputs / "runs" / f"{run}.json", dict(run=run, mass=dict(calibration=method)))
    entries = {}
    for run in RUNS:
        entries[run] = dict(reference_model=dict(path=str(Path(models or results) / f"{run}-model.json")))
        if frame_rows:
            entries[run]["frame_data"] = dict(path=str(results / f"{run}-frames.json"))
    _json(inputs / "geometry_index.json", dict(
        format="darsia-measured-run-geometry-v1", reference_models_complete=True, runs=entries))
    _json(inputs / "pressure_index.json", dict(format="darsia-pressure-dataset-index-v1", runs=dict(demo={
        run: dict(atmosphere_series=dict(path=str(results / f"{run}-pressure.json"))) for run in RUNS})))
    return inputs


@pytest.fixture
def series(tmp_path):
    return _series(tmp_path / "series")


def test_missing_inputs_name_the_place_the_series_keeps_them(series):
    found = sources.phase_one_sources(series)
    assert [source.label for source in found] == ["Run documents", "Measured frame geometry", "Running pressure"]
    assert all(source.missing and source.bound is None for source in found)
    assert found[0].missing == "Not saved yet for 2 of 2 runs: r1, r2."
    folder = series.parent / "results" / "analysis" / "inputs"
    assert Path(found[0].expected.replace("<run>.json", "r1.json")).resolve() == (folder / "runs" / "r1.json").resolve()
    assert str(Path(found[1].expected.split(" (")[0]).resolve()) == str((folder / "geometry_index.json").resolve())
    text = sources.sources_text(found)
    assert text.count("Missing:") == 3 and "Expected:" in text and "Found:" not in text


def test_complete_inputs_are_bound_in_run_order(series):
    inputs = _complete(series)
    runs, geometry, pressure = sources.phase_one_sources(series)
    assert not (runs.missing or geometry.missing or pressure.missing)
    assert [Path(bound["path"]).name for bound in runs.bound] == ["r1.json", "r2.json"]
    assert all(set(bound) == {"path", "sha256", "size_bytes"} for bound in runs.bound)
    assert Path(geometry.bound["path"]) == (inputs / "geometry_index.json").resolve()
    assert Path(pressure.bound["path"]) == (inputs / "pressure_index.json").resolve()
    assert sources.sources_text([runs, geometry, pressure]).count("Found:") == 3


def test_files_outside_the_series_folders_count_as_missing(series, tmp_path):
    _complete(series, models=tmp_path / "local", method=tmp_path / "archive")
    runs, geometry, _pressure = sources.phase_one_sources(series)
    assert runs.missing.startswith("Outside the series folders (or binding a method file there) for 2 of 2")
    assert geometry.missing == "Model or frame rows outside the series folders for 2 of 2 runs: r1, r2."
    assert runs.bound is None and geometry.bound is None


def test_pending_frame_rows_and_runs_without_pressure_are_named(series):
    inputs = _complete(series, frame_rows=False)
    pressure = json.loads((inputs / "pressure_index.json").read_text())
    del pressure["runs"]["demo"]["r2"]
    _json(inputs / "pressure_index.json", pressure)
    _runs, geometry, pressure = sources.phase_one_sources(series)
    assert geometry.missing == "Frame rows not saved yet for 2 of 2 runs: r1, r2."
    assert pressure.missing == "No pressure series for 1 of 2 runs: r2."


def test_an_index_of_another_kind_is_not_used(series):
    inputs = _complete(series)
    _json(inputs / "geometry_index.json", dict(format="darsia-series-display-frame-models-v1"))
    geometry = sources.phase_one_sources(series)[1]
    assert geometry.missing.startswith("Not a measured run geometry file")


def test_a_configured_inputs_folder_is_used(tmp_path):
    series = _series(tmp_path / "series", paths='analysis_inputs = "inputs"\nanalysis_plans = "plans"\n')
    assert sources.analysis_inputs_folder(series) == series / "inputs"
    assert sources.analysis_plans_folder(series) == series / "plans"
    _complete(series)
    assert not any(source.missing for source in sources.phase_one_sources(series))


def test_phase_two_needs_the_completions_and_a_chosen_calibration(series, tmp_path):
    inputs = sources.analysis_inputs_folder(series)
    completions, fitted = sources.phase_two_sources(series)
    assert completions.missing == "Not saved yet."
    assert fitted.missing == "Choose a completed mass calibration job on this page."
    assert "concentration_mass" in fitted.expected
    _json(inputs / "phase_two_inputs.json", dict(format="darsia-analysis-phase2-inputs-v1", runs=[dict(run="r1")]))
    inside = dict(name="Morning", calibration=dict(path=str(series.parent / "results" / "frozen.json")))
    completions, fitted = sources.phase_two_sources(series, inside)
    assert completions.found.startswith("1 runs in") and fitted.bound == inside["calibration"]
    outside = dict(name="Old", calibration=dict(path=str(tmp_path / "archive" / "frozen.json")))
    assert sources.phase_two_sources(series, outside)[1].missing.startswith("Outside the series folders")


def test_prepare_refuses_missing_inputs_and_writes_nothing(series, monkeypatch):
    monkeypatch.setattr(analysis_preparation, "job_identity", lambda _repository: pytest.fail("No job identity"))
    folder, manifest = sources.fresh_plan_paths(series, "phase-maps")
    with pytest.raises(ValueError, match="Analysis inputs are missing; no plan was prepared"):
        sources.prepare_phase_one_plan(series, sources.phase_one_sources(series), folder=folder,
                                       manifest=manifest, gas_probability=False, repository=".", job_id="phase-maps")
    assert not folder.parent.exists()


def test_prepare_uses_the_original_collection_and_preparation(series, monkeypatch):
    _complete(series)
    found = sources.phase_one_sources(series)
    calls = []
    monkeypatch.setattr(analysis_preparation, "job_identity", lambda repository: dict(commit="c", repository=repository))
    monkeypatch.setattr(analysis_preparation, "collect_phase1_source_inputs",
                        lambda *a, **k: calls.append(("collect", a, k)) or dict(inputs=dict(path="manifest")))
    monkeypatch.setattr(analysis_preparation, "prepare_phase1_inputs",
                        lambda *a, **k: calls.append(("prepare", a, k)) or dict(plan=dict(path="plan")))
    folder, manifest = sources.fresh_plan_paths(series, "phase-maps")
    assert manifest.parent == folder.parent and manifest.name == folder.name + "-source-inputs.json"
    result = sources.prepare_phase_one_plan(series, found, folder=folder, manifest=manifest,
                                            gas_probability=True, repository="repo", job_id="phase-maps")
    assert result == dict(plan=dict(path="plan")) and folder.parent.is_dir() and not folder.exists()
    (_, collect_args, collect_kwargs), (_, prepare_args, prepare_kwargs) = calls
    assert collect_args == (found[0].bound, manifest)
    assert collect_kwargs == dict(series_root=series, geometry_index=found[1].bound,
                                  pressure_index=found[2].bound, gas_probability=True)
    assert prepare_args == (dict(path="manifest"), folder)
    assert prepare_kwargs["job"] == dict(commit="c", repository="repo") and prepare_kwargs["job_id"] == "phase-maps"
