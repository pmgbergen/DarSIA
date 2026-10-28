"""Production clocks use metadata only and preserve the calibration choice."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

from darsia.fluidflower.production_image_grid import (
    propose_production_grid, save_production_grid, resolve_production_selection,
    production_grid_targets, validate_production_phase_inputs,
)
from darsia.fluidflower.series_image_selection import RunImageMetadata, _descriptor, save_run_image_selection
from darsia.fluidflower.series_method import ManifestChangedError


def metadata(root, run, times, source=()):
    frames = tuple(dict(source_path=str(root / run / f"image_{index}.png"),
        frame_index=index, image_id=f"image_{index}.png", elapsed_h=time)
        for index, time in enumerate(times))
    return RunImageMetadata(run, frames, 0., 1., tuple(source), (), str(root))


@pytest.fixture
def grid_data(tmp_path):
    root = tmp_path / "series"
    root.mkdir()
    (root / "series.toml").write_text('schema_version=1\nseries_id="generic"\n[runs]\nids=["long","short"]\n')
    source = root / "protocol.csv"
    source.write_text("metadata-only")
    sources = [_descriptor(source)]
    seed = propose_production_grid({"long": metadata(root, "long", [-.1, 100.])})
    times = [-.1, *seed["main_target_hours"], *seed["late_target_hours"]]
    values = {"long": metadata(root, "long", times, sources),
              "short": metadata(root, "short", times[:42], sources)}
    return root, values


def test_shared_clock_has_80_main_10_late_plus_baseline_and_does_not_stretch_short_runs(grid_data, monkeypatch):
    root, values = grid_data
    before = deepcopy(values)
    def no_images(path, *args, **kwargs):
        if str(path).endswith(".png"):
            pytest.fail("Production selection must not open photographs")
    monkeypatch.setattr(Path, "open", no_images)
    result = propose_production_grid(values)
    assert len(result["main_target_hours"]) == 80
    assert result["main_target_hours"][-1] == 48.
    assert len(result["late_target_hours"]) == 10
    assert result["runs"]["long"]["counts"] == dict(baselines=1, main=80, late=10, selected=91, missing=0)
    short = result["runs"]["short"]
    assert short["counts"]["selected"] == 42 and short["counts"]["late"] == 0
    assert [row["target_hour"] for row in short["target_matches"]] == [
        row["target_hour"] for row in result["runs"]["long"]["target_matches"]]
    assert all(row["reason"] == "after_end_of_run" for row in short["target_matches"][41:])
    assert values == before


def test_separate_production_save_and_resolution_preserve_calibration_rows_and_configs(grid_data):
    root, values = grid_data
    current = (root / "series.toml").read_bytes()
    save_run_image_selection(root, values["long"],
        [values["long"].frames[index]["source_path"] for index in (0, 1, 2)], expected_bytes=current)
    old = tomllib.loads((root / "series.toml").read_text())
    raw = (root / "series.toml").read_bytes()
    result = propose_production_grid(values)
    save_production_grid(root, result, values, expected_bytes=raw)
    new = tomllib.loads((root / "series.toml").read_text())
    assert new["image_selection"] == old["image_selection"]
    assert new["data_time"] == old["data_time"]
    row = resolve_production_selection(root, "long", values["long"])
    assert row["purpose"] == "production" and row["selected_frame_count"] == 91
    assert len(new["image_selection"]["runs"]["long"]["selected"]) == 3
    assert row["production_manifest"]["sha256"] == new["production_image_selection"]["selection_manifest"]["sha256"]
    # The saved grid's baseline and target matches are read beside the rows.
    assert "baseline_path" not in row and "target_matches" not in row
    targets = production_grid_targets(root, row)
    assert targets["baseline_path"] == result["runs"]["long"]["baseline_path"]
    assert row["selected"][0]["source_path"] == targets["baseline_path"]
    assert targets["target_matches"] == result["runs"]["long"]["target_matches"]
    copy = Path(row["production_manifest"]["path"]).with_name("grid-copy.json")
    copy.write_bytes(Path(row["production_manifest"]["path"]).read_bytes())
    moved = dict(row, production_manifest=dict(row["production_manifest"], path=str(copy)))
    assert production_grid_targets(root, moved) == targets
    copy.write_text("{}", encoding="utf-8")
    with pytest.raises(ManifestChangedError, match="production grid changed"):
        production_grid_targets(root, moved)
    (root / "protocol.csv").write_text("changed")
    with pytest.raises(ManifestChangedError, match="metadata changed"):
        resolve_production_selection(root, "long", values["long"])


def test_phase_input_preparation_consumes_saved_production_clock_without_changing_sources(grid_data):
    root, values = grid_data
    result = propose_production_grid(values)
    save_production_grid(root, result, values, expected_bytes=(root / "series.toml").read_bytes())
    selected = resolve_production_selection(root, "long", values["long"])["selected"]
    config = dict(frames=[dict(elapsed_h=row["elapsed_h"]) for row in selected[1:]])
    inputs = dict(runs=[dict(run="long", configuration_bound={"path": "config"})])
    reader = SimpleNamespace(json=lambda pin: config if pin["path"] == "config" else inputs)
    assert validate_production_phase_inputs(root, {"path": "inputs"}, reader=reader,
        metadata_reader=lambda root, run: values[run]) is True
    config["frames"].pop()
    with pytest.raises(ValueError, match="Prepare phase maps for the saved production grid"):
        validate_production_phase_inputs(root, {"path": "inputs"}, reader=reader,
            metadata_reader=lambda root, run: values[run])


def test_changed_or_edited_grid_cannot_be_saved_without_a_new_proposal(grid_data):
    root, values = grid_data
    proposal = propose_production_grid(values)
    original = (root / "series.toml").read_bytes()
    proposal["main_target_hours"][0] = .5
    with pytest.raises(ManifestChangedError, match="proposal changed"):
        save_production_grid(root, proposal, values, expected_bytes=original)
    assert (root / "series.toml").read_bytes() == original


def test_ordinary_production_creator_resolves_the_production_purpose(grid_data, monkeypatch):
    from darsia.fluidflower import production_selection, production_resources, staged_jobs
    root, values = grid_data
    result = propose_production_grid(values)
    save_production_grid(root, result, values, expected_bytes=(root / "series.toml").read_bytes())
    monkeypatch.setattr(staged_jobs, "plan_pipeline", lambda pin: "n1")
    monkeypatch.setattr(production_selection, "validate_production_choice", lambda *a, **kw: None)
    import darsia.fluidflower.series_image_selection as selection
    monkeypatch.setattr(selection, "load_run_protocol_metadata", lambda root, run: values[run])
    captured = []
    def creator(*args, **kwargs):
        captured.extend(kwargs["selection_provider"](["long", "short"]))
        return dict(worker_started=False)
    monkeypatch.setattr(production_resources, "create_profiled_analysis_job", creator)
    plan = dict(path="synthetic-plan", sha256="a" * 64, size_bytes=1, production_calibration={})
    production_selection.create_production_job(plan, "synthetic-output", series_root=root, series_id="generic")
    assert [row["purpose"] for row in captured] == ["production", "production"]
    assert captured[0]["selected_frame_count"] == 91 and captured[1]["selected_frame_count"] == 42


def test_camera_tolerance_never_spends_an_extra_late_image_for_the_48_hour_target(tmp_path):
    value = metadata(tmp_path, "generic", [-.1, .25, 47.99, 48.001, 100.])
    result = propose_production_grid({"generic": value}, main_count=2, late_count=1)
    times = [row["elapsed_h"] for row in result["runs"]["generic"]["selected"]]
    assert times == [.25, 47.99, 100.]
    assert sum(time > 48 for time in times) == 1
