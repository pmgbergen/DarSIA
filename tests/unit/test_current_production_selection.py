"""Production photographs enter the current-build selection after the Build baseline.

The method ancestry, protocols and RAW identities are real small files. The
saved production grid and the imaging-protocol rows are supplied directly; the
grid's own saving and resolving are tested with the production grid.
"""
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
import toml

from darsia.fluidflower import production_image_grid, series_image_selection
from darsia.fluidflower.mass.n1_pipeline import _write_json, asset
from darsia.fluidflower.pipeline import current_production_selection as subject

ORIGIN = datetime(2024, 2, 7, 12)


@pytest.fixture
def series(tmp_path, monkeypatch):
    """One run whose Build baseline precedes the grid's own baseline."""
    def write(name, value):
        return _write_json(tmp_path / name, value)
    (tmp_path / "injection.csv").write_text(
        "ID,location_x,location_y,start,end,rate_g/s\n1,0.2,0.1,2024-02-07 12:00:00,2024-02-07 14:00:00,0.01\n")
    (tmp_path / "common.toml").write_text(toml.dumps(dict(protocols=dict(
        injection="injection.csv", pressure_temperature_mode="constant",
        pressure_bar=1.02, temperature_celsius=23.))))
    (tmp_path / "run.toml").write_text("")
    plan = write("plan.json", dict(runs={}))
    fit = write("fit.json", dict(calibration_runs=["sample"], selected_phase_gate=dict(phase_plan=plan)))
    calibration = write("calibration.json", dict(calibration_files={"global": fit}, excluded_runs=["outlier"]))
    (tmp_path / "series.toml").write_text(toml.dumps(dict(
        method=dict(active="phase_model", phase_model=dict(mass_calibration=calibration["path"])),
        paths=dict(common_config="common.toml"),
        runs=dict(ids=["sample", "outlier"], config_files=dict(sample="run.toml", outlier="run.toml")),
        image_selection=dict(color_paths=dict(max_images_per_run=15)))), encoding="utf-8")
    names, times = ["build", "grid", "early", "late", "unused"], [-.5, -.1, .25, 1., 2.]
    photos = []
    for index, (name, time) in enumerate(zip(names, times)):
        path = tmp_path / "raw" / (name + ".ARW")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(("synthetic RAW " + name).encode())
        photos.append(dict(source_path=str(path), image_id=path.name, frame_index=index, elapsed_h=time,
                           blacklisted=False, camera_time=(ORIGIN + timedelta(hours=time)).isoformat()))
    metadata = SimpleNamespace(run_id="sample", frames=tuple(photos))
    grid = dict(selected=photos[1:4], baseline_path=photos[1]["source_path"], production_manifest=dict(
        path=str(tmp_path / "grid.json"), sha256="c" * 64, size_bytes=1),
        target_matches=[dict(target_hour=.25, source_path=photos[2]["source_path"], selected=True),
                        dict(target_hour=1., source_path=photos[3]["source_path"], selected=True),
                        dict(target_hour=50., selected=False, reason="after_end_of_run")])
    monkeypatch.setattr(series_image_selection, "load_run_protocol_metadata", lambda root, run: metadata)
    monkeypatch.setattr(production_image_grid, "resolve_production_selection",
                        lambda root, run, value: {key: grid[key] for key in ("selected", "production_manifest")})
    monkeypatch.setattr(production_image_grid, "production_grid_targets", lambda root, resolved: {
        key: grid[key] for key in ("baseline_path", "target_matches")})
    baselines = {"sample": asset(photos[0]["source_path"])}
    return SimpleNamespace(root=tmp_path, photos=photos, grid=grid, baselines=baselines, fit=fit, plan=plan)


def test_build_baseline_starts_history_and_grid_targets_follow(series):
    record = subject.current_production_selection(series.root, series.baselines, actor="reviewer",
                                                  reason="Prepare production phase maps")
    assert record["format"] == "ffac-local-p1-selection-data-v1" and record["purpose"] == "production"
    assert record["mass_fit_runs"] == [] and record["input_authority"]["global_fit"] == series.fit
    assert record["input_authority"]["phase_plan"] == series.plan
    assert record["optical_selection"] == dict(max_images_per_run=15)
    (run,) = record["runs"]
    assert run["run"] == "sample"  # The calibration's excluded run is left out.
    baseline, frames = run["baseline"], run["selected_frames"]
    assert (baseline["raw_path"], baseline["raw_sha256"]) == (series.photos[0]["source_path"],
                                                              series.baselines["sample"]["sha256"])
    assert baseline["baseline"] is True and baseline["previous_raw_sha256"] is None
    assert run["initial_previous_raw_sha256"] == baseline["raw_sha256"]
    assert [row["raw_path"] for row in frames] == [series.photos[2]["source_path"], series.photos[3]["source_path"]]
    shas = [asset(row["raw_path"])["sha256"] for row in frames]
    assert [row["raw_sha256"] for row in frames] == shas
    assert [row["previous_raw_sha256"] for row in frames] == [baseline["raw_sha256"], shas[0]]
    assert [row["nominal_fit_target_h"] for row in frames] == [.25, 1.]
    assert [row["current_selection_position"] for row in frames] == [1, 2]
    assert not any(row["fit_candidate"] for row in frames)
    assert run["counts"] == dict(unique_acquisitions=3, selected_frames=2, baselines=1,
                                 fit_candidates=0, earlier_history=2)
    assert run["production_grid_baseline"]["raw_path"] == series.photos[1]["source_path"]
    assert run["current_selection_manifest"] == series.grid["production_manifest"]
    protocols = run["current_protocols"]
    assert protocols["format"] == "darsia-current-run-protocols-v1" and protocols["run"] == "sample"
    assert [row["pressure_bar"] for row in protocols["atmosphere"]["rows"]] == [1.02, 1.02]


def test_grid_baseline_equal_to_build_baseline_is_the_reference(series):
    series.grid["selected"] = [series.photos[0], *series.photos[2:4]]
    series.grid["baseline_path"] = series.photos[0]["source_path"]
    record = subject.current_production_selection(series.root, series.baselines, actor="reviewer",
                                                  reason="Prepare production phase maps")
    run = record["runs"][0]
    assert run["production_grid_baseline"] is None and len(run["selected_frames"]) == 2


@pytest.mark.parametrize("change,reason", [
    ("unknown_baseline", "baseline photograph is not one image"),
    ("early_target", "precedes the completed Build's baseline"),
    ("untargeted", "photograph without a target time"),
    ("no_baseline", "no baseline for sample"),
])
def test_unusable_grid_or_build_refuses_with_a_plain_reason(series, change, reason):
    baselines = dict(series.baselines)
    if change == "unknown_baseline":
        baselines["sample"] = dict(baselines["sample"], path=str(series.root / "elsewhere.ARW"))
    elif change == "early_target":
        baselines["sample"] = asset(series.photos[3]["source_path"])
    elif change == "untargeted":
        series.grid["selected"] = [*series.grid["selected"], series.photos[4]]
    else:
        baselines.pop("sample")
    with pytest.raises(ValueError, match=reason):
        subject.current_production_selection(series.root, baselines, actor="reviewer",
                                             reason="Prepare production phase maps")


def test_raw_identities_read_each_photograph_once_in_order(tmp_path):
    paths = []
    for index in range(5):
        path = tmp_path / f"{index}.ARW"
        path.write_bytes(bytes([index]) * (index + 1))
        paths.append(str(path))
    seen = []
    result = subject.raw_identities([*paths, paths[0]], threads=2, on_progress=seen.append)
    assert list(result) == paths and all(result[path] == asset(path) for path in paths)
    assert [event["completed"] for event in seen] == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="positive number"):
        subject.raw_identities(paths, threads=0)
