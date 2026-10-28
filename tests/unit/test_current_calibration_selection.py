"""The current manifest may change while RAW membership/history stays exact."""
from pathlib import Path
from datetime import datetime, timedelta
import toml

from darsia.fluidflower.mass.n1_pipeline import _write_json, asset
from darsia.fluidflower.pipeline.current_calibration_selection import current_calibration_selection


def test_rebind_current_manifest_preserves_early_history_and_original_fit_membership(tmp_path, monkeypatch):
    def write(name, value):
        return _write_json(tmp_path / name, value)
    paths = [str(tmp_path / (name + ".ARW")) for name in ("reference", "early", "candidate")]
    times = [-1., .125, .5]
    origin = datetime(2024, 2, 7, 12)
    (tmp_path / "injection.csv").write_text(
        "ID,location_x,location_y,start,end,rate_g/s\n1,0.2,0.1,2024-02-07 12:00:00,2024-02-07 14:00:00,0.01\n")
    (tmp_path / "common.toml").write_text(toml.dumps(dict(protocols=dict(
        injection="injection.csv", pressure_temperature_mode="constant",
        pressure_bar=1.02, temperature_celsius=23.))))
    (tmp_path / "run.toml").write_text("")
    sources = []
    for index, (path, elapsed) in enumerate(zip(paths, times)):
        sources.append(dict(source_sha256=str(index + 1) * 64,
            previous_source_sha256=str(index) * 64 if index else None,
            v5_raw_path=path, v5_raw_sha256=str(index + 4) * 64,
            elapsed_h=elapsed, selected=bool(index), baseline=index == 0))
    inventory = write("inventory.json", dict(sources=sources))
    original = write("original.json", dict(frame_sources=[dict(raw_path=path, elapsed_h=elapsed)
                                                         for path, elapsed in zip(paths[1:], times[1:])]))
    selection = write("selection.json", dict(frames=[dict(
        chosen_image={"source_path": path}, data_time_h=elapsed,
        camera_time=(origin+timedelta(hours=elapsed)).isoformat())
        for path, elapsed in zip(paths, times)]))
    plan = write("plan.json", dict(runs={"sample": {"source_inventory": inventory}}))
    fit = write("fit.json", dict(calibration_runs=["sample"], selected_phase_gate=dict(
        phase_plan=plan, run_manifests={"sample": original}), frame_decisions=[dict(
        run="sample", source_binding=dict(raw_path=paths[2], raw_sha256="6" * 64),
        elapsed_h=.5, stage="stage1", selected_target_h=.5, included=True)]))
    calibration = write("calibration.json", dict(calibration_files={"global": fit}))
    manifest = dict(method=dict(active="phase_model", phase_model=dict(mass_calibration=calibration["path"])),
        paths=dict(common_config="common.toml"), runs=dict(ids=["sample"], config_files=dict(sample="run.toml")), image_selection=dict(runs={"sample": dict(
            selection_manifest=selection, selected=[dict(source_path=path, elapsed_h=elapsed,
                frame_index=index, image_id=str(index)) for index, (path, elapsed) in enumerate(zip(paths, times))])}))
    series_path = tmp_path / "series.toml"
    series_path.write_text(toml.dumps(manifest), encoding="utf-8")
    before = asset(series_path)
    manifest["geometry_review"] = {"revision": "saved-chart-map"}
    series_path.write_text(toml.dumps(manifest), encoding="utf-8")
    record = current_calibration_selection(tmp_path, actor="reviewer", reason="Use the retained calibration selection")
    assert record["input_authority"]["current_series"] == asset(series_path)
    assert record["input_authority"]["current_series"]["sha256"] != before["sha256"]
    assert record["input_authority"]["global_fit"] == fit
    assert record["counts"] == dict(unique_acquisitions=3, selected_frames=2, baselines=1,
                                    fit_candidates=1, earlier_history=1)
    frames = record["runs"][0]["selected_frames"]
    assert [row["elapsed_h"] for row in frames] == [.125, .5]
    assert [row["previous_raw_sha256"] for row in frames] == ["4" * 64, "5" * 64]
    assert [row["fit_candidate"] for row in frames] == [False, True]
    assert all(not Path(path).exists() for path in paths)
    # A recorded directory rename changes the locator, not RAW identity,
    # chronological predecessors, elapsed time or fit membership.
    from darsia.fluidflower import storage_paths
    relocated = [str(tmp_path / "moved" / Path(path).name) for path in paths]
    mapping = dict(zip(paths, relocated))
    monkeypatch.setattr(storage_paths, "storage_path_remapper",
                        lambda: lambda value: mapping.get(str(value), str(value)))
    selection = write("moved-selection.json", dict(frames=[dict(
        chosen_image={"source_path": path}, data_time_h=elapsed,
        camera_time=(origin+timedelta(hours=elapsed)).isoformat())
        for path, elapsed in zip(relocated, times)]))
    manifest["image_selection"]["runs"]["sample"] = dict(selection_manifest=selection,
        selected=[dict(source_path=path, elapsed_h=elapsed, frame_index=index, image_id=str(index))
                  for index, (path, elapsed) in enumerate(zip(relocated, times))])
    series_path.write_text(toml.dumps(manifest), encoding="utf-8")
    moved = current_calibration_selection(tmp_path, actor="reviewer", reason="Use explicit path relocation")
    assert moved["counts"] == record["counts"]
    assert [row["raw_path"] for row in moved["runs"][0]["selected_frames"]] == relocated[1:]
    assert [row["raw_sha256"] for row in moved["runs"][0]["selected_frames"]] == ["5"*64, "6"*64]
    assert [row["previous_raw_sha256"] for row in moved["runs"][0]["selected_frames"]] == ["4"*64, "5"*64]
    # New weights invalidate the active old calibration, but its explicit
    # cohort/frame choices remain usable for preparing a new fit.
    from darsia.fluidflower.series_method import MethodSelection, save_series_method, read_series_method
    save_series_method(tmp_path, MethodSelection(active="phase_model", model_version="models/new"),
                       prepared_model=True)
    assert read_series_method(tmp_path).mass_calibration == ""
    retrained = current_calibration_selection(tmp_path, actor="reviewer", reason="Use retained image choices for the new model")
    assert retrained["input_authority"]["global_fit"] == fit
    assert retrained["counts"] == moved["counts"]
    assert retrained["runs"][0]["selected_frames"] == moved["runs"][0]["selected_frames"]

    # Non-fit runs still need the same corrected-image/colour-path pipeline.
    # Their observations must remain outside the mass objective.
    mass_fit = ["sample", *["fit-" + str(index) for index in range(25)]]
    controls = ["control-" + str(index) for index in range(13)]
    included = [*mass_fit, *controls]
    plan = write("all-plan.json", dict(runs={run: {"source_inventory": inventory}
                                           for run in included}))
    fit = write("all-fit.json", dict(calibration_runs=mass_fit, selected_phase_gate=dict(
        phase_plan=plan, run_manifests={run: original for run in included}),
        frame_decisions=[dict(run=run, source_binding=dict(raw_path=paths[2], raw_sha256="6" * 64),
            elapsed_h=.5, stage="stage1", selected_target_h=.5, included=True) for run in mass_fit]))
    all_calibration = write("all-calibration.json", dict(calibration_files={"global": fit}, heldout_runs=controls))
    manifest = toml.loads(series_path.read_text(encoding="utf-8"))
    manifest["method"]["phase_model"]["previous_mass_calibrations"] = [all_calibration["path"]]
    manifest["runs"]["ids"] = included
    for run in included[1:]:
        manifest["runs"]["config_files"][run] = "run.toml"
        manifest["image_selection"]["runs"][run] = manifest["image_selection"]["runs"]["sample"]
    series_path.write_text(toml.dumps(manifest), encoding="utf-8")
    all_runs = current_calibration_selection(tmp_path, actor="reviewer", reason="Prepare every included run")
    assert len(all_runs["runs"]) == 39 and [row["run"] for row in all_runs["runs"]] == included
    assert all_runs["mass_fit_runs"] == mass_fit
    assert all_runs["counts"]["fit_candidates"] == 26
    assert all(row["calibration_role"] == "heldout" for row in all_runs["runs"][26:])
    assert all(not frame["fit_candidate"] for row in all_runs["runs"][26:] for frame in row["selected_frames"])

    # The saved nominal camera targets establish statistics eligibility for
    # every run, independently of whether a later mass fit uses that run.
    from darsia.fluidflower.shared_calibration_cohort import save_mass_cohort
    for row in manifest["image_selection"]["runs"].values():
        row.update(target_hours=times, requested_target_hours=times, tolerance_seconds=5.)
    series_path.write_text(toml.dumps(manifest), encoding="utf-8")
    eligible = current_calibration_selection(tmp_path, actor="reviewer", reason="Keep reusable eligible statistics")
    assert eligible["counts"]["fit_candidates"] == 2 * len(included)
    assert all(frame["eligibility_source"] == "saved_camera_target_match"
               for row in eligible["runs"][26:] for frame in row["selected_frames"])
    save_mass_cohort(tmp_path, ["sample", controls[0]], expected_bytes=series_path.read_bytes())
    changed_cohort = current_calibration_selection(tmp_path, actor="reviewer", reason="Use the explicit mass cohort")
    assert changed_cohort["mass_fit_runs"] == ["sample", controls[0]]
    assert changed_cohort["counts"] == eligible["counts"]
    from darsia.fluidflower.pipeline.current_calibration_selection import same_preparation_selection
    assert same_preparation_selection(eligible, changed_cohort)
    modified = dict(changed_cohort, optical_selection={"max_images_per_run": 9})
    assert not same_preparation_selection(eligible, modified)
