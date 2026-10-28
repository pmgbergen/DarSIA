from pathlib import Path
from darsia.fluidflower import image_normalisation
from darsia.fluidflower.native_finish_setup import finish_setup_status
from darsia.fluidflower.mass.n1_pipeline import asset, _write_json
from darsia.fluidflower.pipeline.setup_image_preparation import FORMAT, IMAGE_RUN, OPERATION
import toml


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    return _write_json(path, value)


def test_finish_setup_needs_colour_paths_not_only_geometry_and_ignores_model_choice(tmp_path):
    series = tmp_path / "series.toml"
    document = dict(schema_version=1, series_id="sample", runs=dict(ids=["r1", "r2"]),
        method=dict(active="phase_model"), paths=dict(geometry_facies="geometry"),
        image_normalisation=image_normalisation.choice_record("sample", ["r1", "r2"], "off", None, color_method="off"))
    series.write_text(toml.dumps(document))
    adopted = record(tmp_path / "geometry/overrides/geometry_facies/current.json", dict(revision_id="saved"))
    operations = {run: record(tmp_path / (run + ".json"), dict(run=run)) for run in ("r1", "r2")}
    jobroot = tmp_path / "jobs/current"
    record(jobroot / "job.json", dict(pipeline="series-setup"))
    snapshot = dict(complete=True, errors=[], runs=[dict(run=run, operation="build", state="Complete",
        result=dict(files={"operation_result.json": pin})) for run, pin in operations.items()])
    registrations = [dict(root=str(jobroot), active=True)]
    result = finish_setup_status(tmp_path, registrations, snapshot_reader=lambda row: snapshot)
    assert not result["complete"]
    from darsia.fluidflower.pipeline.setup_colour_paths import SIGNAL_AXIS_CONTRACT
    from darsia.fluidflower.mass.pooled_optical_paths import SETUP_METHOD
    axis = dict(contract=SIGNAL_AXIS_CONTRACT, scales_by_facies={"1":1.},
        source_cohort=list(operations), frozen_for_production=True,
        units="corrected-clearwater-relative-rgb-euclidean-distance",
        scale_rule="maximum-supported-path-length-per-facies")
    packages = {run:dict(metadata=record(tmp_path/(run+"-metadata.json"),dict(
        recipe=dict(method=SETUP_METHOD,fit_runs=[run],runs={run:{}}),source_bindings={run:{}},
        signal_axis=axis,path_audit={"1":dict(status="supported")})),
        flash=record(tmp_path/(run+"-flash.json"),{}),
        color_paths={"1":record(tmp_path/(run+"-path.json"),dict(
            color_path=dict(colors=[[0.,0.,0.],[1.,0.,0.]]),values=[0.,1.]))},
        signals={"1":record(tmp_path/(run+"-signal.json"),{})}) for run in operations}
    colours = record(tmp_path / "colours.json", dict(format="darsia-setup-colour-paths-v1", complete=True,
        operations=operations, choice_authority=asset(series), runs=list(operations), image_count_per_run=15,
        package=packages["r1"], phase_inference_performed=False, model_training_performed=False, mass_fit_performed=False))
    basis = record(tmp_path / "basis.json", dict(format=FORMAT, complete=True,
        choice=document["image_normalisation"], choice_authority=asset(series), operations=operations,
        adopted_revision=adopted, colorpaths_performed=True, colour_paths=colours))
    snapshot["runs"].append(dict(run=IMAGE_RUN, operation=OPERATION, state="Complete", result=dict(image_preparation=basis)))
    assert not finish_setup_status(tmp_path, registrations, snapshot_reader=lambda row: snapshot)["complete"]
    from darsia.fluidflower.pipeline.setup_image_preparation import completed_setup_colour_paths
    assert completed_setup_colour_paths(snapshot, asset(series), require_per_run=False)["complete"]
    current = record(tmp_path / "per-run-colours.json", dict(format="darsia-setup-colour-paths-v2", complete=True,
        scope="per-run",signal_axis=axis,packages_by_run=packages,
        operations=operations, choice_authority=asset(series), runs=list(operations), image_count_per_run=15,
        phase_inference_performed=False, model_training_performed=False, mass_fit_performed=False))
    value = dict(format=FORMAT, complete=True,
        choice=document["image_normalisation"], choice_authority=asset(series), operations=operations,
        adopted_revision=adopted, colorpaths_performed=True, colour_paths=current)
    snapshot["runs"][-1]["result"]["image_preparation"] = record(tmp_path / "per-run-basis.json", value)
    result = finish_setup_status(tmp_path, registrations, snapshot_reader=lambda row: snapshot)
    assert result["complete"] and "all 2 runs" in result["message"]
    document["method"]["phase_model"] = dict(model_version="newly-selected-model")
    series.write_text(toml.dumps(document))
    assert finish_setup_status(tmp_path, registrations, snapshot_reader=lambda row: snapshot)["complete"]
    document["image_selection"] = dict(color_paths=dict(max_images_per_run=16))
    series.write_text(toml.dumps(document))
    assert not finish_setup_status(tmp_path, registrations, snapshot_reader=lambda row: snapshot)["complete"]


def test_no_registered_setup_is_not_complete_and_does_not_scan_images(tmp_path, monkeypatch):
    (tmp_path / "series.toml").write_text('schema_version=1\nseries_id="sample"\n[runs]\nids=[]\n')
    monkeypatch.setattr(Path, "rglob", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("No recursive image scan")))
    result = finish_setup_status(tmp_path, [])
    assert not result["complete"] and result["count"] == 15
