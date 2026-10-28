"""The final Setup operation owns real fits/baselines, not just saved choices."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import toml

from darsia.fluidflower.pipeline import setup_image_preparation as subject
from darsia.fluidflower.pipeline.current_image_preparation import prepare_image_photometry
from darsia.fluidflower.mass.n1_pipeline import asset, _write_json, DataReader
from darsia.fluidflower import image_normalisation
from .test_series_setup_jobs import setup_job, controls, _prepare  # noqa: F401


def test_image_caps_follow_declared_pixels_instead_of_large_old_raw_allowances():
    plan = dict(runs=["r1", "r2"], build_plans=[dict(request_template=dict(output=dict(
        maximum_output_bytes=64 * 1024**3, maximum_file_bytes=32 * 1024**3))) for _ in range(2)])
    native = dict(source=dict(native_shape_hw=[3200, 5000]))
    caps = subject.image_output_caps(plan, native)
    assert caps["maximum_file_bytes"] >= 3200 * 5000 * 3 * np.dtype(np.float32).itemsize
    assert caps["maximum_file_bytes"] < 1024**3
    assert caps["maximum_output_bytes"] < 2 * 1024**3
    larger = subject.image_output_caps(plan, dict(source=dict(native_shape_hw=[6400, 5000])))
    assert larger["maximum_output_bytes"] > caps["maximum_output_bytes"]
    with pytest.raises(ValueError, match="positive native image shape"):
        subject.image_output_caps(plan, dict(source=dict(native_shape_hw=[True, 5000])))


@pytest.mark.parametrize("reuse", [False, True])
def test_typed_image_task_uses_current_choice_and_retains_original_plan(setup_job, reuse):
    case = setup_job
    _prepare(case)
    old = json.loads((case.job_root / "job.json").read_bytes())
    path = Path(case.plan["series_manifest"]["path"])
    document = dict(schema_version=1, series_id="synthetic", method=dict(active="phase_model"),
        runs=dict(ids=case.plan["runs"]), roots=dict(series=".", private=str(case.root)))
    document["image_normalisation"] = image_normalisation.choice_record("synthetic", case.plan["runs"],
        "off", None, color_method="off")
    path.write_text(toml.dumps(document), encoding="utf-8")
    basis = _write_json(case.job_root / "prior-basis.json", {}) if reuse else None
    value = subject.configure_setup_image_preparation_job(old, case.bound_plan, case.plan,
        case.job_root, asset(path), reuse_image_basis=basis)
    assert value["job"]["setup_plan"] == old["setup_plan"]
    assert value["job"]["series"] == old["series"]
    assert value["configuration"]["series_manifest"] == asset(path)
    assert value["task"]["operation"] == "image-preparation"
    assert value["task"]["series"]["sha256"] == asset(path)["sha256"]
    assert value["job"]["expected_runs"] == [*old["expected_runs"], subject.IMAGE_RUN]
    assert value["configuration"]["colour_path_scope"] == "per-run"
    uses = {use for row in value["task"]["input_catalog"]["inputs"] for use in row["uses"]}
    assert ("setup.reused_image_basis" in uses) is reuse
    if reuse:
        assert value["configuration"]["reuse_image_basis"] == basis
    assert json.loads((case.job_root / "job.json").read_bytes()) == old


def test_actual_final_setup_producer_saves_fits_and_corrected_baselines_once(tmp_path, monkeypatch):
    # Use the actual parameter DATA rather than another numerical algorithm.
    params = json.loads((Path(__file__).parents[1] / "fluidflower/fixtures/native_photometry_parameters.json").read_bytes())
    params["native_area_log_int_8"], params["minimum_sensor_signal"] = 2, 0.
    raw = tmp_path / "source.ARW"
    raw.write_bytes(b"synthetic source identity; pixels supplied at the physical decode seam")
    original_series = tmp_path / "original-series.toml"
    original_series.write_text('series_id="synthetic"\n')
    original_pin = asset(original_series)
    snapshot = tmp_path / "series-manifest.toml"
    snapshot.write_bytes(original_series.read_bytes())
    plan = _write_json(tmp_path / "plan.json", dict(series_manifest=original_pin, adopted_revision={}))
    request = _write_json(tmp_path / "request.json", dict(baseline=asset(raw)))
    operations = {run: _write_json(tmp_path / (run + ".json"), dict(
        run=run, setup_plan=plan, raw_build_request=request)) for run in ("r1", "r2")}
    monkeypatch.setattr(subject, "completed_build_operations", lambda *a: operations)
    series = tmp_path / "series.toml"
    choice = image_normalisation.choice_record("synthetic", list(operations), "off", None, color_method="off")
    series.write_text(toml.dumps(dict(series_id="synthetic", runs=dict(ids=list(operations)),
        method=dict(active="phase_model"), image_normalisation=choice)))
    loaded, fits = [], []
    def context(operation, *, reader):
        run = next(run for run in operations if operations[run] == operation)
        return SimpleNamespace(run=run)
    def collect(value, reader, *, require_chart):
        assert not require_chart
        loaded.append(value.run)
        signal = np.full((3, 8, 10), .3 if value.run == "r1" else .4, np.float32)
        return dict(run=value.run, rgb_full=signal * 100, rgb_native=np.moveaxis(signal, 0, -1),
            white_range_dn=100., native_to_camera_u_h=np.eye(3), chart_rgb=None, chart_roi_rc=None,
            source_pins=dict(raw_acquisition=asset(raw), camera_binding=dict(profile_id="camera")),
            shape_assets=[plan], image_metadata={}, parameters=dict(photometry=params,
                support=dict(coverage_tolerance=1e-6)))
    def produce(ops, current, authority, target, **kwargs):
        fits.append(1)
        return prepare_image_photometry(ops, current, authority, target, reader=kwargs["reader"],
            context_loader=context, collector=collect, workers=1,
            materialize_baselines=kwargs["materialize_baselines"])
    config = dict(job_root=str(tmp_path / "job"), runs=list(operations), output_root=str(tmp_path / "image-output"),
        choice_authority=asset(series), adopted_revision={}, native_parameters={})
    colour_calls = []
    def colour_paths(ops, basis, authority, target, **options):
        colour_calls.append((ops, basis, authority, options["workers"]))
        target.mkdir(parents=True, exist_ok=True)
        from darsia.fluidflower.mass.pooled_optical_paths import SETUP_METHOD
        from darsia.fluidflower.pipeline.setup_colour_paths import SIGNAL_AXIS_CONTRACT
        axis=dict(contract=SIGNAL_AXIS_CONTRACT,scales_by_facies={"1":1.},source_cohort=list(ops),
            frozen_for_production=True,units="corrected-clearwater-relative-rgb-euclidean-distance",
            scale_rule="maximum-supported-path-length-per-facies")
        packages={run:dict(metadata=_write_json(target/(run+"-metadata.json"),dict(
            recipe=dict(method=SETUP_METHOD,fit_runs=[run],runs={run:{}}),source_bindings={run:{}},
            signal_axis=axis,path_audit={"1":dict(status="supported")})),
            color_paths={"1":_write_json(target/(run+"-path.json"),dict(
                color_path=dict(colors=[[0.,0.,0.],[1.,0.,0.]]),values=[0.,1.]))},
            signals={"1":_write_json(target/(run+"-signal.json"),{})},
            flash=_write_json(target/(run+"-flash.json"),{})) for run in ops}
        return _write_json(target / "receipt.json", dict(format="darsia-setup-colour-paths-v2", complete=True,
            scope="per-run",packages_by_run=packages,signal_axis=axis,
            operations=ops, choice_authority=authority, runs=list(ops), image_count_per_run=15,
            phase_inference_performed=False, model_training_performed=False, mass_fit_performed=False))
    prepared = subject.run_image_preparation(config, workers=2, producer=produce, colour_path_producer=colour_paths)
    result = DataReader().json(prepared)
    assert result["complete"] and loaded == ["r1", "r2"] and len(fits) == 1
    assert not result["model_training_performed"] and result["colorpaths_performed"]
    assert len(colour_calls) == 1 and colour_calls[0][3] == 2
    for run, expected in (("r1", 30.), ("r2", 40.)):
        cache = DataReader().json(result["runs"][run]["baseline"])
        with np.load(cache["products"]["path"], allow_pickle=False) as archive:
            np.testing.assert_allclose(np.exp(archive["log_linear_rgb"]), expected, rtol=1e-6)
    # A completed operation is reusable without another decode, fit or write.
    assert subject.run_image_preparation(config, workers=2, producer=produce, colour_path_producer=colour_paths) == prepared
    assert len(fits) == 1 and loaded == ["r1", "r2"]
    snapshot = dict(runs=[dict(run=run, operation="build", state="Complete",
        result=dict(files={"operation_result.json": value})) for run, value in operations.items()] + [
        dict(run=subject.IMAGE_RUN, operation=subject.OPERATION, state="Complete",
             result=dict(image_preparation=prepared))])
    accepted = subject.completed_image_preparation(snapshot, asset(series))
    assert accepted["runs"] == result["runs"]
    complete = subject.completed_setup_colour_paths(snapshot, asset(series))
    assert set(complete["runs"]) == set(operations) and complete["complete"]
    before = DataReader().json(plan)
    reader = subject.image_basis_reader(DataReader(), result)
    assert 'series_id="synthetic"' in reader.path(original_pin).read_text()
    assert DataReader().json(plan) == before


def test_image_dependency_does_not_read_retained_tree_when_builds_are_current_done(tmp_path, monkeypatch):
    from darsia.fluidflower import setup_completed_builds
    (tmp_path / "done").mkdir()
    (tmp_path / "done/r1.json").write_text("{}")
    monkeypatch.setattr(setup_completed_builds, "retained_operations", lambda *a: pytest.fail("Unnecessary history replay"))
    job = dict(setup_operations=[dict(run="r1", operation="build")])
    assert subject.build_dependencies_ready(tmp_path, job)
    (tmp_path / "done/r1.json").unlink()
    assert not subject.build_dependencies_ready(tmp_path, job)
