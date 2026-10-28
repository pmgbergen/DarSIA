from copy import deepcopy

import pytest
import numpy as np

from darsia.fluidflower.mass.n1_pipeline import N1Pipeline, asset
from darsia.fluidflower.mass.production_precision import PixelMath
from .test_local_build_inputs import local_build_sources, local_build_case
from .test_staged_data import phase1_data
from .test_local_overflow_sources import saved_case
from .test_pooled_optical_paths import recipe as pooled_recipe, _frame


def test_optical_time_subset_is_shared_bounded_and_independent_of_mass_inputs():
    from darsia.fluidflower.pipeline.current_optical_preparation import optical_frame_selection
    times = [-.25, .001, *[.5 * 1.3 ** index for index in range(35)]]
    selection = dict(mass_fit_runs=["fit"], runs=[dict(run=run,
        baseline=dict(raw_sha256="b" * 64), selected_frames=[dict(
            raw_sha256=f"{index:064x}", current_frame_index=index, elapsed_h=time,
            nominal_fit_target_h=.5 if index == 2 else None,
            fit_candidate=run == "fit") for index, time in enumerate(times) if time <= end])
        for run, end in (("fit", max(times)), ("control", max(times)), ("short", 5.))])
    before = deepcopy(selection)
    chosen = optical_frame_selection(selection)
    assert selection == before  # Every P1/mass image, baseline and role stays intact.
    assert chosen["max_images_per_run"] == 15 and len(chosen["target_hours"]) == 15
    assert chosen["target_hours"][1] == pytest.approx(.5)
    assert [row["target_h"] for row in chosen["runs"]["fit"]] == [
        row["target_h"] for row in chosen["runs"]["control"]]
    assert chosen["runs"]["fit"][-1]["frame"]["elapsed_h"] == max(times)
    assert 10 <= len(chosen["runs"]["fit"]) <= 15
    for run, rows in chosen["runs"].items():
        assert len(rows) <= 15
        assert len({row["frame"]["raw_sha256"] for row in rows}) == len(rows)
        assert all(row["frame"]["elapsed_h"] >= 0 for row in rows)
    assert all(row["target_h"] <= 5. for row in chosen["runs"]["short"])


@pytest.mark.parametrize("count", [True, 0, 1001, 15., "15"])
def test_optical_count_rejects_invalid_saved_settings(count):
    from darsia.fluidflower.pipeline.current_optical_preparation import colour_path_image_count
    with pytest.raises(ValueError, match="representative images"):
        colour_path_image_count(dict(image_selection=dict(color_paths=dict(max_images_per_run=count))))


def test_optical_count_ignores_production_and_uses_its_own_saved_setting():
    from darsia.fluidflower.pipeline.current_optical_preparation import colour_path_image_count
    assert colour_path_image_count(dict(production_image_selection=dict(images=80))) == 15
    assert colour_path_image_count(dict(image_selection=dict(color_paths=dict(max_images_per_run=7)))) == 7


def test_local_frame_loading_keeps_same_recipe_without_old_optical_reads(local_build_case):
    case = local_build_case
    local = case.make_adapter()
    config = local.bind_configuration(case.configuration)
    mass = dict(config["statistics"], run=config["run"], frames=config["frames"], data_aliases=())
    # None is deliberate: a new shared optical package has not been fitted.
    mass["shared"] = deepcopy(mass["shared"])
    mass["shared"]["optics"] = None
    for name, values in (("floor", np.full(3, 2., np.float32)),
                         ("clear", np.ones((3, 3, 3), np.float32)),
                         ("valid", np.ones((3, 3), bool))):
        np.save(case.sources.inputs / (name + ".npy"), values)
    mass["shared"]["floor_curve"] = asset(case.sources.inputs / "floor.npy")
    mass["shared"]["clearwater"].update(rgb=asset(case.sources.inputs / "clear.npy"),
                                      valid=asset(case.sources.inputs / "valid.npy"))
    local.config["statistics"]["shared"] = deepcopy(mass["shared"])
    pipeline = N1Pipeline(mass, precision="float32", pixel_math=PixelMath("float32"),
                          frame_loading_only=True, local_build=local)
    assert pipeline.frame_loading_only
    assert not hasattr(pipeline, "models")
    changed = deepcopy(mass)
    changed["shared"]["porosity_by_facies"]["1"] = .7
    with pytest.raises(ValueError, match="exact selected"):
        N1Pipeline(changed, precision="float32", pixel_math=PixelMath("float32"),
                   frame_loading_only=True, local_build=local)


def test_optical_sampling_uses_original_n1_checked_local_frame(local_build_case, tmp_path):
    from darsia.fluidflower.pipeline import current_optical_preparation as current
    from darsia.fluidflower.mass import pooled_optical_paths as optical
    from darsia.fluidflower.mass.n1_pipeline import _write_npz
    from darsia.fluidflower.pipeline.series import bind_mass_frame
    case = local_build_case
    _live, local, record, _spec = saved_case(case, tmp_path)
    record["phase"]["source_sha256"] = case.spec["source"]["source_sha256"]
    mass_spec = bind_mass_frame(case.spec, record["native"], record["phase"], baseline=case.full)
    mask = _write_npz(tmp_path / "membership.npz", dict(keep=np.ones((3, 3), bool)))
    mass_spec["photo_top"] = dict(mask, array_key="keep")
    mass_spec["original_domain"] = dict(membership=dict(mask, array_key="keep"),
        native_shape_hw=[3, 3], frame_index=mass_spec["frame_index"], H=np.eye(3).tolist())
    mass_spec["inner_polygon_native"] = [[0, 0], [3, 0], [3, 3], [0, 3]]
    config = local.config
    shared = config["statistics"]["shared"]
    shared["optics"] = None
    for name, values in (("floor", np.full(3, 2., np.float32)),
                         ("clear", np.ones((3, 3, 3), np.float32)),
                         ("valid", np.ones((3, 3), bool))):
        np.save(tmp_path / (name + ".npy"), values)
    shared["floor_curve"] = asset(tmp_path / "floor.npy")
    shared["clearwater"].update(rgb=asset(tmp_path / "clear.npy"), valid=asset(tmp_path / "valid.npy"))
    mass = dict(config["statistics"], run=config["run"], frames=config["frames"], data_aliases=())
    pipeline = N1Pipeline(mass, precision="float32", pixel_math=PixelMath("float32"),
                          frame_loading_only=True, local_build=local)
    local.restore(case.spec, record, mass_spec)
    raw = case.spec["source"]["source_binding"]["raw_acquisition"]
    run = config["run"]
    roles = [dict(facies_id=key, name=row["name"], analysis_role=row["analysis_role"])
             for key, row in local.roles.items()]
    value = dict(format=optical.FORMAT, method=optical.METHOD, fit_runs=[run, "other"], excluded_runs=[],
        roles=roles, adopted_revision=local.context.provenance["adopted_revision"],
        build_context=local.operation_result, phase_configuration=config["phase"]["configuration"],
        native_configuration=config["native"]["configuration"],
        runs={r: dict(baseline=raw, clearwater=raw, profile=raw, frames=[dict(
            case_id=mass_spec["case_id"], frame_index=mass_spec["frame_index"],
            elapsed_h=mass_spec["elapsed_h"], target_h=mass_spec["elapsed_h"], source=raw)]) for r in [run, "other"]},
        sampling=dict(stride_px=1, interior_radius_px=0, min_samples=1, min_runs=2),
        flash_bounds=dict(min_value_aq=0., max_value_aq=1., min_value_g=0., max_value_g=1.),
        path_name_prefix="Synthetic ")
    recipe = optical.recipe_from_document(value)
    sample = current.sample_local_frame(pipeline, mass_spec, recipe)
    assert sample.run == run
    assert sample.audit["frame_provenance"]["physical_context_verification"].startswith("original N1")
    saved = current.save_samples(tmp_path / "sample", sample)
    loaded = current.load_samples(saved, pipeline.reader)
    for key in sample.samples:
        np.testing.assert_array_equal(loaded.samples[key], sample.samples[key])


def test_shared_optical_barrier_precedes_any_completed_statistics(pooled_recipe, tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import current_build_job as job_api
    from darsia.fluidflower.pipeline import current_optical_preparation as preparation
    from darsia.fluidflower.mass import pooled_optical_paths as optical
    from darsia.fluidflower.mass.n1_pipeline import DataReader, _write_json
    recipe = pooled_recipe
    expected = recipe.document
    source = expected["runs"]["alpha"]["frames"][0]["source"]
    root = tmp_path / "job"
    root.mkdir()
    selection = dict(runs=[dict(run=run, selected_frames=[dict(
        raw_sha256=source["sha256"], current_frame_index=1,
        elapsed_h=1., nominal_fit_target_h=1., fit_candidate=True)]) for run in expected["fit_runs"]])
    job = dict(root=str(root), results_root=str(tmp_path / "results"),
        selection=_write_json(root / "selection.json", selection), target_machine="test",
        fresh_optical_paths=dict(sampling=expected["sampling"]), inputs=[dict(run=run) for run in expected["fit_runs"]])
    events = []
    def prepare(job, row, *args):
        events.append("prepare:" + row["run"])
        return dict(complete=True, run=row["run"], roles=expected["roles"],
            basis={**{key: expected[key] for key in ("adopted_revision", "build_context", "phase_configuration", "native_configuration")},
                   **{key: expected["runs"][row["run"]][key] for key in ("baseline", "clearwater", "profile")}},
            source_rows={row["run"] + "-" + source["sha256"][:20]: source}, flash_bounds=expected["flash_bounds"])
    def infer(job, row, *args):
        assert not (root / "inputs" / "current_optics.json").exists()
        events.append("infer:" + row["run"])
        current = optical.recipe_from_document(DataReader().json(
            __import__("darsia.fluidflower.mass.n1_pipeline", fromlist=["asset"]).asset(root / "inputs" / "optical_recipe.json")))
        frame = _frame(recipe, row["run"])
        spec = current.document["runs"][row["run"]]["frames"][0]
        frame.provenance["optical_source_contract"] = optical._source_contract(current.document, row["run"], spec)
        sample = optical.sample_frame_for_common_paths(frame, current, case_id=spec["case_id"])
        pin = preparation.save_samples(tmp_path / (row["run"] + "-samples"), sample)
        phase = _write_json(tmp_path / (row["run"] + "-phase.json"), dict(samples=pin))
        return dict(complete=True, run=row["run"], recipe_sha256=current.sha256, frames=[phase])
    def finish(job, row, *args):
        assert (root / "inputs" / "current_optics.json").is_file()
        events.append("statistics:" + row["run"])
        return dict(complete=True, run=row["run"])
    def dispatch(job, commit, stores, workers, log, *, runner, publication_folder="results", **kwargs):
        for row in job["inputs"]:
            assert not (root / "results").exists() or publication_folder == "results"
            result = runner(job, row, commit, stores, None)
            job_api._write_json(root / publication_folder / row["run"] / "completed.json", result)
        return 0
    monkeypatch.setattr(job_api, "_prepare_optical_run", prepare)
    monkeypatch.setattr(job_api, "_infer_optical_run", infer)
    monkeypatch.setattr(job_api, "_finish_optical_run", finish)
    monkeypatch.setattr(job_api, "_dispatch_runs", dispatch)
    from darsia.fluidflower.pipeline import worker
    monkeypatch.setattr(worker, "_job_control_boundary", lambda root: None)
    monkeypatch.setattr(worker, "emit", lambda *args: None)
    assert job_api._run_fresh_optical_job(job, "a" * 40, None, 2, None) == 0
    assert events == ["prepare:alpha", "prepare:beta", "infer:alpha", "infer:beta",
                      "statistics:alpha", "statistics:beta"]


def test_final_statistics_reuses_owned_phase_files_without_inference(phase1_data, tmp_path, monkeypatch):
    """Exercise the final serializer; numerical physical loading is tested above."""
    from types import SimpleNamespace
    from .test_staged_data import _closure
    from darsia.fluidflower.mass.n1_pipeline import _write_json, DataReader
    from darsia.fluidflower.pipeline import phase_statistics as api, staged_data
    from darsia.fluidflower.pipeline.local_overflow_sources import SavedLocalBuildInputs
    from darsia.fluidflower.phase import infer
    from darsia._opencv_runtime import require_ipp_off
    old = _closure(tmp_path, phase1_data, pending=True)
    record = old["record"]
    for key in ("configuration", "completion", "frame"):
        __import__("pathlib").Path(old[key]["path"]).unlink()
    __import__("pathlib").Path(record["statistics"]["path"]).unlink()
    previous = deepcopy(phase1_data)
    previous["statistics"]["shared"]["optics"] = None
    original = _write_json(tmp_path / "optical_configuration.json", previous)
    stage = tmp_path / "preparation" / phase1_data["frames"][0]["case_id"]
    photograph = api._save_original_photo(stage, {"run": phase1_data["run"]},
        phase1_data["frames"][0], None, DataReader())
    source_record = _write_json(stage / "optical_phase_source.json", dict(
        case_id=record["case_id"], source=record["source"], native=record["native"],
        native_receipt=record["native_receipt"], phase=record["phase"],
        before_rules=record["before_rules"], photograph=photograph, probabilities=None))
    sources = dict(configuration=original, phase_sources_complete=True, statistics_complete=False,
                   code_commit="1" * 40, frames=[source_record])
    replay = object.__new__(SavedLocalBuildInputs)
    restored = []
    replay.restore = lambda *args: restored.append(args)
    class Statistics:
        def __init__(self, *args, **kwargs):
            self.last_opencv_ipp = require_ipp_off("synthetic final-statistics transport")
        def extract_statistics(self, spec):
            return dict(statistics=dict(numerical_fixture=1), sandmap={})
    monkeypatch.setattr("darsia.fluidflower.mass.n1_pipeline.N1Pipeline", Statistics)
    monkeypatch.setattr(infer, "infer_labels", lambda *a, **k: pytest.fail("Inference must not repeat"))
    result = api.finish_optical_phase_statistics(phase1_data, tmp_path, sources,
        local_build=replay, code_commit="1" * 40)
    assert result["complete"] and len(restored) == 1
    verified = staged_data.verify_phase1_completion(result["phase1_completion"], result["configuration"],
        expected_commit="1" * 40, reader=DataReader())
    assert len(verified["records"]) == 1
    assert verified["records"][0]["record"]["phase"] == record["phase"]


def test_phase_sources_are_not_reported_as_complete_p1(tmp_path):
    import json
    from darsia.fluidflower.pipeline.current_build_job import current_build_snapshot
    from darsia.fluidflower.jobs_progress import progress_view
    root = tmp_path / "job"
    (root / "logs").mkdir(parents=True)
    marker = root / "optical-inputs" / "one" / "completed.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"format":"darsia-current-optical-input-run-v1","complete":true,"run":"one"}')
    stamp = "2026-10-26T12:00:00+00:00"
    event = dict(kind="progress", run="one", work_stage="phase-inference",
        state="optical_phase_frame_completed", completed_count=2,
        actual_worker=dict(machine="host", pid=123))
    (root / "logs" / "host.log").write_text(stamp + " " + json.dumps(event) + "\n")
    job = dict(inputs=[dict(run="one", selected_frames=2)], fresh_optical_paths={},
               created_utc=stamp, series_id="sample", target_machine="host")
    job["fresh_optical_paths"] = {"sampling": {}}
    (root / "job.json").write_text(json.dumps(job))
    snapshot = current_build_snapshot(dict(root=str(root)), job, {},
                                      reader=lambda path: json.loads(path.read_text()))
    assert snapshot["committed"] == 0 and not snapshot["complete"]
    assert snapshot["work_unit"] == "operations"
    row = snapshot["runs"][0]
    assert row["completed_operations"] == 5 and row["total_operations"] == 7
    assert progress_view(snapshot)["rows"][0]["percent"] == "Estimated 71.4%"


def test_explicit_single_fit_run_uses_one_source_without_duplicate_weight(pooled_recipe):
    from darsia.fluidflower.mass import pooled_optical_paths as optical
    value = pooled_recipe.document
    value["fit_runs"] = ["alpha"]
    value["runs"] = {"alpha": value["runs"]["alpha"]}
    value["sampling"]["min_runs"] = 1
    recipe = optical.recipe_from_document(value)
    batch = optical.sample_frame_for_common_paths(_frame(recipe, "alpha"), recipe, case_id="alpha-1")
    paths, audit, sources = optical.fit_pooled_color_paths(recipe, [batch])
    assert paths and set(sources) == {"alpha"}
    assert all(row["run_count"] == 1 for row in audit.values() if row["status"] == "supported")
    with pytest.raises(ValueError, match="differ|selection"):
        optical.fit_pooled_color_paths(recipe, [batch, batch])


def test_thirteen_nonfit_runs_supply_optical_samples_without_mass_fit_membership(pooled_recipe, tmp_path):
    from darsia.fluidflower.pipeline.current_optical_preparation import recipe_for_runs
    from darsia.fluidflower.mass import pooled_optical_paths as optical
    template = pooled_recipe.document
    raw = template["runs"]["alpha"]["frames"][0]["source"]
    outside = ["control-" + str(index) for index in range(13)]
    rows, selection = [], {"runs": []}
    for run in ["alpha", "beta", *outside]:
        fit = run in {"alpha", "beta"}
        case = run + "-" + raw["sha256"][:20]
        rows.append(dict(run=run, roles=template["roles"], flash_bounds=template["flash_bounds"],
            basis={**{key: template[key] for key in ("adopted_revision", "build_context", "phase_configuration", "native_configuration")},
                   **{key: template["runs"]["alpha"][key] for key in ("baseline", "clearwater", "profile")}},
            source_rows={case: raw}))
        selection["runs"].append(dict(run=run, selected_frames=[dict(raw_sha256=raw["sha256"],
            fit_candidate=fit, current_frame_index=1, elapsed_h=1., nominal_fit_target_h=1.)]))
    recipe = recipe_for_runs(rows, selection, tmp_path / "fresh", sampling=template["sampling"])
    assert recipe.document["fit_runs"] == ["alpha", "beta", *outside]
    assert recipe.document["excluded_runs"] == []
    assert all(not frame["fit_candidate"] for row in selection["runs"][2:] for frame in row["selected_frames"])
    samples = [optical.sample_frame_for_common_paths(_frame(recipe, run), recipe,
        case_id=run + "-" + raw["sha256"][:20]) for run in recipe.document["fit_runs"]]
    paths, audit, sources = optical.fit_pooled_color_paths(recipe, samples)
    assert paths and set(sources) == {"alpha", "beta", *outside}
    assert all(row["run_count"] == 15 for row in audit.values() if row["status"] == "supported")


def test_stop_at_optical_barrier_is_not_a_failed_run(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from darsia.fluidflower.pipeline import current_build_job as job_api, worker
    events = []
    def stop(*args):
        raise worker._JobControlBoundary("control_stopped")
    monkeypatch.setattr(job_api, "_run_fresh_optical_job", stop)
    monkeypatch.setattr(worker, "emit", lambda log, value: events.append(value))
    job = dict(target_machine="host", pipeline="native-phase-statistics",
        expected_runs=["one"], inputs=[dict(run="one")], fresh_optical_paths={"sampling": {}})
    args = SimpleNamespace(backend="cpu", machine="host", job=tmp_path, stores=None, workers=2)
    assert job_api.run_current_build_job(args, job, "a" * 40, None) == 0
    assert not (tmp_path / "failed").exists()
    assert any(row["kind"] == "control_stopped" for row in events)
