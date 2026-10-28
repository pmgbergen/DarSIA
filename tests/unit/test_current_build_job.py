"""Finite current-Build handoff tests; no campaign or RAW decoder is started."""
from copy import deepcopy
from contextlib import nullcontext
import gzip
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.mass.build_native_bridge import BuildNativeFrame
from darsia.fluidflower.mass.n1_pipeline import DataReader, asset, _write_json, _write_npz
from darsia.fluidflower.pipeline import current_build_job as subject
from darsia.fluidflower.pipeline.build_frame_geometry import prepare_build_frame_geometry
from darsia.fluidflower.pipeline.series import SeriesInputs
from darsia.fluidflower.pipeline.staged_data import validate_phase1_run
from .test_local_build_inputs import local_build_sources, local_build_case
from .test_staged_data import phase1_data
from ..fixtures.tiny_saved_model_fixture import tiny_config


@pytest.fixture
def prepared_case(local_build_case, tmp_path):
    case = local_build_case
    reader = DataReader()
    floor_path = tmp_path / "floor.npy"
    np.save(floor_path, np.zeros(3))
    phase = dict(geometry_parameters=dict(case.geometry_parameters, observation_horizon_floor=1e-10),
        parameters=dict(geometry=dict(rig_origin_xy=[0, 0], native_shape_hw=[3, 3],
            source_box_xywh=[0, 0, 3, 3], floor_source_origin_xy=[0, 0]),
            scientific=dict(contract="test-full", storage="test-windows", photograph_canvas_wh=[3, 3],
                            native_shape_hw=[3, 3], native_box_xywh=[0, 0, 3, 3])),
        phase_shape_hw=[3, 3], geometry_fields=dict(reference_to_frame="H_reference_to_frame",
            reference_to_metric="H_reference_to_metric"), chain_assets=dict(floor=asset(floor_path)))
    phase_binding = _write_json(tmp_path / "phase.json", phase)
    case.configuration["phase"]["configuration"] = phase_binding
    adapter = case.make_adapter()
    product = _write_npz(tmp_path / "map.npz", dict(array=case.sources.labels, inside=np.ones((3, 3), bool)))
    specs, originals, produced, native_records = [], [], {}, {}
    previous = case.sources.raw["sha256"]
    for index, (time, target, candidate) in enumerate(((.125, None, False), (.497, .5, True)), 1):
        raw_path = tmp_path / (str(index) + ".ARW")
        raw_path.write_bytes(("TESTONLY " + str(index)).encode())
        raw = asset(raw_path)
        native_row = dict(reader.json(case.spec["native_task"]["geometry"]), acquisition=raw)
        native_binding = _write_json(tmp_path / (str(index) + "-native.json"), native_row)
        audit = deepcopy(case.receipt["correction"])
        audit["sources"]["camera_arw"] = raw
        audit["renderer"]["camera_u_wh"] = [3, 3]
        receipt = dict(case.receipt, native_frame=native_binding, correction=audit)
        receipt_binding = _write_json(tmp_path / (str(index) + "-receipt.json"), receipt)
        buffers = _write_npz(tmp_path / (str(index) + "-buffers.npz"),
            dict(log_linear_rgb=case.native["full_log"], valid_mask=np.ones((3, 3), bool)))
        native = BuildNativeFrame(dict(case.spec["native_task"], geometry=native_binding), buffers, receipt_binding)
        geometry = prepare_build_frame_geometry(adapter, native, map_product=product,
                                                output=tmp_path / (str(index) + "-geometry"))
        spec = deepcopy(case.spec)
        spec.update(case_id="frame-" + str(index), frame_index=index, elapsed_h=time,
                    geometry=geometry, native_task=native.native_task)
        spec["source"].update(source_sha256=raw["sha256"], elapsed_h=time, work_index=index,
            current_frame_index=index, previous_source_sha256=previous,
            source_binding=dict(raw_acquisition=raw, source_sha256=raw["sha256"]))
        spec["phase_roles"].pop("native_row", None)
        spec["phase_roles"].pop("probability_receipt", None)
        spec["phase_roles"].update(photo_sand_reference=asset(floor_path), photo_proof={"fixture": True})
        specs.append(spec)
        produced[spec["case_id"]] = native
        originals.append(dict(nominal_fit_target_h=target, elapsed_h=time, fit_candidate=candidate))
        previous = raw["sha256"]
        source_stage = tmp_path / (str(index) + "-source")
        source_stage.mkdir()
        audit_binding = _write_json(source_stage / "native_audit.json", audit)
        full = deepcopy(case.full)
        full.update(contract="test-full", storage="test-windows", source_sha256=raw["sha256"])
        full["binding"] = dict(geometry=reader.json(geometry), producer_audit=audit_binding)
        record = dict(complete=True, source_sha256=raw["sha256"], processor_kind="raw_rig",
                      full=full, audit=audit_binding)
        native_records[spec["case_id"]] = _write_json(source_stage / "native.json", record)
    reference = dict(full=case.full, native_row=asset(floor_path),
        source=dict(source_sha256=case.sources.raw["sha256"]), current_h=np.eye(3).tolist())
    data = dict(run=case.configuration["run"], native=case.configuration["native"],
        phase=dict(configuration=phase_binding), reference_spec=dict(case_id="baseline"), frames=specs,
        history_policy="strict original acquisition order", selection_index_field="work_index")
    prepared = dict(configuration=data, phase_recipe=dict(phase=phase), acquisitions={"baseline": reference},
        operation_result=case.sources.operation_result, metric_model=case.sources.metric_model,
        produced=produced, prepared_native=native_records)
    template = deepcopy(case.configuration["statistics"])
    for key in ("physical_reference", "reference_matrix", "depth_points", "porosity_by_facies"):
        template["shared"].pop(key, None)
    template.pop("boxes")
    measurements = tmp_path / "depth.csv"
    measurements.write_text("x,y,depth\n0,0,.01\n.003,0,.01\n0,.003,.01\n")
    statistics = dict(statistics=template, atmosphere=dict(rows=[dict(time_hours=0., pressure_bar=1.,
        temperature_c=20., pressure_gradient_bar_per_m=0., temperature_gradient_c_per_m=0.)]),
        dose=dict(dose_kind="nominal", injection_stages=[]), light_epoch="same")
    return SimpleNamespace(prepared=prepared, selection=dict(runs=[dict(run=data["run"], selected_frames=originals)]),
        statistics=statistics, measurements=asset(measurements), phase=phase, case=case)


def test_actual_build_configuration_keeps_early_history_and_original_fit_targets(prepared_case, monkeypatch, tmp_path):
    from darsia.fluidflower.pipeline.refit_jobs import statistics_entries
    f = prepared_case
    monkeypatch.setattr(subject, "_depth_measurements", lambda *args: f.measurements)
    config = subject.phase_configuration(f.prepared, f.selection, f.statistics)
    validate_phase1_run(config)
    from darsia.fluidflower.phase.probability_storage import parse_request, ENCODING
    assert parse_request(config)["encoding"] == ENCODING == "float32-exact-decision-values"
    assert config["storage"]["gas_score_encoding"] == "uint8-la-q255-v1"
    assert [row["elapsed_h"] for row in config["frames"]] == [.125, .497]
    assert [row["mass_spec"]["selected_target_h"] for row in config["frames"]] == [None, .5]
    assert "physical_reference" not in config["statistics"]["shared"]
    assert config["statistics"]["shared"]["porosity_by_facies"]["0"] == 1.
    for frame in config["frames"]:
        path = tmp_path / "fit" / "preparation" / frame["case_id"] / "native_statistics.json.gz"
        path.parent.mkdir(parents=True)
        with gzip.open(path, "wt") as stream:
            json.dump(dict(frame_index=frame["frame_index"], selected_target_h=frame["mass_spec"]["selected_target_h"],
                fit_candidate=frame["mass_spec"]["provenance"]["fit_candidate"]), stream)
    assert [row["frame_index"] for row in statistics_entries(tmp_path / "fit")] == [2]


def test_prepared_native_reuses_exact_raw_products_in_original_commit_namespace(prepared_case, tmp_path, monkeypatch):
    from darsia.fluidflower.native.raw_rig_source import RawRigSourceProcessor
    from darsia.fluidflower.pipeline.runner import _inventory
    from darsia.fluidflower.pipeline.staged_data import _owned
    f = prepared_case
    spec = f.prepared["configuration"]["frames"][0]
    owner = object.__new__(SeriesInputs)
    owner.prepared_native = f.prepared["prepared_native"]
    owner.reader, owner.aliases, owner.phase = DataReader(), {}, f.phase
    owner.parameters, owner.native_parameters = f.phase["parameters"], {}
    monkeypatch.setattr(RawRigSourceProcessor, "correct", lambda *args: pytest.fail("RAW decoded twice"))
    stage = tmp_path / "phase1" / "preparation" / spec["case_id"]
    actual = owner.native(spec, stage)
    np.testing.assert_array_equal(actual["log"], f.case.native["full_log"])
    assert actual["record"]["source_sha256"] == spec["source"]["source_sha256"]
    inventory = _inventory(stage)
    _owned(actual["record_asset"], inventory, where="native receipt")
    _owned(actual["record"]["audit"], inventory, where="native audit")
    for field in actual["record"]["full"]["native_centre"].values():
        if isinstance(field, dict):
            _owned(field, inventory, where="native centre")


def test_run_worker_prepares_then_runs_local_p1_once(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import current_build_preparation, current_statistics_recipe, phase_statistics, local_build
    from darsia.fluidflower import store_map
    root, series = tmp_path / "job", tmp_path / "series"
    root.mkdir(); series.mkdir()
    (series / "series.toml").write_text("# TESTONLY\n")
    selection = _write_json(root / "selection.json", dict(input_authority=dict(current_series=asset(series / "series.toml"))))
    job = dict(job_id="test", execution_mode=subject.MODE, pipeline="native-phase-statistics", target_machine="Main",
        expected_runs=["sample"], inputs=[dict(run="sample", operation_result={}, metric_model={}, selected_frames=1)],
        series_id="test", series_root=str(series), results_root=str(tmp_path / "results" / "test"), selection=selection)
    calls = []
    monkeypatch.setattr(store_map, "read_store_map", lambda _: SimpleNamespace(roots=lambda _: {"results": str(tmp_path / "results")}))
    monkeypatch.setattr(subject, "_capacity", lambda *args: nullcontext())
    monkeypatch.setattr(current_statistics_recipe, "retained_statistics_recipe", lambda *args, **kw: {})
    def prepare(*args, **kwargs):
        calls.append("prepare")
        kwargs["on_progress"](dict(run="sample", state="raw_prepared", position=2, total=2))
        return dict(produced={"one": SimpleNamespace(receipt={})}, prepared_native={"one": {"path": "same"}})
    monkeypatch.setattr(current_build_preparation, "prepare_current_build_acquisitions", prepare)
    monkeypatch.setattr(subject, "phase_configuration", lambda *args: {"frames": [{"case_id": "one"}]})
    monkeypatch.setattr(local_build, "LocalBuildInputs", lambda *args, **kw: "exact-local-build")
    def run(config, output, **kwargs):
        calls.append("p1")
        assert kwargs["prepared_native"] == {"one": {"path": "same"}}
        assert kwargs["local_build"] == "exact-local-build"
        assert kwargs["threads"] == 1
        assert not (root / "results" / "sample" / "completed.json").exists()
        return dict(complete=True, run="sample", ordered_case_ids=["one"])
    monkeypatch.setattr(phase_statistics, "run_phase1_local", run)
    progress = []
    result = subject._run_current_build_run(dict(job, root=str(root)), job["inputs"][0], "a" * 40, None, progress.append)
    assert calls == ["prepare", "p1"]
    assert result["complete"] and progress[0]["state"] == "raw_prepared"
    assert not (root / "results" / "sample" / "completed.json").exists()


def test_raw_sensor_cache_retains_one_acquisition_and_rechecks_bytes(tmp_path, monkeypatch):
    from darsia.fluidflower.native import raw_rig_source
    processor = object.__new__(raw_rig_source.RawRigSourceProcessor)
    processor.parameters, processor.binding = {"registration": {}}, {"profile_id": "test"}
    processor._path = Path
    calls = []
    monkeypatch.setattr(raw_rig_source.raw_positioning, "read_sensor",
        lambda path, **kwargs: calls.append(path) or object())
    a, b = tmp_path / "a.ARW", tmp_path / "b.ARW"
    a.write_bytes(b"TESTONLY a"); b.write_bytes(b"TESTONLY b")
    first, second = asset(a), asset(b)
    sensor = processor._sensor(first)
    assert processor._sensor(first) is sensor and calls == [a]
    processor._sensor(second)
    processor._sensor(first)
    assert calls == [a, b, a]
    a.write_bytes(b"different bytes")
    with pytest.raises(ValueError, match="differs"):
        processor._sensor(first)


def test_metric_and_depth_bind_actual_completed_build_inputs(local_build_sources, tmp_path):
    case = local_build_sources
    csv = tmp_path / "measurements.csv"
    csv.write_text("x,y,depth\n0,0,.01\n.003,0,.011\n0,.003,.012\n")
    toml = tmp_path / "common.toml"
    toml.write_text('[depth]\nmeasurements = "' + csv.as_posix() + '"\n')
    request = _write_json(tmp_path / "request.json", dict(baseline=case.raw,
        native_product=dict(source_model=case.metric_model)))
    operation = _write_json(tmp_path / "operation.json", dict(run=case.context.run,
        raw_build_request=request, baseline_renderer=case.saved.result["baseline_renderer"],
        configuration_files=[asset(toml)]))
    reader = DataReader()
    metric = reader.json(subject._metric_for_build(operation, tmp_path / "metric.json", reader))
    assert metric["H_camera_u_to_metric_mm"] == case.model["H_camera_u_to_metric_mm"]
    assert metric["area_factor"] == case.model["area_factor"]
    assert metric["baseline_native_grid"]["renderer"] == case.saved.result["baseline_renderer"]
    assert subject._depth_measurements(operation, reader) == asset(csv)


def test_depth_source_in_retained_build_configuration_uses_declared_relocation(tmp_path):
    from darsia.fluidflower.pipeline.current_inputs import CurrentInputReader
    moved = tmp_path / "moved" / "depth.csv"
    moved.parent.mkdir()
    moved.write_text("x,y,depth\n0,0,.01\n1,0,.01\n0,1,.01\n")
    old = tmp_path / "old" / "depth.csv"
    config = tmp_path / "retained.toml"
    config.write_text('[depth]\nmeasurements = "' + old.as_posix() + '"\n')
    original = config.read_bytes()
    operation = _write_json(tmp_path / "build.json", dict(configuration_files=[asset(config)]))
    reader = CurrentInputReader(DataReader())
    reader.remap = lambda value: str(moved) if Path(value) == old else value
    assert subject._depth_measurements(operation, reader) == asset(moved)
    assert not old.exists() and config.read_bytes() == original


def test_current_job_uses_existing_capacity_ledger_before_preparation(local_build_sources, tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import capacity
    calls = []
    class Ledger:
        reserve = 0
        def __init__(self, *args, job_root):
            assert job_root == str(tmp_path)
            calls.append("ledger")
        def locked(self): return nullcontext()
        def fresh_volume(self): return dict(free_bytes=10**15)
        def remaining(self): return 0
        def bind(self, root, claim, owner, estimate):
            assert estimate.products_estimated_bytes > 0 and estimate.case_count == 2
            calls.append("reserved")
            return "reservation"
        def release(self, reservation, owner): calls.append("released")
    monkeypatch.setattr(capacity, "CapacityLedger", Ledger)
    with subject._capacity(None, dict(series_id="test", root=str(tmp_path), pipeline="native-phase-statistics", commit="a" * 40),
        dict(run="sample", selected_frames=2, ordered_case_ids=["frame-one", "frame-two"],
             operation_result=local_build_sources.operation_result),
        tmp_path, {}):
        calls.append("producer")
    assert calls == ["ledger", "reserved", "producer", "released"]


def test_original_statistics_keep_null_history_target_and_refit_uses_only_candidate(tiny_config, tmp_path):
    from darsia.fluidflower.mass.n1_pipeline import N1Pipeline
    from darsia.fluidflower.mass.production_precision import PixelMath
    from darsia.fluidflower.pipeline.phase_statistics import _save_statistics
    from darsia.fluidflower.pipeline.refit_jobs import statistics_entries
    config = deepcopy(tiny_config[0])
    config["geometry_fields"] = dict(source_to_native="H_ac28_to_frame_native",
        original_domain_to_native="H_ac28_to_frame_native", carried_homography_provenance="original_H")
    context = N1Pipeline(config, precision="float32", pixel_math=PixelMath("float32"), statistics_only=True)
    for spec, (elapsed, target, candidate) in zip(config["frames"], ((.125, None, False), (.497, .5, True))):
        spec.update(elapsed_h=elapsed, selected_target_h=target, provenance=dict(fit_candidate=candidate))
        result = context.extract_statistics(spec)
        assert result["statistics"]["selected_target_h"] == target
        stage = tmp_path / "actual-prefix" / "preparation" / spec["case_id"]
        stage.mkdir(parents=True)
        _save_statistics(stage, result)
    entries = statistics_entries(tmp_path / "actual-prefix")
    assert [(row["frame_index"], row["selected_target_h"]) for row in entries] == [(3, .5)]


@pytest.mark.parametrize("optical_count", [None, 7])
def test_completed_build_metadata_opens_in_existing_jobs_without_raw_execution(local_build_sources, tmp_path, monkeypatch, optical_count):
    from darsia.fluidflower import analysis_preparation, store_map
    from darsia.fluidflower.pipeline import current_calibration_selection
    from darsia.fluidflower.jobs_metadata import JobSnapshotReader
    from darsia.fluidflower.pipeline import current_model_basis
    monkeypatch.setattr(current_model_basis, "require_current_model_basis", lambda *args, **kwargs: None)
    case = local_build_sources
    request = _write_json(tmp_path / "request-for-p1.json", dict(baseline=case.raw,
        native_product=dict(source_model=case.metric_model)))
    operation = _write_json(tmp_path / "operation-for-p1.json", dict(run=case.context.run,
        raw_build_request=request, baseline_renderer=case.saved.result["baseline_renderer"]))
    monkeypatch.setattr(analysis_preparation, "job_identity", lambda _: dict(commit="a" * 40,
        environment_lock=dict(path="requirements-worker.lock", sha256="b" * 64)))
    monkeypatch.setattr(store_map, "read_store_map", lambda _: SimpleNamespace(roots=lambda _: {"results": str(tmp_path / "local-results")}))
    def selection(root, output, **kwargs):
        value = dict(runs=[dict(run=case.context.run, counts=dict(selected_frames=2),
            selected_frames=[dict(raw_sha256="a" * 64), dict(raw_sha256="b" * 64)])])
        if optical_count is not None:
            import toml
            path = root / "series.toml"
            path.write_text(toml.dumps(dict(image_normalisation=dict(method="off", color_method="off"),
                image_selection=dict(color_paths=dict(max_images_per_run=optical_count)))), encoding="utf-8")
            value["input_authority"] = dict(current_series=asset(path))
        return _write_json(output, value)
    monkeypatch.setattr(current_calibration_selection, "save_current_calibration_selection", selection)
    snapshot = dict(root=str(tmp_path / "setup"), complete=True, job_facts=dict(pipeline="series-setup"),
        runs=[dict(run=case.context.run, operation="build", state="Complete",
                   result=dict(files={"operation_result.json": operation}))])
    registration = subject.prepare_current_build_job(snapshot, series_root=tmp_path, series_id="test", machine="Main")
    actual = JobSnapshotReader()(registration)
    assert actual["state"] == "Waiting" and actual["total_frames"] == 2
    assert actual["job_facts"]["execution_mode"] == subject.MODE
    assert actual["job_facts"]["expected_runs"] == [case.context.run]
    assert "purpose" not in actual["job_facts"] and registration["name"] == "Generate phase maps from model"
    if optical_count is not None:
        assert actual["job_facts"]["fresh_optical_paths"]["frame_selection"]["max_images_per_run"] == optical_count
        assert actual["job_facts"]["inputs"][0]["selected_frames"] == 2  # Independent P1 cohort.


@pytest.mark.parametrize("optical", [False, True])
def test_production_purpose_prepares_the_saved_grid_after_each_build_baseline(local_build_sources, tmp_path,
                                                                              monkeypatch, optical):
    from darsia.fluidflower import analysis_preparation, store_map
    from darsia.fluidflower.pipeline import current_model_basis, current_production_selection
    from darsia.fluidflower.jobs_metadata import JobSnapshotReader
    monkeypatch.setattr(current_model_basis, "require_current_model_basis",
                        lambda *args, **kwargs: dict(model_root="models/new"))
    case = local_build_sources
    request = _write_json(tmp_path / "request-for-p1.json", dict(baseline=case.raw,
        native_product=dict(source_model=case.metric_model)))
    operation = _write_json(tmp_path / "operation-for-p1.json", dict(run=case.context.run,
        raw_build_request=request, baseline_renderer=case.saved.result["baseline_renderer"]))
    monkeypatch.setattr(analysis_preparation, "job_identity", lambda _: dict(commit="a" * 40,
        environment_lock=dict(path="requirements-worker.lock", sha256="b" * 64)))
    monkeypatch.setattr(store_map, "read_store_map", lambda _: SimpleNamespace(roots=lambda _: {"results": str(tmp_path / "local-results")}))
    calls = []
    def production(root, baselines, **kwargs):
        calls.append((baselines, kwargs))
        authority = {}
        if optical:
            import toml
            path = root / "series.toml"
            path.write_text(toml.dumps(dict(image_normalisation=dict(method="off", color_method="off"))),
                            encoding="utf-8")
            authority["current_series"] = asset(path)
        return dict(purpose="production", input_authority=authority, runs=[dict(run=case.context.run,
            counts=dict(selected_frames=3), selected_frames=[dict(raw_sha256=digit * 64) for digit in "cde"])])
    monkeypatch.setattr(current_production_selection, "current_production_selection", production)
    snapshot = dict(root=str(tmp_path / "setup"), complete=True, job_facts=dict(pipeline="series-setup"),
        runs=[dict(run=case.context.run, operation="build", state="Complete",
                   result=dict(files={"operation_result.json": operation}))])
    arguments = dict(series_root=tmp_path, series_id="test", machine="Main")
    with pytest.raises(ValueError, match="calibration or for production"):
        subject.prepare_current_build_job(snapshot, purpose="other", **arguments)
    assert not list(tmp_path.glob("build-p1-*"))
    if optical:
        with pytest.raises(ValueError, match="colour paths from Setup's image correction"):
            subject.prepare_current_build_job(snapshot, purpose="production", **arguments)
        assert not list(tmp_path.glob("build-p1-*"))  # Refused before any job folder.
        return
    registration = subject.prepare_current_build_job(snapshot, purpose="production", **arguments)
    assert calls[0][0] == {case.context.run: case.raw}  # Each run's own Build baseline.
    assert registration["name"] == "Generate production phase maps from model"
    actual = JobSnapshotReader()(registration)
    facts = actual["job_facts"]
    assert facts["purpose"] == "production" and actual["total_frames"] == 3
    assert facts["inputs"][0]["ordered_case_ids"] == [case.context.run + "-" + digit * 20 for digit in "cde"]
    saved = DataReader().json(facts["selection"])
    assert saved["input_authority"]["selected_model"] == dict(model_root="models/new")
    assert saved["purpose"] == "production" and saved["runs"][0]["counts"] == dict(selected_frames=3)


def test_any_connected_machine_needs_shared_inputs_and_ledger(local_build_sources, tmp_path, monkeypatch):
    from darsia.fluidflower import analysis_preparation, store_map
    from darsia.fluidflower.pipeline import (current_calibration_selection, current_inputs,
                                             current_model_basis, current_statistics_recipe)
    from darsia_cache import folders
    from darsia.fluidflower.jobs_metadata import JobSnapshotReader
    monkeypatch.setattr(current_model_basis, "require_current_model_basis", lambda *args, **kwargs: None)
    case = local_build_sources
    request = _write_json(tmp_path / "request-for-p1.json", dict(baseline=case.raw,
        native_product=dict(source_model=case.metric_model)))
    operation = _write_json(tmp_path / "operation-for-p1.json", dict(run=case.context.run,
        raw_build_request=request, baseline_renderer=case.saved.result["baseline_renderer"]))
    monkeypatch.setattr(analysis_preparation, "job_identity", lambda _: dict(commit="a" * 40,
        environment_lock=dict(path="requirements-worker.lock", sha256="b" * 64)))
    monkeypatch.setattr(store_map, "read_store_map", lambda _: SimpleNamespace(roots=lambda _: {"results": str(tmp_path / "local-results")}))
    monkeypatch.setattr(current_calibration_selection, "save_current_calibration_selection",
        lambda root, output, **kwargs: _write_json(output, dict(runs=[dict(run=case.context.run,
            counts=dict(selected_frames=1), selected_frames=[dict(raw_sha256="a" * 64)])])))
    checked = []
    monkeypatch.setattr(current_statistics_recipe, "retained_statistics_recipe",
                        lambda selection, run, **kwargs: checked.append((run, kwargs["reader"].shared_only)))
    snapshot = dict(root=str(tmp_path / "setup"), complete=True, job_facts=dict(pipeline="series-setup"),
        runs=[dict(run=case.context.run, operation="build", state="Complete",
                   result=dict(files={"operation_result.json": operation}))])
    arguments = dict(series_root=tmp_path, series_id="test", machine="Main")
    with pytest.raises(ValueError, match="explicitly"):
        subject.prepare_current_build_job(snapshot, any_connected_machine="yes", **arguments)
    monkeypatch.setattr(folders, "folder_mode", lambda stores, series: False)
    with pytest.raises(ValueError, match="output-space reservations"):
        subject.prepare_current_build_job(snapshot, any_connected_machine=True, **arguments)
    monkeypatch.setattr(folders, "folder_mode", lambda stores, series: True)
    monkeypatch.setattr(current_inputs, "_SHARED_FOLDERS", {})
    with pytest.raises(ValueError, match="not on a shared network folder"):  # These inputs are on this machine.
        subject.prepare_current_build_job(snapshot, any_connected_machine=True, **arguments)
    monkeypatch.setattr(current_inputs, "require_shared_input", lambda path: None)  # As on a network share.
    registration = subject.prepare_current_build_job(snapshot, any_connected_machine=True, **arguments)
    facts = JobSnapshotReader()(registration)["job_facts"]
    assert facts[subject.SHARED_RUNS] is True and facts["target_machine"] == "Main"
    assert checked == [(case.context.run, True)]  # Each run's statistics inputs were read shared-only.
