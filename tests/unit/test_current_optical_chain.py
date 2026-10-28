"""Connected two-run inference, shared optical fit and original N1 statistics.

The finite decoded-camera fixture replaces source I/O. No classifier, optical
fit, physical field reader, N1 calculation or completion writer is replaced.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from darsia.fluidflower.mass.n1_pipeline import DataReader, asset, _write_json, _write_npz
from darsia.fluidflower.phase.selected_model import load_bundle
from darsia.fluidflower.pipeline import current_build_job as job_api, series
from .test_native_optical_source import saved
from .test_local_build_inputs import _local_build_sources, local_build_case
from .test_raw_fresh_training import tiny_cohort  # noqa: F401
from .test_raw_photographic_review import inputs  # noqa: F401
from .test_raw_review_training_route import _studio_request
from .test_staged_data import phase1_data  # noqa: F401


def test_two_runs_infer_once_fit_new_optics_then_finish_real_statistics(
        tmp_path, tiny_cohort, phase1_data, monkeypatch):
    from darsia.fluidflower.phase_training import raw_fresh_training, raw_review_training
    from darsia.fluidflower.pipeline.current_model_basis import require_current_model_basis
    from darsia.fluidflower.pipeline import current_build_preparation, current_statistics_recipe
    from darsia.fluidflower.pipeline import current_clearwater, worker
    from darsia.fluidflower.phase import infer
    from darsia.fluidflower.mass import n1_pipeline
    from darsia.fluidflower.pipeline.staged_data import verify_phase1_completion
    reader = DataReader()
    request = _studio_request(tiny_cohort, tmp_path)
    training = raw_review_training.execution_preparation(asset(request["request_path"]))
    raw_fresh_training.train(training, output_root=Path(request["output_root"]),
                             cache_root=Path(request["cache_root"]))
    root = tmp_path / "series"
    root.mkdir()
    (root / "series.toml").write_text('[method]\nactive="phase_model"\n[method.phase_model]\nmodel_version="old"\n')
    selected = require_current_model_basis(root, model_reference=request["output_root"])
    cases, configs, native_arrays = {}, {}, {}
    events, reads, inferences = [], [], []
    # These are real original saved image archives, ownership receipts, facies
    # and metric models. Only acquisition decoding is a finite 3x3 input seam.
    for index, run in enumerate(("alpha", "beta")):
        path = tmp_path / run
        path.mkdir()
        native_parameters = dict(source=dict(homography_match_atol=1e-8,
            homography_min_scale=1e-12, symbols=dict(reference_to_frame_h_key="H_reference_to_frame")))
        physical = _local_build_sources(saved.__wrapped__(path, native_parameters, run), path)
        if cases:
            from darsia.fluidflower.mass.native_optical_source import load_build_physical_context
            physical.saved.adopted = next(iter(cases.values())).sources.context.provenance["adopted_revision"]
            physical.saved.plan["adopted_revision"] = physical.saved.adopted
            physical.operation_result = physical.saved.seal()
            physical.context = load_build_physical_context(physical.operation_result)
            physical.adopted = physical.saved.adopted
        case = local_build_case.__wrapped__(physical, deepcopy(phase1_data))
        config = case.configuration
        config["statistics"]["physical_correction"] = physical.operation_result
        config["frames"][0]["case_id"] = run + "-" + case.spec["source"]["source_sha256"][:20]
        config["ordered_case_ids"] = [config["frames"][0]["case_id"]]
        case.frame_receipts = {config["ordered_case_ids"][0]: next(iter(case.frame_receipts.values()))}
        case.spec = config["frames"][0]
        phase = dict(model_bundles=dict(family="fresh-native-logistic", selected=selected),
            geometry_parameters=case.geometry_parameters,
            parameters=dict(features=dict(native_decoder=dict(denominator=65535.),
                alignment=dict(homography_floor=1e-12, coverage_floor=1e-6)),
                feature_support=dict(homography_epsilon=1e-12, coverage_tolerance=1e-6, texture_footprint_hw=[1, 1]),
                geometry=dict(source_box_xywh=[0, 0, 3, 3]), domain=dict(tie_fill_contract="nearest")))
        config["phase"]["configuration"] = _write_json(path / "new-phase.json", phase)
        shared = config["statistics"]["shared"]
        shared["optics"] = None
        np.save(path / "floor.npy", np.full(3, 2., np.float32))
        shared["floor_curve"] = asset(path / "floor.npy")
        np.save(path / "clear.npy", np.ones((3, 3, 3), np.float32))
        np.save(path / "clear-valid.npy", np.ones((3, 3), bool))
        shared["clearwater"] = dict(rgb=asset(path / "clear.npy"), valid=asset(path / "clear-valid.npy"),
                                    H_reference_to_grid=np.eye(3).tolist())
        mask = _write_npz(path / "mask.npz", dict(keep=np.ones((3, 3), bool)))
        case.spec["phase_roles"].update(photo_sand_reference=dict(mask, array_key="keep"),
            photo_proof=dict(sha256=mask["sha256"], phase_labels_used=False))
        case.spec["mass_spec"].update(photo_top=dict(mask, array_key="keep"),
            original_domain=dict(membership=dict(mask, array_key="keep"), native_shape_hw=[3, 3],
                frame_index=1, H=np.eye(3).tolist()),
            inner_polygon_native=[[0, 0], [3, 0], [3, 3], [0, 3]])
        current_rgb = np.full((3, 3, 3), .6 + index * .1, np.float32)
        log = np.log(np.moveaxis(current_rgb, -1, 0) * np.float32(65535)).astype(np.float32)
        baseline_log = np.full((3, 3, 3), np.log(.3 * 65535), np.float32)
        # Actual finite source payloads consumed by the original physical loader.
        np.save(path / "baseline-log.npy", baseline_log)
        np.save(path / "baseline-valid.npy", np.ones((3, 3), bool))
        baseline = deepcopy(case.full)
        baseline.update(source_sha256=case.spec["source"]["previous_source_sha256"], native_centre=dict(
            native_shape_hw=[3, 3], log_linear_rgb=asset(path / "baseline-log.npy"), valid_mask=asset(path / "baseline-valid.npy")))
        config["phase"]["reference"]["full"] = baseline
        case.native["full_log"] = log
        case.phase_frame["full_log"] = log
        native_arrays[run] = (log, baseline_log)
        configs[run], cases[run] = config, case

    class DecodedInputs:
        def __init__(self, config, output, **kwargs):
            self.config, self.case = config, cases[config["run"]]
            self.reader = reader
            self.phase = reader.json(config["phase"]["configuration"])
            self.parameters = self.phase["parameters"]
            self.model_bundle = load_bundle(selected, reader)
            base = SimpleNamespace(log=native_arrays[config["run"]][1], valid=np.ones((3, 3), bool))
            self.model_bundle["baseline"] = dict(native_input=base)
            self.mass_baseline = config["phase"]["reference"]["full"]
            self.previous = dict(source=dict(source_sha256=self.case.spec["source"]["previous_source_sha256"]),
                current_h=np.eye(3), temporal_input=SimpleNamespace(log=base.log, valid=base.valid, origin_xy=(0, 0)))
        def native(self, spec, stage):
            reads.append(spec["source"]["run"])
            stage.mkdir(parents=True)
            log = native_arrays[self.config["run"]][0]
            np.save(stage / "native-log.npy", log)
            np.save(stage / "native-valid.npy", np.ones((3, 3), bool))
            native = deepcopy(self.case.native)
            native.update(log=log, valid=np.ones((3, 3), bool), full_valid=np.ones((3, 3), bool))
            native["record"]["audit"] = _write_json(stage / "native_audit.json", native["audit"])
            native["record"]["full"]["binding"]["producer_audit"] = native["record"]["audit"]
            native["record"]["full"]["native_centre"].update(
                log_linear_rgb=asset(stage / "native-log.npy"), valid_mask=asset(stage / "native-valid.npy"))
            native["record_asset"] = _write_json(stage / "native.json", native["record"])
            return native
        def frame(self, spec, native):
            frame = dict(self.case.phase_frame, context=native["context"], full_log=native["full_log"],
                native_input=SimpleNamespace(log=native["log"], valid=native["valid"]),
                temporal_input=SimpleNamespace(log=native["log"], valid=native["valid"], origin_xy=(0, 0), provenance={}),
                photo_sand=np.ones((3, 3), bool), observed=np.ones((3, 3), bool), scientific=native["record"],
                physical_scale_revision={}, history_policy="original")
            return frame
        def advance(self, frame):
            self.previous = frame
        def close(self):
            pass
    monkeypatch.setattr(series, "SeriesInputs", DecodedInputs)
    original_infer = infer.infer_labels
    def counted_infer(frame, *args, **kwargs):
        inferences.append(frame["source"]["run"])
        return original_infer(frame, *args, **kwargs)
    monkeypatch.setattr(infer, "infer_labels", counted_infer)
    # Input/dispatch seams carry the already checked finite source fixture;
    # all three actual job runners and every calculation remain in use.
    def sources(operation, metric, choice, output, **kwargs):
        run = reader.json(operation)["run"]
        case, config = cases[run], configs[run]
        events.append("source:" + run)
        baseline = dict(case_id="baseline")
        return dict(configuration=dict(run=run, reference_spec=baseline),
            produced={case.spec["case_id"]: SimpleNamespace(receipt=case.frame_receipts[case.spec["case_id"]]),
                "baseline": SimpleNamespace(native_task=case.spec["native_task"])},
            acquisitions={"baseline": dict(source=dict(source_binding=case.spec["source"]["source_binding"]))},
            prepared_native={}, phase_recipe={})
    monkeypatch.setattr(current_build_preparation, "prepare_current_build_acquisitions", sources)
    monkeypatch.setattr(current_statistics_recipe, "retained_statistics_recipe", lambda selection, run, **kw:
        dict(statistics=configs[run]["statistics"], clearwater_source=dict(run=run),
            flash_bounds=dict(min_value_aq=0., max_value_aq=1., min_value_g=0., max_value_g=1.)))
    monkeypatch.setattr(current_clearwater, "prepare_current_clearwater", lambda prepared, source, *a, **kw:
        dict(clearwater=configs[source["run"]]["statistics"]["shared"]["clearwater"],
             raw=cases[source["run"]].sources.raw, receipt=cases[source["run"]].sources.operation_result))
    monkeypatch.setattr(job_api, "phase_configuration", lambda prepared, selection, statistics: deepcopy(configs[prepared["configuration"]["run"]]))
    job_root = tmp_path / "job"
    job_root.mkdir()
    selection = dict(input_authority=dict(current_series=asset(root / "series.toml")),
        runs=[dict(run=run, selected_frames=[dict(raw_sha256=case.spec["source"]["source_sha256"],
        current_frame_index=1, elapsed_h=1., nominal_fit_target_h=1., fit_candidate=True)]) for run, case in cases.items()])
    job = dict(root=str(job_root), results_root=str(tmp_path / "results" / "current"),
        series_id="synthetic", series_root=str(root), job_id="current",
        selection=_write_json(job_root / "selection.json", selection), target_machine="synthetic",
        fresh_optical_paths=dict(sampling=dict(stride_px=1, interior_radius_px=0, min_samples=1, min_runs=2)),
        inputs=[dict(run=run, operation_result=case.sources.operation_result, metric_model=case.sources.metric_model)
                for run, case in cases.items()])
    def dispatch(job, commit, stores, workers, log, *, runner, publication_folder="results", **kwargs):
        for row in job["inputs"]:
            if publication_folder != "results":
                assert not (job_root / "results").exists()
                assert not (job_root / "inputs/current_optics.json").exists()
            else:
                assert (job_root / "inputs/current_optics.json").is_file()
                assert reads == ["alpha", "beta"] and inferences == ["alpha", "beta"]
            result = runner(job, row, commit, stores, lambda event:None)
            job_api._write_json(job_root / publication_folder / row["run"] / "completed.json", result)
        return 0
    monkeypatch.setattr(job_api, "_dispatch_runs", dispatch)
    monkeypatch.setattr(worker, "_job_control_boundary", lambda root:None)
    monkeypatch.setattr(worker, "emit", lambda *args:None)
    # Any old model request would now fail; newly fitted paths must be used.
    from darsia.fluidflower import store_map
    monkeypatch.setattr(store_map, "read_store_map", lambda path: SimpleNamespace(
        roots=lambda series_id:dict(results=str(tmp_path / "results"))))
    original_statistics = n1_pipeline.N1Pipeline.extract_statistics
    statistics_runs = []
    def checked_statistics(owner, spec):
        assert owner.config["shared"]["optics"] is not None
        assert "optical_paths" in owner.config["shared"]["optics"]["metadata"]["path"]
        statistics_runs.append(owner.config["run"])
        return original_statistics(owner, spec)
    monkeypatch.setattr(n1_pipeline.N1Pipeline, "extract_statistics", checked_statistics)
    assert job_api._run_fresh_optical_job(job, "a" * 40, None, 2, None) == 0
    assert reads == ["alpha", "beta"] and inferences == ["alpha", "beta"]
    assert statistics_runs == ["alpha", "beta"]
    assert events == ["source:alpha", "source:beta"]
    assert (job_root / "inputs/current_optics.json").is_file()
    for run in cases:
        result = reader.json(asset(job_root / "results" / run / "completed.json"))
        verified = verify_phase1_completion(result["phase1_completion"], result["configuration"],
            expected_commit="a" * 40, reader=reader)
        assert result["complete"] and len(verified["records"]) == 1
