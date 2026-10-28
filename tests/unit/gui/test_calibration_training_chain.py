"""One GUI start through real tiny training, P1 numerics and mass-job creation.

RAW decoding and host launching are finite input/transport seams. The trained
heads, original P1 frame loop, statistics/probability writers, completion reader
and mass-job producer run normally; no real dataset or job is touched.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QTextEdit, QLabel, QPushButton, QTabWidget, QCheckBox, QLineEdit

from darsia.fluidflower import calibration_preparation as preparation
from darsia.fluidflower import refit_job_panel as refit
from darsia.fluidflower.mass.n1_pipeline import DataReader, asset, _write_json
from darsia.fluidflower.phase.selected_model import load_bundle
from darsia.fluidflower.pipeline import current_build_job, phase_statistics, series
from darsia.fluidflower.pipeline.current_model_basis import require_current_model_basis
from darsia.fluidflower.ui.model_training import SharedPhaseModelTrainingPage
from darsia.fluidflower.native_phase_steps import NativePhaseSteps
from ..test_raw_fresh_training import tiny_cohort  # noqa: F401
from ..test_raw_photographic_review import inputs  # noqa: F401
from ..test_raw_review_training_route import _studio_request
from ..test_staged_data import phase1_data  # noqa: F401
from ...fixtures.tiny_saved_model_fixture import tiny_config  # noqa: F401
from .test_refit_job_panel import _rules


def _prepared_cohort_build(cohort, build, output):
    """Bind finite reviewed pixels to the matching finite Build metadata.

    The fixture's original review used one generic processor for both runs.
    A prepared series instead records each actual run/baseline and Build grid.
    No display or scientific pixel array is replaced here.
    """
    reader = DataReader()
    combined = reader.json(cohort.prepare_kwargs["bundles"][0]["combined_manifest"])
    operations = {row["run"]: row for row in build["runs"]}
    manifests = []
    for job in combined["jobs"]:
        run = job["run"]
        (output / run).mkdir(parents=True)
        old_path = Path(job["review_output_dir"]) / "native_frames" / run / "manifest.json"
        manifest = json.loads(old_path.read_text())
        bound = reader.json(manifest["raw_binding"])
        bound.update(run=run, baseline_raw=deepcopy(manifest["reference"]["acquisition"]),
                     native_shape_hw=manifest["native_shape_hw"])
        binding = _write_json(output / run / "raw_binding.json", bound)
        audit = reader.json(manifest["reference"]["audit"])
        renderer = deepcopy(audit["renderer"])
        height, width = manifest["native_shape_hw"]
        renderer.update(profile_id=manifest["reference"]["profile_id"], camera_u_wh=[width, height],
            shape_corrections=deepcopy(bound.get("shape_assets", [])),
            source_grid=dict(num_voxels=[height, width], voxel_size=[.01, .01],
                metadata=dict(space_dim=2, indexing="ij", dimensions=[height*.01, width*.01], origin=[0., height*.01])))
        certificate = _write_json(output / run / "renderer.json", dict(renderer=renderer,
            inputs=dict(raw=bound["baseline_raw"], profile=manifest["native_configuration"]),
            H_rig_corrected_native_to_baseline_camera_u=renderer["native_to_camera_u_h"]))
        original = reader.json(operations[run]["result"]["files"]["operation_result.json"])
        request = reader.json(original["raw_build_request"])
        request["baseline"] = deepcopy(bound["baseline_raw"])
        measured = reader.json(request["native_product"]["source_model"])
        measured["reference_image"] = deepcopy(bound["baseline_raw"])
        request["native_product"]["source_model"] = _write_json(output / run / "measured.json", measured)
        request = _write_json(output / run / "request.json", request)
        original.update(operation="build", complete=True, baseline_renderer=certificate, raw_build_request=request)
        operation = _write_json(output / run / "operation_result.json", original)
        operations[run]["result"]["files"]["operation_result.json"] = operation
        preparation = _write_json(output / run / "prepared_sources.json", dict(
            format="darsia-current-build-source-preparation-v1", run=run,
            operation_result=operation, raw_binding=binding, frames={}))
        manifest.update(raw_binding=binding, current_build_preparation=preparation)
        path = output / "review/native_frames" / run / "manifest.json"
        path.parent.mkdir(parents=True)
        pin = _write_json(path, manifest)
        manifests.append(pin)
        job.update(review_output_dir=str(output / "review"), native_manifest_sha256=pin["sha256"])
    cohort.prepare_kwargs["bundles"][0]["combined_manifest"] = _write_json(output / "combined.json", combined)
    cohort.manifests = manifests


@pytest.mark.parametrize("method", ["off", "match_reference_lighting"])
def test_fresh_training_admits_exact_full_photo_correction_before_crop_choice(tmp_path, method):
    from darsia.fluidflower.image_normalisation import save_choice, saved_choice
    from darsia.fluidflower.calibration_model_training import require_prepared_normalisation
    from darsia.fluidflower.native.full_photometry import ORDER
    from darsia.fluidflower.phase_training.raw_review_training import FAMILY
    import tomllib
    path = tmp_path / "series.toml"
    path.write_text('series_id="synthetic"\n[runs]\nids=["one", "two"]\n')
    save_choice(tmp_path, "synthetic", ["one", "two"], method,
        "two" if method != "off" else None, expected_bytes=path.read_bytes())
    document = tomllib.loads(path.read_text())
    choice = saved_choice(document, "synthetic", ["one", "two"])
    record = _write_json(tmp_path / "normalisation.json", dict(
        format="darsia-prepared-image-normalisation-v1", choice=choice))
    binding = _write_json(tmp_path / "raw_binding.json", dict(correction_order=ORDER,
        normalisation=dict(choice=choice, record=record)))
    request = dict(settings=dict(training_family=FAMILY,
        raw_image_basis=dict(runs=[dict(run="one", raw_binding=binding), dict(run="two", raw_binding=binding)])))
    assert require_prepared_normalisation(request, tmp_path) == choice
    before = path.read_bytes()
    wrong = deepcopy(choice)
    wrong["method"] = "off" if method != "off" else "match_reference_lighting"
    changed = _write_json(tmp_path / "other_binding.json", dict(correction_order=ORDER,
        normalisation=dict(choice=wrong, record=record)))
    request["settings"]["raw_image_basis"]["runs"][0]["raw_binding"] = changed
    with pytest.raises(ValueError, match="another saved image-normalisation"):
        require_prepared_normalisation(request, tmp_path)
    assert path.read_bytes() == before


class _Studio(QWidget):
    training_completed = Signal(str)
    _launch_training_job = SharedPhaseModelTrainingPage._launch_training_job
    _read_output = SharedPhaseModelTrainingPage._read_output
    _training_finished = SharedPhaseModelTrainingPage._training_finished
    _training_error = SharedPhaseModelTrainingPage._training_error


def test_saved_26_run_1039_image_choice_is_not_replaced_by_production_grid(tmp_path):
    import toml
    runs = ["sample" + str(index) for index in range(39)]
    chosen = runs[:26]
    calibration = _write_json(tmp_path / "calibration.json", dict(calibration_runs=chosen))
    document = dict(series_id="synthetic", runs=dict(ids=runs),
        method=dict(active="phase_model", phase_model=dict(mass_calibration=calibration["path"])),
        image_selection=dict(runs={run: dict(selected=[dict(frame_index=i) for i in range(
            40 if run == chosen[-1] else 41 if run in chosen else 90)]) for run in runs}),
        production_image_selection=dict(targets=[1, 2, 3], purpose="production"))
    path = tmp_path / "series.toml"
    path.write_text(toml.dumps(document), encoding="utf-8")
    before = path.read_bytes()
    preview = preparation.selection_preview(tmp_path)
    assert [row["run"] for row in preview["runs"]] == runs
    assert preview["mass_fit_runs"] == chosen
    assert preview["counts"]["selected_frames"] == 1065 + 13 * 90
    fit = [row for row in preview["display_runs"] if row["role"] == "Fit"]
    assert sum(row["selected_images"] for row in fit) == 1039
    assert sum(row["baselines"] for row in fit) == 26
    assert len(preview["display_runs"]) == 39
    assert path.read_bytes() == before


def test_calibration_preview_exposes_time_coverage_and_control_roles(tmp_path):
    import toml
    baseline = _write_json(tmp_path / "calibration.json", dict(calibration_runs=["fit"],
        heldout_runs=["control"], descriptive_runs=["descriptive"], excluded_runs=["excluded"]))
    rows = [dict(frame_index=index, elapsed_h=time) for index, time in enumerate((-.25, .5, 48.))]
    document = dict(series_id="synthetic", runs=dict(ids=["fit", "control", "descriptive", "excluded"]),
        method=dict(active="phase_model", phase_model=dict(mass_calibration=baseline["path"])),
        image_selection=dict(runs={run: dict(selected=rows) for run in ("fit", "control", "descriptive")}))
    (tmp_path / "series.toml").write_text(toml.dumps(document), encoding="utf-8")
    view = preparation.selection_preview(tmp_path)
    assert [row["run"] for row in view["runs"]] == ["fit", "control", "descriptive"]
    assert view["mass_fit_runs"] == ["fit"]
    roles = {row["run"]: row for row in view["display_runs"]}
    assert roles["fit"]["time_range_h"] == [-.25, 48.]
    assert roles["fit"]["selected_images"] == 2 and roles["fit"]["baselines"] == 1
    assert roles["control"]["role"] == "Control / validation"
    assert roles["descriptive"]["role"] == "Descriptive / control"
    assert roles["excluded"]["role"] == "Excluded"


def _studio(qtbot, root):
    studio = _Studio()
    qtbot.addWidget(studio)
    studio.training_process = studio.training_request = None
    studio.bundles, studio.log = QTextEdit(), QTextEdit()
    studio.status, studio.stop_button = QLabel(), QPushButton()
    studio.reviewer, studio.history_toggle = QLineEdit("reviewer"), QCheckBox()
    studio.tabs, studio.evidence_tab = QTabWidget(), QWidget()
    studio.tabs.addTab(studio.evidence_tab, "Evidence")
    studio._update_action_states = lambda: None
    studio._prepare_post_training_views = lambda *a: None
    studio._record_completed_model = lambda *a: None
    studio._select_completed_model = lambda: None
    steps = SimpleNamespace(studio=studio, _ready=True, series_root=root,
        model_name=QLineEdit(), _named_model_context={}, refresh=lambda: None)
    steps.phase_workflow = SimpleNamespace(run_training=lambda call, **_options: call(), before_launch=lambda request: None)
    steps._original_launch = studio._launch_training_job
    studio._launch_training_job = MethodType(NativePhaseSteps._launch_named_job, steps)
    return SimpleNamespace(studio=studio, native_steps=steps)


def _tiny_p1(job_root, selection, binding, numerical, specimen, monkeypatch):
    """Execute the unmodified P1 loop on an explicit 3x3 decoded-source seam."""
    from darsia.fluidflower.phase.probability_storage import parse_request
    from darsia.fluidflower.geometry.context import RigFrameContext
    from darsia.fluidflower.native.sidecar import write_log_linear
    from darsia.fluidflower.mass.n1_pipeline import _write_npz
    shape, eye = (3, 3), np.eye(3)
    reader = DataReader()
    phase_pin = _write_json(job_root / "selected-model.json", dict(model_bundles=dict(
        family="fresh-native-logistic", selected=binding)))
    bundle = load_bundle(binding, reader)
    valid = np.ones(shape, bool)
    log = np.full((3, *shape), np.log(.3 * 65535), np.float32)
    base = SimpleNamespace(log=log, valid=valid)
    bundle["baseline"] = dict(native_input=base)
    parameters = dict(features=dict(native_decoder=dict(denominator=65535.),
        alignment=dict(homography_floor=1e-12, coverage_floor=1e-6)),
        feature_support=dict(homography_epsilon=1e-12, coverage_tolerance=1e-6, texture_footprint_hw=[1, 1]),
        geometry=dict(source_box_xywh=[0, 0, 3, 3]), domain=dict(tie_fill_contract="nearest"))
    context = RigFrameContext(rig_domain=valid.copy(), modelled=valid.copy(),
        filled_observed=np.zeros(shape, bool), filled_unobserved=np.zeros(shape, bool),
        hardware_excluded=np.zeros(shape, bool), excluded_by_name={}, origin_xy=(0, 0),
        native_shape_hw=shape, content_box_xywh=(0, 0, 3, 3),
        pixel_area_mm2=np.ones(shape), parameters={})

    class DecodedInputs:
        def __init__(self, config, output, **kwargs):
            self.reader, self.phase = reader, dict(model_bundles=dict(family="fresh-native-logistic", selected=binding))
            self.model_bundle, self.parameters, self.previous = bundle, parameters, None
            self.mass_baseline = numerical["frames"][0]["baseline"]
        def native(self, spec, stage):
            stage.mkdir(parents=True)
            values = deepcopy(numerical["frames"][0]["current"])
            values.update(complete=True, source_sha256=spec["source"]["source_sha256"])
            write_log_linear(stage / "native_log.npy", log)
            write_log_linear(stage / "native_valid.npy", valid)
            values["native_centre"].update(log_linear_rgb=asset(stage / "native_log.npy"),
                valid_mask=asset(stage / "native_valid.npy"))
            audit = _write_json(stage / "native_audit.json", dict(source=spec["source"], fixture="decoded source"))
            record = dict(complete=True, full=values, audit=audit,
                processor_kind="raw_rig", source_sha256=spec["source"]["source_sha256"])
            return dict(record=record, record_asset=_write_json(stage / "native.json", record))
        def frame(self, spec, native):
            temporal = SimpleNamespace(log=log, valid=valid, origin_xy=(0, 0), provenance={})
            return dict(native_input=base, temporal_input=temporal, source=spec["source"], current_h=eye,
                photo_sand=valid, context=context, observed=valid, physical_scale_revision={},
                history_policy="original", pixel_area_mm2=np.ones(shape), scientific=dict(full=native["record"]["full"]))
        def advance(self, frame):
            self.previous = frame
        def close(self):
            pass
    monkeypatch.setattr(series, "SeriesInputs", DecodedInputs)
    for run in (row["run"] for row in selection["runs"]):
        stats = deepcopy(numerical)
        stats.update(run=run, geometry_fields=dict(source_to_native="H_ac28_to_frame_native",
            original_domain_to_native="H_ac28_to_frame_native", carried_homography_provenance="original_H"))
        frames = []
        for index, original in enumerate(stats.pop("frames")):
            spec = deepcopy(specimen["frames"][0])
            spec.update(case_id=run + "-" + str(index), frame_index=index + 1,
                elapsed_h=original["elapsed_h"], native_task=dict(kind="raw_rig"))
            spec["source"].update(run=run, source_sha256=str(index+1)*64,
                previous_source_sha256=None if index == 0 else str(index)*64)
            spec["mass_spec"] = original
            frames.append(spec)
        config = dict(run=run, frames=frames, precision="float32", statistics=stats,
            phase=dict(configuration=phase_pin), ordered_case_ids=[r["case_id"] for r in frames],
            gas_probability_storage=dict(format="darsia-decision-probability-storage-request-v1",
                encoding="float32-exact-decision-values"))
        output = job_root / "numerical" / run
        output.mkdir(parents=True)
        pin = _write_json(output / "phase1_configuration.json", config)
        manifest = _write_json(output / "manifest.json", dict(configuration=pin))
        task = dict(run=run, precision="float32", commit="a"*40, execution_id="tiny")
        result = phase_statistics._phase1_frames(config, task, output, "cpu", "float32", None,
            pin, manifest, 0., parse_request(config), unc_only=False)
        current_build_job._write_json(job_root / "results" / run / "completed.json", result)


def test_one_gui_start_trains_then_writes_real_p1_and_creates_overflow_mass_job(
        tiny_cohort, tiny_config, phase1_data, qtbot, tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import current_calibration_selection, local_overflow_refit
    from darsia.fluidflower import analysis_preparation, store_map
    from darsia.fluidflower.jobs_metadata import JobSnapshotReader
    root = tmp_path / "series"
    root.mkdir()
    (root / "series.toml").write_text('schema_version=1\nseries_id="synthetic"\n'
        '[method]\nactive="phase_model"\n[method.phase_model]\nmodel_version="old"\n')
    counts = dict(selected_frames=4, fit_candidates=4, earlier_history=0)
    saved = dict(input_authority=dict(current_series=asset(root / "series.toml")), counts=counts,
        runs=[dict(run=run, counts=dict(selected_frames=2, fit_candidates=2, earlier_history=0), selected_frames=[dict(raw_sha256=str(n)*64)
              for n in (1, 2)]) for run in tiny_cohort.runs])
    before = (root / "series.toml").read_bytes()
    build = dict(root=str(tmp_path / "build"), complete=True, errors=[], job_facts=dict(
        pipeline="series-setup", series_id="synthetic", series=saved["input_authority"]["current_series"]), runs=[])
    for index, run in enumerate(tiny_cohort.runs):
        raw = _write_json(tmp_path / (run + "-source.json"), dict(finite_source=run))
        metric = _write_json(tmp_path / (run + "-model.json"), dict(run=run, reference_image=raw, baseline_native_grid={}))
        request = _write_json(tmp_path / (run + "-build.json"), dict(baseline=raw, native_product=dict(source_model=metric)))
        renderer = _write_json(tmp_path / (run + "-renderer.json"), dict(
            H_rig_corrected_native_to_baseline_camera_u=np.eye(3).tolist(), renderer=dict(native_shape_hw=[3, 3])))
        operation = _write_json(tmp_path / (run + "-operation.json"), dict(run=run, raw_build_request=request, baseline_renderer=renderer))
        build["runs"].append(dict(run=run, operation="build", state="Complete", result=dict(files={"operation_result.json": operation})))
    _prepared_cohort_build(tiny_cohort, build, tmp_path / "prepared-cohort")
    request = _studio_request(tiny_cohort, tmp_path)
    pairs = [(row["decisions"]["path"], row["combined_manifest"]["path"])
             for row in tiny_cohort.prepare_kwargs["bundles"]]
    monkeypatch.setattr(preparation, "prepare_calibration_training", lambda *a, **kw: dict(request=request, bundles=pairs))
    monkeypatch.setattr(current_calibration_selection, "current_calibration_selection", lambda *a, **kw: deepcopy(saved))
    def save(root, destination, **kwargs):
        value = deepcopy(saved)
        value["input_authority"]["selected_model"] = kwargs["model_binding"]
        return _write_json(destination, value)
    monkeypatch.setattr(current_calibration_selection, "save_current_calibration_selection", save)
    # This tiny fixture has no project/GeometryRevisionStore. Its finite Build
    # admission accepts only the model-reference save, with every Setup value
    # unchanged; real Setup authority/geometry admission has separate tests.
    from darsia.fluidflower.pipeline import current_training_preparation
    import tomllib
    def same_build_basis(_root, snapshot, _selection, _reader):
        prior = tomllib.loads(before.decode())
        current_manifest = tomllib.loads((root / "series.toml").read_text())
        prior.pop("method"); current_manifest.pop("method")
        return snapshot is build and prior == current_manifest
    monkeypatch.setattr(current_training_preparation, "_same_build_basis", same_build_basis)
    monkeypatch.setattr(preparation, "selection_preview", lambda *a, **kw: dict(deepcopy(saved), model_basis_error=preparation.UNBOUND_MODEL_REASON))
    monkeypatch.setattr(analysis_preparation, "job_identity", lambda *a: dict(commit="a"*40, environment_lock=dict(path="lock", sha256="b"*64)))
    monkeypatch.setattr(store_map, "read_store_map", lambda *a: SimpleNamespace(roots=lambda *a: dict(results=str(tmp_path / "results"))))
    real_reader = JobSnapshotReader()
    monkeypatch.setattr(preparation, "JobSnapshotReader", JobSnapshotReader, raising=False)
    original_prepare = preparation.prepare_calibration_phase
    def prepare_phase(*args, **kwargs):
        return original_prepare(*args, **kwargs, snapshot_reader=lambda row:
            build if row["root"] == build["root"] else real_reader(row))
    monkeypatch.setattr(preparation, "prepare_calibration_phase", prepare_phase)
    panel = refit.RefitJobPanel(machine="test")
    qtbot.addWidget(panel)
    panel.series_root = root
    panel.roles = dict(fit=list(tiny_cohort.runs), heldout=[], descriptive=[], excluded=[])
    panel.rules = _rules()
    rules_file = _write_json(tmp_path / "rules.json", panel.rules)
    panel.rules_edit.setText(rules_file["path"])
    panel.baseline = _write_json(tmp_path / "baseline-calibration.json", dict(parameters=panel.rules["baseline"]))
    panel.jobs_edit.setText(str(tmp_path / "mass-jobs"))
    panel.name_edit.setText("tiny-mass-fit")
    panel.loro_check.setChecked(False)
    overflow = _write_json(tmp_path / "overflow.json", dict(finite_reviewed_source=True))
    panel.overflow_inputs = overflow
    panel.f_check.setChecked(True)
    # Source method admission has its own literal-workbook tests. Keep its
    # chosen pin; exercise the real probability budget and job producer here.
    monkeypatch.setattr(local_overflow_refit, "read_request", lambda *a, **kw: {})
    starts = []
    monkeypatch.setattr(panel, "start", lambda **kw: starts.append(kw) or True)
    page = _studio(qtbot, root)
    registrations = [dict(root=build["root"])]
    current = SimpleNamespace(current=None, local_machine="test")
    current._show_snapshot = lambda value: setattr(current, "current", value)
    shared = SimpleNamespace(automatic=SimpleNamespace(isChecked=lambda: False),
        cpu=SimpleNamespace(value=lambda: 6), _thread=None, status=QLabel())
    def start_phase():
        phase_root = Path(current.current["root"])
        binding = require_current_model_basis(root, saved, model_reference=request["output_root"])
        _tiny_p1(phase_root, saved, binding, tiny_config[0], phase1_data, monkeypatch)
        shared._thread = object()  # Existing host transport was started.
    shared.start_selected_job = start_phase
    jobs = SimpleNamespace(series_id="synthetic", series_root=root,
        registry=SimpleNamespace(jobs=lambda *a: registrations, set_active=lambda *a: None),
        panel=current, stores=SimpleNamespace(path=None), shared_jobs=shared,
        _job_created=lambda row: registrations.append(row) or True)
    window = SimpleNamespace(ensure_jobs=lambda: jobs, open_image_selection_plan=lambda: None,
        _select_row=lambda *a: None, show_phase_model_workflow=lambda: None,
        phase_model_workspace=SimpleNamespace(ensure_studio=lambda: page))
    from darsia.fluidflower.series_method import read_series_method, save_series_method
    from dataclasses import replace
    window._method_manifest_bytes = before
    window._saved_method = read_series_method(root)
    def publish(root, reference, *, expected_bytes, expected_selection, finished):
        chosen = replace(expected_selection, model_version=reference, mass_calibration="")
        window._saved_method = save_series_method(root, chosen,
            expected_bytes=expected_bytes, prepared_model=True)
        window._method_manifest_bytes = (root / "series.toml").read_bytes()
        saved["input_authority"]["current_series"] = asset(root / "series.toml")
        finished(window._saved_method == chosen, "")
    window._publish_prepared_model = publish
    widget = preparation.CalibrationPreparation(window, panel)
    qtbot.addWidget(widget)
    widget.set_series(root)
    qtbot.waitUntil(lambda: not widget._preview.busy)
    assert widget.start.isEnabled(), widget.start.toolTip()
    widget.start.click()
    try:
        qtbot.waitUntil(lambda: bool(starts) or (widget._thread is None and page.studio.training_process is None
            and any(text in widget.status.text().lower() for text in ("failed", "error:", "unavailable"))), timeout=60000)
    except Exception:
        pytest.fail(widget.status.text() + "\n" + panel.job_label.text() + "\n" + page.studio.log.toPlainText())
    assert starts == [dict(confirm=False)], widget.status.text() + panel.job_label.text() + page.studio.log.toPlainText()
    mass_job = DataReader().json(asset(tmp_path / "mass-jobs" / "tiny-mass-fit" / "job.json"))
    assert mass_job["overflow"] == overflow and mass_job["expected_runs"] == ["refit"]
    plan = DataReader().json(mass_job["plans"]["refit"])
    assert plan["runs"]["fit"] == list(tiny_cohort.runs)
    assert all(len(rows) == 2 for rows in plan["statistics"].values())
    assert read_series_method(root).model_version == str(Path(request["output_root"]).resolve())
    assert read_series_method(root).mass_calibration == ""
    assert widget._candidate_reference == str(Path(request["output_root"]).resolve())
    panel._timer.stop(); panel._phase_wait_timer.stop()


def test_independent_corrections_materialize_train_and_run_p1_without_old_fits(
        tiny_cohort, tiny_config, phase1_data, tmp_path, monkeypatch):
    """The current coordinator and review writer feed real fitting and P1.

    Camera decoding and completed Build fixtures supply finite arrays. All
    correction math, annotation reprojection, saved review/request production,
    model fitting and P1 calculations below are the real implementations.
    """
    import toml
    from darsia.fluidflower import image_normalisation, raw_photographic_review, series_native_context
    from darsia.fluidflower.mass.build_native_bridge import BuildNativeFrame
    from darsia.fluidflower.mass.n1_pipeline import _write_npz
    from darsia.fluidflower.native import full_photometry, photometry, raw_positioning
    from darsia.fluidflower.native.raw_rig_source import RawRigSourceProcessor
    from darsia.fluidflower.phase_training import raw_fresh_training, raw_review_training
    from darsia.fluidflower.pipeline import current_build_preparation, current_calibration_selection
    from darsia.fluidflower.pipeline import current_image_preparation, current_training_preparation

    reader = DataReader()
    root = tmp_path / "independent-series"
    root.mkdir()
    runs = tiny_cohort.runs
    old_manifests = {reader.json(pin)["run"]: reader.json(pin) for pin in tiny_cohort.manifests}
    old_sources, cameras = {}, {}
    for run, manifest in old_manifests.items():
        reference = manifest["reference"]
        old_sources[run] = [reader.json(reference["acquisition_manifest"])["records"][0],
                           *[reader.json(frame["native_record"]) for frame in manifest["frames"]]]
        for record in old_sources[run]:
            audit_pin = record.get("audit") or reader.json(record["native_row"])["raw_producer_audit"]
            audit = reader.json(audit_pin)
            raw = dict(audit["sources"]["camera_arw"])
            raw["size_bytes"] = Path(raw["path"]).stat().st_size
            # This is the finite camera-decoding seam: genuine corrected pixel
            # arrays are produced below, never relabelled from old review PNGs.
            camera = reader.array(record["full"]["native_centre"]["log_linear_rgb"])
            cameras[raw["sha256"]] = (raw, np.exp(camera).astype(np.float32)
                * (np.float32(.8) if run == runs[1] else np.float32(1)))

    parameters = deepcopy(reader.json(tiny_cohort.configuration))
    photo_parameters = json.loads((Path(__file__).parents[2] /
        "fluidflower/fixtures/native_photometry_parameters.json").read_bytes())
    photo_parameters.update(fit_observed_field_int_1000=20, native_area_log_int_8=2,
                            minimum_sensor_signal=0.)
    parameters.update(photometry=photo_parameters, source=dict(asset_aliases={},
        profiles={run: "late" for run in runs}, native_shape_hw=[10, 16],
        source_id_pattern="[a-z]+", homography_min_scale=1e-12,
        homography_match_atol=1e-10, symbols=dict(reference_to_frame_h_key="H_ac28_to_frame_native")))
    parameters["support"]["coverage_tolerance"] = 1e-6
    parameters["registration"] = dict(common=dict(sensor=dict(camera_wh=[16, 10], raw_wh=[16, 10]),
        lens=dict(ca_red=[], ca_blue=[]), profiles=dict(late=dict(distortion=[]))))
    native = _write_json(root / "native.json", parameters)
    # Add the literal camera certificate fields absent from the small review
    # rendering fixture. Projection still checks the actual old/new matrices.
    evidence_root = root / "source-review"
    old_template = deepcopy(tiny_cohort.prepare_kwargs["bundles"])
    combined = reader.json(old_template[0]["combined_manifest"])
    for job in combined["jobs"]:
        run = job["run"]
        manifest = deepcopy(old_manifests[run])
        manifest["native_configuration"] = native
        for frame in manifest["frames"]:
            audit = reader.json(frame["audit"])
            audit["renderer"].update(camera_jpeg_wh=[16, 10], raw_visible_wh=[16, 10],
                sony_distortion_table=[], sony_ca_red_table=[], sony_ca_blue_table=[])
            audit_pin = _write_json(root / (run + "-old-audit-" + str(frame["frame_index"]) + ".json"), audit)
            record = reader.json(frame["native_record"])
            record["audit"] = audit_pin
            frame["audit"] = audit_pin
            frame["native_record"] = _write_json(root / (run + "-old-record-" + str(frame["frame_index"]) + ".json"), record)
        (evidence_root / "native_frames" / run).mkdir(parents=True)
        pin = _write_json(evidence_root / "native_frames" / run / "manifest.json", manifest)
        old_manifests[run] = manifest
        job.update(review_output_dir=str(evidence_root), native_manifest_sha256=pin["sha256"])
    old_template[0]["combined_manifest"] = _write_json(root / "old-combined.json", combined)
    settings = tiny_cohort.prepare_kwargs["settings"]
    template = dict(root=str(tmp_path / "old-model"), request=tiny_cohort.preparation,
                    settings=settings, bundles=old_template)
    image_rows = {}
    selection_rows = []
    for run in runs:
        manifest = old_manifests[run]
        sources = [cameras[reader.json((record.get("audit") or
            reader.json(record["native_row"])["raw_producer_audit"]))["sources"]["camera_arw"]["sha256"]][0]
            for record in old_sources[run]]
        image_rows[run] = dict(selected=[dict(source_path=raw["path"], elapsed_h=index * .5 - .5)
                                         for index, raw in enumerate(sources)])
        def source_row(raw, index):
            return dict(raw_path=raw["path"], raw_sha256=raw["sha256"],
                recorded_raw_size_bytes=raw["size_bytes"], current_frame_index=index,
                elapsed_h=index * .5 - .5, baseline=index == 0,
                previous_raw_sha256=None if index == 0 else sources[index-1]["sha256"])
        selection_rows.append(dict(run=run, baseline=source_row(sources[0], 0),
            initial_previous_raw_sha256=sources[0]["sha256"],
            selected_frames=[source_row(raw, index) for index, raw in enumerate(sources[1:], 1)]))
    series_path = root / "series.toml"
    series_path.write_text(toml.dumps(dict(schema_version=1, series_id="synthetic",
        method=dict(active="phase_model", phase_model=dict(model_version="old")),
        runs=dict(ids=list(runs)), image_selection=dict(runs=image_rows))))
    image_normalisation.save_choice(root, "synthetic", runs, "match_reference_lighting", runs[0],
        expected_bytes=series_path.read_bytes(), color_method="off", stable_band_mm=2.)
    manifest = toml.loads(series_path.read_text())
    choice = image_normalisation.saved_choice(manifest, "synthetic", runs)
    selection = dict(format="ffac-local-p1-selection-data-v1", decision=dict(actor="reviewer", reason="GUI start"),
        input_authority=dict(current_series=asset(series_path), phase_plan=tiny_cohort.preparation),
        runs=selection_rows)
    operations, contexts = {}, {}
    for row in selection_rows:
        run = row["run"]
        raw = cameras[row["baseline"]["raw_sha256"]][0]
        request = _write_json(root / (run + "-request.json"), dict(baseline=raw, profile=native))
        renderer = _write_json(root / (run + "-renderer.json"), dict(inputs=dict(raw=raw, profile=native),
            renderer=dict(profile_id="late", native_shape_hw=[10, 16], camera_u_wh=[16, 10],
                shape_corrections=[native], native_to_camera_u_h=np.eye(3).tolist(),
                source_grid=dict(num_voxels=[10, 16], voxel_size=[.01, .01], metadata=dict(
                    space_dim=2, indexing="ij", dimensions=[.1, .16], origin=[0., .1])))))
        operations[run] = _write_json(root / (run + "-build.json"), dict(run=run, operation="build", complete=True,
            raw_build_request=request, baseline_renderer=renderer))
        contexts[run] = SimpleNamespace(run=run, labels=np.ones((10, 16), np.uint8),
            grid=dict(num_voxels=[10, 16]), provenance=dict(raw_acquisition=raw,
            operation_result=operations[run], native_configuration=native,
            baseline_renderer=renderer, camera_binding=dict(profile_id="late")))
    snapshot = dict(root=str(root / "build"), complete=True, errors=[], job_facts=dict(
        pipeline="series-setup", series_id="synthetic", series=asset(series_path)),
        runs=[dict(run=run, operation="build", state="Complete", result=dict(
            files={"operation_result.json": operation})) for run, operation in operations.items()])
    monkeypatch.setattr(current_calibration_selection, "current_calibration_selection", lambda *a, **kw: deepcopy(selection))
    monkeypatch.setattr(current_training_preparation, "selected_training_template", lambda *a, **kw: deepcopy(template))
    monkeypatch.setattr(series_native_context, "read_series_native_binding", lambda *a: SimpleNamespace(
        path=Path(native["path"]), sha256=native["sha256"]))
    monkeypatch.setattr(current_build_job, "_metric_for_build", lambda op, target, reader:
        _write_json(target, dict(area_factor=1.)))
    monkeypatch.setattr(current_image_preparation, "load_image_context",
        lambda operation, **kw: contexts[reader.json(operation)["run"]])
    monkeypatch.setattr(current_image_preparation, "stable_support", lambda context, *a, **kw:
        dict(mask=np.ones((10, 16), bool), width_mm=2.))
    def collect(context, *args, **kwargs):
        pixels = cameras[context.provenance["raw_acquisition"]["sha256"]][1]
        return dict(run=context.run, rgb_full=pixels,
            rgb_native=np.moveaxis(pixels / np.float32(65535), 0, -1),
            native_to_camera_u_h=np.eye(3), white_range_dn=65535., chart_rgb=None,
            chart_roi_rc=None, source_pins=deepcopy(context.provenance), parameters=parameters,
            shape_assets=[native], image_metadata={})
    monkeypatch.setattr(full_photometry, "collect_baseline", collect)
    monkeypatch.setattr(photometry, "load_baseline", lambda *a, **kw: pytest.fail("read an old native/donor fit"))
    monkeypatch.setattr(raw_positioning, "read_sensor", lambda path, **kw: SimpleNamespace(
        source=cameras[kw["expected_sha256"]][0], profile_id="late", distortion=(), ca_red=(), ca_blue=(),
        metadata=dict(white_level=65535, black_level=0, raw_visible_wh=[16, 10])))
    monkeypatch.setattr(raw_positioning, "camera_u", lambda sensor, **kw: cameras[sensor.source["sha256"]][1])
    monkeypatch.setattr(raw_positioning, "load_shape_chain", lambda *a, **kw: [])
    monkeypatch.setattr(raw_positioning, "render_native", lambda *a, **kw: (None,
        dict(native_to_camera_u_h=np.eye(3).tolist(), camera_u_wh=[16, 10])))
    monkeypatch.setattr(raw_photographic_review, "render_original", lambda sources, **kw:
        np.zeros((14, 24, 3), np.uint8))  # Display-only, never training pixels.
    corrections = []
    def materialize(operation, metric, selected_pin, output, review, *, prepared_photometry, **kwargs):
        output.mkdir(parents=True)
        run = reader.json(operation)["run"]
        selected = reader.json(selected_pin)["runs"][0]
        processor = RawRigSourceProcessor(prepared_photometry["source_binding"]["path"], parameters=parameters)
        produced, records, native_records, frames = {}, {}, {}, []
        for index, row in enumerate([selected["baseline"], *selected["selected_frames"]]):
            raw = cameras[row["raw_sha256"]][0]
            case = run + "-" + raw["sha256"][:20]
            linear, audit = processor.correct(dict(contract="darsia-raw-native-frame-v1", run=run,
                acquisition=raw, reference_to_native_h=np.eye(3).tolist(), baseline_to_native_h=np.eye(3).tolist()))
            corrections.append(audit)
            audit["renderer"].update(source_kind="raw", profile_id="late", native_shape_hw=[10, 16],
                camera_jpeg_wh=[16, 10], raw_visible_wh=[16, 10],
                sony_distortion_table=[], sony_ca_red_table=[], sony_ca_blue_table=[])
            folder = output / case
            folder.mkdir()
            log_path, valid_path = folder / "log.npy", folder / "valid.npy"
            np.save(log_path, linear); np.save(valid_path, np.isfinite(linear).all(0))
            audit_pin = _write_json(folder / "audit.json", audit)
            full = deepcopy(old_sources[run][index]["full"])
            full.update(source_sha256=raw["sha256"], native_centre=dict(native_shape_hw=[10, 16],
                    log_linear_rgb=asset(log_path), valid_mask=asset(valid_path)))
            native_row = _write_json(folder / "row.json", dict(raw_producer_audit=audit_pin,
                support_path=str(valid_path), support_sha256=asset(valid_path)["sha256"]))
            records[case] = dict(full=full, native_row=native_row, source=dict(run=run))
            native_records[case] = _write_json(folder / "native.json", dict(complete=True,
                source_sha256=raw["sha256"], processor_kind="raw_rig", audit=audit_pin, full=full))
            geometry = _write_json(folder / "geometry.json", dict(run=run, acquisition=raw))
            products = _write_npz(folder / "products.npz", dict(valid_mask=np.isfinite(linear).all(0)))
            receipt = _write_json(folder / "receipt.json", dict(native_frame=geometry, products=products))
            produced[case] = BuildNativeFrame(dict(geometry=geometry,
                processor=dict(raw_binding=prepared_photometry["source_binding"])), products, receipt)
            if index: frames.append(dict(case_id=case, geometry=geometry))
        value = dict(configuration=dict(run=run, frames=frames), configuration_binding=selected_pin,
            phase_recipe=dict(native_configuration=prepared_photometry["native_configuration"], original_photo={},
                training_context=old_manifests[run]["training_context"],
                phase=dict(parameters=dict(inputs=dict(fields=dict(native=["image", "sha", "support_path", "support_sha256"]))))),
            produced=produced, acquisitions=records, prepared_native=native_records)
        return current_build_preparation.materialize_current_build_review(operation, metric, selected_pin,
            output, review, preparer=lambda *a, **kw: value)
    # Image preparation is a completed typed Setup operation before training,
    # rather than an implicit second fit inside the training coordinator.
    from darsia.fluidflower.pipeline import setup_image_preparation
    monkeypatch.setattr(setup_image_preparation, "completed_build_operations", lambda *a: operations)
    image_receipt = setup_image_preparation.run_image_preparation(dict(
        choice_authority=asset(series_path), job_root=snapshot["root"], runs=list(runs),
        native_parameters=native, adopted_revision=_write_json(root / "adopted.json", dict(revision="finite")),
        output_root=str(root / "setup-image-preparation")))
    snapshot["runs"].append(dict(run=setup_image_preparation.IMAGE_RUN,
        operation=setup_image_preparation.OPERATION, state="Complete", result=dict(image_preparation=image_receipt)))
    prepared = current_training_preparation.prepare_training_from_completed_build(snapshot,
        series_root=root, series_id="synthetic", output=root / "preparation", actor="reviewer",
        materializer=materialize, model_output_parent=root / "models")
    assert len(corrections) == 6 and all(audit["photometry"]["old_photometry_used"] is False for audit in corrections)
    assert all(audit["photometry"]["color"] == {"method": "off"} for audit in corrections)
    native_pins = [reader.json(item["manifest"])["native_configuration"] for item in prepared["reviews"].values()]
    assert len({pin["path"] for pin in native_pins}) == 2
    assert len({pin["sha256"] for pin in native_pins}) == 1
    actual_request = prepared["request"]
    training = raw_review_training.execution_preparation(asset(actual_request["request_path"]))
    summary = raw_fresh_training.train(training, output_root=Path(actual_request["output_root"]),
                                       cache_root=Path(actual_request["cache_root"]))
    assert summary["inventory"]["runs"] == list(runs)
    binding = require_current_model_basis(root, selection, model_reference=actual_request["output_root"], build_snapshot=snapshot)
    for row in reader.json(binding["image_basis"])["runs"]:
        assert reader.json(row["raw_binding"])["photometric_method"] == full_photometry.INDEPENDENT
    p1_root = root / "p1"
    p1_root.mkdir()
    _tiny_p1(p1_root, selection, binding, tiny_config[0], phase1_data, monkeypatch)
    assert all(reader.json(asset(p1_root / "results" / run / "completed.json"))["complete"] for run in runs)
