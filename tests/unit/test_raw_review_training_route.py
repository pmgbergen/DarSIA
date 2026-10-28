"""The ordinary Studio request executes fresh fitting on its prepared RAW basis."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from darsia.fluidflower.phase_saved_assets import asset
from darsia.fluidflower.phase_training import raw_fresh_training
from darsia.fluidflower.phase_training.raw_review_training import (
    FAMILY, execution_preparation, reviewed_input_basis,
)
from darsia.fluidflower.phase_workflow import prepare_phase_training_request, build_phase_training_command
from .test_raw_fresh_training import tiny_cohort  # noqa: F401
from .test_raw_photographic_review import inputs  # noqa: F401


def _studio_request(cohort, tmp_path):
    return prepare_phase_training_request(output_parent=tmp_path / "models",
        cache_parent=tmp_path / "training-cache", actor="reviewer",
        bundles=[(row["decisions"]["path"], row["combined_manifest"]["path"])
                 for row in cohort.prepare_kwargs["bundles"]],
        settings=cohort.prepare_kwargs["settings"])


def test_studio_raw_request_runs_original_mil_and_exports_the_consumed_image_basis(tiny_cohort, tmp_path):
    request = _studio_request(tiny_cohort, tmp_path)
    root = Path(request["output_root"])
    original = Path(request["request_path"]).read_bytes()
    assert request["settings"]["training_family"] == FAMILY
    assert request["settings"]["v5_raw_photometry"]["format"] == "darsia-reviewed-raw-input-photometry-v1"
    command = build_phase_training_command(request)
    assert "darsia.fluidflower.phase_training.raw_fresh_training" in command
    assert "--request" in command and "--resume-completed-heads" not in command
    from darsia.fluidflower.native_phase_steps import save_model_ui_name
    save_model_ui_name(request, "Named RAW model", "reviewer")
    prepared = execution_preparation(asset(request["request_path"]))
    summary = raw_fresh_training.train(prepared, output_root=root, cache_root=Path(request["cache_root"]))
    assert summary["inventory"]["runs"] == list(tiny_cohort.runs)
    basis = json.loads(Path(summary["image_basis"]["path"]).read_text())
    assert basis["input_photometry"] == request["settings"]["v5_raw_photometry"]
    assert {row["run"] for row in basis["runs"]} == set(tiny_cohort.runs)
    assert basis["native_configuration"] == tiny_cohort.configuration
    for head in summary["models"]["generic"]["heads"].values():
        model = json.loads(Path(head["model_path"]).read_text())
        assert model["training_provenance"]["image_basis"] == summary["image_basis"]
        assert head["final_fit"]["backend"] == "cpu"
        assert head["numerical_validation"]["passed"] is True
    assert Path(request["request_path"]).read_bytes() == original
    assert (root / "training_execution.json").is_file()
    assert json.loads((root / "model_ui_name.json").read_text())["display_name"] == "Named RAW model"
    with pytest.raises(FileExistsError):
        raw_fresh_training.train(prepared, output_root=root, cache_root=Path(request["cache_root"]))


def test_changed_declared_crop_is_not_silently_the_same_training_basis(tiny_cohort, tmp_path):
    request = _studio_request(tiny_cohort, tmp_path)
    pairs = [(row["decisions"]["snapshot_path"], row["combined_manifest"]["snapshot_path"])
             for row in request["bundles"]]
    original = reviewed_input_basis(pairs)
    record = original["runs"][0]
    manifest = Path(record["source_manifest"]["path"])
    document = json.loads(manifest.read_text())
    document["native_shape_hw"][1] -= 1
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA differs"):
        execution_preparation(asset(request["request_path"]))
    assert not (Path(request["output_root"]) / "training_started.json").exists()


def test_template_frozen_partitions_and_previous_heads_are_removed_for_current_raw_basis(tiny_cohort):
    from darsia.fluidflower.phase_training.raw_review_training import apply_settings
    bundles = [(row["decisions"]["path"], row["combined_manifest"]["path"])
               for row in tiny_cohort.prepare_kwargs["bundles"]]
    settings = dict(tiny_cohort.prepare_kwargs["settings"],
                    v5_review_fit_partition={"old_ids": ["never-inherit"]},
                    resume_completed_heads=True, inherited_heads={"old": "weights"})
    before = deepcopy(settings)
    result = apply_settings(settings, bundles)
    assert "v5_review_fit_partition" not in result and "inherited_heads" not in result
    assert "resume_completed_heads" not in result
    assert result["training_family"] == FAMILY
    assert settings == before


def test_raw_training_ui_uses_prepared_inputs_even_when_inference_revision_has_no_parent_request(tiny_cohort, qtbot, tmp_path):
    from types import SimpleNamespace
    import shutil
    from PySide6.QtWidgets import QCheckBox, QComboBox, QTextEdit, QVBoxLayout, QWidget
    from darsia.fluidflower.native_training_photometry import NativeTrainingPhotometry

    summary = raw_fresh_training.train(tiny_cohort.preparation,
        output_root=tiny_cohort.output, cache_root=tiny_cohort.cache)
    revision = tmp_path / "inference-revision"
    heads = revision / "models/generic"
    heads.mkdir(parents=True)
    for head in summary["models"]["generic"]["heads"].values():
        source = Path(head["model_path"])
        shutil.copyfile(source, heads / source.name)
    path = heads / "co2_presence_signal.json"
    entry = dict(model_id="inference-revision", root=str(revision),
        library_source_identity=dict(metadata_sha256={
            "models/generic/co2_presence_signal.json": asset(path)["sha256"],
            "models/generic/gas_within_co2_signal.json": asset(heads / "gas_within_co2_signal.json")["sha256"]}))
    body = QWidget()
    qtbot.addWidget(body)
    studio = SimpleNamespace(validation_model=QComboBox(), bundles=QTextEdit(),
                             _selected_validation_model=lambda: entry)
    bundles = [(row["decisions"]["path"], row["combined_manifest"]["path"])
               for row in tiny_cohort.prepare_kwargs["bundles"]]
    studio.bundles.setPlainText("\n".join(str(a) + " | " + str(b) for a, b in bundles))
    existing = QCheckBox()
    existing.setChecked(True)
    steps = SimpleNamespace(studio=studio, layouts={4: QVBoxLayout(body)},
        training_routes={}, series_id="synthetic", refresh=lambda: None, existing_basis=existing)
    controller = NativeTrainingPhotometry(steps)
    controller.refresh()  # The inputs are checked in the background.
    qtbot.waitUntil(lambda: not controller.checking, timeout=10000)
    assert controller.refresh() and controller.uses_prepared_raw_basis
    settings = controller.prepare_settings({}, bundles)
    assert settings["training_family"] == FAMILY
    assert settings["template_model"]["co2_head"]["sha256"] == asset(path)["sha256"]
    assert settings["template_model"]["saved_weights_reused"] is False
    assert settings["v5_raw_photometry"]["prepared_sources"]


def test_resource_limits_bound_run_parallelism_by_ram_and_cpu_without_changing_image_settings():
    options = dict(stage_workers=16, extraction_workers=16, opencv_threads=2)
    result = raw_fresh_training.execution_limits(options, [(1000, 2000)] * 20,
        resources=dict(cores=12, available_bytes=10 * 1024**3))
    assert result["stage_workers"] == result["extraction_workers"] == 4
    assert result["blas_threads"] == 1 and result["cpu_worker_limit"] == 6
    assert options == dict(stage_workers=16, extraction_workers=16, opencv_threads=2)


def test_photo_only_raw_training_runs_the_existing_feature_and_optical_fit(tiny_cohort, tmp_path):
    from darsia.fluidflower.phase_training.raw_fresh_request import prepare_request
    kwargs = dict(tiny_cohort.prepare_kwargs, context_mode="photo-only", context_source=None)
    request = prepare_request(output_root=tmp_path / "photo-prepared", **kwargs)
    result = raw_fresh_training.train(asset(request["request_path"]),
        output_root=tmp_path / "photo-model", cache_root=tmp_path / "photo-cache")
    assert result["models"]["generic"]["heads"]["co2_presence"]["final_fit"]["backend"] == "cpu"
    assert result["deployment_allowed"] is False


def test_parallel_raw_runs_and_independent_cpu_heads_preserve_fitted_parameters(tiny_cohort, tmp_path):
    from darsia.fluidflower.phase_training.raw_fresh_request import prepare_request
    single = raw_fresh_training.train(tiny_cohort.preparation,
        output_root=tiny_cohort.output, cache_root=tiny_cohort.cache)
    settings = dict(tiny_cohort.prepare_kwargs["settings"], stage_workers=2, extraction_workers=2)
    prepared = prepare_request(output_root=tmp_path / "parallel-prepared",
        **dict(tiny_cohort.prepare_kwargs, settings=settings))
    parallel = raw_fresh_training.train(asset(prepared["request_path"]),
        output_root=tmp_path / "parallel-model", cache_root=tmp_path / "parallel-cache")
    assert parallel["execution_resources"]["extraction_workers"] == 2
    assert parallel["execution_resources"]["fit_workers"] == 2
    for name in ("co2_presence", "gas_within_co2"):
        first = json.loads(Path(single["models"]["generic"]["heads"][name]["model_path"]).read_text())
        second = json.loads(Path(parallel["models"]["generic"]["heads"][name]["model_path"]).read_text())
        for key in ("coefficients", "intercept", "feature_mean", "feature_scale", "texture_sigma"):
            assert first[key] == second[key]


def test_unlabelled_prepared_run_is_bound_for_inference_without_training_rows(tiny_cohort, tmp_path):
    import shutil
    combined_path = Path(tiny_cohort.prepare_kwargs["bundles"][0]["combined_manifest"]["path"])
    combined = json.loads(combined_path.read_text())
    source = Path(tiny_cohort.manifests[0]["path"])
    run = "sample-unlabelled"
    target = source.parent.parent / run
    shutil.copytree(source.parent, target)
    manifest = json.loads((target / "manifest.json").read_text())
    manifest["run"] = run
    (target / "manifest.json").write_text(json.dumps(manifest))
    combined["jobs"].append(dict(combined["jobs"][0], run=run,
        native_manifest_sha256=asset(target / "manifest.json")["sha256"]))
    combined_path.write_text(json.dumps(combined))
    request = _studio_request(tiny_cohort, tmp_path)
    prepared = execution_preparation(asset(request["request_path"]))
    summary = raw_fresh_training.train(prepared, output_root=Path(request["output_root"]),
        cache_root=Path(request["cache_root"]))
    basis = json.loads(Path(summary["image_basis"]["path"]).read_text())
    assert {row["run"] for row in basis["runs"]} == {*tiny_cohort.runs, run}
    assert basis["training_runs"] == list(tiny_cohort.runs)
    assert summary["inventory"]["runs"] == list(tiny_cohort.runs)


def test_prepared_calibration_refit_uses_original_studio_process_and_actual_completion(tiny_cohort, qtbot, tmp_path):
    from types import SimpleNamespace, MethodType
    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import QWidget, QTextEdit, QLabel, QPushButton, QTabWidget, QCheckBox, QLineEdit
    from darsia.fluidflower.ui.model_training import SharedPhaseModelTrainingPage
    from darsia.fluidflower.native_phase_steps import NativePhaseSteps
    from darsia.fluidflower.calibration_model_training import launch_prepared_training
    class Studio(QWidget):
        training_completed = Signal(str)
        _launch_training_job = SharedPhaseModelTrainingPage._launch_training_job
        _read_output = SharedPhaseModelTrainingPage._read_output
        _training_finished = SharedPhaseModelTrainingPage._training_finished
        _training_error = SharedPhaseModelTrainingPage._training_error
        _select_completed_model = SharedPhaseModelTrainingPage._select_completed_model
    studio = Studio()
    qtbot.addWidget(studio)
    studio.training_process = None
    studio.training_request = None
    studio.bundles, studio.log = QTextEdit(), QTextEdit()
    studio.status, studio.stop_button = QLabel(), QPushButton()
    studio.reviewer, studio.history_toggle = QLineEdit("reviewer"), QCheckBox()
    studio.tabs, studio.evidence_tab = QTabWidget(), QWidget()
    studio.tabs.addTab(studio.evidence_tab, "Evidence")
    studio._update_action_states = lambda: None
    studio._prepare_post_training_views = lambda *a: None
    registered = []
    studio._record_completed_model = lambda identifier, root: registered.append((identifier, root))
    steps = SimpleNamespace(studio=studio, _ready=True, series_root=tmp_path,
        model_name=QLineEdit(), _named_model_context={}, refresh=lambda: None)
    training_options = []
    def saved_labels_training(call, **options):
        training_options.append(options)
        return call()
    steps.phase_workflow = SimpleNamespace(run_training=saved_labels_training, before_launch=lambda request: None)
    steps._original_launch = studio._launch_training_job
    studio._launch_training_job = MethodType(NativePhaseSteps._launch_named_job, steps)
    request = _studio_request(tiny_cohort, tmp_path)
    pairs = [(row["decisions"]["path"], row["combined_manifest"]["path"])
             for row in tiny_cohort.prepare_kwargs["bundles"]]
    completed = []
    studio.training_completed.connect(completed.append)
    process = launch_prepared_training(steps, dict(request=request, bundles=pairs))
    assert training_options == [dict(saved_labels_only=True)]
    assert process is studio.training_process
    qtbot.waitUntil(lambda: studio.training_process is None, timeout=45000)
    assert completed == [request["output_root"]], studio.log.toPlainText()
    assert registered == [(request["model_id"], Path(request["output_root"]))]
    assert Path(request["output_root"], "summary.json").is_file()
    assert "Completed" in studio.status.text()


def test_repeated_bundles_reuse_each_raw_source_check_within_one_basis_read(tiny_cohort, monkeypatch):
    from darsia.fluidflower.phase_training import raw_review_training
    calls = []
    original = raw_review_training.validate_raw_review_manifest
    def checked(manifest, run):
        calls.append(run)
        return original(manifest, run)
    monkeypatch.setattr(raw_review_training, "validate_raw_review_manifest", checked)
    pairs = [(row["decisions"]["path"], row["combined_manifest"]["path"])
             for row in tiny_cohort.prepare_kwargs["bundles"]] * 20
    first = reviewed_input_basis(pairs)
    assert calls == list(tiny_cohort.runs)
    assert reviewed_input_basis(pairs) == first
    assert calls == list(tiny_cohort.runs) * 2
