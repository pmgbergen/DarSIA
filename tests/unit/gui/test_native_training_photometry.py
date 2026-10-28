from copy import deepcopy
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
from PySide6.QtWidgets import QComboBox, QTextEdit, QVBoxLayout, QWidget

from darsia.fluidflower import phase_review_retractions as withdrawals
from darsia.fluidflower.phase_workflow import prepare_phase_training_request


from darsia.fluidflower import native_training_photometry as ui
from darsia.fluidflower.ui import model_training as training_ui

MODULE = Path(ui.__file__)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def answered(qtbot, controller):
    """Wait until the background check of the current training inputs has answered."""
    qtbot.waitUntil(lambda: not controller.checking, timeout=10000)


def settled(qtbot, controller):
    """Refresh once the background check of the current inputs has answered."""
    controller.refresh()
    answered(qtbot, controller)
    return controller.refresh()


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "saved model"
    proof = tmp_path / "saved source admission.json"
    proof_sha = write(proof, {"complete": True, "nothing_computed_by_test": True})
    photo = {
        "format": ui.MODEL_FORMAT,
        "full_native_resolution": True,
        "unit_denominator": ui.V5_UNIT_DENOMINATOR,
        "display_jpeg_is_training_input": False,
        "ac28_light_anchor": True,
        "materialization_receipt_path": str(proof),
        "materialization_receipt_sha256": proof_sha,
        "unknown_future_source_field": {"kept": [1, 2.0]},
    }
    chemistry = {
        "contract": "ffac-v5-light-only-chemistry-exclusion-v2",
        "excluded_runs": list(ui.EXCLUDED_RUNS),
        "excluded_review_ids": 0,
        "opaque_saved_value": "unchanged",
    }
    document = {
        "model_id": "original inherited request",
        "settings": {"v5_raw_photometry": photo, "v5_chemistry_fit_scope": chemistry},
        "bundles": [],
    }
    checksum = write(root / "training_request.json", document)
    entry = {
        "model_id": "selected inference revision",
        "root": str(root),
        "datasets": ["testseries"],
        "library_source_identity": {
            "model_id": "selected inference revision",
            "metadata_sha256": {"training_request.json": checksum},
        },
    }
    return NS(root=root, entry=entry, document=document, proof=proof, tmp=tmp_path)


@pytest.fixture
def bundle(source):
    root = source.tmp / "saved native source"
    payload = source.tmp / "scientific payload.log_linear_rgb.npy"
    payload.write_bytes(b"NOT AN ARRAY: no scientific payload decoder is permitted")
    valid = source.tmp / "scientific payload.log_linear_valid_mask.npy"
    valid.write_bytes(b"not a mask; only file presence is checked by this UI")
    row = {
        "v5_log_linear_rgb_path": str(payload),
        "v5_log_linear_rgb_sha256": "a" * 64,
        "v5_log_linear_valid_mask_path": str(valid),
        "v5_log_linear_valid_mask_sha256": "b" * 64,
    }
    manifest = {
        "v5_photometry": {"format": ui.SOURCE_FORMAT},
        "v5_float_support_policy": ui.EXACT_VALID_MASK_POLICY,
        "reference": dict(row),
        "frame_sources": [dict(row)],
        "content_box_xywh": [0, 0, 1, 1],
    }
    manifest_path = root / "native_frames/one/manifest.json"
    native_sha = write(manifest_path, manifest)
    decisions = source.tmp / "reviewed decisions.json"
    combined = source.tmp / "combined.json"
    write(
        decisions,
        {"decisions": {"one": {"annotations": [{"id": "inert synthetic region"}]}}},
    )
    document = {
        "jobs": [
            {
                "run": "one",
                "dataset": "testseries",
                "analysis_source_review_root": str(root),
                "review_output_dir": str(root),
                "native_manifest_sha256": native_sha,
            }
        ]
    }
    write(combined, document)
    return NS(
        pairs=[(decisions, combined)],
        root=root,
        manifest=manifest,
        path=manifest_path,
        combined=combined,
        combined_document=document,
        payload=payload,
        valid=valid,
    )


def test_explicit_index_binding_and_inherited_request_are_distinct(source):
    before = (source.root / "training_request.json").read_bytes()
    value = ui.read_saved_training_photometry(source.entry, "TEST_SERIES")
    assert value.model_id == "selected inference revision"
    assert value.request_model_id == "original inherited request"
    assert value.request_sha256 in value.label and value.model_id in value.label
    assert value.settings == source.document["settings"]
    assert (source.root / "training_request.json").read_bytes() == before


@pytest.mark.parametrize(
    "change,match",
    [
        ("other_series", "another series"),
        ("missing_index_sha", "indexed training-request SHA"),
        ("source_changed", "SHA changed"),
        ("reference_changed", "SHA changed"),
        ("reference_sha_missing", "reference SHA"),
        ("uniform24", "legacy radiometry"),
    ],
)
def test_missing_mismatched_source_is_explicitly_unavailable(source, change, match):
    entry = deepcopy(source.entry)
    if change == "other_series":
        entry["datasets"] = ["foreign"]
    elif change == "missing_index_sha":
        entry["library_source_identity"]["metadata_sha256"] = {}
    elif change == "source_changed":
        (source.root / "training_request.json").write_text("{}", encoding="utf-8")
    elif change == "reference_changed":
        source.proof.write_text("{}", encoding="utf-8")
    else:
        document = deepcopy(source.document)
        if change == "reference_sha_missing":
            document["settings"]["v5_raw_photometry"].pop(
                "materialization_receipt_sha256"
            )
        else:
            document["settings"]["radiometry_profile"] = "uniform24-after2014"
        entry["library_source_identity"]["metadata_sha256"]["training_request.json"] = (
            write(source.root / "training_request.json", document)
        )
    with pytest.raises(ValueError, match=match):
        ui.read_saved_training_photometry(entry, "testseries")


def test_sidecar_admission_reads_json_not_payload_and_preserves_original_guard(
    source, bundle, monkeypatch
):
    binding = ui.read_saved_training_photometry(source.entry, "testseries")
    read_bytes = Path.read_bytes

    def no_array_read(path):
        if path.suffix == ".npy":
            raise AssertionError("UI must not read a scientific array")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", no_array_read)
    result = ui.require_saved_sidecars(bundle.pairs, binding)
    assert result["native_manifests"] == 1
    assert result["declared_existing_sidecars"] == 2
    assert result["payload_sha_verified_here"] is False


def test_missing_valid_mask_or_its_sha_closes_admission(source, bundle):
    binding = ui.read_saved_training_photometry(source.entry, "testseries")
    changed = deepcopy(bundle.manifest)
    changed["frame_sources"][0].pop("v5_log_linear_valid_mask_sha256")
    bundle.combined_document["jobs"][0]["native_manifest_sha256"] = write(
        bundle.path, changed
    )
    write(bundle.combined, bundle.combined_document)
    with pytest.raises(ValueError, match="sidecar path/SHA"):
        ui.require_saved_sidecars(bundle.pairs, binding)


def test_unknown_track1_partition_is_never_invented(source, bundle):
    changed = deepcopy(bundle.manifest)
    changed["v5_track1_photo_geometry"] = {"contract": "saved Track1"}
    bundle.combined_document["jobs"][0]["native_manifest_sha256"] = write(
        bundle.path, changed
    )
    write(bundle.combined, bundle.combined_document)
    with pytest.raises(ValueError, match="original saved V5 review-fit partition"):
        ui.require_saved_sidecars(
            bundle.pairs, ui.read_saved_training_photometry(source.entry, "testseries")
        )


def test_original_inert_request_producer_preserves_exact_blocks_and_other_settings(
    source, bundle
):
    binding = ui.read_saved_training_photometry(source.entry, "testseries")
    settings = {
        "model_scope": "single-series",
        "series": ["testseries"],
        "backend": "cpu",
        "mil_max_iters": 77,
        "custom_unrelated_field": {"future": True},
        **deepcopy(binding.settings),
    }
    original = deepcopy(settings)
    result = prepare_phase_training_request(
        output_parent=source.tmp / "inert output",
        cache_parent=source.tmp / "inert cache",
        bundles=bundle.pairs,
        actor="Synthetic only",
        settings=settings,
        withdrawal_snapshot=withdrawals.empty_snapshot("testseries"),
    )
    binding.require_request(result)
    persisted = json.loads(Path(result["request_path"]).read_bytes())
    assert persisted["settings"]["v5_raw_photometry"] == original["v5_raw_photometry"]
    assert (
        persisted["settings"]["v5_chemistry_fit_scope"]
        == original["v5_chemistry_fit_scope"]
    )
    assert (
        persisted["settings"]["custom_unrelated_field"]
        == original["custom_unrelated_field"]
    )
    assert persisted["settings"]["mil_max_iters"] == 77
    assert "radiometry_profile" not in persisted["settings"]
    assert settings == original
    assert not (Path(result["output_root"]) / "models").exists()


def test_native_controller_failure_disables_fresh_route_and_never_launches(
    qtbot, source, bundle
):
    body = QWidget()
    qtbot.addWidget(body)
    layout = QVBoxLayout(body)
    studio = NS(
        validation_model=QComboBox(),
        bundles=QTextEdit(),
        _selected_validation_model=lambda: source.entry,
    )
    studio.bundles.setPlainText(
        str(bundle.pairs[0][0]) + " | " + str(bundle.pairs[0][1])
    )
    steps = NS(
        studio=studio,
        layouts={4: layout},
        training_routes={},
        series_id="testseries",
        refresh=lambda: None,
        _original_start=lambda: pytest.fail("Original training must not be launched"),
    )
    controller = ui.NativeTrainingPhotometry(steps)
    assert settled(qtbot, controller)
    settings = {"radiometry_profile": "uniform24-after2014", "unrelated": {"keep": 1}}
    prepared = controller.prepare_settings(settings, bundle.pairs)
    assert "radiometry_profile" not in prepared
    assert (
        prepared["unrelated"] == settings["unrelated"]
        and settings["radiometry_profile"] == "uniform24-after2014"
    )
    source.proof.write_text("{}", encoding="utf-8")
    assert not controller.refresh(force=True)
    assert "SHA changed" in controller.reason
    assert "scratch" not in steps.training_routes


def test_native_original_start_calls_unchanged_producer_with_copied_v5_not_uniform24(
    qtbot, source, bundle
):
    from darsia.fluidflower.ui import model_training

    candidate = Path(training_ui.__file__)
    tree = ast.parse(candidate.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "SharedPhaseModelTrainingPage"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_start_training"
    )
    scope = dict(model_training.__dict__)
    source_settings, launches = [], []

    def inert_original_producer(**kwargs):
        source_settings.append(deepcopy(kwargs["settings"]))
        return prepare_phase_training_request(
            **kwargs, withdrawal_snapshot=withdrawals.empty_snapshot("testseries")
        )

    scope["prepare_phase_training_request"] = inert_original_producer
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            "actual original start with native UI hook",
            "exec",
        ),
        scope,
    )
    text = lambda value: NS(text=lambda: str(value))
    choice = lambda value: NS(currentData=lambda: value)
    spin = lambda value: NS(value=lambda: value)
    body = QWidget()
    qtbot.addWidget(body)
    studio = NS(
        training_process=None,
        _series_batch_active=False,
        validation_model=QComboBox(),
        bundles=QTextEdit(),
        _selected_validation_model=lambda: source.entry,
        radiometry_mode=choice("uniform24-after2014"),
        radiometry_profile_path=text(""),
        training_series=text("testseries"),
        training_scope=choice("single-series"),
        training_evaluation=choice("whole-run"),
        backend=choice("cpu"),
        stage_workers=spin(1),
        extraction_workers=spin(1),
        pixels=spin(64),
        mil_iterations=spin(10),
        change_ablation=NS(isChecked=lambda: False),
        _models_folder_for_series=lambda _: source.tmp / "native inert output",
        cache_parent=text(source.tmp / "native inert cache"),
        reviewer=text("Synthetic only"),
        _refresh_review_sessions=lambda: None,
        _launch_training_job=lambda *args: launches.append(args),
    )
    studio.bundles.setPlainText(
        str(bundle.pairs[0][0]) + " | " + str(bundle.pairs[0][1])
    )
    steps = NS(
        studio=studio,
        layouts={4: QVBoxLayout(body)},
        training_routes={},
        series_id="testseries",
        refresh=lambda: None,
    )
    controller = ui.NativeTrainingPhotometry(steps)
    steps._original_start = lambda: scope["_start_training"](studio)
    assert settled(qtbot, controller)
    controller.train()
    assert len(source_settings) == len(launches) == 1
    assert "radiometry_profile" not in source_settings[0]
    assert (
        source_settings[0]["v5_raw_photometry"]
        == source.document["settings"]["v5_raw_photometry"]
    )
    assert (
        source_settings[0]["v5_chemistry_fit_scope"]
        == source.document["settings"]["v5_chemistry_fit_scope"]
    )
    assert (
        source_settings[0]["backend"] == "cpu"
        and source_settings[0]["mil_max_iters"] == 10
    )
    request, command, _cwd, _status = launches[0]
    controller.require_request(request)
    assert "--radiometry-profile" not in command
    assert "--fit-excluded-runs" in command
    assert not (Path(request["output_root"]) / "models").exists()


def test_incompatible_persisted_request_blocks_route_and_preserves_the_output(
    qtbot, source, bundle
):
    body = QWidget()
    qtbot.addWidget(body)
    studio = NS(
        validation_model=QComboBox(),
        bundles=QTextEdit(),
        _selected_validation_model=lambda: source.entry,
    )
    studio.bundles.setPlainText(
        str(bundle.pairs[0][0]) + " | " + str(bundle.pairs[0][1])
    )
    steps = NS(
        studio=studio,
        layouts={4: QVBoxLayout(body)},
        training_routes={},
        series_id="testseries",
        refresh=lambda: None,
    )
    controller = ui.NativeTrainingPhotometry(steps)
    assert settled(qtbot, controller)
    path = source.tmp / "preserved incompatible request.json"
    write(path, {"settings": {"v5_raw_photometry": {}, "v5_chemistry_fit_scope": {}}})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="byte-identically"):
        controller.require_request({"request_path": str(path)})
    assert "scratch" not in steps.training_routes and controller.binding is None
    assert "byte-identically" in controller.reason
    assert path.read_bytes() == before
    assert not controller.refresh()


@pytest.fixture
def native_ready(qtbot, source, request):
    """Real candidate NativePhaseSteps, with the retained UI readiness predicate."""
    from types import MethodType
    from PySide6.QtWidgets import QLineEdit, QLabel, QSpinBox, QCheckBox
    from darsia.fluidflower.ui import model_training
    import darsia.fluidflower.native_phase_steps as original_native

    native = original_native
    fixture_path = (
        Path(original_native.__file__).resolve().parents[3]
        / "tests/unit/gui/test_native_phase_steps.py"
    )
    fixture_spec = importlib.util.spec_from_file_location(
        "_retained_photometry_native_fixture", fixture_path
    )
    fixture = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture)
    studio = fixture.FakeStudio()
    studio.bundles = QTextEdit()
    studio.training_scope = QComboBox()
    studio.training_scope.addItem("Single", "single-series")
    studio.training_scope.addItem("Shared", "shared")
    studio.radiometry_mode = QComboBox()
    studio.radiometry_mode.addItem("Custom", "custom")
    studio.radiometry_mode.addItem("Image", "image")
    studio.radiometry_profile_path = model_training.PathField(directory=False)
    studio.radiometry_profile_path.setText("")
    studio.training_effect_summary = QLabel()
    studio.backend = QComboBox()
    studio.backend.addItem("CPU", "cpu")
    studio.backend.addItem("CUDA", "cuda")
    if getattr(request, "param", None) == "cv-controls":
        studio.training_evaluation = QComboBox()
        studio.training_evaluation.addItem("Whole-run CV", "whole-run")
        studio.training_evaluation.addItem("Frame CV", "frame")
    for name, value in (
        ("stage_workers", 2),
        ("extraction_workers", 3),
        ("pixels", 512),
        ("mil_iterations", 40),
    ):
        control = QSpinBox()
        control.setMaximum(100000)
        control.setValue(value)
        setattr(studio, name, control)
    studio.change_ablation = QCheckBox()
    original_summary = MethodType(
        model_training.SharedPhaseModelTrainingPage._update_training_effect_summary,
        studio,
    )
    studio._update_training_effect_summary = original_summary
    studio.training_series.setText("testseries")
    studio.reviewer.setText("Synthetic researcher")
    studio.selected_entry = source.entry
    studio._selected_validation_model = lambda: studio.selected_entry
    # Extract only the unchanged original training-ready statements; unrelated
    # catalogue/session UI is inert, so no actual data or discovery is opened.
    tree = ast.parse(Path(model_training.__file__).read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SharedPhaseModelTrainingPage"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_update_action_states"
    )
    start = next(
        i
        for i, n in enumerate(method.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "training_ready" for t in n.targets)
    )
    end = next(
        i
        for i in range(start, len(method.body))
        if isinstance(method.body[i], ast.Expr)
        and isinstance(method.body[i].value, ast.Call)
        and isinstance(method.body[i].value.func, ast.Attribute)
        and method.body[i].value.func.attr == "setEnabled"
    )
    extracted = ast.FunctionDef(
        name="original_training_ready",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self")],
            vararg=ast.arg(arg="_args"),
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            ast.parse("reviewer_ready = bool(self.reviewer.text().strip())").body[0],
            *method.body[start : end + 1],
        ],
        decorator_list=[],
    )
    scope = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[])),
            "retained original UI training predicate",
            "exec",
        ),
        scope,
    )
    direct = MethodType(scope["original_training_ready"], studio)
    studio._update_action_states = direct
    # These slots are bound before the native instance policy, exactly as in
    # SharedPhaseModelTrainingPage.__init__.
    for signal in (
        studio.reviewer.textChanged,
        studio.bundles.textChanged,
        studio.training_series.textChanged,
        studio.training_scope.currentIndexChanged,
        studio.radiometry_mode.currentIndexChanged,
    ):
        signal.connect(direct)
    studio.radiometry_profile_path.changed.connect(direct)
    for control in (
        studio.radiometry_mode,
        studio.radiometry_profile_path,
        studio.backend,
        studio.stage_workers,
        studio.extraction_workers,
        studio.pixels,
        studio.mil_iterations,
        studio.change_ablation,
    ):
        signal = (
            control.changed
            if hasattr(control, "changed")
            else (
                control.currentIndexChanged
                if isinstance(control, QComboBox)
                else (
                    control.stateChanged
                    if isinstance(control, QCheckBox)
                    else control.valueChanged
                )
            )
        )
        signal.connect(original_summary)
    if hasattr(studio, "training_evaluation"):
        studio.training_evaluation.currentIndexChanged.connect(original_summary)
    direct()
    original_summary()
    normalisation = fixture.FakeNormalisation()
    normalisation.ready = True
    steps = native.NativePhaseSteps(studio, normalisation=normalisation)
    steps.series_id = "testseries"
    controller = ui.NativeTrainingPhotometry(steps)
    steps.training_photometry = controller
    steps.scratch_basis.setChecked(True)
    steps.model_name.setText("Synthetic new named model")
    steps.refresh()
    qtbot.addWidget(studio)
    qtbot.addWidget(steps)
    answered(qtbot, controller)
    return NS(
        qtbot=qtbot,
        studio=studio,
        steps=steps,
        controller=controller,
        original_ready=direct,
        original_summary=original_summary,
    )


@pytest.mark.parametrize("legacy_mode", ["custom", "image"])
def test_real_native_fresh_readiness_ignores_only_unused_blank_legacy_profile(
    native_ready, bundle, legacy_mode
):
    s = native_ready
    s.studio.radiometry_mode.setCurrentIndex(
        s.studio.radiometry_mode.findData(legacy_mode)
    )
    assert not s.studio.start_button.isEnabled()
    assert not s.steps._legacy_start_available
    s.studio.bundles.setPlainText(
        str(bundle.pairs[0][0]) + " | " + str(bundle.pairs[0][1])
    )
    answered(s.qtbot, s.controller)
    # First evidence edit triggers the old bound slot and then the new native
    # hook. No timer, model discovery, Train call or process is needed.
    assert s.controller.binding is not None
    assert not s.steps._legacy_start_available
    assert s.studio.radiometry_profile_path.text() == ""
    assert s.studio.start_button.isEnabled()
    assert "original-training" not in s.studio.calls
    s.steps.existing_basis.setChecked(True)
    assert not s.studio.start_button.isEnabled()  # No invented continuation.


def test_native_fresh_readiness_keeps_reviewer_and_single_series_guards(
    native_ready, bundle
):
    s = native_ready
    s.studio.bundles.setPlainText(
        str(bundle.pairs[0][0]) + " | " + str(bundle.pairs[0][1])
    )
    answered(s.qtbot, s.controller)
    assert s.studio.start_button.isEnabled()
    s.studio.reviewer.clear()
    assert not s.studio.start_button.isEnabled()
    s.studio.reviewer.setText("Synthetic researcher")
    assert s.studio.start_button.isEnabled()
    s.studio.training_series.clear()
    assert not s.studio.start_button.isEnabled()
    s.studio.training_series.setText("testseries,another")
    assert not s.studio.start_button.isEnabled()
    s.studio.training_series.setText("testseries")
    assert s.studio.start_button.isEnabled()
    s.steps.model_name.clear()
    assert not s.studio.start_button.isEnabled()
    s.steps.model_name.setText("Named")
    assert s.studio.start_button.isEnabled()
    s.steps.set_normalisation_ready(False)
    assert not s.studio.start_button.isEnabled()


def test_native_fresh_missing_source_no_bundle_or_busy_stays_disabled(
    native_ready, bundle
):
    s = native_ready
    assert s.controller.binding is None and not s.studio.start_button.isEnabled()
    s.studio.bundles.setPlainText(
        str(bundle.pairs[0][0]) + " | " + str(bundle.pairs[0][1])
    )
    answered(s.qtbot, s.controller)
    assert s.studio.start_button.isEnabled()
    s.studio.training_process = object()
    s.steps.refresh()
    assert not s.studio.start_button.isEnabled()
    s.studio.training_process = None
    s.studio._series_batch_active = True
    s.steps.refresh()
    assert not s.studio.start_button.isEnabled()
    s.studio._series_batch_active = False
    s.steps.refresh()
    assert s.studio.start_button.isEnabled()
    s.studio.selected_entry = None
    s.controller.changed()
    assert s.controller.binding is None and not s.studio.start_button.isEnabled()
    assert "scratch" not in s.steps.training_routes


def test_native_clear_evidence_disables_and_original_unadapted_predicate_is_unchanged(
    native_ready, bundle
):
    s = native_ready
    s.studio.bundles.setPlainText(
        str(bundle.pairs[0][0]) + " | " + str(bundle.pairs[0][1])
    )
    answered(s.qtbot, s.controller)
    assert s.studio.start_button.isEnabled()
    s.original_ready()  # The retained standalone predicate still checks custom profile.
    assert not s.studio.start_button.isEnabled()
    s.steps.refresh()
    assert s.studio.start_button.isEnabled()
    s.studio.bundles.clear()
    assert not s.studio.start_button.isEnabled() and s.controller.binding is None
    assert "original-training" not in s.studio.calls


def test_native_advanced_summary_uses_saved_v5_and_preserves_original_backend_tail(
    native_ready, bundle
):
    s = native_ready
    s.studio.bundles.setPlainText(
        str(bundle.pairs[0][0]) + " | " + str(bundle.pairs[0][1])
    )
    answered(s.qtbot, s.controller)
    assert s.studio.start_button.isEnabled()
    marker = " Current effect: "
    for backend, workers, iterations in (("cpu", 2, 40), ("cuda", 4, 70)):
        s.studio.backend.setCurrentIndex(s.studio.backend.findData(backend))
        s.studio.stage_workers.setValue(workers)
        s.studio.mil_iterations.setValue(iterations)
        final = s.studio.training_effect_summary.text()
        assert final.startswith("Manual new-model training uses saved V5 photometry.")
        assert "source not selected" not in final
        # Compare the rest to the actual retained original formatter, rather
        # than duplicate its backend/concurrency wording in the assertion.
        s.original_summary()
        original = s.studio.training_effect_summary.text()
        assert "saved V5" not in original
        assert final.partition(marker)[2] == original.partition(marker)[2]
        s.steps.refresh()
        assert s.studio.training_effect_summary.text() == final
    tail = s.studio.training_effect_summary.text().partition(marker)[2]
    s.studio.selected_entry = None
    s.controller.changed()
    assert "V5 photometry unavailable" in s.studio.training_effect_summary.text()
    assert s.studio.training_effect_summary.text().partition(marker)[2] == tail
    # A direct retained/standalone formatter remains untouched.
    s.original_summary()
    assert (
        s.studio.radiometry_mode.currentText()
        in s.studio.training_effect_summary.text()
    )
    assert "V5 photometry unavailable" not in s.studio.training_effect_summary.text()


@pytest.mark.parametrize("native_ready", ["cv-controls"], indirect=True)
def test_native_continuation_summary_qualifies_fresh_controls_and_restores_scratch(
    native_ready, bundle, monkeypatch
):
    """Real basis and settings signals describe the route without launching it."""
    from darsia.fluidflower.native_continuation_training import NativeContinuationTraining

    s = native_ready
    s.studio.bundles.setPlainText(
        str(bundle.pairs[0][0]) + " | " + str(bundle.pairs[0][1])
    )
    answered(s.qtbot, s.controller)
    assert s.studio.start_button.isEnabled()
    continuation = NativeContinuationTraining(s.steps)
    # Parent admission is outside this presentation-only case; no model is loaded.
    monkeypatch.setattr(continuation, "refresh", lambda: continuation.reason)
    s.steps.existing_basis.setChecked(True)
    text = s.studio.training_effect_summary.text()
    assert "continuation uses CPU" in text and "parent request" in text
    assert "No new cross-validation is run" in text
    assert "controls apply only to fresh-model training" in text
    assert " Current effect: " not in text
    assert not s.studio.start_button.isEnabled()
    for control, value in ((s.studio.backend, "cuda"), (s.studio.training_evaluation, "frame")):
        control.setCurrentIndex(control.findData(value))
        assert control.currentData() == value
        assert s.studio.training_effect_summary.text() == text
    s.studio.mil_iterations.setValue(70)
    assert s.studio.mil_iterations.value() == 70
    assert s.studio.training_effect_summary.text() == text
    s.steps.scratch_basis.setChecked(True)
    fresh = s.studio.training_effect_summary.text()
    assert fresh.startswith("Manual new-model training uses saved V5 photometry.")
    assert " Current effect: " in fresh
    s.original_summary()
    assert fresh.partition(" Current effect: ")[2] == s.studio.training_effect_summary.text().partition(" Current effect: ")[2]
    s.studio._update_training_effect_summary()
    assert s.studio.training_effect_summary.text() == fresh
    assert s.studio.backend.currentData() == "cuda"
    assert s.studio.training_evaluation.currentData() == "frame"
    assert s.studio.mil_iterations.value() == 70
    assert s.studio.start_button.isEnabled()
    assert s.studio.training_process is None and "original-training" not in s.studio.calls


def test_real_pathfield_signal_cannot_override_native_training_gates(
    native_ready, bundle
):
    s = native_ready
    s.studio.bundles.setPlainText(
        str(bundle.pairs[0][0]) + " | " + str(bundle.pairs[0][1])
    )
    answered(s.qtbot, s.controller)
    assert s.studio.start_button.isEnabled()
    s.studio.radiometry_profile_path.setText("irrelevant-legacy-profile.json")
    assert s.studio.start_button.isEnabled()
    (
        s.studio.radiometry_profile_path.clear()
        if hasattr(s.studio.radiometry_profile_path, "clear")
        else s.studio.radiometry_profile_path.setText("")
    )
    assert s.studio.start_button.isEnabled()
    s.steps.model_name.clear()
    s.studio.radiometry_profile_path.setText("another-irrelevant-legacy-profile.json")
    assert not s.studio.start_button.isEnabled()
    s.steps.model_name.setText("Named")
    s.steps.set_normalisation_ready(False)
    s.studio.radiometry_profile_path.setText("third-irrelevant-legacy-profile.json")
    assert not s.studio.start_button.isEnabled()
    s.steps.set_normalisation_ready(True)
    assert s.studio.start_button.isEnabled()
    s.studio.radiometry_profile_path.setText("")
    assert s.studio.start_button.isEnabled()
    assert "original-training" not in s.studio.calls


def test_input_check_runs_in_the_background_and_a_failure_is_not_repeated(qtbot, monkeypatch):
    """A failed check once read every bundle again on each page refresh, on the GUI thread."""
    import threading
    calls = []

    def failing(entry, bundle_text, series_id, reader):
        calls.append((threading.current_thread() is threading.main_thread(), bundle_text))
        raise ValueError("a projected bundle is incomplete")

    monkeypatch.setattr(ui, "check_training_inputs", failing)
    body = QWidget()
    qtbot.addWidget(body)
    studio = NS(validation_model=QComboBox(), bundles=QTextEdit(), _selected_validation_model=lambda: None)
    studio.bundles.setPlainText("decisions.json | combined.json")
    refreshed = []
    steps = NS(studio=studio, layouts={4: QVBoxLayout(body)}, training_routes={"scratch": object()},
               series_id="testseries", refresh=lambda: refreshed.append(controller.refresh()))
    controller = ui.NativeTrainingPhotometry(steps)
    assert not controller.refresh() and controller.checking
    assert controller.reason == ui.CHECKING_TEXT and "scratch" not in steps.training_routes
    answered(qtbot, controller)
    assert refreshed == [False]  # The answer refreshes the page once.
    for _ in range(5):
        assert not controller.refresh()
    assert calls == [(False, "decisions.json | combined.json")]
    assert controller.reason == "Fresh training is unavailable: a projected bundle is incomplete"
    studio.bundles.setPlainText("other decisions.json | other combined.json")
    assert not controller.refresh() and controller.reason == ui.CHECKING_TEXT
    answered(qtbot, controller)
    assert len(calls) == 2 and not calls[1][0]
    assert not controller.refresh(force=True) and len(calls) == 3 and calls[2][0]

