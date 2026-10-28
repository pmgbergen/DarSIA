"""Native presentation regressions; original routes are replaced only by test spies."""

from types import SimpleNamespace as NS

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit, QPushButton, QSpinBox, QTextEdit, QWidget

from darsia.fluidflower import native_phase_model_workflow as ui
from darsia.fluidflower import native_phase_steps
from .test_native_phase_steps import FakeStudio, FakeNormalisation


@pytest.fixture
def workflow(qtbot):
    studio = FakeStudio()
    studio.review_sessions = []
    studio.model_history = []
    studio._selected_validation_model = lambda: None
    studio.bundles = QTextEdit()
    studio.backend = QComboBox()
    studio.backend.addItem("Automatic", "auto")
    studio.backend.addItem("CUDA float32", "cuda")
    studio.backend.addItem("CPU float64", "cpu")
    studio.open_blind_labeling = QPushButton("Original blind action")
    studio.open_blind_labeling.clicked.connect(lambda: studio._launch_review("blind_region_labeling"))
    studio.evidence_audit_tab = QWidget()
    for name, low, high, value in (("stage_workers", 1, 64, 4), ("extraction_workers", 1, 64, 4),
                                   ("pixels", 64, 1_000_000, 1024), ("mil_iterations", 20, 2000, 160)):
        control = QSpinBox()
        control.setRange(low, high)
        control.setValue(value)
        setattr(studio, name, control)
    studio.change_ablation = QCheckBox()
    studio.change_ablation.setChecked(True)
    studio.radiometry_profile_path = QLineEdit("original reference")
    studio.post_training_scope = QComboBox()
    studio.post_training_scope.addItem("Reviewed runs", "reviewed")
    for name in ("bundles", "backend", "stage_workers", "extraction_workers", "pixels",
                 "mil_iterations", "change_ablation", "radiometry_profile_path", "post_training_scope"):
        studio.training_tab.widget().layout().addWidget(getattr(studio, name))
    normalisation = FakeNormalisation()
    steps = native_phase_steps.NativePhaseSteps(studio, normalisation=normalisation)
    # Source-module tests also work before the narrow shared-file integration.
    if not hasattr(steps, "phase_workflow"):
        steps.phase_workflow = ui.NativePhaseModelWorkflow(steps)
    qtbot.addWidget(studio)
    qtbot.addWidget(steps)
    return NS(studio=studio, steps=steps, workflow=steps.phase_workflow)


def test_removed_settings_preserve_original_values_and_required_controls(workflow):
    s, steps, page = workflow.studio, workflow.steps, workflow.workflow
    assert [steps.phase_tabs.tabText(i) for i in range(4)] == [
        "Basis", "Label phases", "Blindly validate phase model", "Train"]
    assert steps.groups[2].isAncestorOf(s.open_model_correction)
    assert steps.groups[3].isAncestorOf(s.open_blind_labeling)
    assert s.open_model_correction.isHidden() and s.open_blind_labeling.isHidden()
    assert s.evidence_audit_tab.isHidden()
    assert steps.advanced.isHidden() and s.resume_session_button.isHidden()
    assert not steps.organise_button.isHidden()
    assert page.archive_button.text() == "Archive unused models…"
    assert steps.import_button.text() == "Import existing model…"
    assert s.radiometry_profile_path.text() == "original reference"
    assert s.post_training_scope.currentData() == "reviewed"
    assert steps.advanced.isAncestorOf(s.post_training_scope)
    assert s.backend.currentData() == "auto"
    for name, value, bounds in (("stage_workers", 4, (1, 64)), ("extraction_workers", 4, (1, 64)),
                               ("pixels", 1024, (64, 1_000_000)), ("mil_iterations", 160, (20, 2000))):
        widget = getattr(s, name)
        assert page.parameters.isAncestorOf(widget)
        assert widget.value() == value
        assert (widget.minimum(), widget.maximum()) == bounds
    assert s.change_ablation.isChecked()
    assert "small and large boxes more evenly" in s.pixels.toolTip()
    assert s.training_process is None and not s.calls


def test_hidden_role_actions_keep_the_saved_binding_and_normalisation_gates(workflow):
    s, steps, page = workflow.studio, workflow.steps, workflow.workflow
    assert not s.open_model_correction.isEnabled()
    assert not s.open_blind_labeling.isEnabled()
    s._selected_validation_model = lambda: {"model_id": "saved", "validation_binding": {}}
    steps.set_normalisation_ready(True)
    page.refresh()
    assert s.open_model_correction.isHidden() and s.open_blind_labeling.isHidden()
    s.open_blind_labeling.click()
    s.open_model_correction.click()
    assert s.calls == [("original-review", "blind_region_labeling"), ("original-review", "open_correction")]
    steps.set_normalisation_ready(False)
    page.refresh()
    # Verified saved photos can be reviewed without reapplying normalisation.
    assert s.open_model_correction.isEnabled() and s.open_blind_labeling.isEnabled()
    assert s.open_model_correction.isHidden() and s.open_blind_labeling.isHidden()
    s.open_blind_labeling.click()
    s.open_model_correction.click()
    assert len(s.calls) == 4
    for model in (None, {"model_id": "saved"},
                  {"model_id": "saved", "validation_binding": {}, "validation_binding_loading": True},
                  {"model_id": "saved", "validation_binding": {}, "validation_collection_error": "Unverified"}):
        s._selected_validation_model = lambda: model
        steps.refresh()
        assert not s.open_blind_labeling.isEnabled()
        assert s.open_model_correction.isHidden() and s.open_blind_labeling.isHidden()
        s._launch_review("blind_region_labeling")
        assert len(s.calls) == 4
    assert "One active review session" in page.session_notice.text()
    assert "same drawing tools" in page.blind_notice.text()


def test_explicit_precision_does_not_change_the_saved_device_or_open_training_gate(workflow, monkeypatch):
    s, steps, page = workflow.studio, workflow.steps, workflow.workflow
    # This test isolates the device/name gate; session admission has dedicated tests.
    monkeypatch.setattr(page, "run_training", lambda callback: callback())
    steps.training_routes["scratch"] = steps._original_start
    steps.scratch_basis.setChecked(True)
    steps.set_normalisation_ready(True)
    steps.model_name.setText("A new model")
    page.refresh()
    assert s.backend.currentData() == "auto"
    assert page.training_block_reason()
    assert not s.start_button.isEnabled()
    steps.train()
    assert not s.calls
    s.backend.setCurrentIndex(s.backend.findData("cuda"))
    steps.refresh()
    page.refresh()
    assert not page.training_block_reason()
    assert s.start_button.isEnabled()
    steps.train()
    assert s.calls == ["original-training"]
    steps.model_name.clear()
    steps.refresh()
    page.refresh()
    assert not s.start_button.isEnabled()
    steps.train()
    assert s.calls == ["original-training"]
    assert "float32" in page.precision_limit.text()
    assert "newly computed float64 phase maps" in page.precision_limit.text()
    assert s.training_process is None


def test_continuation_explains_inherited_values_instead_of_offering_ignored_settings(workflow):
    s, steps, page = workflow.studio, workflow.steps, workflow.workflow
    s.backend.setCurrentIndex(s.backend.findData("cuda"))
    steps.existing_basis.setChecked(True)
    page.refresh()
    assert s.backend.currentData() == "cuda"
    assert not s.backend.isEnabled()
    assert not s.stage_workers.isEnabled()
    assert not s.pixels.isEnabled()
    assert "CPU float64" in page.precision.text()
    assert "parent's scientific and worker settings" in page.parameter_notice.text()
    steps.scratch_basis.setChecked(True)
    page.refresh()
    assert s.stage_workers.isEnabled() and s.pixels.isEnabled()
    assert s.backend.currentData() == "cuda"


def test_archive_filter_keeps_full_catalogue_for_original_consumers(workflow, monkeypatch, tmp_path):
    from darsia.fluidflower import native_phase_model_archive as archive
    s, steps, page = workflow.studio, workflow.steps, workflow.workflow
    rows = [{"root": "active", "model_id": "current"}, {"root": "old", "model_id": "old"}]
    s.model_history = rows
    s.validation_model.clear()
    for row in rows:
        s.validation_model.addItem(row["model_id"], row["root"])
    seen = []
    s._populate_validation_models = lambda: seen.append(list(s.model_history))
    page._install_archive_filter()
    steps.series_root = tmp_path
    page._saved_context = str(tmp_path)
    monkeypatch.setattr(archive, "read_archive_snapshot", lambda *a, **k: {"models": []})
    monkeypatch.setattr(archive, "filter_snapshot", lambda *a, **k: rows[:1])
    s._populate_validation_models()
    # A binding-only reader launched inside the callback receives every entry.
    assert seen == [rows]
    assert s.model_history is rows
    assert s.validation_model.view().isRowHidden(1)
    assert not s.validation_model.model().item(1).flags() & Qt.ItemFlag.ItemIsEnabled

    def broken(*a, **k):
        raise RuntimeError("Series choice changed")

    monkeypatch.setattr(archive, "filter_snapshot", broken)
    s._populate_validation_models()
    assert seen[-1] == rows
    assert "all model entries remain visible" in page.archive_notice.text()
    assert s.model_history is rows
    assert not s.validation_model.view().isRowHidden(1)
    assert s.validation_model.model().item(1).flags() & Qt.ItemFlag.ItemIsEnabled


def test_removing_an_input_preserves_saved_evidence_and_original_parser_field(workflow, tmp_path):
    s, page = workflow.studio, workflow.workflow
    evidence = tmp_path / "pairwise_review_decisions.json"
    evidence.write_text('{"decisions": {}}', encoding="utf-8")
    before = evidence.read_bytes()
    row = f"{evidence} | manifest.json"
    s.bundles.setPlainText(row + "\nother.json | other-manifest.json")
    page.refresh()
    page.selected_evidence.setCurrentRow(0)
    page.remove_evidence.click()
    assert s.bundles.toPlainText() == "other.json | other-manifest.json"
    assert evidence.read_bytes() == before
    page.refresh()
    page.selected_evidence.setCurrentRow(0)
    s.training_process = object()
    page._remove_evidence()
    assert s.bundles.toPlainText() == "other.json | other-manifest.json"


def test_active_continuation_parent_remains_protected_after_model_selection_changes(workflow, tmp_path):
    s, page = workflow.studio, workflow.workflow
    parent = tmp_path / "parent-model"
    s.training_process = object()
    s.training_request = {"output_root": str(tmp_path / "new-model"),
                          "continuation": {"parent_request": {"path": str(parent / "training_request.json")}}}
    assert str(parent) in page.protected_roots()


def test_training_summary_counts_only_explicit_distinct_inputs(tmp_path):
    chosen, unchosen = str(tmp_path / "chosen.json"), str(tmp_path / "other.json")
    sessions = [dict(decision_path=chosen, manual_regions=320, blind_region_labels=237),
                dict(decision_path=unchosen, manual_regions=10_000, blind_region_labels=5_000)]
    summary = ui.training_summary(f"{chosen} | manifest.json\n{chosen} | manifest.json", sessions)
    assert "320 phase labels and 237 blind validations" in summary
    assert "10,000" not in summary
    assert "held-out-evidence checks" in summary
    assert "No saved review evidence" in ui.training_summary("", sessions)
    assert "incomplete" in ui.training_summary(chosen, sessions)
    assert "refresh saved reviews" in ui.training_summary("unknown.json | manifest.json", sessions)
    assert "unavailable" in ui.training_summary(f"{chosen} | manifest.json", [{"decision_path": chosen}])


def test_precision_provenance_does_not_infer_precision_from_model_name_or_backend():
    assert "not recorded" in ui.recorded_precision_text({"model_id": "float64-reference"})
    text = ui.recorded_precision_text({"heads": [
        dict(adapter="generic", head="co2_presence", backend="cpu", precision="float64"),
        dict(adapter="generic", head="gas_within_co2", backend="cuda", precision="float32"),
        dict(adapter="other", head="gas_within_co2", backend="cpu"),
    ]})
    assert "co2_presence: float64 (cpu)" in text
    assert "gas_within_co2: float32 (cuda)" in text
    assert "not recorded (cpu)" in text


def test_folded_training_log_shows_its_latest_line_and_stays_small(workflow):
    log, page = workflow.studio.log, workflow.workflow
    assert page.log_latest.text() == "No training output yet." and not page.log_group.isChecked()
    log.append("Preparing training images")
    log.append("Epoch 3 of 10")
    log.append("")
    assert page.log_latest.text() == "Latest: Epoch 3 of 10"
    assert log.maximumHeight() == ui.LOG_MAX_HEIGHT
