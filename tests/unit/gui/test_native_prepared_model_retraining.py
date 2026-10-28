from types import SimpleNamespace as NS

from PySide6.QtWidgets import QCheckBox, QSpinBox

from .test_native_phase_model_workflow import workflow  # noqa: F401


def test_train_existing_reuses_saved_labels_on_current_setup_without_new_review(workflow, qtbot, tmp_path, monkeypatch):
    from darsia.fluidflower import calibration_preparation, calibration_model_training
    steps, studio = workflow.steps, workflow.studio
    (tmp_path / "series.toml").write_text('schema_version=1\nseries_id="sample"\n[runs]\nids=[]\n')
    steps.series_root, steps.series_id = tmp_path, "sample"
    template = str(tmp_path / "existing-model")
    studio._selected_validation_model = lambda: dict(root=template, model_id="existing")
    studio.bundles.setPlainText("")
    automatic, cpu = QCheckBox(), QSpinBox()
    automatic.setChecked(True)
    cpu.setValue(6)
    jobs = NS(registry=NS(jobs=lambda root, series: [dict(root=root, series=series)]),
              shared_jobs=NS(automatic=automatic, cpu=cpu))
    monkeypatch.setattr(steps.prepared_retraining, "window_owner", lambda: NS(ensure_jobs=lambda: jobs))
    prepared, launched = [], []
    result = dict(request=dict(output_root=str(tmp_path / "new-model")),
                  bundles=[("projected-old-labels", "current-images")])
    def prepare(root, series, registrations, **options):
        prepared.append((root, series, registrations, options))
        return result
    monkeypatch.setattr(calibration_preparation, "prepare_calibration_training", prepare)
    def launch(actual_steps, actual_result, *, name):
        launched.append((actual_steps, actual_result, name))
        studio.training_process = NS()
    monkeypatch.setattr(calibration_model_training, "launch_prepared_training", launch)
    steps.set_normalisation_ready(True)
    steps.model_name.setText("Current Setup version")
    steps.existing_basis.setChecked(True)
    steps.refresh()
    assert studio.start_button.isEnabled()
    assert "No new boxes or review session" in steps.basis_note.text()
    assert "current Setup" in steps.training_note.text()
    assert workflow.workflow.selected_evidence.isHidden()
    studio.start_button.click()
    qtbot.waitUntil(lambda: bool(launched))
    assert prepared[0][3]["template_reference"] == template
    assert prepared[0][3]["workers"] == 0
    assert launched == [(steps, result, "Current Setup version")]
    assert studio.bundles.toPlainText() == ""
    assert "saved labels" in studio.status.text() and "No new boxes" in studio.status.text()
    assert steps.prepared_retraining.can_close()
    studio.training_process = None
