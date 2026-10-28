"""Explicit model intent controls the existing phase workflow without launching it."""

from .test_native_phase_steps import setup  # noqa: F401
from .test_native_phase_panel_navigation import navigation  # noqa: F401


def test_three_model_choices_disable_training_and_review_without_discarding_inputs(setup, tmp_path):
    steps, studio = setup.steps, setup.studio
    steps.set_context(tmp_path, "series")
    steps.set_normalisation_ready(True)
    steps.model_name.setText("Retained new version")
    initial = list(studio.calls)
    assert [button.text() for button in steps.basis_choices.buttons()] == [
        "New model from new labels", "Retrain on current image correction",
        "Continue with new labels", "Select model for calibration"]
    steps.select_basis.click()
    for section in steps.phase_tabs.sections:
        assert section.isEnabled() is (section.content is steps.groups[0])
    assert steps.phase_tabs.currentWidget() is steps.groups[0]
    assert not studio.start_button.isEnabled()
    assert not studio.open_model_correction.isEnabled()
    steps.train()
    steps._launch_review("open_correction")
    assert studio.calls == initial
    assert steps.model_name.text() == "Retained new version"
    assert studio.original_parameter.text() == "Original numerical setting untouched"
    steps.existing_basis.click()
    assert all(section.isEnabled() for section in steps.phase_tabs.sections)
    assert studio.calls == initial


def test_selecting_an_existing_model_requests_only_its_exact_reference(setup, tmp_path):
    steps, studio = setup.steps, setup.studio
    steps.set_context(tmp_path, "series")
    model = tmp_path / "models" / "saved-model"
    studio._selected_validation_model = lambda: {"root": str(model), "model_id": "saved-model"}
    requested = []
    steps.model_for_calibration_requested.connect(requested.append)
    initial = list(studio.calls)
    steps.select_basis.click()
    assert not steps._ready  # A method/model choice does not require a completed mass calibration.
    assert steps.use_selected_model.isEnabled()
    steps.use_selected_model.click()
    assert requested == [str(model)]
    assert not steps.use_selected_model.isEnabled()
    assert "Saving" in steps.model_selection_status.text()
    assert studio.calls == initial
    assert not list(tmp_path.iterdir())
    steps.set_model_selection_result(True)
    assert "Model selected for calibration" in steps.model_selection_status.text()
    assert steps.use_selected_model.isEnabled()
    steps.use_selected_model.click()
    steps.set_model_selection_result(False, "series changed")
    assert "series changed" in steps.model_selection_status.text()


def test_no_model_means_no_selection_request(setup, tmp_path):
    steps = setup.steps
    steps.set_context(tmp_path, "series")
    steps.studio._selected_validation_model = lambda: None
    requested = []
    steps.model_for_calibration_requested.connect(requested.append)
    steps.select_basis.click()
    steps._request_model_for_calibration()
    assert not steps.use_selected_model.isEnabled()
    assert requested == []


def test_select_mode_cancels_pending_review_and_blocks_programmatic_panel_navigation(navigation):
    n = navigation
    n.state.reason = "The saved model is still loading"
    n.controller.open("open_correction")
    assert n.controller._pending is not None
    n.steps.select_basis.click()
    n.controller._try_open()
    n.controller.open("blind_region_labeling")
    n.controller.show_training()
    assert n.controller._pending is None
    assert n.state.calls == []
    assert n.steps.phase_tabs.currentWidget() is n.steps.groups[0]


def test_retrain_never_continues_and_continue_never_retrains(setup, tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    steps = setup.steps
    # Only the route choice is under test, not the review owner's launch.
    monkeypatch.setattr(steps.phase_workflow, "run_training", lambda callback, **_kw: callback())
    monkeypatch.setattr(steps.phase_workflow, "training_block_reason", lambda: "")
    steps.set_context(tmp_path, "series")
    steps.set_normalisation_ready(True)
    steps.model_name.setText("New version")
    calls = []
    steps.training_routes["existing"] = lambda: calls.append("continue")
    steps.prepared_retraining = NS(available=lambda: True, start=lambda: calls.append("retrain"), busy=False)
    assert steps.existing_choice() == "retrain" and steps.existing_basis.isChecked()
    steps.train()
    steps.continue_basis.click()
    assert steps.existing_basis.isChecked() and steps.existing_choice() == "continue"
    steps.train()
    steps.prepared_retraining = NS(available=lambda: False, start=lambda: calls.append("retrain"), busy=False)
    steps.retrain_basis.click()
    steps.train()  # Nothing prepared to retrain: continuation is never used instead.
    assert calls == ["retrain", "continue"]
    assert "Continue with new labels" in steps.training_note.text()
    steps.scratch_basis.click()
    assert not steps.existing_basis.isChecked() and steps.existing_choice() == ""
    steps.existing_basis.setChecked(True)  # A direct caller keeps the earlier automatic choice.
    assert not steps.scratch_basis.isChecked() and steps.existing_choice() == ""


def test_retrain_with_a_prepared_model_enables_train_and_starts_prepared_retraining(setup, tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    steps, studio = setup.steps, setup.studio
    monkeypatch.setattr(steps.phase_workflow, "training_block_reason", lambda: "")
    started = []
    steps.prepared_retraining = NS(available=lambda: True, start=lambda: started.append("prepared"),
                                   busy=False, can_close=lambda: True)
    steps.training_routes["existing"] = lambda: started.append("continuation")
    steps.set_context(tmp_path, "series")
    studio._selected_validation_model = lambda: {"root": str(tmp_path / "model"), "model_id": "selected-model"}
    steps.set_normalisation_ready(True)  # A completed Image correction.
    steps.model_name.setText("Retrained version")
    steps.refresh()
    assert steps.existing_choice() == "retrain"
    assert studio.start_button.isEnabled()
    studio.start_button.click()
    assert started == ["prepared"]
