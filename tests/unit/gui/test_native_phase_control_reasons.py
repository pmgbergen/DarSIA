"""Native phase actions expose the exact selection and preserve launch gates."""
from types import SimpleNamespace as NS

import pytest
from darsia.fluidflower import native_phase_steps as ui
from darsia.fluidflower.native_darsia_app import NativeFluidFlowerMainWindow
from darsia.fluidflower.normalisation_step import NormalisationStep

from .test_native_phase_steps import FakeNormalisation, FakeStudio


@pytest.fixture
def page(qtbot):
    studio = FakeStudio()
    selection = NS(model=None, run="experiment-a")
    studio._selected_validation_model = lambda: selection.model
    studio._request_async_model_refresh = lambda **kw: studio.calls.append(("saved-metadata", kw))
    normalisation = FakeNormalisation()
    steps = ui.NativePhaseSteps(studio, normalisation=normalisation,
        run_provider=lambda: selection.run)
    steps.series_id = "campaign"
    studio._active_model_series = "campaign"
    qtbot.addWidget(studio)
    qtbot.addWidget(steps)
    return NS(studio=studio, steps=steps, selection=selection, normalisation=normalisation)


def selected(page, **changes):
    page.selection.model = dict(model_id="model-alpha", root="saved-model", **changes)
    page.steps.set_normalisation_ready(True)
    return page.selection.model


def test_missing_maps_name_the_model_series_and_run_and_offer_only_saved_metadata(page):
    selected(page)
    page.studio.open_model_correction.setEnabled(False)
    page.steps._legacy_availability["open_model_correction"] = False
    page.steps.refresh()
    text = page.steps.review_selection_notice.text()
    assert all(value in text for value in ("model-alpha", "campaign", "experiment-a", "no prepared phase maps"))
    assert page.studio.open_model_correction.toolTip() == text
    page.steps.phase_review = lambda: None
    page.steps.refresh()
    assert page.steps.phase_review_button.toolTip() == text
    page.steps.review_route_button.click()
    assert page.studio.calls == [("saved-metadata", {"bindings_only": True, "verify_saved": True})]
    assert page.studio.training_process is None
    assert not page.studio.open_model_correction.isEnabled()


def test_missing_model_route_opens_existing_basis_selector_without_selecting_a_model(page):
    page.steps.set_normalisation_ready(True)
    page.steps.phase_tabs.setCurrentWidget(page.steps.groups[3])
    page.steps.review_route_button.click()
    assert page.steps.phase_tabs.currentWidget() is page.steps.groups[0]
    assert page.selection.model is None
    assert page.studio.calls == []


def test_missing_normalisation_has_visible_reason_and_direct_route(page):
    page.steps.refresh()
    assert "Image correction" in page.steps.review_selection_notice.text()
    assert "Image correction" in page.steps.import_button.toolTip()
    page.steps.review_route_button.click()
    assert page.steps.section == "normalisation"
    assert not page.studio.open_model_correction.isEnabled()


@pytest.mark.parametrize("fields, text", [
    ({"validation_binding_loading": True}, "still being checked"),
    ({"validation_collection_error": "Saved source SHA differs"}, "Saved source SHA differs"),
])
def test_phase_window_never_opens_an_unverified_or_other_run_binding(page, fields, text):
    selected(page, **fields)
    host = NS(phase_model_workspace=NS(studio_page=NS(studio=page.studio, native_steps=page.steps), series_id="campaign"))
    NativeFluidFlowerMainWindow._open_existing_phase_review(host)
    assert text in page.studio.review_status.text()
    assert page.steps.phase_tabs.currentWidget() is page.steps.groups[3]
    assert not page.studio.calls


def test_phase_window_uses_existing_review_route_when_exact_selection_is_ready(page):
    selected(page, validation_binding={"selected_candidates": {"experiment-a": "saved-candidate"}})
    page.studio._active_model_series = "CAM_PAIGN"
    host = NS(phase_model_workspace=NS(studio_page=NS(studio=page.studio, native_steps=page.steps), series_id="campaign"))
    NativeFluidFlowerMainWindow._open_existing_phase_review(host)
    assert page.studio.calls == [("original-review", "open_correction")]
    assert page.studio.training_process is None


def test_exact_saved_review_opens_without_reapplying_current_normalisation(page):
    selected(page, validation_binding={"selected_candidates": {"experiment-a": "saved-candidate"}})
    page.steps.set_normalisation_ready(False)
    assert not page.steps.review_launch_reason()
    assert page.steps.review_route_button.isHidden()
    page.steps._launch_review("open_correction")
    assert page.studio.calls == [("original-review", "open_correction")]
    # Viewing and labelling bound photographs does not make a new fit ready.
    page.steps.train()
    assert page.studio.calls == [("original-review", "open_correction")]
    assert page.studio.status.text() == "Complete Image correction with Finish setup in Setup."


def test_run_change_refreshes_the_reason_without_rebinding_saved_maps(page):
    model = selected(page, validation_binding={"selected_candidates": {"experiment-a": "saved-candidate"}})
    assert not page.steps.review_launch_reason()
    page.selection.run = "experiment-b"
    host = NS(phase_model_workspace=NS(studio_page=NS(native_steps=page.steps)))
    NativeFluidFlowerMainWindow._refresh_phase_review_requirements(host)
    assert "experiment-b" in page.steps.review_selection_notice.text()
    assert model["validation_binding"]["selected_candidates"] == {"experiment-a": "saved-candidate"}
    assert page.studio.calls == []


def test_series_review_keeps_available_runs_when_the_hidden_run_selector_differs(page):
    selected(page, validation_binding={"selected_candidates": {"experiment-b": "saved-candidate"}})
    page.steps.refresh()
    assert "Available runs: experiment-b" in page.steps.review_selection_notice.text()
    assert "Current run selection: experiment-a" in page.steps.review_selection_notice.text()
    host = NS(phase_model_workspace=NS(studio_page=NS(studio=page.studio, native_steps=page.steps), series_id="campaign"))
    NativeFluidFlowerMainWindow._open_existing_phase_review(host)
    assert page.studio.calls == [("original-review", "open_correction")]


def test_phase_window_callback_error_is_visible_in_words(page, monkeypatch):
    errors = []
    monkeypatch.setattr(ui.QMessageBox, "warning", lambda _parent, _title, text: errors.append(text))
    def fail():
        raise OSError("Saved phase-map folder is unavailable")
    page.steps.phase_review = fail
    page.steps.phase_review_button.setEnabled(True)
    page.steps.phase_review_button.clicked.connect(page.steps._open_phase_review)
    page.steps.phase_review_button.click()
    assert errors == ["Could not open phase review: Saved phase-map folder is unavailable"]
    assert page.studio.review_status.text() == errors[0]


def test_review_and_label_callback_error_is_visible_without_changing_the_gate(page, monkeypatch):
    selected(page, validation_binding={})
    errors = []
    monkeypatch.setattr(ui.QMessageBox, "warning", lambda _parent, _title, text: errors.append(text))
    def fail(_mode):
        raise OSError("Review launcher could not read its saved manifest")
    page.steps._original_review = fail
    page.studio.open_model_correction.click()
    assert errors == ["Could not open phase review: Review launcher could not read its saved manifest"]
    assert page.steps.review_selection_notice.text() == errors[0]


def test_saved_map_recheck_error_is_visible_and_never_starts_a_job(page, monkeypatch):
    selected(page)
    errors = []
    monkeypatch.setattr(ui.QMessageBox, "warning", lambda _parent, _title, text: errors.append(text))
    def fail(**_kwargs):
        raise OSError("Saved metadata location is unavailable")
    page.studio._request_async_model_refresh = fail
    page.steps.review_route_button.click()
    assert errors == ["Could not check saved phase maps: Saved metadata location is unavailable"]
    assert page.studio.review_status.text() == errors[0]
    assert page.studio.training_process is None


def test_training_and_saved_session_controls_explain_their_existing_gates(page):
    page.steps._legacy_availability["resume_session_button"] = False
    page.steps.refresh()
    assert "Image correction" in page.studio.resume_session_button.toolTip()
    assert "name the new model version" in page.studio.start_button.toolTip()
    page.steps.set_normalisation_ready(True)
    assert "Select an available saved review session" in page.studio.resume_session_button.toolTip()
    # The default retrains; it never falls back to continuation.
    assert "Retraining needs" in page.studio.start_button.toolTip()
    page.steps.continue_basis.click()
    assert "compatible existing-model training route" in page.studio.start_button.toolTip()


def test_normalisation_disabled_controls_explain_source_and_override_requirements(qtbot):
    step = NormalisationStep(provider=lambda *_args: None)
    qtbot.addWidget(step)
    assert "Choose an experimental series" in step.override.toolTip()
    assert "Choose an experimental series" in step.save_button.toolTip()
    step.series_root, step.series_id = "unused", "campaign"
    step._update_controls()
    assert "Open saved lighting results" in step.reference.toolTip()
    step.snapshot, step.manifest_bytes = object(), b"saved"
    step._update_controls()
    assert "Enable Choose another historical reference" in step.reference.toolTip()


def test_image_correction_reason_says_checking_unreadable_or_incomplete(page):
    steps = page.steps
    steps._prepared_reading, steps._prepared_state = ("root", "campaign"), None
    assert "still being checked" in steps._image_correction_reason()
    steps._prepared_state = dict(kind="error", text="OSError: share unavailable")
    assert steps._image_correction_reason().endswith("could not be read: OSError: share unavailable")
    steps._prepared_state = dict(kind="incomplete", text="No completed image correction for these saved settings.")
    assert steps._image_correction_reason() == (
        "Image correction is not complete: No completed image correction for these saved settings")
    steps._prepared_reading = steps._prepared_state = None
    assert steps._image_correction_reason() == "complete Image correction with Finish setup in Setup"


def test_start_up_check_says_checking_with_a_busy_bar_until_it_answers(page):
    steps, studio = page.steps, page.studio
    studio.status.setText("")
    steps._prepared_reading, steps._prepared_state = ("root", "campaign"), None
    steps.refresh()
    assert studio.status.text().startswith("Checking — the completed Image correction is still being checked")
    assert not steps.check_busy.isHidden() and not studio.start_button.isEnabled()
    assert studio.start_button.toolTip().startswith("the completed Image correction is still being checked")
    assert studio.resume_session_button.toolTip().startswith("The completed Image correction is still being checked")
    steps._prepared_reading = None
    steps._prepared_state = dict(kind="incomplete", text="No completed image correction for these saved settings.")
    steps.refresh()
    assert studio.status.text().startswith(
        "Not done — Image correction is not complete: No completed image correction for these saved settings.")
    assert steps.check_busy.isHidden()
    steps._prepared_state, steps._prepared_ready = dict(kind="complete", text=""), True
    steps.set_normalisation_ready(False)
    assert studio.status.text() == steps.training_note.text() and steps.check_busy.isHidden()


def test_the_busy_bar_also_shows_while_the_training_inputs_are_checked(page):
    steps = page.steps
    photometry = NS(checking=True, binding=None, reason="Checking the selected training inputs.",
                    refresh=lambda: None, fresh_ui_ready=lambda: False,
                    uses_prepared_raw_basis=False, train=lambda: None)
    steps.training_photometry = photometry
    steps.scratch_basis.setChecked(True)
    steps.set_normalisation_ready(True)
    assert not steps.check_busy.isHidden()
    assert steps.check_busy.toolTip() == "Checking the selected training inputs"
    photometry.checking = False
    steps.refresh()
    assert steps.check_busy.isHidden()


def test_the_saved_method_is_parsed_again_only_when_series_toml_changes(page, monkeypatch, tmp_path):
    from darsia.fluidflower import series_method
    manifest = tmp_path / "series.toml"
    manifest.write_text('series_id = "campaign"\n', encoding="utf-8")
    calls = []
    original = series_method.read_series_method
    monkeypatch.setattr(series_method, "read_series_method",
                        lambda root: calls.append(root) or original(root))
    steps = page.steps
    steps.series_root = tmp_path
    for _ in range(3):
        steps.refresh()
    assert calls == [tmp_path]
    manifest.write_text('series_id = "campaign"\n[method]\nactive = "phase_model"\n', encoding="utf-8")
    steps.refresh()
    assert calls == [tmp_path, tmp_path]


def test_missing_review_maps_say_that_training_prepares_them(page):
    selected(page)
    reason = page.steps.review_launch_reason()
    assert "Train model version prepares them" in reason and "blind validation follows training" in reason


def test_prepared_answer_stays_while_the_same_series_is_checked_again(page, qtbot, monkeypatch, tmp_path):
    import threading
    from darsia.fluidflower import image_preparation_status
    (tmp_path / "series.toml").write_text('series_id = "campaign"\n', encoding="utf-8")
    calls, release = [], threading.Event()

    class Source:
        def __call__(self, root, series, original):
            calls.append(series)
            assert release.wait(10)
            return dict(complete=True, text="Image correction completed for 2 runs.")
    monkeypatch.setattr(image_preparation_status, "PreparationStatusSource", Source)
    steps = page.steps
    steps.series_root, steps.series_id = tmp_path, "campaign"
    release.set()
    steps._refresh_prepared_basis()
    qtbot.waitUntil(lambda: steps._prepared_ready, timeout=10000)
    steps._refresh_prepared_basis()  # A recent answer is kept; nothing is read again.
    assert calls == ["campaign"] and steps._ready
    release.clear()
    steps._prepared_checked_at = None
    steps._refresh_prepared_basis()
    steps._refresh_prepared_basis()  # The running reading answers this too.
    assert steps._prepared_ready and steps._ready  # Kept while it is checked again.
    release.set()
    qtbot.waitUntil(lambda: steps._prepared_reading is None, timeout=10000)
    assert calls == ["campaign", "campaign"]
    steps.series_id = "other"
    steps._refresh_prepared_basis()
    assert not steps._prepared_ready


def test_a_model_without_a_current_training_request_is_on_an_older_image_basis(tmp_path, monkeypatch):
    from darsia.fluidflower import calibration_model_training
    model = tmp_path / "model"
    model.mkdir()
    assert "no saved training request" in ui.model_basis_reason(model, tmp_path)
    (model / "training_request.json").write_text('{"settings": {}}', encoding="utf-8")
    assert ui.model_basis_reason(model, tmp_path) == "Saved normalisation is required before launch"
    monkeypatch.setattr(calibration_model_training, "require_prepared_normalisation", lambda request, root: {})
    assert ui.model_basis_reason(model, tmp_path) == ""


def test_review_pages_never_open_a_model_from_an_older_image_basis(page, tmp_path):
    selected(page, validation_binding={"combined_root": "saved"})
    steps = page.steps
    steps.series_root = tmp_path
    steps._model_basis[(str(tmp_path), "saved-model")] = dict(kind="other", text="no request")
    assert ui.OLD_BASIS_REASONS["label"] in steps.review_launch_reason("label")
    assert ui.OLD_BASIS_REASONS["blind"] in steps.review_launch_reason("blind")
    assert ui.OLD_BASIS_REASONS[None] in steps.review_launch_reason()
    steps._model_basis[(str(tmp_path), "saved-model")] = dict(kind="current", text="")
    assert ui.OLD_BASIS_REASONS["label"] not in steps.review_launch_reason("label")
    steps._model_basis.clear()
    steps._model_basis_reading = (str(tmp_path), "saved-model")  # As if its check were running.
    assert "checking whether the selected model" in steps.review_launch_reason("label")
