"""The selected named calibration reaches the original phase-two preparation."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from darsia.fluidflower import analysis_preparation_ui as base_ui
from darsia.fluidflower import jobs_panel, production_preparation_ui as ui
from darsia.fluidflower.pipeline import job_inputs
from .test_production_page import calibration


def test_preparation_passes_selected_bytes_to_original_producer(qtbot, tmp_path, monkeypatch):
    panel = jobs_panel.JobsPanel()
    qtbot.addWidget(panel)
    panel.series_root = tmp_path
    chosen = calibration()
    dialog = ui.ProductionPreparationDialog(panel, chosen, storage_profile="lean")
    qtbot.addWidget(dialog)
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{}", encoding="utf-8")
    dialog.inputs.setText(str(inputs))
    dialog.folder.setText(str(tmp_path / "fresh"))
    assert not dialog.stage.isEnabled()
    assert dialog.calibration.isReadOnly()
    assert not dialog._file_buttons[dialog.calibration].isEnabled()
    assert dialog.stage.currentData() == "phase2"
    assert dialog.stored_resolution.currentData() == "saved"
    dialog.stored_resolution.setCurrentIndex(dialog.stored_resolution.findData(1.0))
    assert dialog.profile.currentData() == "lean"
    calls = []
    queued = []
    dialog._run = lambda call: queued.append(call)
    monkeypatch.setattr(ui, "read_named_calibration", lambda root: deepcopy(chosen))
    monkeypatch.setattr(ui.preparation, "job_identity", lambda root: {"precision": "float32"})
    monkeypatch.setattr(ui.preparation, "prepare_phase2_inputs", lambda *a, **k: calls.append((a, k)))
    dialog.prepare()
    assert len(queued) == 1 and calls == []
    queued[0]()  # Synthetic injected preparation, not the real sealer or queue.
    assert calls[0][1]["fitted_calibration"] == chosen["calibration"]
    assert calls[0][1]["job"] == {"precision": "float32"}
    assert calls[0][1]["job_id"] == "production"
    assert calls[0][1]["storage_profile"] == "lean"
    assert calls[0][1]["stored_resolution"] == 1.0
    assert dialog.prepared_result is None
    dialog._closed()
    assert dialog.calibration.isReadOnly()
    assert not dialog._file_buttons[dialog.calibration].isEnabled()
    dialog.prepared_result = {"plan": {"path": "old-plan"}}
    dialog.adopt_button.setEnabled(True)
    dialog.profile.setCurrentIndex(dialog.profile.findData("full"))
    assert dialog.prepared_result is None and not dialog.adopt_button.isEnabled()
    dialog.prepared_result = {"plan": {"path": "old-plan"}}
    dialog.adopt_button.setEnabled(True)
    dialog.stored_resolution.setCurrentIndex(dialog.stored_resolution.findData("saved"))
    assert dialog.prepared_result is None and not dialog.adopt_button.isEnabled()


@pytest.mark.parametrize("changed", [
    "inputs", "folder", "name", "series", "series_id", "phase1_jobs",
    "profile", "stored_resolution",
])
def test_late_preparation_result_cannot_be_adopted_for_changed_inputs(qtbot, tmp_path, changed):
    panel = jobs_panel.JobsPanel()
    qtbot.addWidget(panel)
    panel.series_root = tmp_path
    callbacks = []
    def start(call, ready, closed):
        callbacks.append((call, ready, closed))
        return SimpleNamespace(wait=lambda: None), object()
    panel._start_call = start
    dialog = ui.ProductionPreparationDialog(panel, calibration())
    qtbot.addWidget(dialog)
    dialog.inputs.setText("source-A.json")
    dialog.folder.setText("plan-A")
    result = dict(plan=dict(path="prepared.toml", sha256="a" * 64), adopted_for_execution=False)
    dialog._run(lambda: result)
    assert not dialog.inputs.isEnabled() and not dialog.folder.isEnabled()
    assert not dialog.name.isEnabled() and not dialog.profile.isEnabled()
    assert not dialog.phase1_jobs.isEnabled()
    assert not dialog._file_buttons[dialog.phase1_jobs].isEnabled()
    if changed == "series":
        panel.series_root = tmp_path / "another-series"
    elif changed == "series_id":
        panel.series_id = "another-series-id"
    elif changed in {"profile", "stored_resolution"}:
        field = getattr(dialog, changed)
        field.setCurrentIndex(field.findData("lean" if changed == "profile" else 1.0))
    else:
        # Programmatic changes are possible even while user editing is locked.
        getattr(dialog, changed).setText("choice-B")
    call, ready, closed = callbacks.pop()
    ready(call())
    closed()
    assert dialog.prepared_result is None and not dialog.adopt_button.isEnabled()
    assert "earlier choices are preserved" in dialog.status.text()
    assert dialog.inputs.isEnabled() and dialog.can_close()
    assert dialog.phase1_jobs.isEnabled()
    assert dialog._file_buttons[dialog.phase1_jobs].isEnabled()


@pytest.mark.parametrize("changed", ["name", "phase1_jobs"])
def test_editing_completed_inputs_invalidates_plan_and_late_review_cannot_adopt(qtbot, tmp_path, changed):
    panel = jobs_panel.JobsPanel()
    qtbot.addWidget(panel)
    panel.series_root = tmp_path
    callbacks = []
    def start(call, ready, closed):
        callbacks.append((call, ready, closed))
        return SimpleNamespace(wait=lambda: None), object()
    panel._start_call = start
    dialog = ui.ProductionPreparationDialog(panel, calibration())
    qtbot.addWidget(dialog)
    dialog._run(lambda: dict(plan=dict(path="prepared.toml", sha256="a" * 64),
                            adopted_for_execution=False))
    call, ready, closed = callbacks.pop()
    ready(call())
    closed()
    assert dialog.adopt_button.isEnabled()
    getattr(dialog, changed).setText("changed after review started")
    assert dialog.prepared_result is None and not dialog.adopt_button.isEnabled()
    dialog._run(lambda: pytest.fail("Changed review must not write adoption"), adopting=True)
    assert callbacks == [] and dialog.can_close()


@pytest.fixture
def deferred_production(qtbot, tmp_path, monkeypatch):
    """Keep the real dialog lifecycle; replace only its finite runner and producer."""
    panel = jobs_panel.JobsPanel()
    qtbot.addWidget(panel)
    panel.series_root, panel.series_id = tmp_path, "series-one"
    callbacks, calls, descriptor_calls = [], [], []

    def start(call, ready, closed):
        callbacks.append((call, ready, closed))
        return SimpleNamespace(wait=lambda: None), object()

    panel._start_call = start
    chosen = calibration()
    dialog = ui.ProductionPreparationDialog(panel, chosen)
    qtbot.addWidget(dialog)
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{}", encoding="utf-8")
    folder = tmp_path / "fresh"
    dialog.inputs.setText(str(inputs))
    dialog.folder.setText(str(folder))
    identity = dict(commit="b" * 40, precision="float32")
    prepared = dict(plan=dict(path=str(folder / "execution.toml"), sha256="c" * 64),
                    adopted_for_execution=False, worker_started=False)

    def producer(*args, **kwargs):
        calls.append((args, kwargs))
        return deepcopy(prepared)  # Inert result; no plan, queue or worker is created.

    source_descriptor = ui.preparation.source_metadata_descriptor

    def bind_source(path):
        descriptor_calls.append(path)
        return source_descriptor(path)

    monkeypatch.setattr(ui, "read_named_calibration", lambda root: deepcopy(chosen))
    monkeypatch.setattr(ui.preparation, "job_identity", lambda root: deepcopy(identity))
    monkeypatch.setattr(ui.preparation, "prepare_phase2_inputs", producer)
    monkeypatch.setattr(ui.preparation, "source_metadata_descriptor", bind_source)

    def finish():
        call, ready, closed = callbacks.pop(0)
        try:
            result = call()
            ready(result)
            return result
        finally:
            closed()

    return SimpleNamespace(panel=panel, dialog=dialog, chosen=chosen, inputs=inputs,
        folder=folder, identity=identity, prepared=prepared, callbacks=callbacks,
        calls=calls, descriptor_calls=descriptor_calls, finish=finish)


@pytest.mark.parametrize("with_origin", [False, True])
@pytest.mark.parametrize("profile,resolution", [("full", "saved"), ("lean", 1.0)])
def test_optional_original_jobs_reach_deferred_producer_with_storage_choices(
        deferred_production, tmp_path, with_origin, profile, resolution):
    case, dialog = deferred_production, deferred_production.dialog
    selected = tmp_path / "original-jobs.json"
    selected.write_text("{}", encoding="utf-8")
    dialog.phase1_jobs.setText(str(selected) if with_origin else "")
    dialog.profile.setCurrentIndex(dialog.profile.findData(profile))
    dialog.stored_resolution.setCurrentIndex(dialog.stored_resolution.findData(resolution))
    dialog.prepare()
    assert len(case.callbacks) == 1 and case.calls == [] and case.descriptor_calls == []
    outcome = case.finish()
    assert "error" not in outcome and len(case.calls) == 1
    assert case.descriptor_calls == ([str(selected)] if with_origin else [])
    args, kwargs = case.calls[0]
    assert args == (ui.descriptor(case.inputs), str(case.folder))
    assert kwargs == dict(series_root=tmp_path, fitted_calibration=case.chosen["calibration"],
        job_id="production", job=case.identity, storage_profile=profile,
        stored_resolution=resolution,
        phase1_jobs=ui.descriptor(selected) if with_origin else None)
    assert dialog.prepared_result == case.prepared and dialog.adopt_button.isEnabled()
    assert not case.folder.exists()


def test_original_jobs_path_is_captured_before_deferred_callback(deferred_production, tmp_path):
    case, dialog = deferred_production, deferred_production.dialog
    original, replacement = tmp_path / "original-jobs.json", tmp_path / "replacement-jobs.json"
    original.write_text("{}", encoding="utf-8")
    replacement.write_text('{"different":true}', encoding="utf-8")
    dialog.phase1_jobs.setText(str(original))
    dialog.prepare()
    assert case.calls == [] and case.descriptor_calls == []
    dialog.phase1_jobs.setText(str(replacement))  # Programmatic edits can bypass disabled widgets.
    case.finish()
    assert len(case.calls) == 1
    assert case.descriptor_calls == [str(original)]
    assert case.calls[0][1]["phase1_jobs"] == ui.descriptor(original)
    assert dialog.prepared_result is None and not dialog.adopt_button.isEnabled()
    assert "earlier choices are preserved" in dialog.status.text()
    assert not case.folder.exists()


def test_original_jobs_controls_lock_until_preparation_and_adoption_retire(
        deferred_production, monkeypatch):
    case, dialog = deferred_production, deferred_production.dialog
    chooser = dialog._file_buttons[dialog.phase1_jobs]
    dialog.prepare()
    assert not dialog.phase1_jobs.isEnabled() and not chooser.isEnabled()
    assert not dialog.can_close()
    case.finish()
    assert dialog.phase1_jobs.isEnabled() and chooser.isEnabled() and dialog.can_close()
    assert dialog.calibration.isReadOnly() and not dialog.stage.isEnabled()
    assert not dialog._file_buttons[dialog.calibration].isEnabled()

    adoptions = []

    def adopt(*args, **kwargs):
        adoptions.append((args, kwargs))
        return dict(case.prepared, adopted_for_execution=True)

    monkeypatch.setattr(base_ui.QInputDialog, "getText", lambda *a, **k: ("test reviewer", True))
    monkeypatch.setattr(base_ui.QMessageBox, "question", lambda *a, **k: base_ui.QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(base_ui, "adopt_prepared_plan", adopt)
    dialog.adopt()
    assert len(case.callbacks) == 1 and adoptions == []
    assert not dialog.phase1_jobs.isEnabled() and not chooser.isEnabled()
    assert not dialog.can_close()
    case.finish()
    assert adoptions == [((case.folder / "execution.toml", case.folder / "execution.adopted.toml"),
        dict(expected_sha256=case.prepared["plan"]["sha256"], actor="test reviewer", reason="test reviewer"))]
    assert dialog.phase1_jobs.isEnabled() and chooser.isEnabled() and dialog.can_close()
    assert dialog.prepared_result["adopted_for_execution"] and not dialog.adopt_button.isEnabled()
    assert dialog.calibration.isReadOnly() and not dialog._file_buttons[dialog.calibration].isEnabled()
    assert not case.folder.exists()


@pytest.mark.parametrize("invalid", ["missing", "non_json", "oversized"])
def test_invalid_original_jobs_metadata_refuses_before_producer(
        deferred_production, tmp_path, monkeypatch, invalid):
    case, dialog = deferred_production, deferred_production.dialog
    selected = tmp_path / ("original-jobs.txt" if invalid == "non_json" else "original-jobs.json")
    if invalid != "missing":
        selected.write_text("{}" + " " * 8, encoding="utf-8")
    if invalid == "oversized":
        monkeypatch.setattr(job_inputs, "MAX_METADATA_BYTES", 8)
    dialog.phase1_jobs.setText(str(selected))
    dialog.prepare()
    assert len(case.callbacks) == 1 and case.calls == [] and case.descriptor_calls == []
    outcome = case.finish()
    assert "error" in outcome and case.calls == []
    assert case.descriptor_calls == [str(selected)]
    expected = "FileNotFoundError" if invalid == "missing" else "existing bounded source-metadata JSON file"
    assert expected in outcome["error"]
    assert dialog.prepared_result is None and not dialog.adopt_button.isEnabled()
    assert dialog.phase1_jobs.isEnabled() and dialog._file_buttons[dialog.phase1_jobs].isEnabled()
    assert dialog.can_close() and not case.folder.exists()
