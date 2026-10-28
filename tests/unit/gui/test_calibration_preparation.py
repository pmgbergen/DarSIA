from copy import deepcopy
from types import SimpleNamespace
from threading import Event

import pytest

from darsia.fluidflower import calibration_preparation as module
from darsia.fluidflower.refit_job_panel import RefitJobPanel
from .test_native_series_method import method_window  # noqa: F401


def selection():
    counts = dict(selected_frames=5, fit_candidates=4, earlier_history=1)
    return dict(input_authority=dict(current_series=dict(sha256="saved-series"), selected_model=None), counts=counts,
        runs=[dict(run="sample", counts=counts)])


def build(root):
    operation = dict(path=str(root / "operation.json"), sha256="operation", size_bytes=4)
    return dict(root=str(root / "build"), complete=True, errors=[], job_facts=dict(
        pipeline="series-setup", series_id="series", series=dict(sha256="saved-series")),
        runs=[dict(run="sample", operation="build", state="Complete", result=dict(
            files={"operation_result.json": operation}))])


def test_existing_matching_phase_result_is_reused_without_preparing_or_launching(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "require_current_model_basis", lambda *a, **kw: None)
    saved = selection()
    completed = build(tmp_path)
    phase = dict(root=str(tmp_path / "phase"), complete=True, job_facts=dict(
        execution_mode="current_build", series_id="series", expected_runs=["sample"],
        inputs=[dict(run="sample", operation_result=completed["runs"][0]["result"]["files"]["operation_result.json"])],
        selection={"path": "saved-choice"}))
    monkeypatch.setattr(module, "saved_selection", lambda *a, **kw: deepcopy(saved))
    monkeypatch.setattr(module, "selection_preview", lambda *a, **kw: deepcopy(saved))
    producer = lambda *a, **kw: pytest.fail("Matching completed P1 must be reused")
    result = module.prepare_calibration_phase(tmp_path, "series", [completed, phase],
        actor="operator", machine="host", snapshot_reader=lambda row: row,
        reader=SimpleNamespace(json=lambda bound: deepcopy(saved)), producer=producer)
    assert result["snapshot"] is phase and result["reused"]
    assert result["registration"] is None


def test_changed_setup_requires_current_build_before_any_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "require_current_model_basis", lambda *a, **kw: None)
    stale = build(tmp_path)
    stale["job_facts"]["series"]["sha256"] = "old-series"
    monkeypatch.setattr(module, "saved_selection", lambda *a, **kw: selection())
    monkeypatch.setattr(module, "selection_preview", lambda *a, **kw: selection())
    with pytest.raises(ValueError, match="Complete Apply and Build"):
        module.prepare_calibration_phase(tmp_path, "series", [stale], actor="operator",
            machine="host", snapshot_reader=lambda row: row,
            producer=lambda *a, **kw: pytest.fail("No current Build"))


def test_active_new_setup_blocks_old_completed_build_and_leads_when_complete(tmp_path):
    old, current = build(tmp_path), build(tmp_path / "new")
    old["active"], current["active"] = False, True
    current["complete"] = False
    with pytest.raises(ValueError, match="active Apply/Build"):
        module.registered_preparation_snapshots([old, current], "series", lambda row: row)
    current["complete"] = True
    assert module.registered_preparation_snapshots([old, current], "series", lambda row: row) == [current]


def test_setup_preparation_stops_before_mass_even_after_switching_presentation(qtbot, tmp_path, monkeypatch):
    from PySide6.QtCore import QObject, Signal
    class JobsPanel(QObject):
        job_selection_changed = Signal()
    panel = RefitJobPanel(machine="host")
    qtbot.addWidget(panel)
    monkeypatch.setattr(panel, "_create_blocker", lambda **kwargs: "Calibration name not chosen")
    monkeypatch.setattr(panel, "start_after_phase_one", lambda *args: pytest.fail("Setup must not start mass calibration"))
    saved = dict(selection(), setup_colour_paths_ready=True)
    monkeypatch.setattr(module, "selection_preview", lambda *args, **kwargs: deepcopy(saved))
    phase = dict(root=str(tmp_path / "phase"), complete=True,
                 job_facts=dict(expected_runs=["sample"], fresh_optical_paths={"format": "real-producer"}))
    started, release, calls = Event(), Event(), []
    def prepare(*args, **kwargs):
        calls.append(kwargs)
        started.set()
        assert release.wait(5.)
        return dict(snapshot=phase, selection=deepcopy(saved), reused=True, registration=None)
    monkeypatch.setattr(module, "prepare_calibration_phase", prepare)
    jobs_panel = JobsPanel()
    jobs_panel.local_machine, jobs_panel.current = "host", phase
    jobs = SimpleNamespace(series_id="series", registry=SimpleNamespace(jobs=lambda *args: []),
                           panel=jobs_panel, stores=SimpleNamespace(path="stores"))
    window = SimpleNamespace(ensure_jobs=lambda: jobs, open_image_selection_plan=lambda: None,
                            _select_row=lambda *args: None, show_phase_model_workflow=lambda: None)
    widget = module.CalibrationPreparation(window, panel)
    qtbot.addWidget(widget)
    widget.set_preparation_only_view(True)
    widget.show()
    widget.set_series(tmp_path)
    qtbot.waitUntil(lambda: not widget._preview.busy)
    assert widget.start.isEnabled()  # No calibration name/overflow choice required.
    try:
        widget.start.click()
        qtbot.waitUntil(started.is_set)
        widget.set_preparation_only_view(False)
    finally:
        release.set()
    qtbot.waitUntil(lambda: widget._thread is None)
    assert calls[0]["require_fresh_paths"] is True
    assert "Colour paths and phase inputs completed" in widget.status.text()
    assert "not been started" in widget.status.text()


def test_missing_phase_uses_original_producer_with_exact_build_and_choices(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "require_current_model_basis", lambda *a, **kw: None)
    saved, completed, calls = selection(), build(tmp_path), []
    monkeypatch.setattr(module, "saved_selection", lambda *a, **kw: deepcopy(saved))
    monkeypatch.setattr(module, "selection_preview", lambda *a, **kw: deepcopy(saved))
    registration = dict(root=str(tmp_path / "prepared"))
    def producer(snapshot, **kwargs):
        calls.append((snapshot, kwargs))
        return registration
    result = module.prepare_calibration_phase(tmp_path, "series", [completed],
        actor="reviewer", machine="chosen-host", stores_path="chosen-stores", producer=producer,
        snapshot_reader=lambda row: row)
    assert result["registration"] is registration and not result["reused"]
    assert calls[0][0] is completed
    assert calls[0][1]["actor"] == "reviewer"
    assert calls[0][1]["machine"] == "chosen-host"
    assert calls[0][1]["stores_path"] == "chosen-stores"


def test_calibration_page_shows_saved_images_and_starts_existing_chain_without_picker(qtbot, tmp_path, monkeypatch):
    panel = RefitJobPanel(machine="host")
    qtbot.addWidget(panel)
    monkeypatch.setattr(panel, "_create_blocker", lambda **kwargs: "")
    monkeypatch.setattr(module, "saved_selection", lambda *a, **kw: selection())
    monkeypatch.setattr(module, "selection_preview", lambda *a, **kw: selection())
    phase = dict(root=str(tmp_path / "phase"), complete=True,
                 job_facts=dict(expected_runs=["sample"]))
    monkeypatch.setattr(module, "prepare_calibration_phase", lambda *a, **kw: dict(
        snapshot=phase, selection=selection(), reused=True, registration=None))
    calls = []
    monkeypatch.setattr(panel, "start_after_phase_one", lambda *args: calls.append(args))
    jobs = SimpleNamespace(series_id="series", registry=SimpleNamespace(jobs=lambda *args: []),
                           panel=SimpleNamespace(local_machine="host"), stores=SimpleNamespace(path="stores"))
    window = SimpleNamespace(ensure_jobs=lambda: jobs, open_image_selection_plan=lambda: None,
        _select_row=lambda *a: None, show_phase_model_workflow=lambda: None)
    widget = module.CalibrationPreparation(window, panel)
    assert widget.scope is panel.calibration_scope_combo
    widget.scope.setCurrentIndex(1)
    assert panel.calibration_scope_combo.currentData() == "individual"
    monkeypatch.setattr(widget, "_basis_blocker", lambda: "")
    qtbot.addWidget(widget)
    widget.show()
    widget.set_series(tmp_path)
    qtbot.waitUntil(lambda: widget._thread is None and not widget._preview.busy)
    assert widget.table.rowCount() == 1
    assert "5 selected images" in widget.summary.text()
    assert "Keep DarSIA open" in widget.summary.text()
    assert widget.start.isEnabled()
    widget.start.click()
    qtbot.waitUntil(lambda: widget._thread is None and not widget._preview.busy)
    assert calls == [(phase["root"], ["sample"])]
    widget.set_series(None)
    assert widget.isHidden() and not widget.start.isEnabled()


def test_missing_current_model_binding_blocks_both_page_and_entry(qtbot, tmp_path, monkeypatch):
    from darsia.fluidflower.series_method import MethodSelection, save_series_method
    (tmp_path / "series.toml").write_text('schema_version=1\nseries_id="sample"\n')
    save_series_method(tmp_path, MethodSelection(active="phase_model", model_version="selected-model",
                                                mass_calibration="saved-calibration.json"))
    monkeypatch.setattr(module, "saved_selection", lambda *a, **kw: selection())
    monkeypatch.setattr(module, "selection_preview", lambda *a, **kw: selection())
    with pytest.raises(ValueError, match="not yet bound"):
        module.prepare_calibration_phase(tmp_path, "sample", [], actor="reviewer", machine="host",
            snapshot_reader=lambda *a: pytest.fail("Unbound model must not reach job preparation"))
    panel = RefitJobPanel(machine="host")
    qtbot.addWidget(panel)
    navigation = []
    window = SimpleNamespace(_select_row=lambda value: navigation.append(value),
        open_image_selection_plan=lambda: None, show_phase_model_workflow=lambda: navigation.append("model"))
    widget = module.CalibrationPreparation(window, panel)
    qtbot.addWidget(widget)
    widget.set_series(tmp_path)
    qtbot.waitUntil(lambda: widget._thread is None and not widget._preview.busy)
    assert not widget.start.isEnabled()
    widget.selection["model_basis_error"] = module.UNBOUND_MODEL_REASON
    widget._actions()
    assert "not yet bound" in widget.basis.text()
    widget.images.click()
    widget.model.click()
    assert navigation == [("phase_model", "normalisation"), "model"]


def test_native_calibration_opens_preparation_without_creating_job(qtbot, method_window, monkeypatch):
    window = method_window
    monkeypatch.setattr(module, "saved_selection", lambda *a, **kw: selection())
    monkeypatch.setattr(module, "selection_preview", lambda *a, **kw: selection())
    monkeypatch.setattr(module, "prepare_calibration_phase", lambda *a, **kw:
                        pytest.fail("Opening Calibration must not prepare or start a job"))
    window._show_mass_calibration_values("phase_model")
    preparation = window.mass_calibration_display.preparation
    qtbot.waitUntil(lambda: preparation._thread is None and not preparation._preview.busy)
    assert preparation.root == window._method_series_root()
    assert preparation.table.rowCount() == 1
    assert not preparation.isHidden()
    window._show_mass_calibration_values("colour")
    assert preparation.root is None and preparation.isHidden()


@pytest.mark.parametrize("known_before_build", [True, False])
def test_unbound_model_refit_completion_automatically_uses_candidate_then_existing_fit_chain(qtbot, tmp_path, monkeypatch, known_before_build):
    from PySide6.QtCore import QObject, Signal
    from darsia.fluidflower import calibration_model_training
    class Studio(QObject):
        training_completed = Signal(str)
    class Process(QObject):
        finished = Signal(int, object)
    studio, process = Studio(), Process()
    studio.status = SimpleNamespace(text=lambda: "fit failed")
    panel = RefitJobPanel(machine="host")
    qtbot.addWidget(panel)
    monkeypatch.setattr(panel, "_create_blocker", lambda **kw: "")
    saved = dict(selection(), model_basis_error=module.UNBOUND_MODEL_REASON if known_before_build else "")
    monkeypatch.setattr(module, "saved_selection", lambda *a, **kw: deepcopy(saved))
    monkeypatch.setattr(module, "selection_preview", lambda *a, **kw: deepcopy(saved))
    request = dict(output_root=str(tmp_path / "new-model"))
    prepared = dict(request=request, bundles=[])
    calls = []
    def prepare_training(*args, **kwargs):
        assert kwargs["workers"] == 6
        return prepared
    monkeypatch.setattr(module, "prepare_calibration_training", prepare_training)
    monkeypatch.setattr(calibration_model_training, "launch_prepared_training",
        lambda steps, result: calls.append(("train", result)) or process)
    phase = dict(root=str(tmp_path / "phase"), complete=True,
                 job_facts=dict(expected_runs=["sample"]))
    missing_grid = []
    def phase_ready(*args, **kwargs):
        if not known_before_build and kwargs.get("model_reference") is None:
            missing_grid.append(True)
            raise ValueError(module.UNBOUND_MODEL_REASON)
        calls.append(("phase", kwargs.get("model_reference")))
        return dict(snapshot=phase, selection=selection(), reused=True, registration=None)
    monkeypatch.setattr(module, "prepare_calibration_phase", phase_ready)
    monkeypatch.setattr(panel, "start_after_phase_one", lambda *a: calls.append(("mass", a)))
    jobs = SimpleNamespace(series_id="series", registry=SimpleNamespace(jobs=lambda *a: []),
        panel=SimpleNamespace(local_machine="host"),
        shared_jobs=SimpleNamespace(automatic=SimpleNamespace(isChecked=lambda: False), cpu=SimpleNamespace(value=lambda: 6)), stores=SimpleNamespace(path="stores"))
    page = SimpleNamespace(studio=studio, native_steps=object())
    window = SimpleNamespace(ensure_jobs=lambda: jobs, open_image_selection_plan=lambda: None,
        _select_row=lambda *a: None, show_phase_model_workflow=lambda: None,
        phase_model_workspace=SimpleNamespace(ensure_studio=lambda: page))
    publications = []
    window._publish_prepared_model = lambda *a, **kw: publications.append(kw["finished"])
    monkeypatch.setattr(module, "require_current_model_basis", lambda *a, **kw: {})
    widget = module.CalibrationPreparation(window, panel)
    qtbot.addWidget(widget)
    widget.set_series(tmp_path)
    qtbot.waitUntil(lambda: widget._thread is None and not widget._preview.busy)
    assert widget.start.isEnabled()
    widget.prepare_and_start()
    qtbot.waitUntil(lambda: widget._thread is None and not widget._preview.busy)
    assert calls == [("train", prepared)]
    assert len(missing_grid) == (0 if known_before_build else 1)
    assert not widget.can_close() and not widget.start.isEnabled()
    studio.training_completed.emit(request["output_root"])
    qtbot.waitUntil(lambda: len(publications) == 1 and widget._thread is None)
    assert calls == [("train", prepared)]
    assert not widget.can_close()
    publications[0](True, "")
    qtbot.waitUntil(lambda: len(calls) == 3 and widget._thread is None)
    assert calls[1] == ("phase", str((tmp_path / "new-model").resolve()))
    assert calls[2] == ("mass", (phase["root"], ["sample"]))
    assert widget.can_close()


def test_model_exit_without_original_completed_artifact_signal_does_not_start_phase(qtbot, tmp_path, monkeypatch):
    from PySide6.QtCore import QObject, Signal
    from darsia.fluidflower import calibration_model_training
    class Studio(QObject):
        training_completed = Signal(str)
    class Process(QObject):
        finished = Signal(int, object)
    studio, process = Studio(), Process()
    studio.status = SimpleNamespace(text=lambda: "model inspection failed")
    panel = RefitJobPanel(machine="host")
    qtbot.addWidget(panel)
    jobs = SimpleNamespace(series_id="series")
    page = SimpleNamespace(studio=studio, native_steps=object())
    window = SimpleNamespace(open_image_selection_plan=lambda: None,
        _select_row=lambda *a: None, show_phase_model_workflow=lambda: None,
        phase_model_workspace=SimpleNamespace(ensure_studio=lambda: page))
    widget = module.CalibrationPreparation(window, panel)
    qtbot.addWidget(widget)
    widget.root = tmp_path
    widget.selection = selection()
    monkeypatch.setattr(calibration_model_training, "launch_prepared_training", lambda *a: process)
    monkeypatch.setattr(widget, "_prepare_phase", lambda *a: pytest.fail("No inspected model"))
    widget._training_prepared(jobs, dict(request=dict(output_root=str(tmp_path / "model"))), tmp_path, "series")
    process.finished.emit(0, None)
    assert widget._training_context is None
    assert "model inspection failed" in widget.status.text()


@pytest.mark.parametrize("changed_context", [False, True])
def test_model_publication_failure_or_changed_context_cannot_prepare_phase(qtbot, tmp_path, monkeypatch, changed_context):
    callbacks, advanced = [], []
    window = SimpleNamespace(open_image_selection_plan=lambda: None, _select_row=lambda *a: None,
        show_phase_model_workflow=lambda: None,
        _publish_prepared_model=lambda *a, **kw: callbacks.append(kw["finished"]))
    panel = RefitJobPanel(machine="host")
    qtbot.addWidget(panel)
    widget = module.CalibrationPreparation(window, panel)
    qtbot.addWidget(widget)
    widget.root, widget.selection = tmp_path, selection()
    widget._model_publication_context = (tmp_path, b"original manifest", object())
    monkeypatch.setattr(widget, "_continue_with_model", lambda *a: advanced.append(a))
    widget._publish_model(object(), tmp_path, "series", "new-model", widget._generation)
    assert widget._model_publication_pending and not advanced
    if changed_context:
        widget._generation += 1
    callbacks[0](changed_context, "" if changed_context else "The save was rejected.")
    assert not widget._model_publication_pending and not advanced
    assert widget._published_candidate is None
    assert "not started" in widget.status.text()
