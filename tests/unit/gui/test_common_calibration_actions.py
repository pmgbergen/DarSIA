from types import SimpleNamespace

from PySide6.QtCore import QProcess, Qt

from darsia.fluidflower.common_calibration_actions import (
    OriginalCalibrationControls, start_original_automatic,
)
from darsia.fluidflower.native_mass_calibration_layout import NativeMassCalibrationDisplay


def test_shared_page_retains_visible_sections_and_only_enables_supported_controls(qtbot, monkeypatch, tmp_path):
    page = NativeMassCalibrationDisplay()
    qtbot.addWidget(page)
    monkeypatch.setattr(page.refit_panel, "set_series", lambda root: None)
    page.resize(1400, 900)
    page.show()
    page.set_refit_series(None)
    page.set_calibration_context("colour_path", tmp_path)
    assert all(not section.isHidden() for section in page.sections.values())
    assert not page.phase_parameters.isVisible()
    assert page.overflow_choice.isHidden()
    assert page.sections["job"].isEnabled()
    page.sections["job"].set_expanded(True)
    assert page.original_controls.isVisible()
    assert not page.job_controls.isVisible()
    calls = []
    page.original_controls.start_requested.connect(calls.append)
    page.original_controls.trials.setValue(75)
    assert calls == [] and page.refit_panel.job_root is None
    qtbot.mouseClick(page.original_controls.start_button, Qt.MouseButton.LeftButton)
    assert calls == [page.original_controls.options()]
    page.set_refit_series(tmp_path)
    page.set_calibration_context("phase_model", tmp_path)
    page.sections["chemistry"].set_expanded(True)
    assert page.refit_panel.calibration_scope_combo.isVisible()
    assert not page.original_controls.isVisible()
    assert page.refit_panel.f_check.isVisible()


def _window(tmp_path):
    config = tmp_path / "run.toml"
    config.write_text('[calibration.mass]\ndata_selection="fit"\n')
    launches, logs = [], []
    process = SimpleNamespace(state=lambda: QProcess.ProcessState.NotRunning)
    def launch(*args, **kwargs):
        launches.append((args, kwargs))
        return process
    window = SimpleNamespace(
        _saved_method=SimpleNamespace(active="colour_path"), _series_loading=False,
        calibration_tab=SimpleNamespace(process=None), _ensure_run_loaded=lambda: None,
        phase_model_workspace=SimpleNamespace(series_id="test-series"),
        _config_series="test-series", config_file=str(config),
        run_combo=SimpleNamespace(currentData=lambda: str(config)),
        toolbar_builder=SimpleNamespace(play_action=object(), stop_action=object()),
        process_runner=SimpleNamespace(start_workflow_process=launch), print_log=logs.append,
    )
    return window, launches, logs


def test_explicit_start_uses_existing_runner_and_saved_selected_run(qtbot, tmp_path, monkeypatch):
    controls = OriginalCalibrationControls()
    qtbot.addWidget(controls)
    window, launches, logs = _window(tmp_path)
    remembers = []
    monkeypatch.setattr(controls, "remember", lambda: remembers.append(True))
    controls.trials.setValue(30)
    controls.seed.setValue(7)
    controls.values.setCurrentIndex(1)
    assert launches == [] and remembers == []
    result = start_original_automatic(window, controls, controls.options())
    assert result is window.calibration_tab.process
    args, kwargs = launches[0]
    assert "darsia.fluidflower.native_mass_calibration" in args[0]
    assert args[0][-1] == "--per-facies"
    assert args[0][args[0].index("--trials") + 1] == "30"
    assert kwargs["config_path"] == tmp_path / "run.toml"
    assert kwargs["actions"] == ["mass"] and remembers == [True]
    assert "launch requested" in controls.status.text()
    assert logs == []


def test_wrong_model_stale_series_and_active_process_never_launch(qtbot, tmp_path):
    controls = OriginalCalibrationControls()
    qtbot.addWidget(controls)
    window, launches, _ = _window(tmp_path)
    window._saved_method.active = "phase_model"
    assert start_original_automatic(window, controls, controls.options()) is None
    window._saved_method.active = "colour_path"
    window._config_series = "previous-series"
    assert start_original_automatic(window, controls, controls.options()) is None
    window._config_series = "test-series"
    window.calibration_tab.process = SimpleNamespace(state=lambda: QProcess.ProcessState.Running)
    assert start_original_automatic(window, controls, controls.options()) is None
    assert launches == []


def test_phase_colour_paths_routes_to_existing_inputs_without_starting_work(qtbot, tmp_path):
    page = NativeMassCalibrationDisplay()
    qtbot.addWidget(page)
    qtbot.addWidget(page.phase_colour_paths)
    calls = []
    window = SimpleNamespace(
        _show_original_mass_calibration_controls=lambda mode: calls.append(("original", mode)),
        open_image_selection_plan=lambda: calls.append("images"),
        _select_row=lambda value: calls.append(value),
        run_combo=SimpleNamespace(currentData=lambda: None))
    page.install_common_actions(window)
    page.set_calibration_context("phase_model", tmp_path)
    paths = page.phase_colour_paths
    assert str(tmp_path) in paths.source.text()
    assert calls == []
    paths.images_button.click()
    assert paths.prepare_button.isHidden()
    assert calls == ["images"]
    assert page.refit_panel.job_root is None and page.refit_panel._thread is None
    page.set_calibration_context("colour_path", tmp_path)
    assert not paths.prepare_button.isEnabled()
