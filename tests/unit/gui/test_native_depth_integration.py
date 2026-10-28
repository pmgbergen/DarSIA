"""Native Depth wiring with local TOML and fake processes; no depth calculation."""

import copy
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import toml
from PySide6.QtCore import QCoreApplication, QEvent, QObject, QProcess, Signal
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget
from shiboken6 import isValid

from .test_native_sidebar import (
    native_window,
    run_window as loading_run_window,
    _select_run,
)
from darsia.fluidflower import depth_setup_ui as depth_ui
from darsia.fluidflower.series_layer_view import StepSection


def install_geometry_host(window):
    """Use real Qt ownership, without loading scientific geometry in this fixture."""
    view = QWidget()
    layout = QVBoxLayout(view)
    view.rig_depth_host = QWidget()
    QVBoxLayout(view.rig_depth_host)
    view.step_depth = StepSection(None, "Depth", view.rig_depth_host, expanded=False)
    layout.addWidget(view.step_depth)
    view.steps_scroll = QScrollArea()
    view.rig_panel = None
    view._native_workflow_scope_connected = True
    view.set_workflow_scope = lambda scope: setattr(view, "_workflow_scope", scope)
    view.wait_for_workers = lambda: None
    workspace = window.geometry_workspace
    workspace.simple_view = view
    workspace.view_stack.addWidget(view)
    return view


@pytest.fixture
def run_window(loading_run_window, qtbot):
    qtbot.waitUntil(
        lambda: not loading_run_window.window._series_loading, timeout=10000
    )
    return loading_run_window


class FakeProcess(QObject):
    finished = Signal(int, object)
    errorOccurred = Signal(object)
    readyReadStandardOutput = Signal()
    MergedChannels = QProcess.ProcessChannelMode.MergedChannels
    CrashExit = QProcess.ExitStatus.CrashExit
    instances = []

    def __init__(self, parent=None):
        super().__init__(parent)
        self.argv = []
        self.started = False
        self.instances.append(self)

    def setProgram(self, program):
        self.argv = [program]

    def setArguments(self, arguments):
        self.argv += list(arguments)

    def setWorkingDirectory(self, cwd):
        self.cwd = cwd

    def setProcessChannelMode(self, mode):
        pass

    def start(self):
        self.started = True

    def processId(self):
        return 0  # The unchanged dashboard also observes this fake process.

    def readAllStandardOutput(self):
        return b""

    def errorString(self):
        return "fake Depth executable failed"


@pytest.fixture(autouse=True)
def original_dialogs(monkeypatch):
    from darsia.gui.ui.process_runner import ProcessRunner

    calls = SimpleNamespace(done=[], error=[])
    monkeypatch.setattr(
        ProcessRunner,
        "_show_done_dialog",
        lambda self, workflow, actions, config: calls.done.append(
            (workflow, actions, config)
        ),
    )
    monkeypatch.setattr(
        ProcessRunner,
        "_show_error_dialog",
        lambda self, workflow, code, lines: calls.error.append((workflow, code, lines)),
    )
    return calls


def prepare_depth(run_window, monkeypatch):
    from darsia.gui.ui import process_runner

    context = run_window
    window = context.window
    config = context.merged["series-a"]["first"]
    with config.open("a", encoding="utf-8") as stream:
        stream.write(
            '\nunknown_marker = "stay"\n'
            '[depth]\nmeasurements_mode = "constant"\n'
            'measurements = "generated_depth.csv"\nconstant_depth = 0.02\n'
            "resolution = [50, 60]\n"
            'depth_map = "explicit_depth_map.npz"\n'
            '[unrelated.inactive]\nmarker = "preserved"\n'
            "numbers = [1, 2, 3]\n"
        )
    initial = {path: path.read_bytes() for path in context.paths}
    monkeypatch.setattr(
        depth_ui,
        "load_saved_depth_map",
        lambda *args, **kwargs: depth_ui.SavedDepthMap(None, None, "Saved range: synthetic"),
    )
    _select_run(window, "series-a", "first")
    depth_before_geometry = copy.deepcopy(window.config_dict)
    install_geometry_host(window)
    window._select_row(("setup", "depth"))
    assert window.config_dict == depth_before_geometry, "Opening embedded Depth changed unrelated configuration"
    FakeProcess.instances = []
    monkeypatch.setattr(process_runner, "QProcess", FakeProcess)
    # Use the original runner with its real save/reload/Done/Error hooks, only
    # replacing the process constructor and never launching an executable.
    monkeypatch.setattr(
        window.process_runner,
        "start_workflow_process",
        lambda *args, **kwargs: process_runner.ProcessRunner.start_workflow_process(
            window.process_runner, *args, **kwargs
        ),
    )
    window.depth_workflow.sequence.resolve_conflict = lambda path: False
    return context, initial, config


def test_depth_is_in_rig_geometry_and_legacy_selection_opens_embedded_pane(native_window):
    window = native_window
    view = install_geometry_host(window)
    labels = [row.label for row in window.sidebar._sections["setup"]._rows.values()]
    assert "Rig Geometry" in labels and "Depth" not in labels
    assert "Depth measurements" not in labels and "Depth map" not in labels
    window._select_row(("preprocessing", "depth_measurements"))
    assert window.sidebar.get_selected() == ("geometry_facies", "rig_frame")
    assert window.content_stack.currentWidget() is window.geometry_workspace
    assert view.rig_depth_host.isAncestorOf(window.depth_workflow.panel)
    assert view.step_depth.is_expanded()
    assert not window.toolbar_builder.play_action.isEnabled()
    assert not window.toolbar_builder.wizard_action.isEnabled()
    assert "Choose a run" in window.depth_workflow.panel.status.text()
    assert "Compute depth map" in window.toolbar_builder.play_action.toolTip()
    window._select_row(("preprocessing", "protocols"))
    assert "Depth pane" not in window.toolbar_builder.play_action.toolTip()


def test_depth_navigation_and_editing_do_not_write_config_or_run(
    run_window, monkeypatch
):
    context, initial, config = prepare_depth(run_window, monkeypatch)
    panel = context.window.depth_workflow.panel
    panel.constant_depth.setText("0.0123")
    context.window._select_row(("preprocessing", "protocols"))
    assert not context.window.depth_workflow._embedded_active
    context.window._select_row(("setup", "depth"))
    assert panel.constant_depth.text() == "0.0123"
    assert config.read_bytes() == initial[config]
    assert not FakeProcess.instances
    for path, contents in initial.items():
        assert path.read_bytes() == contents


def test_embedded_depth_survives_geometry_view_replacement_without_losing_draft(native_window):
    window = native_window
    old = install_geometry_host(window)
    window._select_row(("setup", "depth"))
    panel = window.depth_workflow.panel
    panel.constant_depth.setText("0.0123")
    window.geometry_workspace._discard_page()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(old)
    assert isValid(panel) and panel.parentWidget() is window
    assert panel.constant_depth.text() == "0.0123"
    new = install_geometry_host(window)
    # Retain the same selected run/context; no file load or calculation.
    window._set_geometry_workflow_scope("rig_frame")
    assert new.rig_depth_host.isAncestorOf(panel)
    assert panel.constant_depth.text() == "0.0123"


def test_explicit_compute_preserves_every_other_value_and_sequences_original_commands(
    run_window, monkeypatch, qtbot, original_dialogs
):
    context, initial, config = prepare_depth(run_window, monkeypatch)
    window = context.window
    # Keep pending choices from other editors, along with the authored fields.
    window.config_dict.setdefault("unrelated", {})["pending_setting"] = "retain me"
    values = tomllib.loads(toml.dumps(window.config_dict))
    values["depth"]["constant_depth"] = 0.0123
    panel = window.depth_workflow.panel
    panel.constant_depth.setText("0.0123")
    panel.compute_button.click()
    assert window.depth_workflow.sequence.running
    assert not window.run_combo.isEnabled()
    assert not window.phase_model_workspace.series_selector.isEnabled()
    assert not window.toolbar_builder.play_action.isEnabled()
    assert not panel.compute_button.isEnabled()
    assert tomllib.loads(config.read_text("utf-8")) == values
    first = FakeProcess.instances[0]
    assert first.argv[-1] == "--depth-measurements"
    assert "user_interface_preprocessing" in first.argv[2]
    first.finished.emit(0, QProcess.ExitStatus.NormalExit)
    qtbot.waitUntil(lambda: len(FakeProcess.instances) == 2)
    assert not original_dialogs.done, "Only intermediate success is suppressed"
    assert tomllib.loads(config.read_text("utf-8")) == values
    second = FakeProcess.instances[1]
    assert second.argv[-1] == "--depth"
    assert "user_interface_setup" in second.argv[2]
    second.finished.emit(0, QProcess.ExitStatus.NormalExit)
    assert not window.depth_workflow.sequence.running
    assert original_dialogs.done == [("setup", ["depth"], Path(config))]
    assert panel.compute_button.isEnabled()
    assert not window.toolbar_builder.play_action.isEnabled()
    assert "synthetic" in panel.status.text()
    assert tomllib.loads(config.read_text("utf-8")) == values
    for path, contents in initial.items():
        if path != config:
            assert path.read_bytes() == contents


@pytest.mark.parametrize("exit_code", [1, 2])
def test_measurement_failure_preserves_original_errors_and_never_starts_map(
    run_window, monkeypatch, original_dialogs, exit_code
):
    context, _initial, _config = prepare_depth(run_window, monkeypatch)
    window = context.window
    window.depth_workflow.panel.compute_button.click()
    process = FakeProcess.instances[0]
    process.finished.emit(exit_code, QProcess.ExitStatus.NormalExit)
    assert len(FakeProcess.instances) == 1
    assert original_dialogs.error == [("preprocessing", exit_code, [])]
    assert not original_dialogs.done
    assert not window.depth_workflow.sequence.running
    assert "failed" in window.depth_workflow.panel.status.text()


def test_map_failure_preserves_original_errors_without_done(
    run_window, monkeypatch, qtbot, original_dialogs
):
    context, _initial, _config = prepare_depth(run_window, monkeypatch)
    window = context.window
    window.depth_workflow.panel.compute_button.click()
    FakeProcess.instances[0].finished.emit(0, QProcess.ExitStatus.NormalExit)
    qtbot.waitUntil(lambda: len(FakeProcess.instances) == 2)
    FakeProcess.instances[1].finished.emit(3, QProcess.ExitStatus.NormalExit)
    assert original_dialogs.error == [("setup", 3, [])]
    assert not original_dialogs.done
    assert not window.depth_workflow.sequence.running
    assert "failed" in window.depth_workflow.panel.status.text()


def test_guard_cancelled_launch_does_not_leave_depth_busy(run_window, monkeypatch):
    context, _initial, _config = prepare_depth(run_window, monkeypatch)
    window = context.window
    monkeypatch.setattr(
        window.process_runner, "start_workflow_process", lambda *args, **kwargs: None
    )
    window.depth_workflow.panel.compute_button.click()
    assert not window.depth_workflow.sequence.running
    assert not FakeProcess.instances
    assert window.depth_workflow.panel.compute_button.isEnabled()
    assert "cancelled" in window.depth_workflow.panel.status.text()


def test_failed_to_start_has_original_error_and_restores_depth_controls(
    run_window, monkeypatch, original_dialogs
):
    context, _initial, _config = prepare_depth(run_window, monkeypatch)
    window = context.window
    window.depth_workflow.panel.compute_button.click()
    process = FakeProcess.instances[0]
    process.errorOccurred.emit(QProcess.ProcessError.FailedToStart)
    assert original_dialogs.error == [
        ("preprocessing", -1, ["fake Depth executable failed"])
    ]
    assert not window.depth_workflow.sequence.running
    assert not window.process_runner.active
    assert window.depth_workflow.panel.compute_button.isEnabled()
    assert not window.toolbar_builder.stop_action.isEnabled()
    assert not window.toolbar_builder.play_action.isEnabled()


def test_stop_routes_depth_after_navigation_without_darsia_visible(
    run_window, monkeypatch
):
    context, _initial, _config = prepare_depth(run_window, monkeypatch)
    window = context.window
    aborted = []
    monkeypatch.setattr(
        window.process_runner,
        "abort_workflow_process",
        lambda process: (
            aborted.append(process),
            process.setProperty("darsia_aborted", True),
        ),
    )
    window.depth_workflow.panel.compute_button.click()
    window._select_row(("phase_model", "training"))
    assert not window._darsia_visible()
    window.abort_selected_workflow()
    process = FakeProcess.instances[0]
    assert aborted == [process]
    process.finished.emit(0, QProcess.ExitStatus.NormalExit)
    assert len(FakeProcess.instances) == 1
    assert not window.depth_workflow.sequence.running


def test_configuration_refresh_and_series_change_do_not_recurse_or_keep_old_depth(
    run_window, monkeypatch, qtbot
):
    context, _initial, config = prepare_depth(run_window, monkeypatch)
    window = context.window
    panel = window.depth_workflow.panel
    assert panel.constant_depth.text() == "0.02"
    window.config_controller.load_config()
    assert panel.constant_depth.text() == "0.02"
    window._select_series("series-b")
    qtbot.waitUntil(lambda: not window._series_loading, timeout=10000)
    assert not panel.measurements.text()
    assert "Choose a run" in panel.status.text()
    assert not FakeProcess.instances


def test_changed_file_choice_cancels_before_depth_save(run_window, monkeypatch):
    from darsia.fluidflower import native_darsia_app as native

    context, _initial, config = prepare_depth(run_window, monkeypatch)
    window = context.window
    contents = config.read_bytes() + b"\n# edited outside the window\n"
    config.write_bytes(contents)
    monkeypatch.setattr(native, "_ask_config_reload", lambda *_args: "cancel")
    window.depth_workflow.panel.compute_button.click()
    assert config.read_bytes() == contents
    assert not FakeProcess.instances
    assert "cancelled" in window.depth_workflow.panel.status.text()


def test_saved_map_range_uses_reader_and_reports_absence_without_computation(tmp_path):
    import numpy as np

    missing = {"data": {"results": str(tmp_path)}}
    calls = []
    assert "not saved yet" in depth_ui.saved_depth_map_status(
        missing, reader=lambda path: calls.append(path)
    )
    assert not calls
    path = tmp_path / "saved.npz"
    path.write_bytes(b"synthetic existence marker; reader is mocked")
    text = depth_ui.saved_depth_map_status(
        {"depth": {"depth_map": str(path)}},
        reader=lambda path: SimpleNamespace(img=np.array([0.01, 0.02])),
    )
    assert "0.01–0.02 m" in text
