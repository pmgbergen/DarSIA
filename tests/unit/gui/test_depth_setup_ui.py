"""Depth pane and unchanged-command sequence with fake QProcess only."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QObject, QProcess, Signal
from PySide6.QtWidgets import QPushButton

from darsia.fluidflower import depth_setup_ui as depth_ui


class FakeProcess(QObject):
    finished = Signal(int, object)
    errorOccurred = Signal(object)

    def errorString(self):
        return "fake executable could not start"


class FakeRunner:
    def __init__(self, window):
        self.window = window
        self.active = {}
        self.calls = []
        self.processes = []
        self.errors = []

    def start_workflow_process(self, argv, run_button, abort_button, **kwargs):
        process = FakeProcess()
        self.processes.append(process)
        self.calls.append((argv, kwargs))
        self.active[kwargs["workflow"]] = (process, kwargs["workflow"])
        # Mirrors the existing original handler's retirement before the helper's
        # appended slot. No executable, kernel, workflow or config save is run.
        process.finished.connect(
            lambda *_args: self.active.pop(kwargs["workflow"], None)
        )
        run_button.setEnabled(False)
        abort_button.setVisible(True)
        abort_button.setEnabled(True)
        return process

    def abort_workflow_process(self, process):
        process.setProperty("darsia_aborted", True)

    def _show_error_dialog(self, workflow, code, lines):
        self.errors.append((workflow, code, lines))


@pytest.fixture
def sequence(qtbot, tmp_path):
    button = QPushButton("Play")
    stop = QPushButton("Stop")
    qtbot.addWidget(button)
    qtbot.addWidget(stop)
    path = tmp_path / "run.toml"
    path.write_bytes(b"# synthetic config; no depth computation\n")
    messages = []
    window = SimpleNamespace(
        config_file=str(path),
        print_log=messages.append,
        toolbar_builder=SimpleNamespace(play_action=button, stop_action=stop),
        preprocessing_tab=SimpleNamespace(process=None),
        setup_tab=SimpleNamespace(process=None),
    )
    window.process_runner = FakeRunner(window)
    task = depth_ui.DepthWorkflowSequence(
        window, "python-existing-env", resolve_conflict=lambda _path: False
    )
    return task, window, path, messages


def test_panel_read_existing_csv_defaults_no_constant_or_file_write(qtbot, tmp_path):
    panel = depth_ui.DepthSetupPanel()
    qtbot.addWidget(panel)
    path = tmp_path / "existing-points.csv"
    panel.set_configuration({"measurements": str(path)})
    assert panel.mode.currentData() == "Load from CSV"
    assert panel.constant_depth.text() == ""
    assert panel.compute_button.isEnabled()
    assert panel.requested_settings() == {
        "measurements_mode": "Load from CSV",
        "measurements": str(path),
    }
    assert not path.exists()


def test_panel_constant_is_explicit_positive_and_emits_only_on_compute(qtbot, tmp_path):
    panel = depth_ui.DepthSetupPanel()
    qtbot.addWidget(panel)
    emitted = []
    panel.compute_requested.connect(emitted.append)
    panel.set_configuration(
        {
            "measurements_mode": "constant",
            "measurements": str(tmp_path / "generated.csv"),
        }
    )
    assert not panel.compute_button.isEnabled()
    panel.constant_depth.setText("0.0123")
    assert panel.compute_button.isEnabled() and not emitted
    panel.compute_button.click()
    assert emitted == [
        {
            "measurements_mode": "constant",
            "measurements": str(tmp_path / "generated.csv"),
            "constant_depth": 0.0123,
        }
    ]
    assert not (tmp_path / "generated.csv").exists()
    panel.set_running(True)
    assert not panel.compute_button.isEnabled() and not panel.mode.isEnabled()
    panel.set_running(False)
    assert panel.compute_button.isEnabled()


@pytest.mark.parametrize("number", ["0", "-1", "nan", "inf", "not a depth"])
def test_invalid_constant_never_emits(qtbot, tmp_path, number):
    panel = depth_ui.DepthSetupPanel()
    qtbot.addWidget(panel)
    panel.set_configuration(
        {
            "measurements_mode": "constant",
            "measurements": str(tmp_path / "generated.csv"),
            "constant_depth": number,
        }
    )
    assert not panel.compute_button.isEnabled()


def test_explicit_series_csv_seed_is_used_only_when_config_has_no_path(qtbot):
    panel = depth_ui.DepthSetupPanel()
    qtbot.addWidget(panel)
    panel.set_configuration({}, measured_csv="explicit-existing-series.csv")
    assert panel.measurements.text() == "explicit-existing-series.csv"
    panel.set_configuration(
        {"measurements": "selected-other.csv"},
        measured_csv="explicit-existing-series.csv",
    )
    assert panel.measurements.text() == "selected-other.csv"
    panel.set_status("Saved map range: a supplied status string")
    assert panel.status.text() == "Saved map range: a supplied status string"


def test_argv_uses_original_entrypoints_and_only_depth_flags():
    first, second = depth_ui.depth_workflow_commands(
        "python", "chosen.toml", force=True
    )
    assert first == [
        "python",
        "-m",
        "darsia.presets.workflows.user_interface_preprocessing",
        "--config",
        "chosen.toml",
        "--depth-measurements",
        "--force",
    ]
    assert second == [
        "python",
        "-m",
        "darsia.presets.workflows.user_interface_setup",
        "--config",
        "chosen.toml",
        "--depth",
    ]
    assert "--all" not in first + second
    assert "-c" not in first + second


def test_two_original_commands_are_sequential_and_preserve_selected_config(
    sequence, qtbot
):
    task, window, path, _messages = sequence
    original = path.read_bytes()
    done = []
    busy = []
    task.completed.connect(lambda success, text: done.append((success, text)))
    task.busy_changed.connect(busy.append)
    assert task.start(path)
    runner = window.process_runner
    assert len(runner.calls) == 1
    assert runner.calls[0][1]["workflow"] == "preprocessing"
    assert window.preprocessing_tab.process is runner.processes[0]
    assert window.setup_tab.process is runner.processes[0]
    runner.processes[0].finished.emit(0, QProcess.ExitStatus.NormalExit)
    qtbot.waitUntil(lambda: len(runner.calls) == 2)
    assert not done and task.running
    assert runner.calls[1][1]["workflow"] == "setup"
    assert (
        runner.calls[0][1]["config_path"] == path == runner.calls[1][1]["config_path"]
    )
    runner.processes[1].finished.emit(0, QProcess.ExitStatus.NormalExit)
    assert done == [(True, "Depth map saved.")]
    assert busy == [True, False]
    assert not task.running
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "exit_code,status",
    [(2, QProcess.ExitStatus.NormalExit), (0, QProcess.ExitStatus.CrashExit)],
)
def test_failed_first_step_never_starts_map(sequence, qtbot, exit_code, status):
    task, window, path, _messages = sequence
    done = []
    task.completed.connect(lambda success, text: done.append((success, text)))
    task.start(path)
    window.process_runner.processes[0].finished.emit(exit_code, status)
    qtbot.wait(1)
    assert len(window.process_runner.calls) == 1
    assert done and done[-1][0] is False and not task.running


def test_abort_first_step_does_not_start_second(sequence, qtbot):
    task, window, path, _messages = sequence
    task.start(path)
    first = window.process_runner.processes[0]
    task.abort()
    assert first.property("darsia_aborted")
    first.finished.emit(0, QProcess.ExitStatus.NormalExit)
    qtbot.wait(1)
    assert len(window.process_runner.calls) == 1
    assert not task.running


def test_selected_run_changed_between_steps_stops_sequence(sequence, qtbot):
    task, window, path, _messages = sequence
    done = []
    task.completed.connect(lambda success, text: done.append((success, text)))
    task.start(path)
    window.config_file = str(path.parent / "other-run.toml")
    window.process_runner.processes[0].finished.emit(0, QProcess.ExitStatus.NormalExit)
    qtbot.waitUntil(lambda: bool(done))
    assert len(window.process_runner.calls) == 1
    assert "selected run changed" in done[-1][1]


def test_conflict_cancel_and_existing_other_job_start_nothing(sequence):
    task, window, path, _messages = sequence
    task.resolve_conflict = lambda _path: None
    assert not task.start(path)
    assert not window.process_runner.calls
    task.resolve_conflict = lambda _path: True
    window.process_runner.active["other"] = (object(), "other")
    assert not task.start(path)
    assert not window.process_runner.calls


def test_force_applies_only_to_original_measurement_generation(sequence):
    task, window, path, _messages = sequence
    task.resolve_conflict = lambda _path: True
    task.start(path)
    assert "--force" in window.process_runner.calls[0][0]
    assert "--force" not in task._commands[1]


def test_failed_to_start_keeps_original_error_presentation_and_stops(sequence):
    task, window, path, messages = sequence
    done = []
    task.completed.connect(lambda success, text: done.append((success, text)))
    task.start(path)
    process = window.process_runner.processes[0]
    process.errorOccurred.emit(QProcess.ProcessError.FailedToStart)
    assert len(window.process_runner.calls) == 1
    assert window.process_runner.errors == [
        ("preprocessing", -1, ["fake executable could not start"])
    ]
    assert messages == ["fake executable could not start"]
    assert not window.process_runner.active
    assert window.toolbar_builder.play_action.isEnabled()
    assert not window.toolbar_builder.stop_action.isEnabled()
    assert done[-1][0] is False and not task.running
