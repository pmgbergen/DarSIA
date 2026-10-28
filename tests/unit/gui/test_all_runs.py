"""All runs for the Direct colour method (N141 spec 2.6/3.4, Part 2).

The sequence is driven with fake processes; the strip with the real window on
local manifests. No DarSIA workflow process is started.
"""

from types import SimpleNamespace as NS

import pytest
from PySide6.QtCore import QObject, QProcess, Signal
from PySide6.QtWidgets import QMessageBox

from darsia.fluidflower import all_runs

from .test_native_row_scope import scope_window  # noqa: F401
from .test_native_series_method import method_window  # noqa: F401

OK = QProcess.ExitStatus.NormalExit


class FakeProcess(QObject):
    finished = Signal(int, object)


class FakeTab:
    def __init__(self, window):
        self.window, self.process, self.started = window, None, []

    def on_run_clicked(self):
        self.process = FakeProcess()
        self.started.append(self.window.loaded[-1])


class FakeWindow:
    def __init__(self, *, unreadable=()):
        self.loaded, self.unreadable, self.aborted = [], set(unreadable), []
        self.process_runner = NS(active={}, abort_workflow_process=self._abort)
        self.selected_action, self.selected_checkbox_id = "analysis", "mass"
        self.tab = FakeTab(self)
        self.action_dispatch = {"analysis": self.tab}

    def _load_run_config(self, path):
        self.loaded.append(path)
        return path not in self.unreadable

    def _abort(self, process):
        process.setProperty("darsia_aborted", True)
        self.aborted.append(process)


def _sequence(qtbot, window):
    sequence = all_runs.AllRunsSequence(window)
    events = []
    sequence.started.connect(lambda run: events.append(("started", run)))
    sequence.result.connect(lambda run, outcome: events.append((run, outcome)))
    sequence.completed.connect(lambda ok, message: events.append(("completed", ok, message)))
    return sequence, events


def test_resize_summary_reads_corrections_resize_with_the_workflow_loader():
    assert all_runs.resize_summary(None) == "native"
    assert all_runs.resize_summary({"type": {}}) == "native"
    assert all_runs.resize_summary({"resize": {"mode": "scale", "scale": 0.5}}) == "0.5 × native (resize)"
    assert all_runs.resize_summary({"resize": {"mode": "target_shape", "target_shape": [512, 1024]}}) == (
        "resized to 512 × 1024 pixels (resize)")
    assert all_runs.resize_summary({"active": ["type"], "resize": {"mode": "scale", "scale": 0.5}}) == "native"
    with pytest.raises(ValueError, match="requires 'scale'"):
        all_runs.resize_summary({"resize": {"mode": "scale"}})


def test_runs_resize_summary_counts_each_runs_own_file(tmp_path):
    native = tmp_path / "a.toml"
    native.write_text("[corrections.type]\n", encoding="utf-8")
    half = tmp_path / "b.toml"
    half.write_text('[corrections.resize]\nmode = "scale"\nscale = 0.5\n', encoding="utf-8")
    bare = tmp_path / "c.toml"
    bare.write_text('[data]\nresults = "r"\n', encoding="utf-8")
    text = all_runs.runs_resize_summary(
        [("ac28", native), ("ac47", half), ("ac52", half), ("ac53", None), ("ac54", bare)])
    assert text == "native for 2 runs; 0.5 × native (resize) for 2 runs; ac53: no run configuration"


def test_sequence_runs_each_run_in_order_and_skips_runs_without_configuration(qtbot):
    window = FakeWindow()
    sequence, events = _sequence(qtbot, window)
    assert sequence.start(("analysis", "mass"), [("ac28", "a.toml"), ("ac47", None), ("ac52", "c.toml")])
    assert window.tab.started == ["a.toml"]
    window.tab.process.finished.emit(0, OK)
    qtbot.waitUntil(lambda: window.tab.started == ["a.toml", "c.toml"], timeout=2000)
    window.tab.process.finished.emit(0, OK)
    qtbot.waitUntil(lambda: not sequence.running, timeout=2000)
    assert events == [
        ("started", "ac28"), ("ac28", "done"), ("ac47", "skipped: no run configuration"),
        ("started", "ac52"), ("ac52", "done"),
        ("completed", True, "All runs finished. 1 run(s) had no configuration and were skipped."),
    ]


def test_sequence_stops_at_the_first_error_and_pools_nothing(qtbot):
    window = FakeWindow()
    sequence, events = _sequence(qtbot, window)
    sequence.start(("analysis", "mass"), [("ac28", "a.toml"), ("ac47", "b.toml")])
    window.tab.process.finished.emit(2, OK)
    qtbot.waitUntil(lambda: not sequence.running, timeout=2000)
    assert window.tab.started == ["a.toml"]
    assert events[-2:] == [("ac28", "failed (exit code 2)"),
                           ("completed", False, "ac28 failed. The remaining runs were not started.")]


def test_sequence_stops_when_a_configuration_cannot_open_or_the_step_changed(qtbot):
    window = FakeWindow(unreadable={"b.toml"})
    sequence, events = _sequence(qtbot, window)
    sequence.start(("analysis", "mass"), [("ac28", "a.toml"), ("ac47", "b.toml"), ("ac52", "c.toml")])
    window.tab.process.finished.emit(0, OK)
    qtbot.waitUntil(lambda: not sequence.running, timeout=2000)
    assert ("ac47", "not started: its configuration could not be opened") in events
    window = FakeWindow()
    sequence, events = _sequence(qtbot, window)
    sequence.start(("analysis", "mass"), [("ac28", "a.toml"), ("ac47", "b.toml")])
    window.selected_checkbox_id = "volume"
    window.tab.process.finished.emit(0, OK)
    qtbot.waitUntil(lambda: not sequence.running, timeout=2000)
    assert events[-1] == ("completed", False, "The selected step changed. The remaining runs were not started.")
    assert window.tab.started == ["a.toml"]


def test_sequence_abort_stops_the_running_action_and_starts_no_other(qtbot):
    window = FakeWindow()
    sequence, events = _sequence(qtbot, window)
    sequence.start(("analysis", "mass"), [("ac28", "a.toml"), ("ac47", "b.toml")])
    process = window.tab.process
    sequence.abort()
    assert window.aborted == [process]
    process.finished.emit(1, OK)
    qtbot.waitUntil(lambda: not sequence.running, timeout=2000)
    assert events[-2:] == [("ac28", "stopped"), ("completed", False, "All runs stopped.")]
    window.process_runner.active["analysis"] = object()
    assert not sequence.start(("analysis", "mass"), [("ac28", "a.toml")])  # Another calculation runs.


def test_strip_shows_on_direct_colour_rows_only_with_scoped_buttons(scope_window, qtbot):  # noqa: F811
    w = scope_window
    strip = w.all_runs
    for logical, shown in [
        (("analysis", "mass"), True),
        (("calibration", "color"), True),
        (("calibration", "mass"), False),
        (("calibration", "delete"), False),
        (("phase_model", "training"), False),
        (("jobs", "overview"), False),
    ]:
        w._select_row(logical)
        qtbot.wait(10)
        assert strip.panel.isVisible() is shown, logical
    w._select_row(("analysis", "mass"))
    assert strip.run_button.text() == "Run AC28"
    assert strip.all_button.text() == "Run all runs (2)"
    assert "A resize trades accuracy for speed" in strip.resize_label.text()
    assert not strip.stop_button.isEnabled()
    played = []
    w.run_selected_workflow = lambda: played.append(True)
    strip.run_button.click()  # This run is Play for the selected run.
    assert played == [True]


def test_run_all_confirms_once_and_restores_the_selected_run(scope_window, qtbot, monkeypatch):  # noqa: F811
    w = scope_window
    strip = w.all_runs
    w._select_row(("analysis", "mass"))
    w.selected_action, w.selected_checkbox_id = "analysis", "mass"
    asked, started = [], []
    strip.confirm = lambda question: asked.append(question) or True
    monkeypatch.setattr(w, "_confirm_method_action", lambda method, text: asked.append(method) or True)
    monkeypatch.setattr(strip.sequence, "start", lambda row, runs: started.append((row, runs)) or True)
    strip.run_all()
    assert "for all 2 runs" in asked[0] and "Computed at:" in asked[0] and asked[1] == "colour_path"
    assert started[0][0] == ("analysis", "mass") and [run for run, _path in started[0][1]] == ["ac28", "ac47"]
    assert w._all_runs_confirmed and not w.run_combo.isEnabled()
    assert [strip.results.item(i).text() for i in range(2)] == ["AC28: waiting", "AC47: waiting"]
    strip._result("ac28", "done")
    strip._completed(False, "ac47 failed. The remaining runs were not started.")
    assert [strip.results.item(i).text() for i in range(2)] == ["AC28: done", "AC47: not started"]
    assert not w._all_runs_confirmed and w.run_combo.isEnabled()
    assert strip.status.text() == "ac47 failed. The remaining runs were not started."


def test_one_confirmation_guards_play_stop_and_done_dialogs(method_window, monkeypatch):  # noqa: F811
    w = method_window
    asked = []
    monkeypatch.setattr(w, "_confirm_method_action", lambda *a: asked.append(a) or False)
    w._all_runs_confirmed = True
    w.analysis_tab.run_analysis()  # No configuration: it only logs, and asks nothing.
    assert asked == []
    w._all_runs_confirmed = False
    w.analysis_tab.run_analysis()
    assert len(asked) == 1
    aborted = []
    monkeypatch.setattr(w.all_runs.sequence, "abort", lambda: aborted.append(True))
    w.all_runs.sequence._running = True
    try:
        w._update_run_action()
        assert not w.toolbar_builder.play_action.isEnabled()
        w.abort_selected_workflow()
        assert aborted == [True]
        monkeypatch.setattr(QMessageBox, "exec", lambda self: pytest.fail("No per-run Done dialog"))
        assert w.process_runner._show_done_dialog("analysis", ["mass"], None) is None
    finally:
        w.all_runs.sequence._running = False
