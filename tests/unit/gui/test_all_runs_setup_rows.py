"""All runs for Setup's Build run model and Full setup.

Both steps work on the opened run's configuration, so "All runs" runs them
for each registered run in turn, as for the Direct colour method rows. The
sequence is driven with fake processes; the strip with the real window on
local manifests. No DarSIA workflow process is started.
"""

from darsia.fluidflower import all_runs

from .test_all_runs import FakeWindow, _sequence
from .test_native_row_scope import scope_window  # noqa: F401
from .test_native_series_method import method_window  # noqa: F401


def test_build_run_model_and_full_setup_may_run_for_all_runs():
    assert all_runs.SETUP_ALL_RUNS_ROWS == {("setup", "rig"), ("setup", "all")}
    assert all_runs.SETUP_ALL_RUNS_ROWS <= all_runs.ALL_RUNS_ROWS
    # Depth has its own two-stage page; the danger zone is never repeated.
    assert ("setup", "depth") not in all_runs.ALL_RUNS_ROWS
    assert ("calibration", "delete") not in all_runs.ALL_RUNS_ROWS


def test_the_strip_shows_on_the_setup_rows_without_a_computing_size(scope_window, qtbot):  # noqa: F811
    w = scope_window
    strip = w.all_runs
    # (Selecting Depth opens its own page, which loads a run configuration.)
    for logical, shown in [
        (("setup", "rig"), True),
        (("phase_model", "training"), False),
        (("setup", "all"), True),
    ]:
        w._select_row(logical)
        qtbot.wait(10)
        assert strip.panel.isVisible() is shown, logical
    w._select_row(("setup", "rig"))
    qtbot.wait(10)
    assert strip.resize_label.isHidden()
    assert strip.run_button.text() == "Run AC28"
    assert strip.all_button.text() == "Run all runs (2)"
    # A Direct colour method row still shows its computing size.
    w._select_row(("analysis", "mass"))
    qtbot.wait(10)
    assert not strip.resize_label.isHidden()


def test_run_all_on_build_run_model_confirms_once_without_a_method_question(
    scope_window, monkeypatch  # noqa: F811
):
    w = scope_window
    strip = w.all_runs
    w._select_row(("setup", "rig"))
    w.selected_action, w.selected_checkbox_id = "setup", "rig"
    asked, started = [], []
    strip.confirm = lambda question: asked.append(question) or True
    monkeypatch.setattr(w, "_confirm_method_action", lambda *args: asked.append(args) or True)
    monkeypatch.setattr(strip.sequence, "start", lambda row, runs: started.append((row, runs)) or True)
    strip.run_all()
    assert len(asked) == 1 and "for all 2 runs" in asked[0]
    assert "Computed at" not in asked[0]
    assert started[0][0] == ("setup", "rig")
    assert [run for run, _path in started[0][1]] == ["ac28", "ac47"]
    assert [strip.results.item(i).text() for i in range(2)] == ["AC28: waiting", "AC47: waiting"]
    strip._completed(True, "All runs finished.")
    assert w.run_combo.isEnabled() and strip.status.text() == "All runs finished."


def test_the_sequence_builds_each_runs_model_in_turn(qtbot):
    window = FakeWindow()
    window.selected_action, window.selected_checkbox_id = "setup", "rig"
    window.action_dispatch = {"setup": window.tab}
    sequence, events = _sequence(qtbot, window)
    assert sequence.start(("setup", "rig"), [("ac28", "a.toml"), ("ac47", "b.toml")])
    window.tab.process.finished.emit(0, all_runs.QProcess.ExitStatus.NormalExit)
    qtbot.waitUntil(lambda: window.tab.started == ["a.toml", "b.toml"])
    window.tab.process.finished.emit(0, all_runs.QProcess.ExitStatus.NormalExit)
    qtbot.waitUntil(lambda: not sequence.running)
    assert ("ac28", "done") in events and ("ac47", "done") in events
    assert events[-1] == ("completed", True, "All runs finished.")
