"""The completion-to-next-command gap must not admit another calculation."""

from types import SimpleNamespace

from PySide6.QtCore import QProcess

from .test_native_depth_integration import (
    native_window,
    loading_run_window,
    run_window,
    original_dialogs,
    FakeProcess,
    prepare_depth,
)


def test_depth_sequence_gap_blocks_original_play_and_direct_dispatch(
    run_window, monkeypatch
):
    from darsia.fluidflower import depth_setup_ui
    from darsia.gui.ui.main_window import MainWindow

    context, _initial, _config = prepare_depth(run_window, monkeypatch)
    window = context.window
    pending = []
    monkeypatch.setattr(
        depth_setup_ui,
        "QTimer",
        SimpleNamespace(singleShot=lambda _delay, callback: pending.append(callback)),
    )
    numerical_dispatch = []
    monkeypatch.setattr(
        MainWindow, "run_selected_workflow", lambda self: numerical_dispatch.append(True)
    )
    window.depth_workflow.panel.compute_button.click()
    window._select_row(("preprocessing", "protocols"))
    assert window._darsia_visible()
    FakeProcess.instances[0].finished.emit(0, QProcess.ExitStatus.NormalExit)
    assert pending and not window.process_runner.active
    assert window.depth_workflow.sequence.running
    window._select_row(("preprocessing", "protocols"))
    # Simulate the original runner's completion enabling Play while it has no
    # active entry. The instance change hook must immediately disable it.
    window.toolbar_builder.play_action.setEnabled(True)
    assert not window.toolbar_builder.play_action.isEnabled()
    window.run_selected_workflow()
    assert not numerical_dispatch
    pending.pop()()
    assert len(FakeProcess.instances) == 2
    FakeProcess.instances[1].finished.emit(0, QProcess.ExitStatus.NormalExit)
    assert not window.depth_workflow.sequence.running
