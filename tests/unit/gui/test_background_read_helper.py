"""Background readings run in the helper process when it is on, and on threads otherwise."""
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMainWindow

from darsia.fluidflower import background_read, helper_process
from darsia.fluidflower.background_read import BackgroundRead


@pytest.fixture
def helper(monkeypatch):
    process = helper_process.HelperProcess()
    monkeypatch.setattr(helper_process, "_installed", process)
    yield process
    process.stop()


def test_a_reading_with_a_call_runs_in_the_helper(qtbot, helper):
    reader = BackgroundRead()
    on_thread = []
    assert reader.read("sum", lambda: on_thread.append(True), call=("operator:add", (2, 3)), wait_s=60) == (True, 5, "")
    assert on_thread == [] and not reader.busy()
    done, value, error = reader.read("fails", lambda: None, call=("operator:truediv", (1, 0)), wait_s=60)
    assert done and value is None and error.startswith("ZeroDivisionError")


def test_a_slow_helper_reading_answers_later(qtbot, helper, monkeypatch):
    monkeypatch.setattr(background_read, "QUICK_ANSWER_S", 0.0)
    reader = BackgroundRead()
    with qtbot.waitSignal(reader.answered, timeout=60000) as signal:
        assert reader.read("slow", lambda: None, call=("os.path:basename", ("a/b",))) == (False, None, "")
        assert reader.busy()
    assert signal.args == ["slow", "b", ""] and not reader.busy()


def test_without_the_helper_or_when_it_cannot_answer_the_reading_runs_on_a_thread(qtbot, helper, monkeypatch):
    reader = BackgroundRead()
    helper.enabled = False
    assert reader.read("off", lambda: "thread", call=("operator:add", (2, 3)), wait_s=60) == (True, "thread", "")
    helper.enabled = True
    monkeypatch.setattr(helper, "submit", lambda *args: (_ for _ in ()).throw(helper_process.HelperUnavailable("gone")))
    assert reader.read("gone", lambda: "thread", call=("operator:add", (2, 3)), wait_s=60) == (True, "thread", "")


def test_a_helper_that_ends_before_answering_hands_the_reading_to_a_thread(qtbot, helper, monkeypatch):
    monkeypatch.setattr(background_read, "QUICK_ANSWER_S", 0.0)
    reader = BackgroundRead()
    with qtbot.waitSignal(reader.answered, timeout=60000) as signal:
        assert reader.read("ended", lambda: "thread", call=("os:_exit", (1,)))[0] is False
    assert signal.args == ["ended", "thread", ""]


def test_the_settings_menu_turns_the_helper_on_and_off(qtbot, tmp_path, helper):
    window = QMainWindow()
    qtbot.addWidget(window)
    window.menuBar().addMenu("Settings")
    settings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    action = helper_process.add_helper_process_action(window, settings)
    menu = window.menuBar().actions()[0].menu()
    assert action in menu.actions() and action.isCheckable() and not action.isChecked()
    action.setChecked(True)
    assert helper_process.helper_enabled(settings) and helper.running
    action.setChecked(False)
    assert not helper_process.helper_enabled(settings) and helper_process.current() is None
    assert not helper.running
    action.setChecked(True)
    assert helper_process.helper_enabled(settings) and helper_process.current() is helper and helper.running
