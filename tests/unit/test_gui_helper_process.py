"""The window's helper process: requests, answers, failures and the user's choice.

A real helper process starts with this interpreter; it reads no series.
"""
import os
import time

import pytest

from darsia.fluidflower import helper_process, storage_paths
from darsia.fluidflower.helper_process import HelperProcess, HelperUnavailable


@pytest.fixture
def helper():
    logged = []
    process = HelperProcess(log=logged.append)
    process.logged = logged
    yield process
    process.stop()


def test_a_request_runs_a_module_function_and_returns_its_value(helper):
    helper.start()
    assert helper.submit("operator:add", 2, 3).result(timeout=60) == 5
    assert helper.submit("os.path:basename", "folder/file.txt").result(timeout=60) == "file.txt"


def test_failures_come_back_as_the_exception_type_and_message(helper):
    with pytest.raises(RuntimeError, match="ZeroDivisionError"):
        helper.submit("operator:truediv", 1, 0).result(timeout=60)
    with pytest.raises(RuntimeError, match="could not be sent back"):
        helper.submit("threading:Lock").result(timeout=60)
    with pytest.raises(RuntimeError, match="Not a module function"):
        helper.submit("operator").result(timeout=60)


def test_the_helper_reads_paths_through_the_window_machine_path_map(helper, monkeypatch, tmp_path):
    rows = (("\\\\server\\share", str(tmp_path)),)
    monkeypatch.setattr(storage_paths, "machine_path_map", lambda: rows)
    assert helper.submit("darsia.fluidflower.storage_paths:machine_path_map").result(timeout=60) == rows


def test_the_helper_runs_this_darsia(helper):
    import darsia

    found = helper.submit("darsia.fluidflower.gui_helper:module_file", "darsia").result(timeout=60)
    assert os.path.normcase(os.path.realpath(found)) == os.path.normcase(os.path.realpath(darsia.__file__))
    with pytest.raises(RuntimeError, match="not callable"):
        helper.submit("darsia.fluidflower.gui_helper:resolve", "darsia:__name__").result(timeout=60)
    with pytest.raises(RuntimeError, match="not callable"):
        helper.submit("darsia:__name__").result(timeout=60)


def test_an_ended_helper_starts_once_more_then_reads_move_to_threads(helper):
    with pytest.raises(HelperUnavailable):
        helper.submit("os:_exit", 1).result(timeout=60)
    assert helper.submit("operator:add", 1, 1).result(timeout=60) == 2
    with pytest.raises(HelperUnavailable):
        helper.submit("os:_exit", 1).result(timeout=60)
    with pytest.raises(HelperUnavailable, match="ended twice"):
        helper.submit("operator:add", 1, 1)
    assert helper.failed and "own process" in helper.logged[0]


def test_stop_does_not_wait_for_the_helper(helper):
    helper.start()
    process = helper._process
    slow = helper.submit("time:sleep", 30)
    started = time.monotonic()
    helper.stop()
    assert time.monotonic() - started < 1.0
    process.wait(timeout=20)
    with pytest.raises(HelperUnavailable):
        slow.result(timeout=20)


def test_the_choice_is_off_by_default_and_follows_the_setting(tmp_path):
    from PySide6.QtCore import QSettings

    settings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    assert not helper_process.helper_enabled(settings)
    helper_process.set_helper_enabled(True, settings)
    assert helper_process.helper_enabled(QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat))
    helper_process.set_helper_enabled(False, settings)
    assert not helper_process.helper_enabled(settings)


def test_only_an_installed_helper_that_is_on_is_used(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    settings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(helper_process, "_installed", None)
    assert helper_process.current() is None
    helper_process.set_helper_enabled(False, settings)
    installed = helper_process.install(settings=settings)
    try:
        assert not installed.running and helper_process.current() is None
        installed.enabled = True
        helper_process.set_helper_enabled(True, settings)
        assert helper_process.current() is installed and installed.running
    finally:
        helper_process.uninstall()
    assert helper_process.current() is None
