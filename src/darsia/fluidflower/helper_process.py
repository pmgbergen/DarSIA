"""The window's helper process for the reading that pages start.

When it is on (Settings > «Load pages in a helper process»), the reads that
pages start run in one long-lived process of their own: saved map versions and
the reference point check, for example. Their work then never holds the
window's interpreter. When it is off, the default, the reads run on threads of
the window, and the machine keeps the helper's memory and cores for its
calculations. It is off by default because the reads moved so far mostly wait
on files, and threads keep the window as responsive (measured on a series
share). Heavy computations still go to Jobs.

The helper is a fresh interpreter (``python -m darsia.fluidflower.gui_helper``)
with private pipes, not a multiprocessing child: the window's launcher script
has no main guard, so a spawned child would open a second window. It starts
in the background when the window opens. A helper that ends unexpectedly is
started once more; after a second failure the window reads on threads and
says so in its log.

Only an application that installs the helper (``install``) uses it, so tests
and other programs that build windows keep reading on threads.
"""
from __future__ import annotations

import itertools
import os
import subprocess
import sys
import threading
from concurrent.futures import Future
from pathlib import Path

from darsia.fluidflower import gui_helper

SETTINGS = ("DarSIA", "DarSIA GUI")
SETTING_KEY = "background/helper_process"
ACTION_TEXT = "Load pages in a helper process (uses more memory)"
ACTION_TIP = ("On: pages read their saved files in a separate process, so the window stays responsive. "
              "Off: they read in the window's own process, which leaves more of a small machine "
              "to the calculations.")
# Modules the helper imports right after it starts, so a page's first read does not wait for them.
WARM_UP = ("darsia.fluidflower.layer_painting_version_archive", "darsia.fluidflower.rig_point_review")
# Seconds the window waits for the helper to end before it stops it.
STOP_WAIT_S = 2.0


class HelperUnavailable(RuntimeError):
    """The helper process cannot answer; the read runs on a thread instead."""


class HelperProcess:
    """One helper process and the requests it answers."""

    def __init__(self, *, executable: str | None = None, log=None) -> None:
        self.executable = executable or sys.executable
        self.log = log or (lambda text: None)
        self._lock = threading.Lock()
        self._numbers = itertools.count(1)
        self._pending: dict[int, Future] = {}
        self._process: subprocess.Popen | None = None
        self._sink = None
        self._starts = 0
        self.failed = ""
        # The user's choice (helper_enabled); install sets it.
        self.enabled = True

    @property
    def running(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    def start(self) -> None:
        """Start the helper unless it runs; it then imports WARM_UP in the background."""
        with self._lock:
            if self.running or self.failed:
                return
            self._start_locked()
        try:
            self.submit("darsia.fluidflower.gui_helper:warm_up", *WARM_UP)
        except HelperUnavailable:
            pass

    def _start_locked(self) -> None:
        self._starts += 1
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen([self.executable, "-m", "darsia.fluidflower.gui_helper"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, creationflags=flags, env=helper_environment())
        self._process, self._sink = process, process.stdin
        threading.Thread(target=self._receive, args=(process,), name="darsia-gui-helper-answers",
                         daemon=True).start()

    def submit(self, target: str, *args, **kwargs) -> Future:
        """Run ``target`` (``"package.module:function"``) with these arguments in the helper.

        Returns a Future of its value. It raises HelperUnavailable when the
        helper cannot run or ended before answering, and RuntimeError with the
        exception's type and message when the function raised.
        """
        from darsia.fluidflower import storage_paths

        future: Future = Future()
        with self._lock:
            if self.failed:
                raise HelperUnavailable(self.failed)
            if not self.running:
                if self._starts >= 2:
                    self._fail_locked("The helper process ended twice")
                    raise HelperUnavailable(self.failed)
                self._start_locked()
            number = next(self._numbers)
            self._pending[number] = future
            try:
                gui_helper.write_message(self._sink, ("call", number, target, args, kwargs,
                                                      tuple(storage_paths.machine_path_map())))
            except (OSError, ValueError) as error:
                self._pending.pop(number, None)
                raise HelperUnavailable(f"The helper process cannot be reached: {error}") from error
        return future

    def _receive(self, process) -> None:
        """Hand each answer to its Future; when the helper ends, fail the requests it had."""
        source = process.stdout
        try:
            while True:
                number, ok, value = gui_helper.read_message(source)
                with self._lock:
                    future = self._pending.pop(number, None)
                if future is None:
                    continue
                if ok:
                    future.set_result(value)
                else:
                    future.set_exception(RuntimeError(value))
        except (EOFError, OSError, ValueError):
            pass
        with self._lock:
            if self._process is process:
                self._process = self._sink = None
            pending, self._pending = self._pending, {}
        for future in pending.values():
            future.set_exception(HelperUnavailable("The helper process ended before it answered"))

    def _fail_locked(self, reason: str) -> None:
        if not self.failed:
            self.failed = reason
            self.log(f"{reason}. Pages read their files in the window's own process from now on.")

    def stop(self) -> None:
        """Ask the helper to end, without waiting for it; it is stopped if it does not end soon."""
        with self._lock:
            process, sink = self._process, self._sink
            self._process = self._sink = None
            self._starts = 0
        if process is None:
            return
        try:
            gui_helper.write_message(sink, ("stop",))
            sink.close()
        except (OSError, ValueError):
            pass

        def end():
            try:
                process.wait(STOP_WAIT_S)
            except subprocess.TimeoutExpired:
                process.kill()

        threading.Thread(target=end, name="darsia-gui-helper-stop", daemon=True).start()


def helper_environment() -> dict[str, str]:
    """The window's environment, with the window's own DarSIA first on the import path.

    The helper then runs the same code as the window, also when the window
    runs from a copy of the source tree rather than the installed one.
    """
    import darsia

    source = str(Path(darsia.__file__).resolve().parents[1])
    environment = dict(os.environ)
    paths = [path for path in environment.get("PYTHONPATH", "").split(os.pathsep) if path]
    environment["PYTHONPATH"] = os.pathsep.join([source, *(path for path in paths if path != source)])
    return environment


_installed: HelperProcess | None = None


def _settings(settings=None):
    if settings is not None:
        return settings
    from PySide6.QtCore import QSettings
    return QSettings(*SETTINGS)


def helper_enabled(settings=None) -> bool:
    """The user's choice; off unless it was turned on."""
    value = _settings(settings).value(SETTING_KEY, False)
    return value in (True, "true", "True", "1", 1)


def set_helper_enabled(enabled: bool, settings=None) -> None:
    """Save the user's choice and start or stop the installed helper to follow it."""
    store = _settings(settings)
    store.setValue(SETTING_KEY, bool(enabled))
    store.sync()
    if _installed is None:
        return
    if enabled:
        _installed.failed = ""
        _installed.start()
    else:
        _installed.stop()


def current() -> HelperProcess | None:
    """The installed helper when it is on and has not failed; None means read on threads."""
    helper = _installed
    if helper is None or helper.failed or not helper.enabled:
        return None
    return helper


def install(application=None, *, settings=None, log=None, executable=None) -> HelperProcess:
    """Use a helper process in this application; start it now when it is on.

    ``application`` (the QApplication) stops the helper when it quits.
    """
    global _installed
    if _installed is None:
        _installed = HelperProcess(executable=executable, log=log)
        _installed.enabled = helper_enabled(settings)
        if application is not None:
            application.aboutToQuit.connect(_installed.stop)
    if _installed.enabled:
        _installed.start()
    return _installed


def uninstall() -> None:
    """Stop and forget the installed helper (tests)."""
    global _installed
    helper, _installed = _installed, None
    if helper is not None:
        helper.stop()


def add_helper_process_action(window, settings=None):
    """Add the on/off choice to the window's Settings menu (or its menu bar)."""
    from PySide6.QtGui import QAction

    action = QAction(ACTION_TEXT, window)
    action.setMenuRole(QAction.MenuRole.NoRole)
    action.setCheckable(True)
    action.setChecked(helper_enabled(settings))
    action.setToolTip(ACTION_TIP)
    action.setStatusTip(ACTION_TIP)

    def chosen(enabled):
        if _installed is not None:
            _installed.enabled = bool(enabled)
        set_helper_enabled(enabled, settings)

    action.toggled.connect(chosen)
    menus = [entry.menu() for entry in window.menuBar().actions() if entry.menu() is not None]
    settings_menu = next((menu for menu in menus if menu.title().replace("&", "") == "Settings"), None)
    (settings_menu or window.menuBar()).addAction(action)
    return action
